import os
import sys
import numpy as np
import time
import math
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import logging
from dataclasses import dataclass, asdict
from contextlib import nullcontext
from ast import literal_eval
import platform

from model import GPTLanguageModel


@dataclass
class TrainConfig:
    """ This dataclass holds the configuration for the Trainer class. """
    # General config
    batch_size: int = 12
    eval_iters: int = 200
    eval_interval: int = 2000
    eval_only: bool = False
    max_iters: int = 600000
    gradient_accumulation_steps: int = 40
    iter_num: int = 0
    best_val_loss: float = 1e9
    grad_clip: float = 1.0
    always_save_checkpoint: bool = True
    backend: str = 'nccl'
    compile: bool = True
    init_from: str = 'start'
    dtype: str = 'bfloat16'
    gpu_model: str = 'A100'
    calculate_mfu: bool = True

    # Learning Rate Decay
    warmup_iters: int = 2000
    lr_decay_iters: int = 600000
    decay_lr: bool = True
    min_lr: float = 6e-5

    # Optimizer
    learning_rate: float = 6e-4
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95



class Trainer:
    """ A class to train our GPT language model. """

    def __init__(self, config, model_config, train_data, val_data):      
        self.config = config
        self.model_config = model_config
        self.train_data = train_data
        self.val_data = val_data
        
        # Overwrite config if command line arguments
        if len(sys.argv) > 1:
            self.overwrite_configurations()
        
        # Define checkpoint path
        checkpoint_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
        self.checkpoint_path = os.path.join(checkpoint_dir, "training.pt")

        # Logging
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename="logs/trainer.log")
        self.logger = logging.getLogger()
        
        # Device type and context manager configuration
        self.device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = self.device_type
        self.dtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[self.config.dtype]
        self.ctx = nullcontext() if self.device_type == 'cpu' else torch.amp.autocast(device_type=self.device_type, dtype=self.dtype)

        # Initialize model, optimizer & scaler
        self.model, self.optimizer = self.init_model()
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.config.dtype == 'float16'))
        
        # Check if parallel training (ddp) is available
        self.ddp = int(os.environ.get("RANK", -1)) != -1
        if self.ddp:
            # If available, initialize ddp
            init_process_group(backend=config.backend)
            ddp_rank = int(os.environ["RANK"])
            ddp_local_rank = int(os.environ["LOCAL_RANK"])
            ddp_world_size = int(os.environ["WORLD_SIZE"])
            self.device = f"cuda:{ddp_local_rank}"
            torch.cuda.set_device(self.device)
            self.master_process = ddp_rank == 0
            seed_offset = ddp_rank
            assert self.config.gradient_accumulation_steps % torch.cuda.device_count() == 0
            self.config.gradient_accumulation_steps //= torch.cuda.device_count()
            self.model = DDP(self.model, device_ids=[ddp_local_rank])
        else:
            self.master_process = True
            seed_offset = 0
            ddp_world_size = 1

        # Calculate number of tokens per iteration
        tokens_per_iter = self.config.gradient_accumulation_steps * ddp_world_size * self.config.batch_size * self.model_config.block_size
        print(f"Tokens per iteration will be: {tokens_per_iter:,}")
        # Set reproducible random number generator
        torch.manual_seed(1337 + seed_offset)
        # Enable TensorFloat-32 computation on GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # Compile the model
        if self.config.compile and platform.system() != "Windows":
            print("Compiling model...")
            self.model = torch.compile(self.model)
    

    def init_model(self):
        # Initialize model and optimizer
        model = GPTLanguageModel(self.model_config)
        optimizer = model.configure_optimizers(self.config.weight_decay, self.config.learning_rate, (self.config.beta1, self.config.beta2), self.device_type)

        # Indicate that a new model has been initialized if init_from == 'start'
        if self.config.init_from == "start":
            print("Initializing new model")
            self.logger.info("New model initialized for training.")

        # Recreate model from checkpoint if init_from == 'resume'
        elif self.config.init_from == "resume":
            print("Resuming training from checkpoint")
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)           
            state_dict = checkpoint['model']

            # Remove unwanted prefix from the state_dict keys
            unwanted = "_orig_mod."
            for k,v in list(state_dict.items()):
                if k.startswith(unwanted):
                    state_dict[k[len(unwanted):]] = state_dict.pop(k)

            # Load the model state_dict and reset the iteration number + val loss
            model.load_state_dict(state_dict)
            self.config.iter_num = checkpoint['iter_num']
            self.config.best_val_loss = checkpoint['best_val_loss']
            optimizer.load_state_dict(checkpoint['optimizer'])
            self.logger.info("Training resumed from checkpoint.")

        return model, optimizer
    
    
    def overwrite_configurations(self):
        """ Takes arguments from the command line and overwrites the current config values."""
        for arg in sys.argv[:]:
            if arg.startswith("--"):
                key, val = arg.split("=")
                key = key[2:]
                if hasattr(self.config, key):
                    try:
                        lit_val = literal_eval(val)
                    except (SyntaxError, ValueError):
                        lit_val = val
                    setattr(self.config, key, lit_val)
                else:
                    raise ValueError(f"Unknown config key: {key}")


    def get_batch(self, split):
        data = self.train_data if split == "train" else self.val_data
        ix = torch.randint(len(data) - self.model_config.block_size, (self.config.batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i + self.model_config.block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i + 1:i + 1 + self.model_config.block_size]).astype(np.int64)) for i in ix])

        if self.device_type == 'cuda':
            x, y = x.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(self.device, non_blocking=True)
        else:
            x, y = x.to(self.device), y.to(self.device)
        return x, y


    def get_learning_rate(self, iteration):
        # Linear warm-up for warmup_iters steps
        if iteration < self.config.warmup_iters:
            return self.config.learning_rate * iteration / self.config.warmup_iters
        # If iteration > lr_decay_iters, return minimum learning rate
        if iteration > self.config.lr_decay_iters:
            return self.config.min_lr
        # In between, use cosine decay down to minimum learning rate
        decay_ratio = (iteration - self.config.warmup_iters) / (self.config.lr_decay_iters - self.config.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.config.min_lr + coeff * (self.config.learning_rate - self.config.min_lr)
    

    @torch.no_grad()
    def estimate_loss(self):
        out = {}
        self.model.eval()

        for split in ["train", "val"]:
            losses = torch.zeros(self.config.eval_iters)
            for k in range(self.config.eval_iters):
                X, Y = self.get_batch(split)
                with self.ctx:
                    logits, loss = self.model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.model.train()
        return out


    def run(self):
        """ Run the training loop. """
        # Get initial batch
        X, Y = self.get_batch("train")
        t0 = time.time()
        # Number of iterations in the lifetime of this process
        local_iter_num = 0
        raw_model = self.model.module if self.ddp else self.model
        running_mfu = -1.0

        # Training loop
        while True:
            lr = self.get_learning_rate(self.config.iter_num) if self.config.decay_lr else self.config.learning_rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            
            # Evaluate loss on train/val sets and write checkpoints
            if self.config.iter_num % self.config.eval_interval == 0 and self.master_process:
                losses = self.estimate_loss()
                print(f"Iteration {self.config.iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
                self.logger.info(f"Iteration #{self.config.iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

                if losses['val'] < self.config.best_val_loss or self.config.always_save_checkpoint:
                    self.config.best_val_loss = losses['val']
                    if self.config.iter_num > 0:
                        checkpoint = {
                            'model': raw_model.state_dict(),
                            'optimizer': self.optimizer.state_dict(),
                            'model_args': asdict(self.model_config),
                            'iter_num': self.config.iter_num,
                            'best_val_loss': self.config.best_val_loss,
                            'config': self.config,
                        }
                        torch.save(checkpoint, self.checkpoint_path)
            
            if self.config.iter_num == 0 and self.config.eval_only:
                break
            
            # Forward-Backward update, with optional gradient accumulation to simulate larger batch size
            for micro_step in range(self.config.gradient_accumulation_steps):
                if self.ddp:
                    self.model.require_backward_grad_sync = (micro_step == self.config.gradient_accumulation_steps - 1)
                with self.ctx:
                    logits, loss = self.model(X,Y)
                    loss = loss / self.config.gradient_accumulation_steps
                
                X, Y = self.get_batch("train")
                self.scaler.scale(loss).backward()
            
            # Clip the gradient
            if self.config.grad_clip != 0.0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            # Step the optimizer and scaler if training in fp16
            self.scaler.step(self.optimizer)
            self.scaler.update()
            # Flush the gradients
            self.optimizer.zero_grad(set_to_none=True)

            # Timing
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            if self.master_process and self.config.calculate_mfu:
                lossf = loss.item() * self.config.gradient_accumulation_steps
                if local_iter_num >= 5:
                    mfu = raw_model.estimate_mfu(self.config.batch_size * self.config.gradient_accumulation_steps, dt, self.config.gpu_model)
                    running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
                print(f"Iteration {self.config.iter_num}: loss: {lossf:.4f}, time: {dt:.2f}s, mfu: {running_mfu * 100:.2f}%")
                self.logger.info(f"Iteration {self.config.iter_num}: loss: {lossf:.4f}, time: {dt:.2f}s, mfu: {running_mfu * 100:.2f}%")
            self.config.iter_num += 1
            local_iter_num += 1

            if self.config.iter_num > self.config.max_iters:
                break
        
        if self.ddp:
            destroy_process_group()