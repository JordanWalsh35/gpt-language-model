import os
import sys
import numpy as np
import time
import math
import torch
from torch.distributed import init_process_group, destroy_process_group
from dataclasses import dataclass, asdict
from contextlib import nullcontext
from ast import literal_eval
import wandb

from model import GPTLanguageModel


@dataclass
class TrainConfig:
    eval_iters: int = 200
    eval_interval: int = 2000
    eval_only: bool = False
    gradient_accumulation_steps: int = 40
    iter_num: int = 0
    best_val_loss: float = 1e9
    always_save_checkpoint: bool = True
    backend: str = 'nccl'
    compile: bool = True

    # Logging
    wandb_log: bool = True
    wandb_project = 'gpt'
    wandb_run_name = 'gpt pre-training'
    log_interval: int = 1

    # Optimizer
    learning_rate: float = 6e-4
    weight_decay: float = 1e-1
    max_iters: int = 600000
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    decay_lr: bool = True
    warmup_iters: int = 2000
    lr_decay_iters: int = 600000
    min_lr: float = 6e-5



class Trainer:
    """ A class to train our GPT model. """

    def __init__(self, config, model_config, train_data, val_data, init_from):      
        self.config = config
        self.model_config = model_config
        self.model = self.init_model()
        self.device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.optimizer = self.model.configure_optimizers(self.config.weight_decay, self.config.learning_rate, (self.config.beta1, self.config.beta2), self.device_type)
        self.train_data = train_data
        self.val_data = val_data

        # Define checkpoints
        checkpoint_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
        self.checkpoint_path = os.path.join(checkpoint_dir, "pretraining.pt")
        # Initialize training from scratch or from checkpoint
        self.init_from = init_from
        # Check if parallel training (ddp) is available
        self.ddp = int(os.environ.get("RANK", -1)) != -1
        self.device = self.device_type
        self.dtype = torch.bfloat16
        self.ctx = nullcontext if self.device_type == 'cpu' else torch.amp.autocast(device_type=self.device_type, dtype=self.dtype)

        if self.ddp:
            init_process_group(backend=config.backend)
            ddp_rank = int(os.environ["RANK"])
            ddp_local_rank = int(os.environ["LOCAL_RANK"])
            ddp_world_size = int(os.environ["WORLD_SIZE"])
            self.device = f"cuda:{ddp_local_rank}"
            torch.cuda.set_device(self.device)
            self.master_process = ddp_rank == 0
            seed_offset = ddp_rank
            assert config.gradient_accumulation_steps % torch.cuda.device_count() == 0
            config.gradient_accumulation_steps //= torch.cuda.device_count()
        else:
            self.master_process = True
            seed_offset = 0
            ddp_world_size = 1
        tokens_per_iter = config.gradient_accumulation_steps * ddp_world_size * model_config.batch_size * model_config.block_size
        print(f"Tokens per iteration will be: {tokens_per_iter:,}")
    

    def init_model(self):
        if self.init_from == "start":
            print("Initializing new model")
            model = GPTLanguageModel(self.model_config)
        elif self.init_from == "resume":
            print(f"Resuming training from checkpoint")
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)           
            model = GPTLanguageModel(self.model_config)
            state_dict = checkpoint['model']
            unwanted = "_orig_mod."
            for k,v in list(state_dict.items()):
                if k.startswith(unwanted):
                    state_dict[k[len(unwanted):]] = state_dict.pop(k)
            
            model.load_state_dict(state_dict)
            self.config.iter_num = checkpoint['iter_num']
            self.config.best_val_loss = checkpoint['best_val_loss']
        return model
    
    
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
        ix = torch.randint(len(data) - self.config.block_size)
        x = torch.stack([torch.from_numpy((data[i:i + self.config.block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i + 1:i + 1 + self.block_size]).astype(np.int64)) for i in ix])

        if self.device_type == 'cuda':
            x, y = x.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(self.device, non_blocking=True)
        else:
            x, y = x.to(self.device), y.to(self.device)
        return x, y


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


    def run(self):
        """ Run the training loop. """

        # Logging
        if self.config.wandb_log and self.config.master_process:
            wandb.init(project=self.config.wandb_project, name=self.config.wandb_run_name, config=self.config)

        X, Y = self.get_batch("train")
        t0 = time.time()
        # Number of iterations in the lifetime of this process
        local_iter_num = 0
        raw_model = self.model.module if self.ddp else self.model
        running_mfu = -1.0

        while True:
            lr = self.get_learning_rate(self.config.iter_num) if self.config.decay_lr else self.config.learning_rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            
            # Evaluate loss on train/val sets and write checkpoints
            if self.config.iter_num % self.config.eval_interval == 0 and self.master_process:
                losses = self.estimate_loss()
                print(f"Step {self.config.iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

                if self.config.wandb_log:
                    wandb.log({
                        "iter": self.config.iter_num,
                        "train loss": losses['train'],
                        "val loss": losses['val'],
                        "learning rate": lr,
                        "mfu": running_mfu * 100})
                
                if losses['val'] < self.config.best_val_loss or self.config.always_save_checkpoint:
                    self.config.best_val_loss = losses['val']
                    if self.config.iter_num > 0:
                        checkpoint = {
                            'model': raw_model.state_dict(),
                            'optimizer': self.optimizer.state_dict(),
                            'model_args': asdict(self.model_config),
                            'iter_num': self.config.iter_num,
                            'best_val_loss': self.config.best_val_loss,
                            'config': self.config,   ######## Fix to contain all appropriate variables ##########
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
                # Initialize a GradScaler
                scaler = torch.cuda.amp.GradScaler(enabled=(self.dtype == 'float16'))
                scaler.scale(loss).backward()
            
            # Clip the gradient
            if self.config.grad_clip != 0.0:
                scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            # Step the optimizer and scaler if training in fp16
            scaler.step(self.optimizer)
            scaler.update()
            # Flush the gradients
            self.optimizer.zero_grad(set_to_none=True)

            # Timing and logging
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            if self.config.iter_num % self.config.log_interval == 0 and self.master_process:
                lossf = loss.item() * self.config.gradient_accumulation_steps
                if local_iter_num >= 5:
                    mfu = raw_model.estimate_mfu(self.model_config.batch_size * self.config.gradient_accumulation_steps, dt)
                    running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
                print(f"Iteration: {self.config.iter_num}, loss: {lossf:.4f}, time: {dt * 1000:.2f}ms, mfu: {running_mfu * 100:.2f}%")
            self.config.iter_num += 1
            local_iter_num += 1

            if self.config.iter_num > self.config.max_iters:
                break
        
        if self.ddp:
            destroy_process_group()