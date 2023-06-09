import torch
import torch.nn as nn
from torch.nn import functional as Func
import inspect
import math
from dataclasses import dataclass


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.1
    bias: bool = True 


class LayerNorm(nn.Module):
    """ Layer Normalization with optional bias """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
    
    def forward(self, input):
        return Func.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
    


def gelu(x):
    """ GeLu activation function """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))



class CausalSelfAttention(nn.Module):
    """ Multi-head masked self-attention layer with a projection at the end.
        Alternatively can use nn.MultiheadAttention class. """
    
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # Key, Query, Value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout


    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # Calculate Query, Key, Values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # Flash attention
        y = Func.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y
    


class MultiLayerPerceptron(nn.Module):
    """ A Multilayer Perceptron with two layers and a GeLu activation function in between """

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        x = self.c_fc(x)
        x = gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x



class Block(nn.Module):
    """ Transformer Block """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MultiLayerPerceptron(config)
    
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x



class GPTLanguageModel(nn.Module):
    """ This is our GPT Language Model """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),   # Word token embedding
            wpe = nn.Embedding(config.block_size, config.n_embd),   # Positional Encoding Embedding
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        # Initialize weights
        self.apply(self.initial_weights)
        # Apply special scaled init to residual projections
        for name, parameter in self.named_parameters():
            if name.endswith('c_proj.weight'):
                nn.init.normal_(parameter, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
        
        # Print number of parameters
        print("Number of parameters: %.2fM" % (self.number_of_parameters()/1e6,))


    def initial_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    

    def number_of_parameters(self, non_embedding=True):
        num_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            num_params -= self.transformer.wpe.weight.numel()

        return num_params


    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        # Forward the GPT model
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = Func.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # Only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :])
            loss = None
 
        return logits, loss


    @classmethod
    def from_pretrained(cls, model_type):
        """ Load parameters from pretrained model. """
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024
        config_args['bias'] = True
        config_args['dropout'] = 0.1

        # Initialize model
        config = GPTConfig(**config_args)
        model = GPTLanguageModel(config)
        model_state_dict = model.state_dict()
        sd_keys = model_state_dict.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]

        # Initialize a Huggingface model
        from transformers import GPT2LMHeadModel
        hf_model = GPT2LMHeadModel.from_pretrained(model_type)
        hf_state_dict = hf_model.state_dict()

        # Ensure parameters and shapes align
        sd_keys_hf = hf_state_dict.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        assert len(sd_keys_hf) == len(sd_keys), "Mismatched keys"

        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert hf_state_dict[k].shape[::-1] == model_state_dict[k].shape
                with torch.no_grad():
                    model_state_dict[k].copy_(hf_state_dict[k].t())
            else:
                assert hf_state_dict[k].shape == model_state_dict[k].shape
                with torch.no_grad():
                    model_state_dict[k].copy_(hf_state_dict[k])
        
        return model


    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # Get parameters that require grad.
        param_dict = {pn:p for pn, p in self.named_parameters() if p.requires_grad}
        # Create optimizer groups where any param that is 2 dimensions or greater is weight decayed, otherwise no decay.
        decay_params = [p for n,p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n,p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}]

        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"Number of decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"Number of non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        # Create AdamW Optimizer
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"Using fused AdamW: {use_fused}")

        return optimizer
    

    def estimate_mfu(self, fbpass_per_iter, dt, gpu_model):
        """ Estimate the model flops utilization rate (MFU). References PaLM paper: https://arxiv.org/abs/2204.02311 """
        num_param = self.number_of_parameters()
        config = self.config
        L, H, Q, T =  config.n_layer, config.n_head, config.n_embd//config.n_head, config.block_size
        flops_per_token = 6*num_param + 12*L*H*Q*T
        flops_per_fbpass = flops_per_token * T
        flops_per_iter = flops_per_fbpass * fbpass_per_iter
        flops_achieved = flops_per_iter * (1.0/dt)  # per second
        # Tensor core FLOPS based on different GPU models, at mixed precision (FP16)
        flops_promised = {'A100': 312e12, "V100": 125e12, "T4": 65e12}[gpu_model]
        mfu = flops_achieved / flops_promised
        
        return mfu


    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """ 
        Take input sequence idx (Long Tensor of shape (b,t)) and complete the sequence 
        max_new_tokens times, feeding the predictions back into the model each time. 
        """

        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # Forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            # Optionally crop logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Apply softmax to convert logits to normalized probabilities
            probs = Func.softmax(logits, dim=-1)
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # Append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
