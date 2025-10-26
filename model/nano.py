"""
    This is module from nanoGPT

    https://github.com/karpathy/nanoGPT/blob/master/model.py
"""

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import GPT2LMHeadModel


class LayerNorm(nn.Module):

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, eps=1e-5)



class CausalSelfAttention(nn.Module):
    """
    This does not use PyTorch sdpa implementation yet
    """

    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx

        # projection for qkv
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))

    def forward(self, x, kv_cache):
        B, T, C = x.shape
        # calculate query, key, value
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # B, n_head, T, head_dim
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # B, n_head, T, head_dim
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # B, n_head, T, head_dim

        # apply kv cache
        if kv_cache is not None:
            k, v = kv_cache.insert_kv(self.layer_idx, k, v)  # insert and get the full kv cache
        
        Tq = q.shape[-2]
        Tk = k.shape[-2]

        # no kv cache or this is the prefill where q == k
        if kv_cache is None or Tq == Tk:
            # causal self-attention (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
            # att = (q @ k.transpose(-2, -1)) * (1 / math.sqrt(k.shape[-1]))
            # att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
            # att = F.softmax(att, dim=-1)
            # att = self.attn_dropout(att)
            # y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
            
            # use the pytorch sdpa implementation for consistency
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        elif Tq == 1:
            # during decoding only 1 token is input, attend all past kv attention
            y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        else:
            # during decoding and multiple queries are given, we need a slightly different
            # masking here, all queries could attend all past keys, but the queries are causal
            attn_mask = torch.zeros((Tq, Tk), dtype=torch.bool, device=q.device)
            prefix_len = Tk - Tq
            # all queries could attend all kv cache
            attn_mask[:, :prefix_len] = True
            # but the queries are causal
            attn_mask[:, prefix_len:] = torch.tril(torch.ones((Tq, Tq), dtype=torch.bool, device=q.device))
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

        y = y.transpose(1, 2).contiguous().view(B, T, C)  # reshape back and make it contiguous
        y = self.resid_dropout(self.c_proj(y))

        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc       = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu       = nn.GELU()
        self.c_proj     = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout    = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    
    def __init__(self, config, layer_idx):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config, layer_idx)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, kv_cache):
        x = x + self.attn(self.ln_1(x), kv_cache)
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config, layer_idx=i) for i in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        # init parameters
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
        
        print("number of parameters: %.2fM", self.get_num_params() / 1e6)

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, kv_cache=None):
        device = idx.device
        b, t = idx.shape
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        # if kv cache is not none, we need to shift the position
        if kv_cache is not None:
            pos = pos + kv_cache.get_pos()

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)

        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x, kv_cache)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # calculate loss if targets are provided
            if targets.dim() == 1:
                # this is a classification task, just use teh last token logits
                logits = self.lm_head(x[:, [-1], :])  # B, 1, vocab_size
            else:
                logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))  # (B, T) -> B x T
        else:
            # inference optimization to only calculate the last token logits
            logits = self.lm_head(x[:, [-1], :])  # use [-1] to keep sequence dimension
            loss = None
        
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        """
        simplify this to only small gpt2 model
        """
        from transformers import GPT2LMHeadModel

        config_args = dict(n_layer=12, n_head=12, n_embd=768)
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024  # context window length
        config_args['bias'] = True

        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith(".attn.mask")]  # discard this mask / buffer, not a param

        # hf model
        hf_model = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = hf_model.state_dict()

        # copy and ensure all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.bias")]
        transposed = ["attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight"]
        # basically the openai checkpoint use a Conv1D module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatch: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out that does not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        # Create AdamW optimizer and use the fused version if it is available
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, fused=True)

        return optimizer

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            # context window truncation
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]

            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature  # B, vocab_size
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')

            probs = torch.softmax(logits, dim=-1)
            id_next = torch.multinomial(probs, num_samples=1)
            idx = torch.concat([idx, id_next], dim=1)
        
        return idx
