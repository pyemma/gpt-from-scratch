from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F



class MultiHeadCausalAttention(nn.Module):
    """
    A basic implementation of multi-head causal self-attention

    Support kv cache for efficient inference
    """
    def __init__(self, d_in, d_out, max_seq_len, num_heads, drop_out=0.1, bias=False):
        super().__init__()
        # the project layers for q, k, v; why do we need this? because the self-attention
        # does not comes up with many learnable parameters, thus we need this learnable
        # projection layers to learn from data on how to transform the input
        self.q_w = nn.Linear(d_in, d_out, bias=bias)
        self.k_w = nn.Linear(d_in, d_out, bias=bias)
        self.v_w = nn.Linear(d_in, d_out, bias=bias)

        # the final project layer of the output
        self.o_w = nn.Linear(d_out, d_out, bias=bias)

        self.scale = d_out ** -0.5

        # create the causal mask
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        # add 1 to the batch dimension and num_heads dimension for broadcasting
        # register_buffer will move the mask to the correct device
        self.register_buffer("mask", mask.view(1, 1, max_seq_len, max_seq_len))  # (B, num_heads, T, T)

        # add a dropout layer
        self.attn_dropout = nn.Dropout(p=drop_out)

        # head dimension
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

    def forward(self, x, use_cache=False, kv_cache=None):
        """
        x: (B, T, d_in), when kv_cache is used, T is 1 which is the new token during decoding
        use_cache: whether to use kv cache or not, turned off during training
        kv_cache: past kv tensors, of shape (B, num_heads, T, d_in)
        """
        B, T, _ = x.shape

        q = self.q_w(x)  # B, T, d_out
        k = self.k_w(x)  # B, T, d_out
        v = self.v_w(x)  # B, T, d_out

        # this is more efficient implementation than using multiple causal attention blocks
        # reshape the query, key, value to multiple heads
        q = q.view(B, T, self.num_heads, self.head_dim)
        k = k.view(B, T, self.num_heads, self.head_dim)
        v = v.view(B, T, self.num_heads, self.head_dim)

        # transpose to make the dimension order to be (B, num_head, T, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # if use_cache is True and kv_cache is given, we need to append to pervious kv cache
        new_kv_cache = kv_cache
        if use_cache and kv_cache is not None:
            past_k, past_v = kv_cache  # B, num_heads, past_seq+len, head_dim
            PAST_T = past_k.shape[-2]
            k = torch.cat([past_k, k], dim=-2)  # B, num_heads, PAST_T + T, head_dim
            v = torch.cat([past_v, v], dim=-2)  # B, num_heads, PAST_T + T, head_dim
        
        if use_cache:
            new_kv_cache = (k, v)  # for prefill stage, the kv_cache would be empty and we need to correctly populate it

        attn_scores = torch.matmul(q, k.transpose(-2, -1))  # B, num_heads, T, T (PAST_T + T)
        # the reason that we need the scale here is to handle the large hidden dimension usually
        # used in GPT, otherwise the backward gradient will be too small
        attn_scores = attn_scores / self.scale

        # apply the causal mask, we need to apply the mask to the original attention scores
        # first, otherwise the softmax is not normalized
        # add a slice to handle the case where the input sequence is shorter than the max_seq_len
        if use_cache and kv_cache is not None:
            # the mask should start from row PAST_T to PAST_T+T, up to the PAST_T+T column
            attn_scores = attn_scores.masked_fill(self.mask[:, :, PAST_T:PAST_T+T, :PAST_T+T] == 0 ,float('-inf'))
        else:
            attn_scores = attn_scores.masked_fill(self.mask[:, :, :T, :T] == 0 ,float('-inf'))

        attn_weights = torch.softmax(attn_scores, dim=-1)  # B, num_heads, T, T (PAST_T + T)

        # apply the dropout
        attn_weights = self.attn_dropout(attn_weights)

        # this is essentially using the attention scores to weight the value vectors
        # to compute the new context vector for next layer
        attn_output = torch.matmul(attn_weights, v)  # B, num_heads, T, head_dim

        # reshape attention output back to (B, T, d_out)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, self.num_heads * self.head_dim)

        # transpose to make the dimension order to be (B, T, d_out)
        attn_output = self.o_w(attn_output)  # B, T, d_out
        return attn_output, new_kv_cache


@dataclass
class Config:
    block_size: int = 1024
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True
    n_head: int = 12
    n_kv_head: int = 4

class CausalGroupQueryAttention(nn.Module):
    """
    Group query attention build on top of the causal self-attention
    from nanoGPT implementation
    """

    def __init__(self, config):
        super().__init__()

        self.head_dim = config.n_embd // config.n_head
        # projection for query
        self.q_w = nn.Linear(config.n_embd, self.head_dim * config.n_head, bias=config.bias)
        # projection for key and value
        self.kv_w = nn.Linear(config.n_embd, self.head_dim * config.n_kv_head * 2, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.n_head = config.n_head  # query head
        self.n_kv_head = config.n_kv_head  # kv head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.shape
        # calculate query, key, value
        q = self.q_w(x)  # B, T, head_dim * n_head
        k, v = self.kv_w(x).split(self.head_dim * self.n_kv_head, dim=2)  # B, T, head_dim * n_kv_head

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # B, n_head, T, head_dim
        k = k.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)  # B, n_kv_head, T, head_dim
        v = v.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)  # B, n_kv_head, T, head_dim

        # expand the key and value to match the number of heads
        kv_head_repeat = self.n_head // self.n_kv_head
        k = k.repeat_interleave(kv_head_repeat, dim=1)
        v = v.repeat_interleave(kv_head_repeat, dim=1)

        # causal self-attention (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1 / math.sqrt(k.shape[-1]))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        y = y.transpose(1, 2).contiguous().view(B, T, C)  # reshape back and make it contiguous
        y = self.resid_dropout(self.c_proj(y))

        return y

def get_num_params(module):
    return sum(p.numel() for p in module.parameters())

if __name__ == "__main__":
    config = Config()
    module = CausalGroupQueryAttention(config)
    x = torch.randn(32, 512, config.n_embd)
    out = module(x)
    print(out.shape)

    multi_head_attn = MultiHeadCausalAttention(d_in=config.n_embd, d_out=config.n_embd, max_seq_len=config.block_size, num_heads=config.n_head)
    out, _ = multi_head_attn(x)
    print(out.shape)

    print(f"""
Compare the total number of parameters between the two modules:
GQA: {get_num_params(module)}
Multi-head: {get_num_params(multi_head_attn)}
Ratio: {get_num_params(module) / get_num_params(multi_head_attn)}
        """)