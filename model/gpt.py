import torch
import torch.nn as nn

import tiktoken

from model.attentions import MultiHeadCausalAttention
from model.layers import GELU, LayerNorm

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 2,  # the original GPT-2 has 12 layers
    "dropout": 0.1,
    "qkv_bias": False
}


class FeedForward(nn.Module):
    """
    A basic implementation of feed forward network

    linear -> gelu -> linear
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], cfg["emb_dim"] * 4),
            GELU(),
            nn.Linear(cfg["emb_dim"] * 4, cfg["emb_dim"])
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    """
    GPT-2 is a decoder-only model, thus this transformer block is a decoder block

    Support kv cache for efficient inference
    """
    
    def __init__(self, cfg: dict):
        super().__init__()

        self.attn = MultiHeadCausalAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            max_seq_len=cfg["context_length"],
            num_heads=cfg["n_heads"],
            drop_out=cfg["dropout"],
            bias=cfg["qkv_bias"]
        )
        self.ff = FeedForward(cfg)

        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])

        self.dropout = nn.Dropout(cfg["dropout"])

    def forward(self, x, use_cache=False, kv_cache=None):
        # use per-norm, which provides better performance than post-norm
        x1 = self.norm1(x)
        x1, kv_cache = self.attn(x1, use_cache=use_cache, kv_cache=kv_cache)
        x1 = self.dropout(x1)
        x1 = x + x1  # residual connection


        x2 = self.norm2(x1)
        x2 = self.ff(x2)
        x2 = self.dropout(x2)
        x2 = x1 + x2  # residual connection

        return x2, kv_cache  # return the updated kv_cache


class GPT(nn.Module):

    def __init__(self, cfg: dict):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["dropout"])

        self.blocks = nn.Sequential(*[
            TransformerBlock(cfg) for _ in range(cfg["n_layers"])
        ])

        # one reason that we use layer norm instead of batch norm is the flexibility and
        # stability that layer norm works on individual feature dimension instead of 
        # batch dimension which is subsequential to change in LLM
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, inputs, use_cache=False, kv_caches=None):
        """
        kv_caches is a list of tuples, each tuple contains the past kv tensors for the layer
        """
        B, T = inputs.shape

        tok_embeddings = self.tok_emb(inputs)  # B, T, emb_dim
        # if kv cache is used, the position needs to be shifted
        if use_cache and kv_caches[0] is not None:
            current_kv_len = kv_caches[0][0].shape[-2]
            pos_embeddings = self.pos_emb(torch.arange(current_kv_len, current_kv_len + T, device=inputs.device))
        else:
            pos_embeddings = self.pos_emb(torch.arange(T, device=inputs.device))  # T, emb_dim

        embeddings = tok_embeddings + pos_embeddings  # broadcast pos_embedding
        embeddings = self.drop_emb(embeddings)

        for i, block in enumerate(self.blocks):
            embeddings, new_kv_cache = block(embeddings, use_cache=use_cache, kv_cache=kv_caches[i])
            kv_caches[i] = new_kv_cache

        embeddings = self.final_norm(embeddings)

        logits = self.out_head(embeddings)  # B, T, vocab_size

        return logits, kv_caches


def generate(
    model, 
    idx, 
    max_new_tokens, 
    context_size, 
    top_k=None, 
    temperature=None
):
    """
    Decoding phase of the model

    Implement the top-k sampling, temperature and greedy sampling
    """
    kv_caches = [None] * len(model.blocks)
    
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]  # (B, context_size)
        with torch.no_grad():
            logits, _ = model(idx_cond, use_cache=False, kv_caches=kv_caches)

        # the last hidden state is the probability of the next token
        logits = logits[:, -1, :]  # (B, vocab_size)

        # top-k sampling
        if top_k:
            top_k_logits, _ = torch.topk(logits, top_k)  # B, top_k
            # extract the minimum value of top-k logits for each sample row
            min_val = top_k_logits[:, -1]  # B,
            # filter value less then min_val, set to -inf
            logits = torch.where(logits < min_val, float('-inf'), logits)

        if temperature:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)  # (B, vocab_size)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
        else:
            probs = torch.softmax(logits, dim=-1)  # (B, vocab_size)
            # this is the greedy decoding, which we select the largest prob
            idx_next = torch.argmax(probs, dim=-1, keepdim=True)  # (B, 1)
        idx = torch.concat([idx, idx_next], dim=1)  # (B, context_size + 1)

    return idx


def generate_with_kv_cache(
    model,
    idx, 
    max_new_tokens, 
    context_size, 
    top_k=None, 
    temperature=None,
):
    """
    Using kv cache for efficient decoding
    """
    # initialize of the kv cache based on the model configuration
    num_trans_blocks = len(model.blocks)
    kv_caches = [None] * num_trans_blocks

    # prefill stage
    idx_cond = idx[:, -context_size:]  # (B, context_size)
    with torch.no_grad():
        logits, kv_caches = model(idx_cond, use_cache=True, kv_caches=kv_caches)

    all_tokens = idx.clone()
    new_tokens = []
    
    for _ in range(max_new_tokens):
        logits = logits[:, -1, :]  # (B, vocab_size)

        # the following is the same as the generate function
        # top-k sampling
        if top_k:
            top_k_logits, _ = torch.topk(logits, top_k)  # B, top_k
            # extract the minimum value of top-k logits for each sample row
            min_val = top_k_logits[:, -1]  # B,
            # filter value less then min_val, set to -inf
            logits = torch.where(logits < min_val, float('-inf'), logits)

        if temperature:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)  # (B, vocab_size)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
        else:
            probs = torch.softmax(logits, dim=-1)  # (B, vocab_size)
            # this is the greedy decoding, which we select the largest prob
            idx_next = torch.argmax(probs, dim=-1, keepdim=True)  # (B, 1)
        
        all_tokens = torch.concat([all_tokens, idx_next], dim=1)
        new_tokens.append(idx_next)
        
        # decode the next token
        with torch.no_grad():
            logits, kv_caches = model(idx_next, use_cache=True, kv_caches=kv_caches)

    return all_tokens, torch.concat(new_tokens, dim=1)

def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    # position embedding encodes the context window length
    context_size = model.pos_emb.weight.shape[0]
    idx = torch.tensor(tokenizer.encode(start_context)).unsqueeze(0).to(device)
    idx = generate(model, idx, max_new_tokens=50, context_size=context_size, top_k=25, temperature=1.4)
    decoded_text = tokenizer.decode(idx.squeeze(0).tolist())
    print(decoded_text.replace("\n", " "))
    model.train()


if __name__ == "__main__":
    tokenizer = tiktoken.get_encoding("gpt2")

    txt = "Every effort moves you"

    idx = torch.tensor(tokenizer.encode(txt)).unsqueeze(0)  # (B, T)

    model = GPT(GPT_CONFIG_124M)
    model.eval()

    # idx = generate(model, idx, max_new_tokens=10, context_size=4)
    # print(tokenizer.decode(idx[0].tolist()))

    all_tokens, new_tokens = generate_with_kv_cache(model, idx, max_new_tokens=20, context_size=4)
    print(all_tokens)
    print(new_tokens)
