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
    "n_layers": 3,  # the original GPT-2 has 12 layers
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

    def forward(self, x):
        # use per-norm, which provides better performance than post-norm
        x1 = self.norm1(x)
        x1 = self.attn(x1)
        x1 = self.dropout(x1)
        x1 = x + x1  # residual connection


        x2 = self.norm2(x1)
        x2 = self.ff(x2)
        x2 = self.dropout(x2)
        x2 = x1 + x2  # residual connection

        return x2


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
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"])

    def forward(self, inputs):
        B, T = inputs.shape

        tok_embeddings = self.tok_emb(inputs)  # B, T, emb_dim
        pos_embeddings = self.pos_emb(torch.arange(T, device=inputs.device))  # T, emb_dim

        embeddings = tok_embeddings + pos_embeddings  # broadcast pos_embedding
        embeddings = self.drop_emb(embeddings)

        embeddings = self.blocks(embeddings)
        embeddings = self.final_norm(embeddings)

        logits = self.out_head(embeddings)  # B, T, vocab_size

        return logits


def generate(model, idx, max_new_tokens, context_size):
    """
    Decoding phase of the model
    """
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]  # (B, context_size)
        with torch.no_grad():
            logits = model(idx_cond)

        # the last hidden state is the probability of the next token
        logits = logits[:, -1, :]  # (B, vocab_size)
        probs = torch.softmax(logits, dim=-1)  # (B, vocab_size)
        # this is the greedy decoding, which we select the largest prob
        idx_next = torch.argmax(probs, dim=-1, keepdim=True)  # (B, 1)
        idx = torch.concat([idx, idx_next], dim=1)  # (B, context_size + 1)

    return idx


if __name__ == "__main__":
    tokenizer = tiktoken.get_encoding("gpt2")

    txt = "Every effort moves you"

    idx = torch.tensor(tokenizer.encode(txt)).unsqueeze(0)  # (B, T)

    model = GPT(GPT_CONFIG_124M)
    model.eval()

    idx = generate(model, idx, max_new_tokens=10, context_size=4)
    print(tokenizer.decode(idx[0].tolist()))
