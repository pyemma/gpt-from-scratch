import torch
import torch.nn as nn


class MultiHeadCausalAttention(nn.Module):
    """
    A basic implementation of multi-head causal self-attention
    """
    def __init__(self, d_in, d_out, max_seq_len, num_heads, drop_out=0.1,bias=False):
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
        self.attn_dropout = nn.Dropout(p=0.1)

        # head dimension
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

    def forward(self, x):
        """
        x: (B, T, d_in)
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

        attn_scores = torch.matmul(q, k.transpose(-2, -1))  # B, num_heads, T, T
        # the reason that we need the scale here is to handle the large hidden dimension usually
        # used in GPT, otherwise the backward gradient will be too small
        attn_scores = attn_scores / self.scale

        # apply the causal mask, we need to apply the mask to the original attention scores
        # first, otherwise the softmax is not normalized
        # add a slice to handle the case where the input sequence is shorter than the max_seq_len
        attn_scores = attn_scores.masked_fill(self.mask[:, :, :T, :T] == 0 ,float('-inf'))

        attn_weights = torch.softmax(attn_scores, dim=-1)  # B, num_heads, T, T

        # apply the dropout
        attn_weights = self.attn_dropout(attn_weights)

        # this is essentially using the attention scores to weight the value vectors
        # to compute the new context vector for next layer
        attn_output = torch.matmul(attn_weights, v)  # B, num_heads, T, head_dim

        # reshape attention output back to (B, T, d_out)
        attn_output = attn_output.contiguous().view(B, T, self.num_heads * self.head_dim)

        # transpose to make the dimension order to be (B, T, d_out)
        attn_output = self.o_w(attn_output)  # B, T, d_out
        return attn_output