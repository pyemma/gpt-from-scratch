import torch
import torch.nn as nn


class GELU(nn.Module):
    """
    A better activation function than ReLU

    This is an approximation of the cumulative distribution function
    of the standard Gaussian distribution
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (
            1 + torch.tanh(
                torch.sqrt(torch.tensor(2.0 / torch.pi)) * (
                    x + 0.044715 * torch.pow(x, 3))))


class LayerNorm(nn.Module):
    
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        # normalize each context embedding to be zero mean and unit variance
        # for faster convergence
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)  # keep the original dimension
        var = x.var(dim=-1, keepdim=True, correction=0)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return norm_x * self.scale + self.shift