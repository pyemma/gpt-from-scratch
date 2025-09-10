import torch

import pytest

from model.attentions import MultiHeadCausalAttention


class TestSelfAttention:
    
    @pytest.fixture
    def attn(self):
        return MultiHeadCausalAttention(d_in=128, d_out=128, max_seq_len=1024, num_heads=8)

    def test_forward(self, attn):
        x = torch.randn(32, 512, 128)
        out = attn(x)
        assert out.shape == (32, 512, 128)