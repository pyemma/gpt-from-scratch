import torch

import pytest

from model.attentions import MultiHeadCausalAttention


class TestSelfAttention:
    
    @pytest.fixture
    def attn(self):
        return MultiHeadCausalAttention(d_in=128, d_out=128, max_seq_len=1024, num_heads=8)

    def test_forward(self, attn):
        x = torch.randn(32, 512, 128)
        out, _ = attn(x)
        assert out.shape == (32, 512, 128)

    def test_forward_with_cache(self, attn):
        x = torch.randn(32, 1, 128)
        kv_cache = (torch.randn(32, 8, 12, 16), torch.randn(32, 8, 12, 16))
        out, kv_cache = attn(x, use_cache=True, kv_cache=kv_cache)

        assert out.shape == (32, 1, 128)
        assert kv_cache[0].shape == (32, 8, 13, 16)