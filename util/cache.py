import torch

class KVCache:
    """
    A dedicated class to store the kv cache and work in hands with GPT
    """

    def __init__(self, batch_size, num_heads, seq_len, head_dim, num_layers):
        # each layer has one kv cache, 2 is k & v
        self.kv_shape = (num_layers, 2, batch_size, num_heads, seq_len, head_dim)
        self.kv_cache = None
        self.pos = 0  # this records the position of the new token in the kv cache
    
    def reset(self):
        self.pos = 0

    def get_pos(self):
        return self.pos

    def prefill(self, other):
        """
        Used for prefill stage, copy the kv cache from another instance,
        optionally expand the batch dimension.

        This is used when we prefill with batch 1 prefill and then want to
        generate multiple samples in parallel from there.
        """
        assert self.kv_cache is None, "cannot prefill a non-empty kv cache"
        assert other.kv_cache is not None, "cannot prefill from an empty kv cache"

        # shape validation
        for idx, (dim1, dim2) in enumerate(zip(self.kv_shape, other.kv_shape)):
            if idx in (0, 1, 3, 5):
                # the number of layers, kv, num_heads and head_dim should be the same
                assert dim1 == dim2, f"shape mismatch {idx}: {dim1} != {dim2}"
            elif idx == 2:
                # the batch size could be expanded
                assert dim1 == dim2 or dim2 == 1, f"bath size mismatch: {dim1} != {dim2}"
            elif idx == 4:
                # self length must be longer
                assert dim1 >= dim2, f"self length must be longer: {dim1} < {dim2}"
        
        # initialize the kv cache
        dtype, device = other.kv_cache.dtype, other.kv_cache.device
        self.kv_cache = torch.empty(self.kv_shape, dtype=dtype, device=device)
        # copy data
        self.kv_cache[:, :, :, :, :other.pos, :] = other.kv_cache
        # update pos
        self.pos = other.pos

    def insert_kv(self, layer_idx, k, v):
        if self.kv_cache is None:
            self.kv_cache = torch.empty(self.kv_shape, dtype=k.dtype, device=k.device)
        
        # insert the new kv and return the full cache so far
        B, H, T_add, D = k.size()
        t0, t1 = self.pos, self.pos + T_add  # [start, end)
        # increase the length of the kv cache
        if t1 > self.kv_cache.shape[-2]:  # or 4, which is seq_len dim
            t_needed = t1 + 1024  # buffer size
            t_needed = (t_needed + 1023) & ~1023  # round up to nearest multiple of 1024
            current_shape = list(self.kv_cache.shape)
            current_shape[-2] = t_needed  #  update the seq_len dimension
            self.kv_cache.resize_(current_shape)
        # insert k, v
        self.kv_cache[layer_idx, 0, :, :, t0:t1, :] = k
        self.kv_cache[layer_idx, 1, :, :, t0:t1, :] = v
        key_view = self.kv_cache[layer_idx, 0, :, :, :t1, :]  # stop at t1 as it could be reshaped to longer
        value_view = self.kv_cache[layer_idx, 1, :, :, :t1, :]
        if layer_idx == self.kv_cache.size(0) - 1:  # reach to the last layer, pos could be advanced
            self.pos = t1
        return key_view, value_view