import tiktoken

import torch

from model.nano import GPT
from util.cache import KVCache


model = GPT.from_pretrained("gpt2")

tokenizer = tiktoken.get_encoding("gpt2")
device = "cpu"

tokens = tokenizer.encode("Every effort moves you")

# use kv cache for inference
kv_cache_prefill = KVCache(batch_size=1, num_heads=12, seq_len=len(tokens), head_dim=768 // 12, num_layers=12)
idx = torch.tensor(tokens).unsqueeze(0).to(device)
with torch.no_grad():
    logits, _ = model(idx, kv_cache=kv_cache_prefill)

kv_cache_decode = KVCache(batch_size=2, num_heads=12, seq_len=1024, head_dim=768 // 12, num_layers=12)
kv_cache_decode.prefill(kv_cache_prefill)

output_tokens = torch.tensor(idx).repeat(2, 1)  # expand the batch dimension
logits = logits.repeat(2, 1, 1)

for _ in range(25):
    logits = logits[:, -1, :] / 0.1  # B, vocab_size

    probs = torch.softmax(logits, dim=-1)
    id_next = torch.multinomial(probs, num_samples=1)
    output_tokens = torch.concat([output_tokens, id_next], dim=1)
    with torch.no_grad():
        logits, _ = model(id_next, kv_cache=kv_cache_decode)


# token_ids = model.generate(
#         idx=torch.tensor(tokenizer.encode("Every effort moves you")).unsqueeze(0).to(device),
#         max_new_tokens=25,
#         top_k=50,
#         temperature=1.5
#     )
print("Output text:\n", tokenizer.decode(output_tokens[0].tolist()))
print("Output text:\n", tokenizer.decode(output_tokens[1].tolist()))