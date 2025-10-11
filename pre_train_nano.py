import tiktoken

import torch

from model.nano import GPT


model = GPT.from_pretrained("gpt2")

tokenizer = tiktoken.get_encoding("gpt2")
device = "cpu"

token_ids = model.generate(
        idx=torch.tensor(tokenizer.encode("Every effort moves you")).unsqueeze(0).to(device),
        max_new_tokens=25,
        top_k=50,
        temperature=1.5
    )
print("Output text:\n", tokenizer.decode(token_ids.squeeze(0).tolist()))