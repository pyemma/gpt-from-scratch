import numpy as np
import torch
import tiktoken

from gpt_download import download_and_load_gpt2

from model.gpt import GPT, generate


GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "dropout": 0.1,
    "qkv_bias": True
}

def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch: {left.shape} != {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))

def load_weights_into_gpt(gpt, params):
    # assign the embedding weights
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params["wte"])
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params["wpe"])

    for b in range(len(params["blocks"])):
        # q, k, v weights
        q_w, k_w, v_w = np.split((params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.blocks[b].attn.q_w.weight = assign(gpt.blocks[b].attn.q_w.weight, q_w.T)
        gpt.blocks[b].attn.k_w.weight = assign(gpt.blocks[b].attn.k_w.weight, k_w.T)
        gpt.blocks[b].attn.v_w.weight = assign(gpt.blocks[b].attn.v_w.weight, v_w.T)

        q_b, k_b, v_b = np.split((params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.blocks[b].attn.q_w.bias = assign(gpt.blocks[b].attn.q_w.bias, q_b)
        gpt.blocks[b].attn.k_w.bias = assign(gpt.blocks[b].attn.k_w.bias, k_b)
        gpt.blocks[b].attn.v_w.bias = assign(gpt.blocks[b].attn.v_w.bias, v_b)

        # out projection weights
        gpt.blocks[b].attn.o_w.weight = assign(
            gpt.blocks[b].attn.o_w.weight, 
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.blocks[b].attn.o_w.bias = assign(
            gpt.blocks[b].attn.o_w.bias, 
            params["blocks"][b]["attn"]["c_proj"]["b"])

        # feed forward weights
        gpt.blocks[b].ff.layers[0].weight = assign(
            gpt.blocks[b].ff.layers[0].weight, 
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.blocks[b].ff.layers[0].bias = assign(
            gpt.blocks[b].ff.layers[0].bias, 
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.blocks[b].ff.layers[2].weight = assign(
            gpt.blocks[b].ff.layers[2].weight, 
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.blocks[b].ff.layers[2].bias = assign(
            gpt.blocks[b].ff.layers[2].bias, 
            params["blocks"][b]["mlp"]["c_proj"]["b"])

        # layer norm weights
        gpt.blocks[b].norm1.scale = assign(
            gpt.blocks[b].norm1.scale, 
            params["blocks"][b]["ln_1"]["g"])
        gpt.blocks[b].norm1.shift = assign(
            gpt.blocks[b].norm1.shift, 
            params["blocks"][b]["ln_1"]["b"])
        gpt.blocks[b].norm2.scale = assign(
            gpt.blocks[b].norm2.scale, 
            params["blocks"][b]["ln_2"]["g"])
        gpt.blocks[b].norm2.shift = assign(
            gpt.blocks[b].norm2.shift, 
            params["blocks"][b]["ln_2"]["b"])

    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])

def pre_train_gpt_model():
    settings, params = download_and_load_gpt2(
        model_size="124M", models_dir="gpt2"
    )
    model = GPT(GPT_CONFIG_124M)
    model.eval()
    load_weights_into_gpt(model, params)
    return model

if __name__ == "__main__":
    model = pre_train_gpt_model()

    tokenizer = tiktoken.get_encoding("gpt2")
    device = "cpu"

    torch.manual_seed(123)
    token_ids = generate(
        model=model,
        idx=torch.tensor(tokenizer.encode("Every effort moves you")).unsqueeze(0).to(device),
        max_new_tokens=25,
        context_size=GPT_CONFIG_124M["context_length"],
        top_k=50,
        temperature=1.5
    )
    print("Output text:\n", tokenizer.decode(token_ids.squeeze(0).tolist()))

    print(model)