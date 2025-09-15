import tiktoken
import torch
import torch.optim as optim

from data.dataloader import create_dataloader
from model.gpt import GPT, generate_and_print_sample


GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12, 
    "dropout": 0.1,
    "qkv_bias": False
}

file_path = "the-verdict.txt"
with open(file_path, "r") as file:
    text_data = file.read()

train_ratio = 0.9
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]

train_loader = create_dataloader(
    train_data,  
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"], 
    batch_size=2
)
val_loader = create_dataloader(
    val_data,  
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"], 
    batch_size=2
)

def calc_loss_batch(input_batch, target_batch, model, device):
    """
    Calculate the loss for a batch of data

    Args:
        input_batch: (B, T)
        target_batch: (B, T)
    """
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)

    logits = model(input_batch)
    # the torch cross entropy expects the shape to be (batch, C) where C is the number
    # of classes, thus we need to flatten the B, T dimension to B*T
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten(),
    )
    return loss

def calc_loss_dataloader(dataloader, model, device, num_batches=None):
    total_loss = 0.0
    num_batches = min(num_batches, len(dataloader)) if num_batches else len(dataloader)

    for i, (input_batch, target_batch) in enumerate(dataloader):
        if i >= num_batches:
            break
        loss = calc_loss_batch(input_batch, target_batch, model, device)
        total_loss += loss.item()

    return total_loss / num_batches


def eval_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_dataloader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_dataloader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            # clear the gradients
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()

            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = eval_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Epoch {epoch}, Step {global_step}, Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}")

        generate_and_print_sample(model, tiktoken.get_encoding("gpt2"), device, "Every effort moves you")
    
    return train_losses, val_losses, track_tokens_seen

torch.manual_seed(123)
model = GPT(GPT_CONFIG_124M)
device = "cpu"
model.to(device)

optimizer = optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)

train_losses, val_losses, track_tokens_seen = train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs=10, eval_freq=5, eval_iter=5)
