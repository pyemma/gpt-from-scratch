import torch
import torch.nn as nn
import torch.optim as optim

from pre_train import pre_train_gpt_model, GPT_CONFIG_124M
from data.spam_dataloader import create_dataloader


KV_CACHES = [None] * GPT_CONFIG_124M["n_layers"]

def setup_model():
    model = pre_train_gpt_model()

    # freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    # update the head to binary classification
    model.out_head = nn.Linear(GPT_CONFIG_124M["emb_dim"], 2)

    # also set the last transformer block and layer norm to be trainable
    for param in model.blocks[-1].parameters():
        param.requires_grad = True

    for param in model.final_norm.parameters():
        param.requires_grad = True
    
    return model

def calc_accuracy_loader(dataloader, model, device, num_batches=None):
    model.eval()
    correct_predictions, total_samples = 0, 0

    num_batches = num_batches or len(dataloader)

    for i, (input_batch, label) in enumerate(dataloader):
        if i >= num_batches:
            break

        input_batch = input_batch.to(device)
        label = label.to(device)
        with torch.no_grad():
            logits, _ = model(input_batch, use_cache=False, kv_caches=KV_CACHES)
        
        preds = torch.argmax(logits[:, -1, :], dim=-1)  # B, 1

        total_samples += preds.shape[0]
        correct_predictions += (preds == label).sum().item()

    return correct_predictions / total_samples

def calc_loss_batch(input_batch, label, model, device):
    input_batch = input_batch.to(device)
    label = label.to(device)
    logits, _ = model(input_batch, use_cache=False, kv_caches=KV_CACHES)
    logits = logits[:, -1, :]
    loss = torch.nn.functional.cross_entropy(logits, label)
    return loss


def calc_loss_dataloader(dataloader, model, device, num_batches=None):
    total_loss = 0.0
    num_batches = num_batches or len(dataloader)
    for i, (input_batch, label) in enumerate(dataloader):
        if i >= num_batches:
            break
        loss = calc_loss_batch(input_batch, label, model, device)
        total_loss += loss.item()
    return total_loss / num_batches


def train(num_epochs=4):
    device = "mps"
    model = setup_model()
    model.to(device)
    kv_caches = [None] * len(model.blocks)
    global_step = 0
    train_loader, val_loader, test_loader = create_dataloader("./sms-spam-collection/SMSSpamCollection", batch_size=32, num_workers=0)
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)

    for epoch in range(num_epochs):
        model.train()

        for i, (input_batch, label) in enumerate(train_loader):
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, label, model, device)
            loss.backward()
            optimizer.step()

            global_step += 1

            if global_step % 10 == 0:
                train_loss = running_loss / 100
                print(f"Epoch {epoch}, Step {global_step}, Train loss: {train_loss:.4f}")
                running_loss = 0.0
        
        train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=4)
        val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=4)
        print(f"Train accuracy: {train_accuracy:.4f}, Val accuracy: {val_accuracy:.4f}")

    test_accuracy = calc_accuracy_loader(test_loader, model, device, num_batches=4)
    print(f"Test accuracy: {test_accuracy:.4f}")


if __name__ == "__main__":
    train(num_epochs=10)