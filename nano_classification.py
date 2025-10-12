import torch
import torch.nn as nn
import torch.optim as optim

from model.nano import GPT
from data.spam_dataloader import create_dataloader


def setup_model():
    model = GPT.from_pretrained("gpt2")
    # freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    # update the head to binary classification
    model.lm_head = nn.Linear(model.config.n_embd, 2)

    # also set the last transformer block and layer norm to be trainable
    for param in model.transformer.h[-1].parameters():
        param.requires_grad = True

    for param in model.transformer.ln_f.parameters():
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
            logits, _ = model(input_batch)
        
        preds = torch.argmax(logits[:, -1, :], dim=-1)  # B, 1

        total_samples += preds.shape[0]
        correct_predictions += (preds == label).sum().item()

    return correct_predictions / total_samples


def calc_loss_dataloader(dataloader, model, device, num_batches=None):
    total_loss = 0.0
    num_batches = num_batches or len(dataloader)
    for i, (input_batch, label) in enumerate(dataloader):
        if i >= num_batches:
            break
        input_batch = input_batch.to(device)
        label = label.to(device)
        _, loss = model(input_batch, targets=label)
        total_loss += loss.item()
    return total_loss / num_batches


def eval_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_dataloader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_dataloader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def train(num_epochs=4):
    device = "mps"
    model = setup_model()
    model.to(device)
    global_step = 0
    train_loader, val_loader, test_loader = create_dataloader("./sms-spam-collection/SMSSpamCollection", batch_size=32, num_workers=0)
    optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=5e-5, betas=(0.9, 0.95), device_type=device)

    for epoch in range(num_epochs):
        model.train()

        for i, (input_batch, label) in enumerate(train_loader):
            optimizer.zero_grad()

            input_batch = input_batch.to(device)
            label = label.to(device)
            _, loss = model(input_batch, targets=label)
            loss.backward()
            optimizer.step()

            global_step += 1

            if global_step % 10 == 0:
                train_loss, val_loss = eval_model(model, train_loader, val_loader, device, eval_iter=4)
                print(f"Epoch {epoch}, Step {global_step}, Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}")
        
        train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=4)
        val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=4)
        print(f"Train accuracy: {train_accuracy:.4f}, Val accuracy: {val_accuracy:.4f}")

    test_accuracy = calc_accuracy_loader(test_loader, model, device, num_batches=4)
    print(f"Test accuracy: {test_accuracy:.4f}")


if __name__ == "__main__":
    train(num_epochs=10)