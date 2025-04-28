import torch
import torch.nn as nn
import torch.optim as optim
import os
import matplotlib.pyplot as plt
from code.unet_model import unet_forward  # you need this to forward manually

def setup_training(model, learning_rate=1e-4):
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    return loss_fn, optimizer

def train_one_epoch(model, dataloader, loss_fn, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)

        outputs = unet_forward(model, images)
        loss = loss_fn(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(dataloader)

def validate_one_epoch(model, dataloader, loss_fn, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = unet_forward(model, images)
            loss = loss_fn(outputs, masks)

            running_loss += loss.item()

    return running_loss / len(dataloader)

def save_best_model(model, save_dir, epoch, val_loss, best_val_loss):
    if val_loss < best_val_loss:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"best_unet_epoch{epoch+1}_val{val_loss:.4f}.pt")
        torch.save(model.state_dict(), save_path)
        print(f"✅ Best model saved at Epoch {epoch+1} with val_loss {val_loss:.4f}")
        return val_loss
    else:
        return best_val_loss

def plot_loss(train_losses, val_losses, save_dir=None):
    plt.figure(figsize=(8,6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.grid(True)
    plt.legend()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "loss_curve.png")
        plt.savefig(save_path)
        print(f"✅ Loss curve saved at {save_path}")
    plt.show()
