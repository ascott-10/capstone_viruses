import torch
import torch.nn as nn
import torch.optim as optim
import sys, os


from code.config import *


import matplotlib.pyplot as plt


def setup_training(model, learning_rate=1e-4):
    """
    Sets up the loss function and optimizer for training.
    """
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    return loss_fn, optimizer

def train_one_epoch(model, dataloader, loss_fn, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, masks, _ in dataloader:
        
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = (torch.sigmoid(outputs) > 0.5).float()
        correct += (preds == masks).sum().item()
        total += masks.numel()

    avg_loss = running_loss / len(dataloader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy



def validate_one_epoch(model, dataloader, loss_fn, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, masks, _ in dataloader:
            
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, masks)
            running_loss += loss.item()

            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == masks).sum().item()
            total += masks.numel()

    avg_loss = running_loss / len(dataloader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy


def save_best_model(model, save_dir, epoch, val_loss, best_val_loss):
    """
    Saves the model if validation loss improves.
    """
    if val_loss < best_val_loss:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"best_unet_epoch{epoch+1}_val{val_loss:.4f}.pt")
        torch.save(model.state_dict(), save_path)
        print(f"Best model saved at Epoch {epoch+1} with val_loss {val_loss:.4f}")
        return val_loss
    else:
        return best_val_loss

def plot_loss(train_losses, val_losses, train_accuracies, val_accuracies, save_dir=None):
    """
    Plots training/validation loss and accuracy curves.
    """
    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss", color=TRAIN_COLOR)
    plt.plot(val_losses, label="Validation Loss", color=VAL_COLOR)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.grid(True)
    plt.legend()

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label="Train Accuracy", color=TRAIN_COLOR)
    plt.plot(val_accuracies, label="Validation Accuracy", color=VAL_COLOR)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training vs Validation Accuracy")
    plt.grid(True)
    plt.legend()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "loss_accuracy_curves.png")
        plt.savefig(save_path)
        print(f"Loss and accuracy curves saved at {save_path}")

    plt.show(block=False)
    plt.pause(2)
    plt.close()



import os
import matplotlib.pyplot as plt
import torch
import pandas as pd

import os
import matplotlib.pyplot as plt
import torch
import pandas as pd

def plot_test_predictions(test_loader, model, SAVE_DIR, device, image_paths=None, mask_paths=None, one_image=False, max_examples=5):
    model.eval()
    os.makedirs(os.path.join(SAVE_DIR, "test_predictions"), exist_ok=True)

    num_to_plot = min(max_examples, len(test_loader.dataset))
    count = 0

    all_images = []
    all_masks = []
    all_preds = []
    log_entries = []

    raw_paths = image_paths if image_paths is not None else [None] * len(test_loader.dataset)
    seg_paths = mask_paths if mask_paths is not None else [None] * len(test_loader.dataset)

    with torch.no_grad():
        for images, masks, _ in test_loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)

            if outputs.shape != masks.shape:
                outputs = torch.nn.functional.interpolate(outputs, size=masks.shape[2:], mode='bilinear', align_corners=False)

            preds = torch.sigmoid(outputs)
            preds = (preds > 0.5).float()

            images = images.cpu()
            masks = masks.cpu()
            preds = preds.cpu()

            for i in range(images.shape[0]):
                raw_path = raw_paths[count]
                seg_path = seg_paths[count]

                if one_image:
                    all_images.append(images[i, 0])
                    all_masks.append(masks[i, 0])
                    all_preds.append(preds[i, 0])

                    # Still log it for combined plot
                    log_entries.append({
                        "saved_prediction": f"test_combined.png",
                        "raw_image_path": raw_path,
                        "ground_truth_mask_path": seg_path
                    })
                else:
                    fig, axs = plt.subplots(1, 3, figsize=(12,4))

                    axs[0].imshow(images[i,0], cmap='gray')
                    axs[0].set_title("Raw Image")
                    axs[0].axis('off')

                    axs[1].imshow(masks[i,0], cmap='gray')
                    axs[1].set_title("Ground Truth Segmented Mask")
                    axs[1].axis('off')

                    axs[2].imshow(preds[i,0], cmap='gray')
                    axs[2].set_title("Predicted Mask")
                    axs[2].axis('off')

                    plt.tight_layout()
                    save_name = f"test_{count}.png"
                    save_path = os.path.join(SAVE_DIR, "test_predictions", save_name)
                    plt.savefig(save_path)
                    plt.show()
                    plt.close()

                    log_entries.append({
                        "saved_prediction": save_name,
                        "raw_image_path": raw_path,
                        "ground_truth_mask_path": seg_path
                    })

                    print(f"Saved prediction plot to {save_path}")
                    count += 1

                if count >= num_to_plot:
                    break
            if count >= num_to_plot:
                break

    if one_image:
        total_examples = len(all_images)
        fig, axs = plt.subplots(total_examples, 3, figsize=(12, 4 * total_examples))

        if total_examples == 1:
            axs = [axs]

        for idx in range(total_examples):
            axs[idx][0].imshow(all_images[idx], cmap='gray')
            axs[idx][0].set_title("Raw Image")
            axs[idx][0].axis('off')

            axs[idx][1].imshow(all_masks[idx], cmap='gray')
            axs[idx][1].set_title("Ground Truth Segmented Mask")
            axs[idx][1].axis('off')

            axs[idx][2].imshow(all_preds[idx], cmap='gray')
            axs[idx][2].set_title("Predicted Mask")
            axs[idx][2].axis('off')

        plt.tight_layout()
        save_path = os.path.join(SAVE_DIR, "test_predictions", "test_combined.png")
        plt.savefig(save_path)
        plt.show()
        plt.close()

        print(f"Saved combined prediction plot to {save_path}")

    #Always save log
    if log_entries:
        df = pd.DataFrame(log_entries)
        csv_path = os.path.join(SAVE_DIR, "test_predictions", "prediction_image_map.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved CSV mapping to {csv_path}")


