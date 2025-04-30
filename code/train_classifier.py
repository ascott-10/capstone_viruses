import numpy as np
import pandas as pd
import os
import sys
import pathlib
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import cv2
import PIL
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torch.optim.lr_scheduler import CosineAnnealingLR

# Extend path for local modules
sys.path.append("..")
sys.path.append('./code')

print(torch.__version__)

def plot_loss_and_accuracy(train_losses, val_losses, train_accuracies, val_accuracies, save_dir=None):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, color='tab:blue', label='Train Loss')
    plt.plot(epochs, val_losses, color='tab:orange', label='Validation Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, color='tab:blue', label='Train Accuracy')
    plt.plot(epochs, val_accuracies, color='tab:orange', label='Validation Accuracy')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training vs Validation Accuracy")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "loss_accuracy_curve.png")
        plt.savefig(save_path)
        print(f"Loss/accuracy plot saved at {save_path}")

    plt.show(block=False)
    plt.pause(2)
    plt.close()

def train(model, device, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=15, save_dir=None):
    model.train()

    train_losses = []
    train_accuracies = []
    val_accuracies = []
    val_losses = []  

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for _, masks, labels in train_loader:
            if masks.shape[1] == 1:
                masks = masks.repeat(1, 3, 1, 1)
            masks = masks.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(masks)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_accuracy = 100 * correct / total
        avg_train_loss = running_loss / len(train_loader)
        scheduler.step()

        model.eval()
        val_correct = 0
        val_total = 0
        val_loss_total = 0.0 

        with torch.no_grad():
            for _, val_masks, val_labels in val_loader:
                if val_masks.shape[1] == 1:
                    val_masks = val_masks.repeat(1, 3, 1, 1)
                val_masks = val_masks.to(device)
                val_labels = val_labels.to(device)

                val_outputs = model(val_masks)
                _, val_predicted = torch.max(val_outputs, 1)
                val_total += val_labels.size(0)
                val_correct += (val_predicted == val_labels).sum().item()

                loss = criterion(val_outputs, val_labels)
                val_loss_total += loss.item()

        avg_val_loss = val_loss_total / len(val_loader)
        val_accuracy = 100 * val_correct / val_total
        val_losses.append(avg_val_loss)

        model.train()

        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'  Train Loss: {avg_train_loss:.4f} | Train Accuracy: {train_accuracy:.2f}% | Validation Accuracy: {val_accuracy:.2f}%')

    print("Training complete.")
    plot_loss_and_accuracy(train_losses, val_losses, train_accuracies, val_accuracies, save_dir=save_dir)

def load_classifier(device, num_classes=2):
    model = models.resnet18(pretrained=True)
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, num_classes)
    )
    model.to(device)
    return model

def train_model(model, device, train_loader, val_loader, save_dir):
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True

    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.01,
        momentum=0.9,
        weight_decay=1e-6
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss()

    train(model, device, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=15, save_dir=save_dir)

    for param in model.parameters():
        param.requires_grad = True

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.001,
        momentum=0.9,
        weight_decay=1e-6
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)

    train(model, device, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=15, save_dir=save_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir, f"resnet_weights_{timestamp}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
