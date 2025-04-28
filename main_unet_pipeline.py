################ Import Libraries ################
import numpy as np
import pandas as pd

import pathlib
from pathlib import Path
import glob
import os

from datetime import datetime
print('Imported libraries')

import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import cv2

################ Custom .py files ################

from code.unet_training_setup import load_images_from_dataframe, train_split, create_segmentation_tensor_dataset, create_dataloader
from code.unet_training import (
    setup_training,
    train_one_epoch,
    validate_one_epoch,
    save_best_model,
    plot_loss
)
from code.unet_model import build_unet_functional, unet_forward

#####################################################
save_dir = '/home/ascott10/documents/projects/capstone_viruses/results'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#####################################################

# Load latest raw + segmented data
input_dir = '/home/ascott10/documents/projects/capstone_viruses/data'
file_list = glob.glob(os.path.join(input_dir, "raw_and_segment_*.csv"))
if not file_list:
    raise FileNotFoundError("No raw_and_segment_*.csv files found.")
most_recent_file = max(file_list, key=os.path.getmtime)
raw_and_segmented_df = pd.read_csv(most_recent_file)

# Load images and masks
X_raw, X_seg = load_images_from_dataframe(raw_and_segmented_df, 'file_path', 'segmented_file_path', label_col=None)
X_raw_train, X_raw_val, X_raw_test, X_seg_train, X_seg_val, X_seg_test = train_split(X_raw, X_seg)

# Build datasets and loaders
train_dataset = create_segmentation_tensor_dataset(X_raw_train, X_seg_train)
val_dataset = create_segmentation_tensor_dataset(X_raw_val, X_seg_val)
test_dataset = create_segmentation_tensor_dataset(X_raw_test, X_seg_test)

train_loader = create_dataloader(train_dataset, batch_size=16, shuffle=True)
val_loader = create_dataloader(val_dataset, batch_size=16, shuffle=False)
test_loader = create_dataloader(test_dataset, batch_size=16, shuffle=False)

# Build model
model = build_unet_functional(input_channels=1, output_channels=1)
model = model.to(device)

# Setup optimizer and loss
loss_fn, optimizer = setup_training(model)

########################
# TRAINING SCRIPT
########################
best_val_loss = float('inf')
train_losses = []
val_losses = []
num_epochs = 25

for epoch in range(num_epochs):
    train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
    val_loss = validate_one_epoch(model, val_loader, loss_fn, device)

    print(f"Epoch [{epoch+1}/{num_epochs}] — Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Save losses
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    # Save best model
    best_val_loss = save_best_model(model, save_dir, epoch, val_loss, best_val_loss)

# Plot loss curve after training
plot_loss(train_losses, val_losses, save_dir)
