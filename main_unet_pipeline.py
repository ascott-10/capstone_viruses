################ Import Libraries ################
import os
import glob
import torch
import pandas as pd
import matplotlib.pyplot as plt

from code.config import *

from code.unet_training_setup import (
    combine_dfs,
    load_images_from_dataframe, 
    train_split, 
    create_segmentation_tensor_dataset, 
    create_dataloader
)

from code.unet_training import (
    setup_training, 
    train_one_epoch, 
    validate_one_epoch, 
    save_best_model, 
    plot_loss
)

from code.unet_model import UNet  # <--- corrected

################ Setup ################

# Step 1: Check if any raw_and_segment CSV already exists
file_list = glob.glob(os.path.join(INPUT_DIR, "raw_and_segment_*.csv"))

if not file_list:
    print("[INFO] No raw_and_segment CSV found. Building one now...")
    # No CSV exists, need to build it
    all_files_df = combine_dfs(RAW_IMS_MUT, RAW_IMS_WT, SEGMENTED_MASKS, SAVE_DIR)

    # After building, refresh file list
    file_list = glob.glob(os.path.join(INPUT_DIR, "raw_and_segment_*.csv"))
    if not file_list:
        raise FileNotFoundError("❌ Could not create raw_and_segment CSV. Check folder paths.")

# Step 2: Load the most recent CSV
most_recent_file = max(file_list, key=os.path.getmtime)
print(f"[INFO] Loading metadata from {most_recent_file}")
df = pd.read_csv(most_recent_file)


################## NEW FILTERING SECTION ##################

# Randomly sample 25 mutants and 25 wildtypes
mutants_df = df[df['class'] == 'mutant'].sample(n=25, random_state=42)
wildtypes_df = df[df['class'] == 'wildtype'].sample(n=25, random_state=42)

# Combine
df = pd.concat([mutants_df, wildtypes_df]).reset_index(drop=True)

print(f"[INFO] Using {len(df)} images total (25 mutants + 25 wildtypes)")
print(df.head())


# Load images and masks
X_raw, X_seg, labels = load_images_from_dataframe(df, 'file_path', 'segmented_file_path', 'class')
X_raw_train, X_raw_val, X_raw_test, X_seg_train, X_seg_val, X_seg_test = train_split(X_raw, X_seg)

# Build datasets and dataloaders
train_loader = create_dataloader(create_segmentation_tensor_dataset(X_raw_train, X_seg_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader = create_dataloader(create_segmentation_tensor_dataset(X_raw_val, X_seg_val), batch_size=BATCH_SIZE, shuffle=False)
test_loader = create_dataloader(create_segmentation_tensor_dataset(X_raw_test, X_seg_test), batch_size=BATCH_SIZE, shuffle=False)

# Build model
model = UNet(input_channels=1, output_channels=1)
model = model.to(DEVICE)

# Setup loss and optimizer
loss_fn, optimizer = setup_training(model, learning_rate=LEARNING_RATE)

################ Train ################

train_losses = []
val_losses = []
best_val_loss = float('inf')

for epoch in range(NUM_EPOCHS):
    train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, DEVICE)
    val_loss = validate_one_epoch(model, val_loader, loss_fn, DEVICE)

    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] — Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    best_val_loss = save_best_model(model, SAVE_DIR, epoch, val_loss, best_val_loss)

################ Plot Loss ################

plot_loss(train_losses, val_losses, save_dir=SAVE_DIR)
