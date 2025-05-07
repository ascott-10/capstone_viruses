################ Import Libraries ################
import os
import glob
import torch
import pandas as pd
import matplotlib.pyplot as plt

from config import *
from code.unet_training_setup import (load_segmented_ims, subsample_test,
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
    plot_loss, plot_test_predictions
)
from code.unet_model import UNet

################ Setup ################

#Get all file paths and labels into one dataframe
all_df = combine_dfs(convert_df, RAW_IMS_MUT, RAW_IMS_WT, SEGMENTED_MASKS_MUT, SEGMENTED_MASKS_WT, SAVE_DIR)
print(all_df)
#For testing, only using a portion of the images
if SUBSAMPLE is not None:
    final_df = subsample_test(final_df=all_df, SUBSAMPLE=SUBSAMPLE)
else:
    final_df = all_df

print(final_df)

#Load raw images, segmented images, masks, labels 
X_raw, X_seg, labels = load_images_from_dataframe(final_df, 'file_path', 'segmented_file_path', 'class')
X_raw_train, X_raw_val, X_raw_test,X_seg_train, X_seg_val, X_seg_test, y_train, y_val, y_test = train_split(X_raw, X_seg, labels)

# Build datasets
train_dataset = create_segmentation_tensor_dataset(X_raw_train, X_seg_train, y_train)
val_dataset   = create_segmentation_tensor_dataset(X_raw_val, X_seg_val, y_val)
test_dataset  = create_segmentation_tensor_dataset(X_raw_test, X_seg_test, y_test)

# Build dataloaders
train_loader = create_dataloader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = create_dataloader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader  = create_dataloader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


################ Build or Load Model ################

model = UNet(input_channels=1, output_channels=1)
model = model.to(DEVICE)

best_model_path = os.path.join(SAVE_DIR, "best_unet.pt")

if os.path.exists(best_model_path):
    print(f" Found existing model weights at {best_model_path}. Loading model.")
    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
    model.eval()
    skip_training = True
else:
    print("No saved model found, starting training")
    skip_training = False

# Setup loss and optimizer
loss_fn, optimizer = setup_training(model, learning_rate=LEARNING_RATE)

################ Train ################

if not skip_training:
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

else:
    print(" Skipping training, model weights already exist.")

print("Testing")

plot_test_predictions(test_loader, model, SAVE_DIR, DEVICE, one_image=True, max_examples=5)

