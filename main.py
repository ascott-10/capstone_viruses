
################ Import Libraries ################
import numpy as np
import pandas as pd

import os
import sys
sys.path.append("..")
sys.path.append('./code')
import pathlib
from pathlib import Path

import cv2
import PIL
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


import os
import pandas as pd
import pathlib
from pathlib import Path

import torch
from torchvision import models
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import train_test_split

import paramiko #For remote GPU support
from datetime import datetime

print('Imported libraries') 

######## Custom ########

from code.config import *
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

from code.setup_classifier import load_segmented_ims, transform_data, create_tensor_dataset, create_dataloader, create_and_save_new_df
from code.train_classifier import load_classifier, train_model

from code.customs_stats import load_resnet_weights, make_test_data, make_predictions, display_stats

from code.spike_morpohology import ground_truth_morph, plot_spike_area, calculate_spike_stats
from code.morphology import (
    compare_methods, compare_methods_plotting, compare_methods_stats,
    make_morphology_df, compare_classes_stats, compare_classes_plotting
)

from code.compare_segmentation_methods import compare_load_segmented_ims, compare_combine_dfs, build_compare_df_from_auto_manual, compare_methods_stats, compare_methods_plotting, plot_spike_vs_body_area, export_full_morphology_stats

################ Full Data Setup ################

"""This will be used for both the UNet (segmentation) and ResNet (classification)"""

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
_, X_test_df = train_test_split(
    final_df,
    test_size=0.2,
    random_state=42,
    stratify=final_df['class']
)
X_test_df = X_test_df.reset_index(drop=True)
mask_paths_test = X_test_df['segmented_file_path'].tolist()
# Build datasets
train_dataset = create_segmentation_tensor_dataset(X_raw_train, X_seg_train, y_train)
val_dataset   = create_segmentation_tensor_dataset(X_raw_val, X_seg_val, y_val)
test_dataset  = create_segmentation_tensor_dataset(X_raw_test, X_seg_test, y_test)

# Build dataloaders
train_loader = create_dataloader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = create_dataloader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader  = create_dataloader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

################ UNet ################

"""Use UNet to make predictions of segmentation"""

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

loss_fn, optimizer = setup_training(model, learning_rate=LEARNING_RATE)

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
plot_test_predictions(test_loader, model, SAVE_DIR, DEVICE, image_paths=X_test_df.iloc[:, 0].tolist(), mask_paths=mask_paths_test)


################ ResNet ################
"""Use ResNet to make predictions of classification"""
from code.setup_classifier import load_segmented_ims, transform_data, create_tensor_dataset, create_dataloader, create_and_save_new_df
from code.train_classifier import load_classifier, train_model
#Training pipeline
if NEW_CLASSIFY == True:

    model = load_classifier(DEVICE, num_classes=2)
    model.to(DEVICE)
    train_model(model, DEVICE, train_loader,val_loader, SAVE_DIR)

    print('model trained')
else:
    pre_trained_model = models.resnet18(pretrained=False)
    model = load_resnet_weights(pre_trained_model, SAVE_DIR, DEVICE, num_classes=2)

    print('model trained')

X_test_df_preds = make_predictions(model, DEVICE, X_test_df, test_loader, save_cm=True, save_dir=SAVE_DIR)
display_stats(X_test_df_preds)

############# Get Morphology statistics ###############

#### Morphology ###
# Morphology extraction
df_ground_truth_morph = ground_truth_morph(final_df, segmented_path_label='segmented_file_path', class_label='class')
print(df_ground_truth_morph)

# Regression and spike area statistics
regression_by_class = calculate_spike_stats(df_ground_truth_morph, plot_yes=True)

# Class-to-class comparison
results = compare_classes_stats(
    df_ground_truth_morph,
    plot_yes=True,
    save_dir=SAVE_DIR
)

#############Comparing Manual vs Automatic Segmentation Methods #############
# Combine raw and segmented file info
all_df = combine_dfs(convert_df, RAW_IMS_MUT, RAW_IMS_WT, SEGMENTED_MASKS_MUT, SEGMENTED_MASKS_WT, SAVE_DIR)
print(all_df)

# Optional subsampling
final_df = subsample_test(final_df=all_df, SUBSAMPLE=SUBSAMPLE) if SUBSAMPLE is not None else all_df
print(final_df)

# Compare manual vs automatic segmentations
all_df = compare_combine_dfs(convert_df, SEGMENTED_MASKS_WT, SEGMENTED_MASKS_MUT,
                              AUTO_SEGMENTED_MASKS_WT, AUTO_SEGMENTED_MASKS_MUT, SAVE_DIR)

# Build dataframe with morphological measurements
df_compare = build_compare_df_from_auto_manual(all_df)

# Save comparison statistics and percent error plot
df_wide, stat, p, summary = compare_methods_stats(df_compare, SAVE_DIR, plot_yes=True)
df_wide.to_csv(os.path.join(SAVE_DIR, "comparison_results.csv"), index=False)

# Plot method-by-class comparisons
compare_methods_plotting(df_compare)

# Scatterplot + Pearson r (linear + log scale), saved automatically
plot_spike_vs_body_area(df_compare, SAVE_DIR)

# Save full morphology table with spike/body/perimeter stats
export_full_morphology_stats(df_compare, SAVE_DIR)
