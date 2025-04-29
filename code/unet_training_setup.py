import numpy as np
import pandas as pd
import os
import sys
import pathlib
from pathlib import Path
import cv2
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from datetime import datetime
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

from code.config import RAW_IMS_WT, RAW_IMS_MUT, SEGMENTED_MASKS_WT, SEGMENTED_MASKS_MUT, SAVE_DIR, INPUT_DIR, IMAGE_SIZE


################ Functions ################

import os
import pandas as pd
from pathlib import Path

import os
import pandas as pd
from pathlib import Path

def load_segmented_ims(input_path):
    """Load segmented image paths and labels based on filename patterns."""

    image_filepaths = []
    image_labels = []
    image_ids = []

    mut_labels = ['A2_MHV', 'muimage']
    wt_labels = ['MHVWT', 'wtimage']

    
            
    for files in os.listdir(input_path):
        if files.endswith('png'):
            file_path = os.path.join(input_path, files)
            image_id = Path(files).stem
            

            # Check if filename matches normally
            if any(tag in image_id for tag in mut_labels):
                label = 'mutant'
            elif any(tag in image_id for tag in wt_labels):
                label = 'wildtype'
            else:
                label = 'unknown'
            

            image_labels.append(label)
            image_filepaths.append(file_path)
            image_ids.append(image_id)

            

    return image_labels, image_filepaths, image_ids



def combine_dfs(convert_df, RAW_IMS_MUT, RAW_IMS_WT, SEGMENTED_MASKS_MUT, SEGMENTED_MASKS_WT, SAVE_DIR):
    """Combine raw images and segmented masks into one dataframe."""

    # Load raw images (no conversion needed)
    image_labels_muts, image_filepaths_muts, image_ids_muts = load_segmented_ims(RAW_IMS_MUT)
    image_labels_wts, image_filepaths_wts, image_ids_wts = load_segmented_ims(RAW_IMS_WT)

    # Load segmented masks (conversion needed)
    segmented_image_labels_muts, segmented_image_filepaths_muts, segmented_image_ids_muts = load_segmented_ims(SEGMENTED_MASKS_MUT, convert_df)

    

    segmented_image_labels_wts, segmented_image_filepaths_wts, segmented_image_ids_wts = load_segmented_ims(SEGMENTED_MASKS_WT, convert_df)

    # Build DataFrames
    all_raw_files = pd.DataFrame([
        (image_ids_muts + image_ids_wts),
        (image_filepaths_muts + image_filepaths_wts),
        (image_labels_muts + image_labels_wts)
    ], index=['im_id', 'file_path', 'class']).T

    

    all_segmented_files = pd.DataFrame([
        (segmented_image_ids_muts + segmented_image_ids_wts),
        (segmented_image_filepaths_muts + segmented_image_filepaths_wts),
        (segmented_image_labels_muts + segmented_image_labels_wts)
    ], index=['im_id', 'segmented_file_path', 'class']).T

    print(all_segmented_files)

    # Clean im_id to remove _seg type suffix
    all_segmented_files['im_id'] = all_segmented_files['im_id'].astype(str).str.replace(r'_seg.*', '', regex=True)

    # Merge raw and segmented tables
    all_files_df = all_raw_files.merge(all_segmented_files, on=['im_id', 'class'])

    print('[INFO] All files loaded.')

    # Save the merged DataFrame
    os.makedirs(SAVE_DIR, exist_ok=True)
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M")
    save_path = os.path.join(SAVE_DIR, f"raw_and_segment_{timestamp}.csv")
    all_files_df.to_csv(save_path, index=False)

    return all_files_df


def load_images_from_dataframe(df, raw_image_col, mask_col, label_col=None):
    """Load raw images and masks from a dataframe into tensors."""
    raw_images = []
    mask_images = []
    labels = []

    for idx, row in df.iterrows():
        raw_path = row[raw_image_col]
        mask_path = row[mask_col]

        raw_img = cv2.imread(raw_path, cv2.IMREAD_GRAYSCALE)
        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if raw_img is None:
            print(f"[ERROR] Raw image not found: {raw_path}")
        if mask_img is None:
            print(f"[ERROR] Mask image not found: {mask_path}")

        if raw_img is None or mask_img is None:
            continue  # Skip broken images

        raw_img = cv2.resize(raw_img, IMAGE_SIZE)
        raw_img = torch.tensor(raw_img, dtype=torch.float32) / 255.0
        raw_img = raw_img.unsqueeze(0)  # (1, H, W) grayscale

        mask_img = cv2.resize(mask_img, IMAGE_SIZE)
        mask_img = torch.tensor(mask_img, dtype=torch.float32) / 255.0
        mask_img = mask_img.unsqueeze(0)

        raw_images.append(raw_img)
        mask_images.append(mask_img)

        if label_col is not None:
            labels.append(row[label_col])

    if not raw_images:
        raise ValueError("[FATAL] No valid images loaded. Check your file paths in the CSV.")

    if label_col is not None:
        return torch.stack(raw_images), torch.stack(mask_images), labels
    else:
        return torch.stack(raw_images), torch.stack(mask_images)

def train_split(X_raw, X_segmented, y_class=None):
    """Split raw images and masks into train/val/test sets."""
    if y_class is not None:
        X_raw_trainval, X_raw_test, X_seg_trainval, X_seg_test, y_class_trainval, y_class_test = train_test_split(
            X_raw, X_segmented, y_class, test_size=0.2, random_state=42, stratify=y_class
        )
        X_raw_train, X_raw_val, X_seg_train, X_seg_val, y_class_train, y_class_val = train_test_split(
            X_raw_trainval, X_seg_trainval, y_class_trainval, test_size=0.25, random_state=42, stratify=y_class_trainval
        )
        return (X_raw_train, X_raw_val, X_raw_test,
                X_seg_train, X_seg_val, X_seg_test,
                y_class_train, y_class_val, y_class_test)
    else:
        X_raw_trainval, X_raw_test, X_seg_trainval, X_seg_test = train_test_split(
            X_raw, X_segmented, test_size=0.2, random_state=42
        )
        X_raw_train, X_raw_val, X_seg_train, X_seg_val = train_test_split(
            X_raw_trainval, X_seg_trainval, test_size=0.25, random_state=42
        )
        return (X_raw_train, X_raw_val, X_raw_test,
                X_seg_train, X_seg_val, X_seg_test)

def create_segmentation_tensor_dataset(X_raw, X_mask):
    """Combine raw images and masks into a TensorDataset."""
    return TensorDataset(X_raw, X_mask)

def create_dataloader(dataset, batch_size=32, shuffle=True):
    """Create a DataLoader from a TensorDataset."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
