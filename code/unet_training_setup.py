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

from code.config import *

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

    #Get all file paths for raw images and manually segmented masks
    image_labels_muts, image_filepaths_muts, image_ids_muts = load_segmented_ims(RAW_IMS_MUT)
    image_labels_wts, image_filepaths_wts, image_ids_wts = load_segmented_ims(RAW_IMS_WT)
    segmented_image_labels_muts, segmented_image_filepaths_muts, segmented_image_ids_muts = load_segmented_ims(SEGMENTED_MASKS_MUT)
    segmented_image_labels_wts, segmented_image_filepaths_wts, segmented_image_ids_wts = load_segmented_ims(SEGMENTED_MASKS_WT)


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

    convert_df['Modified Name'] = convert_df['Modified Name'].astype(str).str.replace(r'_corrected.*', '', regex=True)
    all_segmented_files['im_id'] = all_segmented_files['im_id'].astype(str).str.replace(r'_corrected.*', '', regex=True)


    segmented_combined = all_segmented_files.merge(convert_df, left_on='im_id', right_on = 'Modified Name')
    merged_df = all_raw_files.merge(segmented_combined, left_on='im_id', right_on='File_name')

    final_df = pd.DataFrame({
        'im_id': merged_df['im_id_x'],
        'file_path': merged_df['file_path'],
        'segmented_file_path': merged_df['segmented_file_path'],
        'class': merged_df['class_x']
    })

    return final_df

def subsample_test(final_df, SUBSAMPLE, class_1_label = 'mutant', class_2_label = 'wildtype'):

    class_1_label = class_1_label
    class_2_label = class_2_label
    mutants_df = final_df[final_df['class'] == 'mutant']
    wildtypes_df = final_df[final_df['class'] == 'wildtype']

    n_mutants = min(SUBSAMPLE, len(mutants_df))
    n_wildtypes = min(SUBSAMPLE, len(wildtypes_df))

    mutants_sample = mutants_df.sample(n=n_mutants, random_state=42)
    wildtypes_sample = wildtypes_df.sample(n=n_wildtypes, random_state=42)

    df = pd.concat([mutants_sample, wildtypes_sample]).reset_index(drop=True)

    print(f" Using {len(df)} images total ({n_mutants} mutants + {n_wildtypes} wildtypes)")

    return df



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

from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import TensorDataset

from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import TensorDataset

def create_segmentation_tensor_dataset(X_raw, X_mask, labels=None, return_label_mapping=False):
    """Combine raw images, masks, and optionally labels into a TensorDataset.
    
    Args:
        X_raw: Tensor of raw images
        X_mask: Tensor of masks
        labels: List/Tensor of class labels (optional)
        return_label_mapping: bool, if True returns label mapping dict
        
    Returns:
        TensorDataset of (image, mask[, label])
        Optionally also returns label mapping if requested.
    """
    label_mapping = None

    if labels is not None:
        if not isinstance(labels, torch.Tensor):
            if isinstance(labels[0], str):
                encoder = LabelEncoder()
                labels = encoder.fit_transform(labels)
                label_mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
            labels = torch.tensor(labels, dtype=torch.long)
        dataset = TensorDataset(X_raw, X_mask, labels)
    else:
        dataset = TensorDataset(X_raw, X_mask)
    
    if return_label_mapping:
        return dataset, label_mapping
    else:
        return dataset




def create_dataloader(dataset, batch_size=32, shuffle=True):
    """Create a DataLoader from a TensorDataset."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
