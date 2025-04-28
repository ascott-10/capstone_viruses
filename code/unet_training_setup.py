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
import torch
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split

from datetime import datetime


IMAGE_SIZE = (750,750)  # Resize images to match U-Net input size

import cv2
import torch
import torchvision.transforms as transforms

IMAGE_SIZE = (750, 750)

def load_images_from_dataframe(df, raw_image_col, mask_col, label_col=None):
    """
    Load raw images and masks (and optional labels) from a dataframe into tensors.
    """

    raw_images = []
    mask_images = []
    labels = []

    for _, row in df.iterrows():
        # Raw image
        raw_img = cv2.imread(row[raw_image_col], cv2.IMREAD_GRAYSCALE)
        if raw_img is None:
            raise ValueError(f"Raw image not found: {row[raw_image_col]}")
        raw_img = cv2.resize(raw_img, IMAGE_SIZE)
        raw_img = torch.tensor(raw_img, dtype=torch.float32) / 255.0
        raw_img = raw_img.unsqueeze(0)  # (1, H, W) for grayscale

        # Mask image
        mask_img = cv2.imread(row[mask_col], cv2.IMREAD_GRAYSCALE)
        if mask_img is None:
            raise ValueError(f"Mask image not found: {row[mask_col]}")
        mask_img = cv2.resize(mask_img, IMAGE_SIZE)
        mask_img = torch.tensor(mask_img, dtype=torch.float32) / 255.0
        mask_img = mask_img.unsqueeze(0)  # (1, H, W)

        raw_images.append(raw_img)
        mask_images.append(mask_img)

        # Optional: load labels if available
        if label_col is not None:
            labels.append(row[label_col])

    if label_col is not None:
        return torch.stack(raw_images), torch.stack(mask_images), labels
    else:
        return torch.stack(raw_images), torch.stack(mask_images)



from sklearn.model_selection import train_test_split

def train_split(X_raw, X_segmented, y_class=None):
    """
    Splits raw images, masks (and optional labels) into train, val, test.
    """
    if y_class is not None:
        # Step 1
        X_raw_trainval, X_raw_test, X_seg_trainval, X_seg_test, y_class_trainval, y_class_test = train_test_split(
            X_raw, X_segmented, y_class, 
            test_size=0.2, 
            random_state=42, 
            stratify=y_class
        )

        # Step 2
        X_raw_train, X_raw_val, X_seg_train, X_seg_val, y_class_train, y_class_val = train_test_split(
            X_raw_trainval, X_seg_trainval, y_class_trainval, 
            test_size=0.25,  # 60/20/20
            random_state=42,
            stratify=y_class_trainval
        )

        return (X_raw_train, X_raw_val, X_raw_test,
                X_seg_train, X_seg_val, X_seg_test,
                y_class_train, y_class_val, y_class_test)

    else:
        # No labels
        X_raw_trainval, X_raw_test, X_seg_trainval, X_seg_test = train_test_split(
            X_raw, X_segmented, 
            test_size=0.2, 
            random_state=42
        )

        X_raw_train, X_raw_val, X_seg_train, X_seg_val = train_test_split(
            X_raw_trainval, X_seg_trainval, 
            test_size=0.25, 
            random_state=42
        )

        return (X_raw_train, X_raw_val, X_raw_test,
                X_seg_train, X_seg_val, X_seg_test)

from torch.utils.data import TensorDataset

def create_segmentation_tensor_dataset(X_raw, X_mask):
    """
    Combine raw and mask tensors into a TensorDataset for U-Net.
    """
    return TensorDataset(X_raw, X_mask)

from torch.utils.data import DataLoader

def create_dataloader(dataset, batch_size=32, shuffle=True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
