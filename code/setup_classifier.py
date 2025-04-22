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
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import TensorDataset, DataLoader


# Check version of Pytorch
print(torch. __version__)


def load_segmented_ims(input_path):
    """User inputs the folder where their segmented images are stored"""
    
    #Retrieve files
    image_filepaths = []
    image_labels = []
    image_ids = []



    #for each file in the inputs add path to storage list:

    for files in os.listdir(input_path):
        if files.endswith('png'):
            if 'A2_MHV' in files:
                label = 'mutant'
            elif 'MHVWT' in files:
                label = 'wildtype'
            image_labels.append(label)
            image_filepaths.append(os.path.join(input_path,files))
            image_ids.append(Path(files).stem)
            

    all_files = pd.DataFrame([image_ids, image_filepaths, image_labels], index= ['im_id', 'file_path', 'class']).T
    all_files['im_id'] =all_files['im_id'].str.replace('_seg_ver2','')
    all_files_df = all_files.copy()
    
    print('all files loaded')

    return all_files_df

def transform_data(
    image_size=(256, 256),
    normalize_mean=(0.5,), 
    normalize_std=(0.5,),
    rotation_degree=15,
    scale_range=(0.9, 1.0),
    apply_augmentation=True
):
    """
    Returns train and validation transformations with configurable parameters.
    """

    base_transform = [
        transforms.ToTensor(),
        transforms.Normalize(normalize_mean, normalize_std)
    ]

    if apply_augmentation:
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(rotation_degree),
            transforms.RandomResizedCrop(image_size, scale=scale_range),
            *base_transform
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(image_size),
            *base_transform
        ])

    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(image_size),
        *base_transform
    ])

    return train_transform, val_transform

def create_tensor_dataset(df, transform):
    """Using custom transforms for training and validation datasets"""
    images = []
    labels = []

    for _, row in df.iterrows():
        file_path, label = row[1], row[2]

        img = cv2.imread(file_path)

        if img is None:
            raise ValueError(f"Image not found: {file_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (256, 256))

        img = transform(img)  # now using dynamic augmentation

        images.append(img)
        labels.append(1 if label == 'mutant' else 0)

    images_tensor = torch.stack(images)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    return TensorDataset(images_tensor, labels_tensor)


def create_dataloader(dataset, batch_size=64, shuffle=True):
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader

import os
import torch


def general_save(save_dir, timestamp, filename, dataframe=None):
    # If saving a DataFrame
    if dataframe is not None:
        save_path = os.path.join(save_dir, f"{filename}_{timestamp}.csv")
        dataframe.to_csv(save_path, index=False)
        print(f"Saved {filename} split to CSV: {save_path}")
        return save_path

    else:
        raise ValueError("You must provide a dataframe (for .csv).")

    

from sklearn.model_selection import train_test_split

def create_and_save_new_df(input_images_df, timestamp, save_dir, stratify=True):
    # Determine whether to stratify
    stratify_labels = input_images_df['class'] if stratify else None

    # Train/test split → 80% train, 20% test
    X_train_df, X_test_df = train_test_split(
        input_images_df, test_size=0.2, stratify=stratify_labels, random_state=42
    )

    # Train/val split from training data → 75% train, 25% val (so final is 60/20/20)
    X_train_df, X_val_df = train_test_split(
        X_train_df, test_size=0.25, stratify=stratify_labels.loc[X_train_df.index] if stratify else None, random_state=42
    )

    # Save all splits
    general_save(save_dir, timestamp, filename="train", dataframe=X_train_df)
    general_save(save_dir, timestamp, filename="val", dataframe=X_val_df)
    general_save(save_dir, timestamp, filename="test", dataframe=X_test_df)

    return X_train_df, X_test_df, X_val_df


