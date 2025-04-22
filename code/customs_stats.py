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
from torch.utils.data import TensorDataset, DataLoader

import torch
from sklearn.metrics import confusion_matrix

import os
import torch
from torchvision import models
import glob

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from code.setup_classifier import load_segmented_ims, transform_data, create_tensor_dataset, create_dataloader, create_and_save_new_df
# Check version of Pytorch
print(torch. __version__)

import torch.nn as nn
from torchvision import models
"""The default is ResNet18 with binary classification"""
"""Final layer MUST MATCH training final layer"""

def load_resnet_weights(model, save_dir, device, save_path=None, num_classes=2):
    # Step 1: Recreate model architecture
    
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, num_classes)
    )

    # Step 2: Determine which weights file to load
    if save_path is None:
        weight_files = glob.glob(os.path.join(save_dir, "resnet_weights_*.pth"))
        if not weight_files:
            raise FileNotFoundError(f"No weight files found in {save_dir}")
        save_path = max(weight_files, key=os.path.getctime)  # most recently created file
        print(f"[INFO] Using latest weights: {save_path}")
    else:
        if not os.path.exists(save_path):
            raise FileNotFoundError(f"Provided path does not exist: {save_path}")
        print(f"[INFO] Using user-provided weights: {save_path}")

    # Step 3: Load the weights
    model.load_state_dict(torch.load(save_path, map_location=device))
    model.to(device)
    model.eval()

    return model

def make_test_data(dataframe=None, csv_path=None, csv_dir='/home/ascott10/documents/projects/capstone_viruses/data', pattern='test_*.csv'):
    if dataframe is not None:
        X_test_df = dataframe
    else:
        # If no csv_path is provided, find the latest matching CSV
        if csv_path is None:
            matching_files = glob.glob(os.path.join(csv_dir, pattern))
            if not matching_files:
                raise FileNotFoundError(f"No files matching '{pattern}' found in {csv_dir}")
            csv_path = max(matching_files, key=os.path.getctime)
            print(f"[INFO] Using latest CSV: {csv_path}")
        
        X_test_df = pd.read_csv(csv_path)

    _, val_transform = transform_data(
        image_size=(256, 256),
        normalize_mean=(0.5,), 
        normalize_std=(0.5,),
        rotation_degree=15,
        scale_range=(0.9, 1.0),
        apply_augmentation=True
    )    

    test_dataset = create_tensor_dataset(X_test_df, val_transform)
    test_loader = create_dataloader(test_dataset, batch_size=32, shuffle = False)

    

    return X_test_df, test_dataset, test_loader


def make_predictions(model, device, X_test_df, test_loader):

    #Make predictions
    predictions = []
    correct = 0  # Initialize correct count
    total = 0    # Initialize total count

    with torch.no_grad():  
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)  # Move data to same device as model
            
            outputs = model(images)  
            _, predicted = torch.max(outputs, 1)  

            predictions.extend(predicted.tolist())  # Store predictions
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Calculate Accuracy
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

    # Copy test dataframe
    X_test_df_preds = X_test_df.copy()

    # Assign predicted labels
    X_test_df_preds["predicted_label"] = ["mutant" if p == 1 else "wildtype" for p in predictions]

    # Print Confusion Matrix
    print(f"Confusion Matrix: \n {confusion_matrix(X_test_df_preds['class'], X_test_df_preds['predicted_label'])}")
    cm = confusion_matrix(X_test_df_preds['class'], X_test_df_preds['predicted_label'])
    import seaborn as sns
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111)

    # Define labels
    # Dynamically get sorted unique labels from both columns
    # Define labels explicitly
    labels = ["mutant", "wildtype"]

    # Compute confusion matrix with label order
    cm = confusion_matrix(
        X_test_df_preds['class'],
        X_test_df_preds['predicted_label'],
        labels=labels
    )

    # Plot heatmap
    sns.heatmap(cm, cmap='Blues', annot=True, fmt='d',
                xticklabels=labels, yticklabels=labels, ax=ax)


    # Plot heatmap
    sns.heatmap(cm, cmap='Blues', annot=True, fmt='d',
                xticklabels=labels, yticklabels=labels, ax=ax)

    # Label axes
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix')
    # Drop file_path column
    #X_test_df_preds = X_test_df_preds.drop('file_path', axis=1)

    # Get the confusion matrix
    #cf_matrix = confusion_matrix(X_test_df_preds['class'], X_test_df_preds['predicted_label'])
    print(X_test_df_preds['class'].value_counts())
    X_test_df_preds['predicted_label'].value_counts()

    return X_test_df_preds

def display_stats(X_test_df_preds):
    y_test = X_test_df_preds['class']
    y_pred = X_test_df_preds['predicted_label']

    preds_ls = []
    acc_ls = []
    prec_ls = []
    recall_ls = []
    f1_score_ls = []
    probab_ls = []
    confusion_matrices = []

    

    #Metrics
    acc_ls.append(accuracy_score(y_test, y_pred))
    prec_ls.append(precision_score(y_test, y_pred, average= 'weighted',zero_division=0))
    recall_ls.append(recall_score(y_test, y_pred, average='weighted', zero_division=0))
    f1_score_ls.append(f1_score(y_test, y_pred, average='weighted', zero_division=0))

    scores_df = pd.DataFrame({
        'Accuracy': acc_ls,
        'Precision': prec_ls,
        'Recall': recall_ls,
        'F1 Score': f1_score_ls})

    print(scores_df)