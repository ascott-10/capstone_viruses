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
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from code.setup_classifier import load_segmented_ims, transform_data, create_tensor_dataset, create_dataloader, create_and_save_new_df

print(torch.__version__)

import torch.nn as nn
from torchvision import models

def load_resnet_weights(model, save_dir, device, save_path=None, num_classes=2):
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, num_classes)
    )
    if save_path is None:
        weight_files = glob.glob(os.path.join(save_dir, "resnet_weights_*.pth"))
        if not weight_files:
            raise FileNotFoundError(f"No weight files found in {save_dir}")
        save_path = max(weight_files, key=os.path.getctime)
        print(f"[INFO] Using latest weights: {save_path}")
    else:
        if not os.path.exists(save_path):
            raise FileNotFoundError(f"Provided path does not exist: {save_path}")
        print(f"[INFO] Using user-provided weights: {save_path}")
    model.load_state_dict(torch.load(save_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def make_test_data(dataframe=None, csv_path=None, csv_dir='/home/ascott10/documents/projects/capstone_viruses/data', pattern='test_*.csv'):
    if dataframe is not None:
        X_test_df = dataframe
    else:
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
    test_loader = create_dataloader(test_dataset, batch_size=32, shuffle=False)
    return X_test_df, test_dataset, test_loader

def make_predictions(model, device, X_test_df, test_loader, save_cm=True, save_dir=None):
    predictions = []
    correct = 0
    total = 0

    with torch.no_grad():  
        for _, masks, labels in test_loader:
            if masks.shape[1] == 1:
                masks = masks.repeat(1, 3, 1, 1)
            masks = masks.to(device)
            labels = labels.to(device)
            outputs = model(masks)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.tolist())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

    X_test_df_preds = X_test_df.copy()
    encoder = LabelEncoder()
    encoder.fit(["wildtype", "mutant"])
    X_test_df_preds["predicted_label"] = encoder.inverse_transform(np.array(predictions))

    labels = ["mutant", "wildtype"]
    cm = confusion_matrix(
        X_test_df_preds['class'],
        X_test_df_preds['predicted_label'],
        labels=labels
    )

    # ✅ Clean green-themed confusion matrix
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap="Greens",
        xticklabels=labels,
        yticklabels=labels,
        cbar=False,
        square=True,
        linewidths=0.5,
        linecolor='gray',
        ax=ax
    )

    ax.set_xlabel('Predicted Label', fontsize=13)
    ax.set_ylabel('True Label', fontsize=13)
    ax.set_title('Confusion Matrix', fontsize=15)
    ax.tick_params(labelsize=11)

    plt.tight_layout()
    if save_cm and save_dir:
        os.makedirs(save_dir, exist_ok=True)
        cm_path = os.path.join(save_dir, "confusion_matrix.png")
        plt.savefig(cm_path, dpi=300)
        print(f"Saved confusion matrix: {cm_path}")
    plt.show()

    print(X_test_df_preds['class'].value_counts())
    X_test_df_preds['predicted_label'].value_counts()
    return X_test_df_preds



from sklearn.metrics import classification_report

def display_stats(X_test_df_preds):
    y_test = X_test_df_preds['class']
    y_pred = X_test_df_preds['predicted_label']

    print("📊 Classification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    scores_df = pd.DataFrame({
        'Accuracy': [acc],
        'Precision': [prec],
        'Recall': [recall],
        'F1 Score': [f1]
    })

    print("\n✅ Summary Metrics:")
    print(scores_df)
    return scores_df
