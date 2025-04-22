
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
################ Custom ################
from code.setup_classifier import load_segmented_ims, transform_data, create_tensor_dataset, create_dataloader, create_and_save_new_df
from code.train_classifier import load_classifier, train_model

################ Set GPU or CPU ################

#Set GPU or CPU to remain consistent through
use_cuda = torch.cuda.is_available()
device = "cuda" if torch.cuda.is_available() else "cpu"

################ To Set up CLassifier #########

#if user does not have resnet_weights.pth, must generate some
#if user does not have train,test, val dataset, must generate some
#This pipeline uses ResNet18, if the user really wants a different one, they should modify the function

input_path = '/home/ascott10/documents/projects/capstone_viruses/segmented_images/segment_ver2'
input_images_df = load_segmented_ims(input_path)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_dir = '/home/ascott10/documents/projects/capstone_viruses/data'

#Augment the data to account for different sizes and apply transformations
train_transform, val_transform = transform_data(
    image_size=(256, 256),
    normalize_mean=(0.5,), 
    normalize_std=(0.5,),
    rotation_degree=15,
    scale_range=(0.9, 1.0),
    apply_augmentation=True
)

#Create train/test/val datasets
#X_train_df, X_test_df, X_val_df = create_and_save_new_df(input_images_df, timestamp, save_dir, stratify = True)

X_train_df = pd.read_csv('data/train_20250421_151833.csv')
X_test_df = pd.read_csv('data/test_20250421_151833.csv')
X_val_df = pd.read_csv('data/val_20250421_151833.csv')

#Create dataset in Pytorch format
train_dataset = create_tensor_dataset(X_train_df, train_transform)
val_dataset = create_tensor_dataset(X_val_df, val_transform)
test_dataset = create_tensor_dataset(X_test_df, val_transform)

train_loader = create_dataloader(train_dataset, batch_size=64, shuffle = True)
val_loader = create_dataloader(val_dataset, batch_size=64, shuffle = False)
test_loader = create_dataloader(test_dataset, batch_size=32, shuffle = False)

################ To Train CLassifier #########

#Training pipeline

model = load_classifier(device, num_classes=2)
model.to(device)
train_model(model, device, train_loader,val_loader, save_dir)

print('model trained')






