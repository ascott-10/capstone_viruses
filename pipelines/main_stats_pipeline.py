
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

print('Imported libraries') 
################ Custom ################

from code.customs_stats import load_resnet_weights, make_test_data, make_predictions, display_stats
################ Set GPU or CPU ################

#Set GPU or CPU to remain consistent through
use_cuda = torch.cuda.is_available()
device = "cuda" if torch.cuda.is_available() else "cpu"

################ To Load Classifier Model #########



pre_trained_model = models.resnet18(pretrained=False)
save_dir = '/home/ascott10/documents/projects/capstone_viruses/data'

model = load_resnet_weights(pre_trained_model, save_dir, device, num_classes=2)

X_test_df, test_dataset, test_loader =  make_test_data(dataframe=None, csv_path=None, csv_dir=save_dir, pattern='test_*.csv')


################ Make Predictions #########

X_test_df_preds = make_predictions(model, device, X_test_df, test_loader)
display_stats(X_test_df_preds)