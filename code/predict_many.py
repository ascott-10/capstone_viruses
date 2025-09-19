import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname((__file__),'..'))))

import torch
import cv2
import numpy as np
import matplotlib
from pathlib import Path
from matplotlib import pyplot as plt

from unet_model import UNet
from config import *

import re
from glob import glob

from datetime import datetime

#####Define the Functions###########

def find_best_model_by_val_loss(save_dir)
"""Finds best U-Net model file with lowest val loss"""
    
    pattern = os.path.join(save_dir, "best_unet_epoch*_val*.pt")
    model_files = glob(pattern)
    if not model_files:
        raise FileNotFoundError("No model files found.")
    
    def extract_val_loss(fname):
        match = re.search(r"val([\d.]+)\.pt", fname)
        return float(match.group(1)) if match else float("inf")
    
    best_model = min(model_files, key = extract_val_loss)
    
    return best_model

def load_and_preprocess_image(image_path):
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, IMAGE_SIZE)
    img = img.astype(np.float32) / 255.0
    img_tensor = torch.tensor(img).unsqueeze(0).unsqueeze(0) # [1,1,H,W]
    
    
    return img_tensor.to(DEVICE)


def predict_single_image(image_path, model_path = BEST_UNET_PATH, save_dir = "."):
    image_path = Path(image_path)
    save_path = Path(save_dir) / f"{image_path.stem}_predicted_mask.png"
    model = UNet(input_channels=1, output_channels=1).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    img_tensor = load_and_preprocess_image(image_path)
    with torch.no_grad():
        pred = model(img_tensor)
        pred_mask = torch.sigmoid(pred).squeeze().cpu().numpy()
    binary_mask = (pred_mask > 0.5).astype(np.uint8) * 255
    cv2.imwrite(str(save_path), binary_mask)
    print(f"Saved predicted mask to {save_path}")


#####Run the functions###########

#Get image paths
mutant_images = glob(ps.path.join(RAW_IMS_MUT, "*.png"))
wt_images = glob(os.path.join(RAW_IMS_WT, "*.png"))

#Create save directories
timestamp = datetime.now().strftime("batch_%Y_%m_%d_%H_%M")
base_dir = Path(UNET_PREDICTION_SAVE_DIR) / timestamp

mutant_dir = base_dir / "mutant"
wt_dir = base_dir / "wildtype"

mutant_dir.mkdir(parents = True, exist_ok = True)
wt_dir.mkdir(parents = True, exist_ok= True)

#Load model
best_model_path = find_best_model_by_val_loss(SAVE_DIR)

#Predict
for img_path_m in mutant_images:
    predict_single_image(img_path_m, model_path = best_model_path, save_dir = mutant_dir)

for img_path_w in wildtype_images:
    predict_single_image(img_path_w, model_path = best_model_path, save_dir = wt_dir_dir)
    