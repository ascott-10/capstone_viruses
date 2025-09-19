#predict_one.py

import torch
import cv2
import numpy as np
from pathlib import Path
import os
import sys
from config import *

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from unet_model import UNet

def find_best_model_by_val_loss(save_dir):
    from glob import glob
    import re
    pattern = os.path.join(save_dir, "best_unet_epoch*_val*.pt")
    model_files = glob(pattern)
    if not model_files:
        raise FileNotFoundError("No model files found.")
    def extract_val_loss(fname):
        match = re.search(r"val([\d.]+)\.pt", fname)
        return float(match.group(1)) if match else float("inf")
    return min(model_files, key=extract_val_loss)

def load_and_preprocess_image(image_path):
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, IMAGE_SIZE)
    img = img.astype(np.float32) / 255.0
    return torch.tensor(img).unsqueeze(0).unsqueeze(0).to(DEVICE)

def predict_single_image(image_path, model_path, save_dir):
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

def save_composite_image(img_name, raw_dir, sam_dir, manual_dir, predicted_mask_path, save_dir):
    name_stem = Path(img_name).stem

    raw_path      = Path(raw_dir) / img_name
    sam_path      = Path(sam_dir) / img_name.replace(".png", "_seg_ver2.png")
    manual_path   = Path(manual_dir) / img_name.replace(".png", "_corrected.png")
    predicted_path = Path(predicted_mask_path)

    missing = [p for p in [raw_path, sam_path, manual_path, predicted_path] if not p.exists()]
    if missing:
        print(f"Skipping {img_name}, missing: {[str(p) for p in missing]}")
        return

    imgs = []
    for p in [raw_path, sam_path, manual_path, predicted_path]:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Failed to read image: {p}")
            return
        imgs.append(cv2.resize(img, (256, 256)))

    composite = np.hstack(imgs)
    out_path = Path(save_dir) / f"{name_stem}_comparison.png"
    cv2.imwrite(str(out_path), composite)

