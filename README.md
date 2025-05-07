# Capstone Project – Virus Segmentation and Morphology Analysis

## Description

This repository contains a pipeline to perform segmentation and morphological analysis on EM images of viruses using a U-Net model and a ResNet18-based classifier. It isolates viral structures (body and spikes), extracts morphological features, and classifies viral particles as wildtype or mutant.

## Repository Structure

- `main.py`: Full end-to-end pipeline: segmentation, classification, morphology, and evaluation
- `main_sam_pipeline.py`: Alternative pipeline using Segment Anything (SAM) for segmentation
- `main_classify_pipeline.py`: Classification from pre-segmented masks
- `code/`: Core functionality
  - `input_files.py`: Builds dataframe of image paths and labels  
  - `unet_model.py`: U-Net architecture  
  - `unet_training.py`: U-Net training, validation, model saving  
  - `unet_training_setup.py`: Dataset loading and preprocessing for segmentation  
  - `setup_classifier.py`: Data transforms, loading, and dataset splitting  
  - `train_classifier.py`: ResNet classifier training  
  - `customs_stats.py`: Evaluation helpers and confusion matrix  
  - `morphology.py`: Morphology extraction and analysis  
  - `spike_morpohology.py`: Spike-specific analysis  
  - `compare_segmentation_methods.py`: Manual vs automatic segmentation comparison
- `raw_images/`: Input EM images (wildtype and mutant)
- `segmented_images/`: Output binary masks (manual and auto)
- `results/`: Output plots, statistics, and evaluation summaries
- `data/`: Train/test/val splits and model weights

## Installation

git clone https://github.com/ascott-10/capstone-viruses.git
cd capstone-viruses
pip install -r requirements.txt

## Requirements

Ubuntu 22.04  
Python 3.12  
PyTorch 2.1  
CUDA 12.4  
NVIDIA GPU (recommended: RTX 3090)  
(Optional) Segment Anything setup from: https://github.com/facebookresearch/segment-anything/

---

# Full Pipeline Usage

## 1. Run Main Script

python main.py

Main script performs:
- Loads and optionally subsamples raw and segmented image metadata
- Loads or trains U-Net model on image/mask pairs
- Evaluates model on held-out set and visualizes mask predictions
- Loads or trains ResNet classifier on segmented masks
- Extracts morphology from masks
- Compares morphology across wildtype and mutant
- Compares manual vs automatic segmentation methods
- Saves plots and results

## Output Files

Segmented masks → segmented_images/  
Morphology statistics → results/all_component_areas.csv  
Per-image spike stats → results/sam_summary_results_2.csv  
Classifier confusion matrix → results/confusion_matrix.png  
Manual vs automatic comparison → results/comparison_results.csv  
Train/test/val splits → data/train_*.csv, val_*.csv, test_*.csv  
Model weights → data/best_unet.pt, data/resnet_weights_*.pth

## Sample Usage

### Build Segmentation Dataset

from code.unet_training_setup import *
X_raw, X_seg, labels = load_images_from_dataframe(df, ...)
train_dataset = create_segmentation_tensor_dataset(X_raw, X_seg, labels)
train_loader = create_dataloader(train_dataset, batch_size=32)

### Train U-Net

from code.unet_model import UNet
model = UNet(input_channels=1, output_channels=1).to(device)

from code.unet_training import *
train_model(...)  # see main.py for full loop

### Extract Morphology and Compare

from code.morphology import ground_truth_morph, calculate_spike_stats, compare_classes_stats
df = ground_truth_morph(df)
calculate_spike_stats(df, plot_yes=True)
compare_classes_stats(df, plot_yes=True)

### Classify Segmented Images

from code.customs_stats import load_resnet_weights, make_predictions
model = load_resnet_weights(model, save_dir='data', device=device)
X_test_df_preds = make_predictions(model, device, X_test_df, test_loader, save_cm=True)

## Citation

Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Dollár, and Ross Girshick. Segment Anything. arXiv:2304.02643, 2023.
