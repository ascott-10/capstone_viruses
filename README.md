# Capstone Project – Virus Segmentation and Morphology Analysis

## Description

This repository contains a pipeline to perform segmentation and morphological analysis on EM images of viruses using Meta AI's Segment Anything Model (SAM) and classification using a ResNet18-based model. The goal is to isolate the viral body and spikes, compute relevant features (area, perimeter), and classify viral images as wildtype or mutant based on morphology.

## Repository Structure

- `main_sam_pipeline.py`: Segments raw EM images using SAM and computes morphology statistics  
- `main_classify_pipeline.py`: Classifies segmented images into wildtype or mutant  
- `code/`: Contains modular functions for data processing, model training, and image transformation  
  - `input_files.py`: Builds dataframe of image paths and labels  
  - `sam_configure.py`: Loads pretrained SAM model and sets hyperparameters  
  - `segment_workflow.py`: Segmentation, cropping, feature extraction, and saving  
  - `setup_classifier.py`: Loads images, applies transformations, splits into datasets  
  - `train_classifier.py`: Defines model and training logic  
  - `morphology.py`: Morphology analysis and segmentation comparison  
  - `customs_stats.py`: Classifier loading, prediction, and evaluation helpers  
- `raw_images/`: Input folders for wildtype and mutant virus images  
- `segmented_images/`: Output locations for final segmented masks  
- `results/`: Stores CSVs of segmentation outputs and comparison results  
- `data/`: CSVs for train/test/val splits and ResNet model weights  

## Installation

```bash
git clone https://github.com/ascott-10/capstone-viruses.git
cd capstone-viruses
pip install -r requirements.txt
```

## Other Requirements

- Install Segment Anything from https://github.com/facebookresearch/segment-anything/
- Download `sam_vit_h_4b8939.pth` checkpoint and place it in the appropriate directory

## System Requirements

- Ubuntu 22.04  
- Python 3.12  
- CUDA 12.4  
- PyTorch 2.1  
- NVIDIA GPU (e.g. RTX 3090)

---

# Usage Instructions

## Raw Images ➔ Segmented Masks

### 1. Build the input DataFrame

```python
from code.input_files import make_input_df
df = make_input_df('/path/to/raw_images/')
```

### 2. Configure and run SAM

```python
from code.sam_configure import download_sam, custom_mask
sam = download_sam()
mask_generator = custom_mask(sam)
```

### 3. Segment and postprocess images

```python
from code.segment_workflow import run_segmentation_pipeline
run_segmentation_pipeline(df, mask_generator, save_dir='segmented_images/sam_segment_ver3/')
```

### 4. Outputs

- Segmented grayscale masks → `segmented_images/sam_segment_ver3/`
- Per-component stats → `results/all_component_areas.csv`
- Per-image spike counts → `results/sam_summary_results_2.csv`

---

## Segmented Masks ➔ Morphology Comparison

### 1. Extract morphology from two segment sets

```python
from code.morphology import make_morphology_df

df_sam = make_morphology_df('segmented_images/segment_ver2')
df_manual = make_morphology_df('segmented_images/sam_segment_ver1')
```

### 2. Compare per-image spike areas

```python
from code.morphology import compare_methods

df_compare = compare_methods(
    method_a_dir='segmented_images/segment_ver2',
    method_b_dir='segmented_images/sam_segment_ver1',
    suffix_a=r'_seg_ver2$',
    suffix_b=r'_seg$'
)

df_compare.to_csv('results/df_compare.csv')
```

---

## Segmented Masks ➔ Classification

### 1. Prepare train/test/val splits

```python
from code.setup_classifier import load_segmented_ims, create_and_save_new_df

df = load_segmented_ims('segmented_images/segment_ver2')
create_and_save_new_df(df, save_dir='data/', timestamp='20250421', stratify=True)
```

### 2. Transform images and load into DataLoader

```python
from code.setup_classifier import transform_data, create_tensor_dataset, create_dataloader
import pandas as pd

train_df = pd.read_csv('data/train_20250421_151833.csv')
val_df = pd.read_csv('data/val_20250421_151833.csv')
test_df = pd.read_csv('data/test_20250421_151833.csv')

train_tfm, val_tfm = transform_data(image_size=(256, 256))

train_dataset = create_tensor_dataset(train_df, train_tfm)
val_dataset = create_tensor_dataset(val_df, val_tfm)
test_dataset = create_tensor_dataset(test_df, val_tfm)

train_loader = create_dataloader(train_dataset, batch_size=64)
val_loader = create_dataloader(val_dataset, batch_size=64)
test_loader = create_dataloader(test_dataset, batch_size=32)
```

### 3. Train the classifier

```python
from code.train_classifier import load_classifier, train_model
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_classifier(device=device, num_classes=2)
train_model(model, device, train_loader, val_loader, save_dir='data/')
```

---

## Evaluate Trained Classifier on Test Set

### 1. Load pretrained model and test set

```python
from code.customs_stats import load_resnet_weights, make_test_data

from torchvision import models
pre_trained_model = models.resnet18(pretrained=False)
device = "cuda" if torch.cuda.is_available() else "cpu"

model = load_resnet_weights(pre_trained_model, save_dir='data', device=device, num_classes=2)

X_test_df, test_dataset, test_loader = make_test_data(
    dataframe=None,
    csv_path=None,
    csv_dir='data',
    pattern='test_*.csv'
)
```

### 2. Run prediction and display stats

```python
from code.customs_stats import make_predictions, display_stats

X_test_df_preds = make_predictions(model, device, X_test_df, test_loader)
display_stats(X_test_df_preds)
```

---

## Output Files

- Segmented images → `segmented_images/sam_segment_ver3/`
- Per-component morphology → `results/all_component_areas.csv`
- Spike summary per image → `results/sam_summary_results_2.csv`
- Comparison of methods → `results/df_compare.csv`
- Model weights → `data/resnet_weights_*.pth`
- Dataset splits → `data/train_*.csv`, `val_*.csv`, `test_*.csv`

---
## Citation

Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Dollár, and Ross Girshick. Segment Anything. *arXiv preprint arXiv:2304.02643*, 2023.