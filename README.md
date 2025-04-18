# Capstone Project – Virus Segmentation and Morphology Analysis

## Description

This repository contains a pipeline to perform segmentation and morphological analysis on EM images of viruses using Meta AI's Segment Anything Model (SAM). The goal is to isolate the viral body and spikes, compute relevant features (area, perimeter), and export processed masks and statistics.

## Repository Structure

- `main_sam_pipeline.py`: Main script to execute the full workflow  
- `code/`:
  - `input_files.py`: Builds dataframe of image paths and labels  
  - `sam_configure.py`: Loads pretrained SAM model and sets hyperparameters  
  - `segment_workflow.py`: Contains functions for segmentation, cropping, feature extraction, and saving  
- `raw_images/`: Input folders for wildtype and mutant virus images  
- `segmented_images/sam_segment_ver3/`: Output location for final segmented masks  
- `results/`: Stores summary CSVs of segmentation outputs  

# Getting Started

## Installation

```bash
git clone https://github.com/ascott-10/capstone-viruses.git
cd capstone-viruses
pip install -r requirements.txt
```

## Other items needed

- Follow installation instructions for Segment Anything model from https://github.com/facebookresearch/segment-anything/
- Download the checkpoint `.pth` file for the `sam_vit_h_4b8939` model to stay consistent with the model used in this project
- Place the model in the appropriate directory for your project

## System Requirements (Project tested on):

- Ubuntu 22.04  
- Python 3.12  
- CUDA 12.4  
- Pytorch 2.1  
- NVIDIA GPU (RTX 3090)

# Usage Instructions

## Setup

### 1. Set up the file inputs

If the user has a directory of raw images that they wish to convert to segmented masks, they should proceed through the main_sam_pipeline.py script process. If they already have their generated masks and wish to get statistics on them or classify them, they should take note of where they are stored and skip these steps. 

First, the user should have a directory of raw images. This project used EM images of viruses, split into two groups. All images were 750x750, with the virus body centered in the middle.

In `input_files.py`, the user can enter their file paths or set up prompts to enter them interactively.  
This code will output a dataframe of filepaths and classes that will be used later. It also creates a CSV version of this information that can be reused — filename can be customized as needed.

When running `main_sam_pipeline.py`, the user will be prompted whether to use default filepaths, which can be hardcoded in `input_files.py` to avoid typing them on the command line.

### 2. Configure SAM

Make sure the SAM model (`sam_vit_h_4b8939.pth`) has been downloaded.

In `main.py`, `download_sam()` is called once to load the model and returns a SAM instance.  
Then `custom_mask(sam)` returns a configured mask generator object. You can pass `use_defaults=False` to override mask parameters.

### 3. Segmentation

The segmentation step consists of several function calls:

- `load_image(df)`: Loads all images as RGB from the filepath DataFrame.
- `generate_masks(image, mask_generator)`: Uses SAM to segment each image. The largest mask is assumed to be the viral body; masks within a 75-pixel extended bounding box are labeled as spikes. A grayscale mask is returned.
- `postprocess_mask(segmentation_map, body_bbox)`: Crops the mask to focus on the body and nearby spikes. Computes spike count, total spike area, average spike area, and body/spike perimeters using OpenCV.
- `save_masks(im_path, new_seg_map, save_dir)`: Saves each cropped mask to `segmented_images/sam_segment_ver3/` with `_seg_ver3` added to the filename.

Each function is called iteratively on the image set. Metrics are collected in lists and merged into DataFrames.

In the postprocessing step, spike features are stored per component, including area and perimeter. These are written to `results/all_component_areas.csv`. A separate summary with spike count per image is written to `results/sam_summary_results_2.csv`.

### 4. Run a quick test

To test the pipeline on a few images only, modify `df` in `main.py`:

```python
df = df.iloc[:2]  # run on first 2 images
```

## Output Files

- Segmented images: `segmented_images/sam_segment_ver3/`
- CSV summaries:
  - `results/all_component_areas.csv`: Area and perimeter stats for body and spike components per image
  - `results/sam_summary_results_2.csv`: File-level summary of predicted spike count and saved mask path

## Citations

Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Dollár, and Ross Girshick. Segment Anything. arXiv preprint arXiv:2304.02643, 2023.
