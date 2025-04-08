#%%
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


import torch
device = "cuda" if torch.cuda.is_available() else "cpu" 
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import paramiko #For remote GPU support

print('Import libraries good')

############ Input Files #################

import input_files

df = input_files.input_files()

print(df.head())


############ Configure Sam #################

import sam_configure

sam = sam_configure.download_sam()
mask_generator = sam_configure.custom_mask(sam)

############ Segmentation #################

import segment_workflow

num_ims = 8
random_indices = np.random.choice(len(df), size=num_ims, replace=False)

image_ls = segment_workflow.load_image(df, random_indices)
map_ls = []
body_bbox_ls = []
new_seg_map_ls = []
pred_num_spikes = []
for i in range(0,len(image_ls)):
  image = image_ls[i]
  _, mask, body_bbox = segment_workflow.generate_masks(image, mask_generator) #avoid appending twice
  map_ls.append(mask)
  body_bbox_ls.append(body_bbox)

for i in range(0,len(map_ls)):
  segmentation_map = map_ls[i]
  body_bbox = body_bbox_ls[i]
  new_seg_map, num_spikes = segment_workflow.postprocess_mask(segmentation_map, body_bbox)
  new_seg_map_ls.append(new_seg_map)
  pred_num_spikes.append(num_spikes)

segment_workflow.display_images(image_ls, map_ls, new_seg_map_ls, pred_num_spikes)


