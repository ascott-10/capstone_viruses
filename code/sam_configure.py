#%%
################ Import Libraries ################
import numpy as np
import pandas as pd

import os
import sys
sys.path.append("..")
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


#%%
################ Configure SAM ################
def download_sam():

  #Can change the path to where the model is stored
  sam_checkpoint = "/home/ariellescott/Documents/capstone/sam_vit_h_4b8939.pth"  # Pre-downloaded the model already to my folder
  model_type = "vit_h"  # model type is vit_h per the pre-downloaded model
  print('model downloaded')
  
  sam_checkpoint     
  
  sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device)
  sam.to(device=device)
  
  return sam
  
def custom_mask(sam):
  mask_generator = SamAutomaticMaskGenerator(sam)
  
  #pretrained model to generate masks with custom parameters
  mask_generator_ = SamAutomaticMaskGenerator(
  
      model=sam,
      #number of points to be sampled per side of image (more points = denser sampling ~ better segmentation)                              
      points_per_side=32,
      #predicted Intersection over Union (IoU) threshold. higher IoU --> higher quality                     
      pred_iou_thresh=0.95,
      #stability score = measure of quality ~ higher --> better quality masks                    
      stability_score_thresh=0.98,
      # #layers of crops --> size of image crops, improve performance on smaller objects            
      crop_n_layers=1,
      # downscaling factor for #points per side in the crops, controls density of point sampling in the image crops                        
      crop_n_points_downscale_factor=2,
                     
  )
  
  return mask_generator

