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
################ Data Setup ################
#input folder paths

def input_files(im_path_mut, im_path_wt):
  im_path_mut = '/home/ariellescott/Documents/capstone/raw_images/raw_mutant'
  im_path_wt = '/home/ariellescott/Documents/capstone/raw_images/raw_wt'
  
  #Temporary storage
  
  image_filepaths_mut = []
  image_filepaths_wt = []
      
  for files in os.listdir(im_path_mut):
      if files.endswith('png'):
          image_filepaths_mut.append(os.path.join(input_path_mut,files))
          
  for files in os.listdir(im_path_wt):
      if files.endswith('png'):
          image_filepaths_wt.append(os.path.join(input_path_wt,files))
          
  print('Data imported')

#%%
################ Configure SAM ################
def download_sam():

  sam_checkpoint = "/home/ariellescott/Documents/capstone/sam_vit_h_4b8939.pth"  # Pre-downloaded the model already to my folder
  model_type = "vit_h"  # model type is vit_h per the pre-downloaded model
  print('model downloaded')     
  
  sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device)
  sam.to(device=device)
  
  sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device)
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

################ Workflow to preprocess input images images ################
###################
def sam_workflow(im_path):

    #input is image path
    image_orig = cv2.imread(im_path) 
    image = cv2.cvtColor(image_orig, cv2.COLOR_BGR2RGB)

    height, width = image.shape[:2]
    image_center = np.array([width // 2, height // 2])
    
    #Create empty image of same shape as image with zeros in range [0,255]
    segmentation_map = np.zeros((height, width, 3), dtype=np.uint8)

    #Make the mask from the sam model
    masks = mask_generator.generate(image)
    
    plt.figure(figsize=(3,3))
    plt.imshow(image)

    #Rectangle for where the body and spikes are being counted
    plt.gca().add_patch(Rectangle((x_xten,y_xten),(w_xten),(h_xten), fill = False))
    #for k in range(0,len(x_coords)):
    #    plt.gca().add_patch(Rectangle((x_coords[k],y_coords[k]),w_coords[k],h_coords[k], fill = False))
    #plt.axis('off')
        
    #plt.show()
