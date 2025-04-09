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

#Create the dataframe to store the filepaths of the raw datafiles
df = input_files.input_files()

print(df.head())

############ Configure Sam #################

import sam_configure

#Download the sam model
sam = sam_configure.download_sam()

#Create the mask generator object
mask_generator = sam_configure.custom_mask(sam)

############ Segmentation #################
import segment_workflow

### To load raw images ###
#Input is the dataframe of the raw image file filepaths
#Output is the list of images
image_ls = segment_workflow.load_image(df)

### To make original masks ###
map_ls = [] #segmentation maps (without adjustments)
body_bbox_ls = [] #bounding boxes for adjustments
for i in range(0,len(image_ls)): #For i in range (0, length of raw image list)
    image = image_ls[i] #set current image
    mask, body_bbox = segment_workflow.generate_masks(image, mask_generator) #generate masks for objects in current image
  #Save masks, bounding boxes
    map_ls.append(mask) #Original mask without adjustments
    body_bbox_ls.append(body_bbox) #bounding box to make adjustments

### To make processed masks ###
new_seg_map_ls = [] #segmentation maps with adjustments
pred_num_spikes_ls = [] #predicted number of spikes
for i in range(0,len(map_ls)): #for i in range (0, length of mask list)
  segmentation_map = map_ls[i] #set current mask
  body_bbox = body_bbox_ls[i] #set current bounding box
  new_seg_map, num_spikes = segment_workflow.postprocess_mask(segmentation_map, body_bbox) #generate new mask, predicted #spikes
  #Save new masks, predicted number of spikes
  new_seg_map_ls.append(new_seg_map)
  pred_num_spikes_ls.append(num_spikes)

### To save processed masks ###
updated_mask_path_ls = [] #filepaths for new masks
for i in range(0, len(image_ls)): #for i in range(0,len(file paths of original images))
  im_path = df.iloc[i,0] #original file path name
  new_seg_map = new_seg_map_ls[i] #set current image 
  updated_mask_path = segment_workflow.save_masks(im_path, new_seg_map)
  updated_mask_path_ls.append(updated_mask_path)


############ Make a Results df #################

#For the df file name
file_ends = []
for i in range(0,len(updated_mask_path_ls)):
    im_path = df.iloc[i,0]
    filename = os.path.splitext(os.path.basename(im_path))[0]
    file_ends.append(filename)
  

df = pd.DataFrame({'File_name': file_ends, 
'New mask path':updated_mask_path_ls,
'Pred_spike_count': pred_num_spikes_ls})
  

print(df.head())

#Print to csv
output_csv_path = '/home/ariellescott/Documents/capstone/capstone-viruses/data/output/sam_seg_results_ver2.csv'

#save df to csv
df.to_csv(output_csv_path, index=False)  # index=False to avoid writing row numbers as a column


#########Visualize######
segment_workflow.display_images(image_ls, map_ls, new_seg_map_ls, pred_num_spikes_ls)


sys.exit(0)
