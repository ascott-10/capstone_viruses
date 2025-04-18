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

print('Imported libraries')

############ Import custom python files #################

from code.input_files import input_files
from code.sam_configure import download_sam, custom_mask
from code.segment_workflow import load_image, generate_masks, postprocess_mask, save_masks
############ Input Files #################

#Create the dataframe to store the filepaths of the raw datafiles
#df = input_files()

df = pd.DataFrame({
    'filepath': [
        '/home/ascott10/documents/projects/capstone_viruses/raw_images/mutant/A2_MHV_AxA_0047_Ceta_2446_1812.png',
        '/home/ascott10/documents/projects/capstone_viruses/raw_images/wt/MHVWT_A70020_175_2143.png'
    ],
    'class': ['mutant', 'wildtype']
})


print(df.head())

############ Configure Sam #################
#Download sam model, create mask generator object 

sam = download_sam()
mask_generator = custom_mask(sam) #user  can choose to not use defaults

############ Segmentation #################
#load raw images ###
#Input is the dataframe of the raw image file filepaths
#Output is the list of images
image_ls = load_image(df)

### To make original masks ###
map_ls = [] #segmentation maps (without adjustments)
body_bbox_ls = [] #bounding boxes for adjustments

#For each image, get the mask, the bbox coordinates and the area information
for i in range(0,len(image_ls)): #For i in range (0, length of raw image list)
    image = image_ls[i] #set current image

    mask, body_bbox = generate_masks(image, mask_generator) #generate masks for objects in current image
  #Save masks, bounding boxes
    map_ls.append(mask) #Original mask without adjustments
    body_bbox_ls.append(body_bbox) #bounding box to make adjustments

### To make processed masks ###
new_seg_map_ls = [] #segmentation maps with adjustments
pred_num_spikes_ls = [] #predicted number of spikes
all_area_df_ls = []
for i in range(0,len(map_ls)): #for i in range (0, length of mask list)
  segmentation_map = map_ls[i] #set current mask
  body_bbox = body_bbox_ls[i] #set current bounding box
  new_seg_map, num_spikes, total_spike_area, average_spike_area, all_area_df = postprocess_mask(segmentation_map, body_bbox)
  #generate new mask, predicted #spikes
  #Save new masks, predicted number of spikes
  image_id = Path(df.iloc[i, 0]).stem
  all_area_df["image_id"] = image_id
  all_area_df["file_name"] = os.path.basename(df.iloc[i, 0])
  all_area_df_ls.append(all_area_df)

  new_seg_map_ls.append(new_seg_map)
  pred_num_spikes_ls.append(num_spikes)
  
big_area_df = pd.concat(all_area_df_ls, ignore_index=True)

#Save to csv
output_csv_path = '/home/ascott10/documents/projects/capstone_viruses/results/all_component_areas.csv'
big_area_df.to_csv(output_csv_path, index=False)

### To save processed masks ###
updated_mask_path_ls = [] #filepaths for new masks
save_dir = '/home/ascott10/documents/projects/capstone_viruses/segmented_images/sam_segment_ver3'
for i in range(0, len(image_ls)): #for i in range(0,len(file paths of original images))
  im_path = df.iloc[i,0] #original file path name
  new_seg_map = new_seg_map_ls[i] #set current image 
  updated_mask_path = save_masks(im_path, new_seg_map, save_dir)
  updated_mask_path_ls.append(updated_mask_path)


############ Make a Results df #################
summary_df = pd.DataFrame({
    "file_name": [os.path.basename(p) for p in df.iloc[:, 0]],
    "new_mask_path": updated_mask_path_ls,
    "predicted_spike_count": pred_num_spikes_ls
})

#Save to CSV
summary_path = '/home/ascott10/documents/projects/capstone_viruses/results/sam_summary_results_2.csv'
summary_df.to_csv(summary_path, index=False)
print("Saved summary to:", summary_path)


sys.exit(0)
