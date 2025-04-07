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
input_path = '/home/student01/Documents/ascott10/capstone/images/'
print('Data imported')

#Temporary storage
image_filepaths = []

for files in os.listdir(input_path):
    if files.endswith('png'):
        image_filepaths.append(os.path.join(input_path,files))

#%%
################ Configure SAM ################
sam_checkpoint = "/home/student01/Documents/ascott10/capstone/images/sam_vit_h_4b8939.pth"  # Pre-downloaded the model already to my folder
model_type = "vit_h"  # model type is vit_h per the pre-downloaded model
print('model downloaded')     
        


sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device)
sam.to(device=device)

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device)
mask_generator = SamAutomaticMaskGenerator(sam)

mask_generator_ = SamAutomaticMaskGenerator(
#pretrained model to generate masks
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

###################
def sam_workflow(im_path):

    image_orig = cv2.imread(im_path) #input is image path
    image = cv2.cvtColor(image_orig, cv2.COLOR_BGR2RGB)

    height, width = image.shape[:2]
    image_center = np.array([width // 2, height // 2])

    #Create empty image of same shape as image with zeros in range [0,255]
    segmentation_map = np.zeros((height, width, 3), dtype=np.uint8)

    #Make the mask from the sam model
    masks = mask_generator.generate(image)

    
    #compute area
    for mask in masks:
        # Function to compute mask area
        def get_mask_area(mask):
            return np.sum(mask["area"])

        def get_center_distance(mask):
            x, y, w, h = mask["bbox"]  # Get bounding box of the mask
            mask_center = np.array([x + w // 2, y + h // 2])  # Compute mask center
            return np.linalg.norm(mask_center - image_center)  # Euclidean distance

    #Sort masks by euclidean distance then by area 
    masks.sort(key=get_center_distance)  # Then sort by closeness to center
    masks.sort(key=get_mask_area, reverse=True)  # Sort by area (largest first)
    

    ####Spikes#####
    spike = 0 #Spike counter

    ####Body#####
    # Assign the first mask (most centered large object) as "body", rest as "spikes"
    x_body, y_body, w_body, h_body  = masks[1]['bbox']
    extend = 50 #Create box around the body to include the spikes
    x_xten, y_xten, w_xten, h_xten = (x_body-extend),(y_body-extend),(w_body+(2*extend)),(h_body+2*extend)

    #Temporary storage of x and y coordinates for specified range
    x_coords = []
    y_coords = []
    h_coords = []
    w_coords = []
    for i, mask in enumerate(masks):
        if i == 0: 
            #The largest mask (first in sorted) is the background
            segmentation_map[mask["segmentation"] > 0] = (0,0,0) #black
        else:
            if i == 1:
                #The second largest mask is center body
                segmentation_map[mask["segmentation"] > 0] = (150, 150, 150) #dark gray
            
            else:
                x, y, w, h = mask["bbox"] #Grab the coordinates of the spike
                if (x > x_xten) and (x < (x_xten+h_xten)) and (y > y_xten) and (y < (y_xten+w_xten)): #if in range
                    spike = spike + 1 #Add to the counter
                    #make a list of the x-coordinates(first number)
                    x_coords.append(x)                
                    y_coords.append(y)                    
                    h_coords.append(h)                   
                    w_coords.append(w)
                    #The rest of the masks will be the spikes
                    segmentation_map[mask["segmentation"] > 0] = (200, 200, 200) #light gray
                
                #if the bbox outside range of bbox of extended body
                #then it will not count as a spike             
                
                else:
                    spike = spike
                    segmentation_map[mask["segmentation"] > 0] = (100,100,100) #medium gray

       
    
    # Determine the save path based on the filename
    filename = os.path.basename(im_path)
    
    
    if "MHVWT" in filename:
        # If 'MHWT' is in the filename, save to segment_wt folder
        save_dir = "/home/student01/Documents/ascott10/capstone/sam_segment_primary/segment_wt"
    elif "A2_MHV" in filename:
        # If 'A2_MHV' is in the filename, save to segment_mut folder
        save_dir = "/home/student01/Documents/ascott10/capstone/sam_segment_primary/segment_mut"
    else:
        # Default folder if neither is found
        save_dir = "/home/student01/Documents/ascott10/capstone/sam_segment_primary/other_masks"

    os.makedirs(save_dir, exist_ok=True)

    # Modify the filename to append '_seg' before the extension
    filename_without_ext, ext = os.path.splitext(filename)
    new_filename = f"{filename_without_ext}_seg{ext}"
    

    # Save the segmentation mask
    mask_path = os.path.join(save_dir, new_filename)
    cv2.imwrite(mask_path, segmentation_map)
    
    
    
    
    
    return image_orig, segmentation_map, spike
    
    

##########

mask_list = []
spike_list = []
image_list = []
im_path_list = []
file_ends = []


#n = 4  # Number of random numbers
#rand_ = random.sample(range(1, 400), n)

for i in range(0, len(image_filepaths)):
  image, masks, spikes= sam_workflow(image_filepaths[i])
  im_path_list.append(image_filepaths[i])
  spike_list.append(spikes) 

#For the df
for el in im_path_list:
  filename = os.path.splitext(os.path.basename(el))[0]
  file_ends.append(filename)
df = pd.DataFrame({'File_name': file_ends, 'Pred_spike_count': spike_list})
  
print(df.head())

#Print to csv
output_csv_path = '/home/student01/Documents/ascott10/capstone/sam_seg_results_prelim.csv'

#save df to csv
df.to_csv(output_csv_path, index=False)  # index=False to avoid writing row numbers as a column


######################
