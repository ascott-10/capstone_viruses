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


################ Workflow to preprocess input images images #########
def load_image(df = None): 
    image_ls = []
    if df is None:
        df = "/home/ascott10/documents/projects/capstone_viruses/results/raw_filepaths.csv"
    
    for i in range(len(df.index)):
        im_path = df.iloc[i, 0]   
        image_orig = cv2.imread(im_path)
        if image_orig is None:
            print(f"Warning: couldn't load image at {im_path}")
            continue
        image = cv2.cvtColor(image_orig, cv2.COLOR_BGR2RGB)
        image_ls.append(image)
        print('image', str(i), 'loaded')
    
    return image_ls
  
################ Workflow to preprocess input images #########
def generate_masks(image, mask_generator):
    
    #Get the height, width and calculate where the center is
    height, width = image.shape[:2]
    image_center = np.array([width // 2, height // 2])
    
    #Create empty image of same shape as image with zeros in range [0,255]
    segmentation_map = np.zeros((height, width, 3), dtype=np.uint8)

    #Make the mask from the mask generator
    masks = mask_generator.generate(image)

       
    # Function to compute mask area
    def get_mask_area(mask):
        return np.sum(mask["area"])

    def get_center_distance(mask):
        x, y, w, h = mask["bbox"]  # Get bounding box of the mask
        mask_center = np.array([x + w // 2, y + h // 2])  # Compute mask center
        return np.linalg.norm(mask_center - image_center)  # Euclidean distance
        

    #Sort masks by euclidean distance then by area 
    masks.sort(key=get_mask_area, reverse=True)  # Sort by area (largest first)
    masks.sort(key=get_center_distance)  # Then sort by closeness to center
    
    

    ####Spikes#####
    spike = 0 #Spike counter
    
    
    coordinate_dict = {}
    ####Body#####
    # Assign the first mask (most centered large object) as "body", rest as "spikes"
    # Fallback if fewer than 2 masks
    if len(masks) < 2:
        print("⚠️ Not enough masks returned. Skipping image.")
        return segmentation_map, {'x': 0, 'y': 0, 'w': 0, 'h': 0}
    x_body, y_body, w_body, h_body  = masks[1]['bbox']
    extend = 75 #Create box around the body to include the spikes
    
    #Make sure extended bounding box stays within image boundaries 
    #Calculate the extended bounding box and make sure it stays inside image boundaries
    x_xten = max(x_body-extend, 0)
    y_xten = max(y_body-extend, 0)
    w_xten = min(w_body+(2*extend), width-x_xten)
    h_xten = min(h_body+2*extend, height-y_xten)
    
    
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
                    
       
    
    print('mask generated')
      
    body_bbox = {'x': x_xten, 'y': y_xten, 'w': w_xten, 'h': h_xten}
    
    return segmentation_map, body_bbox
################ Workflow to postprocess input images #########   
def postprocess_mask(segmentation_map, body_bbox):

  height, width = segmentation_map.shape[:2]
  #Rectangle for where the body is
  x_xten = body_bbox.get('x') #x + w slices rows (vertical height)
  y_xten = body_bbox.get('y') #y + h slices columns (horizontal width)
  w_xten = body_bbox.get('w')
  h_xten = body_bbox.get('h')
  
  #Make sure staying within bounds
  x_xten = int(max(0, x_xten))
  y_xten = int(max(0, y_xten))
  x_max = int(min(width, x_xten + w_xten))
  y_max = int(min(height, y_xten + h_xten))
  
  
  #Everything outside the extended bounding box is black - essentially make a copy of just the inside bounding box
  
  new_seg_map = np.zeros_like(segmentation_map)
  new_seg_map[y_xten:y_max, x_xten:x_max] =  segmentation_map[y_xten:y_max, x_xten:x_max]
  
  #Count the spikes using cv2 connected components
  spike = 0
  spike_pixels = np.all(segmentation_map == [200, 200, 200], axis=-1) #where pixels match spike color 
  spike_mask = spike_pixels.astype(np.uint8) * 255
  spike_crop = spike_mask[y_xten:y_max, x_xten:x_max] #only include spikes within the bbox_body
  total_num_labels, all_im_labels = cv2.connectedComponents(spike_crop)
  num_spikes = int(total_num_labels - 1) #excluding background
  
  print('Num spikes:  ', num_spikes)
  
  return new_seg_map, num_spikes
################ Workflow to save input images #########   
def save_masks(im_path, new_seg_map, save_dir):
  
    filename = os.path.basename(im_path)
    os.makedirs(save_dir, exist_ok=True)
     

    #Add '_seg_ver2' before the extension
    filename_without_ext, ext = os.path.splitext(filename)
    new_filename = f"{filename_without_ext}_seg_ver2{ext}"
    

    #Save the segmentation mask
    mask_path = os.path.join(save_dir, new_filename)
    cv2.imwrite(mask_path, new_seg_map)
    print('saved', new_filename)
    
    
    
    return mask_path