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
################ Workflow to postprocess input images images #########
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
  spike_crop = spike_mask[y_xten:y_xten+h_xten, x_xten:x_xten+w_xten] #only include spikes within the bbox_body
  total_num_labels, all_im_labels = cv2.connectedComponents(spike_crop)
  num_spikes = int(total_num_labels - 1) #excluding background
  
  print('Num spikes:  ', num_spikes)
  
  
  
  
   
  return new_seg_map, num_spikes

def save_masks(im_path, new_seg_map):
  
    filename = os.path.basename(im_path)
    save_dir = "/home/ariellescott/Documents/capstone/capstone-viruses/data/output/sam_segment_processed/"
    os.makedirs(save_dir, exist_ok=True)
     

    #Add '_seg_ver2' before the extension
    filename_without_ext, ext = os.path.splitext(filename)
    new_filename = f"{filename_without_ext}_seg_ver2{ext}"
    

    #Save the segmentation mask
    mask_path = os.path.join(save_dir, new_filename)
    cv2.imwrite(mask_path, new_seg_map)
    
    return mask_path
    
    




