import os
import pandas as pd
import numpy as np
import pathlib
from pathlib import Path

import torch
from torchvision import models
import cv2
import PIL
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
import skimage
from skimage.measure import regionprops, label
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd

import glob
import os


from scipy.stats import mannwhitneyu
import scipy.stats as stats

def extract_morphology(segmented_image_path):
    image = cv2.imread(segmented_image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        raise ValueError("Error: Image not found or cannot be loaded.")
    
    # extract virus body and spikes 
    body_mask = (image == 150)
    spikes_mask = (image == 200)
    
    if not np.any(body_mask):
        raise ValueError("Error: No virus body detected in the image.")
    if not np.any(spikes_mask):
        print("Warning: No spikes detected in the image.")
    
    body_mask = body_mask.astype(np.uint8)
    spikes_mask = spikes_mask.astype(np.uint8)
    body_labels = label(body_mask)
    spike_labels = label(spikes_mask)
    
    # measure properties of virus body
    body_props = regionprops(body_labels)
    
    # measure properties of spikes
    spike_props = regionprops(spike_labels)
    
    if not body_props:
      body_data = [{
          "ID": 0,
          "Area": 0,
          "Perimeter":0,
          "Major Axis Length": 0,
          "Minor Axis Length": 0}]
    else:
      body_data = [{
          "ID": i + 1,
          "Area": prop.area,
          "Perimeter": prop.perimeter,
          "Major Axis Length": prop.major_axis_length,
          "Minor Axis Length": prop.minor_axis_length
        # Circularity
    } for i, prop in enumerate(body_props)]
    
    if not spike_props:
      spike_data = [{
          "ID": 0,
          "Area": 0,
          "Perimeter":0,
          "Major Axis Length": 0,
          "Minor Axis Length": 0}]
    else:
      spike_data = [{
          "ID": i + 1,
          "Area": prop.area,
          "Perimeter": prop.perimeter,
          "Centroid X": prop.centroid[0],
          "Centroid Y": prop.centroid[1],
          # Clustering
              # 2D distance map (not ideal)
              
          # Distance to nearest
    } for i, prop in enumerate(spike_props)]
    
    body_df = pd.DataFrame(body_data)
    spike_df = pd.DataFrame(spike_data)
    
    return body_df, spike_df

def morphology_df_with_spikes(segmented_image_path):
    im_path = segmented_image_path
    file_path_ls = []
    label_ls = []
    proc_labels = []
    proc_file_paths = []

    for root, _, files in os.walk(im_path):
        for file in files:
            if file.lower().endswith('.png'):
                file_path = os.path.join(root, file)
                file_path_ls.append(file_path)
                
                if 'MHVWT' in file:
                    file_label = 'wt'
                elif 'A2_MHV' in file:
                    file_label = 'mut'
                else:
                    file_label = 'unknown'
                
                label_ls.append(file_label)

    df_input = pd.DataFrame({'filepath': file_path_ls, 'class': label_ls})
    print(df_input)

    sum_spike_area = []
    avg_spike_perim = []
    sum_body_area = []
    avg_body_perim = []
    spike_counts = []

    for i in range(len(df_input)):
        file_path = df_input.iloc[i, 0]
        file_label = df_input.iloc[i, 1]
        
        try:
            body_df, spike_df = extract_morphology(file_path)
        except ValueError as e:
            print(f"Skipping {file_path}: {e}")
            continue
        
        sum_spike_area.append(np.sum(spike_df['Area']))
        avg_spike_perim.append(np.mean(spike_df['Perimeter']))
        sum_body_area.append(np.sum(body_df['Area']))
        avg_body_perim.append(np.mean(body_df['Perimeter']))

        # Spike count = 0 if no real spikes detected
        if len(spike_df) == 1 and spike_df.iloc[0]['ID'] == 0:
            spike_counts.append(0)
        else:
            spike_counts.append(len(spike_df))
        
        proc_file_paths.append(file_path)
        proc_labels.append(file_label)

    # Final dataframe after processing all images
    df = pd.DataFrame({
        'Total_Spike_Area_Per_Particle': sum_spike_area,
        'Total_Body_Area_Per_Particle': sum_body_area,
        'Spike_Count': spike_counts,
        'Class': proc_labels,
        'file_path': proc_file_paths
    })

    print(len(df))
    print(df.head())

    return df