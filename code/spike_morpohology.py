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

def ground_truth_morph(df_final, segmented_path_label, class_label):
    
    proc_labels = []
    proc_file_paths = []

    df_input = df_final[[segmented_path_label, class_label]]

    
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

    
    return df

from scipy.stats import linregress, pearsonr
import pandas as pd
from config import *

def plot_spike_area(ground_truth_morph, SAVE_DIR):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib import cm

    paired = cm.get_cmap('Paired', 6)
    class_colors = {
        'mutant': paired(0),
        'wildtype': paired(1)
    }

    manual_reg_color = 'black'

    fig, ax = plt.subplots(figsize=(7, 6))

    sns.scatterplot(
        data=ground_truth_morph,
        y='Spike_Count',
        x='Total_Spike_Area_Per_Particle',
        hue='Class',
        style='Class',
        palette=class_colors,
        s=60,
        alpha=0.9,
        ax=ax
    )
    sns.regplot(
        data=ground_truth_morph,
        y='Spike_Count',
        x='Total_Spike_Area_Per_Particle',
        scatter=False,
        color=manual_reg_color,
        label='Manual Regression',
        ax=ax
    )

    ax.set_title('Predicted Spike Count vs. Calculated Spike Area')
    ax.set_xlabel('Total Spike Area')
    ax.set_ylabel('Predicted Spike Count')
    ax.grid(True)

    plt.tight_layout()
    save_path = os.path.join(SAVE_DIR, "spike_area_plot.png")
    plt.savefig(save_path)
    plt.show()



def calculate_spike_stats(ground_truth_morph_df, plot_yes=True):
    from scipy.stats import linregress, pearsonr

    results = []

    for cls in ['mutant', 'wildtype']:
        df_subset = ground_truth_morph_df[ground_truth_morph_df['Class'] == cls]

        y_area = df_subset['Total_Spike_Area_Per_Particle']
        x_pred = df_subset['Spike_Count']

        linreg_pred = linregress(x_pred, y_area)
        r_pred, p_pred = pearsonr(x_pred, y_area)

        results.append({
            'Class': cls,
            'Type': 'Predicted',
            'Slope': linreg_pred.slope,
            'Intercept': linreg_pred.intercept,
            'R-squared': linreg_pred.rvalue ** 2,
            'Pearson r': r_pred,
            'p-value': p_pred
        })

    regression_by_class = pd.DataFrame(results)
    print(regression_by_class[['Class', 'Type', 'Slope', 'R-squared', 'Pearson r', 'p-value']])

    if plot_yes:
        plot_spike_area(ground_truth_morph_df, SAVE_DIR)

    return regression_by_class
