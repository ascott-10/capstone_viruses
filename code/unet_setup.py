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
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import TensorDataset, DataLoader
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_dir = '/home/ascott10/documents/projects/capstone_viruses/data'

# Check version of Pytorch
print(torch. __version__)


def load_raw_images(input_path):
    """User inputs the folder where their segmented images are stored"""
    
    #Retrieve files
    image_filepaths = []
    image_labels = []
    image_ids = []

    #for each file in the inputs add path to storage list:

    for files in os.listdir(input_path):
        if files.endswith('png'):
            if 'A2_MHV' in files:
                label = 'mutant'
            elif 'MHVWT' in files:
                label = 'wildtype'
            image_labels.append(label)
            image_filepaths.append(os.path.join(input_path,files))
            image_ids.append(Path(files).stem)
            
    return image_ids, image_filepaths, image_labels

def load_trained_images(input_path):
    
    #Retrieve files
    image_filepaths = []
    image_labels = []
    image_ids = []

    #for each file in the inputs add path to storage list:

    for files in os.listdir(input_path):
        if files.endswith('png'):
            if 'A2_MHV' in files:
                label = 'mutant'
            elif 'MHVWT' in files:
                label = 'wildtype'
            image_labels.append(label)
            image_filepaths.append(os.path.join(input_path,files))
            image_ids.append(Path(files).stem)
            
    return image_ids, image_filepaths, image_labels

def combine_dfs(save_dir, input_path_raw_mutant, input_path_raw_wt, input_path_segmented_1, input_path_segmented_2 = None):
    image_ids_mut, image_filepaths_mut, image_labels_mut = load_raw_images(input_path_raw_mutant)
    image_ids_wt, image_filepaths_wt, image_labels_wt = load_raw_images(input_path_raw_wt)
    seg_image_ids, seg_image_filepaths, seg_image_labels = load_trained_images(input_path_segmented_1)

    all_raw_files_df = pd.DataFrame([(image_ids_mut + image_ids_wt), 
                                (image_filepaths_mut + image_filepaths_wt), 
                                (image_labels_mut + image_labels_wt)],
                                index= ['im_id', 'file_path', 'class']).T

    all_segmented_files_df = pd.DataFrame([seg_image_ids,seg_image_filepaths, seg_image_labels],
                                        index= ['im_id', 'segmented_file_path', 'class']).T

    all_segmented_files_df['im_id'] =all_segmented_files_df['im_id'].str.replace('_seg_ver2','')

    all_files_df = all_raw_files_df.merge(all_segmented_files_df, on = ['im_id', 'class'])

    #Save to csv
    file_name = f"raw_and_segment_{timestamp}.csv"
    output_path = os.path.join(save_dir, file_name)
    all_files_df.to_csv(output_path, index=False)

    return all_files_df

def show_filepaths(raw_and_segmented_df):
    #Randomly select image_ids from both train and test set
    import random
    quick_rand_list = list(raw_and_segmented_df['im_id'])
    quick_rand = random.sample(quick_rand_list, 10)



    #Plot these random 10 images from the whole data set, both sides
    fig, ax = plt.subplots(2, len(quick_rand), figsize=(20,5))
    for i in range(len(quick_rand)):
        rand_id = quick_rand[i]
        raw_path = raw_and_segmented_df.loc[raw_and_segmented_df['im_id'] == rand_id, 'file_path'].values[0]
        seg_path = raw_and_segmented_df.loc[raw_and_segmented_df['im_id'] == rand_id, 'segmented_file_path'].values[0]
        raw_img = cv2.imread(raw_path, cv2.IMREAD_GRAYSCALE)
        seg_img = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)

        ax[0,i].imshow(raw_img,cmap = 'gray')
        ax[1,i].imshow(seg_img,cmap = 'gray')


    plt.setp(ax, xticks=[], yticks=[])

    plt.show()