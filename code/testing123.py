################ Import Libraries ################
import os
import glob
import torch
import pandas as pd
import matplotlib.pyplot as plt

from config import *
from unet_training_setup import (load_segmented_ims,
    combine_dfs,
    load_images_from_dataframe, 
    train_split, 
    create_segmentation_tensor_dataset, 
    create_dataloader
)
from unet_training import (
    setup_training, 
    train_one_epoch, 
    validate_one_epoch, 
    save_best_model, 
    plot_loss
)
from unet_model import UNet

################ Setup ################

# Load conversion dictionary if needed
   
convert_df = pd.read_excel('/home/ascott10/documents/projects/capstone_viruses/data/Segmentation_Progress.xlsx', usecols = [0,1])  # <-- Use read_excel

image_labels_muts, image_filepaths_muts, image_ids_muts = load_segmented_ims(RAW_IMS_MUT)
image_labels_wts, image_filepaths_wts, image_ids_wts = load_segmented_ims(RAW_IMS_WT)
segmented_image_labels_muts, segmented_image_filepaths_muts, segmented_image_ids_muts = load_segmented_ims(SEGMENTED_MASKS_MUT, convert_df)
segmented_image_labels_wts, segmented_image_filepaths_wts, segmented_image_ids_wts = load_segmented_ims(SEGMENTED_MASKS_WT, convert_df)

all_files_df = combine_dfs(convert_df, RAW_IMS_MUT, RAW_IMS_WT, SEGMENTED_MASKS_MUT, SEGMENTED_MASKS_WT, SAVE_DIR)

################## Subsample 25 mutants, 25 wildtypes ##################