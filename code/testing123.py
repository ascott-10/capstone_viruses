################ Import Libraries ################
import os
import glob
import torch
import pandas as pd
import matplotlib.pyplot as plt

from code.config import *
from code.unet_training_setup import (load_segmented_ims, subsample_test,
    combine_dfs,
    load_images_from_dataframe, 
    train_split, 
    create_segmentation_tensor_dataset, 
    create_dataloader
)
from code.unet_training import (
    setup_training, 
    train_one_epoch, 
    validate_one_epoch, 
    save_best_model, 
    plot_loss
)
from code.unet_model import UNet

################ Setup ################

#Get all file paths and labels into one dataframe
all_df = combine_dfs(convert_df, RAW_IMS_MUT, RAW_IMS_WT, SEGMENTED_MASKS_MUT, SEGMENTED_MASKS_WT, SAVE_DIR)

#For testing, only using a portion of the images
if SUBSAMPLE is not None:
    final_df = subsample_test(final_df=all_df, SUBSAMPLE=SUBSAMPLE)
else:
    final_df = all_df