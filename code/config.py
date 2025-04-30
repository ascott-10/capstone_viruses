import torch
import os
import pandas as pd

#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ======== SYSTEM SELECTION ========

# Manually tell the script if you're working at home or at school
ENVIRONMENT = "home"  # options: "home" or "school"

# ======== PATHS (based on environment) ========

if ENVIRONMENT == "school":
    RAW_IMS_WT = '/home/ariellescott/Documents/capstone/raw_ims/raw_wt/'
    RAW_IMS_MUT = '/home/ariellescott/Documents/capstone/raw_ims/raw_mutant/'
    SEGMENTED_MASKS_WT = '/home/ariellescott/Documents/capstone/segmented_ims/'
    SEGMENTED_MASKS_MUT = '/home/ariellescott/Documents/capstone/segmented_ims/'
    AUTO_SEGMENTED_MASKS_MUT = '/home/ariellescott/Documents/capstone/segmented_ims/'
    AUTO_SEGMENTED_MASKS_MUT = '/home/ariellescott/Documents/capstone/segmented_ims/'
    SAVE_DIR = '/home/ariellescott/Documents/capstone/capstone_viruses/results/'
    INPUT_DIR = '/home/ariellescott/Documents/capstone/capstone_viruses/results/'
    convert_df = pd.read_excel('/home/ariellescott/Documents/capstone/capstone_viruses/results/Segmentation_Progress.xlsx', usecols = [0,1]) 

elif ENVIRONMENT == "home":
    RAW_IMS_WT = '/home/ascott10/documents/projects/capstone_viruses/raw_images/wt/'
    RAW_IMS_MUT = '/home/ascott10/documents/projects/capstone_viruses/raw_images/mutant/'
    SEGMENTED_MASKS_WT = '/home/ascott10/documents/projects/capstone_viruses/segmented_images/wildtype_manual_correction/'
    SEGMENTED_MASKS_MUT = '/home/ascott10/documents/projects/capstone_viruses/segmented_images/mutant_manual_correction/'
    AUTO_SEGMENTED_MASKS_WT = '/home/ascott10/documents/projects/capstone_viruses/segmented_images/segment_ver2/'
    AUTO_SEGMENTED_MASKS_MUT = '/home/ascott10/documents/projects/capstone_viruses/segmented_images/segment_ver2/'
    SAVE_DIR = '/home/ascott10/documents/projects/capstone_viruses/data/'
    INPUT_DIR = '/home/ascott10/documents/projects/capstone_viruses/data/'
    convert_df = pd.read_excel('/home/ascott10/documents/projects/capstone_viruses/data/Segmentation_Progress.xlsx', usecols = [0,1]) 


else:
    raise ValueError('"Must be "home" or "school"')

MANUAL_ALL = [SEGMENTED_MASKS_WT, SEGMENTED_MASKS_MUT]
AUTO_ALL   = [AUTO_SEGMENTED_MASKS_WT, AUTO_SEGMENTED_MASKS_MUT]
# ======== TRAINING SETTINGS ========

NUM_EPOCHS = 15
BATCH_SIZE = 4
LEARNING_RATE = 1e-4

# ======== DEVICE SETTING ========

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ======== OTHER SETTINGS ========

SUBSAMPLE = 60
IMAGE_SIZE = (256,256)
SEED = 42
NEW_CLASSIFY = True
