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
    AUTO_SEGMENTED_MASKS_WT = '/home/ariellescott/Documents/capstone/segmented_ims/'
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

# ======== TRAINING SETTINGS ========

NUM_EPOCHS = 10
NUM_EPOCHS_CLASSIFY = 10
BATCH_SIZE = 4
LEARNING_RATE = 1e-4

# ======== DEVICE SETTING ========

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ======== OTHER SETTINGS ========

SUBSAMPLE = 25
IMAGE_SIZE = (256,256)
SEED = 42
NEW_CLASSIFY = True

# Color scheme for plotting
TRAIN_COLOR = 'mediumseagreen'
VAL_COLOR = 'darkgreen'
# General green-themed plotting colors
FONT_SIZE = 14  # or whatever size you want
FONT_SIZE_TITLE = 14
FONT_SIZE_LABEL = 12
FONT_SIZE_TICK = 10
COLOR_GENERAL_1 = "#4CAF50"       # green
COLOR_GENERAL_2 = "#81C784"       # lighter green
COLOR_MUTANT = "#f28cb1"          # default pink
COLOR_MUTANT_BRIGHT = "#ffb6c1"   # brighter pink
COLOR_WILDTYPE = "#66bb6a"        # green
SAVE_DPI = 300

# For classifier confusion matrix (mutant vs wildtype)
CONFUSION_MATRIX_CMAP = ["#F48FB1", "#66BB6A"]  # pink to green

# ======== PATHS TO SAVED MODELS ========
BEST_UNET_PATH = os.path.join(SAVE_DIR, "best_unet.pt")
RESNET_WEIGHT_PATTERN = os.path.join(SAVE_DIR, "resnet_weights_*.pth")

# ======== MODEL CLASSES ========
CLASS_NAMES = ["wildtype", "mutant"]
NUM_CLASSES = len(CLASS_NAMES)

# ======== OUTPUT FILES ========
COMPARE_CSV_FILENAME = "comparison_results.csv"
MORPHOLOGY_CSV_FILENAME = "all_component_areas.csv"

# ======== PLOTTING TOGGLE ========
PLOT_RESULTS = True

# ======== TRANSFORM NORMALIZATION (optional) ========
NORMALIZE_MEAN = 0.5
NORMALIZE_STD = 0.5
