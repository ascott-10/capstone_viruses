import torch
import os

# ======== SYSTEM SELECTION ========

# Manually tell the script if you're working at home or at school
ENVIRONMENT = "home"  # options: "home" or "school"

# ======== PATHS (based on environment) ========

if ENVIRONMENT == "school":
    RAW_IMS_WT = '/home/ariellescott/Documents/capstone/raw_ims/raw_wt/'
    RAW_IMS_MUT = '/home/ariellescott/Documents/capstone/raw_ims/raw_mutant/'
    SEGMENTED_MASKS = '/home/ariellescott/Documents/capstone/segmented_ims/'
    SAVE_DIR = '/home/ariellescott/Documents/capstone/capstone_viruses-2/results/'
    INPUT_DIR = '/home/ariellescott/Documents/capstone/capstone_viruses-2/results/'

elif ENVIRONMENT == "home":
    RAW_IMS_WT = '/home/ascott10/documents/projects/capstone_viruses/raw_images/wt/'
    RAW_IMS_MUT = '/home/ascott10/documents/projects/capstone_viruses/raw_images/mutant/'
    SEGMENTED_MASKS = '/home/ascott10/documents/projects/capstone_viruses/segmented_images/segment_ver2/'
    SAVE_DIR = '/home/ascott10/documents/projects/capstone_viruses/data/'
    INPUT_DIR = '/home/ascott10/documents/projects/capstone_viruses/data/'

else:
    raise ValueError("❌ Invalid ENVIRONMENT. Must be 'home' or 'school'.")

# ======== TRAINING SETTINGS ========

NUM_EPOCHS = 25
BATCH_SIZE = 16
LEARNING_RATE = 1e-4

# ======== DEVICE SETTING ========

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ======== OTHER SETTINGS ========

IMAGE_SIZE = (750, 750)
SEED = 42
