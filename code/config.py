import torch
import os

#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ======== SYSTEM SELECTION ========

# Manually tell the script if you're working at home or at school
ENVIRONMENT = "home"  # options: "home" or "school"

# ======== PATHS (based on environment) ========

if ENVIRONMENT == "school":
    RAW_IMS_WT = '/home/ariellescott/Documents/capstone/raw_ims/raw_wt/'
    RAW_IMS_MUT = '/home/ariellescott/Documents/capstone/raw_ims/raw_mutant/'
    SEGMENTED_MASKS_WT = '/home/ascott10/documents/projects/capstone_viruses/segmented_images/wildtype_manual_correction/'
    SEGMENTED_MASKS_MUT = '/home/ascott10/documents/projects/capstone_viruses/segmented_images/mutant_manual_correction/'
    SAVE_DIR = '/home/ariellescott/Documents/capstone/capstone_viruses-2/results/'
    INPUT_DIR = '/home/ariellescott/Documents/capstone/capstone_viruses-2/results/'

elif ENVIRONMENT == "home":
    RAW_IMS_WT = '/home/ascott10/documents/projects/capstone_viruses/raw_images/wt/'
    RAW_IMS_MUT = '/home/ascott10/documents/projects/capstone_viruses/raw_images/mutant/'
    SEGMENTED_MASKS_WT = '/home/ascott10/documents/projects/capstone_viruses/segmented_images/wildtype_manual_correction/'
    SEGMENTED_MASKS_MUT = '/home/ascott10/documents/projects/capstone_viruses/segmented_images/mutant_manual_correction/'
    SAVE_DIR = '/home/ascott10/documents/projects/capstone_viruses/data/'
    INPUT_DIR = '/home/ascott10/documents/projects/capstone_viruses/data/'

else:
    raise ValueError("❌ Invalid ENVIRONMENT. Must be 'home' or 'school'.")

# ======== TRAINING SETTINGS ========

NUM_EPOCHS = 25
BATCH_SIZE = 4
LEARNING_RATE = 1e-4

# ======== DEVICE SETTING ========

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ======== OTHER SETTINGS ========

IMAGE_SIZE = (256,256)
SEED = 42
