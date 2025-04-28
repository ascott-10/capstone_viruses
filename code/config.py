import torch
import os

# ======== PATHS (change these easily per machine) ========

RAW_IMS_WT = '/home/ariellescott/Documents/capstone/capstone-viruses-1/testing/testing_data/testing_raw_images/testing_raw_wt/'
RAW_IMS_MUT = '/home/ariellescott/Documents/capstone/capstone-viruses-1/testing/testing_data/testing_raw_images/testing_raw_mutant/'
SEGMENTED_MASKS = '/home/ariellescott/Documents/capstone/capstone-viruses-1/testing/testing_data/testing_output/sam_segment_processed_testing/'

SAVE_DIR = '/home/ariellescott/Documents/capstone/capstone_viruses-2/results/'
INPUT_DIR = '/home/ascott10/documents/projects/capstone_viruses/data'

# ======== TRAINING SETTINGS ========

NUM_EPOCHS = 25
BATCH_SIZE = 16
LEARNING_RATE = 1e-4

# ======== DEVICE SETTING ========

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ======== OTHER SETTINGS ========

IMAGE_SIZE = (750, 750) 
SEED = 42               