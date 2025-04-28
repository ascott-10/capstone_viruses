import torch
import os

# ======== PATHS (change these easily per machine) ========

INPUT_DIR = '/path/to/your/data'   # (where your raw_and_segment_*.csv lives)
SAVE_DIR = '/path/to/your/results' # (where to save models, plots, etc.)

# ======== TRAINING SETTINGS ========

NUM_EPOCHS = 25
BATCH_SIZE = 16
LEARNING_RATE = 1e-4

# ======== DEVICE SETTING ========

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ======== OTHER SETTINGS ========

IMAGE_SIZE = (750, 750)  # (height, width) for resizing inputs
SEED = 42                # random seed if you want reproducibility
