
################ Import Libraries ################
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

import pandas as pd
import glob
import os

import os
import pandas as pd
import pathlib
from pathlib import Path

import torch
from torchvision import models
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import train_test_split

from datetime import datetime


import paramiko #For remote GPU support

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_dir = '/home/ascott10/documents/projects/capstone_viruses/results'

############# Get stats for automatics (SAM) pipeline ###############

from code.spike_morpohology import extract_morphology, morphology_df_with_spikes
#segmented_image_path_sam = '/home/ascott10/documents/projects/capstone_viruses/segmented_images/segment_ver2'
df_summary_sam = morphology_df_with_spikes(segmented_image_path_sam)

print(df_summary_sam)