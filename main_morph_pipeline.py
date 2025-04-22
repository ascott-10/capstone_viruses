
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


import os
import pandas as pd
import pathlib
from pathlib import Path

import torch
from torchvision import models
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import train_test_split

import paramiko #For remote GPU support

############# Get stats for automatics (SAM) pipeline ###############


from code.morphology import extract_morphology, make_morphology_df, display_morph_stats, get_base_file_map, compare_methods
segmented_image_path_sam = '/home/ascott10/documents/projects/capstone_viruses/segmented_images/segment_ver2'
df_summary_sam = make_morphology_df(segmented_image_path_sam)

############# Get stats for manually curated pipeline ###############

segmented_image_path_manual = '/home/ascott10/documents/projects/capstone_viruses/segmented_images/sam_segment_ver1'
df_summary_manual = make_morphology_df(segmented_image_path_manual)

############# Compare stats ###############

segment_manual_dir = '/home/ascott10/documents/projects/capstone_viruses/segmented_images/sam_segment_ver1'
method_auto_dir = '/home/ascott10/documents/projects/capstone_viruses/segmented_images/segment_ver2'
df_compare = compare_methods(method_auto_dir, segment_manual_dir, suffix_a=r'_seg$', suffix_b=r'_seg_ver2$')

df_compare.to_csv('/home/ascott10/documents/projects/capstone_viruses/results/df_compare.csv')