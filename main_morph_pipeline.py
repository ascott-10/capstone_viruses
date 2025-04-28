
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

from code.morphology import extract_morphology, make_morphology_df, display_individual_morph_stats, get_base_file_map, compare_methods, compare_methods_plotting, compare_methods_stats
#segmented_image_path_sam = '/home/ascott10/documents/projects/capstone_viruses/segmented_images/segment_ver2'
#df_summary_sam = make_morphology_df(segmented_image_path_sam)

############# Get stats for manually curated pipeline ###############

#segmented_image_path_manual = '/home/ascott10/documents/projects/capstone_viruses/segmented_images/sam_segment_ver1'
#df_summary_manual = make_morphology_df(segmented_image_path_manual)

############# Compare stats ###############

#Input directories for automatic and manual segmentation, assume manual is gold standard
manual_method_dir = '/home/ascott10/documents/projects/capstone_viruses/segmented_images/sam_segment_ver1'
manual_method_suffix = r'_seg$'
automatic_method_dir = '/home/ascott10/documents/projects/capstone_viruses/segmented_images/segment_ver2'
automatic_method_suffix = r'_seg_ver2$'

#Make comparison df and save
df_compare = compare_methods(manual_method_dir, automatic_method_dir, manual_method_suffix, automatic_method_suffix)

save_path = os.path.join(save_dir, f"df_compare_{timestamp}.csv")
df_compare.to_csv(save_path, index=False)

stat, p, summary = compare_methods_stats(save_dir, plot_yes = 'yes')



for cls in ['mutant', 'wildtype']:
    pe = summary[cls]['Percent_Error']
    ad = summary[cls]['Abs_Difference']
    print(f"\n{cls.title()} —")
    print(f"  Percent Error: Mean = {pe[0]:.2f}%, 95% CI = [{pe[1]:.2f}%, {pe[2]:.2f}%]")
    print(f"  Abs Difference: Mean = {ad[0]:.2f} px, 95% CI = [{ad[1]:.2f}, {ad[2]:.2f}]")