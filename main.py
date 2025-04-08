#%%
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


import torch
device = "cuda" if torch.cuda.is_available() else "cpu" 
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import paramiko #For remote GPU support

print('Import libraries good')

############ Input Files #################

import input_files

#Raw image input paths
mut_path = '/home/ariellescott/Documents/capstone/data/raw_images/raw_mutant/'
wt_path = '/home/ariellescott/Documents/capstone/data/raw_images/raw_wt/'

df = input_files.input_files(mut_path,wt_path)

print(df.head())


############ Configure Sam #################

import sam_configure


mask_generator = sam_configure.download_sam()
sam_configure.sam_workflow(mask_generator,11)





