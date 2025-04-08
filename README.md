# Capstone Project

# Description

## Repository Structure

- `code/`: Code for loading data and modeling
- `data/`: Raw image data
- `notebooks/`: Jupyter notebooks
- `main.py`: Main runner script

# Getting Started

## Installation

git clone https://github.com/ascott-10/capstone-viruses.git
cd capstone-viruses
pip install -r requirements.txt

## Other items needed

- Follow installation instructions for Segment Anything model from https://github.com/facebookresearch/segment-anything/
- Downloaded the checkpoint .pth file for the 'sam_vit_h_4b8939' model to stay consistent with the model used in this project
- Place the model in the appropriate directory for your project

## System Requirements (Project tested on):

- Ubuntu 22.04
- Python 3.12
- CUDA 12.4
- Pytorch 2.1
- NVIDIA GPU (RTX 3090)

# Usage Instructions


## Setup

### 1. Set up the file inputs

First, the user should have a directory of raw images. This project used EM images of viruses, split into two groups. All images were 750x750, with the virus body centered in the middle. 

in input_files.py, the user can enter their file paths or setup the prompts to users enter it later. 
This code will output a dataframe of filepaths and classes that will be called on later. It also creates a csv of this information, can change the filename here also. 

When running main.py, the user will be prompted if they will be using the default filepaths, which can be hardcoded into input_files to avoid having to type them on the command line.

### 2. Configure SAM

Next, make sure the SAM model (in this case sam_vit_h_4b8939.pth) has been downloaded. sam_configure can be modified to input the path. 

In main.py, configure_sam.download_sam() is called once to download the model and returns the model sam. With sam as input, configure_sam.custom_mask(sam) is the place to customize all sam inputs. This generates the mask generator object to be used on the images.

### 3. Segmentation

In the next steps, the user applies the segmentation masks to their images. The images are loaded from their df made in earlier steps in main.py

To get the segmentation process started, in main.py, the user first inputs their dataframe of filepaths (or it is automatically loaded) and which files in the load_image module. This outputs a list of images.

This list is then used in the next function, generate_masks, along with the mask_generator that was created previously. The segmentation will run for each image in the list.

During the segmentation, function will use SAM to generate the masks. It then uses the properties from SAM to find each mask's area and the euclidean distance from the center. It sorts the masks by center distance and area. The code then generalizes to say the largest mask is the background. The second largest mask is the body, and the rest are spikes.The masks are colored accordingly; here they are assigned on a grayscale.

Next a box is drawn around the body based on an extended distance of 75, which should also generally include the spikes. The first mask segmentation mask would have included anything not a body as a spike, but instead these will be set as "not a spike, not a body, not a background". 

Optionally, a rectangle is drawn to identify where the extended bounding box is just for visualization and testing purposes is, but for now this part is commented out. 

The output of generate_masks are the masks that correspond to the image inputs ('segmentation_map), which, in the main function, are then fed into display_images for a sample visualization. It also returns the coordinates for the bounding box of the body's mask, which will be used in the post-processing of the mask.

In the main function, the outputs are being appending to lists in order to be used in iterations since the functions are generally built to handle single inputs. 

The next task is to then post-process the mask; that is, to remove the area that is not centered around the virus body and its associated spikes. The inputs are the segmentation map, or the complete mask, and the bounding box which includes the body and its spikes. The outputs of this function are a new mask with the entire background roughly blacked out as well as a predicted number of spikes.

Finally, the original image, the pre-processed mask and the post-processed mask along with the predicted number of spikes is displayed.

Within the main function, the user can choose to display how many images to use, or can choose specific rows from their dataframe. 



## Citations 

Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Dollár, and Ross Girshick. Segment Anything. arXiv preprint arXiv:2304.02643, 2023.


