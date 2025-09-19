
#### Setup ####
 
#import libraries
import sys, os
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import random
import math

import cv2
from PIL import Image
from skimage.measure import regionprops, label
from scipy.ndimage import gaussian_filter1d

import matplotlib.pyplot as plt
import seaborn as sns

from config import *

#### Functions #### 


def extract_coordinates(segmented_image_path):
    """This function will take a segmented image file path as input, remap the pixels and return the image and a dataframe with body and spike coordinates
    Usage: image, body_df, spike_df = extract_coordinates(segmented_image_path)"""
    
    #read in the image, error checking for body and spikes
    image = cv2.imread(segmented_image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Error: Image not found or cannot be loaded.")
    
    body_mask = (image == 150).astype(np.uint8)
    spikes_mask = (image == 200).astype(np.uint8)

    if not np.any(body_mask):
        raise ValueError("No virus body detected in the image.")
    if not np.any(spikes_mask):
        print("No spikes detected in the image.")

    body_labels = label(body_mask)
    spike_labels = label(spikes_mask)

    body_props = regionprops(body_labels)
    spike_props = regionprops(spike_labels)

    if not body_props:
        body_data = [{
            "ID": 0,
            "Area": 0,
            "Perimeter": 0,
            "Major Axis Length": 0,
            "Minor Axis Length": 0
        }]
    else:
        body_data = [{
            "ID": i + 1,
            "Area": prop.area,
            "Perimeter": prop.perimeter,
            "Centroid Y": prop.centroid[0], #row
            "Centroid X": prop.centroid[1] #columns
        } for i, prop in enumerate(body_props)]

    if not spike_props:
        spike_data = [{
            "ID": 0,
            "Area": 0,
            "Perimeter": 0,
            "Centroid Y": 0,
            "Centroid X": 0
        }]
    else:
        spike_data = [{
            "ID": i + 1,
            "Area": prop.area,
            "Perimeter": prop.perimeter,
            "Centroid Y": prop.centroid[0],
            "Centroid X": prop.centroid[1]
        } for i, prop in enumerate(spike_props)]

    body_df = pd.DataFrame(body_data)
    spike_df = pd.DataFrame(spike_data)
    
    return image, body_df, spike_df

def generate_perimeter(image, kernel_size=(7,7), ksize=(7,7)):
    """This function will take the image as input and optional parameters of kernel size and ksize to perform erosion and dilation, to generate virus perimeter
    Usage: perimeter_mask, perim_x_ls, perim_y_ls = generate_perimeter(image)"""

    #Remap image so background =0, virus particle =1, spike =2
    mapped = image.copy()
    mapped[mapped == 150] = 1
    mapped[mapped == 200] = 2
    
    #Focus on virus particle only
    particle_only = (mapped == 1).astype(np.uint8)

    #Define the kernel matrix
    kernel = np.ones(kernel_size, np.uint8) 
    
    #2 rounds of dilation
    img_dilation = cv2.dilate(particle_only,kernel_size,iterations=2) 

    #4 rounds of smoothing - 2D moving average
    img_smooth_1 = cv2.blur(img_dilation.astype(np.float32),ksize=ksize)
    img_smooth_2 = cv2.blur(img_smooth_1,ksize=ksize)
    img_smooth_3 = cv2.blur(img_smooth_2,ksize=ksize)
    img_smooth_4 = cv2.blur(img_smooth_3,ksize=ksize)
    
    #Binarization
    img_binarized = (img_smooth_4 > 0.5).astype(np.uint8)

    #2 rounds of erosion
    img_erosion = cv2.erode(img_binarized,kernel,iterations=2)

    #dilate once
    dilated_once = cv2.dilate(img_erosion,kernel, iterations=1)

    #get perimeter
    perimeter_mask = dilated_once - img_erosion

    #returns parallel (x,y) arrays of all perimeter pixels
    perim_y_ls, perim_x_ls = np.where(perimeter_mask > 0) 

    return perimeter_mask, perim_x_ls, perim_y_ls

def get_body_angle(perim_x_ls, perim_y_ls, body_center_x, body_center_y):
    """This function takes the perimeter mask as input and body centroid coordinates and returns the distance and angle for each perimeter pixel
    Usage: dist_ls, theta_rads_ls, theta_degrees_ls = get_body_angle(perim_x_ls, perim_y_ls, body_center_x, body_center_y)"""
    
    #Use distance formula to calculate distance from centroid of body to each perimeter pixel
    dist_ls = np.sqrt((perim_x_ls - body_center_x)**2 + (perim_y_ls - body_center_y)**2)

    #Use np.arctan function to calculate angle from x-axis for each perimeter pixel (in radians), convert to degrees
    theta_rads_ls = np.arctan2((perim_y_ls - body_center_y),(perim_x_ls - body_center_x))
    theta_degrees_ls = np.degrees(theta_rads_ls)

    return dist_ls, theta_rads_ls, theta_degrees_ls

def get_spike_angle(center_spike_x_ls, center_spike_y_ls, body_center_x, body_center_y):
    """This function takes the body centroid and array of spike centroids and calculates the angle of each spike relative to the body center
    Usage:spike_angles_this_image_ls = get_spike_angle(center_spike_x_ls, center_spike_y_ls, body_center_x, body_center_y)"""

    #Calculate angles for all spikes
    spike_angles_this_image_ls = []
    for i in range(len(center_spike_x_ls)):
        #Iterate through list of spikes
        theta_rads_to_spike = np.arctan2((center_spike_y_ls[i] - body_center_y), (center_spike_x_ls[i] - body_center_x))
        spike_angles_this_image_ls.append(theta_rads_to_spike)

    return spike_angles_this_image_ls

def calculate_interpixel(body_theta_dict, spike_theta_dict):
    """This function takes a dictionary of spike angles calculates angular range, then computes the inter-pixel distance 
    Usage: angular_distance_ls, angle_dist_dict = calculate_interpixel(body_theta_dict, spike_theta_dict)"""

    angular_distance_ls = []
    angle_dist_dict = {}

    #first sort the virus body particle angles
    sorted_spike_angles = sorted(spike_theta_dict.keys())
    sorted_body_angles = sorted(body_theta_dict.keys())
    num_body_angles = len(sorted_body_angles)

    #Loop through target (spike) angles
    for i in range(len(sorted_spike_angles)):
        for j in range(i+1,len(sorted_spike_angles)):

            spike_angle_1 = sorted_spike_angles[i]
            spike_angle_2 = sorted_spike_angles[j]

            #Find index of the angles (where the spike is located on the perimeter)
            idx1 = sorted_body_angles.index(spike_angle_1)
            idx2 = sorted_body_angles.index(spike_angle_2)

            #count distance
            dist_plus = (idx2 - idx1) % num_body_angles
            dist_min = (idx1 - idx2) % num_body_angles

            #shortest distance is smaller of counterclockwise and clockwise distances
            angular_distance=min(dist_plus,dist_min)

            angular_distance_ls.append(angular_distance)
            angle_dist_dict[(spike_angle_1,spike_angle_2)] = (angular_distance)


    return angular_distance_ls, angle_dist_dict

def plot_results(angular_distance_ls, image_for_line, perimeter_mask):
    """This function takes the list of angular distances and the original, mapped image and returns a plot of the spike historgram
      Usage: plot_results(angular_distance_ls, image_for_line, perimeter_mask)"""
    
    

    #make histogram
    

    plt.figure(figsize=(10,10))

    plt.subplot(2,2,1)
    plt.imshow(image_for_line)
    plt.title("Particle with Spike Identification")
    plt.axis("off") 

    plt.subplot(2,2,2)
    plt.imshow(perimeter_mask, cmap = 'gray')
    plt.title("Perimeter Mask")
    plt.axis("off") 

    if len(angular_distance_ls) > 0:
        max_distance = max(angular_distance_ls)
        counts, bins = np.histogram(angular_distance_ls, bins=300, range=(0, max_distance))

        #Normalize distribution by sqrt(N)
        num_spikes = len(spike_theta_dict)
        counts_norm = counts / np.sqrt(num_spikes)

        smooth_counts = gaussian_filter1d(counts_norm, sigma=2)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        plt.subplot(2,1,2)
        plt.plot(bin_centers, smooth_counts, linewidth=2)
        plt.xlim(0, max_distance)
        plt.xlabel("Spike Inter-Pixel Perimeter Distance (nm)")
        plt.title("Spike Distance Distribution")

    plt.tight_layout()
    plt.show()
    


#### Testing One ####

im_path = '/home/ascott10/documents/projects/capstone_viruses/segmented_images/mutant_manual_correction/muimage255_corrected.png'


#Extract morphology
image, body_df, spike_df = extract_coordinates(im_path)

#Generate (x,y) of body
body_center_x = body_df["Centroid X"].astype(int).values[0]
body_center_y= body_df["Centroid Y"].astype(int).values[0]
center_spike_x_ls = body_df["Centroid X"].astype(int).values[0]
center_spike_y_ls= body_df["Centroid Y"].astype(int).values[0]

#Generate parallel lists of (x,y) for each spike (if spike_df exists)
if ((spike_df["Centroid X"].astype(int).iloc[0] > 0) and (spike_df["Centroid Y"].astype(int).iloc[0] > 0)):
    center_spike_x_ls = spike_df["Centroid X"].astype(int).tolist()
    center_spike_y_ls= spike_df["Centroid Y"].astype(int).tolist()

#Generate perimeter mask, body angles and spike angles
perimeter_mask, perim_x_ls, perim_y_ls = generate_perimeter(image)
dist_ls, theta_rads_ls, theta_degrees_ls = get_body_angle(perim_x_ls, perim_y_ls, body_center_x, body_center_y)
spike_angles_this_image_ls = get_spike_angle(center_spike_x_ls, center_spike_y_ls, body_center_x, body_center_y)


#If body angles match spike angles --> this means spikes are in line with perimeter pixels
theta_rads_round = np.round(theta_rads_ls, 2)
theta_rads_spike_round = np.round(spike_angles_this_image_ls,2)
image_for_line = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

#make dictionary for later {theta: (x,y)}
body_theta_dict = {theta:(x,y) for theta,x,y in zip(theta_rads_round, perim_x_ls, perim_y_ls)}
spike_theta_dict = {}

for i in range(0,len(theta_rads_spike_round)):
    spike_angle = theta_rads_spike_round[i]
    if spike_angle in body_theta_dict:
        perim_coord = body_theta_dict[spike_angle]
        spike_theta_dict[spike_angle] = perim_coord
        cv2.line(image_for_line,(body_center_x, body_center_y),(center_spike_x_ls[i], center_spike_y_ls[i]), (0, 0, 100), 3)

#calculate the angular distance/inter-pixel distance and make a histogram
angular_distance_ls, angle_dist_dict = calculate_interpixel(body_theta_dict, spike_theta_dict)

#Plot results
plot_results(angular_distance_ls, image_for_line, perimeter_mask)

#### Testing Many ####

manually_segmented_mutant_dir = "/home/ascott10/documents/projects/capstone_viruses/segmented_images/mutant_manual_correction"
manually_segmented_wt_dir = "/home/ascott10/documents/projects/capstone_viruses/segmented_images/wildtype_manual_correction"
num_samples = 10

all_manual = [os.path.join(directories, files) 
            for directories in (manually_segmented_mutant_dir, manually_segmented_wt_dir)
            for files in os.listdir(directories)
            if files.lower().endswith((".tif",".png",".jpg"))]

for image_file_path in random.sample(all_manual, num_samples):
    #Extract morphology
    image, body_df, spike_df = extract_coordinates(image_file_path)

    #Generate (x,y) of body
    body_center_x = body_df["Centroid X"].astype(int).values[0]
    body_center_y= body_df["Centroid Y"].astype(int).values[0]
    center_spike_x_ls = body_df["Centroid X"].astype(int).values[0]
    center_spike_y_ls= body_df["Centroid Y"].astype(int).values[0]

    #Generate parallel lists of (x,y) for each spike (if spike_df exists)
    if ((spike_df["Centroid X"].astype(int).iloc[0] > 0) and (spike_df["Centroid Y"].astype(int).iloc[0] > 0)):
        center_spike_x_ls = spike_df["Centroid X"].astype(int).tolist()
        center_spike_y_ls= spike_df["Centroid Y"].astype(int).tolist()

        #Generate perimeter mask, body angles and spike angles
        perimeter_mask, perim_x_ls, perim_y_ls = generate_perimeter(image)
        dist_ls, theta_rads_ls, theta_degrees_ls = get_body_angle(perim_x_ls, perim_y_ls, body_center_x, body_center_y)
        spike_angles_this_image_ls = get_spike_angle(center_spike_x_ls, center_spike_y_ls, body_center_x, body_center_y)


        #If body angles match spike angles --> this means spikes are in line with perimeter pixels
        theta_rads_round = np.round(theta_rads_ls, 2)
        theta_rads_spike_round = np.round(spike_angles_this_image_ls,2)
        image_for_line = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        #make dictionary for later {theta: (x,y)}
        body_theta_dict = {theta:(x,y) for theta,x,y in zip(theta_rads_round, perim_x_ls, perim_y_ls)}
        spike_theta_dict = {}

        for i in range(0,len(theta_rads_spike_round)):
            spike_angle = theta_rads_spike_round[i]
            if spike_angle in body_theta_dict:
                perim_coord = body_theta_dict[spike_angle]
                spike_theta_dict[spike_angle] = perim_coord
                cv2.line(image_for_line,(body_center_x, body_center_y),(center_spike_x_ls[i], center_spike_y_ls[i]), (0, 0, 100), 3)

        #calculate the angular distance/inter-pixel distance and make a histogram
        angular_distance_ls, angle_dist_dict = calculate_interpixel(body_theta_dict, spike_theta_dict)

        #Plot results
        spike_count = len(spike_theta_dict)
        for _ in range(spike_count):
            angular_distance_ls.append(0) #give each spike a zero-distance
        plot_results(angular_distance_ls, image_for_line, perimeter_mask)
