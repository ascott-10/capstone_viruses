
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
    print(len(perim_x_ls))

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

def generate_spikes(image, body_center_x, body_center_y):
    """This function will take the image as input and return all distances from body centroid to spike pixel and angle to centroid
    Usage: spike_x_ls, spike_y_ls, dist_spike_ls, theta_spike_rads_ls, theta_spike_degrees_ls  = generate_spikes(image,body_center_x, body_center_y)"""

    #Remap image so background =0, virus particle =1, spike =2
    mapped = image.copy()
    mapped[mapped == 150] = 1
    mapped[mapped == 200] = 2
    
    #Focus on virus particle only
    spikes_only = (mapped == 2).astype(np.uint8)

    #returns parallel (x,y) arrays of all spike pixels
    spike_y_ls, spike_x_ls = np.where(spikes_only > 0)

    #Use distance formula to calculate distance from centroid of body to each spike pixel
    dist_spike_ls = np.sqrt((spike_x_ls - body_center_x)**2 + (spike_y_ls - body_center_y)**2)

    #Use np.arctan function to calculate angle from x-axis for each spike pixel (in radians), convert to degrees
    theta_spike_rads_ls = np.arctan2((spike_y_ls - body_center_y),(spike_x_ls - body_center_x))
    theta_spike_degrees_ls = np.degrees(theta_spike_rads_ls)

    return spike_x_ls, spike_y_ls, dist_spike_ls, theta_spike_rads_ls, theta_spike_degrees_ls


#calculate the angular distance/inter-pixel distance and make a histogram
def calculate_interpixel(theta_rads_ls, perim_x_ls, perim_y_ls, theta_spike_rads_ls):
    """This function takes a dictionary of spike angles calculates angular range, then computes the inter-pixel distance 
    Usage: angular_distance_ls, angle_dist_dict = calculate_interpixel(theta_rads_ls, perim_x_ls, perim_y_ls, theta_spike_rads_ls)"""

    angular_distance_ls = []
    angle_dist_dict = {}

    #Round angles to 2 decimals (or finer if needed)
    theta_rads_round = np.round(theta_rads_ls, 2)
    theta_spike_round = np.round(theta_spike_rads_ls, 2)

    #Map perimeter angles to their index
    body_angle_to_index = {angle: i for i, angle in enumerate(theta_rads_round)}

    #Find perimeter indices for each spike angle (if it lines up)
    spike_indices = []
    for spike_angle in theta_spike_round:
        if spike_angle in body_angle_to_index:
            spike_indices.append(body_angle_to_index[spike_angle])

    #Count distances between each pair of spikes
    num_pixels = len(perim_x_ls)
    for i in range(len(spike_indices)):
        for j in range(i+1, len(spike_indices)):
            idx1 = spike_indices[i]
            idx2 = spike_indices[j]

            dist_forward = (idx2 - idx1) % num_pixels
            dist_backward = (idx1 - idx2) % num_pixels
            shortest_distance = min(dist_forward, dist_backward)

            angular_distance_ls.append(shortest_distance)
            angle_dist_dict[(idx1, idx2)] = shortest_distance

    return angular_distance_ls, angle_dist_dict


def plot_results(angular_distance_ls, image, perimeter_mask, num_spike_pixels):
    """This function takes the list of angular distances and the original, mapped image and returns a plot of the spike historgram
      Usage: plot_results(angular_distance_ls, image, perimeter_mask, num_spike_pixels)"""
    
    plt.figure(figsize=(10,10))

    #Original Image
    plt.subplot(2,2,1)
    plt.imshow(image, cmap='gray')
    plt.title("Particle with Spike Identification")
    plt.axis("off") 

    #Perimeter Mask
    plt.subplot(2,2,2)
    plt.imshow(perimeter_mask, cmap='gray')
    plt.title("Perimeter Mask")
    plt.axis("off") 

    #Spike Distance Histogram
    if len(angular_distance_ls) > 0:
        max_distance = max(angular_distance_ls)

        #Make histogram of inter-pixel distances
        counts, bins = np.histogram(angular_distance_ls, bins=50, range=(0, max_distance))

        #Normalize by sqrt(N spike pixels)
        counts_norm = counts / np.sqrt(num_spike_pixels)

        #Smooth the histogram for nicer plotting
        smooth_counts = gaussian_filter1d(counts_norm, sigma=4)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        plt.subplot(2,1,2)
        plt.plot(bin_centers, smooth_counts, linewidth=2)
        plt.xlim(0, max_distance)
        plt.xlabel("Spike Inter-Pixel Perimeter Distance (pixels)")
        plt.title("Spike Distance Distribution")

    else:
        print("no spike pixels detected")

    plt.tight_layout()
    plt.show()


def plot_perimeter_count(image, perim_x_ls, perim_y_ls, start_coord, end_coord, direction="forward"):
    """
    Plot the perimeter and show how pixels are counted between two points.
    Takes (x, y) for start and end, and counts along the perimeter.
    """

    # Build ordered perimeter coordinate list
    perimeter_pixels = list(zip(perim_x_ls, perim_y_ls))

    # Directly find indices for start and end coords
    idx1 = perimeter_pixels.index(start_coord)
    idx2 = perimeter_pixels.index(end_coord)

    # Convert grayscale image to color for drawing
    vis_img = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2BGR)

    # Mark start (green) and end (red)
    cv2.circle(vis_img, start_coord, 5, (0, 255, 0), -1)
    cv2.circle(vis_img, end_coord, 5, (0, 0, 255), -1)

    # Determine path
    num_pixels = len(perim_x_ls)
    if direction == "forward":
        path_indices = [(idx1 + k) % num_pixels for k in range((idx2 - idx1) % num_pixels)]
        path_color = (255, 0, 0)  # blue
    else:
        path_indices = [(idx1 - k) % num_pixels for k in range((idx1 - idx2) % num_pixels)]
        path_color = (255, 255, 0)  # yellow

    # Color the counted pixels
    for k in path_indices:
        vis_img[perim_y_ls[k], perim_x_ls[k]] = path_color

    # Plot
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
    plt.title(f"Counting {direction.upper()} from {start_coord} to {end_coord} ({len(path_indices)} pixels)")
    plt.axis("off")
    plt.show()


#### Implementation ####
#### Testing Many ####

manually_segmented_mutant_dir = "/home/ascott10/documents/projects/capstone_viruses/segmented_images/wildtype_manual_correction"
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

    #Generate perimeter mask, body angles and spike angles
    perimeter_mask, perim_x_ls, perim_y_ls = generate_perimeter(image)
    dist_ls, theta_rads_ls, theta_degrees_ls = get_body_angle(perim_x_ls, perim_y_ls, body_center_x, body_center_y)
    spike_x_ls, spike_y_ls, dist_spike_ls, theta_spike_rads_ls, theta_spike_degrees_ls = generate_spikes(image,body_center_x, body_center_y)

    #If body angles match spike angles --> this means spikes are in line with perimeter pixels
    angular_distance_ls, angle_dist_dict = calculate_interpixel(theta_rads_ls, perim_x_ls, perim_y_ls, theta_spike_rads_ls)




    plot_results(angular_distance_ls, image, perimeter_mask, len(spike_x_ls))

    # build ordered perimeter pixel list
    perimeter_pixels = list(zip(perim_x_ls, perim_y_ls))

    # pick first and last spike pixel (or any two you want)
    sample_idx1 = (perim_x_ls[0], perim_y_ls[0])
    sample_idx2 = (perim_x_ls[-1], perim_y_ls[-1])



    # plot the counted path
    
    # Pick two perimeter coordinates directly
    start = (perim_x_ls[100], perim_y_ls[100])
    end = (perim_x_ls[50], perim_y_ls[500])

    # Plot counting path
    plot_perimeter_count(image, perim_x_ls, perim_y_ls, start, end, direction="forward")



