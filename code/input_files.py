"""this code will allow the user to input the files paths and save to a csv for future use""" 

import os
import sys
import pandas as pd

def input_files(im_path_mut = None, im_path_wt = None):
    """
    
    Inputs: raw folder path names
    
    Outputs: df with list of all folder paths, and class (useful for classification)
    Also saves to csv
    
    """
    #Either prompt the user or change the paths here

    im_path_mut = im_path_mut
    if im_path_mut is None and im_path_wt is None:
      im_path_wt = input('Enter a file path for wildtype images: ')
      im_path_mut = input('Enter a file path for mutant images: ')
    else:
        im_path_wt = im_path_wt or '/path/to/wt'
        im_path_mut = im_path_mut or '/path/to/mutant'
    
    
    #Can change the output name
    filepath_output = '/home/ascott10/documents/projects/capstone_viruses/results/raw_filepaths.csv'
        
    image_filepaths_mut = []
    image_filepaths_wt = []
    
    for file in os.listdir(im_path_mut):
        if file.endswith('.png'):
            image_filepaths_mut.append(os.path.join(im_path_mut, file))
    
    for file in os.listdir(im_path_wt):
        if file.endswith('.png'):
            image_filepaths_wt.append(os.path.join(im_path_wt, file))
    
    print('Data imported')
    
    data = {
    'filepath': image_filepaths_mut + image_filepaths_wt,
    'class': ['mutant'] * len(image_filepaths_mut) + ['wildtype'] * len(image_filepaths_wt)
}

    df = pd.DataFrame(data)
    
    
    #Enter an output filepath
    df.to_csv(filepath_output, index=False)
    
    print(f"Saved filepaths to csv")
    
    return df

