"""this code will allow the user to input the files paths and save to a csv for future use""" 

import os
import sys
import pandas as pd

def input_files(im_path_mut=None, im_path_wt=None):
    """
    
    Inputs: raw folder path names
    Defaults to personal files for now, inputs are file paths
    
    Outputs: df with list of all folder paths, and class (useful for classification)
    Also saves to csv
    
    """
    if im_path_mut is None:
        im_path_mut = '/home/ariellescott/Documents/capstone/data/raw_images/raw_mutant/'
    if im_path_wt is None:
        im_path_wt = '/home/ariellescott/Documents/capstone/data/raw_images/raw_wt/'
    
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

    df.to_csv('raw_filepaths.csv', index=False)
    
    print(f"Saved csv")
    
    return df

#If user runs from command line
#sys.argv[0] is always the name of the script (input_files.py)
#sys.argv[0] = 'input_files.py'
#sys.argv[1] = '/path/to/mutant'
#sys.argv[2] = '/path/to/wildtype'

if __name__ == "__main__":

    #if user provides 2 commands --> use user input files
    if len(sys.argv) == 3:
        mut_path = sys.argv[1]
        wt_path = sys.argv[2]
        df = input_files(mut_path, wt_path)
    else:
        #otherwise use the hardcoded files
        df = input_files()
    

    print(df.head())
