#%%
################ Import Libraries ################
import torch
device = "cuda" if torch.cuda.is_available() else "cpu" 
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

print('Import libraries good')


#%%
################ Configure SAM ################
def download_sam(sam_checkpoint=None):
    if sam_checkpoint is None:
        sam_checkpoint = "/home/ascott10/documents/projects/capstone_viruses/data/sam_vit_h_4b8939.pth"


    #Can change the path to where the model is stored
    
    model_type = "vit_h"  # model type is vit_h per the pre-downloaded model
    print('model downloaded')
    
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device)
    sam.to(device=device)
    
    return sam
  
def custom_mask(sam, use_defaults=True, **custom_params):
    """
    Returns a SAM mask generator. By default, uses tuned parameters.
    If use_defaults is False, user can override by passing their own values as keyword arguments.
    """
    
    if use_defaults:
        
        mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=32,
            pred_iou_thresh=0.95,
            stability_score_thresh=0.98,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
        )
    else:
        
        mask_generator = SamAutomaticMaskGenerator(model=sam, **custom_params)

    return mask_generator

