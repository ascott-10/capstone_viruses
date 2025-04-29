################ Import Libraries ################
import os
import glob
import torch
import pandas as pd
import matplotlib.pyplot as plt

from code.config import *
from code.unet_training_setup import (load_segmented_ims,
    combine_dfs,
    load_images_from_dataframe, 
    train_split, 
    create_segmentation_tensor_dataset, 
    create_dataloader
)
from code.unet_training import (
    setup_training, 
    train_one_epoch, 
    validate_one_epoch, 
    save_best_model, 
    plot_loss
)
from code.unet_model import UNet

################ Setup ################

# Load conversion dictionary if needed
   
convert_df = pd.read_excel('/home/ascott10/documents/projects/capstone_viruses/data/Segmentation_Progress.xlsx', usecols = [0,1]) 

image_labels_muts, image_filepaths_muts, image_ids_muts = load_segmented_ims(RAW_IMS_MUT)
image_labels_wts, image_filepaths_wts, image_ids_wts = load_segmented_ims(RAW_IMS_WT)
segmented_image_labels_muts, segmented_image_filepaths_muts, segmented_image_ids_muts = load_segmented_ims(SEGMENTED_MASKS_MUT)
segmented_image_labels_wts, segmented_image_filepaths_wts, segmented_image_ids_wts = load_segmented_ims(SEGMENTED_MASKS_WT)

print(convert_df)

# Build DataFrames
all_raw_files = pd.DataFrame([
    (image_ids_muts + image_ids_wts),
    (image_filepaths_muts + image_filepaths_wts),
    (image_labels_muts + image_labels_wts)
], index=['im_id', 'file_path', 'class']).T

    

all_segmented_files = pd.DataFrame([
    (segmented_image_ids_muts + segmented_image_ids_wts),
    (segmented_image_filepaths_muts + segmented_image_filepaths_wts),
    (segmented_image_labels_muts + segmented_image_labels_wts)
], index=['im_id', 'segmented_file_path', 'class']).T

convert_df['Modified Name'] = convert_df['Modified Name'].astype(str).str.replace(r'_corrected.*', '', regex=True)
all_segmented_files['im_id'] = all_segmented_files['im_id'].astype(str).str.replace(r'_corrected.*', '', regex=True)


segmented_combined = all_segmented_files.merge(convert_df, left_on='im_id', right_on = 'Modified Name')
merged_df = all_raw_files.merge(segmented_combined, left_on='im_id', right_on='File_name')

final_df = pd.DataFrame({
    'im_id': merged_df['im_id_x'],
    'file_path': merged_df['file_path'],
    'segmented_file_path': merged_df['segmented_file_path'],
    'class': merged_df['class_x']
})

print(final_df.head())

################## Subsample 25 mutants, 25 wildtypes ##################

mutants_df = final_df[final_df['class'] == 'mutant']
wildtypes_df = final_df[final_df['class'] == 'wildtype']

n_mutants = min(90, len(mutants_df))
n_wildtypes = min(90, len(wildtypes_df))

mutants_sample = mutants_df.sample(n=n_mutants, random_state=42)
wildtypes_sample = wildtypes_df.sample(n=n_wildtypes, random_state=42)

df = pd.concat([mutants_sample, wildtypes_sample]).reset_index(drop=True)

print(f" Using {len(df)} images total ({n_mutants} mutants + {n_wildtypes} wildtypes)")

# Step 3: Load images and masks
X_raw, X_seg, labels = load_images_from_dataframe(df, 'file_path', 'segmented_file_path', 'class')
X_raw_train, X_raw_val, X_raw_test, X_seg_train, X_seg_val, X_seg_test = train_split(X_raw, X_seg)

# Step 4: Build datasets and dataloaders
train_loader = create_dataloader(create_segmentation_tensor_dataset(X_raw_train, X_seg_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader = create_dataloader(create_segmentation_tensor_dataset(X_raw_val, X_seg_val), batch_size=BATCH_SIZE, shuffle=False)
test_loader = create_dataloader(create_segmentation_tensor_dataset(X_raw_test, X_seg_test), batch_size=BATCH_SIZE, shuffle=False)

################ Build or Load Model ################

model = UNet(input_channels=1, output_channels=1)
model = model.to(DEVICE)

best_model_path = os.path.join(SAVE_DIR, "best_unet.pt")

if os.path.exists(best_model_path):
    print(f" Found existing model weights at {best_model_path}. Loading model.")
    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
    model.eval()
    skip_training = True
else:
    print("No saved model found, starting training")
    skip_training = False

# Setup loss and optimizer
loss_fn, optimizer = setup_training(model, learning_rate=LEARNING_RATE)

################ Train ################

if not skip_training:
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    for epoch in range(NUM_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, DEVICE)
        val_loss = validate_one_epoch(model, val_loader, loss_fn, DEVICE)

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] — Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        best_val_loss = save_best_model(model, SAVE_DIR, epoch, val_loss, best_val_loss)

    ################ Plot Loss ################
    plot_loss(train_losses, val_losses, save_dir=SAVE_DIR)

else:
    print(" Skipping training, model weights already exist.")

################ Test ################

print("Testing")

model.eval()

os.makedirs(os.path.join(SAVE_DIR, "test_predictions"), exist_ok=True)
num_to_plot = min(5, len(test_loader))
count = 0

with torch.no_grad():
    for images, masks in test_loader:
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        outputs = model(images)

        if outputs.shape != masks.shape:
            outputs = torch.nn.functional.interpolate(outputs, size=masks.shape[2:], mode='bilinear', align_corners=False)

        preds = torch.sigmoid(outputs)
        preds = (preds > 0.5).float()

        images = images.cpu()
        masks = masks.cpu()
        preds = preds.cpu()

        for i in range(images.shape[0]):
            fig, axs = plt.subplots(1, 3, figsize=(12,4))

            axs[0].imshow(images[i,0], cmap='gray')
            axs[0].set_title("Raw Image")
            axs[0].axis('off')

            axs[1].imshow(masks[i,0], cmap='gray')
            axs[1].set_title("Ground Truth Segmented Mask")
            axs[1].axis('off')

            axs[2].imshow(preds[i,0], cmap='gray')
            axs[2].set_title("Predicted Mask")
            axs[2].axis('off')

            plt.tight_layout()
            save_path = os.path.join(SAVE_DIR, "test_predictions", f"test_{count}.png")
            plt.savefig(save_path)
            plt.show()
            plt.close()

            print(f"✅ Saved prediction plot to {save_path}")
            count += 1

            if count >= num_to_plot:
                break


