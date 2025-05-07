################ Import Libraries ################
import os
from datetime import datetime
import torch
from torchvision import models
from sklearn.model_selection import train_test_split

import pandas as pd

######## Custom ########
from config import *
from code.unet_training_setup import *
from code.unet_training import *
from code.unet_model import UNet

from code.setup_classifier import *
from code.train_classifier import *
from code.customs_stats import *
from code.spike_morpohology import *
from code.morphology import *
from code.compare_segmentation_methods import *

################ Data Setup ################
all_df = combine_dfs(convert_df, RAW_IMS_MUT, RAW_IMS_WT, SEGMENTED_MASKS_MUT, SEGMENTED_MASKS_WT, SAVE_DIR)
final_df = subsample_test(final_df=all_df, SUBSAMPLE=SUBSAMPLE) if SUBSAMPLE else all_df

X_raw, X_seg, labels = load_images_from_dataframe(final_df, 'file_path', 'segmented_file_path', 'class')
X_raw_train, X_raw_val, X_raw_test, X_seg_train, X_seg_val, X_seg_test, y_train, y_val, y_test = train_split(X_raw, X_seg, labels)

_, X_test_df = train_test_split(final_df, test_size=0.2, stratify=final_df['class'], random_state=42)
X_test_df = X_test_df.reset_index(drop=True)
mask_paths_test = X_test_df['segmented_file_path'].tolist()

train_dataset = create_segmentation_tensor_dataset(X_raw_train, X_seg_train, y_train)
val_dataset   = create_segmentation_tensor_dataset(X_raw_val, X_seg_val, y_val)
test_dataset  = create_segmentation_tensor_dataset(X_raw_test, X_seg_test, y_test)

train_loader = create_dataloader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = create_dataloader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader  = create_dataloader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

################ UNet ################
model = UNet(input_channels=1, output_channels=1).to(DEVICE)
best_model_path = os.path.join(SAVE_DIR, "best_unet.pt")

if os.path.exists(best_model_path):
    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
    model.eval()
    skip_training = True
else:
    skip_training = False

loss_fn, optimizer = setup_training(model, learning_rate=LEARNING_RATE)

if not skip_training:
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    best_val_loss = float('inf')

    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, loss_fn, optimizer, DEVICE)
        val_loss, val_acc = validate_one_epoch(model, val_loader, loss_fn, DEVICE)
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] — Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        best_val_loss = save_best_model(model, SAVE_DIR, epoch, val_loss, best_val_loss)

    plot_loss(train_losses, val_losses, train_accuracies, val_accuracies, save_dir=SAVE_DIR)

plot_test_predictions(
    test_loader,
    model,
    SAVE_DIR,
    DEVICE,
    image_paths=X_test_df['file_path'].tolist(),
    mask_paths=mask_paths_test,
    one_image=False,
    max_examples=5
)

################ ResNet ################
if NEW_CLASSIFY:
    model = load_classifier(DEVICE, num_classes=2).to(DEVICE)
    train_model(model, DEVICE, train_loader, val_loader, SAVE_DIR)
else:
    pre_trained_model = models.resnet18(pretrained=False)
    model = load_resnet_weights(pre_trained_model, SAVE_DIR, DEVICE, num_classes=2)

X_test_df_preds = make_predictions(model, DEVICE, X_test_df, test_loader, save_cm=True, save_dir=SAVE_DIR)
display_stats(X_test_df_preds)

print("\n===== U-NET TRAINING SUMMARY =====")
print(f"Trained on: {len(train_dataset)} images")
if not skip_training:
    print(f"Final Training Loss: {train_losses[-1]:.4f}")
    print(f"Final Training Accuracy: {train_accuracies[-1]*100:.2f}%")

print("\n===== CLASSIFIER TRAINING SUMMARY =====")
print(f"Final Test Accuracy: {X_test_df_preds['correct'].sum() / len(X_test_df_preds) * 100:.2f}%")

################ Morphology ################
df_ground_truth_morph = ground_truth_morph(
    final_df,
    segmented_path_label='segmented_file_path',
    class_label='class'
)
calculate_spike_stats(df_ground_truth_morph, plot_yes=True)
compare_classes_stats(df_ground_truth_morph, plot_yes=True, SAVE_DIR=SAVE_DIR)

################ Manual vs Automatic Comparison ################
all_df = compare_combine_dfs(
    convert_df, SEGMENTED_MASKS_WT, SEGMENTED_MASKS_MUT,
    AUTO_SEGMENTED_MASKS_WT, AUTO_SEGMENTED_MASKS_MUT, SAVE_DIR
)

df_compare = build_compare_df_from_auto_manual(all_df)
compare_methods_stats(df_compare, SAVE_DIR, plot_yes=True)
compare_spike_and_body_plotting(df_compare)

df_wide, stat, p, summary = compare_methods_stats(df_compare, SAVE_DIR, plot_yes=True)
df_wide.to_csv(os.path.join(SAVE_DIR, "comparison_results.csv"), index=False)

plot_spike_vs_body_area(df_compare, SAVE_DIR=SAVE_DIR)
