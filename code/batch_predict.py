# batch_predict.py

import os
import sys
import random
import pandas as pd
from glob import glob
from datetime import datetime
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import *
from predict_one import *

def load_segmented_ims(input_path):
    paths, labels, match_ids, filenames = [], [], [], []
    mut_labels = ['muimage']
    wt_labels = ['wtimage']

    for f in os.listdir(input_path):
        if f.lower().endswith(('.png', '.tif', '.tiff')):
            full_path = os.path.join(input_path, f)
            name = Path(f).name
            stem = Path(f).stem
            match_id = stem.replace('_seg_ver2', '').replace('_corrected', '')

            if any(tag in stem for tag in mut_labels):
                label = 'mutant'
            elif any(tag in stem for tag in wt_labels):
                label = 'wildtype'
            else:
                label = 'unknown'

            paths.append(full_path)
            labels.append(label)
            match_ids.append(match_id)
            filenames.append(name)

    return pd.DataFrame({
        'match_id': match_ids,
        'file_path': paths,
        'file_name': filenames,
        'class': labels
    })

N_RANDOM_IMAGES = 100
convert_df = pd.read_excel(os.path.join(SAVE_DIR, "Segmentation_Progress.xlsx"), usecols=[0, 1])
convert_df['match_id'] = convert_df['Modified Name'].str.replace(r'_corrected.*', '', regex=True)

manual_df = pd.concat([
    load_segmented_ims(SEGMENTED_MASKS_MUT),
    load_segmented_ims(SEGMENTED_MASKS_WT)
])

auto_df = pd.concat([
    load_segmented_ims(AUTO_SEGMENTED_MASKS_MUT),
    load_segmented_ims(AUTO_SEGMENTED_MASKS_WT)
])

merged_df = convert_df.merge(manual_df, on='match_id')
merged_df = merged_df.merge(auto_df, on='match_id', suffixes=('_manual', '_auto'))

mutant_images = glob(os.path.join(RAW_IMS_MUT, "*.png"))
wt_images = glob(os.path.join(RAW_IMS_WT, "*.png"))
sampled_mutant = random.sample(mutant_images, min(N_RANDOM_IMAGES, len(mutant_images)))
sampled_wt = random.sample(wt_images, min(N_RANDOM_IMAGES, len(wt_images)))

timestamp = datetime.now().strftime("batch_%Y_%m_%d_%H_%M")
base_dir = Path(PREDICTION_SAVE_DIR) / timestamp
mutant_dir = base_dir / "mutant"
wt_dir = base_dir / "wildtype"
mutant_dir.mkdir(parents=True, exist_ok=True)
wt_dir.mkdir(parents=True, exist_ok=True)

best_model_path = find_best_model_by_val_loss(SAVE_DIR)

results = []

for img_path in sampled_mutant:
    img_name = Path(img_path).name
    img_stem = Path(img_path).stem
    predicted_path = mutant_dir / f"{img_stem}_predicted_mask.png"

    predict_single_image(img_path, model_path=best_model_path, save_dir=mutant_dir)

    row = merged_df[merged_df['File_name'] == img_name]
    if not row.empty:
        manual_path = row.iloc[0]['file_path_manual']
        auto_path = row.iloc[0]['file_path_auto']
        save_composite_image(
            img_name=row.iloc[0]['file_name_manual'],
            raw_dir=Path(img_path).parent,
            sam_dir=Path(auto_path).parent,
            manual_dir=Path(manual_path).parent,
            predicted_mask_path=predicted_path,
            save_dir=mutant_dir
        )
        results.append({"image": img_name, "class": "mutant", "raw": img_path, "manual": manual_path, "auto": auto_path, "predicted": str(predicted_path)})

for img_path in sampled_wt:
    img_name = Path(img_path).name
    img_stem = Path(img_path).stem
    predicted_path = wt_dir / f"{img_stem}_predicted_mask.png"

    predict_single_image(img_path, model_path=best_model_path, save_dir=wt_dir)

    row = merged_df[merged_df['File_name'] == img_name]
    if not row.empty:
        manual_path = row.iloc[0]['file_path_manual']
        auto_path = row.iloc[0]['file_path_auto']
        save_composite_image(
            img_name=row.iloc[0]['file_name_manual'],
            raw_dir=Path(img_path).parent,
            sam_dir=Path(auto_path).parent,
            manual_dir=Path(manual_path).parent,
            predicted_mask_path=predicted_path,
            save_dir=wt_dir
        )
        results.append({"image": img_name, "class": "wildtype", "raw": img_path, "manual": manual_path, "auto": auto_path, "predicted": str(predicted_path)})

summary_df = pd.DataFrame(results)
summary_df.to_csv(base_dir / "prediction_summary.csv", index=False)
