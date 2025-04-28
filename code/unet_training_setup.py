import torch
import cv2
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

IMAGE_SIZE = (750, 750)

def load_images_from_dataframe(df, raw_image_col, mask_col, label_col=None):
    raw_images = []
    mask_images = []
    labels = []

    for _, row in df.iterrows():
        raw_img = cv2.imread(row[raw_image_col], cv2.IMREAD_GRAYSCALE)
        mask_img = cv2.imread(row[mask_col], cv2.IMREAD_GRAYSCALE)

        if raw_img is None or mask_img is None:
            raise ValueError(f"Image not found: {row[raw_image_col]} or {row[mask_col]}")

        raw_img = cv2.resize(raw_img, IMAGE_SIZE)
        mask_img = cv2.resize(mask_img, IMAGE_SIZE)

        raw_img = torch.tensor(raw_img, dtype=torch.float32).unsqueeze(0) / 255.0
        mask_img = torch.tensor(mask_img, dtype=torch.float32).unsqueeze(0) / 255.0

        raw_images.append(raw_img)
        mask_images.append(mask_img)

        if label_col is not None:
            labels.append(row[label_col])

    if label_col is not None:
        return torch.stack(raw_images), torch.stack(mask_images), labels
    else:
        return torch.stack(raw_images), torch.stack(mask_images)

def train_split(X_raw, X_segmented, y_class=None):
    if y_class is not None:
        X_raw_trainval, X_raw_test, X_seg_trainval, X_seg_test, y_class_trainval, y_class_test = train_test_split(
            X_raw, X_segmented, y_class, test_size=0.2, random_state=42, stratify=y_class)
        X_raw_train, X_raw_val, X_seg_train, X_seg_val, y_class_train, y_class_val = train_test_split(
            X_raw_trainval, X_seg_trainval, y_class_trainval, test_size=0.25, random_state=42, stratify=y_class_trainval)

        return X_raw_train, X_raw_val, X_raw_test, X_seg_train, X_seg_val, X_seg_test, y_class_train, y_class_val, y_class_test
    else:
        X_raw_trainval, X_raw_test, X_seg_trainval, X_seg_test = train_test_split(
            X_raw, X_segmented, test_size=0.2, random_state=42)
        X_raw_train, X_raw_val, X_seg_train, X_seg_val = train_test_split(
            X_raw_trainval, X_seg_trainval, test_size=0.25, random_state=42)
        return X_raw_train, X_raw_val, X_raw_test, X_seg_train, X_seg_val, X_seg_test

def create_segmentation_tensor_dataset(X_raw, X_mask):
    return TensorDataset(X_raw, X_mask)

def create_dataloader(dataset, batch_size=32, shuffle=True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
