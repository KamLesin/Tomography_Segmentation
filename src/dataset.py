"""
Dataset class for loading liver CT scans and segmentation masks.
"""
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import cv2


class LiverDataset(Dataset):
    """
    Dataset for liver segmentation from CT scans.
    
    Loads images and labels from NPZ files according to metadata CSV.
    Handles cases where labels are missing (label_path = 'none').
    
    Args:
        metadata_csv: Path to metadata CSV file
        data_root: Root directory containing images/ and labels/ folders
        patient_ids: List of patient IDs to include (for train/val split)
        img_size: Target image size (height, width)
        normalize: Whether to normalize images to [0, 1]
        multiclass: Convert labels to 3-class format (background/liver/tumor)
    """
    def __init__(
        self,
        metadata_csv,
        data_root,
        patient_ids=None,
        img_size=(512, 512),
        normalize=True,
        multiclass=True,
    ):
        self.data_root = data_root
        self.img_size = img_size
        self.normalize = normalize
        self.multiclass = multiclass
        
        # Load metadata (avoid dtype warnings on mixed-type columns)
        self.metadata = pd.read_csv(metadata_csv, low_memory=False)
        
        # Filter by patient IDs if provided
        if patient_ids is not None:
            self.metadata = self.metadata[self.metadata['patient_id'].isin(patient_ids)]
        
        # Only keep rows with valid labels (not 'none')
        self.metadata = self.metadata[self.metadata['label_path'] != 'none'].reset_index(drop=True)
        
        print(f"Dataset initialized with {len(self.metadata)} samples")
        print(f"Patients: {self.metadata['patient_id'].nunique()}")
        # Count distinct series instances per patient (patient+series pairs)
        series_per_patient = self.metadata[['patient_id', 'series_id']].drop_duplicates().shape[0]
        print(f"Series: {series_per_patient}")

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        
        # Load image
        image_path = os.path.join(self.data_root, row['image_path'])
        try:
            image_data = np.load(image_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load image NPZ: {image_path}: {e}")
        # Support both named and default NPZ keys
        if 'image' in image_data.files:
            image = image_data['image'].astype(np.float32)
        elif 'arr_0' in image_data.files:
            image = image_data['arr_0'].astype(np.float32)
        else:
            raise KeyError(f"No supported image key in {image_path}. Available: {image_data.files}")
        
        # Load label
        label_path = os.path.join(self.data_root, row['label_path'])
        try:
            label_data = np.load(label_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load label NPZ: {label_path}: {e}")
        if 'label' in label_data.files:
            label = label_data['label'].astype(np.float32)
        elif 'arr_0' in label_data.files:
            label = label_data['arr_0'].astype(np.float32)
        else:
            raise KeyError(f"No supported label key in {label_path}. Available: {label_data.files}")
        
        # Resize if needed
        if image.shape != self.img_size:
            image = cv2.resize(image, self.img_size, interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, self.img_size, interpolation=cv2.INTER_NEAREST)
        
        # Normalize image
        if self.normalize:
            # Clip to reasonable HU range and normalize to [0, 1]
            image = np.clip(image, -200, 300)
            image = (image + 200) / 500.0
        
        # Convert to multiclass if needed
        if self.multiclass:
            label = self._convert_to_multiclass(label)
        
        # Convert to tensors
        image = torch.from_numpy(image).unsqueeze(0)  # Add channel dimension
        label = torch.from_numpy(label).long()
        
        return {
            'image': image,
            'label': label,
            'patient_id': row['patient_id'],
            'series_id': row['series_id'],
            'slice_id': row['slice_id'],
        }
    
    def _convert_to_multiclass(self, label):
        """
        Convert label to 3-class format:
        0 = background
        1 = liver
        2 = tumor
        """
        multiclass_label = np.zeros_like(label, dtype=np.int64)
        multiclass_label[label == 1] = 1  # Liver
        multiclass_label[label == 2] = 2  # Tumor
        return multiclass_label


def get_data_loaders(
    metadata_csv,
    folds_csv,
    data_root,
    fold,
    batch_size=4,
    img_size=(512, 512),
    normalize=True,
    multiclass=True,
    num_workers=4,
):
    """
    Create train and validation data loaders based on fold assignment.
    
    Args:
        metadata_csv: Path to metadata CSV
        folds_csv: Path to CV folds CSV (patient_id, fold)
        data_root: Root directory with data
        fold: Which fold to use as validation (0-5)
        batch_size: Batch size
        img_size: Target image size
        normalize: Normalize images
        multiclass: Use multiclass labels
        num_workers: Number of data loading workers
    
    Returns:
        train_loader, val_loader
    """
    # Load fold assignments
    folds_df = pd.read_csv(folds_csv)
    
    # Split patient IDs by fold
    val_patients = folds_df[folds_df['fold'] == fold]['patient_id'].tolist()
    train_patients = folds_df[folds_df['fold'] != fold]['patient_id'].tolist()
    
    print(f"\nFold {fold}:")
    print(f"Train patients: {len(train_patients)}")
    print(f"Val patients: {len(val_patients)}")
    
    # Create datasets
    train_dataset = LiverDataset(
        metadata_csv=metadata_csv,
        data_root=data_root,
        patient_ids=train_patients,
        img_size=img_size,
        normalize=normalize,
        multiclass=multiclass,
    )
    
    val_dataset = LiverDataset(
        metadata_csv=metadata_csv,
        data_root=data_root,
        patient_ids=val_patients,
        img_size=img_size,
        normalize=normalize,
        multiclass=multiclass,
    )
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader
