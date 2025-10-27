
# === FILE: dataset.py ===
"""
Data loader for HipMRI 2D slices stored as .nii.gz volumes. We extract slices,
resize them to a fixed size and normalize to [0,1].

Expect the dataset root to look like:
HipMRI_Study_open/keras_slices_data/keras_slices_train/ ... .nii.gz
Additional directories provided (semantic labels etc) are not required for this
proof-of-concept loader but basic hooks are added.
"""

import os
from glob import glob
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import torchvision.transforms.functional as TF
from skimage.transform import resize


def load_nii_slices(path):
    img = nib.load(path).get_fdata()
    # assume shape (H, W, Slices) or (X,Y,Z)
    if img.ndim == 4:
        img = img[...,0]
    return img


class HipMRISlicesDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, target_size=(128,128), max_slices_per_volume=9999):
        self.root = os.path.join(root_dir, 'keras_slices_data')
        self.split = split
        self.dir = os.path.join(self.root, f'keras_slices_{split}')
        self.files = sorted(glob(os.path.join(self.dir, '*.nii.gz')))
        self.transform = transform
        self.target_size = target_size
        self.max_slices_per_volume = max_slices_per_volume
        self.samples = []  # list of (volume_path, slice_idx)
        for f in self.files:
            vol = load_nii_slices(f)
            n_slices = vol.shape[2]
            for s in range(min(n_slices, self.max_slices_per_volume)):
                self.samples.append((f, s))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        f, s = self.samples[idx]
        vol = load_nii_slices(f)
        slice_img = vol[:,:,s]
        # normalize to 0-1
        slice_img = slice_img.astype(np.float32)
        # simple percentile clip
        p1, p99 = np.percentile(slice_img, (1, 99))
        slice_img = np.clip(slice_img, p1, p99)
        slice_img = (slice_img - slice_img.min()) / (slice_img.max() - slice_img.min() + 1e-8)
        # resize
        if slice_img.shape != self.target_size:
            slice_img = resize(slice_img, self.target_size, preserve_range=True, anti_aliasing=True)
        # add channel
        slice_img = np.expand_dims(slice_img, 0)
        tensor = torch.from_numpy(slice_img).float()
        if self.transform:
            tensor = self.transform(tensor)
        return tensor


def make_dataloaders(root, batch_size=16, target_size=(128,128), num_workers=4):
    train_ds = HipMRISlicesDataset(root, split='train', target_size=target_size)
    val_ds = HipMRISlicesDataset(root, split='validate', target_size=target_size)
    test_ds = HipMRISlicesDataset(root, split='test', target_size=target_size)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader

