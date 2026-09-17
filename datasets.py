"""
datasets.py
===========
Loads (image, mask) pairs for both datasets. Rewritten to match their
ACTUAL file naming, confirmed by inspecting the real downloaded folders:

  Sen1Floods11:  1010394_image.tif  <-->  1010394_label.tif
                 (same folder structure, suffix changes: _image -> _label)

  FloodNet:      10165.jpg  <-->  10165_lab.png
                 (split across two class folders: Flooded/ and Non-Flooded/,
                  each with its own image/ and mask/ subfolder)

If you've used OpenCV, a PyTorch Dataset class is just a wrapper that
answers "given index N, give me the N-th (image, mask) pair as arrays" --
everything below is just the specific file-finding and file-reading logic
for these two datasets' real folder layouts.
"""

import glob
import os

import cv2
import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset


def find_pairs_by_rename(image_dir, mask_dir, rename_fn):
    """
    Generic pairing: for each image file found, apply `rename_fn` to its
    filename to get the expected mask filename, then check it exists.

    rename_fn: a function that takes an image filename (e.g. "10165.jpg")
    and returns the matching mask filename (e.g. "10165_lab.png").
    """
    image_paths = sorted(glob.glob(os.path.join(image_dir, "*")))
    pairs = []
    for img_path in image_paths:
        fname = os.path.basename(img_path)
        mask_name = rename_fn(fname)
        mask_path = os.path.join(mask_dir, mask_name)
        if os.path.exists(mask_path):
            pairs.append((img_path, mask_path))
    if not pairs:
        raise RuntimeError(
            f"No matching pairs found between:\n  {image_dir}\n  {mask_dir}\n"
            "Check that rename_fn correctly maps an image filename to its "
            "mask filename -- print a few examples of `mask_name` to debug."
        )
    return pairs


class Sen1FloodsDataset(Dataset):
    """
    Multi-band satellite tiles (SAR + optical, 8 channels) with binary
    flood masks. Filenames: <id>_image.tif <-> <id>_label.tif
    """

    def __init__(self, image_dir, mask_dir, transform=None):
        self.pairs = find_pairs_by_rename(
            image_dir, mask_dir,
            rename_fn=lambda fname: fname.replace("_image.tif", "_label.tif"),
        )
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]

        # rasterio.read() returns (bands, height, width) -- bands FIRST,
        # unlike OpenCV's (height, width, channels).
        with rasterio.open(img_path) as src:
            img = src.read().astype(np.float32)

        with rasterio.open(mask_path) as src:
            mask = src.read(1).astype(np.int64)

        # Per-tile normalization (quick baseline approach). Upgrade later
        # to dataset-wide mean/std for more stable training.
        mean = img.mean(axis=(1, 2), keepdims=True)
        std = img.std(axis=(1, 2), keepdims=True) + 1e-6
        img = (img - mean) / std

        if self.transform:
            img, mask = self.transform(img, mask)

        return torch.from_numpy(img).float(), torch.from_numpy(mask).long()


class FloodNetDataset(Dataset):
    """
    UAV aerial RGB photos with pixel-level flood masks. The real download
    splits examples into two class folders (Flooded/, Non-Flooded/), each
    containing its own image/ and mask/ subfolders -- this class combines
    both into one dataset automatically.

    Filenames: <id>.jpg <-> <id>_lab.png

    IMPORTANT: FloodNet's masks are NOT simple binary flood/no-flood --
    they're full semantic segmentation with 10 classes:
      0 Background, 1 Building-flooded, 2 Building-non-flooded,
      3 Road-flooded, 4 Road-non-flooded, 5 Water, 6 Tree, 7 Vehicle,
      8 Pool, 9 Grass
    To keep this consistent with Sen1Floods11 (a binary water/no-water
    task) and match --num_classes 2, we collapse the flood-related
    classes (1, 3, 5) into "flooded" (1) and everything else into
    "not flooded" (0). If you want the full 10-class problem instead,
    set binary=False and pass --num_classes 10 to train.py.

    Pass the ROOT labeled folder, e.g.:
        ".../FloodNet Challenge - Track 1/Train/Labeled"
    and this will look inside both "Flooded" and "Non-Flooded" subfolders.
    """

    FLOOD_CLASS_IDS = {1, 3, 5}  # Building-flooded, Road-flooded, Water

    def __init__(self, root_dir, img_size=512, transform=None, binary=True, denoise=False):
        self.img_size = img_size
        self.transform = transform
        self.binary = binary
        self.denoise = denoise

        self.pairs = []
        for class_folder in ["Flooded", "Non-Flooded"]:
            image_dir = os.path.join(root_dir, class_folder, "image")
            mask_dir = os.path.join(root_dir, class_folder, "mask")
            if not os.path.isdir(image_dir):
                continue  # skip if this class folder doesn't exist
            class_pairs = find_pairs_by_rename(
                image_dir, mask_dir,
                rename_fn=lambda fname: os.path.splitext(fname)[0] + "_lab.png",
            )
            self.pairs.extend(class_pairs)

        if not self.pairs:
            raise RuntimeError(
                f"No pairs found under {root_dir} in either Flooded/ or "
                "Non-Flooded/ subfolders. Double-check the root_dir path."
            )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]

        # Standard OpenCV pattern: read, BGR->RGB, resize.
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        img = cv2.resize(img, (self.img_size, self.img_size))
        # INTER_NEAREST for masks: normal interpolation would blend class
        # labels (e.g. 0 and 1) into meaningless in-between values like 0.5.
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # (H,W,3) -> (3,H,W)
        if self.denoise:
            from noise_reduction import denoise_optical
            img = denoise_optical(img)
        mask = mask.astype(np.int64)

        if self.binary:
            # Collapse 10 semantic classes -> binary flooded (1) / not (0).
            binary_mask = np.isin(mask, list(self.FLOOD_CLASS_IDS)).astype(np.int64)
            mask = binary_mask

        if self.transform:
            img, mask = self.transform(img, mask)

        return torch.from_numpy(img).float(), torch.from_numpy(mask).long()
