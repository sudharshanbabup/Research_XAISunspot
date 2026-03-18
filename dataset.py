"""
dataset.py
==========
PyTorch Dataset for SDO/HMI continuum image → SILSO ISN regression.

Chronological splits (no data leakage):
  Train : 2010-01-01 to 2018-12-31  (3,011 samples)
  Val   : 2019-01-01 to 2019-12-31  (  319 samples)
  Test  : 2020-01-01 to 2022-12-31  (  901 samples)

Usage:
  from dataset import make_loaders
  loaders = make_loaders('data/image_isn_pairs.csv', 'data/images/')
"""

import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

# ── ImageNet normalization (standard for transfer learning) ──────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ── Chronological split boundaries ──────────────────────────────────────────
SPLIT_DATES = {
    "train": ("2010-01-01", "2018-12-31"),
    "val":   ("2019-01-01", "2019-12-31"),
    "test":  ("2020-01-01", "2022-12-31"),
}


class HMIDataset(Dataset):
    """
    Dataset of pre-processed 224×224 SDO/HMI PNG images paired
    with SILSO daily ISN v2.0 values.

    Parameters
    ----------
    csv_path  : str  -- path to image_isn_pairs.csv
                        columns: image (filename), date (YYYY-MM-DD), ISN (float)
    image_dir : str  -- directory containing the PNG files
    split     : str  -- 'train' | 'val' | 'test'
    augment   : bool -- apply training augmentation (used for train split)
    """

    def __init__(self,
        csv_path: str,
        image_dir: str,
        split: str = "train",
        augment: bool = False,
    ) -> None:
        super().__init__()
        assert split in SPLIT_DATES, f"split must be one of {list(SPLIT_DATES)}"

        df = pd.read_csv(csv_path, parse_dates=["date"])
        start, end = pd.Timestamp(SPLIT_DATES[split][0]), pd.Timestamp(SPLIT_DATES[split][1])
        mask = (
            (df["date"] >= start)
            & (df["date"] <= end)
            & (df["ISN"] >= 0)            # exclude missing-data days (ISN=-1)
        )
        self.df        = df[mask].reset_index(drop=True)
        self.image_dir = image_dir

        # ── Training augmentation (Section III.B of paper) ──────────────────
        train_tfm = T.Compose([
            T.Resize((224, 224)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=5, fill=0),
            T.ColorJitter(brightness=0.1),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        eval_tfm = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        self.transform = train_tfm if augment else eval_tfm

    # ── Properties ───────────────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row   = self.df.iloc[idx]
        path  = os.path.join(self.image_dir, row["image"])
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image not found for row {idx}: {path}")
        img   = Image.open(path).convert("RGB")
        img   = self.transform(img)
        label = torch.tensor(float(row["ISN"]), dtype=torch.float32)
        return img, label

    # ── Convenience ──────────────────────────────────────────────────────────
    @property
    def isn_values(self) -> np.ndarray:
        """Return array of all ISN values in this split (for statistics)."""
        return self.df["ISN"].values.astype(np.float32)

    def class_weights(self, threshold: float = 150.0,
                      lam: float = 2.0) -> torch.Tensor:
        """
        Per-sample weights matching the weighted-MSE training objective.
        w_i = 1 + lam * 1[ISN_i >= threshold]
        """
        isn = torch.from_numpy(self.isn_values)
        return 1.0 + lam * (isn >= threshold).float()


def make_loaders(
    csv_path: str,
    image_dir: str,
    batch_size: int = 32,
    num_workers: int = min(4, os.cpu_count() or 0),
    pin_memory: bool = True,
) -> dict:
    """
    Build and return train/val/test DataLoaders.

    Returns
    -------
    dict with keys 'train', 'val', 'test', and 'train_weights'
    """
    train_ds = HMIDataset(csv_path, image_dir, split="train", augment=True)
    val_ds   = HMIDataset(csv_path, image_dir, split="val",   augment=False)
    test_ds  = HMIDataset(csv_path, image_dir, split="test",  augment=False)

    loaders = {
        "train": DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
            drop_last=True),
        "val": DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory),
        "test": DataLoader(
            test_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory),
        "train_weights": train_ds.class_weights(),
    }

    print(f"Dataset sizes — "
          f"train: {len(train_ds):,}  "
          f"val: {len(val_ds):,}  "
          f"test: {len(test_ds):,}")
    print(f"ISN stats (train) — "
          f"mean: {train_ds.isn_values.mean():.1f}  "
          f"std: {train_ds.isn_values.std():.1f}  "
          f"max: {train_ds.isn_values.max():.0f}")
    return loaders