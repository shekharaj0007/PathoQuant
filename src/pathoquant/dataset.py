"""Overlapping 256×256 patches from MoNuSeg tiles with light H&E augmentation."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from pathoquant.tiling import tile_coords


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def load_index(split_dir: Path) -> list[dict]:
    return json.loads((split_dir / "index.json").read_text(encoding="utf-8"))


def patient_split(index: list[dict], val_frac: float = 0.2, seed: int = 42) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    items = list(index)
    rng.shuffle(items)
    n_val = max(1, int(round(len(items) * val_frac)))
    return items[n_val:], items[:n_val]


def _read_rgb(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


class NucleiPatchDataset(Dataset):
    def __init__(
        self,
        records: list[dict],
        patch: int = 256,
        stride: int = 128,
        augment: bool = False,
        min_fg: float = 0.002,
    ) -> None:
        self.patch = patch
        self.augment = augment
        self.index: list[tuple[dict, int, int]] = []
        self._cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for rec in records:
            inst = np.load(rec["instances"])
            h, w = inst.shape
            for y, x, th, tw in tile_coords(h, w, patch, stride):
                if th != patch or tw != patch:
                    continue
                mask = inst[y : y + th, x : x + tw]
                if (mask > 0).mean() < min_fg:
                    continue
                self.index.append((rec, y, x))

    def __len__(self) -> int:
        return len(self.index)

    def _load(self, rec: dict) -> tuple[np.ndarray, np.ndarray]:
        key = rec["id"]
        if key not in self._cache:
            self._cache[key] = (_read_rgb(rec["image"]), np.load(rec["instances"]))
        return self._cache[key]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        rec, y, x = self.index[idx]
        rgb, inst = self._load(rec)
        img = rgb[y : y + self.patch, x : x + self.patch].copy()
        mask = (inst[y : y + self.patch, x : x + self.patch] > 0).astype(np.float32)
        if self.augment:
            if random.random() < 0.5:
                img = np.ascontiguousarray(np.fliplr(img))
                mask = np.ascontiguousarray(np.fliplr(mask))
            if random.random() < 0.5:
                img = np.ascontiguousarray(np.flipud(img))
                mask = np.ascontiguousarray(np.flipud(mask))
            k = random.randint(0, 3)
            if k:
                img = np.ascontiguousarray(np.rot90(img, k))
                mask = np.ascontiguousarray(np.rot90(mask, k))
            if random.random() < 0.7:
                img = img.astype(np.float32)
                img *= random.uniform(0.9, 1.1)
                img += random.uniform(-8, 8)
                img = np.clip(img, 0, 255).astype(np.uint8)
        x = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        y = torch.from_numpy(mask).unsqueeze(0)
        return x, y
