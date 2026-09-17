"""Constant-VRAM tiled inference for arbitrary H&E (including ~50k WSIs)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from pathoquant.dataset import IMAGENET_MEAN, IMAGENET_STD
from pathoquant.evaluate import _label_binary, overlay, predict_tile
from pathoquant.model import UNet
from pathoquant.quantify import quantify


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--ckpt", type=Path, default=Path("outputs/unet_monuseg.pt"))
    p.add_argument("--out", type=Path, default=Path("outputs/infer"))
    p.add_argument("--base", type=int, default=32)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(base=args.base).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device, weights_only=False)["model"])
    rgb = np.asarray(Image.open(args.image).convert("RGB"))
    print(f"input {rgb.shape[1]}×{rgb.shape[0]}")
    prob = predict_tile(model, rgb, device)
    pred = (prob > 0.5).astype(np.uint8)
    inst = _label_binary(pred)
    stats = quantify(inst)
    args.out.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pred * 255).save(args.out / "mask.png")
    Image.fromarray(overlay(rgb, pred, pred)).save(args.out / "overlay.png")
    print(stats)


if __name__ == "__main__":
    main()
