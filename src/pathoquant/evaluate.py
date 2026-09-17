"""Full-tile Dice / AJI, classical baseline, overlays, quantification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from pathoquant.classical import otsu_watershed
from pathoquant.dataset import IMAGENET_MEAN, IMAGENET_STD, load_index
from pathoquant.metrics import aggregated_jaccard_index, dice_coefficient
from pathoquant.model import UNet
from pathoquant.quantify import quantify
from pathoquant.tiling import stitch_tiles, tile_image


def _label_binary(mask: np.ndarray) -> np.ndarray:
    n, lab = cv2.connectedComponents((mask > 0).astype(np.uint8))
    return lab.astype(np.int32)


def predict_tile(model: UNet, rgb: np.ndarray, device: torch.device, tile: int = 256, stride: int = 128) -> np.ndarray:
    model.eval()
    h, w = rgb.shape[:2]
    patches = []
    with torch.no_grad():
        for patch, y, x in tile_image(rgb, tile, stride):
            t = torch.from_numpy(patch.copy()).permute(2, 0, 1).float() / 255.0
            t = (t - IMAGENET_MEAN) / IMAGENET_STD
            logits = model(t.unsqueeze(0).to(device))
            prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
            patches.append((prob, y, x))
    return stitch_tiles(patches, h, w, tile)


def overlay(rgb: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    vis = rgb.copy()
    gt_e = cv2.Canny((gt > 0).astype(np.uint8) * 255, 40, 120)
    pred_e = cv2.Canny((pred > 0).astype(np.uint8) * 255, 40, 120)
    vis[gt_e > 0] = (40, 220, 80)
    vis[pred_e > 0] = (230, 50, 50)
    return vis


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data/processed"))
    p.add_argument("--ckpt", type=Path, default=Path("outputs/unet_monuseg.pt"))
    p.add_argument("--split", choices=["test", "train"], default="test")
    p.add_argument("--out", type=Path, default=Path("outputs"))
    p.add_argument("--base", type=int, default=32)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(base=args.base).to(device)
    blob = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(blob["model"])
    model.eval()

    records = load_index(args.data / args.split)
    viz_dir = args.out / "overlays"
    viz_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for rec in tqdm(records, desc="evaluate"):
        rgb = np.asarray(Image.open(rec["image"]).convert("RGB"))
        gt = np.load(rec["instances"])
        prob = predict_tile(model, rgb, device)
        pred = (prob > 0.5).astype(np.uint8)
        pred_inst = _label_binary(pred)
        classic = otsu_watershed(rgb)
        dice = dice_coefficient(pred, gt > 0)
        aji = aggregated_jaccard_index(pred_inst, gt)
        aji_cv = aggregated_jaccard_index(classic, gt)
        q = quantify(pred_inst)
        rows.append(
            {
                "id": rec["id"],
                "tissue": rec.get("tissue"),
                "dice": dice,
                "aji": aji,
                "aji_watershed": aji_cv,
                **q,
            }
        )
        vis = np.concatenate(
            [
                rgb,
                overlay(rgb, gt, pred),
            ],
            axis=1,
        )
        Image.fromarray(vis).save(viz_dir / f"{rec['id']}.png")

    dice_m = float(np.mean([r["dice"] for r in rows]))
    aji_m = float(np.mean([r["aji"] for r in rows]))
    aji_w = float(np.mean([r["aji_watershed"] for r in rows]))
    summary = {
        "split": args.split,
        "n": len(rows),
        "dice_mean": dice_m,
        "aji_mean": aji_m,
        "aji_watershed_mean": aji_w,
        "per_image": rows,
    }
    (args.out / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("split", "n", "dice_mean", "aji_mean", "aji_watershed_mean")}, indent=2))
    print(f"overlays → {viz_dir}")


if __name__ == "__main__":
    main()
