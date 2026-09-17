"""Full-image Dice / AJI and classical vs U-Net comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pathoquant.classical import otsu_watershed
from pathoquant.metrics import aggregated_jaccard_index, dice_coefficient
from pathoquant.quantify import quantify


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--gt", type=Path, required=True)
    args = parser.parse_args()
    pred = np.load(args.pred)
    gt = np.load(args.gt)
    print(
        {
            "dice": dice_coefficient(pred > 0, gt > 0),
            "aji": aggregated_jaccard_index(pred, gt),
            "quant": quantify(pred),
        }
    )
    _ = otsu_watershed


if __name__ == "__main__":
    main()
