"""Train residual U-Net on MoNuSeg patches (Dice + BCE)."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from pathoquant.losses import dice_bce_loss
from pathoquant.model import UNet


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data/processed"))
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--out", type=Path, default=Path("outputs/unet_monuseg.pt"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    # Wire MoNuSeg patch DataLoader here (see README).
    _ = DataLoader  # placeholder until local tiles are prepared
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict()}, args.out)
    print(f"checkpoint stub written to {args.out} on {device}")


if __name__ == "__main__":
    main()
