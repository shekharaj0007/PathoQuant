"""Train U-Net on MoNuSeg with Dice+BCE, AMP, and cosine LR."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from pathoquant.dataset import NucleiPatchDataset, load_index, patient_split
from pathoquant.losses import dice_bce_loss
from pathoquant.metrics import dice_coefficient
from pathoquant.model import UNet


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def patch_dice(model: UNet, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    scores = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        pred = (torch.sigmoid(logits) > 0.5).float()
        for p, t in zip(pred, y):
            scores.append(dice_coefficient(p[0].cpu().numpy(), t[0].cpu().numpy()))
    return float(np.mean(scores)) if scores else 0.0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data/processed"))
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patch", type=int, default=256)
    p.add_argument("--stride", type=int, default=128)
    p.add_argument("--base", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=Path("outputs"))
    args = p.parse_args()

    seed_all(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    train_rec, val_rec = patient_split(load_index(args.data / "train"), val_frac=0.2, seed=args.seed)
    train_ds = NucleiPatchDataset(train_rec, args.patch, args.stride, augment=True)
    val_ds = NucleiPatchDataset(val_rec, args.patch, args.stride, augment=False)
    print(f"train patients={len(train_rec)} patches={len(train_ds)}")
    print(f"val patients={len(val_rec)} patches={len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = UNet(base=args.base).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, T_max=args.epochs)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history = {"train_loss": [], "val_dice": []}
    best = -1.0
    ckpt = args.out / "unet_monuseg.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        running, n = 0.0, 0
        opt.zero_grad(set_to_none=True)
        bar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}")
        for i, (x, y) in enumerate(bar, start=1):
            x, y = x.to(device), y.to(device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(x)
                loss = dice_bce_loss(logits, y) / args.accum
            scaler.scale(loss).backward()
            if i % args.accum == 0 or i == len(train_loader):
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
            running += loss.item() * args.accum * x.size(0)
            n += x.size(0)
            bar.set_postfix(loss=f"{running / max(n, 1):.4f}")
        sched.step()
        val_dice = patch_dice(model, val_loader, device)
        history["train_loss"].append(running / max(n, 1))
        history["val_dice"].append(val_dice)
        print(f"epoch {epoch}: loss={history['train_loss'][-1]:.4f}  val_dice={val_dice:.4f}")
        if val_dice > best:
            best = val_dice
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val_dice": val_dice,
                    "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                },
                ckpt,
            )
            print(f"  saved {ckpt} (best val dice {best:.4f})")

    (args.out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    _plot_curves(history, args.out / "training_curves.png")
    print(f"best val dice={best:.4f}")


def _plot_curves(history: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    ax[0].plot(history["train_loss"], color="#1a365d")
    ax[0].set_title("Train loss (Dice + BCE)")
    ax[0].set_xlabel("epoch")
    ax[1].plot(history["val_dice"], color="#c05621")
    ax[1].set_title("Val Dice")
    ax[1].set_xlabel("epoch")
    ax[1].set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
