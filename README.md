# PathoQuant

Gigapixel-aware **nuclei segmentation and quantification** for digital pathology.

A 256×256 residual U-Net never loads a 50,000×50,000 WSI. Overlap tiles → predict → Gaussian stitch → instance count / density. Built for the AIRA-style constraint: high-resolution H&E, segmentation + quantification, constant GPU memory.

## Pipeline

```
MoNuSeg 1000×1000 tiles (or a 50k WSI)
        │
        ▼
 overlap 256 tiles (stride 128)
        │
        ▼
 residual U-Net  (Dice + BCE)
        │
        ▼
 Gaussian blend  →  binary mask
        │
        ▼
 connected components  →  AJI, nuclei count, area, density / mm²
```

Classical baseline on the same tiles: Otsu → distance transform → watershed (shows where DL is required: touching nuclei).

## Dataset

[MoNuSeg](https://monuseg.grand-challenge.org/) via Hugging Face [`RationAI/MoNuSeg`](https://huggingface.co/datasets/RationAI/MoNuSeg): 30 train + 14 test H&E tiles, instance contours (~22k nuclei, 7 organs). Patient-level 80/20 val split from train; official test for the reported number.

## Metrics

| Split | Dice | AJI | Watershed AJI |
|---|---|---|---|
| held-out / test | filled after `evaluate` | filled after `evaluate` | filled after `evaluate` |

AJI (Kumar et al., IEEE TMI) penalizes merged/split nuclei. Semantic U-Net + connected components is an honest instance readout, not HoVer-Net.

## Setup (Windows, RTX 3050)

```powershell
cd PathoQuant
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

CUDA PyTorch (if not already):

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

## Run

```powershell
# 1. Download MoNuSeg and write PNG + instance maps
python -m pathoquant.prepare --out data/processed

# 2. Train U-Net (Dice + BCE, AMP, cosine LR)
python -m pathoquant.train --data data/processed --epochs 40 --batch-size 2 --out outputs

# 3. Test-split Dice / AJI + overlay figures
python -m pathoquant.evaluate --data data/processed --ckpt outputs/unet_monuseg.pt --split test

# 4. Tiled inference on any H&E (VRAM stays ~1–2 GB)
python -m pathoquant.infer --image path\to\slide_or_tile.png --ckpt outputs/unet_monuseg.pt
```

Outputs:

- `outputs/unet_monuseg.pt` — best val-Dice checkpoint  
- `outputs/training_curves.png`  
- `outputs/metrics.json` — the honest test number  
- `outputs/overlays/*.png` — H&E | GT-green / pred-red contours  

## Layout

```
src/pathoquant/
  prepare.py     Hugging Face MoNuSeg → image + instance .npy
  dataset.py     overlapping patches + H&E jitter
  model.py       residual U-Net
  losses.py      Dice + BCE
  metrics.py     Dice, Aggregated Jaccard Index
  tiling.py      50k-safe overlap tile + Gaussian stitch
  classical.py   Otsu + watershed
  quantify.py    count / mean area / density
  train.py
  evaluate.py
  infer.py
```

## Cite

Kumar et al., “A Dataset and a Technique for Generalized Nuclear Segmentation,” IEEE TMI, 2017.  
Kumar et al., “A Multi-organ Nucleus Segmentation Challenge,” IEEE TMI, 2020.
