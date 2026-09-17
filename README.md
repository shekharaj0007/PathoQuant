# PathoQuant

Memory-efficient **nuclei segmentation and quantification** for digital pathology. A 256×256 residual U-Net runs on overlapping tiles so inference VRAM stays flat on ~50k×50k whole-slide images.

**GitHub:** [shekharaj0007/PathoQuant](https://github.com/shekharaj0007/PathoQuant)

## Why this exists

Naive CNNs cannot load a 50,000×50,000 H&E slide. PathoQuant tiles with overlap, predicts per tile, Gaussian-blends the logits, then turns the mask into nuclei counts, mean area, and density.

## Method

| Piece | Choice |
|---|---|
| Model | Residual U-Net, 32–512 channels |
| Loss | Dice + BCE |
| Data | MoNuSeg (30 H&E tiles, ~21.6k nuclear boundaries, 7 organs) |
| Metrics | Dice, Aggregated Jaccard Index (AJI) |
| Inference | 256 tile, 128 stride, Gaussian stitch, constant ~1.2 GB VRAM |
| Baseline | Otsu + distance transform + watershed |
| Quantification | instance count, area, density / mm² |

Held-out MoNuSeg split: **Dice 0.81 / AJI 0.59**. Watershed trails U-Net by ~0.18 AJI on touching nuclei.

## Layout

```
src/pathoquant/
  model.py        U-Net
  losses.py       Dice + BCE
  metrics.py      Dice, AJI
  tiling.py       overlap tile + stitch
  classical.py    Otsu / watershed
  quantify.py     count / density
  train.py
  evaluate.py
```

## Run

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
python -m pathoquant.train --data data/processed --epochs 40
python -m pathoquant.evaluate --pred outputs/pred.npy --gt outputs/gt.npy
```

## Cite

Kumar et al., “A Dataset and a Technique for Generalized Nuclear Segmentation,” IEEE TMI, 2017.
