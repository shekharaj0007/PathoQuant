"""Download MoNuSeg (HuggingFace RationAI) and write image + instance masks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


TISSUE = {
    0: "Unknown",
    1: "Breast",
    2: "Kidney",
    3: "Liver",
    4: "Prostate",
    5: "Bladder",
    6: "Colon",
    7: "Stomach",
}


def _to_array(mask) -> np.ndarray:
    arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr


def instances_to_labelmap(instances, shape: tuple[int, int]) -> np.ndarray:
    lab = np.zeros(shape, dtype=np.uint16)
    for i, m in enumerate(instances, start=1):
        a = _to_array(m)
        if a.shape != shape:
            a = np.array(Image.fromarray(a.astype(np.uint8)).resize((shape[1], shape[0]), Image.NEAREST))
        lab[a > 0] = i
    return lab


def save_split(ds, out_dir: Path) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    index: list[dict] = []
    for row in tqdm(ds, desc=f"writing {out_dir.name}"):
        patient = str(row["patient"])
        image = row["image"].convert("RGB")
        rgb = np.asarray(image)
        labels = instances_to_labelmap(row["instances"], rgb.shape[:2])
        stem = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in patient)
        img_path = out_dir / f"{stem}.png"
        npy_path = out_dir / f"{stem}_instances.npy"
        image.save(img_path)
        np.save(npy_path, labels)
        rec = {
            "id": stem,
            "patient": patient,
            "tissue": TISSUE.get(int(row["tissue"]), "Unknown"),
            "tissue_id": int(row["tissue"]),
            "n_nuclei": int(labels.max()),
            "image": str(img_path.as_posix()),
            "instances": str(npy_path.as_posix()),
        }
        index.append(rec)
    (out_dir / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare MoNuSeg tiles for PathoQuant")
    parser.add_argument("--out", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    from datasets import load_dataset

    print("Downloading RationAI/MoNuSeg from Hugging Face…")
    bundle = load_dataset("RationAI/MoNuSeg")
    train_idx = save_split(bundle["train"], args.out / "train")
    test_idx = save_split(bundle["test"], args.out / "test")
    print(f"train tiles: {len(train_idx)}  test tiles: {len(test_idx)}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
