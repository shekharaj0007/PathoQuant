"""Nuclei count, area, and density from an instance mask."""

from __future__ import annotations

import numpy as np


def quantify(instance_mask: np.ndarray, micron_per_pixel: float = 0.25) -> dict:
    ids = np.unique(instance_mask)
    ids = ids[ids != 0]
    areas_px = [int((instance_mask == i).sum()) for i in ids]
    h, w = instance_mask.shape
    area_mm2 = (h * micron_per_pixel / 1000.0) * (w * micron_per_pixel / 1000.0)
    return {
        "n_nuclei": int(len(ids)),
        "mean_area_px": float(np.mean(areas_px) if areas_px else 0.0),
        "density_per_mm2": float(len(ids) / max(area_mm2, 1e-9)),
    }
