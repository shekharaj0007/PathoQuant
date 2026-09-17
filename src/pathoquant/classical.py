"""Otsu + watershed baseline for touching-nuclei comparison."""

from __future__ import annotations

import cv2
import numpy as np
from skimage.feature import peak_local_max
from skimage.segmentation import watershed


def otsu_watershed(he_rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(he_rgb, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    coords = peak_local_max(dist, min_distance=7, labels=opening)
    markers = np.zeros_like(opening, dtype=np.int32)
    for i, (r, c) in enumerate(coords, start=1):
        markers[r, c] = i
    labels = watershed(-dist, markers, mask=opening)
    return labels.astype(np.int32)
