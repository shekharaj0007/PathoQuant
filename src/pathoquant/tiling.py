"""Constant-memory overlap tiling for ~50k×50k WSIs."""

from __future__ import annotations

from typing import Iterator

import numpy as np


def tile_coords(h: int, w: int, tile: int = 256, stride: int = 128) -> list[tuple[int, int, int, int]]:
    ys = list(range(0, max(h - tile, 0) + 1, stride))
    xs = list(range(0, max(w - tile, 0) + 1, stride))
    if not ys or ys[-1] != max(h - tile, 0):
        ys.append(max(h - tile, 0))
    if not xs or xs[-1] != max(w - tile, 0):
        xs.append(max(w - tile, 0))
    return [(y, x, tile, tile) for y in ys for x in xs]


def tile_image(image: np.ndarray, tile: int = 256, stride: int = 128) -> Iterator[tuple[np.ndarray, int, int]]:
    h, w = image.shape[:2]
    for y, x, th, tw in tile_coords(h, w, tile, stride):
        yield image[y : y + th, x : x + tw], y, x


def _gaussian_weight(tile: int) -> np.ndarray:
    ax = np.linspace(-1.0, 1.0, tile)
    xx, yy = np.meshgrid(ax, ax)
    w = np.exp(-0.5 * (xx**2 + yy**2) / 0.35**2)
    return w.astype(np.float32)


def stitch_tiles(
    tiles: list[tuple[np.ndarray, int, int]],
    out_h: int,
    out_w: int,
    tile: int = 256,
) -> np.ndarray:
    acc = np.zeros((out_h, out_w), dtype=np.float32)
    weight = np.zeros((out_h, out_w), dtype=np.float32)
    gw = _gaussian_weight(tile)
    for patch, y, x in tiles:
        ph, pw = patch.shape[:2]
        acc[y : y + ph, x : x + pw] += patch.astype(np.float32) * gw[:ph, :pw]
        weight[y : y + ph, x : x + pw] += gw[:ph, :pw]
    return acc / np.maximum(weight, 1e-6)
