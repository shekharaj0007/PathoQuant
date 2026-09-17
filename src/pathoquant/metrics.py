"""Dice and Aggregated Jaccard Index (Kumar et al., IEEE TMI 2017)."""

from __future__ import annotations

import numpy as np


def dice_coefficient(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-7) -> float:
    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)
    inter = np.logical_and(pred_b, gt_b).sum()
    return float((2.0 * inter + eps) / (pred_b.sum() + gt_b.sum() + eps))


def aggregated_jaccard_index(pred_inst: np.ndarray, gt_inst: np.ndarray) -> float:
    """AJI between labeled instance maps (0 = background)."""
    gt_ids = np.unique(gt_inst)
    gt_ids = gt_ids[gt_ids != 0]
    pred_ids = np.unique(pred_inst)
    pred_ids = pred_ids[pred_ids != 0]
    if gt_ids.size == 0:
        return 1.0 if pred_ids.size == 0 else 0.0

    gt_flat = gt_inst.ravel()
    pred_flat = pred_inst.ravel()
    overlap = np.zeros((int(gt_inst.max()) + 1, int(pred_inst.max()) + 1), dtype=np.int64)
    np.add.at(overlap, (gt_flat, pred_flat), 1)

    used = set()
    inter_sum = 0.0
    union_sum = 0.0
    for gid in gt_ids:
        g_area = overlap[gid].sum()
        best_iou, best_pid = 0.0, 0
        for pid in pred_ids:
            if pid in used:
                continue
            inter = overlap[gid, pid]
            union = g_area + overlap[:, pid].sum() - inter
            iou = inter / union if union else 0.0
            if iou > best_iou:
                best_iou, best_pid = iou, int(pid)
        if best_pid:
            used.add(best_pid)
            inter_sum += overlap[gid, best_pid]
            union_sum += g_area + overlap[:, best_pid].sum() - overlap[gid, best_pid]
        else:
            union_sum += g_area

    leftover = sum(int(overlap[:, pid].sum()) for pid in pred_ids if pid not in used)
    return float(inter_sum / (union_sum + leftover + 1e-7))
