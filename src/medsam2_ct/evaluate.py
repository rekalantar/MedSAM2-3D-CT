"""Scoring a propagated mask against ground truth."""
from __future__ import annotations

import numpy as np


def dice(a, b) -> float:
    """Volumetric overlap. Two empty masks agree perfectly, by convention."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    total = a.sum() + b.sum()
    return 1.0 if total == 0 else 2 * (a & b).sum() / total


def per_slice_dice(truth: np.ndarray, pred: np.ndarray, key_slice: int | None = None):
    """Dice on every slice the ground truth touches.

    A volume Dice is one number over slices that are not equally hard: the prompted
    slice was seen directly, the rest were inferred. Returns `(offsets, scores)`,
    where offsets are relative to `key_slice` — or absolute indices if it is None.
    """
    truth, pred = np.asarray(truth, bool), np.asarray(pred, bool)
    if truth.shape != pred.shape:
        raise ValueError(f"shape mismatch: {truth.shape} vs {pred.shape}")

    indices = truth.any(axis=(1, 2)).nonzero()[0]
    if len(indices) == 0:
        raise ValueError("ground truth is empty")

    scores = np.array([dice(truth[i], pred[i]) for i in indices])
    offsets = indices if key_slice is None else indices - key_slice
    return offsets, scores
