"""Loading cases and deriving a prompt from them."""
from __future__ import annotations

import numpy as np

from .ct import load_volume, window_hu

DEMO_DATASET = "wanglab/CT_DeepLesion-MedSAM2"


def largest_cross_section(mask: np.ndarray) -> int:
    """Index of the slice carrying the most mask voxels — the clearest one to prompt."""
    mask = np.asarray(mask, bool)
    if not mask.any():
        raise ValueError("mask is empty")
    return int(mask.sum(axis=(1, 2)).argmax())


def box_from_mask(mask_2d: np.ndarray, margin: int = 5) -> list[int]:
    """`[x_min, y_min, x_max, y_max]` around a 2D mask, with a margin.

    Coordinates are in the array's own pixel space. The predictor normalises them
    against the volume's dimensions, so they must not be pre-scaled to the model's
    input size — and must not be rotated for display either.
    """
    mask_2d = np.asarray(mask_2d, bool)
    if not mask_2d.any():
        raise ValueError("mask is empty on this slice")
    ys, xs = np.where(mask_2d)
    height, width = mask_2d.shape
    return [max(0, int(xs.min()) - margin), max(0, int(ys.min()) - margin),
            min(width - 1, int(xs.max()) + margin),
            min(height - 1, int(ys.max()) + margin)]


def load_demo_case(case: str, dataset: str = DEMO_DATASET, width: float = 400,
                   level: float = 40, margin: int = 5) -> dict:
    """Fetch one case of the MedSAM2 demo set, window it, and place a prompt.

    The prompt is derived from the ground-truth label. That is how promptable models
    are normally evaluated — it simulates a perfect user prompt, so what you measure
    is propagation quality rather than whether a human drew a good box.

    Returns a dict with `name`, `hu`, `volume`, `truth`, `spacing`, `key`, `box`
    and `span`.
    """
    from huggingface_hub import hf_hub_download

    image = hf_hub_download(dataset, f"images/{case}_0000.nii.gz", repo_type="dataset")
    label = hf_hub_download(dataset, f"labels/{case}.nii.gz", repo_type="dataset")

    volume_hu, spacing = load_volume(image)
    truth = load_volume(label)[0] > 0
    key = largest_cross_section(truth)

    return {
        "name": case,
        "hu": volume_hu,
        "volume": window_hu(volume_hu, width, level),
        "truth": truth,
        "spacing": spacing,
        "key": key,
        "box": box_from_mask(truth[key], margin),
        "span": int(truth.any(axis=(1, 2)).sum()),
    }
