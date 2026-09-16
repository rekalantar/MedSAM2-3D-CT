"""Mask overlay and animation.

Watch the propagation rather than trusting a summary number. Failure modes are
obvious to the eye and invisible in a mean.
"""
from __future__ import annotations

import numpy as np

CYAN = (63, 193, 201)
CORAL = (255, 107, 91)


def _erode(mask: np.ndarray) -> np.ndarray:
    from scipy.ndimage import binary_erosion, generate_binary_structure
    return binary_erosion(mask, generate_binary_structure(2, 1))


def overlay_mask(slice_u8, mask, color=CYAN, alpha=0.4, outline=True):
    """Blend a binary mask over an 8-bit greyscale slice. Returns RGB uint8."""
    rgb = np.stack([slice_u8] * 3, axis=-1).astype(np.float32)
    mask = np.asarray(mask, dtype=bool)
    if mask.any():
        rgb[mask] = (1 - alpha) * rgb[mask] + alpha * np.array(color, dtype=np.float32)
        if outline:
            rgb[mask ^ _erode(mask)] = color
    return np.clip(rgb, 0, 255).astype(np.uint8)


def save_gif(volume_u8, masks, path, fps=8, color=CYAN, alpha=0.4):
    """Render a volume and its per-slice masks as a scrolling GIF."""
    if len(volume_u8) != len(masks):
        raise ValueError(
            f"volume has {len(volume_u8)} slices but masks has {len(masks)}"
        )

    try:
        import imageio.v2 as imageio
    except ImportError as exc:  # pragma: no cover
        raise ImportError("save_gif needs imageio: pip install imageio") from exc
    frames = [overlay_mask(volume_u8[i], masks[i], color=color, alpha=alpha)
              for i in range(len(volume_u8))]
    imageio.mimsave(path, frames, fps=fps, loop=0)
    return path
