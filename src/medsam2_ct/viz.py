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


def rotate_clockwise(array, quarter_turns=1):
    """Rotate the in-plane axes of a (D, H, W) stack clockwise.

    NIfTI arrays arrive in whatever orientation the affine implies, which is often
    not how a radiologist expects to read an axial slice. This only changes display,
    never the voxel values or their correspondence between volume and mask — so
    rotate both, or neither.
    """
    if quarter_turns % 4 == 0:
        return array
    return np.rot90(array, k=-quarter_turns, axes=(1, 2))


def save_gif(volume_u8, masks, path, fps=8, color=CYAN, alpha=0.4, rotate=0):
    """Render a volume and its per-slice masks as a scrolling GIF.

    `rotate` is the number of clockwise quarter-turns applied for display.
    """
    if len(volume_u8) != len(masks):
        raise ValueError(
            f"volume has {len(volume_u8)} slices but masks has {len(masks)}"
        )

    volume_u8 = rotate_clockwise(volume_u8, rotate)
    masks = rotate_clockwise(masks, rotate)

    try:
        import imageio.v2 as imageio
    except ImportError as exc:  # pragma: no cover
        raise ImportError("save_gif needs imageio: pip install imageio") from exc
    frames = [overlay_mask(volume_u8[i], masks[i], color=color, alpha=alpha)
              for i in range(len(volume_u8))]
    imageio.mimsave(path, frames, fps=fps, loop=0)
    return path


def plot_slices(volume_u8, masks, truth=None, indices=None, rotate=0,
                crop_pad=40, ncols=None, title=None):
    """Contact sheet of slices with contours drawn over them.

    Shows prediction in coral and, when given, ground truth in cyan. Cropped to the
    masked region so the structure is actually legible rather than a speck in a
    512-pixel field. Returns the matplotlib figure.
    """
    import matplotlib.pyplot as plt

    region = masks if truth is None else (masks | truth)
    if indices is None:
        indices = region.any(axis=(1, 2)).nonzero()[0]
    indices = np.atleast_1d(indices)
    if len(indices) == 0:
        raise ValueError("nothing to show: no slice contains a mask")

    ys, xs = np.where(region.any(axis=0))
    y0, y1 = max(0, ys.min() - crop_pad), min(region.shape[1], ys.max() + crop_pad + 1)
    x0, x1 = max(0, xs.min() - crop_pad), min(region.shape[2], xs.max() + crop_pad + 1)

    ncols = ncols or len(indices)
    nrows = int(np.ceil(len(indices) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.2 * ncols, 2.5 * nrows),
                             squeeze=False)
    for ax in axes.flat:
        ax.axis("off")

    def crop(plane):
        return rotate_clockwise(plane[None, y0:y1, x0:x1], rotate)[0]

    # the grid may hold more axes than slices, hence indexing rather than zip
    for n, i in enumerate(indices):
        ax = axes.flat[n]
        ax.imshow(crop(volume_u8[i]), cmap="gray")
        if truth is not None:
            ax.contour(crop(truth[i]), levels=[0.5],
                       colors=[np.array(CYAN) / 255], linewidths=1.4)
        ax.contour(crop(masks[i]), levels=[0.5],
                   colors=[np.array(CORAL) / 255], linewidths=1.4)
        ax.set_title(f"slice {i}", fontsize=9)

    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig
