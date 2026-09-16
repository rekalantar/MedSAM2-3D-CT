"""CT loading and intensity windowing.

Windowing is the step most often skipped, and skipping it is why promptable models
appear to fail on CT. Hounsfield units span several thousand; an 8-bit network input
spans 256. Min-maxing the full range collapses soft tissue into a handful of grey
levels and the model has nothing to work with.
"""
from __future__ import annotations

import numpy as np

# (window_width, window_level) in Hounsfield units
PRESETS = {
    "abdomen": (400, 40),
    "lung": (1500, -600),
    "brain": (80, 40),
    "bone": (1800, 400),
    "mediastinum": (350, 50),
}


def window_hu(volume: np.ndarray, width: float = 400, level: float = 40) -> np.ndarray:
    """Clip a Hounsfield-unit volume to a window and rescale to uint8."""
    if width <= 0:
        raise ValueError("window width must be positive")
    lower, upper = level - width / 2, level + width / 2
    volume = np.clip(volume, lower, upper)
    return ((volume - lower) / (upper - lower) * 255).astype(np.uint8)


def window_preset(volume: np.ndarray, preset: str = "abdomen") -> np.ndarray:
    """Window using a named clinical preset. See PRESETS."""
    if preset not in PRESETS:
        raise KeyError(f"unknown preset {preset!r}; choose from {sorted(PRESETS)}")
    return window_hu(volume, *PRESETS[preset])


def load_volume(path: str) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Load a NIfTI or DICOM-series volume.

    Returns the array in (z, y, x) order and voxel spacing in mm, also (z, y, x).
    SimpleITK reports spacing as (x, y, z), so it is reversed here to match the array
    axes — getting this backwards silently corrupts any physical-distance measurement
    made downstream, with no error and perfectly plausible numbers.
    """
    try:
        import SimpleITK as sitk
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            'load_volume needs SimpleITK: pip install "medsam2-ct[io]"'
        ) from exc

    image = sitk.ReadImage(path)
    return sitk.GetArrayFromImage(image), tuple(reversed(image.GetSpacing()))
