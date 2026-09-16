"""Segment a 3D CT volume by prompting a single slice.

SAM 2 propagates a prompt across video frames with memory attention. A CT volume
has the same structure — adjacent slices differ only slightly — so one box on one
slice can be tracked through the whole stack. MedSAM2 is SAM 2.1 fine-tuned so
this works on medical contrast rather than natural-image contrast.

    from medsam2_ct import window_hu, build_predictor, segment_volume, save_gif

    volume = window_hu(raw_ct, width=400, level=40)
    predictor = build_predictor()
    masks = segment_volume(predictor, volume, box=[180, 180, 260, 260], key_slice=47)
    save_gif(volume, masks, "propagation.gif")
"""
from .ct import PRESETS, load_volume, window_hu, window_preset
from .segment import build_predictor, largest_component, segment_volume
from .viz import overlay_mask, save_gif

__version__ = "0.1.0"
__all__ = [
    "window_hu", "window_preset", "load_volume", "PRESETS",
    "build_predictor", "segment_volume", "largest_component",
    "overlay_mask", "save_gif",
]
