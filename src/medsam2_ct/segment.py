"""One-box 3D propagation with MedSAM2.

Requires the MedSAM2 repository and its weights, which are not redistributed here:

    git clone https://github.com/bowang-lab/MedSAM2.git && cd MedSAM2
    pip install -e ".[dev]"
    bash download.sh

Weights are licensed for research and education use only.
Paper: Ma et al., arXiv:2504.03600
"""
from __future__ import annotations

import numpy as np

CONFIG = "configs/sam2.1_hiera_t512.yaml"
CHECKPOINT = "./checkpoints/MedSAM2_latest.pt"

IMAGE_SIZE = 512
# SAM 2 inherits ImageNet normalisation from its natural-image pretraining.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def resize_grayscale_to_rgb(volume_u8: np.ndarray, size: int = IMAGE_SIZE) -> np.ndarray:
    """(D, H, W) uint8 -> (D, 3, size, size) uint8.

    The encoder is a natural-image backbone: it wants three channels at a fixed
    resolution, so each slice is bilinearly resized and repeated across RGB.
    """
    from PIL import Image

    if volume_u8.ndim != 3:
        raise ValueError(f"expected a (D, H, W) volume, got shape {volume_u8.shape}")
    if volume_u8.dtype != np.uint8:
        raise TypeError(f"expected uint8 (window it first), got {volume_u8.dtype}")

    out = np.zeros((len(volume_u8), 3, size, size), dtype=np.uint8)
    for i, plane in enumerate(volume_u8):
        resized = Image.fromarray(plane).resize((size, size), Image.BILINEAR)
        out[i] = np.asarray(resized)[None].repeat(3, axis=0)
    return out


def preprocess(volume_u8: np.ndarray, size: int = IMAGE_SIZE, device: str | None = None):
    """(D, H, W) uint8 -> normalised float32 tensor of shape (D, 3, size, size).

    This is what `init_state` expects. Passing the raw array instead fails deep
    inside the predictor with an unhelpful AttributeError about `.to`.
    """
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tensor = torch.from_numpy(resize_grayscale_to_rgb(volume_u8, size)).to(device).float()
    tensor /= 255.0
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    tensor -= mean
    tensor /= std
    return tensor


def build_predictor(config: str = CONFIG, checkpoint: str = CHECKPOINT):
    """Build the MedSAM2 video predictor. Note: *video* predictor — that's the point."""
    from sam2.build_sam import build_sam2_video_predictor_npz
    return build_sam2_video_predictor_npz(config, checkpoint)


def init_state(predictor, volume_u8: np.ndarray, size: int = IMAGE_SIZE):
    """Preprocess a volume and open an inference state for it.

    `video_height`/`video_width` are the volume's *original* dimensions, not the
    resized ones. That matters twice: prompts are given in original coordinates
    (the predictor normalises them internally), and masks come back at original
    resolution.
    """
    _, height, width = volume_u8.shape
    return predictor.init_state(preprocess(volume_u8, size), height, width), height, width


def segment_volume(predictor, volume_u8: np.ndarray, box, key_slice: int) -> np.ndarray:
    """Propagate one box prompt through an entire volume, in both directions.

    Args:
        predictor:  from build_predictor()
        volume_u8:  (z, y, x) uint8, already windowed — see window_hu
        box:        [x_min, y_min, x_max, y_max] in the volume's own pixel
                    coordinates on `key_slice`, not resized coordinates
        key_slice:  index of the slice to prompt; pick the largest cross-section

    Returns:
        (z, y, x) boolean mask at the volume's original resolution.
    """
    depth = len(volume_u8)
    if not 0 <= key_slice < depth:
        raise IndexError(f"key_slice {key_slice} outside volume of depth {depth}")

    masks = np.zeros(volume_u8.shape, dtype=bool)
    state, _, _ = init_state(predictor, volume_u8)
    predictor.add_new_points_or_box(
        inference_state=state,
        frame_idx=key_slice,
        obj_id=1,
        box=np.asarray(box, dtype=np.float32),
    )

    # The key slice sits mid-lesion, so forward propagation alone captures only half
    # the volume. Sweeping backward as well is not optional.
    for reverse in (False, True):
        for idx, _obj_ids, logits in predictor.propagate_in_video(state, reverse=reverse):
            masks[idx] = (logits[0] > 0).cpu().numpy().squeeze()

    return masks


def largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest connected component.

    Propagation can leak into neighbouring structures and this cleans that up.
    It will also delete genuine satellite lesions in multifocal disease — know your
    data before switching it on.
    """
    from scipy.ndimage import label

    labelled, count = label(mask)
    if count <= 1:
        return mask
    sizes = np.bincount(labelled.ravel())
    sizes[0] = 0
    return labelled == sizes.argmax()
