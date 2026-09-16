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


def build_predictor(config: str = CONFIG, checkpoint: str = CHECKPOINT):
    """Build the MedSAM2 video predictor. Note: *video* predictor — that's the point."""
    from sam2.build_sam import build_sam2_video_predictor_npz
    return build_sam2_video_predictor_npz(config, checkpoint)


def segment_volume(predictor, volume_u8: np.ndarray, box, key_slice: int) -> np.ndarray:
    """Propagate one box prompt through an entire volume, in both directions.

    Args:
        predictor:  from build_predictor()
        volume_u8:  (z, y, x) uint8, already windowed — see window_hu
        box:        [x_min, y_min, x_max, y_max] on `key_slice`
        key_slice:  index of the slice to prompt; pick the largest cross-section

    Returns:
        (z, y, x) boolean mask.
    """
    depth, height, width = volume_u8.shape
    if not 0 <= key_slice < depth:
        raise IndexError(f"key_slice {key_slice} outside volume of depth {depth}")

    masks = np.zeros(volume_u8.shape, dtype=bool)
    state = predictor.init_state(volume_u8, height, width)
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
