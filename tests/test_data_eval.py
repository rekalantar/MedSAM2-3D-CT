import numpy as np
import pytest

from medsam2_ct import box_from_mask, dice, largest_cross_section, per_slice_dice


def _lesion(shape=(12, 64, 80)):
    m = np.zeros(shape, dtype=bool)
    m[3:9, 20:40, 30:55] = True
    m[5, 18:44, 28:58] = True          # slice 5 is widest
    return m


def test_dice_identical_and_disjoint():
    m = _lesion()
    assert dice(m, m) == 1.0
    assert dice(m, np.zeros_like(m)) == 0.0


def test_dice_two_empty_masks_agree():
    z = np.zeros((4, 8, 8), dtype=bool)
    assert dice(z, z) == 1.0


def test_largest_cross_section_finds_the_widest_slice():
    assert largest_cross_section(_lesion()) == 5


def test_largest_cross_section_rejects_empty():
    with pytest.raises(ValueError):
        largest_cross_section(np.zeros((4, 8, 8), dtype=bool))


def test_box_from_mask_adds_margin():
    mask = np.zeros((64, 80), dtype=bool)
    mask[20:40, 30:55] = True
    assert box_from_mask(mask, margin=5) == [25, 15, 59, 44]


def test_box_from_mask_clamps_at_the_edges():
    mask = np.zeros((20, 20), dtype=bool)
    mask[0:3, 0:3] = True
    x0, y0, x1, y1 = box_from_mask(mask, margin=10)
    assert (x0, y0) == (0, 0)
    assert x1 <= 19 and y1 <= 19


def test_box_from_mask_rejects_empty_slice():
    with pytest.raises(ValueError):
        box_from_mask(np.zeros((8, 8), dtype=bool))


def test_per_slice_dice_offsets_are_relative_to_the_prompt():
    truth = _lesion()
    offsets, scores = per_slice_dice(truth, truth, key_slice=5)
    assert offsets.tolist() == [-2, -1, 0, 1, 2, 3]
    assert np.allclose(scores, 1.0)


def test_per_slice_dice_absolute_indices_without_a_key():
    truth = _lesion()
    offsets, _ = per_slice_dice(truth, truth)
    assert offsets.tolist() == [3, 4, 5, 6, 7, 8]


def test_per_slice_dice_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        per_slice_dice(_lesion(), np.zeros((12, 64, 79), dtype=bool))


def test_per_slice_dice_rejects_empty_truth():
    with pytest.raises(ValueError):
        per_slice_dice(np.zeros((6, 8, 8), dtype=bool), np.zeros((6, 8, 8), dtype=bool))
