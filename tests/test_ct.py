import numpy as np
import pytest

from medsam2_ct import PRESETS, overlay_mask, window_hu, window_preset
from medsam2_ct.segment import largest_component
from medsam2_ct.viz import save_gif


def test_windowing_returns_uint8_full_range():
    volume = np.linspace(-1000, 1000, 1000).reshape(10, 10, 10)
    out = window_hu(volume, width=400, level=40)
    assert out.dtype == np.uint8
    assert out.min() == 0 and out.max() == 255


def test_windowing_clips_outside_the_window():
    volume = np.array([[[-2000, 40, 2000]]], dtype=float)
    out = window_hu(volume, width=400, level=40)
    assert out[0, 0, 0] == 0
    assert out[0, 0, 2] == 255


def test_narrow_window_separates_soft_tissue():
    """The claim that windowing matters, as an executable check.

    A realistic CT phantom: soft tissue around 40 HU, air at -1000, bone at +1000.
    Naive full-range scaling spends most of its 256 levels on the air-to-bone gap
    and leaves soft tissue almost flat. Windowing spends all of them where the
    lesion is.
    """
    rng = np.random.default_rng(0)
    soft = rng.normal(40, 30, 4000).reshape(10, 20, 20)
    air = np.full((10, 20, 20), -1000.0)
    bone = np.full((10, 20, 20), 1000.0)
    volume = np.concatenate([soft, air, bone], axis=0)

    windowed = window_hu(volume, width=400, level=40)[:10]
    lo, hi = volume.min(), volume.max()
    naive = ((volume - lo) / (hi - lo) * 255).astype(np.uint8)[:10]

    # soft tissue gets at least 4x more grey levels to work with
    assert len(np.unique(windowed)) > 4 * len(np.unique(naive))


def test_invalid_width_rejected():
    with pytest.raises(ValueError):
        window_hu(np.zeros((2, 2, 2)), width=0)


def test_unknown_preset_rejected():
    with pytest.raises(KeyError):
        window_preset(np.zeros((2, 2, 2)), "elbow")


def test_presets_are_sane():
    for name, (width, level) in PRESETS.items():
        assert width > 0, name
        assert -1000 < level < 1000, name


def test_largest_component_drops_islands():
    mask = np.zeros((20, 20, 20), dtype=bool)
    mask[2:10, 2:10, 2:10] = True      # big
    mask[15:17, 15:17, 15:17] = True   # island
    out = largest_component(mask)
    assert out.sum() == 8 ** 3
    assert not out[15:17, 15:17, 15:17].any()


def test_largest_component_passes_through_single_object():
    mask = np.zeros((10, 10, 10), dtype=bool)
    mask[2:6, 2:6, 2:6] = True
    assert largest_component(mask).sum() == mask.sum()


def test_overlay_marks_only_the_mask():
    sl = np.zeros((16, 16), dtype=np.uint8)
    mask = np.zeros((16, 16), dtype=bool)
    mask[4:12, 4:12] = True
    out = overlay_mask(sl, mask)
    assert out.shape == (16, 16, 3)
    assert out[mask].any()
    assert not out[~mask].any()


def test_save_gif_rejects_length_mismatch(tmp_path):
    with pytest.raises(ValueError):
        save_gif(np.zeros((5, 8, 8), np.uint8), np.zeros((4, 8, 8), bool),
                 tmp_path / "x.gif")


def test_rotate_clockwise_is_a_quarter_turn():
    from medsam2_ct import rotate_clockwise

    stack = np.zeros((1, 4, 4), dtype=np.uint8)
    stack[0, 0, 0] = 9                       # top-left
    out = rotate_clockwise(stack, 1)
    assert out[0, 0, 3] == 9                 # clockwise -> top-right
    assert out.shape == stack.shape


def test_rotate_four_turns_is_identity():
    from medsam2_ct import rotate_clockwise

    rng = np.random.default_rng(0)
    stack = rng.integers(0, 255, (3, 6, 6), dtype=np.uint8)
    assert np.array_equal(rotate_clockwise(stack, 4), stack)
    assert np.array_equal(rotate_clockwise(stack, 0), stack)


def test_rotate_keeps_volume_and_mask_aligned():
    from medsam2_ct import rotate_clockwise

    mask = np.zeros((2, 8, 8), dtype=bool)
    mask[:, 1, 6] = True
    volume = np.zeros((2, 8, 8), dtype=np.uint8)
    volume[mask] = 255
    r_vol, r_mask = rotate_clockwise(volume, 1), rotate_clockwise(mask, 1)
    assert (r_vol[r_mask] == 255).all()


def test_plot_slices_builds_a_grid(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    from medsam2_ct import plot_slices

    rng = np.random.default_rng(0)
    volume = rng.integers(0, 255, (12, 200, 260), dtype=np.uint8)
    truth = np.zeros((12, 200, 260), dtype=bool)
    truth[3:9, 90:120, 80:115] = True
    pred = np.zeros_like(truth)
    pred[3:9, 92:118, 82:113] = True

    fig = plot_slices(volume, pred, truth, rotate=1)
    assert len(fig.axes) == 6                    # one per masked slice
    out = tmp_path / "sheet.png"
    fig.savefig(out)
    assert out.stat().st_size > 0

    grid = plot_slices(volume, pred, truth, ncols=4)
    assert len(grid.axes) == 8                   # 2 rows x 4, two left blank


def test_plot_slices_rejects_empty_mask():
    import matplotlib
    matplotlib.use("Agg")
    from medsam2_ct import plot_slices

    volume = np.zeros((4, 32, 32), dtype=np.uint8)
    with pytest.raises(ValueError):
        plot_slices(volume, np.zeros((4, 32, 32), dtype=bool))
