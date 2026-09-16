"""The encoder is a natural-image backbone, so volumes need natural-image shaping."""
import numpy as np
import pytest

from medsam2_ct.segment import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, resize_grayscale_to_rgb


def _volume(d=5, h=40, w=60):
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (d, h, w), dtype=np.uint8)


def test_resize_produces_three_channels_at_model_resolution():
    out = resize_grayscale_to_rgb(_volume())
    assert out.shape == (5, 3, IMAGE_SIZE, IMAGE_SIZE)
    assert out.dtype == np.uint8


def test_resize_channels_are_identical():
    out = resize_grayscale_to_rgb(_volume())
    assert np.array_equal(out[:, 0], out[:, 1])
    assert np.array_equal(out[:, 0], out[:, 2])


def test_resize_handles_non_square_input():
    out = resize_grayscale_to_rgb(_volume(d=3, h=17, w=93), size=64)
    assert out.shape == (3, 3, 64, 64)


def test_resize_rejects_non_uint8():
    """Windowing must happen first; float input silently means it didn't."""
    with pytest.raises(TypeError):
        resize_grayscale_to_rgb(np.zeros((2, 8, 8), dtype=np.float32))


def test_resize_rejects_wrong_rank():
    with pytest.raises(ValueError):
        resize_grayscale_to_rgb(np.zeros((8, 8), dtype=np.uint8))


def test_normalisation_constants_are_imagenet():
    assert IMAGENET_MEAN == (0.485, 0.456, 0.406)
    assert IMAGENET_STD == (0.229, 0.224, 0.225)


def test_preprocess_shape_and_range():
    torch = pytest.importorskip("torch")
    from medsam2_ct.segment import preprocess

    out = preprocess(_volume(), device="cpu")
    assert out.shape == (5, 3, IMAGE_SIZE, IMAGE_SIZE)
    assert out.dtype == torch.float32
    # normalised, so values sit roughly in [-2.2, 2.7] rather than [0, 255]
    assert out.min() > -3 and out.max() < 3


def test_preprocess_matches_manual_normalisation():
    pytest.importorskip("torch")
    import torch

    from medsam2_ct.segment import preprocess

    vol = _volume(d=2, h=16, w=16)
    got = preprocess(vol, size=32, device="cpu")

    expected = torch.from_numpy(resize_grayscale_to_rgb(vol, 32)).float() / 255.0
    expected -= torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    expected /= torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    assert torch.allclose(got, expected)
