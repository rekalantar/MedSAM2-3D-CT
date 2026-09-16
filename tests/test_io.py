"""Round-trip through a real NIfTI, because the axis-order trap is silent."""
import numpy as np
import pytest

sitk = pytest.importorskip("SimpleITK")

from medsam2_ct import load_volume  # noqa: E402


def test_load_volume_returns_spacing_in_array_axis_order(tmp_path):
    """SimpleITK reports spacing as (x, y, z); the array is (z, y, x).

    If these ever stop being reversed relative to each other, every physical
    distance computed downstream is silently wrong on anisotropic scans.
    """
    volume = np.zeros((40, 128, 96), dtype=np.int16)
    image = sitk.GetImageFromArray(volume)
    image.SetSpacing((0.7, 0.8, 3.0))  # (x, y, z)

    path = tmp_path / "scan.nii.gz"
    sitk.WriteImage(image, str(path))

    array, spacing = load_volume(str(path))

    assert array.shape == (40, 128, 96)
    assert np.allclose(spacing, (3.0, 0.8, 0.7))    # (z, y, x)
    # the distinct values make a silent transposition impossible to miss
    assert spacing[0] != spacing[-1]


def test_load_volume_preserves_values(tmp_path):
    volume = np.arange(2 * 3 * 4, dtype=np.int16).reshape(2, 3, 4)
    image = sitk.GetImageFromArray(volume)
    image.SetSpacing((1.0, 1.0, 1.0))
    path = tmp_path / "v.nii.gz"
    sitk.WriteImage(image, str(path))

    array, _ = load_volume(str(path))
    assert np.array_equal(array, volume)
