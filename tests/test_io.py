import numpy as np
import SimpleITK as sitk

from branchseed.io import (
    array_to_image,
    image_to_array,
    numpy_index_to_physical_point,
    numpy_zyx_to_sitk_xyz,
    read_image,
    write_image,
)


def test_array_round_trip_preserves_geometry(tmp_path):
    image = sitk.Image((5, 4, 3), sitk.sitkFloat32)
    image.SetSpacing((0.7, 1.2, 2.5))
    image.SetOrigin((10.0, -2.0, 4.0))
    image.SetDirection((0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0))
    array = np.arange(60, dtype=np.float32).reshape(3, 4, 5)

    rebuilt = array_to_image(array, image)
    path = tmp_path / "image.mha"
    write_image(rebuilt, path)
    loaded = read_image(path)

    np.testing.assert_array_equal(image_to_array(loaded), array)
    assert loaded.GetSpacing() == image.GetSpacing()
    assert loaded.GetOrigin() == image.GetOrigin()
    assert loaded.GetDirection() == image.GetDirection()


def test_numpy_indices_are_reordered_before_physical_conversion():
    image = sitk.Image((20, 20, 20), sitk.sitkUInt8)
    image.SetSpacing((2.0, 3.0, 4.0))
    image.SetOrigin((10.0, 20.0, 30.0))

    assert numpy_zyx_to_sitk_xyz((3, 2, 1)) == (1.0, 2.0, 3.0)
    assert numpy_index_to_physical_point(image, (3, 2, 1)) == (12.0, 26.0, 42.0)
