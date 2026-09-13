"""Coordinate handling. Written first, per the spec: the axis-order hazard is
the single most likely source of silent geometric error in the project."""

from __future__ import annotations

import gzip
import shutil

import numpy as np
import pytest
import SimpleITK as sitk

from branchseed import io as bio
from branchseed.interp import points_to_coords, sample_nearest, sample_volume, voxel_indices
from branchseed.roi import build_roi
from branchseed.types import Flags


def test_array_order_is_zyx(standard_phantom):
    """GetArrayFromImage is [z, y, x] while GetSize is (x, y, z)."""
    size_xyz = standard_phantom.image.GetSize()
    arr = sitk.GetArrayFromImage(standard_phantom.image)
    assert arr.shape == size_xyz[::-1]


def test_points_to_coords_flips_axes():
    pts = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    coords = points_to_coords(pts)
    assert coords.shape == (3, 2)
    np.testing.assert_allclose(coords[:, 0], [3.0, 2.0, 1.0])


def test_sampling_uses_xyz_points():
    """A volume that is a ramp in x must read back as a ramp in the x component."""
    arr = np.zeros((4, 5, 6), dtype=np.float32)
    arr[:, :, :] = np.arange(6, dtype=np.float32)[None, None, :]
    pts = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [5.0, 4.0, 3.0]])
    np.testing.assert_allclose(sample_volume(arr, pts, order=1), [0.0, 3.0, 5.0], atol=1e-5)
    np.testing.assert_allclose(sample_nearest(arr, pts), [0.0, 3.0, 5.0])


def test_voxel_indices_clip_in_bounds():
    idx = voxel_indices(np.array([[-3.0, 99.0, 1.4]]), (4, 5, 6))
    np.testing.assert_array_equal(idx[0], [1, 4, 0])   # z, y, x


def test_sitk_index_roundtrip(standard_phantom):
    bio.assert_coordinate_roundtrip(standard_phantom.image)


def test_roi_to_physical_matches_sitk(standard_phantom, cfg):
    """ROI continuous index -> physical must agree with going the long way
    round through the original image, on a grid with non-identity direction
    cosines."""
    image, mask, flags = bio.load_case_from_images(standard_phantom.image, standard_phantom.mask)
    _, mask_roi, grid = build_roi(image, mask, cfg, flags)

    rng = np.random.default_rng(0)
    nz, ny, nx = grid.roi_shape
    probes = rng.random((32, 3)) * np.array([nx - 1, ny - 1, nz - 1])
    ours = grid.to_physical(probes)
    for probe, mine in zip(probes, ours):
        orig_idx = grid.to_original_index(probe)
        theirs = image.TransformContinuousIndexToPhysicalPoint([float(v) for v in orig_idx])
        np.testing.assert_allclose(mine, theirs, atol=1e-6)


def test_roi_mask_centroid_maps_to_original_centroid(standard_phantom, cfg):
    image, mask, flags = bio.load_case_from_images(standard_phantom.image, standard_phantom.mask)
    _, mask_roi, grid = build_roi(image, mask, cfg, flags)

    full_zyx = np.argwhere(mask).mean(axis=0)
    full_xyz = full_zyx[::-1]
    roi_zyx = np.argwhere(mask_roi).mean(axis=0)
    roi_xyz = roi_zyx[::-1]

    expected = np.asarray(
        image.TransformContinuousIndexToPhysicalPoint([float(v) for v in full_xyz])
    )
    np.testing.assert_allclose(grid.to_physical(roi_xyz), expected, atol=1e-6)


def test_direction_rotation_is_not_identity(standard_phantom, cfg):
    """The phantom grid is deliberately rotated, so an index-space direction and
    a physical-space direction genuinely differ. If this ever passes trivially
    the phantom has stopped testing what it is for."""
    image, mask, flags = bio.load_case_from_images(standard_phantom.image, standard_phantom.mask)
    _, _, grid = build_roi(image, mask, cfg, flags)
    index_dir = np.array([1.0, 0.0, 0.0])
    phys = grid.to_physical_direction(index_dir)
    phys /= np.linalg.norm(phys)
    assert not np.allclose(phys, index_dir, atol=1e-3)


def test_gzip_sniffing(tmp_path, standard_phantom):
    """A gzip stream named .nii must load: the supplied case files are exactly
    this, and nibabel raises ImageFileError on them."""
    plain = tmp_path / "volume_plain.nii"
    sitk.WriteImage(standard_phantom.mask, str(plain))
    disguised = tmp_path / "gzip_named_nii.nii"
    with open(plain, "rb") as src, gzip.open(disguised, "wb") as dst:
        shutil.copyfileobj(src, dst)

    loaded = bio.read_image(disguised)
    np.testing.assert_array_equal(
        sitk.GetArrayFromImage(loaded), sitk.GetArrayFromImage(standard_phantom.mask)
    )

    # And the reverse mismatch: uncompressed bytes named .nii.gz.
    mislabelled = tmp_path / "plain_named_gz.nii.gz"
    shutil.copyfile(plain, mislabelled)
    loaded2 = bio.read_image(mislabelled)
    np.testing.assert_array_equal(
        sitk.GetArrayFromImage(loaded2), sitk.GetArrayFromImage(standard_phantom.mask)
    )


def test_mask_grid_mismatch_is_resampled(standard_phantom):
    """A mask on a shifted grid gets resampled onto the image grid, and the
    image is left alone because it is the coordinate authority."""
    flags = Flags()
    shifted = sitk.Image(standard_phantom.mask)
    origin = np.asarray(shifted.GetOrigin()) + 3.0
    shifted.SetOrigin([float(v) for v in origin])
    out = bio.harmonise_grids(standard_phantom.image, shifted, flags)
    assert "mask_grid_resampled" in flags
    np.testing.assert_allclose(out.GetOrigin(), standard_phantom.image.GetOrigin())


def test_binarise_mask_variants():
    flags = Flags()
    arr = np.array([[[0, 255]]], dtype=np.uint8)
    np.testing.assert_array_equal(bio.binarise_mask(arr, flags), [[[False, True]]])
    assert "mask_0_255" in flags

    flags2 = Flags()
    bio.binarise_mask(np.array([[[0, 1, 2, 3]]], dtype=np.uint8), flags2)
    assert "mask_not_binary" in flags2

    flags3 = Flags()
    bio.binarise_mask(np.zeros((2, 2, 2), dtype=np.uint8), flags3)
    assert "empty_mask" in flags3
