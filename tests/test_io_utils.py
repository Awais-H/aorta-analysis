import glob
import os

import numpy as np
import pytest
import SimpleITK as sitk

import io_utils
import config
from conftest import ROOT, make_phantom


def test_only_io_utils_calls_transform_index_to_physical_point():
    """CLAUDE.md rule: io_utils.index_to_mm is the only caller. triage.py is the legacy tool."""
    offenders = []
    for path in glob.glob(os.path.join(ROOT, "*.py")) + glob.glob(os.path.join(ROOT, "tests", "*.py")):
        name = os.path.basename(path)
        if name in ("io_utils.py", "triage.py") or name.startswith("test_"):
            continue
        with open(path, encoding="utf-8") as f:
            if "TransformIndexToPhysicalPoint" in f.read():
                offenders.append(name)
    assert offenders == []


def test_index_to_mm_matches_affine(phantom_aniso):
    img = phantom_aniso["sitk_image"]
    sp = np.array(img.GetSpacing())
    org = np.array(img.GetOrigin())
    for idx_zyx in ([0, 0, 0], [5, 7, 11], [31, 71, 79]):
        expected = org + sp * np.array(idx_zyx)[::-1]
        assert np.allclose(io_utils.index_to_mm(img, idx_zyx), expected)
    # fractional index and round trip
    p = io_utils.index_to_mm(img, [3.5, 2.25, 1.0])
    assert np.allclose(p, org + sp * np.array([1.0, 2.25, 3.5]))
    assert np.allclose(io_utils.mm_to_index(img, p), [3.5, 2.25, 1.0])
    # batch form
    batch = io_utils.index_to_mm(img, np.array([[0, 0, 0], [1, 2, 3]]))
    assert batch.shape == (2, 3)
    assert np.allclose(batch[1], org + sp * np.array([3, 2, 1]))


def test_index_vector_to_mm_is_unit_and_oriented(phantom):
    img = phantom["sitk_image"]
    v = io_utils.index_vector_to_mm(img, [10, 10, 10], [0, 0, 1])  # +x in index space
    assert np.allclose(v, [1, 0, 0], atol=1e-9)


def test_read_gzip_under_nii(tmp_path):
    ph = make_phantom(tmp_path / "gz", gz_under_nii=True)
    with open(ph["image"], "rb") as f:
        assert f.read(2) == io_utils.GZIP_MAGIC
    with pytest.raises(RuntimeError):
        sitk.ReadImage(ph["image"])  # what the judges' file would do to a naive reader
    img, info = io_utils.read_image(ph["image"])
    assert info["gz_under_nii"] is True
    assert img.GetSize() == ph["sitk_image"].GetSize()
    assert np.allclose(img.GetSpacing(), ph["sitk_image"].GetSpacing())


def test_read_non_orthonormal_header_falls_back(tmp_path):
    """Subject 24: a header whose direction cosines are off-orthonormal by rounding."""
    nib = pytest.importorskip("nibabel")
    theta = np.radians(3.5)
    R = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1.0]])
    R[0, 1] += 0.01  # break orthonormality by rounding-scale error
    sp = np.array([0.9, 0.9, 1.5])
    aff = np.eye(4)
    aff[:3, :3] = R * sp
    aff[:3, 3] = [-100.0, -120.0, 50.0]
    arr = np.zeros((20, 22, 24), np.int16)
    arr[10, 11, 12] = 500
    path = str(tmp_path / "tilted.nii")
    nib.save(nib.Nifti1Image(arr, aff), path)

    img, info = io_utils.read_image(path)
    assert info["header_fallback"] == "nibabel_orthonormalized"
    assert 3.0 < info["header_rotation_deg"] < 4.5  # 3.5 deg tilt plus the polar-decomposed shear
    assert img.GetSize() == (20, 22, 24)
    # the bright voxel is at array index (x=10, y=11, z=12) -> zyx (12, 11, 10)
    a = sitk.GetArrayFromImage(img)
    assert a[12, 11, 10] == 500
    lps = np.diag([-1.0, -1.0, 1.0, 1.0]) @ aff
    expected = (lps @ np.array([10, 11, 12, 1.0]))[:3]
    assert np.linalg.norm(io_utils.index_to_mm(img, [12, 11, 10]) - expected) < 0.5


def test_load_case_and_crop_preserves_physical_coordinates(phantom_aniso):
    image, mask_image, info = io_utils.load_case(phantom_aniso["image"], phantom_aniso["mask"])
    assert info["grid_match"]
    mask = sitk.GetArrayViewFromImage(mask_image) > 0
    grid = io_utils.crop_and_resample(image, mask)
    assert grid.resampled
    assert np.allclose(grid.spacing, config.ISO_SPACING_MM)
    assert grid.ct.dtype == np.float32
    assert grid.ct.shape == grid.mask.shape
    assert all(grid.ct.shape[a] < mask.shape[a] * 1.5 / 0.8 + 1 for a in range(3))
    # the mask centroid must land at the same physical point on the working grid
    c_native = np.argwhere(mask).mean(axis=0)
    c_work = np.argwhere(grid.mask).mean(axis=0)
    assert np.linalg.norm(io_utils.index_to_mm(image, c_native) - io_utils.index_to_mm(grid.image, c_work)) < config.ISO_SPACING_MM
    # the padded crop reaches at least the shell radius from the mask on every side
    for a in range(3):
        assert grid.crop[a].start == 0 or grid.crop[a].start <= np.argwhere(mask)[:, a].min() - config.CROP_PAD_MM / grid.native_spacing[a] + 1


def test_crop_without_resample_when_already_iso(phantom):
    image, mask_image, _ = io_utils.load_case(phantom["image"], phantom["mask"])
    grid = io_utils.crop_and_resample(image, sitk.GetArrayViewFromImage(mask_image) > 0)
    assert not grid.resampled
    assert np.allclose(grid.spacing, 0.8)


def test_empty_mask_raises(phantom):
    image, mask_image, _ = io_utils.load_case(phantom["image"], phantom["mask"])
    with pytest.raises(ValueError):
        io_utils.crop_and_resample(image, np.zeros(sitk.GetArrayViewFromImage(mask_image).shape, bool))


def test_array_conversion_matches_simpleitk(phantom):
    """The vectorised affine path for arrays is the same map as TransformIndexToPhysicalPoint."""
    import io_utils
    img = phantom["sitk_image"]
    rng = np.random.default_rng(1)
    idx = np.column_stack([rng.integers(0, s, 50) for s in img.GetSize()[::-1]]).astype(float)
    idx[25:] += rng.uniform(-0.5, 0.5, (25, 3))
    bulk = io_utils.index_to_mm(img, idx)
    single = np.stack([io_utils.index_to_mm(img, row) for row in idx])
    assert np.allclose(bulk, single, atol=1e-9)
    back = io_utils.mm_to_index(img, bulk)
    back_single = np.stack([io_utils.mm_to_index(img, row) for row in bulk])
    assert np.allclose(back, idx, atol=1e-9) and np.allclose(back_single, idx, atol=1e-9)


def test_array_conversion_with_rotated_direction(tmp_path):
    from conftest import make_phantom
    import io_utils
    c, s = np.cos(np.radians(7.0)), np.sin(np.radians(7.0))
    ph = make_phantom(tmp_path / "rot", direction=[[c, -s, 0], [s, c, 0], [0, 0, 1]], size_xyz=(40, 40, 30))
    img = ph["sitk_image"]
    idx = np.array([[0, 0, 0], [29, 39, 39], [10.5, 20.25, 3.75]], float)
    bulk = io_utils.index_to_mm(img, idx)
    single = np.stack([io_utils.index_to_mm(img, row) for row in idx])
    assert np.allclose(bulk, single, atol=1e-9)
    assert np.allclose(io_utils.mm_to_index(img, bulk), idx, atol=1e-9)
