"""Front-half tests on two real dev cases. Skipped when the data directory is absent.

subject024: non-orthonormal header (nibabel fallback), unenhanced (threshold at the 100 HU floor).
subject019: gzip stream under a .nii name (mask19.nii), 1.5 mm isotropic, one reference case.
"""
import glob
import json
import os

import numpy as np
import pytest
import SimpleITK as sitk

import candidates
import config
import io_utils
from conftest import ROOT

DATA = os.path.join(ROOT, "data")
REFS = os.path.join(ROOT, "docs", "references")


def _case_files(n):
    d = os.path.join(DATA, f"subject{n:03d}")
    imgs = glob.glob(os.path.join(d, f"orig{n}.nii*"))
    masks = glob.glob(os.path.join(d, f"mask{n}.nii*"))
    if not imgs or not masks:
        pytest.skip(f"subject{n:03d} data not present")
    return imgs[0], masks[0]


@pytest.fixture(scope="module")
def s24():
    img, msk = _case_files(24)
    image, mask_image, info = io_utils.load_case(img, msk)
    return image, mask_image, info, img


@pytest.fixture(scope="module")
def s19():
    img, msk = _case_files(19)
    image, mask_image, info = io_utils.load_case(img, msk)
    return image, mask_image, info, msk


def test_subject024_header_fallback_fires(s24):
    image, mask_image, info, _ = s24
    for k in ("image", "mask"):
        assert info[k]["header_fallback"] == "nibabel_orthonormalized"
        assert 3.0 < info[k]["header_rotation_deg"] < 5.0  # SPEC D2: tilted about 3.5 deg from standard
    assert info["grid_match"]
    R = np.array(image.GetDirection()).reshape(3, 3)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9), "direction is orthonormal after the fallback"
    assert not np.allclose(np.abs(R), np.eye(3), atol=1e-3), "and the real tilt is kept"


def test_subject024_physical_coordinates_match_the_original_affine(s24):
    nib = pytest.importorskip("nibabel")
    image, _, _, path = s24
    aff = nib.load(path).affine
    lps = np.diag([-1.0, -1.0, 1.0, 1.0]) @ aff
    nx, ny, nz = image.GetSize()
    for i, j, k in ((0, 0, 0), (nx - 1, ny - 1, nz - 1), (nx // 2, ny // 3, nz // 4)):
        expected = (lps @ np.array([i, j, k, 1.0]))[:3]
        got = io_utils.index_to_mm(image, [k, j, i])
        assert np.linalg.norm(got - expected) < 0.5, (i, j, k, got, expected)


def test_subject024_front_half(s24):
    image, mask_image, _, _ = s24
    cand = candidates.build(image, mask_image)
    assert np.allclose(cand.spacing, config.ISO_SPACING_MM) and cand.grid_info["resampled"]
    assert cand.threshold_hu == config.THRESHOLD_FLOOR_HU  # unenhanced: lumen median 78 HU
    assert cand.fragment_log["components"] == 1
    assert cand.bright_shell.any() and (cand.opened <= cand.bright_shell).all()


def test_subject019_mask_is_gzip_under_nii(s19):
    image, mask_image, info, mask_path = s19
    with open(mask_path, "rb") as f:
        assert f.read(2) == io_utils.GZIP_MAGIC
    assert info["mask"]["gz_under_nii"] is True
    assert info["image"]["gz_under_nii"] is False  # orig19.nii.gz is named honestly
    assert info["grid_match"]
    assert np.allclose(image.GetSpacing(), 1.5)
    assert image.GetSize() == (250, 250, 169)


def test_subject019_front_half_matches_atlas_and_spec(s19):
    image, mask_image, _, _ = s19
    cand = candidates.build(image, mask_image)
    assert np.allclose(cand.spacing, config.ISO_SPACING_MM) and cand.grid_info["resampled"]
    assert abs(cand.threshold_hu - 249.4) < 2.0  # atlas_all.csv threshold for subject 19
    assert cand.fragment_log["components"] == 1
    import instances
    inst = instances.build(cand)
    assert inst.n >= 3
    # SPEC section 1: each coarse case has one 440 to 460 mm2 patch, the aorta continuing past the cut
    assert 400 < inst.wall_area_mm2[1] < 520


def test_subject019_reference_ostia_lie_on_opened_wall_patches(s19):
    """SPEC section 1: every reference ostium is within 1.6 mm of an opened wall patch."""
    ref = os.path.join(REFS, "case_19", "annotations.json")
    if not os.path.exists(ref):
        pytest.skip("reference annotations not present")
    image, mask_image, _, _ = s19
    cand = candidates.build(image, mask_image)
    wall = np.argwhere(cand.opened & (cand.distance_mm <= config.WALL_LAYER_MM)) * cand.spacing
    for d in json.load(open(ref))["daughters"]:
        idx = io_utils.mm_to_index(cand.image, d["ostium_xyz_mm"]) * cand.spacing
        assert np.linalg.norm(wall - idx, axis=1).min() <= 2.0, d["instance_id"]
