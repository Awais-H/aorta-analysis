import numpy as np
import SimpleITK as sitk

import candidates
import config
import io_utils
from conftest import LUMEN_HU, TISSUE_HU, make_phantom


def test_contract_shapes_and_types(cand):
    assert cand.ct.dtype == np.float32
    assert cand.mask.dtype == bool and cand.candidates.dtype == bool
    assert cand.ct.shape == cand.mask.shape == cand.candidates.shape == cand.distance_mm.shape
    assert np.allclose(cand.spacing, config.ISO_SPACING_MM)
    assert tuple(cand.image.GetSize())[::-1] == cand.ct.shape
    assert cand.fragment_log["components"] == 1
    assert TISSUE_HU < cand.threshold_hu < LUMEN_HU


def test_threshold_formula():
    ct = np.full((4, 4, 4), 300.0, np.float32)
    ct[0, 0, 0] = 200.0
    mask = np.ones_like(ct, bool)
    thr, stats = candidates.adaptive_threshold(ct, mask)
    hu = ct[mask]
    expected = max(hu.mean() - config.THRESHOLD_STD_MULTIPLIER * hu.std(),
                   config.THRESHOLD_MEDIAN_FRACTION * np.median(hu), config.THRESHOLD_FLOOR_HU)
    assert abs(thr - expected) < 1e-4
    # unenhanced lumen hits the absolute floor
    thr2, _ = candidates.adaptive_threshold(np.full((2, 2, 2), 60.0, np.float32), np.ones((2, 2, 2), bool))
    assert thr2 == config.THRESHOLD_FLOOR_HU


def test_candidates_are_in_shell_and_outside_mask(cand):
    assert not (cand.candidates & cand.mask).any()
    assert cand.distance_mm[cand.candidates].max() <= config.SHELL_MM
    assert cand.distance_mm[cand.mask].max() == 0


def test_branch_is_a_candidate_and_tissue_is_not(cand, phantom):
    idx = np.round(io_utils.mm_to_index(cand.image, phantom["seed_mm"])).astype(int)
    assert cand.candidates[tuple(idx)]
    off = np.round(io_utils.mm_to_index(cand.image, phantom["seed_mm"] + np.array([0.0, 8.0, 0.0]))).astype(int)
    assert not cand.candidates[tuple(off)]


def test_mask_fragment_removed_and_logged(tmp_path):
    ph = make_phantom(tmp_path / "frag", mask_fragment=True)
    image, mask_image, _ = io_utils.load_case(ph["image"], ph["mask"])
    cand = candidates.build(image, mask_image)
    assert cand.fragment_log["components"] == 2
    assert cand.fragment_log["fragment_sizes"] == [1]
    assert cand.fragment_log["warning"] is False
    assert not cand.mask[0, 0, 0]


def test_clean_mask_warns_on_big_blob():
    m = np.zeros((20, 20, 20), bool)
    m[2:12, 2:12, 2:12] = True
    m[15:19, 15:19, 15:19] = True  # 64 voxels vs 1000: 6% discarded
    clean, log = candidates.clean_mask(m)
    assert clean.sum() == 1000 and log["warning"] is True


def test_opening_removes_one_voxel_ring():
    """The ring outside an underfilled mask is one voxel thick; the opening must erase it."""
    sp = np.array([0.8, 0.8, 0.8])
    st = candidates.opening_structure(sp)
    assert st.shape == (3, 3, 3)
    ring = np.zeros((20, 20, 20), bool)
    ring[5:15, 5:15, 10] = True  # a one-voxel-thick sheet
    from scipy import ndimage
    assert not ndimage.binary_opening(ring, structure=st).any()
    slab = np.zeros((20, 20, 20), bool)
    slab[5:15, 5:15, 8:13] = True  # a 5-voxel-thick block survives
    assert ndimage.binary_opening(slab, structure=st).sum() > 0


def test_anisotropic_input_is_resampled(phantom_aniso):
    image, mask_image, _ = io_utils.load_case(phantom_aniso["image"], phantom_aniso["mask"])
    cand = candidates.build(image, mask_image)
    assert cand.grid_info["resampled"] is True
    assert np.allclose(cand.spacing, config.ISO_SPACING_MM)
    idx = np.round(io_utils.mm_to_index(cand.image, phantom_aniso["seed_mm"])).astype(int)
    assert cand.candidates[tuple(idx)]
