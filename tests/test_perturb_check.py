"""perturb_check.py on the phantom: the perturbations are what they say, the threshold shift is
undone afterwards, and the phantom's branch survives every perturbation."""
import numpy as np
import SimpleITK as sitk

import candidates
import coarse_check
import io_utils
import perturb_check
import scorer
from conftest import make_phantom


def test_mask_morphology_changes_volume_by_one_voxel_layer(phantom):
    _, mask, _ = io_utils.load_case(phantom["image"], phantom["mask"])
    base = sitk.GetArrayViewFromImage(mask).sum()
    er = sitk.GetArrayViewFromImage(perturb_check._morph_mask(mask, "erode")).sum()
    di = sitk.GetArrayViewFromImage(perturb_check._morph_mask(mask, "dilate")).sum()
    assert er < base < di
    assert perturb_check._morph_mask(mask, "erode").GetSpacing() == mask.GetSpacing()


def test_threshold_shift_is_applied_and_restored(phantom):
    image, mask, _ = io_utils.load_case(phantom["image"], phantom["mask"])
    orig = candidates.adaptive_threshold
    base = coarse_check.run_stack(image, mask, "p")
    with perturb_check._ShiftedThreshold(-perturb_check.THRESHOLD_SHIFT_HU):
        low = coarse_check.run_stack(image, mask, "p")
    assert candidates.adaptive_threshold is orig
    assert abs((base["cand"].threshold_hu - low["cand"].threshold_hu) - perturb_check.THRESHOLD_SHIFT_HU) < 1e-6
    assert low["cand"].hu_stats["threshold_shift_hu"] == -perturb_check.THRESHOLD_SHIFT_HU


def test_phantom_branch_survives_every_perturbation(tmp_path):
    d = tmp_path / "data" / "subject099"
    make_phantom(str(d))
    (d / "orig.nii").rename(d / "orig99.nii")
    (d / "mask.nii").rename(d / "mask99.nii")
    lines, stats = perturb_check.check_case(99, data_dir=str(tmp_path / "data"))
    assert stats["n_kept"] == 1
    for mode in perturb_check.PERTURBATIONS:
        assert stats["modes"][mode]["tp"] == 1 and stats["modes"][mode]["fn"] == 0, mode
        assert stats["modes"][mode]["ostium"][0] < 2.0, mode
    text = perturb_check.report([99], data_dir=str(tmp_path / "data"))
    assert "cases with no flip under any perturbation: [99]" in text
