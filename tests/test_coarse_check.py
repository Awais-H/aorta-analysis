"""coarse_check.py on the phantom: the resample keeps the physical frame, the pipeline still finds
the branch on the coarse pair, and the coarse-versus-native matching is the scorer's."""
import numpy as np
import SimpleITK as sitk

import coarse_check
import io_utils
import scorer
from conftest import AORTA_R, LUMEN_HU, make_phantom


def test_resample_pair_keeps_the_physical_frame(phantom):
    image, mask, _ = io_utils.load_case(phantom["image"], phantom["mask"])
    ct, m = coarse_check.resample_pair(image, mask, 1.5)
    assert ct.GetSpacing() == (1.5, 1.5, 1.5) and m.GetSpacing() == (1.5, 1.5, 1.5)
    assert np.allclose(ct.GetOrigin(), image.GetOrigin()) and np.allclose(ct.GetDirection(), image.GetDirection())
    assert ct.GetSize() == m.GetSize()
    # the coarse grid covers the native extent
    for k in range(3):
        assert ct.GetSize()[k] * 1.5 >= image.GetSize()[k] * image.GetSpacing()[k] - 1e-6
    # the aorta centre reads as lumen on both, and the mask is still binary there
    cx, cy = phantom["centre_xy"]
    zc = 0.5 * sum(phantom["z_range"])
    idx = ct.TransformPhysicalPointToIndex([float(cx), float(cy), float(zc)])
    assert abs(sitk.GetArrayViewFromImage(ct)[idx[2], idx[1], idx[0]] - LUMEN_HU) < 40
    assert sitk.GetArrayViewFromImage(m)[idx[2], idx[1], idx[0]] == 1
    assert set(np.unique(sitk.GetArrayViewFromImage(m)).tolist()) <= {0, 1}
    # a point one aortic radius outside the lumen is not mask
    idx = m.TransformPhysicalPointToIndex([float(cx), float(cy - 2 * AORTA_R), float(zc)])
    assert sitk.GetArrayViewFromImage(m)[idx[2], idx[1], idx[0]] == 0


def test_branch_survives_coarsening_and_matches_native(phantom):
    image, mask, _ = io_utils.load_case(phantom["image"], phantom["mask"])
    nat = coarse_check.run_stack(image, mask, "phantom")
    ct, m = coarse_check.resample_pair(image, mask, 1.5)
    coa = coarse_check.run_stack(ct, m, "phantom")
    assert len(nat["result"]["daughters"]) == 1 and len(coa["result"]["daughters"]) == 1
    pairs = scorer.match(coa["result"]["daughters"], nat["result"]["daughters"], 5.0)
    assert len(pairs) == 1 and pairs[0][2] < 3.0
    assert np.linalg.norm(np.subtract(coa["result"]["daughters"][0]["ostium_xyz_mm"], phantom["ostium_mm"])) < 3.0


def test_wall_table_and_nearest_wall(phantom):
    image, mask, _ = io_utils.load_case(phantom["image"], phantom["mask"])
    nat = coarse_check.run_stack(image, mask, "phantom")
    table = coarse_check._wall_table(nat)
    lab, dist = coarse_check._nearest_wall(table, phantom["ostium_mm"])
    assert lab == nat["kept"][0] and dist < 2.0
    lab, dist = coarse_check._nearest_wall(table, phantom["ostium_mm"] + np.array([0.0, 0.0, 25.0]))
    assert dist > 10.0


def test_report_runs_on_a_phantom_case(tmp_path):
    d = tmp_path / "data" / "subject099"
    make_phantom(str(d))
    (d / "orig.nii").rename(d / "orig99.nii")
    (d / "mask.nii").rename(d / "mask99.nii")
    text = coarse_check.report([99], str(tmp_path / "work"), str(tmp_path / "data"))
    assert "== subject099" in text and "matched 1, lost 0, gained 0" in text
    assert (tmp_path / "work" / "data" / "subject099" / "orig99.nii.gz").exists()
    assert "AGGREGATE over 1 cases" in text
