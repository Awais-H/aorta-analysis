import numpy as np

import config
import scorer
from conftest import fake_daughter, fake_result


def _ref():
    return fake_result("s", [fake_daughter(1, [0, 0, 0]), fake_daughter(2, [0, 0, 30]), fake_daughter(3, [0, 30, 0])])


def test_perfect_prediction():
    s = scorer.score_case(_ref(), _ref())
    assert s["n_pred"] == s["n_ref"] == 3
    for k in [k for k in s if k.endswith("mm")]:
        assert s[k]["tp"] == 3 and s[k]["fp"] == 0 and s[k]["fn"] == 0
        assert s[k]["f1"] == 1.0 and s[k]["mean_ostium_mm"] == 0.0
        assert s[k]["mean_direction_deg"] == 0.0 and s[k]["mean_radius_abs_mm"] == 0.0


def test_cutoff_and_one_to_one_matching():
    pred = fake_result("s", [fake_daughter(1, [4, 0, 0]), fake_daughter(2, [2, 0, 0]), fake_daughter(3, [0, 0, 30])])
    s = scorer.score_case(pred, _ref(), cutoffs=(3.0, 5.0))
    # at 3 mm: only branch_002 (2 mm) can match ref 1, branch_003 matches ref 2 -> tp 2, fp 1, fn 1
    assert (s["3mm"]["tp"], s["3mm"]["fp"], s["3mm"]["fn"]) == (2, 1, 1)
    # at 5 mm: Hungarian pairs one of the two near candidates with ref 1, the other stays a fp
    assert (s["5mm"]["tp"], s["5mm"]["fp"], s["5mm"]["fn"]) == (2, 1, 1)
    assert s["5mm"]["precision"] == 2 / 3 and s["5mm"]["recall"] == 2 / 3


def test_direction_and_radius_errors():
    pred = fake_result("s", [fake_daughter(1, [0, 0, 0], direction=(0, 1, 0), radius=4.0)])
    ref = fake_result("s", [fake_daughter(1, [0, 0, 0], direction=(1, 0, 0), radius=2.5)])
    s = scorer.score_case(pred, ref, cutoffs=(5.0,))["5mm"]
    assert abs(s["mean_direction_deg"] - 90.0) < 1e-6
    assert abs(s["mean_radius_abs_mm"] - 1.5) < 1e-9


def test_empty_cases():
    s = scorer.score_case(fake_result("s"), fake_result("s"), cutoffs=(5.0,))["5mm"]
    assert s["precision"] == 1.0 and s["recall"] == 1.0
    s = scorer.score_case(fake_result("s"), _ref(), cutoffs=(5.0,))["5mm"]
    assert s["recall"] == 0.0 and s["fn"] == 3


def test_aggregate_is_micro_averaged():
    a = scorer.aggregate({"c1": scorer.score_case(_ref(), _ref(), (5.0,)),
                          "c2": scorer.score_case(fake_result("s"), _ref(), (5.0,))})["5mm"]
    assert a["cases"] == 2 and a["tp"] == 3 and a["fn"] == 3 and a["recall"] == 0.5


def test_invariants_pass_on_valid_result():
    assert scorer.check_invariants(_ref()) == []


def test_invariants_catch_violations():
    bad = _ref()
    bad["daughters"][0]["direction_xyz"] = [0.5, 0.0, 0.0]
    bad["daughters"][1]["radius_mm"] = config.INVARIANT_RADIUS_MAX_MM + 1
    bad["daughters"][2]["ostium_xyz_mm"] = [0, 0, 1.0]  # 1 mm from branch_001
    bad["daughters"][2]["instance_id"] = "branch_007"
    v = scorer.check_invariants(bad)
    joined = "\n".join(v)
    assert "direction norm" in joined
    assert "radius" in joined
    assert "closer than" in joined
    assert "expected branch_003" in joined


def test_invariants_with_image(cand, phantom):
    good = fake_result("s", [fake_daughter(1, phantom["ostium_mm"], phantom["direction"])])
    assert scorer.check_invariants(good, cand) == []
    away = fake_result("s", [fake_daughter(1, phantom["ostium_mm"], -phantom["direction"])])
    v = scorer.check_invariants(away, cand)
    assert any("not bright" in x or "away" in x for x in v)


def test_seed_labels_and_seed_on_branch(phantom, tmp_path):
    """A label volume on the phantom grid: the branch is label 1, so a seed on the branch scores."""
    import SimpleITK as sitk
    img = phantom["sitk_image"]
    ct = sitk.GetArrayFromImage(img)
    mask = sitk.GetArrayFromImage(phantom["sitk_mask"]) > 0
    lab = ((ct > 200) & ~mask).astype(np.uint8)  # branch voxels only
    lab_img = sitk.GetImageFromArray(lab)
    lab_img.CopyInformation(img)
    path = str(tmp_path / "daughters_draft.nii.gz")
    sitk.WriteImage(lab_img, path)
    on = fake_daughter(1, phantom["ostium_mm"], phantom["direction"])
    off = fake_daughter(2, phantom["ostium_mm"], -phantom["direction"])  # seed inside the aorta
    assert scorer.seed_labels([on, off], path) == [1, 0]
    ref = {"daughters": [dict(fake_daughter(1, phantom["ostium_mm"], phantom["direction"], radius=2.0), label_value=1)]}
    s = scorer.score_case({"daughters": [on]}, ref, cutoffs=(5.0,), label_path=path)["5mm"]
    assert s["seed_on_branch"] == (1, 1) and s["n_radius"] == 1
    s = scorer.score_case({"daughters": [off]}, ref, cutoffs=(5.0,), label_path=path)["5mm"]
    assert s["seed_on_branch"] == (0, 1)


def test_radius_scored_only_where_reference_has_one():
    ref = {"daughters": [dict(fake_daughter(1, [0, 0, 0]), radius_mm=None), fake_daughter(2, [0, 0, 30], radius=3.0)]}
    pred = fake_result("s", [fake_daughter(1, [0, 0, 0], radius=2.0), fake_daughter(2, [0, 0, 30], radius=2.0)])
    s = scorer.score_case(pred, ref, cutoffs=(5.0,))["5mm"]
    assert s["tp"] == 2 and s["n_radius"] == 1 and abs(s["mean_radius_abs_mm"] - 1.0) < 1e-9
    assert s["seed_on_branch"] == (0, 0)  # no label volume given


def test_load_reference_schema():
    import os
    ref = scorer.load_reference(19)
    if ref is None:
        import pytest
        pytest.skip("docs/references not present")
    assert ref["parent"]["instance_id"] == "aorta"
    for d in ref["daughters"]:
        assert {"instance_id", "ostium_xyz_mm", "seed_xyz_mm", "direction_xyz", "radius_mm", "label_value"} <= set(d)
        assert np.allclose(d["direction_xyz"], scorer.np.subtract(d["seed_xyz_mm"], d["ostium_xyz_mm"]) / np.linalg.norm(np.subtract(d["seed_xyz_mm"], d["ostium_xyz_mm"])), atol=1e-3)
    assert scorer.case_files(19)["labels"] is not None or not os.path.isdir(scorer.DATA_DIR)


def test_format_scores_runs():
    cs = {"case_x": scorer.score_case(_ref(), _ref())}
    text = scorer.format_scores(cs, scorer.aggregate(cs), ["hdr"])
    assert "AGGREGATE" in text and "case_x" in text and "3/3" not in text  # no label volume: 0/0
