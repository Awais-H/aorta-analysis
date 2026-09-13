import json

import numpy as np

import config
import pipeline
import scorer


def _valid_schema(result):
    assert set(result) == {"case_id", "parent", "daughters"}
    assert result["parent"] == {"instance_id": "aorta"}
    for k, d in enumerate(result["daughters"], 1):
        assert set(d) == {"instance_id", "parent_instance_id", "ostium_xyz_mm", "seed_xyz_mm", "radius_mm", "direction_xyz"}
        assert d["instance_id"] == f"branch_{k:03d}" and d["parent_instance_id"] == "aorta"
        assert len(d["ostium_xyz_mm"]) == 3 and len(d["seed_xyz_mm"]) == 3 and len(d["direction_xyz"]) == 3
        assert isinstance(d["radius_mm"], float)
    json.dumps(result)


def test_phantom_with_branch(phantom, tmp_path):
    result, meta = pipeline.run(phantom["image"], phantom["mask"], "subject000", report_dir=str(tmp_path / "rep"))
    _valid_schema(result)
    assert len(result["daughters"]) == 1
    d = result["daughters"][0]
    assert np.linalg.norm(np.subtract(d["ostium_xyz_mm"], phantom["ostium_mm"])) < 3.0
    assert np.linalg.norm(np.subtract(d["seed_xyz_mm"], phantom["seed_mm"])) < 3.0
    assert np.dot(d["direction_xyz"], phantom["direction"]) > 0.8
    assert scorer.check_invariants(result) == []
    assert {"load", "candidates", "instances", "frame", "ostium", "tracing", "filters", "report", "total"} <= set(meta["timings_s"])
    assert meta["wall_patches"] == 1 and meta["rejections"] == []
    assert meta["label_to_branch"] == {"1": "branch_001"}
    assert meta["report_files"]["png"].endswith("subject000_check.png")
    s = scorer.score_case(result, {"daughters": [{"instance_id": "ref", "ostium_xyz_mm": list(phantom["ostium_mm"]),
                                                  "direction_xyz": list(phantom["direction"]), "radius_mm": phantom["radius_mm"]}]})
    assert s["5mm"]["f1"] == 1.0


def test_phantom_without_bright_voxels_outside_mask_gives_empty_daughters(phantom_no_branch):
    result, meta = pipeline.run(phantom_no_branch["image"], phantom_no_branch["mask"], "subject000")
    _valid_schema(result)
    assert result["daughters"] == []
    assert meta["wall_patches"] == 0


def test_anisotropic_phantom_end_to_end(phantom_aniso):
    result, meta = pipeline.run(phantom_aniso["image"], phantom_aniso["mask"], "subject000")
    _valid_schema(result)
    assert meta["grid"]["resampled"] is True
    assert len(result["daughters"]) == 1
    assert np.linalg.norm(np.subtract(result["daughters"][0]["ostium_xyz_mm"], phantom_aniso["ostium_mm"])) < 3.0
