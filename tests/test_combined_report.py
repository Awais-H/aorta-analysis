"""Fake-data checks for the combined dashboard payload (challenge JSON per daughter)."""
from combined_report import challenge_case, challenge_daughter


def test_challenge_daughter_keeps_only_required_fields():
    raw = {
        "instance_id": "branch_001",
        "parent_instance_id": "aorta",
        "ostium_xyz_mm": [1.0, 2.0, 3.0],
        "seed_xyz_mm": [1.5, 2.5, 3.5],
        "radius_mm": 2.25,
        "direction_xyz": [0.0, 1.0, 0.0],
        "clock": "12:00",
        "flags": ["near_cut_face"],
    }
    out = challenge_daughter(raw)
    assert set(out) == {
        "instance_id",
        "parent_instance_id",
        "ostium_xyz_mm",
        "seed_xyz_mm",
        "radius_mm",
        "direction_xyz",
    }
    assert out["instance_id"] == "branch_001"
    assert out["radius_mm"] == 2.25


def test_challenge_case_wraps_parent_and_daughters():
    pred = {
        "case_id": "subject000",
        "parent": {"instance_id": "aorta"},
        "daughters": [{
            "instance_id": "branch_001",
            "parent_instance_id": "aorta",
            "ostium_xyz_mm": [0.0, 0.0, 0.0],
            "seed_xyz_mm": [0.0, 0.0, 5.0],
            "radius_mm": 1.5,
            "direction_xyz": [0.0, 0.0, 1.0],
        }],
        "meta": {"threshold_hu": 200},
    }
    out = challenge_case(pred)
    assert out["case_id"] == "subject000"
    assert out["parent"] == {"instance_id": "aorta"}
    assert len(out["daughters"]) == 1
    assert "meta" not in out
    assert out["daughters"][0]["seed_xyz_mm"] == [0.0, 0.0, 5.0]
