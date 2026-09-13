import json
import subprocess
import sys

import numpy as np
import pytest

from branchseed.aorta import detect_end_caps
from branchseed.ablation import run_synthetic_ablation
from branchseed.confidence import estimate_perturbation_confidence
from branchseed.config import PipelineConfig
from branchseed.models import CasePrediction, DaughterPrediction
from branchseed.output import challenge_dict
from branchseed.phantoms import make_phantom, write_phantom
from branchseed.pipeline import run_pipeline_images


def test_challenge_output_is_exact_and_empty_is_valid():
    empty = challenge_dict(CasePrediction("empty"))
    assert empty == {
        "case_id": "empty", "parent": {"instance_id": "aorta"}, "daughters": []
    }
    daughter = DaughterPrediction(
        (1, 2, 3), (2, 2, 3), (1, 0, 0), 1.5,
        metadata={"must_not_leak": True},
    )
    payload = challenge_dict(CasePrediction("case", (daughter,)))
    assert set(payload) == {"case_id", "parent", "daughters"}
    assert payload["parent"] == {"instance_id": "aorta"}
    assert set(payload["daughters"][0]) == {
        "instance_id", "parent_instance_id", "ostium_xyz_mm",
        "seed_xyz_mm", "radius_mm", "direction_xyz",
    }
    assert payload["daughters"][0]["instance_id"] == "branch_001"
    assert CasePrediction.from_dict(payload).daughters[0].ostium_xyz == (1.0, 2.0, 3.0)


def test_challenge_writer_normalizes_and_rejects_zero_direction():
    prediction = CasePrediction(
        "case", (DaughterPrediction((1, 2, 3), (2, 2, 3), (3, 0, 0), 1.5),)
    )
    assert challenge_dict(prediction)["daughters"][0]["direction_xyz"] == [1.0, 0.0, 0.0]
    invalid = CasePrediction(
        "case", (DaughterPrediction((1, 2, 3), (2, 2, 3), (0, 0, 0), 1.5),)
    )
    with pytest.raises(ValueError, match="nonzero"):
        challenge_dict(invalid)


def test_flat_internal_mask_ends_are_caps_without_axis_rule():
    z, y, x = np.indices((32, 38, 42))
    mask = ((z - 16) ** 2 + (y - 19) ** 2 <= 5**2) & (x >= 7) & (x <= 34)
    rotation = (0, 0, 1, 0, 1, 0, -1, 0, 0)
    caps = detect_end_caps(mask, (1, 1.4, 2), direction=rotation)
    assert caps[:, :, 7].sum() > 20
    assert caps[:, :, 34].sum() > 20
    assert caps[:, 19, 20].sum() == 0


def test_no_daughter_pipeline_returns_empty_prediction():
    phantom = make_phantom("no_daughter")
    result = run_pipeline_images(
        phantom.image,
        phantom.aorta_mask,
        case_id="none",
        config=PipelineConfig(objectness_scales_mm=(1.0,), max_candidates=8),
    )
    assert result.prediction.case_id == "none"
    assert result.candidate_count <= 8
    assert result.prediction.daughters == ()
    assert result.profiles["total"].seconds > 0
    assert result.profiles["total"].rss_mb > 0


def test_config_values_reach_pipeline_and_halo_is_safely_raised(monkeypatch):
    captured = {}

    def no_candidates(*args, **kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr("branchseed.pipeline.generate_candidates", no_candidates)
    phantom = make_phantom("no_daughter")
    config = PipelineConfig(
        objectness_scales_mm=(2.0,),
        roi_halo_mm=5.0,
        fallback_far_mm=31.0,
        merge_distance_mm=2.25,
    )
    result = run_pipeline_images(
        phantom.image, phantom.aorta_mask, case_id="config", config=config
    )
    assert captured["merge_distance_mm"] == 2.25
    assert result.prediction.metadata["effective_roi_halo_mm"] == 39.0


def test_seed_distance_and_confidence_are_integrated(monkeypatch):
    import branchseed.pipeline as pipeline_module

    seen = []
    original = pipeline_module.instance_to_prediction

    def recording_measurement(*args, **kwargs):
        seen.append(kwargs["seed_distance_mm"])
        return original(*args, **kwargs)

    monkeypatch.setattr(pipeline_module, "instance_to_prediction", recording_measurement)
    phantom = make_phantom("straight_multiple")
    result = run_pipeline_images(
        phantom.image,
        phantom.aorta_mask,
        case_id="confidence",
        config=PipelineConfig(
            objectness_scales_mm=(1.0,),
            seed_distance_mm=6.0,
            confidence_perturbations=1,
        ),
    )
    assert seen and set(seen) == {6.0}
    assert result.prediction.metadata["confidence"]["repetitions"] == 1
    assert "confidence_perturbations" in result.profiles
    assert result.debug["confidence"] == result.prediction.metadata["confidence"]


def test_confidence_uses_one_to_one_matching_with_misses():
    baseline = (
        DaughterPrediction((0, 0, 0)),
        DaughterPrediction((10, 0, 0)),
    )

    def duplicate_first(_rng):
        return (
            DaughterPrediction((0.1, 0, 0)),
            DaughterPrediction((0.2, 0, 0)),
        )

    confidence = estimate_perturbation_confidence(
        baseline, duplicate_first, repetitions=1, match_distance_mm=1.0
    )
    assert confidence.detection_fraction == 0.5


def test_common_trunk_has_one_direct_daughter_reference():
    assert len(make_phantom("common_trunk").reference.daughters) == 1


def test_path_ablation_variants_disable_fallback():
    result = run_synthetic_ablation(("no_daughter",))
    path_rows = [
        row for row in result["results"] if row["variant_kind"] == "path"
    ]
    assert len(path_rows) == 2
    assert all(row["verification_fallback"] is False for row in path_rows)


def test_pipeline_import_does_not_eagerly_import_matplotlib():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, branchseed.pipeline; "
            "print(any(name.startswith('matplotlib') for name in sys.modules))",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert completed.stdout.strip() == "False"


@pytest.mark.parametrize(
    "changes",
    [
        {"roi_halo_mm": float("nan")},
        {"primary_distance_mm": 0},
        {"fallback_far_mm": float("inf")},
        {"min_component_mm3": -1},
        {"merge_distance_mm": 0},
        {"seed_distance_mm": -1},
        {"objectness_scales_mm": (1.0, float("nan"))},
        {"min_candidate_score": float("nan")},
        {"max_candidates": 1.5},
        {"verification_fallback": 1},
        {"confidence_perturbations": 21},
    ],
)
def test_config_rejects_invalid_numeric_ranges(changes):
    with pytest.raises(ValueError):
        PipelineConfig(**changes)


def test_root_cli_writes_machine_readable_challenge_json(tmp_path):
    phantom = make_phantom("no_daughter")
    image, mask, _ = write_phantom(phantom, tmp_path)
    output = tmp_path / "prediction.json"
    completed = subprocess.run(
        [
            sys.executable, "run.py", "--image", str(image),
            "--aorta-mask", str(mask), "--output", str(output),
        ],
        check=False, capture_output=True, text=True,
    )
    assert completed.returncode == 0, completed.stderr
    summary = json.loads(completed.stdout)
    assert "total" in summary["profiles"]
    payload = json.loads(output.read_text())
    assert payload == {
        "case_id": "no_daughter",
        "parent": {"instance_id": "aorta"},
        "daughters": [],
    }
    assert CasePrediction.load_json(output).daughters == ()
