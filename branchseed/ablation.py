"""Executable synthetic feature/path/radius ablation benchmark."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from typing import Sequence

from .config import PipelineConfig
from .evaluation import evaluate_case
from .models import DaughterPrediction
from .phantoms import make_phantom
from .pipeline import run_pipeline_images


def _metrics(predictions: Sequence[DaughterPrediction], references: Sequence[DaughterPrediction]) -> dict[str, object]:
    result = evaluate_case(predictions, references)
    return {
        "precision": result.precision,
        "recall": result.recall,
        "f1": result.f1,
        "ostium_error_mm": result.ostium_error_mm,
        "radius_error_mm": result.radius_error_mm,
    }


def run_synthetic_ablation(
    scenarios: Sequence[str] = ("straight_multiple", "curved", "no_daughter"),
) -> dict[str, object]:
    """Run maintained CPU variants; advanced filters are intentionally excluded."""
    variants = [
        ("feature", "small_scales", PipelineConfig(objectness_scales_mm=(0.8, 1.2))),
        ("feature", "full_scales", PipelineConfig()),
        (
            "path",
            "segmentation",
            PipelineConfig(verification="segmentation", verification_fallback=False),
        ),
        (
            "path",
            "minimal_path",
            PipelineConfig(verification="minimal_path", verification_fallback=False),
        ),
    ]
    rows: list[dict[str, object]] = []
    cached: dict[str, tuple[tuple[DaughterPrediction, ...], tuple[DaughterPrediction, ...]]] = {}
    for kind, name, config in variants:
        for scenario in scenarios:
            phantom = make_phantom(scenario)
            result = run_pipeline_images(
                phantom.image, phantom.aorta_mask, case_id=scenario, config=config
            )
            rows.append({
                "variant_kind": kind,
                "variant": name,
                "scenario": scenario,
                "candidate_count": result.candidate_count,
                "verified_path_count": result.verified_path_count,
                "verification_fallback": config.verification_fallback,
                **_metrics(result.prediction.daughters, phantom.reference.daughters),
                "runtime_seconds": result.profiles["total"].seconds,
                "peak_rss_mb": result.profiles["total"].rss_mb,
            })
            if name == "full_scales":
                cached[scenario] = (result.prediction.daughters, phantom.reference.daughters)
    for radius_name, metadata_key in (
        ("3d_edt", "radius_3d_edt_mm"),
        ("2d_inscribed", "radius_2d_inscribed_mm"),
        ("2d_area_equivalent", "radius_2d_area_equivalent_mm"),
        ("consensus_median", None),
    ):
        for scenario, (predictions, references) in cached.items():
            changed = tuple(
                item if metadata_key is None else replace(
                    item, radius_mm=float(item.metadata.get(metadata_key, item.radius_mm or 0.0))
                )
                for item in predictions
            )
            rows.append({
                "variant_kind": "radius", "variant": radius_name,
                "scenario": scenario, **_metrics(changed, references),
            })
    return {
        "benchmark": "branchseed_synthetic_ablation",
        "scientific_scope": (
            "Synthetic engineering regression only; it is not clinical validation. "
            "Jerman, OOF, and RORPO remain optional and were not evaluated."
        ),
        "scenarios": list(scenarios),
        "results": rows,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--scenarios", nargs="*")
    arguments = parser.parse_args(argv)
    result = run_synthetic_ablation(tuple(arguments.scenarios) if arguments.scenarios else (
        "straight_multiple", "curved", "common_trunk", "nearby_ostia",
        "anisotropic_rotated", "low_proximal_contrast", "crop_ends", "no_daughter",
    ))
    destination = Path(arguments.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
