"""Strict challenge-output serialization, isolated from internal metadata."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .models import CasePrediction


def _finite_vector(value: object, name: str) -> list[float]:
    if value is None:
        raise ValueError(f"{name} is required for challenge output")
    vector = [float(v) for v in value]  # type: ignore[union-attr]
    if len(vector) != 3 or not all(math.isfinite(v) for v in vector):
        raise ValueError(f"{name} must contain three finite values")
    return vector


def _unit_direction(value: object) -> list[float]:
    vector = _finite_vector(value, "direction_xyz")
    magnitude = math.sqrt(sum(component * component for component in vector))
    if not math.isfinite(magnitude) or magnitude <= 1e-12:
        raise ValueError("direction_xyz must be a nonzero finite vector")
    return [component / magnitude for component in vector]


def challenge_dict(prediction: CasePrediction) -> dict[str, Any]:
    """Return exactly the challenge schema, dropping all internal fields."""
    daughters: list[dict[str, Any]] = []
    for index, daughter in enumerate(prediction.daughters, 1):
        radius = daughter.radius_mm
        if radius is None or not math.isfinite(radius) or radius < 0:
            raise ValueError("radius_mm is required and must be finite and non-negative")
        daughters.append(
            {
                "instance_id": f"branch_{index:03d}",
                "parent_instance_id": "aorta",
                "ostium_xyz_mm": _finite_vector(daughter.ostium_xyz, "ostium_xyz_mm"),
                "seed_xyz_mm": _finite_vector(daughter.seed_xyz, "seed_xyz_mm"),
                "radius_mm": float(radius),
                "direction_xyz": _unit_direction(daughter.direction_xyz),
            }
        )
    return {
        "case_id": prediction.case_id,
        "parent": {"instance_id": "aorta"},
        "daughters": daughters,
    }


def write_challenge_json(prediction: CasePrediction, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(challenge_dict(prediction), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return destination
