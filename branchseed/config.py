"""Validated configuration for the CPU prediction pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
import json
import math
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    roi_halo_mm: float = 35.0
    objectness_scales_mm: tuple[float, ...] = (0.8, 1.2, 1.8, 2.5)
    primary_distance_mm: float = 8.0
    fallback_near_mm: float = 12.0
    fallback_far_mm: float = 30.0
    min_component_mm3: float = 3.0
    max_candidates: int = 24
    verification: str = "segmentation"
    verification_fallback: bool = True
    min_candidate_score: float = 0.30
    merge_distance_mm: float = 5.0
    seed_distance_mm: float = 5.0
    confidence_perturbations: int = 0

    def __post_init__(self) -> None:
        finite_positive = {
            "roi_halo_mm": self.roi_halo_mm,
            "primary_distance_mm": self.primary_distance_mm,
            "fallback_near_mm": self.fallback_near_mm,
            "fallback_far_mm": self.fallback_far_mm,
            "min_component_mm3": self.min_component_mm3,
            "merge_distance_mm": self.merge_distance_mm,
            "seed_distance_mm": self.seed_distance_mm,
        }
        for name, value in finite_positive.items():
            if isinstance(value, bool) or not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not self.objectness_scales_mm or any(
            isinstance(v, bool) or not math.isfinite(float(v)) or float(v) <= 0
            for v in self.objectness_scales_mm
        ):
            raise ValueError("objectness_scales_mm must contain finite positive values")
        if not 0 < self.fallback_near_mm <= self.fallback_far_mm:
            raise ValueError("fallback distances must be positive and ordered")
        if isinstance(self.max_candidates, bool) or not isinstance(self.max_candidates, int):
            raise ValueError("max_candidates must be an integer")
        if self.max_candidates < 1:
            raise ValueError("max_candidates must be at least one")
        if self.verification not in {"segmentation", "minimal_path"}:
            raise ValueError("verification must be 'segmentation' or 'minimal_path'")
        if (
            isinstance(self.min_candidate_score, bool)
            or not math.isfinite(float(self.min_candidate_score))
            or not 0 <= self.min_candidate_score <= 1
        ):
            raise ValueError("min_candidate_score must be finite and lie in [0, 1]")
        if not isinstance(self.verification_fallback, bool):
            raise ValueError("verification_fallback must be boolean")
        if (
            isinstance(self.confidence_perturbations, bool)
            or not isinstance(self.confidence_perturbations, int)
            or not 0 <= self.confidence_perturbations <= 20
        ):
            raise ValueError("confidence_perturbations must be an integer in [0, 20]")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "PipelineConfig":
        known = {item.name for item in fields(cls)}
        unexpected = set(values) - known
        if unexpected:
            raise ValueError(f"unknown configuration fields: {sorted(unexpected)}")
        data = dict(values)
        if "objectness_scales_mm" in data:
            data["objectness_scales_mm"] = tuple(float(v) for v in data["objectness_scales_mm"])
        return cls(**data)

    @classmethod
    def load(cls, path: str | Path | None) -> "PipelineConfig":
        if path is None:
            return cls()
        try:
            value = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read pipeline config {path}: {exc}") from exc
        if not isinstance(value, Mapping):
            raise ValueError("pipeline config must be a JSON object")
        return cls.from_mapping(value)

    def updated(self, **changes: Any) -> "PipelineConfig":
        return replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
