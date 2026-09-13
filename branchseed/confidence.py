"""Deterministic confidence summaries and optional perturbation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence, TypeVar

import numpy as np
from scipy.optimize import linear_sum_assignment

from .models import DaughterPrediction

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class PerturbationConfidence:
    repetitions: int
    detection_fraction: float
    median_ostium_spread_mm: float


def perturb_intensity(
    image_zyx: np.ndarray, rng: np.random.Generator, *, noise_sigma: float = 2.0
) -> np.ndarray:
    """Return a reproducible finite Gaussian intensity perturbation."""
    image = np.asarray(image_zyx, np.float32)
    scale = max(float(np.std(image[np.isfinite(image)])), 1.0)
    return image + rng.normal(0.0, noise_sigma * scale / 100.0, image.shape).astype(np.float32)


def estimate_perturbation_confidence(
    baseline: Sequence[DaughterPrediction],
    run: Callable[[np.random.Generator], Sequence[DaughterPrediction]],
    repetitions: int,
    *,
    seed: int = 0,
    match_distance_mm: float = 5.0,
) -> PerturbationConfidence:
    """Measure repeat detection and ostium spread without claiming calibration."""
    if repetitions <= 0:
        return PerturbationConfidence(0, 1.0, 0.0)
    rng = np.random.default_rng(seed)
    matched_total = 0
    possible_total = 0
    spreads: list[float] = []
    base = np.asarray([item.ostium_xyz for item in baseline], float)
    for _ in range(repetitions):
        predicted = list(run(rng))
        points = np.asarray([item.ostium_xyz for item in predicted], float)
        if len(base) == 0:
            possible_total += max(1, len(points))
            matched_total += int(len(points) == 0)
            continue
        possible_total += max(len(base), len(points))
        if not len(points):
            continue
        distances = np.linalg.norm(base[:, None, :] - points[None, :, :], axis=2)
        rows, columns = linear_sum_assignment(distances)
        accepted = distances[rows, columns] <= match_distance_mm
        matched_total += int(np.count_nonzero(accepted))
        spreads.extend(float(v) for v in distances[rows[accepted], columns[accepted]])
    return PerturbationConfidence(
        repetitions,
        matched_total / max(possible_total, 1),
        float(np.median(spreads)) if spreads else 0.0,
    )
