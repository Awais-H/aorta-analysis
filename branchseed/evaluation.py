"""One-to-one matching and metrics for daughter-vessel predictions."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from .models import DaughterPrediction


@dataclass(frozen=True, slots=True)
class OstiumMatch:
    prediction_index: int
    reference_index: int
    distance_mm: float


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    matches: tuple[OstiumMatch, ...]
    unmatched_prediction_indices: tuple[int, ...]
    unmatched_reference_indices: tuple[int, ...]
    precision: float
    recall: float
    f1: float
    ostium_error_mm: float | None
    seed_distance_mm: float | None
    direction_angular_error_deg: float | None
    radius_error_mm: float | None

    @property
    def true_positives(self) -> int:
        return len(self.matches)

    @property
    def false_positives(self) -> int:
        return len(self.unmatched_prediction_indices)

    @property
    def false_negatives(self) -> int:
        return len(self.unmatched_reference_indices)


def match_ostia(
    predictions: Sequence[DaughterPrediction],
    references: Sequence[DaughterPrediction],
    *,
    max_distance_mm: float = 10.0,
) -> tuple[OstiumMatch, ...]:
    """Find a maximum-cardinality, minimum-distance one-to-one assignment."""
    if max_distance_mm < 0 or not math.isfinite(max_distance_mm):
        raise ValueError("max_distance_mm must be a finite non-negative value")
    if not predictions or not references:
        return ()

    predicted_points = np.asarray([item.ostium_xyz for item in predictions], dtype=float)
    reference_points = np.asarray([item.ostium_xyz for item in references], dtype=float)
    distances = np.linalg.norm(
        predicted_points[:, np.newaxis, :] - reference_points[np.newaxis, :, :],
        axis=2,
    )

    # A forbidden edge costs more than every possible valid edge combined, so
    # assignment first maximizes valid match count and then minimizes distance.
    assignment_size = min(len(predictions), len(references))
    forbidden_cost = (assignment_size + 1) * (max_distance_mm + 1.0)
    costs = np.where(distances <= max_distance_mm, distances, forbidden_cost)
    rows, columns = linear_sum_assignment(costs)
    matches = [
        OstiumMatch(int(row), int(column), float(distances[row, column]))
        for row, column in zip(rows, columns, strict=True)
        if distances[row, column] <= max_distance_mm
    ]
    return tuple(sorted(matches, key=lambda match: match.prediction_index))


def _mean(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _distance(first: Sequence[float], second: Sequence[float]) -> float:
    return float(np.linalg.norm(np.asarray(first, dtype=float) - np.asarray(second, dtype=float)))


def _angular_error_degrees(first: Sequence[float], second: Sequence[float]) -> float | None:
    first_vector = np.asarray(first, dtype=float)
    second_vector = np.asarray(second, dtype=float)
    denominator = np.linalg.norm(first_vector) * np.linalg.norm(second_vector)
    if denominator == 0:
        return None
    cosine = float(np.clip(np.dot(first_vector, second_vector) / denominator, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def evaluate_case(
    predictions: Sequence[DaughterPrediction],
    references: Sequence[DaughterPrediction],
    *,
    max_distance_mm: float = 10.0,
) -> EvaluationResult:
    """Evaluate one case; optional attribute errors use available matched pairs."""
    matches = match_ostia(
        predictions, references, max_distance_mm=max_distance_mm
    )
    matched_predictions = {match.prediction_index for match in matches}
    matched_references = {match.reference_index for match in matches}
    unmatched_predictions = tuple(
        index for index in range(len(predictions)) if index not in matched_predictions
    )
    unmatched_references = tuple(
        index for index in range(len(references)) if index not in matched_references
    )

    tp = len(matches)
    # An empty denominator means there were no errors of that type.
    precision = tp / len(predictions) if predictions else 1.0
    recall = tp / len(references) if references else 1.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    seed_errors: list[float] = []
    direction_errors: list[float] = []
    radius_errors: list[float] = []
    for match in matches:
        prediction = predictions[match.prediction_index]
        reference = references[match.reference_index]
        if prediction.seed_xyz is not None and reference.seed_xyz is not None:
            seed_errors.append(_distance(prediction.seed_xyz, reference.seed_xyz))
        if prediction.direction_xyz is not None and reference.direction_xyz is not None:
            angular_error = _angular_error_degrees(
                prediction.direction_xyz, reference.direction_xyz
            )
            if angular_error is not None:
                direction_errors.append(angular_error)
        if prediction.radius_mm is not None and reference.radius_mm is not None:
            radius_errors.append(abs(prediction.radius_mm - reference.radius_mm))

    return EvaluationResult(
        matches=matches,
        unmatched_prediction_indices=unmatched_predictions,
        unmatched_reference_indices=unmatched_references,
        precision=precision,
        recall=recall,
        f1=f1,
        ostium_error_mm=_mean([match.distance_mm for match in matches]),
        seed_distance_mm=_mean(seed_errors),
        direction_angular_error_deg=_mean(direction_errors),
        radius_error_mm=_mean(radius_errors),
    )
