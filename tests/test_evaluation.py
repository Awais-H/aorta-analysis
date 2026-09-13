import pytest

from branchseed.evaluation import evaluate_case, match_ostia
from branchseed.models import DaughterPrediction


def daughter(ostium, *, seed=None, direction=None, radius=None):
    return DaughterPrediction(
        ostium_xyz=ostium,
        seed_xyz=seed,
        direction_xyz=direction,
        radius_mm=radius,
    )


def test_hungarian_matching_is_one_to_one_and_thresholded():
    predictions = [daughter((0, 0, 0)), daughter((1, 0, 0)), daughter((100, 0, 0))]
    references = [daughter((0.2, 0, 0)), daughter((2, 0, 0))]

    matches = match_ostia(predictions, references, max_distance_mm=2)

    assert [(match.prediction_index, match.reference_index) for match in matches] == [
        (0, 0),
        (1, 1),
    ]


def test_metrics_and_matched_attribute_errors():
    predictions = [
        daughter((1, 0, 0), seed=(2, 0, 0), direction=(1, 0, 0), radius=3),
        daughter((50, 0, 0)),
    ]
    references = [
        daughter((0, 0, 0), seed=(0, 0, 0), direction=(0, 1, 0), radius=1)
    ]

    result = evaluate_case(predictions, references, max_distance_mm=5)

    assert result.precision == 0.5
    assert result.recall == 1.0
    assert result.f1 == pytest.approx(2 / 3)
    assert result.ostium_error_mm == 1.0
    assert result.seed_distance_mm == 2.0
    assert result.direction_angular_error_deg == 90.0
    assert result.radius_error_mm == 2.0
    assert result.unmatched_prediction_indices == (1,)


def test_empty_sets_are_well_defined():
    both_empty = evaluate_case([], [])
    assert (both_empty.precision, both_empty.recall, both_empty.f1) == (1.0, 1.0, 1.0)
    assert both_empty.ostium_error_mm is None

    predictions_only = evaluate_case([daughter((0, 0, 0))], [])
    assert (predictions_only.precision, predictions_only.recall, predictions_only.f1) == (
        0.0,
        1.0,
        0.0,
    )


def test_invalid_matching_distance_is_rejected():
    with pytest.raises(ValueError, match="finite non-negative"):
        match_ostia([], [], max_distance_mm=-1)
