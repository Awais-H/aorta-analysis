import json

import pytest

from branchseed.models import CasePrediction, DaughterPrediction


def test_prediction_json_round_trip(tmp_path):
    prediction = CasePrediction(
        case_id="case-1",
        daughters=(
            DaughterPrediction(
                ostium_xyz=(1, 2, 3),
                seed_xyz=(4, 5, 6),
                direction_xyz=(1, 0, 0),
                radius_mm=2.5,
                candidate_path_xyz=((1, 2, 3), (2, 2, 3)),
                metadata={"method": "test"},
            ),
        ),
    )
    path = tmp_path / "prediction.json"
    prediction.save_json(path)

    assert CasePrediction.load_json(path) == prediction
    assert CasePrediction.from_json(prediction.to_json()) == prediction
    assert json.loads(prediction.to_json())["daughters"][0]["ostium_xyz"] == [1.0, 2.0, 3.0]


@pytest.mark.parametrize(
    "payload, message",
    [
        ({"case_id": "x", "daughters": [{"seed_xyz": [1, 2, 3]}]}, "ostium_xyz"),
        ({"case_id": "x", "daughters": [{"ostium_xyz": [1, 2]}]}, "three numbers"),
        ({"case_id": "x", "daughters": "invalid"}, "JSON array"),
        ({"case_id": "x", "unknown": True}, "unexpected case fields"),
    ],
)
def test_invalid_json_shapes_raise_clear_errors(payload, message):
    with pytest.raises(ValueError, match=message):
        CasePrediction.from_dict(payload)


def test_non_serializable_metadata_is_rejected():
    prediction = CasePrediction("case", metadata={"bad": object()})
    with pytest.raises(ValueError, match="not JSON serializable"):
        prediction.to_json()
