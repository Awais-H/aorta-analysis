import numpy as np
import pytest

from branchseed.aorta import BloodModel
from branchseed.features import (
    blood_similarity,
    compose_score,
    gradient_magnitude,
    multiscale_hessian_objectness,
    optional_advanced_filter,
    tubularity_from_eigenvalues,
)


def test_patient_adaptive_blood_similarity():
    model = BloodModel(median=180, mad=5, low=165, high=195, sample_count=100)
    values = np.array([20, 170, 180, 190, 350], np.float32)
    score = blood_similarity(values, model)
    assert score[2] > 0.99
    assert score[1] >= 0.75
    assert score[0] < 0.01


def test_spacing_aware_gradient_and_tubularity_helper():
    z, y, x = np.indices((8, 9, 10))
    physical_ramp = z * 3.0 + y * 2.0 + x
    gradient = gradient_magnitude(physical_ramp, (1, 2, 3), sigma_mm=0)
    np.testing.assert_allclose(gradient[2:-2, 2:-2, 2:-2], np.sqrt(3), atol=1e-5)
    eigenvalues = np.array([[[[-0.01, -2.0, -3.0], [0.01, 2.0, 3.0]]]])
    response = tubularity_from_eigenvalues(eigenvalues)
    assert response[0, 0, 0] > 0
    assert response[0, 0, 1] == 0


def test_sitk_multiscale_objectness_uses_local_crop_float32():
    image = np.zeros((25, 25, 25), np.float32)
    image[:, 11:14, 11:14] = 200
    crop = (slice(3, 22), slice(5, 20), slice(5, 20))
    response = multiscale_hessian_objectness(
        image, (1.0, 1.0, 1.5), (1.0, 2.0), crop_zyx=crop
    )
    assert response.dtype == np.float32
    assert response.shape == image.shape
    assert response[12, 12, 12] > 0
    assert not response[:3].any()


def test_composable_score_and_optional_failure():
    a = np.array([0, 1], np.float32)
    score = compose_score({"blood": a, "tube": np.ones(2)}, {"blood": 3, "tube": 1})
    np.testing.assert_allclose(score, [0.25, 1])
    with pytest.warns(RuntimeWarning):
        result = optional_advanced_filter(np.ones((2, 2, 2)), (1, 1, 1), implementation="missing")
    assert not result.any()
