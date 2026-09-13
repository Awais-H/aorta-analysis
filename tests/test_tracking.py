import numpy as np
from dataclasses import replace

from branchseed.aorta import analyze_aorta
from branchseed.candidates import Candidate, CandidateEvidence
from branchseed.tracking import (
    physical_path_length,
    track_local_segmentation,
    track_minimal_path,
    verify_path,
    voxel_to_physical,
)
from branchseed.instances import derive_instances
from branchseed.measurements import measure_instance


def _case(direction=(1, 0, 0, 0, 1, 0, 0, 0, 1)):
    shape = (32, 48, 56)
    z, y, x = np.indices(shape)
    aorta = (y - 24) ** 2 + (x - 20) ** 2 <= 7**2
    daughter = ((z - 16) ** 2 + (y - 24) ** 2 <= 2.5**2) & (x >= 27) & (x <= 44)
    image = np.where(aorta | daughter, 180, 20).astype(np.float32)
    geometry = analyze_aorta(image, aorta, (1, 1, 1.5))
    ostium = voxel_to_physical((16, 24, 27), geometry.spacing_xyz, direction=direction)
    seed = voxel_to_physical((16, 24, 32), geometry.spacing_xyz, direction=direction)
    direction_xyz = np.asarray(direction).reshape(3, 3) @ np.asarray((1, 0, 0))
    candidate = Candidate(
        tuple(ostium), tuple(seed), tuple(direction_xyz), 2.5, 0.9, "synthetic",
        CandidateEvidence(0.9, 0.9, 1, 1, 17),
        tuple(tuple(voxel_to_physical((16, 24, x), geometry.spacing_xyz, direction=direction))
              for x in range(27, 45)),
    )
    blood = daughter.astype(np.float32) * 0.9
    tube = daughter.astype(np.float32) * 0.8
    return image, geometry, candidate, blood, tube


def test_independent_tracking_methods_verify_direct_supported_paths():
    image, geometry, candidate, blood, tube = _case()
    segmented = track_local_segmentation(
        image, geometry, candidate, blood_similarity_zyx=blood,
        tubularity_zyx=tube, use_kimimaro=False,
    )
    minimal = track_minimal_path(geometry, candidate, blood, tube)
    assert segmented is not None
    assert minimal is not None
    assert segmented.method != minimal.method
    assert segmented.verification.passed and minimal.verification.passed
    assert segmented.length_mm >= 5 and minimal.length_mm >= 5
    assert minimal.vessel_mask_zyx is not None
    assert minimal.metadata["vessel_mask_source"] == "local_feature_component"


def test_verification_rejects_short_and_unsupported_paths():
    _, geometry, _, blood, tube = _case()
    short = [(16, 24, 27), (16, 24, 29)]
    result = verify_path(short, geometry, blood_similarity_zyx=blood, tubularity_zyx=tube)
    assert not result.passed
    assert "path_too_short" in result.reasons
    unsupported = [(16, 24, x) for x in range(27, 40)]
    result = verify_path(
        unsupported, geometry, blood_similarity_zyx=np.zeros_like(blood),
        tubularity_zyx=np.zeros_like(tube),
    )
    assert not result.image_support


def test_rotated_anisotropic_physical_path():
    rotation = (0, -1, 0, 1, 0, 0, 0, 0, 1)
    image, geometry, candidate, blood, tube = _case(rotation)
    result = track_minimal_path(
        geometry, candidate, blood, tube, direction=rotation
    )
    assert result is not None
    displacement = np.asarray(result.path_xyz[-1]) - result.path_xyz[0]
    assert displacement[1] > 5
    assert abs(physical_path_length(result.path_xyz) - result.length_mm) < 1e-6


def test_minimal_path_radius_uses_image_component_not_candidate_radius():
    _, geometry, candidate, blood, tube = _case()
    inflated = replace(candidate, radius_mm=9.0)
    result = track_minimal_path(geometry, inflated, blood, tube)
    assert result is not None and result.vessel_mask_zyx is not None
    instance = derive_instances([result], geometry)[0]
    measurement = measure_instance(instance, geometry.spacing_xyz)
    assert 1.5 < measurement.radius_mm < 4.0
    assert not np.isclose(measurement.radius_mm, inflated.radius_mm)


def test_minimal_path_enforces_length_and_expansion_bounds():
    _, geometry, candidate, blood, tube = _case()
    assert track_minimal_path(
        geometry, candidate, blood, tube, max_length_mm=4.0
    ) is None
    assert track_minimal_path(
        geometry, candidate, blood, tube, max_expansions=1
    ) is None

