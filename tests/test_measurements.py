import numpy as np

from branchseed.aorta import analyze_aorta
from branchseed.candidates import Candidate, CandidateEvidence
from branchseed.instances import derive_instances
from branchseed.measurements import (
    anchor_path_at_ostium,
    estimate_radii,
    instance_to_prediction,
    interpolate_at_arc_length,
    measure_instance,
    robust_direction,
)
from branchseed.tracking import PathVerification, VerifiedPath, voxel_to_physical


def test_exact_five_mm_seed_on_curved_nonuniform_path():
    path = np.asarray([
        (0, 0, 0), (1, 0.05, 0), (2.8, 0.39, 0), (5.5, 1.51, 0),
        (8.5, 3.61, 0),
    ], float)
    seed, segment, fraction = interpolate_at_arc_length(path, 5.0)
    reconstructed = np.vstack([path[:segment + 1], seed])
    length = np.linalg.norm(np.diff(reconstructed, axis=0), axis=1).sum()
    assert abs(length - 5.0) < 1e-10
    assert 0 <= fraction <= 1
    direction = robust_direction(path, 5.0)
    assert np.isclose(np.linalg.norm(direction), 1)
    assert direction[0] > 0.8 and direction[1] > 0


def test_anisotropic_rotated_orthogonal_radii():
    spacing = (1.0, 1.5, 2.0)
    rotation = (0, -1, 0, 1, 0, 0, 0, 0, 1)
    z, y, x = np.indices((25, 35, 50))
    vessel = (((z - 12) * spacing[2]) ** 2 + ((y - 17) * spacing[1]) ** 2 <= 3.2**2)
    seed = voxel_to_physical((12, 17, 25), spacing, direction=rotation)
    direction = np.asarray(rotation).reshape(3, 3) @ np.asarray((1, 0, 0))
    radii = estimate_radii(
        vessel, seed, direction, spacing, image_direction=rotation,
        plane_resolution_mm=0.25,
    )
    assert all(2.4 < radius < 4.0 for radius in radii)
    assert max(radii) - min(radii) < 1.2


def test_measurement_prediction_contains_seed_direction_radii_and_flags():
    shape = (32, 48, 58)
    z, y, x = np.indices(shape)
    aorta = (y - 24) ** 2 + (x - 20) ** 2 <= 7**2
    geometry = analyze_aorta(aorta.astype(np.float32) * 180, aorta, (1, 1, 1))
    points = np.asarray([(16, 24, x) for x in range(27, 46)])
    vessel = ((z - 16) ** 2 + (y - 24) ** 2 <= 3**2) & (x >= 27) & (x <= 47)
    xyz = tuple(tuple(voxel_to_physical(point, geometry.spacing_xyz)) for point in points)
    candidate = Candidate(
        xyz[0], xyz[5], (1, 0, 0), 3, 0.9, "synthetic",
        CandidateEvidence(0.9, 0.8, 1, 1, 18), xyz,
    )
    verification = PathVerification(True, True, True, True, 18, 1, 18)
    path = VerifiedPath(
        "synthetic", candidate, tuple(tuple(float(v) for v in p) for p in points),
        xyz, verification, vessel,
    )
    instance = derive_instances([path], geometry)[0]
    measured = measure_instance(instance, geometry.spacing_xyz)
    assert np.isclose(
        np.linalg.norm(np.asarray(measured.seed_xyz) - np.asarray(measured.ostium_xyz)),
        5.0,
    )
    assert measured.direction_xyz[0] > 0.99
    assert 2.5 < measured.radius_mm < 4
    assert "single_method_support" in measured.quality_flags
    prediction = instance_to_prediction(instance, geometry.spacing_xyz)
    assert prediction.seed_xyz == measured.seed_xyz
    assert prediction.radius_mm == measured.radius_mm
    assert prediction.metadata["seed_arc_length_mm"] == 5
    assert prediction.candidate_path_xyz[0] == prediction.ostium_xyz
    anchored = anchor_path_at_ostium(path.path_xyz, instance.ostium_xyz)
    expected_seed, _, _ = interpolate_at_arc_length(anchored, 5.0)
    assert np.allclose(prediction.seed_xyz, expected_seed)

