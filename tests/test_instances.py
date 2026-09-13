import numpy as np
import pytest
from scipy import ndimage

from branchseed.aorta import analyze_aorta
from branchseed.candidates import Candidate, CandidateEvidence
from branchseed.instances import derive_instances, proximal_path_overlap
from branchseed.tracking import (
    PathVerification,
    VerifiedPath,
    verify_path,
    voxel_to_physical,
)


def _geometry():
    z, y, x = np.indices((36, 52, 60))
    mask = (y - 26) ** 2 + (x - 20) ** 2 <= 7**2
    return analyze_aorta(mask.astype(np.float32) * 200, mask, (1, 1, 1.2))


def _verified(geometry, points, method, mask):
    xyz = tuple(tuple(voxel_to_physical(p, geometry.spacing_xyz)) for p in points)
    candidate = Candidate(
        xyz[0], xyz[min(5, len(xyz) - 1)], (1, 0, 0), 2, 0.8, method,
        CandidateEvidence(0.8, 0.8, 1, 1, 10), xyz,
    )
    verification = PathVerification(True, True, True, True, 10, 1, 10)
    return VerifiedPath(
        method, candidate, tuple(tuple(float(v) for v in p) for p in points),
        xyz, verification, mask,
    )


def _tube(shape, points, radius=2):
    result = np.zeros(shape, bool)
    rounded = np.rint(points).astype(int)
    result[tuple(rounded.T)] = True
    return np.asarray(ndimage.binary_dilation(result, iterations=radius), bool)


def test_common_trunk_methods_merge_into_one_instance():
    geometry = _geometry()
    first = np.asarray([(18, 26, x) for x in range(27, 38)] + [(18, 27, x) for x in range(38, 43)])
    second = np.asarray([(18, 26, x) for x in range(27, 38)] + [(18, 25, x) for x in range(38, 43)])
    paths = [
        _verified(geometry, first, "segmentation", _tube(geometry.mask.shape, first)),
        _verified(geometry, second, "astar", _tube(geometry.mask.shape, second)),
    ]
    assert proximal_path_overlap(*paths) > 0.6
    instances = derive_instances(paths, geometry)
    assert len(instances) == 1
    assert len(instances[0].paths) == 2
    assert len(instances[0].trunk_path_xyz) < len(first)


def test_identical_contact_merges_even_when_paths_diverge_downstream():
    geometry = _geometry()
    first = np.asarray([(18, 26, 27), (18, 26, 28)] + [
        (18 + offset // 2, 26, 28 + offset) for offset in range(1, 12)
    ])
    second = np.asarray([(18, 26, 27), (18, 26, 28)] + [
        (18 - offset // 2, 26, 28 + offset) for offset in range(1, 12)
    ])
    paths = [
        _verified(geometry, first, "segmentation", _tube(geometry.mask.shape, first, 1)),
        _verified(geometry, second, "astar", _tube(geometry.mask.shape, second, 1)),
    ]
    assert proximal_path_overlap(*paths) < 0.60
    instances = derive_instances(paths, geometry)
    assert len(instances) == 1
    assert len(instances[0].paths) == 2


def test_nearby_distinct_ostia_are_preserved():
    geometry = _geometry()
    first = np.asarray([(14, 26, x) for x in range(27, 43)])
    second = np.asarray([(18, 26, x) for x in range(27, 43)])
    paths = [
        _verified(geometry, first, "segmentation", _tube(geometry.mask.shape, first, 1)),
        _verified(geometry, second, "segmentation", _tube(geometry.mask.shape, second, 1)),
    ]
    assert len(derive_instances(paths, geometry)) == 2


def test_indirect_daughter_cannot_be_a_verified_path():
    geometry = _geometry()
    points = [(18, 26, x) for x in range(35, 44)]
    result = verify_path(points, geometry)
    assert not result.direct_aorta_contact
    candidate = Candidate(
        (35, 26, 18), (38, 26, 18), (1, 0, 0), 2, 0.5, "indirect",
        CandidateEvidence(),
    )
    xyz = tuple(tuple(voxel_to_physical(p, geometry.spacing_xyz)) for p in points)
    with pytest.raises(ValueError, match="passing verification"):
        VerifiedPath("indirect", candidate, tuple(points), xyz, result)

