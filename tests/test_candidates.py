import numpy as np

from branchseed.aorta import analyze_aorta
from branchseed.candidates import (
    Candidate,
    CandidateEvidence,
    generate_candidates,
    merge_candidates,
    robust_wall_normal,
)


def _geometry():
    shape = (40, 64, 64)
    z, y, x = np.indices(shape)
    mask = (y - 32) ** 2 + (x - 25) ** 2 <= 8**2
    intensity = np.where(mask, 180, 0).astype(np.float32)
    return analyze_aorta(intensity, mask, (1, 1, 1.5))


def test_primary_wall_crossing_side_daughter():
    geometry = _geometry()
    similarity = np.zeros(geometry.mask.shape, np.float32)
    tubular = np.zeros_like(similarity)
    similarity[17:23, 29:36, 33:43] = 0.9
    tubular[17:23, 30:35, 33:43] = 0.8
    candidates = generate_candidates(geometry, similarity, tubular)
    assert candidates
    candidate = max(candidates, key=lambda item: item.score)
    assert candidate.source in {"primary", "merged"}
    assert candidate.direction_xyz[0] > 0.5
    assert candidate.evidence.persistence_mm >= 2


def test_low_contrast_proximal_gap_uses_wider_fallback():
    geometry = _geometry()
    similarity = np.zeros(geometry.mask.shape, np.float32)
    tubular = np.zeros_like(similarity)
    # The tube starts beyond the primary wall-contact range but remains within
    # the configured 15--30 mm fallback search.
    similarity[18:22, 30:35, 42:58] = 0.30
    tubular[18:22, 30:35, 42:58] = 0.85
    candidates = generate_candidates(geometry, similarity, tubular)
    fallback = [item for item in candidates if item.source in {"fallback", "merged"}]
    assert fallback
    assert fallback[0].evidence.tubularity > 0.5
    assert fallback[0].evidence.persistence_mm >= 15


def test_noisy_sdf_normal_is_rejected_and_empty_features_produce_none():
    geometry = _geometry()
    rng = np.random.default_rng(4)
    normal, coherence = robust_wall_normal(
        rng.normal(size=geometry.mask.shape).astype(np.float32),
        (20, 32, 33),
        geometry.spacing_xyz,
        min_coherence=0.8,
    )
    assert normal is None
    assert coherence < 0.8
    zeros = np.zeros(geometry.mask.shape, np.float32)
    assert generate_candidates(geometry, zeros, zeros) == []


def test_candidate_coordinates_respect_rotated_direction():
    geometry = _geometry()
    similarity = np.zeros(geometry.mask.shape, np.float32)
    tubular = np.zeros_like(similarity)
    similarity[17:23, 29:36, 33:43] = 0.9
    tubular[17:23, 30:35, 33:43] = 0.8
    rotation = (0, -1, 0, 1, 0, 0, 0, 0, 1)
    candidate = generate_candidates(
        geometry, similarity, tubular, origin_xyz=(10, 20, 30), direction=rotation
    )[0]
    # Array +x maps to physical +y under this direction matrix.
    assert candidate.direction_xyz[1] > 0.5


def test_candidate_merge_preserves_nearby_same_channel_ostia():
    evidence = CandidateEvidence(wall_contact=1.0)
    first = Candidate((0, 0, 0), (5, 0, 0), (1, 0, 0), 2, 0.9, "primary", evidence)
    nearby = Candidate((0, 0, 3), (5, 0, 3), (1, 0, 0), 2, 0.8, "primary", evidence)
    assert len(merge_candidates([first, nearby])) == 2


def test_candidate_merge_requires_independent_aligned_duplicate_evidence():
    evidence = CandidateEvidence(wall_contact=1.0)
    primary = Candidate((0, 0, 0), (5, 0, 0), (1, 0, 0), 2, 0.9, "primary", evidence)
    duplicate = Candidate(
        (0.5, 0, 0), (5, 0, 0), (1, 0, 0), 2, 0.8, "fallback", evidence
    )
    opposing = Candidate(
        (0.5, 0, 0), (-5, 0, 0), (-1, 0, 0), 2, 0.7, "fallback", evidence
    )
    assert len(merge_candidates([primary, duplicate])) == 1
    assert len(merge_candidates([primary, opposing])) == 2
