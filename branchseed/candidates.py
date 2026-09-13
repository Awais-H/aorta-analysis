"""Typed, evidence-preserving aortic daughter candidate proposals."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from .aorta import AortaGeometry


@dataclass(frozen=True, slots=True)
class CandidateEvidence:
    blood_similarity: float = 0.0
    tubularity: float = 0.0
    wall_contact: float = 0.0
    outward: float = 0.0
    persistence_mm: float = 0.0
    channels: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Candidate:
    """A branch proposal in physical x-y-z coordinates."""

    ostium_xyz: tuple[float, float, float]
    seed_xyz: tuple[float, float, float]
    direction_xyz: tuple[float, float, float]
    radius_mm: float
    score: float
    source: str
    evidence: CandidateEvidence
    path_xyz: tuple[tuple[float, float, float], ...] = ()
    voxel_count: int = 0


def _sampling(spacing_xyz: Sequence[float]) -> np.ndarray:
    spacing = np.asarray(tuple(spacing_xyz), float)
    if spacing.shape != (3,) or np.any(~np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError("spacing_xyz must contain three positive finite values")
    return spacing[::-1]


def _structure(radius_mm: float, spacing_xyz: Sequence[float]) -> np.ndarray:
    sampling = _sampling(spacing_xyz)
    radii = np.ceil(radius_mm / sampling).astype(int)
    z, y, x = np.ogrid[
        -radii[0] : radii[0] + 1,
        -radii[1] : radii[1] + 1,
        -radii[2] : radii[2] + 1,
    ]
    return sum((grid * step) ** 2 for grid, step in zip((z, y, x), sampling)) <= radius_mm**2


def _to_physical(
    index_zyx: np.ndarray,
    spacing_xyz: Sequence[float],
    origin_xyz: Sequence[float],
    direction: Sequence[float],
) -> np.ndarray:
    scaled_xyz = np.asarray(index_zyx, float)[::-1] * np.asarray(spacing_xyz, float)
    return np.asarray(origin_xyz, float) + np.asarray(direction, float).reshape(3, 3) @ scaled_xyz


def robust_wall_normal(
    signed_distance_mm: np.ndarray,
    point_zyx: Sequence[float],
    spacing_xyz: Sequence[float],
    *,
    radius_mm: float = 3.0,
    min_coherence: float = 0.35,
) -> tuple[np.ndarray | None, float]:
    """Estimate an outward normal by robust local averaging.

    The SDF is positive inside, so its gradient points inward. Individual raw
    gradients are never exposed: finite, non-trivial vectors in a physical
    neighbourhood are normalized, median-averaged, and rejected when their
    directional coherence is low.
    """
    sdf = np.asarray(signed_distance_mm, dtype=np.float32)
    sampling = _sampling(spacing_xyz)
    gradients = np.gradient(ndimage.gaussian_filter(sdf, 0.7 / sampling), *sampling)
    centre = np.rint(point_zyx).astype(int)
    radii = np.ceil(radius_mm / sampling).astype(int)
    starts = np.maximum(centre - radii, 0)
    stops = np.minimum(centre + radii + 1, sdf.shape)
    slices = tuple(slice(a, b) for a, b in zip(starts, stops))
    vectors = np.stack([component[slices].ravel() for component in gradients], axis=1)
    norm = np.linalg.norm(vectors, axis=1)
    vectors = vectors[np.isfinite(norm) & (norm > 0.15)] / norm[np.isfinite(norm) & (norm > 0.15), None]
    if len(vectors) < 5:
        return None, 0.0
    inward_zyx = np.median(vectors, axis=0)
    coherence = float(np.linalg.norm(inward_zyx))
    if coherence < min_coherence:
        return None, coherence
    outward_zyx = -inward_zyx / coherence
    return outward_zyx, coherence


def _line_indices(start: np.ndarray, stop: np.ndarray, shape: Sequence[int]) -> np.ndarray:
    count = max(2, int(np.ceil(np.linalg.norm(stop - start))) + 1)
    line = np.rint(np.linspace(start, stop, count)).astype(int)
    return np.clip(line, 0, np.asarray(shape) - 1)


def _candidate_from_component(
    component: np.ndarray,
    lateral_wall: np.ndarray,
    similarity: np.ndarray,
    tubularity: np.ndarray,
    geometry: AortaGeometry,
    origin_xyz: Sequence[float],
    direction: Sequence[float],
    source: str,
) -> Candidate | None:
    sampling = _sampling(geometry.spacing_xyz)
    points = np.argwhere(component)
    wall_points = np.argwhere(lateral_wall)
    if not len(points) or not len(wall_points):
        return None
    # Exact nearest wall voxel is cheap for the local synthetic/clinical crop.
    tree = None
    try:
        tree = cKDTree(wall_points * sampling)
        distances, nearest = tree.query(points * sampling)
        index = int(np.argmin(distances))
        ostium = wall_points[int(nearest[index])]
    except Exception:
        distance = np.linalg.norm(
            (points[:, None, :] - wall_points[None, :, :]) * sampling, axis=2
        )
        index = int(np.unravel_index(np.argmin(distance), distance.shape)[0])
        ostium = wall_points[int(np.argmin(distance[index]))]
        distances = distance.min(axis=1)
    far_index = int(np.argmax(distances))
    farthest = points[far_index]
    persistence = float(distances[far_index])
    normal, coherence = robust_wall_normal(
        geometry.signed_distance_mm, ostium, geometry.spacing_xyz
    )
    data_direction = (farthest - ostium) * sampling
    data_norm = np.linalg.norm(data_direction)
    if data_norm < 1e-6:
        return None
    data_direction /= data_norm
    outward = float(max(0.0, np.dot(normal, data_direction))) if normal is not None else 0.0
    direction_zyx = data_direction if normal is None else data_direction + coherence * normal
    direction_zyx /= max(np.linalg.norm(direction_zyx), 1e-6)
    seed_distance = min(max(persistence * 0.5, 2.0), 8.0)
    seed_index = ostium + direction_zyx * seed_distance / sampling
    # Components may be wider than their radius. Estimate volume-equivalent radius
    # per unit length and keep it within the proposal search range.
    volume = len(points) * float(np.prod(geometry.spacing_xyz))
    radius = float(np.sqrt(volume / (np.pi * max(persistence, min(sampling)))))
    blood = float(np.mean(similarity[component]))
    tube = float(np.mean(tubularity[component]))
    contact = float(np.exp(-float(np.min(distances)) / 3.0))
    persistence_score = min(persistence / (8.0 if source == "primary" else 20.0), 1.0)
    path_indices = _line_indices(ostium, farthest, component.shape)
    # Explicitly sample the whole outward trace. A bright blob near the wall is
    # not wall-crossing evidence unless support persists while wall distance
    # increases along the proposed direction.
    trace_support = np.maximum(
        similarity[tuple(path_indices.T)], tubularity[tuple(path_indices.T)]
    )
    if tree is not None:
        trace_distance = tree.query(path_indices * sampling)[0]
    else:
        trace_distance = np.min(
            np.linalg.norm(
                (path_indices[:, None, :] - wall_points[None, :, :]) * sampling,
                axis=2,
            ),
            axis=1,
        )
    continuity = float(np.mean(trace_support[1:] >= 0.18)) if len(path_indices) > 1 else 0.0
    monotonic = float(np.mean(np.diff(trace_distance) >= -0.75)) if len(path_indices) > 1 else 0.0
    score = float(np.clip(
        0.28 * blood + 0.23 * tube + 0.16 * contact + 0.11 * outward
        + 0.08 * persistence_score + 0.09 * continuity + 0.05 * monotonic,
        0, 1,
    ))
    path = tuple(
        tuple(float(v) for v in _to_physical(p, geometry.spacing_xyz, origin_xyz, direction))
        for p in path_indices[:: max(1, len(path_indices) // 8)]
    )
    direction_xyz = np.asarray(direction, float).reshape(3, 3) @ direction_zyx[::-1]
    direction_xyz /= max(np.linalg.norm(direction_xyz), 1e-6)
    return Candidate(
        ostium_xyz=tuple(float(v) for v in _to_physical(ostium, geometry.spacing_xyz, origin_xyz, direction)),
        seed_xyz=tuple(float(v) for v in _to_physical(seed_index, geometry.spacing_xyz, origin_xyz, direction)),
        direction_xyz=tuple(float(v) for v in direction_xyz),
        radius_mm=float(np.clip(radius, 0.5, 15.0)),
        score=score,
        source=source,
        evidence=CandidateEvidence(
            blood_similarity=blood,
            tubularity=tube,
            wall_contact=contact,
            outward=outward,
            persistence_mm=persistence,
            channels={
                "normal_coherence": coherence,
                "outward_trace_continuity": continuity,
                "outward_distance_monotonicity": monotonic,
            },
        ),
        path_xyz=path,
        voxel_count=int(len(points)),
    )


def _components(mask: np.ndarray, min_voxels: int) -> list[np.ndarray]:
    labels, count = ndimage.label(mask, ndimage.generate_binary_structure(3, 2))
    sizes = np.bincount(labels.ravel())
    return [labels == number for number in range(1, count + 1) if sizes[number] >= min_voxels]


def merge_candidates(
    candidates: Sequence[Candidate],
    *,
    merge_distance_mm: float = 5.0,
) -> list[Candidate]:
    """Merge only strongly justified cross-channel duplicate evidence."""
    if not np.isfinite(merge_distance_mm) or merge_distance_mm <= 0:
        raise ValueError("merge_distance_mm must be finite and positive")
    merged: list[Candidate] = []
    for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
        match = next(
            (
                i
                for i, existing in enumerate(merged)
                if _duplicate_candidate(existing, candidate, merge_distance_mm)
            ),
            None,
        )
        if match is None:
            merged.append(candidate)
            continue
        existing = merged[match]
        channels = dict(existing.evidence.channels)
        channels[f"{candidate.source}_score"] = candidate.score
        evidence = CandidateEvidence(
            blood_similarity=max(existing.evidence.blood_similarity, candidate.evidence.blood_similarity),
            tubularity=max(existing.evidence.tubularity, candidate.evidence.tubularity),
            wall_contact=max(existing.evidence.wall_contact, candidate.evidence.wall_contact),
            outward=max(existing.evidence.outward, candidate.evidence.outward),
            persistence_mm=max(existing.evidence.persistence_mm, candidate.evidence.persistence_mm),
            channels=channels,
        )
        merged[match] = Candidate(
            ostium_xyz=existing.ostium_xyz,
            seed_xyz=existing.seed_xyz,
            direction_xyz=existing.direction_xyz,
            radius_mm=existing.radius_mm,
            score=max(existing.score, candidate.score),
            source="merged",
            evidence=evidence,
            path_xyz=existing.path_xyz,
            voxel_count=existing.voxel_count + candidate.voxel_count,
        )
    return merged


def _duplicate_candidate(
    first: Candidate, second: Candidate, merge_distance_mm: float
) -> bool:
    """Require independent channels plus matching contact, direction, and trace."""
    if first.source == second.source or "merged" in {first.source, second.source}:
        return False
    distance = float(
        np.linalg.norm(np.asarray(first.ostium_xyz) - np.asarray(second.ostium_xyz))
    )
    if distance > merge_distance_mm:
        return False
    first_direction = np.asarray(first.direction_xyz, float)
    second_direction = np.asarray(second.direction_xyz, float)
    cosine = float(np.dot(first_direction, second_direction)) / max(
        float(np.linalg.norm(first_direction) * np.linalg.norm(second_direction)), 1e-9
    )
    if cosine < 0.90 or min(first.evidence.wall_contact, second.evidence.wall_contact) < 0.5:
        return False
    if distance <= min(1.5, merge_distance_mm):
        return True
    if not first.path_xyz or not second.path_xyz:
        return False
    first_path = np.asarray(first.path_xyz, float)
    second_path = np.asarray(second.path_xyz, float)
    first_overlap = np.mean(cKDTree(second_path).query(first_path)[0] <= 1.5)
    second_overlap = np.mean(cKDTree(first_path).query(second_path)[0] <= 1.5)
    return bool(min(first_overlap, second_overlap) >= 0.70)


def generate_candidates(
    geometry: AortaGeometry,
    blood_similarity_zyx: np.ndarray,
    tubularity_zyx: np.ndarray,
    *,
    origin_xyz: Sequence[float] = (0.0, 0.0, 0.0),
    direction: Sequence[float] = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
    primary_distance_mm: float = 8.0,
    fallback_distance_mm: tuple[float, float] = (15.0, 30.0),
    min_component_mm3: float = 3.0,
    merge_distance_mm: float = 5.0,
) -> list[Candidate]:
    """Generate wall-crossing primary and wider tubular fallback proposals."""
    similarity = np.asarray(blood_similarity_zyx, np.float32)
    tubular = np.asarray(tubularity_zyx, np.float32)
    if similarity.shape != geometry.mask.shape or tubular.shape != geometry.mask.shape:
        raise ValueError("feature arrays must match the aorta geometry shape")
    sampling = _sampling(geometry.spacing_xyz)
    outside = ~geometry.mask
    wall_distance, nearest = ndimage.distance_transform_edt(
        ~geometry.lateral_wall, sampling=sampling, return_indices=True
    )
    voxel_volume = float(np.prod(geometry.spacing_xyz))
    min_voxels = max(2, int(np.ceil(min_component_mm3 / voxel_volume)))
    primary_mask = (
        outside
        & (wall_distance <= primary_distance_mm)
        & (similarity >= 0.48)
        & ((tubular >= 0.08) | (similarity >= 0.72))
    )
    proposed: list[Candidate] = []
    for component in _components(primary_mask, min_voxels):
        # Connected patches must approach the lateral wall and persist outward.
        if float(wall_distance[component].min()) > max(sampling) * 1.8:
            continue
        candidate = _candidate_from_component(
            component, geometry.lateral_wall, similarity, tubular,
            geometry, origin_xyz, direction, "primary"
        )
        if candidate is not None and candidate.evidence.persistence_mm >= 2.0:
            proposed.append(candidate)

    near, far = (float(v) for v in fallback_distance_mm)
    if not (0 < near <= far):
        raise ValueError("fallback_distance_mm must be a positive ordered pair")
    # Lower similarity tolerates a proximal contrast gap, but requires connected
    # tubular evidence and a physical trace no farther than `far` from the wall.
    fallback_mask = (
        outside
        & (wall_distance <= far)
        & (tubular >= 0.18)
        & (similarity >= 0.18)
        & ~primary_mask
    )
    for component in _components(fallback_mask, min_voxels):
        distances = wall_distance[component]
        if distances.max() < near or distances.min() > far:
            continue
        candidate = _candidate_from_component(
            component, geometry.lateral_wall, similarity, tubular,
            geometry, origin_xyz, direction, "fallback"
        )
        if candidate is not None:
            proposed.append(candidate)
    return merge_candidates(proposed, merge_distance_mm=merge_distance_mm)
