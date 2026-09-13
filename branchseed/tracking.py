"""CPU-only, independent daughter-vessel path trackers.

Arrays use ``(z, y, x)`` while physical coordinates use ``(x, y, z)``.
The segmentation/skeleton tracker and continuous-cost A* tracker intentionally
do not call one another, so agreement between them is useful evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import heapq
from typing import Mapping, Sequence

import numpy as np
from scipy import ndimage, sparse
from scipy.sparse.csgraph import dijkstra

from .aorta import AortaGeometry
from .candidates import Candidate, robust_wall_normal

Point3 = tuple[float, float, float]
Voxel3 = tuple[float, float, float]


def _sampling(spacing_xyz: Sequence[float]) -> np.ndarray:
    value = np.asarray(tuple(spacing_xyz), dtype=float)
    if value.shape != (3,) or np.any(~np.isfinite(value)) or np.any(value <= 0):
        raise ValueError("spacing_xyz must contain three positive finite values")
    return value[::-1]


def voxel_to_physical(
    point_zyx: Sequence[float],
    spacing_xyz: Sequence[float],
    origin_xyz: Sequence[float] = (0, 0, 0),
    direction: Sequence[float] = (1, 0, 0, 0, 1, 0, 0, 0, 1),
) -> np.ndarray:
    scaled = np.asarray(point_zyx, float)[::-1] * np.asarray(spacing_xyz, float)
    return np.asarray(origin_xyz, float) + np.asarray(direction, float).reshape(3, 3) @ scaled


def physical_to_voxel(
    point_xyz: Sequence[float],
    spacing_xyz: Sequence[float],
    origin_xyz: Sequence[float] = (0, 0, 0),
    direction: Sequence[float] = (1, 0, 0, 0, 1, 0, 0, 0, 1),
) -> np.ndarray:
    local = np.asarray(direction, float).reshape(3, 3).T @ (
        np.asarray(point_xyz, float) - np.asarray(origin_xyz, float)
    )
    return (local / np.asarray(spacing_xyz, float))[::-1]


def physical_path_length(path_xyz: Sequence[Sequence[float]]) -> float:
    points = np.asarray(path_xyz, float)
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


@dataclass(frozen=True, slots=True)
class PathVerification:
    direct_aorta_contact: bool
    length_ok: bool
    outward_progression: bool
    image_support: bool
    path_length_mm: float
    support_fraction: float
    outward_displacement_mm: float
    reasons: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return (
            self.direct_aorta_contact
            and self.length_ok
            and self.outward_progression
            and self.image_support
        )


@dataclass(frozen=True, slots=True)
class VerifiedPath:
    """A path carrying the same geometry in voxel and physical coordinates."""

    method: str
    candidate: Candidate
    path_zyx: tuple[Voxel3, ...]
    path_xyz: tuple[Point3, ...]
    verification: PathVerification
    vessel_mask_zyx: np.ndarray | None = field(default=None, repr=False, compare=False)
    metadata: Mapping[str, float | str | bool | int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.path_zyx) != len(self.path_xyz) or len(self.path_xyz) < 2:
            raise ValueError("voxel and physical paths must have equal length >= 2")
        if not self.verification.passed:
            raise ValueError("VerifiedPath requires passing verification")
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def length_mm(self) -> float:
        return self.verification.path_length_mm


_OFFSETS = np.asarray(
    [(z, y, x) for z in (-1, 0, 1) for y in (-1, 0, 1) for x in (-1, 0, 1)
     if (z, y, x) != (0, 0, 0)],
    dtype=int,
)


def _ordered_physical(
    path_zyx: np.ndarray,
    spacing_xyz: Sequence[float],
    origin_xyz: Sequence[float],
    direction: Sequence[float],
) -> tuple[tuple[Voxel3, ...], tuple[Point3, ...]]:
    voxel = tuple(tuple(float(v) for v in point) for point in path_zyx)
    physical = tuple(
        tuple(float(v) for v in voxel_to_physical(point, spacing_xyz, origin_xyz, direction))
        for point in path_zyx
    )
    return voxel, physical


def verify_path(
    path_zyx: Sequence[Sequence[float]],
    geometry: AortaGeometry,
    *,
    blood_similarity_zyx: np.ndarray | None = None,
    tubularity_zyx: np.ndarray | None = None,
    origin_xyz: Sequence[float] = (0, 0, 0),
    direction: Sequence[float] = (1, 0, 0, 0, 1, 0, 0, 0, 1),
    min_length_mm: float = 5.0,
    min_support_fraction: float = 0.55,
) -> PathVerification:
    """Verify contact, physical length, outward ordering, and image evidence."""
    points = np.asarray(path_zyx, float)
    if points.ndim != 2 or points.shape[1:] != (3,) or len(points) < 2:
        return PathVerification(False, False, False, False, 0, 0, 0, ("invalid_path",))
    indices = np.clip(np.rint(points).astype(int), 0, np.asarray(geometry.mask.shape) - 1)
    wall_distance = ndimage.distance_transform_edt(
        ~geometry.lateral_wall, sampling=_sampling(geometry.spacing_xyz)
    )
    contact = bool(wall_distance[tuple(indices[0])] <= 1.75 * max(_sampling(geometry.spacing_xyz)))
    xyz = np.asarray(
        [voxel_to_physical(p, geometry.spacing_xyz, origin_xyz, direction) for p in points]
    )
    length = physical_path_length(xyz)
    normal_zyx, _ = robust_wall_normal(
        geometry.signed_distance_mm, points[0], geometry.spacing_xyz, min_coherence=0.15
    )
    if normal_zyx is None:
        displacement = float(wall_distance[tuple(indices[-1])])
        radial = wall_distance[tuple(indices)]
    else:
        physical_zyx = points * _sampling(geometry.spacing_xyz)
        radial = (physical_zyx - physical_zyx[0]) @ normal_zyx
        displacement = float(radial[-1])
    increments = np.diff(radial)
    outward = bool(
        displacement >= min(3.0, 0.5 * min_length_mm)
        and np.count_nonzero(increments >= -0.75) >= 0.75 * len(increments)
        and wall_distance[tuple(indices[-1])] > wall_distance[tuple(indices[0])]
    )
    evidence = np.zeros(len(indices), dtype=float)
    supplied = 0
    if blood_similarity_zyx is not None:
        evidence = np.maximum(evidence, np.asarray(blood_similarity_zyx)[tuple(indices.T)])
        supplied += 1
    if tubularity_zyx is not None:
        evidence = np.maximum(evidence, np.asarray(tubularity_zyx)[tuple(indices.T)])
        supplied += 1
    # Exclude the wall sample, where partial-volume support is expected to be low.
    support = float(np.mean(evidence[1:] >= 0.18)) if supplied else 1.0
    image_ok = bool(support >= min_support_fraction)
    reasons = []
    if not contact:
        reasons.append("no_direct_aorta_contact")
    if length < min_length_mm:
        reasons.append("path_too_short")
    if not outward:
        reasons.append("not_outward")
    if not image_ok:
        reasons.append("insufficient_image_support")
    return PathVerification(
        contact, length >= min_length_mm, outward, image_ok, length, support,
        displacement, tuple(reasons)
    )


def _roi(
    candidate: Candidate,
    geometry: AortaGeometry,
    margin_mm: float,
    origin_xyz: Sequence[float],
    direction: Sequence[float],
) -> tuple[slice, ...]:
    sampling = _sampling(geometry.spacing_xyz)
    points = np.asarray(
        [physical_to_voxel(candidate.ostium_xyz, geometry.spacing_xyz, origin_xyz, direction),
         physical_to_voxel(candidate.seed_xyz, geometry.spacing_xyz, origin_xyz, direction)]
    )
    if candidate.path_xyz:
        points = np.vstack([
            points,
            [physical_to_voxel(p, geometry.spacing_xyz, origin_xyz, direction)
             for p in candidate.path_xyz],
        ])
    pad = np.ceil(margin_mm / sampling).astype(int)
    lo = np.maximum(np.floor(points.min(axis=0)).astype(int) - pad, 0)
    hi = np.minimum(np.ceil(points.max(axis=0)).astype(int) + pad + 1, geometry.mask.shape)
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))


def _skeleton_path(mask: np.ndarray, start: np.ndarray, sampling: np.ndarray) -> np.ndarray | None:
    """Deterministic skeleton graph path used only by the segmentation method."""
    try:
        from skimage.morphology import skeletonize  # type: ignore[import-not-found]

        skeleton = skeletonize(mask)
    except ImportError:
        distance = ndimage.distance_transform_edt(mask, sampling=sampling)
        ridge = distance >= ndimage.maximum_filter(distance, size=3) - 1e-6
        skeleton = mask & ndimage.binary_dilation(ridge, iterations=1)
    # Keep exact start connected to the thinned centreline.
    nodes = np.argwhere(skeleton)
    if not len(nodes):
        return None
    start_node = int(np.argmin(np.linalg.norm((nodes - start) * sampling, axis=1)))
    lookup = {tuple(point): i for i, point in enumerate(nodes)}
    rows: list[int] = []
    cols: list[int] = []
    weights: list[float] = []
    for i, point in enumerate(nodes):
        for offset in _OFFSETS:
            j = lookup.get(tuple(point + offset))
            if j is not None:
                rows.append(i)
                cols.append(j)
                weights.append(float(np.linalg.norm(offset * sampling)))
    graph = sparse.csr_matrix((weights, (rows, cols)), shape=(len(nodes), len(nodes)))
    distances, predecessors = dijkstra(
        graph, directed=False, indices=start_node, return_predecessors=True
    )
    finite = np.flatnonzero(np.isfinite(distances))
    if not len(finite):
        return None
    endpoint = int(finite[np.argmax(distances[finite])])
    order = [endpoint]
    while order[-1] != start_node:
        previous = int(predecessors[order[-1]])
        if previous < 0:
            return None
        order.append(previous)
    return nodes[np.asarray(order[::-1])]


def track_local_segmentation(
    intensity_zyx: np.ndarray,
    geometry: AortaGeometry,
    candidate: Candidate,
    *,
    blood_similarity_zyx: np.ndarray | None = None,
    tubularity_zyx: np.ndarray | None = None,
    origin_xyz: Sequence[float] = (0, 0, 0),
    direction: Sequence[float] = (1, 0, 0, 0, 1, 0, 0, 0, 1),
    roi_margin_mm: float = 12.0,
    use_kimimaro: bool = True,
) -> VerifiedPath | None:
    """Track by local adaptive hysteresis segmentation and centreline thinning."""
    image = np.asarray(intensity_zyx, np.float32)
    if image.shape != geometry.mask.shape:
        raise ValueError("intensity and geometry shapes must match")
    roi = _roi(candidate, geometry, roi_margin_mm, origin_xyz, direction)
    local = image[roi]
    similarity = (
        np.asarray(blood_similarity_zyx, np.float32)[roi]
        if blood_similarity_zyx is not None
        else np.exp(-0.5 * ((local - (geometry.blood.median if geometry.blood else np.median(local)))
                           / (geometry.blood.robust_sigma if geometry.blood else max(np.std(local), 1))) ** 2)
    )
    tube = (
        np.asarray(tubularity_zyx, np.float32)[roi]
        if tubularity_zyx is not None else np.zeros_like(local)
    )
    low = ((similarity >= 0.20) | (tube >= 0.12)) & ~geometry.mask[roi]
    high = ((similarity >= 0.58) | (tube >= 0.35)) & low
    labels, count = ndimage.label(low, ndimage.generate_binary_structure(3, 2))
    seed_global = np.rint(
        physical_to_voxel(candidate.seed_xyz, geometry.spacing_xyz, origin_xyz, direction)
    ).astype(int)
    starts = np.asarray([sl.start for sl in roi])
    seed = np.clip(seed_global - starts, 0, np.asarray(local.shape) - 1)
    selected = int(labels[tuple(seed)])
    if selected == 0:
        high_labels = np.unique(labels[high])
        high_labels = high_labels[high_labels > 0]
        if len(high_labels):
            centres = np.asarray([np.mean(np.argwhere(labels == value), axis=0) for value in high_labels])
            selected = int(high_labels[np.argmin(np.linalg.norm((centres - seed) * _sampling(geometry.spacing_xyz), axis=1))])
    if selected == 0 or selected > count:
        return None
    segment = labels == selected
    if not np.any(segment & high):
        return None
    wall_distance = ndimage.distance_transform_edt(
        ~geometry.lateral_wall[roi], sampling=_sampling(geometry.spacing_xyz)
    )
    if not np.any(segment & (wall_distance <= 1.8 * max(_sampling(geometry.spacing_xyz)))):
        return None
    ostium_global = np.rint(
        physical_to_voxel(candidate.ostium_xyz, geometry.spacing_xyz, origin_xyz, direction)
    ).astype(int)
    ostium = np.clip(ostium_global - starts, 0, np.asarray(local.shape) - 1)
    path_local: np.ndarray | None = None
    used_kimimaro = False
    if use_kimimaro:
        try:
            import kimimaro  # type: ignore[import-not-found]

            skeletons = kimimaro.skeletonize(segment.astype(np.uint8), teasar_params={
                "scale": 1.0, "const": 4.0, "pdrf_exponent": 4, "pdrf_scale": 100000,
                "soma_detection_threshold": 1100, "soma_acceptance_threshold": 3500,
                "soma_invalidation_scale": 1.0, "soma_invalidation_const": 300,
                "max_paths": 50,
            }, anisotropy=tuple(float(v) for v in geometry.spacing_xyz))
            vertices = [s.vertices[:, ::-1] / _sampling(geometry.spacing_xyz) for s in skeletons.values()
                        if len(s.vertices) >= 2]
            if vertices:
                path_local = max(vertices, key=len)
                used_kimimaro = True
        except (ImportError, OSError, RuntimeError, ValueError):
            pass
    if path_local is None:
        path_local = _skeleton_path(segment, ostium, _sampling(geometry.spacing_xyz))
    if path_local is None or len(path_local) < 2:
        return None
    # Put the actual wall contact first and orient away from it.
    if np.linalg.norm((path_local[-1] - ostium) * _sampling(geometry.spacing_xyz)) < np.linalg.norm(
        (path_local[0] - ostium) * _sampling(geometry.spacing_xyz)
    ):
        path_local = path_local[::-1]
    path_global = np.vstack([ostium_global, path_local + starts])
    path_global = path_global[np.r_[True, np.any(np.diff(path_global, axis=0), axis=1)]]
    verification = verify_path(
        path_global, geometry, blood_similarity_zyx=blood_similarity_zyx,
        tubularity_zyx=tubularity_zyx, origin_xyz=origin_xyz, direction=direction,
    )
    if not verification.passed:
        return None
    voxel, physical = _ordered_physical(path_global, geometry.spacing_xyz, origin_xyz, direction)
    full_segment = np.zeros_like(geometry.mask)
    full_segment[roi] = segment
    return VerifiedPath(
        "local_hysteresis_teasar" if used_kimimaro else "local_hysteresis_skeleton",
        candidate, voxel, physical, verification, full_segment,
        {"kimimaro": used_kimimaro},
    )


def track_minimal_path(
    geometry: AortaGeometry,
    candidate: Candidate,
    blood_similarity_zyx: np.ndarray,
    tubularity_zyx: np.ndarray,
    *,
    origin_xyz: Sequence[float] = (0, 0, 0),
    direction: Sequence[float] = (1, 0, 0, 0, 1, 0, 0, 0, 1),
    roi_margin_mm: float = 15.0,
    max_length_mm: float = 45.0,
    max_expansions: int = 200_000,
) -> VerifiedPath | None:
    """Track independently with A* through a bounded continuous cost field."""
    blood = np.asarray(blood_similarity_zyx, np.float32)
    tube = np.asarray(tubularity_zyx, np.float32)
    if blood.shape != geometry.mask.shape or tube.shape != geometry.mask.shape:
        raise ValueError("feature arrays must match geometry")
    if not np.isfinite(max_length_mm) or max_length_mm <= 0:
        raise ValueError("max_length_mm must be finite and positive")
    if isinstance(max_expansions, bool) or not isinstance(max_expansions, int) or max_expansions < 1:
        raise ValueError("max_expansions must be a positive integer")
    roi = _roi(candidate, geometry, roi_margin_mm, origin_xyz, direction)
    starts = np.asarray([sl.start for sl in roi])
    shape = np.asarray(blood[roi].shape)
    sampling = _sampling(geometry.spacing_xyz)
    start_global = np.rint(
        physical_to_voxel(candidate.ostium_xyz, geometry.spacing_xyz, origin_xyz, direction)
    ).astype(int)
    seed_global = np.rint(
        physical_to_voxel(candidate.seed_xyz, geometry.spacing_xyz, origin_xyz, direction)
    ).astype(int)
    start = np.clip(start_global - starts, 0, shape - 1)
    target_hint = np.clip(seed_global - starts, 0, shape - 1)
    outside_distance = ndimage.distance_transform_edt(~geometry.mask[roi], sampling=sampling)
    centrality = np.clip(outside_distance / max(candidate.radius_mm, 0.5), 0, 1)
    allowed = (
        (~geometry.mask[roi])
        & ((blood[roi] >= 0.10) | (tube[roi] >= 0.08))
        & (outside_distance <= max_length_mm)
    )
    allowed[tuple(start)] = True
    # Goal is evidence-supported, outward, and at least 5 mm from the wall.
    direction_zyx = (
        np.asarray(direction, float).reshape(3, 3).T @ np.asarray(candidate.direction_xyz, float)
    )[::-1]
    direction_zyx /= max(np.linalg.norm(direction_zyx), 1e-9)
    grid = np.indices(tuple(shape)).reshape(3, -1).T
    progression = ((grid - start) * sampling) @ direction_zyx
    goal_score = (
        0.45 * blood[roi].ravel() + 0.35 * tube[roi].ravel()
        + 0.20 * centrality.ravel() + 0.01 * progression
    )
    valid_goals = allowed.ravel() & (outside_distance.ravel() >= 5.0) & (progression >= 5.0)
    if not np.any(valid_goals):
        return None
    goal_candidates = np.flatnonzero(valid_goals)
    hint_distance = np.linalg.norm((grid[goal_candidates] - target_hint) * sampling, axis=1)
    utility = goal_score[goal_candidates] + np.clip(hint_distance, 0, 20) * 0.01
    goal = grid[int(goal_candidates[np.argmax(utility)])]
    cost = (
        0.46 * (1 - blood[roi]) + 0.31 * (1 - tube[roi])
        + 0.18 * (1 - centrality)
    )
    forbidden = geometry.mask[roi].copy()
    forbidden[tuple(start)] = False
    cost[forbidden | ~allowed] = np.inf
    start_key, goal_key = tuple(start), tuple(goal)
    queue: list[tuple[float, float, float, tuple[int, int, int]]] = [
        (0.0, 0.0, 0.0, start_key)
    ]
    previous: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    best = {start_key: 0.0}
    expansions = 0
    while queue:
        _, accumulated_cost, physical_length, current = heapq.heappop(queue)
        if accumulated_cost != best.get(current):
            continue
        expansions += 1
        if expansions > max_expansions:
            return None
        if current == goal_key:
            break
        point = np.asarray(current)
        for offset in _OFFSETS:
            nxt = point + offset
            if np.any(nxt < 0) or np.any(nxt >= shape):
                continue
            key = tuple(int(v) for v in nxt)
            if not np.isfinite(cost[key]):
                continue
            step_vector = offset * sampling
            step = float(np.linalg.norm(step_vector))
            next_length = physical_length + step
            if next_length > max_length_mm:
                continue
            turn_penalty = max(0.0, -float(np.dot(step_vector / step, direction_zyx)))
            value = accumulated_cost + step * (
                0.1 + float(cost[key]) + 0.25 * turn_penalty
            )
            if value < best.get(key, np.inf):
                best[key] = value
                previous[key] = current
                heuristic = float(np.linalg.norm((nxt - goal) * sampling)) * 0.1
                heapq.heappush(queue, (value + heuristic, value, next_length, key))
    if goal_key not in best:
        return None
    path = [goal_key]
    while path[-1] != start_key:
        path.append(previous[path[-1]])
    path_global = np.asarray(path[::-1], dtype=float) + starts
    physical_length = physical_path_length([
        voxel_to_physical(point, geometry.spacing_xyz, origin_xyz, direction)
        for point in path_global
    ])
    if physical_length > max_length_mm + 1e-6:
        return None
    verification = verify_path(
        path_global, geometry, blood_similarity_zyx=blood,
        tubularity_zyx=tube, origin_xyz=origin_xyz, direction=direction,
    )
    if not verification.passed:
        return None
    voxel, physical = _ordered_physical(path_global, geometry.spacing_xyz, origin_xyz, direction)
    local_path = np.rint(path_global - starts).astype(int)
    image_supported = (
        (~geometry.mask[roi])
        & ((blood[roi] >= 0.18) | (tube[roi] >= 0.12))
    )
    labels, count = ndimage.label(
        image_supported, ndimage.generate_binary_structure(3, 2)
    )
    path_labels = labels[tuple(local_path.T)]
    positive_labels = path_labels[path_labels > 0]
    vessel_mask = None
    selected_label = 0
    if len(positive_labels):
        selected_label = int(np.bincount(positive_labels).argmax())
        component = labels == selected_label
        support_fraction = float(np.mean(path_labels[1:] == selected_label))
        if (
            count
            and component.sum() >= max(8, len(local_path) // 2)
            and support_fraction >= 0.55
        ):
            vessel_mask = np.zeros_like(geometry.mask)
            vessel_mask[roi] = component
    return VerifiedPath(
        "targeted_continuous_astar",
        candidate,
        voxel,
        physical,
        verification,
        vessel_mask,
        {
            "search_cost": float(best[goal_key]),
            "bounded_roi": True,
            "node_expansions": expansions,
            "physical_length_mm": physical_length,
            "vessel_mask_source": (
                "local_feature_component" if vessel_mask is not None else "unavailable"
            ),
            "selected_component": selected_label,
        },
    )


# Explicit aliases make the two public strategies easy to discover.
track_candidate_local = track_local_segmentation
track_candidate_minimal_path = track_minimal_path

