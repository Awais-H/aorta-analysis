"""Physical seed, direction, and radius measurements for verified instances."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import ndimage

from .instances import VerifiedInstance
from .models import DaughterPrediction
from .tracking import Point3, physical_to_voxel


@dataclass(frozen=True, slots=True)
class InstanceMeasurement:
    ostium_xyz: Point3
    seed_xyz: Point3
    direction_xyz: Point3
    seed_arc_length_mm: float
    radius_3d_edt_mm: float
    radius_2d_inscribed_mm: float
    radius_2d_area_equivalent_mm: float
    radius_mm: float
    quality_flags: tuple[str, ...] = ()


def _arc(path_xyz: Sequence[Sequence[float]]) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(path_xyz, dtype=float)
    if points.ndim != 2 or points.shape[1:] != (3,) or len(points) < 2:
        raise ValueError("path must contain at least two physical 3-D points")
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return points, np.r_[0.0, np.cumsum(lengths)]


def interpolate_at_arc_length(
    path_xyz: Sequence[Sequence[float]], distance_mm: float
) -> tuple[np.ndarray, int, float]:
    """Interpolate at an exact physical polyline arc length."""
    if distance_mm < 0:
        raise ValueError("distance_mm must be non-negative")
    points, arc = _arc(path_xyz)
    if distance_mm > arc[-1] + 1e-8:
        raise ValueError("requested arc length exceeds path length")
    if distance_mm >= arc[-1]:
        return points[-1].copy(), len(points) - 2, 1.0
    segment = int(np.searchsorted(arc, distance_mm, side="right") - 1)
    segment = max(0, min(segment, len(points) - 2))
    denominator = arc[segment + 1] - arc[segment]
    fraction = 0.0 if denominator <= 1e-12 else (distance_mm - arc[segment]) / denominator
    return points[segment] + fraction * (points[segment + 1] - points[segment]), segment, float(fraction)


def anchor_path_at_ostium(
    path_xyz: Sequence[Sequence[float]],
    ostium_xyz: Sequence[float],
    *,
    search_prefix_mm: float = 10.0,
) -> np.ndarray:
    """Project the final contact centroid onto the proximal path and prepend it."""
    points, arc = _arc(path_xyz)
    ostium = np.asarray(ostium_xyz, float)
    best_distance = np.inf
    best_segment = 0
    best_projection = points[0]
    for index, (first, second) in enumerate(zip(points[:-1], points[1:])):
        if arc[index] > search_prefix_mm:
            break
        delta = second - first
        denominator = float(np.dot(delta, delta))
        fraction = 0.0 if denominator <= 1e-12 else float(
            np.clip(np.dot(ostium - first, delta) / denominator, 0, 1)
        )
        projection = first + fraction * delta
        distance = float(np.linalg.norm(projection - ostium))
        if distance < best_distance:
            best_distance = distance
            best_segment = index
            best_projection = projection
    anchored = np.vstack([ostium, best_projection, points[best_segment + 1 :]])
    return anchored[np.r_[True, np.linalg.norm(np.diff(anchored, axis=0), axis=1) > 1e-9]]


def robust_direction(
    path_xyz: Sequence[Sequence[float]],
    at_mm: float = 5.0,
    *,
    window_mm: float = 3.0,
) -> np.ndarray:
    """Robust local line fit, oriented from ostium toward the daughter."""
    points, arc = _arc(path_xyz)
    if not 0 <= at_mm <= arc[-1]:
        raise ValueError("at_mm must lie on the path")
    local = points[np.abs(arc - at_mm) <= window_mm]
    if len(local) < 3:
        lo = max(0, np.searchsorted(arc, at_mm) - 2)
        hi = min(len(points), lo + 4)
        local = points[lo:hi]
    centre = np.median(local, axis=0)
    retained = local
    for _ in range(2):
        _, _, vh = np.linalg.svd(retained - centre, full_matrices=False)
        axis = vh[0]
        residual = np.linalg.norm(
            (retained - centre) - ((retained - centre) @ axis)[:, None] * axis, axis=1
        )
        cutoff = np.median(residual) + 3 * max(
            np.median(np.abs(residual - np.median(residual))), 1e-3
        )
        kept = retained[residual <= cutoff]
        if len(kept) < 2 or len(kept) == len(retained):
            break
        retained = kept
        centre = np.median(retained, axis=0)
    _, _, vh = np.linalg.svd(retained - centre, full_matrices=False)
    axis = vh[0]
    before, _, _ = interpolate_at_arc_length(points, max(0.0, at_mm - min(1.0, at_mm)))
    after, _, _ = interpolate_at_arc_length(points, min(float(arc[-1]), at_mm + 1.0))
    if np.dot(axis, after - before) < 0:
        axis = -axis
    return axis / max(np.linalg.norm(axis), 1e-12)


def _plane_basis(direction_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    direction_xyz = direction_xyz / np.linalg.norm(direction_xyz)
    reference = np.asarray((1.0, 0.0, 0.0))
    if abs(np.dot(reference, direction_xyz)) > 0.85:
        reference = np.asarray((0.0, 1.0, 0.0))
    first = np.cross(direction_xyz, reference)
    first /= np.linalg.norm(first)
    second = np.cross(direction_xyz, first)
    return first, second / np.linalg.norm(second)


def estimate_radii(
    vessel_mask_zyx: np.ndarray,
    seed_xyz: Sequence[float],
    direction_xyz: Sequence[float],
    spacing_xyz: Sequence[float],
    *,
    origin_xyz: Sequence[float] = (0, 0, 0),
    image_direction: Sequence[float] = (1, 0, 0, 0, 1, 0, 0, 0, 1),
    plane_half_width_mm: float = 12.0,
    plane_resolution_mm: float | None = None,
) -> tuple[float, float, float]:
    """Estimate 3-D EDT, orthogonal inscribed, and area-equivalent radii."""
    mask = np.asarray(vessel_mask_zyx, bool)
    spacing = np.asarray(tuple(spacing_xyz), float)
    if mask.ndim != 3 or spacing.shape != (3,) or np.any(spacing <= 0):
        raise ValueError("mask and spacing must describe a 3-D physical image")
    seed_voxel = physical_to_voxel(seed_xyz, spacing, origin_xyz, image_direction)
    edt3 = ndimage.distance_transform_edt(mask, sampling=spacing[::-1])
    radius3 = float(ndimage.map_coordinates(
        edt3, np.asarray(seed_voxel)[:, None], order=1, mode="constant", cval=0
    )[0])
    resolution = float(plane_resolution_mm or min(0.5, spacing.min() / 2))
    coordinates = np.arange(-plane_half_width_mm, plane_half_width_mm + resolution / 2, resolution)
    uu, vv = np.meshgrid(coordinates, coordinates, indexing="ij")
    first, second = _plane_basis(np.asarray(direction_xyz, float))
    physical = (
        np.asarray(seed_xyz, float)[None, None, :]
        + uu[..., None] * first + vv[..., None] * second
    )
    rotation = np.asarray(image_direction, float).reshape(3, 3)
    local_xyz = (physical - np.asarray(origin_xyz, float)) @ rotation
    voxel = (local_xyz / spacing)[..., ::-1]
    section = ndimage.map_coordinates(
        mask.astype(np.float32), np.moveaxis(voxel, -1, 0), order=1,
        mode="constant", cval=0,
    ) >= 0.5
    centre = np.asarray(section.shape) // 2
    labels, _ = ndimage.label(section, ndimage.generate_binary_structure(2, 2))
    label = int(labels[tuple(centre)])
    if label == 0:
        nearby = np.argwhere(section)
        if not len(nearby):
            return radius3, 0.0, 0.0
        nearest = nearby[np.argmin(np.linalg.norm(nearby - centre, axis=1))]
        label = int(labels[tuple(nearest)])
    component = labels == label
    edt2 = ndimage.distance_transform_edt(component, sampling=(resolution, resolution))
    inscribed = float(edt2.max())
    area_radius = float(np.sqrt(component.sum() * resolution**2 / np.pi))
    return radius3, inscribed, area_radius


def measure_instance(
    instance: VerifiedInstance,
    spacing_xyz: Sequence[float],
    *,
    seed_distance_mm: float = 5.0,
    origin_xyz: Sequence[float] = (0, 0, 0),
    image_direction: Sequence[float] = (1, 0, 0, 0, 1, 0, 0, 0, 1),
    disagreement_fraction: float = 0.35,
) -> InstanceMeasurement:
    """Measure a verified instance exactly 5 mm along its physical path."""
    if instance.vessel_mask_zyx is None:
        raise ValueError("instance requires a vessel mask for radius measurement")
    path = max(instance.paths, key=lambda item: item.length_mm)
    anchored_path = anchor_path_at_ostium(path.path_xyz, instance.ostium_xyz)
    _, arc = _arc(anchored_path)
    if arc[-1] < seed_distance_mm:
        raise ValueError("verified path is shorter than requested seed distance")
    seed, _, _ = interpolate_at_arc_length(anchored_path, seed_distance_mm)
    direction = robust_direction(anchored_path, seed_distance_mm)
    radius3, inscribed, area = estimate_radii(
        instance.vessel_mask_zyx, seed, direction, spacing_xyz,
        origin_xyz=origin_xyz, image_direction=image_direction,
    )
    radii = np.asarray([radius3, inscribed, area])
    positive = radii[radii > 0]
    radius = float(np.median(positive)) if len(positive) else 0.0
    flags: list[str] = []
    if len(positive) < 3:
        flags.append("incomplete_radius_support")
    if len(positive) >= 2 and (positive.max() - positive.min()) / max(radius, 1e-6) > disagreement_fraction:
        flags.append("radius_disagreement")
    if radius <= 0:
        flags.append("invalid_radius")
    if len(instance.paths) == 1:
        flags.append("single_method_support")
    return InstanceMeasurement(
        instance.ostium_xyz,
        tuple(float(v) for v in seed),
        tuple(float(v) for v in direction),
        float(seed_distance_mm),
        radius3,
        inscribed,
        area,
        radius,
        tuple(flags),
    )


def instance_to_prediction(
    instance: VerifiedInstance,
    spacing_xyz: Sequence[float],
    *,
    origin_xyz: Sequence[float] = (0, 0, 0),
    image_direction: Sequence[float] = (1, 0, 0, 0, 1, 0, 0, 0, 1),
    label: str | None = None,
    seed_distance_mm: float = 5.0,
) -> DaughterPrediction:
    measurement = measure_instance(
        instance,
        spacing_xyz,
        seed_distance_mm=seed_distance_mm,
        origin_xyz=origin_xyz,
        image_direction=image_direction,
    )
    score = float(max(path.candidate.score for path in instance.paths))
    representative = max(instance.paths, key=lambda item: item.length_mm)
    anchored_path = anchor_path_at_ostium(
        representative.path_xyz, measurement.ostium_xyz
    )
    return DaughterPrediction(
        ostium_xyz=measurement.ostium_xyz,
        seed_xyz=measurement.seed_xyz,
        direction_xyz=measurement.direction_xyz,
        radius_mm=measurement.radius_mm,
        label=label,
        score=score,
        candidate_path_xyz=tuple(tuple(float(v) for v in point) for point in anchored_path),
        metadata={
            "tracking_methods": sorted({path.method for path in instance.paths}),
            "seed_arc_length_mm": measurement.seed_arc_length_mm,
            "radius_3d_edt_mm": measurement.radius_3d_edt_mm,
            "radius_2d_inscribed_mm": measurement.radius_2d_inscribed_mm,
            "radius_2d_area_equivalent_mm": measurement.radius_2d_area_equivalent_mm,
            "quality_flags": list(measurement.quality_flags),
        },
    )


def instances_to_predictions(
    instances: Sequence[VerifiedInstance],
    spacing_xyz: Sequence[float],
    **kwargs: object,
) -> tuple[DaughterPrediction, ...]:
    return tuple(instance_to_prediction(instance, spacing_xyz, **kwargs) for instance in instances)


# More explicit synonym for downstream callers.
convert_verified_instances = instances_to_predictions

