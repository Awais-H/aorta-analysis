"""Spacing-aware preparation of an aortic lumen mask.

NumPy arrays in this module are always indexed ``(z, y, x)``.  Physical
points and vectors are returned in SimpleITK's ``(x, y, z)`` convention.
The signed distance returned here is *positive inside* the cleaned lumen and
negative outside; it is therefore the negative of SciPy's usual level-set
sign convention.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import ndimage


def _spacing_zyx(spacing_xyz: Sequence[float]) -> tuple[float, float, float]:
    spacing = tuple(float(v) for v in spacing_xyz)
    if len(spacing) != 3 or any(not np.isfinite(v) or v <= 0 for v in spacing):
        raise ValueError("spacing_xyz must contain three positive finite values")
    return spacing[2], spacing[1], spacing[0]


def _ball(radius_mm: float, spacing_xyz: Sequence[float]) -> np.ndarray:
    sz, sy, sx = _spacing_zyx(spacing_xyz)
    rz, ry, rx = (int(np.ceil(radius_mm / s)) for s in (sz, sy, sx))
    z, y, x = np.ogrid[-rz : rz + 1, -ry : ry + 1, -rx : rx + 1]
    return (z * sz) ** 2 + (y * sy) ** 2 + (x * sx) ** 2 <= radius_mm**2


def _largest_component(mask: np.ndarray) -> np.ndarray:
    labels, count = ndimage.label(mask, structure=ndimage.generate_binary_structure(3, 2))
    if not count:
        return np.zeros_like(mask, dtype=bool)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels == int(np.argmax(sizes))


@dataclass(frozen=True, slots=True)
class BloodModel:
    """Robust patient-specific blood intensity summary."""

    median: float
    mad: float
    low: float
    high: float
    sample_count: int

    @property
    def robust_sigma(self) -> float:
        return max(1.4826 * self.mad, (self.high - self.low) / 3.29, 1.0)


@dataclass(frozen=True, slots=True)
class PhysicalBoundingBox:
    """Inclusive voxel bounds and their eight-corner physical envelope."""

    start_zyx: tuple[int, int, int]
    stop_zyx: tuple[int, int, int]
    minimum_xyz: tuple[float, float, float]
    maximum_xyz: tuple[float, float, float]

    @property
    def slices_zyx(self) -> tuple[slice, slice, slice]:
        return tuple(slice(a, b) for a, b in zip(self.start_zyx, self.stop_zyx))  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class AortaGeometry:
    mask: np.ndarray
    signed_distance_mm: np.ndarray
    interior_wall: np.ndarray
    lateral_wall: np.ndarray
    end_caps: np.ndarray
    blood: BloodModel | None
    bounding_box: PhysicalBoundingBox | None
    centreline_xyz: np.ndarray
    tangents_xyz: np.ndarray
    spacing_xyz: tuple[float, float, float]


def clean_aorta_mask(
    mask_zyx: np.ndarray,
    spacing_xyz: Sequence[float],
    *,
    fill_holes_mm: float = 2.0,
    min_volume_mm3: float = 20.0,
) -> np.ndarray:
    """Validate and conservatively clean the largest connected component."""
    mask = np.asarray(mask_zyx)
    if mask.ndim != 3:
        raise ValueError("mask_zyx must be three-dimensional")
    if not np.issubdtype(mask.dtype, np.bool_) and not np.all(np.isin(mask, (0, 1))):
        raise ValueError("aorta mask must be binary")
    spacing = tuple(float(v) for v in spacing_xyz)
    _spacing_zyx(spacing)
    result = _largest_component(mask.astype(bool))
    if not result.any():
        return result
    if result.sum() * np.prod(spacing) < min_volume_mm3:
        return np.zeros_like(result)
    if fill_holes_mm > 0:
        # A closing only bridges defects smaller than the configured physical scale.
        footprint = _ball(fill_holes_mm, spacing)
        closed = ndimage.binary_closing(result, structure=footprint)
        result |= closed & ndimage.binary_fill_holes(result)
    return result


def signed_distance_mm(mask_zyx: np.ndarray, spacing_xyz: Sequence[float]) -> np.ndarray:
    """Return signed Euclidean distance in mm, positive inside the mask."""
    mask = np.asarray(mask_zyx, dtype=bool)
    if mask.ndim != 3:
        raise ValueError("mask_zyx must be three-dimensional")
    sampling = _spacing_zyx(spacing_xyz)
    # Padding makes the image crop boundary an explicit exterior. Without it,
    # SciPy can overestimate distance for a lumen truncated by the image crop.
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    inside = ndimage.distance_transform_edt(padded, sampling=sampling)[1:-1, 1:-1, 1:-1]
    outside = ndimage.distance_transform_edt(~padded, sampling=sampling)[1:-1, 1:-1, 1:-1]
    return (inside - outside).astype(np.float32)


def derive_walls(
    mask_zyx: np.ndarray,
    spacing_xyz: Sequence[float],
    *,
    thickness_mm: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the inner wall shell and, initially, all boundary as lateral wall."""
    if thickness_mm <= 0:
        raise ValueError("thickness_mm must be positive")
    sdf = signed_distance_mm(mask_zyx, spacing_xyz)
    interior = np.asarray(mask_zyx, bool) & (sdf <= thickness_mm)
    return interior, interior.copy()


def _physical_points(
    indices_zyx: np.ndarray,
    spacing_xyz: Sequence[float],
    origin_xyz: Sequence[float],
    direction: Sequence[float],
) -> np.ndarray:
    if len(origin_xyz) != 3 or len(direction) != 9:
        raise ValueError("origin_xyz and direction must describe 3-D geometry")
    xyz_index = indices_zyx[:, ::-1].astype(float)
    scaled = xyz_index * np.asarray(spacing_xyz, dtype=float)
    return np.asarray(origin_xyz, float) + scaled @ np.asarray(direction, float).reshape(3, 3).T


def detect_end_caps(
    mask_zyx: np.ndarray,
    spacing_xyz: Sequence[float],
    *,
    origin_xyz: Sequence[float] = (0.0, 0.0, 0.0),
    direction: Sequence[float] = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
    cap_depth_mm: float = 3.0,
) -> np.ndarray:
    """Detect flat terminal surfaces using local geometry, never a global axis.

    Both image-face truncations and flat supplied-mask ends inside the image are
    covered.  Candidate terminal centres come from the two ends of the mask's
    principal physical extent; local PCA supplies the inward direction.
    """
    mask = np.asarray(mask_zyx, dtype=bool)
    caps = np.zeros_like(mask)
    if not mask.any():
        return caps
    line, _ = coarse_centreline(
        mask,
        spacing_xyz,
        origin_xyz=origin_xyz,
        direction=direction,
        step_mm=max(2.0, cap_depth_mm),
    )
    if len(line) < 2:
        return caps
    sdf = signed_distance_mm(mask, spacing_xyz)
    sampling = np.asarray(_spacing_zyx(spacing_xyz))
    boundary = mask & (sdf <= max(sampling) * 1.75)
    boundary_indices = np.argwhere(boundary)
    boundary_points = _physical_points(
        boundary_indices, spacing_xyz, origin_xyz, direction
    )
    gradients_zyx = np.stack(
        np.gradient(ndimage.gaussian_filter(sdf, 0.6 / sampling), *sampling),
        axis=-1,
    )
    gradient_xyz = (
        gradients_zyx[tuple(boundary_indices.T)][:, ::-1]
        @ np.asarray(direction, float).reshape(3, 3).T
    )
    gradient_xyz /= np.maximum(np.linalg.norm(gradient_xyz, axis=1, keepdims=True), 1e-8)
    all_indices = np.argwhere(mask)
    all_points = _physical_points(all_indices, spacing_xyz, origin_xyz, direction)
    for endpoint, neighbour in ((line[0], line[1]), (line[-1], line[-2])):
        inward = np.asarray(neighbour - endpoint, float)
        inward /= max(float(np.linalg.norm(inward)), 1e-8)
        nearby = np.linalg.norm(all_points - endpoint, axis=1) <= 12.0
        local_radius = max(
            float(np.max(sdf[tuple(all_indices[nearby].T)])) if np.any(nearby) else 0.0,
            float(max(sampling)),
        )
        relative = boundary_points - endpoint
        axial = relative @ inward
        radial = np.linalg.norm(relative - axial[:, None] * inward, axis=1)
        terminal_candidates = radial <= 1.35 * local_radius
        if not np.any(terminal_candidates):
            continue
        plane = float(np.min(axial[terminal_candidates]))
        orientation = gradient_xyz @ inward
        selected = (
            (axial >= plane - max(sampling))
            & (axial <= plane + cap_depth_mm)
            & (radial <= 1.35 * local_radius)
            & (orientation >= 0.55)
        )
        caps[tuple(boundary_indices[selected].T)] = True
    return caps


def estimate_blood_model(
    intensity_zyx: np.ndarray,
    aorta_mask_zyx: np.ndarray,
    spacing_xyz: Sequence[float],
    *,
    erosion_mm: float = 2.5,
) -> BloodModel | None:
    """Estimate blood statistics from a physically eroded aortic core."""
    values = np.asarray(intensity_zyx, dtype=np.float32)
    mask = np.asarray(aorta_mask_zyx, dtype=bool)
    if values.shape != mask.shape:
        raise ValueError("intensity and mask shapes must match")
    core = mask & (signed_distance_mm(mask, spacing_xyz) >= erosion_mm)
    samples = values[core & np.isfinite(values)]
    if samples.size < 16:
        samples = values[mask & np.isfinite(values)]
    if samples.size == 0:
        return None
    median = float(np.median(samples))
    return BloodModel(
        median=median,
        mad=float(np.median(np.abs(samples - median))),
        low=float(np.percentile(samples, 5)),
        high=float(np.percentile(samples, 95)),
        sample_count=int(samples.size),
    )


def physical_bounding_box(
    mask_zyx: np.ndarray,
    spacing_xyz: Sequence[float],
    *,
    origin_xyz: Sequence[float] = (0.0, 0.0, 0.0),
    direction: Sequence[float] = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
    margin_mm: float = 0.0,
) -> PhysicalBoundingBox | None:
    mask = np.asarray(mask_zyx, bool)
    indices = np.argwhere(mask)
    if not len(indices):
        return None
    pad = np.ceil(margin_mm / np.asarray(_spacing_zyx(spacing_xyz))).astype(int)
    start = np.maximum(indices.min(axis=0) - pad, 0)
    stop = np.minimum(indices.max(axis=0) + pad + 1, mask.shape)
    corners = np.array(np.meshgrid(*zip(start, stop - 1), indexing="ij")).reshape(3, -1).T
    points = _physical_points(corners, spacing_xyz, origin_xyz, direction)
    return PhysicalBoundingBox(
        tuple(int(v) for v in start),
        tuple(int(v) for v in stop),
        tuple(float(v) for v in points.min(axis=0)),
        tuple(float(v) for v in points.max(axis=0)),
    )


def coarse_centreline(
    mask_zyx: np.ndarray,
    spacing_xyz: Sequence[float],
    *,
    origin_xyz: Sequence[float] = (0.0, 0.0, 0.0),
    direction: Sequence[float] = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
    step_mm: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate an optional coarse PCA-binned centreline and tangent sequence."""
    indices = np.argwhere(mask_zyx)
    if len(indices) < 8:
        empty = np.empty((0, 3), dtype=np.float32)
        return empty, empty.copy()
    points = _physical_points(indices, spacing_xyz, origin_xyz, direction)
    centre = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points - centre, full_matrices=False)
    axis = vh[0]
    projection = (points - centre) @ axis
    edges = np.arange(projection.min(), projection.max() + step_mm, step_mm)
    centres = [np.median(points[(projection >= a) & (projection < b)], axis=0)
               for a, b in zip(edges[:-1], edges[1:])
               if np.count_nonzero((projection >= a) & (projection < b)) >= 4]
    line = np.asarray(centres, dtype=np.float32)
    if len(line) < 2:
        empty = np.empty((0, 3), dtype=np.float32)
        return empty, empty.copy()
    tangent = np.gradient(line, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-6)
    return line, tangent.astype(np.float32)


def analyze_aorta(
    intensity_zyx: np.ndarray,
    mask_zyx: np.ndarray,
    spacing_xyz: Sequence[float],
    *,
    origin_xyz: Sequence[float] = (0.0, 0.0, 0.0),
    direction: Sequence[float] = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
    wall_thickness_mm: float = 2.0,
) -> AortaGeometry:
    """Run the complete conservative aorta preparation stage."""
    spacing = tuple(float(v) for v in spacing_xyz)
    mask = clean_aorta_mask(mask_zyx, spacing)
    sdf = signed_distance_mm(mask, spacing)
    interior, lateral = derive_walls(mask, spacing, thickness_mm=wall_thickness_mm)
    caps = detect_end_caps(mask, spacing, origin_xyz=origin_xyz, direction=direction)
    lateral &= ~caps
    line, tangents = coarse_centreline(
        mask, spacing, origin_xyz=origin_xyz, direction=direction
    )
    return AortaGeometry(
        mask=mask,
        signed_distance_mm=sdf,
        interior_wall=interior,
        lateral_wall=lateral,
        end_caps=caps,
        blood=estimate_blood_model(intensity_zyx, mask, spacing),
        bounding_box=physical_bounding_box(
            mask, spacing, origin_xyz=origin_xyz, direction=direction
        ),
        centreline_xyz=line,
        tangents_xyz=tangents,
        spacing_xyz=spacing,
    )
