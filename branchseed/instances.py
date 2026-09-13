"""Convert independently verified paths into wall-contact vessel instances."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from .aorta import AortaGeometry
from .tracking import Point3, VerifiedPath, voxel_to_physical

Index3 = tuple[int, int, int]


def _sampling(spacing_xyz: Sequence[float]) -> np.ndarray:
    return np.asarray(tuple(spacing_xyz), float)[::-1]


@dataclass(frozen=True, slots=True)
class WallContactPatch:
    voxels_zyx: tuple[Index3, ...]
    points_xyz: tuple[Point3, ...]
    area_mm2: float
    centroid_xyz: Point3

    def __post_init__(self) -> None:
        if not self.voxels_zyx or len(self.voxels_zyx) != len(self.points_xyz):
            raise ValueError("a wall-contact patch needs matching voxel and physical points")


@dataclass(frozen=True, slots=True)
class VerifiedInstance:
    """One direct daughter, possibly supported by both tracking methods."""

    paths: tuple[VerifiedPath, ...]
    contact_patch: WallContactPatch
    interface_voxels_zyx: tuple[Index3, ...]
    interface_points_xyz: tuple[Point3, ...]
    trunk_path_zyx: tuple[tuple[float, float, float], ...]
    trunk_path_xyz: tuple[Point3, ...]
    vessel_mask_zyx: np.ndarray | None = field(default=None, repr=False, compare=False)
    metadata: Mapping[str, float | int | str | bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.paths or not all(path.verification.passed for path in self.paths):
            raise ValueError("instances require at least one verified path")
        if len(self.interface_voxels_zyx) != len(self.interface_points_xyz):
            raise ValueError("interface coordinates must correspond")
        if len(self.trunk_path_zyx) != len(self.trunk_path_xyz):
            raise ValueError("trunk voxel and physical paths must correspond")
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def ostium_xyz(self) -> Point3:
        return self.contact_patch.centroid_xyz


def _contact_path_mask(path: VerifiedPath, shape: Sequence[int]) -> np.ndarray:
    if path.vessel_mask_zyx is not None:
        return np.asarray(path.vessel_mask_zyx, bool).copy()
    mask = np.zeros(shape, bool)
    points = np.rint(np.asarray(path.path_zyx)).astype(int)
    points = np.clip(points, 0, np.asarray(shape) - 1)
    mask[tuple(points.T)] = True
    return ndimage.binary_dilation(mask, ndimage.generate_binary_structure(3, 1))


def derive_wall_contact_patch(
    path: VerifiedPath,
    geometry: AortaGeometry,
    *,
    origin_xyz: Sequence[float] = (0, 0, 0),
    direction: Sequence[float] = (1, 0, 0, 0, 1, 0, 0, 0, 1),
) -> tuple[WallContactPatch, np.ndarray, np.ndarray | None]:
    """Return lateral wall patch, vessel-side interface, and vessel mask."""
    contact_vessel = _contact_path_mask(path, geometry.mask.shape) & ~geometry.mask
    vessel = (
        np.asarray(path.vessel_mask_zyx, bool) & ~geometry.mask
        if path.vessel_mask_zyx is not None
        else None
    )
    connectivity = ndimage.generate_binary_structure(3, 2)
    vessel_neighbourhood = ndimage.binary_dilation(contact_vessel, connectivity)
    patch_mask = geometry.lateral_wall & vessel_neighbourhood
    # Restrict broad segmented contacts to the component nearest the path origin.
    labels, count = ndimage.label(patch_mask, connectivity)
    origin = np.rint(np.asarray(path.path_zyx[0])).astype(int)
    if count:
        patch_points = np.argwhere(patch_mask)
        nearest = patch_points[np.argmin(
            np.linalg.norm((patch_points - origin) * _sampling(geometry.spacing_xyz), axis=1)
        )]
        patch_mask = labels == labels[tuple(nearest)]
    if not patch_mask.any():
        # The verified contact can be sub-voxel relative to a thin wall shell.
        wall_points = np.argwhere(geometry.lateral_wall)
        if not len(wall_points):
            raise ValueError("aorta geometry has no lateral wall")
        nearest = wall_points[np.argmin(
            np.linalg.norm((wall_points - origin) * _sampling(geometry.spacing_xyz), axis=1)
        )]
        patch_mask[tuple(nearest)] = True
    interface_mask = contact_vessel & ndimage.binary_dilation(patch_mask, connectivity)
    patch_indices = np.argwhere(patch_mask)
    patch_xyz = np.asarray([
        voxel_to_physical(p, geometry.spacing_xyz, origin_xyz, direction) for p in patch_indices
    ])
    # Boundary voxel area depends on orientation; this isotropic-equivalent
    # estimate is stable for small, oblique patches.
    voxel_area = float(np.prod(geometry.spacing_xyz) ** (2 / 3))
    patch = WallContactPatch(
        tuple(tuple(int(v) for v in p) for p in patch_indices),
        tuple(tuple(float(v) for v in p) for p in patch_xyz),
        float(len(patch_indices) * voxel_area),
        tuple(float(v) for v in np.mean(patch_xyz, axis=0)),
    )
    return patch, interface_mask, vessel


def _arc_prefix(points: np.ndarray, distance_mm: float) -> np.ndarray:
    if len(points) < 2:
        return points
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    end = np.searchsorted(arc, distance_mm, side="right")
    return points[:max(2, min(end, len(points)))]


def proximal_path_overlap(
    first: VerifiedPath,
    second: VerifiedPath,
    *,
    proximal_mm: float = 8.0,
    tolerance_mm: float = 1.5,
) -> float:
    """Symmetric fraction of proximal samples lying on the same trunk."""
    a = _arc_prefix(np.asarray(first.path_xyz, float), proximal_mm)
    b = _arc_prefix(np.asarray(second.path_xyz, float), proximal_mm)
    if not len(a) or not len(b):
        return 0.0
    ab = np.mean(cKDTree(b).query(a)[0] <= tolerance_mm)
    ba = np.mean(cKDTree(a).query(b)[0] <= tolerance_mm)
    return float(min(ab, ba))


def _patch_overlap(first: WallContactPatch, second: WallContactPatch) -> float:
    a, b = set(first.voxels_zyx), set(second.voxels_zyx)
    return len(a & b) / max(1, min(len(a), len(b)))


def _patch_distance(first: WallContactPatch, second: WallContactPatch) -> float:
    a, b = np.asarray(first.points_xyz), np.asarray(second.points_xyz)
    return float(min(cKDTree(a).query(b)[0].min(), cKDTree(b).query(a)[0].min()))


def _common_trunk(paths: Sequence[VerifiedPath], tolerance_mm: float = 1.5) -> VerifiedPath:
    """Choose the longest trace through the mutually shared proximal trunk."""
    if len(paths) == 1:
        return paths[0]
    reference = max(paths, key=lambda path: path.length_mm)
    points = np.asarray(reference.path_xyz)
    shared = np.ones(len(points), bool)
    for other in paths:
        shared &= cKDTree(np.asarray(other.path_xyz)).query(points)[0] <= tolerance_mm
    # Shared points must form a prefix; stop before the first genuine split.
    stop = 1
    for index, value in enumerate(shared):
        if value:
            stop = index + 1
        elif index > 1:
            break
    if stop < 2:
        return reference
    # Return the representative path; callers slice it using the computed common
    # prefix to preserve an exact voxel/physical correspondence.
    return reference


def derive_instances(
    paths: Sequence[VerifiedPath],
    geometry: AortaGeometry,
    *,
    origin_xyz: Sequence[float] = (0, 0, 0),
    direction: Sequence[float] = (1, 0, 0, 0, 1, 0, 0, 0, 1),
    patch_overlap_threshold: float = 0.35,
    surface_distance_mm: float = 1.5,
    proximal_overlap_threshold: float = 0.60,
) -> list[VerifiedInstance]:
    """Deduplicate direct daughters while preserving distinct nearby ostia."""
    records = []
    for path in paths:
        if not path.verification.passed or not path.verification.direct_aorta_contact:
            continue  # Explicitly reject indirect daughters.
        patch, interface, vessel = derive_wall_contact_patch(
            path, geometry, origin_xyz=origin_xyz, direction=direction
        )
        records.append((path, patch, interface, vessel))
    groups: list[list[int]] = []
    for index, (path, patch, _, _) in enumerate(records):
        selected = None
        for group_index, group in enumerate(groups):
            representative, other_patch, _, _ = records[group[0]]
            overlap = proximal_path_overlap(path, representative)
            same_contact = _patch_overlap(patch, other_patch) >= patch_overlap_threshold
            nearby_contact = _patch_distance(patch, other_patch) <= surface_distance_mm
            # Surface proximity alone never collapses two neighboring ostia.
            if same_contact or (nearby_contact and overlap >= proximal_overlap_threshold):
                selected = group_index
                break
        if selected is None:
            groups.append([index])
        else:
            groups[selected].append(index)
    instances: list[VerifiedInstance] = []
    for group in groups:
        selected = [records[i] for i in group]
        member_paths = tuple(item[0] for item in selected)
        patch_voxels = sorted(set().union(*(set(item[1].voxels_zyx) for item in selected)))
        patch_points = np.asarray([
            voxel_to_physical(p, geometry.spacing_xyz, origin_xyz, direction) for p in patch_voxels
        ])
        patch = WallContactPatch(
            tuple(patch_voxels),
            tuple(tuple(float(v) for v in p) for p in patch_points),
            float(sum(item[1].area_mm2 for item in selected) / len(selected)),
            tuple(float(v) for v in patch_points.mean(axis=0)),
        )
        interface = np.logical_or.reduce([item[2] for item in selected])
        image_vessels = [item[3] for item in selected if item[3] is not None]
        vessel = np.logical_or.reduce(image_vessels) if image_vessels else None
        interface_indices = np.argwhere(interface)
        interface_points = tuple(
            tuple(float(v) for v in voxel_to_physical(
                point, geometry.spacing_xyz, origin_xyz, direction
            )) for point in interface_indices
        )
        trunk = _common_trunk(member_paths)
        # Determine the maximal prefix shared by every method.
        trunk_xyz = np.asarray(trunk.path_xyz)
        shared_count = len(trunk_xyz)
        for other in member_paths:
            distances = cKDTree(np.asarray(other.path_xyz)).query(trunk_xyz)[0]
            failures = np.flatnonzero(distances > 1.5)
            if len(failures):
                shared_count = min(shared_count, max(2, int(failures[0])))
        instances.append(VerifiedInstance(
            member_paths,
            patch,
            tuple(tuple(int(v) for v in p) for p in interface_indices),
            interface_points,
            tuple(trunk.path_zyx[:shared_count]),
            tuple(trunk.path_xyz[:shared_count]),
            vessel,
            {"method_count": len(member_paths), "common_trunk_points": shared_count},
        ))
    return instances


# Naming used by some pipeline clients.
build_instances = derive_instances

