"""Sub-voxel sampling helpers.

This module is the *only* place in the package where an (x, y, z) point is
turned into a [z, y, x] array index. Every other module hands points here and
never touches the axis order itself. All entry points are batched: there are no
per-point ``map_coordinates`` calls anywhere in the pipeline.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def points_to_coords(pts_xyz: np.ndarray) -> np.ndarray:
    """``(..., 3)`` xyz points -> ``(3, M)`` zyx coordinate array.

    The single axis-order flip in the codebase.
    """
    pts = np.asarray(pts_xyz, dtype=np.float64)
    if pts.shape[-1] != 3:
        raise ValueError(f"points must have trailing axis 3 (x, y, z), got {pts.shape}")
    flat = pts.reshape(-1, 3)
    return np.stack([flat[:, 2], flat[:, 1], flat[:, 0]], axis=0)


def sample_volume(
    arr_zyx: np.ndarray,
    pts_xyz: np.ndarray,
    order: int = 3,
    mode: str = "nearest",
    cval: float = 0.0,
    prefilter: bool | None = None,
) -> np.ndarray:
    """Interpolate ``arr_zyx`` at ``pts_xyz`` with one batched call.

    Returns an array shaped like ``pts_xyz[..., 0]``.
    """
    pts = np.asarray(pts_xyz, dtype=np.float64)
    coords = points_to_coords(pts)
    if prefilter is None:
        prefilter = order > 1
    values = ndimage.map_coordinates(
        arr_zyx, coords, order=order, mode=mode, cval=cval, prefilter=prefilter
    )
    return values.reshape(pts.shape[:-1])


def sample_volume_prefiltered(
    spline_zyx: np.ndarray, pts_xyz: np.ndarray, order: int = 3, mode: str = "nearest"
) -> np.ndarray:
    """As :func:`sample_volume` but on an already spline-prefiltered volume.

    Use :func:`prefilter_volume` once and this many times when sampling the
    same volume repeatedly; the prefilter is the expensive half of a cubic
    interpolation on a multi-million-voxel array.
    """
    return sample_volume(spline_zyx, pts_xyz, order=order, mode=mode, prefilter=False)


def prefilter_volume(arr_zyx: np.ndarray, order: int = 3) -> np.ndarray:
    if order <= 1:
        return arr_zyx
    return ndimage.spline_filter(arr_zyx.astype(np.float32), order=order, output=np.float32)


def voxel_indices(pts_xyz: np.ndarray, shape_zyx) -> np.ndarray:
    """Round xyz points to in-bounds ``(..., 3)`` integer zyx voxel indices."""
    pts = np.asarray(pts_xyz, dtype=np.float64)
    idx = np.empty(pts.shape[:-1] + (3,), dtype=np.intp)
    for out_axis, in_axis in enumerate((2, 1, 0)):          # z<-2, y<-1, x<-0
        limit = shape_zyx[out_axis] - 1
        idx[..., out_axis] = np.clip(np.rint(pts[..., in_axis]), 0, limit).astype(np.intp)
    return idx


def sample_nearest(arr_zyx: np.ndarray, pts_xyz: np.ndarray) -> np.ndarray:
    """Nearest-neighbour lookup by rounding. Much faster than order=0 sampling
    on large point sets, and exact for boolean masks."""
    idx = voxel_indices(pts_xyz, arr_zyx.shape)
    return arr_zyx[idx[..., 0], idx[..., 1], idx[..., 2]]


def in_bounds(pts_xyz: np.ndarray, shape_zyx, margin: float = 0.0) -> np.ndarray:
    """Boolean mask of points inside the array, with an optional margin."""
    pts = np.asarray(pts_xyz, dtype=np.float64)
    ok = np.ones(pts.shape[:-1], dtype=bool)
    for in_axis, out_axis in ((0, 2), (1, 1), (2, 0)):
        ok &= pts[..., in_axis] >= margin
        ok &= pts[..., in_axis] <= shape_zyx[out_axis] - 1 - margin
    return ok


def orthonormal_basis(direction: np.ndarray) -> tuple:
    """Two unit vectors spanning the plane perpendicular to ``direction``."""
    d = np.asarray(direction, dtype=np.float64)
    d = d / max(np.linalg.norm(d), 1e-9)
    seed = np.array([0.0, 0.0, 1.0]) if abs(d[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(d, seed)
    e1 /= max(np.linalg.norm(e1), 1e-9)
    e2 = np.cross(d, e1)
    e2 /= max(np.linalg.norm(e2), 1e-9)
    return e1, e2
