"""Stage 3 - aortic geometry: centreline, radius profile, surface, end caps,
rotation-minimising frame.

Nothing here assumes the supplied segment is abdominal, straight, or monotonic
in z. Coverage varies between cases and an arch is a perfectly ordinary input:
an axial slice through one contains two disconnected aortic cross-sections, so
there is no per-slice reasoning and no superior-inferior ordering anywhere in
this module.

Directions (tangents, frame vectors, surface normals) are unit vectors in
*millimetre-scaled index space*: the direction cosine matrix is orthonormal, so
these differ from physical-space directions by a fixed rotation and by nothing
else. Points remain (x, y, z) ROI continuous indices.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
from scipy import interpolate, ndimage
from scipy.sparse import csgraph
from scipy.spatial import cKDTree
from skimage import morphology

from .calibrate import signed_distance_mm
from .config import Config
from .frames import align_to_reference, rotation_minimising_frame
from .interp import sample_nearest, sample_volume
from .types import AortaGeometry, CaseGrid, Flags
from .voxelgraph import voxel_graph

log = logging.getLogger(__name__)

class GeometryError(RuntimeError):
    """The mask does not support a usable centreline."""


# --------------------------------------------------------------------------- #
# Skeleton graph
# --------------------------------------------------------------------------- #

def _skeleton_graph(skel: np.ndarray, spacing_zyx: np.ndarray):
    """Sparse 26-connectivity graph over skeleton voxels with physical weights."""
    return voxel_graph(skel, spacing_zyx)


def _largest_component(coords: np.ndarray, graph) -> Tuple[np.ndarray, object, np.ndarray]:
    n_comp, labels = csgraph.connected_components(graph, directed=False)
    if n_comp <= 1:
        return coords, graph, np.arange(coords.shape[0])
    sizes = np.bincount(labels)
    keep = np.nonzero(labels == int(np.argmax(sizes)))[0]
    log.info("skeleton has %d components; keeping the largest (%d of %d voxels)",
             n_comp, keep.size, coords.shape[0])
    return coords[keep], graph[keep][:, keep], keep


def prune_spurs(skel: np.ndarray, edt_mm: np.ndarray, grid: CaseGrid, cfg: Config) -> np.ndarray:
    """Delete dead-end branches shorter than the local vessel radius.

    Surface irregularity on a real aorta produces short skeleton spurs - the
    same non-anatomic branches Riffaud describes. Left in place they add
    spurious endpoints and can capture the longest path.
    """
    skel = skel.copy()
    spacing_zyx = grid.spacing_zyx
    for iteration in range(10):
        coords, graph = _skeleton_graph(skel, spacing_zyx)
        if coords.shape[0] == 0:
            return skel
        degree = np.diff(graph.indptr)
        endpoints = np.nonzero(degree == 1)[0]
        if endpoints.size == 0:
            return skel

        indices, indptr = graph.indices, graph.indptr
        removed: List[int] = []
        for start in endpoints:
            if degree[start] != 1:
                continue
            chain = [start]
            prev = -1
            node = int(start)
            length = 0.0
            while True:
                neighbours = indices[indptr[node]: indptr[node + 1]]
                nxt = [int(v) for v in neighbours if v != prev]
                if len(nxt) != 1:
                    break                    # junction or isolated: stop here
                step = nxt[0]
                length += float(np.linalg.norm((coords[step] - coords[node]) * spacing_zyx))
                if degree[step] != 2:
                    break                    # reached a junction; do not delete it
                chain.append(step)
                prev, node = node, step
                if length > 200.0:
                    break
            z, y, x = coords[start]
            local_radius = float(edt_mm[z, y, x])
            if length < cfg.geometry.spur_prune_factor * max(local_radius, 1e-3):
                removed.extend(chain)

        if not removed:
            return skel
        rm = coords[np.unique(removed)]
        skel[rm[:, 0], rm[:, 1], rm[:, 2]] = False
        log.debug("spur prune pass %d removed %d voxels", iteration + 1, rm.shape[0])
    return skel


def longest_path(skel: np.ndarray, grid: CaseGrid) -> np.ndarray:
    """Voxel path between the two extremes of the skeleton, in xyz index order.

    The standard tree-diameter trick: farthest node from an arbitrary start,
    then farthest node from that. Exact on a tree and close enough on the
    near-tree a pruned vessel skeleton gives.
    """
    coords, graph = _skeleton_graph(skel, grid.spacing_zyx)
    if coords.shape[0] == 0:
        raise GeometryError("skeleton is empty")
    coords, graph, _ = _largest_component(coords, graph)
    if coords.shape[0] == 1:
        raise GeometryError("skeleton has a single voxel")

    first = csgraph.dijkstra(graph, indices=0, directed=False)
    first[~np.isfinite(first)] = -1.0
    node_a = int(np.argmax(first))
    second, predecessors = csgraph.dijkstra(
        graph, indices=node_a, directed=False, return_predecessors=True
    )
    second[~np.isfinite(second)] = -1.0
    node_b = int(np.argmax(second))

    path = [node_b]
    while path[-1] != node_a:
        nxt = int(predecessors[path[-1]])
        if nxt < 0:
            raise GeometryError("skeleton path reconstruction failed")
        path.append(nxt)
    path.reverse()

    zyx = coords[np.asarray(path)]
    log.info("centreline raw path: %d voxels, %.1f mm", len(path), float(second[node_b]))
    return zyx[:, ::-1].astype(np.float64)          # -> xyz


# --------------------------------------------------------------------------- #
# Spline and resampling
# --------------------------------------------------------------------------- #

def fit_centreline(
    path_xyz: np.ndarray, grid: CaseGrid, cfg: Config
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Smooth the voxel path and resample it at uniform physical arc length.

    Returns ``(points_xyz, arc_length_mm, tangent_mm)``. Tangents are unit
    vectors in millimetre-scaled index space.
    """
    n_raw = path_xyz.shape[0]
    if n_raw < 4:
        raise GeometryError(f"centreline path too short ({n_raw} voxels)")

    degree = min(3, n_raw - 1)
    smoothing = cfg.geometry.spline_smoothing * n_raw
    try:
        tck, _ = interpolate.splprep(path_xyz.T, s=smoothing, k=degree)
    except (TypeError, ValueError) as exc:      # duplicate points, degenerate path
        log.warning("spline fit failed (%s); falling back to an interpolating fit", exc)
        tck, _ = interpolate.splprep(path_xyz.T, s=0.0, k=degree)

    # Two passes: dense evaluation to measure arc length, then re-evaluation at
    # parameter values that give uniform spacing in millimetres.
    dense_u = np.linspace(0.0, 1.0, max(10 * n_raw, 200))
    dense = np.stack(interpolate.splev(dense_u, tck), axis=1)
    steps = np.linalg.norm(np.diff(dense, axis=0) * grid.roi_spacing_mm, axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(steps)])
    total_mm = float(cumulative[-1])
    if total_mm < 2 * cfg.geometry.centreline_step_mm:
        raise GeometryError(f"centreline is only {total_mm:.1f} mm long")

    n_samples = max(int(np.floor(total_mm / cfg.geometry.centreline_step_mm)) + 1, 4)
    targets = np.linspace(0.0, total_mm, n_samples)
    u_at = np.interp(targets, cumulative, dense_u)

    points = np.stack(interpolate.splev(u_at, tck), axis=1)
    derivative = np.stack(interpolate.splev(u_at, tck, der=1), axis=1)
    tangent = derivative * grid.roi_spacing_mm            # into mm-space
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-9)

    log.info("centreline: %.1f mm, %d samples at %.1f mm",
             total_mm, n_samples, cfg.geometry.centreline_step_mm)
    return points, targets, tangent


def extend_to_cut_faces(
    points: np.ndarray,
    arc_length: np.ndarray,
    tangent: np.ndarray,
    mask: np.ndarray,
    grid: CaseGrid,
    cfg: Config,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extrapolate the centreline along its end tangents up to the mask edge.

    A 3D skeleton retracts several millimetres from a flat cut face, so the
    fitted centreline stops short of the mask. Rays in Stage 4 are cast
    perpendicular to the tangent, which would leave that band of wall
    unsampled - and a real origin a few millimetres from the cut face is a case
    the challenge calls out explicitly. Extending along the straight end
    tangent restores coverage right up to the terminal plane, where the
    end-cap tests then take over.
    """
    step = cfg.geometry.centreline_step_mm
    max_steps = int(round(cfg.geometry.max_extension_mm / step))
    prefix: List[np.ndarray] = []
    suffix: List[np.ndarray] = []

    for direction_mm, anchor, sink in ((-tangent[0], points[0], prefix),
                                       (tangent[-1], points[-1], suffix)):
        for k in range(1, max_steps + 1):
            candidate = anchor + grid.mm_to_index_delta(direction_mm, k * step)
            if not sample_nearest(mask, candidate):
                break
            sink.append(candidate)

    if not prefix and not suffix:
        return points, arc_length, tangent

    head = np.asarray(prefix[::-1]).reshape(-1, 3)
    tail = np.asarray(suffix).reshape(-1, 3)
    new_points = np.concatenate([head, points, tail], axis=0)
    new_tangent = np.concatenate(
        [np.repeat(tangent[:1], head.shape[0], axis=0), tangent,
         np.repeat(tangent[-1:], tail.shape[0], axis=0)], axis=0
    )
    steps = np.linalg.norm(np.diff(new_points, axis=0) * grid.roi_spacing_mm, axis=1)
    new_arc = np.concatenate([[0.0], np.cumsum(steps)])
    log.info("centreline extended by %.1f mm / %.1f mm at the two ends (total %.1f mm)",
             head.shape[0] * step, tail.shape[0] * step, float(new_arc[-1]))
    return new_points, new_arc, new_tangent


def stabilise_end_radius(
    radius_mm: np.ndarray, arc_length: np.ndarray, window_mm: float
) -> np.ndarray:
    """Stop the local radius collapsing against a flat cut face.

    The EDT measures distance to the nearest background voxel, and within a
    voxel or two of a terminal plane the nearest background *is* that plane.
    Left alone the profile falls to half a voxel at both ends, which shrinks
    the ray cap and the end-cap windows exactly where they matter most. A
    vessel does not change calibre abruptly over a centimetre, so each sample
    near an end is raised to the largest radius seen between it and the
    interior. The interior profile, including any genuine taper, is untouched.
    """
    if radius_mm.size < 3 or window_mm <= 0:
        return radius_mm
    out = radius_mm.copy()
    n = int(np.searchsorted(arc_length, arc_length[0] + window_mm))
    n = int(min(max(n, 1), radius_mm.size // 2))
    out[:n] = np.maximum.accumulate(radius_mm[:n][::-1])[::-1]
    out[-n:] = np.maximum.accumulate(radius_mm[-n:])
    return out


# --------------------------------------------------------------------------- #
# Surface, normals, end caps
# --------------------------------------------------------------------------- #

def surface_and_normals(
    mask: np.ndarray, grid: CaseGrid, cfg: Config
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One-voxel surface shell plus outward unit normals in mm-space.

    Normals come from the gradient of the signed EDT, lightly smoothed to
    suppress the staircase artefacts of a voxelised surface. The signed EDT
    increases inward, so the outward normal is the negated gradient.
    """
    surface = mask & ~ndimage.binary_erosion(mask)
    signed = signed_distance_mm(mask, grid)
    smoothed = signed
    if cfg.geometry.normal_smooth_voxels > 0:
        smoothed = ndimage.gaussian_filter(signed, cfg.geometry.normal_smooth_voxels)

    grad_z, grad_y, grad_x = np.gradient(smoothed)
    # d/d(mm) = (1 / spacing) * d/d(index), and mm-space components are xyz.
    sx, sy, sz = grid.roi_spacing_mm
    vec = np.stack([-grad_x / sx, -grad_y / sy, -grad_z / sz]).astype(np.float32)
    norm = np.linalg.norm(vec, axis=0)
    np.divide(vec, np.maximum(norm, 1e-9), out=vec)
    return surface, vec, signed


def exclude_end_caps(
    surface: np.ndarray,
    normals: np.ndarray,
    centreline: np.ndarray,
    arc_length: np.ndarray,
    tangent: np.ndarray,
    radius_mm: np.ndarray,
    grid: CaseGrid,
    cfg: Config,
    flags: Flags,
) -> np.ndarray:
    """Mandatory: the flat faces created by cropping are not branch origins.

    Three tests applied together. Test 1 alone catches nothing when the mask
    sits in the volume interior, which is the common case, so tests 2 and 3 are
    the primary mechanism rather than a refinement.
    """
    eligible = surface.copy()
    voxels = np.argwhere(surface)
    if voxels.size == 0:
        return eligible
    n_surface = voxels.shape[0]

    # --- Test 1: on or within one voxel of a ROI face ------------------------
    shape = np.asarray(surface.shape)
    on_face = ((voxels <= 0) | (voxels >= shape - 1)).any(axis=1)

    # --- Test 2: outward normal parallel to the tangent near an endpoint -----
    pts_mm = centreline * grid.roi_spacing_mm
    tree = cKDTree(pts_mm)
    voxel_xyz = voxels[:, ::-1].astype(np.float64)
    _, nearest = tree.query(voxel_xyz * grid.roi_spacing_mm)

    normal_vecs = normals[:, voxels[:, 0], voxels[:, 1], voxels[:, 2]].T
    arc_at = arc_length[nearest]
    radius_at = radius_mm[nearest]
    total_mm = float(arc_length[-1])

    cos_limit = np.cos(np.radians(cfg.geometry.endcap_normal_deg))
    parallel = np.zeros(n_surface, dtype=bool)
    near_plane = np.zeros(n_surface, dtype=bool)

    for end, (endpoint_arc, endpoint_pt, outward_t, endpoint_radius) in enumerate(
        (
            (0.0, centreline[0], -tangent[0], radius_mm[0]),
            (total_mm, centreline[-1], tangent[-1], radius_mm[-1]),
        )
    ):
        window = cfg.geometry.endcap_arc_factor * np.maximum(radius_at, 1e-3)
        near_end = np.abs(arc_at - endpoint_arc) <= window
        cosine = normal_vecs @ outward_t
        parallel |= near_end & (cosine >= cos_limit)

        # --- Test 3: within a few mm of the terminal cross-sectional plane ---
        offset_mm = (voxel_xyz - endpoint_pt) * grid.roi_spacing_mm
        along = offset_mm @ outward_t
        radial = np.linalg.norm(offset_mm - along[:, None] * outward_t, axis=1)
        near_plane |= (np.abs(along) <= cfg.geometry.endcap_plane_mm) & (
            radial <= cfg.geometry.endcap_radial_factor * max(float(endpoint_radius), 1e-3)
        )

    excluded = on_face | parallel | near_plane
    eligible[voxels[excluded, 0], voxels[excluded, 1], voxels[excluded, 2]] = False

    log.info(
        "end caps: %d of %d surface voxels excluded (face %d, normal %d, plane %d)",
        int(excluded.sum()), n_surface, int(on_face.sum()), int(parallel.sum()),
        int(near_plane.sum()),
    )
    if on_face.any():
        flags.add("mask_touches_roi_boundary")
    if excluded.sum() > 0.5 * n_surface:
        log.warning("end-cap exclusion removed over half the surface")
        flags.add("large_endcap_exclusion")
    return eligible


# --------------------------------------------------------------------------- #
# Frame
# --------------------------------------------------------------------------- #

def anterior_reference(grid: CaseGrid) -> np.ndarray:
    """Anterior direction expressed in mm-scaled index space.

    SimpleITK physical space is LPS, so anterior is -y. Rotating it back
    through the direction cosines is what makes theta = 0 mean 12 o'clock on
    the display rather than an arbitrary direction that changes per case.
    """
    anterior_physical = np.array([0.0, -1.0, 0.0])
    return grid.roi_direction.T @ anterior_physical


def build_frame(centreline: np.ndarray, tangent: np.ndarray, grid: CaseGrid):
    pts_mm = centreline * grid.roi_spacing_mm
    u, v = rotation_minimising_frame(pts_mm, tangent)
    mid = centreline.shape[0] // 2
    return align_to_reference(u, v, tangent, anterior_reference(grid), mid)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def build_geometry(
    mask: np.ndarray, edt_mm: np.ndarray, grid: CaseGrid, cfg: Config, flags: Flags
) -> AortaGeometry:
    skeleton = morphology.skeletonize(mask, method="lee").astype(bool)
    n_skel = int(skeleton.sum())
    if n_skel < cfg.geometry.min_skeleton_voxels:
        raise GeometryError(f"skeleton has only {n_skel} voxels")
    skeleton = prune_spurs(skeleton, edt_mm, grid, cfg)
    log.info("skeleton: %d voxels (%d after pruning)", n_skel, int(skeleton.sum()))

    path_xyz = longest_path(skeleton, grid)
    centreline, arc_length, tangent = fit_centreline(path_xyz, grid, cfg)
    centreline, arc_length, tangent = extend_to_cut_faces(
        centreline, arc_length, tangent, mask, grid, cfg
    )

    radius_mm = sample_volume(edt_mm, centreline, order=1).astype(np.float64)
    radius_mm = stabilise_end_radius(
        np.maximum(radius_mm, 1e-3), arc_length, cfg.geometry.end_radius_window_mm
    )
    log.info("radius profile: median %.1f mm, min %.1f mm, max %.1f mm",
             float(np.median(radius_mm)), float(radius_mm.min()), float(radius_mm.max()))

    surface, normals, signed_edt = surface_and_normals(mask, grid, cfg)
    eligible_wall = exclude_end_caps(
        surface, normals, centreline, arc_length, tangent, radius_mm, grid, cfg, flags
    )
    frame_u, frame_v = build_frame(centreline, tangent, grid)

    return AortaGeometry(
        centreline=centreline, arc_length=arc_length, tangent=tangent,
        frame_u=frame_u, frame_v=frame_v, radius_mm=radius_mm,
        mask=mask, surface=surface, eligible_wall=eligible_wall,
        edt_mm=edt_mm, signed_edt_mm=signed_edt, normals=normals,
    )
