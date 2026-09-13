"""D5 proximal tracing: seed, direction, radius.

Slice-and-centroid march (option C): start at the ostium with the axis direction, step
TRACE_STEP_MM, take the cross-section of the branch's own voxels (its watershed region through
the raw shell, see instances.branch_voxels) perpendicular to the current direction, update the
direction toward the cross-section centroid, repeat. Stop at TRACE_MAX_MM of path length, when the
cross-section splits into two blobs (first bifurcation: the trunk's direction is reported, not an
average of its children), when it vanishes, when the direction flips, or at the volume edge.
Fallback (option A): if the march yields under MIN_TRACE_MM, use the straight axis line as far as
the branch voxels reach; if that is also under MIN_TRACE_MM the branch is ineligible (seed None).

Outputs: seed = point at SEED_DISTANCE_MM of path length; direction = normalised ostium-to-seed
chord (the reference convention); radius = area-equivalent radius of the thresholded cross-section
on a plane perpendicular to the chord at the seed, resampled at RADIUS_PLANE_SPACING_MM, with the
inscribed-circle fallback when it exceeds RADIUS_AORTA_FRACTION_MAX of the local aortic radius,
clamped to RADIUS_CLAMP_MM. PCA of the path is kept as a diagnostic (angle to the chord).

Output contract (SPEC.md D9): per label, path points (mm), seed (mm), direction (unit), radius
(mm), path length, bifurcation flag.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

import config
import io_utils
from candidates import Candidates
from instances import Instances, branch_voxels
from ostium import Ostium

log = logging.getLogger("branchseed.tracing")


@dataclass
class Trace:
    label: int
    path_mm: np.ndarray            # (k, 3) xyz mm, path_mm[0] is the ostium
    seed_mm: np.ndarray | None     # None when the path is shorter than SEED_DISTANCE_MM
    direction_xyz: np.ndarray      # unit vector from the ostium into the branch (chord)
    radius_mm: float
    path_length_mm: float
    bifurcation: bool
    method: str                    # "march", "axis" or "none"
    stop_reason: str = ""          # why the march ended: max, bifurcation, vanished, flip, edge
    radius_flag: str | None = None  # "inscribed_fallback" when the area radius failed the sanity check, "no_cross_section" when none was found
    radius_area_mm: float | None = None
    radius_inscribed_mm: float | None = None
    section_fills_window: bool = False  # the seed cross-section reaches the CROSS_SECTION_HALF_WIDTH_MM window edge (D6 rule 4)
    section_cortex_fraction: float = 0.0  # fraction of the seed cross-section over BONE_HU_LUMEN_RATIO x the lumen median: cortical bone or calcium (D6 rule 4)
    pca_chord_angle_deg: float | None = None  # diagnostic: angle between the PCA axis of the first 5 mm of path and the chord
    march_length_mm: float = 0.0   # what the march achieved before any fallback
    section_areas_mm2: list | None = None  # cross-section area (main blob voxels x voxel area) at each march step, 1 mm apart; D6 rule 4


# ------------------------------------------------------------------ geometry helpers


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def chord_direction(ostium_mm, seed_mm) -> np.ndarray:
    """D5: the reported direction is normalise(seed - ostium), the answer key's convention."""
    return _unit(np.asarray(seed_mm, float) - np.asarray(ostium_mm, float))


def point_along(path_mm: np.ndarray, s_mm: float) -> np.ndarray | None:
    """Point at arc length s_mm along a polyline, or None if the polyline is shorter."""
    if len(path_mm) < 2:
        return None
    seg = np.linalg.norm(np.diff(path_mm, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    if s_mm > arc[-1] + 1e-9:
        return None
    return np.array([np.interp(s_mm, arc, path_mm[:, a]) for a in range(3)])


def pca_direction(points: np.ndarray, origin: np.ndarray) -> np.ndarray | None:
    """Principal axis of `points` with the sign fixed to point away from `origin` (diagnostic)."""
    if len(points) < 3:
        return None
    c = points.mean(axis=0)
    w, V = np.linalg.eigh(np.cov((points - c).T))
    axis = _unit(V[:, int(np.argmax(w))])
    if np.sum((points - origin) @ axis) < 0:
        axis = -axis
    return axis


def _perp_basis(d: np.ndarray):
    a = np.array([1.0, 0.0, 0.0]) if abs(d[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = _unit(np.cross(d, a))
    w = _unit(np.cross(d, u))
    return u, w


# ------------------------------------------------------------------ the march


def cross_section(V: np.ndarray, q: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Indices into V (mm-scaled voxel coordinates) of voxels in the slab perpendicular to d at q."""
    rel = V - q
    t = rel @ d
    lat = np.linalg.norm(rel - t[:, None] * d, axis=1)
    return np.where((np.abs(t) <= config.TRACE_SLAB_HALF_MM) & (lat <= config.CROSS_SECTION_HALF_WIDTH_MM))[0]


def blobs(idx: np.ndarray) -> list:
    """26-connected components of a set of voxel indices; list of index arrays into `idx`."""
    lo = idx.min(axis=0)
    local = idx - lo
    arr = np.zeros(tuple(local.max(axis=0) + 1), bool)
    arr[tuple(local.T)] = True
    lab, n = ndimage.label(arr, structure=np.ones((3, 3, 3), bool))
    labs = lab[tuple(local.T)]
    return [np.where(labs == k)[0] for k in range(1, n + 1)]


def march(cand: Candidates, region_idx: np.ndarray, ost: Ostium):
    """Returns (points in mm-scaled index space, bifurcation flag, stop reason, section areas mm2)."""
    sp = cand.spacing
    V = region_idx * sp
    p = ost.index_zyx * sp
    d = _unit(ost.axis_zyx.astype(float))
    pts = [p]
    areas = []
    vox_area = float(sp[1] * sp[2])
    upper = (np.array(cand.mask.shape) - 1) * sp
    n_steps = int(round(config.TRACE_MAX_MM / config.TRACE_STEP_MM))
    for _ in range(n_steps):
        q = p + config.TRACE_STEP_MM * d
        sel = cross_section(V, q, d)
        if len(sel) < config.MIN_CROSS_SECTION_VOXELS:
            return pts, False, "vanished", areas
        parts = blobs(region_idx[sel])
        lat = np.linalg.norm((V[sel] - q) - ((V[sel] - q) @ d)[:, None] * d, axis=1)
        nearest = int(np.argmin(lat))
        main = next(b for b in parts if nearest in set(b.tolist()))
        if len(main) < config.MIN_CROSS_SECTION_VOXELS:
            return pts, False, "vanished", areas
        others = [b for b in parts if b is not main and len(b) >= config.MIN_CROSS_SECTION_VOXELS]
        # a split counts as the first downstream bifurcation only once the slab is clear of the
        # wall layer: inside it the cross-section still contains fragments of the origin itself
        q_vox = np.clip(np.round(q / sp).astype(int), 0, np.array(cand.mask.shape) - 1)
        if others and cand.distance_mm[tuple(q_vox)] > config.WALL_LAYER_MM:
            return pts, True, "bifurcation", areas
        c = V[sel][main].mean(axis=0)
        d_new = _unit(c - p)
        if np.degrees(np.arccos(np.clip(np.dot(d_new, d), -1, 1))) > config.TRACE_MAX_TURN_DEG:
            return pts, False, "flip", areas
        p_new = p + config.TRACE_STEP_MM * d_new
        if np.any(p_new < 0) or np.any(p_new > upper):
            return pts, False, "edge", areas
        p, d = p_new, d_new
        pts.append(p)
        areas.append(len(main) * vox_area)
    return pts, False, "max", areas


def axis_path(cand: Candidates, region_idx: np.ndarray, ost: Ostium) -> list:
    """Fallback A: straight line along the axis for as long as the line itself stays inside the
    branch voxels (a branch voxel within one working voxel of every step point). Returns points
    in mm-scaled index space."""
    from scipy.spatial import cKDTree
    sp = cand.spacing
    p0 = ost.index_zyx * sp
    d = _unit(ost.axis_zyx.astype(float))
    pts = [p0]
    if len(region_idx) == 0:
        return pts
    tree = cKDTree(region_idx * sp)
    upper = (np.array(cand.mask.shape) - 1) * sp
    n_steps = int(round(config.TRACE_MAX_MM / config.TRACE_STEP_MM))
    for k in range(1, n_steps + 1):
        p = p0 + k * config.TRACE_STEP_MM * d
        if np.any(p < 0) or np.any(p > upper):
            break
        if tree.query(p)[0] > config.ISO_SPACING_MM:
            break
        pts.append(p)
    return pts


# ------------------------------------------------------------------ radius


def radius_at(cand: Candidates, seed_idx: np.ndarray, dir_idx: np.ndarray, local_aortic_radius_mm: float | None):
    """Area-equivalent and inscribed-circle radius of the thresholded cross-section on a plane
    perpendicular to dir_idx at seed_idx, sampled at RADIUS_PLANE_SPACING_MM.
    Returns (radius_mm, area_radius, inscribed_radius, flag, fills_window, cortex_fraction)."""
    sp = float(cand.spacing[0])
    h = config.RADIUS_PLANE_SPACING_MM
    u, w = _perp_basis(_unit(dir_idx.astype(float)))
    n = int(round(2 * config.CROSS_SECTION_HALF_WIDTH_MM / h)) + 1
    off = (np.arange(n) - n // 2) * h
    S, T = np.meshgrid(off, off, indexing="ij")
    P = seed_idx * sp + S[..., None] * u + T[..., None] * w      # mm-scaled index space
    coords = np.moveaxis(P / sp, -1, 0)
    ct = ndimage.map_coordinates(cand.ct, coords, order=1, cval=config.CT_BACKGROUND_HU)
    inside = ndimage.map_coordinates(cand.mask.astype(np.uint8), coords, order=0, cval=0)
    bright = (ct > cand.threshold_hu) & (inside == 0)
    lab, k = ndimage.label(bright, structure=np.ones((3, 3), bool))
    if k == 0:
        return float(config.RADIUS_CLAMP_MM[0]), None, None, "no_cross_section", False, 0.0
    centre = (n // 2, n // 2)
    if lab[centre] > 0:
        comp = lab == lab[centre]
    else:
        dist_to_bright = ndimage.distance_transform_edt(lab == 0) * h
        if dist_to_bright[centre] > config.ISO_SPACING_MM:
            return float(config.RADIUS_CLAMP_MM[0]), None, None, "no_cross_section", False, 0.0
        yy, xx = np.where(lab > 0)
        j = int(np.argmin((yy - centre[0]) ** 2 + (xx - centre[1]) ** 2))
        comp = lab == lab[yy[j], xx[j]]
    r_area = float(np.sqrt(comp.sum() * h * h / np.pi))
    r_ins = float(ndimage.distance_transform_edt(comp).max() * h)
    fills = bool(comp[0, :].any() or comp[-1, :].any() or comp[:, 0].any() or comp[:, -1].any())
    cortex = float((ct[comp] > config.BONE_HU_LUMEN_RATIO * cand.hu_stats["hu_in_mask_median"]).mean())
    r, flag = r_area, None
    if local_aortic_radius_mm is not None and r_area > config.RADIUS_AORTA_FRACTION_MAX * local_aortic_radius_mm:
        r, flag = r_ins, "inscribed_fallback"
    return float(np.clip(r, *config.RADIUS_CLAMP_MM)), r_area, r_ins, flag, fills, cortex


def local_aortic_radius(inside_edt: np.ndarray, spacing: np.ndarray, at_idx: np.ndarray) -> float | None:
    """Largest inscribed-sphere radius of the mask within SHELL_MM of `at_idx` (box approximation)."""
    r = np.ceil(config.SHELL_MM / spacing).astype(int)
    lo = np.maximum(at_idx - r, 0)
    hi = np.minimum(at_idx + r + 1, np.array(inside_edt.shape))
    sub = inside_edt[tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))]
    return float(sub.max()) if sub.size else None


# ------------------------------------------------------------------ per branch


def trace_one(cand: Candidates, ost: Ostium, region_idx: np.ndarray, inside_edt: np.ndarray | None = None) -> Trace:
    sp = cand.spacing
    if len(region_idx) == 0:
        return Trace(label=ost.label, path_mm=io_utils.index_to_mm(cand.image, ost.index_zyx)[None, :], seed_mm=None,
                     direction_xyz=ost.normal_mm.copy(), radius_mm=float(config.RADIUS_CLAMP_MM[0]),
                     path_length_mm=0.0, bifurcation=False, method="none", stop_reason="no_voxels")
    pts, bif, reason, areas = march(cand, region_idx, ost)
    march_len = float((len(pts) - 1) * config.TRACE_STEP_MM)
    method = "march"
    if march_len < config.MIN_TRACE_MM:
        pts, bif, method = axis_path(cand, region_idx, ost), False, "axis"
        areas = None
    length = float((len(pts) - 1) * config.TRACE_STEP_MM)
    pts_idx = np.array(pts) / sp
    path_mm = io_utils.index_to_mm(cand.image, pts_idx)
    seed = point_along(path_mm, config.SEED_DISTANCE_MM)
    if seed is None:
        return Trace(label=ost.label, path_mm=path_mm, seed_mm=None, direction_xyz=ost.axis_mm.copy(),
                     radius_mm=float(config.RADIUS_CLAMP_MM[0]), path_length_mm=length, bifurcation=bif,
                     method=method if length > 0 else "none", stop_reason=reason, march_length_mm=march_len,
                     section_areas_mm2=areas)
    direction = chord_direction(ost.mm, seed)
    seed_idx = io_utils.mm_to_index(cand.image, seed)
    dir_idx = io_utils.mm_vector_to_index(cand.image, seed, direction)
    aortic_r = local_aortic_radius(inside_edt, sp, ost.index_zyx) if inside_edt is not None else None
    radius, r_area, r_ins, flag, fills, cortex = radius_at(cand, seed_idx, dir_idx, aortic_r)
    first = path_mm[np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(path_mm, axis=0), axis=1))]) <= config.SEED_DISTANCE_MM + 1e-9]
    pca = pca_direction(first, path_mm[0])
    pca_angle = float(np.degrees(np.arccos(np.clip(np.dot(pca, direction), -1, 1)))) if pca is not None else None
    return Trace(label=ost.label, path_mm=path_mm, seed_mm=seed, direction_xyz=direction, radius_mm=radius,
                 path_length_mm=length, bifurcation=bif, method=method, stop_reason=reason, radius_flag=flag,
                 radius_area_mm=r_area, radius_inscribed_mm=r_ins, pca_chord_angle_deg=pca_angle, march_length_mm=march_len,
                 section_areas_mm2=areas, section_fills_window=fills, section_cortex_fraction=cortex)


def trace_all(cand: Candidates, inst: Instances, ostia: dict) -> dict:
    """label -> Trace for every ostium."""
    if not ostia:
        return {}
    regions = branch_voxels(cand, inst)
    inside_edt = ndimage.distance_transform_edt(cand.mask, sampling=cand.spacing).astype(np.float32)
    out = {label: trace_one(cand, ost, regions.get(label, np.zeros((0, 3), int)), inside_edt) for label, ost in ostia.items()}
    n_march = sum(t.method == "march" for t in out.values())
    n_bif = sum(t.bifurcation for t in out.values())
    n_short = sum(t.seed_mm is None for t in out.values())
    log.info("traced %d branches: %d by march, %d by axis fallback, %d bifurcations, %d under %g mm",
             len(out), n_march, sum(t.method == "axis" for t in out.values()), n_bif, n_short, config.MIN_TRACE_MM)
    return out
