"""D4 ostium localisation.

Per label: (D) initial estimate = centre of the largest inscribed circle in the wall patch, i.e.
the patch voxel furthest from the patch's lateral edge; (C) refinement = PCA axis of the branch
voxels within AXIS_FIT_MM of the mask, walked back to the aorta, intersected with the mask
surface; if C is within OSTIUM_AGREEMENT_MM of D use C, else D (a bad axis fit never beats the
baseline). C is only reported when config.OSTIUM_USE_AXIS_REFINEMENT is set: on the coarse
labelled cases it lost to D even inside the gate, so by default it is computed for the log only.
Finally snap to the nearest raw-mask boundary voxel, which is the annotators' origin convention.
Outward wall normal from the local gradient of the distance-from-mask field. The fitted axis
(when the fit succeeded and agreed) is handed to D5 as the tracer's initial direction.

Output contract (SPEC.md D9): per label, ostium in working index space and in mm, outward normal.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

import config
import io_utils
from candidates import Candidates
from instances import Instances, branch_voxels

log = logging.getLogger("branchseed.ostium")


@dataclass
class Ostium:
    label: int
    index_zyx: np.ndarray     # int (3,), a mask-boundary voxel on the working grid
    mm: np.ndarray            # (3,) xyz physical mm
    normal_zyx: np.ndarray    # unit outward wall normal, index space
    normal_mm: np.ndarray     # unit outward wall normal, xyz physical
    axis_zyx: np.ndarray      # unit initial branch direction for the tracer (axis fit, else the normal)
    axis_mm: np.ndarray
    method: str               # "axis_intersection" or "inscribed_circle"
    inscribed_idx: np.ndarray | None = None  # D estimate (float zyx), for the failure gallery
    axis_idx: np.ndarray | None = None       # C estimate (float zyx) or None if the fit failed
    agreement_mm: float | None = None        # |C - D| in mm, or None


def mask_boundary_voxels(mask: np.ndarray) -> np.ndarray:
    """(k, 3) zyx indices of mask voxels with a non-mask 6-neighbour."""
    inner = ndimage.binary_erosion(mask, structure=ndimage.generate_binary_structure(3, 1), border_value=0)
    return np.argwhere(mask & ~inner)


def _local_gradient(field: np.ndarray, at_zyx: np.ndarray, spacing: np.ndarray) -> np.ndarray:
    """Central-difference gradient of `field` at the voxel nearest `at_zyx`, clipped at the edges."""
    c = np.clip(np.round(at_zyx).astype(int), 0, np.array(field.shape) - 1)
    g = np.zeros(3)
    for a in range(3):
        lo, hi = c.copy(), c.copy()
        lo[a] = max(0, c[a] - 1)
        hi[a] = min(field.shape[a] - 1, c[a] + 1)
        span = (hi[a] - lo[a]) * spacing[a]
        g[a] = (float(field[tuple(hi)]) - float(field[tuple(lo)])) / span if span > 0 else 0.0
    return g


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def inscribed_centre(cand: Candidates, wall_idx: np.ndarray) -> np.ndarray:
    """D: the wall-patch voxel furthest from the patch's lateral edge (float zyx index).

    The edge is every wall-layer voxel that is not in the patch, so the distance is measured along
    the wall rather than across the 2 mm layer; the slab faces do not cap it.
    """
    pad = int(np.ceil(config.WALL_LAYER_MM / cand.spacing.min())) + 1
    lo = np.maximum(wall_idx.min(axis=0) - pad, 0)
    hi = np.minimum(wall_idx.max(axis=0) + pad + 1, np.array(cand.mask.shape))
    box = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    local = wall_idx - lo
    patch = np.zeros(tuple(hi - lo), bool)
    patch[tuple(local.T)] = True
    dist = cand.distance_mm[box]
    edge = (dist > 0) & (dist <= config.WALL_LAYER_MM) & ~patch
    if not edge.any():
        return wall_idx.mean(axis=0)
    d = ndimage.distance_transform_edt(~edge, sampling=cand.spacing)
    vals = d[tuple(local.T)]
    # an elongated patch has a ridge of equally deep voxels: take the ridge voxel nearest the
    # ridge's own centroid, so the estimate sits mid-patch instead of at whichever end comes first
    ridge = wall_idx[vals >= vals.max() - cand.spacing.min() / 2]
    centre = ridge.mean(axis=0)
    return ridge[int(np.argmin(np.linalg.norm((ridge - centre) * cand.spacing, axis=1)))].astype(float)


def axis_fit(cand: Candidates, region_idx: np.ndarray, from_idx: np.ndarray):
    """C: PCA axis of the branch voxels within AXIS_FIT_MM of the mask, oriented away from the
    aorta, walked back until it enters the mask. Returns (axis_zyx unit, crossing float index) or
    None when the fit is impossible or the axis never meets the aorta."""
    near = region_idx[cand.distance_mm[tuple(region_idx.T)] <= config.AXIS_FIT_MM]
    if len(near) < config.AXIS_FIT_MIN_VOXELS:
        return None
    P = near * cand.spacing
    c = P.mean(axis=0)
    cov = np.cov((P - c).T)
    w, V = np.linalg.eigh(cov)
    axis = _unit(V[:, int(np.argmax(w))])
    if np.dot(c - from_idx * cand.spacing, axis) < 0:
        axis = -axis
    upper = (np.array(cand.mask.shape) - 1) * cand.spacing
    p = c.copy()
    travelled = 0.0
    prev = None
    while travelled <= config.SHELL_MM + config.AXIS_FIT_MM:
        if np.any(p < 0) or np.any(p > upper):
            return None
        vox = np.round(p / cand.spacing).astype(int)
        if cand.mask[tuple(vox)]:
            crossing = (p + prev) / 2 if prev is not None else p
            return axis, crossing / cand.spacing
        prev = p
        p = p - config.AXIS_WALK_STEP_MM * axis
        travelled += config.AXIS_WALK_STEP_MM
    return None


def locate_one(cand: Candidates, label: int, wall_idx: np.ndarray, region_idx: np.ndarray,
               boundary: np.ndarray, tree: cKDTree) -> Ostium:
    d_est = inscribed_centre(cand, wall_idx)
    fit = axis_fit(cand, region_idx, d_est) if len(region_idx) else None
    chosen, method, axis_idx, agreement = d_est, "inscribed_circle", None, None
    axis = None
    if fit is not None:
        axis, c_est = fit
        axis_idx = c_est
        agreement = float(np.linalg.norm((c_est - d_est) * cand.spacing))
        if agreement <= config.OSTIUM_AGREEMENT_MM and config.OSTIUM_USE_AXIS_REFINEMENT:
            chosen, method = c_est, "axis_intersection"
    _, j = tree.query(chosen * cand.spacing)
    b = boundary[j]
    n = _local_gradient(cand.distance_mm, d_est, cand.spacing)
    if np.linalg.norm(n) == 0:
        n = (wall_idx.mean(axis=0) - b) * cand.spacing
    if np.linalg.norm(n) == 0:
        n = np.array([0.0, 0.0, 1.0])
    n = _unit(n)
    # the tracer starts with the fitted axis whenever the fit is trustworthy (inside the agreement
    # gate), whether or not C's point is the reported ostium; otherwise with the wall normal
    a = axis if (axis is not None and agreement <= config.OSTIUM_AGREEMENT_MM) else n
    return Ostium(label=int(label), index_zyx=b.astype(int), mm=io_utils.index_to_mm(cand.image, b),
                  normal_zyx=n, normal_mm=io_utils.index_vector_to_mm(cand.image, b, n),
                  axis_zyx=a, axis_mm=io_utils.index_vector_to_mm(cand.image, b, a),
                  method=method, inscribed_idx=d_est, axis_idx=axis_idx, agreement_mm=agreement)


def locate(cand: Candidates, inst: Instances) -> dict:
    """label -> Ostium for every instance."""
    if inst.n == 0:
        return {}
    boundary = mask_boundary_voxels(cand.mask)
    tree = cKDTree(boundary * cand.spacing)
    regions = branch_voxels(cand, inst)
    out = {}
    for label, wall_idx in inst.wall_indices.items():
        out[label] = locate_one(cand, label, wall_idx, regions.get(label, np.zeros((0, 3), int)), boundary, tree)
    n_axis = sum(o.method == "axis_intersection" for o in out.values())
    log.info("located %d ostia (%d by axis intersection, %d by inscribed circle)", len(out), n_axis, len(out) - n_axis)
    return out
