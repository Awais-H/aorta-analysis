"""D4 ostium localisation.

STUB. Current behaviour: wall-patch centroid (option A) snapped to the nearest raw-mask boundary
voxel, outward wall normal from the local gradient of the distance-from-mask field. The real
version (inscribed-circle centre, axis-intersection refinement, 4 mm agreement check, then snap)
per SPEC.md D4 replaces `locate_one` without changing the contract.

Output contract (SPEC.md D9): per label, ostium in working index space and in mm, outward normal.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

import io_utils
from candidates import Candidates
from instances import Instances

log = logging.getLogger("branchseed.ostium")


@dataclass
class Ostium:
    label: int
    index_zyx: np.ndarray     # int (3,), a mask-boundary voxel on the working grid
    mm: np.ndarray            # (3,) xyz physical mm
    normal_zyx: np.ndarray    # unit outward wall normal, index space
    normal_mm: np.ndarray     # unit outward wall normal, xyz physical
    method: str = "stub_centroid_snap"


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


def locate_one(cand: Candidates, label: int, wall_idx: np.ndarray, boundary: np.ndarray, tree: cKDTree) -> Ostium:
    centroid = wall_idx.mean(axis=0)
    _, j = tree.query(centroid * cand.spacing)
    b = boundary[j]
    n = _local_gradient(cand.distance_mm, centroid, cand.spacing)
    if np.linalg.norm(n) == 0:
        n = (centroid - b) * cand.spacing
    if np.linalg.norm(n) == 0:
        n = np.array([0.0, 0.0, 1.0])
    n = n / np.linalg.norm(n)
    return Ostium(label=int(label), index_zyx=b.astype(int), mm=io_utils.index_to_mm(cand.image, b),
                  normal_zyx=n, normal_mm=io_utils.index_vector_to_mm(cand.image, b, n))


def locate(cand: Candidates, inst: Instances) -> dict:
    """label -> Ostium for every instance."""
    if inst.n == 0:
        return {}
    boundary = mask_boundary_voxels(cand.mask)
    tree = cKDTree(boundary * cand.spacing)
    out = {}
    for label, wall_idx in inst.wall_indices.items():
        out[label] = locate_one(cand, label, wall_idx, boundary, tree)
    log.info("located %d ostia", len(out))
    return out
