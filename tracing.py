"""D5 proximal tracing: seed, direction, radius.

STUB. Current behaviour: a straight line from the ostium along the outward wall normal, stepped
TRACE_STEP_MM at a time up to TRACE_MAX_MM and bounds-checked at the volume edge (a branch within
5 mm of the boundary cannot be traced 5 mm and is ineligible by the PDF's own rule). Seed at
SEED_DISTANCE_MM along the path; direction = the normal; radius = equivalent radius of the wall
patch. The slice-and-centroid march with bifurcation stop and inscribed-circle radius per SPEC.md
D5 replaces `trace_one` without changing the contract.

Output contract (SPEC.md D9): per label, path points (mm), seed (mm), direction (unit), radius
(mm), path length, bifurcation flag.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

import config
import io_utils
from candidates import Candidates
from instances import Instances
from ostium import Ostium

log = logging.getLogger("branchseed.tracing")


@dataclass
class Trace:
    label: int
    path_mm: np.ndarray        # (k, 3) xyz mm, path_mm[0] is the ostium
    seed_mm: np.ndarray | None  # None when the path is shorter than SEED_DISTANCE_MM
    direction_xyz: np.ndarray  # unit vector from the ostium into the branch
    radius_mm: float
    path_length_mm: float
    bifurcation: bool
    method: str = "stub_straight_normal"


def point_along(path_mm: np.ndarray, s_mm: float) -> np.ndarray | None:
    """Point at arc length s_mm along a polyline, or None if the polyline is shorter."""
    if len(path_mm) < 2:
        return None
    seg = np.linalg.norm(np.diff(path_mm, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    if s_mm > arc[-1] + 1e-9:
        return None
    return np.array([np.interp(s_mm, arc, path_mm[:, a]) for a in range(3)])


def trace_one(cand: Candidates, ost: Ostium, wall_area_mm2: float) -> Trace:
    step_idx = ost.normal_zyx * (config.TRACE_STEP_MM / cand.spacing)
    n_steps = int(round(config.TRACE_MAX_MM / config.TRACE_STEP_MM))
    upper = np.array(cand.mask.shape) - 1
    pts = []
    for k in range(n_steps + 1):
        p = ost.index_zyx + k * step_idx
        if np.any(p < 0) or np.any(p > upper):
            break
        pts.append(p)
    path_mm = io_utils.index_to_mm(cand.image, np.array(pts)) if pts else np.zeros((0, 3))
    length = float((len(pts) - 1) * config.TRACE_STEP_MM) if pts else 0.0
    seed = point_along(path_mm, config.SEED_DISTANCE_MM)
    radius = float(np.sqrt(max(wall_area_mm2, 0.0) / np.pi))
    return Trace(label=ost.label, path_mm=path_mm, seed_mm=seed, direction_xyz=ost.normal_mm.copy(),
                 radius_mm=radius, path_length_mm=length, bifurcation=False)


def trace_all(cand: Candidates, inst: Instances, ostia: dict) -> dict:
    """label -> Trace for every ostium."""
    out = {label: trace_one(cand, ost, inst.wall_area_mm2.get(label, 0.0)) for label, ost in ostia.items()}
    log.info("traced %d branches", len(out))
    return out
