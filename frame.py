"""Centreline, aortic frame, mm -> (height, clock).

STUB. Current behaviour: the centreline is the per-slice mask centroid path (the triage.py
tortuosity path, lines 113 to 121), ordered superior to inferior; the two endpoints are its ends;
the clock angle is measured in the axial plane. The geodesic centreline and rotation-minimising
frame per SPEC.md D8 replace `build` without changing the contract. The endpoints are the first
deliverable because D2 and D6 rule 1 need them to identify end faces.

Clock convention (D8): 12 = anterior, 3 = patient's left, 6 = posterior, 9 = patient's right.
In LPS mm, anterior is -y and patient's left is +x. Height = arc length from the superior end.

Output contract (SPEC.md D9): centreline points (mm), per-point frame, function mm -> (height, clock).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

import io_utils
from candidates import Candidates

log = logging.getLogger("branchseed.frame")


@dataclass
class Frame:
    centreline_mm: np.ndarray      # (k, 3) xyz mm, superior -> inferior
    centreline_idx: np.ndarray     # (k, 3) zyx working-grid indices (float)
    tangents: np.ndarray           # (k, 3) unit xyz, pointing inferior
    arc_mm: np.ndarray             # (k,) arc length from the superior end
    length_mm: float
    endpoints_mm: np.ndarray       # (2, 3): [superior, inferior]
    endpoint_idx_zyx: np.ndarray   # (2, 3)
    tortuosity: float              # path length / chord
    method: str = "stub_slice_centroids"

    def end_faces(self) -> list:
        """[(endpoint_mm, outward unit tangent)] for the superior and inferior ends. D2/D6 rule 1:
        the end faces are the mask boundary regions nearest these two points."""
        if len(self.centreline_mm) < 2:
            return [(self.endpoints_mm[0], np.array([0.0, 0.0, 1.0])), (self.endpoints_mm[1], np.array([0.0, 0.0, -1.0]))]
        return [(self.endpoints_mm[0], -self.tangents[0]), (self.endpoints_mm[1], self.tangents[-1])]

    def nearest(self, xyz_mm) -> int:
        d = np.linalg.norm(self.centreline_mm - np.asarray(xyz_mm, float)[None, :], axis=1)
        return int(np.argmin(d))

    def height_clock(self, xyz_mm) -> tuple[float, float]:
        """(height mm from the superior end, clock hours in [0, 12)) of a physical point."""
        p = np.asarray(xyz_mm, float)
        i = self.nearest(p)
        v = p - self.centreline_mm[i]
        angle = np.arctan2(v[0], -v[1])           # 0 at anterior (-y), +pi/2 at patient's left (+x)
        clock = (np.degrees(angle) / 30.0) % 12.0  # 30 degrees per hour
        return float(self.arc_mm[i]), float(clock)


def build(cand: Candidates) -> Frame:
    m = cand.mask
    zs = np.where(m.any(axis=(1, 2)))[0]
    cents = np.array([ndimage.center_of_mass(m[z]) for z in zs], dtype=float).reshape(-1, 2)
    idx = np.column_stack([zs.astype(float), cents])
    mm = io_utils.index_to_mm(cand.image, idx)
    if len(mm) > 1 and mm[0, 2] < mm[-1, 2]:       # superior (+z in LPS) first
        idx, mm = idx[::-1], mm[::-1]
    if len(mm) > 1:
        seg = np.linalg.norm(np.diff(mm, axis=0), axis=1)
        arc = np.concatenate([[0.0], np.cumsum(seg)])
        tang = np.gradient(mm, axis=0)
        norms = np.linalg.norm(tang, axis=1, keepdims=True)
        tang = np.where(norms > 0, tang / np.maximum(norms, 1e-12), np.array([[0.0, 0.0, -1.0]]))
        chord = float(np.linalg.norm(mm[-1] - mm[0]))
    else:
        arc = np.zeros(len(mm))
        tang = np.tile([[0.0, 0.0, -1.0]], (len(mm), 1))
        chord = 0.0
    length = float(arc[-1]) if len(arc) else 0.0
    fr = Frame(centreline_mm=mm, centreline_idx=idx, tangents=tang, arc_mm=arc, length_mm=length,
               endpoints_mm=np.array([mm[0], mm[-1]]), endpoint_idx_zyx=np.array([idx[0], idx[-1]]),
               tortuosity=float(length / chord) if chord > 0 else 1.0)
    log.info("centreline: %d points, %.1f mm, tortuosity %.3f", len(mm), length, fr.tortuosity)
    return fr
