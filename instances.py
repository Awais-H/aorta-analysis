"""D3 instance separation: wall-layer connected components define instances (B); a watershed
seeded from them through the candidate voxels grows each one (D).

Wall-patch labelling ported from triage.py lines 146 to 152 (the atlas wall_patches column is this
count before any filtering). Labels are ordered by wall-patch area, largest first, so the ordering
is deterministic across runs.

Output contract (SPEC.md D9): label array (0 background, 1..N branches), per-label wall-patch voxel
indices.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage
from skimage.segmentation import watershed

import config
from candidates import Candidates

log = logging.getLogger("branchseed.instances")


@dataclass
class Instances:
    labels: np.ndarray                 # int32 (z, y, x): watershed region of each branch, 0 background
    wall_labels: np.ndarray            # int32 (z, y, x): the wall-patch component of each label
    wall_indices: dict = field(default_factory=dict)   # label -> (k, 3) int zyx voxel indices of the wall patch
    wall_area_mm2: dict = field(default_factory=dict)  # label -> wall-patch voxel count x in-plane voxel area (atlas convention)
    region_volume_ml: dict = field(default_factory=dict)  # label -> watershed region volume (D6 rule 4)
    n: int = 0

    @property
    def labels_list(self) -> list:
        return list(range(1, self.n + 1))


def build(cand: Candidates) -> Instances:
    wall = cand.candidates & (cand.distance_mm <= config.WALL_LAYER_MM)
    wl, nw = ndimage.label(wall)
    if nw == 0:
        log.info("no wall patches")
        return Instances(labels=np.zeros(cand.mask.shape, np.int32), wall_labels=np.zeros(cand.mask.shape, np.int32))

    sizes = np.asarray(ndimage.sum(wall, wl, range(1, nw + 1)))
    order = np.argsort(-sizes, kind="stable")          # largest patch first
    relabel = np.zeros(nw + 1, np.int32)
    relabel[order + 1] = np.arange(1, nw + 1, dtype=np.int32)
    wl = relabel[wl]
    sizes = sizes[order]

    labels = watershed(cand.distance_mm, markers=wl, mask=cand.candidates).astype(np.int32)

    a_mm2 = float(cand.spacing[1] * cand.spacing[2])
    vox_ml = float(np.prod(cand.spacing)) / 1000.0
    idx = np.argwhere(wl > 0)
    labs = wl[tuple(idx.T)]
    srt = np.argsort(labs, kind="stable")
    idx, labs = idx[srt], labs[srt]
    bounds = np.searchsorted(labs, np.arange(1, nw + 2))
    wall_indices = {int(k): idx[bounds[k - 1]:bounds[k]] for k in range(1, nw + 1)}
    region_counts = np.bincount(labels.ravel(), minlength=nw + 1)

    inst = Instances(
        labels=labels, wall_labels=wl, n=int(nw),
        wall_indices=wall_indices,
        wall_area_mm2={int(k): float(sizes[k - 1] * a_mm2) for k in range(1, nw + 1)},
        region_volume_ml={int(k): float(region_counts[k] * vox_ml) for k in range(1, nw + 1)},
    )
    log.info("%d wall patches; areas mm2 (top 8): %s", nw,
             [round(v, 1) for v in list(inst.wall_area_mm2.values())[:8]])
    return inst
