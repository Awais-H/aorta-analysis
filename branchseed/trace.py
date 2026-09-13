"""Stage 5 - the proximal trace.

Ten millimetres is shorter than Wala's optimised cylinder height of 15 mm, so
iterative cylinder tracking is the wrong tool: it is built for following a
vessel across many generations, and here we need one stub measured precisely.
Instead each candidate gets a small subvolume upsampled to 0.5 mm - the only
place in the pipeline we upsample - and a geodesic distance field walked
outward in level sets. That gives 20 samples across the 10 mm budget against
the three or four a cylinder tracker would manage at 1.5 mm native spacing.

Geodesic rather than Euclidean distance matters: a branch that curves sharply
would fail a straight-line 5 mm test.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy import ndimage
from scipy.sparse import csgraph

from .config import Config
from .interp import sample_nearest, sample_volume_prefiltered, voxel_indices
from .types import AortaGeometry, Calibration, Candidate, CaseGrid
from .voxelgraph import NEIGHBOURHOOD_26, voxel_graph

log = logging.getLogger(__name__)


@dataclass
class Subvolume:
    """A local cube resampled to 0.5 mm, plus the affine back to ROI space."""

    ct: np.ndarray                 # (n, n, n) float32, [z, y, x]
    aorta: np.ndarray              # (n, n, n) bool
    centre_roi: np.ndarray         # (3,) ROI continuous index, xyz
    spacing_mm: float
    n: int
    grid: CaseGrid

    @property
    def half_index(self) -> float:
        return (self.n - 1) / 2.0

    @property
    def spacing_zyx(self) -> np.ndarray:
        return np.full(3, self.spacing_mm)

    @property
    def voxel_volume_mm3(self) -> float:
        return self.spacing_mm ** 3

    def to_roi(self, pts_sub_xyz: np.ndarray) -> np.ndarray:
        """Subvolume continuous index -> ROI continuous index."""
        pts = np.asarray(pts_sub_xyz, dtype=np.float64)
        offset_mm = (pts - self.half_index) * self.spacing_mm
        return self.centre_roi + offset_mm / self.grid.roi_spacing_mm

    def from_roi(self, pts_roi_xyz: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts_roi_xyz, dtype=np.float64)
        offset_mm = (pts - self.centre_roi) * self.grid.roi_spacing_mm
        return offset_mm / self.spacing_mm + self.half_index

    def voxels_to_roi(self, coords_zyx: np.ndarray) -> np.ndarray:
        return self.to_roi(np.asarray(coords_zyx, dtype=np.float64)[:, ::-1])


def extract_subvolume(
    centre_roi: np.ndarray,
    image_spline: np.ndarray,
    aorta_mask: np.ndarray,
    grid: CaseGrid,
    cfg: Config,
) -> Subvolume:
    """Resample a cube around ``centre_roi`` to the trace spacing."""
    spacing = cfg.roi.subvolume_spacing_mm
    n = int(round(2.0 * cfg.roi.subvolume_half_mm / spacing)) + 1
    offsets_mm = (np.arange(n) - (n - 1) / 2.0) * spacing

    # Build a [z, y, x] grid of ROI continuous indices in one shot.
    oz, oy, ox = np.meshgrid(offsets_mm, offsets_mm, offsets_mm, indexing="ij")
    offset_mm = np.stack([ox, oy, oz], axis=-1)
    pts = centre_roi + offset_mm / grid.roi_spacing_mm

    ct = sample_volume_prefiltered(image_spline, pts, order=3).astype(np.float32)
    aorta = sample_nearest(aorta_mask, pts)
    return Subvolume(ct=ct, aorta=aorta, centre_roi=np.asarray(centre_roi, dtype=np.float64),
                     spacing_mm=spacing, n=n, grid=grid)


def geodesic_distance(
    region: np.ndarray, source: np.ndarray, spacing_zyx: np.ndarray
) -> np.ndarray:
    """Geodesic distance in mm from ``source`` through ``region``.

    Dijkstra on the 26-connectivity voxel graph rather than fast marching. The
    lattice bias is the classic few-percent overestimate on diagonals, which is
    immaterial over a 10 mm budget, and it removes a dependency from an
    offline install - worth more than 5% on a diagonal.
    """
    out = np.full(region.shape, np.inf, dtype=np.float64)
    coords, graph = voxel_graph(region, spacing_zyx)
    if coords.shape[0] == 0:
        return out
    node_of = np.full(region.shape, -1, dtype=np.int64)
    node_of[coords[:, 0], coords[:, 1], coords[:, 2]] = np.arange(coords.shape[0])

    source_nodes = node_of[source & region]
    source_nodes = source_nodes[source_nodes >= 0]
    if source_nodes.size == 0:
        return out

    distances = csgraph.dijkstra(graph, directed=False, indices=source_nodes, min_only=True)
    out[coords[:, 0], coords[:, 1], coords[:, 2]] = distances
    return out


@dataclass
class Level:
    distance_mm: float
    voxels: np.ndarray            # (K, 3) zyx subvolume voxels retained
    centroid_zyx: np.ndarray
    radius_mm: float
    n_components: int


def _equivalent_radius(n_voxels: int, sub: Subvolume, step_mm: float) -> float:
    """Cross-sectional equivalent radius of a level band.

    The band is a shell of thickness ``step_mm``, so its cross-sectional area
    is its volume divided by that thickness.
    """
    area = n_voxels * sub.voxel_volume_mm3 / max(step_mm, 1e-6)
    return float(np.sqrt(max(area, 0.0) / np.pi))


def walk_levels(
    distance: np.ndarray,
    source: np.ndarray,
    sub: Subvolume,
    calibration: Calibration,
    cfg: Config,
) -> Tuple[List[Level], str, Optional[float], List[str]]:
    """Follow the geodesic level sets outward and decide where to stop."""
    step = cfg.trace.step_mm
    n_levels = int(round(cfg.trace.max_mm / step))
    persist_levels = max(int(round(cfg.trace.split_persist_mm / step)), 1)

    levels: List[Level] = []
    flags: List[str] = []
    retained = source.copy()
    stop_reason = "max"
    split_at: Optional[float] = None
    pending_split: Optional[int] = None
    following_dominant = False

    for k in range(1, n_levels + 1):
        d = k * step
        band = (distance >= d) & (distance < d + step)
        if not band.any():
            stop_reason = "weak_match"
            break

        labels, n_labels = ndimage.label(band, structure=NEIGHBOURHOOD_26)
        neighbourhood = ndimage.binary_dilation(retained, structure=NEIGHBOURHOOD_26)
        connected = np.unique(labels[neighbourhood & band])
        connected = connected[connected > 0]
        if connected.size == 0:
            stop_reason = "weak_match"
            break

        keep = np.isin(labels, connected)
        sizes = [int((labels == c).sum()) for c in connected]

        # --- weak match ------------------------------------------------------
        # Propagation runs at a permissive threshold so partial-volume dimming
        # in a small vessel does not kill the trace; the band itself must still
        # be predominantly true lumen.
        total = int(keep.sum())
        strong = float((sub.ct[keep] > calibration.t_lumen).mean()) if total else 0.0
        if total < cfg.trace.min_component_voxels or strong < cfg.trace.strong_match_frac:
            stop_reason = "weak_match"
            break

        voxels = np.argwhere(keep)
        radius = _equivalent_radius(total, sub, step)
        levels.append(
            Level(distance_mm=d, voxels=voxels, centroid_zyx=voxels.mean(axis=0),
                  radius_mm=radius, n_components=int(connected.size))
        )

        # --- leak ------------------------------------------------------------
        lookback = int(round(cfg.trace.leak_lookback_mm / step))
        if len(levels) > lookback:
            past = levels[-1 - lookback].radius_mm
            if radius > cfg.trace.leak_ratio * max(past, 1e-6):
                stop_reason = "leak"
                break

        # --- bifurcation ------------------------------------------------------
        if d < cfg.trace.stable_from_mm:
            pending_split = None
        elif not following_dominant:
            if connected.size >= 2:
                if pending_split is None:
                    pending_split = k
                elif k - pending_split >= persist_levels:
                    split_at = pending_split * step
                    stop_reason = "bifurcation"
                    if split_at >= cfg.trace.eligible_mm:
                        break
                    # The trunk divides before the seed distance. Follow the
                    # dominant child to reach 5 mm, as a human tracing a
                    # centreline would, and say so in the flags.
                    flags.append("seed_beyond_bifurcation")
                    following_dominant = True
                    dominant = connected[int(np.argmax(sizes))]
                    keep = labels == dominant
                    levels[-1] = Level(
                        distance_mm=d, voxels=np.argwhere(keep),
                        centroid_zyx=np.argwhere(keep).mean(axis=0),
                        radius_mm=_equivalent_radius(int(keep.sum()), sub, step),
                        n_components=1,
                    )
            else:
                pending_split = None
        elif connected.size >= 2:
            # Already following one child: stay with the larger sub-component.
            dominant = connected[int(np.argmax(sizes))]
            keep = labels == dominant
            voxels = np.argwhere(keep)
            levels[-1] = Level(
                distance_mm=d, voxels=voxels, centroid_zyx=voxels.mean(axis=0),
                radius_mm=_equivalent_radius(int(keep.sum()), sub, step), n_components=1,
            )

        if following_dominant and d >= cfg.trace.eligible_mm:
            break

        retained = keep

    # --- secondary radius check, diagnostic only ---------------------------
    # Wala's ratio test, kept as a diagnostic only: at this spacing the radius
    # estimate is too noisy to be a primary stop signal, which is why component
    # splitting is the primary one. A sharp drop without a topological split
    # usually means the trace slipped onto a neighbouring structure.
    lookback = int(round(cfg.trace.leak_lookback_mm / step))
    first = int(round(cfg.trace.stable_from_mm / step)) + lookback
    for k in range(first, len(levels)):
        if levels[k].radius_mm < cfg.trace.radius_ratio * levels[k - lookback].radius_mm:
            flags.append("radius_drop")
            break

    return levels, stop_reason, split_at, flags


def trace_candidate(
    candidate: Candidate,
    image_spline: np.ndarray,
    geometry: AortaGeometry,
    calibration: Calibration,
    grid: CaseGrid,
    cfg: Config,
) -> None:
    """Trace one candidate outward from its wall patch, in place."""
    if candidate.patch_points.shape[0] == 0:
        candidate.reject("empty_patch")
        return

    weights = np.maximum(candidate.patch_weights - calibration.t_bg, 1e-6)
    centre_roi = np.average(candidate.patch_points, axis=0, weights=weights)
    sub = extract_subvolume(centre_roi, image_spline, geometry.mask, grid, cfg)

    # --- geodesic source: the wall patch itself ------------------------------
    patch_sub = sub.from_roi(candidate.patch_points)
    idx = voxel_indices(patch_sub, sub.ct.shape)
    source = np.zeros(sub.ct.shape, dtype=bool)
    inside = ((patch_sub >= 0) & (patch_sub <= sub.n - 1)).all(axis=1)
    if not inside.any():
        candidate.reject("patch_outside_subvolume")
        return
    idx = idx[inside]
    source[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    if cfg.trace.source_dilate_voxels > 0:
        source = ndimage.binary_dilation(
            source, structure=NEIGHBOURHOOD_26, iterations=cfg.trace.source_dilate_voxels
        )

    # Excluding the aorta mask is essential: otherwise the field floods back
    # into the parent and every candidate connects to every other.
    t_propagate = calibration.t_lumen + cfg.trace.propagate_sigma_offset * calibration.sigma_ao
    lumen = (sub.ct > t_propagate) & (sub.ct < calibration.t_calcium) & ~sub.aorta
    region = lumen | (source & ~sub.aorta)
    source = source & region

    distance = geodesic_distance(region, source, sub.spacing_zyx)
    levels, stop_reason, split_at, flags = walk_levels(distance, source, sub, calibration, cfg)

    for flag in flags:
        candidate.flags.add(flag)
    candidate.stop_reason = stop_reason
    candidate.debug["subvolume_centre_roi"] = centre_roi

    if not levels:
        candidate.path = np.zeros((0, 3))
        candidate.path_radius_mm = np.zeros(0)
        candidate.path_distance_mm = np.zeros(0)
        candidate.path_length_mm = 0.0
        candidate.reached_mm = 0.0
        candidate.eligible = False
        return

    reached = levels[-1].distance_mm
    candidate.path = np.stack([sub.to_roi(lv.centroid_zyx[::-1]) for lv in levels])
    candidate.path_radius_mm = np.array([lv.radius_mm for lv in levels])
    candidate.path_distance_mm = np.array([lv.distance_mm for lv in levels])
    # The honest proximal-path length is the distance to the split, even when
    # the point list continues past it to reach the seed.
    candidate.path_length_mm = float(split_at if split_at is not None else reached)
    candidate.reached_mm = float(reached)
    candidate.eligible = bool(reached >= cfg.trace.eligible_mm and stop_reason != "leak")
    candidate.lumen_points = sub.voxels_to_roi(
        np.concatenate([lv.voxels for lv in levels], axis=0)
    )
    candidate.lumen_distance_mm = np.concatenate(
        [np.full(lv.voxels.shape[0], lv.distance_mm) for lv in levels]
    )
    candidate.debug["reached_mm"] = reached

    log.debug(
        "candidate %d: %d levels, reached %.1f mm, stop=%s, eligible=%s",
        candidate.id, len(levels), reached, stop_reason, candidate.eligible,
    )


def trace_all(
    candidates: List[Candidate],
    image_spline: np.ndarray,
    geometry: AortaGeometry,
    calibration: Calibration,
    grid: CaseGrid,
    cfg: Config,
) -> None:
    for candidate in candidates:
        if candidate.rejected_by is not None:
            continue
        try:
            trace_candidate(candidate, image_spline, geometry, calibration, grid, cfg)
        except Exception:                                  # noqa: BLE001
            log.exception("candidate %d failed during tracing", candidate.id)
            candidate.reject("exception")
        if candidate.rejected_by is None and not candidate.eligible:
            candidate.reject("not_eligible" if candidate.stop_reason != "leak" else "leak")
    kept = sum(1 for c in candidates if c.rejected_by is None)
    log.info("trace: %d of %d candidates eligible", kept, len(candidates))
