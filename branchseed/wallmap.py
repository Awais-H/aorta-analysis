"""Stage 4a - the unrolled aortic wall map.

Elattar's cylindrical intensity map for coronary ostia, stripped of its
anatomical prior. Rows are arc length along the aortic centreline, columns are
angle around it; each pixel holds the mean CT intensity along a short outward
probe starting just beyond the aortic surface.

This is the working representation the detector runs on *and* the clinician
display, which is the point: the reader sees the same picture the algorithm
saw.
"""

from __future__ import annotations

import logging
import warnings
from typing import Tuple

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from .config import Config
from .interp import in_bounds, prefilter_volume, sample_nearest, sample_volume_prefiltered
from .types import AortaGeometry, Calibration, CaseGrid, WallMap

log = logging.getLogger(__name__)

_MARCH_BLOCK = 24        # ray-march steps evaluated per batch
_PROBE_BLOCK = 48        # centreline samples probed per batch


def ray_directions(geometry: AortaGeometry, angular_bins: int) -> Tuple[np.ndarray, np.ndarray]:
    """``(theta, w)`` with ``w`` shaped (N, B, 3), unit vectors in mm-space."""
    theta = np.linspace(0.0, 2.0 * np.pi, angular_bins, endpoint=False)
    cos_t = np.cos(theta)[None, :, None]
    sin_t = np.sin(theta)[None, :, None]
    w = cos_t * geometry.frame_u[:, None, :] + sin_t * geometry.frame_v[:, None, :]
    return theta, w


def _march_to_wall(
    mask: np.ndarray, geometry: AortaGeometry, w: np.ndarray, grid: CaseGrid, cfg: Config
) -> Tuple[np.ndarray, np.ndarray]:
    """First wall crossing along each ray.

    The crossing is the last in-mask sample before a run of at least
    ``wall_gap_mm`` of background; requiring the run suppresses single-voxel
    holes in the mask, which would otherwise stop every ray early.
    """
    step = cfg.wallmap.ray_step_mm
    ray_max = cfg.wallmap.ray_max_factor * geometry.radius_mm
    n_steps = int(np.ceil(float(ray_max.max()) / step)) + 1
    gap_steps = max(int(round(cfg.wallmap.wall_gap_mm / step)), 1)

    n, b = w.shape[0], w.shape[1]
    hit = np.zeros((n_steps + gap_steps, n, b), dtype=bool)
    centre = geometry.centreline[:, None, :]

    for start in range(0, n_steps, _MARCH_BLOCK):
        stop = min(start + _MARCH_BLOCK, n_steps)
        distances = np.arange(start, stop, dtype=np.float64) * step
        # (S, N, B, 3): centre + w * distance, converted to index units
        pts = centre[None] + (w[None] * distances[:, None, None, None]) / grid.roi_spacing_mm
        inside = sample_nearest(mask, pts) & in_bounds(pts, mask.shape)
        hit[start:stop] = inside

    # A run of background of at least gap_steps immediately after the sample.
    clear = np.ones_like(hit)
    for k in range(1, gap_steps + 1):
        clear[: n_steps] &= ~hit[k: n_steps + k]

    steps_axis = np.arange(n_steps)[:, None, None]
    in_range = steps_axis * step <= ray_max[None, :, None]
    crossing = hit[:n_steps] & clear[:n_steps] & in_range

    found = crossing.any(axis=0)
    index = np.argmax(crossing, axis=0)
    wall_radius = np.where(found, index * step, np.nan).astype(np.float32)
    return wall_radius, found


def _probe_offsets(w: np.ndarray, tangent: np.ndarray, radius_mm: float, n_offsets: int):
    """``n_offsets`` points on a disc of the given radius perpendicular to each ray."""
    e1 = np.broadcast_to(tangent[:, None, :], w.shape)
    e2 = np.cross(w, e1)
    e2 /= np.maximum(np.linalg.norm(e2, axis=-1, keepdims=True), 1e-9)
    angles = np.linspace(0.0, 2.0 * np.pi, n_offsets, endpoint=False)
    disc = [np.zeros_like(w)]
    for ang in angles:
        disc.append(radius_mm * (np.cos(ang) * e1 + np.sin(ang) * e2))
    return np.stack(disc, axis=2)          # (N, B, 1 + n_offsets, 3)


def _probe_outward(spline, shape, geometry, w, safe_radius, calibration, grid, cfg):
    """Mean peri-aortic intensity per map pixel, and the calcium fraction.

    One batched ``map_coordinates`` call per block of centreline samples: a
    per-point loop would dominate the runtime, while a single call over the
    whole map would allocate a coordinate array that grows without bound with
    the length of the supplied segment. Blocking keeps peak memory flat.
    """
    n, b = w.shape[0], w.shape[1]
    offsets = _probe_offsets(
        w, geometry.tangent, cfg.wallmap.probe_radius_mm, cfg.wallmap.probe_disc_samples
    )
    depths = np.linspace(
        cfg.wallmap.probe_inner_mm, cfg.wallmap.probe_outer_mm, cfg.wallmap.probe_samples
    )
    intensity = np.empty((n, b), dtype=np.float32)
    calcium_fraction = np.empty((n, b), dtype=np.float32)

    for start in range(0, n, _PROBE_BLOCK):
        stop = min(start + _PROBE_BLOCK, n)
        block = slice(start, stop)
        # (rows, B, P, D, 3): the displacement is assembled in mm-space and
        # converted to index units exactly once, before it is added to the
        # index-space centreline.
        along = (safe_radius[block, :, None] + depths[None, None, :])[..., None] \
            * w[block, :, None, :]
        offset_mm = along[:, :, :, None, :] + offsets[block, :, None, :, :]
        probe_idx = geometry.centreline[block, None, None, None, :] \
            + offset_mm / grid.roi_spacing_mm

        values = sample_volume_prefiltered(spline, probe_idx, order=3)
        values = np.where(in_bounds(probe_idx, shape), values, np.nan)

        # Deliberate NaN arithmetic: probe points outside the ROI and samples
        # dropped as calcium are meant to propagate as NaN and be excluded.
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            disc_mean = np.nanmean(values, axis=3)
            # Calcium sits exactly where we are looking for ostia and is
            # brighter than contrast; without masking it, mural plaque
            # produces confident false positives.
            calcium = disc_mean > calibration.t_calcium
            intensity[block] = np.nanmean(np.where(calcium, np.nan, disc_mean), axis=2)
        calcium_fraction[block] = calcium.mean(axis=2)

    return intensity, calcium_fraction


def build_wall_map(
    image: np.ndarray,
    geometry: AortaGeometry,
    calibration: Calibration,
    grid: CaseGrid,
    cfg: Config,
    image_spline: np.ndarray | None = None,
) -> WallMap:
    """Construct the (N, B) unrolled map.

    ``image_spline`` is the cubic spline prefilter of ``image``. The prefilter
    is the expensive half of a cubic interpolation on a multi-million-voxel
    array, so the pipeline computes it once and shares it with the trace and
    the seed cross-sections; pass None and it is computed here.
    """
    if image_spline is None:
        image_spline = prefilter_volume(image, order=3)
    theta, w = ray_directions(geometry, cfg.wallmap.angular_bins)
    n, b = w.shape[0], w.shape[1]

    wall_radius, found = _march_to_wall(geometry.mask, geometry, w, grid, cfg)
    safe_radius = np.where(found, wall_radius, 0.0)
    wall_point = (
        geometry.centreline[:, None, :] + (w * safe_radius[..., None]) / grid.roi_spacing_mm
    ).astype(np.float32)

    valid = found.copy()

    # --- the ray must still belong to this part of the vessel ----------------
    # On the inner curvature of an arch, normal planes from different samples
    # intersect and a ray crosses into a different segment of the aorta.
    pts_mm = geometry.centreline * grid.roi_spacing_mm
    tree = cKDTree(pts_mm)
    _, nearest = tree.query(wall_point.reshape(-1, 3).astype(np.float64) * grid.roi_spacing_mm)
    arc_at_wall = geometry.arc_length[nearest].reshape(n, b)
    arc_self = geometry.arc_length[:, None]
    frame_ok = np.abs(arc_at_wall - arc_self) <= cfg.wallmap.frame_validate_arc_mm
    valid &= frame_ok

    # --- the wall point must sit on eligible wall, not on an end cap ---------
    excluded = geometry.surface & ~geometry.eligible_wall
    eligible_near = ndimage.binary_dilation(geometry.eligible_wall, iterations=1)
    on_eligible = sample_nearest(eligible_near, wall_point)
    on_excluded = sample_nearest(excluded, wall_point)
    valid &= on_eligible & ~on_excluded

    # --- probe outward -------------------------------------------------------
    intensity, calcium_fraction = _probe_outward(
        image_spline, image.shape, geometry, w, safe_radius, calibration, grid, cfg
    )

    valid &= calcium_fraction <= cfg.wallmap.calcium_reject_frac
    valid &= np.isfinite(intensity)

    intensity = np.where(valid, intensity, np.nan).astype(np.float32)
    wall_radius = np.where(valid, wall_radius, np.nan).astype(np.float32)

    invalid_frac = 1.0 - float(valid.mean())
    log.info(
        "wall map %d x %d: %.1f%% invalid (no crossing %.1f%%, frame %.1f%%, "
        "end cap %.1f%%, calcium %.1f%%)",
        n, b, 100 * invalid_frac, 100 * (1 - found.mean()), 100 * (1 - frame_ok.mean()),
        100 * (1 - (on_eligible & ~on_excluded).mean()),
        100 * float((calcium_fraction > cfg.wallmap.calcium_reject_frac).mean()),
    )
    if invalid_frac > 0.20:
        log.warning("over 20%% of wall-map pixels are invalid; check the frame and ray casting")

    return WallMap(
        intensity=intensity, wall_radius_mm=wall_radius, wall_point=wall_point,
        valid=valid, theta=theta,
    )
