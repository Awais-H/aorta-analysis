"""Stage 4b - candidate generation from the unrolled map.

Genuine 2D local-maximum detection, not Elattar's 1D max projections: they
collapse each axis and read two peaks off one curve because there are exactly
two coronary ostia at roughly one level. Here the branch count is unknown and
the branches sit at many levels, so projecting would smear them together.

The map is cyclic in theta, so every neighbourhood operation runs on a
circularly padded copy. A branch at 12 o'clock must yield one candidate, not
two half-candidates at either edge.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np
from scipy import ndimage
from skimage import segmentation

from .config import Config
from .types import AortaGeometry, Calibration, Candidate, WallMap

log = logging.getLogger(__name__)


def _circular_pad(arr: np.ndarray, pad: int) -> np.ndarray:
    return np.concatenate([arr[:, -pad:], arr, arr[:, :pad]], axis=1)


def map_threshold(calibration: Calibration, cfg: Config) -> float:
    """Detection threshold, derived from this patient's own contrast level."""
    return calibration.t_lumen + cfg.detect.threshold_sigma_offset * calibration.sigma_ao


def _pixel_distance_mm(
    a: Tuple[int, int], b: Tuple[int, int], geometry: AortaGeometry,
    wall_map: WallMap, n_bins: int
) -> float:
    """Physical distance between two map pixels: arc length one way, arc along
    the circumference the other."""
    d_arc = float(geometry.arc_length[a[0]] - geometry.arc_length[b[0]])
    dj = abs(a[1] - b[1]) % n_bins
    dj = min(dj, n_bins - dj)
    radius = np.nanmean([wall_map.wall_radius_mm[a], wall_map.wall_radius_mm[b]])
    if not np.isfinite(radius):
        radius = float(np.nanmedian(geometry.radius_mm))
    d_circ = radius * (2.0 * np.pi * dj / n_bins)
    return float(np.hypot(d_arc, d_circ))


def _split_by_watershed(
    region_mask: np.ndarray, intensity: np.ndarray, maxima: List[Tuple[int, int]]
) -> np.ndarray:
    """Watershed a component that holds two well-separated maxima.

    This is the paired-renal case. Riffaud's 239-case series gives a mean
    inter-renal origin distance of 9.5 mm with SD 5.1, so pairs routinely sit
    close enough to merge under naive labelling - and the challenge requires
    them back as two instances.
    """
    markers = np.zeros(region_mask.shape, dtype=np.int32)
    for n, (i, j) in enumerate(maxima, start=1):
        markers[i, j] = n
    filled = np.where(np.isfinite(intensity), intensity, np.nanmin(intensity))
    return segmentation.watershed(-filled, markers, mask=region_mask)


def detect_candidates(
    wall_map: WallMap,
    geometry: AortaGeometry,
    calibration: Calibration,
    cfg: Config,
) -> List[Candidate]:
    """Suprathreshold connected components containing a local maximum."""
    intensity = wall_map.intensity
    n_rows, n_bins = intensity.shape
    pad = max(n_bins // 8, 1)
    threshold = map_threshold(calibration, cfg)

    padded = _circular_pad(intensity, pad)
    padded_valid = _circular_pad(wall_map.valid, pad)
    # Fill invalid pixels below threshold so the maximum filter behaves; the
    # validity mask still rejects any region whose peak lands on one.
    filled = np.where(np.isfinite(padded), padded, calibration.t_bg)

    footprint = np.ones(tuple(cfg.detect.peak_footprint), dtype=bool)
    local_max = (filled == ndimage.maximum_filter(filled, footprint=footprint)) & padded_valid
    above = (filled > threshold) & padded_valid
    labels, n_labels = ndimage.label(above, structure=np.ones((3, 3), dtype=bool))

    seen: Dict[Tuple, int] = {}
    candidates: List[Candidate] = []

    for label in range(1, n_labels + 1):
        pixels = np.argwhere(labels == label)
        if pixels.shape[0] < cfg.detect.min_region_px:
            continue
        peaks = [tuple(p) for p in pixels if local_max[p[0], p[1]]]
        if not peaks:
            continue

        groups = [(pixels, peaks)]
        if len(peaks) > 1:
            groups = _maybe_split(pixels, peaks, labels == label, filled, geometry,
                                  wall_map, cfg, pad, n_bins)

        for group_pixels, group_peaks in groups:
            unwrapped = _unwrap(group_pixels, pad, n_bins)
            key = tuple(sorted(map(tuple, unwrapped.tolist())))
            if key in seen:
                continue                      # the padded copy of a component
            seen[key] = 1
            best = max(group_peaks, key=lambda p: filled[p[0], p[1]])
            peak_ij = (int(best[0]), int((best[1] - pad) % n_bins))
            candidates.append(_make_candidate(len(candidates), unwrapped, peak_ij, wall_map))

    log.info("detection: %d candidate regions above %.0f HU", len(candidates), threshold)
    return candidates


def _unwrap(pixels: np.ndarray, pad: int, n_bins: int) -> np.ndarray:
    out = pixels.copy()
    out[:, 1] = (out[:, 1] - pad) % n_bins
    return np.unique(out, axis=0)


def _maybe_split(pixels, peaks, region_mask, filled, geometry, wall_map, cfg, pad, n_bins):
    """Split a component whose maxima are further apart than a branch is wide."""
    orig = [(int(p[0]), int((p[1] - pad) % n_bins)) for p in peaks]
    far = [
        (peaks[a], peaks[b])
        for a in range(len(peaks)) for b in range(a + 1, len(peaks))
        if _pixel_distance_mm(orig[a], orig[b], geometry, wall_map, n_bins)
        > cfg.detect.split_distance_mm
    ]
    if not far:
        return [(pixels, peaks)]

    # Keep only maxima that are mutually well separated, so noise-level maxima
    # a pixel apart do not each seed a basin.
    kept: List[Tuple[int, int]] = []
    for peak, peak_orig in sorted(zip(peaks, orig), key=lambda t: -filled[t[0][0], t[0][1]]):
        if all(
            _pixel_distance_mm(peak_orig, (int(k[0]), int((k[1] - pad) % n_bins)),
                               geometry, wall_map, n_bins) > cfg.detect.split_distance_mm
            for k in kept
        ):
            kept.append(peak)
    if len(kept) < 2:
        return [(pixels, peaks)]

    basins = _split_by_watershed(region_mask, filled, kept)
    groups = []
    for n, peak in enumerate(kept, start=1):
        sub = np.argwhere(basins == n)
        if sub.shape[0] >= cfg.detect.min_region_px:
            groups.append((sub, [peak]))
    log.debug("split a component into %d basins", len(groups))
    return groups or [(pixels, peaks)]


def _make_candidate(index: int, pixels: np.ndarray, peak_ij, wall_map: WallMap) -> Candidate:
    rows, cols = pixels[:, 0], pixels[:, 1]
    weights = wall_map.intensity[rows, cols].astype(np.float64)
    points = wall_map.wall_point[rows, cols].astype(np.float64)
    finite = np.isfinite(weights) & np.isfinite(points).all(axis=1)
    return Candidate(
        id=index,
        map_region=pixels,
        peak_ij=peak_ij,
        peak_value=float(wall_map.intensity[peak_ij]),
        patch_points=points[finite],
        patch_weights=weights[finite],
    )
