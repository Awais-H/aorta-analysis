"""Stage 6 - the four quantities the challenge actually asks for.

Ostium centre, initial direction, seed at 5 mm and local radius at that seed.
Between them these carry 40% of the score, and each has a specific definition
that a plausible-looking alternative would get systematically wrong.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree
from skimage import measure as skmeasure

from .config import Config
from .interp import orthonormal_basis, sample_volume, sample_volume_prefiltered
from .types import AortaGeometry, Calibration, Candidate, CaseGrid

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Ostium
# --------------------------------------------------------------------------- #

def project_to_surface(
    point_roi: np.ndarray,
    geometry: AortaGeometry,
    surface_tree: cKDTree,
    surface_xyz: np.ndarray,
    grid: CaseGrid,
    cfg: Config,
) -> np.ndarray:
    """Move a point onto the aortic surface along the local outward normal.

    The centroid of points sampled on a curved surface sits slightly inside it.
    A short bisection on the signed distance puts the ostium exactly on the
    wall - not inside, not outside - to well under a tenth of a millimetre.
    """
    _, nearest = surface_tree.query(point_roi * grid.roi_spacing_mm)
    voxel = surface_xyz[int(nearest)]
    normal = geometry.normals[
        :, int(round(voxel[2])), int(round(voxel[1])), int(round(voxel[0]))
    ].astype(np.float64)
    norm = np.linalg.norm(normal)
    if norm < 1e-6:
        return point_roi
    normal /= norm

    def signed_at(offset_mm: float) -> float:
        pt = point_roi + normal * offset_mm / grid.roi_spacing_mm
        return float(sample_volume(geometry.signed_edt_mm, pt, order=1))

    lo, hi = 0.0, 0.0
    start = signed_at(0.0)
    if start > 0:                       # inside: search outward for the zero
        hi = 1.0
        for _ in range(8):
            if signed_at(hi) <= 0:
                break
            lo, hi = hi, hi * 2.0
        else:
            return point_roi
    else:                               # outside: search inward
        lo = -1.0
        for _ in range(8):
            if signed_at(lo) >= 0:
                break
            hi, lo = lo, lo * 2.0
        else:
            return point_roi

    for _ in range(cfg.measure.surface_projection_iters + 5):
        mid = 0.5 * (lo + hi)
        if signed_at(mid) > 0:
            lo = mid
        else:
            hi = mid
    return point_roi + normal * (0.5 * (lo + hi)) / grid.roi_spacing_mm


def ostium_centre(
    candidate: Candidate,
    geometry: AortaGeometry,
    calibration: Calibration,
    surface_tree: cKDTree,
    surface_xyz: np.ndarray,
    grid: CaseGrid,
    cfg: Config,
) -> np.ndarray:
    """Intensity-weighted centroid of the wall contact patch, on the wall.

    Deliberately *not* a centreline node. Riffaud's branch location is the
    bifurcation node on the aortic centreline, one aortic radius inside the
    wall; with a median aortic radius around 9 mm that is a systematic ~9 mm
    error against a metric worth 25% of the score.
    """
    weights = np.maximum(candidate.patch_weights - calibration.t_bg, 1e-6)
    centroid = np.average(candidate.patch_points, axis=0, weights=weights)
    return project_to_surface(centroid, geometry, surface_tree, surface_xyz, grid, cfg)


# --------------------------------------------------------------------------- #
# Direction
# --------------------------------------------------------------------------- #

def initial_direction(
    path_roi: np.ndarray, ostium_roi: np.ndarray, grid: CaseGrid
) -> np.ndarray:
    """Riffaud's Definition 5, computed entirely in physical millimetres.

    A least-squares fit over all ~20 path samples, which beats an
    ostium-to-seed difference vector taken from two noisy endpoints.

    The sign term is what makes it work: an eigenvector's polarity is
    arbitrary, and without it roughly half the detections come back with a
    correctly-oriented line and a randomly-flipped arrow.

    Doing the whole computation in physical space is not optional. With
    non-identity direction cosines an index-space direction is simply a
    different vector, and normalising it hides the error.
    """
    points_mm = grid.to_physical(path_roi)
    ostium_mm = grid.to_physical(ostium_roi)
    a = points_mm - ostium_mm
    if a.shape[0] < 2:
        raise ValueError("need at least two path points for a direction")

    _, vectors = np.linalg.eigh(a.T @ a)
    e = vectors[:, -1]
    sign = np.sign(np.ones(a.shape[0]) @ a @ e)
    if sign == 0:
        sign = 1.0
    d = sign * e
    return d / max(np.linalg.norm(d), 1e-9)


# --------------------------------------------------------------------------- #
# Seed and radius
# --------------------------------------------------------------------------- #

def point_at_distance(candidate: Candidate, target_mm: float) -> Optional[np.ndarray]:
    """Interpolate along the traced path to an exact geodesic distance."""
    distances = candidate.path_distance_mm
    points = candidate.path
    if distances is None or points is None or distances.size == 0:
        return None
    if target_mm <= distances[0]:
        return points[0]
    if target_mm >= distances[-1]:
        return points[-1]
    upper = int(np.searchsorted(distances, target_mm))
    lower = upper - 1
    span = distances[upper] - distances[lower]
    frac = 0.0 if span <= 0 else (target_mm - distances[lower]) / span
    return points[lower] + frac * (points[upper] - points[lower])


def _cross_section(
    centre_roi: np.ndarray,
    direction_mm: np.ndarray,
    image_spline: np.ndarray,
    grid: CaseGrid,
    cfg: Config,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Resample a plane perpendicular to the branch at the given point."""
    e1, e2 = orthonormal_basis(direction_mm)
    half = cfg.measure.xsec_half_mm
    step = cfg.measure.xsec_spacing_mm
    axis = np.arange(-half, half + 0.5 * step, step)
    aa, bb = np.meshgrid(axis, axis, indexing="ij")
    offset_mm = aa[..., None] * e1 + bb[..., None] * e2
    pts = centre_roi + offset_mm / grid.roi_spacing_mm
    values = sample_volume_prefiltered(image_spline, pts, order=3)
    return values, aa, bb, pts


def _component_containing_centre(binary: np.ndarray) -> Optional[np.ndarray]:
    labels = skmeasure.label(binary, connectivity=2)
    centre = (labels.shape[0] // 2, labels.shape[1] // 2)
    label = labels[centre]
    if label == 0:
        # The centre pixel just missed; accept a component touching it.
        window = labels[max(centre[0] - 1, 0): centre[0] + 2,
                        max(centre[1] - 1, 0): centre[1] + 2]
        nonzero = window[window > 0]
        if nonzero.size == 0:
            return None
        label = int(np.bincount(nonzero).argmax())
    return labels == label


def seed_and_radius(
    candidate: Candidate,
    image_spline: np.ndarray,
    calibration: Calibration,
    grid: CaseGrid,
    cfg: Config,
) -> Tuple[Optional[np.ndarray], Optional[float]]:
    """The seed at 5 mm, recentred, and the FWHM radius on the same plane."""
    raw_seed = point_at_distance(candidate, cfg.measure.seed_mm)
    if raw_seed is None:
        return None, None
    direction_mm = grid.to_mm_direction(candidate.direction)

    values, aa, bb, pts = _cross_section(raw_seed, direction_mm, image_spline, grid, cfg)
    radial = np.hypot(aa, bb)

    peak_region = radial <= cfg.measure.peak_window_mm
    peak = float(values[peak_region].max()) if peak_region.any() else float(values.max())
    lo, hi = cfg.measure.bg_annulus_mm
    annulus = (radial >= lo) & (radial <= hi) & (values < calibration.t_calcium)
    background = float(np.median(values[annulus])) if annulus.any() else calibration.t_bg
    half_max = 0.5 * (peak + background)

    component = _component_containing_centre(values >= half_max)
    if component is None or not component.any():
        candidate.flags.add("seed_recentre_failed")
        return raw_seed, None

    # --- recentre: this is what guarantees the seed lies *on* the daughter ---
    weights = np.maximum(values[component] - background, 1e-6)
    seed = np.average(pts[component], axis=0, weights=weights)

    # --- FWHM radius ---------------------------------------------------------
    area = float(component.sum()) * cfg.measure.xsec_spacing_mm ** 2
    radius = float(np.sqrt(area / np.pi))
    candidate.debug["xsec_peak_hu"] = peak
    candidate.debug["xsec_background_hu"] = background
    candidate.debug["xsec_eccentricity"] = _eccentricity(component)
    candidate.debug["xsec_mean_hu"] = float(values[component].mean())
    return seed, radius


def _eccentricity(component: np.ndarray) -> float:
    """Major/minor axis ratio of the cross-section, a cheap tubularity proxy."""
    props = skmeasure.regionprops(component.astype(np.uint8))
    if not props:
        return 1.0
    minor = float(props[0].axis_minor_length)
    major = float(props[0].axis_major_length)
    return major / max(minor, 1e-6)


# --------------------------------------------------------------------------- #
# Display quantities
# --------------------------------------------------------------------------- #

def clock_position(theta: float, left_is_positive: bool) -> str:
    """Clock face with 12 o'clock anterior, running towards the patient's left.

    Defined in the *local vessel frame*, which diverges from the patient axial
    plane in the arch - noted in the display legend for exactly that reason.
    """
    angle = theta if left_is_positive else (2.0 * np.pi - theta)
    hours_total = (angle / (2.0 * np.pi)) * 12.0
    hours = int(np.floor(hours_total)) % 12
    minutes = int(round((hours_total - np.floor(hours_total)) * 60.0))
    if minutes == 60:
        minutes = 0
        hours = (hours + 1) % 12
    return f"{12 if hours == 0 else hours}:{minutes:02d}"


def _confidence(candidate: Candidate, calibration: Calibration, cfg: Config) -> float:
    """Composite score used to order the display and drive the flags.

    Not a rejection criterion: that decision belongs in Stage 7, where it can
    be tuned against F1.
    """
    span = max(calibration.mu_ao - calibration.t_lumen, 1e-6)
    peak_term = np.clip((candidate.peak_value - calibration.t_lumen) / span, 0.0, 1.0)
    length_term = np.clip(candidate.path_length_mm / cfg.trace.max_mm, 0.0, 1.0)

    taper_term = 1.0
    radii = candidate.path_radius_mm
    if radii is not None and radii.size > 2:
        first = float(radii[min(1, radii.size - 1)])
        last = float(radii[-1])
        taper_term = float(np.clip(1.0 - abs(last - first) / max(first, 1e-6), 0.0, 1.0))
    return float(np.clip(peak_term * length_term * taper_term, 0.0, 1.0))


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def measure_all(
    candidates: List[Candidate],
    image_spline: np.ndarray,
    geometry: AortaGeometry,
    calibration: Calibration,
    wall_theta: np.ndarray,
    grid: CaseGrid,
    cfg: Config,
) -> None:
    surface_voxels = np.argwhere(geometry.surface)
    if surface_voxels.size == 0:
        for candidate in candidates:
            candidate.reject("no_surface")
        return
    surface_xyz = surface_voxels[:, ::-1].astype(np.float64)
    surface_tree = cKDTree(surface_xyz * grid.roi_spacing_mm)

    mid = geometry.n_samples // 2
    v_physical = grid.rotate_to_physical(geometry.frame_v[mid])
    left_is_positive = bool(v_physical[0] >= 0.0)     # +x is the patient's left in LPS

    for candidate in candidates:
        if candidate.rejected_by is not None:
            continue
        try:
            _measure_one(candidate, image_spline, geometry, calibration, wall_theta,
                         surface_tree, surface_xyz, left_is_positive, grid, cfg)
        except Exception:                              # noqa: BLE001
            log.exception("candidate %d failed during measurement", candidate.id)
            candidate.reject("exception")


def _measure_one(
    candidate, image_spline, geometry, calibration, wall_theta,
    surface_tree, surface_xyz, left_is_positive, grid, cfg,
) -> None:
    candidate.ostium_xyz_roi = ostium_centre(
        candidate, geometry, calibration, surface_tree, surface_xyz, grid, cfg
    )

    # Initial direction, so only the truncated proximal path. The seed distance
    # is a floor: a trunk that divides at 2 mm still needs enough points to fit.
    limit = max(candidate.path_length_mm, cfg.measure.seed_mm)
    keep = candidate.path_distance_mm <= limit + 1e-6
    proximal = candidate.path[keep]
    if proximal.shape[0] < 2:
        proximal = candidate.path
    candidate.direction = initial_direction(proximal, candidate.ostium_xyz_roi, grid)

    seed, radius = seed_and_radius(candidate, image_spline, calibration, grid, cfg)
    if seed is None:
        candidate.reject("no_seed")
        return
    candidate.seed_xyz_roi = seed

    row, col = candidate.peak_ij
    candidate.arc_length_mm = float(geometry.arc_length[row])
    candidate.parent_radius_mm = float(geometry.radius_mm[row])
    candidate.clock_position = clock_position(float(wall_theta[col]), left_is_positive)

    if radius is None:
        candidate.radius_mm = None
        candidate.flags.add("radius_unavailable")
        candidate.reject("no_radius")
        return

    clamped = float(np.clip(radius, cfg.measure.radius_min_mm, cfg.measure.radius_max_mm))
    if abs(clamped - radius) > 1e-9:
        candidate.flags.add("radius_out_of_bounds")
    if clamped > cfg.measure.radius_parent_frac * candidate.parent_radius_mm:
        # A daughter as wide as its parent is a leak that escaped Stage 5.
        candidate.flags.add("radius_near_parent")
    candidate.radius_mm = clamped
    candidate.confidence = _confidence(candidate, calibration, cfg)
