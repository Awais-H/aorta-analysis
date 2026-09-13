"""Stage 7 - instance resolution. This is where the 45% is won or lost.

Rules are applied in order and each sets ``rejected_by`` rather than deleting
the candidate, so the sweep harness can attribute a lost detection to a named
rule. That breakdown is the most actionable diagnostic in the project.
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np
from scipy import ndimage

from .config import Config
from .interp import sample_nearest, sample_volume_prefiltered, voxel_indices
from .types import AortaGeometry, Calibration, Candidate, CaseGrid

log = logging.getLogger(__name__)


def merge_by_contiguity(candidates: List[Candidate]) -> None:
    """Rule 1 - merge only when wall patches are contiguous on eligible wall.

    Stage 4 already labels the map's suprathreshold regions with circular
    padding and splits genuinely separate maxima by watershed, so contiguity is
    settled there: do not merge what Stage 4 separated, and do not re-merge
    what its watershed split.

    The rule that matters here is the one that is *absent*. Never merge on
    centroid proximity: the challenge requires two nearby origins back as two
    instances when they are separate at the aortic wall, and Riffaud's 239-case
    series puts the mean inter-renal origin distance at 9.5 mm with SD 5.1, so
    any proximity merge with a sensible radius silently halves renal recall.

    The common-trunk case needs no rule either - the Stage 5 bifurcation stop
    terminates the trace at the split while keeping exactly one ostium.
    """
    regions = [tuple(sorted(map(tuple, c.map_region.tolist()))) for c in candidates]
    if len(set(regions)) != len(regions):
        log.warning("duplicate map regions survived detection; this should not happen")


def reject_secondary_branches(
    candidates: List[Candidate], geometry: AortaGeometry, grid: CaseGrid, cfg: Config
) -> None:
    """Rule 2 - a vessel arising from another daughter is not a direct daughter.

    Candidates are processed in descending wall-patch area so the true parent
    is established first; ties break on confidence.
    """
    claimed = np.zeros(geometry.mask.shape, dtype=bool)
    dilate = max(int(round(cfg.resolve.claimed_dilate_mm / float(grid.roi_spacing_mm.min()))), 1)

    order = sorted(
        (c for c in candidates if c.rejected_by is None),
        key=lambda c: (-c.patch_points.shape[0], -c.confidence, c.id),
    )
    for candidate in order:
        if candidate.lumen_points is None or candidate.lumen_points.shape[0] == 0:
            continue
        proximal = candidate.lumen_points[
            candidate.lumen_distance_mm <= cfg.resolve.secondary_probe_mm
        ]
        if proximal.shape[0] and claimed.any():
            overlap = float(sample_nearest(claimed, proximal).mean())
            if overlap > cfg.resolve.secondary_overlap_frac:
                candidate.reject("secondary_branch")
                candidate.debug["secondary_overlap"] = overlap
                continue

        idx = voxel_indices(candidate.lumen_points, claimed.shape)
        mine = np.zeros_like(claimed)
        mine[idx[:, 0], idx[:, 1], idx[:, 2]] = True
        claimed |= ndimage.binary_dilation(mine, iterations=dilate)


def reject_non_arterial(
    candidates: List[Candidate],
    image_spline: np.ndarray,
    calibration: Calibration,
    grid: CaseGrid,
    cfg: Config,
) -> None:
    """Rule 3 - three tests, all of which must pass.

    In a well-timed CTA the arterial/venous HU gap does most of the work. In a
    poorly timed one it will not, which is why degenerate cases are flagged
    rather than silently trusted.
    """
    lo, hi = cfg.resolve.contact_probe_mm
    depths = np.linspace(lo, hi, 5)

    for candidate in candidates:
        if candidate.rejected_by is not None or candidate.ostium_xyz_roi is None:
            continue

        # --- contact brightness: a vein abutting the aorta has no bright
        # --- connection *at the wall*.
        direction_mm = grid.to_mm_direction(candidate.direction)
        probe = candidate.ostium_xyz_roi + np.outer(depths, direction_mm) / grid.roi_spacing_mm
        contact = float(np.median(sample_volume_prefiltered(image_spline, probe, order=3)))
        candidate.debug["contact_hu"] = contact
        if contact < calibration.t_lumen:
            candidate.reject("dim_contact")
            continue

        # --- HU band around the parent's own contrast level
        mean_hu = candidate.debug.get("xsec_mean_hu")
        if mean_hu is None and candidate.lumen_points is not None:
            mean_hu = float(
                sample_volume_prefiltered(image_spline, candidate.lumen_points, order=3).mean()
            )
        if mean_hu is not None:
            band = cfg.resolve.hu_band_sigma * calibration.sigma_ao
            candidate.debug["lumen_mean_hu"] = mean_hu
            if abs(mean_hu - calibration.mu_ao) > band:
                candidate.reject("hu_out_of_band")
                continue

        # --- tubularity
        eccentricity = candidate.debug.get("xsec_eccentricity")
        if eccentricity is not None and eccentricity > cfg.resolve.eccentricity_max:
            candidate.reject("not_tubular")


def reject_artefactual_bumps(candidates: List[Candidate], cfg: Config) -> None:
    """Rule 4 - the aorta is not a perfect cylinder.

    Riffaud's feature space: traced length against length divided by the local
    parent radius, where real arteries and surface artefacts separate close to
    linearly. Their own constants are not usable here - a test of "length < 3r
    or length < r + 20 mm" was tuned for named major arteries and would delete
    essentially every eligible daughter under a 5 mm rule. Relative length is
    the discriminative axis: a bump on a 15 mm-radius aorta and one on a 6 mm
    aorta are not comparable in absolute millimetres.

    The feature is how far the lumen was *followable*, not the length reported
    after truncating at a bifurcation. A branch that divides at 5.5 mm is
    strong evidence of a real vessel; scoring it as a 5.5 mm stub would let
    this rule undo the eligibility test that the candidate already passed.
    """
    w_len, w_rel, bias = cfg.resolve.bump_boundary
    for candidate in candidates:
        if candidate.rejected_by is not None:
            continue
        length = max(candidate.reached_mm, candidate.path_length_mm)
        relative = length / max(candidate.parent_radius_mm, 1e-6)
        score = w_len * length + w_rel * relative + bias
        candidate.debug["bump_score"] = float(score)
        if score < 0.0:
            candidate.reject("artefactual_bump")


def assign_ids(candidates: List[Candidate], geometry: AortaGeometry) -> List[Candidate]:
    """Order by arc length along the centreline, not by z.

    Superior-to-inferior ordering is ambiguous across an arch; arc length is
    well defined for any coverage and makes IDs stable across runs.
    """
    accepted = [c for c in candidates if c.rejected_by is None]
    accepted.sort(key=lambda c: (c.arc_length_mm, c.peak_ij[1], c.id))
    for n, candidate in enumerate(accepted, start=1):
        candidate.instance_id = f"branch_{n:03d}"
    return accepted


def resolve(
    candidates: List[Candidate],
    image_spline: np.ndarray,
    geometry: AortaGeometry,
    calibration: Calibration,
    grid: CaseGrid,
    cfg: Config,
) -> List[Candidate]:
    before = sum(1 for c in candidates if c.rejected_by is None)
    merge_by_contiguity(candidates)
    reject_secondary_branches(candidates, geometry, grid, cfg)
    reject_non_arterial(candidates, image_spline, calibration, grid, cfg)
    reject_artefactual_bumps(candidates, cfg)
    accepted = assign_ids(candidates, geometry)
    log.info("resolution: %d of %d candidates accepted", len(accepted), before)
    for reason, count in rejection_counts(candidates).items():
        log.info("  rejected by %s: %d", reason, count)
    return accepted


def rejection_counts(candidates: List[Candidate]) -> dict:
    counts: dict = {}
    for candidate in candidates:
        if candidate.rejected_by is not None:
            counts[candidate.rejected_by] = counts.get(candidate.rejected_by, 0) + 1
    return dict(sorted(counts.items()))
