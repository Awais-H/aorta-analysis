"""Stage 2 - HU statistics derived from the aorta mask.

The mask is a calibration source, not just an anchor. It tells us this
patient's contrast level, so almost every threshold downstream is derived from
it rather than hard-coded. A fixed threshold fitted on one scanner and contrast
timing is the failure mode Elattar names as their own unvalidated limitation.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy import ndimage

from .config import Config
from .types import Calibration, CaseGrid, Flags

log = logging.getLogger(__name__)


def euclidean_distance_mm(mask_zyx: np.ndarray, grid: CaseGrid) -> np.ndarray:
    """Distance from every voxel to the nearest background voxel, in mm.

    ``sampling`` takes spacing in array order, which is [z, y, x].
    """
    return ndimage.distance_transform_edt(
        mask_zyx, sampling=grid.spacing_zyx
    ).astype(np.float32)


def signed_distance_mm(mask_zyx: np.ndarray, grid: CaseGrid) -> np.ndarray:
    """Positive inside the mask, negative outside, zero at the surface."""
    inside = ndimage.distance_transform_edt(mask_zyx, sampling=grid.spacing_zyx)
    outside = ndimage.distance_transform_edt(~mask_zyx, sampling=grid.spacing_zyx)
    return (inside - outside).astype(np.float32)


def calibrate(
    image_zyx: np.ndarray,
    mask_zyx: np.ndarray,
    edt_mm: np.ndarray,
    grid: CaseGrid,
    cfg: Config,
    flags: Flags,
) -> Calibration:
    """Derive the intensity thresholds the rest of the pipeline runs on."""
    interior = edt_mm > cfg.calibrate.erosion_mm
    n_interior = int(interior.sum())

    if n_interior < cfg.calibrate.min_interior_voxels:
        # A very thin or short mask. Better a noisier calibration than none.
        log.warning("only %d interior voxels after %.1f mm erosion; using the whole mask",
                    n_interior, cfg.calibrate.erosion_mm)
        flags.add("thin_mask_calibration")
        interior = mask_zyx
        n_interior = int(interior.sum())

    hu = image_zyx[interior]
    mu = float(hu.mean())
    sigma = float(hu.std())
    t_lumen = float(np.percentile(hu, cfg.calibrate.lumen_percentile))
    t_calcium = mu + cfg.calibrate.calcium_sigma * sigma

    degenerate = False
    if mu > 1e-6 and sigma / abs(mu) > cfg.calibrate.max_sigma_ratio:
        # Thrombus in the mask, poor contrast timing, a leaked mask, or a
        # non-contrast scan. An over-permissive threshold with a flag beats a
        # crash, so widen rather than abort.
        log.warning("degenerate interior distribution: mu=%.0f sigma=%.0f (ratio %.2f)",
                    mu, sigma, sigma / abs(mu))
        flags.add("degenerate_calibration")
        degenerate = True
        t_lumen = mu - 1.5 * sigma

    if mu < cfg.calibrate.min_contrast_hu:
        log.warning("mean interior HU %.0f suggests no arterial contrast; expect poor results", mu)
        flags.add("no_arterial_contrast")

    t_bg = _background_level(image_zyx, mask_zyx, edt_mm, t_lumen, cfg)

    cal = Calibration(
        mu_ao=mu, sigma_ao=sigma, t_lumen=t_lumen, t_calcium=t_calcium, t_bg=t_bg,
        n_interior_voxels=n_interior, degenerate=degenerate,
    )
    log.info("calibration: mu=%.0f sigma=%.0f t_lumen=%.0f t_calcium=%.0f t_bg=%.0f (n=%d)",
             mu, sigma, t_lumen, t_calcium, t_bg, n_interior)
    return cal


def _background_level(
    image_zyx: np.ndarray,
    mask_zyx: np.ndarray,
    edt_mm: np.ndarray,
    t_lumen: float,
    cfg: Config,
) -> float:
    """Median HU of the dark perivascular tissue just outside the aorta."""
    shell = (~mask_zyx) & (image_zyx < t_lumen)
    # Restrict to a band close to the wall rather than the whole ROI, so lung
    # or air far from the aorta does not drag the estimate down.
    outside = ndimage.distance_transform_edt(~mask_zyx)
    shell &= outside <= max(cfg.roi.dilation_mm / 3.0, 3.0)
    if not shell.any():
        return float(image_zyx.min())
    return float(np.median(image_zyx[shell]))
