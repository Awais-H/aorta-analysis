"""D1 + D2: mask cleanup, adaptive HU threshold, 15 mm search shell, ring-removing opening.

Ported from triage.py: mask components (lines 84 to 90), adaptive threshold (96 to 101), shell and
opening (136 to 145). The crop and resample live in io_utils; this module calls them.

Output contract (SPEC.md D9): working CT (float32, ISO mm), working mask (largest component),
the raw thresholded shell (bool, bright voxels in the 15 mm shell), the opened array (bool, the
same after the 0.8 mm opening; used ONLY to define the wall layer, D2), distance-from-mask array
(mm), spacing, the working SimpleITK image handle, the mask fragment log, the threshold.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import SimpleITK as sitk
from scipy import ndimage

import config
import io_utils

log = logging.getLogger("branchseed.candidates")


@dataclass
class Candidates:
    ct: np.ndarray            # float32 (z, y, x) HU on the working grid
    mask: np.ndarray          # bool, largest connected component of the supplied mask
    bright_shell: np.ndarray  # bool, raw thresholded shell: what growth (D3) and tracing (D5) consume
    opened: np.ndarray        # bool, bright_shell after the opening: defines the wall layer only (D2, D3)
    distance_mm: np.ndarray   # float32, Euclidean distance from the mask in mm (0 inside)
    spacing: np.ndarray       # (3,) zyx mm
    image: sitk.Image         # working image handle, the argument for io_utils.index_to_mm
    threshold_hu: float
    fragment_log: dict
    hu_stats: dict
    grid_info: dict = field(default_factory=dict)


def clean_mask(mask) -> tuple[np.ndarray, dict]:
    """Keep the largest connected component; log fragments; warn loudly on a big discard (D2)."""
    m = np.asarray(mask) > 0
    lab, n = ndimage.label(m)
    frag = {"components": int(n), "fragment_sizes": [], "discarded_voxels": 0,
            "discarded_fraction": 0.0, "warning": False}
    if n <= 1:
        return m, frag
    sizes = ndimage.sum(m, lab, range(1, n + 1))
    keep = int(np.argmax(sizes)) + 1
    discarded = int(sizes.sum() - sizes.max())
    frag["fragment_sizes"] = sorted([int(s) for i, s in enumerate(sizes) if i + 1 != keep], reverse=True)[:20]
    frag["discarded_voxels"] = discarded
    frag["discarded_fraction"] = float(discarded / sizes.sum())
    if frag["discarded_fraction"] > config.MASK_FRAGMENT_WARN_FRACTION:
        frag["warning"] = True
        log.warning("mask has %d components; discarding %d voxels (%.1f%% of the mask): possible mislabelled blob",
                    n, discarded, 100 * frag["discarded_fraction"])
    else:
        log.info("mask has %d components; discarding %d stray voxels (%.2f%%)", n, discarded, 100 * frag["discarded_fraction"])
    return lab == keep, frag


def adaptive_threshold(ct: np.ndarray, mask: np.ndarray) -> tuple[float, dict]:
    """threshold = max(mean - k std, f x median, floor) over HU inside the mask (D1)."""
    hu = ct[mask]
    mean, std, med = float(hu.mean()), float(hu.std()), float(np.median(hu))
    thr = max(mean - config.THRESHOLD_STD_MULTIPLIER * std, config.THRESHOLD_MEDIAN_FRACTION * med, config.THRESHOLD_FLOOR_HU)
    stats = {"hu_in_mask_mean": round(mean, 1), "hu_in_mask_std": round(std, 1),
             "hu_in_mask_median": round(med, 1), "threshold_hu": round(thr, 1)}
    return thr, stats


def opening_structure(spacing) -> np.ndarray:
    """Ball of OPENING_RADIUS_MM in voxels, never smaller than OPENING_MIN_RADIUS_VOX per axis."""
    r = [max(config.OPENING_MIN_RADIUS_VOX, int(round(config.OPENING_RADIUS_MM / s))) for s in spacing]
    st = np.zeros([2 * x + 1 for x in r], bool)
    idx = np.indices(st.shape).reshape(3, -1).T
    st.flat[np.where((((idx - np.array(r)) / np.maximum(r, 1)) ** 2).sum(1) <= 1.0)[0]] = True
    return st


def build(image: sitk.Image, mask_image: sitk.Image) -> Candidates:
    """Full front half: clean mask -> crop + resample -> threshold -> shell -> opening."""
    mask_full, frag = clean_mask(sitk.GetArrayViewFromImage(mask_image))
    grid = io_utils.crop_and_resample(image, mask_full)
    del mask_full

    thr, hu_stats = adaptive_threshold(grid.ct, grid.mask)
    dist = ndimage.distance_transform_edt(~grid.mask, sampling=grid.spacing).astype(np.float32)
    shell = (dist > 0) & (dist <= config.SHELL_MM)
    bright = grid.ct > thr
    cand = bright & shell
    cand_open = ndimage.binary_opening(cand, structure=opening_structure(grid.spacing))
    log.info("threshold %.1f HU; %d bright shell voxels, %d after opening", thr, int(cand.sum()), int(cand_open.sum()))
    return Candidates(ct=grid.ct, mask=grid.mask, bright_shell=cand, opened=cand_open, distance_mm=dist,
                      spacing=grid.spacing, image=grid.image, threshold_hu=float(thr),
                      fragment_log=frag, hu_stats=hu_stats, grid_info=grid.info)
