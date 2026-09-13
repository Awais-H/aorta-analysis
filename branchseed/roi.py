"""Stage 1 - crop to a working ROI and normalise the grid.

The single biggest compute win in the pipeline. A typical aorta mask occupies a
couple of percent of the volume; even after a 30 mm dilation the ROI is a few
million voxels instead of twenty-odd million.

We deliberately do **not** globally upsample. Sub-voxel precision comes from
on-demand interpolation at the points we actually need and from the small
per-candidate subvolumes in Stage 5. Upsampling the whole ROI to 0.5 mm would
cost ~60 M voxels per array for no accuracy gain in the EDT or the skeleton.
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import SimpleITK as sitk
from scipy import ndimage

from .config import Config
from .io import sitk_geometry
from .types import CaseGrid, Flags

log = logging.getLogger(__name__)


def mask_bbox_slices(mask_zyx: np.ndarray, dilation_mm: float, spacing_xyz) -> Tuple[slice, ...]:
    """Tight mask bounding box, dilated by a physical margin and clipped."""
    found = ndimage.find_objects(mask_zyx.astype(np.uint8))
    if not found or found[0] is None:
        raise ValueError("cannot compute a bounding box for an empty mask")
    box = found[0]
    spacing_zyx = np.asarray(spacing_xyz, dtype=np.float64)[::-1]
    out = []
    for axis, sl in enumerate(box):
        pad = int(np.ceil(dilation_mm / max(spacing_zyx[axis], 1e-6)))
        start = max(0, sl.start - pad)
        stop = min(mask_zyx.shape[axis], sl.stop + pad)
        out.append(slice(start, stop))
    return tuple(out)


def _needs_resampling(spacing_xyz: np.ndarray, tolerance: float) -> bool:
    return bool(spacing_xyz.max() / max(spacing_xyz.min(), 1e-9) > tolerance)


def _resample_isotropic(
    image: sitk.Image, mask: sitk.Image, target_mm: float
) -> Tuple[sitk.Image, sitk.Image]:
    spacing = np.asarray(image.GetSpacing(), dtype=np.float64)
    size = np.asarray(image.GetSize(), dtype=np.float64)
    extent = spacing * size
    new_size = np.maximum(np.ceil(extent / target_mm).astype(int), 1)
    new_spacing = [float(target_mm)] * 3

    def run(img: sitk.Image, interpolator: int, default: float) -> sitk.Image:
        r = sitk.ResampleImageFilter()
        r.SetOutputOrigin(img.GetOrigin())
        r.SetOutputDirection(img.GetDirection())
        r.SetOutputSpacing(new_spacing)
        r.SetSize([int(v) for v in new_size])
        r.SetInterpolator(interpolator)
        r.SetDefaultPixelValue(default)
        return r.Execute(img)

    return run(image, sitk.sitkBSpline, -1024.0), run(mask, sitk.sitkNearestNeighbor, 0.0)


def build_roi(
    image_sitk: sitk.Image, mask_zyx: np.ndarray, cfg: Config, flags: Flags
) -> Tuple[np.ndarray, np.ndarray, CaseGrid]:
    """Crop both volumes to the dilated mask bbox and return the ROI grid.

    Returns ``(image_roi_zyx float32, mask_roi_zyx bool, grid)``.
    """
    _, _, orig_spacing = sitk_geometry(image_sitk)
    sl_z, sl_y, sl_x = mask_bbox_slices(mask_zyx, cfg.roi.dilation_mm, orig_spacing)

    full = np.asarray(image_sitk.GetSize(), dtype=np.int64)
    log.info(
        "ROI crop x=%d:%d y=%d:%d z=%d:%d of %s (%.1f%% of volume)",
        sl_x.start, sl_x.stop, sl_y.start, sl_y.stop, sl_z.start, sl_z.stop,
        tuple(full), 100.0 * (sl_x.stop - sl_x.start) * (sl_y.stop - sl_y.start)
        * (sl_z.stop - sl_z.start) / float(np.prod(full)),
    )

    crop_origin_idx = (int(sl_x.start), int(sl_y.start), int(sl_z.start))
    image_roi = image_sitk[sl_x.start:sl_x.stop, sl_y.start:sl_y.stop, sl_z.start:sl_z.stop]

    mask_roi_arr = np.ascontiguousarray(mask_zyx[sl_z, sl_y, sl_x])
    mask_roi = sitk.GetImageFromArray(mask_roi_arr.astype(np.uint8))
    mask_roi.CopyInformation(image_roi)

    resampled = False
    if _needs_resampling(orig_spacing, cfg.roi.isotropy_tolerance):
        target = float(orig_spacing.min())
        log.info("anisotropic input %s; resampling ROI to %.3f mm isotropic",
                 tuple(round(s, 3) for s in orig_spacing), target)
        flags.add("roi_resampled_isotropic")
        image_roi, mask_roi = _resample_isotropic(image_roi, mask_roi, target)
        resampled = True

    image_arr = sitk.GetArrayFromImage(image_roi).astype(np.float32, copy=False)
    mask_arr = sitk.GetArrayFromImage(mask_roi) > 0
    del mask_roi_arr, mask_roi

    origin, direction, spacing = sitk_geometry(image_roi)
    grid = CaseGrid(
        sitk_ref=image_sitk,
        crop_origin_idx=crop_origin_idx,
        roi_origin_mm=origin,
        roi_direction=direction,
        roi_spacing_mm=spacing,
        orig_spacing_mm=orig_spacing,
        roi_shape=tuple(int(v) for v in image_arr.shape),
        resampled=resampled,
    )
    _assert_grid_consistent(grid, image_roi)
    log.info("ROI shape (z, y, x) = %s, spacing %s, %.2f M voxels",
             grid.roi_shape, tuple(round(s, 3) for s in spacing), image_arr.size / 1e6)
    return image_arr, mask_arr, grid


def _assert_grid_consistent(grid: CaseGrid, roi_image: sitk.Image, atol: float = 1e-4) -> None:
    """Cross-check the vectorised transform against SimpleITK, and the ROI
    index chain against the original image. Cheap, and it catches an axis-order
    or crop-offset error at the moment it is introduced rather than three
    stages later as a plausible-looking wrong answer."""
    probes = np.array(
        [[0.0, 0.0, 0.0],
         [1.0, 2.0, 3.0],
         [(grid.roi_shape[2] - 1) / 2.0, (grid.roi_shape[1] - 1) / 2.0,
          (grid.roi_shape[0] - 1) / 2.0]]
    )
    ours = grid.to_physical(probes)
    for probe, mine in zip(probes, ours):
        theirs = roi_image.TransformContinuousIndexToPhysicalPoint(
            [float(v) for v in probe]
        )
        if not np.allclose(mine, theirs, atol=atol):
            raise AssertionError(f"ROI to_physical disagrees with SimpleITK: {mine} vs {theirs}")
        orig_idx = grid.to_original_index(probe)
        via_orig = grid.sitk_ref.TransformContinuousIndexToPhysicalPoint(
            [float(v) for v in orig_idx]
        )
        if not np.allclose(mine, via_orig, atol=atol):
            raise AssertionError(
                f"ROI->original index chain disagrees: {mine} vs {via_orig}"
            )
