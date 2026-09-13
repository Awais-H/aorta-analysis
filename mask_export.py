"""Render a run.py prediction JSON, together with the supplied aorta mask, into one coloured
NIfTI segmentation.

Standalone visual check, separate from the pipeline: an ITK-SNAP / 3D Slicer overlay that a
clinician can load next to the CT without a browser or Plotly, on the same grid as the supplied
aorta mask so it lines up voxel for voxel. Two labels only:

    1 = the supplied parent aorta (red)
    2 = every daughter's ostium sphere and direction segment (teal)

Each daughter contributes a sphere of radius `radius_mm` at the ostium, plus a thin segment from
the ostium to ostium + direction_xyz * TRACE_MAX_MM so the reported direction is visible too (a
bare point cannot show direction, and direction is one of the five required outputs). The seed
point falls on that segment by construction (PDF: seed = ostium + 5 mm along the daughter path),
so it needs no separate marker. Daughters are painted over the aorta label, so the label at an
ostium always reads as "daughter", not "aorta". Branch identity is not preserved in the label
values (out of scope here; run.py's JSON is still the authority on which is branch_001 vs
branch_002) - the point of this file is one glance at what belongs to the parent vs what does not.

Geometry is computed in physical mm via io_utils.mm_to_index / io_utils.index_to_mm, so it is
correct under a non-orthonormal direction matrix (subject 24) and goes through io_utils for every
index/physical conversion, per CLAUDE.md's rule that io_utils owns that mapping.

Also writes a sibling "<output stem>_labels.txt": ITK-SNAP label descriptions (Segmentation ->
Import Label Descriptions) so the two labels open already coloured red and teal, matching the
convention in review_markers.py.

    python mask_export.py --json prediction.json --aorta-mask mask1.nii --output aorta_daughters1.nii
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import numpy as np
import SimpleITK as sitk

import config
import io_utils

log = logging.getLogger("branchseed.mask_export")

AORTA_LABEL = 1
DAUGHTER_LABEL = 2
AORTA_COLOR = (220, 30, 30)     # red
DAUGHTER_COLOR = (0, 170, 170)  # teal


def _point_segment_distance(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance from each row of `points` (N, 3) to the segment a->b (both (3,)), physical mm."""
    ab = b - a
    denom = float(ab @ ab)
    if denom < 1e-12:
        return np.linalg.norm(points - a[None, :], axis=1)
    t = np.clip(((points - a[None, :]) @ ab) / denom, 0.0, 1.0)
    closest = a[None, :] + t[:, None] * ab[None, :]
    return np.linalg.norm(points - closest, axis=1)


def _paint_branch(label_arr: np.ndarray, image: sitk.Image, ostium_mm: np.ndarray, direction: np.ndarray,
                   radius_mm: float, label: int) -> None:
    direction = np.asarray(direction, dtype=np.float64)
    n = np.linalg.norm(direction)
    direction = direction / n if n > 0 else np.array([1.0, 0.0, 0.0])
    end_mm = ostium_mm + direction * config.TRACE_MAX_MM
    line_r = config.MASK_EXPORT_DIRECTION_RADIUS_MM
    extent = max(radius_mm, line_r)

    idx_o = io_utils.mm_to_index(image, ostium_mm)
    idx_e = io_utils.mm_to_index(image, end_mm)
    spacing_zyx = np.array(image.GetSpacing())[::-1]
    pad_vox = int(np.ceil(extent / max(spacing_zyx.min(), 1e-6))) + 2

    shape = label_arr.shape  # (z, y, x)
    lo = np.floor(np.minimum(idx_o, idx_e)).astype(int) - pad_vox
    hi = np.ceil(np.maximum(idx_o, idx_e)).astype(int) + pad_vox
    lo = np.clip(lo, 0, np.array(shape) - 1)
    hi = np.clip(hi, 0, np.array(shape) - 1)
    if np.any(hi < lo):
        return

    zz, yy, xx = np.meshgrid(np.arange(lo[0], hi[0] + 1), np.arange(lo[1], hi[1] + 1),
                              np.arange(lo[2], hi[2] + 1), indexing="ij")
    idx_pts = np.stack([zz.ravel(), yy.ravel(), xx.ravel()], axis=1).astype(np.float64)
    mm_pts = io_utils.index_to_mm(image, idx_pts)

    in_sphere = np.linalg.norm(mm_pts - ostium_mm[None, :], axis=1) <= radius_mm
    in_line = _point_segment_distance(mm_pts, ostium_mm, end_mm) <= line_r
    hit = in_sphere | in_line
    if not np.any(hit):
        return

    z, y, x = idx_pts[hit, 0].astype(int), idx_pts[hit, 1].astype(int), idx_pts[hit, 2].astype(int)
    label_arr[z, y, x] = label  # daughters always win over the aorta label underneath


def build_mask(prediction: dict, aorta_mask_image: sitk.Image) -> sitk.Image:
    """One uint8 volume on `aorta_mask_image`'s grid: 0 background, 1 the supplied aorta mask,
    2 every daughter's ostium sphere and direction segment (painted last, so it always wins where
    a daughter overlaps the aortic wall)."""
    aorta_arr = sitk.GetArrayFromImage(aorta_mask_image) != 0
    label_arr = np.zeros(aorta_arr.shape, dtype=np.uint8)
    label_arr[aorta_arr] = AORTA_LABEL

    for d in prediction.get("daughters", []):
        ostium_mm = np.asarray(d["ostium_xyz_mm"], dtype=np.float64)
        _paint_branch(label_arr, aorta_mask_image, ostium_mm, d["direction_xyz"], float(d["radius_mm"]), DAUGHTER_LABEL)

    out = sitk.GetImageFromArray(label_arr)
    out.CopyInformation(aorta_mask_image)
    return out


def write_label_description(path: str) -> None:
    """ITK-SNAP 'Segmentation -> Import Label Descriptions' file: IDX R G B A VIS MSH LABEL."""
    with open(path, "w", encoding="utf-8") as f:
        f.write("# ITK-SNAP label descriptions: IDX R G B A VIS MSH LABEL\n")
        f.write("0 0 0 0 0 0 0 \"Clear Label\"\n")
        f.write(f"{AORTA_LABEL} {AORTA_COLOR[0]} {AORTA_COLOR[1]} {AORTA_COLOR[2]} 1 1 1 \"aorta\"\n")
        f.write(f"{DAUGHTER_LABEL} {DAUGHTER_COLOR[0]} {DAUGHTER_COLOR[1]} {DAUGHTER_COLOR[2]} 1 1 1 \"daughter branches\"\n")


def json_to_mask(json_path: str, aorta_mask_path: str, output_path: str) -> sitk.Image:
    with open(json_path, encoding="utf-8") as f:
        prediction = json.load(f)
    aorta_mask_image, _ = io_utils.read_image(aorta_mask_path)
    out = build_mask(prediction, aorta_mask_image)
    sitk.WriteImage(out, output_path)

    stem = output_path[:-len(".nii.gz")] if output_path.endswith(".nii.gz") else os.path.splitext(output_path)[0]
    write_label_description(stem + "_labels.txt")

    log.info("wrote %s (+ %s_labels.txt): aorta + %d daughters from %s onto the grid of %s",
              output_path, stem, len(prediction.get("daughters", [])), json_path, aorta_mask_path)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Render a branchseed prediction JSON, with the aorta mask, as one coloured NIfTI segmentation.")
    ap.add_argument("--json", required=True, help="prediction JSON written by run.py")
    ap.add_argument("--aorta-mask", required=True, help="the supplied parent aorta mask (mask*.nii), also used for the output grid")
    ap.add_argument("--output", required=True, help="output segmentation (.nii or .nii.gz)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    json_to_mask(args.json, args.aorta_mask, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
