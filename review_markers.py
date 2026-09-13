"""Review aids for looking at a case by eye in ITK-SNAP or 3D Slicer.

Writes, on the CT's native grid:
  <out>/subjectNNN_pred_markers.nii.gz   2 mm spheres at each predicted ostium, label = branch number
  <out>/subjectNNN_ref_markers.nii.gz    the same for the reference ostia (label = reference number), if any
  <out>/subjectNNN_review.txt            one line per predicted branch: native voxel index (x, y, z) of the
                                         ostium for the slice sliders, clock, height, diameter, path, flags,
                                         nearest reference and its distance

    python review_markers.py --cases 21 22 --pred-dir out/review --out out/review
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import SimpleITK as sitk

import config
import io_utils
import scorer

MARKER_RADIUS_MM = 2.0  # sphere radius for the overlay: visible at 1.5 mm voxels, smaller than any renal


def paint_spheres(image: sitk.Image, points_mm: list, labels: list) -> sitk.Image:
    size = list(image.GetSize())[::-1]  # z, y, x
    arr = np.zeros(size, np.uint8)
    sp = np.array(image.GetSpacing())[::-1]
    r_vox = np.ceil(MARKER_RADIUS_MM / sp).astype(int)
    for p, lab in zip(points_mm, labels):
        c = np.round(io_utils.mm_to_index(image, p)).astype(int)
        lo = np.maximum(c - r_vox, 0)
        hi = np.minimum(c + r_vox + 1, size)
        zz, yy, xx = np.mgrid[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        d = np.sqrt(((zz - c[0]) * sp[0]) ** 2 + ((yy - c[1]) * sp[1]) ** 2 + ((xx - c[2]) * sp[2]) ** 2)
        sub = arr[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        sub[d <= MARKER_RADIUS_MM] = lab
    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(image)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", required=True)
    ap.add_argument("--pred-dir", default=scorer.PRED_DIR)
    ap.add_argument("--out", default=os.path.join(scorer.ROOT, "out", "review"))
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    for n in scorer._parse_cases(args.cases):
        cf = scorer.case_files(n)
        image, _ = io_utils.read_image(cf["mask"])  # mask geometry == CT geometry, and it is small to read
        pred_path = os.path.join(args.pred_dir, f"{cf['case_id']}.json")
        meta_path = os.path.join(args.pred_dir, f"{cf['case_id']}_meta.json")
        with open(pred_path, encoding="utf-8") as f:
            pred = json.load(f)
        meta = json.load(open(meta_path, encoding="utf-8")) if os.path.exists(meta_path) else {}
        ds = pred["daughters"]
        sitk.WriteImage(paint_spheres(image, [d["ostium_xyz_mm"] for d in ds], list(range(1, len(ds) + 1))),
                        os.path.join(args.out, f"{cf['case_id']}_pred_markers.nii.gz"))
        ref = scorer.load_reference(n)
        refs = ref["daughters"] if ref else []
        if refs:
            sitk.WriteImage(paint_spheres(image, [r["ostium_xyz_mm"] for r in refs], list(range(1, len(refs) + 1))),
                            os.path.join(args.out, f"{cf['case_id']}_ref_markers.nii.gz"))
        # frame for clock/height: rebuild the cheap front half
        import candidates
        import frame as frame_mod
        img_full, mask_img, _ = io_utils.load_case(cf["image"], cf["mask"])
        cand = candidates.build(img_full, mask_img)
        fr = frame_mod.build(cand)
        branch_of_label = {v: k for k, v in meta.get("label_to_branch", {}).items()}
        lines = [f"{cf['case_id']}: {len(ds)} predicted branches, {len(refs)} reference branches. "
                 "Voxel index is (x, y, z) zero-based on the native grid, for the ITK-SNAP slice sliders.",
                 "Clock: 12 anterior, 3 patient's left, 6 posterior, 9 patient's right. Height from the superior end of the mask.", ""]
        for k, d in enumerate(ds, 1):
            idx = np.round(io_utils.mm_to_index(image, d["ostium_xyz_mm"])).astype(int)[::-1]
            h, c = fr.height_clock(d["ostium_xyz_mm"])
            lab = branch_of_label.get(d["instance_id"])
            m = meta.get("measurements", {}).get(str(lab), {}) if lab else {}
            flags = meta.get("flags", {}).get(str(lab), []) if lab else []
            near = ""
            if refs:
                dist = [np.linalg.norm(np.subtract(d["ostium_xyz_mm"], r["ostium_xyz_mm"])) for r in refs]
                j = int(np.argmin(dist))
                near = f"nearest ref {refs[j]['instance_id']} at {dist[j]:.1f} mm" + ("  <-- MATCH" if dist[j] <= config.MATCH_CUTOFF_MM else "  (no reference within 5 mm)")
            lines.append(f"{d['instance_id']}  voxel (x={idx[0]}, y={idx[1]}, z={idx[2]})  clock {c:4.1f}  height {h:5.1f} mm  "
                         f"diam {m.get('origin_diameter_mm')} mm  radius {d['radius_mm']:.1f}  path {m.get('path_mm')} mm  "
                         f"departure {m.get('departure_mm')}  flags {flags}  {near}")
        with open(os.path.join(args.out, f"{cf['case_id']}_review.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print("\n".join(lines))
        print()


if __name__ == "__main__":
    main()
