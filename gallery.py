"""D7 failure gallery: image crops for every kept branch (and optionally every rejected wall
patch) of a case, so a reviewer can answer one question per candidate without opening a viewer:
is there a contrast-filled tube leaving the aortic lumen here that can be followed for 5 mm?

Per candidate, four panels on the native grid, 40 mm wide, CT windowed to [-100, 600] HU, the
aorta mask outlined in cyan: axial through the ostium, axial through the seed (5 mm out),
coronal and sagittal through the ostium. Green dot = ostium, yellow dot = seed, red arrow =
direction projected into the panel. Six candidates per sheet.

    python gallery.py --cases 22 --pred-dir out/review --out out/gallery [--rejected]
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk

import config
import io_utils
import scorer

CROP_MM = 40.0          # panel width: the aorta plus 10 mm of surroundings on each side
WINDOW_HU = (-100.0, 600.0)  # soft tissue to contrast-filled lumen; bone saturates, which is the point
ROWS_PER_SHEET = 6


def _slice(ct, mask, idx_zyx, axis, sp):
    """2D crop of ct and mask through idx along `axis` (0 axial, 1 coronal, 2 sagittal), plus the
    (row, col) of idx in the crop and the pixel aspect."""
    z, y, x = [int(round(v)) for v in idx_zyx]
    half = [int(round(CROP_MM / 2 / s)) for s in sp]
    if axis == 0:
        img, m = ct[z], mask[z]
        r, c, hr, hc, sr, sc = y, x, half[1], half[2], sp[1], sp[2]
    elif axis == 1:
        img, m = ct[:, y, :], mask[:, y, :]
        r, c, hr, hc, sr, sc = z, x, half[0], half[2], sp[0], sp[2]
    else:
        img, m = ct[:, :, x], mask[:, :, x]
        r, c, hr, hc, sr, sc = z, y, half[0], half[1], sp[0], sp[1]
    r0, r1 = max(0, r - hr), min(img.shape[0], r + hr + 1)
    c0, c1 = max(0, c - hc), min(img.shape[1], c + hc + 1)
    return img[r0:r1, c0:c1], m[r0:r1, c0:c1], (r - r0, c - c0), sr / sc, (r0, c0)


def _panel(ax, ct, mask, centre_idx, mark_idx, axis, sp, direction_idx, title):
    img, m, (r, c), aspect, (r0, c0) = _slice(ct, mask, centre_idx, axis, sp)
    ax.imshow(img, cmap="gray", vmin=WINDOW_HU[0], vmax=WINDOW_HU[1], aspect=aspect, origin="lower" if axis else "upper")
    if m.any():
        ax.contour(m.astype(float), levels=[0.5], colors="cyan", linewidths=0.8)
    comp = {0: (1, 2), 1: (0, 2), 2: (0, 1)}[axis]  # (row axis, col axis) in zyx
    for idx, colour in mark_idx:
        rr, cc = idx[comp[0]] - r0, idx[comp[1]] - c0
        if 0 <= rr < img.shape[0] and 0 <= cc < img.shape[1]:
            ax.plot(cc, rr, "o", color=colour, ms=5, mec="black", mew=0.5)
    d = np.array([direction_idx[comp[0]], direction_idx[comp[1]]])
    L = 8.0 / np.array([sp[comp[0]], sp[comp[1]]])  # 8 mm arrow in pixels per axis
    ax.annotate("", xy=(c + d[1] * L[1], r + d[0] * L[0]), xytext=(c, r), arrowprops=dict(arrowstyle="->", color="red", lw=1.2))
    ax.set_title(title, fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])


def render(case_n, pred_dir, out_dir, include_rejected=False):
    cf = scorer.case_files(case_n)
    image, mask_image, _ = io_utils.load_case(cf["image"], cf["mask"])
    ct = sitk.GetArrayViewFromImage(image).astype(np.float32)
    mask = sitk.GetArrayViewFromImage(mask_image) > 0
    sp = np.array(image.GetSpacing())[::-1]
    pred = json.load(open(os.path.join(pred_dir, f"{cf['case_id']}.json"), encoding="utf-8"))
    meta_p = os.path.join(pred_dir, f"{cf['case_id']}_meta.json")
    meta = json.load(open(meta_p, encoding="utf-8")) if os.path.exists(meta_p) else {}
    ref = scorer.load_reference(case_n)
    refs = ref["daughters"] if ref else []
    review = {}
    rp = os.path.join(pred_dir, f"{cf['case_id']}_review.txt")
    if os.path.exists(rp):
        for line in open(rp, encoding="utf-8"):
            if line.startswith("branch_"):
                review[line.split()[0]] = line.strip()

    items = []
    for d in pred["daughters"]:
        o = io_utils.mm_to_index(image, d["ostium_xyz_mm"])
        s = io_utils.mm_to_index(image, d["seed_xyz_mm"])
        dv = io_utils.mm_vector_to_index(image, d["ostium_xyz_mm"], d["direction_xyz"])
        near = ""
        if refs:
            dist = [np.linalg.norm(np.subtract(d["ostium_xyz_mm"], r["ostium_xyz_mm"])) for r in refs]
            j = int(np.argmin(dist))
            near = f"ref {refs[j]['instance_id'][-3:]} {dist[j]:.1f}mm" if dist[j] <= config.MATCH_CUTOFF_MM else "no ref"
        line = review.get(d["instance_id"], "")
        info = " ".join(t for t in line.split("  ") if t.startswith(("clock", "height", "diam", "flags")))
        items.append((d["instance_id"], o, s, dv, f"{d['instance_id']}  r={d['radius_mm']:.1f}  {near}\n{info}"))
    if include_rejected:
        import candidates
        import instances
        import ostium
        import tracing
        cand = candidates.build(image, mask_image)
        inst = instances.build(cand)
        ostia = ostium.locate(cand, inst)
        traces = tracing.trace_all(cand, inst, ostia)
        kept = set(meta.get("label_to_branch", {}).keys())
        for l in inst.labels_list:
            if str(l) in kept:
                continue
            ost, tr = ostia[l], traces[l]
            o = io_utils.mm_to_index(image, ost.mm)
            s = io_utils.mm_to_index(image, tr.seed_mm) if tr.seed_mm is not None else o
            dv = io_utils.mm_vector_to_index(image, ost.mm, tr.direction_xyz)
            hits = ",".join(sorted({rr for ll, rr, v in meta.get("rejections", []) if ll == l}))
            m = meta.get("measurements", {}).get(str(l), {})
            items.append((f"rejected patch {l}", o, s, dv, f"REJECTED patch {l}: {hits}\nwall {m.get('wall_voxels')} vox diam {m.get('origin_diameter_mm')} path {m.get('path_mm')}"))

    os.makedirs(out_dir, exist_ok=True)
    sheets = []
    for start in range(0, len(items), ROWS_PER_SHEET):
        chunk = items[start:start + ROWS_PER_SHEET]
        fig, axes = plt.subplots(len(chunk), 4, figsize=(11, 2.6 * len(chunk)), squeeze=False)
        for row, (name, o, s, dv, title) in zip(axes, chunk):
            marks = [(o, "lime"), (s, "yellow")]
            _panel(row[0], ct, mask, o, marks, 0, sp, dv, title + "\naxial @ ostium")
            _panel(row[1], ct, mask, s, marks, 0, sp, dv, "axial @ seed (5 mm out)")
            _panel(row[2], ct, mask, o, marks, 1, sp, dv, "coronal @ ostium")
            _panel(row[3], ct, mask, o, marks, 2, sp, dv, "sagittal @ ostium")
        fig.suptitle(f"{cf['case_id']}  candidates {start + 1} to {start + len(chunk)} of {len(items)}", fontsize=10)
        plt.tight_layout()
        path = os.path.join(out_dir, f"{cf['case_id']}_sheet{start // ROWS_PER_SHEET + 1}.png")
        fig.savefig(path, dpi=100)
        plt.close(fig)
        sheets.append(path)
    return sheets


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", required=True)
    ap.add_argument("--pred-dir", default=scorer.PRED_DIR)
    ap.add_argument("--out", default=os.path.join(scorer.ROOT, "out", "gallery"))
    ap.add_argument("--rejected", action="store_true", help="also render every rejected wall patch")
    args = ap.parse_args(argv)
    for n in scorer._parse_cases(args.cases):
        for p in render(n, args.pred_dir, args.out, args.rejected):
            print(p)


if __name__ == "__main__":
    main()
