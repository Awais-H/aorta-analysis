"""Dense review views for one candidate: the views that decide "is there a contrast-filled tube
leaving the aortic lumen here that can be followed 5 mm" when the gallery sheets (gallery.py,
one axial slice at the ostium and one at the seed) are too coarse for a 2 to 3 mm vessel at
1.5 mm voxels. Used for the 13 Sep false-positive adjudication (SPEC section 1).

Per candidate (a wall-patch label as printed by the ledger and review_markers, or a branch id):

  <case>_patch<NN>.png       twelve native axial slices from 7.5 mm below to 9 mm above the ostium
                             with the traced path drawn where it crosses each slice; two oblique
                             longitudinal reformats through the ostium along the traced chord
                             (chord x outward wall normal, chord x the other perpendicular);
                             native coronal and sagittal; perpendicular sections every 2 mm along
                             the path out to 10 mm (16 mm wide). A tube shows as a round bright
                             dot at the centre of the perpendicular sections that persists to 5 mm.
  <case>_patch<NN>_wall.png  reformats along the local aortic axis: (axis x outward normal) through
                             the ostium, and two planes parallel to the wall 1.5 and 3.5 mm outside
                             it, 50 mm long, plus a 90 mm wide axial. A vessel passing in contact
                             shows as a band running through the ostium on both sides; a daughter's
                             lumen has an end at its ostium.

Also prints the HU profile along the traced path and along the straight chord (native trilinear),
the distance from the mask at each millimetre and the bright non-mask voxel count in a 5 x 5 x 5
native neighbourhood.

    python review_dense.py --case 22 --labels 23 28 --out out/review/dense
    python review_dense.py --case 21 --branches branch_005 branch_008
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from scipy import ndimage

import candidates
import filters
import frame as frame_mod
import instances
import io_utils
import ostium
import scorer
import tracing

WINDOW_LOW_HU = -100.0      # display: fat and soft tissue stay visible
WINDOW_HIGH_FRACTION = 1.15  # display: the window tops out just above the lumen median so a partial-volume vessel is grey, not lost
PLANE_STEP_MM = 0.4         # display: reformat sampling step (interpolation adds no resolution; it only draws the voxel grid smoothly)


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n else v


def _perp(d):
    a = np.array([1.0, 0, 0]) if abs(d[0]) < 0.9 else np.array([0, 1.0, 0])
    u = _unit(np.cross(d, a))
    return u, _unit(np.cross(d, u))


def sample_plane(image, arr, centre, e1, e2, r1, r2, step, order):
    """Resample `arr` (native zyx) on the plane centre + a*e1 + b*e2, a in r1, b in r2 (mm)."""
    a = np.arange(r1[0], r1[1] + 1e-6, step)
    b = np.arange(r2[0], r2[1] + 1e-6, step)
    A, B = np.meshgrid(a, b, indexing="ij")
    pts = centre[None, :] + A.reshape(-1, 1) * e1[None, :] + B.reshape(-1, 1) * e2[None, :]
    idx = io_utils.mm_to_index(image, pts)
    vals = ndimage.map_coordinates(arr, idx.T, order=order, mode="constant", cval=-1000 if order else 0)
    return vals.reshape(A.shape), a, b


def _show_plane(ax, img, a, b, m, vmin, vmax, title):
    ax.imshow(img.T, cmap="gray", vmin=vmin, vmax=vmax, origin="lower", extent=[a[0], a[-1], b[0], b[-1]], interpolation="nearest")
    if m.any():
        ax.contour(a, b, m.T, levels=[0.5], colors="cyan", linewidths=0.8)
    ax.set_title(title, fontsize=8)


def profile_lines(cand, image, ct, mask, ost, tr, fr) -> list:
    """HU and distance profile along the traced path and the straight chord."""
    sp = np.array(image.GetSpacing())[::-1]
    bright = ct > cand.threshold_hu
    dist_native = ndimage.distance_transform_edt(mask == 0, sampling=sp)
    o, d = ost.mm, _unit(tr.direction_xyz)
    h, c = fr.height_clock(o)
    oi = io_utils.mm_to_index(image, o)
    lines = [f"patch {ost.label}: clock {c:.1f}, height {h:.1f} mm, ostium native voxel (x,y,z)={np.round(oi[::-1]).astype(int).tolist()}, "
             f"direction xyz={np.round(d, 2).tolist()}, path {tr.path_length_mm} mm ({tr.method}/{tr.stop_reason}), threshold {cand.threshold_hu:.0f} HU, "
             f"lumen median {cand.hu_stats.get('hu_in_mask_median', float('nan')):.0f} HU",
             "  s_mm  HU@path  dist_mask_mm  bright_nomask_5x5x5 | HU@chord  dist_chord_mm"]

    def stats(pt):
        idx = io_utils.mm_to_index(image, pt)
        hu = float(ndimage.map_coordinates(ct, idx.reshape(3, 1), order=1)[0])
        dm = float(ndimage.map_coordinates(dist_native, idx.reshape(3, 1), order=1)[0])
        zi, yi, xi = np.round(idx).astype(int)
        sl = tuple(slice(max(0, v - 2), v + 3) for v in (zi, yi, xi))
        return hu, dm, int((bright[sl] & (mask[sl] == 0)).sum())

    for s in np.arange(0, 10.5, 1.0):
        p = tracing.point_along(tr.path_mm, s)
        huc, dmc, _ = stats(o + s * d)
        if p is not None:
            hu, dm, b = stats(p)
            lines.append(f"  {s:4.0f}  {hu:7.0f}  {dm:12.1f}  {b:19d} | {huc:7.0f}  {dmc:13.1f}")
        else:
            lines.append(f"  {s:4.0f}  (path ended)                              | {huc:7.0f}  {dmc:13.1f}")
    return lines


def render_candidate(cand, image, ct, mask, ost, tr, fr, out_dir, case_id):
    vmin = WINDOW_LOW_HU
    vmax = WINDOW_HIGH_FRACTION * cand.hu_stats.get("hu_in_mask_median", 300.0)
    sp = np.array(image.GetSpacing())[::-1]
    o, d, n = ost.mm, _unit(tr.direction_xyz), _unit(ost.normal_mm)
    h, c = fr.height_clock(o)
    u = _unit(n - np.dot(n, d) * d)
    if np.linalg.norm(u) < 1e-6:
        u, _ = _perp(d)
    w = _unit(np.cross(d, u))
    oi = io_utils.mm_to_index(image, o)
    z0, yc, xc = [int(round(v)) for v in oi]
    path_idx = io_utils.mm_to_index(image, tr.path_mm)
    seed_idx = io_utils.mm_to_index(image, tr.seed_mm) if tr.seed_mm is not None else None
    rel = tr.path_mm - o
    head = f"{case_id} patch {ost.label}: clock {c:.1f}, height {h:.1f} mm, radius {tr.radius_mm:.1f} mm, path {tr.path_length_mm} mm ({tr.method}/{tr.stop_reason}); window [{vmin:.0f},{vmax:.0f}] HU; green ostium, yellow seed, red path"

    # ---- figure 1: axial series, chord reformats, native coronal/sagittal, perpendicular sections
    fig = plt.figure(figsize=(16, 13))
    gs = fig.add_gridspec(4, 6)
    half = int(round(15 / sp[1]))
    for k, dz in enumerate(range(-5, 7)):
        ax = fig.add_subplot(gs[k // 6, k % 6])
        z = z0 + dz
        if not (0 <= z < ct.shape[0]):
            ax.axis("off")
            continue
        img = ct[z, yc - half:yc + half + 1, xc - half:xc + half + 1]
        m = mask[z, yc - half:yc + half + 1, xc - half:xc + half + 1]
        ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax, origin="upper", interpolation="nearest")
        if m.any():
            ax.contour(m.astype(float), levels=[0.5], colors="cyan", linewidths=0.8)
        on = np.abs(path_idx[:, 0] - z) <= 0.5
        ax.plot(path_idx[on, 2] - (xc - half), path_idx[on, 1] - (yc - half), "r.", ms=4)
        if dz == 0:
            ax.plot(oi[2] - (xc - half), oi[1] - (yc - half), "o", color="lime", ms=6, mec="k")
        if seed_idx is not None and abs(seed_idx[0] - z) <= 0.5:
            ax.plot(seed_idx[2] - (xc - half), seed_idx[1] - (yc - half), "o", color="yellow", ms=6, mec="k")
        ax.set_title(f"axial z={z} ({dz * sp[0]:+.1f} mm)", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    for k, (e2, name) in enumerate(((u, "chord x wall-normal"), (w, "chord x other"))):
        ax = fig.add_subplot(gs[2, 2 * k:2 * k + 2])
        img, a, b = sample_plane(image, ct, o, d, e2, (-12, 12), (-15, 15), PLANE_STEP_MM, 1)
        m, _, _ = sample_plane(image, mask, o, d, e2, (-12, 12), (-15, 15), PLANE_STEP_MM, 0)
        _show_plane(ax, img, a, b, m, vmin, vmax, f"longitudinal: {name} (x = along chord, mm)")
        ax.plot(rel @ d, rel @ e2, "r.-", ms=3, lw=0.8)
        ax.plot(0, 0, "o", color="lime", ms=6, mec="k")
        if tr.seed_mm is not None:
            ax.plot((tr.seed_mm - o) @ d, (tr.seed_mm - o) @ e2, "o", color="yellow", ms=6, mec="k")
        ax.axvline(5, color="yellow", lw=0.5, ls="--")
        ax.axvline(10, color="orange", lw=0.5, ls="--")
    for k, axis in enumerate((1, 2)):
        ax = fig.add_subplot(gs[2, 4 + k])
        if axis == 1:
            img, m = ct[z0 - half:z0 + half + 1, yc, xc - half:xc + half + 1], mask[z0 - half:z0 + half + 1, yc, xc - half:xc + half + 1]
            pr, pc, sel = path_idx[:, 0] - (z0 - half), path_idx[:, 2] - (xc - half), np.abs(path_idx[:, 1] - yc) <= 0.5
            title = "coronal (native)"
        else:
            img, m = ct[z0 - half:z0 + half + 1, yc - half:yc + half + 1, xc], mask[z0 - half:z0 + half + 1, yc - half:yc + half + 1, xc]
            pr, pc, sel = path_idx[:, 0] - (z0 - half), path_idx[:, 1] - (yc - half), np.abs(path_idx[:, 2] - xc) <= 0.5
            title = "sagittal (native)"
        ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax, origin="lower", interpolation="nearest")
        if m.any():
            ax.contour(m.astype(float), levels=[0.5], colors="cyan", linewidths=0.8)
        ax.plot(pc[sel], pr[sel], "r.", ms=4)
        ax.plot(pc[0], pr[0], "o", color="lime", ms=6, mec="k")
        ax.set_title(title, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    for k, s in enumerate((0, 2, 4, 6, 8, 10)):
        ax = fig.add_subplot(gs[3, k])
        p = tracing.point_along(tr.path_mm, s)
        if p is None:
            p, t, tag = o + s * d, d, " (chord)"
        else:
            p2 = tracing.point_along(tr.path_mm, min(s + 1, tr.path_length_mm))
            p1 = tracing.point_along(tr.path_mm, max(s - 1, 0))
            t = _unit(p2 - p1) if p2 is not None and p1 is not None and np.linalg.norm(p2 - p1) > 0 else d
            tag = ""
        e1, e2 = _perp(t)
        img, a, b = sample_plane(image, ct, p, e1, e2, (-8, 8), (-8, 8), PLANE_STEP_MM, 1)
        m, _, _ = sample_plane(image, mask, p, e1, e2, (-8, 8), (-8, 8), PLANE_STEP_MM, 0)
        _show_plane(ax, img, a, b, m, vmin, vmax, f"perp section at {s} mm{tag}, 16 mm wide")
        ax.plot(0, 0, "+", color="red", ms=8)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(head, fontsize=10)
    plt.tight_layout()
    p1 = os.path.join(out_dir, f"{case_id}_patch{ost.label:02d}.png")
    fig.savefig(p1, dpi=110)
    plt.close(fig)

    # ---- figure 2: along the aortic axis, wall-parallel planes, wide axial
    mask_mm = io_utils.index_to_mm(image, np.argwhere(mask > 0))
    near = mask_mm[np.linalg.norm(mask_mm - o, axis=1) < 20]
    _, _, vt = np.linalg.svd(near - near.mean(axis=0), full_matrices=False)
    t = _unit(vt[0])
    if t[2] < 0:
        t = -t
    n2 = _unit(n - np.dot(n, t) * t)
    bt = _unit(np.cross(t, n2))
    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    img, a, b = sample_plane(image, ct, o, t, n2, (-25, 25), (-10, 15), PLANE_STEP_MM, 1)
    m, _, _ = sample_plane(image, mask, o, t, n2, (-25, 25), (-10, 15), PLANE_STEP_MM, 0)
    _show_plane(axes[0], img, a, b, m, vmin, vmax, "aortic axis (x, +superior) x outward wall normal (y)")
    axes[0].plot(rel @ t, rel @ n2, "r.-", ms=3, lw=0.8)
    axes[0].plot(0, 0, "o", color="lime", ms=6, mec="k")
    for k, off in enumerate((1.5, 3.5)):
        img, a, b = sample_plane(image, ct, o + off * n2, t, bt, (-25, 25), (-15, 15), PLANE_STEP_MM, 1)
        m, _, _ = sample_plane(image, mask, o + off * n2, t, bt, (-25, 25), (-15, 15), PLANE_STEP_MM, 0)
        _show_plane(axes[1 + k], img, a, b, m, vmin, vmax, f"wall-parallel plane {off} mm outside the wall (x along the axis, y around the wall)")
        axes[1 + k].plot(rel @ t, rel @ bt, "r.-", ms=3, lw=0.8)
        axes[1 + k].plot(0, 0, "o", color="lime", ms=6, mec="k")
    ax = axes[3]
    halfw = int(round(45 / sp[1]))
    y0, y1 = max(0, yc - halfw), min(ct.shape[1], yc + halfw + 1)
    x0, x1 = max(0, xc - halfw), min(ct.shape[2], xc + halfw + 1)
    ax.imshow(ct[z0, y0:y1, x0:x1], cmap="gray", vmin=vmin, vmax=vmax, origin="upper", interpolation="nearest")
    if mask[z0, y0:y1, x0:x1].any():
        ax.contour(mask[z0, y0:y1, x0:x1].astype(float), levels=[0.5], colors="cyan", linewidths=0.8)
    ax.plot(oi[2] - x0, oi[1] - y0, "o", color="lime", ms=6, mec="k")
    ax.set_title(f"wide axial z={z0} (up = posterior, right = patient's right)", fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.suptitle(head, fontsize=10)
    plt.tight_layout()
    p2 = os.path.join(out_dir, f"{case_id}_patch{ost.label:02d}_wall.png")
    fig.savefig(p2, dpi=100)
    plt.close(fig)
    return [p1, p2]


def render(case_n: int, labels: list | None, branches: list | None, out_dir: str) -> list:
    cf = scorer.case_files(case_n)
    image, mask_image, _ = io_utils.load_case(cf["image"], cf["mask"])
    ct = sitk.GetArrayViewFromImage(image).astype(np.float32)
    mask = (sitk.GetArrayViewFromImage(mask_image) > 0).astype(np.uint8)
    cand = candidates.build(image, mask_image)
    inst = instances.build(cand)
    ostia = ostium.locate(cand, inst)
    traces = tracing.trace_all(cand, inst, ostia)
    fr = frame_mod.build(cand)
    labels = list(labels or [])
    if branches:
        res = filters.apply(cand, inst, ostia, traces, fr)
        kept = [l for l in res.kept if traces[l].seed_mm is not None]
        ids = {f"branch_{k:03d}": l for k, l in enumerate(kept, 1)}
        labels += [ids[b] for b in branches]
    os.makedirs(out_dir, exist_ok=True)
    out = []
    for l in labels:
        print("\n".join(profile_lines(cand, image, ct, mask, ostia[l], traces[l], fr)))
        out += render_candidate(cand, image, ct, mask, ostia[l], traces[l], fr, out_dir, cf["case_id"])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", type=int, required=True)
    ap.add_argument("--labels", type=int, nargs="*", help="wall-patch labels (as in the ledger and review tables)")
    ap.add_argument("--branches", nargs="*", help="branch ids of the current prediction, e.g. branch_005")
    ap.add_argument("--out", default=os.path.join(scorer.ROOT, "out", "review", "dense"))
    args = ap.parse_args(argv)
    if not args.labels and not args.branches:
        ap.error("give --labels or --branches")
    for p in render(args.case, args.labels, args.branches, args.out):
        print(p)


if __name__ == "__main__":
    main()
