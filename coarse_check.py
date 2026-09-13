"""The 1.5 mm pseudo-labelled check (SPEC D7, section 1): resample each fine dev case (subjects 1
to 15, 0.8 mm z) to 1.5 mm isotropic, run the pipeline on the coarse pair, and score the coarse
predictions against the native predictions as pseudo-references with the scorer's matching at
the working cutoff. It is a measurement of what the hidden set's resolution costs, not a tuning
set: nothing here changes a constant.

Resampling: CT linear, mask nearest neighbour, SimpleITK, same physical frame (origin and
direction kept, size = native extent / 1.5 mm), so a coarse prediction and its native counterpart
are compared in the same mm coordinates.

Per case: matched branches with ostium, direction, radius and seed error; branches lost at coarse
resolution with the fate of the nearest coarse wall patch (which rule rejected it, or no patch);
branches gained with the fate of the nearest native patch. Aggregate: TP/FP/FN, errors, rule
flips, and the origin diameter of what was lost.

    python coarse_check.py --out results/coarse_check.txt [--cases 1-15] [--work out/coarse]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback

import numpy as np
import SimpleITK as sitk

import candidates
import config
import filters
import frame as frame_mod
import instances
import io_utils
import ostium
import pipeline
import scorer
import tracing

COARSE_SPACING_MM = 1.5  # the coarse acquisition of subjects 16 to 25 and of all five references
FINE_CASES = tuple(range(1, 16))  # the 0.8 mm cases (docs/atlas_all.csv)
CONTACT_MM = 3.0  # a wall patch whose voxels come within 3 mm (two coarse voxels) of the other resolution's ostium is the same contact
DIAMETER_BANDS = ((0.0, 2.5), (2.5, 3.5), (3.5, 5.0), (5.0, 99.0))  # origin diameter bands for the lost/kept histogram


def resample_pair(image: sitk.Image, mask: sitk.Image, spacing: float) -> tuple:
    """Resample CT (linear) and mask (nearest) onto a `spacing` isotropic grid covering the same
    physical extent, same origin and direction."""
    size = [max(1, int(np.ceil(sz * sp / spacing))) for sz, sp in zip(image.GetSize(), image.GetSpacing())]
    ref = sitk.Image(size, sitk.sitkFloat32)
    ref.SetSpacing([spacing] * 3)
    ref.SetOrigin(image.GetOrigin())
    ref.SetDirection(image.GetDirection())
    ct = sitk.Resample(image, ref, sitk.Transform(), sitk.sitkLinear, config.CT_BACKGROUND_HU, sitk.sitkFloat32)
    m = sitk.Resample(mask, ref, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    return ct, m


def run_stack(image: sitk.Image, mask_image: sitk.Image, case_id: str) -> dict:
    """The pipeline in-process, keeping every intermediate the report needs."""
    t0 = time.perf_counter()
    cand = candidates.build(image, mask_image)
    inst = instances.build(cand)
    ostia = ostium.locate(cand, inst)
    traces = tracing.trace_all(cand, inst, ostia)
    fr = frame_mod.build(cand)
    res = filters.apply(cand, inst, ostia, traces, fr)
    kept = [l for l in res.kept if traces[l].seed_mm is not None]
    result = pipeline.assemble(case_id, kept, ostia, traces)
    return {"cand": cand, "inst": inst, "ostia": ostia, "traces": traces, "frame": fr, "res": res,
            "kept": kept, "result": result, "seconds": time.perf_counter() - t0}


def _rules(res, label) -> str:
    hits = sorted({r for l, r, v in res.rejections if l == label})
    return ",".join(hits) if hits else "kept"


def _nearest_patch(ostia: dict, point_mm) -> tuple:
    if not ostia:
        return None, float("inf")
    lab = min(ostia, key=lambda l: np.linalg.norm(ostia[l].mm - point_mm))
    return lab, float(np.linalg.norm(ostia[lab].mm - point_mm))


def _wall_table(stack: dict) -> tuple:
    """(mm positions of every wall-patch voxel, their labels) for nearest-contact lookups."""
    inst, cand = stack["inst"], stack["cand"]
    labs, pts = [], []
    for l, idx in inst.wall_indices.items():
        if len(idx):
            pts.append(io_utils.index_to_mm(cand.image, np.asarray(idx, float)))
            labs.append(np.full(len(idx), int(l)))
    if not pts:
        return np.zeros((0, 3)), np.zeros(0, int)
    return np.vstack(pts), np.concatenate(labs)


def _nearest_wall(table: tuple, point_mm) -> tuple:
    """(label, distance) of the wall patch whose voxels come nearest to point_mm."""
    pts, labs = table
    if not len(pts):
        return None, float("inf")
    d = np.linalg.norm(pts - np.asarray(point_mm, float)[None, :], axis=1)
    k = int(np.argmin(d))
    return int(labs[k]), float(d[k])


def _in_region(stack: dict, label: int, point_mm) -> bool:
    """Is `point_mm` inside the watershed region of `label` on this stack's working grid?"""
    cand, inst = stack["cand"], stack["inst"]
    idx = np.round(io_utils.mm_to_index(cand.image, point_mm)).astype(int)
    if np.any(idx < 0) or np.any(idx >= np.array(inst.labels.shape)):
        return False
    return bool(inst.labels[tuple(idx)] == label)


def _band(d):
    if d is None:
        return "n/a"
    for lo, hi in DIAMETER_BANDS:
        if lo <= d < hi:
            return f"{lo:g}-{hi:g}" if hi < 99 else f">={lo:g}"
    return "n/a"


def check_case(n: int, work_dir: str, data_dir: str = scorer.DATA_DIR) -> tuple[list, dict]:
    cf = scorer.case_files(n, data_dir)
    cid = cf["case_id"]
    lines = []
    image, mask_image, _ = io_utils.load_case(cf["image"], cf["mask"])
    native_sp = image.GetSpacing()
    ct_c, m_c = resample_pair(image, mask_image, COARSE_SPACING_MM)
    cdir = os.path.join(work_dir, "data", cid)
    os.makedirs(cdir, exist_ok=True)
    sitk.WriteImage(ct_c, os.path.join(cdir, f"orig{n}.nii.gz"))
    sitk.WriteImage(m_c, os.path.join(cdir, f"mask{n}.nii.gz"))
    nat = run_stack(image, mask_image, cid)
    coa = run_stack(ct_c, m_c, cid)
    pdir = os.path.join(work_dir, "predictions")
    os.makedirs(pdir, exist_ok=True)
    import json
    with open(os.path.join(pdir, f"{cid}_coarse.json"), "w", encoding="utf-8") as f:
        json.dump(coa["result"], f, indent=2)
    with open(os.path.join(pdir, f"{cid}_native.json"), "w", encoding="utf-8") as f:
        json.dump(nat["result"], f, indent=2)

    P, R = coa["result"]["daughters"], nat["result"]["daughters"]
    pairs = scorer.match(P, R, config.MATCH_CUTOFF_MM)
    tp, fp, fn = len(pairs), len(P) - len(pairs), len(R) - len(pairs)
    coa_wall, nat_wall = _wall_table(coa), _wall_table(nat)
    coarse_match_of = {coa["kept"][i]: R[j]["instance_id"] for i, j, _ in pairs}   # coarse label -> matched native branch id
    native_match_of = {nat["kept"][j]: P[i]["instance_id"] for i, j, _ in pairs}   # native label -> matched coarse branch id
    lines.append(f"== {cid}: native {native_sp[0]:.2f}x{native_sp[1]:.2f}x{native_sp[2]:.2f} mm -> {COARSE_SPACING_MM} mm iso; "
                 f"native {len(R)} kept of {nat['inst'].n} patches ({nat['seconds']:.1f} s), coarse {len(P)} kept of {coa['inst'].n} patches ({coa['seconds']:.1f} s); "
                 f"threshold {nat['cand'].threshold_hu:.0f} -> {coa['cand'].threshold_hu:.0f} HU")
    lines.append(f"  matched {tp}, lost {fn}, gained {fp}")
    stats = {"tp": tp, "fp": fp, "fn": fn, "ostium": [], "direction": [], "radius": [], "seed_dist": [], "seed_in_native": [],
             "lost_bands": [], "kept_bands": [], "lost_fates": [], "gained_fates": [], "invariants": 0}
    for i, j, d in pairs:
        p, r = P[i], R[j]
        ang = scorer.angle_deg(p["direction_xyz"], r["direction_xyz"])
        dr = abs(float(p["radius_mm"]) - float(r["radius_mm"]))
        sd = float(np.linalg.norm(np.subtract(p["seed_xyz_mm"], r["seed_xyz_mm"])))
        nat_label = nat["kept"][j]
        on = _in_region(nat, nat_label, p["seed_xyz_mm"])
        m = nat["res"].measurements.get(nat_label, {})
        stats["ostium"].append(d)
        stats["direction"].append(ang)
        stats["radius"].append(dr)
        stats["seed_dist"].append(sd)
        stats["seed_in_native"].append(on)
        stats["kept_bands"].append(_band(m.get("origin_diameter_mm")))
        lines.append(f"    {p['instance_id']} <- native {r['instance_id']} (patch {nat_label}, diam {m.get('origin_diameter_mm')}): "
                     f"ostium {d:.2f} mm, direction {ang:.0f} deg, radius {float(p['radius_mm']):.2f} vs {float(r['radius_mm']):.2f}, "
                     f"seed {sd:.2f} mm apart, seed in native region {'yes' if on else 'no'}"
                     f"{'  flags ' + str(coa['res'].flags.get(coa['kept'][i])) if coa['res'].flags.get(coa['kept'][i]) else ''}")
    matched_ref = {j for _, j, _ in pairs}
    for j, r in enumerate(R):
        if j in matched_ref:
            continue
        nat_label = nat["kept"][j]
        m = nat["res"].measurements.get(nat_label, {})
        lab, dist = _nearest_patch(coa["ostia"], np.array(r["ostium_xyz_mm"]))
        wlab, wdist = _nearest_wall(coa_wall, np.array(r["ostium_xyz_mm"]))
        if lab is None or dist > config.MATCH_CUTOFF_MM:
            if wlab is None or wdist > CONTACT_MM:
                fate = f"contact erased: no coarse wall voxel within {CONTACT_MM:g} mm of the native ostium (nearest {wdist:.1f} mm)"
                key = "contact_erased"
            else:
                wr = _rules(coa["res"], wlab)
                if wr == "kept":
                    other = coarse_match_of.get(wlab)
                    fate = (f"merged: coarse patch {wlab} touches the native ostium ({wdist:.1f} mm) and was kept" + (f", matched to native {other}" if other else ", unmatched") + f"; its own ostium is {dist:.1f} mm away")
                    key = "merged_into_kept"
                else:
                    cm = coa["res"].measurements.get(wlab, {})
                    fate = f"coarse patch {wlab} touches the native ostium ({wdist:.1f} mm) but was REJECTED {wr} (wall {cm.get('wall_voxels')} vox, diam {cm.get('origin_diameter_mm')}, path {cm.get('path_mm')})"
                    key = wr
        else:
            rules = _rules(coa["res"], lab)
            if rules == "kept":
                fate = f"coarse patch {lab} at {dist:.1f} mm was kept but matched another native branch"
                key = "kept_elsewhere"
            else:
                cm = coa["res"].measurements.get(lab, {})
                fate = f"coarse patch {lab} at {dist:.1f} mm REJECTED {rules} (wall {cm.get('wall_voxels')} vox, diam {cm.get('origin_diameter_mm')}, path {cm.get('path_mm')})"
                key = rules
        stats["lost_bands"].append(_band(m.get("origin_diameter_mm")))
        stats["lost_fates"].append(key)
        lines.append(f"    LOST native {r['instance_id']} (patch {nat_label}, diam {m.get('origin_diameter_mm')}, wall {m.get('wall_voxels')} vox, "
                     f"flags {nat['res'].flags.get(nat_label, [])}): {fate}")
    matched_pred = {i for i, _, _ in pairs}
    for i, p in enumerate(P):
        if i in matched_pred:
            continue
        coa_label = coa["kept"][i]
        cm = coa["res"].measurements.get(coa_label, {})
        lab, dist = _nearest_patch(nat["ostia"], np.array(p["ostium_xyz_mm"]))
        wlab, wdist = _nearest_wall(nat_wall, np.array(p["ostium_xyz_mm"]))
        if lab is None or dist > config.MATCH_CUTOFF_MM:
            if wlab is None or wdist > CONTACT_MM:
                fate = f"new contact: no native wall voxel within {CONTACT_MM:g} mm of the coarse ostium (nearest {wdist:.1f} mm)"
                key = "new_contact"
            else:
                wr = _rules(nat["res"], wlab)
                nm = nat["res"].measurements.get(wlab, {})
                if wr == "kept":
                    other = native_match_of.get(wlab)
                    fate = f"split: native patch {wlab} touches the coarse ostium ({wdist:.1f} mm) and was kept" + (f", matched to coarse {other}" if other else ", unmatched") + f"; its own ostium is {dist:.1f} mm away"
                    key = "split_from_kept"
                else:
                    fate = f"native patch {wlab} touches the coarse ostium ({wdist:.1f} mm) but was REJECTED {wr} (wall {nm.get('wall_voxels')} vox, diam {nm.get('origin_diameter_mm')}, path {nm.get('path_mm')})"
                    key = wr
        else:
            rules = _rules(nat["res"], lab)
            nm = nat["res"].measurements.get(lab, {})
            if rules == "kept":
                fate = f"native patch {lab} at {dist:.1f} mm was kept but matched another coarse branch"
                key = "kept_elsewhere"
            else:
                fate = f"native patch {lab} at {dist:.1f} mm was REJECTED {rules} (wall {nm.get('wall_voxels')} vox, diam {nm.get('origin_diameter_mm')}, path {nm.get('path_mm')})"
                key = rules
        stats["gained_fates"].append(key)
        lines.append(f"    GAINED coarse {p['instance_id']} (patch {coa_label}, diam {cm.get('origin_diameter_mm')}, wall {cm.get('wall_voxels')} vox, "
                     f"trace {coa['traces'][coa_label].method}/{coa['traces'][coa_label].stop_reason}, flags {coa['res'].flags.get(coa_label, [])}): {fate}")
    viol = scorer.check_invariants(coa["result"], coa["cand"], coa["inst"], {"label_to_branch": {str(l): d["instance_id"] for l, d in zip(coa["kept"], P)}})
    reach = [v for v in viol if "watershed region reaches" in v]
    viol = [v for v in viol if v not in reach]
    stats["invariants"] = len(viol)
    lines.append(("  coarse invariants: clean" if not viol else f"  coarse invariants: {len(viol)} violation(s): " + "; ".join(viol))
                 + (f" (plus {len(reach)} watershed regions reaching over {config.INVARIANT_REGION_MAX_PATH_MM:g} mm, information only)" if reach else ""))
    lines.append("")
    return lines, stats


def _mean(xs):
    return float(np.mean(xs)) if xs else float("nan")


def report(cases: list, work_dir: str, data_dir: str = scorer.DATA_DIR) -> str:
    lines = [f"1.5 mm pseudo-labelled check, commit {scorer._git_head()}, {time.strftime('%Y-%m-%d %H:%M')}",
             f"Each fine case is resampled to {COARSE_SPACING_MM} mm isotropic (CT linear, mask nearest neighbour, same physical frame) and the",
             f"coarse predictions are matched to the native predictions (pseudo-references) with the scorer at the {config.MATCH_CUTOFF_MM:g} mm cutoff.",
             "The native predictions are not ground truth: a branch 'lost' may be a native false positive and one 'gained' may be real.",
             "This is a measurement of the resolution regime of the hidden set, not a tuning set (SPEC section 3).", ""]
    all_stats = {}
    for n in cases:
        try:
            case_lines, stats = check_case(n, work_dir, data_dir)
        except Exception as e:  # noqa: BLE001
            case_lines = [f"== subject{n:03d}: FAILED {type(e).__name__}: {e}", "   " + traceback.format_exc().strip().splitlines()[-1], ""]
            stats = None
        print("\n".join(case_lines), file=sys.stderr)
        lines += case_lines
        if stats is not None:
            all_stats[n] = stats
    tp = sum(s["tp"] for s in all_stats.values())
    fp = sum(s["fp"] for s in all_stats.values())
    fn = sum(s["fn"] for s in all_stats.values())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    ost = [d for s in all_stats.values() for d in s["ostium"]]
    ang = [a for s in all_stats.values() for a in s["direction"]]
    rad = [r for s in all_stats.values() for r in s["radius"]]
    sd = [r for s in all_stats.values() for r in s["seed_dist"]]
    on = [r for s in all_stats.values() for r in s["seed_in_native"]]
    lines.append(f"AGGREGATE over {len(all_stats)} cases (coarse vs native, {config.MATCH_CUTOFF_MM:g} mm cutoff)")
    lines.append(f"  matched {tp}, lost {fn}, gained {fp}: precision {prec:.2f}, recall {rec:.2f}, F1 {f1:.2f}")
    lines.append(f"  matched branches: ostium error mean {_mean(ost):.2f} mm (median {np.median(ost) if ost else float('nan'):.2f}, max {max(ost) if ost else float('nan'):.2f}); "
                 f"direction error mean {_mean(ang):.1f} deg (median {np.median(ang) if ang else float('nan'):.1f}, over 30 deg: {sum(a > 30 for a in ang)}); "
                 f"radius abs error mean {_mean(rad):.2f} mm; seed-to-seed mean {_mean(sd):.2f} mm; coarse seed inside the native branch region {sum(on)}/{len(on)}")
    lines.append(f"  per case (matched/lost/gained): " + ", ".join(f"{n}: {s['tp']}/{s['fn']}/{s['fp']}" for n, s in all_stats.items()))
    bands = [f"{lo:g}-{hi:g}" if hi < 99 else f">={lo:g}" for lo, hi in DIAMETER_BANDS]
    lines.append("  native origin diameter (mm) of matched vs lost branches:")
    for b in bands + ["n/a"]:
        k = sum(s["kept_bands"].count(b) for s in all_stats.values())
        l_ = sum(s["lost_bands"].count(b) for s in all_stats.values())
        if k or l_:
            lines.append(f"    {b:>8}: matched {k:3d}, lost {l_:3d}  ({l_ / (k + l_) * 100:.0f}% lost)")
    from collections import Counter
    lf = Counter(f for s in all_stats.values() for f in s["lost_fates"])
    gf = Counter(f for s in all_stats.values() for f in s["gained_fates"])
    lines.append("  why a native branch is lost at 1.5 mm (fate of the nearest coarse wall patch): " + ", ".join(f"{k} {v}" for k, v in lf.most_common()))
    lines.append("  where a coarse-only branch comes from (fate of the nearest native wall patch): " + ", ".join(f"{k} {v}" for k, v in gf.most_common()))
    inv = sum(s["invariants"] for s in all_stats.values())
    lines.append(f"  coarse invariant violations: {inv}")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", nargs="*", default=[f"{FINE_CASES[0]}-{FINE_CASES[-1]}"])
    ap.add_argument("--work", default=os.path.join(scorer.ROOT, "out", "coarse"), help="resampled volumes and both prediction sets go here")
    ap.add_argument("--data-dir", default=scorer.DATA_DIR)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    text = report(scorer._parse_cases(args.cases), args.work, args.data_dir)
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
