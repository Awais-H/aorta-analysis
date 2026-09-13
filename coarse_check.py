"""The 1.5 mm pseudo-labelled check (SPEC D7, section 1): resample each fine dev case (subjects 1
to 15, 0.8 mm z) to 1.5 mm isotropic, run the pipeline on the coarse pair, and score the coarse
predictions against the native predictions as pseudo-references with the scorer's matching at
the working cutoff. It is a measurement of what the hidden set's resolution costs, not a tuning
set: nothing here changes a constant.

Resampling: CT linear, mask nearest neighbour, SimpleITK, same physical frame (origin and
direction kept, size = native extent / 1.5 mm), so a coarse prediction and its native counterpart
are compared in the same mm coordinates.

Per case: matched branches with ostium, direction, radius and seed error; branches lost at coarse
resolution with the fate of the nearest coarse wall patch (which rule rejected it, merged into a
kept neighbour, or no contact at all); branches gained with the fate of the nearest native patch.
Aggregate: TP/FP/FN, errors, rule flips, and the origin diameter of what was lost.

`compare_stacks` is the shared core: perturb_check.py uses it with a perturbed run in place of
the coarse one.

    python coarse_check.py --out results/coarse_check.txt [--cases 1-15] [--work out/coarse]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from collections import Counter

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
CONTACT_MM = 3.0  # a wall patch whose voxels come within 3 mm (two coarse voxels) of the other run's ostium is the same contact
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


def compare_stacks(ref: dict, alt: dict, ref_name: str, alt_name: str) -> tuple[list, dict]:
    """Match `alt`'s kept branches to `ref`'s (pseudo-references) at the working cutoff and
    explain every branch that is lost or gained by the fate of the wall patch under it."""
    lines = []
    P, R = alt["result"]["daughters"], ref["result"]["daughters"]
    pairs = scorer.match(P, R, config.MATCH_CUTOFF_MM)
    tp, fp, fn = len(pairs), len(P) - len(pairs), len(R) - len(pairs)
    alt_wall, ref_wall = _wall_table(alt), _wall_table(ref)
    alt_match_of = {alt["kept"][i]: R[j]["instance_id"] for i, j, _ in pairs}
    ref_match_of = {ref["kept"][j]: P[i]["instance_id"] for i, j, _ in pairs}
    lines.append(f"  matched {tp}, lost {fn}, gained {fp}")
    stats = {"tp": tp, "fp": fp, "fn": fn, "ostium": [], "direction": [], "radius": [], "seed_dist": [], "seed_in_ref": [],
             "lost_bands": [], "kept_bands": [], "lost_fates": [], "gained_fates": [], "invariants": 0,
             "pairs": [(P[i]["instance_id"], R[j]["instance_id"]) for i, j, _ in pairs]}
    for i, j, d in pairs:
        p, r = P[i], R[j]
        ang = scorer.angle_deg(p["direction_xyz"], r["direction_xyz"])
        dr = abs(float(p["radius_mm"]) - float(r["radius_mm"]))
        sd = float(np.linalg.norm(np.subtract(p["seed_xyz_mm"], r["seed_xyz_mm"])))
        ref_label = ref["kept"][j]
        on = _in_region(ref, ref_label, p["seed_xyz_mm"])
        m = ref["res"].measurements.get(ref_label, {})
        stats["ostium"].append(d)
        stats["direction"].append(ang)
        stats["radius"].append(dr)
        stats["seed_dist"].append(sd)
        stats["seed_in_ref"].append(on)
        stats["kept_bands"].append(_band(m.get("origin_diameter_mm")))
        af = alt["res"].flags.get(alt["kept"][i])
        lines.append(f"    {p['instance_id']} <- {ref_name} {r['instance_id']} (patch {ref_label}, diam {m.get('origin_diameter_mm')}): "
                     f"ostium {d:.2f} mm, direction {ang:.0f} deg, radius {float(p['radius_mm']):.2f} vs {float(r['radius_mm']):.2f}, "
                     f"seed {sd:.2f} mm apart, seed in {ref_name} region {'yes' if on else 'no'}" + (f"  flags {af}" if af else ""))
    matched_ref = {j for _, j, _ in pairs}
    for j, r in enumerate(R):
        if j in matched_ref:
            continue
        ref_label = ref["kept"][j]
        m = ref["res"].measurements.get(ref_label, {})
        lab, dist = _nearest_patch(alt["ostia"], np.array(r["ostium_xyz_mm"]))
        wlab, wdist = _nearest_wall(alt_wall, np.array(r["ostium_xyz_mm"]))
        if lab is None or dist > config.MATCH_CUTOFF_MM:
            if wlab is None or wdist > CONTACT_MM:
                fate = f"contact erased: no {alt_name} wall voxel within {CONTACT_MM:g} mm of the {ref_name} ostium (nearest {wdist:.1f} mm)"
                key = "contact_erased"
            else:
                wr = _rules(alt["res"], wlab)
                if wr == "kept":
                    other = alt_match_of.get(wlab)
                    fate = (f"merged: {alt_name} patch {wlab} touches the {ref_name} ostium ({wdist:.1f} mm) and was kept"
                            + (f", matched to {ref_name} {other}" if other else ", unmatched") + f"; its own ostium is {dist:.1f} mm away")
                    key = "merged_into_kept"
                else:
                    cm = alt["res"].measurements.get(wlab, {})
                    fate = (f"{alt_name} patch {wlab} touches the {ref_name} ostium ({wdist:.1f} mm) but was REJECTED {wr} "
                            f"(wall {cm.get('wall_voxels')} vox, diam {cm.get('origin_diameter_mm')}, path {cm.get('path_mm')})")
                    key = wr
        else:
            rules = _rules(alt["res"], lab)
            if rules == "kept":
                fate = f"{alt_name} patch {lab} at {dist:.1f} mm was kept but matched another {ref_name} branch"
                key = "kept_elsewhere"
            else:
                cm = alt["res"].measurements.get(lab, {})
                fate = f"{alt_name} patch {lab} at {dist:.1f} mm REJECTED {rules} (wall {cm.get('wall_voxels')} vox, diam {cm.get('origin_diameter_mm')}, path {cm.get('path_mm')})"
                key = rules
        stats["lost_bands"].append(_band(m.get("origin_diameter_mm")))
        stats["lost_fates"].append(key)
        lines.append(f"    LOST {ref_name} {r['instance_id']} (patch {ref_label}, diam {m.get('origin_diameter_mm')}, wall {m.get('wall_voxels')} vox, "
                     f"flags {ref['res'].flags.get(ref_label, [])}): {fate}")
    matched_alt = {i for i, _, _ in pairs}
    for i, p in enumerate(P):
        if i in matched_alt:
            continue
        alt_label = alt["kept"][i]
        cm = alt["res"].measurements.get(alt_label, {})
        lab, dist = _nearest_patch(ref["ostia"], np.array(p["ostium_xyz_mm"]))
        wlab, wdist = _nearest_wall(ref_wall, np.array(p["ostium_xyz_mm"]))
        if lab is None or dist > config.MATCH_CUTOFF_MM:
            if wlab is None or wdist > CONTACT_MM:
                fate = f"new contact: no {ref_name} wall voxel within {CONTACT_MM:g} mm of the {alt_name} ostium (nearest {wdist:.1f} mm)"
                key = "new_contact"
            else:
                wr = _rules(ref["res"], wlab)
                nm = ref["res"].measurements.get(wlab, {})
                if wr == "kept":
                    other = ref_match_of.get(wlab)
                    fate = (f"split: {ref_name} patch {wlab} touches the {alt_name} ostium ({wdist:.1f} mm) and was kept"
                            + (f", matched to {alt_name} {other}" if other else ", unmatched") + f"; its own ostium is {dist:.1f} mm away")
                    key = "split_from_kept"
                else:
                    fate = (f"{ref_name} patch {wlab} touches the {alt_name} ostium ({wdist:.1f} mm) but was REJECTED {wr} "
                            f"(wall {nm.get('wall_voxels')} vox, diam {nm.get('origin_diameter_mm')}, path {nm.get('path_mm')})")
                    key = wr
        else:
            rules = _rules(ref["res"], lab)
            nm = ref["res"].measurements.get(lab, {})
            if rules == "kept":
                fate = f"{ref_name} patch {lab} at {dist:.1f} mm was kept but matched another {alt_name} branch"
                key = "kept_elsewhere"
            else:
                fate = f"{ref_name} patch {lab} at {dist:.1f} mm was REJECTED {rules} (wall {nm.get('wall_voxels')} vox, diam {nm.get('origin_diameter_mm')}, path {nm.get('path_mm')})"
                key = rules
        stats["gained_fates"].append(key)
        lines.append(f"    GAINED {alt_name} {p['instance_id']} (patch {alt_label}, diam {cm.get('origin_diameter_mm')}, wall {cm.get('wall_voxels')} vox, "
                     f"trace {alt['traces'][alt_label].method}/{alt['traces'][alt_label].stop_reason}, flags {alt['res'].flags.get(alt_label, [])}): {fate}")
    viol = scorer.check_invariants(alt["result"], alt["cand"], alt["inst"], {"label_to_branch": {str(l): d["instance_id"] for l, d in zip(alt["kept"], P)}})
    reach = [v for v in viol if "watershed region reaches" in v]
    viol = [v for v in viol if v not in reach]
    stats["invariants"] = len(viol)
    lines.append((f"  {alt_name} invariants: clean" if not viol else f"  {alt_name} invariants: {len(viol)} violation(s): " + "; ".join(viol))
                 + (f" (plus {len(reach)} watershed regions reaching over {config.INVARIANT_REGION_MAX_PATH_MM:g} mm, information only)" if reach else ""))
    return lines, stats


def check_case(n: int, work_dir: str, data_dir: str = scorer.DATA_DIR) -> tuple[list, dict]:
    cf = scorer.case_files(n, data_dir)
    cid = cf["case_id"]
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
    with open(os.path.join(pdir, f"{cid}_coarse.json"), "w", encoding="utf-8") as f:
        json.dump(coa["result"], f, indent=2)
    with open(os.path.join(pdir, f"{cid}_native.json"), "w", encoding="utf-8") as f:
        json.dump(nat["result"], f, indent=2)
    head = (f"== {cid}: native {native_sp[0]:.2f}x{native_sp[1]:.2f}x{native_sp[2]:.2f} mm -> {COARSE_SPACING_MM} mm iso; "
            f"native {len(nat['kept'])} kept of {nat['inst'].n} patches ({nat['seconds']:.1f} s), coarse {len(coa['kept'])} kept of {coa['inst'].n} patches ({coa['seconds']:.1f} s); "
            f"threshold {nat['cand'].threshold_hu:.0f} -> {coa['cand'].threshold_hu:.0f} HU")
    lines, stats = compare_stacks(nat, coa, "native", "coarse")
    return [head] + lines + [""], stats


def _mean(xs):
    return float(np.mean(xs)) if xs else float("nan")


def aggregate_lines(all_stats: dict, ref_name: str, alt_name: str) -> list:
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
    on = [r for s in all_stats.values() for r in s["seed_in_ref"]]
    lines = [f"AGGREGATE over {len(all_stats)} cases ({alt_name} vs {ref_name}, {config.MATCH_CUTOFF_MM:g} mm cutoff)",
             f"  matched {tp}, lost {fn}, gained {fp}: precision {prec:.2f}, recall {rec:.2f}, F1 {f1:.2f}",
             f"  matched branches: ostium error mean {_mean(ost):.2f} mm (median {np.median(ost) if ost else float('nan'):.2f}, max {max(ost) if ost else float('nan'):.2f}); "
             f"direction error mean {_mean(ang):.1f} deg (median {np.median(ang) if ang else float('nan'):.1f}, over 30 deg: {sum(a > 30 for a in ang)}); "
             f"radius abs error mean {_mean(rad):.2f} mm; seed-to-seed mean {_mean(sd):.2f} mm; {alt_name} seed inside the {ref_name} branch region {sum(on)}/{len(on)}",
             "  per case (matched/lost/gained): " + ", ".join(f"{n}: {s['tp']}/{s['fn']}/{s['fp']}" for n, s in all_stats.items())]
    bands = [f"{lo:g}-{hi:g}" if hi < 99 else f">={lo:g}" for lo, hi in DIAMETER_BANDS]
    lines.append(f"  {ref_name} origin diameter (mm) of matched vs lost branches:")
    for b in bands + ["n/a"]:
        k = sum(s["kept_bands"].count(b) for s in all_stats.values())
        l_ = sum(s["lost_bands"].count(b) for s in all_stats.values())
        if k or l_:
            lines.append(f"    {b:>8}: matched {k:3d}, lost {l_:3d}  ({l_ / (k + l_) * 100:.0f}% lost)")
    lf = Counter(f for s in all_stats.values() for f in s["lost_fates"])
    gf = Counter(f for s in all_stats.values() for f in s["gained_fates"])
    lines.append(f"  why a {ref_name} branch is lost (fate of the nearest {alt_name} wall patch): " + ", ".join(f"{k} {v}" for k, v in lf.most_common()))
    lines.append(f"  where a {alt_name}-only branch comes from (fate of the nearest {ref_name} wall patch): " + ", ".join(f"{k} {v}" for k, v in gf.most_common()))
    lines.append(f"  {alt_name} invariant violations: {sum(s['invariants'] for s in all_stats.values())}")
    return lines


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
    lines += aggregate_lines(all_stats, "native", "coarse")
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
