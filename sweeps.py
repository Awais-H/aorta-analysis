"""D7 / section 3 sensitivity sweeps: re-run the filters over a range of one constant at a time
and report matched references (TP) and false positives at the working cutoff on the labelled
cases. The point is the shape of the curve, not the best value: flat means the constant does not
matter, a sharp peak is the dev set talking. Constants are NOT changed here.

    python sweeps.py --out results/sweeps.txt
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

import candidates
import config
import filters
import frame as frame_mod
import instances
import io_utils
import ostium
import scorer
import tracing

SWEEPS = {
    "DEPARTURE_MM": [0.0, 1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0],
    "TANGENCY_ANGLE_DEG": [60.0, 65.0, 70.0, 75.0, 80.0],
    "MIN_ORIGIN_DIAMETER_MM": [1.5, 2.0, 2.5, 3.0],
    "REGION_VOLUME_CAP_ML": [0.5, 0.75, 1.0, 1.5, 2.0],
    "END_FACE_ANGLE_DEG": [10.0, 15.0, 20.0, 25.0, 30.0],
    "DUPLICATE_MM": [3.0, 4.0, 5.0, 6.0],
    "MIN_WALL_PATCH_VOXELS": [1, 2, 3, 4, 5],
    "ILIAC_LUMEN_MIN_DIAMETER_MM": [4.0, 5.0, 6.0, 7.0, 8.0],
    "ILIAC_CANDIDATE_MIN_DIAMETER_MM": [3.0, 4.0, 5.0, 6.0, 7.0],
    "BLOB_INSCRIBED_RADIUS_MM": [5.0, 6.0, 7.0, 8.0],
}


def prepare(cases):
    prepped = {}
    for n in cases:
        cf = scorer.case_files(n)
        ref = scorer.load_reference(n)
        if ref is None or cf["image"] is None:
            continue
        image, mask_image, _ = io_utils.load_case(cf["image"], cf["mask"])
        cand = candidates.build(image, mask_image)
        inst = instances.build(cand)
        ostia = ostium.locate(cand, inst)
        traces = tracing.trace_all(cand, inst, ostia)
        fr = frame_mod.build(cand)
        prepped[n] = (cand, inst, ostia, traces, fr, ref)
    return prepped


def evaluate(prepped):
    tp = fp = fn = 0
    per_case = {}
    for n, (cand, inst, ostia, traces, fr, ref) in prepped.items():
        res = filters.apply(cand, inst, ostia, traces, fr)
        pred = [{"instance_id": f"L{l}", "ostium_xyz_mm": list(ostia[l].mm)} for l in res.kept]
        pairs = scorer.match(pred, ref["daughters"], config.MATCH_CUTOFF_MM)
        t, f = len(pairs), len(pred) - len(pairs)
        tp += t
        fp += f
        fn += len(ref["daughters"]) - t
        per_case[n] = (t, f)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return tp, fp, fn, f1, per_case


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="*", default=["labelled"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    cases = scorer._parse_cases(args.cases)
    prepped = prepare(cases)
    lines = [f"Sensitivity sweeps at the {config.MATCH_CUTOFF_MM:g} mm cutoff on cases {sorted(prepped)}, commit {scorer._git_head()}, {time.strftime('%Y-%m-%d %H:%M')}",
             "One constant varied at a time, all others at their config values. Current value marked with *.",
             "Columns: value, TP, FP, FN, F1, then (TP, FP) per case.", ""]
    for name, values in SWEEPS.items():
        current = getattr(config, name)
        lines.append(f"{name} (config: {current})")
        for v in values:
            setattr(config, name, v)
            tp, fp, fn, f1, pc = evaluate(prepped)
            mark = "*" if v == current else " "
            lines.append(f"  {mark}{v:>6}  TP {tp:>2}  FP {fp:>3}  FN {fn:>2}  F1 {f1:.2f}   " + "  ".join(f"s{n}:({t},{f})" for n, (t, f) in pc.items()))
        setattr(config, name, current)
        lines.append("")
    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)


if __name__ == "__main__":
    sys.exit(main())
