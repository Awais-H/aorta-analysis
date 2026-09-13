"""D7 / section 3 sensitivity sweeps: re-run the filters over a range of one constant at a time
and report matched references (TP) and false positives at the working cutoff on the labelled
cases. The point is the shape of the curve, not the best value: flat means the constant does not
matter, a sharp peak is the dev set talking. Constants are NOT changed here.

    python sweeps.py --out results/sweeps.txt
    python sweeps.py --loo --out results/leave_one_out.txt   (SPEC section 3, defence 3)
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


# constants that change the traces themselves: the trace is recomputed per value and the
# instance-quality metrics (direction error, seed on the reference daughter) are reported too.
# 13 Sep: a lateral cap on the section centroid and a per-step turn clamp were evaluated here
# and rejected (SPEC D5): every setting lost a labelled true positive or added false positives
TRACE_SWEEPS = {
    "TRACE_MAX_TURN_DEG": [45.0, 60.0, 90.0, 120.0],
    "MIN_CROSS_SECTION_VOXELS": [2, 3, 4, 5],
    "TRACE_SLAB_HALF_MM": [0.5, 0.8, 1.0],
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
        prepped[n] = (cand, inst, ostia, traces, fr, ref, scorer.case_files(n).get("labels"))
    return prepped


def evaluate_traces(prepped):
    """Re-trace every case with the current config and score TP, FP, direction error and seed on
    branch at the working cutoff (scorer.score_case does the matching)."""
    import pipeline
    tp = fp = fn = 0
    dirs, seeds = [], []
    per_case = {}
    for n, (cand, inst, ostia, _, fr, ref, labels) in prepped.items():
        traces = tracing.trace_all(cand, inst, ostia)
        res = filters.apply(cand, inst, ostia, traces, fr)
        result = pipeline.assemble(f"subject{n:03d}", res.kept, ostia, traces)
        sc = scorer.score_case(result, ref, cutoffs=(config.MATCH_CUTOFF_MM,), label_path=labels)[f"{config.MATCH_CUTOFF_MM:g}mm"]
        tp += sc["tp"]; fp += sc["fp"]; fn += sc["fn"]
        d = sc.get("mean_direction_deg")
        if sc["tp"] and d == d:
            dirs.append((d, sc["tp"]))
        hits, tot = sc.get("seed_on_branch", (0, 0))
        seeds.append((hits, tot))
        per_case[n] = (sc["tp"], sc["fp"], d, f"{hits}/{tot}")
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    dmean = sum(d * w for d, w in dirs) / sum(w for _, w in dirs) if dirs else float("nan")
    return tp, fp, fn, f1, per_case, dmean, (sum(h for h, _ in seeds), sum(t for _, t in seeds))


def evaluate(prepped):
    tp = fp = fn = 0
    per_case = {}
    for n, (cand, inst, ostia, traces, fr, ref, _) in prepped.items():
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


def _case_f1(t, f, n_ref):
    fn = n_ref - t
    p = t / (t + f) if t + f else 0.0
    r = t / (t + fn) if t + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def leave_one_out(prepped, sweeps=SWEEPS) -> str:
    """SPEC section 3, defence 3: for each constant, pick the value that scores best on the other
    labelled cases (sum of F1) and apply it to the held-out case; rotate. If the held-out score at
    the tuned value does not beat the held-out score at the config (physical) value, tuning the
    constant to the dev set buys nothing and the physical value is not costing us. Reported per
    constant: mean held-out F1 at the config value, at the tuned value, and how often the tuned
    value differs from the config value."""
    cases = sorted(prepped)
    n_ref = {n: len(prepped[n][5]["daughters"]) for n in cases}
    lines = [f"Leave-one-out over the labelled cases {cases} at the {config.MATCH_CUTOFF_MM:g} mm cutoff.",
             "For each constant: the value that maximises the summed F1 of the other cases is applied to the held-out case.",
             "Columns: constant, mean held-out F1 at the config value, at the tuned value, cases where the tuned value differs (value chosen).", ""]
    grand_cfg, grand_tuned = [], []
    for name, values in sweeps.items():
        current = getattr(config, name)
        table = {}   # value -> {case: (tp, fp)}
        for v in values:
            setattr(config, name, v)
            _, _, _, _, pc = evaluate(prepped)
            table[v] = pc
        setattr(config, name, current)
        cfg_scores, tuned_scores, diffs = [], [], []
        for held in cases:
            others = [c for c in cases if c != held]
            def summed(v):
                return sum(_case_f1(*table[v][c], n_ref[c]) for c in others)
            # ties go to the config value, then to the value nearest it: the physical value wins a draw
            best = max(values, key=lambda v: (round(summed(v), 6), v == current, -abs(v - current)))
            f_cfg = _case_f1(*table[current][held], n_ref[held]) if current in table else float("nan")
            f_tuned = _case_f1(*table[best][held], n_ref[held])
            cfg_scores.append(f_cfg)
            tuned_scores.append(f_tuned)
            if best != current:
                diffs.append(f"s{held}:{best:g}({f_tuned - f_cfg:+.2f})")
        mc, mt = float(np.mean(cfg_scores)), float(np.mean(tuned_scores))
        grand_cfg.append(mc)
        grand_tuned.append(mt)
        lines.append(f"{name:32s} config {current:>6g}  held-out F1 {mc:.3f}  tuned {mt:.3f}  {'differs in ' + ', '.join(diffs) if diffs else 'tuned value = config value on every fold'}")
    lines += ["", f"Mean over constants: held-out F1 at config values {np.mean(grand_cfg):.3f}, at per-fold tuned values {np.mean(grand_tuned):.3f}.",
              "In-sample reference: " + f"F1 {evaluate(prepped)[3]:.3f} at the config values on all five cases."]
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="*", default=["labelled"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--loo", action="store_true", help="leave-one-out check of the filter constants instead of the sweeps")
    args = ap.parse_args(argv)
    cases = scorer._parse_cases(args.cases)
    prepped = prepare(cases)
    if args.loo:
        text = leave_one_out(prepped)
        print(text)
        if args.out:
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            with open(args.out, "w", encoding="utf-8") as f:
                f.write(text)
        return
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
    lines.append("Tracing constants (traces recomputed per value; columns add mean direction error and seed-on-branch at the cutoff):")
    for name, values in TRACE_SWEEPS.items():
        current = getattr(config, name)
        lines.append(f"{name} (config: {current})")
        for v in values:
            setattr(config, name, v)
            tp, fp, fn, f1, pc, dmean, (sh, sn) = evaluate_traces(prepped)
            mark = "*" if v == current else " "
            lines.append(f"  {mark}{v:>6}  TP {tp:>2}  FP {fp:>3}  FN {fn:>2}  F1 {f1:.2f}  dir {dmean:5.1f} deg  seed {sh}/{sn}   "
                         + "  ".join(f"s{n}:({t},{f},{'n/a' if d is None or d != d else round(d)}deg,{s})" for n, (t, f, d, s) in pc.items()))
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
