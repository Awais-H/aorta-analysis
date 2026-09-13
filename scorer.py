"""D7 validation: Hungarian matching against the draft references, per-case and aggregate
metrics, invariant checks on unlabelled cases, and the CLI that produces the committed result
files.

References: docs/references/case_NN/annotations.json (same schema as our output plus guides and
per-branch masks) and data/subjectNNN/daughtersNN_draft.nii.gz (label volume, 1..N = branches, on
the CT grid). Seed-on-branch = the predicted seed voxel carries the matched reference label.
Radius error is computed only where the reference radius_mm is not null. Direction error is the
angle to the reference direction_xyz, which is the ostium-to-seed chord in every reference.

    python scorer.py predict    --cases 19 20 21 22 23        # run.py per case -> out/predictions/
    python scorer.py score      --cases 19 20 21 22 23 --out results/dev_scores.txt
    python scorer.py invariants --cases all --out results/invariants.txt
    python scorer.py ledger     --cases 19 20 21 22 23 --out results/reference_ledger.txt
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

import config

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
REFERENCE_DIR = os.path.join(ROOT, "docs", "references")
PRED_DIR = os.path.join(ROOT, "out", "predictions")


# ------------------------------------------------------------------ references and cases


def case_files(n: int, data_dir: str = DATA_DIR) -> dict:
    """Paths for subject n: image, mask, optional daughter label volume."""
    d = os.path.join(data_dir, f"subject{n:03d}")
    img = sorted(glob.glob(os.path.join(d, f"orig{n}.nii*")))
    msk = sorted(glob.glob(os.path.join(d, f"mask{n}.nii*")))
    lab = sorted(glob.glob(os.path.join(d, f"daughters{n}_draft.nii*")))
    return {"case_id": f"subject{n:03d}", "image": img[0] if img else None, "mask": msk[0] if msk else None,
            "labels": lab[0] if lab else None}


def load_reference(n: int, reference_dir: str = REFERENCE_DIR) -> dict | None:
    p = os.path.join(reference_dir, f"case_{n}", "annotations.json")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _daughters(obj) -> list:
    return obj.get("daughters", []) if isinstance(obj, dict) else list(obj)


# ------------------------------------------------------------------ matching and metrics


def match(pred: list, ref: list, cutoff_mm: float) -> list:
    """One-to-one pairs (pred_i, ref_j, distance_mm) with distance <= cutoff (Hungarian)."""
    if not pred or not ref:
        return []
    P = np.array([d["ostium_xyz_mm"] for d in pred], float)
    R = np.array([d["ostium_xyz_mm"] for d in ref], float)
    D = cdist(P, R)
    # cutoff-aware: pairs beyond the cutoff cost more than any number of in-cutoff pairs, so the
    # assignment first maximises the number of matches within the cutoff and only then minimises
    # distance. A plain assignment can pair a prediction with a far reference to lower the total
    # and then lose that pair to the cutoff.
    cost = np.where(D <= cutoff_mm, D, cutoff_mm * (D.size + 1))
    rows, cols = linear_sum_assignment(cost)
    return [(int(i), int(j), float(D[i, j])) for i, j in zip(rows, cols) if D[i, j] <= cutoff_mm]


def angle_deg(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))))


def seed_labels(pred: list, label_path: str | None) -> list:
    """Label value under each predicted seed in the reference daughter volume (0 = background,
    -1 = outside the volume, None = no label volume)."""
    if label_path is None or not pred:
        return [None] * len(pred)
    import io_utils
    lab_img, _ = io_utils.read_image(label_path)
    arr = io_utils.sitk.GetArrayViewFromImage(lab_img)
    out = []
    for d in pred:
        idx = np.round(io_utils.mm_to_index(lab_img, d["seed_xyz_mm"])).astype(int)
        if np.all(idx >= 0) and np.all(idx < np.array(arr.shape)):
            out.append(int(arr[tuple(idx)]))
        else:
            out.append(-1)
    return out


def _mean(xs) -> float:
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return float(np.mean(xs)) if xs else float("nan")


def score_case(pred, ref, cutoffs=config.MATCH_CUTOFFS_MM, label_path: str | None = None) -> dict:
    """Per-cutoff metrics for one case. `pred`/`ref` are result dicts or daughter lists."""
    pred, ref = _daughters(pred), _daughters(ref)
    labels = seed_labels(pred, label_path)
    out = {"n_pred": len(pred), "n_ref": len(ref), "n_ref_with_radius": sum(r.get("radius_mm") is not None for r in ref)}
    for c in cutoffs:
        pairs = match(pred, ref, c)
        tp, fp, fn = len(pairs), len(pred) - len(pairs), len(ref) - len(pairs)
        precision = tp / (tp + fp) if tp + fp else (1.0 if not ref else 0.0)
        recall = tp / (tp + fn) if tp + fn else 1.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        dists = [d for _, _, d in pairs]
        angles = [angle_deg(pred[i]["direction_xyz"], ref[j]["direction_xyz"]) for i, j, _ in pairs]
        radii = [abs(float(pred[i]["radius_mm"]) - float(ref[j]["radius_mm"]))
                 for i, j, _ in pairs if ref[j].get("radius_mm") is not None]
        on_branch = [labels[i] == ref[j].get("label_value") for i, j, _ in pairs if labels[i] is not None]
        out[f"{c:g}mm"] = {
            "tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1,
            "mean_ostium_mm": _mean(dists),
            "mean_direction_deg": _mean(angles),
            "seed_on_branch": (sum(on_branch), len(on_branch)),
            "mean_radius_abs_mm": _mean(radii), "n_radius": len(radii),
            "pairs": [(pred[i]["instance_id"], ref[j]["instance_id"], round(d, 2)) for i, j, d in pairs],
        }
    return out


def aggregate(case_scores: dict) -> dict:
    """Micro-averaged precision/recall/F1 and pooled distance metrics over cases."""
    out = {}
    keys = sorted({k for s in case_scores.values() for k in s if k.endswith("mm")})
    for k in keys:
        rows = [s[k] for s in case_scores.values() if k in s]
        tp, fp, fn = (sum(r[m] for r in rows) for m in ("tp", "fp", "fn"))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        seed = (sum(r["seed_on_branch"][0] for r in rows), sum(r["seed_on_branch"][1] for r in rows))
        n_rad = sum(r["n_radius"] for r in rows)
        out[k] = {
            "cases": len(rows), "tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "mean_ostium_mm": _mean([r["mean_ostium_mm"] for r in rows if r["tp"]]),
            "mean_direction_deg": _mean([r["mean_direction_deg"] for r in rows if r["tp"]]),
            "seed_on_branch": seed,
            "mean_radius_abs_mm": (sum(r["mean_radius_abs_mm"] * r["n_radius"] for r in rows if r["n_radius"]) / n_rad) if n_rad else float("nan"),
            "n_radius": n_rad,
        }
    return out


def format_scores(case_scores: dict, agg: dict, header_lines: list) -> str:
    def fmt(v, w=6, p=2):
        return f"{v:{w}.{p}f}" if isinstance(v, float) and not np.isnan(v) else f"{'n/a':>{w}}"

    cols = "cutoff   tp  fp  fn   prec    rec     f1  ostium_mm  dir_deg  seed_on_branch  radius_abs_mm(n)"
    lines = list(header_lines) + [""]
    for case, s in case_scores.items():
        lines.append(f"{case}: {s['n_pred']} predicted, {s['n_ref']} reference ({s['n_ref_with_radius']} with radius)")
        lines.append(cols)
        for k, r in s.items():
            if not k.endswith("mm"):
                continue
            sb = f"{r['seed_on_branch'][0]}/{r['seed_on_branch'][1]}"
            lines.append(f"{k:>6} {r['tp']:>4} {r['fp']:>3} {r['fn']:>3} {fmt(r['precision'])} {fmt(r['recall'])} {fmt(r['f1'])}"
                         f" {fmt(r['mean_ostium_mm'], 10)} {fmt(r['mean_direction_deg'], 8, 1)} {sb:>15}"
                         f" {fmt(r['mean_radius_abs_mm'], 12)}({r['n_radius']})")
        m = s.get(f"{config.MATCH_CUTOFF_MM:g}mm", {})
        if m.get("pairs"):
            lines.append(f"   matched at {config.MATCH_CUTOFF_MM:g} mm: " + ", ".join(f"{p}->{r} {d} mm" for p, r, d in m["pairs"]))
        lines.append("")
    lines.append(f"AGGREGATE over {len(case_scores)} cases (micro-averaged)")
    lines.append(cols)
    for k, r in agg.items():
        sb = f"{r['seed_on_branch'][0]}/{r['seed_on_branch'][1]}"
        lines.append(f"{k:>6} {r['tp']:>4} {r['fp']:>3} {r['fn']:>3} {fmt(r['precision'])} {fmt(r['recall'])} {fmt(r['f1'])}"
                     f" {fmt(r['mean_ostium_mm'], 10)} {fmt(r['mean_direction_deg'], 8, 1)} {sb:>15}"
                     f" {fmt(r['mean_radius_abs_mm'], 12)}({r['n_radius']})")
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ invariants


def check_invariants(result: dict, cand=None, inst=None, meta: dict | None = None) -> list:
    """Violations of the D7 invariants. JSON-only checks always run; image checks need `cand`,
    the region-extent check needs `inst` and `meta['label_to_branch']`, runtime needs `meta`."""
    v = []
    ds = _daughters(result)
    if result.get("parent", {}).get("instance_id") != "aorta":
        v.append("parent.instance_id != 'aorta'")
    ids = [d.get("instance_id") for d in ds]
    if len(set(ids)) != len(ids):
        v.append("duplicate instance ids")
    for k, d in enumerate(ds, 1):
        i = d.get("instance_id")
        if i != f"branch_{k:03d}":
            v.append(f"{i}: expected branch_{k:03d}")
        if d.get("parent_instance_id") != "aorta":
            v.append(f"{i}: parent_instance_id != 'aorta'")
        n = float(np.linalg.norm(d["direction_xyz"]))
        if abs(n - 1.0) > config.INVARIANT_UNIT_NORM_TOL:
            v.append(f"{i}: direction norm {n:.4f}")
        r = float(d["radius_mm"])
        if not (config.INVARIANT_RADIUS_MIN_MM <= r <= config.INVARIANT_RADIUS_MAX_MM):
            v.append(f"{i}: radius {r:.2f} mm outside [{config.INVARIANT_RADIUS_MIN_MM}, {config.INVARIANT_RADIUS_MAX_MM}]")
        chord = float(np.linalg.norm(np.subtract(d["seed_xyz_mm"], d["ostium_xyz_mm"])))
        if chord <= 0 or chord > config.SEED_DISTANCE_MM + config.INVARIANT_SEED_CHORD_TOL_MM:
            v.append(f"{i}: seed is {chord:.2f} mm from the ostium")
        if cand is not None:
            import io_utils
            seed_idx = np.round(io_utils.mm_to_index(cand.image, d["seed_xyz_mm"])).astype(int)
            if np.all(seed_idx >= 0) and np.all(seed_idx < np.array(cand.ct.shape)):
                if cand.ct[tuple(seed_idx)] <= cand.threshold_hu:
                    v.append(f"{i}: seed voxel is not bright ({cand.ct[tuple(seed_idx)]:.0f} HU)")
                o_idx = np.round(io_utils.mm_to_index(cand.image, d["ostium_xyz_mm"])).astype(int)
                o_idx = np.clip(o_idx, 0, np.array(cand.ct.shape) - 1)
                if cand.distance_mm[tuple(seed_idx)] <= cand.distance_mm[tuple(o_idx)]:
                    v.append(f"{i}: direction does not point away from the aorta")
            else:
                v.append(f"{i}: seed outside the working volume")
    O = np.array([d["ostium_xyz_mm"] for d in ds], float)
    if len(O) > 1:
        D = cdist(O, O)
        np.fill_diagonal(D, np.inf)
        if D.min() < config.INVARIANT_MIN_OSTIUM_SEPARATION_MM:
            close = int((D < config.INVARIANT_MIN_OSTIUM_SEPARATION_MM).sum() // 2)
            a, b = np.unravel_index(np.argmin(D), D.shape)
            v.append(f"{close} ostium pair(s) closer than {config.INVARIANT_MIN_OSTIUM_SEPARATION_MM:g} mm (closest {ids[a]}, {ids[b]}: {D.min():.2f} mm)")
    if meta is not None:
        frag = meta.get("fragment_log", {})
        if frag.get("components", 1) != 1:
            v.append(f"warning: mask has {frag['components']} components, {frag.get('discarded_voxels')} voxels discarded")
        total = meta.get("timings_s", {}).get("total")
        if total is not None and total > config.RUNTIME_CAP_S:
            v.append(f"runtime {total:.1f} s over the {config.RUNTIME_CAP_S:g} s cap")
        if inst is not None and cand is not None:
            import io_utils
            for label, bid in meta.get("label_to_branch", {}).items():
                d = next((x for x in ds if x["instance_id"] == bid), None)
                if d is None:
                    continue
                vox = np.argwhere(inst.labels == int(label))
                if len(vox) == 0:
                    continue
                o_idx = io_utils.mm_to_index(cand.image, d["ostium_xyz_mm"])
                reach = float(np.linalg.norm((vox - o_idx) * cand.spacing, axis=1).max())
                if reach > config.INVARIANT_REGION_MAX_PATH_MM:
                    v.append(f"{bid}: watershed region reaches {reach:.1f} mm from its ostium")
    if meta and meta.get("peak_memory_mb") is not None and meta["peak_memory_mb"] > config.PEAK_MEMORY_CAP_GB * 1024:
        v.append(f"peak memory {meta['peak_memory_mb']:.0f} MB over the {config.PEAK_MEMORY_CAP_GB:g} GB cap")
    return v


# ------------------------------------------------------------------ CLI


def _git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    except Exception:
        return "unknown"


def _parse_cases(tokens: list) -> list:
    if not tokens or tokens == ["all"]:
        return list(range(1, 26))
    if tokens == ["labelled"]:
        return list(config.LABELLED_CASES)
    if tokens == ["regression"]:
        return list(config.REGRESSION_ORDER)
    out = []
    for t in tokens:
        if "-" in t:
            a, b = t.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(t))
    return out


def predict(cases: list, pred_dir: str = PRED_DIR, data_dir: str = DATA_DIR) -> dict:
    """Run run.py (subprocess, so the never-crash path is what is measured) per case."""
    os.makedirs(pred_dir, exist_ok=True)
    runs = {}
    for n in cases:
        cf = case_files(n, data_dir)
        out = os.path.join(pred_dir, f"{cf['case_id']}.json")
        meta = os.path.join(pred_dir, f"{cf['case_id']}_meta.json")
        rec = {"case_id": cf["case_id"], "output": out, "meta": meta, "returncode": None, "wall_s": None,
               "valid_json": False, "stderr_tail": ""}
        if cf["image"] is None or cf["mask"] is None:
            rec["stderr_tail"] = "input files missing"
            runs[n] = rec
            continue
        t0 = time.perf_counter()
        p = subprocess.run([sys.executable, os.path.join(ROOT, "run.py"), "--image", cf["image"], "--aorta-mask", cf["mask"],
                            "--output", out, "--meta-output", meta, "--case-id", cf["case_id"]],
                           capture_output=True, text=True, cwd=ROOT)
        rec["wall_s"] = round(time.perf_counter() - t0, 1)
        rec["returncode"] = p.returncode
        rec["stderr_tail"] = "\n".join(p.stderr.strip().splitlines()[-3:])
        try:
            with open(out, encoding="utf-8") as f:
                r = json.load(f)
            rec["valid_json"] = isinstance(r, dict) and "daughters" in r and check_invariants({**r, "daughters": []}) == []
            rec["n_daughters"] = len(r["daughters"])
        except Exception as e:  # noqa: BLE001
            rec["stderr_tail"] += f"\n[invalid JSON: {e}]"
        print(f"{cf['case_id']}: rc={rec['returncode']} {rec['wall_s']} s daughters={rec.get('n_daughters')} valid={rec['valid_json']}", file=sys.stderr)
        runs[n] = rec
    return runs


def score(cases: list, pred_dir: str = PRED_DIR, data_dir: str = DATA_DIR, reference_dir: str = REFERENCE_DIR) -> tuple:
    case_scores = {}
    for n in cases:
        ref = load_reference(n, reference_dir)
        if ref is None:
            print(f"case {n}: no reference, skipped", file=sys.stderr)
            continue
        cf = case_files(n, data_dir)
        pred_path = os.path.join(pred_dir, f"{cf['case_id']}.json")
        with open(pred_path, encoding="utf-8") as f:
            pred = json.load(f)
        case_scores[cf["case_id"]] = score_case(pred, ref, label_path=cf["labels"])
    return case_scores, aggregate(case_scores)


def invariants_report(cases: list, pred_dir: str = PRED_DIR, data_dir: str = DATA_DIR) -> str:
    import candidates
    import instances
    import io_utils
    runs = predict(cases, pred_dir, data_dir)
    lines = [f"D7 invariant checks on {len(cases)} cases, commit {_git_head()}, {time.strftime('%Y-%m-%d %H:%M')}",
             f"Runtime is run.py wall time including interpreter start-up; peak is the run.py process's peak resident memory (cap {config.PEAK_MEMORY_CAP_GB:g} GB, D10).",
             "Region-reach violations are the raw watershed basin flooding neighbouring bright tissue (information only; rule 4 uses the proximal 10 mm).", ""]
    summary = []
    for n, rec in runs.items():
        cid = rec["case_id"]
        head = f"{cid}: rc={rec['returncode']} wall={rec['wall_s']} s"
        if not rec["valid_json"]:
            lines += [head + "  INVALID OR MISSING JSON", "   " + rec["stderr_tail"].replace("\n", "\n   "), ""]
            summary.append((cid, "INVALID"))
            continue
        with open(rec["output"], encoding="utf-8") as f:
            result = json.load(f)
        meta = {}
        if os.path.exists(rec["meta"]):
            with open(rec["meta"], encoding="utf-8") as f:
                meta = json.load(f)
        crashed = "error" in meta
        cand = inst = None
        if not crashed:
            try:
                cf = case_files(n, data_dir)
                image, mask_image, _ = io_utils.load_case(cf["image"], cf["mask"])
                cand = candidates.build(image, mask_image)
                inst = instances.build(cand)
            except Exception as e:  # noqa: BLE001
                lines.append(f"   (image-based checks skipped: {type(e).__name__}: {e})")
        v = check_invariants(result, cand, inst, meta)
        t = meta.get("timings_s", {}).get("total")
        mem = meta.get("peak_memory_mb")
        head += f" pipeline={t} s peak={'n/a' if mem is None else f'{mem:.0f} MB'} daughters={len(result['daughters'])} wall_patches={meta.get('wall_patches')}"
        if crashed:
            head += "  CRASHED (empty daughters written)"
            lines += [head, "   " + meta["error"].strip().splitlines()[-1], ""]
            summary.append((cid, "CRASH"))
            continue
        lines.append(head + ("  OK" if not v else f"  {len(v)} violation(s)"))
        lines += ["   - " + x for x in v]
        lines.append("")
        summary.append((cid, "OK" if not v else f"{len(v)} violations"))
    lines.append("SUMMARY")
    lines += [f"  {cid}: {s}" for cid, s in summary]
    mems = [(rec["case_id"], m) for rec in runs.values() for m in [_meta_peak(rec)] if m is not None]
    walls = [(rec["case_id"], rec["wall_s"]) for rec in runs.values() if rec.get("wall_s") is not None]
    if mems:
        worst = max(mems, key=lambda x: x[1])
        lines.append(f"  peak memory: max {worst[1]:.0f} MB ({worst[0]}), median {np.median([m for _, m in mems]):.0f} MB over {len(mems)} cases; cap {config.PEAK_MEMORY_CAP_GB * 1024:.0f} MB")
    if walls:
        worst = max(walls, key=lambda x: x[1])
        lines.append(f"  wall time: max {worst[1]:.1f} s ({worst[0]}), mean {np.mean([w for _, w in walls]):.1f} s; organiser target {config.RUNTIME_CAP_S:.0f} s average")
    return "\n".join(lines) + "\n"


def _meta_peak(rec: dict):
    try:
        with open(rec["meta"], encoding="utf-8") as f:
            return json.load(f).get("peak_memory_mb")
    except Exception:  # noqa: BLE001
        return None


def reference_ledger(cases: list, data_dir: str = DATA_DIR, reference_dir: str = REFERENCE_DIR) -> str:
    """For every reference branch: the nearest wall patch, its fate in the filters, and every
    measurement the rules saw; then the surviving false positives. D7's failure ledger in text."""
    import candidates
    import filters
    import frame as frame_mod
    import instances
    import io_utils
    import ostium
    import tracing
    keys = ("wall_voxels", "wall_area_mm2", "proximal_ml", "path_mm", "image_edge_mm", "departure_mm", "origin_diameter_mm",
            "area_growth", "end_face_height_mm", "end_face_angle_deg", "area_ratio", "tangency_deg", "aspect_ratio")
    lines = [f"Reference ledger, commit {_git_head()}, {time.strftime('%Y-%m-%d %H:%M')}",
             "For each reference branch: nearest wall patch by ostium distance, kept or rejected (rule = value), measurements, flags.",
             "Then every kept patch further than the match cutoff from all references (false positives).", ""]
    for n in cases:
        ref = load_reference(n, reference_dir)
        cf = case_files(n, data_dir)
        if ref is None or cf["image"] is None:
            continue
        image, mask_image, _ = io_utils.load_case(cf["image"], cf["mask"])
        cand = candidates.build(image, mask_image)
        inst = instances.build(cand)
        ostia = ostium.locate(cand, inst)
        traces = tracing.trace_all(cand, inst, ostia)
        fr = frame_mod.build(cand)
        res = filters.apply(cand, inst, ostia, traces, fr)
        lines.append(f"== {cf['case_id']}: {inst.n} wall patches, {len(res.kept)} kept, {len(ref['daughters'])} references")
        for r in ref["daughters"]:
            R = np.array(r["ostium_xyz_mm"])
            if not ostia:
                lines.append(f"  {r['instance_id']}: no wall patches at all")
                continue
            lab = min(ostia, key=lambda l: np.linalg.norm(ostia[l].mm - R))
            d = float(np.linalg.norm(ostia[lab].mm - R))
            hits = [f"{rr}={v:.2f}" for l, rr, v in res.rejections if l == lab]
            fate = "KEPT" if lab in res.kept else "REJECTED " + ", ".join(hits)
            m = res.measurements.get(lab, {})
            tr = traces.get(lab)
            lines.append(f"  {r['instance_id']} (ref diam {r.get('origin_diameter_estimate_mm')}, {r.get('confidence')}) -> patch {lab} at {d:.2f} mm: {fate}")
            lines.append("      " + ", ".join(f"{k}={m[k]}" for k in keys if k in m)
                         + (f" | trace {tr.method}/{tr.stop_reason}" if tr is not None else "") + f" | flags {res.flags.get(lab, [])}")
        fps = [l for l in res.kept if all(np.linalg.norm(ostia[l].mm - np.array(r["ostium_xyz_mm"])) > config.MATCH_CUTOFF_MM for r in ref["daughters"])]
        lines.append(f"  false positives kept: {len(fps)}")
        for l in fps:
            m = res.measurements[l]
            lines.append(f"    patch {l}: " + ", ".join(f"{k}={m[k]}" for k in keys if k in m) + f" | trace {traces[l].method}/{traces[l].stop_reason} | flags {res.flags.get(l, [])}")
        lines.append("")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("predict", "score", "invariants", "ledger"):
        s = sub.add_parser(name)
        s.add_argument("--cases", nargs="*", default=["labelled"], help="numbers, ranges (19-23), 'labelled', 'regression' or 'all'")
        s.add_argument("--pred-dir", default=PRED_DIR)
        s.add_argument("--data-dir", default=DATA_DIR)
        s.add_argument("--out", default=None, help="write the report here (score, invariants)")
    args = ap.parse_args(argv)
    cases = _parse_cases(args.cases)
    if args.cmd == "predict":
        predict(cases, args.pred_dir, args.data_dir)
        return 0
    if args.cmd == "score":
        cs, agg = score(cases, args.pred_dir, args.data_dir)
        text = format_scores(cs, agg, [f"Dev-set scores, commit {_git_head()}, {time.strftime('%Y-%m-%d %H:%M')}",
                                       "References: docs/references (draft, expert review pending). Matching: Hungarian on ostium distance.",
                                       "seed_on_branch = predicted seed voxel carries the matched reference label; radius scored only where the reference radius is not null."])
    elif args.cmd == "ledger":
        text = reference_ledger(cases, args.data_dir)
    else:
        text = invariants_report(cases, args.pred_dir, args.data_dir)
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
