"""D7 validation: one-to-one Hungarian matching on ostium distance, per-case and aggregate
metrics, and invariant checks that need no reference.

Reference JSONs use the same schema as predictions (docs/Branchseed_challenge.pdf). Metrics per
cutoff: precision, recall, F1, mean matched ostium distance, mean direction angle error, mean
radius absolute error. Seed-inside-branch needs the image and is left to the invariants.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

import config


def _daughters(obj) -> list:
    return obj.get("daughters", []) if isinstance(obj, dict) else list(obj)


def match(pred: list, ref: list, cutoff_mm: float) -> list:
    """One-to-one pairs (pred_i, ref_j, distance_mm) with distance <= cutoff."""
    if not pred or not ref:
        return []
    P = np.array([d["ostium_xyz_mm"] for d in pred], float)
    R = np.array([d["ostium_xyz_mm"] for d in ref], float)
    D = cdist(P, R)
    rows, cols = linear_sum_assignment(D)
    return [(int(i), int(j), float(D[i, j])) for i, j in zip(rows, cols) if D[i, j] <= cutoff_mm]


def _angle_deg(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))))


def score_case(pred, ref, cutoffs=config.MATCH_CUTOFFS_MM) -> dict:
    pred, ref = _daughters(pred), _daughters(ref)
    out = {}
    for c in cutoffs:
        pairs = match(pred, ref, c)
        tp, fp, fn = len(pairs), len(pred) - len(pairs), len(ref) - len(pairs)
        precision = tp / (tp + fp) if tp + fp else (1.0 if not ref else 0.0)
        recall = tp / (tp + fn) if tp + fn else 1.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        dists = [d for _, _, d in pairs]
        angles = [_angle_deg(pred[i]["direction_xyz"], ref[j]["direction_xyz"]) for i, j, _ in pairs]
        radii = [abs(float(pred[i]["radius_mm"]) - float(ref[j]["radius_mm"])) for i, j, _ in pairs]
        out[f"{c:g}mm"] = {
            "tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1,
            "mean_ostium_mm": float(np.mean(dists)) if dists else float("nan"),
            "mean_direction_deg": float(np.nanmean(angles)) if angles else float("nan"),
            "mean_radius_abs_mm": float(np.mean(radii)) if radii else float("nan"),
            "pairs": [(pred[i]["instance_id"], ref[j]["instance_id"], round(d, 2)) for i, j, d in pairs],
        }
    return out


def aggregate(case_scores: dict) -> dict:
    """Micro-averaged precision/recall/F1 over cases, mean of the per-case distance metrics."""
    out = {}
    keys = set(k for s in case_scores.values() for k in s)
    for k in sorted(keys):
        rows = [s[k] for s in case_scores.values() if k in s]
        tp, fp, fn = (sum(r[m] for r in rows) for m in ("tp", "fp", "fn"))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        out[k] = {
            "cases": len(rows), "tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "mean_ostium_mm": float(np.nanmean([r["mean_ostium_mm"] for r in rows])) if rows else float("nan"),
            "mean_direction_deg": float(np.nanmean([r["mean_direction_deg"] for r in rows])) if rows else float("nan"),
            "mean_radius_abs_mm": float(np.nanmean([r["mean_radius_abs_mm"] for r in rows])) if rows else float("nan"),
        }
    return out


def check_invariants(result: dict, cand=None) -> list:
    """Violations of the D7 invariants that can be checked on the JSON (plus the image if given)."""
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
            a, b = np.unravel_index(np.argmin(D), D.shape)
            v.append(f"{ids[a]} and {ids[b]} are {D.min():.2f} mm apart")
    return v
