#!/usr/bin/env python3
"""Score predictions against reference annotations.

Implements the organisers' rubric as closely as the brief allows, so that the
sweep harness optimises the right thing:

* Discovery (45%): one-to-one matching, so duplicates count as false positives.
* Localisation (25%): physical distance between matched ostium centres.
* Quality (15%): whether the seed lies on the matched daughter, whether the
  direction follows the proximal path, and whether the radius is consistent.

The match radius is not published. Report at more than one radius and say which
you used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

DEFAULT_MATCH_RADII = (5.0, 10.0)


def _points(records: Sequence[dict], key: str) -> np.ndarray:
    if not records:
        return np.zeros((0, 3))
    return np.array([r[key] for r in records], dtype=np.float64)


def match_one_to_one(pred: Sequence[dict], ref: Sequence[dict], radius_mm: float):
    """Hungarian assignment on ostium distance, capped at ``radius_mm``."""
    if not pred or not ref:
        return [], list(range(len(pred))), list(range(len(ref)))
    cost = np.linalg.norm(
        _points(pred, "ostium_xyz_mm")[:, None, :] - _points(ref, "ostium_xyz_mm")[None, :, :],
        axis=2,
    )
    big = cost.max() + 1e6
    rows, cols = linear_sum_assignment(np.where(cost <= radius_mm, cost, big))
    pairs = [(int(r), int(c)) for r, c in zip(rows, cols) if cost[r, c] <= radius_mm]
    matched_pred = {r for r, _ in pairs}
    matched_ref = {c for _, c in pairs}
    return (pairs,
            [i for i in range(len(pred)) if i not in matched_pred],
            [j for j in range(len(ref)) if j not in matched_ref])


def _unit(vec) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float64)
    return v / max(np.linalg.norm(v), 1e-9)


def _distance_to_reference_path(seed: np.ndarray, reference: dict) -> float:
    """Distance from the predicted seed to the reference proximal centreline.

    Falls back to the ray from the ostium along the reference direction when
    the reference gives no explicit polyline.
    """
    poly = reference.get("proximal_centreline")
    if poly:
        pts = np.asarray(poly, dtype=np.float64)
        if pts.shape[0] == 1:
            return float(np.linalg.norm(seed - pts[0]))
        best = np.inf
        for a, b in zip(pts[:-1], pts[1:]):
            ab = b - a
            denom = float(ab @ ab)
            t = 0.0 if denom < 1e-12 else float(np.clip(((seed - a) @ ab) / denom, 0.0, 1.0))
            best = min(best, float(np.linalg.norm(seed - (a + t * ab))))
        return best
    ostium = np.asarray(reference["ostium_xyz_mm"], dtype=np.float64)
    direction = _unit(reference["direction_xyz"])
    along = float(np.clip((seed - ostium) @ direction, 0.0, 10.0))
    return float(np.linalg.norm(seed - (ostium + along * direction)))


def score_case(pred: Sequence[dict], ref: Sequence[dict], radius_mm: float) -> Dict:
    pairs, false_positives, false_negatives = match_one_to_one(pred, ref, radius_mm)
    tp = len(pairs)
    precision = tp / len(pred) if pred else (1.0 if not ref else 0.0)
    recall = tp / len(ref) if ref else (1.0 if not pred else 0.0)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

    ostium_errors, direction_errors, radius_errors, radius_rel, seed_errors = [], [], [], [], []
    for p, r in pairs:
        pp, rr = pred[p], ref[r]
        ostium_errors.append(float(np.linalg.norm(
            np.asarray(pp["ostium_xyz_mm"]) - np.asarray(rr["ostium_xyz_mm"]))))
        cos = float(np.clip(_unit(pp["direction_xyz"]) @ _unit(rr["direction_xyz"]), -1.0, 1.0))
        direction_errors.append(float(np.degrees(np.arccos(cos))))
        radius_errors.append(abs(float(pp["radius_mm"]) - float(rr["radius_mm"])))
        radius_rel.append(radius_errors[-1] / max(float(rr["radius_mm"]), 1e-6))
        seed_errors.append(_distance_to_reference_path(
            np.asarray(pp["seed_xyz_mm"], dtype=np.float64), rr))

    def stats(values: List[float]) -> Dict[str, Optional[float]]:
        if not values:
            return {"mean": None, "median": None, "max": None}
        return {"mean": float(np.mean(values)), "median": float(np.median(values)),
                "max": float(np.max(values))}

    return {
        "match_radius_mm": radius_mm,
        "n_pred": len(pred), "n_ref": len(ref),
        "tp": tp, "fp": len(false_positives), "fn": len(false_negatives),
        "precision": precision, "recall": recall, "f1": f1,
        "ostium_mm": stats(ostium_errors),
        "direction_deg": stats(direction_errors),
        "radius_mm": stats(radius_errors),
        "radius_relative": stats(radius_rel),
        "seed_on_daughter_mm": stats(seed_errors),
    }


def composite_score(case_scores: Sequence[Dict]) -> float:
    """Single number mirroring the rubric's weights, for the sweep harness."""
    if not case_scores:
        return 0.0
    f1 = float(np.mean([s["f1"] for s in case_scores]))
    loc = [s["ostium_mm"]["mean"] for s in case_scores if s["ostium_mm"]["mean"] is not None]
    qual = [s["seed_on_daughter_mm"]["mean"] for s in case_scores
            if s["seed_on_daughter_mm"]["mean"] is not None]
    loc_score = float(np.clip(1.0 - np.mean(loc) / 10.0, 0.0, 1.0)) if loc else 0.0
    qual_score = float(np.clip(1.0 - np.mean(qual) / 5.0, 0.0, 1.0)) if qual else 0.0
    return 0.45 * f1 + 0.25 * loc_score + 0.15 * qual_score


def rejection_breakdown(prediction: Dict) -> Dict[str, int]:
    """Which rule dropped candidates. Tells you where recall is being lost."""
    return prediction.get("meta", {}).get("candidates_rejected", {})


def _load(path: Path) -> Dict:
    with open(path) as handle:
        return json.load(handle)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Score Branchseed predictions.")
    parser.add_argument("--predictions", required=True, help="directory of prediction JSON files")
    parser.add_argument("--references", required=True, help="directory of reference JSON files")
    parser.add_argument("--match-radius", type=float, nargs="*", default=list(DEFAULT_MATCH_RADII))
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args(argv)

    pred_dir, ref_dir = Path(args.predictions), Path(args.references)
    report: Dict[str, Dict] = {}
    for radius in args.match_radius:
        per_case = {}
        # Files beginning with "_" are indices and summaries, not cases.
        for ref_path in sorted(ref_dir.glob("[!_]*.json")):
            pred_path = pred_dir / ref_path.name
            reference = _load(ref_path)
            prediction = _load(pred_path) if pred_path.exists() else {"daughters": []}
            per_case[ref_path.stem] = score_case(
                prediction.get("daughters", []), reference.get("daughters", []), radius
            )
            per_case[ref_path.stem]["rejected_by"] = rejection_breakdown(prediction)
        scores = list(per_case.values())
        report[f"match_{radius:g}mm"] = {
            "per_case": per_case,
            "aggregate": {
                "precision": float(np.mean([s["precision"] for s in scores])) if scores else 0.0,
                "recall": float(np.mean([s["recall"] for s in scores])) if scores else 0.0,
                "f1": float(np.mean([s["f1"] for s in scores])) if scores else 0.0,
                "composite": composite_score(scores),
            },
        }

    text = json.dumps(report, indent=2, sort_keys=True)
    if args.json_out:
        Path(args.json_out).write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
