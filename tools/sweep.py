#!/usr/bin/env python3
"""One-parameter-at-a-time tuning, following Wala's methodology.

Sweep one parameter over a range, re-run the pipeline on a development subset,
and maximise a single score. Print the whole curve rather than only the
argmax: Wala found that both too-small and too-large values of their cylinder
height degraded performance sharply, and the same shape is expected from the
probe range. Choose a value in the middle of a flat region, not on a narrow
peak.

Suggested order, by sensitivity:

  1. wallmap.probe_outer_mm   4 to 8      the highest-sensitivity parameter
  2. wallmap.probe_inner_mm   0.5 to 2
  3. detect.threshold_sigma_offset  -1 to 1
  4. trace.leak_ratio         1.3 to 2.0
  5. geometry.endcap_normal_deg  25 to 45

``resolve.bump_boundary`` is fitted in its feature space rather than swept -
plot path length against length/parent radius for matched and unmatched
candidates and place the line by hand or with a linear discriminant.

Hold two or three development cases out of fitting entirely. With ~25 cases in
total and a small annotated subset, overfitting is a real risk; prefer fewer,
coarser parameters to many fine ones.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from branchseed.config import Config, load_config     # noqa: E402
from branchseed.pipeline import process_case          # noqa: E402
from tools.evaluate import composite_score, score_case  # noqa: E402

log = logging.getLogger("branchseed.sweep")


def _cases(data: Path, references: Path, holdout: Sequence[str]):
    found = []
    for ref_path in sorted(references.glob("*.json")):
        name = ref_path.stem
        if name in holdout:
            continue
        case_dir = data / name
        if not case_dir.is_dir():
            log.warning("no data directory for reference %s", name)
            continue
        image = next(iter(sorted(case_dir.glob("orig*.nii*")) + sorted(case_dir.glob("image*.nii*"))), None)
        mask = next(iter(sorted(case_dir.glob("mask*.nii*"))), None)
        if image and mask:
            found.append((name, image, mask, json.loads(ref_path.read_text())))
    return found


def evaluate_config(cfg: Config, cases, match_radius: float) -> Dict:
    scores = []
    for name, image, mask, reference in cases:
        payload = process_case(image, mask, cfg, case_id=name).payload
        scores.append(score_case(payload["daughters"], reference.get("daughters", []),
                                 match_radius))
    return {
        "f1": float(np.mean([s["f1"] for s in scores])) if scores else 0.0,
        "precision": float(np.mean([s["precision"] for s in scores])) if scores else 0.0,
        "recall": float(np.mean([s["recall"] for s in scores])) if scores else 0.0,
        "composite": composite_score(scores),
        "ostium_mm": float(np.mean([s["ostium_mm"]["mean"] for s in scores
                                    if s["ostium_mm"]["mean"] is not None])) if scores else None,
    }


def parse_values(raw: Sequence[str]) -> List:
    import yaml
    return [yaml.safe_load(v) for v in raw]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Sweep one Branchseed parameter.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--references", required=True)
    parser.add_argument("--param", required=True, help="dotted path, e.g. wallmap.probe_outer_mm")
    parser.add_argument("--values", nargs="+", required=True)
    parser.add_argument("--objective", default="composite", choices=["f1", "composite"])
    parser.add_argument("--match-radius", type=float, default=5.0)
    parser.add_argument("--holdout", nargs="*", default=[])
    parser.add_argument("--config", default=None)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, stream=sys.stderr,
                        format="%(levelname)-7s %(name)s: %(message)s")
    base = load_config(args.config)
    cases = _cases(Path(args.data), Path(args.references), args.holdout)
    if not cases:
        print("no cases found; check --data and --references", file=sys.stderr)
        return 1

    rows = []
    for value in parse_values(args.values):
        result = evaluate_config(base.override(args.param, value), cases, args.match_radius)
        result["value"] = value
        rows.append(result)
        print("%-28s = %-8s  F1 %.3f  P %.3f  R %.3f  ostium %5s mm  composite %.3f"
              % (args.param, value, result["f1"], result["precision"], result["recall"],
                 "n/a" if result["ostium_mm"] is None else "%.2f" % result["ostium_mm"],
                 result["composite"]))

    best = max(rows, key=lambda r: r[args.objective])
    print("\n%d cases, %d held out. Best %s = %s at %s = %s"
          % (len(cases), len(args.holdout), args.objective, f"{best[args.objective]:.3f}",
             args.param, best["value"]))
    print("Prefer a value in the middle of a flat region over a narrow peak.")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(
            {"param": args.param, "match_radius_mm": args.match_radius,
             "holdout": args.holdout, "rows": rows}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
