#!/usr/bin/env python3
"""Run the pipeline over a dataset directory and write one JSON per case.

Expects the layout the challenge describes::

    data/
      subject001/
        orig1.nii
        mask1.nii

The image is whichever file matches ``--image-glob``; the mask is whichever
matches ``--mask-glob``. Regenerate the committed predictions with this as the
last step before submission so they cannot drift from the code.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from branchseed.config import load_config          # noqa: E402
from branchseed.output import write_output         # noqa: E402
from branchseed.pipeline import process_case       # noqa: E402

log = logging.getLogger("branchseed.predict_all")


def _find(directory: Path, patterns) -> Optional[Path]:
    for pattern in patterns:
        matches = sorted(directory.glob(pattern))
        if matches:
            return matches[0]
    return None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Batch-run Branchseed over a dataset.")
    parser.add_argument("--data", required=True, help="dataset root, one directory per case")
    parser.add_argument("--out", required=True, help="output directory for predictions")
    parser.add_argument("--viz-dir", default=None, help="also write figures here")
    parser.add_argument("--viz-cases", type=int, default=3,
                        help="number of cases to render figures for (0 = all)")
    parser.add_argument("--config", default=None)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--image-glob", nargs="*", default=["orig*.nii*", "image*.nii*", "*_0000.nii*"])
    parser.add_argument("--mask-glob", nargs="*", default=["mask*.nii*", "*aorta*.nii*", "*label*.nii*"])
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level), stream=sys.stderr,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    cfg = load_config(args.config, overrides=args.set)
    data, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    viz_dir = Path(args.viz_dir) if args.viz_dir else None

    cases = sorted(d for d in data.iterdir() if d.is_dir())
    summary = []
    for n, case_dir in enumerate(cases):
        image = _find(case_dir, args.image_glob)
        mask = _find(case_dir, args.mask_glob)
        if image is None or mask is None:
            log.warning("skipping %s: could not find an image/mask pair", case_dir.name)
            continue

        result = process_case(image, mask, cfg, case_id=case_dir.name)
        write_output(result.payload, out / f"{case_dir.name}.json")

        if viz_dir is not None and (args.viz_cases == 0 or n < args.viz_cases):
            try:
                from branchseed.viz import write_figures
                write_figures(result, viz_dir, cfg)
            except Exception:                                   # noqa: BLE001
                log.exception("figures failed for %s", case_dir.name)

        meta = result.payload["meta"]
        summary.append({
            "case": case_dir.name,
            "daughters": len(result.payload["daughters"]),
            "runtime_s": meta.get("runtime_seconds"),
            "peak_mb": meta.get("peak_memory_mb"),
            "flags": meta.get("case_flags", []),
            "rejected": meta.get("candidates_rejected", {}),
        })
        log.info("%s: %d daughters in %.1f s", case_dir.name,
                 summary[-1]["daughters"], summary[-1]["runtime_s"] or 0.0)

    (out / "_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if summary:
        runtimes = [s["runtime_s"] for s in summary if s["runtime_s"]]
        log.info("%d cases, mean runtime %.1f s, max %.1f s, mean %.1f daughters",
                 len(summary), sum(runtimes) / len(runtimes), max(runtimes),
                 sum(s["daughters"] for s in summary) / len(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
