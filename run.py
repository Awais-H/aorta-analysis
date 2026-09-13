#!/usr/bin/env python3
"""Branchseed command-line entry point.

    python run.py --image image.nii.gz --aorta-mask aorta_mask.nii.gz \
        --output prediction.json

Thin by design: parse arguments, load the config, call one pipeline function,
write JSON. All logic lives in the package so the test suite can drive any
stage on its own.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")            # before any pyplot import: no GUI, no server


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run.py", description="Detect direct aortic branch origins in a CTA volume."
    )
    parser.add_argument("--image", required=True, help="CT volume (.nii or .nii.gz)")
    parser.add_argument("--aorta-mask", required=True, help="binary parent-aorta mask")
    parser.add_argument("--output", required=True, help="output JSON path")
    parser.add_argument("--config", default=None, help="override config.yaml")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="override one config value, e.g. wallmap.probe_outer_mm=7")
    parser.add_argument("--case-id", default=None, help="case identifier for the output")
    parser.add_argument("--viz-dir", default=None,
                        help="optional directory for the wall map and verification figures")
    parser.add_argument("--voxel-output", default=None,
                        help="optional path for a companion output in original-image voxel "
                             "units (continuous indices), alongside the mandated mm output")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    # stderr only: the CLI's stdout stays clean.
    logging.basicConfig(
        level=getattr(logging, args.log_level), stream=sys.stderr,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    log = logging.getLogger("branchseed.run")

    from branchseed.config import load_config
    from branchseed.output import build_voxel_output, write_output
    from branchseed.pipeline import process_case

    cfg = load_config(args.config, overrides=args.set)
    viz_dir = Path(args.viz_dir) if args.viz_dir else None
    if viz_dir is not None:
        viz_dir.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(viz_dir / "branchseed.log", mode="w")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
        logging.getLogger().addHandler(handler)

    result = process_case(args.image, args.aorta_mask, cfg, case_id=args.case_id)
    write_output(result.payload, args.output)

    if args.voxel_output:
        try:
            voxel_payload = build_voxel_output(
                result.payload.get("case_id", "case"), result.accepted, result.grid, cfg
            )
            write_output(voxel_payload, args.voxel_output)
        except Exception:                                    # noqa: BLE001
            # Optional output: never let it take down the mandated result.
            log.exception("voxel-unit output failed; the mm output is unaffected")

    if viz_dir is not None:
        try:
            from branchseed.viz import write_figures
            write_figures(result, viz_dir, cfg)
        except Exception:                                    # noqa: BLE001
            # Figures are optional: never let them take down a valid result.
            log.exception("figure generation failed; the JSON output is unaffected")

    meta = result.payload.get("meta", {})
    log.info("case %s: %d daughters in %.1f s (peak %.0f MB)",
             result.payload.get("case_id"), len(result.payload.get("daughters", [])),
             meta.get("runtime_seconds", 0.0), meta.get("peak_memory_mb", 0.0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
