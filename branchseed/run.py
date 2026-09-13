"""Command-line entry point for the complete Branchseed CPU pipeline."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
import sys

from . import __version__
from .config import PipelineConfig
from .output import write_challenge_json
from .pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="branchseed")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--image", required=True, help="input 3-D image")
    parser.add_argument("--aorta-mask", required=True, help="binary aortic lumen mask")
    parser.add_argument("--output", required=True, help="challenge JSON destination")
    parser.add_argument("--config", help="optional JSON pipeline configuration")
    parser.add_argument("--visual", help="optional orthogonal overlay PNG")
    parser.add_argument("--debug-dir", help="optional feature/profile output directory")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        result = run_pipeline(
            arguments.image,
            arguments.aorta_mask,
            config=PipelineConfig.load(arguments.config),
            visual_path=arguments.visual,
            debug_dir=arguments.debug_dir,
        )
        write_challenge_json(result.prediction, arguments.output)
        print(json.dumps({
            "output": arguments.output,
            "daughters": len(result.prediction.daughters),
            "candidates": result.candidate_count,
            "profiles": {
                name: {"seconds": value.seconds, "rss_mb": value.rss_mb}
                for name, value in result.profiles.items()
            },
            "warnings": list(result.warnings),
        }, sort_keys=True))
        return 0
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"branchseed: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
