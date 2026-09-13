"""Stage 8 - serialisation.

Every emitted point traverses exactly one chain, in one function: subvolume
index -> ROI continuous index -> physical millimetres. The mandated output
(:func:`build_output`) never emits voxel indices.

:func:`build_voxel_output` produces a companion, non-mandated payload with
coordinates in the original image's voxel units instead, for callers who want
to overlay results on the source NIfTI directly. It is written to a separate
file via ``--voxel-output`` and never substitutes for the mm output.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import __version__
from .config import Config
from .types import Candidate, CaseGrid

log = logging.getLogger(__name__)

MANDATED_FIELDS = (
    "instance_id", "parent_instance_id", "ostium_xyz_mm", "seed_xyz_mm",
    "radius_mm", "direction_xyz",
)


def _round(value, decimals: int):
    if isinstance(value, (list, tuple, np.ndarray)):
        return [round(float(v), decimals) for v in value]
    return round(float(value), decimals)


def daughter_record(candidate: Candidate, grid: CaseGrid, cfg: Config) -> Optional[dict]:
    """One daughter instance, or None if any coordinate failed to validate."""
    decimals = cfg.output.round_decimals
    ostium = grid.to_physical(candidate.ostium_xyz_roi)
    seed = grid.to_physical(candidate.seed_xyz_roi)
    direction = np.asarray(candidate.direction, dtype=np.float64)
    norm = np.linalg.norm(direction)
    if norm < 1e-9:
        return None
    direction = direction / norm

    for array in (ostium, seed, direction):
        if not np.all(np.isfinite(array)):
            log.warning("candidate %d produced a non-finite coordinate; dropping it",
                        candidate.id)
            return None
    if candidate.radius_mm is None or not np.isfinite(candidate.radius_mm):
        return None

    return {
        "instance_id": candidate.instance_id,
        "parent_instance_id": "aorta",
        "ostium_xyz_mm": _round(ostium, decimals),
        "seed_xyz_mm": _round(seed, decimals),
        "radius_mm": _round(candidate.radius_mm, decimals),
        "direction_xyz": _round(direction, cfg.output.direction_decimals),
        # Additive fields. Extra keys do not break a schema check and they are
        # what the clinician display and the tuning harness run on.
        "confidence": _round(candidate.confidence, decimals),
        "path_length_mm": _round(candidate.path_length_mm, decimals),
        "followed_mm": _round(candidate.reached_mm, decimals),
        "clock_position": candidate.clock_position,
        "arc_length_mm": _round(candidate.arc_length_mm, decimals),
        "parent_radius_mm": _round(candidate.parent_radius_mm, decimals),
        "stop_reason": candidate.stop_reason,
        "flags": candidate.flags.as_list(),
    }


def build_output(
    case_id: str,
    accepted: Sequence[Candidate],
    grid: Optional[CaseGrid],
    cfg: Config,
    meta: Dict,
) -> dict:
    daughters: List[dict] = []
    if grid is not None:
        for candidate in accepted:
            record = daughter_record(candidate, grid, cfg)
            if record is not None:
                daughters.append(record)
    # IDs must be unique and contiguous even if a record was dropped above.
    for n, record in enumerate(daughters, start=1):
        record["instance_id"] = f"branch_{n:03d}"

    return {
        "case_id": case_id,
        "parent": {"instance_id": "aorta"},
        "daughters": daughters,
        "meta": meta,
    }


def daughter_voxel_record(candidate: Candidate, grid: CaseGrid, cfg: Config) -> Optional[dict]:
    """The same daughter, with coordinates in the *original* image's voxel
    units instead of physical millimetres.

    This is not the mandated output - the brief requires physical millimetres
    via ``TransformIndexToPhysicalPoint`` - but a voxel-space file is handy for
    overlaying on the original NIfTI in a plain array viewer. Coordinates are
    continuous indices (equivalent to
    ``sitk_ref.TransformPhysicalPointToContinuousIndex``), not rounded to
    integer voxels, so the mapping back to the mm output is exact.
    """
    decimals = cfg.output.round_decimals
    ostium_ijk = grid.physical_to_original_index(grid.to_physical(candidate.ostium_xyz_roi))
    seed_ijk = grid.physical_to_original_index(grid.to_physical(candidate.seed_xyz_roi))
    direction = np.asarray(candidate.direction, dtype=np.float64)
    norm = np.linalg.norm(direction)
    if norm < 1e-9:
        return None
    direction = direction / norm

    for array in (ostium_ijk, seed_ijk, direction):
        if not np.all(np.isfinite(array)):
            return None
    if candidate.radius_mm is None or not np.isfinite(candidate.radius_mm):
        return None

    # A direction has no single length in voxel space when spacing is
    # anisotropic, so this is offered purely for drawing an arrow in an image
    # viewer that works in voxel coordinates, renormalised to unit length in
    # that (distorted) space rather than claiming physical equivalence.
    direction_ijk = grid.physical_direction_to_original_index(direction)
    direction_ijk = direction_ijk / max(np.linalg.norm(direction_ijk), 1e-9)

    mean_spacing = float(np.mean(grid.orig_spacing_mm))

    return {
        "instance_id": candidate.instance_id,
        "parent_instance_id": "aorta",
        "ostium_ijk_voxel": _round(ostium_ijk, decimals),
        "seed_ijk_voxel": _round(seed_ijk, decimals),
        "radius_voxels": _round(candidate.radius_mm / max(mean_spacing, 1e-9), decimals),
        "direction_ijk": _round(direction_ijk, cfg.output.direction_decimals),
        # Kept for cross-reference against the mandated mm output.
        "radius_mm": _round(candidate.radius_mm, decimals),
    }


def build_voxel_output(
    case_id: str,
    accepted: Sequence[Candidate],
    grid: Optional[CaseGrid],
    cfg: Config,
) -> dict:
    """A companion output with coordinates in the original image's voxel
    units. Not the mandated schema - see :func:`build_output` for that."""
    daughters: List[dict] = []
    if grid is not None:
        for candidate in accepted:
            record = daughter_voxel_record(candidate, grid, cfg)
            if record is not None:
                daughters.append(record)
    for n, record in enumerate(daughters, start=1):
        record["instance_id"] = f"branch_{n:03d}"

    return {
        "case_id": case_id,
        "parent": {"instance_id": "aorta"},
        "units": "voxel indices (continuous) in the original --image grid",
        "daughters": daughters,
    }


def build_meta(
    cfg: Config,
    flags: Sequence[str],
    timings: Dict[str, float],
    runtime_s: float,
    peak_memory_mb: float,
    candidate_count: int,
    rejections: Dict[str, int],
    calibration=None,
    error: Optional[str] = None,
) -> dict:
    meta = {
        "software": "branchseed",
        "version": __version__,
        "config_hash": cfg.hash(),
        "runtime_seconds": round(runtime_s, 3),
        "peak_memory_mb": round(peak_memory_mb, 1),
        "stage_seconds": timings,
        "case_flags": list(flags),
        "candidates_generated": candidate_count,
        "candidates_rejected": rejections,
    }
    if calibration is not None:
        meta["calibration"] = {
            "mu_ao": round(calibration.mu_ao, 1),
            "sigma_ao": round(calibration.sigma_ao, 1),
            "t_lumen": round(calibration.t_lumen, 1),
            "t_calcium": round(calibration.t_calcium, 1),
            "t_bg": round(calibration.t_bg, 1),
            "degenerate": bool(calibration.degenerate),
        }
    if error is not None:
        meta["pipeline_error"] = error
    return meta


def write_output(payload: dict, path: Path | str) -> None:
    path = Path(path)
    if path.parent and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    # sort_keys makes the bytes reproducible for the determinism test.
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    log.info("wrote %d daughters to %s", len(payload.get("daughters", [])), path)
