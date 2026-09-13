"""End-to-end orchestration.

The pipeline must never traceback on the hidden set. A crashed case scores zero
on everything; a case that emits an empty daughters list scores zero on
discovery but keeps reproducibility intact and does not abort a batch run. So
every required stage is wrapped, optional stages are best-effort, and a
per-candidate failure rejects that candidate and continues with the rest.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import SimpleITK as sitk

from . import io as bio
from .calibrate import calibrate, euclidean_distance_mm
from .config import Config
from .detect import detect_candidates
from .geometry import GeometryError, build_geometry
from .interp import prefilter_volume
from .measure import measure_all
from .output import build_meta, build_output
from .resolve import rejection_counts, resolve
from .roi import build_roi
from .timing import Timer, peak_memory_mb
from .trace import trace_all
from .wallmap import build_wall_map
from .types import AortaGeometry, Calibration, Candidate, CaseGrid, Flags, WallMap

log = logging.getLogger(__name__)


@dataclass
class CaseResult:
    """Everything a caller might want, including the intermediates the
    visualiser needs. ``payload`` is the JSON-ready dictionary."""

    payload: dict
    grid: Optional[CaseGrid] = None
    geometry: Optional[AortaGeometry] = None
    wall_map: Optional[WallMap] = None
    calibration: Optional[Calibration] = None
    candidates: List[Candidate] = None            # type: ignore[assignment]
    accepted: List[Candidate] = None              # type: ignore[assignment]
    image: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        if self.candidates is None:
            self.candidates = []
        if self.accepted is None:
            self.accepted = []


def process_case(
    image_path: Path | str,
    mask_path: Path | str,
    cfg: Config,
    case_id: Optional[str] = None,
) -> CaseResult:
    """Run the whole pipeline on one case. Never raises."""
    timer = Timer()
    flags = Flags()
    case_id = case_id or _infer_case_id(image_path)

    try:
        with timer.stage("load"):
            image_sitk, mask_full, load_flags = bio.load_case(image_path, mask_path)
        for flag in load_flags:
            flags.add(flag)
        return _run(image_sitk, mask_full, cfg, case_id, flags, timer)
    except Exception as exc:                                   # noqa: BLE001
        log.exception("case %s failed", case_id)
        flags.add("pipeline_error")
        meta = build_meta(cfg, flags.as_list(), timer.as_dict(), timer.total_seconds,
                          peak_memory_mb(), 0, {}, error=f"{type(exc).__name__}: {exc}")
        return CaseResult(payload=build_output(case_id, [], None, cfg, meta))


def process_images(
    image_sitk: sitk.Image,
    mask_sitk: sitk.Image,
    cfg: Config,
    case_id: str = "case",
) -> CaseResult:
    """As :func:`process_case` but for images already in memory."""
    timer = Timer()
    flags = Flags()
    try:
        with timer.stage("load"):
            image_sitk, mask_full, load_flags = bio.load_case_from_images(image_sitk, mask_sitk)
        for flag in load_flags:
            flags.add(flag)
        return _run(image_sitk, mask_full, cfg, case_id, flags, timer)
    except Exception as exc:                                   # noqa: BLE001
        log.exception("case %s failed", case_id)
        flags.add("pipeline_error")
        meta = build_meta(cfg, flags.as_list(), timer.as_dict(), timer.total_seconds,
                          peak_memory_mb(), 0, {}, error=f"{type(exc).__name__}: {exc}")
        return CaseResult(payload=build_output(case_id, [], None, cfg, meta))


def _empty_result(case_id, cfg, flags, timer, calibration=None, grid=None, reason="") -> CaseResult:
    if reason:
        log.warning("case %s: %s; emitting an empty daughters list", case_id, reason)
    meta = build_meta(cfg, flags.as_list(), timer.as_dict(), timer.total_seconds,
                      peak_memory_mb(), 0, {}, calibration=calibration)
    return CaseResult(payload=build_output(case_id, [], grid, cfg, meta), grid=grid,
                      calibration=calibration)


def _run(image_sitk, mask_full, cfg, case_id, flags, timer) -> CaseResult:
    if not mask_full.any():
        flags.add("empty_mask")
        return _empty_result(case_id, cfg, flags, timer, reason="the aorta mask is empty")

    with timer.stage("roi"):
        image, mask, grid = build_roi(image_sitk, mask_full, cfg, flags)
        del mask_full

    with timer.stage("calibrate"):
        edt_mm = euclidean_distance_mm(mask, grid)
        calibration = calibrate(image, mask, edt_mm, grid, cfg, flags)

    try:
        with timer.stage("geometry"):
            geometry = build_geometry(mask, edt_mm, grid, cfg, flags)
    except GeometryError as exc:
        flags.add("no_centreline")
        return _empty_result(case_id, cfg, flags, timer, calibration, grid, str(exc))

    with timer.stage("prefilter"):
        # One cubic prefilter of the ROI, shared by the wall map, the trace,
        # the seed cross-sections and the contact probes.
        image_spline = prefilter_volume(image, order=3)

    with timer.stage("wall_map"):
        wall_map = build_wall_map(image, geometry, calibration, grid, cfg, image_spline)

    with timer.stage("detect"):
        candidates = detect_candidates(wall_map, geometry, calibration, cfg)

    with timer.stage("trace"):
        trace_all(candidates, image_spline, geometry, calibration, grid, cfg)

    with timer.stage("measure"):
        measure_all(candidates, image_spline, geometry, calibration, wall_map.theta, grid, cfg)

    with timer.stage("resolve"):
        accepted = resolve(candidates, image_spline, geometry, calibration, grid, cfg)

    meta = build_meta(
        cfg, flags.as_list(), timer.as_dict(), timer.total_seconds, peak_memory_mb(),
        len(candidates), rejection_counts(candidates), calibration=calibration,
    )
    payload = build_output(case_id, accepted, grid, cfg, meta)
    return CaseResult(
        payload=payload, grid=grid, geometry=geometry, wall_map=wall_map,
        calibration=calibration, candidates=candidates, accepted=accepted, image=image,
    )


def _infer_case_id(image_path: Path | str) -> str:
    path = Path(image_path)
    parent = path.parent.name
    return parent if parent and parent not in (".", "/") else path.stem.split(".")[0]
