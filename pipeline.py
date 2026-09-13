"""Orchestrates the stages with per-stage wall time (D10: log per-stage time on every run).

run.py calls run(); nothing here catches exceptions, run.py does.
"""
from __future__ import annotations

import logging
import time
from contextlib import contextmanager

import numpy as np

import candidates
import config
import filters
import frame as frame_mod
import instances
import io_utils
import ostium
import report
import tracing

log = logging.getLogger("branchseed.pipeline")


@contextmanager
def _timed(meta: dict, stage: str):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        meta["timings_s"][stage] = round(time.perf_counter() - t0, 3)
        log.info("stage %-10s %6.2f s", stage, meta["timings_s"][stage])


def empty_result(case_id: str) -> dict:
    return {"case_id": case_id, "parent": {"instance_id": "aorta"}, "daughters": []}


def format_daughter(k: int, ost, tr) -> dict:
    r = config.OUTPUT_MM_DECIMALS
    return {
        "instance_id": f"branch_{k:03d}",
        "parent_instance_id": "aorta",
        "ostium_xyz_mm": [round(float(v), r) for v in ost.mm],
        "seed_xyz_mm": [round(float(v), r) for v in tr.seed_mm],
        "radius_mm": round(float(tr.radius_mm), r),
        "direction_xyz": [round(float(v), config.OUTPUT_DIRECTION_DECIMALS) for v in tr.direction_xyz],
    }


def assemble(case_id: str, kept: list, ostia: dict, traces: dict) -> dict:
    result = empty_result(case_id)
    k = 0
    for label in kept:
        tr = traces[label]
        if tr.seed_mm is None:
            continue
        k += 1
        result["daughters"].append(format_daughter(k, ostia[label], tr))
    return result


def run(image_path: str, mask_path: str, case_id: str, report_dir: str | None = None) -> tuple[dict, dict]:
    """Returns (result JSON dict, meta dict with timings, threshold, fragment log, rejections)."""
    meta = {"case_id": case_id, "timings_s": {}}
    t_all = time.perf_counter()
    with _timed(meta, "load"):
        image, mask_image, load_info = io_utils.load_case(image_path, mask_path)
    meta["load"] = load_info
    with _timed(meta, "candidates"):
        cand = candidates.build(image, mask_image)
    del image, mask_image  # D10: drop full-resolution data once cropped
    meta.update({"threshold_hu": cand.threshold_hu, "hu_stats": cand.hu_stats,
                 "fragment_log": cand.fragment_log, "grid": cand.grid_info})
    with _timed(meta, "instances"):
        inst = instances.build(cand)
    meta["wall_patches"] = inst.n
    meta["wall_area_mm2"] = {str(k): round(v, 1) for k, v in inst.wall_area_mm2.items()}
    with _timed(meta, "frame"):
        fr = frame_mod.build(cand)
    meta["segment_length_mm"] = round(fr.length_mm, 1)
    meta["frame"] = {"method": fr.method, "tortuosity": round(fr.tortuosity, 3), "twist_deg": round(fr.twist_deg, 1),
                     "endpoints_mm": [[round(float(v), 1) for v in e] for e in fr.endpoints_mm], **fr.info}
    with _timed(meta, "ostium"):
        ostia = ostium.locate(cand, inst)
    with _timed(meta, "tracing"):
        traces = tracing.trace_all(cand, inst, ostia)
    meta["ostium_methods"] = {m: sum(o.method == m for o in ostia.values()) for m in ("axis_intersection", "inscribed_circle", "strip_end")}
    meta["trace_methods"] = {m: sum(t.method == m for t in traces.values()) for m in ("march", "axis", "none")}
    meta["trace_stop_reasons"] = {r: sum(t.stop_reason == r for t in traces.values())
                                 for r in sorted({t.stop_reason for t in traces.values()})}
    meta["bifurcations"] = sum(t.bifurcation for t in traces.values())
    meta["radius_flags"] = {f: sum(t.radius_flag == f for t in traces.values())
                           for f in sorted({t.radius_flag for t in traces.values() if t.radius_flag})}
    meta["per_branch"] = {str(l): {"ostium_method": ostia[l].method, "trace_method": t.method, "stop": t.stop_reason,
                                   "path_mm": t.path_length_mm, "march_mm": t.march_length_mm, "bifurcation": t.bifurcation,
                                   "radius_mm": round(t.radius_mm, 2), "radius_flag": t.radius_flag, "fills_window": t.section_fills_window, "cortex": round(t.section_cortex_fraction, 3),
                                   "pca_chord_deg": None if t.pca_chord_angle_deg is None else round(t.pca_chord_angle_deg, 1)}
                          for l, t in traces.items()}
    with _timed(meta, "filters"):
        fres = filters.apply(cand, inst, ostia, traces, fr)
    meta["rejections"] = [(int(l), r, round(float(v), 3)) for l, r, v in fres.rejections]
    meta["rejection_counts"] = {r: sum(1 for _, rr, _ in fres.rejections if rr == r) for r in sorted({r for _, r, _ in fres.rejections})}
    meta["flags"] = {str(l): f for l, f in fres.flags.items() if f}
    meta["measurements"] = {str(l): m for l, m in fres.measurements.items()}
    meta["label_to_branch"] = {}
    result = assemble(case_id, fres.kept, ostia, traces)
    for d, label in zip(result["daughters"], [l for l in fres.kept if traces[l].seed_mm is not None]):
        meta["label_to_branch"][str(label)] = d["instance_id"]
    if report_dir:
        # the report is display only: a failure here is logged and must never cost the JSON
        # (run.py replaces the result with an empty daughters list on any exception)
        with _timed(meta, "report"):
            try:
                meta["report_files"] = report.write_all(cand, result, fr, report_dir, case_id, meta=meta)
            except Exception as e:  # noqa: BLE001
                log.exception("report generation failed; the prediction is unaffected")
                meta["report_error"] = f"{type(e).__name__}: {e}"
    meta["timings_s"]["total"] = round(time.perf_counter() - t_all, 3)
    log.info("%s: %d daughters in %.1f s", case_id, len(result["daughters"]), meta["timings_s"]["total"])
    return result, meta
