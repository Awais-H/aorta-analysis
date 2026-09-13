"""CLI entry point.

    python run.py --image image.nii.gz --aorta-mask aorta_mask.nii.gz --output prediction.json

Never crashes on a case (SPEC.md D9 robustness): if a file cannot be loaded or any stage raises,
the traceback goes to stderr, valid JSON with an empty daughters list is written, and the exit
code is 0. A wrong answer on one case costs that case; a crash can cost the run.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
import traceback


def peak_memory_mb() -> float | None:
    """Peak resident memory of this process in MB (D10: the organisers score peak memory on an
    8 GB machine). Windows: PeakWorkingSetSize from psapi; elsewhere ru_maxrss. None if unknown.
    No third-party dependency, and never raises."""
    try:
        if sys.platform == "win32":
            import ctypes
            import ctypes.wintypes as w

            class PMC(ctypes.Structure):
                _fields_ = [("cb", w.DWORD), ("PageFaultCount", w.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t),
                            ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
                            ("PeakPagefileUsage", ctypes.c_size_t)]
            psapi, k32 = ctypes.WinDLL("psapi"), ctypes.WinDLL("kernel32")
            k32.GetCurrentProcess.restype = w.HANDLE
            psapi.GetProcessMemoryInfo.argtypes = [w.HANDLE, ctypes.POINTER(PMC), w.DWORD]
            psapi.GetProcessMemoryInfo.restype = w.BOOL
            pmc = PMC()
            pmc.cb = ctypes.sizeof(PMC)
            if not psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
                return None
            return round(pmc.PeakWorkingSetSize / 2 ** 20, 1)
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # kB on Linux, bytes on macOS
        return round(rss / (2 ** 20 if sys.platform == "darwin" else 1024), 1)
    except Exception:  # noqa: BLE001
        return None


def infer_case_id(image_path: str) -> str:
    parent = os.path.basename(os.path.dirname(os.path.abspath(image_path)))
    if re.match(r"^subject\d+$", parent):
        return parent
    name = os.path.basename(image_path)
    for ext in (".nii.gz", ".nii", ".gz"):
        if name.endswith(ext):
            name = name[: -len(ext)]
    return name


def write_json(path: str, obj: dict) -> None:
    """Atomic write (SPEC.md D9): dump to <path>.tmp, then os.replace onto the final name, so a
    killed process never leaves a half-written file that looks valid."""
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Branchseed: find arteries leaving the supplied aorta.")
    ap.add_argument("--image", required=True)
    ap.add_argument("--aorta-mask", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--case-id", default=None, help="defaults to the subjectNNN folder name or the image stem")
    ap.add_argument("--report-dir", default=None, help="also write the verification PNG and HTML report here")
    ap.add_argument("--meta-output", default=None, help="write timings, threshold and rejection log as JSON")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    case_id = args.case_id or infer_case_id(args.image)
    t0 = time.perf_counter()
    result = {"case_id": case_id, "parent": {"instance_id": "aorta"}, "daughters": []}
    meta = {"case_id": case_id}
    try:
        import pipeline
        result, meta = pipeline.run(args.image, args.aorta_mask, case_id, report_dir=args.report_dir)
    except BaseException as e:  # noqa: BLE001 - deliberately catches everything, including MemoryError
        print(f"ERROR on {case_id}: {type(e).__name__}: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        meta["error"] = traceback.format_exc()
        result = {"case_id": case_id, "parent": {"instance_id": "aorta"}, "daughters": []}

    try:
        write_json(args.output, result)
    except Exception:
        traceback.print_exc(file=sys.stderr)
        try:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(json.dumps({"case_id": case_id, "parent": {"instance_id": "aorta"}, "daughters": []}))
        except Exception:
            traceback.print_exc(file=sys.stderr)

    meta["peak_memory_mb"] = peak_memory_mb()
    if args.meta_output:
        try:
            write_json(args.meta_output, meta)
        except Exception:
            traceback.print_exc(file=sys.stderr)

    elapsed = time.perf_counter() - t0
    mem = f", peak {meta['peak_memory_mb']:.0f} MB" if meta.get("peak_memory_mb") else ""
    print(f"{case_id}: {len(result['daughters'])} daughters, {elapsed:.1f} s{mem}, wrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
