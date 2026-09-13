"""D8 clinician display and the mandatory verification PNG.

STUB. Current behaviour: the verification PNG (coronal and sagittal maximum-intensity projections
of the candidate voxels with the mask overlaid, ostia as dots, direction arrows), ported from the
triage.py render (lines 169 to 192), plus a minimal self-contained HTML page with the branch table
in clinician terms (clock, height, radius). The unrolled clock map with inter-branch spacing and
the Plotly 3D view per SPEC.md D8 are added here. Generation sits behind a flag in run.py so the
scored run does not pay for it.
"""
from __future__ import annotations

import html
import logging
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config
import io_utils
from candidates import Candidates
from frame import Frame

log = logging.getLogger("branchseed.report")

ARROW_LENGTH_MM = config.TRACE_MAX_MM  # arrows drawn as long as the traced path


def _projection_axes(cand: Candidates):
    """Coronal (rows z, cols x) and sagittal (rows z, cols y) MIPs of the candidate voxels."""
    cor = cand.candidates.max(axis=1).astype(float)
    sag = cand.candidates.max(axis=2).astype(float)
    mcor = cand.mask.max(axis=1).astype(float)
    msag = cand.mask.max(axis=2).astype(float)
    return cor, sag, mcor, msag


def write_verification_png(cand: Candidates, result: dict, path: str) -> str:
    cor, sag, mcor, msag = _projection_axes(cand)
    fig, axes = plt.subplots(1, 2, figsize=(12, 7))
    for ax, img, mk, title, col in ((axes[0], cor, mcor, "coronal: candidates + mask, ostia, directions", 2),
                                    (axes[1], sag, msag, "sagittal", 1)):
        ax.imshow(img, cmap="gray", origin="lower", aspect=cand.spacing[0] / cand.spacing[col])
        ax.imshow(np.ma.masked_where(mk == 0, mk), cmap="spring", alpha=0.35, origin="lower",
                  aspect=cand.spacing[0] / cand.spacing[col])
        for d in result.get("daughters", []):
            o = io_utils.mm_to_index(cand.image, d["ostium_xyz_mm"])
            tip = io_utils.mm_to_index(cand.image, np.asarray(d["ostium_xyz_mm"]) + ARROW_LENGTH_MM * np.asarray(d["direction_xyz"]))
            ax.plot(o[col], o[0], "o", color="cyan", ms=6)
            ax.annotate("", xy=(tip[col], tip[0]), xytext=(o[col], o[0]),
                        arrowprops=dict(arrowstyle="->", color="red", lw=1.5))
            ax.text(o[col], o[0], " " + d["instance_id"][-3:], color="yellow", fontsize=8)
        ax.set_title(title)
    fig.suptitle(f"{result.get('case_id', '')}: {len(result.get('daughters', []))} daughters")
    plt.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def branch_rows(result: dict, frame: Frame | None) -> list:
    rows = []
    for d in result.get("daughters", []):
        h, c = frame.height_clock(d["ostium_xyz_mm"]) if frame is not None else (float("nan"), float("nan"))
        rows.append({"instance_id": d["instance_id"], "clock": c, "height_mm": h, "radius_mm": d["radius_mm"],
                     "direction_xyz": d["direction_xyz"], "ostium_xyz_mm": d["ostium_xyz_mm"]})
    rows.sort(key=lambda r: r["height_mm"])
    return rows


def write_html(result: dict, frame: Frame | None, path: str, png_name: str | None = None) -> str:
    rows = branch_rows(result, frame)
    length = f"{frame.length_mm:.0f} mm" if frame is not None else "n/a"
    parts = [f"<h1>{html.escape(result.get('case_id', ''))}</h1>",
             f"<p>Supplied aortic segment: {length}. Daughters: {len(rows)}. "
             "Clock: 12 anterior, 3 patient's left, 6 posterior, 9 patient's right. Height from the superior end.</p>",
             "<table border='1' cellpadding='4'><tr><th>ID</th><th>Clock</th><th>Height (mm)</th><th>Radius (mm)</th>"
             "<th>Direction xyz</th><th>Ostium xyz (mm)</th></tr>"]
    for r in rows:
        parts.append(f"<tr><td>{html.escape(r['instance_id'])}</td><td>{r['clock']:.1f}</td><td>{r['height_mm']:.1f}</td>"
                     f"<td>{r['radius_mm']:.1f}</td><td>{[round(v, 2) for v in r['direction_xyz']]}</td>"
                     f"<td>{[round(v, 1) for v in r['ostium_xyz_mm']]}</td></tr>")
    parts.append("</table>")
    if png_name:
        parts.append(f"<p><img src='{html.escape(png_name)}' style='max-width:100%'></p>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("<!doctype html><html><head><meta charset='utf-8'><title>"
                f"{html.escape(result.get('case_id', ''))}</title></head><body>" + "\n".join(parts) + "</body></html>")
    return path


def write_all(cand: Candidates, result: dict, frame: Frame | None, out_dir: str, case_id: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    png = os.path.join(out_dir, f"{case_id}_check.png")
    htm = os.path.join(out_dir, f"{case_id}_report.html")
    write_verification_png(cand, result, png)
    write_html(result, frame, htm, png_name=os.path.basename(png))
    log.info("wrote %s and %s", png, htm)
    return {"png": png, "html": htm}
