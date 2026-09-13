"""D8 clinician display and the mandatory verification PNG.

Per case, behind run.py's --report-dir flag so the scored run never pays for it:

1. Verification PNG (the PDF's required visual check): coronal and sagittal projections of the
   candidate voxels with the aorta mask overlaid, the centreline, every reported ostium as a dot,
   the direction arrow drawn as long as the traced path, and the branch ID.
2. Clock map PNG: the aortic wall unrolled. Horizontal axis is the clock position (6 posterior
   at both edges, 9 patient's right, 12 anterior in the middle, 3 patient's left: the radiological
   view, patient's right on the viewer's left), vertical axis is height from the superior end of
   the supplied segment. Each branch is a marker sized by its radius, green when it passed every
   rule cleanly and amber when it carries a flag, labelled with its ID and radius. Brackets on
   the right give the inter-branch spacing along the centreline (the room to land a stent), and
   a panel on the left gives the local aortic diameter along the same height axis.
3. HTML report, one self-contained file: summary, branch table in clinician terms (clock, height,
   radius, direction, origin diameter, flags), the two PNGs embedded, an interactive Plotly 3D
   view (aorta mesh, centreline, ostia sized by radius, direction arrows; Plotly's JavaScript is
   inlined so it opens offline), and the JSON as written.

No anatomical names anywhere (CLAUDE.md). Height and clock come from frame.py.
"""
from __future__ import annotations

import base64
import html
import json
import logging
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config
import frame as frame_mod
import io_utils
from candidates import Candidates
from frame import Frame

log = logging.getLogger("branchseed.report")

ARROW_LENGTH_MM = config.TRACE_MAX_MM  # arrows drawn as long as the traced path

FLAG_GLOSSARY = {
    "near_cut_face": "wall patch within 3 mm of a cut end of the supplied mask (organiser question 1)",
    "large_patch_at_cut": "patch at a cut end over 40% of the aortic cross-section, direction not along the aorta",
    "borderline_diameter": "origin diameter 1.5 to 2.5 mm: at 1.5 mm voxels the 2 mm eligibility limit is uncertain",
    "no_departure": "traced path still within 3 mm of the aorta after 10 mm: a wall-hugging vessel",
    "tangential": "direction more than 70 deg from the wall normal: leaves at a shallow angle",
    "elongated_patch": "contact patch longer than 3:1: a vessel running along the wall",
    "section_merged": "cross-section at the seed touches the 16 mm window edge: merges with a neighbour at this resolution",
    "area_growth": "cross-section more than doubles between 3 and 5 mm out",
    "axis_fallback": "the march failed; seed and direction from the straight axis fit",
    "bifurcation": "the traced path met a division before 10 mm; direction is the trunk's",
    "radius_inscribed_fallback": "area radius exceeded half the aortic radius; inscribed-circle radius reported",
    "radius_no_cross_section": "no cross-section found at the seed; radius from the inscribed circle",
    "origin_no_cross_section": "no cross-section at 2 mm out; origin diameter not measured",
}


# ------------------------------------------------------------------ data


def _branch_meta(meta: dict | None) -> tuple[dict, dict]:
    """branch_id -> flags, branch_id -> measurements, from the pipeline meta (label keyed)."""
    if not meta:
        return {}, {}
    l2b = meta.get("label_to_branch", {})
    flags = {b: meta.get("flags", {}).get(l, []) for l, b in l2b.items()}
    meas = {b: meta.get("measurements", {}).get(l, {}) for l, b in l2b.items()}
    return flags, meas


def branch_rows(result: dict, frame: Frame | None, meta: dict | None = None) -> list:
    flags, meas = _branch_meta(meta)
    rows = []
    for d in result.get("daughters", []):
        h, c = frame.height_clock(d["ostium_xyz_mm"]) if frame is not None else (float("nan"), float("nan"))
        m = meas.get(d["instance_id"], {})
        rows.append({"instance_id": d["instance_id"], "clock": c, "height_mm": h, "radius_mm": d["radius_mm"],
                     "direction_xyz": d["direction_xyz"], "ostium_xyz_mm": d["ostium_xyz_mm"], "seed_xyz_mm": d["seed_xyz_mm"],
                     "flags": list(flags.get(d["instance_id"], [])),
                     "origin_diameter_mm": m.get("origin_diameter_mm"), "departure_mm": m.get("departure_mm")})
    rows.sort(key=lambda r: (r["height_mm"] if r["height_mm"] == r["height_mm"] else 1e9))
    return rows


def clock_label(c: float) -> str:
    """12.0 -> '12:00', 2.5 -> '2:30' (12 o'clock for values that round to 0)."""
    if c != c:
        return "n/a"
    hours = int(np.floor(c)) % 12
    minutes = int(round((c - np.floor(c)) * 60))
    if minutes == 60:
        hours, minutes = (hours + 1) % 12, 0
    return f"{12 if hours == 0 else hours}:{minutes:02d}"


# ------------------------------------------------------------------ verification PNG


def _projection_axes(cand: Candidates):
    """Coronal (rows z, cols x) and sagittal (rows z, cols y) MIPs of the candidate voxels."""
    cor = cand.bright_shell.max(axis=1).astype(float)
    sag = cand.bright_shell.max(axis=2).astype(float)
    mcor = cand.mask.max(axis=1).astype(float)
    msag = cand.mask.max(axis=2).astype(float)
    return cor, sag, mcor, msag


def write_verification_png(cand: Candidates, result: dict, path: str, frame: Frame | None = None) -> str:
    cor, sag, mcor, msag = _projection_axes(cand)
    fig, axes = plt.subplots(1, 2, figsize=(12, 7))
    n = len(result.get("daughters", []))
    for ax, img, mk, title, col in ((axes[0], cor, mcor, "coronal (viewer's left = patient's right)", 2),
                                    (axes[1], sag, msag, "sagittal (left = anterior)", 1)):
        asp = cand.spacing[0] / cand.spacing[col]
        ax.imshow(img, cmap="gray", origin="lower", aspect=asp)
        ax.imshow(np.ma.masked_where(mk == 0, mk), cmap="spring", alpha=0.35, origin="lower", aspect=asp)
        if frame is not None and len(frame.centreline_idx):
            ax.plot(frame.centreline_idx[:, col], frame.centreline_idx[:, 0], "-", color="deepskyblue", lw=1.0, label="centreline")
            ax.plot(frame.endpoint_idx_zyx[:, col], frame.endpoint_idx_zyx[:, 0], "s", color="deepskyblue", ms=4)
        for d in result.get("daughters", []):
            o = io_utils.mm_to_index(cand.image, d["ostium_xyz_mm"])
            tip = io_utils.mm_to_index(cand.image, np.asarray(d["ostium_xyz_mm"]) + ARROW_LENGTH_MM * np.asarray(d["direction_xyz"]))
            ax.plot(o[col], o[0], "o", color="cyan", ms=6, mec="black")
            ax.annotate("", xy=(tip[col], tip[0]), xytext=(o[col], o[0]),
                        arrowprops=dict(arrowstyle="->", color="red", lw=1.5))
            ax.text(o[col], o[0], " " + d["instance_id"][-3:], color="yellow", fontsize=8)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"{result.get('case_id', '')}: {n} daughter{'s' if n != 1 else ''}. "
                 "Grey: candidate voxels; pink: aorta mask; cyan dots: ostia; red: direction (10 mm); blue: centreline")
    plt.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


# ------------------------------------------------------------------ clock map


def clock_x(c: float) -> float:
    """Map clock hours to the unrolled axis: 6 -> 0, 9 -> 3, 12 -> 6, 3 -> 9, 6 -> 12."""
    return float((c - 6.0) % 12.0)


def write_clock_map_png(result: dict, frame: Frame | None, path: str, meta: dict | None = None) -> str:
    rows = branch_rows(result, frame, meta)
    length = float(frame.length_mm) if frame is not None else 100.0
    fig, (axd, ax) = plt.subplots(1, 2, figsize=(11, max(6.0, min(14.0, 3.0 + length / 25.0))),
                                  gridspec_kw={"width_ratios": [1, 4]}, sharey=True)
    # diameter panel
    if frame is not None and frame.radius_mm is not None and len(frame.radius_mm) > 2:
        diam = 2.0 * np.asarray(frame.radius_mm)
        axd.plot(diam, frame.arc_mm, color="0.3", lw=1.2)
        axd.fill_betweenx(frame.arc_mm, 0, diam, color="0.85")
        axd.set_xlim(0, max(10.0, float(diam.max()) * 1.2))
    axd.set_xlabel("aortic diameter (mm)")
    axd.set_ylabel("height from superior end of the supplied segment (mm)")
    axd.grid(True, alpha=0.3)
    # branches
    heights = {}
    for r in rows:
        x, y = clock_x(r["clock"]), r["height_mm"]
        heights[r["instance_id"]] = y
        colour = "#2e8b57" if not r["flags"] else "#e69f00"
        size = max(30.0, (r["radius_mm"] * 6.0) ** 2)
        ax.scatter([x], [y], s=size, color=colour, edgecolor="black", zorder=3, alpha=0.9)
        label = f"{r['instance_id'][-3:]}  r={r['radius_mm']:.1f}"
        if r["flags"]:
            label += "  *"
        ax.annotate(label, (x, y), xytext=(8, 4), textcoords="offset points", fontsize=8, zorder=4)
        if x < 1.5:  # a marker near the seam appears at both edges
            ax.scatter([x + 12.0], [y], s=size, color=colour, edgecolor="black", zorder=3, alpha=0.35)
        elif x > 10.5:
            ax.scatter([x - 12.0], [y], s=size, color=colour, edgecolor="black", zorder=3, alpha=0.35)
    # spacing brackets
    gaps = frame_mod.spacing_along(heights, length)
    xb = 12.6
    for g in gaps:
        y0 = 0.0 if g["from"] == "superior end" else heights[g["from"]]
        y1 = length if g["to"] == "inferior end" else heights[g["to"]]
        if y1 - y0 < 0.5:
            continue
        ax.plot([xb, xb], [y0, y1], color="0.4", lw=1.0, clip_on=False)
        ax.plot([xb - 0.15, xb + 0.15], [y0, y0], color="0.4", lw=1.0, clip_on=False)
        ax.plot([xb - 0.15, xb + 0.15], [y1, y1], color="0.4", lw=1.0, clip_on=False)
        ax.text(xb + 0.25, (y0 + y1) / 2, f"{g['gap_mm']:.0f} mm", fontsize=8, va="center", color="0.25", clip_on=False)
    ax.set_xlim(0, 12)
    ax.set_ylim(length + 5.0, -5.0)
    ax.set_xticks([0, 3, 6, 9, 12])
    ax.set_xticklabels(["6\nposterior", "9\npatient's right", "12\nanterior", "3\npatient's left", "6\nposterior"])
    ax.set_xlabel("clock position on the aortic wall (unrolled; radiological view)")
    ax.axhline(0.0, color="0.5", lw=0.8, ls="--")
    ax.axhline(length, color="0.5", lw=0.8, ls="--")
    ax.text(0.1, 0.0, "superior cut", fontsize=7, va="bottom", color="0.4")
    ax.text(0.1, length, "inferior cut", fontsize=7, va="top", color="0.4")
    ax.grid(True, alpha=0.3)
    n = len(rows)
    fig.suptitle(f"{result.get('case_id', '')}: {n} daughter{'s' if n != 1 else ''} on a {length:.0f} mm segment. "
                 "Marker area = radius; green = passed every rule, amber = flagged (*). Brackets: spacing along the centreline.",
                 fontsize=9)
    plt.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


# ------------------------------------------------------------------ Plotly 3D


def plotly_3d_div(cand: Candidates, result: dict, frame: Frame | None, downsample: int = 2) -> str | None:
    """Self-contained HTML div (Plotly JS inlined) or None when plotly is unavailable."""
    try:
        import plotly.graph_objects as go
        from skimage import measure
    except Exception as e:  # noqa: BLE001
        log.warning("plotly 3D view skipped: %s", e)
        return None
    m = cand.mask[::downsample, ::downsample, ::downsample]
    traces = []
    if m.any() and m.shape[0] > 1 and m.shape[1] > 1 and m.shape[2] > 1:
        mp = np.pad(m, 1)
        verts, faces, _, _ = measure.marching_cubes(mp.astype(np.uint8), level=0.5)
        verts = (verts - 1.0) * downsample  # working-grid zyx indices
        vmm = io_utils.index_to_mm(cand.image, verts)
        traces.append(go.Mesh3d(x=vmm[:, 0], y=vmm[:, 1], z=vmm[:, 2], i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                                color="lightcoral", opacity=0.35, name="aorta mask", showscale=False))
    if frame is not None and len(frame.centreline_mm) > 1:
        c = frame.centreline_mm
        traces.append(go.Scatter3d(x=c[:, 0], y=c[:, 1], z=c[:, 2], mode="lines", line=dict(color="royalblue", width=4), name="centreline"))
    ds = result.get("daughters", [])
    if ds:
        o = np.array([d["ostium_xyz_mm"] for d in ds], float)
        r = np.array([d["radius_mm"] for d in ds], float)
        v = np.array([d["direction_xyz"] for d in ds], float)
        rows = branch_rows(result, frame)
        text = [f"{d['instance_id']}<br>clock {clock_label(rw['clock'])}, height {rw['height_mm']:.0f} mm<br>radius {d['radius_mm']:.1f} mm"
                for d, rw in zip(ds, sorted(rows, key=lambda x: x["instance_id"]))]
        traces.append(go.Scatter3d(x=o[:, 0], y=o[:, 1], z=o[:, 2], mode="markers+text", text=[d["instance_id"][-3:] for d in ds],
                                   textposition="top center", hovertext=text, hoverinfo="text",
                                   marker=dict(size=np.clip(4 + 3 * r, 4, 20), color="gold", line=dict(color="black", width=1)), name="ostia"))
        xs, ys, zs = [], [], []
        for oi, vi in zip(o, v):
            tip = oi + ARROW_LENGTH_MM * vi
            xs += [oi[0], tip[0], None]
            ys += [oi[1], tip[1], None]
            zs += [oi[2], tip[2], None]
        traces.append(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=dict(color="red", width=6), name="direction (10 mm)"))
        tips = o + ARROW_LENGTH_MM * v
        traces.append(go.Scatter3d(x=tips[:, 0], y=tips[:, 1], z=tips[:, 2], mode="markers",
                                   marker=dict(size=4, color="red", symbol="diamond"), name="10 mm along the path", showlegend=False))
    fig = go.Figure(data=traces)
    fig.update_layout(scene=dict(xaxis_title="x (mm, LPS: +x = patient's left)", yaxis_title="y (mm, +y = posterior)",
                                 zaxis_title="z (mm, +z = superior)", aspectmode="data"),
                      margin=dict(l=0, r=0, t=30, b=0), height=650, legend=dict(orientation="h"),
                      title=f"{result.get('case_id', '')}: aorta mask, centreline, ostia and directions")
    return fig.to_html(full_html=False, include_plotlyjs=True, div_id="view3d")


# ------------------------------------------------------------------ HTML


def _img_tag(path: str, alt: str) -> str:
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"<img alt='{html.escape(alt)}' src='data:image/png;base64,{b64}' style='max-width:100%'>"


def write_html(result: dict, frame: Frame | None, path: str, png_name: str | None = None, meta: dict | None = None,
               clock_png: str | None = None, verification_png: str | None = None, plotly_div: str | None = None) -> str:
    rows = branch_rows(result, frame, meta)
    case = html.escape(result.get("case_id", ""))
    length = f"{frame.length_mm:.0f} mm" if frame is not None else "n/a"
    tort = f"{frame.tortuosity:.2f}" if frame is not None else "n/a"
    timing = ""
    if meta and meta.get("timings_s", {}).get("total") is not None:
        t = meta["timings_s"]
        timing = f" Pipeline {t['total']:.1f} s" + (f" (report excluded: {t['total'] - t.get('report', 0):.1f} s)." if "report" in t else ".")
    parts = [f"<h1>{case}</h1>",
             f"<p class='sum'>Supplied aortic segment {length} along the centreline, tortuosity {tort}. "
             f"<b>{len(rows)} daughter{'s' if len(rows) != 1 else ''}</b> reported.{html.escape(timing)}</p>",
             "<p class='conv'>Clock: 12 = anterior, 3 = patient's left, 6 = posterior, 9 = patient's right, measured in the plane "
             "perpendicular to the local centreline. Height = distance along the centreline from the superior end of the supplied mask. "
             "Branch IDs carry no anatomical meaning.</p>",
             "<h2>Branches</h2>",
             "<table><tr><th>ID</th><th>Clock</th><th>Height (mm)</th><th>Radius at seed (mm)</th><th>Origin diameter (mm)</th>"
             "<th>Direction xyz</th><th>Ostium xyz (mm)</th><th>Flags</th></tr>"]
    for r in rows:
        od = "n/a" if r["origin_diameter_mm"] is None else f"{r['origin_diameter_mm']:.1f}"
        fl = ", ".join(r["flags"]) if r["flags"] else "clean"
        cls = " class='flag'" if r["flags"] else ""
        parts.append(f"<tr{cls}><td>{html.escape(r['instance_id'])}</td><td>{clock_label(r['clock'])} ({r['clock']:.1f})</td>"
                     f"<td>{r['height_mm']:.1f}</td><td>{r['radius_mm']:.1f}</td><td>{od}</td>"
                     f"<td>{[round(v, 2) for v in r['direction_xyz']]}</td><td>{[round(v, 1) for v in r['ostium_xyz_mm']]}</td>"
                     f"<td>{html.escape(fl)}</td></tr>")
    parts.append("</table>")
    if frame is not None:
        gaps = frame_mod.spacing_along({r["instance_id"]: r["height_mm"] for r in rows}, frame.length_mm)
        parts.append("<h2>Spacing along the centreline</h2><table><tr><th>From</th><th>To</th><th>Gap (mm)</th></tr>" +
                     "".join(f"<tr><td>{html.escape(g['from'])}</td><td>{html.escape(g['to'])}</td><td>{g['gap_mm']:.1f}</td></tr>" for g in gaps) +
                     "</table>")
    if clock_png and os.path.exists(clock_png):
        parts.append("<h2>Clock map</h2>" + _img_tag(clock_png, "clock map"))
    if plotly_div:
        parts.append("<h2>3D view</h2>" + plotly_div)
    if verification_png and os.path.exists(verification_png):
        parts.append("<h2>Verification projections</h2>" + _img_tag(verification_png, "verification"))
    elif png_name:
        parts.append(f"<h2>Verification projections</h2><p><img src='{html.escape(png_name)}' style='max-width:100%'></p>")
    used = sorted({f for r in rows for f in r["flags"]})
    if used:
        parts.append("<h2>Flags</h2><dl>" + "".join(f"<dt>{html.escape(f)}</dt><dd>{html.escape(FLAG_GLOSSARY.get(f, ''))}</dd>" for f in used) + "</dl>")
    parts.append("<h2>Output JSON</h2><details><summary>prediction.json</summary><pre>" +
                 html.escape(json.dumps(result, indent=2)) + "</pre></details>")
    css = ("body{font-family:system-ui,sans-serif;margin:24px;max-width:1200px;color:#222}"
           "table{border-collapse:collapse;margin:8px 0}th,td{border:1px solid #bbb;padding:4px 8px;font-size:14px;text-align:left}"
           "th{background:#eee}tr.flag td{background:#fff6e0}.sum{font-size:16px}.conv{color:#555;font-size:13px}"
           "dt{font-weight:bold;margin-top:6px}dd{margin-left:16px;color:#444}pre{background:#f6f6f6;padding:8px;overflow:auto}")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"<!doctype html><html><head><meta charset='utf-8'><title>{case}</title><style>{css}</style></head><body>"
                + "\n".join(parts) + "</body></html>")
    return path


def write_all(cand: Candidates, result: dict, frame: Frame | None, out_dir: str, case_id: str, meta: dict | None = None) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    png = os.path.join(out_dir, f"{case_id}_check.png")
    clock = os.path.join(out_dir, f"{case_id}_clock.png")
    htm = os.path.join(out_dir, f"{case_id}_report.html")
    write_verification_png(cand, result, png, frame)
    write_clock_map_png(result, frame, clock, meta)
    div = plotly_3d_div(cand, result, frame)
    write_html(result, frame, htm, png_name=os.path.basename(png), meta=meta, clock_png=clock, verification_png=png, plotly_div=div)
    log.info("wrote %s, %s and %s", png, clock, htm)
    return {"png": png, "clock_png": clock, "html": htm, "plotly": div is not None}
