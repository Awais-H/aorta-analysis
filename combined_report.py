"""Combined browser across all 25 subjects: one HTML page with a patient dropdown, built purely
from existing results/ (per-case prediction JSON, the PNGs and the Plotly 3D figures report.py
already wrote) -- no pipeline re-run. Read-only aggregation, no anatomical names, branch IDs only
(CLAUDE.md).

    python combined_report.py

Writes, all into results/visual_checks/:
  index.html          the page
  plotly.min.js       the Plotly bundle lifted once out of a per-case report (they each inline
                      their own 4.3 MB copy; sharing one costs ~4 MB instead of ~107 MB)
  subjectNNN_3d.js    that case's figure, assigned to window.__AORTA3D__, loaded on demand

Everything is referenced relatively, so the page must stay in results/visual_checks/ next to those
files. It needs no server and no network: the 3D sidecars are loaded as <script> tags rather than
fetch(), which file:// blocks.
"""
from __future__ import annotations

import glob
import json
import os
import re

ROOT = os.path.dirname(os.path.abspath(__file__))
PRED_DIR = os.path.join(ROOT, "results", "predictions")
VC_DIR = os.path.join(ROOT, "results", "visual_checks")
OUT_PATH = os.path.join(VC_DIR, "index.html")
PLOTLY_PATH = os.path.join(VC_DIR, "plotly.min.js")

# palette: brand cyan, black, white only -- no gradients (per user request)
CYAN = "#55C6DE"
INK = "#0E0E0E"
PAPER = "#FFFFFF"
LINE = "#E6E6E6"
MUTED = "#6B6B6B"


def _scan_value(text: str, start: int) -> int:
    """End index (exclusive) of the JSON array or object beginning at `start`, string-aware."""
    depth, i, in_str, esc = 0, start, False, False
    while i < len(text):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch in "[{":
            depth += 1
        elif ch in "]}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    raise ValueError("unterminated JSON value")


def _extract_plotly(report_path: str) -> tuple[str, str] | None:
    """(data, layout) as raw JS text from a report's Plotly.newPlot call, or None if it has none.

    The arguments are copied verbatim rather than re-encoded: Plotly writes typed arrays as
    base64 ({"dtype":"i4","bdata":...}), and the same bundle decodes them on the way back in.
    """
    with open(report_path, encoding="utf-8") as f:
        html_text = f.read()
    i = html_text.find("Plotly.newPlot(")
    if i < 0:
        return None
    i = html_text.index(",", i + len("Plotly.newPlot(")) + 1  # skip the div id argument
    d0 = html_text.index("[", i)
    d1 = _scan_value(html_text, d0)
    l0 = html_text.index("{", d1)
    l1 = _scan_value(html_text, l0)
    return html_text[d0:d1], html_text[l0:l1]


def _extract_bundle(report_path: str) -> str | None:
    """The plotly.js bundle inlined in a report, without its <script> wrapper."""
    with open(report_path, encoding="utf-8") as f:
        html_text = f.read()
    m = re.search(r"<script>(/\*\*\s*\n\* plotly\.js)", html_text)
    if not m:
        return None
    start = m.start(1)
    return html_text[start:html_text.index("</script>", start)]


def _write_3d_assets(cases: list[dict]) -> None:
    """plotly.min.js once, plus one subjectNNN_3d.js per case that has a figure."""
    for c in cases:
        report = os.path.join(VC_DIR, f"{c['case_id']}_report.html")
        if not os.path.exists(report):
            continue
        if not os.path.exists(PLOTLY_PATH):
            bundle = _extract_bundle(report)
            if bundle:
                with open(PLOTLY_PATH, "w", encoding="utf-8") as f:
                    f.write(bundle)
        fig = _extract_plotly(report)
        if fig is None:
            continue
        data, layout = fig
        name = f"{c['case_id']}_3d.js"
        with open(os.path.join(VC_DIR, name), "w", encoding="utf-8") as f:
            f.write("window.__AORTA3D__=window.__AORTA3D__||{};\n"
                    f"window.__AORTA3D__[{json.dumps(c['case_id'])}]=" +
                    "{data:" + data + ",layout:" + layout + "};\n")
        c["fig_js"] = name


def challenge_daughter(d: dict) -> dict:
    """The challenge JSON object for one daughter — nothing internal, nothing extra."""
    return {
        "instance_id": d["instance_id"],
        "parent_instance_id": d.get("parent_instance_id", "aorta"),
        "ostium_xyz_mm": d["ostium_xyz_mm"],
        "seed_xyz_mm": d["seed_xyz_mm"],
        "radius_mm": d["radius_mm"],
        "direction_xyz": d["direction_xyz"],
    }


def challenge_case(pred: dict) -> dict:
    """The case-level challenge JSON (parent + daughters), for the viewer and copy control."""
    return {
        "case_id": pred["case_id"],
        "parent": pred.get("parent") or {"instance_id": "aorta"},
        "daughters": [challenge_daughter(d) for d in pred.get("daughters", [])],
    }


def _load_cases() -> list[dict]:
    """case_id, daughters and the PNGs that exist for it, ordered by case number."""
    cases = []
    for p in sorted(glob.glob(os.path.join(PRED_DIR, "subject*.json"))):
        if p.endswith("_meta.json"):
            continue
        with open(p, encoding="utf-8") as f:
            pred = json.load(f)
        cid = pred["case_id"]
        daughters = []
        for d in pred.get("daughters", []):
            payload = challenge_daughter(d)
            daughters.append({
                "id": payload["instance_id"],
                "radius_mm": payload["radius_mm"],
                "ostium": payload["ostium_xyz_mm"],
                "seed": payload["seed_xyz_mm"],
                "direction": payload["direction_xyz"],
                "json": payload,
            })
        cases.append({
            "case_id": cid,
            "daughters": daughters,
            "json": challenge_case(pred),
            "clock_png": f"{cid}_clock.png" if os.path.exists(os.path.join(VC_DIR, f"{cid}_clock.png")) else None,
            "check_png": f"{cid}_check.png" if os.path.exists(os.path.join(VC_DIR, f"{cid}_check.png")) else None,
            "report": f"{cid}_report.html" if os.path.exists(os.path.join(VC_DIR, f"{cid}_report.html")) else None,
            "fig_js": None,  # filled in by _write_3d_assets
        })
    return cases


CSS = f"""
*, *::before, *::after {{ box-sizing: border-box; }}
html, body {{ height: 100%; }}
body {{
  margin: 0; background: {PAPER}; color: {INK};
  font-family: -apple-system, BlinkMacSystemFont, 'Inter', 'Helvetica Neue', Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
}}
a {{ color: inherit; }}
[hidden] {{ display: none !important; }}

/* ---------- top bar ---------- */
.topbar {{
  position: sticky; top: 0; z-index: 20; background: {PAPER};
  border-bottom: 1px solid {LINE}; display: flex; align-items: center; gap: 16px;
  padding: 0 20px; height: 56px;
}}
.brand {{ margin-right: auto; }}
.brand h1 {{ font-size: 28px; font-weight: 750; margin: 0; letter-spacing: -0.03em; color: {CYAN}; }}

.picker {{ display: flex; align-items: center; gap: 8px; }}
.picker label {{ font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em; color: {MUTED}; }}

/* searchable patient combobox */
.combo {{ position: relative; }}
.combo input {{
  font: inherit; font-size: 14px; font-weight: 550; color: {INK};
  background: {PAPER}; border: 1.5px solid {INK}; border-radius: 8px;
  padding: 8px 34px 8px 36px; width: 240px;
  background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='14' height='14' viewBox='0 0 14 14'><circle cx='6' cy='6' r='4.4' fill='none' stroke='%236B6B6B' stroke-width='1.6'/><path d='M9.4 9.4 12.6 12.6' stroke='%236B6B6B' stroke-width='1.6' stroke-linecap='round'/></svg>");
  background-repeat: no-repeat; background-position: left 12px center;
}}
.combo input::placeholder {{ color: {MUTED}; font-weight: 450; }}
.combo input:focus {{ outline: none; border-color: {CYAN}; box-shadow: 0 0 0 3px rgba(85,198,222,0.28); }}
.combo .caret {{
  position: absolute; right: 12px; top: 50%; transform: translateY(-50%);
  pointer-events: none; color: {MUTED};
}}
.options {{
  display: none; position: absolute; top: calc(100% + 6px); left: 0; right: 0; z-index: 40;
  background: {PAPER}; border: 1px solid {LINE}; border-radius: 10px; padding: 4px;
  max-height: 320px; overflow-y: auto; box-shadow: 0 10px 28px rgba(0,0,0,0.13);
}}
.options.open {{ display: block; }}
.opt {{
  display: flex; align-items: baseline; justify-content: space-between; gap: 12px;
  padding: 9px 11px; border-radius: 7px; font-size: 13px; cursor: pointer;
}}
.opt .n {{ font-size: 11.5px; color: {MUTED}; white-space: nowrap; }}
.opt.active {{ background: {CYAN}; }}
.opt.active .n {{ color: {INK}; }}
.opt mark {{ background: transparent; color: inherit; font-weight: 750; text-decoration: underline; }}
.opt.current .name {{ font-weight: 700; }}
.options .none {{ padding: 14px 12px; text-align: center; font-size: 12.5px; color: {MUTED}; }}
.stepper {{ display: flex; }}
.stepper button {{
  font: inherit; background: {PAPER}; color: {INK}; border: 1.5px solid {INK};
  width: 38px; height: 38px; cursor: pointer; display: grid; place-items: center;
}}
.stepper button:first-child {{ border-radius: 8px 0 0 8px; border-right-width: 0.75px; }}
.stepper button:last-child {{ border-radius: 0 8px 8px 0; border-left-width: 0.75px; }}
.stepper button:hover:not(:disabled) {{ background: {CYAN}; border-color: {CYAN}; }}
.stepper button:disabled {{ opacity: 0.3; cursor: not-allowed; }}

/* ---------- case header ---------- */
.casebar {{
  display: flex; align-items: center; gap: 16px; flex-wrap: wrap;
  padding: 14px 20px; border-bottom: 1px solid {LINE};
}}
.casebar h2 {{ margin: 0; font-size: 20px; font-weight: 680; letter-spacing: -0.02em; }}
.pill {{
  background: {CYAN}; color: {INK}; border-radius: 999px;
  padding: 5px 13px; font-size: 13px; font-weight: 600;
}}
.pill.zero {{ background: {PAPER}; border: 1.5px solid {LINE}; color: {MUTED}; }}
.stats {{ display: flex; gap: 26px; margin-left: auto; flex-wrap: wrap; }}
.stat .n {{ display: block; font-size: 17px; font-weight: 650; font-variant-numeric: tabular-nums; }}
.stat .l {{ display: block; font-size: 10px; text-transform: uppercase; letter-spacing: 0.08em; color: {MUTED}; margin-top: 2px; }}
.ghost {{
  text-decoration: none; font-size: 13px; font-weight: 550; border: 1.5px solid {INK};
  border-radius: 8px; padding: 8px 14px; white-space: nowrap;
}}
.ghost:hover {{ background: {CYAN}; border-color: {CYAN}; }}

/* ---------- layout ---------- */
.layout {{ display: grid; grid-template-columns: minmax(0,1fr) 420px; gap: 0; align-items: start; }}
@media (max-width: 1080px) {{ .layout {{ grid-template-columns: minmax(0,1fr); }} }}

.viewer {{ padding: 16px 20px 32px; min-width: 0; }}
.segmented {{
  display: inline-flex; background: #F4F4F4; border-radius: 9px; padding: 3px; margin-bottom: 16px;
}}
.segmented button {{
  font: inherit; font-size: 13px; font-weight: 550; border: 0; background: transparent; color: {MUTED};
  padding: 7px 16px; border-radius: 7px; cursor: pointer;
}}
.segmented button[aria-selected="true"] {{ background: {PAPER}; color: {INK}; box-shadow: 0 1px 2px rgba(0,0,0,0.14); }}
.frame {{
  border: 1px solid {LINE}; border-radius: 12px; overflow: hidden; background: {PAPER};
  display: grid; place-items: center; padding: 8px;
}}
.frame img {{ display: block; max-width: 100%; max-height: 74vh; object-fit: contain; }}
.frame .empty {{ padding: 64px 24px; text-align: center; color: {MUTED}; font-size: 13px; }}
.frame .loading {{ padding: 64px 24px; text-align: center; color: {MUTED}; font-size: 13px; }}
.frame .loading .spin {{
  display: block; width: 22px; height: 22px; margin: 0 auto 12px; border-radius: 50%;
  border: 2.5px solid {LINE}; border-top-color: {CYAN}; animation: spin 0.8s linear infinite;
}}
@keyframes spin {{ to {{ transform: rotate(360deg); }} }}
.frame.is3d {{ display: block; padding: 0; }}
#plot3d {{ width: 100%; height: 68vh; min-height: 400px; }}
.caption {{ font-size: 12px; color: {MUTED}; margin: 8px 2px 0; line-height: 1.5; max-width: 72ch; }}

/* ---------- branch panel ---------- */
.panel {{
  border-left: 1px solid {LINE}; padding: 16px 16px 28px;
  position: sticky; top: 56px; max-height: calc(100vh - 56px); overflow: auto;
}}
@media (max-width: 1080px) {{
  .panel {{ position: static; max-height: none; border-left: 0; border-top: 1px solid {LINE}; }}
}}
.panel-head {{
  display: flex; align-items: center; justify-content: space-between; gap: 10px;
  margin: 0 0 12px; flex-wrap: wrap;
}}
.panel-head h3 {{
  font-size: 11px; text-transform: uppercase; letter-spacing: 0.09em; color: {MUTED};
  margin: 0; font-weight: 650;
}}
.panel-actions {{
  display: flex; align-items: center; gap: 8px; flex-wrap: wrap; margin-left: auto;
}}
.panel-head .segmented {{ margin-bottom: 0; }}
.jsonwrap.casejson pre {{
  max-height: calc(100vh - 220px); padding-right: 12px;
}}
.copybtn {{
  font: inherit; font-size: 11px; font-weight: 600; color: {INK};
  background: {PAPER}; border: 1px solid {LINE}; border-radius: 6px;
  padding: 4px 8px; cursor: pointer; white-space: nowrap;
}}
.copybtn:hover {{ border-color: {INK}; }}
.copybtn.done {{ border-color: {CYAN}; background: {CYAN}; }}
.branch {{ border: 1px solid {LINE}; border-radius: 10px; margin-bottom: 8px; overflow: hidden; }}
.branch > summary {{
  list-style: none; cursor: pointer; display: flex; align-items: center; gap: 10px;
  padding: 11px 12px; user-select: none;
}}
.branch > summary::-webkit-details-marker {{ display: none; }}
.branch:hover {{ border-color: {CYAN}; }}
.branch[open] {{ border-color: {INK}; }}
.branch .idx {{
  width: 24px; height: 24px; border-radius: 50%; background: {CYAN}; color: {INK}; flex: none;
  display: grid; place-items: center; font-size: 11px; font-weight: 700; font-variant-numeric: tabular-nums;
}}
.branch .label {{ display: flex; flex-direction: column; gap: 1px; min-width: 0; }}
.branch .name {{ font-size: 13px; font-weight: 600; }}
.branch .sub {{ font-size: 11px; color: {MUTED}; }}
.branch .arrowslot {{ margin-left: auto; flex: none; display: flex; opacity: 0.85; }}
.branch .chev {{ flex: none; transition: transform 0.15s ease; }}
.branch[open] .chev {{ transform: rotate(180deg); }}
.branch .body {{ border-top: 1px solid {LINE}; padding: 10px; }}
.fmt {{ display: flex; justify-content: flex-end; margin-bottom: 8px; }}
.segmented.mini {{ margin: 0; }}
.segmented.mini button {{ font-size: 11px; padding: 5px 10px; }}
.kv {{ display: grid; grid-template-columns: 74px 1fr; gap: 6px 10px; font-size: 12px; padding: 2px 2px 4px; }}
.kv dt {{ color: {MUTED}; }}
.kv dd {{
  margin: 0; font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 11.5px; font-variant-numeric: tabular-nums;
}}
.jsonwrap {{
  position: relative; background: #F6F6F6; border-radius: 8px; overflow: hidden;
}}
.jsonwrap .copybtn {{ position: absolute; top: 8px; right: 8px; z-index: 1; }}
.jsonwrap pre {{
  margin: 0; padding: 10px 76px 10px 12px; overflow-x: auto;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 11px; line-height: 1.45; color: {INK};
  font-variant-numeric: tabular-nums;
}}
.jsonwrap .jk {{ color: {MUTED}; }}
.jsonwrap .jn {{ color: {CYAN}; }}
.jsonwrap .js {{ color: {INK}; }}
.empty-branches {{
  border: 1px dashed {LINE}; border-radius: 10px; padding: 22px 16px; text-align: center;
  font-size: 12.5px; color: {MUTED};
}}
"""

JS = """
const CASES = __DATA__;
let current = 0, view = '3d';
let panelView = 'branches';       // 'branches' | 'casejson'
let plotlyReady = null;           // promise, resolved once the shared bundle is in
const figLoaded = {};             // case_id -> promise for that case's figure sidecar
let renderToken = 0;              // guards against a slow 3D load painting over a newer case

const $ = (s) => document.querySelector(s);
const fmt = (v, n = 1) => Number(v).toFixed(n);
const vec = (a, n = 2) => a.map((v) => Number(v).toFixed(n)).join(', ');
let detailFmt = 'json';   // 'fields' | 'json' — last choice, applied to every card on render

function escapeHtml(s) {
  return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function prettyJson(obj) {
  const esc = escapeHtml(JSON.stringify(obj, null, 2));
  return esc.replace(
    /("(?:\\\\u[a-fA-F0-9]{4}|\\\\[^u]|[^\\\\"])*")(\\s*:)?|\\b(-?\\d+(?:\\.\\d+)?(?:[eE][+-]?\\d+)?)\\b|\\b(true|false|null)\\b/g,
    (m, str, colon, num, kw) => {
      if (str) return colon ? `<span class="jk">${str}</span>${colon}` : `<span class="js">${str}</span>`;
      if (num) return `<span class="jn">${num}</span>`;
      return `<span class="js">${kw}</span>`;
    }
  );
}

function copyJson(btn, obj) {
  const text = JSON.stringify(obj, null, 2);
  const done = () => {
    const prev = btn.textContent;
    btn.textContent = 'Copied';
    btn.classList.add('done');
    setTimeout(() => { btn.textContent = prev; btn.classList.remove('done'); }, 1100);
  };
  if (navigator.clipboard && window.isSecureContext) {
    navigator.clipboard.writeText(text).then(done).catch(() => fallbackCopy(text, done));
  } else {
    fallbackCopy(text, done);
  }
}

function fallbackCopy(text, done) {
  const t = document.createElement('textarea');
  t.value = text;
  t.setAttribute('readonly', '');
  t.style.position = 'fixed';
  t.style.left = '-9999px';
  document.body.appendChild(t);
  t.select();
  try { document.execCommand('copy'); } catch (e) { /* ignore */ }
  t.remove();
  done();
}

// The panel arrow is the initial direction as the 3D view currently shows it, so the two agree at
// a glance. Screen basis from the camera: right = normalise(forward x up), up = right x forward,
// with world up = +z. The scene is aspectmode 'data' (equal scale on every axis), so a direction
// vector projects without distortion.
const DEFAULT_EYE = { x: 1.4, y: -1.4, z: 0.6 };   // report.py pins this; scaling it changes nothing here
let camEye = DEFAULT_EYE;

function camBasis(eye) {
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const unit = (a) => { const n = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / n, a[1] / n, a[2] / n]; };
  const fwd = unit([-eye.x, -eye.y, -eye.z]);
  const up = [0, 0, 1];
  let right = cross(fwd, up);
  if (Math.hypot(right[0], right[1], right[2]) < 1e-6) right = [1, 0, 0];  // looking straight down the z axis
  right = unit(right);
  return { right: right, up: unit(cross(right, fwd)) };
}

function arrowSvg(dir) {
  const b = camBasis(camEye);
  const sx = dir[0] * b.right[0] + dir[1] * b.right[1] + dir[2] * b.right[2];
  const sy = dir[0] * b.up[0] + dir[1] * b.up[1] + dir[2] * b.up[2];
  const ang = -Math.atan2(sy, sx) * 180 / Math.PI;   // SVG y grows downward
  return `<svg class="arrow" width="20" height="20" viewBox="0 0 20 20" aria-hidden="true">
    <g transform="rotate(${ang.toFixed(1)} 10 10)">
      <line x1="3" y1="10" x2="15" y2="10" stroke="#0E0E0E" stroke-width="1.6" stroke-linecap="round"/>
      <path d="M15 10 11.5 7.4 M15 10 11.5 12.6" fill="none" stroke="#0E0E0E" stroke-width="1.6"
            stroke-linecap="round" stroke-linejoin="round"/>
    </g></svg>`;
}

function refreshArrows() {
  // redraw the glyphs in place, so rotating the 3D view does not collapse an expanded branch
  const ds = CASES[current].daughters;
  document.querySelectorAll('#branches .arrowslot').forEach((slot, i) => {
    if (ds[i]) slot.innerHTML = arrowSvg(ds[i].direction);
  });
}

const CHEV = `<svg class="chev" width="11" height="7" viewBox="0 0 12 8" aria-hidden="true">
  <path d="M1 1.5 6 6.5 11 1.5" fill="none" stroke="#6B6B6B" stroke-width="1.8"
        stroke-linecap="round" stroke-linejoin="round"/></svg>`;

function loadScript(src) {
  return new Promise((resolve, reject) => {
    const s = document.createElement('script');
    s.src = src;
    s.onload = resolve;
    s.onerror = () => reject(new Error('could not load ' + src));
    document.head.appendChild(s);
  });
}

function ensurePlotly() {
  // file:// blocks fetch(), so the bundle and the figures come in as plain <script> tags.
  if (!plotlyReady) plotlyReady = loadScript('plotly.min.js');
  return plotlyReady;
}

function ensureFigure(c) {
  if (!figLoaded[c.case_id]) figLoaded[c.case_id] = loadScript(c.fig_js);
  return figLoaded[c.case_id];
}

// Recolour report.py's figure into this page's palette without touching its geometry. The aorta
// keeps the report's red so the parent vessel reads the same in both views; everything reported
// on top of it is page palette, with the ostium the one saturated marker.
function repaint(trace) {
  const t = Object.assign({}, trace);
  if (t.type === 'mesh3d') {
    t.color = 'lightcoral';
    t.opacity = 0.35;
    t.showlegend = true;
  } else if (t.name === 'centreline') {
    t.line = Object.assign({}, t.line, { color: '#0E0E0E', width: 3 });
  } else if (t.name === 'ostia') {
    t.marker = Object.assign({}, t.marker, {
      color: '#55C6DE', line: { color: '#0E0E0E', width: 1.5 },
    });
    t.textfont = { color: '#0E0E0E', size: 11 };
  } else if (t.mode === 'lines') {
    t.line = Object.assign({}, t.line, { color: '#0E0E0E', width: 5 });
  } else if (t.mode === 'markers') {
    t.marker = Object.assign({}, t.marker, { color: '#0E0E0E' });
  }
  return t;
}

function render3d(c, token) {
  const frame = document.getElementById('frame');
  if (!c.fig_js) {
    frame.innerHTML = '<div class="empty">no 3D figure for this case</div>';
    return;
  }
  frame.innerHTML = '<div class="loading"><span class="spin"></span>building the 3D view\\u2026</div>';
  ensurePlotly()
    .then(() => ensureFigure(c))
    .then(() => {
      if (token !== renderToken) return;   // the user moved on while this was loading
      const fig = window.__AORTA3D__[c.case_id];
      frame.innerHTML = '<div id="plot3d"></div>';
      // aspectmode 'data' fits the bounding cube, so a long thin segment overruns the frame;
      // pull the camera back along the same axis report.py pinned it to.
      const scene = Object.assign({}, fig.layout.scene);
      const eye = (scene.camera && scene.camera.eye) || { x: 1.4, y: -1.4, z: 0.6 };
      scene.camera = Object.assign({}, scene.camera, {
        eye: { x: eye.x * 1.5, y: eye.y * 1.5, z: eye.z * 1.5 },
      });
      const layout = Object.assign({}, fig.layout, {
        scene: scene,
        title: null, height: null, autosize: true,
        margin: { l: 0, r: 0, t: 0, b: 0 },
        paper_bgcolor: '#FFFFFF', plot_bgcolor: '#FFFFFF',
        legend: { orientation: 'h', y: 1.04, x: 0, bgcolor: 'rgba(255,255,255,0.85)' },
      });
      camEye = scene.camera.eye;
      Plotly.newPlot('plot3d', fig.data.map(repaint), layout, { responsive: true, displaylogo: false })
        .then((gd) => {
          refreshArrows();
          gd.on('plotly_relayout', (e) => {
            const eye = (e['scene.camera'] && e['scene.camera'].eye)
              || (e.scene && e.scene.camera && e.scene.camera.eye);
            if (eye) { camEye = eye; refreshArrows(); }
          });
        });
    })
    .catch((e) => {
      if (token !== renderToken) return;
      frame.innerHTML = '<div class="empty">3D view unavailable: ' + e.message + '</div>';
    });
}

function render(opts) {
  const panelOnly = opts && opts.panelOnly;
  const c = CASES[current];
  const n = c.daughters.length;
  const radii = c.daughters.map((d) => d.radius_mm);

  if (!comboOpen) $('#patientSearch').value = c.case_id;   // don't clobber what is being typed
  $('#prev').disabled = current === 0;
  $('#next').disabled = current === CASES.length - 1;

  $('#caseName').textContent = c.case_id;
  const pill = $('#casePill');
  pill.textContent = n === 0 ? 'none detected' : `${n} branch${n === 1 ? '' : 'es'} detected`;
  pill.className = n === 0 ? 'pill zero' : 'pill';

  $('#statRadius').textContent = n ? `${fmt(Math.min(...radii))}–${fmt(Math.max(...radii))} mm` : '—';
  $('#statCase').textContent = `${current + 1} / ${CASES.length}`;
  const link = $('#reportLink');
  if (c.report) { link.style.display = ''; link.href = c.report; } else { link.style.display = 'none'; }

  // viewer — skip when only the branch/case-JSON toggle changed, so the 3D plot stays put
  if (!panelOnly) {
    const token = ++renderToken;
    const frame = $('#frame');
    frame.classList.toggle('is3d', view === '3d');
    if (view === '3d') {
      render3d(c, token);
      $('#caption').textContent = 'Drag to rotate. Red: supplied aorta. Black: centreline and 10 mm directions. Cyan: ostia.';
    } else {
      const src = view === 'clock' ? c.clock_png : c.check_png;
      frame.innerHTML = src
        ? `<img src="${src}" alt="${c.case_id} ${view === 'clock' ? 'clock map' : 'verification projections'}">`
        : `<div class="empty">no ${view === 'clock' ? 'clock map' : 'verification'} image for this case</div>`;
      $('#caption').textContent = view === 'clock'
        ? 'Wall unrolled: 12 anterior, 3 patient left, 6 posterior. Height is from the superior cut. Marker size is radius.'
        : 'Pink mask, cyan ostia, red 10 mm directions, blue centreline.';
    }
    document.querySelectorAll('[data-view]').forEach((b) =>
      b.setAttribute('aria-selected', String(b.dataset.view === view)));
  }

  // branches — each card is the official challenge JSON for that instance
  $('#branchCount').textContent = panelView === 'casejson'
    ? 'Case JSON'
    : (n ? `Detected branches (${n})` : 'Detected branches');
  document.querySelectorAll('[data-panel]').forEach((b) =>
    b.setAttribute('aria-selected', String(b.dataset.panel === panelView)));
  const copyCase = $('#copyCase');
  copyCase.onclick = () => copyJson(copyCase, c.json);
  $('#branches').hidden = panelView === 'casejson';
  $('#caseJson').hidden = panelView === 'branches';
  $('#caseJsonPre').innerHTML = prettyJson(c.json);
  if (panelOnly && panelView === 'casejson') return;
  $('#branches').innerHTML = n === 0
    ? `<div class="empty-branches">No eligible daughters on the supplied segment.<br>
       Anatomy outside the supplied coverage cannot be assessed.</div>`
    : c.daughters.map((d, i) => `
      <details class="branch"${i === 0 ? ' open' : ''}>
        <summary>
          <span class="idx">${String(i + 1).padStart(2, '0')}</span>
          <span class="label">
            <span class="name">${d.id}</span>
            <span class="sub">radius ${fmt(d.radius_mm)} mm</span>
          </span>
          <span class="arrowslot" title="initial direction, as oriented in the 3D view">${arrowSvg(d.direction)}</span>
          ${CHEV}
        </summary>
        <div class="body">
          <div class="fmt">
            <div class="segmented mini" role="tablist">
              <button type="button" data-fmt="fields" aria-selected="${detailFmt === 'fields'}">Fields</button>
              <button type="button" data-fmt="json" aria-selected="${detailFmt === 'json'}">JSON</button>
            </div>
          </div>
          <dl class="kv"${detailFmt === 'json' ? ' hidden' : ''}>
            <dt>Ostium</dt><dd>${vec(d.ostium, 1)} mm</dd>
            <dt>Seed</dt><dd>${vec(d.seed, 1)} mm</dd>
            <dt>Direction</dt><dd>${vec(d.direction, 3)}</dd>
            <dt>Radius</dt><dd>${fmt(d.radius_mm, 2)} mm</dd>
          </dl>
          <div class="jsonwrap"${detailFmt === 'fields' ? ' hidden' : ''}>
            <button type="button" class="copybtn" data-copy="${i}">Copy</button>
            <pre>${prettyJson(d.json)}</pre>
          </div>
        </div>
      </details>`).join('');
}

function go(i) {
  current = Math.max(0, Math.min(CASES.length - 1, i));
  if (comboOpen) closeCombo();
  render();
}

// ---------- patient search ----------
let comboOpen = false, comboActive = 0, comboMatches = [];

function matchesFor(query) {
  const q = query.trim().toLowerCase();
  return CASES.map((c, i) => ({ c: c, i: i }))
              .filter((m) => !q || m.c.case_id.toLowerCase().includes(q));
}

function drawOptions(query) {
  const q = query.trim().toLowerCase();
  const list = document.getElementById('patientList');
  if (!comboMatches.length) {
    list.innerHTML = '<div class="none">No patient matches that search.</div>';
    return;
  }
  list.innerHTML = comboMatches.map((m, k) => {
    const id = m.c.case_id;
    const at = q ? id.toLowerCase().indexOf(q) : -1;
    const label = at < 0 ? id
      : id.slice(0, at) + '<mark>' + id.slice(at, at + q.length) + '</mark>' + id.slice(at + q.length);
    const n = m.c.daughters.length;
    return `<div class="opt${k === comboActive ? ' active' : ''}${m.i === current ? ' current' : ''}"
                 role="option" aria-selected="${k === comboActive}" data-i="${m.i}">
              <span class="name">${label}</span>
              <span class="n">${n} branch${n === 1 ? '' : 'es'}</span>
            </div>`;
  }).join('');
  const active = list.querySelector('.opt.active');
  if (active) active.scrollIntoView({ block: 'nearest' });
}

function openCombo(query) {
  comboOpen = true;
  comboMatches = matchesFor(query);
  comboActive = Math.max(0, comboMatches.findIndex((m) => m.i === current));
  document.getElementById('patientList').classList.add('open');
  document.getElementById('patientSearch').setAttribute('aria-expanded', 'true');
  drawOptions(query);
}

function closeCombo() {
  comboOpen = false;
  document.getElementById('patientList').classList.remove('open');
  const input = document.getElementById('patientSearch');
  input.setAttribute('aria-expanded', 'false');
  input.value = CASES[current].case_id;   // always settle back on the case actually shown
}

document.addEventListener('DOMContentLoaded', () => {
  const input = $('#patientSearch');
  const list = $('#patientList');

  input.addEventListener('focus', () => { input.select(); openCombo(''); });
  input.addEventListener('input', () => openCombo(input.value));
  input.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
      e.preventDefault();
      if (!comboOpen) return openCombo('');
      if (comboMatches.length) {
        comboActive = (comboActive + (e.key === 'ArrowDown' ? 1 : -1) + comboMatches.length) % comboMatches.length;
        drawOptions(input.value);
      }
    } else if (e.key === 'Enter') {
      e.preventDefault();
      if (comboOpen && comboMatches[comboActive]) { go(comboMatches[comboActive].i); input.blur(); }
    } else if (e.key === 'Escape') {
      closeCombo();
      input.blur();
    }
  });
  list.addEventListener('mousedown', (e) => {   // mousedown: fires before the input's blur
    const opt = e.target.closest('.opt');
    if (!opt) return;
    e.preventDefault();
    go(Number(opt.dataset.i));
    input.blur();
  });
  list.addEventListener('mousemove', (e) => {
    const opt = e.target.closest('.opt');
    if (!opt) return;
    const k = comboMatches.findIndex((m) => m.i === Number(opt.dataset.i));
    if (k >= 0 && k !== comboActive) { comboActive = k; drawOptions(input.value); }
  });
  input.addEventListener('blur', closeCombo);

  $('#prev').addEventListener('click', () => go(current - 1));
  $('#next').addEventListener('click', () => go(current + 1));
  document.querySelectorAll('[data-view]').forEach((b) =>
    b.addEventListener('click', () => { view = b.dataset.view; render(); }));
  document.querySelectorAll('[data-panel]').forEach((b) =>
    b.addEventListener('click', () => {
      if (panelView === b.dataset.panel) return;
      panelView = b.dataset.panel;
      render({ panelOnly: true });
    }));
  document.addEventListener('keydown', (e) => {
    if (e.target === input) return;
    if (e.key === 'ArrowLeft') go(current - 1);
    if (e.key === 'ArrowRight') go(current + 1);
  });
  $('#branches').addEventListener('click', (e) => {
    const fmtBtn = e.target.closest('[data-fmt]');
    if (fmtBtn) {
      e.preventDefault();
      detailFmt = fmtBtn.dataset.fmt;
      const body = fmtBtn.closest('.body');
      if (!body) return;
      body.querySelectorAll('[data-fmt]').forEach((b) =>
        b.setAttribute('aria-selected', String(b.dataset.fmt === detailFmt)));
      const kv = body.querySelector('.kv');
      const jw = body.querySelector('.jsonwrap');
      if (kv) kv.hidden = detailFmt === 'json';
      if (jw) jw.hidden = detailFmt === 'fields';
      return;
    }
    const btn = e.target.closest('[data-copy]');
    if (!btn) return;
    e.preventDefault();
    const d = CASES[current].daughters[Number(btn.dataset.copy)];
    if (d) copyJson(btn, d.json);
  });
  $('#branches').addEventListener('toggle', (e) => {
    const card = e.target;
    if (!(card instanceof HTMLDetailsElement) || !card.open) return;
    $('#branches').querySelectorAll('details.branch').forEach((el) => {
      if (el !== card) el.open = false;
    });
  }, true);
  render();
});
"""


def build() -> str:
    cases = _load_cases()
    _write_3d_assets(cases)
    data = json.dumps(cases, separators=(",", ":"))
    script = JS.replace("__DATA__", data)

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Devar</title>
<style>{CSS}</style></head>
<body>

<div class="topbar">
  <div class="brand"><h1>Devar</h1></div>
  <div class="picker">
    <label for="patientSearch">Patient</label>
    <div class="combo">
      <input id="patientSearch" type="text" role="combobox" aria-expanded="false"
             aria-controls="patientList" aria-autocomplete="list" autocomplete="off"
             spellcheck="false" placeholder="Search patients&hellip;">
      <span class="caret">
        <svg width="12" height="8" viewBox="0 0 12 8" aria-hidden="true"><path d="M1 1.5 6 6.5 11 1.5"
          fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>
      </span>
      <div id="patientList" class="options" role="listbox"></div>
    </div>
    <div class="stepper">
      <button id="prev" title="Previous patient (&larr;)" aria-label="Previous patient">
        <svg width="8" height="12" viewBox="0 0 8 12"><path d="M6.5 1 1.5 6l5 5" fill="none"
          stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>
      </button>
      <button id="next" title="Next patient (&rarr;)" aria-label="Next patient">
        <svg width="8" height="12" viewBox="0 0 8 12"><path d="M1.5 1l5 5-5 5" fill="none"
          stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>
      </button>
    </div>
  </div>
</div>

<div class="casebar">
  <h2 id="caseName"></h2>
  <span id="casePill" class="pill"></span>
  <div class="stats">
    <div class="stat"><span class="n" id="statRadius"></span><span class="l">radius range</span></div>
    <div class="stat"><span class="n" id="statCase"></span><span class="l">case</span></div>
  </div>
  <a id="reportLink" class="ghost" href="#">Full report &rarr;</a>
</div>

<div class="layout">
  <div class="viewer">
    <div class="segmented" role="tablist">
      <button data-view="3d" role="tab" aria-selected="true">3D view</button>
      <button data-view="clock" role="tab" aria-selected="false">Clock map</button>
      <button data-view="check" role="tab" aria-selected="false">Projections</button>
    </div>
    <div class="frame" id="frame"></div>
    <p class="caption" id="caption"></p>
  </div>

  <aside class="panel">
    <div class="panel-head">
      <h3 id="branchCount">Detected branches</h3>
      <div class="panel-actions">
        <div class="segmented mini" role="tablist">
          <button type="button" data-panel="branches" aria-selected="true">Branches</button>
          <button type="button" data-panel="casejson" aria-selected="false">Case JSON</button>
        </div>
        <button type="button" class="copybtn" id="copyCase">Copy case JSON</button>
      </div>
    </div>
    <div id="branches"></div>
    <div id="caseJson" hidden>
      <div class="jsonwrap casejson">
        <pre id="caseJsonPre"></pre>
      </div>
    </div>
  </aside>
</div>

<script>{script}</script>
</body></html>
"""


def main() -> None:
    os.makedirs(VC_DIR, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(build())
    print(OUT_PATH)


if __name__ == "__main__":
    main()
