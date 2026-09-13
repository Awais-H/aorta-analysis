"""Visualisation: one clinician-facing display and the mandated verification
figures.

The clinician view is the unrolled wall map. Circumferential clock position and
longitudinal distance from a reference are the two numbers that go on a
fenestrated stent-graft order form, and the elegant part is that this *is* the
working representation the detector ran on rather than a separate rendering -
the reader sees the same picture the algorithm saw.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")                       # no GUI, no server, no GPU

import matplotlib.pyplot as plt             # noqa: E402
import numpy as np                          # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402
from skimage import measure as skmeasure    # noqa: E402

log = logging.getLogger(__name__)

_CLOCK_TICKS = np.arange(0, 12)


def _superior_first(geometry, grid) -> bool:
    """True when centreline sample 0 is the superior end of the segment.

    The centreline runs between the two skeleton extremes and its direction is
    arbitrary, so orient the display rather than assuming.
    """
    z_start = grid.to_physical(geometry.centreline[0])[2]
    z_end = grid.to_physical(geometry.centreline[-1])[2]
    return bool(z_start >= z_end)


def _clock_axis(ax, theta, left_is_positive: bool) -> None:
    ax.set_xlabel("clock position (12 = anterior, local vessel frame)")
    positions = _CLOCK_TICKS / 12.0 * 360.0
    labels = [f"{12 if h == 0 else h}" for h in _CLOCK_TICKS]
    if not left_is_positive:
        positions = (360.0 - positions) % 360.0
    order = np.argsort(positions)
    ax.set_xticks(positions[order])
    ax.set_xticklabels([labels[i] for i in order])


def wall_map_figure(result, path: Path, cfg) -> None:
    """The clinician display: arc length against clock position."""
    geometry, wall_map, grid = result.geometry, result.wall_map, result.grid
    if geometry is None or wall_map is None:
        return

    arc = geometry.arc_length
    superior_first = _superior_first(geometry, grid)
    # The centreline runs between the two skeleton extremes in an arbitrary
    # direction. Flip the whole map, not just the axis, so that row 0 of what
    # is drawn is always the superior end of the supplied segment.
    intensity = np.ma.masked_invalid(
        wall_map.intensity if superior_first else wall_map.intensity[::-1]
    )
    y = arc if superior_first else (arc[-1] - arc)[::-1]

    mid = geometry.n_samples // 2
    left_is_positive = bool(grid.rotate_to_physical(geometry.frame_v[mid])[0] >= 0.0)

    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("0.82")                     # invalid pixels: flat light grey

    fig, ax = plt.subplots(figsize=(10, 11))
    # Scale from this patient's own background to this patient's own lumen, so
    # the display means the same thing on every case and the detection
    # threshold sits at a fixed place on the bar.
    calibration = result.calibration
    vmin, vmax = float(calibration.t_bg), float(calibration.mu_ao)
    image = ax.imshow(
        intensity, aspect="auto", origin="upper", cmap=cmap, vmin=vmin, vmax=vmax,
        extent=(0.0, 360.0, float(y.max()), float(y.min())),
    )
    bar = fig.colorbar(image, ax=ax, label="mean peri-aortic HU, 1-6 mm outward")
    bar.ax.axhline(calibration.t_lumen, color="#39d3ff", linewidth=1.4)
    bar.ax.text(1.6, calibration.t_lumen, " detection\n threshold", va="center",
                fontsize=7, color="#1c7ea0", transform=bar.ax.get_yaxis_transform())

    _mark_endcap_bands(ax, wall_map, y, superior_first)

    for candidate in result.candidates:
        _mark_candidate(ax, candidate, wall_map, geometry, y, grid, superior_first, False)
    for candidate in result.accepted:
        _mark_candidate(ax, candidate, wall_map, geometry, y, grid, superior_first, True)

    _clock_axis(ax, wall_map.theta, left_is_positive)
    ax.set_ylabel("distance along the supplied aorta (mm from its superior end)")
    flags = result.payload.get("meta", {}).get("case_flags", [])
    title = "%s - unrolled aortic wall map, %d daughters" % (
        result.payload.get("case_id", "case"), len(result.payload.get("daughters", []))
    )
    if flags:
        title += "\nflags: " + ", ".join(flags)
    ax.set_title(title, fontsize=11)
    ax.text(
        0.5, -0.075,
        "Clock position is defined in the local vessel frame, which diverges from the "
        "patient axial plane in the arch.\nGrey pixels are invalid: cropped end caps, "
        "rays crossing to another part of the vessel, or calcium.",
        transform=ax.transAxes, ha="center", va="top", fontsize=8, color="0.3",
    )
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        seen, uniq_h, uniq_l = set(), [], []
        for handle, label in zip(handles, labels):
            if label not in seen:
                seen.add(label)
                uniq_h.append(handle)
                uniq_l.append(label)
        ax.legend(uniq_h, uniq_l, loc="upper right", fontsize=8, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(path, dpi=cfg.viz.dpi, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", path)


def _mark_endcap_bands(ax, wall_map, y, superior_first: bool) -> None:
    """Label the invalid bands at the two ends, where the cut faces are."""
    valid = wall_map.valid if superior_first else wall_map.valid[::-1]
    invalid_fraction = 1.0 - valid.mean(axis=1)
    band = invalid_fraction > 0.5
    if not band.any():
        return
    rows = np.nonzero(band)[0]
    for group in np.split(rows, np.nonzero(np.diff(rows) > 1)[0] + 1):
        if group.size == 0:
            continue
        ax.text(5.0, float(y[group].mean()), "cropped end - not a branch origin",
                fontsize=7, color="0.25", va="center")


def _mark_candidate(ax, candidate, wall_map, geometry, y, grid, superior_first, accepted) -> None:
    row, col = candidate.peak_ij
    n_bins = wall_map.intensity.shape[1]
    n_rows = wall_map.intensity.shape[0]
    display_row = row if superior_first else n_rows - 1 - row
    x = 360.0 * col / n_bins
    yy = float(y[display_row])

    if not accepted:
        if candidate.rejected_by is None:
            return
        ax.plot(x, yy, marker="o", markersize=7, markerfacecolor="none",
                markeredgecolor="0.45", markeredgewidth=1.0, linestyle="none",
                label=f"rejected: {candidate.rejected_by}")
        return

    # Marker size in points, not data units: the two axes are degrees and
    # millimetres, so a data-space circle would render as an ellipse.
    radius = candidate.radius_mm or 1.0
    ax.plot(x, yy, marker="o", markersize=6.0 * radius, markerfacecolor="none",
            markeredgecolor="#39d3ff", markeredgewidth=1.8, linestyle="none",
            label="detected ostium (disc area proportional to radius)", zorder=5)
    ax.plot(x, yy, marker="+", color="#39d3ff", markersize=6, linestyle="none", zorder=6)

    # Direction, projected into the map plane: the tangential component moves
    # along the arc axis, the circumferential component around the clock.
    direction_mm = grid.to_mm_direction(candidate.direction)
    theta = wall_map.theta[col]
    circumferential = -np.sin(theta) * geometry.frame_u[row] + np.cos(theta) * geometry.frame_v[row]
    d_arc = float(direction_mm @ geometry.tangent[row])
    d_circ = float(direction_mm @ circumferential)
    scale_deg, scale_mm = 22.0, 9.0
    sign = 1.0 if superior_first else -1.0
    ax.annotate(
        "", xy=(x + d_circ * scale_deg, yy + sign * d_arc * scale_mm), xytext=(x, yy),
        arrowprops=dict(arrowstyle="->", color="#39d3ff", linewidth=1.6), zorder=6,
    )
    ax.text(x + 6.0, yy - 3.0, f"{candidate.instance_id.replace('branch_', 'b')} "
            f"{candidate.clock_position} r{candidate.radius_mm:.1f}",
            fontsize=7, color="#39d3ff", zorder=6)


# --------------------------------------------------------------------------- #
# Verification figures
# --------------------------------------------------------------------------- #

def render_3d(result, path: Path, cfg, stride: int = 2) -> None:
    """Mask surface with ostium markers and direction arrows, three views."""
    geometry, grid = result.geometry, result.grid
    if geometry is None:
        return
    try:
        verts, faces, _, _ = skmeasure.marching_cubes(
            geometry.mask.astype(np.float32), level=0.5, step_size=stride
        )
    except (RuntimeError, ValueError):
        log.warning("marching cubes failed; skipping the 3D render")
        return

    verts_mm = grid.to_physical(verts[:, ::-1])          # zyx verts -> xyz -> mm
    ostia = np.array([grid.to_physical(c.ostium_xyz_roi) for c in result.accepted]) \
        if result.accepted else np.zeros((0, 3))
    directions = np.array([c.direction for c in result.accepted]) \
        if result.accepted else np.zeros((0, 3))

    views = [("anterior", 0, -90), ("lateral", 0, 0), ("oblique", 25, -55)]
    fig = plt.figure(figsize=(15, 6))
    for n, (name, elev, azim) in enumerate(views, start=1):
        ax = fig.add_subplot(1, 3, n, projection="3d")
        mesh = Poly3DCollection(verts_mm[faces], alpha=0.18, linewidths=0)
        mesh.set_facecolor("#c94f5c")
        ax.add_collection3d(mesh)
        if ostia.size:
            ax.scatter(ostia[:, 0], ostia[:, 1], ostia[:, 2], color="#0d47a1", s=28, depthshade=False)
            ax.quiver(ostia[:, 0], ostia[:, 1], ostia[:, 2],
                      directions[:, 0], directions[:, 1], directions[:, 2],
                      length=14.0, color="#0d47a1", linewidth=1.6, arrow_length_ratio=0.3)
        _equal_aspect(ax, verts_mm)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f"{name} view", fontsize=10)
        ax.set_xlabel("x (mm)", fontsize=7)
        ax.set_ylabel("y (mm)", fontsize=7)
        ax.set_zlabel("z (mm)", fontsize=7)
        ax.tick_params(labelsize=6)
    fig.suptitle(
        "%s - aorta mask, detected ostia and daughter directions (SimpleITK LPS mm)"
        % result.payload.get("case_id", "case")
    )
    fig.tight_layout()
    fig.savefig(path, dpi=cfg.viz.dpi, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", path)


def _equal_aspect(ax, points: np.ndarray, margin_mm: float = 12.0) -> None:
    """True millimetre proportions without padding a slim vessel into a cube."""
    lo = points.min(axis=0) - margin_mm
    hi = points.max(axis=0) + margin_mm
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(tuple(hi - lo))


def mip_panels(result, path: Path, cfg, slab_mm: float = 10.0) -> None:
    """Orthogonal MIP slabs through each ostium.

    The 3D render alone does not let a reader confirm the branch is actually
    there; this does.
    """
    if result.image is None or not result.accepted:
        return
    image, grid = result.image, result.grid
    half = np.maximum((0.5 * slab_mm / grid.roi_spacing_mm).astype(int), 1)
    window = int(round(30.0 / float(grid.roi_spacing_mm.min())))

    n = len(result.accepted)
    fig, axes = plt.subplots(n, 3, figsize=(10, 3.4 * n), squeeze=False)
    for r, candidate in enumerate(result.accepted):
        ostium = np.rint(candidate.ostium_xyz_roi).astype(int)
        seed = np.rint(candidate.seed_xyz_roi).astype(int)
        for c, (axis_name, array_axis) in enumerate(
            (("axial (array z)", 0), ("coronal (array y)", 1), ("sagittal (array x)", 2))
        ):
            centre = ostium[2 - array_axis]
            lo = max(centre - half[2 - array_axis], 0)
            hi = min(centre + half[2 - array_axis] + 1, image.shape[array_axis])
            slab = np.take(image, np.arange(lo, hi), axis=array_axis).max(axis=array_axis)

            # In-plane coordinates of the ostium and the seed for this view.
            plane_axes = [a for a in (0, 1, 2) if a != array_axis]      # array order
            oy, ox = [ostium[2 - a] for a in plane_axes]
            sy, sx = [seed[2 - a] for a in plane_axes]

            ax = axes[r][c]
            ax.imshow(slab, cmap="gray", origin="lower",
                      vmin=float(np.percentile(slab, 1)), vmax=float(np.percentile(slab, 99.7)))
            ax.plot([ox], [oy], marker="+", color="#39d3ff", markersize=11, markeredgewidth=1.6)
            ax.annotate("", xy=(sx, sy), xytext=(ox, oy),
                        arrowprops=dict(arrowstyle="->", color="#39d3ff", linewidth=1.4))
            ax.set_xlim(max(ox - window, 0), min(ox + window, slab.shape[1] - 1))
            ax.set_ylim(max(oy - window, 0), min(oy + window, slab.shape[0] - 1))
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(f"{axis_name} MIP, {slab_mm:.0f} mm slab", fontsize=9)
            if c == 0:
                ax.set_ylabel("%s  %s  r=%.1f mm" % (candidate.instance_id,
                                                     candidate.clock_position,
                                                     candidate.radius_mm or float("nan")),
                              fontsize=8)
    fig.suptitle("%s - orthogonal MIP slabs through each detected ostium"
                 % result.payload.get("case_id", "case"))
    fig.tight_layout()
    fig.savefig(path, dpi=cfg.viz.dpi, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", path)


def write_figures(result, directory: Path, cfg) -> List[Path]:
    """Best effort: a failure here must never take down a valid JSON result."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    case = result.payload.get("case_id", "case")
    written: List[Path] = []
    jobs = (
        (f"{case}_wallmap.png", wall_map_figure),
        (f"{case}_3d.png", render_3d),
        (f"{case}_mip.png", mip_panels),
    )
    for name, function in jobs:
        target = directory / name
        try:
            function(result, target, cfg)
            if target.exists():
                written.append(target)
        except Exception:                                  # noqa: BLE001
            log.exception("failed to write %s", target)
    return written
