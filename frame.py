"""Centreline, aortic frame, mm -> (height, clock).

D8 as built: geodesic centreline with a per-level anatomical clock frame.

1. Endpoints. The two mask voxels furthest apart along the mask (double geodesic sweep with
   skimage.graph.MCP_Geometric through the mask at unit cost). They sit on the rim of the two
   end faces, which is fine as a start: the face centroid replaces them in step 3.
2. Path. Minimum-cost path between them with cost = (inside distance)^-CENTRELINE_COST_POWER, so
   the path runs along the axis instead of cutting the inside of a bend.
3. End faces. The first stretch of the path at each end runs from the rim to the axis. It is cut
   where the path becomes central (inside distance at least CENTRELINE_CORE_FRACTION x the path
   median) and replaced by the centroid of the end face: boundary voxels beyond the last central
   point, within END_FACE_LATERAL_FACTOR x the local radius of the extended axis, whose outward
   normal (minus the gradient of the inside distance) is within END_FACE_NORMAL_DEG of the outward
   tangent. A lateral-wall voxel's normal is near 90 deg from the tangent, so the face is picked
   out whether the cut is flat, oblique (subject 3) or a tapered point (subject 20).
4. Resample at CENTRELINE_STEP_MM, Gaussian-smooth along the arc (CENTRELINE_SMOOTH_MM, endpoints
   pinned), orient superior first (larger z in LPS).
5. Frame. Tangents by central differences, pointing inferior. 12 o'clock at each point is the
   patient's anterior (-y in LPS) projected into the plane perpendicular to the tangent: the
   surgeon's convention at every level, which a tilted aorta needs (a naive axial angle is wrong
   by the tilt) and which never drifts from the anatomy. The first draft transported the anterior
   vector from the inferior end with the double-reflection rotation-minimising frame (Wang et al.
   2008); the two agree on a planar curve, but on subject 3 (an S-bend with a forward bow) the
   transported vector ends 43 deg from the patient's anterior and would call a posterior lumbar
   4:30 o'clock. The transported frame is still computed (`clock12_rmf`), its maximum deviation
   is logged as `twist_deg`, and it is used only where the tangent runs antero-posteriorly and
   the projection vanishes (CLOCK_ANTERIOR_MIN_SIN; the arch on subject 25, out of scope).
   3 o'clock = 12 o'clock x tangent, which is patient's left (+x) on a vertical aorta.

Fallback: per-slice mask centroids (the first-draft method) if the geodesic step fails, so the
engine always has endpoints for D6 rule 1.

Clock convention (D8): 12 = anterior, 3 = patient's left, 6 = posterior, 9 = patient's right.
Height = arc length from the superior end.

Output contract (SPEC.md D9): centreline points (mm), per-point frame, function mm -> (height, clock).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage
from skimage.graph import MCP_Geometric

import config
import io_utils
from candidates import Candidates
from ostium import mask_boundary_voxels

log = logging.getLogger("branchseed.frame")

ANTERIOR_LPS = np.array([0.0, -1.0, 0.0])   # 12 o'clock on the patient: -y in the LPS frame
INFERIOR_LPS = np.array([0.0, 0.0, -1.0])   # default tangent for a degenerate (single-point) centreline


@dataclass
class Frame:
    centreline_mm: np.ndarray      # (k, 3) xyz mm, superior -> inferior, CENTRELINE_STEP_MM apart
    centreline_idx: np.ndarray     # (k, 3) zyx working-grid indices (float)
    tangents: np.ndarray           # (k, 3) unit xyz, pointing inferior
    arc_mm: np.ndarray             # (k,) arc length from the superior end
    length_mm: float
    endpoints_mm: np.ndarray       # (2, 3): [superior, inferior]
    endpoint_idx_zyx: np.ndarray   # (2, 3)
    tortuosity: float              # path length / chord
    method: str = "geodesic"
    clock12: np.ndarray = None     # (k, 3) unit xyz: 12 o'clock, the patient's anterior projected perpendicular to the tangent
    clock3: np.ndarray = None      # (k, 3) unit xyz: 3 o'clock (patient's left) = clock12 x tangent
    clock12_rmf: np.ndarray = None  # (k, 3) the rotation-minimising transport of anterior from the inferior end (diagnostic)
    radius_mm: np.ndarray = None   # (k,) inside distance at each centreline point: the local aortic radius
    twist_deg: float = 0.0         # max angle between the transported (RMF) anterior and the projected anterior: zero on a planar curve
    info: dict = field(default_factory=dict)

    def end_faces(self) -> list:
        """[(endpoint_mm, outward unit tangent)] for the superior and inferior ends. D2/D6 rule 1:
        the end faces are the mask boundary regions nearest these two points. The outward tangent
        is the chord over the first aortic radius of centreline (median radius), which is steadier
        than a one-sided difference at a pinned endpoint."""
        if len(self.centreline_mm) < 2:
            return [(self.endpoints_mm[0], -INFERIOR_LPS), (self.endpoints_mm[1], INFERIOR_LPS)]
        reach = max(float(np.median(self.radius_mm)) if self.radius_mm is not None and len(self.radius_mm) else 0.0,
                    3 * config.CENTRELINE_STEP_MM)
        reach = min(reach, self.length_mm / 2)
        j_top = int(np.searchsorted(self.arc_mm, reach))
        j_bot = int(np.searchsorted(self.arc_mm, self.length_mm - reach)) - 1
        j_top = min(max(j_top, 1), len(self.centreline_mm) - 1)
        j_bot = max(min(j_bot, len(self.centreline_mm) - 2), 0)
        t_top = _unit(self.centreline_mm[0] - self.centreline_mm[j_top])
        t_bot = _unit(self.centreline_mm[-1] - self.centreline_mm[j_bot])
        if np.linalg.norm(t_top) == 0:
            t_top = -self.tangents[0]
        if np.linalg.norm(t_bot) == 0:
            t_bot = self.tangents[-1]
        return [(self.endpoints_mm[0], t_top), (self.endpoints_mm[1], t_bot)]

    def nearest(self, xyz_mm) -> int:
        d = np.linalg.norm(self.centreline_mm - np.asarray(xyz_mm, float)[None, :], axis=1)
        return int(np.argmin(d))

    def project(self, xyz_mm) -> tuple:
        """Foot of the perpendicular from a point onto the polyline.
        Returns (segment index j, fraction u along it, foot point mm)."""
        P = self.centreline_mm
        p = np.asarray(xyz_mm, float)
        if len(P) < 2:
            return 0, 0.0, P[0].copy()
        seg = P[1:] - P[:-1]
        L2 = np.maximum((seg ** 2).sum(axis=1), 1e-12)
        u = np.clip(((p[None, :] - P[:-1]) * seg).sum(axis=1) / L2, 0.0, 1.0)
        foot = P[:-1] + u[:, None] * seg
        j = int(np.argmin(np.linalg.norm(foot - p[None, :], axis=1)))
        return j, float(u[j]), foot[j]

    def frame_at(self, j: int, u: float) -> tuple:
        """(tangent, clock12, clock3) interpolated at fraction u along segment j."""
        if len(self.centreline_mm) < 2:
            return self.tangents[0], self.clock12[0], self.clock3[0]
        t = _unit((1 - u) * self.tangents[j] + u * self.tangents[j + 1])
        e12 = (1 - u) * self.clock12[j] + u * self.clock12[j + 1]
        e12 = _unit(e12 - np.dot(e12, t) * t)
        return t, e12, np.cross(e12, t)

    def height_clock(self, xyz_mm) -> tuple[float, float]:
        """(height mm from the superior end, clock hours in [0, 12)) of a physical point.
        Clock angle measured in the plane perpendicular to the local tangent, 12 = patient anterior."""
        p = np.asarray(xyz_mm, float)
        j, u, foot = self.project(p)
        seglen = float(np.linalg.norm(self.centreline_mm[j + 1] - self.centreline_mm[j])) if len(self.centreline_mm) > 1 else 0.0
        height = float(self.arc_mm[j] + u * seglen)
        _, e12, e3 = self.frame_at(j, u)
        v = p - foot
        angle = np.arctan2(np.dot(v, e3), np.dot(v, e12))   # 0 at 12 o'clock, +pi/2 at 3 o'clock
        return height, float((np.degrees(angle) / 30.0) % 12.0)

    def radius_at(self, height_mm: float) -> float:
        """Local aortic radius (inside distance) at a height along the centreline."""
        if self.radius_mm is None or len(self.arc_mm) == 0:
            return float("nan")
        return float(np.interp(height_mm, self.arc_mm, self.radius_mm))


# ------------------------------------------------------------------ helpers


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def inside_distance(mask: np.ndarray, spacing: np.ndarray) -> np.ndarray:
    """Distance from each mask voxel to the nearest non-mask voxel, in mm. The array is padded
    with one empty layer first so a mask cut by the image boundary (subjects 4 to 12 are cropped
    to the mask's z extent) still has an end face at distance one voxel, not a face the transform
    cannot see."""
    padded = np.pad(mask, 1)
    d = ndimage.distance_transform_edt(padded, sampling=spacing)
    return d[1:-1, 1:-1, 1:-1].astype(np.float32)


def _farthest(cost: np.ndarray, mask: np.ndarray, spacing: np.ndarray, start_zyx) -> np.ndarray:
    """Mask voxel geodesically farthest from `start_zyx` through the mask."""
    mcp = MCP_Geometric(cost, sampling=tuple(float(s) for s in spacing))
    costs, _ = mcp.find_costs([tuple(int(v) for v in start_zyx)])
    reach = np.where(mask & np.isfinite(costs), costs, -1.0)
    return np.array(np.unravel_index(int(np.argmax(reach)), mask.shape))


def geodesic_path(mask: np.ndarray, din: np.ndarray, spacing: np.ndarray) -> np.ndarray:
    """(k, 3) int zyx voxel chain from one end of the mask to the other along the axis."""
    f = int(config.CENTRELINE_SWEEP_DOWNSAMPLE)
    m_low = mask[::f, ::f, ::f]
    if m_low.sum() < 2:
        f, m_low = 1, mask
    unit_cost = np.where(m_low, 1.0, np.inf)
    idx = np.argwhere(m_low)
    start = idx[int(np.argmax(idx[:, 0]))]                  # any voxel; the top slice is as good as another
    a = _farthest(unit_cost, m_low, spacing * f, start)
    b = _farthest(unit_cost, m_low, spacing * f, a)
    a, b = _snap_to_mask(mask, a * f), _snap_to_mask(mask, b * f)
    central_cost = np.where(mask, (float(din.max()) / np.maximum(din, 1e-6)) ** config.CENTRELINE_COST_POWER, np.inf)
    mcp = MCP_Geometric(central_cost, sampling=tuple(float(s) for s in spacing))
    mcp.find_costs([tuple(int(v) for v in a)], [tuple(int(v) for v in b)])
    return np.array(mcp.traceback(tuple(int(v) for v in b)), dtype=int)


def _snap_to_mask(mask: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Nearest mask voxel to an index (the downsampled sweep lands between working voxels)."""
    idx = np.minimum(idx, np.array(mask.shape) - 1)
    if mask[tuple(idx)]:
        return idx
    lo = np.maximum(idx - 3, 0)
    hi = np.minimum(idx + 4, np.array(mask.shape))
    box = tuple(slice(int(l), int(h)) for l, h in zip(lo, hi))
    local = np.argwhere(mask[box])
    if len(local) == 0:
        local = np.argwhere(mask)
        return local[int(np.argmin(np.linalg.norm(local - idx, axis=1)))]
    return lo + local[int(np.argmin(np.linalg.norm(local + lo - idx, axis=1)))]


def _boundary_normals(din: np.ndarray, bidx: np.ndarray, spacing: np.ndarray) -> np.ndarray:
    """Unit outward normals at boundary voxels: minus the central-difference gradient of the
    inside distance (index space scaled to mm)."""
    g = np.zeros((len(bidx), 3))
    shape = np.array(din.shape)
    for a in range(3):
        lo, hi = bidx.copy(), bidx.copy()
        lo[:, a] = np.maximum(bidx[:, a] - 1, 0)
        hi[:, a] = np.minimum(bidx[:, a] + 1, shape[a] - 1)
        span = (hi[:, a] - lo[:, a]) * spacing[a]
        g[:, a] = np.where(span > 0, (din[tuple(hi.T)] - din[tuple(lo.T)]) / np.maximum(span, 1e-9), 0.0)
    n = -g
    norms = np.linalg.norm(n, axis=1, keepdims=True)
    return np.where(norms > 0, n / np.maximum(norms, 1e-12), 0.0)


def end_face_centroid(bmm: np.ndarray, bnorm: np.ndarray, c_mm: np.ndarray, t_out: np.ndarray, radius_mm: float):
    """Centroid (mm) of the end-face voxels beyond the last central point `c_mm` in the outward
    direction `t_out`, or None if no voxel qualifies. Returns (centroid, count)."""
    d = bmm - c_mm[None, :]
    along = d @ t_out
    lateral = np.linalg.norm(d - along[:, None] * t_out[None, :], axis=1)
    cosang = bnorm @ t_out
    sel = (along > 0) & (lateral <= config.END_FACE_LATERAL_FACTOR * radius_mm) & \
          (cosang >= np.cos(np.radians(config.END_FACE_NORMAL_DEG)))
    if not sel.any():
        return None, 0
    return bmm[sel].mean(axis=0), int(sel.sum())


def _resample(pts: np.ndarray, step: float) -> np.ndarray:
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    keep = np.concatenate([[True], seg > 1e-9])           # drop repeated points
    pts = pts[keep]
    if len(pts) < 2:
        return pts
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    s = np.arange(0.0, arc[-1], step)
    if arc[-1] - s[-1] > 1e-9:
        s = np.append(s, arc[-1])
    return np.column_stack([np.interp(s, arc, pts[:, a]) for a in range(3)])


def smooth_polyline(pts_mm: np.ndarray, step: float = config.CENTRELINE_STEP_MM,
                    sigma_mm: float = config.CENTRELINE_SMOOTH_MM) -> np.ndarray:
    """Resample at `step`, Gaussian-smooth along the arc with the endpoints pinned, resample again."""
    pts = _resample(np.asarray(pts_mm, float), step)
    if len(pts) < 3:
        return pts
    sm = ndimage.gaussian_filter1d(pts, sigma=sigma_mm / step, axis=0, mode="nearest")
    sm[0], sm[-1] = pts[0], pts[-1]
    return _resample(sm, step)


def rotation_minimising_frame(pts: np.ndarray, tang: np.ndarray, r0: np.ndarray) -> np.ndarray:
    """Double-reflection transport of the reference vector r0 (given at pts[0]) along the
    polyline. Returns (k, 3) unit vectors perpendicular to the tangents."""
    R = np.zeros_like(pts)
    r = _unit(r0 - np.dot(r0, tang[0]) * tang[0])
    R[0] = r
    for i in range(len(pts) - 1):
        v1 = pts[i + 1] - pts[i]
        c1 = float(v1 @ v1)
        if c1 < 1e-12:
            R[i + 1] = r
            continue
        rL = r - (2.0 / c1) * float(v1 @ r) * v1
        tL = tang[i] - (2.0 / c1) * float(v1 @ tang[i]) * v1
        v2 = tang[i + 1] - tL
        c2 = float(v2 @ v2)
        r = rL if c2 < 1e-12 else rL - (2.0 / c2) * float(v2 @ rL) * v2
        r = _unit(r - np.dot(r, tang[i + 1]) * tang[i + 1])
        R[i + 1] = r
    return R


# ------------------------------------------------------------------ centreline candidates


def slice_centroid_polyline(cand: Candidates) -> np.ndarray:
    """First-draft centreline: per-slice mask centroids (fallback only)."""
    m = cand.mask
    zs = np.where(m.any(axis=(1, 2)))[0]
    cents = np.array([ndimage.center_of_mass(m[z]) for z in zs], dtype=float).reshape(-1, 2)
    idx = np.column_stack([zs.astype(float), cents])
    return io_utils.index_to_mm(cand.image, idx)


def geodesic_polyline(cand: Candidates, din: np.ndarray, info: dict) -> np.ndarray:
    """Steps 1 to 3 of the module docstring: rim-to-rim geodesic path, cut to its central part and
    capped with the two end-face centroids. Returns (k, 3) mm points, unordered in z.

    The rim-to-axis stretch at each end is about one aortic radius long (the path travels at most
    a radius laterally while it converges), so the core is cut one radius (the median inside
    distance along the path) in from where it first becomes central, and the end tangent is the
    chord between one and three radii. Joining the face centroid to the cut point then gives a
    segment at least a radius long that runs along the axis instead of a kink."""
    sp = cand.spacing
    path = geodesic_path(cand.mask, din, sp)
    d_path = din[tuple(path.T)]
    med = float(np.median(d_path))
    central = np.where(d_path >= config.CENTRELINE_CORE_FRACTION * med)[0]
    if len(central) == 0:
        central = np.array([0, len(path) - 1])
    k0, k1 = int(central[0]), int(central[-1])
    core = path[k0:k1 + 1]
    core_mm = io_utils.index_to_mm(cand.image, core.astype(float))
    info["path_voxels"], info["core_start"], info["core_end"] = int(len(path)), k0, int(len(path) - 1 - k1)
    if len(core_mm) < 3:
        return core_mm

    bidx = mask_boundary_voxels(cand.mask)
    bmm = io_utils.index_to_mm(cand.image, bidx.astype(float))
    bnorm_idx = _boundary_normals(din, bidx, sp)
    # index-space normals to mm: the working image direction matrix (row-major xyz) applied to the xyz vector
    D = np.array(cand.image.GetDirection()).reshape(3, 3)
    bnorm = (D @ bnorm_idx[:, ::-1].T).T

    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(core_mm, axis=0), axis=1))])
    total = float(arc[-1])
    ends, cuts, face_counts = [], [], []
    for which in (0, 1):
        s = arc if which == 0 else total - arc                 # arc distance from this end
        s1 = min(med, total / 4)
        s3 = min(3 * med, total / 2)
        inside1 = np.where(s >= s1)[0]
        inside3 = np.where(s >= s3)[0]
        if which == 0:
            i1 = int(inside1[0]) if len(inside1) else 0
            i3 = int(inside3[0]) if len(inside3) else len(core_mm) - 1
        else:
            i1 = int(inside1[-1]) if len(inside1) else len(core_mm) - 1
            i3 = int(inside3[-1]) if len(inside3) else 0
        if i1 == i3:
            i3 = min(i1 + 1, len(core_mm) - 1) if which == 0 else max(i1 - 1, 0)
        t_out = _unit(core_mm[i1] - core_mm[i3])
        if np.linalg.norm(t_out) == 0:
            t_out = -INFERIOR_LPS if which == 0 else INFERIOR_LPS
        c = core_mm[i1]
        r_local = float(d_path[k0 + i1])
        e, n_face = end_face_centroid(bmm, bnorm, c, t_out, max(r_local, med))
        if e is None:                                          # no face found: the rim voxel the sweep gave us
            e = io_utils.index_to_mm(cand.image, path[0 if which == 0 else -1].astype(float))
        ends.append(e)
        cuts.append(i1)
        face_counts.append(n_face)
    info["end_face_voxels"] = face_counts
    info["cut_mm"] = [round(float(arc[cuts[0]]), 1), round(float(total - arc[cuts[1]]), 1)]
    if cuts[1] < cuts[0]:                                      # a segment shorter than two radii: keep the middle point
        cuts = [len(core_mm) // 2, len(core_mm) // 2]
    return np.vstack([ends[0][None, :], core_mm[cuts[0]:cuts[1] + 1], ends[1][None, :]])


# ------------------------------------------------------------------ build


def _finish(cand: Candidates, pts_mm: np.ndarray, din: np.ndarray, method: str, info: dict) -> Frame:
    pts = smooth_polyline(pts_mm)
    if len(pts) > 1 and pts[0, 2] < pts[-1, 2]:        # superior (+z in LPS) first
        pts = pts[::-1].copy()
    idx = io_utils.mm_to_index(cand.image, pts)
    k = len(pts)
    if k > 1:
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        arc = np.concatenate([[0.0], np.cumsum(seg)])
        tang = np.gradient(pts, axis=0)
        norms = np.linalg.norm(tang, axis=1, keepdims=True)
        tang = np.where(norms > 0, tang / np.maximum(norms, 1e-12), INFERIOR_LPS[None, :])
        chord = float(np.linalg.norm(pts[-1] - pts[0]))
        # 12 o'clock = patient anterior projected into the perpendicular plane (D8 as built); the
        # rotation-minimising transport from the inferior end is kept as the twist diagnostic and
        # takes over only where the tangent runs antero-posteriorly and the projection vanishes
        e12_rmf = rotation_minimising_frame(pts[::-1], tang[::-1], ANTERIOR_LPS)[::-1]
        proj = ANTERIOR_LPS[None, :] - (tang @ ANTERIOR_LPS)[:, None] * tang
        pn = np.linalg.norm(proj, axis=1)
        ok = pn >= config.CLOCK_ANTERIOR_MIN_SIN
        e12 = np.where(ok[:, None], proj / np.maximum(pn, 1e-12)[:, None], e12_rmf)
        twist = float(np.degrees(np.arccos(np.clip((e12_rmf[ok] * e12[ok]).sum(axis=1), -1, 1))).max()) if ok.any() else 0.0
    else:
        arc = np.zeros(k)
        tang = np.tile(INFERIOR_LPS[None, :], (k, 1))
        chord = 0.0
        e12 = e12_rmf = np.tile(ANTERIOR_LPS[None, :], (k, 1))
        twist = 0.0
    e3 = np.cross(e12, tang)
    length = float(arc[-1]) if k else 0.0
    ci = np.clip(idx, 0, np.array(din.shape) - 1)
    radius = ndimage.map_coordinates(din, ci.T, order=1, mode="nearest") if k else np.zeros(0)
    fr = Frame(centreline_mm=pts, centreline_idx=idx, tangents=tang, arc_mm=arc, length_mm=length,
               endpoints_mm=np.array([pts[0], pts[-1]]), endpoint_idx_zyx=np.array([idx[0], idx[-1]]),
               tortuosity=float(length / chord) if chord > 0 else 1.0, method=method,
               clock12=e12, clock3=e3, clock12_rmf=e12_rmf, radius_mm=np.asarray(radius, float), twist_deg=twist, info=info)
    log.info("centreline (%s): %d points, %.1f mm, tortuosity %.3f, twist %.1f deg, radius %.1f to %.1f mm",
             method, k, length, fr.tortuosity, twist, float(radius.min()) if k else 0.0, float(radius.max()) if k else 0.0)
    return fr


def build(cand: Candidates) -> Frame:
    din = inside_distance(cand.mask, cand.spacing)
    info = {}
    try:
        pts = geodesic_polyline(cand, din, info)
        method = "geodesic"
    except Exception as e:  # noqa: BLE001 - the engine needs endpoints whatever happens
        log.warning("geodesic centreline failed (%s: %s); falling back to slice centroids", type(e).__name__, e)
        pts = slice_centroid_polyline(cand)
        method = "slice_centroids"
        info["fallback_error"] = f"{type(e).__name__}: {e}"
    return _finish(cand, pts, din, method, info)


# ------------------------------------------------------------------ spacing


def spacing_along(heights: dict, length_mm: float) -> list:
    """Inter-branch spacing along the centreline (D8: the room to land a stent).
    heights: {branch_id: height_mm}. Returns [{'from', 'to', 'gap_mm'}] from the superior end,
    through the branches in height order, to the inferior end."""
    order = sorted(heights.items(), key=lambda kv: kv[1])
    stops = [("superior end", 0.0)] + order + [("inferior end", float(length_mm))]
    return [{"from": a, "to": b, "gap_mm": round(float(hb - ha), 1)}
            for (a, ha), (b, hb) in zip(stops[:-1], stops[1:])]
