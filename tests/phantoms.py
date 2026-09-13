"""Synthetic volumes with exact ground truth.

Lets the whole pipeline be exercised before any annotated data exists, and
separates algorithmic error from data messiness. Geometry is defined in
physical millimetres and rasterised through a deliberately non-identity
direction-cosine matrix, so any index/physical confusion shows up immediately
as a failed ground-truth comparison rather than as a plausible wrong number.

The variants map one-to-one onto the challenge's "Important cases" list.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
import SimpleITK as sitk

LUMEN_HU = 485.0
BACKGROUND_HU = -50.0
NOISE_HU = 30.0


@dataclass
class BranchSpec:
    """One side branch, positioned by arc length along the aorta and clock angle."""

    arc_mm: float
    clock_deg: float
    radius_mm: float
    length_mm: float                    # length beyond the aortic wall
    tilt_deg: float = 0.0               # tilt towards +arc direction
    # (distance from ostium, turn angle in deg, radius, length) for each child
    children: Sequence[Tuple[float, float, float, float]] = ()
    blob_at_mm: Optional[float] = None  # sphere centre distance from the ostium
    blob_radius_mm: float = 6.0
    eligible: bool = True               # what the detector *should* report


@dataclass
class PhantomSpec:
    aorta_radius_mm: float = 9.0
    aorta_length_mm: float = 180.0
    curvature: float = 0.0              # 0 = straight, 1.0 = a full arch
    branches: Sequence[BranchSpec] = field(default_factory=tuple)
    spacing_mm: float = 1.5
    noise_hu: float = NOISE_HU
    n_distractors: int = 3
    seed: int = 0
    margin_mm: float = 25.0
    rotate_grid: bool = True


@dataclass
class Phantom:
    image: sitk.Image
    mask: sitk.Image
    truth: List[dict]
    spec: PhantomSpec


def _rotation(rx: float, ry: float, rz: float) -> np.ndarray:
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    mx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    my = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    mz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return mz @ my @ mx


def _axis_curve(spec: PhantomSpec, n: int = 400) -> Tuple[np.ndarray, np.ndarray]:
    """Aortic axis in physical mm, plus unit tangents, sampled uniformly in arc."""
    s = np.linspace(0.0, spec.aorta_length_mm, n)
    if spec.curvature <= 0.0:
        pts = np.stack([np.zeros_like(s), np.zeros_like(s), s], axis=1)
    else:
        # A planar arch: bend through up to 180 degrees over the segment.
        total = np.pi * spec.curvature
        radius = spec.aorta_length_mm / max(total, 1e-6)
        ang = s / radius
        pts = np.stack([radius * (1.0 - np.cos(ang)), np.zeros_like(s), radius * np.sin(ang)], 1)
    tan = np.gradient(pts, axis=0)
    tan /= np.maximum(np.linalg.norm(tan, axis=1, keepdims=True), 1e-9)
    return pts, tan


def _frame_at(pts: np.ndarray, tan: np.ndarray, arc_mm: float, spec: PhantomSpec):
    """Point, tangent and a reference (u, v) pair at a given arc length."""
    s = np.linspace(0.0, spec.aorta_length_mm, pts.shape[0])
    idx = int(np.clip(np.searchsorted(s, arc_mm), 0, pts.shape[0] - 1))
    t = tan[idx]
    seed = np.array([0.0, -1.0, 0.0])           # anterior in LPS
    u = seed - np.dot(seed, t) * t
    u /= max(np.linalg.norm(u), 1e-9)
    v = np.cross(t, u)
    return pts[idx], t, u, v


def _segment_distance(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    denom = float(ab @ ab)
    if denom < 1e-12:
        return np.linalg.norm(points - a, axis=-1)
    t = np.clip(((points - a) @ ab) / denom, 0.0, 1.0)
    closest = a + t[..., None] * ab
    return np.linalg.norm(points - closest, axis=-1)


def _polyline_distance(points: np.ndarray, poly: np.ndarray) -> np.ndarray:
    best = np.full(points.shape[:-1], np.inf)
    for i in range(poly.shape[0] - 1):
        np.minimum(best, _segment_distance(points, poly[i], poly[i + 1]), out=best)
    return best


def _fill_fraction(distance: np.ndarray, radius: float, voxel_mm: float) -> np.ndarray:
    """Soft rasterisation: a partial-volume ramp one voxel wide at the edge."""
    return np.clip(0.5 + (radius - distance) / voxel_mm, 0.0, 1.0)


def build_phantom(spec: PhantomSpec) -> Phantom:
    rng = np.random.default_rng(spec.seed)
    axis_pts, axis_tan = _axis_curve(spec)

    # --- collect geometry in physical mm -------------------------------------
    tubes: List[Tuple[np.ndarray, float]] = [(axis_pts, spec.aorta_radius_mm)]
    spheres: List[Tuple[np.ndarray, float]] = []
    truth: List[dict] = []

    for br in spec.branches:
        centre, tangent, u, v = _frame_at(axis_pts, axis_tan, br.arc_mm, spec)
        ang = np.radians(br.clock_deg)
        radial = np.cos(ang) * u + np.sin(ang) * v
        tilt = np.radians(br.tilt_deg)
        direction = np.cos(tilt) * radial + np.sin(tilt) * tangent
        direction /= np.linalg.norm(direction)

        # Ostium: where the branch axis leaves the aortic surface. Found by
        # marching, so it stays correct for a curved parent and a tilted branch.
        march = centre + np.outer(np.arange(0.0, 60.0, 0.05), direction)
        dist = _polyline_distance(march, axis_pts)
        outside = np.nonzero(dist > spec.aorta_radius_mm)[0]
        cross = int(outside[0]) if outside.size else 0
        ostium = march[cross]

        end = ostium + direction * br.length_mm
        tubes.append((np.stack([centre, end]), br.radius_mm))

        for offset, turn_deg, child_r, child_len in br.children:
            base = ostium + direction * offset
            turn = np.radians(turn_deg)
            child_dir = np.cos(turn) * direction + np.sin(turn) * radial_orthogonal(direction, u, v)
            child_dir /= np.linalg.norm(child_dir)
            tubes.append((np.stack([base, base + child_dir * child_len]), child_r))

        if br.blob_at_mm is not None:
            spheres.append((ostium + direction * br.blob_at_mm, br.blob_radius_mm))

        truth.append(
            {
                "ostium_xyz_mm": ostium,
                "seed_xyz_mm": ostium + direction * 5.0,
                "direction_xyz": direction,
                "radius_mm": br.radius_mm,
                "eligible": br.eligible,
                "arc_mm": br.arc_mm,
                "clock_deg": br.clock_deg,
            }
        )

    # --- grid ----------------------------------------------------------------
    all_pts = np.concatenate([p for p, _ in tubes] + ([s[0][None] for s in spheres] or []))
    lo = all_pts.min(axis=0) - (spec.aorta_radius_mm + spec.margin_mm)
    hi = all_pts.max(axis=0) + (spec.aorta_radius_mm + spec.margin_mm)
    size_xyz = np.maximum(np.ceil((hi - lo) / spec.spacing_mm).astype(int) + 1, 8)

    direction = _rotation(np.radians(7.0), np.radians(-5.0), np.radians(11.0)) if spec.rotate_grid \
        else np.eye(3)
    spacing = np.array([spec.spacing_mm] * 3)
    origin = lo

    grid_idx = np.stack(
        np.meshgrid(
            np.arange(size_xyz[2]), np.arange(size_xyz[1]), np.arange(size_xyz[0]), indexing="ij"
        ),
        axis=-1,
    ).astype(np.float64)                       # (nz, ny, nx, 3) in (z, y, x)
    idx_xyz = grid_idx[..., ::-1]
    coords = (idx_xyz * spacing) @ direction.T + origin   # physical mm per voxel centre

    # --- rasterise -----------------------------------------------------------
    lumen_fill = np.zeros(coords.shape[:-1], dtype=np.float32)
    aorta_poly, aorta_radius = tubes[0]
    aorta_dist = _polyline_distance(coords, aorta_poly)
    aorta_fill = _fill_fraction(aorta_dist, aorta_radius, spec.spacing_mm)

    # Flat cut faces at both ends, both well inside the volume.
    start_p, start_t = aorta_poly[0], axis_tan[0]
    end_p, end_t = aorta_poly[-1], axis_tan[-1]
    inside_caps = (((coords - start_p) @ start_t) >= 0.0) & (((coords - end_p) @ end_t) <= 0.0)
    aorta_fill = np.where(inside_caps, aorta_fill, 0.0).astype(np.float32)
    np.maximum(lumen_fill, aorta_fill, out=lumen_fill)

    for poly, radius in tubes[1:]:
        d = _polyline_distance(coords, poly)
        np.maximum(lumen_fill, _fill_fraction(d, radius, spec.spacing_mm).astype(np.float32),
                   out=lumen_fill)
    for centre, radius in spheres:
        d = np.linalg.norm(coords - centre, axis=-1)
        np.maximum(lumen_fill, _fill_fraction(d, radius, spec.spacing_mm).astype(np.float32),
                   out=lumen_fill)

    # Bright distractors, deliberately not connected to the tube.
    for _ in range(spec.n_distractors):
        for _attempt in range(50):
            centre = origin + rng.random(3) * (hi - lo)
            if _polyline_distance(centre[None], aorta_poly)[0] > aorta_radius + 12.0:
                break
        radius = float(rng.uniform(3.0, 7.0))
        d = np.linalg.norm(coords - centre, axis=-1)
        np.maximum(lumen_fill, _fill_fraction(d, radius, spec.spacing_mm).astype(np.float32),
                   out=lumen_fill)

    image = BACKGROUND_HU * (1.0 - lumen_fill) + LUMEN_HU * lumen_fill
    image = image + rng.normal(0.0, spec.noise_hu, size=image.shape)
    mask = (aorta_fill > 0.5)

    img_sitk = sitk.GetImageFromArray(image.astype(np.float32))
    mask_sitk = sitk.GetImageFromArray(mask.astype(np.uint8))
    for img in (img_sitk, mask_sitk):
        img.SetSpacing([float(v) for v in spacing])
        img.SetOrigin([float(v) for v in origin])
        img.SetDirection([float(v) for v in direction.reshape(-1)])

    return Phantom(image=img_sitk, mask=mask_sitk, truth=truth, spec=spec)


def radial_orthogonal(direction: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """A unit vector perpendicular to ``direction``, used to turn child branches."""
    cand = u - float(u @ direction) * direction
    if np.linalg.norm(cand) < 1e-6:
        cand = v - float(v @ direction) * direction
    return cand / max(np.linalg.norm(cand), 1e-9)


# --------------------------------------------------------------------------- #
# Named variants: one per "Important case" in the challenge document.
# --------------------------------------------------------------------------- #

def standard(**kwargs) -> PhantomSpec:
    """Four well-separated branches at different levels and clock positions."""
    return PhantomSpec(
        branches=(
            BranchSpec(arc_mm=50.0, clock_deg=0.0, radius_mm=3.5, length_mm=14.0),
            BranchSpec(arc_mm=80.0, clock_deg=100.0, radius_mm=2.5, length_mm=14.0),
            BranchSpec(arc_mm=110.0, clock_deg=250.0, radius_mm=2.0, length_mm=12.0),
            BranchSpec(arc_mm=140.0, clock_deg=180.0, radius_mm=3.0, length_mm=14.0),
        ),
        **kwargs,
    )


def arched(**kwargs) -> PhantomSpec:
    """A curved parent, so the frame must survive a tangent sweeping through
    every global direction."""
    return PhantomSpec(
        curvature=0.9,
        aorta_length_mm=200.0,
        branches=(
            BranchSpec(arc_mm=60.0, clock_deg=0.0, radius_mm=3.0, length_mm=14.0),
            BranchSpec(arc_mm=120.0, clock_deg=90.0, radius_mm=3.0, length_mm=14.0),
        ),
        **kwargs,
    )


def nearby_pair(separation_mm: float = 8.0, **kwargs) -> PhantomSpec:
    """Two origins 8 mm apart: must stay two instances."""
    return PhantomSpec(
        branches=(
            BranchSpec(arc_mm=90.0, clock_deg=70.0, radius_mm=2.5, length_mm=14.0),
            BranchSpec(arc_mm=90.0 + separation_mm, clock_deg=290.0, radius_mm=2.5,
                       length_mm=14.0),
        ),
        **kwargs,
    )


def common_trunk(**kwargs) -> PhantomSpec:
    """One origin that divides 3 mm out: one instance, seed beyond the split."""
    return PhantomSpec(
        branches=(
            BranchSpec(
                arc_mm=90.0, clock_deg=0.0, radius_mm=3.0, length_mm=3.0,
                children=((3.0, 35.0, 2.4, 12.0), (3.0, -35.0, 2.4, 12.0)),
            ),
        ),
        **kwargs,
    )


def daughter_of_daughter(**kwargs) -> PhantomSpec:
    """A vessel arising from another daughter, 8 mm out: not a direct daughter."""
    return PhantomSpec(
        branches=(
            BranchSpec(
                arc_mm=90.0, clock_deg=0.0, radius_mm=3.5, length_mm=20.0,
                children=((8.0, 80.0, 2.0, 12.0),),
            ),
        ),
        **kwargs,
    )


def short_stub(**kwargs) -> PhantomSpec:
    """A bump that dies at 3 mm: below the eligibility rule, must be rejected."""
    return PhantomSpec(
        branches=(
            BranchSpec(arc_mm=90.0, clock_deg=0.0, radius_mm=2.5, length_mm=3.0, eligible=False),
        ),
        **kwargs,
    )


def leaking(**kwargs) -> PhantomSpec:
    """A stub opening into a large blob: the leak detector must discard it."""
    return PhantomSpec(
        branches=(
            BranchSpec(arc_mm=90.0, clock_deg=0.0, radius_mm=2.5, length_mm=6.0,
                       blob_at_mm=9.0, blob_radius_mm=8.0, eligible=False),
        ),
        **kwargs,
    )


def near_cut_face(**kwargs) -> PhantomSpec:
    """An origin 14 mm from the inferior cut face. End-cap exclusion and true
    detection compete here; the branch must still be found."""
    return PhantomSpec(
        branches=(
            BranchSpec(arc_mm=166.0, clock_deg=40.0, radius_mm=3.0, length_mm=14.0),
        ),
        **kwargs,
    )


def no_branches(**kwargs) -> PhantomSpec:
    """A bare tube: the output must be a valid, empty daughters list."""
    return PhantomSpec(branches=(), **kwargs)
