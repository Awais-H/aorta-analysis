import numpy as np

import frame as frame_mod


def test_centreline_contract(cand, phantom):
    fr = frame_mod.build(cand)
    k = len(fr.centreline_mm)
    assert fr.centreline_mm.shape == (k, 3) and fr.tangents.shape == (k, 3) and fr.arc_mm.shape == (k,)
    assert fr.endpoints_mm.shape == (2, 3)
    assert np.allclose(np.linalg.norm(fr.tangents, axis=1), 1.0)
    # superior first: z decreases along the line in LPS
    assert fr.centreline_mm[0, 2] > fr.centreline_mm[-1, 2]
    assert np.allclose(fr.endpoints_mm[0], fr.centreline_mm[0])
    # a straight vertical cylinder: length equals the z extent, tortuosity 1, centred on the axis
    z0, z1 = phantom["z_range"]
    assert abs(fr.length_mm - (z1 - z0)) < 1.0
    assert abs(fr.tortuosity - 1.0) < 1e-3
    cx, cy = phantom["centre_xy"]
    assert np.allclose(fr.centreline_mm[:, 0], cx, atol=0.5)
    assert np.allclose(fr.centreline_mm[:, 1], cy, atol=0.5)


def test_height_and_clock_convention(cand, phantom):
    fr = frame_mod.build(cand)
    cx, cy = phantom["centre_xy"]
    z_top = fr.centreline_mm[0, 2]
    z_mid = fr.centreline_mm[len(fr.centreline_mm) // 2, 2]
    h, c = fr.height_clock([cx, cy - 10.0, z_mid])  # anterior (-y)
    assert abs(c - 0.0) < 0.05 or abs(c - 12.0) < 0.05
    assert abs(h - (z_top - z_mid)) < 1.0
    _, c = fr.height_clock([cx + 10.0, cy, z_mid])  # patient's left (+x)
    assert abs(c - 3.0) < 0.05
    _, c = fr.height_clock([cx, cy + 10.0, z_mid])  # posterior
    assert abs(c - 6.0) < 0.05
    _, c = fr.height_clock([cx - 10.0, cy, z_mid])  # patient's right
    assert abs(c - 9.0) < 0.05
    h, _ = fr.height_clock([cx, cy - 10.0, z_top])
    assert abs(h) < 1e-6


def test_phantom_branch_is_at_three_oclock(cand, phantom):
    fr = frame_mod.build(cand)
    _, c = fr.height_clock(phantom["ostium_mm"])
    assert abs(c - 3.0) < 0.1


# ------------------------------------------------------------------ geodesic centreline and clock frame
# Synthetic aortas built straight into a Candidates object: a bent tube (circular arc in the
# sagittal plane, bowing anteriorly like subject 2), a straight tube tilted 30 deg from the
# z axis and cut by axial planes (oblique end faces, subject 3's situation), and a vertical tube
# cut by the image boundary (subjects 4 to 12: no empty slice beyond the mask).

import SimpleITK as sitk
from candidates import Candidates


def _tube_candidates(axis_fn, s_range, radius=8.0, spacing=1.0, shape_xyz=(60, 100, 105), origin=(-25.0, -50.0, 0.0)):
    """Candidates whose mask is a tube of `radius` around the curve axis_fn(s), s in s_range (mm)."""
    sx = sy = sz = spacing
    nx, ny, nz = shape_xyz
    ox, oy, oz = origin
    Z, Y, X = np.meshgrid(oz + sz * np.arange(nz), oy + sy * np.arange(ny), ox + sx * np.arange(nx), indexing="ij")
    P = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    s = np.linspace(s_range[0], s_range[1], 41)
    poly = np.array([axis_fn(v) for v in s])
    best = np.full(len(P), np.inf)
    for a, b in zip(poly[:-1], poly[1:]):
        ab = b - a
        t = np.clip(((P - a) @ ab) / max(ab @ ab, 1e-12), 0, 1)
        best = np.minimum(best, np.linalg.norm(P - (a + t[:, None] * ab), axis=1))
    mask = (best <= radius).reshape(Z.shape)
    img = sitk.GetImageFromArray(np.where(mask, 300.0, 40.0).astype(np.float32))
    img.SetSpacing([sx, sy, sz])
    img.SetOrigin([ox, oy, oz])
    from scipy import ndimage as ndi
    dist = ndi.distance_transform_edt(~mask, sampling=(sz, sy, sx)).astype(np.float32)
    return Candidates(ct=np.where(mask, 300.0, 40.0).astype(np.float32), mask=mask, bright_shell=np.zeros_like(mask),
                      opened=np.zeros_like(mask), distance_mm=dist, spacing=np.array([sz, sy, sx]), image=img,
                      threshold_hu=150.0, fragment_log={}, hu_stats={}), poly


def test_bent_aorta_length_endpoints_and_clock():
    # arc of radius 60 mm in the y-z plane from z=88 down to about z=8, bowing anteriorly (-y)
    R = 60.0
    axis = lambda s: np.array([5.0, 20.0 - R * np.sin(s / R), 88.0 - R * (1 - np.cos(s / R))])
    cand, poly = _tube_candidates(axis, (0.0, 80.0))
    fr = frame_mod.build(cand)
    assert fr.method == "geodesic"
    # the union-of-segments tube has hemispherical caps, so the face centroids sit up to 0.75 R
    # beyond the arc ends: the length is the 80 mm arc plus the caps, never the 76 mm chord
    assert 80.0 < fr.length_mm < 80.0 + 2 * 0.75 * 8.0 + 3.0
    assert fr.tortuosity > 1.02
    assert np.linalg.norm(fr.endpoints_mm[0] - axis(0.0)) < 8.0
    assert np.linalg.norm(fr.endpoints_mm[1] - axis(80.0)) < 8.0
    # every centreline point within a voxel and a half of the true axis
    d = np.array([np.linalg.norm(poly - p, axis=1).min() for p in fr.centreline_mm[8:-8]])
    assert d.max() < 1.5
    # clock: at mid arc the tangent is tilted about 38 deg; a point offset along the local
    # anterior-perpendicular direction reads 12, one offset to patient's left reads 3, and the
    # frame agrees with a naive axial angle only up to that tilt
    s = 40.0
    c = axis(s)
    t = np.array([0.0, -np.cos(s / R), -np.sin(s / R)])   # d(axis)/ds
    ant = np.array([0.0, -1.0, 0.0])
    ant_perp = ant - (ant @ t) * t
    ant_perp /= np.linalg.norm(ant_perp)
    _, clk = fr.height_clock(c + 5.0 * ant_perp)
    assert min(clk, 12 - clk) < 0.15
    _, clk = fr.height_clock(c + np.array([5.0, 0.0, 0.0]))
    assert abs(clk - 3.0) < 0.15
    h, _ = fr.height_clock(c)
    cap = np.linalg.norm(fr.endpoints_mm[0] - axis(0.0))   # the hemispherical cap adds up to 0.75 R of height
    assert abs(h - (s + cap)) < 2.5
    assert fr.twist_deg < 3.0                           # planar curve: transported and projected frames agree
    assert abs(fr.radius_at(40.0) - 8.0) < 1.0


def test_tilted_aorta_oblique_cut_faces():
    # straight tube tilted 30 deg from z in the x-z plane, running the full z extent so the
    # image's axial planes cut it obliquely: the end tangent must follow the tube, not z
    ang = np.radians(30.0)
    axis = lambda s: np.array([-12.0 + s * np.sin(ang), -8.0, -5.0 + s * np.cos(ang)])
    cand, poly = _tube_candidates(axis, (0.0, 130.0), shape_xyz=(80, 40, 100), origin=(-25.0, -25.0, 0.0))
    fr = frame_mod.build(cand)
    assert fr.method == "geodesic"
    (e_top, t_top), (e_bot, t_bot) = fr.end_faces()
    axis_dir = np.array([np.sin(ang), 0.0, np.cos(ang)])
    assert np.degrees(np.arccos(np.clip(t_top @ axis_dir, -1, 1))) < 8.0
    assert np.degrees(np.arccos(np.clip(t_bot @ -axis_dir, -1, 1))) < 8.0
    # endpoints on the image's top and bottom faces (z of the first and last slice), on the axis
    z_lo, z_hi = 0.0, 1.0 * 99
    assert abs(e_top[2] - z_hi) < 1.0 and abs(e_bot[2] - z_lo) < 1.0
    for e in (e_top, e_bot):
        assert np.linalg.norm(poly - e, axis=1).min() < 1.5
    # a point straight anterior of the axis (-y) reads 12 o'clock everywhere; naive axial angle would agree here,
    # but a point offset in +x (which has a component along the tilted tangent) must still read 3
    c = axis(60.0)
    _, clk = fr.height_clock(c + np.array([0.0, -6.0, 0.0]))
    assert min(clk, 12 - clk) < 0.1
    left = np.cross(np.array([0.0, -1.0, 0.0]), -axis_dir)   # 3 o'clock = anterior x tangent; the tangent points inferior
    _, clk = fr.height_clock(c + 6.0 * left)
    assert abs(clk - 3.0) < 0.1


def test_mask_cut_by_image_boundary_has_end_faces():
    # vertical tube spanning every slice: no empty slice beyond the mask (subjects 4 to 12)
    axis = lambda s: np.array([2.0, 3.0, -20.0 + s])
    cand, _ = _tube_candidates(axis, (0.0, 200.0), shape_xyz=(50, 50, 80), origin=(-20.0, -20.0, 0.0))
    assert cand.mask[0].any() and cand.mask[-1].any()
    fr = frame_mod.build(cand)
    assert fr.method == "geodesic"
    assert fr.info["end_face_voxels"][0] > 50 and fr.info["end_face_voxels"][1] > 50
    (e_top, t_top), (e_bot, t_bot) = fr.end_faces()
    assert abs(e_top[2] - 79.0) < 1.0 and abs(e_bot[2] - 0.0) < 1.0
    assert t_top[2] > 0.98 and t_bot[2] < -0.98
    assert abs(fr.length_mm - 79.0) < 1.5


def test_spacing_along():
    gaps = frame_mod.spacing_along({"branch_002": 30.0, "branch_001": 12.5}, 100.0)
    assert [g["from"] for g in gaps] == ["superior end", "branch_001", "branch_002"]
    assert [g["to"] for g in gaps] == ["branch_001", "branch_002", "inferior end"]
    assert [g["gap_mm"] for g in gaps] == [12.5, 17.5, 70.0]
    assert frame_mod.spacing_along({}, 50.0) == [{"from": "superior end", "to": "inferior end", "gap_mm": 50.0}]


def test_fallback_to_slice_centroids(cand, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("no path")
    monkeypatch.setattr(frame_mod, "geodesic_path", boom)
    fr = frame_mod.build(cand)
    assert fr.method == "slice_centroids" and "fallback_error" in fr.info
    assert fr.centreline_mm[0, 2] > fr.centreline_mm[-1, 2]
    assert len(fr.end_faces()) == 2
