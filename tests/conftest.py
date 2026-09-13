"""Synthetic phantom shared by every module test.

A vertical bright cylinder (the aorta, radius AORTA_R mm) with the mask exactly on it, plus an
optional horizontal side branch (radius BRANCH_R mm) leaving toward +x (patient's left) at mid
height. Built in physical mm so any voxel spacing gives the same anatomy. Ground truth: ostium at
the aorta surface on the branch axis, direction (1, 0, 0), seed 5 mm further along +x.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import SimpleITK as sitk

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

AORTA_R = 6.0
BRANCH_R = 2.0
BRANCH_LEN = 30.0
BIFURCATION_MM = 5.0  # trunk length beyond the wall in the bifurcating phantom; the children's oblique cross-sections separate about 4 mm further out
LUMEN_HU = 300.0
LUMEN_NOISE_HU = 10.0
TISSUE_HU = 40.0


def branch_polyline(cx, cy, zc, kind):
    """Centre-line(s) of the side branch in mm, starting at the aorta axis and leaving toward +x.

    straight: a straight tube along +x.
    curved: leaves along +x then bends toward +z on an 8 mm radius, so a straight line from the
            ostium misses the lumen by the time it is 5 mm out.
    bifurcating: a straight trunk that splits BIFURCATION_MM beyond the wall into two children at
            plus and minus 60 degrees in the x-z plane, so the two 2 mm lumens separate within 2 mm of the split.
    """
    x0 = cx + AORTA_R
    if kind == "straight":
        return [np.array([[cx, cy, zc], [x0 + BRANCH_LEN, cy, zc]])]
    if kind == "curved":
        R = 8.0
        ang = np.linspace(0, np.pi / 2, 40)
        arc = np.column_stack([x0 + 2.0 + R * np.sin(ang), np.full_like(ang, cy), zc + R * (1 - np.cos(ang))])
        tail = arc[-1] + np.array([0.0, 0.0, 12.0])
        return [np.vstack([[[cx, cy, zc], [x0 + 2.0, cy, zc]], arc, [tail]])]
    if kind == "bifurcating":
        xb = x0 + BIFURCATION_MM
        trunk = np.array([[cx, cy, zc], [xb, cy, zc]])
        a = np.radians(60.0)
        c1 = np.array([[xb, cy, zc], [xb + 15 * np.cos(a), cy, zc + 15 * np.sin(a)]])
        c2 = np.array([[xb, cy, zc], [xb + 15 * np.cos(a), cy, zc - 15 * np.sin(a)]])
        return [trunk, c1, c2]
    raise ValueError(kind)


def _dist_to_polyline(P, poly):
    """Min distance of points P (n, 3) to a polyline (k, 3)."""
    best = np.full(len(P), np.inf)
    for a, b in zip(poly[:-1], poly[1:]):
        ab = b - a
        t = np.clip(((P - a) @ ab) / max(ab @ ab, 1e-12), 0, 1)
        best = np.minimum(best, np.linalg.norm(P - (a + t[:, None] * ab), axis=1))
    return best


def make_phantom(out_dir, spacing_xyz=(0.8, 0.8, 0.8), origin_xyz=(-25.0, -30.0, 100.0),
                 size_xyz=(72, 64, 60), with_branch=True, gz_under_nii=False, mask_fragment=False,
                 direction=None, branch_kind="straight", mask_z_fraction=1.0, parallel_vessel=False,
                 blob=False, hugging=False, iliac_split=False, blob_radius_mm=9.0):
    """Write orig.nii and mask.nii into out_dir; return a dict with paths and ground truth.

    mask_z_fraction < 1 keeps the mask on the middle fraction of the aorta only, so the bright
    aorta continues past both mask ends (D6 rule 1). parallel_vessel adds a bright 2.5 mm tube
    running along the aorta in contact with it (rule 3). blob adds a bright 9 mm sphere touching
    the aorta (rule 4: bone-sized, 3 ml).
    """
    os.makedirs(out_dir, exist_ok=True)
    sx, sy, sz = spacing_xyz
    nx, ny, nz = size_xyz
    ox, oy, oz = origin_xyz
    x = ox + sx * np.arange(nx)
    y = oy + sy * np.arange(ny)
    z = oz + sz * np.arange(nz)
    Z, Y, X = np.meshgrid(z, y, x, indexing="ij")  # arrays are (z, y, x)
    cx, cy, zc = ox + sx * nx / 2, oy + sy * ny / 2, oz + sz * nz / 2

    aorta = (X - cx) ** 2 + (Y - cy) ** 2 <= AORTA_R ** 2
    rng = np.random.default_rng(0)
    ct = np.full(aorta.shape, TISSUE_HU, np.float32)
    ct[aorta] = LUMEN_HU + rng.normal(0.0, LUMEN_NOISE_HU, int(aorta.sum()))
    polys = branch_polyline(cx, cy, zc, branch_kind) if with_branch else []
    if with_branch:
        P = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        near = np.zeros(len(P), bool)
        for poly in polys:
            near |= _dist_to_polyline(P, poly) <= BRANCH_R
        branch = near.reshape(aorta.shape)
        ct[branch & ~aorta] = LUMEN_HU
    if parallel_vessel:
        px = cx - AORTA_R - 2.5 + 0.5  # 2.5 mm tube overlapping the aorta surface by 0.5 mm
        ct[((X - px) ** 2 + (Y - cy) ** 2 <= 2.5 ** 2) & ~aorta] = LUMEN_HU
    if hugging:
        # a 1.5 mm-radius tube leaving the aorta at (cx, cy + AORTA_R, zc - 10) and running along
        # the wall toward +z, lifting off linearly: its axis sits 0.5 mm outside the surface at the
        # proximal end (2 mm of lumen protruding, enough to survive the opening) and 3 mm outside
        # it 20 mm later (SPEC D3/D4: the branch that hugs the wall)
        t = np.clip((Z - (zc - 10.0)) / 20.0, 0.0, 1.0)
        yaxis = cy + AORTA_R + 0.5 + 2.5 * t
        hug_tube = ((Y - yaxis) ** 2 + (X - cx) ** 2 <= 1.5 ** 2) & (Z >= zc - 10.0) & (Z <= zc + 10.0)
        ct[hug_tube & ~aorta] = LUMEN_HU
    if blob:
        by = cy + AORTA_R + blob_radius_mm - 0.5  # sphere overlapping the aorta surface by 0.5 mm
        ct[((X - cx) ** 2 + (Y - by) ** 2 + (Z - zc) ** 2 <= blob_radius_mm ** 2) & ~aorta] = LUMEN_HU
    mask = aorta.astype(np.uint8)
    if mask_z_fraction < 1.0:
        keep = np.abs(Z - zc) <= mask_z_fraction * (z[-1] - z[0]) / 2
        mask[~keep] = 0
    if iliac_split:
        # below the mask's inferior end the bright aorta is replaced by two 4 mm-radius tubes
        # diverging at 35 deg in the x-z plane (the common iliacs); the aorta above is untouched
        z_lo = zc - mask_z_fraction * (z[-1] - z[0]) / 2
        below = Z < z_lo
        ct[below & aorta] = TISSUE_HU
        for sgn in (-1.0, 1.0):
            ax_x = cx + sgn * (2.0 + (z_lo - Z) * np.tan(np.radians(35.0)))
            tube = below & ((X - ax_x) ** 2 + (Y - cy) ** 2 <= 4.0 ** 2)
            ct[tube] = LUMEN_HU
    if mask_fragment:
        mask[1, 1, 1] = 1  # a stray voxel far from the aorta

    img = sitk.GetImageFromArray(ct)
    img.SetSpacing([float(sx), float(sy), float(sz)])
    img.SetOrigin([float(ox), float(oy), float(oz)])
    if direction is not None:
        img.SetDirection([float(v) for v in np.asarray(direction).flatten()])
    msk = sitk.GetImageFromArray(mask)
    msk.CopyInformation(img)

    paths = {}
    for name, im in (("orig", img), ("mask", msk)):
        if gz_under_nii:
            tmp = os.path.join(out_dir, f"{name}.nii.gz")
            sitk.WriteImage(im, tmp)
            final = os.path.join(out_dir, f"{name}.nii")
            os.replace(tmp, final)
        else:
            final = os.path.join(out_dir, f"{name}.nii")
            sitk.WriteImage(im, final)
        paths[name] = final

    seed = point_along_polylines(polys[0], cx + AORTA_R, 5.0) if with_branch else None
    ostium = np.array([cx + AORTA_R, cy, zc])
    return {
        "image": paths["orig"], "mask": paths["mask"], "sitk_image": img, "sitk_mask": msk,
        "ostium_mm": ostium, "direction": (seed - ostium) / np.linalg.norm(seed - ostium) if with_branch else None,
        "seed_mm": seed, "radius_mm": BRANCH_R, "branch_kind": branch_kind,
        "polylines": polys, "bifurcation_mm": BIFURCATION_MM if branch_kind == "bifurcating" else None,
        "centre_xy": (cx, cy), "z_range": (z[0], z[-1]), "with_branch": with_branch,
        "hugging_ostium_mm": np.array([cx, cy + AORTA_R, zc - 10.0]) if hugging else None,
    }


def point_along_polylines(poly, x_wall, s_mm):
    """Point s_mm of arc length along `poly` measured from where it crosses x = x_wall."""
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    # arc length at the wall crossing (first point with x >= x_wall, linear within the segment)
    xs = poly[:, 0]
    k = int(np.argmax(xs >= x_wall))
    f = (x_wall - xs[k - 1]) / max(xs[k] - xs[k - 1], 1e-12) if k > 0 else 0.0
    s0 = arc[k - 1] + f * seg[k - 1] if k > 0 else 0.0
    target = s0 + s_mm
    return np.array([np.interp(target, arc, poly[:, a]) for a in range(3)])


@pytest.fixture
def phantom(tmp_path):
    return make_phantom(tmp_path / "iso")


@pytest.fixture
def phantom_curved(tmp_path):
    return make_phantom(tmp_path / "curved", branch_kind="curved")


@pytest.fixture
def phantom_bifurcating(tmp_path):
    return make_phantom(tmp_path / "bif", branch_kind="bifurcating")


@pytest.fixture
def phantom_no_branch(tmp_path):
    return make_phantom(tmp_path / "nobranch", with_branch=False)


@pytest.fixture
def phantom_aniso(tmp_path):
    return make_phantom(tmp_path / "aniso", spacing_xyz=(0.7, 0.7, 1.5), size_xyz=(80, 72, 32))


@pytest.fixture
def cand(phantom):
    import candidates
    import io_utils
    image, mask_image, _ = io_utils.load_case(phantom["image"], phantom["mask"])
    return candidates.build(image, mask_image)


@pytest.fixture
def inst(cand):
    import instances
    return instances.build(cand)


@pytest.fixture
def ostia(cand, inst):
    import ostium
    return ostium.locate(cand, inst)


@pytest.fixture
def traces(cand, inst, ostia):
    import tracing
    return tracing.trace_all(cand, inst, ostia)


def fake_daughter(k, ostium, direction=(1.0, 0.0, 0.0), radius=2.5, seed_dist=5.0):
    o = np.asarray(ostium, float)
    d = np.asarray(direction, float)
    d = d / np.linalg.norm(d)
    return {"instance_id": f"branch_{k:03d}", "parent_instance_id": "aorta",
            "ostium_xyz_mm": [float(v) for v in o], "seed_xyz_mm": [float(v) for v in o + seed_dist * d],
            "radius_mm": float(radius), "direction_xyz": [float(v) for v in d]}


def fake_result(case_id="subject000", daughters=()):
    return {"case_id": case_id, "parent": {"instance_id": "aorta"}, "daughters": list(daughters)}
