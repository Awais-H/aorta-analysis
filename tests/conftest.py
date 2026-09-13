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
LUMEN_HU = 300.0
LUMEN_NOISE_HU = 10.0
TISSUE_HU = 40.0


def make_phantom(out_dir, spacing_xyz=(0.8, 0.8, 0.8), origin_xyz=(-25.0, -30.0, 100.0),
                 size_xyz=(72, 64, 60), with_branch=True, gz_under_nii=False, mask_fragment=False,
                 direction=None):
    """Write orig.nii and mask.nii into out_dir; return a dict with paths and ground truth."""
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
    if with_branch:
        branch = ((Y - cy) ** 2 + (Z - zc) ** 2 <= BRANCH_R ** 2) & (X >= cx) & (X <= cx + AORTA_R + BRANCH_LEN)
        ct[branch & ~aorta] = LUMEN_HU
    mask = aorta.astype(np.uint8)
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

    return {
        "image": paths["orig"], "mask": paths["mask"], "sitk_image": img, "sitk_mask": msk,
        "ostium_mm": np.array([cx + AORTA_R, cy, zc]), "direction": np.array([1.0, 0.0, 0.0]),
        "seed_mm": np.array([cx + AORTA_R + 5.0, cy, zc]), "radius_mm": BRANCH_R,
        "centre_xy": (cx, cy), "z_range": (z[0], z[-1]), "with_branch": with_branch,
    }


@pytest.fixture
def phantom(tmp_path):
    return make_phantom(tmp_path / "iso")


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
