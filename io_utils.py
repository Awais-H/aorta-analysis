"""NIfTI loading, crop, resample and index <-> mm conversion.

Data contract (SPEC.md D9): reads NIfTI by sniffing the gzip magic (not by extension), survives the
non-orthonormal header of subject 24 via a nibabel fallback, owns the crop and the resample to
ISO_SPACING_MM, and exposes index_to_mm(), the ONLY caller of TransformIndexToPhysicalPoint in the
codebase. Nothing else converts coordinates.

Index convention: every array in the pipeline is numpy (z, y, x). Every index passed into or out of
this module is (z, y, x). Every physical point is (x, y, z) mm in SimpleITK's LPS frame, which is
what the challenge's "physical coordinate system returned by SimpleITK" means.

File reading is ported from triage.py (read_any); crop and resample from triage.py lines 56 to 73.
"""
from __future__ import annotations

import logging
import os
import shutil
import tempfile
from dataclasses import dataclass, field

import numpy as np
import SimpleITK as sitk
from scipy import ndimage

import config

log = logging.getLogger("branchseed.io")

GZIP_MAGIC = b"\x1f\x8b"


# --------------------------------------------------------------------------- reading


def _is_gzip(path: str) -> bool:
    with open(path, "rb") as f:
        return f.read(2) == GZIP_MAGIC


def _read_with_nibabel_fallback(path: str, info: dict) -> sitk.Image:
    """Load a NIfTI whose direction cosines SimpleITK rejects as non-orthonormal (subject 24).

    nibabel affine (RAS) -> LPS, spacing = column norms, direction = nearest orthonormal matrix by
    polar decomposition (SVD), origin = translation. Keeps the real rotation of the grid.
    """
    import nibabel as nib

    nii = nib.load(path)
    arr = np.asanyarray(nii.dataobj)  # x, y, z
    aff = nii.affine.astype(np.float64)
    lps = np.diag([-1.0, -1.0, 1.0, 1.0]) @ aff  # RAS -> LPS
    M = lps[:3, :3]
    spacing = np.linalg.norm(M, axis=0)
    U, _, Vt = np.linalg.svd(M / spacing)
    R = U @ Vt  # nearest orthonormal
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    im = sitk.GetImageFromArray(np.ascontiguousarray(arr.transpose(2, 1, 0)))
    im.SetSpacing([float(x) for x in spacing])
    im.SetDirection([float(x) for x in R.flatten()])
    im.SetOrigin([float(x) for x in lps[:3, 3]])
    # tilt = rotation away from the nearest axis-aligned (signed permutation) orientation, so an
    # array stored in RAS order does not read as a 180 degree "rotation"
    P = np.zeros((3, 3))
    for j in range(3):
        i = int(np.argmax(np.abs(R[:, j])))
        P[i, j] = np.sign(R[i, j])
    info["header_fallback"] = "nibabel_orthonormalized"
    info["header_rotation_deg"] = round(float(np.degrees(np.arccos(np.clip((np.trace(R.T @ P) - 1) / 2, -1, 1)))), 2)
    log.warning("%s: non-orthonormal header, loaded via nibabel fallback (grid rotated %.2f deg)",
                path, info["header_rotation_deg"])
    return im


def read_image(path: str) -> tuple[sitk.Image, dict]:
    """Read a NIfTI regardless of gzip-under-.nii, and survive non-orthonormal headers.

    Returns (image, info) where info records what had to be done to open the file.
    """
    info = {"path": path, "gz_under_nii": False, "header_fallback": None}
    tmpdir = None
    read_path = path
    try:
        if _is_gzip(path) and not path.endswith(".gz"):
            # SimpleITK picks its reader by extension and refuses a gzip stream under .nii.
            tmpdir = tempfile.mkdtemp(prefix="branchseed_")
            read_path = os.path.join(tmpdir, os.path.basename(path) + ".gz")
            shutil.copyfile(path, read_path)
            info["gz_under_nii"] = True
            log.info("%s: gzip stream under a .nii name, reading as .nii.gz", path)
        try:
            return sitk.ReadImage(read_path), info
        except RuntimeError as e:
            if "orthonormal" not in str(e):
                raise
            return _read_with_nibabel_fallback(read_path, info), info
    finally:
        if tmpdir is not None:
            shutil.rmtree(tmpdir, ignore_errors=True)


def load_case(image_path: str, mask_path: str) -> tuple[sitk.Image, sitk.Image, dict]:
    """Load the CT and the aorta mask. Returns (image, mask_image, info)."""
    image, info_i = read_image(image_path)
    mask, info_m = read_image(mask_path)
    info = {"image": info_i, "mask": info_m}
    info["grid_match"] = bool(
        image.GetSize() == mask.GetSize()
        and np.allclose(image.GetSpacing(), mask.GetSpacing())
        and np.allclose(image.GetOrigin(), mask.GetOrigin(), atol=1e-3)
    )
    if not info["grid_match"]:
        log.warning("image and mask grids differ: image %s/%s, mask %s/%s; using the image grid",
                    image.GetSize(), image.GetSpacing(), mask.GetSize(), mask.GetSpacing())
    if image.GetSize() != mask.GetSize():
        raise ValueError(f"image size {image.GetSize()} != mask size {mask.GetSize()}")
    return image, mask, info


# ------------------------------------------------------------------ index <-> mm


def index_to_mm(image: sitk.Image, idx_zyx) -> np.ndarray:
    """Map (z, y, x) indices on `image`'s grid to (x, y, z) physical mm.

    THE only place TransformIndexToPhysicalPoint is called (CLAUDE.md rule). Integer indices go
    through TransformIndexToPhysicalPoint as the challenge PDF requires; fractional indices (path
    points, centroids) through the continuous variant, which is the same affine map.
    Accepts one index of shape (3,) or an array of shape (N, 3); returns the same leading shape.
    """
    idx = np.asarray(idx_zyx, dtype=np.float64)
    if idx.ndim == 1:
        xyz = idx[::-1]
        if np.all(np.abs(xyz - np.round(xyz)) < 1e-9):
            return np.asarray(image.TransformIndexToPhysicalPoint([int(round(v)) for v in xyz]), dtype=np.float64)
        return np.asarray(image.TransformContinuousIndexToPhysicalPoint([float(v) for v in xyz]), dtype=np.float64)
    return np.stack([index_to_mm(image, row) for row in idx], axis=0) if len(idx) else np.zeros((0, 3))


def mm_to_index(image: sitk.Image, xyz_mm) -> np.ndarray:
    """Inverse of index_to_mm: (x, y, z) mm -> continuous (z, y, x) index on `image`'s grid."""
    p = np.asarray(xyz_mm, dtype=np.float64)
    if p.ndim == 1:
        return np.asarray(image.TransformPhysicalPointToContinuousIndex([float(v) for v in p]), dtype=np.float64)[::-1]
    return np.stack([mm_to_index(image, row) for row in p], axis=0) if len(p) else np.zeros((0, 3))


def index_vector_to_mm(image: sitk.Image, idx_zyx, vec_zyx) -> np.ndarray:
    """Unit (x, y, z) physical direction of an index-space vector `vec_zyx` placed at `idx_zyx`."""
    idx = np.asarray(idx_zyx, dtype=np.float64)
    vec = np.asarray(vec_zyx, dtype=np.float64)
    d = index_to_mm(image, idx + vec) - index_to_mm(image, idx)
    n = np.linalg.norm(d)
    return d / n if n > 0 else d


# ------------------------------------------------------------------ crop and resample


@dataclass
class WorkingGrid:
    """The cropped, resampled block everything downstream runs on."""

    ct: np.ndarray            # float32 (z, y, x) HU
    mask: np.ndarray          # bool (z, y, x)
    image: sitk.Image         # working image handle: the argument for index_to_mm
    spacing: np.ndarray       # (3,) zyx mm, all ISO_SPACING_MM after resampling
    crop: tuple               # slices into the native grid (z, y, x)
    native_spacing: np.ndarray  # (3,) zyx mm
    native_shape: tuple
    resampled: bool
    info: dict = field(default_factory=dict)


def crop_and_resample(image: sitk.Image, mask: np.ndarray,
                      pad_mm: float = config.CROP_PAD_MM,
                      iso: float = config.ISO_SPACING_MM) -> WorkingGrid:
    """Crop the CT and mask to the padded mask bounding box, then resample to `iso` mm isotropic.

    Crop first (D10: the full-volume distance transform is what killed the triage process). The
    resample preserves physical coordinates, so index_to_mm on the returned image is exact.
    """
    mask = np.asarray(mask) > 0
    sp = np.array(image.GetSpacing())[::-1]  # z, y, x
    full_shape = tuple(int(v) for v in mask.shape)
    if not mask.any():
        raise ValueError("aorta mask is empty")
    pad_vox = [int(np.ceil(pad_mm / s)) for s in sp]
    obj = ndimage.find_objects(mask.astype(np.int8))[0]
    crop = tuple(slice(max(0, o.start - pv), min(n, o.stop + pv)) for o, pv, n in zip(obj, pad_vox, full_shape))

    ct_c = np.ascontiguousarray(sitk.GetArrayViewFromImage(image)[crop]).astype(np.float32)
    m_c = np.ascontiguousarray(mask[crop])
    origin_idx_zyx = [int(c.start) for c in crop]

    cimg = sitk.GetImageFromArray(ct_c)
    cimg.SetSpacing(image.GetSpacing())
    cimg.SetDirection(image.GetDirection())
    cimg.SetOrigin([float(v) for v in index_to_mm(image, origin_idx_zyx)])
    cm = sitk.GetImageFromArray(m_c.astype(np.uint8))
    cm.CopyInformation(cimg)

    resampled = False
    if abs(max(sp) - iso) > config.RESAMPLE_TOLERANCE_MM or abs(min(sp) - iso) > config.RESAMPLE_TOLERANCE_MM:
        newsize = [max(1, int(round(sz * s_ / iso))) for sz, s_ in zip(cimg.GetSize(), cimg.GetSpacing())]
        ref = sitk.Image(newsize, sitk.sitkFloat32)
        ref.SetSpacing([iso] * 3)
        ref.SetOrigin(cimg.GetOrigin())
        ref.SetDirection(cimg.GetDirection())
        cimg = sitk.Resample(cimg, ref, sitk.Transform(), sitk.sitkLinear, config.CT_BACKGROUND_HU, sitk.sitkFloat32)
        cm = sitk.Resample(cm, ref, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
        ct_c = sitk.GetArrayFromImage(cimg)
        m_c = sitk.GetArrayFromImage(cm) > 0
        resampled = True

    work_sp = np.array(cimg.GetSpacing())[::-1]
    info = {
        "native_shape_zyx": list(full_shape),
        "native_spacing_zyx_mm": [round(float(x), 4) for x in sp],
        "crop_zyx": [[int(c.start), int(c.stop)] for c in crop],
        "working_shape_zyx": list(ct_c.shape),
        "working_spacing_zyx_mm": [round(float(x), 4) for x in work_sp],
        "resampled": resampled,
    }
    log.info("crop %s -> %s, working spacing %s (resampled=%s)", full_shape, ct_c.shape, work_sp, resampled)
    return WorkingGrid(ct=ct_c, mask=m_c, image=cimg, spacing=work_sp, crop=crop,
                       native_spacing=sp, native_shape=full_shape, resampled=resampled, info=info)
