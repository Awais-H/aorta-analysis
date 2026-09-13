"""Quick per-case triage for the Branchseed challenge.
Usage: python triage.py <image.nii> <mask.nii> <out_prefix>
Prints stats and writes <out_prefix>_triage.png
"""
import sys, json
import numpy as np
import SimpleITK as sitk
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

img_p, mask_p, out = sys.argv[1], sys.argv[2], sys.argv[3]
import gzip, shutil, os, tempfile
def read_any(path):
    """Read NIfTI regardless of gzip-under-.nii, and survive non-orthonormal headers.
    Fallback path: nibabel affine (RAS) -> LPS, spacing = column norms, direction = polar-
    decomposed rotation (nearest orthonormal), origin = translation. Keeps the real rotation."""
    with open(path, "rb") as f:
        magic = f.read(2)
    if magic == b"\x1f\x8b" and not path.endswith(".gz"):
        tmp = os.path.join(tempfile.gettempdir(), os.path.basename(path) + ".gz")
        shutil.copyfile(path, tmp)
        path = tmp
    try:
        return sitk.ReadImage(path)
    except RuntimeError as e:
        if "orthonormal" not in str(e):
            raise
        import nibabel as nib
        nii = nib.load(path)
        arr = np.asanyarray(nii.dataobj)              # x, y, z
        aff = nii.affine.astype(np.float64)
        lps = np.diag([-1.0, -1.0, 1.0, 1.0]) @ aff   # RAS -> LPS
        M = lps[:3, :3]
        spacing = np.linalg.norm(M, axis=0)
        U, _, Vt = np.linalg.svd(M / spacing)
        R = U @ Vt                                     # nearest orthonormal
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1; R = U @ Vt
        im = sitk.GetImageFromArray(np.ascontiguousarray(arr.transpose(2, 1, 0)))
        im.SetSpacing([float(x) for x in spacing])
        im.SetDirection([float(x) for x in R.flatten()])
        im.SetOrigin([float(x) for x in lps[:3, 3]])
        stats_extra["header_fallback"] = "nibabel_orthonormalized"
        stats_extra["header_rotation_deg"] = round(float(np.degrees(np.arccos((np.trace(R) - 1) / 2))), 2)
        return im
stats_extra = {}
img = read_any(img_p)
msk = read_any(mask_p)
ct = sitk.GetArrayFromImage(img).astype(np.float32)   # z, y, x
m = sitk.GetArrayFromImage(msk) > 0
sp = np.array(img.GetSpacing())[::-1]                # z, y, x spacing in mm
full_shape = ct.shape
stats_pre = {"full_size_zyx": list(full_shape), "mask_touches_z_boundary_full": [bool(m[0].any()), bool(m[-1].any())]}
pad_vox = [int(np.ceil(30.0 / s)) for s in sp]
obj = ndimage.find_objects(m.astype(np.int8))[0]
crop = tuple(slice(max(0, o.start - pv), min(n, o.stop + pv)) for o, pv, n in zip(obj, pad_vox, full_shape))
ct = np.ascontiguousarray(ct[crop]); m = np.ascontiguousarray(m[crop])
ISO = 0.8
stats_pre["native_spacing_zyx_mm"] = [round(float(x), 3) for x in sp]
if abs(max(sp) - ISO) > 0.05 or abs(min(sp) - ISO) > 0.05:
    # build sitk images for the crop, resample to isotropic ISO mm (linear CT, nearest mask)
    origin_idx = [int(c.start) for c in crop][::-1]
    cimg = sitk.GetImageFromArray(ct); cimg.SetSpacing(img.GetSpacing()); cimg.SetDirection(img.GetDirection())
    cimg.SetOrigin(img.TransformIndexToPhysicalPoint(origin_idx))
    cm = sitk.GetImageFromArray(m.astype(np.uint8)); cm.CopyInformation(cimg)
    newsize = [int(round(sz * s_ / ISO)) for sz, s_ in zip(cimg.GetSize(), cimg.GetSpacing())]
    ref = sitk.Image(newsize, sitk.sitkFloat32); ref.SetSpacing([ISO] * 3); ref.SetOrigin(cimg.GetOrigin()); ref.SetDirection(cimg.GetDirection())
    ct = sitk.GetArrayFromImage(sitk.Resample(cimg, ref, sitk.Transform(), sitk.sitkLinear, -1000.0, sitk.sitkFloat32))
    m = sitk.GetArrayFromImage(sitk.Resample(cm, ref, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)) > 0
    sp = np.array([ISO] * 3)
    stats_pre["resampled_to_iso_mm"] = ISO

stats = dict(stats_pre)
stats.update(stats_extra)
stats["crop_size_zyx"] = list(ct.shape)
stats["size_zyx"] = list(ct.shape)
stats["spacing_zyx_mm"] = [round(float(s), 3) for s in sp]
stats["direction"] = [round(d, 2) for d in img.GetDirection()]
stats["mask_voxels"] = int(m.sum())
stats["mask_volume_ml"] = round(float(m.sum() * np.prod(sp) / 1000), 1)

lab, n = ndimage.label(m)
stats["mask_components"] = int(n)
if n > 1:
    sizes_m = ndimage.sum(m, lab, range(1, n + 1))
    stats["mask_fragment_sizes"] = sorted([int(x) for x in sizes_m], reverse=True)[:8]
    stats["mask_largest_fraction"] = round(float(sizes_m.max() / sizes_m.sum()), 4)
    m = lab == (int(np.argmax(sizes_m)) + 1)
stats["grid_match"] = bool(img.GetSize()==msk.GetSize() and np.allclose(img.GetSpacing(),msk.GetSpacing()) and np.allclose(img.GetOrigin(),msk.GetOrigin(),atol=1e-3))
stats["dtypes"] = [img.GetPixelIDTypeAsString(), msk.GetPixelIDTypeAsString()]
stats["mask_unique_values"] = [int(v) for v in np.unique(sitk.GetArrayFromImage(msk))[:5]]


# HU inside mask -> adaptive threshold
hu = ct[m]
stats["hu_in_mask_mean"] = round(float(hu.mean()), 1)
stats["hu_in_mask_std"] = round(float(hu.std()), 1)
stats["hu_in_mask_median"] = round(float(np.median(hu)), 1)
thr = max(float(hu.mean() - 2.0 * hu.std()), 0.45 * float(np.median(hu)), 100.0)
stats["adaptive_threshold_hu"] = round(thr, 1)

# z extent of mask, per-slice area -> diameter estimate, aneurysm check
zs = np.where(m.any(axis=(1, 2)))[0]
stats["mask_z_slices"] = [int(zs[0]), int(zs[-1])]
stats["mask_z_extent_mm"] = round(float((zs[-1] - zs[0] + 1) * sp[0]), 1)
areas = m.sum(axis=(1, 2))[zs] * sp[1] * sp[2]
diam = 2 * np.sqrt(areas / np.pi)
stats["equiv_diam_mm_min_med_max"] = [round(float(diam.min()), 1), round(float(np.median(diam)), 1), round(float(diam.max()), 1)]
stats["aneurysm_suspect"] = bool(diam.max() > 30 or diam.max() > 1.5 * np.median(diam))

# tortuosity: per-slice centroid path length vs straight distance
cents = np.array([ndimage.center_of_mass(m[z]) for z in zs])
pts = np.column_stack([zs * sp[0], cents[:, 0] * sp[1], cents[:, 1] * sp[2]])
path = np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))
chord = np.linalg.norm(pts[-1] - pts[0])
stats["centroid_path_mm"] = round(float(path), 1)
stats["tortuosity_path_over_chord"] = round(float(path / max(chord, 1e-6)), 3)
lateral_dev = np.linalg.norm(pts[:, 1:] - pts[:, 1:].mean(axis=0), axis=1).max()
stats["max_lateral_centroid_deviation_mm"] = round(float(lateral_dev), 1)

# underfill check: bright ring right outside mask
bright = ct > thr
dil1 = ndimage.binary_dilation(m, iterations=1)
ring = dil1 & ~m
stats["bright_fraction_in_1vox_ring"] = round(float(bright[ring].mean()), 3)
dil3 = ndimage.binary_dilation(m, iterations=3)
ring3 = dil3 & ~dil1
stats["bright_fraction_in_2to3vox_ring"] = round(float(bright[ring3].mean()), 3)

# very bright voxels adjacent to mask -> plaque / calcification suspect
stats["voxels_over_600HU_in_3vox_ring"] = int((ct[dil3 & ~m] > 600).sum())


# wall patches: bright voxels in a 15mm shell, morphological opening (0.8mm ball) removes the
# thin partial-volume ring outside the mask, then take what remains within 2mm of the mask
dist = ndimage.distance_transform_edt(~m, sampling=sp)
shell = (dist > 0) & (dist <= 15.0)
cand = bright & shell
r = [max(1, int(round(0.8 / s))) for s in sp]
st = np.zeros([2 * x + 1 for x in r], bool)
idx = np.indices(st.shape).reshape(3, -1).T
st.flat[np.where((((idx - np.array(r)) / np.maximum(r, 1)) ** 2).sum(1) <= 1.0)[0]] = True
cand_open = ndimage.binary_opening(cand, structure=st)
wall = cand_open & (dist <= 2.0)
wl, nw = ndimage.label(wall)
a_mm2 = sp[1] * sp[2]
sizes = ndimage.sum(wall, wl, range(1, nw + 1)) * a_mm2 if nw else np.array([])
stats["wall_patches_total"] = int(nw)
stats["wall_patches_over_5mm2"] = int((sizes > 5).sum()) if nw else 0
stats["wall_patch_areas_mm2_top8"] = sorted([round(float(x), 1) for x in sizes if x > 5], reverse=True)[:8]
# bone-sized blobs in the shell
cl, nc = ndimage.label(cand_open)
vols = ndimage.sum(cand_open, cl, range(1, nc + 1)) * np.prod(sp) / 1000 if nc else np.array([])
stats["aorta_xsec_mm2_median"] = round(float(np.median(areas)), 1)
stats["largest_patch_over_40pct_aorta"] = bool(len(sizes) and sizes.max() > 0.4 * np.median(areas))
stats["shell_blobs_over_1ml"] = int((vols > 1.0).sum()) if nc else 0

# end-face check: bright continuation beyond top and bottom z of mask
for name, z, step in (("top", zs[-1], 1), ("bottom", zs[0], -1)):
    zz = z + step * 2
    if 0 <= zz < ct.shape[0]:
        sl_m = m[z]
        stats[f"bright_beyond_{name}_face_fraction"] = round(float(bright[zz][sl_m].mean()), 3)

print(json.dumps(stats, indent=1))

# ---- render: coronal + sagittal MIP of bright voxels near the aorta, mask overlay
box = ndimage.find_objects(dil3.astype(int))[0]
pad = tuple(slice(max(0, s.start - 40), s.stop + 40) for s in box)
sub = ct[pad]
subm = m[pad]
subshell = (bright & (dist <= 25.0))[pad]
fig, axes = plt.subplots(1, 3, figsize=(16, 7))
asp_cor = sp[0] / sp[2]
asp_sag = sp[0] / sp[1]
cor = subshell.max(axis=1)
sag = subshell.max(axis=2)
axes[0].imshow(cor[::-1], cmap="gray", aspect=asp_cor)
mm = subm.max(axis=1)[::-1].astype(float)
axes[0].imshow(np.ma.masked_where(mm == 0, mm), cmap="spring", alpha=0.35, aspect=asp_cor)
axes[0].set_title("coronal MIP, bright voxels within 25mm of mask (mask overlay)")
axes[1].imshow(sag[::-1], cmap="gray", aspect=asp_sag)
ms = subm.max(axis=2)[::-1].astype(float)
axes[1].imshow(np.ma.masked_where(ms == 0, ms), cmap="spring", alpha=0.35, aspect=asp_sag)
axes[1].set_title("sagittal MIP (anterior on left)")
axes[2].plot(diam)
axes[2].set_title("equiv diameter per slice (mm), bottom->top")
axes[2].set_xlabel("slice index within mask")
plt.tight_layout()
plt.savefig(f"{out}_triage.png", dpi=110)
print("wrote", f"{out}_triage.png")
