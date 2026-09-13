# Branchseed — Implementation Specification

Companion to the system design. This document says *how to build it*: module boundaries, data contracts, exact library calls, numerical details, and the failure modes of each step. No code, but enough detail that writing the code is mechanical.

---

# Part A — Project setup

## A.1 Repository layout

```
branchseed/
  run.py                    # CLI entry point, the mandated interface
  config.yaml               # every tunable constant, one file
  branchseed/
    __init__.py
    io.py                   # Stage 0: load, validate, coordinate mapping
    roi.py                  # Stage 1: crop, grid normalisation
    calibrate.py            # Stage 2: HU statistics from the mask
    geometry.py             # Stage 3: centreline, EDT, surface, frame, end caps
    wallmap.py              # Stage 4: unrolled map construction
    detect.py               # Stage 4: 2D peak detection -> candidates
    trace.py                # Stage 5: geodesic level-set tracing
    measure.py              # Stage 6: ostium, direction, seed, radius
    resolve.py              # Stage 7: instance resolution rules
    output.py               # Stage 8: JSON serialisation
    viz.py                  # unrolled display + verification figures
    frames.py               # rotation-minimising frame (shared utility)
    interp.py               # sub-voxel sampling helpers (shared utility)
    timing.py               # per-stage timing and memory instrumentation
  tests/
    test_coords.py
    test_phantom.py
    phantoms.py             # synthetic volume generator
    test_endcaps.py
    test_output_schema.py
  tools/
    sweep.py                # parameter tuning harness
    evaluate.py             # dev-set scoring against reference JSON
  vendor/                   # pre-downloaded wheels for offline install
  environment.yml
  requirements.txt
  README.md
```

`run.py` stays thin: parse args, load config, call a single `pipeline.process_case()`, write JSON. All logic lives in the package so the test suite can call stages independently.

## A.2 Dependencies

| Package | Version pin | Used for |
|---|---|---|
| SimpleITK | 2.3+ | I/O, physical coordinates, resampling |
| numpy | 1.24+ | everything |
| scipy | 1.10+ | ndimage, interpolate, sparse.csgraph |
| scikit-image | 0.21+ | skeletonize, marching_cubes, regionprops |
| matplotlib | 3.7+ | figures (Agg backend only) |
| PyYAML | any | config |
| scikit-fmm | 2023.4+ | geodesic distance (optional, see C.5.3) |
| numba | 0.57+ | optional, only if profiling demands it |

**Offline install.** The evaluation box has no internet. Run `pip download -r requirements.txt -d vendor/` on a connected machine, commit `vendor/`, and make the README's setup command `pip install --no-index --find-links vendor -r requirements.txt`. Test this in a network-disabled container before submitting — a setup command that silently requires PyPI is a reproducibility zero.

Avoid: anything with pretrained weights, anything that downloads data on first import, anything GPU-conditional.

## A.3 Config file

One `config.yaml`, loaded once, passed as a frozen dataclass. No constant appears inline in the code. Structure it by stage so the sweep harness can address parameters by dotted path:

```yaml
roi:
  dilation_mm: 30.0
  subvolume_half_mm: 15.0
  subvolume_spacing_mm: 0.5
calibrate:
  erosion_mm: 2.0
  lumen_percentile: 5.0
  calcium_sigma: 4.0
  max_sigma_ratio: 0.25        # guard: sigma/mu above this -> degenerate flag
geometry:
  spur_prune_factor: 1.0       # x local radius
  centreline_step_mm: 1.0
  spline_smoothing: 0.5
  endcap_normal_deg: 35.0
  endcap_plane_mm: 3.0
wallmap:
  probe_inner_mm: 1.0
  probe_outer_mm: 6.0
  probe_radius_mm: 1.0
  probe_samples: 11
  angular_bins: 128
  ray_step_mm: 0.25
  ray_max_factor: 2.5          # x local radius
  frame_validate_arc_mm: 3.0
detect:
  peak_footprint: [3, 3]
  min_region_px: 4
trace:
  step_mm: 0.5
  max_mm: 10.0
  eligible_mm: 5.0
  leak_ratio: 1.5
  leak_lookback_mm: 3.0
  radius_ratio: 0.90
  strong_match_frac: 0.5
  split_persist_mm: 1.0
measure:
  seed_mm: 5.0
  xsec_half_mm: 6.0
  xsec_spacing_mm: 0.25
resolve:
  bump_boundary: [a, b, c]     # fitted coefficients
  hu_band_sigma: 6.0
```

Add a `--config` override flag for the sweep harness, but keep the default path baked in so the mandated CLI works unmodified.

---

# Part B — Data contracts

Define these as frozen dataclasses in `branchseed/types.py`. Every stage takes and returns these, which makes stages independently testable.

## B.1 `CaseGrid`

Carries everything needed to map back to physical space. Created once in Stage 1, never mutated.

- `sitk_ref`: the original SimpleITK image (for `TransformContinuousIndexToPhysicalPoint`)
- `crop_origin_idx`: `(i0, j0, k0)` integer index offset of the ROI within the original
- `roi_spacing_mm`: `(sx, sy, sz)` of the ROI array
- `orig_spacing_mm`: `(sx, sy, sz)` of the original
- `roi_shape`: array shape in **numpy order** `(nz, ny, nx)`

Provide one method: `to_physical(roi_continuous_index_xyz) -> (x, y, z) mm`. Everything that emits a coordinate calls this and nothing else.

## B.2 `Calibration`

`mu_ao`, `sigma_ao`, `t_lumen`, `t_calcium`, `t_bg`, `n_interior_voxels`, `degenerate: bool`.

## B.3 `AortaGeometry`

- `centreline`: `(N, 3)` float array, ROI continuous indices, 1 mm arc spacing
- `arc_length`: `(N,)` cumulative mm from the first sample
- `tangent`: `(N, 3)` unit vectors
- `frame_u`, `frame_v`: `(N, 3)` rotation-minimising orthonormal frame
- `radius_mm`: `(N,)` local inscribed radius from EDT
- `mask`: boolean ROI array
- `surface`: boolean ROI array
- `eligible_wall`: boolean ROI array
- `edt_mm`: float32 ROI array

## B.4 `WallMap`

- `intensity`: `(N, B)` float32, mean probe intensity, NaN where invalid
- `wall_radius_mm`: `(N, B)` float32, distance from centreline to wall along the ray
- `wall_point`: `(N, B, 3)` float32, ROI continuous index of the wall crossing
- `valid`: `(N, B)` bool
- `theta`: `(B,)` angles in radians, 0 = anterior

## B.5 `Candidate`

Mutable through Stages 4–7, then frozen.

`id`, `map_region` (list of (i,j) pixels), `peak_ij`, `peak_value`, `patch_points` `(P,3)`, `patch_weights` `(P,)`, `ostium_xyz_roi`, `path` `(M,3)`, `path_radius_mm` `(M,)`, `path_length_mm`, `stop_reason`, `eligible`, `direction`, `seed_xyz_roi`, `radius_mm`, `confidence`, `flags`, `rejected_by`.

Keep rejected candidates in the list with `rejected_by` set rather than deleting them. Essential for debugging and for the sweep harness to report *why* recall was lost.

---

# Part C — Module specifications

## C.0 `io.py` — loading

### C.0.1 Format sniffing

```
read first 2 bytes
  == b'\x1f\x8b'  -> gzip-compressed regardless of extension
```

SimpleITK's `ReadImage` dispatches on content for NIfTI in most builds, but do not rely on it. If the extension and the content disagree, copy to a temp file with the correct extension and read that. Case 25's uploads are gzip named `.nii`, and `nibabel.load` raises `ImageFileError: Cannot work out file type` on exactly this.

Accept `.nii`, `.nii.gz`, and mismatches of both.

### C.0.2 Axis-order hazard

**This is the single most likely source of silent geometric error in the whole project.**

`sitk.GetArrayFromImage(img)` returns a numpy array in **`[z, y, x]`** order. `img.GetSize()`, `GetSpacing()`, `GetOrigin()` are in **`(x, y, z)`** order. `TransformContinuousIndexToPhysicalPoint` takes **`(x, y, z)`**.

Adopt one convention and enforce it:

- All numpy arrays are `[z, y, x]`.
- All *points* — centreline samples, path points, wall points, directions — are `(x, y, z)` ROI continuous index.
- Any function that indexes an array from a point does the flip explicitly, in one place, in `interp.py`.

Name variables accordingly (`pt_xyz`, `arr_zyx`). Write the round-trip test in C.0.4 before writing anything else.

### C.0.3 Grid validation

Compare image and mask on `GetSize`, `GetSpacing`, `GetOrigin`, `GetDirection`. Tolerance 1e-4 relative for floats.

On mismatch: resample the mask onto the image grid with `sitk.ResampleImageFilter`, `SetInterpolator(sitk.sitkNearestNeighbor)`, `SetReferenceImage(image)`. Log a warning and add a case-level flag. Do not resample the image — the spec says image and mask share the grid, so a mismatch means something upstream is odd and the image is the authority for coordinates.

Check the mask is binary. If it has more than two values, threshold at `>0` and flag. If it has exactly `{0, 255}`, normalise to `{0, 1}`.

Check the mask is non-empty. An empty mask means an empty daughters list and a clean exit, not a crash.

### C.0.4 Coordinate round-trip test

Write this first and run it on every case:

```
pick voxel index (i,j,k) at the mask centroid
p = sitk_ref.TransformIndexToPhysicalPoint((i,j,k))
q = sitk_ref.TransformPhysicalPointToContinuousIndex(p)
assert allclose(q, (i,j,k), atol=1e-6)
```

Then the ROI version: ROI continuous index → original continuous index (add `crop_origin_idx`, scale by spacing ratio if resampled) → physical. Assert the ROI centre maps to the same physical point computed directly.

If you prototype anything in nibabel, remember it reports RAS while SimpleITK reports LPS: x and y flip sign. Never mix the two in one code path.

---

## C.1 `roi.py` — crop and grid

### C.1.1 Bounding box

`scipy.ndimage.find_objects(mask.astype(int))[0]` gives the tight bbox as a tuple of slices. Expand by `ceil(dilation_mm / spacing)` per axis, clip to array bounds.

Case 25 reference: mask bbox 39 × 65 × 229 voxels out of 236 × 236 × 421 — about 2.5% of the volume. After 30 mm dilation at 1.5 mm you add 20 voxels per side, giving roughly 79 × 105 × 269 ≈ 2.2 M voxels. That is the working size.

### C.1.2 Isotropy

If `max(spacing) / min(spacing) > 1.05`, resample the ROI to isotropic at `min(spacing)` using `sitk.ResampleImageFilter` with `sitkBSpline` for the image and `sitkNearestNeighbor` for the mask. Update `CaseGrid` accordingly.

If already isotropic (case 25 is), skip — resampling costs time and introduces interpolation error for nothing.

**Do not globally upsample below native spacing.** Sub-voxel precision comes from `map_coordinates` at the points you actually need, and from per-candidate subvolumes in Stage 5. Upsampling a 2.2 M-voxel ROI to 0.5 mm gives ~60 M voxels and 240 MB per array — affordable but wasteful, and it makes the EDT and skeletonisation an order of magnitude slower for no accuracy gain in those steps.

### C.1.3 Memory discipline

Hold at most: the ROI image (float32), the ROI mask (bool), the EDT (float32), and one working array. Delete the uncropped arrays immediately after cropping — `del` plus letting the SimpleITK image go out of scope. Convert the image from int16 to float32 once, at crop time.

---

## C.2 `calibrate.py` — HU statistics

### C.2.1 Interior extraction

`scipy.ndimage.binary_erosion(mask, iterations=round(erosion_mm / spacing))`. With a 2 mm erosion at 1.5 mm spacing that is 1 iteration, which is thin. Prefer distance-based: `edt_mm > erosion_mm`, computed from `distance_transform_edt(mask, sampling=spacing_zyx)`. Note `sampling` takes spacing in **array order**, `[z, y, x]`.

You need the EDT in Stage 3 anyway, so compute it here once and pass it forward.

Case 25: 40,285 interior voxels from 58,520 total. Plenty for stable percentiles. If the interior has fewer than ~500 voxels (a very thin or short mask), fall back to the full mask interior and flag.

### C.2.2 Thresholds

- `mu_ao = mean(hu_interior)`, `sigma_ao = std(hu_interior)`
- `t_lumen = percentile(hu_interior, 5)`
- `t_calcium = mu_ao + 4 * sigma_ao`
- `t_bg = median(hu[dilated_shell & ~mask & (hu < t_lumen)])`

Case 25 values: mu 485, sigma 32, t_lumen 433, t_calcium ≈ 613.

### C.2.3 Degeneracy guard

If `sigma_ao / mu_ao > 0.25`, the interior distribution is not a clean contrast peak. Causes: thrombus included in the mask, poor contrast timing, a mask that leaked into surrounding tissue, or a non-contrast scan.

Response: set `t_lumen` to `mu_ao - 1.5 * sigma_ao` instead of the percentile (more permissive), set `degenerate = True`, and propagate a case-level flag into the output and the display. Do not abort — an over-permissive threshold with a flag beats a crash.

Also guard the other way: if `mu_ao < 150 HU` the scan probably has no arterial contrast at all. Flag prominently; expect poor results and say so.

---

## C.3 `geometry.py` — aortic geometry

### C.3.1 Skeleton

`skimage.morphology.skeletonize(mask, method='lee')` for 3D. Returns a boolean array of a one-voxel-wide medial axis.

Known behaviour: the skeleton of a real aorta has spurs — short dead-end branches caused by surface irregularity, the same phenomenon Riffaud calls non-anatomic branches. Prune before use.

**Pruning.** Build a graph over skeleton voxels with 26-connectivity. Find endpoints (degree 1) and junctions (degree ≥ 3). For each endpoint, walk to the nearest junction; if the path length is less than `spur_prune_factor * local_radius`, delete it. Iterate until stable (usually 2–3 passes).

### C.3.2 Longest path

Build a sparse graph: nodes are skeleton voxels, edges connect 26-neighbours with weight equal to the physical distance between them (1, √2, or √3 times spacing, or the anisotropic equivalent).

`scipy.sparse.csgraph.dijkstra` twice: from an arbitrary skeleton voxel find the farthest node A; from A find the farthest node B. The A–B path is the centreline. (This is the standard tree-diameter trick. It is exact on a tree and close enough on a near-tree.)

Extract the path with `predecessors` from the second Dijkstra call.

**Arch check.** If the path's two endpoints are both at low z and the path has a high-z excursion, you have an arch. This is not an error. Do not assert monotonic z anywhere.

### C.3.3 Spline fit and resampling

`scipy.interpolate.splprep(path_xyz.T, s=smoothing, k=3)` then `splev` at parameter values that give 1 mm arc spacing.

Getting uniform arc length requires two passes: evaluate densely (say 10× oversampled), compute cumulative chord length, then interpolate parameter values at the desired arc positions and re-evaluate.

Choose `s` so the spline smooths voxelisation noise without cutting corners at the arch. Start with `s = 0.5 * N` where N is the number of path points, and inspect visually on a dev case. Too much smoothing shortens the arch and shifts the frame; too little leaves the tangent jittering by a few degrees per sample, which propagates into the wall map as a wobble.

**Tangent** from `splev(..., der=1)`, normalised.

### C.3.4 Radius profile

For each centreline sample, `radius_mm[i] = edt_mm` interpolated at that point (`map_coordinates`, order=1). Case 25: median 9.1 mm, max 15.7 mm, min 1.5 mm at the tapered ends.

### C.3.5 Surface and normals

`surface = mask & ~binary_erosion(mask, iterations=1)`.

Normals: compute `signed_edt = distance_transform_edt(mask, sampling=...) - distance_transform_edt(~mask, sampling=...)`, then `numpy.gradient` of that, normalised. Gradient points inward (increasing distance is inward), so negate for outward. Smooth with a small Gaussian (σ ≈ 1 voxel) before differentiating to suppress staircase artefacts.

Alternative: `skimage.measure.marching_cubes` gives vertices and vertex normals directly, with sub-voxel surface positions. Better geometry but introduces a mesh representation you then have to relate back to the voxel grid. The EDT-gradient approach is simpler and adequate; use marching cubes only if you find the normals too noisy at 1.5 mm.

### C.3.6 End-cap exclusion

Start with `eligible_wall = surface.copy()`, then clear:

**Test 1 — volume boundary.** Any surface voxel within 1 voxel of any ROI face. On case 25 this clears nothing (the mask sits at z = 147–375 of 421), which is exactly why the other two tests are mandatory rather than optional.

**Test 2 — normal parallel to tangent.** For each surface voxel, find the nearest centreline sample (build a KD-tree over centreline points with `scipy.spatial.cKDTree`). If that sample is within `3 * radius_mm` arc length of either centreline endpoint, compute the angle between the voxel's outward normal and the endpoint-directed tangent. Clear if under 35°.

**Test 3 — terminal plane.** Define a plane at each centreline endpoint with normal equal to the local tangent. Clear any surface voxel within 3 mm of that plane and within `1.5 * radius_mm` of the endpoint.

Case 25 reference: both terminal slices have ~82–84 voxels of area, roughly 185 mm², a full-calibre flat disc. If your exclusion is working, the wall map (Stage 4) shows a band of invalid pixels at each end and no peaks there.

**Verify visually before proceeding.** Render `eligible_wall` in 3D with the excluded voxels in a different colour. This is the most common silent failure in the pipeline and it costs you precision on every case.

### C.3.7 Rotation-minimising frame — `frames.py`

You need an orthonormal `(u, v)` at each centreline sample, both perpendicular to the tangent, varying smoothly with no twist.

**Do not** use a fixed global vector projected into the normal plane. `u = normalize(g - (g·t)t)` for fixed `g` degenerates when `t` becomes parallel to `g`, which is guaranteed somewhere on an arch, and produces a discontinuous jump in the map.

**Do not** use the Frenet frame. It flips at inflection points and is undefined where curvature is zero, which is most of a straight aortic segment.

**Use the double-reflection method** (Wang et al., "Computation of rotation minimizing frames", ACM TOG 2008). The algorithm propagates a frame from sample *i* to *i+1* with two reflections and is stable, second-order accurate, and about ten lines. Initialise `u_0` as any unit vector perpendicular to `t_0`.

**Anterior alignment.** After propagation, apply one global rotation so that θ = 0 points anterior. Compute the anterior direction in patient space (in LPS, anterior is −y), map it into the ROI index frame, project into the normal plane at a mid-centreline sample, and rotate the whole frame by the angle between that and `u`. Now θ maps to clock position with 12 o'clock anterior, which is what a surgeon reads.

Because the frame is rotation-minimising and not re-referenced per sample, the clock position stays consistent along the vessel even round the arch. Note in the display legend that clock position is defined relative to the local vessel frame, not the patient axial plane, since these diverge in the arch.

---

## C.4 `wallmap.py` and `detect.py` — candidate generation

### C.4.1 Map construction

Output is an `(N, B)` image: N centreline samples at 1 mm, B = 128 angular bins.

For each `(i, j)`:

1. Ray direction `w = cos(θ_j) * u_i + sin(θ_j) * v_i`.
2. **Find the wall crossing.** March from `c_i` along `w` in 0.25 mm steps, sampling the mask with nearest-neighbour interpolation. The wall is the last point where mask = 1 before a run of at least 1 mm of mask = 0 (the run requirement suppresses single-voxel holes). Record `R(i,j)` and the wall point.
3. **Validity check.** Query the centreline KD-tree with the wall point. If the nearest centreline sample is more than `frame_validate_arc_mm` (3 mm) of arc length from sample *i*, this ray has crossed to a different part of the vessel — which happens on the inner curvature of the arch, where normal planes from different samples intersect. Mark invalid.
4. **End-cap check.** If the wall point falls in an excluded region, mark invalid.
5. **Probe.** Sample the CT at 11 points from `R + 1 mm` to `R + 6 mm` along `w`. For each sample, average over a small disc of radius 1 mm perpendicular to `w` (4 offset samples plus the centre is enough) to suppress noise.
6. **Calcium exclusion.** Drop any probe sample above `t_calcium` from the mean. If more than half the samples are calcium, mark the pixel invalid rather than reporting a misleading low mean.
7. Store the mean as `intensity[i, j]`.

All sampling via `scipy.ndimage.map_coordinates(arr_zyx, coords_zyx, order=3, mode='nearest')`. Batch it: build the full `(N, B, S, 5, 3)` coordinate array and make **one** `map_coordinates` call. Per-point calls in a Python loop will dominate your runtime; one batched call on ~1 M sample points takes well under a second.

**Cap the ray length** at `ray_max_factor * radius_mm[i]` (2.5×). If no wall crossing is found within that, mark invalid. Prevents runaway rays in degenerate geometry.

### C.4.2 Why 1–6 mm

- Starting at 1 mm rather than 0 avoids partial-volume contamination from the aortic wall itself. At 1.5 mm spacing the wall voxel is a blend of lumen and surrounding tissue, and including it biases every pixel upward uniformly, compressing your dynamic range.
- Ending at 6 mm samples past the 5 mm eligibility distance, so a stub that dies at 3 mm yields a weaker mean than a real branch. The map partially pre-filters for eligibility.
- Elattar used 2.5 mm with a 0.75 mm radius for coronaries at 0.5 mm spacing. Scaling for 1.5 mm data and the 5 mm rule gives roughly this range.

**This is the highest-sensitivity parameter in the system. Tune it first.** Too short and you fire on wall irregularity; too long and you integrate over whatever happens to pass nearby.

### C.4.3 Peak detection

Operating on `intensity` with NaNs at invalid pixels.

1. **Circular padding.** The map is cyclic in θ. Pad by `angular_bins // 8` columns wrapped from the other side before any neighbourhood operation, then crop back. A branch at θ ≈ 0 must not split into two components.
2. **Fill NaNs** with a value below threshold (e.g. `t_bg`) so the maximum filter behaves, but keep the `valid` mask to reject any region whose peak lands on an invalid pixel.
3. **Local maxima**: `scipy.ndimage.maximum_filter(img, size=peak_footprint)` then `img == filtered`.
4. **Threshold** at `t_lumen`. A pixel is a candidate seed only if it is both a local max and above threshold.
5. **Region growing**: threshold the whole map at `t_lumen`, `scipy.ndimage.label` with 8-connectivity, keep components containing at least one local max and at least `min_region_px` pixels.
6. Each surviving component is one `Candidate`. Its `patch_points` are the wall points of its pixels; `patch_weights` are the map intensities.

**Two adjacent branches in one component.** If a component contains two local maxima separated by more than ~4 mm in physical distance (convert from map coordinates using `arc_length` and `R * Δθ`), split it with a watershed (`skimage.segmentation.watershed` on the negated intensity, seeded at the maxima). This is the paired-renal case and it is worth the extra step: Riffaud's series gives a mean inter-renal origin distance of 9.5 mm with SD 5.1, so pairs routinely sit close enough to merge in a naive labelling.

### C.4.4 Optional evidence fusion

If dev-set F1 plateaus, add layers and multiply:

- **Vesselness.** `skimage.filters.sato(image, sigmas=[1.5, 2.0, 2.8, 4.0], black_ridges=False)` computed once on the ROI, then sampled along the same probes and mapped to `(N, B)`. Sato is generally better behaved than Frangi for tubular structures and has one fewer parameter. Restrict computation to the dilated shell to save time.
- **Outward wall curvature.** The wall bulges outward at a real ostium. Compute mean curvature from the marching-cubes mesh or from the Hessian of the signed EDT, sample at wall points.

Multiply the normalised maps rather than averaging: multiplication requires all criteria simultaneously, which is the right behaviour when precision binds. This is Elattar's hinge-point construction.

Budget ~20 s for vesselness. Measure whether it improves F1 before keeping it.

---

## C.5 `trace.py` — proximal trace

### C.5.1 Subvolume extraction

For each candidate: take the patch centroid, extract a cube of half-width `subvolume_half_mm` (15 mm), resample to `subvolume_spacing_mm` (0.5 mm) with `map_coordinates(order=3)`.

At 0.5 mm a 30 mm cube is 60³ = 216 k voxels. Negligible memory, fast operations. This is the *only* place you upsample, and it is where sub-voxel precision actually matters.

Carry the affine relating subvolume indices to ROI continuous indices. Everything computed here maps back through it.

### C.5.2 Seed region

Rasterise the candidate's wall patch into the subvolume. Dilate by one voxel. This is the geodesic source, φ = 0.

Restrict propagation to `lumen = (ct > t_lumen) & (ct < t_calcium) & ~aorta_mask`. Excluding the aorta mask is important — otherwise the geodesic field floods back into the parent and every candidate connects to every other.

### C.5.3 Geodesic distance

**Preferred: fast marching.** `skfmm.distance(phi, dx=0.5, narrow=...)` where `phi` is a signed array with the source at zero, masked to the lumen region (`numpy.ma.MaskedArray` — scikit-fmm respects masks). Gives a first-order-accurate isotropic distance field.

**Fallback without the dependency: Dijkstra.** Build a sparse graph over the 216 k lumen voxels with 26-connectivity and physical edge weights, run `scipy.sparse.csgraph.dijkstra` from the source set. Slightly anisotropic (the classic √2/√3 lattice bias, ~5% overestimate on diagonals) but entirely adequate over a 10 mm distance, and it removes a dependency from the offline install. **Recommend this** unless you measure a real accuracy difference — one fewer vendored wheel is worth more than 5% on a diagonal.

Either way the output is `D`, geodesic distance in mm from the aortic wall, infinite outside the lumen.

### C.5.4 Level-set walk

For `d` in `0.5, 1.0, ..., 10.0`:

1. `band = (D >= d) & (D < d + 0.5)`
2. `lbl, n = scipy.ndimage.label(band, structure=ones((3,3,3)))`
3. Keep components 26-adjacent to the previous level's retained component(s). At `d = 0.5`, keep components adjacent to the source patch.
4. Record the centroid and `equivalent radius = sqrt(area_mm2 / π)` where area is taken on the band's cross-section — in practice use `(voxel_count * voxel_volume) / band_thickness` as the cross-sectional area estimate.

This gives 20 samples across the 10 mm budget, versus the three or four an iterative cylinder tracker would manage at 1.5 mm native spacing. That sample density is the whole reason for choosing this over cylinder fitting.

### C.5.5 Stop conditions

Evaluated in this order at each level:

**Leak.** `r(d) > leak_ratio * r(d - leak_lookback_mm)`. Ratio 1.5, lookback 3 mm. Real aortic daughters taper; anything that balloons is the trace entering a cardiac chamber, the IVC, or bowel. **Leaked candidates are discarded entirely, not truncated** — the leak indicates the candidate was never a branch. This is Wala's leak detector and it is the single highest-value borrowed rule, given that case 25's shell contained 38 bright components ≥5 voxels with the largest at 1788.

**Bifurcation.** The retained component at level `d` splits into ≥2 components at `d + 0.5` which remain separate through `d + split_persist_mm` (1.0 mm, i.e. two further levels). The persistence requirement suppresses noise-driven single-level splits. Record `stop_reason = 'bifurcation'` and the split location.

**Weak match.** Fewer than `strong_match_frac` (0.5) of the band's voxels exceed `t_lumen`, or the component has fewer than ~8 voxels. Adapted from Wala's strong-match criterion.

**Max.** `d >= 10 mm`. Record `stop_reason = 'max'`.

**Secondary radius check.** Wala's ratio test — current radius below 0.90 × radius at 3 mm upstream — is retained *only* as a diagnostic. A sharp drop without a topological split usually means the trace slipped off the vessel onto a neighbouring structure. Flag it; do not stop on it. At this spacing the radius estimate is too noisy to be a primary stop signal, which is why component splitting is the primary one.

### C.5.6 Eligibility

`eligible = (max reached distance >= 5 mm) and (stop_reason != 'leak')`.

Note this is **independent of the stop reason** otherwise. A trace stopping at a bifurcation at 7 mm is eligible. A trace dying weak at 3 mm is not.

### C.5.7 Sub-5 mm bifurcation

If the trace bifurcated before 5 mm but the lumen continues past 5 mm:

- Follow the **dominant child** — larger cross-sectional area at the split level — until 5 mm.
- Set `flags += ['seed_beyond_bifurcation']`.
- Keep `path_length_mm` as the distance to the split, since that is the honest proximal-path length, but extend the point list to the seed.

The spec is genuinely ambiguous here: the branch is eligible (5 mm of lumen exists), it is one instance (common-trunk rule), but the trace rule stopped before the seed distance. Dominant-child is the best guess because it is what a human tracing a centreline does. **Ask the organisers.** If you cannot, state the assumption in the README and demo. An explicit documented assumption reads far better than a silent one.

---

## C.6 `measure.py` — the four quantities

### C.6.1 Ostium centre

1. Weighted centroid of `patch_points` using `patch_weights` (map intensity as a proxy for how much lumen sits behind each wall pixel).
2. The centroid of points on a curved surface sits slightly inside it. Project back: find the nearest surface voxel with the KD-tree, take its outward normal, and move the centroid along that normal until the signed EDT is zero (a short bisection, 5 iterations, sub-0.05 mm).
3. Convert to physical mm via `CaseGrid.to_physical`.

**Do not use a centreline node.** Riffaud's branch location is the bifurcation node on the aortic centreline, one aortic radius inside the wall. On case 25 with a median radius of 9.1 mm that is a systematic ~9 mm error against a metric worth 25% of the score. Elattar's benchmark — 2.0–2.4 mm mean error against interobserver variation of 2.4–3.2 mm — is what you are aiming for, adjusted upward for 1.5 mm spacing.

### C.6.2 Direction

Riffaud's Definition 5, on the traced path:

1. `A = path_points - ostium` (an `(M, 3)` matrix).
2. `w, V = numpy.linalg.eigh(A.T @ A)`; take `e = V[:, -1]` (largest eigenvalue).
3. `d = sign(ones(M) @ A @ e) * e`.
4. Normalise. Emit as `direction_xyz`.

Step 3 is the critical one. An eigenvector's polarity is arbitrary, and the sign term forces it to point away from the ostium into the branch. Omitting it gives you a correctly-oriented line and a randomly-flipped vector on roughly half your detections.

This beats an ostium-to-seed difference vector because it is a least-squares fit over all 20 path samples rather than two noisy endpoints.

**Direction must be expressed in physical space, not index space.** If the ROI spacing is isotropic and the direction cosines are identity, index-space and physical-space directions agree up to a uniform scale and normalisation hides it. If the direction cosines are *not* identity — which they can be on real data — you must rotate. Safest: convert two path points to physical mm, take their difference, and do the PCA in physical coordinates from the start. Do the whole computation in physical mm and the problem disappears.

### C.6.3 Seed

1. Interpolate along the path to geodesic distance exactly 5 mm (linear between the bracketing level centroids).
2. **Recentre.** Build an orthonormal basis `(e1, e2)` perpendicular to `d`. Sample a plane at that point, ±6 mm at 0.25 mm. Threshold at the half-maximum (C.6.4). Take the connected component containing the centre. Compute the intensity-weighted centroid of that component and move the seed there.

Recentring is what guarantees the seed lies *on* the daughter, which is explicitly what the 15% quality category checks. A level-set centroid can sit slightly off-lumen where the band is asymmetric.

If the component containing the centre is empty (the seed fell off), fall back to the un-recentred point and flag `seed_recentre_failed`.

### C.6.4 Radius

On the same resampled cross-section at the seed:

1. `peak` = maximum within 1 mm of the centre.
2. `bg` = median of an annulus at 5–6 mm radius, excluding anything above `t_calcium`.
3. `half_max = (peak + bg) / 2`.
4. Threshold, label, take the component containing the centre.
5. `radius_mm = sqrt(area_mm2 / π)`.

**Why FWHM and not EDT.** An EDT on a binarised mask inherits your threshold's bias and quantises to the grid: at 1.5 mm native spacing the quantisation step is ~0.75 mm, which on a 2.5 mm-radius renal artery is a 30% error. FWHM on an interpolated cross-section is the standard sub-voxel vessel-calibre estimator and is what the 15% category is checking.

**Why not average along the path.** The spec asks for the local radius *at the seed*. Riffaud's mean-over-branch definition is the wrong quantity.

**Sanity bounds.** Clamp to [0.4, 8.0] mm. Anything outside suggests the cross-section caught a neighbour; flag it. Also check `radius_mm < 0.5 * local_aortic_radius` — a "daughter" as wide as the parent is a leak that escaped C.5.5.

### C.6.5 Confidence

Not required by the spec but valuable for the display and for the sweep harness. Compose from:

- normalised map peak value: `(peak - t_lumen) / (mu_ao - t_lumen)`, clipped to [0, 1]
- path completeness: `min(path_length_mm / 10, 1)`
- taper consistency: `1 - |r(5mm) - r(1mm)| / r(1mm)`, clipped
- optionally, normalised vesselness at the seed

Multiply, then clip. Report as `confidence`. Do not threshold on it for the main output — that decision belongs in Stage 7 where it can be tuned against F1 — but do use it to order the display and to drive the flags.

---

## C.7 `resolve.py` — instance resolution

Apply in this order. Each rule sets `rejected_by` rather than deleting, so the sweep harness can attribute recall loss.

### C.7.1 Merge by wall-patch contiguity

Two candidates merge **only if** their wall patches are contiguous on `eligible_wall` — i.e. there is a path between them through surface voxels that are themselves above the map threshold.

Implement on the map, not in 3D: two map regions merge if they are 8-adjacent after circular padding. Since Stage 4 already labelled connected components, this means: do not merge anything Stage 4 separated, and do not re-merge what the C.4.3 watershed split.

**Never merge on centroid proximity.** The spec requires two nearby origins to be returned as two instances when they are separate at the aortic wall. Riffaud's 239-case series gives a mean inter-renal origin distance of 9.5 mm (SD 5.1, minimum well below). Any proximity merge with a sensible radius silently halves your renal recall — and renals are a large fraction of the branches in an abdominal segment.

The common-trunk case needs no rule here: the C.5.5 bifurcation stop already terminates the trace at the split while keeping exactly one ostium.

### C.7.2 Reject daughters-of-daughters

Process candidates in descending order of wall-patch area. For each, check whether its geodesic path back to the aorta passes through voxels already claimed by an earlier-processed candidate's trace before reaching `eligible_wall`.

Implementation: maintain a cumulative `claimed` boolean array in ROI space. After accepting a candidate, rasterise its traced lumen into it (dilated by 1 mm). For each subsequent candidate, if more than ~30% of its first 2 mm of trace overlaps `claimed`, reject with `rejected_by = 'secondary_branch'`.

The area ordering matters: the true parent should be established first. Ties broken by confidence.

### C.7.3 Reject non-arterial structures

Three tests, all must pass:

- **Contact brightness.** Median CT over the wall-patch voxels' outward 1–2 mm must exceed `t_lumen`. A vein abutting the aorta has no bright connection *at the wall*.
- **HU band.** Mean CT over the traced lumen within `hu_band_sigma * sigma_ao` of `mu_ao`. In a well-timed CTA this cleanly separates arteries from veins; in a poorly timed one it will not, which is why `degenerate` cases get flagged.
- **Tubularity.** If vesselness is computed, require it above a fitted threshold at the seed. Otherwise use an eccentricity proxy: the ratio of the cross-section's major to minor axis from `skimage.measure.regionprops` should be under ~3.

### C.7.4 Reject artefactual bumps

Riffaud's feature space: plot `path_length_mm` against `path_length_mm / local_aortic_radius_mm`. Real arteries and surface artefacts separate near-linearly (their Fig. 12).

Fit a boundary on the dev subset — a simple linear discriminant or a hand-placed line is sufficient and more defensible than a learned model on a handful of cases.

**Use the feature space, not their constants.** Their rule — length < 3r, or length < r + 20 mm — was tuned for named major arteries and would delete essentially every eligible daughter under a 5 mm rule. Relative length is the discriminative axis: a bump on a 15 mm-radius aorta and one on a 6 mm-radius aorta are not comparable in absolute millimetres.

### C.7.5 ID assignment

Order surviving candidates by `arc_length` of their nearest centreline sample. Assign `branch_001`, `branch_002`, …

**Not by z.** Superior-to-inferior ordering is ambiguous across an arch — case 25's mask covers one. Arc length is well-defined for any coverage and makes IDs stable and reproducible across runs.

---

## C.8 `output.py` — serialisation

### C.8.1 Coordinate chain

Every emitted point traverses: subvolume continuous index → ROI continuous index (via the subvolume affine) → original continuous index (add `crop_origin_idx`, scale by spacing ratio if C.1.2 resampled) → physical mm via `sitk_ref.TransformContinuousIndexToPhysicalPoint`.

Write this as **one function**, used everywhere. Assert in it that the input is `(x, y, z)` ordered.

Round to 3 decimal places on output — sub-micron precision is noise and makes diffs unreadable.

### C.8.2 Schema

Mandated fields exactly as specified. Additive fields (`confidence`, `path_length_mm`, `clock_position`, `arc_length_mm`, `flags`, `stop_reason`) do not break a schema check and feed the display.

Include a top-level `meta` block: software version, config hash, runtime seconds, peak memory MB, case-level flags, number of candidates generated and rejected by each rule. This costs nothing and is exactly what the reproducibility category rewards.

### C.8.3 Empty case

`"daughters": []` must serialise correctly and the process must exit 0. Test this explicitly with a synthetic mask that has no branches — it is an easy path to leave broken, since every real case has branches.

Also handle: empty mask, mask with no valid centreline (fewer than ~10 skeleton voxels), all candidates rejected. All produce an empty list plus a flag, never a traceback.

### C.8.4 Determinism

Same input must give byte-identical output. Sources of non-determinism to eliminate:

- `scipy.ndimage.label` component ordering is deterministic; `skimage.segmentation.watershed` is too, given the same seeds.
- Any iteration over a Python `set` or `dict` built from array values — sort explicitly.
- Multi-threaded reductions in numpy with non-associative float addition. Not usually a problem at this scale; if you parallelise, parallelise over candidates and merge in sorted order.

Add a test that runs a case twice and diffs the JSON.

---

## C.9 `viz.py`

### C.9.1 Unrolled wall map — the clinician view

Render the Stage 4 `intensity` map directly: arc length on the vertical axis, clock position on the horizontal.

- Grayscale or a perceptually-uniform colormap for the intensity.
- Invalid pixels in a distinct flat colour (light grey), with the end-cap bands labelled.
- Each accepted ostium as a circle of radius proportional to `radius_mm`, with a short arrow for the direction projected into the map plane.
- Rejected candidates as faint hollow markers with their `rejected_by` reason in a legend — this makes the figure a debugging tool as well as a display.
- Horizontal axis labelled in clock positions (12, 1, 2, …), vertical in mm from the superior end of the supplied segment.
- Case-level flags as a banner.

The value here is that clock position and longitudinal distance from a reference are the two numbers that go on a fenestrated stent-graft order form. And this *is* the working representation the detector ran on, not a separate rendering — the clinician sees what the algorithm saw.

Add a legend note that clock position is defined in the local vessel frame, which diverges from the patient axial plane in the arch.

### C.9.2 Verification figures (≥3 cases, mandated)

**3D render.** `skimage.measure.marching_cubes` on the mask → `matplotlib` `Poly3DCollection` at low alpha, plus ostium markers and direction arrows via `quiver`. Decimate the mesh (`skimage.measure.decimate` or simple vertex subsampling) to keep the file small and rendering fast. Three views: anterior, lateral, oblique.

**Orthogonal MIP slabs.** For each ostium, a 3 × 1 panel: axial, coronal, sagittal MIP over a 10 mm slab centred on the ostium, with the ostium marked and the direction arrow overlaid. This lets a reader confirm the branch is actually there, which the 3D render alone does not.

Backend: `matplotlib.use('Agg')` before any pyplot import. Output PNG at 150 dpi, or bundle into a single self-contained HTML with base64-embedded images. No GPU, no interactive backend, no server.

---

# Part D — Cross-cutting concerns

## D.1 Timing and memory

Decorate each stage with a context manager that records wall time and peak RSS (`resource.getrusage(RUSAGE_SELF).ru_maxrss` on Linux, or `tracemalloc` for Python-level allocation).

Emit to the `meta` block and to a log line. You have to report runtime in the demo, and the 10% compute category is scored on the organisers' hardware — knowing your own per-stage breakdown is how you find what to cut if you are over.

Target breakdown on 4 cores:

| Stage | Budget |
|---|---|
| Load, crop, calibrate | 3 s |
| Skeleton, longest path, spline, EDT, surface, end caps, frame | 6 s |
| Wall map (batched interpolation) | 8 s |
| Peak detection | <1 s |
| Per-candidate trace, 5–15 candidates | 8 s |
| Measurement | 2 s |
| Output + figures | 5 s |
| **Total** | **~33 s** |

Vesselness adds ~20 s and is the first cut if you are over.

## D.2 Logging

`logging` at INFO for stage boundaries and counts, DEBUG for per-candidate detail. Never print to stdout — the CLI's stdout should stay clean. Write a per-case log file next to the output when `--viz-dir` is given.

Log the things you will want when a hidden-set case behaves oddly: calibration values, centreline length and endpoint positions, number of map pixels marked invalid and why, candidate count before and after each rejection rule.

## D.3 Error handling

The pipeline must never traceback on the hidden set. Wrap `process_case` so that:

- A failure in an optional stage (vesselness, figures) is caught, logged, flagged, and the pipeline continues.
- A failure in a required stage produces a valid JSON with an empty daughters list, a `pipeline_error` flag, and the exception text in `meta`. Exit 0.
- A per-candidate failure rejects that candidate with `rejected_by = 'exception'` and continues with the rest.

A crashed case scores zero on everything. A case that emits an empty list scores zero on discovery but keeps reproducibility intact and does not abort a batch run.

## D.4 Numerical hygiene

- float32 for volumes, float64 for point arithmetic and the PCA.
- Guard every normalisation: `norm = max(norm, 1e-9)`.
- Guard every division by a radius or count.
- `numpy.errstate(invalid='ignore')` around deliberate NaN arithmetic in the wall map, but never globally.
- Check for NaN in every emitted coordinate before serialisation and reject the candidate if found.

---

# Part E — Testing

## E.1 Synthetic phantom — build this early

`tests/phantoms.py` generates a volume with known ground truth. This lets you test the entire pipeline before you have annotated dev data, and it isolates algorithmic error from data messiness.

Generate:

- A cylinder (or gently curved tube) of radius 9 mm along a prescribed centreline, filled at 485 HU.
- N side branches at prescribed arc lengths, clock angles, radii (1.5–4 mm) and directions, filled at the same intensity.
- Additive Gaussian noise at σ = 30 HU.
- Background at −50 HU, plus a few bright distractors *not* connected to the tube.
- Rasterise at 1.5 mm isotropic with a known affine including non-identity direction cosines.
- Flat-cut both ends, positioned in the volume interior.

Assertions:

- All N branches detected, no false positives from the distractors or the end caps.
- Ostium error under 2 mm.
- Direction angular error under 10°.
- Radius error under 0.5 mm.
- Output coordinates round-trip to the prescribed physical positions.

Variants to generate: a curved/arched phantom, a phantom with two branches 8 mm apart (must yield two instances), a phantom with a trunk that bifurcates at 3 mm (must yield one instance with the `seed_beyond_bifurcation` flag), a phantom with a branch that dies at 3 mm (must be rejected), a phantom with a branch that opens into a large blob at 6 mm (must be leak-rejected), and a phantom with zero branches (must emit an empty list).

That last set maps one-to-one onto the spec's "Important cases" list. Being able to demonstrate them in the five-minute demo is worth real marks.

## E.2 Unit tests

- **Coordinates** (C.0.4). Run on every case as an assertion, not just in tests.
- **Frame.** Propagate along a helix and assert the frame stays orthonormal and shows no net twist; assert it does not degenerate where the tangent passes through the global up vector.
- **End caps.** On the phantom, assert zero candidates within 5 mm of either cut face.
- **Circular wrap.** Place a phantom branch at θ = 0 and assert it yields one candidate, not two.
- **Schema.** Validate output against a JSON schema; assert all mandated fields present, `direction_xyz` is unit to 1e-6, `parent_instance_id == "aorta"`, IDs unique.
- **Determinism.** Same input twice, identical bytes.

## E.3 Real-data checks

On each dev case, assert and log:

- Calibration values within plausible ranges.
- Centreline length consistent with the mask's z-extent (case 25: 342 mm of z-extent, so expect a centreline longer than that if there is an arch).
- Fraction of wall-map pixels marked invalid — if above ~20%, something is wrong with the frame or the ray casting.
- Candidate count before and after each rejection rule.

## E.4 Offline install test

Build a container with networking disabled, run the README's setup command and then the run command on a dev case. This is the exact scenario the organisers will execute. A surprising number of submissions fail here.

---

# Part F — Tuning harness

## F.1 `tools/evaluate.py`

Implement the organisers' scoring as closely as the spec allows so you are optimising the right thing.

- **Matching.** One-to-one, greedy or Hungarian on ostium distance, with a match radius (try 5 mm and 10 mm and report both — the organisers have not published theirs). Duplicates count as false positives.
- **Discovery.** Precision, recall, F1.
- **Localisation.** Mean and median ostium distance over matched pairs.
- **Quality.** Seed-on-daughter (distance from seed to the reference proximal centreline), direction angular error, radius absolute and relative error.

Report per-case and aggregate. Also report a breakdown of false negatives by rejection rule, using the `rejected_by` field — this tells you which rule is costing recall, which is the most actionable diagnostic in the whole project.

## F.2 `tools/sweep.py`

Wala's methodology: one parameter at a time, a table of range and step, maximising a single score.

Score: F1 for the discovery-dominated sweep, or a weighted composite `0.45*F1 + 0.25*loc_score + 0.15*qual_score` mirroring the rubric.

Sweep order, by sensitivity:

1. `wallmap.probe_inner_mm` / `probe_outer_mm` — the highest-sensitivity pair. Sweep the outer bound over 4–8 mm, the inner over 0.5–2 mm.
2. `detect` threshold offset relative to `t_lumen` — try `t_lumen ± k*sigma_ao` for k in −1…+1.
3. `resolve.bump_boundary` — fit rather than sweep.
4. `trace.leak_ratio` — 1.3 to 2.0.
5. `geometry.endcap_normal_deg` — 25° to 45°.

Hold out 2–3 dev cases entirely from fitting so you have an honest estimate before the hidden set. With only ~25 total cases and a small annotated subset, overfitting is a real risk; prefer fewer, coarser parameters over many fine ones.

Wala's finding is worth internalising: their cylinder height had the largest effect, and both too-small and too-large values degraded performance sharply. Expect the same shape from the probe range — plot the curve rather than just taking the argmax, and pick a value in the middle of a flat region rather than on a narrow peak.

## F.3 The precision/recall operating point

Wala states the tradeoff directly: relaxing the bifurcation model gives few missed segments but more false positives.

Sweep a global confidence threshold and plot the PR curve on the dev set. With one-to-one matching and duplicates counting against you, lean conservative. Report where you chose to sit and why — that reasoning is exactly what the demo should contain.

---

# Part G — Deliverables

## G.1 README

Must contain exactly one setup command and one run command, per the spec.

```
## Setup
pip install --no-index --find-links vendor -r requirements.txt

## Run
python run.py --image image.nii.gz --aorta-mask aorta_mask.nii.gz --output prediction.json
```

Then: method summary in a paragraph, the parameter table with provenance, runtime and memory measured on your own hardware with the spec stated, known failure modes, the `seed_beyond_bifurcation` assumption stated explicitly, and a pointer to the dev-set predictions and visual checks.

Keep the two commands literally first and literally alone under their headings. Anything that makes an evaluator hunt for the command costs you reproducibility marks.

## G.2 Dev-set predictions

One JSON per dev case, in a `predictions/` directory, generated by the committed code with the committed config. Regenerate them as the last step before submission so they cannot drift from the code.

## G.3 Five-minute demo

Suggested structure:

- 0:00 — the problem in one sentence and the key constraint: 1.5 mm spacing, a 5 mm eligibility rule, an unknown number of unnamed branches.
- 0:30 — the pipeline in one diagram.
- 1:30 — the unrolled wall map, live, showing that the display and the detector are the same object.
- 2:30 — the phantom suite demonstrating each of the spec's "Important cases": end caps, two nearby origins, common trunk, daughter-of-daughter, absent vessel.
- 3:30 — dev-set numbers, the PR curve, and where you chose to operate.
- 4:15 — known failure modes, named honestly, with the literature benchmarks for context (Elattar 2.0–2.4 mm ostium localisation against 2.4–3.2 mm interobserver; Tahoces 91.8/98.8 recall/precision; Riffaud 70.8% on polar renals even with a full segmentation).
- 4:45 — runtime and memory.

The honesty in the second-to-last section is worth more than an extra point of F1. Every paper reviewed names its own failure modes; matching that standard signals you understand the problem rather than just the dataset.
