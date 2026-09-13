# Branchseed — End-to-End System Design

**Task.** Given a CT volume and a binary mask of the parent aorta, detect every eligible artery that leaves the aorta directly, and return each as a separate daughter instance with ostium centre, seed, local radius and initial direction in SimpleITK physical millimetres.

---

## 0. Design principles

Five facts drive every decision below.

**The mask is a calibration source, not just an anchor.** It tells you this patient's contrast level, this patient's vessel-wall appearance, and this patient's vessel calibre. Almost every threshold should be derived from it rather than hard-coded. Case 25 gave mean 485 HU with SD 32 inside the lumen; another scanner or contrast timing could easily give 300 or 600. A fixed threshold would fail on the hidden set. Elattar's paper names exactly this as its own unvalidated limitation — that accuracy was never tested against different institutions' blood Hounsfield values. Deriving from the mask removes the risk.

**Resolution is the binding constraint.** Case 25 is 1.5 mm isotropic. The 5 mm eligibility distance is 3.3 voxels; the 10 mm trace cap is 6.7. A 2.5 mm-radius renal artery is about 3 voxels across. Every measurement must be sub-voxel, by interpolation, or it will quantise into uselessness.

**The trace is short, so tracking machinery is the wrong tool.** Ten millimetres is shorter than Wala's optimised cylinder height of 15 mm. Iterative cylinder trackers are built for following vessels across many generations; you need one stub. Use a dense, fine-grained representation of a short distance instead.

**Precision is the hard half.** A 6 mm shell around case 25's mask, thresholded at the calibrated lumen value, yields 106 connected components, 38 of them ≥5 voxels, the largest at 1788 — heart chambers, great vessels, bowel contrast. Detection is 45% of the score with one-to-one matching, so duplicates and leaks cost twice.

**Coverage is unpredictable.** Case 25's mask runs from the aortic arch down to just above the iliac bifurcation, not the abdominal segment the brief describes. Both cut faces sit in the volume interior, not at its boundary. An axial slice through the arch contains two disconnected aortic cross-sections. No per-slice reasoning, no superior-inferior ordering, no abdominal anatomical priors.

---

## 1. Pipeline

### Stage 0 — Load

Sniff the first two bytes for the gzip magic number `1f 8b` rather than trusting the extension. The uploaded case-25 files are gzip-compressed but named `.nii`; `nibabel.load` raises `ImageFileError` on them. This is a five-line fix that would otherwise be a zero-score crash on the hidden set.

Read with **SimpleITK**, not nibabel. The spec mandates SimpleITK physical coordinates, which are LPS. Nibabel reports RAS, and the x and y signs differ. Prototype however you like, but every emitted coordinate goes through `TransformContinuousIndexToPhysicalPoint`. Add a unit test asserting a known voxel round-trips to the expected millimetre triple.

Verify the mask shares the image's size, spacing, origin and direction cosines. If not, resample the mask onto the image grid with nearest-neighbour and log a warning.

### Stage 1 — Crop and normalise the grid

Compute the mask bounding box, dilate by 30 mm in physical units, crop both volumes.

This is the single biggest compute win. Case 25's mask bbox is 39 × 65 × 229 voxels against 236 × 236 × 421 in the volume — roughly 2.5%. Even after dilation you are working on a few million voxels instead of 23 million.

Resample the crop to isotropic spacing if the input is anisotropic; leave it at native spacing if already isotropic. **Do not globally upsample.** Sub-voxel precision comes from on-demand interpolation (`scipy.ndimage.map_coordinates`, cubic) at exactly the points you need, and from small per-candidate subvolumes in Stage 5. Upsampling the whole ROI to 0.5 mm would mean ~60 M voxels and 240 MB per array, which is affordable but wasteful when you only ever sample a thin shell and a few short paths.

Record the crop offset. Keep at most two full ROI arrays alive; float32 throughout.

### Stage 2 — Calibrate from the aorta

Erode the mask by 2 mm, take the HU distribution of the interior.

| Quantity | Definition | Case 25 |
|---|---|---|
| `mu_ao`, `sigma_ao` | mean, SD of eroded interior | 485, 32 |
| `T_lumen` | 5th percentile of interior HU | 433 |
| `T_calcium` | `mu_ao + 4·sigma_ao` | ~613 |
| `T_bg` | median of non-lumen shell voxels | — |

`T_lumen` is the contrast threshold for everything downstream. `T_calcium` masks out calcified plaque before any probing — Elattar masks high intensities due to calcification in their segmentation step, and calcium sits exactly where you are looking for ostia and is brighter than contrast. Without this, mural calcification produces confident false positives.

The tight SD in case 25 confirms that percentile-based calibration is stable on a well-timed CTA. Guard against a degenerate case (SD very large, suggesting the mask includes thrombus or the scan is poorly timed) by falling back to a wider threshold and raising a case-level flag.

### Stage 3 — Aortic geometry

**Centreline.** 3D skeletonise the mask (`skimage.morphology.skeletonize_3d`), prune spurs shorter than the local radius, keep the longest path between the two extreme endpoints, fit a smoothing spline, resample at 1 mm arc length. This handles the arch correctly — it is still a single curved tube — where per-slice centroids would fail.

**Radius profile.** Euclidean distance transform of the mask; `r(s)` is the EDT value at each centreline sample. Needed for scale-relative thresholds and for the unrolled display.

**Surface and normals.** `surface = mask XOR binary_erosion(mask)`. Outward normals from the gradient of the signed EDT, or from marching-cubes vertex normals.

**End-cap exclusion — mandatory.** Three tests, applied together:

1. Surface voxels on or within one voxel of the ROI/volume boundary.
2. Surface voxels whose outward normal is within 35° of the local centreline tangent, evaluated near the two centreline endpoints.
3. Surface voxels within 3 mm of the terminal cross-sectional plane at each endpoint.

Test 1 alone is insufficient and would catch nothing on case 25, where the mask occupies z = 147–375 of 421 slices. Tests 2 and 3 are the primary mechanism. Both caps there are full-calibre (~185 mm²) and a naive detector fires on them immediately.

The output is `eligible_wall`. Every candidate must attach to it.

**Rotation-minimising frame.** To unroll the wall in Stage 4 you need a reference direction at each centreline sample. Do not use a fixed global vector projected into the normal plane — it degenerates where the centreline turns parallel to it, which is guaranteed at the arch. Use parallel transport (a rotation-minimising frame) propagated along the centreline, then apply a single global rotation at the end so that θ = 0 corresponds to anterior. This gives a twist-free map, and the clock positions it produces are the ones a surgeon expects.

### Stage 4 — Candidate generation: the unrolled wall map

This is Elattar's coronary ostium detector, stripped of its anatomical prior.

Build a 2D image indexed by arc length *s* along the centreline (rows, 1 mm) and angle θ around it (columns, 128 bins ≈ 2.8°). Each pixel holds the mean CT intensity along a short outward probe starting at the aortic surface, with calcium-masked voxels excluded from the mean.

Elattar used a 2.5 mm probe of 0.75 mm radius. Scale to your problem: **probe from 1 mm to 6 mm outward**, radius ~1 mm. Starting at 1 mm avoids partial-volume contamination from the aortic wall itself; ending at 6 mm samples beyond the 5 mm eligibility distance so a branch that stops short produces a weaker response. Sample by interpolation, not voxel lookup.

Three departures from the paper, all forced by the open-set problem:

- **No Gaussian weighting.** Their prior centres on the distal extent because coronary ostia are always near the sinotubular junction. You have no positional prior.
- **No 1D max projections.** They collapse each axis and read two peaks off one curve and one off the other. That works only because there are exactly two ostia at roughly the same level. Your branches sit at many levels; projecting would smear them together. Use genuine **2D local-maximum detection**: maximum filter, threshold at `T_lumen`, connected-component labelling of the suprathreshold regions. Each region is a candidate with both coordinates read directly.
- **Wrap θ.** The map is cyclic in the column axis; a branch at 0° must not be split into two components. Pad circularly before labelling.

Each candidate carries: peak (s, θ), the extent of its suprathreshold region, the corresponding 3D wall patch on `eligible_wall`, and peak intensity.

**Optional evidence fusion.** Elattar's hinge-point branch multiplies three 2D maps and thresholds the product, which enforces all criteria at once rather than averaging them away. The analogous stack here is mean outward intensity × vesselness × outward wall curvature. Multiplication is the right operator when precision binds. Treat this as an experiment: build the intensity map first, measure F1 on the dev subset, and only add layers that improve it. Multiscale Frangi/Sato over scales 1.5–4 mm is the most expensive component in the pipeline and may not earn its runtime.

### Stage 5 — Proximal trace

For each candidate, extract a local subvolume (about 30 mm cube) around the wall patch and upsample it to **0.5 mm** with cubic interpolation. Small, cheap, and this is where sub-voxel precision is actually needed.

Compute a geodesic distance field outward from the wall patch, restricted to voxels above `T_lumen`. Fast marching, or a masked EDT. Geodesic rather than Euclidean matters: a branch that curves sharply would fail a straight-line 5 mm test.

Walk level sets outward in 0.5 mm steps, taking the connected component belonging to this candidate. That gives **20 samples across the 10 mm budget** instead of the three or four an iterative cylinder tracker would manage at this spacing. Record each level's centroid and equivalent radius.

Stop on the first of:

- **Bifurcation.** The component at level *d* splits into two or more sub-components that remain separate at *d* + 0.5. This detects the topological event itself rather than inferring it from a noisy scalar. Wala's radius-ratio test (current radius below 0.90 × radius half a cylinder-length upstream) is retained only as a secondary check: a sharp area drop without a split usually means the trace slipped off the vessel.
- **10 mm**, the spec cap.
- **Leak.** Equivalent radius exceeds 150% of its value one probe-length upstream. This is Wala's leak detector and it is the single most valuable borrowed rule. Real aortic daughters taper; anything that balloons is the trace falling into a cardiac chamber, the IVC, or bowel. Leaked candidates are discarded, not truncated.
- **Weak match.** Fewer than half the component voxels above `T_lumen`, adapting Wala's strong-match criterion.

Record `path_length_mm`.

**Eligibility** is a separate test from the stop condition: the lumen was followable to at least 5 mm without leaking. A trace that stopped at a bifurcation at 7 mm is eligible; one that died at 3 mm is not.

### Stage 6 — The four required quantities

**Ostium centre.** Intensity-weighted centroid of the wall contact patch, projected onto the aortic surface along the local outward normal so the point lies on the wall — not inside, not outside.

Do **not** use a centreline node. Riffaud's branch location is the bifurcation node on the aortic centreline, which sits one aortic radius inside the wall. On case 25, with a median local radius of 9 mm, that is a systematic 9 mm error against a metric worth 25% of the score.

**Direction.** Riffaud's Definition 5, applied to the proximal path. Build A with rows (pᵢ − ostium), take the eigenvector **e** of AᵀA with the largest eigenvalue, fix polarity with

```
d = sign(1ᵀAe) · e
```

then normalise. The sign term is what makes this work — an eigenvector's polarity is arbitrary, and this forces it to point away from the ostium into the branch. Better than an ostium-to-seed difference vector because it is a least-squares fit over all 20 path samples rather than two noisy endpoints.

Use the truncated proximal path only. Branches curve; the spec asks for the *initial* direction.

**Seed.** The path point at geodesic distance 5 mm, then recentred: resample a plane perpendicular to **d** at that location (~0.25 mm sampling) and take the intensity-weighted centroid of the lumen cross-section. Recentring is what guarantees the seed lies *on* the daughter, which is explicitly what the 15% quality category checks.

**Radius.** One FWHM measurement on that same resampled cross-section. Threshold at the half-maximum between the local lumen peak and local background, take the connected region containing the seed, report `sqrt(area/π)`.

Do not use EDT on a binarised mask — it inherits your threshold's bias and quantises to the grid. Do not average along the path; the spec asks for the local radius *at the seed*. Riffaud's mean-over-branch definition is the wrong quantity here.

**The sub-5 mm bifurcation case.** A trunk that splits at 3 mm is eligible (5 mm of lumen exists), is one instance (common-trunk rule), but its trace stopped before the seed distance. Follow the **dominant child** — the larger-area sub-component — to reach 5 mm, since that is what a human tracing a centreline does naturally. Record a `seed_beyond_bifurcation` flag. Ask the organisers if you can; if not, state the assumption in the demo. Making it explicit is worth more than guessing silently.

### Stage 7 — Instance resolution

This is where the 45% is won or lost. Four rules, in order.

**1. Merge only by wall-patch contiguity.** Two candidates merge only if their contact patches are contiguous on `eligible_wall`. Never merge on centroid proximity.

The spec requires two nearby origins to be returned as two instances when they are separate at the aortic wall. Riffaud's 239-case series gives a mean inter-renal origin distance of 9.5 mm with SD 5.1 and a minimum well below that. Any proximity-based merge with a sensible radius would silently halve your renal recall. Contiguity is the criterion the spec actually states.

The common-trunk case needs no separate rule: the Stage 5 bifurcation stop already terminates the trace at the split while keeping one ostium.

**2. Reject daughters-of-daughters.** If a candidate's geodesic path back to the aorta passes through another candidate's traced voxels before reaching `eligible_wall`, drop it. Process candidates in descending order of wall-contact area so the true parent is established first.

**3. Reject non-arterial structures.** Require a bright contact patch (contact voxels above `T_lumen`), a component mean HU within a band of `mu_ao`, and adequate tubularity. In a well-timed CTA the arterial/venous HU gap does most of the work; the contact-patch test and the leak detector do the rest.

**4. Reject artefactual bumps.** The aorta is not a perfect cylinder, and surface irregularity generates spurious protrusions. Riffaud characterises this well: plot traced length against length divided by local parent radius, and real arteries separate from artefacts almost linearly (their Fig. 12).

Use the **feature space, not their constants**. Their test — length < 3r, or length < r + 20 mm — would delete nearly every eligible daughter under a 5 mm rule. Fit your own boundary on the dev subset. Relative length is the discriminative axis: a bump on a 15 mm aorta and one on a 6 mm aorta are not comparable in absolute millimetres.

Assign IDs `branch_001`, `branch_002`, … ordered by **arc length along the centreline**, not by z. Superior-to-inferior ordering is ambiguous across the arch; arc length is well-defined for any coverage and makes outputs stable across runs.

### Stage 8 — Output

Map every point: local subvolume continuous index → ROI index → original image index → physical mm via `TransformContinuousIndexToPhysicalPoint`. Never emit voxel indices.

```json
{
  "case_id": "subject001",
  "parent": { "instance_id": "aorta" },
  "daughters": [
    {
      "instance_id": "branch_001",
      "parent_instance_id": "aorta",
      "ostium_xyz_mm": [12.4, -31.8, 184.6],
      "seed_xyz_mm": [15.1, -29.7, 181.2],
      "radius_mm": 2.7,
      "direction_xyz": [0.56, 0.43, -0.71],
      "confidence": 0.87,
      "path_length_mm": 9.5,
      "clock_position": "11:30",
      "arc_length_mm": 142.3,
      "flags": []
    }
  ]
}
```

The first six fields are mandated. The rest are additive — extra keys do not break a schema check, and they feed the visualisation and the clinician display. `clock_position` and `arc_length_mm` in particular are the two numbers that matter for stent-graft planning.

An empty `daughters` list must produce valid JSON, not a crash. Test that path explicitly.

---

## 2. Visualisation

### The clinician-facing view: unrolled aortic wall map

The spec asks for information displayed "in a unique way that will be useful for clinicians", separately from the three required verification figures.

Render the Stage 4 map directly: arc length along the centreline on one axis, clock position (12 o'clock = anterior) on the other, each detected ostium as a disc sized by `radius_mm` with a short vector for the projected direction.

This is genuinely what a vascular surgeon needs for fenestrated stent-graft planning — circumferential clock position and longitudinal distance from a reference point are the two numbers that go on the device order form. It turns the output into something with a clinical use rather than a debug view.

The elegant part is that this *is* the working representation the detector runs on, not a separate rendering. The clinician sees the same picture the algorithm saw.

Overlay case-level flags on the display: "two origins 2.1 mm apart — verify separation", "detection within 5 mm of cropped superior face", "low confidence, faint lumen". Riffaud does exactly this with their 75 mm warning, and it is the cheapest way to be useful rather than falsely certain.

### Required verification figures (≥3 cases)

- 3D render: mask surface, ostium markers, direction arrows (matplotlib or a static trimesh/VTK snapshot).
- Orthogonal MIP slabs through each detected ostium so a reader can confirm the branch is really there.

Static PNG or a self-contained HTML page. No GPU, no server.

---

## 3. Parameter table

Every value is either derived from the mask at runtime or fitted once on the dev subset and frozen. One config file, no case-specific edits.

| Parameter | Value | Source | Sensitivity |
|---|---|---|---|
| ROI dilation | 30 mm | fixed | low |
| Local subvolume upsampling | 0.5 mm | fixed | medium |
| Erosion for calibration | 2 mm | fixed | low |
| `T_lumen` | interior p5 | **derived** | high |
| `T_calcium` | `mu_ao + 4·sigma_ao` | **derived** | medium |
| Probe range | 1–6 mm outward | fixed | **high — tune first** |
| Probe radius | 1 mm | fixed | medium |
| Angular bins | 128 | fixed | low |
| Arc-length sampling | 1 mm | fixed | low |
| End-cap normal angle | 35° | fixed | medium |
| End-cap plane distance | 3 mm | fixed | medium |
| Geodesic step | 0.5 mm | fixed | low |
| Leak ratio | 1.5 | Wala | medium |
| Radius-ratio secondary check | 0.90 | Wala | low |
| Strong-match fraction | 0.5 | Wala | medium |
| Bump rejection boundary | fitted | Riffaud feature space | **high** |
| Map peak threshold | `T_lumen` | **derived** | **high** |

Sweep the three high-sensitivity fitted parameters one at a time on the dev subset, maximising F1, following Wala's methodology: a small table of parameter / range / step / optimised value, with a held-out subset. That discipline is directly presentable in the README and the demo and feeds the 5% reproducibility category.

---

## 4. Constraint compliance

**Runtime (10% of score).** Target well under 60 s on four cores.

| Stage | Budget |
|---|---|
| Load, crop, calibrate | 3 s |
| Skeleton, EDT, surface, end caps | 6 s |
| Unrolled map (interpolated probes) | 8 s |
| 2D peak detection | <1 s |
| Per-candidate trace (n ≈ 5–15) | 8 s |
| Outputs and figures | 5 s |
| **Total** | **~30 s** |

Vesselness, if used, adds ~20 s and is the first thing to cut. Parallelise scales across cores. Log per-stage timings — you have to report runtime in the demo anyway.

**Memory (8 GB).** Working on a cropped ROI at native spacing with float32 and per-candidate subvolumes, peak should sit near 1–2 GB. The decision not to globally upsample is what buys this headroom for cases with unusually long coverage.

**No internet.** Vendor all wheels. Dependencies: SimpleITK, numpy, scipy, scikit-image, matplotlib, optionally numba for the geodesic march. Nothing that phones home, no pretrained weights to fetch.

**CLI.** Exactly `python run.py --image image.nii.gz --aorta-mask aorta_mask.nii.gz --output prediction.json`. Accept both `.nii` and `.nii.gz` and gzip-despite-extension. Add `--viz-dir` as an optional extra so the mandated interface stays clean. Zero prompts, zero manual seeding.

---

## 5. Build order

Always have something that runs end to end.

1. I/O, crop, calibration, stub emitting an empty daughters list. Valid output from day one.
2. Geometry: centreline, radius, surface, end-cap exclusion, rotation-minimising frame. **Visualise this before going further** — end-cap handling is the most common silent failure.
3. Unrolled map and 2D peak detection. Measure precision/recall on the dev subset now, before adding anything.
4. Trace and the four output quantities.
5. Instance resolution rules, one at a time, re-measuring F1 after each. Drop any rule that does not help.
6. Vesselness, only if step 5 has plateaued.
7. Visualisation and README.

Hold out a few dev cases from threshold fitting so you have an honest estimate before the hidden set.

---

## 6. Expected failure modes

For the demo's "known failure cases" section. Naming these accurately is worth more than claiming they do not exist.

- **Small accessory branches near the eligibility boundary.** Resolution-limited, not algorithm-limited. Even with a full segmentation and a pre-built tree, Riffaud caught polar renal arteries only 70.8% of the time.
- **Venous structures in a poorly timed scan.** The HU separation that carries rule 3 collapses.
- **Cardiac chambers and great vessels near the ascending aorta.** The leak detector is the defence; expect residual false positives if coverage extends high.
- **Heavy mural calcification or stent artefact** at the wall, partially handled by `T_calcium`.
- **Origins within a few millimetres of a cut face**, where end-cap exclusion and true detection compete.
- **Trifurcations**, structurally the same problem as the spec's "two nearby origins" case. Wala names this as a known miss mode for model-based detectors.

---

## 7. Benchmarks

| Source | Task | Result | Relevance |
|---|---|---|---|
| Elattar 2016 | Coronary ostium localisation | 2.37 ± 1.44 mm (RCA), 1.99 ± 1.30 mm (LCA); interobserver 2.38 ± 1.56 and 3.21 ± 4.89 | Target for the 25% localisation category. They worked at 0.44–0.68 mm in-plane; expect worse at 1.5 mm. |
| Tahoces 2020 | Supra-aortic + visceral branch detection from a segmented aorta | recall 91.8%, precision 98.8% (F1 ≈ 0.95) | Closest comparable to the 45% category. Named list, 33 tuning cases — harder for you on both counts. |
| Riffaud 2022 | Named-artery identification from a pre-built vascular tree | 89.1% of 239 segmentations fully correct; polar RAs 70.8% | Ceiling estimate for small branches even with perfect input. |
| Wala 2011 | Pulmonary artery segmentation, low-dose CT | 64% sensitivity, 90% specificity; 0.15 mm bias, 0.63 mm RMS surface error | Closest spacing to yours (1.25 mm slices). Note localisation far exceeded detection — expect the same shape. |

Wala also names the tradeoff you will face directly: relaxing the bifurcation model gives few missed segments but more false positives. Characterise that curve on your dev set and report where you chose to sit and why. With one-to-one matching, lean conservative.

---

## 8. Requirements checklist

| Requirement | Where handled |
|---|---|
| No manual point placement | Stages 0–8 fully automatic |
| Valid JSON, empty list when nothing eligible | Stage 8, explicitly tested |
| Unique clinician-useful display | Unrolled wall map |
| Variable number of instances | Open-set 2D peak detection |
| Physical coordinate system preserved | SimpleITK throughout, LPS, round-trip test |
| No case-specific edits | Single config; all thresholds derived or frozen |
| CPU-only, no GPU | Classical IP only |
| Cropped ends are not origins | Stage 3, three tests |
| Two nearby origins stay separate | Stage 7 rule 1, contiguity not proximity |
| Common trunk is one origin | Stage 5 bifurcation stop |
| Daughter-of-daughter excluded | Stage 7 rule 2 |
| Do not guess absent vessels | Eligibility test + leak detector |
| Iliac division out of core scope | Not detected; flagged if encountered |
| No anatomical names | `branch_NNN` by arc length |
| ≥3 visual checks | Stage 8 |
| ~60 s, 8 GB, 4 cores | Section 4 |
| CLI contract | Section 4 |

---

## 9. Provenance

Worth stating in the demo — it shows you read the closest prior work and understood precisely where each piece stops applying.

**Elattar 2016 (Int J Cardiovasc Imaging).** The cylindrical intensity map is the candidate generator. Calcium masking. Localisation benchmark. Dropped: the Gaussian distal-extent prior and the 1D max projections, both of which assume a known count at a known location.

**Riffaud 2022 (Med Biol Eng Comput).** The PCA direction formula with sign fix. The non-anatomic-branch feature space. Dropped: all anatomical distance priors (their Table 1 describes the abdominal segment, which may not even be in the supplied coverage), the named-artery matching framework, and their centreline-node branch location, which carries a systematic one-radius ostium error.

**Wala 2011 (arXiv).** The leak detector. The asymmetric similarity penalty concept. The radius-ratio bifurcation check as a secondary signal. The parameter-optimisation methodology. Dropped: the iterative cylinder-tracking loop, whose 15 mm optimised cylinder height exceeds the entire permitted trace.

**Tahoces 2020.** Benchmark only; paywalled, method not available.

Everything in front of Stage 5 — crop, calibration, the geodesic survival test, end-cap exclusion by normal-versus-tangent, the rotation-minimising frame, wall-patch-contiguity merging, FWHM radius at the seed, LPS handling, gzip sniffing — has no ancestor in the literature reviewed. That gap is the challenge: every one of these papers is handed either a pre-built vascular tree or a known, named, fixed-count list of targets.
