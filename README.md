# Branchseed

Detects every eligible artery that leaves a supplied parent aorta and returns
each as a separate daughter instance, with ostium centre, seed, local radius
and initial direction in SimpleITK physical millimetres.

## Setup

```
./setup.sh
```

## Run

```
python run.py --image image.nii.gz --aorta-mask aorta_mask.nii.gz --output prediction.json
```

---

## Method

The aorta mask is treated as a calibration source rather than only an anchor.
Every intensity threshold in the pipeline is derived from the HU distribution
inside the supplied mask, so nothing is tuned to one scanner or one contrast
timing. The volume is cropped to the mask bounding box dilated by 30 mm, which
is typically a few percent of the input, and is never globally upsampled:
sub-voxel precision comes from on-demand cubic interpolation at the points that
need it and from small per-candidate subvolumes at 0.5 mm.

A pruned 3D skeleton gives the aortic centreline, resampled to 1 mm of arc, and
a **rotation-minimising frame** (double reflection, Wang et al. 2008) is
propagated along it, then rotated once so that θ = 0 is anterior. A fixed
global reference vector would degenerate wherever the centreline turns parallel
to it, which is guaranteed on an arch, and the Frenet frame is undefined along
a straight segment. Candidates are generated from an **unrolled wall map**:
arc length against angle, each pixel holding the mean CT intensity along a
1–6 mm outward probe from the aortic surface, with calcified voxels excluded.
Peaks are found by genuine 2D local-maximum detection on a circularly padded
copy, so a branch at 12 o'clock is one candidate rather than two.

Each candidate is then traced. A 30 mm subvolume around its wall patch is
resampled to 0.5 mm, a geodesic distance field is grown outward from the patch
through peri-aortic lumen with the parent excluded, and its level sets are
walked in 0.5 mm steps. That gives 20 samples across the 10 mm budget where an
iterative cylinder tracker would manage three or four at 1.5 mm native spacing.
The trace stops on a leak, a persistent topological split, a weak match, or the
10 mm cap. Eligibility — at least 5 mm of followable lumen, no leak — is a
separate test from the stop reason, so a trace that stops at a bifurcation at
7 mm is eligible while one that dies at 3 mm is not.

The four reported quantities each have a specific definition:

* **Ostium** — the intensity-weighted centroid of the wall contact patch,
  projected onto the surface by bisection on the signed distance. Explicitly
  *not* a centreline node, which sits one aortic radius inside the wall and
  would be a systematic ~9 mm error on a metric worth 25%.
* **Direction** — Riffaud's Definition 5, the leading eigenvector of `AᵀA` with
  the sign fixed by `sign(1ᵀAe)`, computed over the whole proximal path and
  entirely in physical millimetres. Without the sign term roughly half the
  arrows come back flipped; in index space, non-identity direction cosines make
  the vector simply wrong and normalisation hides it.
* **Seed** — the path point at 5 mm of geodesic distance, then recentred on the
  intensity-weighted centroid of the lumen cross-section.
* **Radius** — one FWHM measurement on that same 0.25 mm cross-section, not an
  EDT on a binarised mask, which inherits the threshold's bias and quantises to
  ~0.75 mm — a 30% error on a 2.5 mm renal artery.

Instance resolution merges **only** by wall-patch contiguity, never by centroid
proximity: two nearby origins that are separate at the wall must be returned as
two instances, and Riffaud's series puts the mean inter-renal origin distance
at 9.5 mm with SD 5.1, so any proximity merge would silently halve renal
recall. Daughters-of-daughters, dim or non-arterial contacts and artefactual
surface bumps are then rejected in that order. IDs are assigned by arc length
along the centreline, not by z, because superior-to-inferior ordering is
ambiguous across an arch.

## Clinician display

`--viz-dir DIR` writes the unrolled wall map with each detected ostium marked
by clock position, longitudinal distance and radius. Circumferential clock
position and distance from a reference are the two numbers that go on a
fenestrated stent-graft order form, and the display *is* the representation the
detector ran on rather than a separate rendering — the reader sees the picture
the algorithm saw, including the pixels it discarded and why. Rejected
candidates appear as hollow markers labelled with the rule that dropped them.

Clock position is defined in the local vessel frame, which diverges from the
patient axial plane in the arch. This is stated on the figure.

## Optional voxel-unit output

The mandated output reports physical millimetres, per the brief. For pointing
a viewer such as ITK-SNAP at the result directly, `--voxel-output PATH` writes
a second, non-mandated file alongside it with the same daughters in the
**original image's** voxel grid: `ostium_ijk_voxel` and `seed_ijk_voxel` are
integer, non-negative, in-bounds indices — the cursor-position convention
ITK-SNAP uses — with `*_continuous` variants alongside them for anyone who
wants the sub-voxel position instead (equivalent to
`sitk_ref.TransformPhysicalPointToContinuousIndex`, unrounded). `radius_voxels`
and `direction_ijk` are also included. This file never replaces `--output`,
and a failure to write it never affects the mandated result.

```
python run.py --image image.nii.gz --aorta-mask aorta_mask.nii.gz \
    --output prediction.json --voxel-output prediction_voxel.json
```

## Verification figures

`figures/` holds three views per case for four synthetic cases: the unrolled
map, a 3D render of the mask with ostium markers and direction arrows in three
projections, and orthogonal MIP slabs through each ostium so a reader can
confirm the branch is actually there.

Regenerate with:

```
python tools/predict_all.py --data data/phantoms --out predictions/phantoms \
    --viz-dir figures --viz-cases 4
```

## Parameters

Every value is either derived from the mask at runtime or fixed in
`config.yaml`. There are no case-specific edits and no inline constants.

| Parameter | Value | Source | Sensitivity |
|---|---|---|---|
| ROI dilation | 30 mm | fixed | low |
| Subvolume upsampling | 0.5 mm | fixed | medium |
| Calibration erosion | 2 mm | fixed | low |
| `t_lumen` | interior 5th percentile | **derived** | high |
| `t_calcium` | `mu_ao + 4·sigma_ao` | **derived** | medium |
| Probe range | 1–6 mm outward | fixed | **high — tune first** |
| Probe radius | 1 mm | fixed | medium |
| Angular bins | 128 | fixed | low |
| Arc sampling | 1 mm | fixed | low |
| Minimum map region | 2 px | phantom suite | medium |
| End-cap normal angle | 35° | fixed | medium |
| End-cap plane distance | 3 mm | fixed | medium |
| Geodesic step | 0.5 mm | fixed | low |
| Trace propagation threshold | `t_lumen − 2·sigma_ao` | **derived** | medium |
| Leak ratio | 1.5 | Wala 2011 | medium |
| Strong-match fraction | 0.5 | Wala 2011 | medium |
| Radius-ratio check | 0.90, diagnostic only | Wala 2011 | low |
| Bump boundary | `[0.4, 10.0, −7.0]` | **unfitted, see below** | **high** |
| Map peak threshold | `t_lumen` | **derived** | **high** |

Two of these deserve their provenance stated plainly.

**`detect.min_region_px = 2`** was lowered from 4 on the synthetic suite. A
2 mm-radius branch produces an unmistakable 495 HU peak but clears the absolute
threshold on only about three pixels, so rejecting on area penalises exactly
the small accessory branches that are hardest to recall. Precision is defended
downstream instead, by eligibility, the leak detector and the contact tests.

**`resolve.bump_boundary` has not been fitted**, because no annotated case was
available in this repository. The committed line passes through (5 mm, 0.50)
and (10 mm, 0.30) in Riffaud's feature space — followed length against length
divided by the local parent radius — so a 5 mm reach is kept on an aorta of
radius up to about 10 mm and rejected on a 15 mm one. It is deliberately
weighted towards the relative axis and deliberately permissive: an unfitted
discriminant that deletes a true positive cannot be recovered, while a false
positive still has to survive three principled tests. **Fit it on the
development subset before relying on it**; `tools/sweep.py` documents how.

## Tuning

```
python tools/sweep.py --data data/... --references references/... \
    --param wallmap.probe_outer_mm --values 4 5 6 7 8 --holdout caseA caseB
```

One parameter at a time, maximising a single score, following Wala's
methodology. The harness prints the whole curve rather than only the argmax:
both too-small and too-large values are expected to degrade performance, and a
value in the middle of a flat region is worth more than one on a narrow peak.
Hold two or three cases out of fitting entirely.

`tools/evaluate.py` scores predictions against references with one-to-one
Hungarian matching, so duplicates count as false positives, and reports a
breakdown of losses by rejection rule — which tells you which rule is costing
recall, the most actionable diagnostic here.

## Results

**No development-set data was supplied in this repository**, so the committed
predictions are for a synthetic suite with exact ground truth rather than for
real cases. The suite is generated deterministically by
`tools/make_phantom_dataset.py` and each case corresponds to one of the
challenge's "Important cases".

```
python tools/make_phantom_dataset.py
python tools/predict_all.py --data data/phantoms --out predictions/phantoms
python tools/evaluate.py --predictions predictions/phantoms --references references/phantoms
```

| Case | What it tests | Ref | Pred | Ostium | Direction | Radius | Seed |
|---|---|---|---|---|---|---|---|
| standard | four branches, distractors, two cut faces | 4 | 4 | 0.46 mm | 1.4° | 0.10 mm | 0.06 mm |
| arched | curved parent, frame must not degenerate | 2 | 2 | 0.23 mm | 0.8° | 0.03 mm | 0.03 mm |
| nearby_pair | two origins 8 mm apart | 2 | 2 | 0.29 mm | 0.7° | 0.16 mm | 0.08 mm |
| common_trunk | divides at 3 mm, one origin | 1 | 1 | 0.36 mm | 1.4° | 0.48 mm | 0.11 mm |
| daughter_of_daughter | vessel off a daughter | 1 | 1 | 0.17 mm | 4.4° | 0.06 mm | 0.11 mm |
| short_stub | dies at 3 mm | 0 | 0 | — | — | — | — |
| leaking | opens into a blob at 9 mm | 0 | 0 | — | — | — | — |
| near_cut_face | real origin 14 mm from a cut face | 1 | 1 | 0.32 mm | 1.8° | 0.06 mm | 0.09 mm |
| no_branches | bare tube | 0 | 0 | — | — | — | — |

Precision, recall and F1 are 1.00 across the suite at both 5 mm and 10 mm
match radii.

**These numbers are not a estimate of hidden-set performance.** A phantom has
no venous contamination, no calcified plaque, no motion, no bowel gas and a
perfect mask. What the suite does establish is that there is no systematic
geometric error — no axis-order confusion, no index/physical mix-up, no flipped
direction vectors, no end-cap false positives — and that each of the specified
edge cases behaves as required. Expect real-data ostium error in the range
Elattar reports, 2.0–2.4 mm against 2.4–3.2 mm interobserver variation, made
worse by 1.5 mm spacing.

## Runtime and memory

Measured on a 2.36 M-voxel ROI (340 mm of arched aorta, 7 branches), on an
Apple M-series laptop, single process, via the mandated command line:

| Stage | Seconds |
|---|---|
| load | 0.07 |
| crop | 0.01 |
| calibrate | 0.16 |
| geometry | 0.28 |
| prefilter | 0.05 |
| wall map | 0.48 |
| detect | 0.00 |
| trace (8 candidates) | 1.01 |
| measure | 0.02 |
| resolve | 0.08 |
| **total** | **2.15 s wall, 504 MB peak RSS** |

Against a 60 s and 8 GB budget on four cores. Peak memory does not grow with
the length of the supplied segment: the wall-map probe and the ray march are
both evaluated in blocks. The organisers' CPU will differ, but there is a
large margin. Vesselness fusion is implemented nowhere by design — it would add
roughly 20 s and the intensity map alone saturates the synthetic suite; add it
only if measured F1 on real data plateaus.

## Assumptions and known failure modes

**Stated assumption — the sub-5 mm bifurcation.** If a trunk divides before the
5 mm seed distance, the branch is eligible (5 mm of lumen exists) and is one
instance (common-trunk rule), but the trace rule stopped before the seed. We
follow the **dominant child**, the larger cross-section at the split, out to
5 mm, because that is what a human tracing a centreline does. `path_length_mm`
still reports the honest distance to the split and the daughter is flagged
`seed_beyond_bifurcation`. The brief is genuinely ambiguous here; this is a
documented choice, not a silent one, and it should be put to the organisers.

Failure modes we expect, named rather than denied:

* **Small accessory branches near the 5 mm boundary.** Resolution-limited, not
  algorithm-limited. Riffaud caught polar renal arteries only 70.8% of the time
  with a full segmentation and a pre-built tree.
* **Venous structures in a poorly timed scan.** The arterial/venous HU
  separation that carries the contact and band tests collapses. Detected as
  `degenerate_calibration` and flagged rather than silently trusted.
* **Cardiac chambers and great vessels** where coverage extends high. The leak
  detector is the defence; expect residual false positives.
* **Heavy mural calcification or stent artefact** at the wall, only partly
  handled by `t_calcium`.
* **Origins within a few millimetres of a cut face**, where end-cap exclusion
  and true detection compete directly. An origin 14 mm out is tested and
  survives; closer than that is untested.
* **Trifurcations**, structurally the same problem as two nearby origins.
* **The bump boundary is unfitted** (above). This is the largest single source
  of uncertainty in the precision of the hidden-set result.

## Provenance

* **Elattar 2016** — the cylindrical intensity map as candidate generator, and
  calcium masking. Dropped: the Gaussian distal-extent prior and the 1D max
  projections, both of which assume a known count at a known location.
* **Riffaud 2022** — the PCA direction formula with the sign fix, and the
  non-anatomic-branch feature space. Dropped: all anatomical distance priors,
  the named-artery framework, and their centreline-node branch location.
* **Wala 2011** — the leak detector, the radius-ratio check as a secondary
  signal, and the parameter-optimisation methodology. Dropped: the iterative
  cylinder-tracking loop, whose optimised 15 mm cylinder height exceeds the
  entire permitted trace.

Everything ahead of the trace — crop, mask-derived calibration, end-cap
exclusion by normal-versus-tangent, the rotation-minimising frame, the geodesic
survival test, wall-patch-contiguity merging, FWHM radius at the seed, LPS
handling and gzip sniffing — has no direct ancestor in that literature, which
is the gap the challenge occupies: each of those papers is handed either a
pre-built vascular tree or a known, named, fixed-count list of targets.

## Repository layout

```
run.py                  the mandated CLI, deliberately thin
config.yaml             every tunable constant
setup.sh                the one setup command
branchseed/
  io.py                 load, gzip sniffing, grid validation, coordinate round-trip
  roi.py                crop and grid normalisation
  calibrate.py          HU statistics derived from the mask
  geometry.py           centreline, radius, surface, end caps, frame
  frames.py             rotation-minimising frame (double reflection)
  wallmap.py            the unrolled map
  detect.py             2D peak detection to candidates
  trace.py              geodesic level-set trace
  measure.py            ostium, direction, seed, radius
  resolve.py            instance resolution rules
  output.py             JSON serialisation
  viz.py                clinician display and verification figures
  pipeline.py           orchestration and error containment
  interp.py             the only place an (x,y,z) point becomes a [z,y,x] index
  voxelgraph.py         shared 26-connectivity graph
  timing.py             per-stage time and peak RSS
tests/                  41 tests: coordinates, frames, end caps, phantoms, schema, CLI
tools/                  evaluate, sweep, predict_all, phantom dataset, vendoring
```

Run the tests with `python -m pytest tests/ -q` (about a minute).

## Conventions

Two rules hold everywhere and are asserted in tests, because breaking either
produces plausible-looking numbers that are simply wrong:

* every numpy volume is indexed `[z, y, x]`, every *point* is `(x, y, z)` in
  ROI continuous-index space, and the single conversion lives in `interp.py`;
* every emitted coordinate passes through `CaseGrid.to_physical`, which is
  SimpleITK LPS. nibabel is never imported — it reports RAS, and mixing the two
  flips the sign of x and y.

Output is deterministic: the same input produces byte-identical JSON, which is
tested.
