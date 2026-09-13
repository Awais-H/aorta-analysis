# Branchseed Partner Handoff

## Instructions for the receiving agent

Use this document as the source of truth for the current research direction and implementation
state. Before changing the algorithm:

1. Read `README.md` and the modules listed below.
2. Run `uv sync --extra test && uv run pytest`.
3. Preserve the challenge output contract and SimpleITK physical-coordinate handling.
4. Treat all current thresholds as uncalibrated until evaluated against organizer references.
5. Change one algorithmic component at a time and compare it through the shared evaluator.
6. Do not claim that a detection is anatomically correct without reference data or qualified review.
7. Do not replace topology-aware instance formation with anatomical naming or fixed expected branch
   locations.

The goal of collaboration is not to merge two complete pipelines blindly. Compare each pipeline
stage, retain the stronger component, and verify that the combined system improves the same
one-to-one branch metrics within the CPU and memory limits.

## Executive summary

Branchseed receives:

- A contrast-enhanced abdominal CT/CTA volume.
- A binary mask containing only the parent abdominal aorta.

It must discover every eligible artery that connects directly to that aorta and return, for each
unique daughter:

- The ostium centre.
- A seed exactly 5 mm along the daughter path.
- The local radius at the seed.
- A unit direction vector pointing into the daughter.

The two largest evaluation categories are branch discovery (45%) and ostium localization (25%).
Duplicate detections are false positives.

Our current implementation is a deterministic, CPU-only research baseline. Its distinguishing
architecture is:

> Use the known aortic wall to propose likely branch openings, retain a wider fallback search for
> weak ostia, verify every candidate with a local outward path, and form daughter instances from
> wall-contact topology.

The code passes 62 automated tests, works on synthetic vessel phantoms, and has now executed on one
real organizer-format CTA case (`subject001`). That case does not include a reference annotation in
the available directory, so no branch-level accuracy, clinical, or challenge-performance claim is
justified.

## How we reached this design

### Initial standard approach

The obvious medical-imaging baseline was:

1. Crop around the supplied aorta.
2. Identify bright, tube-like structures.
3. Segment candidate vessels.
4. Skeletonize them.
5. Convert the skeleton into a graph.
6. Find graph branches attached to the aorta.
7. Measure ostia, seeds, directions, and radii.

This remains useful, but it depends on obtaining a reliable vessel segmentation before using the
strongest prior in the task: the parent aorta is already known.

### Useful ideas from the original notes

- **Segmentation:** identify voxels likely to belong to daughter vessels.
- **Pixel intensity:** contrast-filled arterial blood is bright, but brightness cannot distinguish
  arteries from bone, calcification, veins, and artifacts by itself.
- **Connected components:** reject isolated noise and group vessel candidates, using physical
  geometry rather than voxel count alone.
- **Skeletonization:** reduce a thick vessel mask to a centre path for topology and measurement.
- **Graph analysis:** represent parent connections, branch paths, and bifurcations.
- **Width thresholds:** useful only in physical millimetres and only as one source of evidence.

Brain diseases, amyloid biology, and brain atlases were not directly relevant to abdominal CTA.
SPIMquant was useful as an example of reproducible image-analysis workflow, not as a transferable
branch detector.

### Product insight from Toralis

Toralis describes converting imaging anatomy into structured, machine-readable geometry for
downstream morphometrics, device matching, and digital-twin modelling.

Branchseed is an upstream translation problem:

```text
CT voxels
  -> direct branch instances
  -> centreline and radius geometry
  -> vascular graph
  -> possible future EVAR/device modelling
```

This encouraged us to preserve an internal graph and evidence trail rather than returning
unexplained points only.

### Gemini deep research

Gemini strongly recommended:

- Ostium-first detection.
- Signed-distance wall regions.
- Jerman and Optimally Oriented Flux vessel filters.
- RORPO path operators.
- APP2/fast-marching ideas from neurite tracing.
- TEASAR/Kimimaro skeletonization.
- Cross-sectional radius measurement.
- Perturbation confidence.

We source-checked those recommendations before implementation.

### What survived source checking

- Searching from the known aortic wall is a strong task-specific hypothesis.
- Patient-specific intensity calibration is valuable.
- Real physical crops are important for CPU and memory.
- Jerman, OOF, and RORPO deserve controlled experiments.
- TEASAR and minimal paths are useful alternative path strategies.
- Radius should be measured at the 5 mm seed, not inside the ostium confluence.
- Small deterministic perturbations can expose unstable detections.

### What was corrected or rejected

- No vessel filter guarantees junction preservation or calcification resistance.
- Masking a full Hessian response with a signed-distance shell does not reduce convolution cost;
  the input must actually be cropped or tiled.
- `scikit-fmm` is not a complete APP2 implementation.
- Fast marching followed by TEASAR is redundant: use one path formulation at a time.
- A parallel-transport frame is unnecessary for one perpendicular cross-section.
- Fixed-distance DBSCAN can merge genuine nearby ostia and cannot solve topology by itself.
- Global image Z cannot identify aortic end caps safely.
- Branchseed does not prohibit external development data in the supplied brief; it prohibits GPU
  and internet use during evaluation.

## Most useful research cases and sources

### Directly relevant vascular studies

- **Automatic abdominal aortic branch identification:** Riffaud et al. convert an already-segmented
  abdominal arterial tree into a centreline graph and identify direct branches using topology,
  position, direction, length, and diameter. This supports graph reasoning after discovery, but does
  not solve discovery from a parent-only mask.
  - <https://doi.org/10.1007/s11517-022-02603-2>
  - <https://hal.science/hal-03520790v2/file/main.pdf>
- **Coronary ostium detection from an aortic surface:** supports treating an ostium as a localized
  wall-adjacent intensity event and searching outward from a known parent surface. Abdominal
  geometry is more variable, so fixed coronary-root assumptions must not transfer.
  - <https://pmc.ncbi.nlm.nih.gov/articles/PMC4751164/>
- **BRAVE EVAR pipeline:** supports centreline-constrained search and combining intensity,
  connectivity, and anatomical evidence. Its learned segmentation and naming assumptions are
  heavier than the Branchseed core task.
  - <https://pmc.ncbi.nlm.nih.gov/articles/PMC12069616/>
- **VMTK centreline and branch decomposition:** useful reference for radius-weighted centrelines,
  maximal-inscribed-sphere radii, and surface branch groups. It expects a suitable
  branch-inclusive surface and seeds, so it is not the initial detector here.
  - <https://github.com/vmtk/vmtk>
  - <http://www.vmtk.org/tutorials/Centerlines.html>

### Candidate advanced techniques

- **Jerman vesselness:** may retain more uniform responses around bifurcations and aneurysms than
  traditional Frangi, but does not guarantee connected ostia.
  - <https://doi.org/10.1109/TMI.2016.2550102>
- **Optimally Oriented Flux:** uses regional boundary-flux evidence and may be more robust to noise
  and nearby structures than pointwise Hessian features. Calcification resistance is unproven here.
  - <https://doi.org/10.1007/978-3-540-88693-8_27>
- **RORPO:** path-opening morphology detects bright structures that persist along discrete paths
  and may complement local vesselness around weak branches.
  - <https://doi.org/10.1109/TPAMI.2017.2672972>
  - <https://github.com/path-openings/RORPO>
- **TEASAR/Kimimaro:** distance-penalized tree skeletonization for a binary local vessel candidate.
  It is an alternative to a continuous minimal path, not a cleanup stage after one.
  - <https://github.com/seung-lab/kimimaro>
- **APP2/neurite tracing:** demonstrates grayscale-weighted paths and hierarchical pruning through
  weak 3D signal. `scikit-fmm` alone does not implement the complete method.
  - <https://doi.org/10.1093/bioinformatics/btt170>
- **Vascular junction correction and radius change points:** reinforces that raw skeleton junctions
  can be displaced from physiological ostia and radius should be measured outside the confluence.
  - <https://pmc.ncbi.nlm.nih.gov/articles/PMC10836077/>

### Public datasets worth investigating

- **AortaSeg24:** whole-aorta CTA with separately labeled aortic regions and branches. Potentially
  useful for deriving approximate branch/aorta interfaces; access and DUA restrictions apply.
  - <https://aortaseg24.grand-challenge.org/>
- **ColonVessels:** detailed abdominal visceral arterial labels with downloadable data and published
  segmentation models.
  - <https://doi.org/10.5281/zenodo.17407158>
- **AVT/SEG.A.:** binary complete aortic-vessel-tree CTA masks, useful for connectivity and graph
  extraction even though individual ostia are not labeled.
  - <https://doi.org/10.6084/m9.figshare.14806362>
- **AAA-100:** meshes and centrelines for geometry and topology testing, but no source CTA intensity.
  - <https://zenodo.org/records/10932957>

No reviewed public resource provides the exact Branchseed combination of explicit ostium points,
5 mm seeds, directions, and local radii. Approximate targets would need to be derived and must not be
treated as identical to organizer annotations.

## Final pipeline

### 1. Validate physical geometry

The image and mask must have matching:

- Grid size.
- Voxel spacing.
- Physical origin.
- Direction matrix.

Arrays use NumPy `z, y, x`; outputs use physical SimpleITK `x, y, z` millimetres.

### 2. Crop a safe periaortic region

The implementation creates an actual crop around the aorta. The halo is large enough to include:

- The wider fallback search.
- Gaussian support for the largest vesselness scale.

This reduces unrelated anatomy and bounds memory use.

### 3. Build the aortic model

The pipeline:

- Retains the largest connected parent-mask component.
- Computes a spacing-aware signed distance field.
- Identifies the interior wall.
- Identifies flat terminal cap regions without assuming a global image axis.
- Retains the lateral wall as the valid origin surface.
- Estimates patient-specific blood intensity from an eroded aortic core.

### 4. Compute image features

The maintained baseline calculates:

- Similarity to the patient's aortic blood intensity.
- Multiscale Hessian objectness.

Objectness is evaluated one scale at a time and only the maximum response is retained.

Jerman, OOF, and RORPO are not enabled by default. They should be added only after real-data
ablation demonstrates value.

### 5. Generate two proposal channels

#### Surface-first channel

Search near the lateral aortic wall for components with:

- Blood-like intensity.
- Tubular evidence.
- Proximity to the wall.
- Outward continuity.
- Sufficient physical persistence.

#### Wider fallback channel

Search farther from the wall for strong distal vessel evidence when the proximal ostium is weak.
Trace that evidence back toward the aorta.

Candidates retain their evidence source so the channels can be compared.

### 6. Verify each candidate locally

Only a small candidate-centred region is processed.

Two independent strategies are implemented:

1. Local adaptive/hysteresis segmentation followed by skeleton extraction.
2. Bounded continuous-cost A* minimal-path tracking.

The strategies are alternatives. They are isolated during ablation and may be used as runtime
fallbacks in normal inference.

A candidate must:

- Contact the valid lateral aortic wall.
- Continue for at least 5 physical millimetres.
- Move outward.
- Retain sufficient image support.
- Stay within bounded search and path limits.

### 7. Form direct daughter instances

Instances are formed from connected wall-contact patches and proximal path overlap.

This explicitly handles:

- One opening followed by a downstream split: one common-trunk daughter.
- Two separate nearby openings: two daughters.
- Multiple detections of one opening: merge.
- A vessel first connected through another daughter: reject as indirect.

Euclidean distance is supporting evidence only.

### 8. Measure required geometry

The final path is anchored at the reported wall-contact ostium.

- Seed: exactly 5 mm of physical polyline arc length from that ostium.
- Direction: robust local line/tangent fit oriented away from the parent.
- Radius candidates:
  - 3D Euclidean distance-transform radius.
  - 2D orthogonal inscribed radius.
  - 2D area-equivalent radius.
- Final radius: current consensus estimate with disagreement flags.

For a minimal-path result, the radius mask must come from local image evidence. It is never
fabricated by dilating the path with the proposal radius.

### 9. Confidence, output, and visualization

Optional perturbations vary feature values and rerun only bounded downstream stages. One-to-one
matching measures detection stability and ostium spread.

The strict output contains only:

```json
{
  "case_id": "case",
  "parent": {"instance_id": "aorta"},
  "daughters": []
}
```

Internal features, confidence, warnings, timing, and memory stay in debug output.

## What has been implemented

### Algorithm modules

- `branchseed/aorta.py`: mask cleanup, signed distance, walls, caps, blood model, coarse centreline.
- `branchseed/features.py`: blood similarity, gradients, Hessian objectness, score composition.
- `branchseed/candidates.py`: surface and wider fallback proposals.
- `branchseed/tracking.py`: local segmentation/skeleton and bounded A* tracking.
- `branchseed/instances.py`: wall contacts, common trunks, nearby openings, deduplication.
- `branchseed/measurements.py`: exact seed, direction, and radius estimators.
- `branchseed/confidence.py`: deterministic perturbation stability.

### Product and engineering modules

- `branchseed/pipeline.py`: end-to-end orchestration and profiling.
- `branchseed/config.py`: validated runtime configuration.
- `branchseed/models.py`: internal and challenge JSON models.
- `branchseed/output.py`: strict challenge serialization.
- `branchseed/evaluation.py`: Hungarian one-to-one matching and metrics.
- `branchseed/visualize.py`: axial, coronal, and sagittal overlays.
- `branchseed/phantoms.py`: synthetic vessel scenarios.
- `branchseed/ablation.py`: machine-readable feature/path/radius experiments.
- `branchseed/run.py` and root `run.py`: challenge CLI.

## Current evidence

- 62 automated tests pass.
- Lint, compilation, and dependency-lock checks pass.
- Synthetic tests cover:
  - Straight and curved aortas.
  - Multiple side daughters.
  - One common trunk.
  - Separate nearby ostia.
  - Anisotropic and rotated image geometry.
  - Weak proximal contrast.
  - Flat crop ends.
  - No visible daughter.
- The demonstrated straight synthetic case generated four proposals and correctly consolidated
  them to two daughters.
- The demonstrated small case ran in approximately 0.70 seconds with visualization and reached
  approximately 193 MB process high-water RSS.

These are engineering results, not clinical validation.

## First real-case run: subject001

### Inputs

- Image: `subject001/orig1.nii`
- Parent mask: `subject001/mask1.nii`
- Image size: `(512, 512, 174)` voxels in SimpleITK `x, y, z` order.
- Spacing: approximately `(0.782, 0.782, 0.800)` mm.
- Origin: approximately `(194.915, 230.071, -1086.440)` mm.
- Direction: diagonal `(-1, -1, +1)`.

### Execution result

- Initial candidates retained: 24, equal to the configured maximum.
- Final predicted daughters: 3.
- Total wall time with visualization: approximately 30.08 seconds.
- Process high-water RSS: approximately 1.88 GB.
- Runtime warnings: none.

Outputs:

- `results/subject001/prediction.json`
- `results/subject001/overlay.png`
- `results/subject001/debug/profile.json`
- `results/subject001/debug/features.npz`

### Predicted physical coordinates

The JSON uses SimpleITK/ITK physical `x, y, z` coordinates in millimetres, normally LPS.

- Branch 1:
  - Ostium: approximately `(16.650, 37.887, -1053.896)` mm.
  - Seed: approximately `(19.055, 39.391, -1049.779)` mm.
  - Nearest ostium voxel: `(228, 246, 41)`.
  - Nearest seed voxel: `(225, 244, 46)`.
- Branch 2:
  - Ostium: approximately `(17.479, 33.789, -961.240)` mm.
  - Seed: approximately `(15.999, 30.374, -964.578)` mm.
  - Nearest ostium voxel: `(227, 251, 157)`.
  - Nearest seed voxel: `(229, 255, 152)`.
- Branch 3:
  - Ostium: approximately `(19.702, 34.671, -991.538)` mm.
  - Seed: approximately `(22.332, 35.353, -994.684)` mm.
  - Nearest ostium voxel: `(224, 250, 119)`.
  - Nearest seed voxel: `(221, 249, 115)`.

Physical-to-voxel conversion uses:

```python
image.TransformPhysicalPointToContinuousIndex(point_xyz_mm)
```

Conceptually:

```text
physical = origin + direction * (voxel_index * spacing)
voxel_index = inverse(direction) * (physical - origin) / spacing
```

### Findings exposed by the real case

1. **Candidate generation is too permissive:** retaining exactly 24 proposals means the configured
   cap was saturated. Lower-ranked real branches could be dropped before verification.
2. **Predictions are unverified:** no reference file was available, so three outputs must not be
   described as three correct arteries.
3. **Case ID needs correction:** output currently derives `case_id` as `orig1`; the expected case
   identity is likely `subject001`.
4. **Visualization needs refinement:** paths are projected into all orthogonal views rather than
   filtered to portions intersecting each displayed slice. Per-daughter centred views or a slab/MIP
   representation would be more trustworthy.
5. **Branch 1 direction is suspicious:** the fitted direction differs from the normalized
   ostium-to-seed displacement by approximately 74.4 degrees. Branch 2 differs by about 2.5 degrees
   and Branch 3 by about 19.9 degrees. This requires path inspection and a direction-consistency
   quality gate.
6. **Runtime and memory are currently within the stated limits for this case:** approximately 30
   seconds and 1.88 GB, but one case does not establish worst-case compliance.

These findings are more important than synthetic success because they identify the first real
engineering priorities.

## How to run

```bash
cd /Users/brandonli/Documents/projects/aorta-analysis
git switch feature/branchseed-pipeline
uv sync --extra test

uv run python run.py \
  --image image.nii.gz \
  --aorta-mask aorta_mask.nii.gz \
  --output prediction.json \
  --visual overlay.png \
  --debug-dir debug/
```

Run tests:

```bash
uv run pytest
```

Generate synthetic data and ablations:

```bash
uv run python -m branchseed.phantoms --output-dir synthetic-output
uv run python -m branchseed.ablation --output synthetic-output/ablation.json
```

## Best ways to combine partner work

### If the partner has another detector

Integrate its probability map as another feature/proposal channel in:

- `branchseed/features.py`
- `branchseed/candidates.py`

Retain the existing path validation, topology, measurement, output, and evaluation stages.

### If the partner has a segmentation model

Use its local vessel mask as input to:

- `branchseed/tracking.py`
- `branchseed/instances.py`
- `branchseed/measurements.py`

Do not discard physical-coordinate and topology tests.

### If the partner has a centreline method

Add it as a third isolated tracker in `branchseed/tracking.py`. Compare it with the existing
segmentation/skeleton and bounded minimal-path strategies using the ablation harness.

### If the partner has a UI

Consume:

- Strict challenge JSON for finalized findings.
- Debug profile JSON for timing and evidence.
- Feature arrays for diagnostic overlays.
- Internal candidate paths for interactive centreline rendering.

### If the partner has medical expertise

Prioritize review and labeling of real failure categories:

- Calcification.
- Veins.
- Weak contrast.
- Small branch caliber.
- Closely spaced ostia.
- Common trunks.
- Incomplete coverage.
- Parent-mask errors.

## Immediate next steps

1. Obtain the example reference output for `subject001` or another annotated case.
2. Fix case-ID derivation and add an optional explicit `--case-id`.
3. Replace projected path overlays with slice-aware or per-daughter centred diagnostic views.
4. Add a direction-consistency gate comparing the robust tangent with proximal displacement.
5. Remove the hard candidate-cap risk by improving ranking, processing in batches, or proving that
   discarded candidates cannot be eligible.
6. Evaluate `subject001` predictions manually in ITK-SNAP and record whether each candidate is a
   direct artery, false positive, duplicate, or indirect branch.
7. Run the unchanged baseline on every available case and create a failure ledger.
8. Audit whether reference ostia are visible within the near-wall region.
9. Tune only physical parameters supported by development data.
10. Run isolated experiments for Jerman, OOF, RORPO, path method, and radius method.
11. Optimize discovery F1 and duplicate control before polishing geometry.
12. Profile all full-size cases against the four-core, 8 GB, and runtime limits.
13. Decide which partner component measurably improves the shared baseline.

## Prioritized recombination protocol

### Phase A: compare both systems without merging

For the same cases, run:

- Current Branchseed baseline.
- Partner baseline.
- Any available organizer baseline.

Use one evaluator and retain raw predictions, timings, memory, and visual checks.

### Phase B: compare stage by stage

For each system, document:

1. Aorta representation.
2. Intensity normalization.
3. Feature generation.
4. Candidate proposal logic.
5. Path or segmentation method.
6. Direct-parent topology.
7. Duplicate/common-trunk handling.
8. Ostium, seed, direction, and radius measurement.
9. Runtime and memory.
10. Known failure cases.

### Phase C: integrate through explicit seams

- Alternative feature map: add behind an interface in `features.py`.
- Alternative candidate generator: add a separate proposal channel in `candidates.py`.
- Alternative tracker or segmentation: add an isolated method in `tracking.py`.
- Alternative instance logic: compare in `instances.py` without changing the evaluator.
- Alternative radius/direction method: add an estimator in `measurements.py`.
- Alternative UI: consume strict JSON and debug artifacts without changing algorithm outputs.

### Phase D: require an ablation decision

For every added component, record:

- Hypothesis.
- Cases and failure category targeted.
- Baseline configuration.
- Changed component.
- Precision, recall, F1, duplicate count, and ostium error.
- Seed, direction, and radius quality.
- Runtime and peak memory.
- Keep, revise, or reject decision.

Do not retain a component merely because it is novel or uses machine learning.

## Suggested tasks for the partner's agent

1. Reproduce the 62-test baseline before editing.
2. Review the `subject001` prediction in ITK-SNAP using the voxel coordinates above.
3. Search for or request the organizer reference annotation.
4. Fix the known case-ID, overlay, direction-consistency, and candidate-cap issues with regression
   tests.
5. Add a structured experiment manifest and per-case result ledger.
6. Compare the partner's strongest detector against the current proposal stage without replacing
   downstream topology and physical measurements.
7. Prototype only one advanced feature family at a time, beginning with the failure observed most
   often on real cases.
8. Preserve a frozen baseline configuration and save every experiment result in machine-readable
   form.
9. Keep the strict challenge writer isolated from debug metadata.
10. Report unsupported assumptions and missing annotations explicitly.

## Current repository state

- Repository: `Awais-H/aorta-analysis`
- Local path: `/Users/brandonli/Documents/projects/aorta-analysis`
- Local branch: `feature/branchseed-pipeline`
- GitHub rejected publishing this branch because the connected account does not have write access.
- The implementation is currently uncommitted locally.

Before collaboration, either:

- Grant the connected account collaborator access.
- Push from an authorized account.
- Fork the repository and publish the feature branch there.

