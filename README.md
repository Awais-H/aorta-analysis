Branchseed
==========

Branchseed is a deterministic, CPU-only research pipeline that proposes direct
aortic daughter-vessel ostia, seeds, directions, and radii from a 3-D image and
a supplied binary aortic lumen mask.

Setup
-----

```bash
uv sync --extra test
```

Run
---

The challenge command is exactly:

```bash
python run.py --image image.nii.gz --aorta-mask aorta_mask.nii.gz --output prediction.json
```

Optional flags are `--config config.json`, `--visual overlay.png`, and
`--debug-dir debug/`. The JSON config accepts fields from
`branchseed.config.PipelineConfig`; unknown fields fail clearly.

Challenge output contains exactly `case_id`, `parent`, and `daughters`:

```json
{"case_id": "case", "parent": {"instance_id": "aorta"}, "daughters": []}
```

Architecture
------------

1. Validate matching 3-D size, spacing, origin, and direction and a binary mask.
2. Crop an actual periaortic ROI. The effective halo is automatically raised
   to `fallback_far_mm + 4 * max(objectness_scales_mm)` so fallback reach and
   Gaussian support are not truncated.
3. Clean and analyze the aorta, flat end caps, wall, and patient-specific blood.
4. Compute blood similarity and maintained Hessian objectness one scale at a
   time, retaining only the maximum response.
5. Generate near-wall and low-proximal-contrast fallback proposals, explicitly
   sampling outward continuity and bounding their count.
6. Verify locally with segmentation/skeleton tracking or bounded minimal path;
   an unavailable optional skeletonizer falls back to the built-in method.
7. Deduplicate direct wall-contact instances, measure at 5 mm, sort
   deterministically, and emit strict challenge JSON. Debug output retains
   internal evidence, profiles stage runtime/RSS, and never leaks into the
   challenge schema.

Candidate deduplication preserves nearby same-channel ostia and merges only
strongly aligned cross-channel duplicate evidence. Instance deduplication is
based primarily on wall-contact patch overlap, so one opening remains one
daughter even if tracked paths diverge downstream. The reported path is anchored
at the final wall-contact centroid before the seed is measured at exactly 5 mm.
Minimal-path radius measurement uses a local feature-derived vessel component;
if none is reliable, that radius/prediction is skipped rather than reconstructed
from the proposal radius.

The CLI `profiles.total.seconds` is measured wall time. Its `rss_mb` is the
process high-water RSS; individual stage `rss_mb` values are point-in-time
samples. Matplotlib is imported only when `--visual` is requested.

Setting `confidence_perturbations` from 1 through 20 deterministically reruns
only bounded candidate generation, local verification, and instance grouping
with small feature perturbations. Its stability summary is internal
metadata/debug only. Runtime grows approximately linearly with that count;
objectness is not recomputed.

Synthetic benchmark and ablation
--------------------------------

```bash
python -m branchseed.phantoms --output-dir synthetic-output
python -m branchseed.ablation --output synthetic-output/ablation.json
```

The phantoms cover straight and curved aortas, multiple side daughters, a common
trunk, nearby ostia, anisotropic rotated metadata, low proximal contrast, crop
ends, and no daughter. Ablation output is machine-readable and compares feature
scales, local path methods, and radius estimators.

Limitations
-----------

This is an engineering research baseline, not a medical device, and no clinical
accuracy is claimed. A supplied aortic mask is mandatory. The coarse aortic
geometry and threshold-based proposal/verification stages can fail on severe
artifact, unusual anatomy, segmentation defects, or branches outside the halo.
Synthetic results are not evidence of clinical generalization. Jerman, OOF,
RORPO, and other advanced filters remain optional hooks and should not become
defaults until evaluated by a controlled ablation on representative data.
Minimal-path feature segmentation can be unavailable for weak or discontinuous
evidence; in that case Branchseed intentionally omits circular radius estimates.
