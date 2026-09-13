# Branchseed Challenge

Read SPEC.md before doing anything. It contains every design decision and the reasoning.
The challenge PDF is docs/Branchseed_challenge.pdf. SPEC.md wins if they conflict.

## Rules
- All numeric constants live in config.py with a justification comment. Never hardcode a threshold elsewhere.
- io_utils.index_to_mm is the only place TransformIndexToPhysicalPoint is called.
- Every module gets a fake-data test in tests/ before it is merged.
- Do not change constants to improve a score on one case. See SPEC.md section 3.
- Do not add anatomical names anywhere. Branches are branch_001, branch_002, ...
- run.py must never crash on a case: log, write empty daughters, exit 0.

## Data
- data/subjectNNN/origN.nii and maskN.nii. Some are gzipped despite the extension. Subject 24 has a non-orthonormal header.
- atlas_all.csv has per-case stats; expected wall-patch counts are in the wall_patches column.
- Draft reference annotations for cases 19 to 23 are in docs/references/case_NN/annotations.json; the per-case daughter label volumes are data/subjectNNN/daughtersNN_draft.nii.gz.

## Smoke test
python run.py --image data/subject010/orig10.nii --aorta-mask data/subject010/mask10.nii --output /tmp/s10.json
Expected: 4 to 6 branches, runtime under 10 s.

## Current status
(update this line as modules land)
Skeleton aligned with the updated SPEC (opening for the wall layer only, watershed and tracing on the raw shell, chord direction, 2.0 mm origin diameter, atomic JSON). Scorer reads docs/references and writes results/dev_scores.txt; invariants sweep writes results/invariants.txt. All 25 cases run without a crash. Real logic: io_utils, candidates, instances, scorer. Stubs: ostium (centroid snap), tracing (straight line along the wall normal), filters (eligibility rule only: precision 0.12 on cases 19-23), frame (slice centroids), report (PNG + table). Next per SPEC section 5: ostium and tracing, then filters rules 1, 3, 4, 7.
