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
- Reference annotations for cases 19-23 are in docs/references/case_XX/annotations.json
  (SimpleITK LPS mm; schema is a superset of the required output). Read docs/references/README.md
  for the annotation policy: 2.0 mm minimum origin diameter, ostium on the supplied mask boundary,
  direction = normalised (seed - ostium). These are drafts; see each case's review_notes.md.
- Daughter instance masks for the seed-on-branch test: data/subjectXXX/daughtersXX_draft.nii.gz
  (0 = background, 1..N = branch_001..branch_N, same grid as the CT).
- docs/atlas_all.csv has per-case stats for all 25; wall_patches is the expected candidate count
  before filtering. Regression priority is cases 19-23, then 16-17, then 18 and 24 (allowed to fail).
- The scorer (scorer.py) is a day-one deliverable and reads docs/references/ directly.

## Smoke test
python run.py --image data/subject010/orig10.nii --aorta-mask data/subject010/mask10.nii --output /tmp/s10.json
Expected: 4 to 6 branches, runtime under 10 s.

## Current status
(update this line as modules land)
Engine complete through D6. D4 handles the hugging branch (elongated patch: ostium at the end that hugs the wall, tracer along the strip). On cases 19-23 at 5 mm: TP 15 / FP 24 / FN 4, F1 0.52, ostium 1.39 mm, seed on branch 12/15. Known misses: 19/b2 (1-voxel patch, rule 7 noise floor), 20/b1 and 23/b1 (wall-hugging lumbars: ostium now within 2.3 mm but rejected by D6 rule 3 at 1.9 mm departure, exactly like the reference guides themselves; decision pending, see results/sweeps.txt DEPARTURE_MM), 23/b2 (merged with b3 at the wall at 1.5 mm voxels: resolution limit). Remaining stubs: frame (slice centroids), report (PNG + table). The axis refinement and the area-growth rule are switched off with the measurements recorded in config and filters.
