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
Engine complete through D6, reviewed by eye on the gallery sheets (gallery.py) for subjects 16, 17, 21, 22. D6 now: rule 3 is a flag (hugging lumbars and the IMA are daughters); rule 4 rejects bone by "seed cross-section runs out of the 16 mm window AND carries cortex over 1.5 x lumen" (edge without cortex = section_merged flag); rule 6 also merges same-voxel ostia and survivors whose 10 mm paths come within 4 mm (one lumen, one daughter). On cases 19-23 at 5 mm: TP 17 / FP 16 / FN 2, F1 0.65, ostium 1.35 mm, seed on branch 14/17. Remaining false positives are mostly 2 to 3 mm posterior/anterior-left vessels in unannotated lumbar and IMA territory, plus one candidate per case at the inferior cut (iliac division or cut-adjacent; decision pending, flagged near_cut_face). Known misses: 19/b2 (1-voxel patch), 23/b2 (merged with b3). Remaining stubs: frame (slice centroids), report (PNG + table). Review tools: review_markers.py (ITK-SNAP overlays), gallery.py (crop sheets).
