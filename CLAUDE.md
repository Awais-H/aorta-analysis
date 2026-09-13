# Branchseed Challenge

Read docs/HANDOFF.md first (current state, decisions, what remains), then SPEC.md (design and
reasoning, up to date with the code). The challenge PDF is docs/Branchseed_challenge.pdf. Where
SPEC.md and the PDF conflict, the PDF's definitions win; where SPEC.md's first-draft methods and
the anatomy conflict, the anatomy wins (see the principle at the top of SPEC.md).

## Rules
- All numeric constants live in config.py with a justification comment. Never hardcode a threshold elsewhere.
- io_utils.index_to_mm is the only place TransformIndexToPhysicalPoint is called.
- Every module gets a fake-data test in tests/ before it is merged.
- Do not change constants to improve a score on one case. See SPEC.md section 3. Run sweeps.py and keep the physical value.
- Judge every rule against real vascular anatomy and the reference annotations before the numbers. A rule that rejects things real vessels do gets changed, with the evidence written next to it.
- Do not add anatomical names to the output. Branches are branch_001, branch_002, ... (names may appear in comments and the ledger as reasoning).
- run.py must never crash on a case: log, write empty daughters, exit 0.
- After any pipeline change, regenerate and commit results/ (scores, ledger, sweeps, invariants). A change must improve or hold every labelled case or come with a written anatomical argument.

## Data
- data/subjectNNN/origN.nii or origN.nii.gz and maskN.nii. Some are gzipped despite the extension. Subject 24 has a non-orthonormal header. Subjects 1 to 15 are 0.8 mm, 16 to 25 are 1.5 mm isotropic; the hidden set is assumed to be 1.5 mm like the references.
- docs/atlas_all.csv has per-case stats for all 25; wall_patches is the expected candidate count before filtering.
- Reference annotations for cases 19-23 are in docs/references/case_XX/annotations.json
  (SimpleITK LPS mm; schema is a superset of the required output). Read docs/references/README.md
  for the annotation policy: 2.0 mm minimum origin diameter, ostium on the supplied mask boundary,
  direction = normalised (seed - ostium). These are drafts; see each case's review_notes.md.
- Daughter instance masks for the seed-on-branch test: data/subjectXXX/daughtersXX_draft.nii.gz
  (0 = background, 1..N = branch_001..branch_N, same grid as the CT).
- Regression priority is cases 19-23 (scored), then 16-17, then 18 and 24 (allowed to fail), then 1, 6, 8. Subject 25 (whole aorta with arch) is out of scope.
- The scorer (scorer.py) reads docs/references/ directly.

## Smoke test
python run.py --image data/subject010/orig10.nii --aorta-mask data/subject010/mask10.nii --output /tmp/s10.json
Expected: 4 daughters (celiac and SMA at 12 to 1 o'clock, right renal at 10.5, left renal at 3), pipeline about 2 s, about 5 s wall with imports.

## Current status
(update this line as modules land)
Engine complete through D6 and reviewed by eye (gallery.py) on subjects 16, 17, 21, 22; frame (geodesic centreline, anatomical clock) and report (clock map, flags, Plotly 3D, verification PNG) built. On cases 19-23 at 5 mm: TP 17 / FP 13 / FN 2, F1 0.69, ostium 1.35 mm, seed on branch 14/17 (rule 1 iliac division, rule 4 blob at seed, rule 6 same-origin added after the false-positive review, SPEC section 1); invariants clean on all 25, under 7 s and 1.3 GB peak per case; leave-one-out (results/leave_one_out.txt) shows the physical constants are the held-out optimum. Dev-set predictions in results/predictions, visual checks in results/visual_checks. Rule 4 area growth reviewed on subjects 8 and 12 and kept as a flag. Next in order: vendored wheels and offline install test (needs the organisers' OS/Python answer), demo material. Six questions are with the organisers (SPEC section 6); their answers may reopen rule 1 at the inferior cut and the borderline-diameter handling. Code is on the fork nikolakr7/aorta-analysis, branch skeleton.
