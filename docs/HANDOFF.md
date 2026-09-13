# Branchseed handoff (state at the end of 13 Sep 2026)

Read this, then CLAUDE.md, then SPEC.md. SPEC.md is the design document and is up to date with the
code; the decision log there says "as built" wherever the code differs from the first draft and
records the evidence. This file is the short version: where things are, what was decided and why,
what remains, and how to work.

## Where the code is

- Repository: fork `https://github.com/nikolakr7/aorta-analysis` (remote `fork`). The team origin
  is `https://github.com/Awais-H/aorta-analysis` (remote `origin`); nothing has been pushed there
  since the initial skeleton, by request.
- Branch `skeleton` on the fork holds everything. The branches `ostium-tracing`, `filters` and
  `instances-split` are earlier stops on the same line and are fully contained in `skeleton`.
- Data is in `data/subjectNNN/` (gitignored, 25 cases). Draft references for cases 19 to 23 are
  in `docs/references/` (committed), with the daughter label volumes in `data/`.
- Every module lives flat at the repository root. `python run.py --image ... --aorta-mask ...
  --output ...` is the judges' command and never crashes (traceback to stderr, empty daughters,
  exit 0). 87 tests: `python -m pytest tests`.

## What is built

| Stage | File | State |
|---|---|---|
| Reading, crop, resample, index to mm | `io_utils.py` | done; gzip sniffing, tilted-header fallback (subject 24) |
| Candidates (threshold, shell, opening) | `candidates.py` | done |
| Instances (wall patches, watershed) | `instances.py` | done; `branch_voxels` is the per-branch voxel set downstream |
| Ostium | `ostium.py` | done; inscribed circle snapped to the mask; axis refinement off by default; contact-end rule for hugging strips |
| Tracing | `tracing.py` | done; march with bifurcation stop, axis fallback, chord direction, 0.25 mm plane radius |
| Filters | `filters.py` | done; rules 1, 2, 4 (volume, bone), 6, 7 reject; rule 3 and area growth are flags |
| Frame | `frame.py` | STUB: slice centroids, axial clock; endpoints used by rule 1 |
| Report | `report.py` | STUB: verification PNG (coronal/sagittal projections) + HTML table |
| Scorer and tools | `scorer.py`, `sweeps.py`, `gallery.py`, `review_markers.py` | done |

Scores on the labelled cases (19 to 23) at the 5 mm cutoff: 17 of 19 references found, 16 false
positives, F1 0.65, mean ostium error 1.35 mm, direction error 26 degrees, seed on the right
branch 14 of 17. Invariants on all 25 cases: no crash, under 9 s per case. Files:
`results/dev_scores.txt`, `results/reference_ledger.txt`, `results/sweeps.txt`,
`results/invariants.txt`, and `results/dev_scores_skeleton_baseline.txt` for the before/after.

## Decisions made against the spec's first draft, with the evidence

All of these are written into SPEC.md D4 to D7 and section 1; the one-line versions:

1. **Rule 3 (path departure) is a flag, not a rejection.** Branches that leave the aorta and run
   along it are daughters: the PDF definition has no angle condition, two of 19 references do it,
   and the only such strip on the cases the rule was written for is subject 2's inferior
   mesenteric artery. No vein clears the arterial-phase threshold, so the rule had no benefit.
2. **Hugging strips get their ostium at the end that hugs tightest** (`ostium.strip_end`). On
   both reference strips that end is wider, brighter, tighter, with no vessel body beyond the
   wall; the other end is where the vessel lifts off.
3. **Rule 4 volume is the proximal 10 mm, not the watershed basin.** The basin floods every
   connected bright voxel (1 to 4 ml on real renals); the opened component is one sheet round the
   aorta (5 to 17 ml). Within 10 mm every reference is under 0.8 ml.
4. **Rule 4 bone = seed section runs out of the 16 mm window AND contains cortex over 1.5 x the
   lumen median.** Both signs on every vertebral contact on subjects 16 and 22, neither on any
   real vessel. Edge without cortex is a flag (reference 19/b3 merges with a neighbour).
5. **Rule 4 area growth is a flag** until the bowel cases (subjects 8, 12) are looked at: its
   only hits are real branches whose first 3 mm read narrow at 1.5 mm voxels.
6. **Rule 6 also merges same-voxel ostia and survivors whose 10 mm paths come within 4 mm.**
   Two daughters cannot share a lumen; one hugging vessel leaves several contact patches.
7. **D4 axis refinement is off** (`config.OSTIUM_USE_AXIS_REFINEMENT`): worse than the inscribed
   centre on 6 of 8 branches where it passed its own gate, at 1.5 mm resolution.
8. **Tracer fixes found by the ledger:** a split inside the wall layer is not a bifurcation; the
   axis fallback must keep the line inside the lumen (the loose version made 22 of 38 false
   positives).
9. **Scorer matching is cutoff-aware.** Plain Hungarian paired a near prediction with a far
   reference and then lost it to the cutoff.
10. **Two misses are resolution limits and are not chased:** 19/b2 (2.3 mm, one opened voxel) and
    23/b2 (two origins merged at the wall). The alternatives were measured and rejected.

## Open, waiting on the organisers (sent 13 Sep, SPEC section 6)

- Small vessels just above the inferior cut: annotated or not. One `near_cut_face` candidate per
  coarse case rides on this (the iliac division on 22; 7 to 8.5 mm vessels on 21 and 23).
- Borderline 2 to 2.5 mm detections: false positive, ignored, or hit.
- Whether the final references are swept for completeness. About ten of our 16 remaining false
  positives are 1 to 3 mm tubes at lumbar and IMA positions that look real on the gallery sheets.
- Matching cutoff (we assume 5 mm), subjects 18/24/25 in or out, scoring machine OS and Python.

## What remains, in order

1. `frame.py`: geodesic centreline (skimage route_through_array on the mask, cost from the inside
   distance), rotation-minimising frame, height and inter-branch spacing. Rule 1 picks up the
   endpoints automatically. Keep `end_faces()` and `height_clock()` as the interface.
2. `report.py`: unrolled clock map with spacing brackets, branch table with the flags, Plotly 3D,
   and verification PNGs for at least three cases committed (the PDF requires them).
3. Submission mechanics: dev-set predictions committed (`python scorer.py predict --cases all`
   into a committed folder), vendored wheels once the platform is known, an install test with
   networking off, README setup and run commands checked.
4. Rule 4 area growth: `python gallery.py --cases 8 12 --rejected`, look, decide.
5. Demo material: failure gallery sheets, the known-failure slide (subjects 18 and 24 are allowed
   to fail; 19/b2 and 23/b2 are the honest misses), runtime figure.

## How to work

- After any pipeline change: `python -m pytest tests`, then `python scorer.py predict --cases
  19-23`, `python scorer.py score --cases 19-23 --out results/dev_scores.txt`, `python scorer.py
  ledger --cases 19-23 --out results/reference_ledger.txt`, `python sweeps.py --out
  results/sweeps.txt`, `python scorer.py invariants --cases all --pred-dir out/predictions_all
  --out results/invariants.txt`. Commit the results files with the change. A change must improve
  or hold every labelled case (SPEC section 3), or come with a written anatomical argument.
- To look at a case: `python run.py ... --meta-output out/review/subjectNNN_meta.json --output
  out/review/subjectNNN.json`, then `python gallery.py --cases NN --pred-dir out/review` and read
  the sheets in `out/gallery/`; `review_markers.py` writes ITK-SNAP overlays (recipe in README).
- Judge every rule against anatomy and the annotations first, the numbers second. Do not change a
  constant to improve a score; run the sweep and keep the physical value unless the curve says
  the constant does not matter.
- All constants are in `config.py` with their justification; switched-off items carry the
  measurement that switched them off.
- Windows note: the repository has CRLF line endings on several files; git warns and converts.
  Long shell heredocs with quotes fail in this environment; write scripts to a file and run them.
