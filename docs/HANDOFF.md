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
  exit 0). 99 tests: `python -m pytest tests`.

## What is built

| Stage | File | State |
|---|---|---|
| Reading, crop, resample, index to mm | `io_utils.py` | done; gzip sniffing, tilted-header fallback (subject 24) |
| Candidates (threshold, shell, opening) | `candidates.py` | done |
| Instances (wall patches, watershed) | `instances.py` | done; `branch_voxels` is the per-branch voxel set downstream |
| Ostium | `ostium.py` | done; inscribed circle snapped to the mask; axis refinement off by default; contact-end rule for hugging strips |
| Tracing | `tracing.py` | done; march with bifurcation stop, axis fallback, chord direction, 0.25 mm plane radius |
| Filters | `filters.py` | done; rules 1, 2, 4 (volume, bone), 6, 7 reject; rule 3 and area growth are flags |
| Frame | `frame.py` | done; geodesic centreline, end-face endpoints (rule 1), anatomical clock, radius profile, spacing |
| Report | `report.py` | done; verification PNG, clock map PNG, self-contained HTML with flags, spacing and Plotly 3D |
| Scorer and tools | `scorer.py`, `sweeps.py`, `gallery.py`, `review_markers.py` | done |

Scores on the labelled cases (19 to 23) at the 5 mm cutoff: 17 of 19 references found, 13 false
positives, F1 0.69, mean ostium error 1.35 mm, direction error 26 degrees, seed on the right
branch 14 of 17 (the three misses and the 26 deg mean are explained in SPEC section 1, instance-quality
review: merged pair on 23, a bend on 19, a strip end on 20; not chased). Of the 13: three are one looping vessel on subject 21 that the annotators also
could not tie to the aorta (known failure class), three are 2.0 to 2.4 mm borderline origins,
two are at the cut faces, five are unflagged 3 mm vessels at lumbar positions (SPEC section 1,
false-positive review). Invariants on all 25 cases: no crash, under 10 s per case. Files:
`results/dev_scores.txt`, `results/reference_ledger.txt`, `results/sweeps.txt`,
`results/invariants.txt`, `results/dev_scores_skeleton_baseline.txt` for the before/after, `results/predictions/` (the
JSON for all 25 dev cases) and `results/visual_checks/` (verification and clock-map PNGs for 3, 16, 17, 19 to 23).

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
5. **Rule 4 area growth stays a flag.** Its only labelled hits are real branches whose first
   3 mm read narrow at 1.5 mm voxels; the bowel cases were checked on 13 Sep (subject 12 flags
   nothing, subject 8's two hits are already rejected as duplicate and bone), so arming it
   removes nothing anywhere in the dev set.
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
11. **The clock's 12 o'clock is the patient's anterior projected at each level, not a transported
    frame.** The rotation-minimising transport the spec first asked for drifts 43 deg from the
    patient's anterior on subject 3 and 35 deg on subject 6 (torsion in the S-bends); it is kept
    as the `twist_deg` diagnostic. The two agree on a planar bend. SPEC D8.
13. **The terminal iliac division is rejected** (rule 1 `iliac_division`): two alike lumens over
    6 mm 5 mm beyond the inferior face mean the segment ends at the bifurcation, and a candidate
    on that face with an origin over 5 mm is an iliac. PDF and the reviewer checklist both
    exclude the division; measured on all 25 cases (fires on 17, 22, 23; not on 19 to 21).
14. **A seed inside a bright region over 16 mm across is not a vessel** (rule 4 `blob_at_seed`):
    the disc contact on subject 22 and the heart and arch on subject 25; no reference.
15. **A survivor on the origin voxel of a structurally rejected patch goes with it** (rule 6
    `same_origin_rejected`), otherwise rejecting a blob releases its twin patch.
16. **Subject 21's looping vessel is a recorded failure class**, not a rule: hug ratio, opposed
    directions and origin-to-seed taper were each measured and each also fires on a real vessel.
17. **The three off-label seeds and subject 23's direction error are resolution and reference
    limits, not tracer bugs** (SPEC section 1, instance-quality review): 23 is the merged lumbar
    pair (seed on the neighbour's mask, matched to the other), 19 a bend the straight fallback
    misses by 2 mm, 20 a strip whose two ends are alike. All 17 seeds lie in our own lumen.
18. **Tracer stabilisation (lateral cap, turn clamp) was implemented, swept and reverted**: every
    setting lost a labelled true positive or added false positives (the grid is in SPEC D5).
    The tracer's fragility is doing eligibility work; a smoother march needs an explicit
    followability test first.
12. **Centreline endpoints are end-face centroids**, not the geodesic rim points: the rim-to-axis
    stretch of the path is cut one radius in and the face (boundary voxels with normals within
    60 deg of the tangent) is averaged. Rule 1 picks the endpoints up automatically; the labelled
    scores did not move.

## Open, waiting on the organisers (sent 13 Sep, SPEC section 6)

- Small vessels just above the inferior cut: annotated or not. Two `near_cut_face` candidates ride
  on this (an 8.5 mm vessel at 21's cut, a 2.5 mm posterior track at 20's superior crop limit);
  the iliac divisions on 22 and 23 are now rejected by policy (rule 1). The reference checklist
  and notes exclude or leave unresolved every cap-adjacent candidate they mention (SPEC section 6).
- Borderline 2 to 2.5 mm detections: false positive, ignored, or hit.
- Whether the final references are swept for completeness. Five of our 13 remaining false
  positives are unflagged 3 mm tubes at lumbar and IMA positions that look real on the gallery
  sheets; one coincides with a structure case 21's notes list as unresolved.
- Matching cutoff (we assume 5 mm), subjects 18/24/25 in or out, scoring machine OS and Python.

## What remains, in order

1. Submission mechanics that wait on organiser question 6: vendored wheels (`pip download -r
   requirements.txt -d vendor/` on the right platform), the offline install test with networking
   off, and the README setup command switched to `--no-index --find-links vendor`. Everything
   else is in place: dev-set predictions in `results/predictions/`, visual checks in
   `results/visual_checks/`, README setup and run commands checked.
2. Demo material: failure gallery sheets, the known-failure slide (subjects 18 and 24 are allowed
   to fail; 19/b2 and 23/b2 are the honest misses), runtime figure, and the clock maps and 3D
   views from `python run.py ... --report-dir` for three cases.
3. When the organisers answer (SPEC section 6): question 1 decides the `near_cut_face` candidates
   (rule 1 at the inferior cut), question 2 the `borderline_diameter` handling, question 3 whether
   the remaining false positives are ours, question 4 the scorer cutoff.

## How to work

- After any pipeline change: `python -m pytest tests`, then `python scorer.py predict --cases
  19-23`, `python scorer.py score --cases 19-23 --out results/dev_scores.txt`, `python scorer.py
  ledger --cases 19-23 --out results/reference_ledger.txt`, `python sweeps.py --out
  results/sweeps.txt`, `python scorer.py invariants --cases all --pred-dir out/predictions_all
  --out results/invariants.txt`. Commit the results files with the change. A change must improve
  or hold every labelled case (SPEC section 3), or come with a written anatomical argument.
- To look at a case: `python run.py ... --meta-output out/review/subjectNNN_meta.json --output
  out/review/subjectNNN.json --report-dir out/review` gives the clock map, the projections and the
  HTML with the 3D view; then `python gallery.py --cases NN --pred-dir out/review` and read
  the sheets in `out/gallery/`; `review_markers.py` writes ITK-SNAP overlays (recipe in README).
- Judge every rule against anatomy and the annotations first, the numbers second. Do not change a
  constant to improve a score; run the sweep and keep the physical value unless the curve says
  the constant does not matter.
- All constants are in `config.py` with their justification; switched-off items carry the
  measurement that switched them off.
- Windows note: the repository has CRLF line endings on several files; git warns and converts.
  Long shell heredocs with quotes fail in this environment; write scripts to a file and run them.
