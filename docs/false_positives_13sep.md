# The twelve false positives on the labelled cases (13 Sep 2026, night)

Scores on cases 19 to 23 at the 5 mm cutoff: 17 true positives, 12 false positives, 2 false
negatives; precision 0.59, recall 0.89, F1 0.71; mean ostium error 1.35 mm; direction error
26 deg; seed inside the reference vessel on 14 of 17. Each false positive below was judged by eye
with `review_dense.py` (figures in `results/visual_checks/precision_review/`); "false positive"
means "no draft reference within 5 mm", nothing more. Details and evidence: SPEC.md section 1.

| Case, branch (wall patch) | Where | What the views show | Anatomically |
|---|---|---|---|
| 21 branch_003 (patch 4) | posterior-left, hugging the wall | one contact of the looping vessel | a real vessel, but neither the annotators nor we could tie it to the aorta; at most one of its three contacts can be an origin, so as three daughters it is false |
| 21 branch_004 (patch 5) | same vessel, 4 mm away | second contact of the looping vessel | same |
| 21 branch_007 (patch 10) | same vessel | third contact | same |
| 21 branch_005 (patch 7) | right posterolateral, 3.0 mm | tube followed 7 mm, continues over 15 mm in the coronal | real vessel; the case notes list this exact voxel as unresolved |
| 21 branch_008 (patch 12) | posterior-right, 2.9 mm | round dot in every section to 10 mm | real vessel, unannotated |
| 21 branch_010 (patch 14) | posterior, 3.0 mm, 8 mm below the left renal | the cleanest of all | real vessel, unannotated |
| 21 branch_006 (patch 8) | anterior corner of the inferior cut | vessel running down the anterior wall past the cut | real vessel, wrong point: not an origin here |
| 22 branch_008 (patch 23) | posterior-right, 3.2 mm | faint tube in the fat between aorta and vertebra | real vessel, unannotated; the least certain of the real ones |
| 22 branch_010 (patch 28) | anterior-left, 15 mm above the division | the inferior mesenteric artery 26 mm below its own origin, which we already report | real vessel, wrong point: a second contact of a reported daughter |
| 22 branch_009 (patch 26) | posterior-right, reads 2.0 mm | round dot in every section from 2 to 10 mm, reaching lumen brightness | real vessel; the 2.0 mm reading is partial volume at the origin |
| 22 branch_007 (patch 15) | anterior, reads 2.4 mm | small round dot to 6 mm, then lost | a small vessel at the resolution limit; the annotators withhold this band by policy |
| 22 branch_011 (patch 30) | anterior, reads 2.2 mm | straight-line fallback into homogeneous soft tissue at 100 to 120 HU, no tube in any section | genuinely false: bowel or mesentery clearing subject 22's 105 HU threshold |

Summary: six real unannotated vessels (the four 3 mm lumbars and the two borderline tubes on
22), two real vessels reported at a point that is not their origin, three contacts of one real
vessel with no established origin, one soft-tissue detection. If the final references add the six
unannotated vessels, precision on this set becomes about 0.79 with no code change.

## The one removed on 13 Sep (night)

Subject 20's fourth prediction (patch 4) sat on the top slice of the volume: its whole path lay
in the last voxel layer and every cross-section along it was half outside the image. Its 5 mm
followability cannot be verified in the data, which is the PDF's own eligibility test, and the
reference notes excluded it for the same reason. Rule 2's `path_at_image_edge` rejects any path
running within one native voxel of an image face in its first 5 mm; across all 25 cases it
removes only this one.

## The two false negatives

Both are resolution limits, present since the filters landed, recorded in SPEC.md section 1 and
not chased:

- **19 branch_002** (2.3 mm): its wall contact is a single opened voxel; the morphological opening
  erases it inside the wall layer and the march cannot start. Dropping the voxel floor recovers
  nothing; an outward-support wall layer was prototyped and rejected.
- **23 branch_002** (2.5 mm): two posterior lumbars with ostia 4 mm apart whose lumens merge at the
  wall at 1.5 mm voxels. We produce one instance with its ostium at branch_003's origin, so
  branch_002 is unmatched. Splitting the instance at the wall was tried and rejected (no neck at
  any prominence). The annotators note this pair as "separate ostia versus a short common origin".
