# Branchseed Challenge: Team Spec

Toralis Labs healthcare track, Battle of the Schools hackathon. Reference document is the updated Branchseed Challenge PDF (the version with "could be in JSON, or something creative" and the clinician-display bullet). Team of three. Written 12 Sep 2026.

---

## 1. Task in one paragraph

Input: a contrast-enhanced abdominal CT (NIfTI) and a binary mask of the aortic lumen on the same grid. Output: one JSON per case listing every artery that leaves the aorta directly, with an ostium centre (mm), a seed point 5 mm into the branch (mm), a unit direction vector, and a local radius (mm). No anatomical names. Must run from one command on a CPU-only laptop (4 cores, 8 GB, no internet) in about 60 s per case. Also required: a visual check for at least three cases, and a display of the results that is useful to a clinician.

Scoring: branch discovery F1 (45%), ostium distance (25%), seed/direction/radius quality (15%), runtime and memory (10%), reproducibility (5%).

What we learned from looking at subjects 1 to 3 in 3D Slicer (volume rendering, CT-AAA preset, mask overlaid):

- Real anatomy is far messier than the diagram. Many small branches (lumbar arteries) come off the back of the aorta. These may count as daughters if they clear the organiser's minimum origin size. No size heuristics of our own.
- Large vessels run parallel to the aorta. Subject 2: the SMA runs in contact with the aorta for several cm with five or six jejunal side branches of its own inside the search shell. Subject 1: a large parallel vessel, possibly the IVC. These sit inside any search shell without being ostia at that height. D6 rule 3 exists because of subject 2.
- The supplied mask underfills the true lumen by a variable amount. Subject 1 has a solid bright ring outside the mask; subjects 4 to 12 range from 2% to 37% of the one-voxel ring being bright. The ring-removal step in D2 (morphological opening) exists because of this; a fixed dilation or a connectivity-based grow both fail.
- Coverage varies. Bright vessels continue above and below the mask and must not be reported.
- The aorta curves strongly (subject 2 bows forward; subject 3 winds side to side around the spine in a full S). Axial slices can cut it obliquely or twice. End faces of the mask are not axial planes. The centreline method (frame.py) and the cropped-end rule (D2) were changed because of subject 3.
- A branch can originate right at a cropped end (subject 3: a large bright vessel leaves the aorta at the superior cut). A blanket exclusion zone near end faces would drop it. See D2 and open question 6.
- Later contrast phase in some scans (subject 3: strongly enhanced kidneys, left renal vein bright and crossing in front of the aorta). Veins that touch the aorta tangentially are a false-positive class; see D6 rule 3.
- Masks so far are single connected components (verified for subject 2), but edges are jagged, suggesting semi-automatic segmentation. Expect single-voxel wall bumps that survive thresholding; D6 rule 7 handles them.
- Tortuous branches (subject 2: a renal that loops hard within the first cm). D5's marching tracer is required; a straight-line axis would misplace the seed.
- Known limitations we accept rather than fix: recall on vessels under about 2 mm at 1.5 mm native resolution (partial volume makes them faint smears, and the annotators faced the same limit); radius precision on the coarse cases; the opening's minimum survivable vessel size (about 1.6 mm), which is revisited once the organisers announce the minimum origin size with the final dataset.
- Some volumes are pre-cropped to the mask's z-range (subject 4: 180 slices, mask spans 0 to 179), so there is no aorta continuation beyond the end faces. Others (subject 1) have bright vessels continuing above the mask. The end-face logic must handle both: an end face at the image boundary rejects nothing; an end face inside the image needs D6 rule 1. One test case for each.
- Bone sits inside the 15 mm shell. Subject 4 has six bright blobs of 1 to 2 ml within 15 mm of the aorta, all vertebral bodies, all clearing the threshold. They do not touch the 2 mm wall layer on that case, but will on any case where the aorta lies on the spine. D6 rule 4 gained a volume cap because of this.
- Aortic diameter varies widely (subject 4: 11 to 18 mm, small and non-aneurysmal). No fixed size assumptions about the parent either.
- The triage pass (triage.py) runs in about 5 s per case and produces the stats above plus coronal and sagittal projections. It is the tool for building the case atlas across all 25 cases. Results for subjects 4 to 12 are in atlas_04_12.json.

Findings from the triage batch (subjects 4 to 12):

- Every one of the nine volumes is pre-cropped in z to exactly the mask's extent (mask touches slice 0 and the last slice in all cases). So far there is no "aorta continuing past the cut" in the image at all; D6 rule 1 stays as a guard but is probably never exercised. What does matter at the ends: a branch leaving within 5 mm of the image boundary cannot be traced 5 mm and is ineligible by the spec's own rule, and all tracing code must bounds-check at the volume edge. In-plane, some volumes are cropped (240 to 440 px) and some are full 512 × 512 (subjects 8, 12); the full ones carry bowel and organs into the shell.
- Spacing: z is 0.8 mm in all nine; in-plane ranges 0.63 to 0.96 mm. Mild anisotropy, but the 0.96 mm cases lose small vessels to partial volume.
- Masks are int16 or uint8 with values 0/1 and grids always match the CT. Two of nine have stray fragments.
- Inside-mask HU ranges 242 to 440 with std 27 to 83 on subjects 4 to 12. A fixed 150 HU floor was tested and merged patches on the low-contrast cases; the fixed floor was later replaced by the relative floor in D1 once subjects 16 to 20 arrived.
- No aneurysm in subjects 4 to 12 (max equivalent diameter 23 mm) and almost no calcification (at most 11 voxels over 600 HU near the wall). The dataset may be largely non-aneurysmal; keep the plaque and aneurysm handling but do not over-invest.
- Tortuosity 1.01 to 1.15; subjects 6, 8 and 9 curve the most. Subject 6 is the tortuous regression case.
- Subject 12's mask includes a small bulge at one branch root (the ostium funnel). The ostium snap to the raw mask boundary still works, but the bulge slightly shifts where "the wall" is; worth checking against the reference if subject 12 is in the dev set.
- Runtime and memory: a full-volume distance transform on the two 512 × 512 × 180 cases got the triage process killed in an 8 GB sandbox. Cropping to the padded mask bounding box first fixed it and brought each case to about 5 s. This is D10's "crop before anything expensive" rule demonstrated.

Findings from the second triage batch (subjects 13 to 20):

- Subjects 13 to 15 match the first acquisition (0.8 mm z, 512 × 512 in-plane, one stray single-voxel fragment on 15). Nothing new.
- **Subjects 16 to 20 are a different dataset.** 1.5 mm isotropic voxels, gzipped files under a `.nii` extension, uint8 masks, and much wider variation: lumen median HU from 110 (subject 18) to 556 (subject 19); segment lengths of 48 to 69 mm (17, 19, 20) versus 285 mm (18); aortic diameter 26 to 29 mm on subject 16 (ectatic) and up to 31 mm on 18. At 1.5 mm a renal artery is three voxels wide and a lumbar is one. Expect the hidden set to contain more of these.
- Subject 18 is a known bad case: lumen at 110 HU (barely enhanced), 1.5 mm voxels, a mislabelled 4241-voxel blob in the mask, and 41 wall patches at the 100 HU floor. It belongs in the regression set as the case that is allowed to score badly, and in the demo as the honest failure example. Do not tune anything to make it look good.
- Subjects 19 and 20: the mask ends mid-vessel with the aorta continuing inside the image, producing a single 440 to 460 mm² wall patch at the cut. Subject 20's mask also tapers to a 5 mm point at that end rather than a flat face. These are the cases D6 rule 1 exists for.
- Subject 16: coarse, dilated, and low-contrast (inside-mask HU 5th percentile is 16, so the mask includes some non-enhancing voxels, possibly thrombus). Eleven patches after resampling, three of them renal-sized. A reasonable hard case.
- Wall-patch counts after the full front half (crop, resample, threshold, opening): 3 to 11 on every case except subject 18. The top three or four patches per case are 60 to 240 mm², consistent with celiac, SMA and renal origins.

Findings from the reference release (draft annotations for cases 19 to 23, under `docs/references/`):

- **All five references are coarse (1.5 mm) cases.** Assume the hidden evaluation set is drawn from the same acquisition. Cases 1 to 15 are still useful for robustness but are no longer the priority; the regression set and every tuning decision centre on 16 to 25.
- **The branches are small.** 19 branches across five cases, origin diameters 2.2 to 6.5 mm, ten of them under 3 mm. Cases 19, 20 and 23 are short infrarenal segments whose only daughters are lumbar-sized posterior vessels. Detection of faint 2 to 3 mm vessels at 1.5 mm resolution is where the 45% lives, not renal and SMA origins.
- **Direction is the ostium-to-seed chord** in every reference branch (angle to `normalise(seed − ostium)` is 0°). D5 changed to match.
- **Radius is null on 16 of 19 references.** Low effort on radius; see D5.
- **Ostium is on the supplied mask boundary** and seeds are 5 mm along the guide; straight-line ostium-to-seed distance is 4.1 to 5.0 mm.
- **Our threshold formula matches the annotators' per-branch thresholds:** ours 217 to 254 HU on 19 to 21 and 105 to 164 on 22 and 23; theirs 220 to 252 and 135 to 180. D1 needs no change.
- **Front-half recall on the references is 19/19:** every reference ostium is within 1.6 mm (median under 1 mm) of an opened wall patch, and every one reaches the full shell through raw thresholded voxels. The opening must not be applied to the growth region (D2 changed).
- **Precision is the whole problem:** 83 wall patches survive a 3-voxel minimum and a 5 mm reach test across the five cases, against 19 references. Each case has one 700 to 900 voxel patch that is the aorta continuing past the mask end. D6 rules 1, 3 and 4 must remove about 60 patches while keeping all 19; this is the first ablation now that labels exist.
- **The answer key is a draft and admits it.** "Every case requires expert review"; "does not guarantee that every eligible origin has been found"; excluded borderline vessels are listed per case. Expect noise in the hidden references too. Do not chase individual dev-case misses that look like annotation gaps; record them in the failure ledger with the reference's own uncertainty note.
- **The annotation schema is a superset of the required output** (`ostium_xyz_mm`, `seed_xyz_mm`, `radius_mm`, `direction_xyz` plus guides and per-branch masks). The scorer reads it directly. The daughter mask volume gives an exact "seed inside the matched daughter" test; the tracing guide gives "direction follows the proximal path".

Findings from the final batch (subjects 1 to 3, 21 to 25), completing the atlas (atlas_all.csv, atlas_all.json):

- Subjects 1 to 3 confirm the first dataset (0.8 mm z). Subject 1's ring bright fraction is 0.48, the heaviest underfill in the set, and the opening handles it (11 patches, top three at renal scale). Subject 3's mask does not reach the top of its volume, so the aorta continues above the cut inside the image, matching what the Slicer screenshot showed.
- **Dataset split is 15 fine (subjects 1 to 15, 0.8 mm z) and 10 coarse (16 to 25, 1.5 mm isotropic, gzipped, uint8 masks).** In the coarse set, the mask usually ends mid-vessel at the inferior end (17, 19, 20, 21, 22, 23) and on subject 25 at both ends, so the aorta-continuing case (D6 rule 1) is the norm there, not the exception.
- **Subject 25's mask is the entire aorta**: ascending aorta, arch, descending and abdominal, 343 mm long, aneurysmal at the root (45 mm), tortuosity 1.22, with the heart and pulmonary artery pressed against the ascending segment. The PDF says "abdominal aorta"; this case is not that. 38 wall patches, the largest 1015 mm² (heart contact or arch underfill). The pipeline is anatomy-agnostic so it will run, but the clock convention (D8) loses meaning around the arch, and the frame must not assume a roughly vertical vessel. Ask the organisers whether arch branches are in its reference (open question 9). Decision: it is out of scope. It runs like any other case, its output is whatever it is, and it is not in the regression set.
- **Subject 24 is unenhanced** (lumen median 78 HU), dilated (26 to 29 mm), and has the tilted non-orthonormal header. Like subject 18, it is a known-bad case: 27 patches at the 100 HU floor, mostly noise. Second entry on the allowed-to-fail list.
- **Subject 21 is the calcified, tortuous case**: 191 voxels over 600 HU within 3 mm of the wall (real plaque, the only substantial example in the set), tortuosity 1.21, lumen 564 HU. Regression case for D6 rule 2.
- Subject 22: 210 mm segment, low enhancement (233 HU), 28 patches at a 105 HU threshold. Noisy but workable; the second-largest patch (279 mm²) is worth checking by eye.
- Subject 23: 50 mm segment, three tiny mask fragments, one continuation patch. Routine after cleanup.
- Across all 25: lumen median HU 78 to 564; segment length 48 to 343 mm; median aortic diameter 12 to 26 mm, max 45 mm; tortuosity 1.01 to 1.22; ring underfill 2% to 48%; substantial plaque on one case; two unenhanced cases; one non-orthonormal header; four masks with fragments; one mask with a large mislabelled blob; five files gzipped under a .nii name. No two assumptions from the PDF alone would have survived contact with this set.

---

## 2. Decision log

Each decision lists the options weighed, the choice, and why. The reasoning is the point: every rule in the code should trace back to a line here.

### D1. Candidate generation

Options: fixed HU threshold (A), Frangi vesselness (B), vesselFM foundation model (C), adaptive threshold from the aorta mask's own intensity statistics (D).

**Choice: D primary, B secondary.**

- Contrast timing varies between patients, so a fixed HU cutoff is unreliable. The mask tells us exactly what enhanced blood looks like in this scan. **Threshold = max(mean − 2.0 × std, 0.45 × median, 100 HU)**, all computed over HU inside the mask.
  - History: started as mean − 1.5 std floored at 120. Subject 4 (440 ± 83) showed 1.5 std was too strict (316 HU, 8 wall blobs) versus 2.0 std (275 HU, 14 blobs, the extras being 250 to 300 HU lumbar-sized vessels), so it became 2.0 std floored at 200. Subjects 16 to 20 then broke the fixed floor: their inside-mask HU std is 100 to 165 (coarse voxels, poor enhancement), so mean − 2 std goes negative, and subject 18's lumen median is only 110 HU, below the old floor, giving zero candidates. A relative floor of 0.45 × median adapts to the enhancement level; the absolute 100 HU floor keeps unenhanced soft tissue (40 to 80) out. Median − 2 MAD was also tested and rejected: on well-enhanced homogeneous lumens the MAD is tiny and the threshold lands too high (386 on subject 4), missing dim branches.
  - Result across subjects 4 to 20: thresholds from 100 (subject 18) to 313 (subject 15); fine-resolution cases unchanged from before.
- Frangi is not used to find candidates. It is computed on the cropped block and used later (D6) as a tube-ness score to reject non-vessel candidates.
- vesselFM rejected as primary: unknown CPU runtime on a 3D volume, torch dependency, weights must be bundled for the offline judge machine, and unknown zero-shot quality on abdominal CTA. Downstream code treats "vessel mask" as a swappable input, so it can be tried later if someone finishes early.

### D2. Search region and cropping

Options: dilation in voxels (A), dilation in mm via distance transform (B), bounding-box crop with no shell (C).

**Choice: B, 15 mm shell, crop first.**

- Voxel spacing varies between scans, so all distances are in mm.
- Sequence: bounding box of mask, pad 15 mm per axis, crop CT and mask to that box. Distance transform on the cropped mask with real spacing. Shell = 0 < distance <= 15 mm.
- 15 mm because the spec traces up to 10 mm, and we need margin to see whether a common trunk splits before or after that.
- Cropping first is where most of the runtime and memory saving comes from.
- **File reading:** subjects 16 to 20 are gzip-compressed files with a plain `.nii` extension. SimpleITK chooses its reader by extension and refuses them. io_utils must sniff the first two bytes (`1f 8b`) and treat the file as `.nii.gz` regardless of name. The judges' command uses `.nii.gz`, the dev data uses `.nii`, and at least one of those lies.
- **Header fallback (added after subject 24):** subject 24's NIfTI header has direction cosines that are off-orthonormal by rounding, and SimpleITK refuses the file outright ("ITK only supports orthonormal direction cosines"). The rotation underneath is real: the grid is tilted about 3.5° from standard, so treating it as axis-aligned would put physical coordinates 5 to 6 mm off at the far end of the volume. io_utils catches that error, loads with nibabel, converts the affine from RAS to LPS, takes spacing as the column norms, replaces the rotation with its nearest orthonormal matrix (polar decomposition via SVD), sets the origin from the translation, and rebuilds a SimpleITK image. TransformIndexToPhysicalPoint on that image then gives the same mm as the original header to within rounding. Log when the fallback fires.
- **Mask cleanup (added after subjects 8, 10, 18, 23):** masks can contain stray fragments (subject 10: twelve fragments of 1 to 6 voxels; subject 8: one 80-voxel island) and occasionally a large mislabelled blob (subject 18: a 4241-voxel component 47 mm from the aorta, 9% of the mask). Keep the largest connected component as the parent; log fragment count and sizes; warn loudly if the discarded voxels exceed 1% of the mask. Mask dtype varies (int16 or uint8); always test `> 0`.
- **Resample the cropped block to 0.8 mm isotropic (added after subjects 16 to 20):** the dev set contains two acquisitions: 0.8 mm z with 0.63 to 0.96 mm in-plane (subjects 4 to 15) and 1.5 mm isotropic (subjects 16 to 20). Every mm-based parameter in this spec (2 mm wall layer, 0.8 mm opening, 5 mm seed, 1 mm march step) converts to a different voxel count on the two grids, and at 1.5 mm the opening's minimum 1-voxel radius becomes 1.5 mm and erased every wall patch on subjects 16 and 18. Fix: after cropping (never before, for memory), resample the block to 0.8 mm isotropic with linear interpolation for the CT and nearest-neighbour for the mask, then run everything on that grid. Physical coordinates are preserved by the resample, so TransformIndexToPhysicalPoint on the resampled image gives correct mm and the spec's coordinate requirement is still met. Upsampling adds no information (coarse cases stay blurry) but it makes the parameters mean what they say. Cost: crops on the coarse cases are small, 0.4 to 5.4 s per case including the resample.
- **Ring removal (revised after subjects 4 to 12; replaces the earlier "grown aorta" step):** the mask underfills the lumen by a variable amount (bright fraction of the one-voxel ring outside the mask ranges from 2% to 37% across cases, and subject 1 had a solid ring). A connectivity-based grow step was tried and rejected: it cannot tell a lumen ring from a branch root, since both are bright and touch the mask, so it swallowed the first 2 mm of every branch and on subject 10 produced zero wall patches. The fix is morphological: take the bright voxels in the 15 mm shell and apply a binary opening with a 0.8 mm ball (1 voxel radius on these spacings). The ring is one to two voxels thick and vanishes; branch roots are 3 mm or wider and survive. Tested on subjects 4 to 12: wall-patch counts dropped from 137 to 703 raw fragments per case to 4 to 18 patches, with the four largest per case consistently 50 to 170 mm², which is the scale of celiac, SMA and renal origins. Lumbar arteries near 1.6 mm diameter are at risk from the opening; if the organiser's minimum origin size is below that, drop the ball radius to 0.6 mm and re-test.
- **The opening defines wall patches only; outward growth and tracing use the raw thresholded voxels (added after the reference release).** Tested against the 19 reference ostia on cases 19 to 23: all 19 sit within 1.6 mm of an opened wall patch, but four of the small (2.3 to 2.6 mm) branches fail to reach 5 mm when their blob is measured through the opened voxels, because the opening fragments thin vessels beyond the wall. Through the unopened voxels at the same threshold, all 19 reach the full 15 mm shell. So D3's watershed and D5's tracer operate on `bright & shell`, seeded from the opened wall patches.
- All detection geometry (shell, wall layer, distance transform) is relative to the raw largest-component mask. The reported ostium is snapped to that mask's boundary, which the reference package confirms is the annotators' convention ("the supplied parent-lumen mask boundary is the operational origin boundary"; every guide start was adjusted onto it).
- **Cropped ends (revised after subject 3):** end faces are not axial planes and a real branch can originate within a few mm of a cut, so a blanket exclusion zone is wrong. Instead, identify the end faces as the mask boundary regions nearest the two centreline endpoints, and reject a candidate as "the aorta continuing" only if (a) its wall-contact patch area exceeds 40% of the aorta's local cross-sectional area and (b) its direction is within 20° of the local centreline tangent. A 3 mm renal next to the cut survives; the aorta's continuation does not. This check runs in D6 as rule 1 once direction is known; the patch-area part can run early to save tracing time.
- One helper owns the crop offset and index-to-mm conversion (see data contract) so coordinate bugs have exactly one place to live.

### D3. Instance separation

Options: 3D connected components on all candidates (A), connected components on the thin wall-contact layer only (B), unrolled-wall 2D map (C), watershed from wall seeds (D).

**Choice: B defines instances, D grows them. C is built as a separate coordinate and display module, not as the grouping step.**

- The spec defines instances by separation at the aortic wall: two nearby origins are two instances if separate at the wall; a common trunk is one instance even if it splits later. B implements that definition literally: wall layer = opened candidate voxels (D2) within 2 mm of the mask surface; connected components there are the instances. Measured patch counts on subjects 4 to 12 after opening: 4 to 18 per case, before any D6 filtering.
- D (watershed seeded from those components through the full 15 mm shell of raw thresholded voxels, not the opened set) assigns every candidate voxel to exactly one branch, so two branches that touch further out stay separate. Those voxels feed D4 and D5.
- C produces the same instances as B (same wall voxels, same connectivity) with added projection risk (seam at 0/360, smearing where the aorta curves, centreline noise in aneurysms). So it buys no accuracy in grouping. What C provides that B cannot is a coordinate frame (height along the aorta, clock angle) that the clinician display needs. It is therefore built separately (D8), consumes the engine's outputs, and the engine never depends on it. Optional late cross-check: blob count on the map vs component count from B; disagreement flags a case for review.
- Known weakness to state in the demo: a branch that hugs the wall before leaving gives a long thin wall component. D4 handles the ostium for that case; D6 handles the false-positive version.

### D4. Ostium localisation

Options: centroid of wall patch (A), centroid snapped to surface (B), branch-axis intersection with aorta surface (C), centre of largest inscribed circle in the wall patch (D).

**Choice: D for the initial estimate, C to refine, agreement check between them, snap to raw mask boundary last.**

- D: distance transform restricted to the wall patch, take the voxel furthest from the patch edge. Handles irregular and elongated openings better than a centroid at the same cost. A is dropped because D replaces it.
- C: fit the branch axis (PCA on watershed voxels within the first 10 mm), walk it back to the aorta, take the intersection with the surface. Fixes shallow-angle branches where the wall patch is smeared along the wall (common for renals).
- If C's point is within 4 mm of D's point, use C. Otherwise fall back to D. The check means a bad axis fit never makes things worse than the baseline.
- Final step: snap the ostium to the nearest raw-mask boundary voxel. The reference annotator clicked on the aortic wall; scoring is straight-line distance; a point 3 mm inside the lumen or 3 mm into the branch loses 3 mm for nothing.

### D5. Proximal tracing (seed, direction, radius)

Options: straight line from the axis fit (A), skeletonize the watershed region (B), slice-and-centroid marching (C), fast marching from the ostium (D).

**Choice: C, with A as fallback.**

- C directly implements the spec sentence "trace up to 10 mm or until the first bifurcation, whichever occurs first." Start at the ostium with the axis direction, step 1 mm, take the cross-section of branch voxels perpendicular to the current direction, update direction toward the cross-section centroid, repeat. Stop at 10 mm of path length or when the cross-section splits into two blobs.
- All three outputs from one loop: seed = point at 5 mm path length; direction = the chord from ostium to seed, normalised. This replaces the earlier PCA tangent: in all 19 reference branches on cases 19 to 23, `direction_xyz` equals `normalise(seed − ostium)` to the degree, so the chord is the answer key's convention and a path tangent would score worse on curved branches. (PCA with the sign fix is kept in the code as a diagnostic only; a large angle between PCA axis and chord flags a path that doubled back.)
- Radius = area-equivalent radius sqrt(area / pi) of the thresholded cross-section on a plane perpendicular to the chord at the seed, resampled at 0.25 mm, which is the reference package's own method. Radius is deliberately low effort: 16 of the 19 reference branches have `radius_mm: null` ("search envelope limited"), so radius can only be scored on three branches in the dev references and probably few in the hidden set. Report it, clamp to [0.5, 8] mm, and do not tune it.
- The bifurcation stop matters: a common trunk splitting at 7 mm must report the trunk's direction, not an average of its children. B and D would average or pick a child.
- B rejected: skeletons are noisy exactly at the junction, where precision matters most. D rejected: needs extra machinery for radius and bifurcation with no accuracy gain.
- A fallback: if the march fails (under 5 mm of voxels, cross-section vanishes, direction flips), use the axis fit. If even that gives under 5 mm, the branch fails eligibility and is dropped.
- Radius sanity: if the area-equivalent radius exceeds half the local aortic radius, the cross-section grabbed adjacent tissue; report the largest inscribed circle instead and flag it.
- Step in mm and resample cross-sections with real spacing. Anisotropic voxels bias area otherwise.

### D6. False positive filtering

Options: hard rules only (A), tiny trained classifier (B), rules for structural classes plus classifier for the residual (C).

**Choice: A, with a fixed threshold. B held in reserve only if the dev set turns out large enough.**

- With 3 to 5 labelled cases, a learned boundary overfits and we cannot detect it until the hidden set. Rules encode the definition of a branch, not the dev set's quirks, so they generalize.
- Every rule maps to a bullet in the spec's "important cases" list, which is the demo story.
- Default on borderline: keep if the branch passes the 5 mm trace, reject only on a clear rule hit. Precision and recall are weighted equally, and a candidate that survives 5 mm of tracing is more likely real than fake.

Rules, one per false-positive class:

1. Cropped ends / aorta continuing past the mask. Reject if the candidate's wall patch touches an end face (within 3 mm of a centreline endpoint) AND its direction is within 20° of the local centreline tangent. Patch area over 40% of the aortic cross-section is a supporting signal only, not a discriminator: renal origin patches measure 150 to 240 mm² on this data, the same as the aortic cross-section. The continuation patches on subjects 19 and 20 are 440 to 460 mm² (about twice the cross-section) so a ratio above 1.5 can short-circuit the direction test. No distance-based exclusion zone (subject 3 has a real branch at the cut). This rule is exercised by subjects 17, 19 and 20, where the mask ends mid-vessel with the aorta continuing inside the image, and by subject 20 whose mask tapers to a 5 mm point at one end rather than a flat cut.
2. Calcified plaque. Very bright, touches the wall, not a tube. Reject if traced path length is under 5 mm (this is also the spec's eligibility rule). Secondary: mean Frangi response below threshold.
3. Adjacent parallel or crossing vessels (SMA running alongside, IVC, left renal vein crossing anteriorly). Primary test: reject if the traced path (D5) is still within 3 mm of the aorta surface at its far end (10 mm, or the bifurcation if earlier). A real branch, even a shallow-angle renal, has moved more than 3 mm away from the aorta by the time you have walked 10 mm along it; a vein or the SMA lying against the wall never leaves. Secondary signals: angle between direction and outward wall normal above 70°, patch aspect ratio above 3. The path-departure test is primary because it uses the whole 10 mm of geometry rather than the angle at the ostium, which is measured exactly where the geometry is least reliable. This is the highest-priority rule and the one most likely to cost real renals if wrong; it gets the first sensitivity sweep (departure distance 2 to 4 mm, angle 60° to 80°) and is tested on subjects 2 and 3 first.
4. Bowel, organ tissue and bone. Low Frangi, cross-section grows instead of staying constant. Reject if cross-section area more than doubles over the first 5 mm. Also reject if the candidate's watershed region exceeds 1 ml: no proximal branch segment within 15 mm is anywhere near that volume, but a vertebral body touching the wall layer is (subject 4 has 1 to 2 ml bone blobs inside the shell), and so are the heart chambers and pulmonary artery pressed against the ascending aorta on subject 25 (14 blobs over 1 ml in its shell).
5. Branches of branches. Impossible by construction: instances require wall contact with the parent. Stated invariant, no rule.
6. Duplicates. Two ostia within 4 mm with directions within 20° are one branch. Merge, keep the larger wall patch. Genuinely separate nearby origins survive because they have separate wall patches and diverging directions.
7. Noise and sub-threshold vessels. The organisers' minimum origin diameter is 2.0 mm, estimated in a plane perpendicular to the path at the origin. Estimate the origin diameter the same way (area-equivalent, first perpendicular cross-section outside the mask boundary) and reject below 2.0 mm; flag 1.5 to 2.5 mm as `borderline_diameter` in the display. Both excluded-candidate files in the reference package are vessels the annotators saw at 1.5 to 2.5 mm and withheld; detecting those counts as a false positive, so the estimate must be honest, not generous. Also drop wall patches under 3 voxels on the 0.8 mm grid as pure noise.

Every rejection is logged as (label, rule name, value). That log is the tuning tool and the "known failure cases" slide.

### D7. Validation strategy

Options: eyeball (A), reimplement the scorer (B), scorer plus invariant checks on unlabelled cases (C).

**Choice: C.**

- Scorer: one-to-one matching via Hungarian assignment (scipy.optimize.linear_sum_assignment) on ostium distance with a cutoff. Report at 3, 5 and 8 mm cutoffs; 5 mm is the working value since the organisers have not specified one. The scorer reads the reference `annotations.json` files for cases 19 to 23 directly and uses the per-case daughter mask volume for the seed-on-daughter test and the tracing guide for the direction test. Build it on day one; nothing merges without a scorer run from then on. Metrics per case and aggregated: precision, recall, F1, mean matched ostium distance, direction angle error, radius absolute error, seed-inside-branch.
- Invariants on all 25 cases: every seed in a bright voxel, every direction points away from the aorta, radius between 1 and 8 mm, no two ostia within 4 mm, runtime and memory under budget, JSON validates against the schema, mask is a single connected component (warn otherwise), and no watershed region extends more than 15 mm along its path from its own ostium (a leak into a neighbouring vessel, e.g. SMA side branches in subject 2).
- Dev-set scores committed to the repo as a text file. Any change that drops F1 or raises ostium distance on any case is reviewed before merge.
- Failure gallery: every rejection and every scorer false positive or miss dumps an image crop. Reviewing that folder is how rules get tuned.
- Case atlas: run triage.py on all 25 cases on day one and keep the output as a table in the repo (atlas_04_20.json covers 17 of them). Flag for manual review any case with mask_components not 1, native spacing not 0.8 mm, lumen median HU under 200, largest_patch_over_40pct_aorta true, tortuosity above 1.1, or wall patch count above 15. Regression set, in priority order now that the references are coarse cases: the five labelled cases 19 to 23 (scored), then the unlabelled coarse cases 16, 17 (invariants only), then the allowed-to-fail pair 18 and 24, then fine-resolution robustness cases 1 (heaviest underfill), 6 (tortuous), 8 (uncropped field of view). Subject 21 is both labelled and the calcified case. Every change runs on all of these; invariants apply even where there is no reference. Do not tune anything to make 18 or 24 look good. Subject 25 (whole aorta with arch) runs like every other case but is deliberately excluded from the regression set: the task is abdominal, and nothing is designed or tuned for the arch. Every change runs on all of these; invariants apply even where there is no reference.
- Do not tune to noise: a change that helps one case and hurts another is left out.

### D8. Clinician display

Options: static matplotlib (A), interactive 3D only (B), unrolled clock map (C), per-case HTML report combining B and C with a branch table (D).

**Choice: D, with C as the centrepiece. A is generated by the same code path as the mandatory verification PNG.**

- The end user is a vascular surgeon planning EVAR. They think in clock position and height along the aorta, not scanner xyz. The clock map converts our output into their coordinates, which is the actual "useful for clinicians" content.
- Centreline method (chosen after subject 3): slice centroids fail on a tortuous aorta because an axial slice can cut it obliquely or twice. Use a geodesic path: take the two mask voxels furthest apart along the mask as endpoints, then find the minimum-cost path between them through the mask interior with cost inversely related to distance-from-boundary, so the path stays central (skimage.graph.route_through_array). Smooth the result. Fallback: skeletonize_3d, prune to the longest path. The centreline endpoints also define the end faces used by D2 and D6 rule 1, so frame.py's endpoint step is shared with the engine even though the rest of the frame is display-only.
- Clock convention: 12 = anterior, 3 = patient's left, 6 = posterior, 9 = patient's right. Angle measured in the plane perpendicular to the local centreline tangent using a rotation-minimizing frame so the clock does not twist along a curved aorta. Not optional: subjects 2 and 3 curve enough that a naive axial angle would be wrong by tens of degrees. Initialise the frame at the inferior end of the segment with 12 o'clock pointing to patient-anterior and transport it upward; on a whole-aorta case (subject 25) the clock stays internally consistent through the arch but stops matching the surgical convention there; that case is out of scope and nothing is adjusted for it.
- Height: distance along the centreline from the superior end of the supplied mask, in mm. Show total segment length.
- Inter-branch spacing along the centreline, shown as brackets on the map. This is the "room to land a stent" number.
- Branch table: ID, clock, height, radius, direction. Same data as the JSON in readable terms.
- Interactive 3D (Plotly, self-contained HTML): aorta mesh, spheres at ostia sized by radius, direction arrows. For orientation and the demo.
- No anatomical names. The spec says not to assign them, and a wrong guess in front of a surgeon is worse than none.
- Optional: colour branches by confidence (passed all rules cleanly vs passed via fallback).
- Report generation is behind a flag so the scored run does not pay for it.

### D9. Code structure and team split

Options: one script (A), modules with a fixed data contract (B), notebooks merged at the end (C).

**Choice: B.** The pipeline is sequential with clean handoffs. Once the contract is agreed, each stage is testable on fake inputs before the upstream stage exists. run.py falls out for free.

Layout:

```
branchseed/
  run.py            CLI entry, calls pipeline.run(), writes JSON
  pipeline.py       orchestrates stages, per-stage timing
  config.py         every numeric constant, one per line, with its physical justification (see section 3)
  io_utils.py       load NIfTI, crop/uncrop, index <-> mm (the ONLY caller of TransformIndexToPhysicalPoint)
  candidates.py     D1 + D2
  instances.py      D3
  ostium.py         D4
  tracing.py        D5
  filters.py        D6
  frame.py          centreline, rotation-minimizing frame, mm -> (height, clock)
  report.py         D8 HTML report, clock map, verification PNG
  scorer.py         D7 matching, metrics, invariants
  tests/            one fake-data test per module
  requirements.txt
  README.md
```

Data contract (stage outputs):

- io_utils: reads NIfTI by sniffing gzip magic (not by extension); returns the SimpleITK images. Owns the crop and the resample to 0.8 mm isotropic and exposes one function, index_to_mm(idx), that maps an index on the working (cropped, resampled) grid to physical mm via the working image's TransformIndexToPhysicalPoint. Nothing else in the codebase converts coordinates.
- candidates: working CT (float32, 0.8 mm iso), working mask (largest component), opened candidate array (bool, bright voxels in the 15 mm shell after the 0.8 mm opening), distance-from-mask array (mm), spacing (3 floats, all 0.8), the working SimpleITK image handle, plus the mask fragment log and the chosen threshold.
- instances: label array (0 background, 1..N branches), per-label wall-patch voxel indices.
- ostium: per label, ostium in original index space and in mm, outward wall normal.
- tracing: per label, path points (mm), seed (mm), direction (unit), radius (mm), path length, bifurcation flag.
- filters: surviving labels, rejection log of (label, rule, value).
- frame: centreline points (mm), per-point frame, function mm -> (height, clock).
- Everything downstream of candidates works in cropped index space and converts to mm only through io_utils.

Team (three people):

- Person 1, engine front half: candidates, instances, ostium. Critical path.
- Person 2, engine back half and ops: scorer first (no dependencies), then tracing and filters, then run.py, pipeline timing, requirements, README, invariant checks on all 25 cases.
- Person 3, frame and report: centreline, clock frame, HTML report, verification PNG, demo slides. Parallel from hour one using fake ostia. First deliverable is the centreline with its two endpoints, because D2 and D6 rule 1 need the endpoints to identify end faces. Until it exists, person 1 stubs end faces as the top and bottom z-slices.

Rules: commit to main behind a passing fake-data test; full integration with the dumbest version of every stage by end of day one; after that, replace stages, never restructure.

Robustness requirements for run.py (the reproducibility bucket is lost by a single crash on a hidden case):

- A case that yields zero branches after filtering still writes valid JSON with `"daughters": []`, and the visual check for that case renders the aorta alone without error.
- If a file cannot be loaded even with the gzip and header fallbacks, or any stage raises, run.py catches the exception, logs the traceback to stderr, writes valid JSON with an empty daughters list for that case, and exits 0. A wrong answer on one case costs that case; a crash can cost the run.
- Both paths are covered by tests: one synthetic case with no bright voxels outside the mask, one with a deliberately corrupt input file.
- JSON is written atomically: write to `<output>.tmp`, then `os.replace` onto the final name. A killed process never leaves a half-written file that looks valid.
- Offline install is tested, not assumed. `pip download -r requirements.txt -d vendor/` on a connected machine, commit `vendor/`, and the README's setup command is `pip install --no-index --find-links vendor -r requirements.txt`. Before submission, run that setup and one dev case inside a container or VM with networking disabled. The judges' machine has no internet, and a package that quietly fetches something on first import is a zero on the run.

### D10. Runtime budget

Target under 35 s per case, hard cap 60 s, peak memory under 4 GB.

- Load NIfTIs: 2 to 4 s.
- Crop, resample to 0.8 mm, distance transform, threshold: 1 to 5 s. Uncropped, the distance transform alone is 20 to 40 s and most of the memory (the triage process was killed on 512 × 512 volumes when this was skipped). Resampling the coarse 1.5 mm cases multiplies voxel count by about 6.6, but their crops are small; subject 18 (285 mm segment) is the worst at about 5 s.
- Frangi on the cropped block: 3 to 10 s. Three scales max (about 1, 2, 3 mm), computed only inside the shell's bounding box. First thing to drop if a case runs long.
- Components and watershed: under 2 s.
- Ostium, tracing, filtering: under 1 s per branch.
- Centreline and frame: 1 to 3 s.
- Report: 2 to 5 s, behind a flag.

Rules: crop before anything expensive; float32 everywhere; delete full-resolution arrays after cropping; log per-stage wall time on every run; no multiprocessing inside a case (parallelize across cases at batch level if wanted); test on the slowest dev case and assume the hidden set has one worse.

---

## 3. Overfitting controls

We have 25 dev cases, a handful with references, and a hidden test set. Everything in section 2 was tuned to the inputs (file formats, voxel sizes, enhancement levels, mask defects), not to any reference annotation, so nothing so far is fit to the answer key. The risk begins when the scorer runs. These rules apply from that point.

Where the risk lives: the spec carries about a dozen numeric constants (0.8 mm opening, 2 mm wall layer, 15 mm shell, 0.45 × median and 100 HU threshold floors, 4 mm ostium agreement, 70° tangency, 20° along-centreline, aspect ratio 3, 1 ml volume cap, 4 mm duplicate merge, 40% area ratio). Each is a place where a value could be nudged to fix one dev case and break two hidden ones.

Defences, strongest first:

1. **All constants live in one file, `config.py`, one per line, each with its justification as a comment.** No numeric threshold is defined anywhere else in the codebase; modules import from config. This makes the freeze a single file, lets a reviewer see every magic number on one screen, and the README's constants table is generated from it. Each constant gets a physical justification. 2 mm wall layer because lumbar arteries are 2 mm; 0.8 mm opening because it is one voxel and the ring is one to two voxels; 100 HU floor because unenhanced soft tissue tops out near 80; 5 mm and 10 mm because the PDF says so. A constant that can be explained without naming a subject is safe. A constant whose only explanation is "it worked on subject 7" is a suspect and gets flagged in code review.
2. **Sensitivity sweeps, not point tuning.** For each constant, run the scorer across a range and look at the curve. First sweeps, in order: D6 rule 3 departure distance (2 to 4 mm) and angle (60° to 80°), the opening radius (0.6 to 1.2 mm), the threshold multiplier (1.5 to 2.5 std), and the duplicate-merge distance (3 to 6 mm). Flat means the choice does not matter; pick the physically motivated value and stop. A sharp peak at one value is the dev set talking; treat the peak with suspicion, not as a finding, and prefer the physically motivated value even if it scores slightly lower.
3. **Leave-one-out on the labelled cases.** Tune on N − 1, score on the held-out case, rotate through all N. If held-out scores track in-sample scores, the rules generalize. If they drop, we have been fitting. Report both numbers in the demo.

Process rules:

- A parameter change must improve or hold every labelled dev case. A change that helps one and hurts another is fitting to noise and is not merged (already stated in D7; enforced here).
- Every rule in D6 maps to a bullet in the PDF's "important cases" list. A proposed rule that does not correspond to a phenomenon described in the spec needs a written argument for why the hidden set will contain that phenomenon.
- Subjects 18 and 24 (unenhanced) are allowed to fail. No rule is added or adjusted to rescue them.
- All constants freeze 24 hours before submission. After the freeze, `config.py` is not touched; bug fixes only elsewhere. The README's constants table is generated from `config.py` at freeze time.
- The failure gallery stays in the demo. Showing cases we get wrong, and why, is more credible than a clean-looking slide.
- Rules over learned models at this sample size (the D6 argument) is itself an overfitting control: the system encodes the definition of a branch (bright, tube-shaped, attached to the wall, pointing away from it, followable for 5 mm), which is the same definition the reference annotators applied, and that definition does not change on the hidden set.

---

## 4. Parallel implementation and what to evaluate against it

Two teammate documents arrived after this spec was written: a handoff describing an existing, tested implementation of essentially the same architecture (crop, wall-first search, path verification, wall-contact instances, measurement at 5 mm; 62 tests, phantom suite, evaluator, run on subject001), and a literature-based ideas ledger worked out on subject 25. Both live under `docs/`.

Decision: build this spec as written and run the two implementations side by side, with one shared scorer and a decision date at the end of day two, rather than merging code nobody has fully read. This is also the protocol the handoff itself proposes (compare stage by stage, integrate through explicit seams, decide by ablation).

Adopted now, without comparison, because they are bug avoidance with reasoning checkable in a minute rather than design choices:

- PCA direction with the sign fix, kept as a diagnostic (D5); the reported direction is the ostium-to-seed chord, which is the reference convention.
- Gzip sniffing regardless of extension (D2, already present).
- Atomic JSON writes (D9).
- Offline install test with a vendored wheel directory (D9).

To evaluate through the side-by-side comparison, each with the metric that decides it:

| Idea | Replaces or adds to | Deciding metric |
|---|---|---|
| Leak detector: reject a trace when radius at d exceeds 1.5 × radius at d − 3 mm | D6 rule 4 area-doubling test | False positives on subjects 8 and 12 (bowel), 4 (bone); recall unchanged on labelled cases |
| Contiguity-only merging of wall patches, no distance rule | D6 rule 6 (4 mm, 20° merge) | Recall on close renal pairs; duplicate count |
| FWHM radius on an interpolated cross-section at the seed | D5 area-equivalent radius | Deprioritised: only 3 of 19 reference branches have a radius value |
| Bifurcation by component split that persists for two further steps | D5 stop rule | Fraction of traces truncated before 5 mm on the regression set |
| Wall-contact brightness: median HU in the 1 to 2 mm outside the patch must exceed the threshold | Addition to D6 rule 3 | False positives on subjects with venous enhancement (3, 18, 22) |
| Degenerate calibration flag when std/mean inside the mask exceeds 0.25 | Addition to D1 | Fires on 16, 18, 24 and nowhere else |
| Geodesic level-set tracing at 0.5 mm in a per-candidate subvolume | D5 1 mm marching tracer | Seed-on-branch rate and direction error; runtime |
| Per-candidate subvolume upsampling instead of global resample of the crop | D2 resample step | Runtime and memory on subjects 8, 12, 18; recall on 16 to 23 |
| Hessian objectness as a candidate feature (in the existing implementation) | D1 threshold-only candidates | Runtime (the handoff reports 30 s on subject001; our candidate stage runs in 1 to 5 s without it) versus any recall gain |
| Output flags (near_cut_face, close_pair, low_confidence, degenerate_calibration) and rejected_by on every candidate | D8 display, D7 failure gallery | No metric; adopt if the display team has time |
| Circular padding and ray-validity checks on the unrolled map | D8 frame module | Visual: a branch at 12 o'clock renders as one blob; arch rows on subject 25 are not scrambled |

Explicitly not evaluated: Jerman, OOF and RORPO vessel filters, TEASAR skeletonization, external datasets, perturbation-based confidence, and any constant taken from the literature (the ideas ledger itself shows Riffaud's length rule would delete every eligible branch). Reason: no time, and the handoff's own protocol says not to keep a component merely because it is novel.

Known issues in the existing implementation to check for in ours: case_id derived from the filename instead of the folder; a hard cap of 24 candidates that saturated on subject001; a direction that disagreed with ostium-to-seed by 74° on one branch (consistent with the eigenvector sign problem above).

---

## 5. Timeline

Day 1 (today): environment, repo, data contract agreed, every stage stubbed and integrated end to end with the dumbest version. run.py produces valid JSON for case 19 by end of day. Scorer built against the five reference cases and run on that JSON, however bad the numbers. Person 3 has the centreline with endpoints, then the clock map on fake ostia. Person 2 has scorer running on the dev references. triage.py run on all 25 cases, atlas table committed, hard cases identified.

Day 2: replace stubs with real implementations in order of score weight: instances and ostium first, tracing second, filters third. Scorer run after every merge. Failure gallery reviewed together at end of day. End of day: run both implementations (this spec and the existing package) on the regression set and the labelled dev cases with the shared scorer, review both failure ledgers, and choose the base for day three per section 4.

Day 3: tuning against the dev set, invariant checks on all 25, runtime on the slowest case, report polish, README, demo run-through with a failure case.

---

## 6. Open questions for the organisers

1. Is the JSON schema still required for scoring, given "or any format you find fit"? (We are producing it regardless.)
2. ~~What is the minimum origin size for eligibility?~~ Answered by the reference package: 2.0 mm origin diameter, estimated perpendicular to the path at native resolution.
3. What ostium distance cutoff is used for matching predictions to references? (Working value: 5 mm.)
4. Are the hackathon judges scoring the PDF rubric, a separate creativity rubric, or both?
5. ~~Is the reference ostium annotated on the aortic wall surface, or at the lumen centre of the opening?~~ Answered: on the supplied parent-mask boundary.
6. ~~If a daughter's ostium sits within a few mm of a cropped end of the mask, is it included?~~ Partly answered: case 20's notes exclude posterior tracks near the superior crop as uncertain, and the terminal iliac division is excluded (case 23). Still worth asking how a confident branch at a cap would be treated.
7. ~~What fraction of the hidden set is coarse-resolution?~~ Effectively answered: all five references are 1.5 mm cases annotated at native resolution. Still confirm the hidden set is the same source.
11. Will the hidden references be adjudicated by an expert before scoring, or scored as drafts? And will excluded borderline candidates (1.5 to 2.5 mm) be ignored in scoring, or counted as false positives if detected?
8. Subjects 18 and 24 have lumens of about 110 and 78 HU (unenhanced). Are they in the evaluation set, and if so, how many eligible daughters do their references contain? (We expect to do poorly on them and want to know whether that is shared.)
9. Subject 25's mask covers the whole aorta including the arch and ascending segment. Are arch and coronary origins expected in its reference, or only abdominal branches? The PDF scope says abdominal.
10. Subject 24's header is non-orthonormal and SimpleITK cannot open it as supplied. Was the reference for that case generated in the tilted physical frame (as the header implies) or on an axis-aligned grid?

---

## 7. Environment

Python 3.10 or 3.11. `pip install SimpleITK numpy scipy scikit-image nibabel matplotlib plotly`. No torch. 3D Slicer for human inspection only (not a dependency). One setup command and one run command in the README. Input files may be gzipped regardless of extension; the reader handles both.