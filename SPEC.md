# Branchseed Challenge: Team Spec

Toralis Labs healthcare track, Battle of the Schools hackathon. Reference document is the updated Branchseed Challenge PDF (the version with "could be in JSON, or something creative" and the clinician-display bullet). Team of three. Written 12 Sep 2026, brought up to date with the built system on 13 Sep 2026 (see `docs/HANDOFF.md` for the current state and what remains).

Guiding principle, adopted 13 Sep after rule 3 rejected real branches: no loyalty to any method in this document. Every rule is judged against real vascular anatomy and the reference annotations, and a rule that contradicts what vessels actually do is changed with the evidence recorded next to it. Decisions below marked "as built" describe what the code does now.

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

Findings from the review by eye (13 Sep; gallery sheets from `gallery.py` for subjects 16, 17, 21, 22, plus the strip diagnostics on 1 to 3 and 20, 23):

- **Wall-hugging branches are daughters.** Two of the 19 reference daughters (20/b1, 23/b1) are 2.6 to 2.8 mm lumbars that leave at a shallow angle and run along the wall; their own 10 mm guides end 1.8 and 2.5 mm from the mask. The only elongated contact strip on subjects 1 to 3 is a 2.9 mm vessel at 1 o'clock, 125 mm down subject 2's segment, descending steeply, with a clear origin end: the inferior mesenteric artery. No vein in the dev set clears the arterial-phase threshold. The PDF's definition ("an artery whose lumen connects directly to the supplied parent aorta") has no angle condition. D6 rule 3 therefore became a flag.
- **A hugging strip's origin is the end still in contact with the lumen.** On both reference strips that end is tighter to the wall, wider, brighter, and has almost no vessel body beyond the wall layer; the other end is where the vessel lifts off. D4 uses the tightest-hugging end.
- **Bone is the dominant false positive on poorly enhanced cases.** Subject 22 (lumen 232 HU, threshold 105) lets cancellous bone clear the threshold, so eight of its seventeen surviving false positives sat on the vertebral body with the seed on the cortical arc; subject 16 the same for four. Every bone contact shows two signs and no real vessel shows either: the seed cross-section runs out of the 16 mm window, and it contains cortex over 1.5 x the lumen median. D6 rule 4 uses both together; the edge alone also fires on reference 19/b3, whose section merges with an adjacent vessel at lumen brightness.
- **Neither the raw-shell watershed basin nor the opened component is a per-branch volume.** Real renals carry 1 to 4 ml basins (the basin floods every connected bright voxel) and the opened set is one 5 to 17 ml sheet round most of the aorta. The branch's voxels within 10 mm of the ostium are under 0.8 ml on every labelled reference and separate cleanly from bone.
- **The remaining false positives are mostly plausible unannotated vessels.** On subjects 21 and 22 about ten 1 to 3 mm tubes sit at lumbar and IMA positions where the draft README admits its sweep was incomplete; one candidate per case sits at the inferior cut (the iliac division on 22, visible as two lumens in the seed panel). Organiser questions 1 and 3 (section 6) decide these.
- **Subject 16** shows a textbook celiac and SMA at 12 o'clock 23 mm apart; its renals are not candidates because the mask ends near the renal level. **Subject 17's** two "branches" 8 mm apart are one hugging vessel producing two contact patches (now merged by D6 rule 6).
- **Resolution-limit misses, recorded and not chased:** 19/b2 (2.3 mm, one opened voxel: the opening erases it inside the wall layer and the march cannot start; dropping the voxel floor recovers nothing, an "outward support" wall layer was prototyped and rejected) and 23/b2 (merged with b3 at the wall, no neck at any prominence).

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
- As built: `instances.build` labels the wall layer (opened voxels within 2 mm), orders labels by patch area, and runs `skimage.segmentation.watershed` on the distance field through `bright_shell`. `instances.branch_voxels` is the voxel set D4, D5 and D6 operate on: inside the wall layer only opened voxels (the partial-volume ring is a one-to-two-voxel sheet hugging the mask, and the opened set is by definition what survived ring removal there), beyond it the raw thresholded voxels. Per label it also records the wall-patch area (atlas convention) and the whole-basin volume (information only; see D6 rule 4).
- A branch that hugs the wall before leaving gives a long thin wall component (aspect ratio over 3). D4 places its ostium at the contact end (strip_end); D6 rule 6 merges the several contact patches one hugging vessel can leave along its length.
- Lobe splitting of merged patches (h-maxima of the lateral distance field) was measured and rejected: at any prominence it fragments five real renal patches on subject 22 before it separates anything useful. The one merged reference pair (23/b2 and b3, 4 mm apart, no neck) is a resolution limit.

### D4. Ostium localisation

Options: centroid of wall patch (A), centroid snapped to surface (B), branch-axis intersection with aorta surface (C), centre of largest inscribed circle in the wall patch (D).

**Choice as built: D, snapped to the raw mask boundary; C computed and logged but not reported; a contact-end rule for hugging strips.**

- D: the lateral distance transform of the wall patch (edge = every other wall-layer voxel, so the 2 mm slab faces do not cap it), take the voxel furthest from the edge. An elongated patch has a ridge of equally deep voxels; the ridge voxel nearest the ridge centroid is used so the estimate sits mid-patch. A is dropped because D replaces it.
- C: fit the branch axis (PCA on the branch voxels within the first 10 mm, at least AXIS_FIT_MIN_VOXELS), walk it back to the aorta in AXIS_WALK_STEP_MM steps, take the surface crossing. If within OSTIUM_AGREEMENT_MM of D it is trusted. **Measured on all five labelled cases it lost to D on 6 of the 8 branches where it passed the gate, by 1 to 2 mm, and every case's mean ostium distance improved with D alone**: at 1.5 mm native resolution a few millimetres of blurry lumen give an axis too noisy to walk back. It is off by default (`config.OSTIUM_USE_AXIS_REFINEMENT`), still computed and logged, and the fitted axis is still handed to D5 as the tracer's initial direction when it passes the gate.
- Hugging branch (patch aspect ratio over PATCH_ASPECT_RATIO_MAX): the origin is the STRIP_END_WINDOW_MM end window with the smaller mean distance from the mask; the ostium is D restricted to that window and the tracer starts along the strip axis away from it (`ostium.strip_end`). Evidence in section 1. Moved reference 20/b1 from 6.8 mm to 2.3 mm and 23/b1 from 6.7 mm to 0.4 mm, and recovered subject 20's shallow-angle branch 4.
- Final step: snap the ostium to the nearest raw-mask boundary voxel. The reference package confirms this is the annotators' convention ("the supplied parent-lumen mask boundary is the operational origin boundary"); scoring is straight-line distance.
- Mean matched ostium distance on the labelled cases: 1.35 mm (best possible with any snap: 0.2 to 0.9 mm, the nearest boundary voxel to each reference).

### D5. Proximal tracing (seed, direction, radius)

Options: straight line from the axis fit (A), skeletonize the watershed region (B), slice-and-centroid marching (C), fast marching from the ostium (D).

**Choice: C, with A as fallback.**

- C directly implements the spec sentence "trace up to 10 mm or until the first bifurcation, whichever occurs first." As built (`tracing.march`): start at the ostium with the initial direction from D4, step TRACE_STEP_MM, take the branch's own voxels (`instances.branch_voxels`, so neighbouring branches are excluded) in a slab TRACE_SLAB_HALF_MM thick and CROSS_SECTION_HALF_WIDTH_MM wide perpendicular to the current direction, label its 26-connected blobs, follow the blob containing the voxel nearest the probe point, update the direction toward its centroid, repeat. Stop at TRACE_MAX_MM, when the section vanishes (under MIN_CROSS_SECTION_VOXELS), when a second blob of that size appears **once the slab is clear of the wall layer** (inside it the section still holds fragments of the origin, which produced false bifurcations at step one), on a turn over TRACE_MAX_TURN_DEG, or at the volume edge. Per-step section areas are kept for D6.
- All three outputs from one loop: seed = point at 5 mm path length; direction = the chord from ostium to seed, normalised. This replaces the earlier PCA tangent: in all 19 reference branches on cases 19 to 23, `direction_xyz` equals `normalise(seed − ostium)` to the degree, so the chord is the answer key's convention and a path tangent would score worse on curved branches. (PCA with the sign fix is kept in the code as a diagnostic only; a large angle between PCA axis and chord flags a path that doubled back.)
- Radius = area-equivalent radius sqrt(area / pi) of the thresholded cross-section on a plane perpendicular to the chord at the seed, resampled at 0.25 mm, which is the reference package's own method. Radius is deliberately low effort: 16 of the 19 reference branches have `radius_mm: null` ("search envelope limited"), so radius can only be scored on three branches in the dev references and probably few in the hidden set. Report it, clamp to [0.5, 8] mm, and do not tune it.
- The bifurcation stop matters: a common trunk splitting at 7 mm must report the trunk's direction, not an average of its children. B and D would average or pick a child.
- B rejected: skeletons are noisy exactly at the junction, where precision matters most. D rejected: needs extra machinery for radius and bifurcation with no accuracy gain.
- A fallback: if the march yields under MIN_TRACE_MM, use the straight axis line **for as long as the line itself stays within one working voxel of branch voxels**. If even that gives under 5 mm, the branch fails eligibility and is dropped. The first version counted any branch voxel within 8 mm of the line as reach; that manufactured 22 of 38 surviving false positives (10 mm of "path" through nothing) and was fixed after the filters ledger exposed it.
- Radius as built (`tracing.radius_at`): the plane at the seed is sampled at RADIUS_PLANE_SPACING_MM over a CROSS_SECTION_HALF_WIDTH_MM window, thresholded at the case threshold with mask voxels excluded, the component containing the centre (or the nearest within one voxel) taken. It also reports whether that component reaches the window edge and what fraction of it exceeds BONE_HU_LUMEN_RATIO x the lumen median; both feed D6 rule 4.
- Radius sanity: if the area-equivalent radius exceeds half the local aortic radius, the cross-section grabbed adjacent tissue; report the largest inscribed circle instead and flag it.
- Step in mm and resample cross-sections with real spacing. Anisotropic voxels bias area otherwise.

### D6. False positive filtering

Options: hard rules only (A), tiny trained classifier (B), rules for structural classes plus classifier for the residual (C).

**Choice: A, with a fixed threshold. B held in reserve only if the dev set turns out large enough.**

- With 3 to 5 labelled cases, a learned boundary overfits and we cannot detect it until the hidden set. Rules encode the definition of a branch, not the dev set's quirks, so they generalize.
- Every rule maps to a bullet in the spec's "important cases" list, which is the demo story.
- Default on borderline: keep if the branch passes the 5 mm trace, reject only on a clear rule hit. Precision and recall are weighted equally, and a candidate that survives 5 mm of tracing is more likely real than fake.
- As built (`filters.apply`): every rule is evaluated for every branch, all hits are logged as (label, rule, value), soft signals become flags, every measurement goes into a per-branch ledger (`meta["measurements"]`), and a branch survives only with no hit. The status of each rule below is what the code does on 13 Sep.

Rules, one per false-positive class:

1. Cropped ends / aorta continuing past the mask. **Rejecting, as built.** Reject if the candidate's wall patch touches an end face (its highest voxel within END_FACE_MM of the plane through the centreline endpoint with the outward tangent as normal; endpoints from frame.py, still the slice-centroid stub) AND its direction is within 20° of the outward tangent. Patch area over 40% of the aortic cross-section is a supporting signal only, not a discriminator: renal origin patches measure 150 to 240 mm² on this data, the same as the aortic cross-section. The continuation patches on subjects 19 and 20 are 440 to 460 mm² (about twice the cross-section) so a ratio above 1.5 can short-circuit the direction test. No distance-based exclusion zone (subject 3 has a real branch at the cut). This rule is exercised by subjects 17, 19 and 20, where the mask ends mid-vessel with the aorta continuing inside the image, and by subject 20 whose mask tapers to a 5 mm point at one end rather than a flat cut. It removes the continuation patch on every coarse case. Open: one further candidate per case touches the inferior face at an angle (the iliac division on 22, visible as two lumens 5 mm out; 7 to 8.5 mm vessels on 21 and 23; a 2.5 mm one on 20). They carry the `near_cut_face` flag and are kept until organiser question 1 is answered.
2. Calcified plaque. **Rejecting, as built** (`min_trace_mm`). Very bright, touches the wall, not a tube. Reject if traced path length is under 5 mm (this is also the spec's eligibility rule). Secondary: mean Frangi response below threshold. Frangi is not computed; FRANGI_MIN_RESPONSE = 0 disables it.
3. Adjacent parallel or crossing vessels. **Flag only (`no_departure`, `tangential`, `elongated_patch`), changed 13 Sep.** The original test rejected a branch whose traced path was still within 3 mm of the aorta at its far end, on the premise that a real branch has moved away by 10 mm and a vein or the SMA lying on the wall never leaves. The premise is false: a branch that leaves the aorta and then runs along it is still a branch (the PDF's definition has no angle condition), two of the 19 reference daughters do exactly that, and the only elongated contact strip on the cases the rule was written for is subject 2's IMA (section 1). Veins in the arterial phase do not clear the threshold, so the rule had no demonstrated benefit anywhere in the dev set and cost three real daughters. The biologically meaningful question for a contact strip is whether it has an origin end (one end tighter, wider, brighter, with no body beyond the wall) rather than two alike ends as a vein touching in passing would; that asymmetry is logged per branch (`strip_hug_ratio`) so the rule can be re-armed if a negative example ever appears. Sweep: at every departure distance from 0 to 5 mm F1 stays 0.50 to 0.55 on the labelled cases.
4. Bowel, organ tissue and bone. **Three tests as built.** (a) `proximal_volume_ml`, rejecting: the branch's voxels within TRACE_MAX_MM of the ostium exceed REGION_VOLUME_CAP_ML. The whole watershed basin was tried first and rejected real renals at 1 to 4 ml (it floods every connected bright voxel in the shell); the opened component too (one 5 to 17 ml sheet round the aorta). Within 10 mm every labelled reference is under 0.8 ml. (b) `bone`, rejecting: the seed cross-section reaches the edge of the 16 mm window AND contains cortex (voxels over BONE_HU_LUMEN_RATIO x the lumen median). Both signs on every vertebral contact reviewed on subjects 16 and 22, neither on any real vessel; the edge alone would also reject reference 19/b3, whose section merges with a neighbouring vessel at lumen brightness, so edge-without-cortex is the `section_merged` flag. (c) `area_growth`, flag only: section at 4 to 5 mm over the first step clear of the wall layer exceeding AREA_GROWTH_MAX. Its only hits on the labelled cases are real 4.5 mm branches whose first 3 mm read narrow at 1.5 mm voxels, and no labelled false positive grows; it becomes a rejection once the bowel cases (subjects 8, 12) have been reviewed on the gallery sheets. Frangi is not computed.
5. Branches of branches. Impossible by construction: instances require wall contact with the parent. Stated invariant, no rule.
6. Duplicates. **Rejecting, as built, widened 13 Sep.** Two survivors are one branch if their ostia are within DUPLICATE_MM with directions within DUPLICATE_ANGLE_DEG, or on the same working voxel whatever their directions (one origin is one instance; a trunk that splits still has one ostium), or their traced 10 mm paths come within DUPLICATE_MM of each other (two daughters cannot share a lumen; a wall-hugging vessel leaves several contact patches along its length, e.g. subject 17). The larger wall patch wins. Genuinely separate nearby origins survive because their paths diverge.
7. Noise and sub-threshold vessels. **Rejecting, as built.** The organisers' minimum origin diameter is 2.0 mm (the reference package's policy; the PDF says the number comes with the final dataset), estimated in a plane perpendicular to the path at the origin. The code measures it area-equivalent on the first plane clear of the wall layer (WALL_LAYER_MM along the path: closer planes cut the partial-volume ring and read a 2.5 mm branch as 4 mm) and rejects below MIN_ORIGIN_DIAMETER_MM; the BORDERLINE_ORIGIN_DIAMETER_MM band sets the `borderline_diameter` flag. Both excluded-candidate files in the reference package are vessels the annotators saw at 1.5 to 2.5 mm and withheld (organiser question 2 asks how a detection there is scored). Wall patches under MIN_WALL_PATCH_VOXELS are dropped as noise; the sweep shows a 1-voxel floor adds 15 false positives and recovers nothing, and the one reference lost here (19/b2) is a resolution limit, not the floor (section 1).

Every rejection is logged as (label, rule name, value). That log is the tuning tool and the "known failure cases" slide.

### D7. Validation strategy

Options: eyeball (A), reimplement the scorer (B), scorer plus invariant checks on unlabelled cases (C).

**Choice: C.**

- Scorer, as built (`scorer.py`): one-to-one matching via Hungarian assignment on ostium distance, **cutoff-aware** (pairs beyond the cutoff cost more than any number of in-cutoff pairs, otherwise a near prediction can be paired with a far reference and then lost to the cutoff; this happened on subject 23). Report at 3, 5 and 8 mm; 5 mm is the working value. Reads `docs/references/case_NN/annotations.json` directly; seed-on-branch = the predicted seed voxel carries the matched label in `data/subjectNNN/daughtersNN_draft.nii.gz`; direction error against the reference `direction_xyz` (the ostium-to-seed chord); radius error only where the reference radius is not null (3 of 19). Metrics per case and aggregated: precision, recall, F1, mean matched ostium distance, direction angle error, radius absolute error, seed-inside-branch.
- Tooling, all committed under `results/` after every pipeline change: `python scorer.py predict|score|invariants|ledger`, `python sweeps.py` (one constant at a time, the curve not the peak), `python gallery.py` (the failure gallery: per-candidate CT crops, six per sheet, `--rejected` for the rejected patches), `python review_markers.py` (ITK-SNAP marker volumes, label file, per-branch table with voxel indices). Reviewing a case means reading the gallery sheets and asking one question per candidate: is there a contrast-filled tube leaving the lumen that can be followed 5 mm.
- Scores on the labelled cases at 5 mm, 13 Sep: TP 17 / FP 16 / FN 2, precision 0.52, recall 0.89, F1 0.65, mean ostium 1.35 mm, direction 26 deg, seed on branch 14/17, radius error 0.29 mm (n=3). History: skeleton 0.22, ostium+tracing 0.25, filters 0.47, hugging-branch ostium 0.52, rule 3 as flag 0.55, review-driven bone and shared-lumen rules 0.65.
- Invariants on all 25 cases: every seed in a bright voxel, every direction points away from the aorta, radius between 1 and 8 mm, no two ostia within 4 mm, runtime and memory under budget, JSON validates against the schema, mask is a single connected component (warn otherwise), and no watershed region extends more than 15 mm along its path from its own ostium (a leak into a neighbouring vessel, e.g. SMA side branches in subject 2).
- Dev-set scores committed to the repo as a text file. Any change that drops F1 or raises ostium distance on any case is reviewed before merge.
- Failure gallery: `gallery.py` renders every kept candidate (and with `--rejected` every rejected patch) as CT crops; `results/reference_ledger.txt` records every reference branch's fate and every surviving false positive with all measurements. Reviewing those is how rules get changed; three rules changed that way on 13 Sep.
- Case atlas: run triage.py on all 25 cases on day one and keep the output as a table in the repo (atlas_04_20.json covers 17 of them). Flag for manual review any case with mask_components not 1, native spacing not 0.8 mm, lumen median HU under 200, largest_patch_over_40pct_aorta true, tortuosity above 1.1, or wall patch count above 15. Regression set, in priority order now that the references are coarse cases: the five labelled cases 19 to 23 (scored), then the unlabelled coarse cases 16, 17 (invariants only), then the allowed-to-fail pair 18 and 24, then fine-resolution robustness cases 1 (heaviest underfill), 6 (tortuous), 8 (uncropped field of view). Subject 21 is both labelled and the calcified case. Every change runs on all of these; invariants apply even where there is no reference. Do not tune anything to make 18 or 24 look good. Subject 25 (whole aorta with arch) runs like every other case but is deliberately excluded from the regression set: the task is abdominal, and nothing is designed or tuned for the arch. Every change runs on all of these; invariants apply even where there is no reference.
- Do not tune to noise: a change that helps one case and hurts another is left out.

### D8. Clinician display

Options: static matplotlib (A), interactive 3D only (B), unrolled clock map (C), per-case HTML report combining B and C with a branch table (D).

**Choice: D, with C as the centrepiece. A is generated by the same code path as the mandatory verification PNG.**

Status 13 Sep: frame.py is the slice-centroid stub (endpoints, arc length, axial clock; adequate for the straight coarse cases and for rule 1's end faces) and report.py writes the verification PNG (coronal and sagittal projections with ostia and arrows) plus an HTML branch table with clock and height. The geodesic centreline, rotation-minimising frame, clock map with spacing brackets and the Plotly 3D view are not built. Flags to show per branch: `near_cut_face`, `borderline_diameter`, `no_departure`, `tangential`, `elongated_patch`, `section_merged`, `area_growth`, `axis_fallback`, `bifurcation`, radius fallbacks.

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
  scorer.py         D7 matching, metrics, invariants, reference ledger, batch prediction
  sweeps.py         section 3 sensitivity sweeps
  gallery.py        D7 failure gallery (CT crop sheets per candidate)
  review_markers.py ITK-SNAP marker volumes and per-branch review tables
  tests/            one fake-data test per module (87 tests) plus real-data tests on subjects 19 and 24
  results/          dev_scores.txt, dev_scores_skeleton_baseline.txt, reference_ledger.txt, sweeps.txt, invariants.txt
  docs/HANDOFF.md   current state, decisions, what remains
  requirements.txt
  README.md
```

The modules live flat at the repository root (no `branchseed/` package): CLAUDE.md's smoke test runs `python run.py` from the root.

Data contract (stage outputs):

- io_utils: reads NIfTI by sniffing gzip magic (not by extension); returns the SimpleITK images. Owns the crop and the resample to 0.8 mm isotropic and exposes one function, index_to_mm(idx), that maps an index on the working (cropped, resampled) grid to physical mm via the working image's TransformIndexToPhysicalPoint. Nothing else in the codebase converts coordinates.
- candidates: working CT (float32, 0.8 mm iso), working mask (largest component), `bright_shell` (bool, raw thresholded voxels in the 15 mm shell), `opened` (the same after the 0.8 mm opening; defines the wall layer only), distance-from-mask array (mm), spacing (3 floats, all 0.8), the working SimpleITK image handle, plus the mask fragment log, HU statistics and the chosen threshold.
- instances: label array (watershed through `bright_shell`, 0 background, 1..N branches, ordered by wall-patch area), wall-label array, per-label wall-patch voxel indices, wall-patch area (mm2), basin volume (ml); `branch_voxels()` gives the per-label voxel set used downstream.
- ostium: per label, ostium as a mask-boundary voxel on the working grid and in mm, outward wall normal, the tracer's initial axis, method (`inscribed_circle`, `axis_intersection`, `strip_end`), the D and C estimates, the agreement distance, and for strips the two end-window hug distances.
- tracing: per label, path points (mm), seed (mm), direction (unit chord), radius (mm), path length, bifurcation flag, method (`march`, `axis`, `none`), stop reason, march length, per-step section areas, radius fallback flag, PCA-to-chord angle, and whether the seed section fills the window and its cortex fraction.
- filters: surviving labels, rejection log of (label, rule, value), per-label flags, per-label measurements.
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
- Offline install is tested, not assumed. `pip download -r requirements.txt -d vendor/` on a connected machine, commit `vendor/`, and the README's setup command is `pip install --no-index --find-links vendor -r requirements.txt`. Before submission, run that setup and one dev case inside a container or VM with networking disabled. The judges' machine has no internet, and a package that quietly fetches something on first import is a zero on the run. **Not done: wheels are platform-specific and the scoring machine's OS and Python version are organiser question 6.** Atomic JSON writes are done.

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
- Anatomy over method: a rule is judged first against what real vessels do and what the annotators marked, then against the numbers. Three rules changed that way on 13 Sep (rule 3 to a flag, rule 4 to the proximal volume and the cortex test, rule 6 widened to shared lumens), each with a written argument in D6 and the evidence in section 1.
- Switched off with the measurement recorded next to the switch, to be re-armed only on new evidence: the D4 axis refinement (`OSTIUM_USE_AXIS_REFINEMENT`), rule 3 departure, rule 4 area growth. The sweeps in `results/sweeps.txt` show where a constant would score better on the dev set and the physical value was kept anyway (MIN_ORIGIN_DIAMETER 2.5 and REGION_VOLUME_CAP 0.75 each drop 5 to 6 false positives at no labelled cost).
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

Status at the end of 13 Sep: days 1 and 2 are done (engine complete through D6, reviewed by eye, scored, invariants clean on all 25 cases, runtime under 9 s per case). Remaining, in the order agreed: (1) frame.py geodesic centreline and rotation-minimising clock; (2) report.py clock map, flags in the table, Plotly 3D, verification PNGs for three cases committed; (3) submission mechanics: dev-set predictions committed, vendored wheels once the platform is known, offline install test, README commands; (4) rule 4 area growth decision after a look at subjects 8 and 12 on the gallery sheets; (5) demo material from the ledger and gallery. Organiser answers (section 6) may reopen rule 1 at the inferior cut and the borderline-diameter handling. See `docs/HANDOFF.md`.

---

## 6. Open questions for the organisers

Answered by the PDF itself (checked 13 Sep): a direct daughter is "an artery whose lumen connects directly to the supplied parent aorta" with no angle condition, so wall-hugging branches count; "the flat superior and inferior ends created by cropping are not branch origins"; the terminal iliac division is outside the core task; the minimum origin size "will be specified with the final dataset"; matching is one-to-one and duplicates are false positives; the machine is four cores, 8 GB, no GPU, no internet, 60 s per case.

Answered by the reference package: origin diameter policy 2.0 mm measured perpendicular to the path at native resolution; the reference ostium is on the supplied mask boundary; the references are drafts pending expert review and their sweep for small posterior origins is incomplete; all five are 1.5 mm cases.

Sent to the organisers on 13 Sep (awaiting answers):

1. When the aorta mask ends near the iliac bifurcation, we sometimes find a small vessel starting a couple of millimetres above the cut. Do those get annotated, or is anything that close to the cut face left out? (Decides the `near_cut_face` candidates, one per coarse case.)
2. The draft annotations use a 2 mm minimum origin size and left out a couple of vessels around 2.1 mm as borderline. If we detect one of those borderline ones, does it count against us, get ignored, or count as a hit? And is the size measured the same way as in the drafts, at native resolution? (Decides the `borderline_diameter` handling.)
3. The draft README says the sweep for small posterior branches wasn't complete. Will the final references be checked for completeness before scoring? On the labelled cases we find about ten small posterior and anterior-left vessels that aren't in the drafts. (Decides whether the remaining false positives are ours or theirs.)
4. How close does a predicted ostium need to be to the reference one to count as a match? We've been assuming 5 mm. And is the reference ostium placed on the supplied mask boundary, like in the drafts?
5. Subjects 18 and 24 look unenhanced, and subject 25 is the whole aorta including the arch. Are those three in the evaluation set, and if 25 is, are arch branches expected?
6. What OS and Python version will the scoring machine run? We want to bundle the right wheels for the offline install.

Not sent, low priority: whether the JSON schema is required given "or any format you find fit" (we produce it regardless); whether a creativity rubric is scored alongside the PDF rubric; whether subject 24's reference was made in its tilted physical frame (our reader keeps the tilt, so either answer works within a voxel).

---

## 7. Environment

Python 3.10 or newer (developed and tested on 3.13.7 with SimpleITK 2.5, numpy 2.2, scipy 1.18, scikit-image 0.26). `pip install -r requirements.txt`. No torch. 3D Slicer for human inspection only (not a dependency). One setup command and one run command in the README. Input files may be gzipped regardless of extension; the reader handles both.