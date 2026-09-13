# Branchseed reference-annotation review record

Complete this record for each case before designating its masks as ground truth. Review the original CT, supplied aorta mask, daughter labels and case notes together. The current files and their qualitative confidence ratings are draft annotations.

Case ID: __________  Reviewer: __________  Review date: __________

Input/version or file hashes: __________  Edited-output version: __________

- [ ] Open the matching CT and segmentation in ITK-SNAP. Import the matching label descriptions. Check overlay alignment in axial, coronal and sagittal views.
- [ ] Read the case notes, excluded/uncertain candidates and validation report. Inspect all flagged locations and partial-volume or measurement failures.
- [ ] Confirm that dimensions, 1.5 mm spacing and physical geometry remain unchanged after any edits. Preserve integer instance labels.

For **every retained branch**, record its ID and one decision: **accept, edit, exclude or unresolved**.

| Branch ID | Decision | Origin diameter / method | Visible path length | Distal stopping point | Edits or unresolved issue |
|---|---|---|---|---|---|
| __________ | __________ | __________ | __________ | __________ | __________ |
| __________ | __________ | __________ | __________ | __________ | __________ |
| __________ | __________ | __________ | __________ | __________ | __________ |

- [ ] Establish continuous contrast-filled lumen from the supplied parent aorta into the daughter using consecutive slices and orthogonal views. A nearby bright structure alone does not establish an origin.
- [ ] Check that this is a direct aortic daughter. A downstream branch of another daughter must not receive an additional direct-origin instance. A common trunk has one direct origin; two separate ostia receive two instances.
- [ ] Locate the ostium at the supplied parent-lumen boundary. Check whether a mask imperfection shifts the proposed origin; document any correction and the boundary convention used.
- [ ] Estimate diameter at the origin in a plane perpendicular to the local path. Confirm the 2 mm eligibility threshold with an explicit method. Record uncertainty near the cutoff; do not treat the seed diameter or a search radius as the origin diameter.
- [ ] Follow the lumen for at least 5 mm beyond the origin, measuring along its path. Confirm that the proposed 5 mm seed lies within that daughter's lumen and inspect whether its voxel mask supports the landmark.
- [ ] Inspect the full proximal segment and identify the first downstream bifurcation. Stop the annotation at 10 mm or that bifurcation, whichever occurs first. If bifurcation occurs before 5 mm, record the ambiguity and the organizer's adopted eligibility/seed convention; do not silently choose a downstream daughter.
- [ ] Compare the segmentation with actual lumen width throughout the segment. Correct spill into parent aorta, veins, bone, calcification or surrounding tissue. Check holes, disconnected fragments and overlapping or duplicated daughter instances.
- [ ] Review tracing-guide placement, direction and mask coverage. A reported guide length alone does not verify the segmented lumen's extent.
- [ ] Review each radius/diameter measurement and its flags. Replace unresolved values only when supported by a recorded measurement; retain `null` when the quantity cannot be established. Do not infer measurement accuracy from decimal places.

After reviewing existing labels, **sweep the entire supplied parent segment for missed origins**.

- [ ] Inspect the full circumference across consecutive slices, including anterior, lateral and posterior surfaces. Revisit thin or oblique vessels that a coarse slice montage could miss.
- [ ] Check possible separate nearby ostia and common trunks. Search actual image coverage without assuming a fixed count or list of named vessels.
- [ ] Exclude flat superior/inferior crop caps as origins. Apply the agreed exclusion of the terminal iliac division. Do not invent an origin outside the visible or supplied coverage.
- [ ] Adjudicate every listed excluded/uncertain candidate. Add newly found candidates with coordinates, supporting views and a reason for inclusion, exclusion or uncertainty.
- [ ] Record any region where image quality or sampling prevents a confident completeness judgement. Specify how unresolved regions/branches will be handled in scoring before release.

Overlooked origins added: __________

Candidates excluded, with reasons: __________

Unresolved eligibility or boundary decisions: __________

- [ ] Save edited masks as a new reviewed version. Update branch IDs, label descriptions, annotations, notes and previews consistently.
- [ ] Reopen saved files and rerun geometry, label, connectivity, parent-contact, overlap and guide/seed-support checks. Reinspect changes in the image viewer.
- [ ] Confirm that the intended scoring interpretation matches the released masks and notes. Retain the earlier draft and an edit record for provenance.

Case disposition: **approved reference / further edits required / unresolved**

Approved branch IDs or explicit scoring exclusions: __________

Reviewer sign-off and date: __________

Dataset release owner and version: __________

## Sign-off register

No branch below has been approved. Replace pending only after the review described above.

| Case | Branch | Decision | Reviewer | Date |
|---|---|---|---|---|
| 19 | branch_001 | Pending | | |
| 19 | branch_002 | Pending | | |
| 19 | branch_003 | Pending | | |
| 20 | branch_001 | Pending | | |
| 20 | branch_002 | Pending | | |
| 20 | branch_003 | Pending | | |
| 20 | branch_004 | Pending | | |
| 21 | branch_001 | Pending | | |
| 21 | branch_002 | Pending | | |
| 21 | branch_003 | Pending | | |
| 22 | branch_001 | Pending | | |
| 22 | branch_002 | Pending | | |
| 22 | branch_003 | Pending | | |
| 22 | branch_004 | Pending | | |
| 22 | branch_005 | Pending | | |
| 22 | branch_006 | Pending | | |
| 23 | branch_001 | Pending | | |
| 23 | branch_002 | Pending | | |
| 23 | branch_003 | Pending | | |
