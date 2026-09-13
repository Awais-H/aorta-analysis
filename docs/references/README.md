# Branchseed: five draft daughter-vessel annotation sets

This package contains five CT-guided draft instance masks for cases 19–23, one daughter-label NIfTI volume per case. They are intended for viewing, editing and developing the hackathon's reference annotations. **Every case requires expert review before it is designated ground truth or used as an authoritative scoring reference.** The current annotations do not guarantee that every eligible origin has been found.

Each retained direct daughter has its own label, named `branch_001`, `branch_002`, and so on. IDs are local to a case and do not identify the same anatomical vessel across different cases. The number of retained daughters is recorded in that case's `annotations.json`.

The annotation policy uses the actual visible contrast-filled lumen. **2 mm is the minimum estimated diameter at the origin; it is not a fixed mask diameter.** A daughter must be followable for at least **5 mm along its path beyond the origin**. The seed landmark is placed 5 mm along the tracing guide. Proximal guides and masks extend toward 10 mm or the first downstream bifurcation, whichever comes first, according to visual review. The 5 mm requirement is a minimum eligibility distance, not a 5 mm radial shell around the aorta. The supplied parent-lumen mask boundary is the operational origin boundary; an outer aortic-wall segmentation was not supplied.

The five source grids have **1.5 × 1.5 × 1.5 mm voxels**. Masks retain the corresponding CT's native dimensions and physical geometry. A 2 mm lumen is only about 1.3 voxels wide, so borderline eligibility and precise lumen boundaries are uncertain. Potential origins that could not be established confidently are recorded in the review notes where identified and are not silently counted as confirmed branches. Those notes are also incomplete until expert review covers the whole supplied aortic segment.

For case number `#`, the case folder contains:

| File | Purpose |
|---|---|
| `orig#.nii.gz` | Source CT; load as the main image. |
| `aorta#.nii.gz` | Supplied parent-aorta mask. |
| `daughters#_draft.nii.gz` | Daughter instances only: 0 = background; 1…N = branches. |
| `aorta_and_daughters#_draft.nii.gz` | Combined viewing mask: 0 = background; 1 = parent aorta; 2…N+1 = branches. |
| `daughter_labels.txt` | Names and colours for the daughter-only mask. |
| `combined_labels.txt` | Names and colours for the combined mask. |
| `annotations.json` | Instance IDs, tracing guides, landmark estimates, measurements and review status. |
| `review_preview.png` | Selected orthogonal slices for orientation and preliminary inspection. |

To view a case in ITK-SNAP:

1. Select **File → Open Main Image…**, choose `orig#.nii.gz`, and complete the wizard. Select NIfTI if format detection needs assistance.
2. Select **Segmentation → Open Segmentation…**. Choose `daughters#_draft.nii.gz` for daughters alone, or `aorta_and_daughters#_draft.nii.gz` for the aorta and daughters together. Use the same case number as the CT.
3. Select **Segmentation → Import Label Descriptions…**. Import `daughter_labels.txt` for the daughter-only mask, or `combined_labels.txt` for the combined mask.
4. Adjust overlay opacity so the underlying lumen remains visible. Scroll through all three slice views. In the 3D pane, use **Update** to create a surface view when useful.

These steps follow the [official ITK-SNAP training handout](https://www.itksnap.org/pmwiki/uploads/Train/rsna_handout.pdf). The files were checked programmatically; this package does not claim a completed ITK-SNAP interface test. A preview image does not replace inspection through the full volume.

The masks were built from case-specific, visually reviewed tracing guides with CT-intensity constraints, exclusion of the parent mask, and connected-component selection. This is an annotation-construction workflow, not a general-purpose daughter detector. Confidence values are qualitative review judgements, not calibrated probabilities.

Fields named `centerline_*` contain approximate tracing guides. Their length describes the guide, which can differ from the exact extent of the voxel mask. Ostia and seeds inherit guide-placement and native-resolution uncertainty. Physical landmarks use **SimpleITK LPS millimetres**; voxel guide coordinates use zero-based x, y, z array indices. Do not directly compare LPS coordinates with RAS coordinates without the corresponding coordinate conversion.

Manual origin-diameter estimates are distinct from CT-threshold measurements. Threshold-derived cross-sectional diameters and radii are approximate area-equivalent measurements; they may be affected by partial volume, the parent boundary and the search envelope. A `null` measurement means the estimate was unresolved or limited by the search envelope, not zero. Measurement and partial-volume flags should be read alongside the numeric fields. Submillimetre numerical sampling does not add spatial resolution to these scans.

The source files named `mask#.nii` were already gzip-compressed NIfTI streams. Packaged `aorta#.nii.gz` files are byte-identical copies with a matching extension; the source files were not edited. Source CT copies are also byte-identical. Geometry was checked with NiBabel and SimpleITK, including dimensions, spacing, physical origin/direction, affine, and qform/sform. See `validation_report.json` for the recorded structural checks and their results. A structural pass does not establish anatomical correctness, exhaustive detection or expert approval. NiBabel explains the coordinate transforms stored in NIfTI in its [official NIfTI documentation](https://nipy.org/nibabel/nifti_images.html).

Use the accompanying reviewer checklist to adjudicate each branch and sweep for overlooked origins. Record edits and unresolved cases before releasing a scored reference version.

## Case inventory

| Case | Draft daughter labels | CT shape (x, y, z) |
|---|---:|---|
| 19 | 3 | 250 x 250 x 169 |
| 20 | 4 | 299 x 299 x 201 |
| 21 | 3 | 217 x 217 x 202 |
| 22 | 6 | 247 x 247 x 280 |
| 23 | 3 | 244 x 244 x 179 |

There are 19 provisional daughter instances across five cases. This is a draft inventory, not an adjudicated reference count. All saved tracing guides are 10 mm long after adjusting their starting point to the supplied aorta boundary.

## Archive choices

`branchseed_masks_only.zip` contains the new segmentations, parent masks, label colors, previews, metadata and review documents. Open it alongside your existing matching `orig19.nii.gz` through `orig23.nii.gz` CTs. `branchseed_complete_dataset.zip` additionally includes byte-identical copies of those five CTs.
