"""Every numeric constant in the pipeline, one per line, each with its physical justification.

SPEC.md section 3: no numeric threshold is defined anywhere else in the codebase; modules import
from here. Constants freeze 24 h before submission. A constant that can be explained without
naming a subject is safe; one whose only explanation is "it worked on subject 7" is a suspect.
Do not change a constant to improve the score on one case (CLAUDE.md, SPEC.md section 3).
"""

# ---------------------------------------------------------------- grid, crop, resample (D2, D10)
ISO_SPACING_MM = 0.8        # working grid: the finer of the two dev acquisitions (0.8 mm z); every mm parameter below becomes a clean voxel count here
RESAMPLE_TOLERANCE_MM = 0.05  # skip the resample when native spacing is already within this of ISO on every axis (avoids a no-op interpolation blur)
CROP_PAD_MM = 15.0          # pad the mask bounding box by the shell radius so the whole search shell is inside the crop
CT_BACKGROUND_HU = -1000.0  # fill value outside the volume when resampling: air, never mistaken for enhanced blood
MASK_FRAGMENT_WARN_FRACTION = 0.01  # discarded mask voxels above 1% of the mask means a mislabelled blob (subject 18: 9%), not jagged-edge crumbs

# ------------------------------------------------------------------- candidate threshold (D1)
THRESHOLD_STD_MULTIPLIER = 2.0  # threshold = mean - 2 std of HU inside the mask: 1.5 std missed 250 to 300 HU lumbar-sized vessels
THRESHOLD_MEDIAN_FRACTION = 0.45  # relative floor 0.45 x median adapts to the enhancement level when std is huge on coarse, poorly enhanced scans
THRESHOLD_FLOOR_HU = 100.0  # absolute floor: unenhanced soft tissue tops out near 80 HU, so 100 keeps it out on any scan

# ------------------------------------------------------------- search shell and wall layer (D2, D3)
SHELL_MM = 15.0             # search shell around the mask: the spec traces 10 mm, plus margin to see whether a common trunk splits before or after that
WALL_LAYER_MM = 2.0         # candidate voxels within 2 mm of the mask surface define instances; lumbar arteries are about 2 mm across
OPENING_RADIUS_MM = 0.8     # binary opening ball: one working voxel; the partial-volume ring outside the mask is one to two voxels thick, branch roots are 3 mm or wider. The opening defines the wall layer ONLY (D2); growth (D3 watershed) and tracing (D5) use the raw thresholded shell so a thin branch is not eroded once its origin is established
OPENING_MIN_RADIUS_VOX = 1  # the opening ball is never smaller than one voxel, otherwise it is a no-op

# ------------------------------------------------------------------------ ostium (D4)
AXIS_FIT_MM = 10.0          # PCA for the branch axis uses watershed voxels within the first 10 mm, the spec's tracing horizon
AXIS_FIT_MIN_VOXELS = 5     # PCA on fewer voxels than this has no meaningful principal axis (a 2 mm lumen is about 5 voxels per 0.8 mm slice); fall back to the wall normal
AXIS_WALK_STEP_MM = 0.25    # the axis is walked back toward the aorta in quarter-voxel steps so the surface crossing is located to sub-voxel precision before the snap
OSTIUM_AGREEMENT_MM = 4.0   # axis-intersection ostium is used only if within 4 mm of the inscribed-circle ostium, so a bad axis fit never beats the baseline
OSTIUM_USE_AXIS_REFINEMENT = False  # D4 option C is computed and logged but not reported. Measured on all five labelled cases (docs/references): where C passed the 4 mm gate it was worse than D on 6 of 8 branches, by 1 to 2 mm, and every case's mean ostium distance improved with D alone (D7: a change must improve or hold every labelled case). At 1.5 mm native resolution a few millimetres of blurry lumen give an axis too noisy to walk back to the wall. Flip to True to re-evaluate on finer data

# ------------------------------------------------------------------------ tracing (D5)
SEED_DISTANCE_MM = 5.0      # PDF definition: the daughter seed is 5 mm outward from the ostium along the daughter path
TRACE_MAX_MM = 10.0         # PDF: trace up to 10 mm beyond the ostium or until the first bifurcation
TRACE_STEP_MM = 1.0         # march step: fine enough to follow a looping renal, coarse enough that one step spans more than one working voxel
MIN_TRACE_MM = 5.0          # PDF eligibility: a branch must be followable for at least 5 mm beyond the wall
TRACE_SLAB_HALF_MM = 0.5    # the cross-section perpendicular to the current direction is a slab one working voxel thick (plus and minus half a voxel)
CROSS_SECTION_HALF_WIDTH_MM = 8.0  # lateral window of a cross-section: the largest daughter radius accepted (INVARIANT_RADIUS_MAX_MM), so a real lumen is never truncated
MIN_CROSS_SECTION_VOXELS = 3  # a cross-section blob under 3 voxels is noise: a 2 mm lumen gives about 5 voxels per 0.8 mm slab. Used for 'cross-section vanishes' and for the second blob of a bifurcation
TRACE_MAX_TURN_DEG = 90.0   # a direction change over 90 deg in one 1 mm step is a flip (the march ran back into the aorta or jumped to a neighbour), not anatomy: stop and fall back to the axis
RADIUS_PLANE_SPACING_MM = 0.25  # D5: radius = sqrt(area / pi) of the thresholded cross-section on a plane perpendicular to the chord at the seed, resampled at 0.25 mm; the reference package's own method, so radius errors are comparable. Interpolation adds no resolution
RADIUS_CLAMP_MM = (0.5, 8.0)  # D5: reported radius is clamped to this range; below 0.5 mm nothing is resolvable, above 8 mm nothing but the iliacs leaves the abdominal aorta. Radius is low effort (16 of 19 reference radii are null) and is not tuned
RADIUS_AORTA_FRACTION_MAX = 0.5  # D5 sanity: a daughter radius over half the local aortic radius means the cross-section grabbed adjacent tissue; report the inscribed circle instead and flag it

# ------------------------------------------------------------------------ filters (D6)
END_FACE_MM = 3.0           # rule 1: a wall patch within 3 mm of a centreline endpoint touches an end face (end faces are not axial planes, so this is a distance, not a slice index)
END_FACE_ANGLE_DEG = 20.0   # rule 1: direction within 20 deg of the local centreline tangent means the aorta continuing, not a branch
END_FACE_AREA_FRACTION = 0.40  # rule 1 supporting signal: patch area over 40% of the aortic cross-section (renal patches reach this too, so support only)
END_FACE_AREA_SHORTCUT = 1.5  # rule 1: a patch over 1.5 x the aortic cross-section is a continuation regardless of direction (subjects 19, 20: about 2 x)
FRANGI_SCALES_MM = (1.0, 2.0, 3.0)  # rule 2 secondary / rule 4: three vesselness scales spanning lumbar to renal radii; three scales max for runtime
FRANGI_MIN_RESPONSE = 0.0   # rule 2 secondary: mean Frangi response floor. PLACEHOLDER, not yet set by the spec; 0.0 disables the rule until a sweep sets it
DEPARTURE_MM = 3.0          # rule 3, FLAG ONLY (no_departure): path far end still within 3 mm of the aorta. Not a rejection: wall-hugging lumbars and the IMA are real daughters that never leave the wall within 10 mm (2 of 19 references; subject 2), and no vein in the dev set clears the threshold. See filters.py rule 3
TANGENCY_ANGLE_DEG = 70.0   # rule 3 secondary, FLAG ONLY (tangential): direction more than 70 deg from the outward wall normal, i.e. running along the wall
PATCH_ASPECT_RATIO_MAX = 3.0  # rule 3 secondary, FLAG ONLY (elongated_patch), and the D4 trigger for the hugging-branch ostium (strip_end): a wall patch elongated more than 3:1 is a vessel running along the wall
STRIP_END_WINDOW_MM = 4.0   # D4 hugging branch: the ostium of an elongated patch (aspect over PATCH_ASPECT_RATIO_MAX) lies in the end window that hugs the wall most tightly; 4 mm covers the strip's own width (2 to 4 mm at lumbar scale) and is under half of any strip that passes the aspect test
AREA_GROWTH_MAX = 2.0       # rule 4, FLAG ONLY (area_growth): cross-section at 4 to 5 mm over the first step clear of the wall layer. Its only hits on the labelled cases are real 4.5 mm branches whose first 3 mm read narrow at 1.5 mm voxels. Bowel cases looked at 13 Sep: subject 12 (uncropped field of view) flags nothing and its kept branches grow 0.3 to 1.1 x; subject 8's two flagged patches (2.3 and 2.5 x) are already rejected, one a duplicate contact of a kept 4.1 mm branch and one a bone contact. Arming it would remove nothing on the bowel cases and lose real branches on the labelled ones, so it stays a flag
REGION_VOLUME_CAP_ML = 1.0  # rule 4: applied to the branch's voxels within TRACE_MAX_MM of the ostium (the proximal segment). No proximal segment approaches 1 ml (every labelled reference is under 0.8 ml), a vertebral body or heart chamber does. Not the whole watershed basin, which floods every connected bright voxel in the shell
BONE_HU_LUMEN_RATIO = 1.5   # rule 4: contrast-filled blood is the brightest soft tissue in the scan; voxels over 1.5 x the lumen median in a cross-section are cortical bone or calcium. Bone only becomes a candidate on poorly enhanced scans (lumen about 230 HU, threshold about 105, so cancellous bone clears it), and there the cortex reads 400 to 1200 HU
DUPLICATE_MM = 4.0          # rule 6: two ostia within 4 mm (with directions within DUPLICATE_ANGLE_DEG) are one opening, and two survivors whose 10 mm paths come within 4 mm share a lumen; separate lumbar pairs are further apart than that. Ostia on the same working voxel merge regardless of direction
DUPLICATE_ANGLE_DEG = 20.0  # rule 6: and their directions within 20 deg; genuinely separate nearby origins diverge
MIN_ORIGIN_DIAMETER_MM = 2.0  # rule 7: the reference policy's minimum estimated lumen diameter at the origin (docs/references, minimum_origin_diameter_mm); equivalent diameter of the wall patch
BORDERLINE_ORIGIN_DIAMETER_MM = (1.5, 2.5)  # rule 7: at 1.5 mm voxels a 2 mm lumen is 1.3 voxels wide, so origins in this band are flagged borderline_diameter in the display; both excluded candidates in the reference package sit in this band
MIN_WALL_PATCH_VOXELS = 3   # rule 7: a wall patch under 3 voxels on the 0.8 mm grid is smaller than any 2 mm origin can be and is pure noise. Sweep: 1 voxel adds 15 false positives and recovers nothing (reference 19/b2 fails the 5 mm test regardless)

# ------------------------------------------------------------------------ scorer (D7)
LABELLED_CASES = (19, 20, 21, 22, 23)  # dev cases with draft reference annotations in docs/references (all 1.5 mm isotropic)
REGRESSION_ORDER = (19, 20, 21, 22, 23, 16, 17, 18, 24, 1, 6, 8)  # D7 priority order: labelled coarse cases (scored), unlabelled coarse cases (invariants only), the allowed-to-fail unenhanced pair, then fine-resolution robustness cases (underfill, tortuous, uncropped field of view); 25 is out of scope
MATCH_CUTOFFS_MM = (3.0, 5.0, 8.0)  # ostium-distance cutoffs reported; the organisers have not specified one
MATCH_CUTOFF_MM = 5.0       # working cutoff for one-to-one matching (open question 3)
INVARIANT_RADIUS_MIN_MM = 1.0  # invariant: nothing under 1 mm is resolvable at 0.8 mm voxels
INVARIANT_RADIUS_MAX_MM = 8.0  # invariant: no direct daughter of the abdominal aorta has a radius over 8 mm except the iliacs, which are out of scope
INVARIANT_MIN_OSTIUM_SEPARATION_MM = 4.0  # invariant: mirrors the duplicate-merge distance, so two reported ostia are never closer than a merge would allow
INVARIANT_REGION_MAX_PATH_MM = 15.0  # invariant: a watershed region reaching more than 15 mm from its own ostium has leaked into a neighbouring vessel
INVARIANT_UNIT_NORM_TOL = 1e-3  # invariant: a direction vector is "unit" if its norm is within this of 1 after JSON rounding
INVARIANT_SEED_CHORD_TOL_MM = 0.5  # invariant: the seed-to-ostium chord may exceed SEED_DISTANCE_MM by JSON rounding plus half a working voxel, no more

# ------------------------------------------------------------------------ runtime (D10)
RUNTIME_TARGET_S = 35.0     # target per case; leaves headroom under the organiser 60 s average
RUNTIME_CAP_S = 60.0        # organiser initial per-case limit
PEAK_MEMORY_CAP_GB = 4.0    # half of the organiser 8 GB machine, so the OS and a second process fit

# ------------------------------------------------------------------------ output
OUTPUT_MM_DECIMALS = 3      # mm values written to JSON: sub-voxel precision is meaningless beyond this
OUTPUT_DIRECTION_DECIMALS = 4  # direction components: keeps the unit norm within 1e-3 after rounding

# ------------------------------------------------------------------------ frame (D8)
CENTRELINE_STEP_MM = 1.0    # centreline resampled at 1 mm arc length: finer than any working voxel, coarse enough that heights and clock lookups are cheap
CENTRELINE_SMOOTH_MM = 3.0  # Gaussian sigma along the arc applied to the voxel-chain path: removes the one-voxel zigzag of a grid path without flattening a bend (the tightest aortic bends in the set have radii of several centimetres)
CENTRELINE_COST_POWER = 2.0  # geodesic cost = (inside distance)^-2: a wall-hugging shortcut across a bend costs more than the extra length of staying central, so the path follows the axis (1st power lets the path cut corners on the tortuous cases)
CENTRELINE_CORE_FRACTION = 0.5  # a path point is 'central' once its inside distance reaches half the median along the path; the geodesic endpoints sit on the end-face rim and the first stretch of the path runs from the rim to the axis, so those points are replaced by the end-face centroid
END_FACE_NORMAL_DEG = 60.0  # an end-face voxel's outward normal is within 60 deg of the outward centreline tangent (a lateral-wall voxel's normal is near 90 deg); accommodates a tapered end (subject 20) and an oblique cut
END_FACE_LATERAL_FACTOR = 1.5  # end-face voxels lie within 1.5 x the local aortic radius of the extended axis, so a neighbouring structure fused to the mask at the cut is not counted
CLOCK_ANTERIOR_MIN_SIN = 0.25  # 12 o'clock is the patient's anterior projected into the plane perpendicular to the local tangent (the surgeon's convention at every level; the transported frame drifts 43 deg from it on subject 3). The projection is undefined when the tangent runs antero-posteriorly; below this sine (tangent within about 15 deg of the y axis, which the abdominal aorta never does) the rotation-minimising transport is used instead
CENTRELINE_SWEEP_DOWNSAMPLE = 2  # the two farthest-point sweeps that find the end rims run on a mask downsampled by this factor (endpoints identical on the longest case, 6 x faster); only the central path itself runs at working resolution
