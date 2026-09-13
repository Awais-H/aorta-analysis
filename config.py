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
OPENING_RADIUS_MM = 0.8     # binary opening ball: one working voxel; the partial-volume ring outside the mask is one to two voxels thick, branch roots are 3 mm or wider
OPENING_MIN_RADIUS_VOX = 1  # the opening ball is never smaller than one voxel, otherwise it is a no-op

# ------------------------------------------------------------------------ ostium (D4)
AXIS_FIT_MM = 10.0          # PCA for the branch axis uses watershed voxels within the first 10 mm, the spec's tracing horizon
OSTIUM_AGREEMENT_MM = 4.0   # axis-intersection ostium is used only if within 4 mm of the inscribed-circle ostium, so a bad axis fit never beats the baseline

# ------------------------------------------------------------------------ tracing (D5)
SEED_DISTANCE_MM = 5.0      # PDF definition: the daughter seed is 5 mm outward from the ostium along the daughter path
TRACE_MAX_MM = 10.0         # PDF: trace up to 10 mm beyond the ostium or until the first bifurcation
TRACE_STEP_MM = 1.0         # march step: fine enough to follow a looping renal, coarse enough that one step spans more than one working voxel
MIN_TRACE_MM = 5.0          # PDF eligibility: a branch must be followable for at least 5 mm beyond the wall
RADIUS_SAMPLE_OFFSETS_MM = (4.0, 5.0, 6.0)  # radius is averaged over the cross-sections around the seed because one 0.8 mm slice of a 3 mm vessel is only 10 to 15 voxels
RADIUS_DISAGREEMENT_FRACTION = 0.30  # if inscribed-circle and sqrt(area/pi) radii differ by more than 30%, the cross-section grabbed adjacent tissue: use the smaller

# ------------------------------------------------------------------------ filters (D6)
END_FACE_MM = 3.0           # rule 1: a wall patch within 3 mm of a centreline endpoint touches an end face (end faces are not axial planes, so this is a distance, not a slice index)
END_FACE_ANGLE_DEG = 20.0   # rule 1: direction within 20 deg of the local centreline tangent means the aorta continuing, not a branch
END_FACE_AREA_FRACTION = 0.40  # rule 1 supporting signal: patch area over 40% of the aortic cross-section (renal patches reach this too, so support only)
END_FACE_AREA_SHORTCUT = 1.5  # rule 1: a patch over 1.5 x the aortic cross-section is a continuation regardless of direction (subjects 19, 20: about 2 x)
FRANGI_SCALES_MM = (1.0, 2.0, 3.0)  # rule 2 secondary / rule 4: three vesselness scales spanning lumbar to renal radii; three scales max for runtime
FRANGI_MIN_RESPONSE = 0.0   # rule 2 secondary: mean Frangi response floor. PLACEHOLDER, not yet set by the spec; 0.0 disables the rule until a sweep sets it
DEPARTURE_MM = 3.0          # rule 3: a real branch has moved more than 3 mm from the aorta surface after 10 mm of path; a parallel vein or the SMA lying on the wall never leaves. First sensitivity sweep: 2 to 4 mm
TANGENCY_ANGLE_DEG = 70.0   # rule 3 secondary: direction more than 70 deg from the outward wall normal is tangential, i.e. running along the wall
PATCH_ASPECT_RATIO_MAX = 3.0  # rule 3 secondary: a wall patch elongated more than 3:1 is a vessel hugging the wall, not an opening
AREA_GROWTH_MAX = 2.0       # rule 4: a cross-section that more than doubles over the first 5 mm is organ or bowel, not a tube
REGION_VOLUME_CAP_ML = 1.0  # rule 4: no proximal branch segment within 15 mm approaches 1 ml, but a vertebral body or heart chamber does
DUPLICATE_MM = 4.0          # rule 6: two ostia within 4 mm are one opening; separate lumbar pairs are further apart than that
DUPLICATE_ANGLE_DEG = 20.0  # rule 6: and their directions within 20 deg; genuinely separate nearby origins diverge
MIN_ORIGIN_RADIUS_MM = 1.5  # rule 7: PLACEHOLDER for the organiser minimum origin size (open question 2); equivalent radius of the wall patch

# ------------------------------------------------------------------------ scorer (D7)
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
