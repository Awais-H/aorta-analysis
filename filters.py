"""D6 false-positive filtering: hard rules, one per false-positive class in the PDF's list.

Every rule is evaluated for every branch (even after an earlier hit) so the log carries all the
measurements; a branch survives only with no hit. Rejections are (label, rule, value); flags are
soft signals for the display and the failure gallery. Default on borderline: keep.

1. end_face        aorta continuing past the mask: wall patch touches an end face (within
                   END_FACE_MM of the end plane at a centreline endpoint) AND direction within
                   END_FACE_ANGLE_DEG of the outward centreline tangent; a patch over
                   END_FACE_AREA_SHORTCUT x the aortic cross-section short-circuits the angle test.
   iliac_division  the terminal division of the aorta is outside the task (PDF; the reference
                   checklist applies "the agreed exclusion of the terminal iliac division", and
                   case 23's notes exclude its iliac split). Signature, measured on all 25 cases:
                   in the slab LUMEN_PROBE_MM beyond the inferior face, within
                   LUMEN_PROBE_LATERAL_FACTOR aortic radii of the axis, two bright lumens both over
                   ILIAC_LUMEN_MIN_DIAMETER_MM and alike to ILIAC_LUMEN_RATIO_MIN (subjects 17, 22,
                   23: 9 to 13 mm each; 19, 20, 21: one 17 to 20 mm lumen, the aorta continuing).
                   Then a candidate touching the inferior face with an origin over
                   ILIAC_CANDIDATE_MIN_DIAMETER_MM is one of the iliacs. Rule 1's continuation
                   test already removes the larger one; this removes the other (22 patch 24, 23
                   patch 5: the two false positives the eye review had called the iliac division).
2. min_trace_mm    plaque / eligibility: traced path under MIN_TRACE_MM (Frangi secondary is
                   disabled while FRANGI_MIN_RESPONSE is 0).
   path_at_image_edge  eligibility at the volume boundary, added 13 Sep (night): the first
                   SEED_DISTANCE_MM of the traced path run within IMAGE_EDGE_MARGIN_VOXELS native
                   voxels of a face where the native image ends. Every cross-section there is
                   truncated by the volume, so the 5 mm followability the PDF requires cannot be
                   verified; the reference notes for case 20 exclude the posterior tracks at its
                   superior crop limit for exactly that reason. Counted on all 25 cases: 2 of 147
                   kept branches (20 patch 4, the excluded track; 6 patch 5, the aortic lumen
                   outside the mask at the inferior image edge), no reference, and the next
                   nearest kept path is more than two voxels from any image face. Not the same
                   thing as the mask's end faces (rule 1): a cut face inside the volume leaves the
                   vessel's cross-sections intact and is judged on its own evidence.
3. departure_mm    FLAG ONLY (no_departure): the far end of the traced path is still within
                   DEPARTURE_MM of the aorta surface. Rethought against the anatomy and the
                   references: a branch that leaves the aorta and then runs along it is still a
                   branch. Two of the 19 reference daughters (subjects 20, 23) are wall-hugging
                   lumbars whose own 10 mm guides end 1.8 and 2.5 mm from the mask, and the only
                   contact strip on subjects 1 to 3 (the cases the rule was written for) is a
                   2.9 mm vessel at 1 o'clock near the inferior end of subject 2 with a clear origin
                   end, i.e. the inferior mesenteric artery, not the SMA or a vein. Veins in the
                   arterial phase do not clear the threshold, so the rule had no demonstrated
                   benefit and cost three real daughters. The biologically meaningful test is
                   whether a contact strip HAS AN ORIGIN END (D4 strip_end: one end tighter, wider,
                   brighter, no body beyond the wall; a vein touching in passing has two alike
                   ends). That asymmetry is logged (strip_hug_ratio) so the rule can be re-armed on
                   a negative example; none exists in the dev set. Tangency angle and patch aspect
                   ratio stay as flags.
4. proximal_volume_ml bowel, organ, bone: the branch's voxels within TRACE_MAX_MM of the ostium
                   (the proximal segment) exceed REGION_VOLUME_CAP_ML. The whole watershed basin is
                   not used: through the raw shell it floods every connected bright voxel (renals on
                   the labelled cases carry 1 to 4 ml basins), and the opened component is one sheet
                   around most of the aorta (5 to 17 ml). Within 10 mm of the ostium every labelled
                   reference is under 0.8 ml.
   bone            the thresholded cross-section at the seed reaches the edge of the
                   CROSS_SECTION_HALF_WIDTH_MM window (16 mm across) AND contains cortex (voxels over
                   BONE_HU_LUMEN_RATIO x the lumen median). A daughter's lumen 5 mm out is a closed
                   region no wider than the largest daughter; a vertebral body runs out of the
                   window and carries a cortical shell far brighter than blood. Reviewed on the
                   gallery sheets: on subject 22 (threshold 105 HU, so cancellous bone is "bright")
                   all eight vertebral-contact candidates show both signs and none of the eleven
                   real vessels shows either; on subject 16 the four posterior candidates at 500 to
                   1200 HU. Reaching the edge WITHOUT cortex is flagged section_merged and kept:
                   reference 19/b3 (2.7 mm) does that because at 1.5 mm voxels its section merges
                   with an adjacent vessel at lumen brightness.
   blob_at_seed    the inscribed circle of the seed cross-section reaches BLOB_INSCRIBED_RADIUS_MM:
                   the seed sits in a bright region at least 16 mm across, which no daughter lumen
                   is 5 mm from the aorta. An intervertebral disc at a 105 HU threshold (subject
                   22 patch 32, no cortex so the bone test cannot fire) and the heart and arch
                   contacts of subject 25; no reference.
   area_growth     FLAG ONLY for now: cross-section more than AREA_GROWTH_MAX x larger at 4 to 5 mm
                   than at the first step clear of the wall layer. On the labelled cases the only
                   hits are real 4.5 mm branches whose first 3 mm read narrow at 1.5 mm voxels, and
                   no labelled false positive grows. The bowel cases were reviewed on 13 Sep
                   (gallery sheets of subjects 8 and 12 with the rejected patches): subject 12
                   flags nothing, subject 8's two hits are already rejected as a duplicate and as
                   bone. No case in the dev set needs the rule, so it stays a flag.
5. (invariant, no rule) branches of branches cannot occur: instances need wall contact.
6. same_origin_rejected  a survivor whose ostium lies on the same working voxel (within
                   ISO_SPACING_MM) as a patch rejected for what it is (bone, blob_at_seed,
                   proximal_volume_ml, iliac_division, end_face) is the same opening: one origin
                   is one instance, and if that origin is a disc contact so is its twin patch
                   (subject 22 patches 32 and 35, 0.8 mm apart, both 12 mm origins into the disc;
                   35 surfaced when 32 was rejected because rule 6 only merged survivors).
   duplicate       two surviving ostia within DUPLICATE_MM with directions within
                   DUPLICATE_ANGLE_DEG, or on the same working voxel whatever their directions (one
                   origin is one instance; a trunk that splits still has one ostium), or whose
                   traced paths come within DUPLICATE_MM of each other (two daughters cannot share
                   a lumen: a wall-hugging vessel can produce several contact patches along its
                   length, e.g. subject 17, and each traces into the same vessel): keep the larger
                   wall patch.
7. tiny_patch      wall patch under MIN_WALL_PATCH_VOXELS;
   origin_diameter_mm  area-equivalent diameter of the thresholded cross-section perpendicular to
                   the path at the first plane clear of the wall layer (WALL_LAYER_MM out; closer
                   planes cut the partial-volume ring) under MIN_ORIGIN_DIAMETER_MM; the
                   BORDERLINE_ORIGIN_DIAMETER_MM band sets the borderline_diameter flag.
                   Checked against the anatomy on reference 19/b2 (2.3 mm, one opened voxel, the
                   only rule 7 loss): the voxel floor is not what loses it. The vessel is 1.5 native
                   voxels wide, so the ring-removing opening erases it inside the wall layer and the
                   march cannot start (it also fails rule 2). Dropping the floor to 1 voxel adds 15
                   false positives and recovers nothing; an 'outward support' wall layer instead
                   of the opening was prototyped and rejected (17 -> 12 true positives: the ring
                   skirt it keeps merges neighbouring origins). Recorded as a resolution limit.

Output contract (SPEC.md D9): surviving labels, rejection log of (label, rule, value).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

import config
import io_utils
import tracing
from candidates import Candidates
from instances import Instances, branch_voxels, patch_aspect_ratio

log = logging.getLogger("branchseed.filters")


@dataclass
class FilterResult:
    kept: list = field(default_factory=list)          # labels that survive, in instance order
    rejections: list = field(default_factory=list)    # (label, rule, value)
    flags: dict = field(default_factory=dict)         # label -> [flag, ...]
    measurements: dict = field(default_factory=dict)  # label -> {name: value}, the tuning ledger
    terminal_division: dict = field(default_factory=dict)  # rule 1: lumens beyond the inferior face and whether they are the iliacs


# ------------------------------------------------------------------ helpers


def _angle(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))))


def aortic_cross_section_mm2(cand: Candidates) -> float:
    """Median per-slice mask area (the atlas convention, so the rule 1 numbers in SPEC apply)."""
    areas = cand.mask.sum(axis=(1, 2)) * float(cand.spacing[1] * cand.spacing[2])
    areas = areas[areas > 0]
    return float(np.median(areas)) if len(areas) else float("nan")


def end_face_contact(cand: Candidates, wall_idx: np.ndarray, end_faces: list):
    """(max height of the patch above the nearest end plane in mm, outward tangent of that end,
    index of that end: 0 superior, 1 inferior). Height is (voxel - endpoint) . outward tangent;
    a patch touches the face when its highest voxel is within END_FACE_MM below the plane or
    beyond it."""
    best = (-np.inf, None, -1)
    W = wall_idx * cand.spacing
    for k, (e_mm, t_mm) in enumerate(end_faces):
        e = io_utils.mm_to_index(cand.image, e_mm) * cand.spacing
        t = io_utils.mm_vector_to_index(cand.image, e_mm, t_mm)
        h = float(((W - e) @ t).max())
        if h > best[0]:
            best = (h, t_mm, k)
    return best


def lumens_beyond_face(cand: Candidates, end_mm: np.ndarray, t_out_mm: np.ndarray, aortic_radius_mm: float) -> list:
    """Equivalent diameters (mm, largest first) of the bright lumens in a one-voxel slab
    LUMEN_PROBE_MM beyond an end face, within LUMEN_PROBE_LATERAL_FACTOR aortic radii of the
    extended axis. Raw thresholded voxels outside the mask, 26-connected in the slab."""
    q = np.asarray(end_mm, float) + config.LUMEN_PROBE_MM * np.asarray(t_out_mm, float)
    qi = io_utils.mm_to_index(cand.image, q)
    ti = io_utils.mm_vector_to_index(cand.image, q, t_out_mm)
    ti = ti / max(np.linalg.norm(ti), 1e-12)
    V = np.argwhere(cand.bright_shell)
    if len(V) == 0:
        return []
    d = (V - qi) * cand.spacing
    along = d @ ti
    lat = np.linalg.norm(d - along[:, None] * ti[None, :], axis=1)
    sel = (np.abs(along) <= config.TRACE_SLAB_HALF_MM) & (lat <= config.LUMEN_PROBE_LATERAL_FACTOR * aortic_radius_mm)
    if not sel.any():
        return []
    pts = V[sel]
    lo = pts.min(axis=0)
    box = np.zeros(tuple(pts.max(axis=0) - lo + 1), bool)
    box[tuple((pts - lo).T)] = True
    lab, k = ndimage.label(box, structure=np.ones((3, 3, 3), bool))
    sizes = np.bincount(lab[box])[1:]
    area = float(cand.spacing[1] * cand.spacing[2])
    return sorted((2.0 * np.sqrt(s_ * area / np.pi) for s_ in sizes), reverse=True)


def terminal_division(cand: Candidates, frame) -> dict:
    """Whether the supplied segment ends at the aortic bifurcation: two alike lumens over
    ILIAC_LUMEN_MIN_DIAMETER_MM beyond the inferior face. Returns the measurement."""
    out = {"detected": False, "lumens_mm": []}
    if frame is None or frame.radius_mm is None or len(frame.radius_mm) == 0:
        return out
    e, t = frame.end_faces()[1]
    lumens = lumens_beyond_face(cand, e, t, float(np.median(frame.radius_mm)))
    out["lumens_mm"] = [round(v, 1) for v in lumens[:4]]
    if len(lumens) >= 2 and lumens[1] >= config.ILIAC_LUMEN_MIN_DIAMETER_MM and lumens[1] >= config.ILIAC_LUMEN_RATIO_MIN * lumens[0]:
        out["detected"] = True
    return out


def image_edge_distance_mm(cand: Candidates, path_mm: np.ndarray, reach_mm: float = config.SEED_DISTANCE_MM) -> float:
    """Smallest distance from the traced path's first `reach_mm` to a face of the native image.
    A working-grid face counts only where the crop reached the native volume's edge; elsewhere
    the crop's CROP_PAD_MM of padding lies beyond it and the face is not an image boundary.
    Returns inf when the grid information is missing or no face is an image boundary."""
    info = getattr(cand, "grid_info", None) or {}
    crop, native = info.get("crop_zyx"), info.get("native_shape_zyx")
    if not crop or not native or path_mm is None or len(path_mm) == 0:
        return float("inf")
    pts = [tracing.point_along(path_mm, s) for s in np.arange(0.0, reach_mm + 1e-6, config.TRACE_STEP_MM / 2)]
    pts = np.array([p for p in pts if p is not None])
    if len(pts) == 0:
        return float("inf")
    idx = io_utils.mm_to_index(cand.image, pts)  # (N, 3) zyx continuous, working grid
    shape = np.array(cand.mask.shape, float)
    best = float("inf")
    for ax in range(3):
        if crop[ax][0] == 0:
            best = min(best, float((idx[:, ax] * cand.spacing[ax]).min()))
        if crop[ax][1] >= native[ax]:
            best = min(best, float(((shape[ax] - 1 - idx[:, ax]) * cand.spacing[ax]).min()))
    return max(best, 0.0)


def departure_mm(cand: Candidates, path_mm: np.ndarray) -> float:
    """Distance from the aorta surface at the far end of the traced path (trilinear)."""
    idx = io_utils.mm_to_index(cand.image, path_mm[-1])
    return float(ndimage.map_coordinates(cand.distance_mm, idx.reshape(3, 1), order=1, cval=0.0)[0])


def area_growth(areas) -> float | None:
    """Cross-section area at 4 to 5 mm over the area at the first march step clear of the wall
    layer (WALL_LAYER_MM + one step), or None. Steps inside the wall layer run through opened
    voxels and are systematically smaller than the raw-voxel steps beyond it."""
    first = int(round(config.WALL_LAYER_MM / config.TRACE_STEP_MM))  # 0-based index of the step at WALL_LAYER_MM + 1 step
    last = int(round(config.SEED_DISTANCE_MM / config.TRACE_STEP_MM))
    if not areas or len(areas) < last:
        return None
    a0 = float(areas[first])
    a1 = float(np.mean(areas[first + 1:last]))
    return a1 / a0 if a0 > 0 else None


def proximal_volume_ml(cand: Candidates, region_idx: np.ndarray, ostium_idx: np.ndarray) -> float:
    """Volume of the branch's voxels within TRACE_MAX_MM of the ostium: the proximal segment."""
    if len(region_idx) == 0:
        return 0.0
    d = np.linalg.norm((region_idx - ostium_idx) * cand.spacing, axis=1)
    return float((d <= config.TRACE_MAX_MM).sum() * np.prod(cand.spacing) / 1000.0)


def origin_diameter_mm(cand: Candidates, trace) -> tuple:
    """Area-equivalent diameter at the first cross-section clear of the wall layer.
    Returns (diameter_mm or None, flag)."""
    p = tracing.point_along(trace.path_mm, config.WALL_LAYER_MM)
    if p is None:
        return None, "no_cross_section"
    d = tracing.chord_direction(trace.path_mm[0], p)
    idx = io_utils.mm_to_index(cand.image, p)
    d_idx = io_utils.mm_vector_to_index(cand.image, p, d)
    _, r_area, r_ins, flag, _, _ = tracing.radius_at(cand, idx, d_idx, None)
    if r_area is None:
        return None, flag
    return 2.0 * r_area, flag


def _path_gap_mm(p: np.ndarray, q: np.ndarray) -> float:
    """Smallest distance between two polylines' points (mm); paths are sampled every TRACE_STEP_MM."""
    if p is None or q is None or len(p) == 0 or len(q) == 0:
        return float("inf")
    return float(np.linalg.norm(p[:, None, :] - q[None, :, :], axis=2).min())


def merge_duplicates(survivors: list) -> tuple:
    """survivors: [(label, ostium_mm, direction_xyz, wall_area_mm2[, path_mm])]. Larger patch wins.
    Returns (kept labels, [(label, 'duplicate', distance, kept_label)])."""
    order = sorted(survivors, key=lambda s: -s[3])
    kept, rejected = [], []
    for item in order:
        label, o, d, a = item[:4]
        path = item[4] if len(item) > 4 else None
        dup = None
        for k in kept:
            k_label, k_o, k_d, _ = k[:4]
            k_path = k[4] if len(k) > 4 else None
            dist = float(np.linalg.norm(np.subtract(o, k_o)))
            if dist <= config.ISO_SPACING_MM or (dist < config.DUPLICATE_MM and _angle(d, k_d) < config.DUPLICATE_ANGLE_DEG):
                dup = (k_label, dist)
                break
            gap = _path_gap_mm(path, k_path)
            if gap < config.DUPLICATE_MM:
                dup = (k_label, gap)
                break
        if dup is None:
            kept.append(item)
        else:
            rejected.append((label, "duplicate", dup[1], dup[0]))
    return [k[0] for k in kept], rejected


# ------------------------------------------------------------------ apply


def apply(cand: Candidates, inst: Instances, ostia: dict, traces: dict, frame=None) -> FilterResult:
    res = FilterResult()
    xsec = aortic_cross_section_mm2(cand)
    end_faces = frame.end_faces() if frame is not None else []
    division = terminal_division(cand, frame) if end_faces else {"detected": False, "lumens_mm": []}
    res.terminal_division = division
    if division["detected"]:
        log.info("segment ends at the aortic bifurcation: lumens %s mm beyond the inferior face", division["lumens_mm"])
    regions = branch_voxels(cand, inst)
    survivors = []
    for label in inst.labels_list:
        hits, flags, m = [], [], {}
        wall_idx = inst.wall_indices[label]
        m["wall_voxels"] = int(len(wall_idx))
        m["wall_area_mm2"] = round(inst.wall_area_mm2[label], 1)
        m["region_ml"] = round(inst.region_volume_ml[label], 3)

        # 7a: noise
        if len(wall_idx) < config.MIN_WALL_PATCH_VOXELS:
            hits.append(("tiny_patch", float(len(wall_idx))))

        # 2: eligibility (traced path length)
        tr = traces.get(label)
        ost = ostia.get(label)
        length = tr.path_length_mm if tr is not None else 0.0
        m["path_mm"] = length
        if tr is None or ost is None or tr.seed_mm is None or length < config.MIN_TRACE_MM:
            hits.append(("min_trace_mm", float(length)))
            res.rejections += [(label, r, v) for r, v in hits]
            res.flags[label], res.measurements[label] = flags, m
            continue
        # 2: the first 5 mm run along the image boundary, so followability cannot be verified
        edge = image_edge_distance_mm(cand, tr.path_mm)
        m["image_edge_mm"] = None if not np.isfinite(edge) else round(edge, 2)
        native_sp = (getattr(cand, "grid_info", None) or {}).get("native_spacing_zyx_mm")
        if np.isfinite(edge) and native_sp and edge < config.IMAGE_EDGE_MARGIN_VOXELS * max(native_sp):
            hits.append(("path_at_image_edge", float(edge)))
        if tr.method == "axis":
            flags.append("axis_fallback")
        if tr.bifurcation:
            flags.append("bifurcation")
        if tr.radius_flag:
            flags.append(f"radius_{tr.radius_flag}")

        # 4: volume cap on the proximal segment
        pv = proximal_volume_ml(cand, regions.get(label, np.zeros((0, 3), int)), ost.index_zyx)
        m["proximal_ml"] = round(pv, 3)
        if pv > config.REGION_VOLUME_CAP_ML:
            hits.append(("proximal_volume_ml", pv))

        # 1: end faces
        on_inferior_face = False
        if end_faces:
            h, t_out, which_end = end_face_contact(cand, wall_idx, end_faces)
            ratio = inst.wall_area_mm2[label] / xsec if xsec > 0 else 0.0
            ang = _angle(tr.direction_xyz, t_out)
            m["end_face_height_mm"], m["end_face_angle_deg"], m["area_ratio"] = round(h, 2), round(ang, 1), round(ratio, 2)
            if h >= -config.END_FACE_MM:
                flags.append("near_cut_face")
                on_inferior_face = which_end == 1
                if ratio > config.END_FACE_AREA_SHORTCUT:
                    hits.append(("end_face", float(ratio)))
                elif ang <= config.END_FACE_ANGLE_DEG:
                    hits.append(("end_face", float(ang)))
                elif ratio > config.END_FACE_AREA_FRACTION:
                    flags.append("large_patch_at_cut")

        # 3: departure and its secondary signals (all flags, see module docstring)
        dep = departure_mm(cand, tr.path_mm)
        m["departure_mm"] = round(dep, 2)
        if dep < config.DEPARTURE_MM:
            flags.append("no_departure")
        if ost.method == "strip_end" and ost.strip_hug_mm is not None:
            a, b = ost.strip_hug_mm
            m["strip_hug_ratio"] = round(max(a, b) / max(min(a, b), 1e-6), 2)
        tang = _angle(tr.direction_xyz, ost.normal_mm)
        asp = patch_aspect_ratio(cand, wall_idx)
        m["tangency_deg"], m["aspect_ratio"] = round(tang, 1), round(asp, 2)
        if tang > config.TANGENCY_ANGLE_DEG:
            flags.append("tangential")
        if asp > config.PATCH_ASPECT_RATIO_MAX:
            flags.append("elongated_patch")

        # 4: the seed sits inside a bright region at least 2 x BLOB_INSCRIBED_RADIUS_MM across
        m["seed_inscribed_mm"] = None if tr.radius_inscribed_mm is None else round(tr.radius_inscribed_mm, 2)
        if tr.radius_inscribed_mm is not None and tr.radius_inscribed_mm >= config.BLOB_INSCRIBED_RADIUS_MM:
            hits.append(("blob_at_seed", float(tr.radius_inscribed_mm)))

        # 4: the seed cross-section runs out of the window with a cortical shell: bone
        m["cortex_fraction"] = round(tr.section_cortex_fraction, 3)
        if tr.section_fills_window and tr.section_cortex_fraction > 0:
            hits.append(("bone", float(tr.section_cortex_fraction)))
        elif tr.section_fills_window:
            flags.append("section_merged")

        # 4: area growth along the march (flag only, see module docstring)
        g = area_growth(tr.section_areas_mm2)
        m["area_growth"] = None if g is None else round(g, 2)
        if g is not None and g > config.AREA_GROWTH_MAX:
            flags.append("area_growth")

        # 7b: origin diameter
        diam, dflag = origin_diameter_mm(cand, tr)
        m["origin_diameter_mm"] = None if diam is None else round(diam, 2)
        if diam is not None:
            if diam < config.MIN_ORIGIN_DIAMETER_MM:
                hits.append(("origin_diameter_mm", diam))
            # 1: one of the iliacs on a segment that ends at its terminal division
            if division["detected"] and on_inferior_face and diam >= config.ILIAC_CANDIDATE_MIN_DIAMETER_MM:
                hits.append(("iliac_division", diam))
            lo, hi = config.BORDERLINE_ORIGIN_DIAMETER_MM
            if lo <= diam <= hi:
                flags.append("borderline_diameter")
        elif dflag:
            flags.append(f"origin_{dflag}")

        res.rejections += [(label, r, v) for r, v in hits]
        res.flags[label], res.measurements[label] = flags, m
        if not hits:
            survivors.append((label, ost.mm, tr.direction_xyz, inst.wall_area_mm2[label], tr.path_mm))

    # 6: a survivor on the same origin voxel as a structurally rejected patch is that structure
    structural = ("bone", "blob_at_seed", "proximal_volume_ml", "iliac_division", "end_face")
    bad_origins = [(l, ostia[l].index_zyx) for l, r, _ in res.rejections if r in structural and l in ostia]
    still = []
    for item in survivors:
        label = item[0]
        oi = ostia[label].index_zyx
        # the same or a 26-adjacent working voxel (a distance test at exactly one voxel is a coin flip in floating point)
        twin = next(((bl, float(np.linalg.norm((oi - bi) * cand.spacing))) for bl, bi in bad_origins
                     if np.abs(oi - bi).max() <= 1), None)
        if twin is None:
            still.append(item)
        else:
            res.rejections.append((label, "same_origin_rejected", twin[1]))
            res.measurements[label]["duplicate_of"] = twin[0]
    survivors = still

    # 6: duplicates among survivors
    kept, dups = merge_duplicates(survivors)
    for label, rule, dist, of in dups:
        res.rejections.append((label, rule, dist))
        res.measurements[label]["duplicate_of"] = of
    res.kept = [l for l in inst.labels_list if l in set(kept)]
    for label, rule, value in res.rejections:
        log.info("reject label %d: %s = %.2f", label, rule, value)
    log.info("%d kept, %d rejected (%d labels)", len(res.kept), len({l for l, _, _ in res.rejections}), inst.n)
    return res
