"""D6 false-positive filtering: hard rules, one per false-positive class in the PDF's list.

Every rule is evaluated for every branch (even after an earlier hit) so the log carries all the
measurements; a branch survives only with no hit. Rejections are (label, rule, value); flags are
soft signals for the display and the failure gallery. Default on borderline: keep.

1. end_face        aorta continuing past the mask: wall patch touches an end face (within
                   END_FACE_MM of the end plane at a centreline endpoint) AND direction within
                   END_FACE_ANGLE_DEG of the outward centreline tangent; a patch over
                   END_FACE_AREA_SHORTCUT x the aortic cross-section short-circuits the angle test.
2. min_trace_mm    plaque / eligibility: traced path under MIN_TRACE_MM (Frangi secondary is
                   disabled while FRANGI_MIN_RESPONSE is 0).
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
   area_growth     FLAG ONLY for now: cross-section more than AREA_GROWTH_MAX x larger at 4 to 5 mm
                   than at the first step clear of the wall layer. On the labelled cases the only
                   hits are real 4.5 mm branches whose first 3 mm read narrow at 1.5 mm voxels, and
                   no labelled false positive grows; it becomes a rejection once the bowel cases
                   (subjects 8, 12) have been reviewed in the failure gallery.
5. (invariant, no rule) branches of branches cannot occur: instances need wall contact.
6. duplicate       two surviving ostia within DUPLICATE_MM with directions within
                   DUPLICATE_ANGLE_DEG: keep the larger wall patch.
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
    """(max height of the patch above the nearest end plane in mm, outward tangent of that end).
    Height is (voxel - endpoint) . outward tangent; a patch touches the face when its highest
    voxel is within END_FACE_MM below the plane or beyond it."""
    best = (-np.inf, None)
    W = wall_idx * cand.spacing
    for e_mm, t_mm in end_faces:
        e = io_utils.mm_to_index(cand.image, e_mm) * cand.spacing
        t = io_utils.mm_vector_to_index(cand.image, e_mm, t_mm)
        h = float(((W - e) @ t).max())
        if h > best[0]:
            best = (h, t_mm)
    return best


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
    _, r_area, r_ins, flag = tracing.radius_at(cand, idx, d_idx, None)
    if r_area is None:
        return None, flag
    return 2.0 * r_area, flag


def merge_duplicates(survivors: list) -> tuple:
    """survivors: [(label, ostium_mm, direction_xyz, wall_area_mm2)]. Larger patch wins.
    Returns (kept labels, [(label, 'duplicate', distance, kept_label)])."""
    order = sorted(survivors, key=lambda s: -s[3])
    kept, rejected = [], []
    for label, o, d, a in order:
        dup = None
        for k_label, k_o, k_d, _ in kept:
            dist = float(np.linalg.norm(np.subtract(o, k_o)))
            if dist < config.DUPLICATE_MM and _angle(d, k_d) < config.DUPLICATE_ANGLE_DEG:
                dup = (k_label, dist)
                break
        if dup is None:
            kept.append((label, o, d, a))
        else:
            rejected.append((label, "duplicate", dup[1], dup[0]))
    return [k[0] for k in kept], rejected


# ------------------------------------------------------------------ apply


def apply(cand: Candidates, inst: Instances, ostia: dict, traces: dict, frame=None) -> FilterResult:
    res = FilterResult()
    xsec = aortic_cross_section_mm2(cand)
    end_faces = frame.end_faces() if frame is not None else []
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
        if end_faces:
            h, t_out = end_face_contact(cand, wall_idx, end_faces)
            ratio = inst.wall_area_mm2[label] / xsec if xsec > 0 else 0.0
            ang = _angle(tr.direction_xyz, t_out)
            m["end_face_height_mm"], m["end_face_angle_deg"], m["area_ratio"] = round(h, 2), round(ang, 1), round(ratio, 2)
            if h >= -config.END_FACE_MM:
                flags.append("near_cut_face")
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
            lo, hi = config.BORDERLINE_ORIGIN_DIAMETER_MM
            if lo <= diam <= hi:
                flags.append("borderline_diameter")
        elif dflag:
            flags.append(f"origin_{dflag}")

        res.rejections += [(label, r, v) for r, v in hits]
        res.flags[label], res.measurements[label] = flags, m
        if not hits:
            survivors.append((label, ost.mm, tr.direction_xyz, inst.wall_area_mm2[label]))

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
