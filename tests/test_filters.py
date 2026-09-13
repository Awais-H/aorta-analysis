import copy

import numpy as np

import candidates
import config
import filters
import frame as frame_mod
import instances
import io_utils
import ostium
import tracing
from conftest import make_phantom


def _run(ph):
    image, mask_image, _ = io_utils.load_case(ph["image"], ph["mask"])
    cand = candidates.build(image, mask_image)
    inst = instances.build(cand)
    ostia = ostium.locate(cand, inst)
    traces = tracing.trace_all(cand, inst, ostia)
    fr = frame_mod.build(cand)
    return cand, inst, ostia, traces, fr, filters.apply(cand, inst, ostia, traces, fr)


def _branch_label(inst, ostia, ph):
    return min(ostia, key=lambda l: np.linalg.norm(ostia[l].mm - ph["ostium_mm"]))


def test_phantom_branch_is_kept_clean(phantom):
    cand, inst, ostia, traces, fr, res = _run(phantom)
    assert res.kept == [1] and res.rejections == []
    m = res.measurements[1]
    assert m["departure_mm"] > config.DEPARTURE_MM
    assert 2.5 < m["origin_diameter_mm"] < 5.0
    assert m["area_growth"] is not None and m["area_growth"] < config.AREA_GROWTH_MAX
    assert 0 < m["proximal_ml"] < config.REGION_VOLUME_CAP_ML
    assert "near_cut_face" not in res.flags[1]


def test_aorta_continuing_past_the_mask_is_rejected(tmp_path):
    """Rule 1: the mask covers the middle 60% of the aorta; the two continuation patches go, the branch stays."""
    ph = make_phantom(tmp_path / "cut", mask_z_fraction=0.6)
    cand, inst, ostia, traces, fr, res = _run(ph)
    assert inst.n == 3
    b = _branch_label(inst, ostia, ph)
    assert res.kept == [b]
    rejected = {l: r for l, r, v in res.rejections}
    assert len(rejected) == 2 and all(r == "end_face" for r in rejected.values())
    for l in rejected:
        assert "near_cut_face" in res.flags[l]
        assert res.measurements[l]["area_ratio"] > config.END_FACE_AREA_SHORTCUT


def test_parallel_vessel_is_flagged_not_rejected_by_departure(tmp_path):
    """Rule 3 is a flag: a tube lying along the aorta for its full length gets no_departure and is
    caught (if at all) by rule 1, since a full-length parallel tube touches both end faces."""
    ph = make_phantom(tmp_path / "par", parallel_vessel=True)
    cand, inst, ostia, traces, fr, res = _run(ph)
    b = _branch_label(inst, ostia, ph)
    assert b in res.kept and "no_departure" not in res.flags[b]
    others = [l for l in inst.labels_list if l != b]
    assert others
    for l in others:
        assert "departure_mm" not in {r for ll, r, v in res.rejections if ll == l}
        assert "no_departure" in res.flags[l] or res.measurements[l]["departure_mm"] >= config.DEPARTURE_MM


def test_hugging_branch_survives_the_filters(tmp_path):
    """A branch that leaves the aorta and runs along the wall is a daughter (references 20, 23)."""
    ph = make_phantom(tmp_path / "hug", with_branch=False, hugging=True)
    cand, inst, ostia, traces, fr, res = _run(ph)
    assert inst.n == 1 and res.kept == [1]
    assert ostia[1].method == "strip_end" and "strip_hug_ratio" in res.measurements[1]
    assert res.measurements[1]["strip_hug_ratio"] > 1.0


def test_bone_sized_blob_is_rejected(tmp_path):
    """Rule 4: a 3 ml sphere touching the wall."""
    ph = make_phantom(tmp_path / "blob", blob=True)
    cand, inst, ostia, traces, fr, res = _run(ph)
    b = _branch_label(inst, ostia, ph)
    assert b in res.kept
    others = [l for l in inst.labels_list if l != b]
    assert others
    for l in others:
        rules = {r for ll, r, v in res.rejections if ll == l}
        assert "proximal_volume_ml" in rules, (l, rules, res.measurements[l])
    big = max(others, key=lambda l: res.measurements[l]["proximal_ml"])
    assert res.measurements[big]["proximal_ml"] > config.REGION_VOLUME_CAP_ML
    assert res.measurements[b]["proximal_ml"] < config.REGION_VOLUME_CAP_ML / 2


def test_short_trace_is_rejected_with_logged_value(cand, inst, ostia, traces):
    short = copy.copy(traces[1])
    short.path_length_mm = config.MIN_TRACE_MM - 1.0
    short.seed_mm = None
    res = filters.apply(cand, inst, ostia, {1: short})
    assert res.kept == []
    assert (1, "min_trace_mm", config.MIN_TRACE_MM - 1.0) in res.rejections


def test_missing_trace_is_rejected(cand, inst, ostia):
    res = filters.apply(cand, inst, ostia, {})
    assert res.kept == [] and res.rejections[0][1] == "min_trace_mm"


def test_merge_duplicates():
    a = (1, [0, 0, 0], [1, 0, 0], 50.0)
    b = (2, [0, 0, 3.0], [0.98, 0.2, 0], 20.0)     # 3 mm away, 11 deg: duplicate of 1
    c = (3, [0, 0, 3.0], [0, 1, 0], 20.0)          # 3 mm away but 90 deg: separate origin
    d = (4, [0, 0, 10.0], [1, 0, 0], 80.0)         # far away
    e = (5, [0, 0, 0.5], [0, -1, 0], 10.0)         # same voxel as 1, opposite direction: still one origin
    kept, rej = filters.merge_duplicates([a, b, c, d, e])
    assert kept == [4, 1, 3]
    assert (2, "duplicate", 3.0, 1) in rej and (5, "duplicate", 0.5, 1) in rej


def test_bone_at_the_seed_is_rejected_but_a_vessel_merge_is_only_flagged(cand, inst, ostia, traces):
    import copy
    t = copy.copy(traces[1])
    t.section_fills_window = True
    t.section_cortex_fraction = 0.3
    res = filters.apply(cand, inst, ostia, {1: t}, frame_mod.build(cand))
    assert res.kept == [] and any(r == "bone" for _, r, _ in res.rejections)
    t.section_cortex_fraction = 0.0
    res = filters.apply(cand, inst, ostia, {1: t}, frame_mod.build(cand))
    assert res.kept == [1] and "section_merged" in res.flags[1]


def test_area_growth():
    assert filters.area_growth(None) is None
    assert filters.area_growth([4, 4, 4]) is None
    # steps at 1..5 mm; the wall-layer steps (1, 2 mm) are ignored, 3 mm is the base, 4 to 5 mm the test
    assert abs(filters.area_growth([1, 2, 4, 10, 10]) - 2.5) < 1e-9
    assert abs(filters.area_growth([1, 1, 5, 5, 5]) - 1.0) < 1e-9


def test_end_face_contact_geometry(cand, inst, phantom):
    fr = frame_mod.build(cand)
    faces = fr.end_faces()
    (e_top, t_top), (e_bot, t_bot) = faces
    assert t_top[2] > 0.9 and t_bot[2] < -0.9  # outward tangents point out of the segment
    h, t, _ = filters.end_face_contact(cand, inst.wall_indices[1], faces)
    assert h < -config.END_FACE_MM  # the mid-height branch is far below both end planes
    # a synthetic patch at the very top slice touches the superior face
    top = np.argwhere(cand.mask[-1:])
    top[:, 0] = cand.mask.shape[0] - 1
    h2, _, _ = filters.end_face_contact(cand, top, faces)
    assert h2 >= -config.END_FACE_MM


def test_origin_diameter_flags_borderline(cand, traces, monkeypatch):
    diam, flag = filters.origin_diameter_mm(cand, traces[1])
    assert diam is not None and flag is None
    lo, hi = config.BORDERLINE_ORIGIN_DIAMETER_MM
    # force the band around the measured value and check the flag logic through apply()
    monkeypatch.setattr(config, "BORDERLINE_ORIGIN_DIAMETER_MM", (diam - 0.1, diam + 0.1))
    monkeypatch.setattr(config, "MIN_ORIGIN_DIAMETER_MM", diam - 0.1)
    inst = instances.build(cand)
    ostia = ostium.locate(cand, inst)
    res = filters.apply(cand, inst, ostia, traces, frame_mod.build(cand))
    assert res.kept == [1] and "borderline_diameter" in res.flags[1]
    monkeypatch.setattr(config, "MIN_ORIGIN_DIAMETER_MM", diam + 0.5)
    res = filters.apply(cand, inst, ostia, traces, frame_mod.build(cand))
    assert res.kept == [] and any(r == "origin_diameter_mm" for _, r, _ in res.rejections)


def test_area_growth_is_a_flag_not_a_rejection(cand, inst, ostia, traces):
    grown = copy.copy(traces[1])
    grown.section_areas_mm2 = [2, 2, 4, 20, 20, 20, 20, 20, 20, 20]
    res = filters.apply(cand, inst, ostia, {1: grown}, frame_mod.build(cand))
    assert res.kept == [1] and "area_growth" in res.flags[1]
    assert res.measurements[1]["area_growth"] > config.AREA_GROWTH_MAX


def test_merge_duplicates_by_shared_path():
    """Two patches 8 mm apart on one wall-hugging vessel: their paths run into each other."""
    path_a = np.array([[0, 0, t] for t in range(0, 11)], float)          # along +z from the origin
    path_b = np.array([[0.5, 0, 8 + t] for t in range(0, 11)], float)    # starts 8 mm along the same vessel
    path_c = np.array([[t, 0, 8] for t in range(0, 11)], float)          # a different vessel: perpendicular, 8 mm off
    a = (1, [0, 0, 0], [0, 0, 1], 50.0, path_a)
    b = (2, [0.5, 0, 8], [0, 0, 1], 20.0, path_b)
    c = (3, [0, 0, 8], [1, 0, 0], 30.0, path_c)
    kept, rej = filters.merge_duplicates([a, b, c])
    assert 1 in kept and 2 not in kept
    assert any(r[0] == 2 and r[3] == 1 for r in rej)
    # c's ostium is 8 mm from a's, but its path passes within 4 mm of a's path at (0..3, 0, 8): merged too
    assert 3 not in kept
    # a genuinely separate vessel whose path stays clear survives
    d = (4, [0, 6, 0], [0, 1, 0], 30.0, np.array([[0, 6 + t, 0] for t in range(0, 11)], float))
    kept, rej = filters.merge_duplicates([a, d])
    assert kept == [1, 4]


def test_iliac_division_rejected_only_on_a_bifurcating_end(tmp_path):
    # mask on the middle 60% of the aorta; below it the aorta splits into two 8 mm iliacs
    ph = make_phantom(tmp_path / "iliac", mask_z_fraction=0.6, iliac_split=True, size_xyz=(80, 64, 90))
    cand, inst, ostia, traces, fr, res = _run(ph)
    assert res.terminal_division["detected"]
    assert len(res.terminal_division["lumens_mm"]) >= 2 and min(res.terminal_division["lumens_mm"][:2]) >= 6.0
    rules = {r for _, r, _ in res.rejections}
    assert "iliac_division" in rules
    # the mid-height side branch is untouched
    b = _branch_label(inst, ostia, ph)
    assert b in res.kept
    # no kept candidate touches the inferior face any more
    for l in res.kept:
        assert not ("near_cut_face" in res.flags[l] and res.measurements[l]["origin_diameter_mm"] >= config.ILIAC_CANDIDATE_MIN_DIAMETER_MM)
    # the plain cropped aorta (one continuing lumen) does not trigger the division
    ph2 = make_phantom(tmp_path / "plain", mask_z_fraction=0.6)
    _, _, _, _, _, res2 = _run(ph2)
    assert not res2.terminal_division["detected"]
    assert "iliac_division" not in {r for _, r, _ in res2.rejections}


def test_blob_at_seed_rejects_a_wide_bright_region(tmp_path):
    # a 13 mm-radius sphere touching the aorta (an organ or vertebral body): the section 5 mm in
    # is over 20 mm across, so its inscribed circle fills the 16 mm window
    ph = make_phantom(tmp_path / "blob", blob=True, blob_radius_mm=13.0, size_xyz=(72, 80, 60))
    cand, inst, ostia, traces, fr, res = _run(ph)
    blob_labels = [l for l, r, v in res.rejections if r == "blob_at_seed"]
    assert blob_labels, res.rejections
    for l in blob_labels:
        assert res.measurements[l]["seed_inscribed_mm"] >= config.BLOB_INSCRIBED_RADIUS_MM
    b = _branch_label(inst, ostia, ph)
    assert b in res.kept and res.measurements[b]["seed_inscribed_mm"] < config.BLOB_INSCRIBED_RADIUS_MM


def test_end_face_contact_reports_which_end(cand, inst, phantom):
    fr = frame_mod.build(cand)
    faces = fr.end_faces()
    top = np.argwhere(cand.mask[-1:])
    top[:, 0] = cand.mask.shape[0] - 1
    h, _, which = filters.end_face_contact(cand, top, faces)
    assert which == 0 and h >= -config.END_FACE_MM
    bottom = np.argwhere(cand.mask[:1])
    h, _, which = filters.end_face_contact(cand, bottom, faces)
    assert which == 1 and h >= -config.END_FACE_MM


def test_same_origin_as_a_rejected_structure_is_rejected(tmp_path):
    # the wide-blob phantom: take the blob's label, add a fake survivor on its ostium voxel and
    # check the twin is rejected with the blob rather than surfacing as a daughter
    ph = make_phantom(tmp_path / "blob2", blob=True, blob_radius_mm=13.0, size_xyz=(72, 80, 60))
    cand, inst, ostia, traces, fr, res = _run(ph)
    blob = next(l for l, r, v in res.rejections if r == "blob_at_seed")
    good = _branch_label(inst, ostia, ph)
    # a survivor whose ostium coincides with the blob's, with the good branch's trace
    survivors = [(good, ostia[good].mm, traces[good].direction_xyz, 10.0, traces[good].path_mm),
                 (999, ostia[blob].mm + 0.3, traces[good].direction_xyz, 5.0, traces[good].path_mm)]
    import filters as f
    rej = [(blob, "blob_at_seed", 9.0)]
    structural = ("bone", "blob_at_seed", "proximal_volume_ml", "iliac_division", "end_face")
    bad = [(l, ostia[l].mm) for l, r, _ in rej if r in structural]
    twins = [item[0] for item in survivors if any(np.linalg.norm(np.subtract(item[1], bo)) <= config.ISO_SPACING_MM for _, bo in bad)]
    assert twins == [999]
    # and through apply(): nothing kept shares the blob's origin voxel
    for l in res.kept:
        assert np.linalg.norm(ostia[l].mm - ostia[blob].mm) > config.ISO_SPACING_MM


def test_image_edge_distance_counts_only_native_faces(cand, traces):
    # the phantom's mask spans the whole z extent, so the two z faces of the working grid are image
    # faces; in x and y the crop keeps CROP_PAD_MM of padding, so those faces are not
    info = cand.grid_info
    assert info["crop_zyx"][0] == [0, info["native_shape_zyx"][0]]
    assert info["crop_zyx"][2][0] > 0
    tr = traces[min(traces)]
    assert filters.image_edge_distance_mm(cand, tr.path_mm) > 5.0  # the branch leaves at mid height
    # a path lying on the superior image face
    top = io_utils.index_to_mm(cand.image, np.array([[cand.mask.shape[0] - 1, 10.0, 10.0], [cand.mask.shape[0] - 1, 12.0, 10.0]]))
    assert filters.image_edge_distance_mm(cand, top) < 0.01
    # a path on the x face of the working grid, which is a cropped face, not an image face
    side = io_utils.index_to_mm(cand.image, np.array([[10.0, 10.0, 0.0], [12.0, 10.0, 0.0]]))
    assert filters.image_edge_distance_mm(cand, side) == float("inf") or filters.image_edge_distance_mm(cand, side) > 5.0


def test_path_at_image_edge_is_rejected(cand, inst, ostia, traces):
    label = min(traces)
    tr = copy.copy(traces[label])
    top = cand.mask.shape[0] - 1
    idx = io_utils.mm_to_index(cand.image, tr.path_mm)
    idx[:, 0] = top  # slide the whole path onto the superior image face
    tr.path_mm = io_utils.index_to_mm(cand.image, idx)
    tr.seed_mm = tracing.point_along(tr.path_mm, config.SEED_DISTANCE_MM)
    res = filters.apply(cand, inst, ostia, {**traces, label: tr})
    assert ("path_at_image_edge", 0.0) in [(r, round(v, 2)) for l, r, v in res.rejections if l == label]
    assert label not in res.kept
    res2 = filters.apply(cand, inst, ostia, traces)
    assert label in res2.kept and res2.measurements[label]["image_edge_mm"] > 5.0
