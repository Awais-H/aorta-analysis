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
    kept, rej = filters.merge_duplicates([a, b, c, d])
    assert kept == [4, 1, 3]
    assert rej == [(2, "duplicate", 3.0, 1)]


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
    h, t = filters.end_face_contact(cand, inst.wall_indices[1], faces)
    assert h < -config.END_FACE_MM  # the mid-height branch is far below both end planes
    # a synthetic patch at the very top slice touches the superior face
    top = np.argwhere(cand.mask[-1:])
    top[:, 0] = cand.mask.shape[0] - 1
    h2, _ = filters.end_face_contact(cand, top, faces)
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
