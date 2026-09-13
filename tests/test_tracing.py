import numpy as np

import candidates
import config
import instances
import io_utils
import ostium
import tracing
from conftest import BRANCH_R


def _trace(ph):
    image, mask_image, _ = io_utils.load_case(ph["image"], ph["mask"])
    cand = candidates.build(image, mask_image)
    inst = instances.build(cand)
    ostia = ostium.locate(cand, inst)
    return cand, inst, ostia, tracing.trace_all(cand, inst, ostia)


def test_trace_contract(traces, ostia):
    assert set(traces) == {1}
    t = traces[1]
    assert t.path_mm.shape[1] == 3
    assert np.allclose(t.path_mm[0], ostia[1].mm)
    assert t.seed_mm is not None
    assert abs(np.linalg.norm(t.direction_xyz) - 1) < 1e-6
    assert config.RADIUS_CLAMP_MM[0] <= t.radius_mm <= config.RADIUS_CLAMP_MM[1]
    assert t.method in ("march", "axis", "none") and t.stop_reason


def test_straight_branch(traces, phantom):
    t = traces[1]
    assert t.method == "march" and t.stop_reason == "max"
    assert t.path_length_mm == config.TRACE_MAX_MM
    assert t.bifurcation is False
    assert np.linalg.norm(t.seed_mm - phantom["seed_mm"]) < 1.0
    assert np.degrees(np.arccos(np.dot(t.direction_xyz, phantom["direction"]))) < 8.0
    assert abs(t.radius_mm - BRANCH_R) < 0.5
    assert t.radius_flag is None and t.pca_chord_angle_deg is not None and t.pca_chord_angle_deg < 10.0


def test_direction_is_the_ostium_to_seed_chord(traces, ostia):
    t = traces[1]
    assert np.allclose(t.direction_xyz, tracing.chord_direction(ostia[1].mm, t.seed_mm))
    assert np.allclose(tracing.chord_direction([0, 0, 0], [0, 3, 4]), [0, 0.6, 0.8])
    assert abs(np.linalg.norm(t.seed_mm - ostia[1].mm) - config.SEED_DISTANCE_MM) < 0.3


def test_curved_branch_march_follows_the_lumen(phantom_curved):
    cand, inst, ostia, traces = _trace(phantom_curved)
    t = traces[1]
    assert t.method == "march"
    assert t.path_length_mm >= config.SEED_DISTANCE_MM
    assert np.linalg.norm(t.seed_mm - phantom_curved["seed_mm"]) < 1.5
    seed_idx = np.round(io_utils.mm_to_index(cand.image, t.seed_mm)).astype(int)
    assert cand.bright_shell[tuple(seed_idx)], "seed sits in the lumen"
    # the path bends: its far end is well above the ostium plane
    assert t.path_mm[-1][2] - t.path_mm[0][2] > 2.0
    assert abs(t.radius_mm - BRANCH_R) < 0.6


def test_bifurcating_branch_stops_at_the_split(phantom_bifurcating):
    cand, inst, ostia, traces = _trace(phantom_bifurcating)
    t = traces[1]
    assert t.bifurcation is True and t.stop_reason == "bifurcation"
    # the trunk is traced in full; the stop comes where the children's cross-sections separate
    assert phantom_bifurcating["bifurcation_mm"] <= t.path_length_mm < config.TRACE_MAX_MM
    assert np.linalg.norm(t.seed_mm - phantom_bifurcating["seed_mm"]) < 1.0
    assert np.dot(t.direction_xyz, phantom_bifurcating["direction"]) > 0.98  # the trunk, not a child


def test_point_along():
    path = np.array([[0, 0, 0], [3, 0, 0], [3, 4, 0]], float)
    assert np.allclose(tracing.point_along(path, 0.0), [0, 0, 0])
    assert np.allclose(tracing.point_along(path, 5.0), [3, 2, 0])
    assert tracing.point_along(path, 7.5) is None
    assert tracing.point_along(path[:1], 1.0) is None


def test_pca_direction_sign_points_away_from_origin():
    pts = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0.1], [3, 0, 0]], float)
    d = tracing.pca_direction(pts, np.array([0.0, 0, 0]))
    assert d[0] > 0.99
    d = tracing.pca_direction(pts[::-1], np.array([3.0, 0, 0]))
    assert d[0] < -0.99
    assert tracing.pca_direction(pts[:2], pts[0]) is None


def test_trace_stops_at_volume_edge(cand, inst, ostia):
    """A branch leaving within 5 mm of the boundary cannot be traced 5 mm (PDF eligibility)."""
    import copy
    o = copy.copy(ostia[1])
    o.index_zyx = np.array([cand.mask.shape[0] - 2, o.index_zyx[1], o.index_zyx[2]])
    o.axis_zyx = np.array([1.0, 0.0, 0.0])  # straight into the +z edge
    o.normal_zyx = o.axis_zyx
    region = instances.branch_voxels(cand, inst)[1]
    t = tracing.trace_one(cand, o, region)
    assert t.path_length_mm < config.MIN_TRACE_MM
    assert t.seed_mm is None


def test_no_voxels_gives_no_seed(cand, ostia):
    t = tracing.trace_one(cand, ostia[1], np.zeros((0, 3), int))
    assert t.seed_mm is None and t.method == "none"


def test_radius_at_on_the_phantom(cand, phantom):
    seed_idx = io_utils.mm_to_index(cand.image, phantom["seed_mm"])
    d = io_utils.mm_vector_to_index(cand.image, phantom["seed_mm"], phantom["direction"])
    r, r_area, r_ins, flag, fills, cortex = tracing.radius_at(cand, seed_idx, d, local_aortic_radius_mm=6.0)
    assert fills is False and cortex == 0.0
    # the phantom threshold (280 HU on a 300 HU lumen) sits at the voxel centres, so the interpolated
    # cross-section is about half a voxel small; real thresholds are far below the lumen value
    assert abs(r_area - BRANCH_R) < 0.5 and abs(r_ins - BRANCH_R) < 0.5
    assert flag is None and abs(r - r_area) < 1e-9
    # sanity rule: a tiny "aorta" makes the branch look too big, so the inscribed circle is used
    r2, _, _, flag2, _, _ = tracing.radius_at(cand, seed_idx, d, local_aortic_radius_mm=1.0)
    assert flag2 == "inscribed_fallback" and abs(r2 - r_ins) < 1e-9
    # far from any bright voxel there is no cross-section
    far = seed_idx + np.array([0.0, 12.0, 0.0]) / cand.spacing
    r3, _, _, flag3, _, _ = tracing.radius_at(cand, far, d, None)
    assert flag3 == "no_cross_section" and r3 == config.RADIUS_CLAMP_MM[0]


def test_section_fills_window_on_a_slab(cand):
    """A plane through a wide bright slab (bone-like) runs out of the window; the phantom branch does not."""
    import copy
    c = copy.copy(cand)
    ct = cand.ct.copy()
    ct[:, :, 5:9] = 500.0  # a bright slab 4 voxels thick spanning the whole y-z extent, away from the aorta
    c.ct = ct
    seed_idx = np.array([cand.mask.shape[0] / 2, cand.mask.shape[1] / 2, 7.0])
    _, _, _, _, fills, cortex = tracing.radius_at(c, seed_idx, np.array([0.0, 0.0, 1.0]), None)
    assert fills is True and cortex > 0.9  # 500 HU against a 300 HU lumen: cortical bone
    ct[:, :, 5:9] = 320.0  # the same slab at lumen brightness: reaches the edge but is not bone
    c.ct = ct
    _, _, _, _, fills, cortex = tracing.radius_at(c, seed_idx, np.array([0.0, 0.0, 1.0]), None)
    assert fills is True and cortex == 0.0
