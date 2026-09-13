import numpy as np

import config
import tracing


def test_trace_contract(traces, ostia, phantom):
    assert set(traces) == {1}
    t = traces[1]
    assert t.path_mm.shape[1] == 3
    assert np.allclose(t.path_mm[0], ostia[1].mm)
    assert t.path_length_mm == config.TRACE_MAX_MM
    assert t.seed_mm is not None
    assert abs(np.linalg.norm(t.seed_mm - ostia[1].mm) - config.SEED_DISTANCE_MM) < 1e-6
    assert abs(np.linalg.norm(t.direction_xyz) - 1) < 1e-6
    assert t.radius_mm > 0
    assert t.bifurcation is False


def test_seed_and_direction_near_truth(traces, phantom):
    t = traces[1]
    assert np.linalg.norm(t.seed_mm - phantom["seed_mm"]) < 3.0
    assert np.dot(t.direction_xyz, phantom["direction"]) > 0.8


def test_direction_is_the_ostium_to_seed_chord(traces, ostia):
    t = traces[1]
    assert np.allclose(t.direction_xyz, tracing.chord_direction(ostia[1].mm, t.seed_mm))
    assert np.allclose(tracing.chord_direction([0, 0, 0], [0, 3, 4]), [0, 0.6, 0.8])
    assert config.RADIUS_CLAMP_MM[0] <= t.radius_mm <= config.RADIUS_CLAMP_MM[1]


def test_point_along():
    path = np.array([[0, 0, 0], [3, 0, 0], [3, 4, 0]], float)
    assert np.allclose(tracing.point_along(path, 0.0), [0, 0, 0])
    assert np.allclose(tracing.point_along(path, 5.0), [3, 2, 0])
    assert tracing.point_along(path, 7.5) is None
    assert tracing.point_along(path[:1], 1.0) is None


def test_trace_stops_at_volume_edge(cand, ostia, inst):
    """A branch leaving within 5 mm of the boundary cannot be traced 5 mm (PDF eligibility)."""
    import copy
    o = copy.copy(ostia[1])
    o.index_zyx = np.array([cand.mask.shape[0] - 2, o.index_zyx[1], o.index_zyx[2]])
    o.normal_zyx = np.array([1.0, 0.0, 0.0])  # straight into the +z edge
    t = tracing.trace_one(cand, o, inst.wall_area_mm2[1])
    assert t.path_length_mm < config.MIN_TRACE_MM
    assert t.seed_mm is None
