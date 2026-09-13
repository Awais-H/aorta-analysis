import numpy as np

import ostium


def test_ostium_contract(cand, inst, ostia, phantom):
    assert set(ostia) == {1}
    o = ostia[1]
    assert o.label == 1
    assert o.index_zyx.shape == (3,) and o.index_zyx.dtype.kind == "i"
    assert cand.mask[tuple(o.index_zyx)], "ostium is snapped onto the raw mask"
    assert abs(np.linalg.norm(o.normal_zyx) - 1) < 1e-6
    assert abs(np.linalg.norm(o.normal_mm) - 1) < 1e-6


def test_ostium_near_truth_and_normal_outward(ostia, phantom):
    o = ostia[1]
    assert np.linalg.norm(o.mm - phantom["ostium_mm"]) < 3.0
    assert np.dot(o.normal_mm, phantom["direction"]) > 0.8


def test_boundary_voxels_are_on_the_surface(cand):
    b = ostium.mask_boundary_voxels(cand.mask)
    assert len(b) > 0
    assert cand.mask[tuple(b.T)].all()
    inner = np.argwhere(cand.mask)
    assert len(b) < len(inner)


def test_no_instances_gives_no_ostia(cand, inst):
    import instances
    import copy
    c = copy.copy(cand)
    c.candidates = np.zeros_like(cand.candidates)
    assert ostium.locate(c, instances.build(c)) == {}
