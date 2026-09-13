import types

import numpy as np

import candidates
import config
import instances
import io_utils
import ostium


def test_ostium_contract(cand, inst, ostia, phantom):
    assert set(ostia) == {1}
    o = ostia[1]
    assert o.label == 1
    assert o.index_zyx.shape == (3,) and o.index_zyx.dtype.kind == "i"
    assert cand.mask[tuple(o.index_zyx)], "ostium is snapped onto the raw mask"
    for v in (o.normal_zyx, o.normal_mm, o.axis_zyx, o.axis_mm):
        assert abs(np.linalg.norm(v) - 1) < 1e-6
    assert o.method in ("axis_intersection", "inscribed_circle")
    assert o.inscribed_idx is not None


def test_straight_branch_lands_on_truth(ostia, phantom):
    o = ostia[1]
    assert o.method == "inscribed_circle"  # OSTIUM_USE_AXIS_REFINEMENT is off by default
    assert o.axis_idx is not None, "the axis fit still runs and is logged"
    assert o.agreement_mm is not None and o.agreement_mm <= config.OSTIUM_AGREEMENT_MM
    assert np.linalg.norm(o.mm - phantom["ostium_mm"]) < 1.2  # within 1.5 working voxels
    assert np.dot(o.normal_mm, phantom["direction"]) > 0.9


def test_axis_refinement_when_enabled(cand, inst, phantom, monkeypatch):
    monkeypatch.setattr(config, "OSTIUM_USE_AXIS_REFINEMENT", True)
    o = ostium.locate(cand, inst)[1]
    assert o.method == "axis_intersection"
    assert np.linalg.norm(o.mm - phantom["ostium_mm"]) < 1.2
    assert np.dot(o.axis_mm, phantom["direction"]) > 0.95


def test_curved_branch_ostium(phantom_curved):
    image, mask_image, _ = io_utils.load_case(phantom_curved["image"], phantom_curved["mask"])
    cand = candidates.build(image, mask_image)
    inst = instances.build(cand)
    o = ostium.locate(cand, inst)[1]
    assert np.linalg.norm(o.mm - phantom_curved["ostium_mm"]) < 1.6
    assert cand.mask[tuple(o.index_zyx)]


def test_inscribed_centre_of_elongated_patch():
    """A 20 x 4 voxel patch smeared along the wall: the centre must be mid-way, not at an end."""
    shape = (6, 30, 30)
    sp = np.array([0.8, 0.8, 0.8])
    dist = np.zeros(shape, np.float32)
    for z in range(1, shape[0]):
        dist[z] = z * sp[0]          # mask is the z = 0 face; wall layer is z in {1, 2}
    fake = types.SimpleNamespace(distance_mm=dist, spacing=sp, mask=np.zeros(shape, bool))
    zz, yy, xx = np.meshgrid([1, 2], np.arange(5, 25), np.arange(10, 14), indexing="ij")
    wall_idx = np.column_stack([zz.ravel(), yy.ravel(), xx.ravel()])
    c = ostium.inscribed_centre(fake, wall_idx)
    assert 13 <= c[1] <= 16 and 11 <= c[2] <= 12
    assert c[0] in (1, 2)


def test_axis_fit_fails_gracefully_on_too_few_voxels(cand):
    assert ostium.axis_fit(cand, np.zeros((0, 3), int), np.zeros(3)) is None
    assert ostium.axis_fit(cand, np.array([[1, 1, 1], [1, 1, 2]]), np.zeros(3)) is None


def test_boundary_voxels_are_on_the_surface(cand):
    b = ostium.mask_boundary_voxels(cand.mask)
    assert len(b) > 0
    assert cand.mask[tuple(b.T)].all()
    assert len(b) < len(np.argwhere(cand.mask))


def test_no_instances_gives_no_ostia(cand):
    import copy
    c = copy.copy(cand)
    c.opened = np.zeros_like(cand.opened)
    c.bright_shell = np.zeros_like(cand.bright_shell)
    assert ostium.locate(c, instances.build(c)) == {}
