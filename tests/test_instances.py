import numpy as np

import config
import instances
import io_utils


def test_single_branch_gives_one_instance(cand, inst, phantom):
    assert inst.n == 1
    assert inst.labels_list == [1]
    assert inst.labels.dtype == np.int32 and inst.labels.shape == cand.mask.shape
    assert set(np.unique(inst.labels)) == {0, 1}
    # wall patch sits within the wall layer, and the watershed region contains the seed
    wi = inst.wall_indices[1]
    assert wi.shape[1] == 3 and len(wi) > 0
    assert cand.distance_mm[tuple(wi.T)].max() <= config.WALL_LAYER_MM
    assert (inst.wall_labels[tuple(wi.T)] == 1).all()
    seed_idx = np.round(io_utils.mm_to_index(cand.image, phantom["seed_mm"])).astype(int)
    assert inst.labels[tuple(seed_idx)] == 1
    assert inst.wall_area_mm2[1] > 0 and inst.region_volume_ml[1] > 0


def test_labels_are_subset_of_candidates(cand, inst):
    assert not (inst.labels > 0)[~cand.candidates].any()
    assert ((inst.wall_labels > 0) <= (inst.labels > 0)).all()


def test_no_candidates_gives_empty_instances(cand):
    import copy
    c = copy.copy(cand)
    c.candidates = np.zeros_like(cand.candidates)
    inst = instances.build(c)
    assert inst.n == 0 and inst.labels_list == [] and inst.wall_indices == {}
    assert inst.labels.shape == cand.mask.shape


def test_two_separate_patches_are_two_instances_ordered_by_area(cand):
    """Two blobs separated at the wall are two instances; the larger is label 1."""
    import copy
    c = copy.copy(cand)
    wall = cand.candidates & (cand.distance_mm <= config.WALL_LAYER_MM)
    zs, ys, xs = np.where(wall)
    # duplicate the branch's wall patch on the opposite (-x) side of the aorta, smaller
    cx = int(round(np.argwhere(cand.mask).mean(axis=0)[2]))
    mirror = np.zeros_like(wall)
    keep = zs <= np.median(zs)  # the lower half of the patch: connected, and smaller
    mirror[zs[keep], ys[keep], 2 * cx - xs[keep]] = True
    c.candidates = cand.candidates | mirror
    inst = instances.build(c)
    assert inst.n == 2
    assert inst.wall_area_mm2[1] >= inst.wall_area_mm2[2]
