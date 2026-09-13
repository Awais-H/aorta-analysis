import numpy as np

from branchseed.aorta import analyze_aorta, clean_aorta_mask, signed_distance_mm


def _cylinder(shape=(32, 48, 48), radius=9):
    z, y, x = np.indices(shape)
    return (y - 24) ** 2 + (x - 24) ** 2 <= radius**2


def test_anisotropic_distance_and_largest_component():
    mask = np.zeros((9, 15, 15), bool)
    mask[2:7, 4:11, 4:11] = True
    mask[0, 0, 0] = True
    cleaned = clean_aorta_mask(mask, (1.0, 1.0, 3.0), min_volume_mm3=1)
    assert not cleaned[0, 0, 0]
    distance = signed_distance_mm(cleaned, (1.0, 1.0, 3.0))
    assert distance[4, 7, 7] >= 4.0
    assert distance[0, 7, 7] < 0


def test_crop_caps_are_geometry_based_and_side_daughter_is_lateral():
    mask = _cylinder()
    # A small side daughter reaches a different image face.
    mask[14:19, 21:27, 24:] = True
    image = np.where(mask, 220.0, 20.0).astype(np.float32)
    result = analyze_aorta(image, mask, (1.0, 1.0, 1.5))
    assert result.end_caps[0].sum() > 20
    assert result.end_caps[-1].sum() > 20
    assert result.lateral_wall[16, 24, -1]
    assert result.blood is not None
    assert abs(result.blood.median - 220) < 1


def test_rotated_direction_changes_physical_bbox_and_centreline():
    mask = _cylinder(shape=(24, 40, 40), radius=6)
    image = mask.astype(np.float32) * 150
    rotation = (0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    result = analyze_aorta(
        image, mask, (2.0, 1.0, 1.5), origin_xyz=(10, 20, 30), direction=rotation
    )
    assert result.bounding_box is not None
    assert result.bounding_box.minimum_xyz[0] < 0
    assert result.centreline_xyz.shape[1] == 3
    direction = result.tangents_xyz[len(result.tangents_xyz) // 2]
    assert abs(direction[2]) > 0.8


def test_empty_mask_returns_empty_geometry():
    result = analyze_aorta(
        np.random.default_rng(2).normal(size=(8, 9, 10)).astype(np.float32),
        np.zeros((8, 9, 10), bool),
        (1, 1, 2),
    )
    assert not result.mask.any()
    assert result.blood is None
    assert result.bounding_box is None


def test_wide_internal_cylinder_caps_exclude_lateral_mid_wall():
    z, y, x = np.indices((70, 64, 64))
    mask = ((y - 32) ** 2 + (x - 32) ** 2 <= 12**2) & (z >= 8) & (z <= 61)
    result = analyze_aorta(mask.astype(np.float32) * 180, mask, (1, 1, 1))
    assert result.end_caps[8].sum() > 300
    assert result.end_caps[61].sum() > 300
    assert not result.end_caps[35, 32, 44]
    assert result.lateral_wall[35, 32, 44]
