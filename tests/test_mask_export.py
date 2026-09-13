import os

import numpy as np
import SimpleITK as sitk

import config
import io_utils
import mask_export
from conftest import fake_daughter, fake_result


def test_aorta_label_matches_supplied_mask(phantom):
    image, mask_image, _ = io_utils.load_case(phantom["image"], phantom["mask"])
    result = fake_result(daughters=[])  # no daughters: label 2 should never appear

    out = mask_export.build_mask(result, mask_image)
    arr = sitk.GetArrayFromImage(out)
    mask_arr = sitk.GetArrayFromImage(mask_image) != 0

    assert set(np.unique(arr)) <= {0, mask_export.AORTA_LABEL}
    assert np.array_equal(arr == mask_export.AORTA_LABEL, mask_arr)


def test_daughter_label_covers_ostium_sphere_and_wins_over_aorta(phantom):
    image, mask_image, _ = io_utils.load_case(phantom["image"], phantom["mask"])
    ostium = phantom["ostium_mm"]
    daughter = fake_daughter(1, ostium, direction=phantom["direction"], radius=2.0)
    result = fake_result(daughters=[daughter])

    out = mask_export.build_mask(result, mask_image)
    arr = sitk.GetArrayFromImage(out)
    assert set(np.unique(arr)) <= {0, mask_export.AORTA_LABEL, mask_export.DAUGHTER_LABEL}

    idx = np.argwhere(arr == mask_export.DAUGHTER_LABEL)
    assert len(idx) > 0
    mm = io_utils.index_to_mm(image, idx.astype(np.float64))
    dist_ostium = np.linalg.norm(mm - np.asarray(ostium), axis=1)
    end = np.asarray(ostium) + np.asarray(phantom["direction"]) * config.TRACE_MAX_MM
    dist_line = mask_export._point_segment_distance(mm, np.asarray(ostium), end)
    assert np.all((dist_ostium <= 2.0 + 1e-6) | (dist_line <= config.MASK_EXPORT_DIRECTION_RADIUS_MM + 1e-6))
    assert dist_ostium.min() < 1.0

    # the ostium sits on the aortic wall, so the daughter label must have overwritten the aorta
    # label there rather than leaving the aorta label underneath
    mask_arr = sitk.GetArrayFromImage(mask_image) != 0
    ostium_idx = tuple(np.round(io_utils.mm_to_index(image, ostium)).astype(int))
    assert mask_arr[ostium_idx]  # sanity: the phantom's ostium is on the mask
    assert arr[ostium_idx] == mask_export.DAUGHTER_LABEL


def test_direction_segment_reaches_toward_seed(phantom):
    image, mask_image, _ = io_utils.load_case(phantom["image"], phantom["mask"])
    ostium = phantom["ostium_mm"]
    daughter = fake_daughter(1, ostium, direction=phantom["direction"], radius=1.0)
    result = fake_result(daughters=[daughter])

    out = mask_export.build_mask(result, mask_image)
    arr = sitk.GetArrayFromImage(out)
    idx = np.argwhere(arr == mask_export.DAUGHTER_LABEL)
    mm = io_utils.index_to_mm(image, idx.astype(np.float64))
    seed = np.asarray(daughter["seed_xyz_mm"])
    assert np.linalg.norm(mm - seed[None, :], axis=1).min() < max(image.GetSpacing())


def test_grid_matches_aorta_mask(phantom):
    image, mask_image, _ = io_utils.load_case(phantom["image"], phantom["mask"])
    out = mask_export.build_mask(fake_result(daughters=[]), mask_image)
    assert out.GetSize() == mask_image.GetSize()
    assert np.allclose(out.GetSpacing(), mask_image.GetSpacing())
    assert np.allclose(out.GetOrigin(), mask_image.GetOrigin())
    assert np.allclose(out.GetDirection(), mask_image.GetDirection())


def test_no_daughters_gives_aorta_only_mask(phantom):
    image, mask_image, _ = io_utils.load_case(phantom["image"], phantom["mask"])
    out = mask_export.build_mask(fake_result(daughters=[]), mask_image)
    arr = sitk.GetArrayFromImage(out)
    assert mask_export.DAUGHTER_LABEL not in np.unique(arr)


def test_json_to_mask_round_trip_and_label_file(phantom, tmp_path):
    import json

    ostium = phantom["ostium_mm"]
    daughter = fake_daughter(1, ostium, direction=phantom["direction"], radius=2.0)
    result = fake_result(case_id="subject000", daughters=[daughter])
    json_path = tmp_path / "prediction.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f)

    out_path = tmp_path / "aorta_daughters.nii.gz"
    mask_export.json_to_mask(str(json_path), phantom["mask"], str(out_path))
    out = sitk.ReadImage(str(out_path))
    arr = sitk.GetArrayFromImage(out)
    assert np.any(arr == mask_export.AORTA_LABEL)
    assert np.any(arr == mask_export.DAUGHTER_LABEL)

    labels_path = tmp_path / "aorta_daughters_labels.txt"
    assert os.path.exists(labels_path)
    text = labels_path.read_text(encoding="utf-8")
    assert '"aorta"' in text and '"daughter branches"' in text
    assert f"{mask_export.AORTA_LABEL} {mask_export.AORTA_COLOR[0]} {mask_export.AORTA_COLOR[1]} {mask_export.AORTA_COLOR[2]}" in text
    assert f"{mask_export.DAUGHTER_LABEL} {mask_export.DAUGHTER_COLOR[0]} {mask_export.DAUGHTER_COLOR[1]} {mask_export.DAUGHTER_COLOR[2]}" in text
