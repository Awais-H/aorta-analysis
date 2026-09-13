"""Output contract, including the mandated command line.

An empty daughters list must serialise correctly and the process must exit 0.
It is an easy path to leave broken, because every interesting case has
branches.
"""

from __future__ import annotations

import gzip
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from branchseed.pipeline import process_images

ROOT = Path(__file__).resolve().parent.parent
MANDATED_DAUGHTER_FIELDS = {
    "instance_id", "parent_instance_id", "ostium_xyz_mm", "seed_xyz_mm",
    "radius_mm", "direction_xyz",
}


def _validate(payload: dict) -> None:
    assert isinstance(payload["case_id"], str)
    assert payload["parent"] == {"instance_id": "aorta"}
    assert isinstance(payload["daughters"], list)

    ids = set()
    for daughter in payload["daughters"]:
        assert MANDATED_DAUGHTER_FIELDS <= set(daughter), MANDATED_DAUGHTER_FIELDS - set(daughter)
        assert daughter["parent_instance_id"] == "aorta"
        assert daughter["instance_id"] not in ids
        ids.add(daughter["instance_id"])

        for key in ("ostium_xyz_mm", "seed_xyz_mm", "direction_xyz"):
            value = daughter[key]
            assert isinstance(value, list) and len(value) == 3
            assert all(isinstance(v, float) and np.isfinite(v) for v in value)

        norm = float(np.linalg.norm(daughter["direction_xyz"]))
        assert abs(norm - 1.0) < 1e-6, f"direction is not unit: {norm}"
        assert isinstance(daughter["radius_mm"], float)
        assert daughter["radius_mm"] > 0.0


def test_schema_on_a_populated_case(standard_phantom, cfg):
    payload = process_images(standard_phantom.image, standard_phantom.mask, cfg, "s").payload
    _validate(payload)
    assert len(payload["daughters"]) == 4


def test_schema_on_an_empty_case(no_branch_phantom, cfg):
    payload = process_images(no_branch_phantom.image, no_branch_phantom.mask, cfg, "e").payload
    _validate(payload)
    assert payload["daughters"] == []
    assert json.loads(json.dumps(payload)) == payload


def test_empty_mask_is_not_a_crash(standard_phantom, cfg):
    """An empty mask means an empty daughters list and a clean exit."""
    empty = sitk.GetImageFromArray(
        np.zeros(sitk.GetArrayFromImage(standard_phantom.mask).shape, dtype=np.uint8)
    )
    empty.CopyInformation(standard_phantom.mask)
    payload = process_images(standard_phantom.image, empty, cfg, "empty").payload
    _validate(payload)
    assert payload["daughters"] == []
    assert "empty_mask" in payload["meta"]["case_flags"]


def test_unusable_mask_is_not_a_crash(standard_phantom, cfg):
    """Too few voxels for a centreline: a flag, not a traceback."""
    array = np.zeros(sitk.GetArrayFromImage(standard_phantom.mask).shape, dtype=np.uint8)
    array[10:12, 10:12, 10:12] = 1
    tiny = sitk.GetImageFromArray(array)
    tiny.CopyInformation(standard_phantom.mask)
    payload = process_images(standard_phantom.image, tiny, cfg, "tiny").payload
    _validate(payload)
    assert payload["daughters"] == []
    flags = payload["meta"]["case_flags"]
    assert "no_centreline" in flags or "pipeline_error" in flags


def test_meta_block_records_provenance(standard_phantom, cfg):
    meta = process_images(standard_phantom.image, standard_phantom.mask, cfg, "s").payload["meta"]
    for key in ("version", "config_hash", "runtime_seconds", "peak_memory_mb",
                "stage_seconds", "case_flags", "candidates_generated",
                "candidates_rejected", "calibration"):
        assert key in meta, key
    assert meta["runtime_seconds"] > 0.0


@pytest.mark.parametrize("suffix", [".nii", ".nii.gz", "gzip_named.nii"])
def test_mandated_cli(tmp_path, standard_phantom, suffix):
    """Exactly the command the organisers will run, on all three file-naming
    variants the hidden set might contain."""
    image_path = tmp_path / f"image{suffix if suffix.startswith('.') else '.nii'}"
    mask_path = tmp_path / f"aorta_mask{suffix if suffix.startswith('.') else '.nii'}"
    if suffix == "gzip_named.nii":
        for image, target in ((standard_phantom.image, image_path),
                              (standard_phantom.mask, mask_path)):
            plain = tmp_path / (target.stem + "_plain.nii")
            sitk.WriteImage(image, str(plain))
            with open(plain, "rb") as src, gzip.open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
    else:
        sitk.WriteImage(standard_phantom.image, str(image_path))
        sitk.WriteImage(standard_phantom.mask, str(mask_path))

    output = tmp_path / "prediction.json"
    completed = subprocess.run(
        [sys.executable, "run.py", "--image", str(image_path),
         "--aorta-mask", str(mask_path), "--output", str(output)],
        cwd=ROOT, capture_output=True, text=True, timeout=900,
    )
    assert completed.returncode == 0, completed.stderr[-3000:]
    assert completed.stdout == "", f"stdout must stay clean, got: {completed.stdout!r}"

    payload = json.loads(output.read_text())
    _validate(payload)
    assert len(payload["daughters"]) == 4


def test_cli_writes_figures_when_asked(tmp_path, standard_phantom):
    image_path, mask_path = tmp_path / "image.nii.gz", tmp_path / "mask.nii.gz"
    sitk.WriteImage(standard_phantom.image, str(image_path))
    sitk.WriteImage(standard_phantom.mask, str(mask_path))
    viz_dir = tmp_path / "viz"
    completed = subprocess.run(
        [sys.executable, "run.py", "--image", str(image_path), "--aorta-mask", str(mask_path),
         "--output", str(tmp_path / "p.json"), "--viz-dir", str(viz_dir), "--case-id", "demo"],
        cwd=ROOT, capture_output=True, text=True, timeout=900,
    )
    assert completed.returncode == 0, completed.stderr[-3000:]
    for name in ("demo_wallmap.png", "demo_3d.png", "demo_mip.png", "branchseed.log"):
        assert (viz_dir / name).exists(), name


def test_voxel_output_round_trips_against_sitk(standard_phantom, cfg):
    """Voxel-unit coordinates must match SimpleITK's own physical->index
    transform on the original image (rounded to the nearest voxel, the
    ITK-SNAP cursor-position convention), and the continuous variants must
    recover the mandated mm output exactly, up to that rounding."""
    from branchseed.output import build_voxel_output

    result = process_images(standard_phantom.image, standard_phantom.mask, cfg, "v")
    voxel_payload = build_voxel_output(
        result.payload["case_id"], result.accepted, result.grid, cfg
    )
    assert voxel_payload["daughters"], "expected daughters on the standard phantom"

    image = standard_phantom.image
    size = np.asarray(image.GetSize())
    for mm_daughter, voxel_daughter in zip(result.payload["daughters"], voxel_payload["daughters"]):
        assert voxel_daughter["instance_id"] == mm_daughter["instance_id"]
        for mm_key, voxel_key, continuous_key in (
            ("ostium_xyz_mm", "ostium_ijk_voxel", "ostium_ijk_voxel_continuous"),
            ("seed_xyz_mm", "seed_ijk_voxel", "seed_ijk_voxel_continuous"),
        ):
            expected = np.asarray(
                image.TransformPhysicalPointToContinuousIndex(mm_daughter[mm_key])
            )
            np.testing.assert_allclose(voxel_daughter[continuous_key], expected, atol=1e-2)
            # The rounded field must be the nearest integer to *our own*
            # continuous value - not independently re-derived from
            # ``expected``, which can land on the opposite side of a .5
            # boundary from a value that agrees with it to 1e-6.
            np.testing.assert_allclose(
                voxel_daughter[voxel_key], voxel_daughter[continuous_key], atol=0.500001
            )
            # Rounded voxel coordinates: integer, in-bounds, never negative.
            for component in voxel_daughter[voxel_key]:
                assert isinstance(component, int)
            assert (np.asarray(voxel_daughter[voxel_key]) >= 0).all()
            assert (np.asarray(voxel_daughter[voxel_key]) <= size - 1).all()

        # radius_voxels must convert back to (approximately) radius_mm.
        spacing = np.mean(image.GetSpacing())
        np.testing.assert_allclose(
            voxel_daughter["radius_voxels"] * spacing, mm_daughter["radius_mm"], atol=1e-2
        )
        np.testing.assert_allclose(np.linalg.norm(voxel_daughter["direction_ijk"]), 1.0, atol=1e-6)


def test_voxel_output_via_cli(tmp_path, standard_phantom):
    """--voxel-output writes a second file without disturbing the mandated one."""
    from branchseed.config import load_config

    image_path, mask_path = tmp_path / "image.nii.gz", tmp_path / "mask.nii.gz"
    sitk.WriteImage(standard_phantom.image, str(image_path))
    sitk.WriteImage(standard_phantom.mask, str(mask_path))
    mm_out = tmp_path / "prediction.json"
    voxel_out = tmp_path / "prediction_voxel.json"

    completed = subprocess.run(
        [sys.executable, "run.py", "--image", str(image_path), "--aorta-mask", str(mask_path),
         "--output", str(mm_out), "--voxel-output", str(voxel_out)],
        cwd=ROOT, capture_output=True, text=True, timeout=900,
    )
    assert completed.returncode == 0, completed.stderr[-3000:]

    mm_payload = json.loads(mm_out.read_text())
    voxel_payload = json.loads(voxel_out.read_text())
    assert len(voxel_payload["daughters"]) == len(mm_payload["daughters"]) == 4
    assert "units" in voxel_payload
    for daughter in voxel_payload["daughters"]:
        assert "ostium_ijk_voxel" in daughter
        assert "seed_ijk_voxel" in daughter
        assert len(daughter["ostium_ijk_voxel"]) == 3
