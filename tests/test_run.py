import json
import os
import subprocess
import sys

import run
from conftest import ROOT


def _run(args):
    return subprocess.run([sys.executable, os.path.join(ROOT, "run.py")] + args, capture_output=True, text=True, cwd=ROOT)


def test_cli_on_phantom(phantom, tmp_path):
    out = tmp_path / "pred.json"
    meta = tmp_path / "meta.json"
    p = _run(["--image", phantom["image"], "--aorta-mask", phantom["mask"], "--output", str(out),
              "--meta-output", str(meta), "--case-id", "subject999"])
    assert p.returncode == 0, p.stderr
    result = json.load(open(out))
    assert result["case_id"] == "subject999"
    assert len(result["daughters"]) == 1
    assert "timings_s" in json.load(open(meta))


def test_corrupt_input_writes_empty_json_and_exits_zero(tmp_path):
    bad = tmp_path / "corrupt.nii"
    bad.write_bytes(b"this is not a nifti file" * 100)
    out = tmp_path / "sub" / "pred.json"  # output directory does not exist yet
    p = _run(["--image", str(bad), "--aorta-mask", str(bad), "--output", str(out)])
    assert p.returncode == 0
    assert "Traceback" in p.stderr
    result = json.load(open(out))
    assert result == {"case_id": "corrupt", "parent": {"instance_id": "aorta"}, "daughters": []}


def test_missing_file_exits_zero(tmp_path):
    out = tmp_path / "pred.json"
    p = _run(["--image", str(tmp_path / "nope.nii.gz"), "--aorta-mask", str(tmp_path / "nope_mask.nii.gz"), "--output", str(out)])
    assert p.returncode == 0
    assert json.load(open(out))["daughters"] == []


def test_write_json_is_atomic(tmp_path):
    out = tmp_path / "a" / "b.json"
    run.write_json(str(out), {"x": 1})
    assert json.load(open(out)) == {"x": 1}
    assert not os.path.exists(str(out) + ".tmp")


def test_infer_case_id():
    assert run.infer_case_id("data/subject010/orig10.nii") == "subject010"
    assert run.infer_case_id("/x/y/image.nii.gz") == "image"
    assert run.infer_case_id("case7.nii") == "case7"
