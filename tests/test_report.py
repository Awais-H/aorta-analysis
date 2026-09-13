import os

import frame as frame_mod
import report
from conftest import fake_daughter, fake_result


def test_write_all_produces_png_and_html(cand, phantom, tmp_path):
    fr = frame_mod.build(cand)
    result = fake_result("subject000", [fake_daughter(1, phantom["ostium_mm"], phantom["direction"])])
    files = report.write_all(cand, result, fr, str(tmp_path / "rep"), "subject000")
    assert os.path.getsize(files["png"]) > 1000
    html = open(files["html"], encoding="utf-8").read()
    assert "branch_001" in html and "subject000" in html
    assert "3.0" in html  # the phantom branch sits at 3 o'clock


def test_report_with_no_daughters_renders(cand, tmp_path):
    fr = frame_mod.build(cand)
    files = report.write_all(cand, fake_result("empty"), fr, str(tmp_path / "rep"), "empty")
    assert os.path.exists(files["png"]) and os.path.exists(files["html"])
    assert "Daughters: 0" in open(files["html"], encoding="utf-8").read()


def test_branch_rows_sorted_by_height(cand, phantom):
    fr = frame_mod.build(cand)
    low = phantom["ostium_mm"] - [0, 0, 10.0]
    result = fake_result("s", [fake_daughter(1, low), fake_daughter(2, phantom["ostium_mm"])])
    rows = report.branch_rows(result, fr)
    assert [r["instance_id"] for r in rows] == ["branch_002", "branch_001"]
    assert rows[0]["height_mm"] < rows[1]["height_mm"]
