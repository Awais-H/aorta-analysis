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
    assert "3:00" in html  # the phantom branch sits at 3 o'clock
    assert os.path.getsize(files["clock_png"]) > 1000


def test_report_with_no_daughters_renders(cand, tmp_path):
    fr = frame_mod.build(cand)
    files = report.write_all(cand, fake_result("empty"), fr, str(tmp_path / "rep"), "empty")
    assert os.path.exists(files["png"]) and os.path.exists(files["html"])
    assert "<b>0 daughters</b>" in open(files["html"], encoding="utf-8").read()


def test_branch_rows_sorted_by_height(cand, phantom):
    fr = frame_mod.build(cand)
    low = phantom["ostium_mm"] - [0, 0, 10.0]
    result = fake_result("s", [fake_daughter(1, low), fake_daughter(2, phantom["ostium_mm"])])
    rows = report.branch_rows(result, fr)
    assert [r["instance_id"] for r in rows] == ["branch_002", "branch_001"]
    assert rows[0]["height_mm"] < rows[1]["height_mm"]


def test_flags_and_measurements_reach_the_table(cand, phantom, tmp_path):
    fr = frame_mod.build(cand)
    result = fake_result("subject000", [fake_daughter(1, phantom["ostium_mm"], phantom["direction"])])
    meta = {"label_to_branch": {"1": "branch_001"}, "flags": {"1": ["near_cut_face", "borderline_diameter"]},
            "measurements": {"1": {"origin_diameter_mm": 2.3, "departure_mm": 7.1}}, "timings_s": {"total": 1.5}}
    files = report.write_all(cand, result, fr, str(tmp_path / "rep"), "subject000", meta=meta)
    html = open(files["html"], encoding="utf-8").read()
    assert "near_cut_face, borderline_diameter" in html and "2.3" in html
    assert "clock map" in html and "Spacing along the centreline" in html
    rows = report.branch_rows(result, fr, meta)
    assert rows[0]["flags"] == ["near_cut_face", "borderline_diameter"] and rows[0]["origin_diameter_mm"] == 2.3


def test_plotly_view_is_self_contained(cand, phantom):
    import pytest
    pytest.importorskip("plotly")
    fr = frame_mod.build(cand)
    result = fake_result("subject000", [fake_daughter(1, phantom["ostium_mm"], phantom["direction"])])
    div = report.plotly_3d_div(cand, result, fr)
    assert div is not None and "plotly" in div.lower() and "cdn" not in div[:2000].lower()
    assert "view3d" in div and "branch_001" not in div[:100]  # the div is markup, the data is inside
    assert len(div) > 100000  # the JavaScript library is inlined, so the page opens offline


def test_clock_label_and_unrolled_axis():
    assert report.clock_label(12.0) == "12:00" and report.clock_label(0.0) == "12:00"
    assert report.clock_label(2.5) == "2:30" and report.clock_label(11.99) == "11:59"
    assert report.clock_x(6.0) == 0.0 and report.clock_x(12.0) == 6.0 and report.clock_x(3.0) == 9.0 and report.clock_x(9.0) == 3.0
