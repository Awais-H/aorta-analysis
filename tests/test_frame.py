import numpy as np

import frame as frame_mod


def test_centreline_contract(cand, phantom):
    fr = frame_mod.build(cand)
    k = len(fr.centreline_mm)
    assert fr.centreline_mm.shape == (k, 3) and fr.tangents.shape == (k, 3) and fr.arc_mm.shape == (k,)
    assert fr.endpoints_mm.shape == (2, 3)
    assert np.allclose(np.linalg.norm(fr.tangents, axis=1), 1.0)
    # superior first: z decreases along the line in LPS
    assert fr.centreline_mm[0, 2] > fr.centreline_mm[-1, 2]
    assert np.allclose(fr.endpoints_mm[0], fr.centreline_mm[0])
    # a straight vertical cylinder: length equals the z extent, tortuosity 1, centred on the axis
    z0, z1 = phantom["z_range"]
    assert abs(fr.length_mm - (z1 - z0)) < 1.0
    assert abs(fr.tortuosity - 1.0) < 1e-3
    cx, cy = phantom["centre_xy"]
    assert np.allclose(fr.centreline_mm[:, 0], cx, atol=0.5)
    assert np.allclose(fr.centreline_mm[:, 1], cy, atol=0.5)


def test_height_and_clock_convention(cand, phantom):
    fr = frame_mod.build(cand)
    cx, cy = phantom["centre_xy"]
    z_top = fr.centreline_mm[0, 2]
    z_mid = fr.centreline_mm[len(fr.centreline_mm) // 2, 2]
    h, c = fr.height_clock([cx, cy - 10.0, z_mid])  # anterior (-y)
    assert abs(c - 0.0) < 0.05 or abs(c - 12.0) < 0.05
    assert abs(h - (z_top - z_mid)) < 1.0
    _, c = fr.height_clock([cx + 10.0, cy, z_mid])  # patient's left (+x)
    assert abs(c - 3.0) < 0.05
    _, c = fr.height_clock([cx, cy + 10.0, z_mid])  # posterior
    assert abs(c - 6.0) < 0.05
    _, c = fr.height_clock([cx - 10.0, cy, z_mid])  # patient's right
    assert abs(c - 9.0) < 0.05
    h, _ = fr.height_clock([cx, cy - 10.0, z_top])
    assert abs(h) < 1e-6


def test_phantom_branch_is_at_three_oclock(cand, phantom):
    fr = frame_mod.build(cand)
    _, c = fr.height_clock(phantom["ostium_mm"])
    assert abs(c - 3.0) < 0.1
