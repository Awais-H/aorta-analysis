"""The rotation-minimising frame and the circular wrap of the map.

Both are failure modes that produce a plausible-looking map with a
discontinuity or a duplicated branch, rather than an obvious error.
"""

from __future__ import annotations

import numpy as np

from branchseed import io as bio
from branchseed.calibrate import calibrate, euclidean_distance_mm
from branchseed.detect import detect_candidates
from branchseed.frames import align_to_reference, rotation_minimising_frame
from branchseed.geometry import build_geometry
from branchseed.roi import build_roi
from branchseed.wallmap import build_wall_map


def _helix(n=500, turns=3.0, pitch=0.3):
    t = np.linspace(0.0, turns * 2 * np.pi, n)
    points = np.stack([np.cos(t), np.sin(t), pitch * t], axis=1)
    tangent = np.stack([-np.sin(t), np.cos(t), pitch * np.ones_like(t)], axis=1)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    return points, tangent


def test_frame_is_orthonormal_and_right_handed():
    points, tangent = _helix()
    u, v = rotation_minimising_frame(points, tangent)
    assert np.abs((u * tangent).sum(axis=1)).max() < 1e-9
    assert np.abs((v * tangent).sum(axis=1)).max() < 1e-9
    assert np.abs((u * v).sum(axis=1)).max() < 1e-9
    np.testing.assert_allclose(np.linalg.norm(u, axis=1), 1.0, atol=1e-9)
    np.testing.assert_allclose(np.cross(u, v), tangent, atol=1e-9)


def test_frame_has_no_net_twist():
    """The frame must rotate no faster than the curve does. A fixed-vector
    frame would spin once per turn of the helix."""
    points, tangent = _helix()
    u, _ = rotation_minimising_frame(points, tangent)
    turn = np.arccos(np.clip((tangent[:-1] * tangent[1:]).sum(axis=1), -1, 1))
    twist = np.arccos(np.clip((u[:-1] * u[1:]).sum(axis=1), -1, 1))
    assert twist.max() <= turn.max() + 1e-9


def test_frame_survives_a_tangent_through_the_global_up_vector():
    """A fixed global vector projected into the normal plane degenerates here,
    which is guaranteed to happen somewhere on an arch."""
    t = np.linspace(-1.0, 1.0, 401)
    tangent = np.stack([np.sin(t * np.pi / 2), np.zeros_like(t), np.cos(t * np.pi / 2)], axis=1)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    points = np.cumsum(tangent * 0.05, axis=0)
    u, v = rotation_minimising_frame(points, tangent)
    assert np.isfinite(u).all() and np.isfinite(v).all()
    np.testing.assert_allclose(np.linalg.norm(u, axis=1), 1.0, atol=1e-9)
    step = np.arccos(np.clip((u[:-1] * u[1:]).sum(axis=1), -1, 1))
    assert np.degrees(step).max() < 5.0, "the frame jumped, which would tear the map"


def test_global_rotation_puts_theta_zero_on_the_reference():
    points, tangent = _helix()
    u, v = rotation_minimising_frame(points, tangent)
    reference = np.array([0.0, -1.0, 0.0])
    u2, v2 = align_to_reference(u, v, tangent, reference, at=250)
    projected = reference - float(reference @ tangent[250]) * tangent[250]
    projected /= np.linalg.norm(projected)
    assert float(projected @ u2[250]) > 1.0 - 1e-9
    assert abs(float(projected @ v2[250])) < 1e-6
    np.testing.assert_allclose(np.cross(u2, v2), tangent, atol=1e-9)


def test_branch_at_theta_zero_is_one_candidate(standard_phantom, cfg):
    """The map is cyclic in theta. Without circular padding a branch at 12
    o'clock splits into two half-regions at either edge and is reported twice.
    The standard phantom has a branch at exactly clock 0."""
    image, mask, flags = bio.load_case_from_images(standard_phantom.image, standard_phantom.mask)
    image_roi, mask_roi, grid = build_roi(image, mask, cfg, flags)
    edt = euclidean_distance_mm(mask_roi, grid)
    calibration = calibrate(image_roi, mask_roi, edt, grid, cfg, flags)
    geometry = build_geometry(mask_roi, edt, grid, cfg, flags)
    wall_map = build_wall_map(image_roi, geometry, calibration, grid, cfg)
    candidates = detect_candidates(wall_map, geometry, calibration, cfg)

    n_bins = wall_map.intensity.shape[1]
    wrapping = [
        c for c in candidates
        if min(c.peak_ij[1], n_bins - c.peak_ij[1]) < n_bins // 8
    ]
    assert len(wrapping) == 1, [c.peak_ij for c in wrapping]
    # ...and its region really does straddle the seam.
    columns = set(wrapping[0].map_region[:, 1].tolist())
    assert any(c < 4 for c in columns) and any(c > n_bins - 5 for c in columns)
