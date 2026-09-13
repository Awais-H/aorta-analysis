"""End-cap exclusion.

The flat superior and inferior faces created by cropping are not branch
origins, and a naive detector fires on them immediately: both caps are
full-calibre discs of bright lumen surrounded by background, which is exactly
what an ostium looks like from the outside.

This is the most common silent failure in the pipeline. It costs precision on
every case and nothing else in the output looks wrong when it breaks.
"""

from __future__ import annotations

import numpy as np

from branchseed import io as bio
from branchseed.calibrate import calibrate, euclidean_distance_mm
from branchseed.detect import detect_candidates
from branchseed.geometry import build_geometry
from branchseed.pipeline import process_images
from branchseed.roi import build_roi
from branchseed.wallmap import build_wall_map
from branchseed.types import Flags


def _stages(phantom, cfg):
    image, mask, flags = bio.load_case_from_images(phantom.image, phantom.mask)
    image_roi, mask_roi, grid = build_roi(image, mask, cfg, flags)
    edt = euclidean_distance_mm(mask_roi, grid)
    calibration = calibrate(image_roi, mask_roi, edt, grid, cfg, flags)
    geometry = build_geometry(mask_roi, edt, grid, cfg, flags)
    wall_map = build_wall_map(image_roi, geometry, calibration, grid, cfg)
    return grid, geometry, calibration, wall_map


def test_terminal_discs_are_excluded(standard_phantom, cfg):
    """Surface voxels on the two flat faces must not be eligible wall."""
    grid, geometry, _, _ = _stages(standard_phantom, cfg)
    excluded = geometry.surface & ~geometry.eligible_wall
    assert excluded.any(), "nothing was excluded; the end-cap tests are not firing"

    voxels = np.argwhere(geometry.surface)
    xyz_mm = voxels[:, ::-1].astype(np.float64) * grid.roi_spacing_mm
    keep = geometry.eligible_wall[voxels[:, 0], voxels[:, 1], voxels[:, 2]]

    for endpoint, outward in ((geometry.centreline[0], -geometry.tangent[0]),
                              (geometry.centreline[-1], geometry.tangent[-1])):
        offset = xyz_mm - endpoint * grid.roi_spacing_mm
        along = offset @ outward
        radial = np.linalg.norm(offset - along[:, None] * outward, axis=1)
        on_cap = (np.abs(along) <= 1.0) & (radial <= 6.0)
        assert on_cap.any(), "the phantom should have a flat cap here"
        assert not keep[on_cap].any(), "cap voxels survived as eligible wall"


def test_wall_map_is_invalid_at_both_ends(standard_phantom, cfg):
    _, geometry, _, wall_map = _stages(standard_phantom, cfg)
    invalid = 1.0 - wall_map.valid.mean(axis=1)
    assert invalid[0] > 0.9
    assert invalid[-1] > 0.9
    # ...and the middle of the vessel is almost entirely valid.
    middle = slice(len(invalid) // 3, 2 * len(invalid) // 3)
    assert invalid[middle].max() < 0.1


def test_no_candidates_near_the_cut_faces(standard_phantom, cfg):
    grid, geometry, calibration, wall_map = _stages(standard_phantom, cfg)
    candidates = detect_candidates(wall_map, geometry, calibration, cfg)
    total = float(geometry.arc_length[-1])
    for candidate in candidates:
        arc = float(geometry.arc_length[candidate.peak_ij[0]])
        assert min(arc, total - arc) > 5.0, f"candidate at {arc:.1f} mm of {total:.1f} mm"


def test_overall_invalid_fraction_is_small(standard_phantom, cfg):
    """A high invalid fraction means the frame or the ray casting is wrong,
    not that the end caps are being handled well."""
    _, _, _, wall_map = _stages(standard_phantom, cfg)
    assert (1.0 - wall_map.valid.mean()) < 0.20


def test_a_real_origin_near_a_cut_face_survives(near_cut_face_phantom, cfg):
    """Exclusion must be targeted, not a blanket margin: an origin 14 mm from
    the inferior face is a real branch and must still be reported."""
    payload = process_images(near_cut_face_phantom.image, near_cut_face_phantom.mask,
                             cfg, "near_cut_face").payload
    assert len(payload["daughters"]) == 1, payload["daughters"]
    truth = near_cut_face_phantom.truth[0]["ostium_xyz_mm"]
    error = np.linalg.norm(np.asarray(payload["daughters"][0]["ostium_xyz_mm"]) - truth)
    assert error < 2.0, error
