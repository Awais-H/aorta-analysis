"""Data contracts shared between pipeline stages.

Conventions enforced everywhere in this package:

* every numpy volume is indexed ``[z, y, x]``;
* every *point* is ``(x, y, z)`` in ROI continuous-index space;
* every emitted physical coordinate goes through :meth:`CaseGrid.to_physical`.

Stages take and return these objects and nothing else, which is what lets the
test suite drive any stage in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


class Flags:
    """Ordered, de-duplicated set of case- or candidate-level flag strings."""

    def __init__(self, initial: Sequence[str] = ()) -> None:
        self._items: List[str] = []
        for item in initial:
            self.add(item)

    def add(self, flag: str) -> None:
        if flag not in self._items:
            self._items.append(flag)

    def __contains__(self, flag: object) -> bool:
        return flag in self._items

    def __iter__(self):
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __repr__(self) -> str:
        return f"Flags({self._items!r})"

    def as_list(self) -> List[str]:
        return list(self._items)


@dataclass(frozen=True)
class CaseGrid:
    """Maps ROI continuous indices to SimpleITK physical millimetres.

    ``roi_origin_mm``/``roi_direction``/``roi_spacing_mm`` fully describe the
    ROI grid, so :meth:`to_physical` is pure arithmetic and vectorises. The
    original image is retained for validation and for the round-trip assertion
    in :func:`branchseed.io.assert_coordinate_roundtrip`.
    """

    sitk_ref: Any                      # the original SimpleITK image
    crop_origin_idx: Tuple[int, int, int]   # (i, j, k) offset in the ORIGINAL grid, xyz
    roi_origin_mm: np.ndarray          # (3,) physical origin of ROI voxel (0,0,0)
    roi_direction: np.ndarray          # (3, 3) direction cosines, columns are axes
    roi_spacing_mm: np.ndarray         # (3,) xyz
    orig_spacing_mm: np.ndarray        # (3,) xyz
    roi_shape: Tuple[int, int, int]    # numpy order (nz, ny, nx)
    resampled: bool = False

    def to_physical(self, pt_xyz: np.ndarray) -> np.ndarray:
        """ROI continuous index ``(..., 3)`` in xyz -> physical mm ``(..., 3)``.

        The single exit point from index space. Everything that emits a
        coordinate calls this and nothing else.
        """
        pts = np.asarray(pt_xyz, dtype=np.float64)
        if pts.shape[-1] != 3:
            raise ValueError(f"expected trailing axis of size 3 (x, y, z), got {pts.shape}")
        scaled = pts * self.roi_spacing_mm
        return scaled @ self.roi_direction.T + self.roi_origin_mm

    def to_physical_direction(self, vec_xyz: np.ndarray) -> np.ndarray:
        """Rotate a *direction* from ROI index space into physical space.

        Not normalised: anisotropic spacing legitimately changes the direction,
        which is exactly why index-space directions must never be emitted.
        """
        vecs = np.asarray(vec_xyz, dtype=np.float64)
        return (vecs * self.roi_spacing_mm) @ self.roi_direction.T

    def to_original_index(self, pt_xyz: np.ndarray) -> np.ndarray:
        """ROI continuous index -> original-image continuous index (xyz)."""
        pts = np.asarray(pt_xyz, dtype=np.float64)
        ratio = self.roi_spacing_mm / self.orig_spacing_mm
        return pts * ratio + np.asarray(self.crop_origin_idx, dtype=np.float64)

    def physical_to_original_index(self, pt_mm: np.ndarray) -> np.ndarray:
        """Physical mm -> original-image continuous index (xyz).

        Equivalent to ``sitk_ref.TransformPhysicalPointToContinuousIndex``, but
        vectorised. Used only for the voxel-unit output variant; the mandated
        output never calls this.
        """
        return self.to_original_index(self.to_index(pt_mm))

    def physical_direction_to_original_index(self, vec_phys: np.ndarray) -> np.ndarray:
        """A physical unit direction, expressed as a displacement in the
        original image's continuous-index units.

        Not unit length in general: anisotropic spacing changes a direction's
        length under this map, which is exactly why the mandated
        ``direction_xyz`` is physical and this method exists only for the
        voxel-unit output variant, where it is renormalised for display.
        """
        return self.to_mm_direction(vec_phys) / self.orig_spacing_mm

    def to_index(self, pt_mm: np.ndarray) -> np.ndarray:
        """Physical mm -> ROI continuous index (xyz). Inverse of to_physical."""
        pts = np.asarray(pt_mm, dtype=np.float64) - self.roi_origin_mm
        return (pts @ self.roi_direction) / self.roi_spacing_mm

    def to_mm_direction(self, vec_phys: np.ndarray) -> np.ndarray:
        """Physical direction -> millimetre-scaled index space. Inverse of
        rotate_to_physical."""
        return np.asarray(vec_phys, dtype=np.float64) @ self.roi_direction

    def rotate_to_physical(self, vec_mm: np.ndarray) -> np.ndarray:
        """Rotate a vector already expressed in millimetre-scaled index space
        into physical space. Direction cosines are orthonormal, so this is a
        pure rotation and preserves length."""
        return np.asarray(vec_mm, dtype=np.float64) @ self.roi_direction.T

    def mm_to_index_delta(self, dir_mm: np.ndarray, distance_mm) -> np.ndarray:
        """A step of ``distance_mm`` along a unit mm-space direction, expressed
        as an offset in ROI continuous index units."""
        d = np.asarray(dir_mm, dtype=np.float64)
        dist = np.asarray(distance_mm, dtype=np.float64)
        return d * dist[..., None] / self.roi_spacing_mm if dist.ndim else \
            d * float(dist) / self.roi_spacing_mm

    def index_delta_to_mm(self, delta_idx: np.ndarray) -> np.ndarray:
        """An index-space offset expressed in millimetre-scaled index space."""
        return np.asarray(delta_idx, dtype=np.float64) * self.roi_spacing_mm

    @property
    def spacing_zyx(self) -> np.ndarray:
        """ROI spacing in numpy array order, for scipy's ``sampling`` kwarg."""
        return self.roi_spacing_mm[::-1].copy()

    @property
    def voxel_volume_mm3(self) -> float:
        return float(np.prod(self.roi_spacing_mm))


@dataclass(frozen=True)
class Calibration:
    mu_ao: float
    sigma_ao: float
    t_lumen: float
    t_calcium: float
    t_bg: float
    n_interior_voxels: int
    degenerate: bool = False


@dataclass(frozen=True)
class AortaGeometry:
    centreline: np.ndarray      # (N, 3) ROI continuous index, xyz, 1 mm arc spacing
    arc_length: np.ndarray      # (N,) cumulative mm
    tangent: np.ndarray         # (N, 3) unit, xyz index space
    frame_u: np.ndarray         # (N, 3) rotation-minimising frame, theta = 0
    frame_v: np.ndarray         # (N, 3)
    radius_mm: np.ndarray       # (N,) inscribed radius from the EDT
    mask: np.ndarray            # (nz, ny, nx) bool
    surface: np.ndarray         # (nz, ny, nx) bool
    eligible_wall: np.ndarray   # (nz, ny, nx) bool
    edt_mm: np.ndarray          # (nz, ny, nx) float32, positive inside
    signed_edt_mm: np.ndarray   # (nz, ny, nx) float32, negative outside
    normals: np.ndarray         # (3, nz, ny, nx) float32, outward, xyz components

    @property
    def n_samples(self) -> int:
        return int(self.centreline.shape[0])


@dataclass(frozen=True)
class WallMap:
    intensity: np.ndarray       # (N, B) float32, NaN where invalid
    wall_radius_mm: np.ndarray  # (N, B) float32
    wall_point: np.ndarray      # (N, B, 3) float32, ROI continuous index xyz
    valid: np.ndarray           # (N, B) bool
    theta: np.ndarray           # (B,) radians, 0 = anterior


@dataclass
class Candidate:
    """Mutable through stages 4-7, then serialised.

    Rejected candidates stay in the list with ``rejected_by`` set: the sweep
    harness attributes lost recall to a named rule, which is the single most
    actionable diagnostic in the project.
    """

    id: int
    map_region: np.ndarray                    # (P, 2) map pixel indices (i, j)
    peak_ij: Tuple[int, int]
    peak_value: float
    patch_points: np.ndarray                  # (P, 3) ROI continuous index xyz
    patch_weights: np.ndarray                 # (P,)
    ostium_xyz_roi: Optional[np.ndarray] = None
    path: Optional[np.ndarray] = None         # (M, 3) ROI continuous index xyz
    path_radius_mm: Optional[np.ndarray] = None
    path_distance_mm: Optional[np.ndarray] = None
    path_length_mm: float = 0.0      # honest proximal length, truncated at a split
    reached_mm: float = 0.0          # how far the lumen was followable in total
    stop_reason: str = "none"
    eligible: bool = False
    direction: Optional[np.ndarray] = None    # (3,) unit, PHYSICAL space
    seed_xyz_roi: Optional[np.ndarray] = None
    radius_mm: Optional[float] = None
    confidence: float = 0.0
    arc_length_mm: float = 0.0
    clock_position: str = ""
    parent_radius_mm: float = 0.0
    lumen_points: Optional[np.ndarray] = None  # (K, 3) traced lumen, ROI index
    lumen_distance_mm: Optional[np.ndarray] = None  # (K,) geodesic distance per point
    flags: Flags = field(default_factory=Flags)
    rejected_by: Optional[str] = None
    debug: Dict[str, Any] = field(default_factory=dict)
    instance_id: str = ""

    @property
    def accepted(self) -> bool:
        return self.rejected_by is None

    def reject(self, reason: str) -> None:
        if self.rejected_by is None:
            self.rejected_by = reason
