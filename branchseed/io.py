"""Image and coordinate I/O helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import SimpleITK as sitk

from .models import Point3D


def read_image(path: str | Path, *, pixel_type: int = sitk.sitkUnknown) -> sitk.Image:
    """Read a medical image without changing its spatial metadata."""
    return sitk.ReadImage(str(path), pixel_type)


def write_image(image: sitk.Image, path: str | Path, *, compress: bool = True) -> None:
    """Write an image while retaining spacing, origin, and direction."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(destination), compress)


def image_to_array(image: sitk.Image, *, copy: bool = True) -> np.ndarray:
    """Return pixels in NumPy z-y-x order."""
    if image.GetDimension() != 3:
        raise ValueError("expected a 3D image")
    getter = sitk.GetArrayFromImage if copy else sitk.GetArrayViewFromImage
    return np.asarray(getter(image))


def array_to_image(array_zyx: np.ndarray, reference: sitk.Image) -> sitk.Image:
    """Create an image from a z-y-x array and copy reference geometry."""
    array = np.asarray(array_zyx)
    if array.ndim != 3:
        raise ValueError("array_zyx must be three-dimensional")
    if tuple(reversed(array.shape)) != reference.GetSize():
        raise ValueError(
            f"array shape {array.shape} does not match reference size {reference.GetSize()}"
        )
    image = sitk.GetImageFromArray(array)
    image.CopyInformation(reference)
    return image


def numpy_zyx_to_sitk_xyz(index_zyx: Sequence[float]) -> Point3D:
    """Reverse a NumPy z-y-x index into SimpleITK x-y-z order."""
    if len(index_zyx) != 3:
        raise ValueError("index_zyx must contain exactly three values")
    z, y, x = index_zyx
    return float(x), float(y), float(z)


def sitk_xyz_to_numpy_zyx(index_xyz: Sequence[float]) -> Point3D:
    """Reverse a SimpleITK x-y-z index into NumPy z-y-x order."""
    if len(index_xyz) != 3:
        raise ValueError("index_xyz must contain exactly three values")
    x, y, z = index_xyz
    return float(z), float(y), float(x)


def numpy_index_to_physical_point(
    image: sitk.Image, index_zyx: Sequence[float]
) -> Point3D:
    """Convert a continuous NumPy z-y-x index to physical x-y-z millimetres."""
    index_xyz = numpy_zyx_to_sitk_xyz(index_zyx)
    return tuple(
        float(value) for value in image.TransformContinuousIndexToPhysicalPoint(index_xyz)
    )  # type: ignore[return-value]


def physical_point_to_numpy_index(
    image: sitk.Image, point_xyz: Sequence[float]
) -> Point3D:
    """Convert a physical x-y-z point to a continuous NumPy z-y-x index."""
    if len(point_xyz) != 3:
        raise ValueError("point_xyz must contain exactly three values")
    index_xyz = image.TransformPhysicalPointToContinuousIndex(
        tuple(float(value) for value in point_xyz)
    )
    return sitk_xyz_to_numpy_zyx(index_xyz)
