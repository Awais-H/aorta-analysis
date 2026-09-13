"""Headless orthogonal image visualization."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk

from .io import image_to_array
from .models import DaughterPrediction, Point3D

Arrow3D = tuple[Point3D, Point3D]


def _continuous_index(image: sitk.Image, point_xyz: Sequence[float]) -> np.ndarray:
    return np.asarray(
        image.TransformPhysicalPointToContinuousIndex(
            tuple(float(value) for value in point_xyz)
        ),
        dtype=float,
    )


def _draw_geometry(
    axis: plt.Axes,
    image: sitk.Image,
    horizontal_axis: int,
    vertical_axis: int,
    slice_axis: int,
    slice_index: float,
    points_xyz: Sequence[Point3D],
    paths_xyz: Sequence[Sequence[Point3D]],
    arrows_xyz: Sequence[Arrow3D],
) -> None:
    tolerance = 0.75
    for point in points_xyz:
        index = _continuous_index(image, point)
        if abs(index[slice_axis] - slice_index) <= tolerance:
            axis.scatter(index[horizontal_axis], index[vertical_axis], c="red", s=22)

    for path in paths_xyz:
        if not path:
            continue
        indices = np.asarray([_continuous_index(image, point) for point in path])
        axis.plot(
            indices[:, horizontal_axis],
            indices[:, vertical_axis],
            color="cyan",
            linewidth=1.5,
        )

    for start, vector in arrows_xyz:
        start_index = _continuous_index(image, start)
        end_physical = np.asarray(start, dtype=float) + np.asarray(vector, dtype=float)
        end_index = _continuous_index(image, end_physical)
        delta = end_index - start_index
        axis.quiver(
            start_index[horizontal_axis],
            start_index[vertical_axis],
            delta[horizontal_axis],
            delta[vertical_axis],
            color="yellow",
            angles="xy",
            scale_units="xy",
            scale=1,
        )


def save_orthogonal_overlays(
    image: sitk.Image,
    output_path: str | Path,
    *,
    overlay: sitk.Image | None = None,
    center_xyz: Point3D | None = None,
    points_xyz: Sequence[Point3D] = (),
    paths_xyz: Sequence[Sequence[Point3D]] = (),
    arrows_xyz: Sequence[Arrow3D] = (),
    predictions: Sequence[DaughterPrediction] = (),
    dpi: int = 150,
) -> Path:
    """Save axial, coronal, and sagittal views with optional geometry overlays."""
    volume = image_to_array(image)
    if overlay is not None:
        if (
            overlay.GetSize() != image.GetSize()
            or overlay.GetSpacing() != image.GetSpacing()
            or overlay.GetOrigin() != image.GetOrigin()
            or overlay.GetDirection() != image.GetDirection()
        ):
            raise ValueError("overlay geometry must match image geometry")
        overlay_volume = image_to_array(overlay)
    else:
        overlay_volume = None

    if center_xyz is None:
        center_index = (np.asarray(image.GetSize(), dtype=float) - 1.0) / 2.0
    else:
        center_index = _continuous_index(image, center_xyz)
    center_index = np.clip(center_index, 0, np.asarray(image.GetSize()) - 1)
    x, y, z = np.rint(center_index).astype(int)

    all_points = list(points_xyz)
    all_paths = list(paths_xyz)
    all_arrows = list(arrows_xyz)
    for prediction in predictions:
        all_points.append(prediction.ostium_xyz)
        if prediction.candidate_path_xyz:
            all_paths.append(prediction.candidate_path_xyz)
        if prediction.direction_xyz is not None:
            all_arrows.append((prediction.ostium_xyz, prediction.direction_xyz))

    views = (
        ("Axial", volume[z, :, :], None if overlay_volume is None else overlay_volume[z, :, :], 0, 1, 2, z),
        ("Coronal", volume[:, y, :], None if overlay_volume is None else overlay_volume[:, y, :], 0, 2, 1, y),
        ("Sagittal", volume[:, :, x], None if overlay_volume is None else overlay_volume[:, :, x], 1, 2, 0, x),
    )
    figure, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for axis, (title, pixels, mask, horizontal, vertical, sliced, selected) in zip(
        axes, views, strict=True
    ):
        axis.imshow(pixels, cmap="gray", origin="lower")
        if mask is not None:
            axis.imshow(np.ma.masked_where(mask == 0, mask), cmap="autumn", alpha=0.4, origin="lower")
        _draw_geometry(
            axis,
            image,
            horizontal,
            vertical,
            sliced,
            selected,
            all_points,
            all_paths,
            all_arrows,
        )
        axis.set_title(title)
        axis.set_axis_off()

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return destination
