"""Small deterministic 3-D phantoms for integration and regression tests."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy import ndimage
import SimpleITK as sitk

from .io import write_image
from .models import CasePrediction, DaughterPrediction


@dataclass(frozen=True, slots=True)
class Phantom:
    name: str
    image: sitk.Image
    aorta_mask: sitk.Image
    reference: CasePrediction


def _line_mask(shape: tuple[int, ...], points: Sequence[Sequence[float]], radius: int = 2) -> np.ndarray:
    mask = np.zeros(shape, bool)
    samples: list[np.ndarray] = []
    for first, second in zip(points[:-1], points[1:]):
        count = max(2, int(np.linalg.norm(np.asarray(second) - first) * 2))
        samples.extend(np.linspace(first, second, count))
    if not samples:
        samples = [np.asarray(points[0])]
    indices = np.clip(np.rint(samples).astype(int), 0, np.asarray(shape) - 1)
    mask[tuple(indices.T)] = True
    return ndimage.binary_dilation(mask, iterations=radius)


def make_phantom(
    name: str = "straight_multiple",
    *,
    shape: tuple[int, int, int] = (40, 56, 56),
) -> Phantom:
    """Create one named scenario with physical metadata and reference daughters."""
    supported = {
        "straight_multiple", "curved", "common_trunk", "nearby_ostia",
        "anisotropic_rotated", "low_proximal_contrast", "crop_ends", "no_daughter",
    }
    if name not in supported:
        raise ValueError(f"unknown phantom {name!r}; choose from {sorted(supported)}")
    z, y, x = np.indices(shape)
    centre_x = np.full(shape[0], 24.0)
    if name == "curved":
        centre_x = 24.0 + 4.0 * np.sin(np.linspace(-1.0, 1.0, shape[0]) * np.pi / 2)
    aorta = (y - 28.0) ** 2 + (x - centre_x[:, None, None]) ** 2 <= 7.0**2
    if name != "crop_ends":
        aorta[:3] = False
        aorta[-3:] = False
    image = np.full(shape, 20.0, np.float32)
    image[aorta] = 220.0
    branch_paths: list[list[tuple[float, float, float]]] = []
    if name not in {"no_daughter"}:
        levels = [14, 26] if name in {"straight_multiple", "anisotropic_rotated"} else [20]
        if name == "nearby_ostia":
            levels = [19, 23]
        for level in levels:
            start_x = float(centre_x[level] + 6)
            branch_paths.append([(level, 28, start_x), (level, 28, 34), (level, 28, 46)])
    if name == "common_trunk":
        branch_paths = [
            [(20, 28, 30), (20, 28, 38), (17, 27, 47)],
            [(20, 28, 30), (20, 28, 38), (23, 29, 47)],
        ]
    reference_paths = branch_paths
    if name == "common_trunk":
        # Two distal branches share one direct aortic opening: the challenge
        # target is the single proximal daughter trunk, not both descendants.
        reference_paths = [[(20, 28, 30), (20, 28, 38), (20, 28, 46)]]
    branch_masks = [_line_mask(shape, path, 2) & ~aorta for path in branch_paths]
    for vessel in branch_masks:
        image[vessel] = 220.0
    if name == "low_proximal_contrast" and branch_masks:
        # Retain a distal bright tube but weaken the first several millimetres.
        image[20, 26:31, 30:37] = 75.0
    spacing = (1.0, 1.0, 1.2)
    origin = (5.0, -8.0, 12.0)
    direction = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    if name == "anisotropic_rotated":
        spacing = (0.8, 1.3, 2.0)
        direction = (0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    image_itk = sitk.GetImageFromArray(image)
    mask_itk = sitk.GetImageFromArray(aorta.astype(np.uint8))
    for item in (image_itk, mask_itk):
        item.SetSpacing(spacing)
        item.SetOrigin(origin)
        item.SetDirection(direction)

    def physical(point_zyx: Sequence[float]) -> tuple[float, float, float]:
        return tuple(float(v) for v in image_itk.TransformContinuousIndexToPhysicalPoint(
            tuple(float(v) for v in point_zyx[::-1])
        ))

    daughters: list[DaughterPrediction] = []
    for path in reference_paths:
        ostium = physical(path[0])
        seed = physical(np.asarray(path[0]) + np.asarray((0, 0, 5)))
        vector = np.asarray(physical(path[-1])) - np.asarray(ostium)
        vector /= np.linalg.norm(vector)
        daughters.append(DaughterPrediction(
            ostium_xyz=ostium, seed_xyz=seed,
            direction_xyz=tuple(float(v) for v in vector), radius_mm=2.0,
            candidate_path_xyz=tuple(physical(point) for point in path),
        ))
    return Phantom(name, image_itk, mask_itk, CasePrediction(name, tuple(daughters)))


def write_phantom(phantom: Phantom, output_dir: str | Path) -> tuple[Path, Path, Path]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    image_path = destination / f"{phantom.name}.nii.gz"
    mask_path = destination / f"{phantom.name}_aorta.nii.gz"
    reference_path = destination / f"{phantom.name}_reference.json"
    write_image(phantom.image, image_path)
    write_image(phantom.aorta_mask, mask_path)
    phantom.reference.save_json(reference_path)
    return image_path, mask_path, reference_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scenario", choices=[
        "straight_multiple", "curved", "common_trunk", "nearby_ostia",
        "anisotropic_rotated", "low_proximal_contrast", "crop_ends", "no_daughter",
    ])
    arguments = parser.parse_args(argv)
    names = [arguments.scenario] if arguments.scenario else [
        "straight_multiple", "curved", "common_trunk", "nearby_ostia",
        "anisotropic_rotated", "low_proximal_contrast", "crop_ends", "no_daughter",
    ]
    for name in names:
        write_phantom(make_phantom(name), arguments.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
