"""Patient-adaptive, spacing-aware image features for branch proposals."""

from __future__ import annotations

import warnings
from typing import Mapping, Sequence

import numpy as np
from scipy import ndimage
import SimpleITK as sitk

from .aorta import BloodModel


def blood_similarity(
    intensity_zyx: np.ndarray,
    model: BloodModel | None,
    *,
    floor_sigma: float = 8.0,
) -> np.ndarray:
    """Map intensity to [0, 1] similarity using a robust per-patient model."""
    values = np.asarray(intensity_zyx, dtype=np.float32)
    if model is None:
        return np.zeros_like(values, dtype=np.float32)
    sigma = max(model.robust_sigma, floor_sigma)
    z = (values - model.median) / sigma
    score = np.exp(-0.5 * z * z)
    # Preserve a broad plateau across the observed central blood distribution.
    score[(values >= model.low) & (values <= model.high)] = np.maximum(
        score[(values >= model.low) & (values <= model.high)], 0.75
    )
    score[~np.isfinite(values)] = 0
    return score.astype(np.float32)


def gradient_magnitude(
    image_zyx: np.ndarray,
    spacing_xyz: Sequence[float],
    *,
    sigma_mm: float = 0.8,
) -> np.ndarray:
    """Physical gradient magnitude after optional Gaussian smoothing."""
    spacing_zyx = np.asarray(tuple(spacing_xyz)[::-1], dtype=float)
    if spacing_zyx.shape != (3,) or np.any(spacing_zyx <= 0):
        raise ValueError("spacing_xyz must contain three positive values")
    image = np.asarray(image_zyx, dtype=np.float32)
    smooth = ndimage.gaussian_filter(image, sigma_mm / spacing_zyx) if sigma_mm > 0 else image
    gradients = np.gradient(smooth, *spacing_zyx)
    return np.sqrt(sum(component * component for component in gradients)).astype(np.float32)


def tubularity_from_eigenvalues(
    eigenvalues: np.ndarray,
    *,
    bright: bool = True,
    alpha: float = 0.5,
    beta: float = 0.5,
    gamma: float | None = None,
) -> np.ndarray:
    """Frangi-like 3-D line response from Hessian eigenvalues.

    Eigenvalues may have any order; they are sorted by absolute magnitude.
    For a bright tube the two larger-magnitude eigenvalues must be negative.
    """
    eig = np.asarray(eigenvalues, dtype=np.float32)
    if eig.shape[-1] != 3:
        raise ValueError("eigenvalues must end in an axis of length three")
    eig = np.take_along_axis(eig, np.argsort(np.abs(eig), axis=-1), axis=-1)
    l1, l2, l3 = np.moveaxis(eig, -1, 0)
    eps = np.finfo(np.float32).eps
    ra = np.abs(l2) / (np.abs(l3) + eps)
    rb = np.abs(l1) / np.sqrt(np.abs(l2 * l3) + eps)
    norm = np.sqrt(l1 * l1 + l2 * l2 + l3 * l3)
    scale = float(gamma) if gamma is not None else max(float(np.percentile(norm, 90)), eps)
    response = (
        (1.0 - np.exp(-(ra * ra) / (2 * alpha * alpha)))
        * np.exp(-(rb * rb) / (2 * beta * beta))
        * (1.0 - np.exp(-(norm * norm) / (2 * scale * scale)))
    )
    valid = (l2 < 0) & (l3 < 0) if bright else (l2 > 0) & (l3 > 0)
    return np.where(valid, response, 0).astype(np.float32)


def _sitk_objectness_scale(
    image_zyx: np.ndarray,
    spacing_xyz: Sequence[float],
    sigma_mm: float,
    *,
    bright: bool,
) -> np.ndarray:
    # SimpleITK exposes ITK's maintained ObjectnessMeasure filter but not the
    # HessianRecursiveGaussian wrapper on every build. Form the scale-normalized
    # physical Hessian locally with SciPy, then pass the symmetric tensor image
    # to SimpleITK's maintained objectness implementation.
    array = np.asarray(image_zyx, dtype=np.float32)
    spacing_zyx = np.asarray(tuple(spacing_xyz)[::-1], dtype=float)
    pixel_sigma = sigma_mm / spacing_zyx
    components = []
    for first, second in ((2, 2), (2, 1), (2, 0), (1, 1), (1, 0), (0, 0)):
        order = [0, 0, 0]
        order[first] += 1
        order[second] += 1
        derivative = ndimage.gaussian_filter(array, pixel_sigma, order=order)
        derivative *= sigma_mm**2 / (spacing_zyx[first] * spacing_zyx[second])
        components.append(derivative.astype(np.float32, copy=False))
    # ObjectnessMeasure's wrapped 3-D dispatch accepts vector float64 (not
    # vector float32) in current SimpleITK. The source, derivatives, running
    # maximum, and returned feature remain float32; only this per-scale tensor
    # bridge is promoted.
    hessian = sitk.GetImageFromArray(
        np.stack(components, axis=-1).astype(np.float64), isVector=True
    )
    hessian.SetSpacing(tuple(float(v) for v in spacing_xyz))
    measure = sitk.ObjectnessMeasureImageFilter()
    measure.SetObjectDimension(1)
    measure.SetBrightObject(bright)
    measure.SetScaleObjectnessMeasure(True)
    measure.SetAlpha(0.5)
    measure.SetBeta(0.5)
    measure.SetGamma(5.0)
    try:
        return sitk.GetArrayFromImage(measure.Execute(hessian)).astype(np.float32, copy=False)
    except RuntimeError as exc:
        # Some SimpleITK wheels expose ObjectnessMeasure but no wrapped
        # symmetric-tensor pixel type. Use the same Hessian eigenvalue measure
        # rather than making the required pipeline depend on a particular wheel.
        if "Pixel type" not in str(exc):
            raise
        xx, xy, xz, yy, yz, zz = components
        tensor = np.empty(array.shape + (3, 3), dtype=np.float32)
        tensor[..., 0, 0], tensor[..., 0, 1], tensor[..., 0, 2] = xx, xy, xz
        tensor[..., 1, 0], tensor[..., 1, 1], tensor[..., 1, 2] = xy, yy, yz
        tensor[..., 2, 0], tensor[..., 2, 1], tensor[..., 2, 2] = xz, yz, zz
        return tubularity_from_eigenvalues(np.linalg.eigvalsh(tensor), bright=bright)


def multiscale_hessian_objectness(
    image_zyx: np.ndarray,
    spacing_xyz: Sequence[float],
    scales_mm: Sequence[float] = (0.8, 1.2, 1.8, 2.5),
    *,
    crop_zyx: tuple[slice, slice, slice] | None = None,
    bright: bool = True,
) -> np.ndarray:
    """Compute maintained SimpleITK objectness one scale at a time.

    Only a float32 local crop is sent to SimpleITK and only the running maximum
    is retained, bounding memory independently of the number of scales.
    """
    image = np.asarray(image_zyx)
    if image.ndim != 3:
        raise ValueError("image_zyx must be three-dimensional")
    if not scales_mm or any(float(scale) <= 0 for scale in scales_mm):
        raise ValueError("scales_mm must contain positive values")
    crop = crop_zyx or (slice(None), slice(None), slice(None))
    local = np.asarray(image[crop], dtype=np.float32)
    result = np.zeros(local.shape, dtype=np.float32)
    for scale in scales_mm:
        result = np.maximum(
            result, _sitk_objectness_scale(local, spacing_xyz, float(scale), bright=bright)
        )
    if crop_zyx is None:
        return result
    full = np.zeros(image.shape, dtype=np.float32)
    full[crop] = result
    return full


def optional_advanced_filter(
    image_zyx: np.ndarray,
    spacing_xyz: Sequence[float],
    *,
    implementation: str = "itk",
) -> np.ndarray:
    """Try an optional advanced filter and return zeros when it is unavailable."""
    try:
        if implementation != "itk":
            raise ImportError(f"unsupported optional implementation: {implementation}")
        import itk  # type: ignore[import-not-found]  # noqa: F401
    except (ImportError, OSError) as exc:
        warnings.warn(f"optional advanced filter unavailable: {exc}", RuntimeWarning)
        return np.zeros_like(image_zyx, dtype=np.float32)
    # Keep the optional dependency out of the required path; callers can replace
    # this compatibility hook with a site-specific maintained ITK filter.
    warnings.warn("optional ITK filter is not configured; returning zeros", RuntimeWarning)
    return np.zeros_like(image_zyx, dtype=np.float32)


def compose_score(
    channels: Mapping[str, np.ndarray],
    weights: Mapping[str, float] | None = None,
    *,
    geometric_mean: bool = False,
) -> np.ndarray:
    """Compose named [0, 1] evidence channels into one bounded score."""
    if not channels:
        raise ValueError("at least one channel is required")
    arrays = {name: np.clip(np.asarray(value, np.float32), 0, 1) for name, value in channels.items()}
    shape = next(iter(arrays.values())).shape
    if any(value.shape != shape for value in arrays.values()):
        raise ValueError("all score channels must have the same shape")
    selected = {name: float((weights or {}).get(name, 1.0)) for name in arrays}
    if any(value < 0 or not np.isfinite(value) for value in selected.values()):
        raise ValueError("weights must be finite and non-negative")
    total = sum(selected.values())
    if total <= 0:
        return np.zeros(shape, dtype=np.float32)
    if geometric_mean:
        log_score = sum(selected[name] * np.log(np.maximum(value, 1e-6))
                        for name, value in arrays.items()) / total
        return np.exp(log_score).astype(np.float32)
    return (sum(selected[name] * value for name, value in arrays.items()) / total).astype(np.float32)
