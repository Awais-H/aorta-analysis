"""Complete bounded-memory CPU Branchseed prediction pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import json
from pathlib import Path
import resource
import sys
import time
from typing import Any, Callable, TypeVar

import numpy as np
import psutil
import SimpleITK as sitk

from .aorta import analyze_aorta
from .candidates import Candidate, generate_candidates
from .confidence import estimate_perturbation_confidence
from .config import PipelineConfig
from .features import blood_similarity, multiscale_hessian_objectness
from .instances import derive_instances
from .io import image_to_array, read_image
from .measurements import instance_to_prediction
from .models import CasePrediction, DaughterPrediction
from .tracking import track_local_segmentation, track_minimal_path

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class StageProfile:
    seconds: float
    rss_mb: float


@dataclass(frozen=True, slots=True)
class PipelineResult:
    prediction: CasePrediction
    profiles: dict[str, StageProfile]
    candidate_count: int
    verified_path_count: int
    warnings: tuple[str, ...] = ()
    debug: dict[str, Any] = field(default_factory=dict, repr=False)


class _Profiler:
    def __init__(self) -> None:
        self.records: dict[str, StageProfile] = {}
        self.process = psutil.Process()

    def run(self, name: str, function: Callable[[], T]) -> T:
        started = time.perf_counter()
        value = function()
        self.records[name] = StageProfile(
            time.perf_counter() - started,
            self.process.memory_info().rss / (1024 * 1024),
        )
        return value


def _peak_rss_mb() -> float:
    maximum = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return maximum / (1024 * 1024) if sys.platform == "darwin" else maximum / 1024


def _same_geometry(image: sitk.Image, mask: sitk.Image) -> bool:
    return (
        image.GetDimension() == mask.GetDimension() == 3
        and image.GetSize() == mask.GetSize()
        and np.allclose(image.GetSpacing(), mask.GetSpacing(), rtol=0, atol=1e-6)
        and np.allclose(image.GetOrigin(), mask.GetOrigin(), rtol=0, atol=1e-5)
        and np.allclose(image.GetDirection(), mask.GetDirection(), rtol=0, atol=1e-6)
    )


def validate_inputs(image: sitk.Image, mask: sitk.Image) -> None:
    if image.GetDimension() != 3 or mask.GetDimension() != 3:
        raise ValueError("image and aorta mask must both be 3-D")
    if not _same_geometry(image, mask):
        raise ValueError(
            "image and aorta mask geometry must match exactly "
            "(size, spacing, origin, and direction)"
        )
    spacing = np.asarray(image.GetSpacing(), float)
    if np.any(~np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError("image spacing must be finite and positive")


def _case_id(path: str | Path) -> str:
    name = Path(path).name
    return name[:-7] if name.lower().endswith(".nii.gz") else Path(name).stem


def _crop_slices(mask: np.ndarray, spacing_xyz: tuple[float, ...], halo_mm: float) -> tuple[slice, ...] | None:
    points = np.argwhere(mask)
    if not len(points):
        return None
    padding = np.ceil(halo_mm / np.asarray(spacing_xyz[::-1])).astype(int)
    start = np.maximum(points.min(axis=0) - padding, 0)
    stop = np.minimum(points.max(axis=0) + padding + 1, mask.shape)
    return tuple(slice(int(a), int(b)) for a, b in zip(start, stop))


def _local_origin(image: sitk.Image, crop: tuple[slice, ...]) -> tuple[float, float, float]:
    start_zyx = tuple(int(item.start or 0) for item in crop)
    return tuple(
        float(v)
        for v in image.TransformIndexToPhysicalPoint(tuple(reversed(start_zyx)))
    )


def _track(
    method: str,
    image: np.ndarray,
    geometry: Any,
    candidate: Candidate,
    blood: np.ndarray,
    objectness: np.ndarray,
    origin: tuple[float, float, float],
    direction: tuple[float, ...],
) -> Any:
    if method == "segmentation":
        return track_local_segmentation(
            image, geometry, candidate, blood_similarity_zyx=blood,
            tubularity_zyx=objectness, origin_xyz=origin, direction=direction,
        )
    return track_minimal_path(
        geometry, candidate, blood, objectness,
        origin_xyz=origin, direction=direction,
    )


def run_pipeline_images(
    image: sitk.Image,
    mask_image: sitk.Image,
    *,
    case_id: str,
    config: PipelineConfig | None = None,
    visual_path: str | Path | None = None,
    debug_dir: str | Path | None = None,
) -> PipelineResult:
    """Run all stages on already-loaded images."""
    pipeline_started = time.perf_counter()
    cfg = config or PipelineConfig()
    validate_inputs(image, mask_image)
    profiler = _Profiler()
    full_image = profiler.run("read_image_array", lambda: image_to_array(image).astype(np.float32))
    full_mask_raw = profiler.run("read_mask_array", lambda: image_to_array(mask_image))
    finite = np.isfinite(full_image)
    if not finite.all():
        raise ValueError("image contains NaN or infinite voxel values")
    if not np.all(np.isin(full_mask_raw, (0, 1))):
        raise ValueError("aorta mask must be binary (0/1)")
    full_mask = full_mask_raw.astype(bool)
    spacing = tuple(float(v) for v in image.GetSpacing())
    direction = tuple(float(v) for v in image.GetDirection())
    required_halo_mm = cfg.fallback_far_mm + 4.0 * max(cfg.objectness_scales_mm)
    effective_halo_mm = max(cfg.roi_halo_mm, required_halo_mm)
    crop = _crop_slices(full_mask, spacing, effective_halo_mm)
    if crop is None:
        prediction = CasePrediction(case_id=case_id, daughters=(), metadata={"empty_aorta_mask": True})
        profiler.records["total"] = StageProfile(
            time.perf_counter() - pipeline_started, _peak_rss_mb()
        )
        return PipelineResult(prediction, profiler.records, 0, 0, ("empty aorta mask",))
    local_image = np.ascontiguousarray(full_image[crop])
    local_mask = np.ascontiguousarray(full_mask[crop])
    origin = _local_origin(image, crop)
    geometry = profiler.run(
        "analyze_aorta",
        lambda: analyze_aorta(
            local_image, local_mask, spacing, origin_xyz=origin, direction=direction
        ),
    )
    if not geometry.mask.any() or geometry.blood is None:
        prediction = CasePrediction(case_id=case_id, daughters=(), metadata={"empty_cleaned_mask": True})
        profiler.records["total"] = StageProfile(
            time.perf_counter() - pipeline_started, _peak_rss_mb()
        )
        return PipelineResult(prediction, profiler.records, 0, 0, ("empty cleaned aorta mask",))
    blood = profiler.run(
        "blood_similarity", lambda: blood_similarity(local_image, geometry.blood)
    )
    objectness = profiler.run(
        "multiscale_objectness",
        lambda: multiscale_hessian_objectness(
            local_image, spacing, cfg.objectness_scales_mm
        ),
    )
    def propose(feature_blood: np.ndarray, feature_objectness: np.ndarray) -> list[Candidate]:
        proposed = generate_candidates(
            geometry,
            feature_blood,
            feature_objectness,
            origin_xyz=origin,
            direction=direction,
            primary_distance_mm=cfg.primary_distance_mm,
            fallback_distance_mm=(cfg.fallback_near_mm, cfg.fallback_far_mm),
            min_component_mm3=cfg.min_component_mm3,
            merge_distance_mm=cfg.merge_distance_mm,
        )
        return sorted(
            (item for item in proposed if item.score >= cfg.min_candidate_score),
            key=lambda item: (-item.score, item.ostium_xyz),
        )[: cfg.max_candidates]

    candidates = profiler.run("candidate_generation", lambda: propose(blood, objectness))

    def verify(
        selected_candidates: list[Candidate],
        feature_blood: np.ndarray,
        feature_objectness: np.ndarray,
    ) -> list[Any]:
        paths: list[Any] = []
        fallback = "minimal_path" if cfg.verification == "segmentation" else "segmentation"
        for candidate in selected_candidates:
            path = _track(
                cfg.verification, local_image, geometry, candidate,
                feature_blood, feature_objectness, origin, direction,
            )
            if path is None and cfg.verification_fallback:
                path = _track(
                    fallback, local_image, geometry, candidate,
                    feature_blood, feature_objectness, origin, direction,
                )
            if path is not None:
                paths.append(path)
        return paths

    paths = profiler.run(
        "local_verification", lambda: verify(candidates, blood, objectness)
    )
    instances = profiler.run(
        "instance_derivation",
        lambda: derive_instances(paths, geometry, origin_xyz=origin, direction=direction),
    )
    warnings: list[str] = []
    predictions: list[DaughterPrediction] = []
    for instance in instances:
        try:
            predictions.append(
                instance_to_prediction(
                    instance,
                    spacing,
                    origin_xyz=origin,
                    image_direction=direction,
                    seed_distance_mm=cfg.seed_distance_mm,
                )
            )
        except (ValueError, np.linalg.LinAlgError) as exc:
            warnings.append(f"measurement skipped: {exc}")
    predictions.sort(key=lambda item: tuple(round(v, 6) for v in item.ostium_xyz))
    predictions = [
        replace(
            item,
            instance_id=f"branch_{index:03d}",
            parent_instance_id="aorta",
        )
        for index, item in enumerate(predictions, 1)
    ]
    confidence = None
    if cfg.confidence_perturbations:
        def perturbation_run(rng: np.random.Generator) -> tuple[DaughterPrediction, ...]:
            perturbed_blood = np.clip(
                blood + rng.normal(0.0, 0.025, blood.shape).astype(np.float32), 0, 1
            )
            perturbed_objectness = np.clip(
                objectness
                + rng.normal(0.0, 0.015, objectness.shape).astype(np.float32),
                0,
                1,
            )
            perturbed_candidates = propose(perturbed_blood, perturbed_objectness)
            perturbed_paths = verify(
                perturbed_candidates, perturbed_blood, perturbed_objectness
            )
            perturbed_instances = derive_instances(
                perturbed_paths, geometry, origin_xyz=origin, direction=direction
            )
            return tuple(
                DaughterPrediction(ostium_xyz=instance.ostium_xyz)
                for instance in perturbed_instances
            )

        confidence = profiler.run(
            "confidence_perturbations",
            lambda: estimate_perturbation_confidence(
                predictions,
                perturbation_run,
                cfg.confidence_perturbations,
                seed=0,
            ),
        )
    prediction = CasePrediction(
        case_id=case_id,
        daughters=tuple(predictions),
        metadata={
            "pipeline": "branchseed_cpu",
            "crop_start_zyx": [item.start for item in crop],
            "candidate_count": len(candidates),
            "verified_path_count": len(paths),
            "requested_roi_halo_mm": cfg.roi_halo_mm,
            "effective_roi_halo_mm": effective_halo_mm,
            "confidence": None if confidence is None else asdict(confidence),
        },
        instance_id="aorta",
    )
    if visual_path is not None:
        from .visualize import save_orthogonal_overlays

        profiler.run(
            "visualization",
            lambda: save_orthogonal_overlays(
                image,
                visual_path,
                overlay=mask_image,
                predictions=prediction.daughters,
            ),
        )
    if debug_dir is not None:
        destination = Path(debug_dir)
        destination.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination / "features.npz",
            blood_similarity=blood,
            objectness=objectness,
            aorta_mask=geometry.mask,
        )
    profiler.records["total"] = StageProfile(
        time.perf_counter() - pipeline_started, _peak_rss_mb()
    )
    if debug_dir is not None:
        (destination / "profile.json").write_text(
            json.dumps(
                {name: asdict(record) for name, record in profiler.records.items()},
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    return PipelineResult(
        prediction, profiler.records, len(candidates), len(paths), tuple(warnings),
        {
            "crop": crop,
            "origin_xyz": origin,
            "confidence": None if confidence is None else asdict(confidence),
        },
    )


def run_pipeline(
    image_path: str | Path,
    aorta_mask_path: str | Path,
    *,
    config: PipelineConfig | None = None,
    visual_path: str | Path | None = None,
    debug_dir: str | Path | None = None,
) -> PipelineResult:
    """Load and execute a case, wrapping reader failures with clear context."""
    try:
        image = read_image(image_path, pixel_type=sitk.sitkFloat32)
    except RuntimeError as exc:
        raise ValueError(f"cannot read image {image_path}: {exc}") from exc
    try:
        mask = read_image(aorta_mask_path)
    except RuntimeError as exc:
        raise ValueError(f"cannot read aorta mask {aorta_mask_path}: {exc}") from exc
    return run_pipeline_images(
        image, mask, case_id=_case_id(image_path), config=config,
        visual_path=visual_path, debug_dir=debug_dir,
    )
