"""Configuration loading.

One YAML file, parsed once into nested frozen dataclasses. Nothing in the
pipeline may hard-code a tunable constant; if it is tunable it lives here.
Unknown or missing keys are errors, so a typo in config.yaml fails loudly at
startup rather than silently falling back to a default.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Sequence, Tuple, get_type_hints

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


@dataclass(frozen=True)
class RoiConfig:
    dilation_mm: float
    subvolume_half_mm: float
    subvolume_spacing_mm: float
    isotropy_tolerance: float


@dataclass(frozen=True)
class CalibrateConfig:
    erosion_mm: float
    lumen_percentile: float
    calcium_sigma: float
    max_sigma_ratio: float
    min_contrast_hu: float
    min_interior_voxels: int


@dataclass(frozen=True)
class GeometryConfig:
    spur_prune_factor: float
    centreline_step_mm: float
    spline_smoothing: float
    endcap_normal_deg: float
    endcap_plane_mm: float
    endcap_arc_factor: float
    endcap_radial_factor: float
    end_radius_window_mm: float
    max_extension_mm: float
    min_skeleton_voxels: int
    normal_smooth_voxels: float


@dataclass(frozen=True)
class WallmapConfig:
    probe_inner_mm: float
    probe_outer_mm: float
    probe_radius_mm: float
    probe_samples: int
    probe_disc_samples: int
    angular_bins: int
    ray_step_mm: float
    ray_max_factor: float
    wall_gap_mm: float
    frame_validate_arc_mm: float
    calcium_reject_frac: float


@dataclass(frozen=True)
class DetectConfig:
    peak_footprint: Tuple[int, int]
    min_region_px: int
    threshold_sigma_offset: float
    split_distance_mm: float


@dataclass(frozen=True)
class TraceConfig:
    step_mm: float
    propagate_sigma_offset: float
    max_mm: float
    eligible_mm: float
    leak_ratio: float
    leak_lookback_mm: float
    radius_ratio: float
    strong_match_frac: float
    stable_from_mm: float
    min_component_voxels: int
    split_persist_mm: float
    source_dilate_voxels: int


@dataclass(frozen=True)
class MeasureConfig:
    seed_mm: float
    xsec_half_mm: float
    xsec_spacing_mm: float
    peak_window_mm: float
    bg_annulus_mm: Tuple[float, float]
    radius_min_mm: float
    radius_max_mm: float
    radius_parent_frac: float
    surface_projection_iters: int


@dataclass(frozen=True)
class ResolveConfig:
    bump_boundary: Tuple[float, float, float]
    hu_band_sigma: float
    contact_probe_mm: Tuple[float, float]
    secondary_overlap_frac: float
    secondary_probe_mm: float
    claimed_dilate_mm: float
    eccentricity_max: float


@dataclass(frozen=True)
class OutputConfig:
    round_decimals: int
    direction_decimals: int


@dataclass(frozen=True)
class VizConfig:
    dpi: int


@dataclass(frozen=True)
class Config:
    roi: RoiConfig
    calibrate: CalibrateConfig
    geometry: GeometryConfig
    wallmap: WallmapConfig
    detect: DetectConfig
    trace: TraceConfig
    measure: MeasureConfig
    resolve: ResolveConfig
    output: OutputConfig
    viz: VizConfig

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)

    def hash(self) -> str:
        """Stable short hash of the whole configuration, for the meta block."""
        blob = json.dumps(self.as_dict(), sort_keys=True, default=list)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]

    def override(self, dotted_path: str, value: Any) -> "Config":
        """Return a copy with one dotted-path parameter replaced.

        Used by tools/sweep.py so parameters can be addressed as
        ``wallmap.probe_outer_mm`` without the harness knowing the schema.
        """
        stage, _, key = dotted_path.partition(".")
        if not key:
            raise KeyError(f"expected 'stage.key', got {dotted_path!r}")
        if stage not in {f.name for f in fields(self)}:
            raise KeyError(f"unknown config stage {stage!r}")
        sub = getattr(self, stage)
        if key not in {f.name for f in fields(sub)}:
            raise KeyError(f"unknown config key {dotted_path!r}")
        return dataclasses.replace(self, **{stage: dataclasses.replace(sub, **{key: value})})


def _coerce(value: Any, target: Any) -> Any:
    """Coerce YAML scalars/sequences into the declared field type."""
    if target is float:
        return float(value)
    if target is int:
        return int(value)
    if target is bool:
        return bool(value)
    origin = getattr(target, "__origin__", None)
    if origin is tuple:
        return tuple(_coerce(v, target.__args__[0]) for v in value)
    return value


def _build(cls: Any, data: dict, prefix: str) -> Any:
    # get_type_hints, not f.type: PEP 563 string annotations are in force here.
    hints = get_type_hints(cls)
    declared = {f.name: hints[f.name] for f in fields(cls)}
    unknown = set(data) - set(declared)
    if unknown:
        raise KeyError(f"unknown config keys under {prefix or '<root>'}: {sorted(unknown)}")
    missing = set(declared) - set(data)
    if missing:
        raise KeyError(f"missing config keys under {prefix or '<root>'}: {sorted(missing)}")
    kwargs = {}
    for name, ftype in declared.items():
        value = data[name]
        if is_dataclass(ftype):
            kwargs[name] = _build(ftype, value, f"{prefix}{name}.")
        else:
            kwargs[name] = _coerce(value, ftype)
    return cls(**kwargs)


def load_config(path: Path | str | None = None, overrides: Sequence[str] = ()) -> Config:
    """Load config.yaml. ``overrides`` are ``dotted.path=value`` strings."""
    path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with open(path, "r") as handle:
        data = yaml.safe_load(handle)
    cfg = _build(Config, data, "")
    for item in overrides:
        key, _, raw = item.partition("=")
        cfg = cfg.override(key.strip(), yaml.safe_load(raw))
    return cfg
