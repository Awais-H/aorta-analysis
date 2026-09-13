"""Typed, JSON-serializable prediction data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

Point3D = tuple[float, float, float]


def _point3(value: object, name: str, *, optional: bool = False) -> Point3D | None:
    if value is None and optional:
        return None
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        raise ValueError(f"{name} must contain exactly three numbers")
    try:
        point = tuple(float(component) for component in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain exactly three numbers") from exc
    if not all(math.isfinite(component) for component in point):
        raise ValueError(f"{name} must contain only finite numbers")
    return point  # type: ignore[return-value]


def _path(value: object) -> tuple[Point3D, ...]:
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("candidate_path_xyz must be a sequence of 3D points")
    return tuple(
        _point3(point, f"candidate_path_xyz[{index}]")  # type: ignore[arg-type]
        for index, point in enumerate(value)
    )


@dataclass(frozen=True, slots=True)
class DaughterPrediction:
    """Prediction for one daughter vessel, in physical x-y-z coordinates."""

    ostium_xyz: Point3D
    seed_xyz: Point3D | None = None
    direction_xyz: Point3D | None = None
    radius_mm: float | None = None
    label: str | None = None
    score: float | None = None
    candidate_path_xyz: tuple[Point3D, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    instance_id: str | None = None
    parent_instance_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "ostium_xyz", _point3(self.ostium_xyz, "ostium_xyz"))
        object.__setattr__(self, "seed_xyz", _point3(self.seed_xyz, "seed_xyz", optional=True))
        object.__setattr__(
            self,
            "direction_xyz",
            _point3(self.direction_xyz, "direction_xyz", optional=True),
        )
        radius = None if self.radius_mm is None else float(self.radius_mm)
        if radius is not None and (not math.isfinite(radius) or radius < 0):
            raise ValueError("radius_mm must be finite and non-negative")
        score = None if self.score is None else float(self.score)
        if score is not None and not math.isfinite(score):
            raise ValueError("score must be finite")
        if self.label is not None and not isinstance(self.label, str):
            raise ValueError("label must be a string or null")
        if self.instance_id is not None and not isinstance(self.instance_id, str):
            raise ValueError("instance_id must be a string or null")
        if self.parent_instance_id is not None and not isinstance(self.parent_instance_id, str):
            raise ValueError("parent_instance_id must be a string or null")
        object.__setattr__(self, "radius_mm", radius)
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "candidate_path_xyz", _path(self.candidate_path_xyz))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DaughterPrediction:
        if not isinstance(data, Mapping):
            raise ValueError("daughter prediction must be a JSON object")
        internal_fields = {
            "ostium_xyz",
            "seed_xyz",
            "direction_xyz",
            "radius_mm",
            "label",
            "score",
            "candidate_path_xyz",
            "metadata",
            "instance_id",
            "parent_instance_id",
        }
        challenge_fields = {
            "instance_id",
            "parent_instance_id",
            "ostium_xyz_mm",
            "seed_xyz_mm",
            "radius_mm",
            "direction_xyz",
        }
        is_challenge = "ostium_xyz_mm" in data
        known = challenge_fields if is_challenge else internal_fields
        unexpected = set(data) - known
        if unexpected:
            raise ValueError(f"unexpected daughter fields: {sorted(unexpected)}")
        ostium_key = "ostium_xyz_mm" if is_challenge else "ostium_xyz"
        seed_key = "seed_xyz_mm" if is_challenge else "seed_xyz"
        if ostium_key not in data:
            raise ValueError(f"daughter prediction is missing {ostium_key}")
        metadata = data.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ValueError("metadata must be a JSON object")
        return cls(
            ostium_xyz=_point3(data[ostium_key], ostium_key),  # type: ignore[arg-type]
            seed_xyz=_point3(data.get(seed_key), seed_key, optional=True),
            direction_xyz=_point3(
                data.get("direction_xyz"), "direction_xyz", optional=True
            ),
            radius_mm=data.get("radius_mm"),
            label=data.get("label"),
            score=data.get("score"),
            candidate_path_xyz=_path(data.get("candidate_path_xyz")),
            metadata=dict(metadata),
            instance_id=data.get("instance_id"),
            parent_instance_id=data.get("parent_instance_id"),
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "ostium_xyz": list(self.ostium_xyz),
            "seed_xyz": None if self.seed_xyz is None else list(self.seed_xyz),
            "direction_xyz": (
                None if self.direction_xyz is None else list(self.direction_xyz)
            ),
            "radius_mm": self.radius_mm,
            "label": self.label,
            "score": self.score,
            "candidate_path_xyz": [list(point) for point in self.candidate_path_xyz],
            "metadata": dict(self.metadata),
            "instance_id": self.instance_id,
            "parent_instance_id": self.parent_instance_id,
        }
        return result


@dataclass(frozen=True, slots=True)
class CasePrediction:
    """All daughter-vessel predictions for one image case."""

    case_id: str
    daughters: tuple[DaughterPrediction, ...] = ()
    image_path: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    instance_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.case_id, str) or not self.case_id:
            raise ValueError("case_id must be a non-empty string")
        object.__setattr__(self, "daughters", tuple(self.daughters))
        if not all(isinstance(item, DaughterPrediction) for item in self.daughters):
            raise ValueError("daughters must contain DaughterPrediction objects")
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CasePrediction:
        if not isinstance(data, Mapping):
            raise ValueError("case prediction must be a JSON object")
        known = {
            "case_id", "daughters", "image_path", "metadata", "instance_id", "parent"
        }
        unexpected = set(data) - known
        if unexpected:
            raise ValueError(f"unexpected case fields: {sorted(unexpected)}")
        daughters = data.get("daughters", [])
        metadata = data.get("metadata", {})
        if not isinstance(daughters, Sequence) or isinstance(daughters, (str, bytes)):
            raise ValueError("daughters must be a JSON array")
        if not isinstance(metadata, Mapping):
            raise ValueError("metadata must be a JSON object")
        parent = data.get("parent")
        if parent is not None:
            if (
                not isinstance(parent, Mapping)
                or set(parent) != {"instance_id"}
                or parent.get("instance_id") != "aorta"
            ):
                raise ValueError("parent must be exactly {'instance_id': 'aorta'}")
        return cls(
            case_id=data.get("case_id"),
            daughters=tuple(DaughterPrediction.from_dict(item) for item in daughters),
            image_path=data.get("image_path"),
            metadata=dict(metadata),
            instance_id=parent.get("instance_id") if parent is not None else data.get("instance_id"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "daughters": [daughter.to_dict() for daughter in self.daughters],
            "image_path": self.image_path,
            "metadata": dict(self.metadata),
            "instance_id": self.instance_id,
        }

    @classmethod
    def from_json(cls, source: str | bytes | Path) -> CasePrediction:
        """Parse a JSON string, or read JSON from a Path."""
        if isinstance(source, Path):
            text = source.read_text(encoding="utf-8")
        elif isinstance(source, bytes):
            text = source.decode("utf-8")
        else:
            text = source
        try:
            value = json.loads(text)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError(f"invalid prediction JSON: {exc}") from exc
        return cls.from_dict(value)

    def to_json(self, *, indent: int | None = 2) -> str:
        try:
            return json.dumps(self.to_dict(), indent=indent, sort_keys=True, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"prediction is not JSON serializable: {exc}") from exc

    def save_json(self, path: str | Path, *, indent: int | None = 2) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(self.to_json(indent=indent) + "\n", encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> CasePrediction:
        return cls.from_json(Path(path))
