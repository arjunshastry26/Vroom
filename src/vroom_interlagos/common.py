"""Shared utilities: logging, YAML/JSON I/O, hashing and path helpers."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, TypeVar

import numpy as np
import yaml

T = TypeVar("T")

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def setup_logging(level: str = "INFO", log_file: Path | None = None) -> None:
    """Configure root logging to the console and, optionally, to a file."""
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(level=level.upper(), format=LOG_FORMAT, handlers=handlers, force=True)


def utc_now_iso() -> str:
    """Current UTC time as an ISO-8601 string (second resolution)."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML mapping. An empty file yields an empty dict."""
    with Path(path).open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a YAML mapping at the top level")
    return data


def dataclass_from_dict(cls: type[T], data: Mapping[str, Any] | None) -> T:
    """Build a (possibly nested) dataclass from a mapping.

    Unknown keys raise ``ValueError`` so that typos in config files are caught
    instead of being silently ignored.
    """
    data = dict(data or {})
    fields = {f.name: f for f in dataclasses.fields(cls)}  # type: ignore[arg-type]
    unknown = set(data) - set(fields)
    if unknown:
        raise ValueError(f"Unknown config keys for {cls.__name__}: {sorted(unknown)}")
    kwargs: dict[str, Any] = {}
    for name, value in data.items():
        factory = fields[name].default_factory
        default = factory() if factory is not dataclasses.MISSING else None
        if dataclasses.is_dataclass(default) and isinstance(value, Mapping):
            kwargs[name] = dataclass_from_dict(type(default), value)
        else:
            kwargs[name] = value
    return cls(**kwargs)


class _NumpyEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:  # noqa: D401 - json API
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return o.as_posix()
        if dataclasses.is_dataclass(o) and not isinstance(o, type):
            return dataclasses.asdict(o)
        return super().default(o)


def _sanitize(obj: Any) -> Any:
    """Replace NaN/inf floats by ``None`` so the output is strict JSON."""
    if isinstance(obj, (float, np.floating)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, Mapping):
        return {str(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _sanitize(obj.tolist())
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return _sanitize(dataclasses.asdict(obj))
    return obj


def write_json(path: Path, obj: Any) -> None:
    """Write strict JSON atomically (temp file + replace)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(_sanitize(obj), fh, indent=2, cls=_NumpyEncoder, allow_nan=False)
        fh.write("\n")
    os.replace(tmp, path)


def read_json(path: Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def sha256_file(path: Path, block_size: int = 1 << 20) -> str:
    """SHA-256 of a file, read in blocks so large videos never sit in RAM."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while block := fh.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def config_hash(obj: Any, length: int = 8) -> str:
    """Short, stable hash of a JSON-serialisable configuration."""
    canonical = json.dumps(_sanitize(obj), sort_keys=True, cls=_NumpyEncoder)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:length]


def portable_path(path: Path, root: Path) -> str:
    """Path relative to ``root`` (POSIX separators) when possible, else absolute."""
    path = Path(path).resolve()
    try:
        return path.relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return path.as_posix()
