"""Chunk-boundary score: how bad is it to cut the video between two frames?

For the pair of consecutive frames ``i`` (frames ``i`` and ``i + 1``)::

    score[i] = sum_k w_k * component_k[i] / sum_k w_k

with every component mapped to [0, 1] (0 = stable, 1 = unstable):

``motion_change``              change of dense-flow magnitude versus the neighbouring pairs
``flow_variance``              spatial standard deviation of the dense flow
``rotation_change``            inter-frame camera rotation (essential matrix)
``tracking_instability``       lost Lucas-Kanade tracks and low epipolar inlier ratio
``blur_penalty``               loss of sharpness relative to the clip median
``reconstruction_uncertainty`` optional, supplied from a previous reconstruction

Components that are unavailable (e.g. no reconstruction feedback yet) are left
out and the remaining weights are renormalised. Motion change, flow variance
and rotation are normalised with robust percentiles *of the clip itself*, so a
score is relative to that clip and not comparable across clips.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

COMPONENTS = (
    "motion_change",
    "flow_variance",
    "rotation_change",
    "tracking_instability",
    "blur_penalty",
    "reconstruction_uncertainty",
)


@dataclass
class ScoreWeights:
    motion_change: float = 1.0
    flow_variance: float = 0.5
    rotation_change: float = 1.0
    tracking_instability: float = 1.0
    blur_penalty: float = 0.5
    reconstruction_uncertainty: float = 1.0


@dataclass
class ScoreConfig:
    weights: ScoreWeights = field(default_factory=ScoreWeights)
    low_percentile: float = 5.0
    high_percentile: float = 95.0
    invalid_pose_value: float = 1.0  # rotation component when the pose could not be estimated


def robust_normalize(values: np.ndarray, low_percentile: float, high_percentile: float) -> np.ndarray:
    """Map values to [0, 1] using percentiles of the finite entries. NaN stays NaN."""
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    out = np.full_like(values, np.nan)
    if finite.size == 0:
        return out
    lo, hi = np.percentile(finite, [low_percentile, high_percentile])
    if hi - lo < 1e-12:
        out[np.isfinite(values)] = 0.0  # constant signal: no evidence of instability
        return out
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0)


def _column(rows: Sequence[dict[str, Any]], name: str) -> np.ndarray:
    return np.asarray([row[name] for row in rows], dtype=np.float64)


def compute_boundary_scores(
    rows: Sequence[dict[str, Any]],
    cfg: ScoreConfig,
    reconstruction_uncertainty: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Per-pair components and the combined ``score`` (NaN for discontinuous pairs).

    ``reconstruction_uncertainty``, when given, must hold one value in [0, 1]
    per pair. It is never synthesised when absent.
    """
    n = len(rows)
    continuous = _column(rows, "continuous") > 0
    lo, hi = cfg.low_percentile, cfg.high_percentile

    mag = _column(rows, "flow_mag_median")
    change = np.zeros(n)
    for i in range(n):
        neighbours = [j for j in (i - 1, i + 1) if 0 <= j < n and continuous[j] and continuous[i]]
        if neighbours:
            change[i] = max(abs(mag[i] - mag[j]) for j in neighbours)
    change[~continuous] = np.nan

    rotation = robust_normalize(_column(rows, "rot_deg"), lo, hi)
    rotation[continuous & ~np.isfinite(rotation)] = cfg.invalid_pose_value

    survival = np.nan_to_num(_column(rows, "lk_survival"), nan=0.0)
    inlier_ratio = np.nan_to_num(_column(rows, "inlier_ratio"), nan=0.0)
    instability = 0.5 * (1.0 - survival) + 0.5 * (1.0 - inlier_ratio)

    sharp_a, sharp_b = _column(rows, "sharpness_a"), _column(rows, "sharpness_b")
    reference = float(np.median(np.concatenate([sharp_a, sharp_b]))) if n else 0.0
    blur = np.clip(1.0 - np.minimum(sharp_a, sharp_b) / reference, 0.0, 1.0) if reference > 0 else np.zeros(n)

    components: dict[str, np.ndarray] = {
        "motion_change": robust_normalize(change, lo, hi),
        "flow_variance": robust_normalize(_column(rows, "flow_mag_std"), lo, hi),
        "rotation_change": rotation,
        "tracking_instability": instability,
        "blur_penalty": blur,
    }
    if reconstruction_uncertainty is not None:
        unc = np.asarray(reconstruction_uncertainty, dtype=np.float64)
        if unc.shape != (n,):
            raise ValueError(f"reconstruction_uncertainty must have shape ({n},), got {unc.shape}")
        components["reconstruction_uncertainty"] = np.clip(unc, 0.0, 1.0)

    total = np.zeros(n)
    weight_sum = 0.0
    for name, values in components.items():
        weight = float(getattr(cfg.weights, name))
        if weight < 0:
            raise ValueError(f"Weight for {name} must be non-negative")
        total += weight * np.nan_to_num(values, nan=0.0)
        weight_sum += weight
    if weight_sum <= 0:
        raise ValueError("At least one boundary-score weight must be positive")
    score = total / weight_sum
    score[~continuous] = np.nan
    for values in components.values():
        values[~continuous] = np.nan
    return {**components, "score": score}
