"""Cheap per-frame quality checks run during decoding.

All checks operate on a small thumbnail so they add little cost to the decode
loop and can therefore be evaluated at the *source* frame rate, where cuts and
frame duplication are much easier to detect than after temporal subsampling.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

THUMB_SIZE = (160, 90)  # (width, height)


@dataclass
class SceneCutConfig:
    """A cut is declared when the colour-histogram distance to the previous
    source frame is large in absolute terms *and* relative to recent history,
    and the thumbnails also differ at pixel level."""

    enabled: bool = True
    hist_threshold: float = 0.45  # Bhattacharyya distance in [0, 1]
    ratio_to_recent: float = 4.0  # must exceed this multiple of the recent median distance
    history: int = 15  # number of recent source frames in the running median
    min_pixel_diff: float = 10.0  # mean absolute grey difference (0-255)


@dataclass
class DuplicateConfig:
    enabled: bool = True
    max_mean_abs_diff: float = 0.6  # thumbnails closer than this (0-255) count as duplicates


@dataclass
class InvalidFrameConfig:
    enabled: bool = True
    min_mean: float = 4.0  # darker than this -> black frame
    min_std: float = 2.0  # flatter than this -> blank/uniform frame


@dataclass
class FrameSignature:
    """Thumbnail-level description of a frame."""

    gray: np.ndarray  # float32 thumbnail
    hist: np.ndarray  # normalised HSV histogram

    @classmethod
    def from_bgr(cls, frame: np.ndarray) -> "FrameSignature":
        small = cv2.resize(frame, THUMB_SIZE, interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [16, 8, 8], [0, 180, 0, 256, 0, 256])
        cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L1)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
        return cls(gray=gray, hist=hist)


def pixel_difference(a: FrameSignature, b: FrameSignature) -> float:
    """Mean absolute grey-level difference between two thumbnails."""
    return float(np.mean(np.abs(a.gray - b.gray)))


def is_invalid_frame(sig: FrameSignature, cfg: InvalidFrameConfig) -> bool:
    """True for black or (near-)uniform frames."""
    if not cfg.enabled:
        return False
    return bool(sig.gray.mean() < cfg.min_mean or sig.gray.std() < cfg.min_std)


class SceneCutDetector:
    """Streaming hard-cut detector over consecutive source frames."""

    def __init__(self, cfg: SceneCutConfig) -> None:
        self.cfg = cfg
        self._prev: FrameSignature | None = None
        self._recent: deque[float] = deque(maxlen=max(1, cfg.history))

    def update(self, sig: FrameSignature) -> tuple[bool, float]:
        """Feed the next frame. Returns ``(is_cut, histogram_distance)``."""
        prev, self._prev = self._prev, sig
        if prev is None:
            return False, 0.0
        dist = float(cv2.compareHist(prev.hist, sig.hist, cv2.HISTCMP_BHATTACHARYYA))
        if not self.cfg.enabled:
            return False, dist
        recent = float(np.median(self._recent)) if self._recent else 0.0
        is_cut = (
            dist > self.cfg.hist_threshold
            and dist > self.cfg.ratio_to_recent * recent
            and pixel_difference(prev, sig) > self.cfg.min_pixel_diff
        )
        if is_cut:
            self._recent.clear()  # statistics of the old shot do not describe the new one
        else:
            self._recent.append(dist)
        return is_cut, dist
