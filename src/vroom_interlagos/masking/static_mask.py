"""Data-driven mask of image regions that do not change while the car moves.

On an onboard camera the car's own bodywork, halo and helmet, and any
broadcast logos, stay fixed in the image. Their pixels therefore vary little
over time, while the track and surroundings change constantly. The mask is
obtained by thresholding the per-pixel temporal standard deviation of the
grey level, computed incrementally (one frame in memory).

Limitations, by construction:

* Regions that are uniform *and* moving (clear sky, featureless asphalt) also
  vary little and are masked out. They carry no trackable features, so this
  costs little for motion analysis, but the mask is not a segmentation of the
  car.
* Parts of the car that do move in the image (rotating tyres, the driver's
  helmet) are only caught when they form small islands surrounded by static
  pixels (``fill_holes_max_fraction``); otherwise they stay unmasked.
* Frames in which the car is stationary make everything look static; use a
  clip window in which the car is moving.

Always inspect the preview image before using the mask.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np


@dataclass
class StaticMaskConfig:
    std_threshold: float = 12.0  # grey levels (0-255); below this a pixel counts as static
    blur_sigma: float = 2.0  # spatial smoothing of each frame, suppresses compression noise
    open_px: int = 5  # remove isolated static specks smaller than this
    close_px: int = 15  # fill holes inside static regions (e.g. logos on the bodywork)
    fill_holes_max_fraction: float = 0.03  # non-static islands smaller than this share of the image
    # are absorbed into the static region (tyres, helmet and other moving parts of the car)
    min_frames: int = 10


def estimate_static_mask(gray_frames: Iterable[np.ndarray], cfg: StaticMaskConfig) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(mask, temporal_std)``.

    ``mask`` is uint8 with 255 = pixel changes over time (analyse it) and
    0 = static (ignore it), the convention expected by ``motion.mask_path``.
    """
    count = 0
    mean = sq_mean = None
    for gray in gray_frames:
        frame = gray.astype(np.float32)
        if cfg.blur_sigma > 0:
            frame = cv2.GaussianBlur(frame, (0, 0), cfg.blur_sigma)
        if mean is None:
            mean, sq_mean = np.zeros_like(frame), np.zeros_like(frame)
        elif frame.shape != mean.shape:
            raise ValueError("All frames must have the same size")
        count += 1
        mean += (frame - mean) / count
        sq_mean += (frame * frame - sq_mean) / count
    if mean is None or count < cfg.min_frames:
        raise ValueError(f"Need at least {cfg.min_frames} frames to estimate a static mask, got {count}")

    std = np.sqrt(np.maximum(sq_mean - mean * mean, 0.0))
    static = (std < cfg.std_threshold).astype(np.uint8)
    if cfg.open_px > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.open_px, cfg.open_px))
        static = cv2.morphologyEx(static, cv2.MORPH_OPEN, kernel)
    if cfg.close_px > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.close_px, cfg.close_px))
        static = cv2.morphologyEx(static, cv2.MORPH_CLOSE, kernel)
    if cfg.fill_holes_max_fraction > 0:
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats((1 - static).astype(np.uint8), connectivity=4)
        max_area = cfg.fill_holes_max_fraction * static.size
        for label in range(1, n_labels):
            if stats[label, cv2.CC_STAT_AREA] < max_area:
                static[labels == label] = 1
    return (1 - static) * 255, std
