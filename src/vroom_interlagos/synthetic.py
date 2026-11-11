"""Synthetic test clip with known motion events.

A camera window slides over a procedurally generated texture: slowly on
"straights" and fast, with motion blur, in "corners". Optional events inject a
hard scene cut, repeated (duplicate) frames and black frames.

This clip exists to exercise the pipeline plumbing without any copyrighted
footage. It is a 2-D pan, not a 3-D scene: it says nothing about
reconstruction quality and its epipolar geometry is degenerate by design.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass
class SyntheticClipSpec:
    width: int = 640
    height: int = 360
    fps: float = 30.0
    duration_s: float = 12.0
    slow_speed_px: float = 3.0  # horizontal pixels per frame on straights
    fast_speed_px: float = 45.0  # in corners
    corners_s: list[tuple[float, float]] = field(default_factory=lambda: [(4.0, 5.5)])
    cut_at_s: float | None = None  # hard cut to a different texture
    duplicate_at_s: tuple[float, float] | None = None  # frozen picture during this interval
    black_at_s: tuple[float, float] | None = None  # black frames during this interval
    seed: int = 0


def _texture(rng: np.random.Generator, width: int, height: int, tint: tuple[float, float, float]) -> np.ndarray:
    """Feature-rich texture: multi-scale noise plus random high-contrast shapes."""
    canvas = np.zeros((height, width, 3), dtype=np.float32)
    for cell in (64, 16, 4):
        noise = rng.random((height // cell + 1, width // cell + 1, 3), dtype=np.float32)
        canvas += cv2.resize(noise, (width, height), interpolation=cv2.INTER_CUBIC)
    canvas = canvas / 3.0 * np.asarray(tint, dtype=np.float32)
    img = np.clip(canvas * 255.0, 0, 255).astype(np.uint8)
    for _ in range(width // 8):
        x, y = int(rng.integers(0, width)), int(rng.integers(0, height))
        colour = tuple(int(c) for c in rng.integers(0, 256, 3))
        if rng.random() < 0.5:
            size = (int(rng.integers(6, 40)), int(rng.integers(6, 40)))
            cv2.rectangle(img, (x, y), (x + size[0], y + size[1]), colour, -1)
        else:
            cv2.circle(img, (x, y), int(rng.integers(4, 22)), colour, -1)
    return img


def _in(t: float, interval: tuple[float, float] | None) -> bool:
    return interval is not None and interval[0] <= t < interval[1]


def make_synthetic_clip(path: Path, spec: SyntheticClipSpec | None = None) -> dict[str, Any]:
    """Write the clip to ``path`` (``.mp4`` -> mp4v, ``.avi`` -> MJPG) and return its ground truth."""
    spec = spec or SyntheticClipSpec()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(spec.seed)
    n_frames = int(round(spec.duration_s * spec.fps))
    tex_w = 4096
    textures = [
        _texture(rng, tex_w, spec.height, (1.0, 1.0, 1.0)),
        _texture(rng, tex_w, spec.height, (0.35, 0.6, 1.0)),  # visibly different colour statistics
    ]

    fourcc = cv2.VideoWriter_fourcc(*("MJPG" if path.suffix.lower() == ".avi" else "mp4v"))
    writer = cv2.VideoWriter(str(path), fourcc, spec.fps, (spec.width, spec.height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open a video writer for {path}")

    x = 0.0
    frozen: np.ndarray | None = None
    try:
        for i in range(n_frames):
            t = i / spec.fps
            in_corner = any(_in(t, c) for c in spec.corners_s)
            speed = spec.fast_speed_px if in_corner else spec.slow_speed_px
            tex = textures[1] if spec.cut_at_s is not None and t >= spec.cut_at_s else textures[0]
            cols = (int(round(x)) + np.arange(spec.width)) % tex_w
            frame = np.ascontiguousarray(tex[:, cols])
            if in_corner:
                kernel = np.full((1, int(speed)), 1.0 / int(speed), dtype=np.float32)
                frame = cv2.filter2D(frame, -1, kernel)
            if _in(t, spec.duplicate_at_s):
                frozen = frame if frozen is None else frozen
                frame = frozen
            else:
                frozen = None
                x += speed
            if _in(t, spec.black_at_s):
                frame = np.zeros_like(frame)
            writer.write(frame)
    finally:
        writer.release()
    return {
        "path": path.as_posix(),
        "n_frames": n_frames,
        "fps": spec.fps,
        "width": spec.width,
        "height": spec.height,
        "corners_s": [list(c) for c in spec.corners_s],
        "cut_at_s": spec.cut_at_s,
        "duplicate_at_s": spec.duplicate_at_s,
        "black_at_s": spec.black_at_s,
    }
