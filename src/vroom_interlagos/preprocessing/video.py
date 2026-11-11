"""Video metadata inspection and per-frame presentation timestamps.

``ffprobe`` is preferred because it reports container-level information and
exact presentation timestamps (PTS) without decoding. When it is not on PATH
the module falls back to OpenCV, which only exposes nominal values.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".ts"}


class VideoError(RuntimeError):
    """Raised when a video cannot be opened or probed."""


@dataclass
class VideoMetadata:
    """Container/stream properties of the first video stream."""

    path: str
    width: int
    height: int
    fps: float
    n_frames: int | None
    duration_s: float | None
    codec: str | None
    pix_fmt: str | None
    container: str | None
    bit_rate: int | None
    r_frame_rate: str | None
    file_size_bytes: int
    probe_backend: str


def ffprobe_available() -> bool:
    return shutil.which("ffprobe") is not None


def _fraction(text: str | None) -> float | None:
    if not text or text in {"0/0", "N/A"}:
        return None
    try:
        return float(Fraction(text))
    except (ValueError, ZeroDivisionError):
        return None


def _number(text: object, cast: type = float) -> float | int | None:
    try:
        return cast(text)  # type: ignore[call-arg]
    except (TypeError, ValueError):
        return None


def _run_ffprobe(args: list[str]) -> str:
    cmd = ["ffprobe", "-v", "error", *args]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as exc:
        raise VideoError(f"ffprobe failed: {exc.stderr.strip()}") from exc
    return result.stdout


def _probe_ffprobe(path: Path) -> VideoMetadata:
    out = _run_ffprobe(
        [
            "-select_streams", "v:0",
            "-show_entries",
            "stream=codec_name,width,height,avg_frame_rate,r_frame_rate,nb_frames,duration,bit_rate,pix_fmt"
            ":format=format_name,duration,bit_rate",
            "-of", "json",
            str(path),
        ]
    )
    info = json.loads(out)
    streams = info.get("streams") or []
    if not streams:
        raise VideoError(f"{path}: no video stream found")
    stream, fmt = streams[0], info.get("format", {})
    fps = _fraction(stream.get("avg_frame_rate")) or _fraction(stream.get("r_frame_rate"))
    if not fps:
        raise VideoError(f"{path}: could not determine frame rate")
    duration = _number(stream.get("duration")) or _number(fmt.get("duration"))
    n_frames = _number(stream.get("nb_frames"), int)
    if n_frames is None and duration:
        n_frames = int(round(duration * fps))
    return VideoMetadata(
        path=path.as_posix(),
        width=int(stream["width"]),
        height=int(stream["height"]),
        fps=float(fps),
        n_frames=n_frames,
        duration_s=duration,
        codec=stream.get("codec_name"),
        pix_fmt=stream.get("pix_fmt"),
        container=fmt.get("format_name"),
        bit_rate=_number(stream.get("bit_rate"), int) or _number(fmt.get("bit_rate"), int),
        r_frame_rate=stream.get("r_frame_rate"),
        file_size_bytes=path.stat().st_size,
        probe_backend="ffprobe",
    )


def _probe_opencv(path: Path) -> VideoMetadata:
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise VideoError(f"{path}: OpenCV could not open the video")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            raise VideoError(f"{path}: could not determine frame rate")
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join(chr((fourcc >> (8 * i)) & 0xFF) for i in range(4)).strip("\x00 ") or None
        return VideoMetadata(
            path=path.as_posix(),
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=float(fps),
            n_frames=count if count > 0 else None,
            duration_s=count / fps if count > 0 else None,
            codec=codec,
            pix_fmt=None,
            container=path.suffix.lstrip(".").lower() or None,
            bit_rate=None,
            r_frame_rate=None,
            file_size_bytes=path.stat().st_size,
            probe_backend="opencv",
        )
    finally:
        cap.release()


def probe_video(path: Path) -> VideoMetadata:
    """Inspect a video file (ffprobe if available, otherwise OpenCV)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Video not found: {path}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        logger.warning("%s: extension not in the tested set %s", path.name, sorted(SUPPORTED_EXTENSIONS))
    if ffprobe_available():
        return _probe_ffprobe(path)
    logger.warning("ffprobe not found on PATH; falling back to OpenCV metadata (less detailed)")
    return _probe_opencv(path)


def read_frame_timestamps(path: Path) -> np.ndarray | None:
    """Presentation timestamps (seconds) of every video frame, in display order.

    Reads packet PTS with ffprobe (no decoding, so it is fast even for long
    files). Returns ``None`` when ffprobe is unavailable or the container does
    not carry usable timestamps; callers then fall back to ``index / fps``.
    """
    if not ffprobe_available():
        return None
    try:
        out = _run_ffprobe(
            ["-select_streams", "v:0", "-show_entries", "packet=pts_time", "-of", "csv=p=0", str(path)]
        )
    except VideoError as exc:
        logger.warning("Could not read packet timestamps: %s", exc)
        return None
    values: list[float] = []
    for line in out.splitlines():
        token = line.strip().strip(",")
        if not token:
            continue
        try:
            values.append(float(token))
        except ValueError:  # "N/A"
            logger.warning("%s: packets without PTS; timestamps will use index / fps", Path(path).name)
            return None
    if not values:
        return None
    # Packets are stored in decode order; B-frames make that differ from display order.
    return np.sort(np.asarray(values, dtype=np.float64))
