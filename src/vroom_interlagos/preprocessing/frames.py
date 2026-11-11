"""Streaming frame extraction with timestamp preservation.

The video is decoded sequentially, one frame in memory at a time. Frames are
subsampled to a target rate, resized, written as numbered images, and described
in ``frames.csv``. Every parameter and every detected anomaly (scene cuts,
timestamp gaps, duplicates, invalid frames) is recorded in ``metadata.json``.
The source video is never modified.
"""

from __future__ import annotations

import csv
import logging
import platform
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from vroom_interlagos import __version__
from vroom_interlagos.common import utc_now_iso, write_json
from vroom_interlagos.preprocessing.quality import (
    DuplicateConfig,
    FrameSignature,
    InvalidFrameConfig,
    SceneCutConfig,
    SceneCutDetector,
    is_invalid_frame,
    pixel_difference,
)
from vroom_interlagos.preprocessing.video import VideoError, probe_video, read_frame_timestamps

logger = logging.getLogger(__name__)

FRAMES_CSV = "frames.csv"
METADATA_JSON = "metadata.json"
SCHEMA_VERSION = 1

_INTERPOLATION = {
    "area": cv2.INTER_AREA,
    "linear": cv2.INTER_LINEAR,
    "cubic": cv2.INTER_CUBIC,
    "lanczos": cv2.INTER_LANCZOS4,
}


@dataclass
class ClipConfig:
    """Time window to process, in seconds relative to the first video frame."""

    start_s: float = 0.0
    duration_s: float | None = None  # None = until the end of the video


@dataclass
class CropRegion:
    """Region of the source frame to keep, as fractions of its width and height.

    Applied before resizing. Used to cut away image areas that are fixed to the
    camera (car bodywork, recording borders), because the reconstruction model
    has no way to ignore pixels.
    """

    x0: float = 0.0
    y0: float = 0.0
    x1: float = 1.0
    y1: float = 1.0

    def is_full(self) -> bool:
        return (self.x0, self.y0, self.x1, self.y1) == (0.0, 0.0, 1.0, 1.0)


@dataclass
class PreprocessConfig:
    target_width: int = 512
    target_height: int = 288
    resize_mode: str = "stretch"  # "stretch" | "center_crop" (crop to target aspect, then resize)
    interpolation: str = "area"
    sample_fps: float | None = 3.0  # None = keep every source frame
    image_format: str = "jpg"  # "jpg" | "png"
    jpeg_quality: int = 95
    timestamp_source: str = "auto"  # "auto" | "ffprobe" | "index"
    gap_factor: float = 1.5  # PTS step > gap_factor * nominal interval => missing frame(s)
    num_threads: int = 0  # OpenCV threads; 0 = library default
    overwrite: bool = False
    clip: ClipConfig = field(default_factory=ClipConfig)
    crop_region: CropRegion = field(default_factory=CropRegion)
    scene_cut: SceneCutConfig = field(default_factory=SceneCutConfig)
    duplicate: DuplicateConfig = field(default_factory=DuplicateConfig)
    invalid: InvalidFrameConfig = field(default_factory=InvalidFrameConfig)

    def validate(self) -> None:
        if self.target_width <= 0 or self.target_height <= 0:
            raise ValueError("target_width and target_height must be positive")
        if self.resize_mode not in {"stretch", "center_crop"}:
            raise ValueError(f"Unknown resize_mode: {self.resize_mode}")
        if self.interpolation not in _INTERPOLATION:
            raise ValueError(f"Unknown interpolation: {self.interpolation}")
        if self.image_format not in {"jpg", "png"}:
            raise ValueError("image_format must be 'jpg' or 'png'")
        if self.sample_fps is not None and self.sample_fps <= 0:
            raise ValueError("sample_fps must be positive or null")
        if self.timestamp_source not in {"auto", "ffprobe", "index"}:
            raise ValueError(f"Unknown timestamp_source: {self.timestamp_source}")
        c = self.crop_region
        if not (0.0 <= c.x0 < c.x1 <= 1.0 and 0.0 <= c.y0 < c.y1 <= 1.0):
            raise ValueError("crop_region must satisfy 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1")
        if self.clip.start_s < 0 or (self.clip.duration_s is not None and self.clip.duration_s <= 0):
            raise ValueError("clip.start_s must be >= 0 and clip.duration_s positive or null")


@dataclass
class FrameRecord:
    """One row of ``frames.csv``."""

    frame_index: int  # index in the extracted sequence (0-based)
    filename: str
    source_frame_index: int  # index in the original video (0-based, display order)
    timestamp_s: float  # original presentation timestamp
    clip_time_s: float  # seconds since the first frame of the video
    is_invalid: bool  # black / blank frame
    is_duplicate: bool  # (near-)identical to the previous extracted frame
    cut_before: bool  # a scene cut lies between the previous extracted frame and this one
    gap_before: bool  # missing source frames (PTS gap or decode failure) since the previous one


def _resize(frame: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    target_w, target_h = cfg.target_width, cfg.target_height
    if not cfg.crop_region.is_full():
        h, w = frame.shape[:2]
        c = cfg.crop_region
        frame = frame[int(round(c.y0 * h)) : int(round(c.y1 * h)), int(round(c.x0 * w)) : int(round(c.x1 * w))]
    if cfg.resize_mode == "center_crop":
        h, w = frame.shape[:2]
        target_aspect = target_w / target_h
        if w / h > target_aspect:
            new_w = int(round(h * target_aspect))
            x0 = (w - new_w) // 2
            frame = frame[:, x0 : x0 + new_w]
        else:
            new_h = int(round(w / target_aspect))
            y0 = (h - new_h) // 2
            frame = frame[y0 : y0 + new_h]
    if frame.shape[1] == target_w and frame.shape[0] == target_h:
        return frame
    return cv2.resize(frame, (target_w, target_h), interpolation=_INTERPOLATION[cfg.interpolation])


def _load_timestamps(video_path: Path, cfg: PreprocessConfig) -> tuple[np.ndarray | None, str]:
    if cfg.timestamp_source == "index":
        return None, "index_over_fps"
    pts = read_frame_timestamps(video_path)
    if pts is None:
        if cfg.timestamp_source == "ffprobe":
            raise VideoError("timestamp_source='ffprobe' requested but packet timestamps are unavailable")
        logger.warning("Exact timestamps unavailable; using index / fps (assumes constant frame rate)")
        return None, "index_over_fps"
    return pts, "ffprobe_pts"


def extract_frames(
    video_path: Path,
    out_dir: Path,
    cfg: PreprocessConfig,
    video_id: str | None = None,
) -> dict[str, Any]:
    """Extract, resize and index frames from ``video_path`` into ``out_dir``.

    Returns the metadata dictionary that is also written to ``metadata.json``.
    Raises ``FileExistsError`` if ``out_dir`` already holds an extraction and
    ``cfg.overwrite`` is false, so earlier results are never silently replaced.
    """
    cfg.validate()
    video_path, out_dir = Path(video_path), Path(out_dir)
    meta_path = out_dir / METADATA_JSON
    if meta_path.exists() and not cfg.overwrite:
        raise FileExistsError(f"{out_dir} already contains an extraction; pass --overwrite (or set overwrite: true) to replace it")
    out_dir.mkdir(parents=True, exist_ok=True)
    if cfg.overwrite:
        for old in out_dir.glob("frame_*.*"):
            old.unlink()
    if cfg.num_threads > 0:
        cv2.setNumThreads(cfg.num_threads)

    video = probe_video(video_path)
    pts, ts_source = _load_timestamps(video_path, cfg)
    nominal_dt = 1.0 / video.fps
    t0 = float(pts[0]) if pts is not None else 0.0
    start = cfg.clip.start_s
    end = None if cfg.clip.duration_s is None else start + cfg.clip.duration_s
    sample_dt = None if cfg.sample_fps is None else 1.0 / cfg.sample_fps
    eps = 1e-6

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise VideoError(f"OpenCV could not open {video_path}")

    write_params = [cv2.IMWRITE_JPEG_QUALITY, int(cfg.jpeg_quality)] if cfg.image_format == "jpg" else []
    cut_detector = SceneCutDetector(cfg.scene_cut)
    records: list[FrameRecord] = []
    scene_cuts: list[dict[str, Any]] = []
    gaps: list[dict[str, Any]] = []
    decode_failures: list[int] = []
    n_scanned = n_source_duplicates = 0
    pts_exhausted = False
    prev_source_sig: FrameSignature | None = None
    prev_saved_sig: FrameSignature | None = None
    prev_t: float | None = None
    next_sample_t = start
    pending_cut = pending_gap = False

    src_idx = -1
    try:
        while cap.grab():  # grab() advances without the cost of colour conversion
            src_idx += 1
            if pts is not None and src_idx < len(pts):
                t = float(pts[src_idx])
            else:
                if pts is not None and not pts_exhausted:
                    pts_exhausted = True
                    logger.warning("More decoded frames than packet timestamps; extrapolating from frame %d", src_idx)
                t = t0 + src_idx * nominal_dt if pts is None else float(pts[-1]) + (src_idx - len(pts) + 1) * nominal_dt
            rel = t - t0
            if rel < start - eps:
                continue
            if end is not None and rel >= end - eps:
                break

            ok, frame = cap.retrieve()
            if not ok or frame is None:
                decode_failures.append(src_idx)
                pending_gap = True
                continue
            n_scanned += 1

            if prev_t is not None and t - prev_t > cfg.gap_factor * nominal_dt:
                gaps.append({"source_frame_index": src_idx, "timestamp_s": t, "gap_s": t - prev_t})
                pending_gap = True
            prev_t = t

            sig = FrameSignature.from_bgr(frame)
            is_cut, dist = cut_detector.update(sig)
            if is_cut:
                scene_cuts.append({"source_frame_index": src_idx, "timestamp_s": t, "hist_distance": dist})
                pending_cut = True
            if (
                cfg.duplicate.enabled
                and prev_source_sig is not None
                and pixel_difference(prev_source_sig, sig) < cfg.duplicate.max_mean_abs_diff
            ):
                n_source_duplicates += 1
            prev_source_sig = sig

            if sample_dt is not None:
                if rel < next_sample_t - eps:
                    continue
                while next_sample_t <= rel + eps:  # also skips slots when the source is slower than sample_fps
                    next_sample_t += sample_dt

            is_dup = (
                cfg.duplicate.enabled
                and prev_saved_sig is not None
                and not pending_cut
                and pixel_difference(prev_saved_sig, sig) < cfg.duplicate.max_mean_abs_diff
            )
            index = len(records)
            filename = f"frame_{index:06d}.{cfg.image_format}"
            if not cv2.imwrite(str(out_dir / filename), _resize(frame, cfg), write_params):
                raise OSError(f"Failed to write {out_dir / filename}")
            records.append(
                FrameRecord(
                    frame_index=index,
                    filename=filename,
                    source_frame_index=src_idx,
                    timestamp_s=t,
                    clip_time_s=rel,
                    is_invalid=is_invalid_frame(sig, cfg.invalid),
                    is_duplicate=bool(is_dup),
                    cut_before=pending_cut and index > 0,
                    gap_before=pending_gap and index > 0,
                )
            )
            prev_saved_sig = sig
            pending_cut = pending_gap = False
    finally:
        cap.release()

    if not records:
        raise VideoError(f"No frames extracted from {video_path}; check the clip window ({start}s, {end}s)")

    write_frame_table(out_dir / FRAMES_CSV, records)
    span = records[-1].timestamp_s - records[0].timestamp_s
    metadata: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "video_id": video_id or video_path.stem,
        "created_utc": utc_now_iso(),
        "source_video": asdict(video),
        "preprocessing": asdict(cfg),
        "timestamp_source": ts_source,
        "output": {
            "frames_csv": FRAMES_CSV,
            "n_frames": len(records),
            "width": cfg.target_width,
            "height": cfg.target_height,
            "first_timestamp_s": records[0].timestamp_s,
            "last_timestamp_s": records[-1].timestamp_s,
            "achieved_fps": (len(records) - 1) / span if span > 0 else None,
        },
        "scan": {
            "n_source_frames_scanned": n_scanned,
            "n_source_duplicates": n_source_duplicates,
            "decode_failures": decode_failures,
            "scene_cuts": scene_cuts,
            "timestamp_gaps": gaps,
            "n_invalid_extracted": sum(r.is_invalid for r in records),
            "n_duplicate_extracted": sum(r.is_duplicate for r in records),
        },
        "software": {
            "vroom_interlagos": __version__,
            "opencv": cv2.__version__,
            "numpy": np.__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
    }
    write_json(meta_path, metadata)
    logger.info(
        "Extracted %d frames (%d source frames scanned, %d cuts, %d gaps, %d invalid, %d duplicate) -> %s",
        len(records), n_scanned, len(scene_cuts), len(gaps),
        metadata["scan"]["n_invalid_extracted"], metadata["scan"]["n_duplicate_extracted"], out_dir,
    )
    return metadata


def write_frame_table(path: Path, records: list[FrameRecord]) -> None:
    fieldnames = list(FrameRecord.__dataclass_fields__)
    with Path(path).open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            row = asdict(rec)
            row["timestamp_s"] = f"{rec.timestamp_s:.6f}"
            row["clip_time_s"] = f"{rec.clip_time_s:.6f}"
            for key in ("is_invalid", "is_duplicate", "cut_before", "gap_before"):
                row[key] = int(row[key])
            writer.writerow(row)


def load_frame_table(frames_dir: Path) -> list[FrameRecord]:
    """Read ``frames.csv`` from an extraction directory."""
    path = Path(frames_dir) / FRAMES_CSV
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; run scripts/extract_frames.py first")
    records: list[FrameRecord] = []
    with path.open("r", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            records.append(
                FrameRecord(
                    frame_index=int(row["frame_index"]),
                    filename=row["filename"],
                    source_frame_index=int(row["source_frame_index"]),
                    timestamp_s=float(row["timestamp_s"]),
                    clip_time_s=float(row["clip_time_s"]),
                    is_invalid=bool(int(row["is_invalid"])),
                    is_duplicate=bool(int(row["is_duplicate"])),
                    cut_before=bool(int(row["cut_before"])),
                    gap_before=bool(int(row["gap_before"])),
                )
            )
    return records
