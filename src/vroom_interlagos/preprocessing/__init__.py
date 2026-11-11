"""Module A: video inspection, frame extraction and frame-quality checks."""

from vroom_interlagos.preprocessing.frames import (
    FrameRecord,
    PreprocessConfig,
    extract_frames,
    load_frame_table,
)
from vroom_interlagos.preprocessing.video import VideoError, VideoMetadata, probe_video

__all__ = [
    "FrameRecord",
    "PreprocessConfig",
    "VideoError",
    "VideoMetadata",
    "extract_frames",
    "load_frame_table",
    "probe_video",
]
