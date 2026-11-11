"""Shared fixtures: small synthetic clips (no real footage is needed for the tests)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest

from vroom_interlagos.chunking.motion import COLUMNS
from vroom_interlagos.preprocessing.frames import FrameRecord, PreprocessConfig, extract_frames
from vroom_interlagos.synthetic import SyntheticClipSpec, make_synthetic_clip

SMALL = dict(width=320, height=180, fps=30.0)


@pytest.fixture(scope="session")
def clips_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("clips")


@pytest.fixture(scope="session")
def plain_clip(clips_dir: Path) -> Path:
    """6 s slow pan, no events."""
    path = clips_dir / "plain.mp4"
    make_synthetic_clip(path, SyntheticClipSpec(duration_s=6.0, corners_s=[], **SMALL))
    return path


@pytest.fixture(scope="session")
def corner_clip(clips_dir: Path) -> Path:
    """20 s clip with one fast, blurred 'corner' between 9 s and 11 s."""
    path = clips_dir / "corner.mp4"
    make_synthetic_clip(path, SyntheticClipSpec(duration_s=20.0, corners_s=[(9.0, 11.0)], **SMALL))
    return path


@pytest.fixture(scope="session")
def cut_clip(clips_dir: Path) -> Path:
    """8 s clip with a hard scene cut at 4 s."""
    path = clips_dir / "cut.mp4"
    make_synthetic_clip(path, SyntheticClipSpec(duration_s=8.0, corners_s=[], cut_at_s=4.0, **SMALL))
    return path


@pytest.fixture(scope="session")
def defect_clip(clips_dir: Path) -> Path:
    """8 s clip with a frozen picture (2-3 s) and black frames (5-6 s)."""
    path = clips_dir / "defects.mp4"
    spec = SyntheticClipSpec(duration_s=8.0, corners_s=[], duplicate_at_s=(2.0, 3.0), black_at_s=(5.0, 6.0), **SMALL)
    make_synthetic_clip(path, spec)
    return path


@pytest.fixture(scope="session")
def small_config() -> PreprocessConfig:
    return PreprocessConfig(target_width=256, target_height=144, sample_fps=5.0)


@pytest.fixture(scope="session")
def corner_frames(corner_clip: Path, clips_dir: Path) -> Path:
    out = clips_dir / "frames_corner"
    extract_frames(corner_clip, out, PreprocessConfig(target_width=256, target_height=144, sample_fps=3.0))
    return out


@pytest.fixture(scope="session")
def cut_frames(cut_clip: Path, clips_dir: Path, small_config: PreprocessConfig) -> Path:
    out = clips_dir / "frames_cut"
    extract_frames(cut_clip, out, small_config)
    return out


def make_records(n: int, fps: float = 3.0, cuts: tuple[int, ...] = (), invalid: tuple[int, ...] = (),
                 duplicates: tuple[int, ...] = ()) -> list[FrameRecord]:
    """Frame table without images, for chunker unit tests."""
    return [
        FrameRecord(
            frame_index=i, filename=f"frame_{i:06d}.jpg", source_frame_index=i * 10, timestamp_s=i / fps,
            clip_time_s=i / fps, is_invalid=i in invalid, is_duplicate=i in duplicates,
            cut_before=i in cuts, gap_before=False,
        )
        for i in range(n)
    ]


def make_motion_rows(records: list[FrameRecord], unstable: tuple[int, ...] = ()) -> list[dict[str, Any]]:
    """Motion rows for consecutive usable frames; pairs listed in ``unstable`` look like a corner."""
    from vroom_interlagos.chunking.motion import is_continuous, usable_frames

    frames = usable_frames(records)
    rows = []
    for i, (a, b) in enumerate(zip(frames[:-1], frames[1:])):
        bad = i in unstable
        row: dict[str, Any] = {name: math.nan for name in COLUMNS}
        row.update(frame_a=a.frame_index, frame_b=b.frame_index, t_a=a.timestamp_s, t_b=b.timestamp_s,
                   dt=b.timestamp_s - a.timestamp_s, continuous=int(is_continuous(records, a, b)),
                   lk_n_init=0, lk_n_tracked=0, n_matches=0, n_inliers=0, pose_valid=0,
                   sharpness_a=100.0, sharpness_b=100.0)
        if row["continuous"]:
            row.update(
                flow_mag_mean=30.0 if bad else 5.0, flow_mag_median=30.0 if bad else 5.0,
                flow_mag_std=12.0 if bad else 1.0, lk_n_init=300, lk_n_tracked=40 if bad else 280,
                lk_survival=0.13 if bad else 0.93, lk_fb_error=0.3, n_matches=60 if bad else 400,
                n_inliers=20 if bad else 360, inlier_ratio=0.33 if bad else 0.9, pose_valid=1,
                rot_deg=9.0 if bad else 0.5, yaw_deg=9.0 if bad else 0.5, parallax_px=3.0,
                sharpness_a=30.0 if bad else 100.0, sharpness_b=30.0 if bad else 100.0,
            )
        rows.append(row)
    return rows
