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


# ----------------------------------------------------------------------------
# Constructed 3-D scene with a known answer, written in the worker's artifact
# format. Used to test alignment; it is test data, not a reconstruction.

def write_chunk_arrays(chunk_dir: Path, chunk: dict, arrays: dict) -> None:
    import numpy as np

    from vroom_interlagos.common import sha256_file, write_json

    chunk_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, arr in arrays.items():
        np.save(chunk_dir / f"{name}.npy", arr)
        files[f"{name}.npy"] = {"sha256": sha256_file(chunk_dir / f"{name}.npy"), "shape": list(arr.shape),
                                "dtype": str(arr.dtype)}
    write_json(chunk_dir / "metadata.json", {
        "chunk_id": chunk["chunk_id"], "status": "completed", "frame_indices": chunk["frame_indices"],
        "frame_files": chunk["frame_files"], "model": "constructed-scene", "reconstruction_config": {},
        "reconstruction_config_hash": "testhash", "files": files, "gpu": {"available": False},
        "runtime_s": 0.0, "peak_gpu_memory_gb": None, "final_loss": None, "error": None,
    })


def scene_frame(i: int, h: int = 36, w: int = 64):
    """World pose, intrinsics and ray-cast depth of frame ``i`` in a corridor (ground + two walls + end cap)."""
    import cv2
    import numpy as np

    K = np.array([[40.0, 0, w / 2], [0, 40.0, h / 2], [0, 0, 1]])
    R, _ = cv2.Rodrigues(np.array([0.0, 0.25 * np.sin(i * 0.07), 0.0]))
    centre = np.array([1.5 * np.sin(i * 0.05), 0.0, 1.0 * i])
    pose = np.eye(4)
    pose[:3, :3], pose[:3, 3] = R, centre
    u, v = np.meshgrid(np.arange(w, dtype=float), np.arange(h, dtype=float))
    rays = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1) @ R.T
    depth = np.full((h, w), np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        for axis, value in ((1, 1.5), (0, 5.0), (0, -5.0), (2, centre[2] + 40.0), (1, -6.0)):
            t = (value - centre[axis]) / rays[..., axis]
            depth = np.where((t > 1e-6) & (t < depth), t, depth)
    return pose, K, depth


def write_scene_run(run_dir: Path, manifest: dict, depth_noise: float = 0.0, seed: int = 0) -> dict:
    """Write every chunk of ``manifest`` in its own random similarity frame. Returns ``{chunk_id: Sim3}``
    (world -> chunk), the ground truth for alignment tests."""
    import cv2
    import numpy as np

    from vroom_interlagos.alignment.sim3 import Sim3
    from vroom_interlagos.reconstruction.artifacts import unproject_depth

    rng = np.random.default_rng(seed)
    truth = {}
    for chunk in manifest["chunks"]:
        R, _ = cv2.Rodrigues(rng.normal(0, 0.8, 3))
        Q = Sim3(float(rng.uniform(0.3, 3.0)), R, rng.normal(0, 20, 3))
        truth[chunk["chunk_id"]] = Q
        poses, Ks, depths, points = [], [], [], []
        for frame in chunk["frame_indices"]:
            pose, K, depth = scene_frame(frame)
            depth = depth * (1.0 + depth_noise * rng.normal(size=depth.shape))
            pose_c = Q.transform_pose(pose)
            depth_c = Q.s * depth
            poses.append(pose_c), Ks.append(K), depths.append(depth_c)
            points.append(unproject_depth(depth_c, K, pose_c))
        n = len(poses)
        write_chunk_arrays(run_dir / chunk["chunk_id"], chunk, {
            "camera_poses": np.asarray(poses, np.float32), "intrinsics": np.asarray(Ks, np.float32),
            "depth": np.asarray(depths, np.float32), "pointmaps": np.asarray(points, np.float32),
            "confidence": np.full((n, 36, 64), 3.0, np.float32),
        })
    return truth
