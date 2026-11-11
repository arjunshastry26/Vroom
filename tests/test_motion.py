from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from vroom_interlagos.chunking.motion import (
    DenseFlowConfig, MotionConfig, PoseConfig, TrackingConfig, analyze_motion, dense_flow_stats,
    estimate_relative_pose, intrinsics_from_hfov, lk_tracking_stats, load_motion_table, sharpness,
)
from vroom_interlagos.preprocessing.frames import load_frame_table
from vroom_interlagos.synthetic import _texture


@pytest.fixture(scope="module")
def texture() -> np.ndarray:
    img = _texture(np.random.default_rng(1), 600, 288, (1.0, 1.0, 1.0))
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def test_relative_pose_recovers_known_rotation() -> None:
    rng = np.random.default_rng(0)
    K = intrinsics_from_hfov(512, 288, 80.0)
    R_true, _ = cv2.Rodrigues(np.array([0.0, math.radians(5.0), 0.0]))
    t_true = np.array([0.3, 0.0, 0.05])
    pts_a3d = np.column_stack([rng.uniform(-6, 6, 400), rng.uniform(-3, 3, 400), rng.uniform(4, 14, 400)])
    pts_b3d = pts_a3d @ R_true.T + t_true  # X_b = R X_a + t

    def project(points: np.ndarray) -> np.ndarray:
        uv = points @ K.T
        return uv[:, :2] / uv[:, 2:3]

    pose = estimate_relative_pose(project(pts_a3d), project(pts_b3d), K, PoseConfig())
    assert pose.valid
    assert pose.rot_deg == pytest.approx(5.0, abs=0.1)
    assert pose.yaw_deg == pytest.approx(5.0, abs=0.1)
    assert pose.inlier_ratio > 0.95
    assert np.allclose(pose.rotation, R_true, atol=1e-2)
    direction = t_true / np.linalg.norm(t_true)
    assert float(pose.translation_dir @ direction) > 0.99  # direction only; scale is unobservable
    assert pose.parallax_px > 1.0


def test_relative_pose_rejects_few_matches() -> None:
    K = intrinsics_from_hfov(512, 288, 80.0)
    pts = np.random.default_rng(0).uniform(0, 200, (8, 2))
    pose = estimate_relative_pose(pts, pts + 1.0, K, PoseConfig())
    assert not pose.valid and math.isnan(pose.rot_deg)


def test_dense_flow_measures_known_shift(texture: np.ndarray) -> None:
    a, b = texture[:, 0:512], texture[:, 6:518]  # scene content moves 6 px
    stats = dense_flow_stats(a, b, DenseFlowConfig())
    assert stats["flow_mag_median"] == pytest.approx(6.0, abs=0.6)
    assert stats["flow_mag_std"] < 1.5


def test_lk_survival_high_for_small_motion_low_for_unrelated(texture: np.ndarray) -> None:
    a, b = texture[:, 0:512], texture[:, 6:518]
    good = lk_tracking_stats(a, b, TrackingConfig())
    assert good["lk_n_init"] > 100 and good["lk_survival"] > 0.85
    unrelated = cv2.cvtColor(_texture(np.random.default_rng(7), 512, 288, (1.0, 1.0, 1.0)), cv2.COLOR_BGR2GRAY)
    bad = lk_tracking_stats(a, unrelated, TrackingConfig())
    assert bad["lk_survival"] < 0.2


def test_sharpness_drops_with_blur(texture: np.ndarray) -> None:
    assert sharpness(cv2.GaussianBlur(texture, (0, 0), 3.0)) < 0.25 * sharpness(texture)


def test_analyze_motion_on_corner_clip(corner_frames: Path) -> None:
    rows = analyze_motion(corner_frames, MotionConfig())
    records = load_frame_table(corner_frames)
    assert len(rows) == len(records) - 1
    assert all(r["continuous"] for r in rows)

    corner = [r for r in rows if 9.0 <= r["t_a"] and r["t_b"] <= 11.0]
    straight = [r for r in rows if r["t_b"] <= 8.5 or r["t_a"] >= 11.5]
    assert len(corner) >= 4
    assert np.mean([r["lk_survival"] for r in corner]) < 0.3
    # At this tiny test resolution ~10% of the image leaves the view between samples.
    assert np.mean([r["lk_survival"] for r in straight]) > 0.55
    assert np.mean([r["sharpness_b"] for r in corner]) < 0.6 * np.mean([r["sharpness_b"] for r in straight])

    loaded = load_motion_table(corner_frames)
    assert len(loaded) == len(rows)
    assert loaded[3]["frame_a"] == rows[3]["frame_a"]
    assert loaded[3]["flow_mag_median"] == pytest.approx(rows[3]["flow_mag_median"], abs=1e-5)
    assert (corner_frames / "motion_meta.json").is_file()


def test_no_measurements_across_scene_cut(cut_frames: Path) -> None:
    rows = analyze_motion(cut_frames, MotionConfig())
    broken = [r for r in rows if not r["continuous"]]
    assert len(broken) == 1
    assert math.isnan(broken[0]["flow_mag_median"]) and math.isnan(broken[0]["rot_deg"])
    loaded = load_motion_table(cut_frames)
    assert math.isnan(loaded[rows.index(broken[0])]["lk_survival"])
