from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from vroom_interlagos.common import dataclass_from_dict, load_yaml, read_json
from vroom_interlagos.preprocessing.frames import (
    ClipConfig, CropRegion, PreprocessConfig, _resize, extract_frames, load_frame_table,
)
from vroom_interlagos.preprocessing.video import ffprobe_available, probe_video

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_probe_video(plain_clip: Path) -> None:
    meta = probe_video(plain_clip)
    assert (meta.width, meta.height) == (320, 180)
    assert meta.fps == pytest.approx(30.0, rel=1e-3)
    assert meta.n_frames == 180
    assert meta.duration_s == pytest.approx(6.0, abs=0.05)
    assert meta.file_size_bytes > 0


def test_probe_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        probe_video(tmp_path / "nope.mp4")


def test_extract_frames_basic(plain_clip: Path, tmp_path: Path, small_config: PreprocessConfig) -> None:
    out = tmp_path / "frames"
    meta = extract_frames(plain_clip, out, small_config, video_id="plain")
    records = load_frame_table(out)

    assert len(records) == 30  # 6 s at 5 fps
    assert [r.frame_index for r in records] == list(range(30))
    assert [r.source_frame_index for r in records] == list(range(0, 180, 6))
    ts = np.array([r.timestamp_s for r in records])
    assert np.all(np.diff(ts) > 0)
    assert np.allclose(np.diff(ts), 0.2, atol=1e-3)
    assert not any(r.is_invalid or r.is_duplicate or r.cut_before or r.gap_before for r in records)

    for rec in (records[0], records[-1]):
        img = cv2.imread(str(out / rec.filename))
        assert img is not None and img.shape == (144, 256, 3)

    on_disk = read_json(out / "metadata.json")
    assert on_disk["video_id"] == "plain" and meta["video_id"] == "plain"
    assert on_disk["preprocessing"]["sample_fps"] == 5.0
    assert on_disk["source_video"]["width"] == 320
    assert on_disk["output"]["n_frames"] == 30
    assert on_disk["output"]["achieved_fps"] == pytest.approx(5.0, rel=1e-2)
    assert on_disk["scan"]["n_source_frames_scanned"] == 180
    assert on_disk["scan"]["scene_cuts"] == []
    if ffprobe_available():
        assert on_disk["timestamp_source"] == "ffprobe_pts"


def test_extract_refuses_to_overwrite(plain_clip: Path, tmp_path: Path, small_config: PreprocessConfig) -> None:
    out = tmp_path / "frames"
    extract_frames(plain_clip, out, small_config)
    with pytest.raises(FileExistsError):
        extract_frames(plain_clip, out, small_config)
    cfg = PreprocessConfig(target_width=256, target_height=144, sample_fps=2.0, overwrite=True)
    extract_frames(plain_clip, out, cfg)
    assert len(list(out.glob("frame_*.jpg"))) == 12  # stale frames from the 5 fps run were removed


def test_clip_window(plain_clip: Path, tmp_path: Path) -> None:
    cfg = PreprocessConfig(sample_fps=5.0, clip=ClipConfig(start_s=2.0, duration_s=2.0))
    extract_frames(plain_clip, tmp_path / "f", cfg)
    records = load_frame_table(tmp_path / "f")
    assert len(records) == 10
    assert records[0].clip_time_s == pytest.approx(2.0, abs=1e-3)
    assert all(2.0 - 1e-3 <= r.clip_time_s < 4.0 for r in records)
    assert records[0].source_frame_index == 60


def test_index_timestamps_match_pts(plain_clip: Path, tmp_path: Path) -> None:
    cfg = PreprocessConfig(sample_fps=5.0, timestamp_source="index")
    meta = extract_frames(plain_clip, tmp_path / "f", cfg)
    assert meta["timestamp_source"] == "index_over_fps"
    ts = [r.timestamp_s for r in load_frame_table(tmp_path / "f")]
    assert np.allclose(ts, np.arange(30) * 0.2, atol=1e-3)


def test_scene_cut_detected(cut_frames: Path) -> None:
    meta = read_json(cut_frames / "metadata.json")
    cuts = meta["scan"]["scene_cuts"]
    assert len(cuts) == 1
    assert cuts[0]["timestamp_s"] == pytest.approx(4.0, abs=0.05)
    flagged = [r for r in load_frame_table(cut_frames) if r.cut_before]
    assert len(flagged) == 1
    assert flagged[0].timestamp_s == pytest.approx(4.0, abs=0.21)


def test_fast_pan_is_not_a_cut(corner_frames: Path) -> None:
    assert read_json(corner_frames / "metadata.json")["scan"]["scene_cuts"] == []


def test_duplicate_and_black_frames(defect_clip: Path, tmp_path: Path, small_config: PreprocessConfig) -> None:
    meta = extract_frames(defect_clip, tmp_path / "f", small_config)
    records = load_frame_table(tmp_path / "f")
    duplicates = [r for r in records if r.is_duplicate and not r.is_invalid]
    invalid = [r for r in records if r.is_invalid]
    # Frozen picture 2-3 s: the first frozen frame is new; the 4 later samples repeat it, and so does
    # the sample at 3.0 s because the synthetic camera did not move while frozen.
    assert len(duplicates) == 5
    assert all(2.0 < r.clip_time_s < 3.0 + 1e-3 for r in duplicates)
    assert len(invalid) == 5
    assert all(5.0 - 1e-3 <= r.clip_time_s < 6.0 for r in invalid)
    assert meta["scan"]["n_source_duplicates"] >= 29  # frozen source frames (black frames also repeat)
    assert len(list((tmp_path / "f").glob("frame_*.jpg"))) == len(records)  # nothing is discarded


def test_center_crop_resize() -> None:
    frame = np.zeros((300, 400, 3), dtype=np.uint8)
    frame[:, 100:300] = 255
    cfg = PreprocessConfig(target_width=100, target_height=150, resize_mode="center_crop")
    out = _resize(frame, cfg)
    assert out.shape == (150, 100, 3)
    assert out.min() == 255  # only the central 200x300 region survives the crop


def test_config_validation_and_yaml() -> None:
    cfg = dataclass_from_dict(PreprocessConfig, load_yaml(CONFIGS / "preprocessing.yaml"))
    cfg.validate()
    assert (cfg.target_width, cfg.target_height) == (512, 288)
    assert cfg.scene_cut.hist_threshold == 0.45
    with pytest.raises(ValueError, match="Unknown config keys"):
        dataclass_from_dict(PreprocessConfig, {"target_widht": 512})
    with pytest.raises(ValueError):
        PreprocessConfig(sample_fps=-1).validate()


def test_crop_region_keeps_only_the_requested_part() -> None:
    frame = np.zeros((200, 400, 3), dtype=np.uint8)
    frame[:100] = 200  # bright top half ("scene"), dark bottom half ("car")
    cfg = PreprocessConfig(target_width=64, target_height=16, crop_region=CropRegion(0.0, 0.0, 1.0, 0.5))
    out = _resize(frame, cfg)
    assert out.shape == (16, 64, 3) and out.min() == 200
    with pytest.raises(ValueError, match="crop_region"):
        PreprocessConfig(crop_region=CropRegion(0.5, 0.0, 0.4, 1.0)).validate()


def test_topcrop_config_is_valid_for_monst3r() -> None:
    cfg = dataclass_from_dict(PreprocessConfig, load_yaml(CONFIGS / "preprocessing_topcrop.yaml"))
    cfg.validate()
    assert cfg.target_width % 16 == 0 and cfg.target_height % 16 == 0
    assert cfg.crop_region.y1 < 0.6 and cfg.resize_mode == "center_crop"
    out = _resize(np.zeros((710, 1238, 3), dtype=np.uint8), cfg)
    assert out.shape == (144, 512, 3)
