from __future__ import annotations

import itertools
import math
from pathlib import Path

import numpy as np
import pytest

from conftest import make_motion_rows, make_records
from vroom_interlagos.chunking.chunkers import (
    ChunkingConfig, adaptive_partition, build_chunk_manifest, default_manifest_path, fixed_partition,
    split_segments, window_cost,
)
from vroom_interlagos.chunking.motion import MotionConfig, analyze_motion
from vroom_interlagos.chunking.plots import plot_chunk_boundaries
from vroom_interlagos.chunking.scoring import ScoreConfig, ScoreWeights, compute_boundary_scores, robust_normalize
from vroom_interlagos.common import dataclass_from_dict, load_yaml, read_json, write_json
from vroom_interlagos.preprocessing.frames import load_frame_table

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
CFG = ChunkingConfig(min_frames=10, max_frames=30, target_frames=20, overlap_frames=4)


# ---------------------------------------------------------------- partitions

def test_split_segments() -> None:
    assert split_segments([True, True, True]) == [(0, 3)]
    assert split_segments([True, False, True, True, False]) == [(0, 1), (2, 4), (5, 5)]
    assert split_segments([]) == [(0, 0)]


def test_fixed_partition() -> None:
    assert fixed_partition(48, 16, 6, 30) == [16, 16, 16]
    assert fixed_partition(10, 16, 6, 30) == [10]
    assert fixed_partition(52, 16, 6, 30) == [16, 16, 20]  # short remainder merged
    assert fixed_partition(52, 16, 6, 18) == [16, 16, 16, 4]  # merging would exceed the limit
    assert fixed_partition(56, 16, 6, 30) == [16, 16, 16, 8]  # remainder long enough to stand alone


def _brute_force(cost: np.ndarray, lo: int, hi: int, target: int, lam: float) -> float:
    length = len(cost) + 1
    best = math.inf
    for k in range(1, length // lo + 1):
        for sizes in itertools.product(range(lo, hi + 1), repeat=k):
            if sum(sizes) != length:
                continue
            ends = np.cumsum(sizes)[:-1]
            total = float(cost[ends - 1].sum()) + lam * sum(((s - target) / target) ** 2 for s in sizes)
            best = min(best, total)
    return best


def _objective(cost: np.ndarray, sizes: list[int], target: int, lam: float) -> float:
    ends = np.cumsum(sizes)[:-1]
    return float(cost[ends - 1].sum()) + lam * sum(((s - target) / target) ** 2 for s in sizes)


@pytest.mark.parametrize("seed", range(8))
def test_adaptive_partition_is_optimal(seed: int) -> None:
    rng = np.random.default_rng(seed)
    length = int(rng.integers(9, 19))
    cost = rng.random(length - 1)
    sizes, used_min = adaptive_partition(cost, 3, 6, 4, 0.5)
    assert used_min == 3
    assert sum(sizes) == length and all(3 <= s <= 6 for s in sizes)
    assert _objective(cost, sizes, 4, 0.5) == pytest.approx(_brute_force(cost, 3, 6, 4, 0.5))


def test_adaptive_partition_cuts_in_valleys() -> None:
    cost = np.ones(59)
    cost[[17, 41]] = 0.0  # two clearly stable places
    sizes, _ = adaptive_partition(cost, 10, 30, 20, 0.1)
    assert sizes == [18, 24, 18]


def test_adaptive_partition_edge_cases() -> None:
    assert adaptive_partition(np.zeros(4), 10, 30, 20, 0.5) == ([5], 5)  # shorter than min: one chunk
    # 17 frames with cores in [10, 15] has no valid partition -> min is relaxed just enough.
    sizes, used_min = adaptive_partition(np.zeros(16), 10, 15, 12, 0.5)
    assert sum(sizes) == 17 and used_min == 8 and min(sizes) >= 8 and max(sizes) <= 15
    sizes, _ = adaptive_partition(np.array([np.nan, 0.0, np.nan, np.nan, np.nan]), 2, 4, 3, 0.0)
    assert sizes == [2, 4]  # NaN costs are treated as worst-case


def test_window_cost() -> None:
    score = np.array([0.0, 1.0, 1.0, 0.0, 0.0, np.nan, 0.5])
    assert np.allclose(window_cost(score, 0, 5, 2), [0.5, 1.0, 0.5, 0.0, 0.0])
    assert window_cost(score, 6, 7, 3)[0] == 0.5


# ---------------------------------------------------------------- scoring

def test_robust_normalize() -> None:
    out = robust_normalize(np.array([0.0, 5.0, 10.0, np.nan, 1000.0]), 0, 75)
    assert out[0] == 0.0 and out[4] == 1.0 and np.isnan(out[3]) and 0 < out[1] < out[2] <= 1.0
    assert np.all(robust_normalize(np.full(5, 3.0), 5, 95) == 0.0)


def test_boundary_scores_separate_stable_and_unstable() -> None:
    records = make_records(60, cuts=(40,))
    rows = make_motion_rows(records, unstable=tuple(range(20, 26)))
    scores = compute_boundary_scores(rows, ScoreConfig())
    s = scores["score"]
    assert np.isnan(s[39]) and np.isnan(scores["blur_penalty"][39])  # pair across the cut
    finite = s[np.isfinite(s)]
    assert finite.min() >= 0.0 and finite.max() <= 1.0
    assert s[21:25].min() > 0.6 and np.nanmax(s[:18]) < 0.15
    for name in ("motion_change", "flow_variance", "rotation_change", "tracking_instability", "blur_penalty"):
        vals = scores[name][np.isfinite(scores[name])]
        assert vals.min() >= 0.0 and vals.max() <= 1.0
    assert "reconstruction_uncertainty" not in scores  # never synthesised


def test_boundary_scores_optional_uncertainty_and_weights() -> None:
    records = make_records(30)
    rows = make_motion_rows(records)
    base = compute_boundary_scores(rows, ScoreConfig())["score"]
    unc = np.zeros(29)
    unc[10] = 1.0
    with_unc = compute_boundary_scores(rows, ScoreConfig(), reconstruction_uncertainty=unc)
    assert with_unc["score"][10] > base[10] + 0.1
    with pytest.raises(ValueError):
        compute_boundary_scores(rows, ScoreConfig(), reconstruction_uncertainty=np.zeros(5))
    zero = ScoreWeights(0, 0, 0, 0, 0, 0)
    with pytest.raises(ValueError):
        compute_boundary_scores(rows, ScoreConfig(weights=zero))


def test_invalid_pose_counts_as_unstable() -> None:
    rows = make_motion_rows(make_records(30))
    rows[12].update(pose_valid=0, rot_deg=math.nan)
    scores = compute_boundary_scores(rows, ScoreConfig())
    assert scores["rotation_change"][12] == 1.0


# ---------------------------------------------------------------- manifests

def _check_manifest(manifest: dict, cfg: ChunkingConfig, usable: list[int]) -> None:
    chunks = manifest["chunks"]
    assert [c["chunk_id"] for c in chunks] == [f"chunk_{i + 1:03d}" for i in range(len(chunks))]
    cores: list[int] = []
    for c in chunks:
        idx = c["frame_indices"]
        assert idx == sorted(idx) and set(idx) <= set(usable)
        assert c["n_frames"] == len(idx) == len(c["frame_files"])
        assert (c["start_frame"], c["end_frame"]) == (idx[0], idx[-1])
        assert c["n_frames"] <= cfg.max_frames or "above_max_length" in c["flags"]
        assert c["end_timestamp_s"] >= c["start_timestamp_s"]
        cores.extend(i for i in idx if c["core_start_frame"] <= i <= c["core_end_frame"])
    in_chunked_segments = [
        i for i in usable
        if any(s["status"] == "chunked" and s["start_frame"] <= i <= s["end_frame"] for s in manifest["segments"])
    ]
    assert cores == in_chunked_segments  # cores partition the frames: complete, ordered, no repeats
    for a, b in zip(chunks[:-1], chunks[1:]):
        if a["segment_id"] != b["segment_id"]:
            assert a["overlap"]["next"] is None and b["overlap"]["prev"] is None
            assert not set(a["frame_indices"]) & set(b["frame_indices"])
            continue
        shared = a["overlap"]["next"]
        assert shared == {**b["overlap"]["prev"], "chunk_id": b["chunk_id"]}
        assert a["overlap"]["next"]["chunk_id"] == b["chunk_id"] and b["overlap"]["prev"]["chunk_id"] == a["chunk_id"]
        assert shared["n_frames"] == cfg.overlap_frames
        assert shared["frame_indices"] == a["frame_indices"][-cfg.overlap_frames:] == b["frame_indices"][: cfg.overlap_frames]


@pytest.mark.parametrize("strategy", ["fixed", "adaptive"])
def test_manifest_structure(strategy: str) -> None:
    records = make_records(100)
    rows = make_motion_rows(records, unstable=tuple(range(14, 20)) + tuple(range(47, 53)))
    manifest = build_chunk_manifest(records, CFG, strategy, motion_rows=rows, video_id="vid")
    _check_manifest(manifest, CFG, list(range(100)))
    assert manifest["strategy"] == strategy and manifest["video_id"] == "vid"
    assert manifest["config"]["overlap_frames"] == 4
    assert all(c["motion_stats"] is not None for c in manifest["chunks"])
    assert manifest["summary"]["n_chunks"] == len(manifest["chunks"])
    assert manifest["summary"]["overlap_tracking"]["n_pairs"] == 4 * (len(manifest["chunks"]) - 1)


def test_fixed_and_adaptive_manifests_are_compatible() -> None:
    records = make_records(100)
    rows = make_motion_rows(records, unstable=tuple(range(14, 20)))
    fixed = build_chunk_manifest(records, CFG, "fixed", motion_rows=rows)
    adaptive = build_chunk_manifest(records, CFG, "adaptive", motion_rows=rows)
    assert set(fixed) == set(adaptive)
    assert set(fixed["chunks"][0]) == set(adaptive["chunks"][0])
    assert fixed["config_hash"] != adaptive["config_hash"]
    assert default_manifest_path(Path("x"), fixed) != default_manifest_path(Path("x"), adaptive)


def test_fixed_baseline_layout() -> None:
    manifest = build_chunk_manifest(make_records(100), CFG, "fixed")  # works without motion analysis
    chunks = manifest["chunks"]
    assert [(c["start_frame"], c["end_frame"]) for c in chunks] == [
        (0, 19), (16, 35), (32, 51), (48, 67), (64, 83), (80, 99),
    ]
    assert all(c["motion_stats"] is None and c["boundary_score"] == {"start": None, "end": None} for c in chunks)
    assert manifest["motion_analysis_used"] is False


def test_adaptive_avoids_unstable_region_fixed_does_not() -> None:
    records = make_records(100)
    unstable = tuple(range(14, 20))  # a "corner" right where the fixed baseline cuts (frame 15/16)
    rows = make_motion_rows(records, unstable=unstable)
    fixed = build_chunk_manifest(records, CFG, "fixed", motion_rows=rows)
    adaptive = build_chunk_manifest(records, CFG, "adaptive", motion_rows=rows)

    def overlaps_touching_corner(manifest: dict) -> int:
        return sum(
            1 for c in manifest["chunks"]
            if c["overlap"]["next"] and set(c["overlap"]["next"]["frame_indices"]) & set(range(14, 21))
        )

    assert overlaps_touching_corner(fixed) >= 1
    assert overlaps_touching_corner(adaptive) == 0
    assert adaptive["summary"]["cut_cost_max"] < 0.2 < fixed["summary"]["cut_cost_max"]


@pytest.mark.parametrize("strategy", ["fixed", "adaptive"])
def test_chunks_never_span_cuts_or_invalid_frames(strategy: str) -> None:
    records = make_records(90, cuts=(40,), invalid=(70,), duplicates=(5, 6))
    rows = make_motion_rows(records)
    manifest = build_chunk_manifest(records, CFG, strategy, motion_rows=rows)
    usable = [i for i in range(90) if i not in (5, 6, 70)]
    _check_manifest(manifest, CFG, usable)
    assert [(s["start_frame"], s["end_frame"]) for s in manifest["segments"]] == [(0, 39), (40, 69), (71, 89)]
    for c in manifest["chunks"]:
        assert not (c["start_frame"] < 40 <= c["end_frame"])
        assert not (c["start_frame"] < 70 < c["end_frame"])
    assert manifest["excluded_frames"] == {"invalid": [70], "duplicate": [5, 6]}


def test_short_segments_are_dropped_and_reported() -> None:
    records = make_records(38, cuts=(1, 30))  # frame 0 is isolated; the tail has only 8 frames
    manifest = build_chunk_manifest(records, CFG, "fixed")
    assert [s["status"] for s in manifest["segments"]] == ["dropped_too_short", "chunked", "chunked"]
    assert manifest["chunks"][0]["start_frame"] == 1
    assert "below_min_length" in manifest["chunks"][-1]["flags"]


def test_manifest_errors() -> None:
    records = make_records(50)
    with pytest.raises(ValueError, match="needs motion analysis"):
        build_chunk_manifest(records, CFG, "adaptive")
    with pytest.raises(ValueError, match="Unknown strategy"):
        build_chunk_manifest(records, CFG, "random")
    stale = make_motion_rows(make_records(50, duplicates=(3,)))
    with pytest.raises(ValueError, match="does not match"):
        build_chunk_manifest(records, CFG, "adaptive", motion_rows=stale)
    with pytest.raises(ValueError):
        ChunkingConfig(min_frames=10, max_frames=30, target_frames=20, overlap_frames=6).validate()
    with pytest.raises(ValueError):
        ChunkingConfig(min_frames=20, max_frames=10, target_frames=15).validate()


def test_default_yaml_config_is_valid() -> None:
    raw = load_yaml(CONFIGS / "chunking.yaml")
    cfg = dataclass_from_dict(ChunkingConfig, raw["chunking"])
    cfg.validate()
    assert cfg.score.weights.rotation_change == 1.0
    motion = dataclass_from_dict(MotionConfig, raw["motion"])
    assert motion.pose.assumed_hfov_deg == 80.0 and motion.drop_duplicates == cfg.drop_duplicates


# ---------------------------------------------------------------- end to end

def test_end_to_end_on_synthetic_clip(corner_frames: Path, tmp_path: Path) -> None:
    """Video -> frames -> motion -> both manifests -> JSON round trip -> plot."""
    rows = analyze_motion(corner_frames, MotionConfig())
    records = load_frame_table(corner_frames)
    # A 2-D pan has degenerate epipolar geometry, so the rotation term is switched off here.
    weights = ScoreWeights(rotation_change=0.0)
    cfg = ChunkingConfig(min_frames=10, max_frames=30, target_frames=20, overlap_frames=4,
                         score=ScoreConfig(weights=weights))
    manifests = [build_chunk_manifest(records, cfg, s, motion_rows=rows, video_id="corner") for s in ("fixed", "adaptive")]
    for manifest in manifests:
        _check_manifest(manifest, cfg, [r.frame_index for r in records])
        path = default_manifest_path(tmp_path / "chunks", manifest)
        write_json(path, manifest)
        assert read_json(path)["chunks"][0]["frame_files"][0] == "frame_000000.jpg"

    adaptive = manifests[1]
    for chunk in adaptive["chunks"]:
        if chunk["overlap"]["next"] is None:
            continue
        times = [records[i].timestamp_s for i in chunk["overlap"]["next"]["frame_indices"]]
        assert max(times) < 9.0 or min(times) > 11.0  # the overlap stays out of the corner (9-11 s)
    assert adaptive["summary"]["cut_cost_mean"] <= manifests[0]["summary"]["cut_cost_mean"] + 1e-9

    plot = plot_chunk_boundaries(rows, compute_boundary_scores(rows, cfg.score), manifests, tmp_path / "plot.png")
    assert plot.is_file() and plot.stat().st_size > 10_000
