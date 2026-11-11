"""Temporal chunking: fixed-length baseline and motion-aware adaptive strategy.

Terminology
-----------
*Usable frames* are the extracted frames that are neither invalid nor (by
default) duplicates. *Positions* index that usable sequence. The sequence is
first split into *segments* at scene cuts, timestamp gaps and invalid frames;
no chunk ever spans such a break.

Inside a segment every chunk has a *core* (cores partition the segment) and is
extended by ``overlap_frames`` frames into the next chunk's core, so adjacent
chunks share frames for later alignment. ``min_frames`` / ``max_frames`` /
``target_frames`` refer to the full chunk length including that overlap, which
is what determines GPU memory in the reconstruction worker.

Both strategies emit the same manifest structure.
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from vroom_interlagos.chunking.motion import is_continuous, usable_frames
from vroom_interlagos.chunking.scoring import ScoreConfig, compute_boundary_scores
from vroom_interlagos.common import config_hash, utc_now_iso
from vroom_interlagos.preprocessing.frames import FrameRecord

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
STRATEGIES = ("fixed", "adaptive")


@dataclass
class ChunkingConfig:
    min_frames: int = 15
    max_frames: int = 45
    target_frames: int = 30
    overlap_frames: int = 5
    fixed_chunk_frames: int | None = None  # fixed baseline length; None = target_frames
    length_penalty: float = 0.5  # adaptive: weight of ((length - target) / target)^2 per chunk
    min_cost_window: int = 3  # adaptive: a cut is scored over max(overlap_frames, this) pairs
    min_segment_frames: int = 2  # shorter segments cannot be reconstructed and are dropped
    drop_duplicates: bool = True
    score: ScoreConfig = field(default_factory=ScoreConfig)

    def validate(self) -> None:
        if self.overlap_frames < 0:
            raise ValueError("overlap_frames must be >= 0")
        if not (0 < self.min_frames <= self.target_frames <= self.max_frames):
            raise ValueError("Require 0 < min_frames <= target_frames <= max_frames")
        if self.min_frames - self.overlap_frames < max(1, self.overlap_frames):
            raise ValueError(
                "min_frames must be at least 2 * overlap_frames (and > overlap_frames) so that a chunk "
                "only overlaps its direct neighbours"
            )
        fixed = self.fixed_chunk_frames or self.target_frames
        if not (self.min_frames <= fixed <= self.max_frames):
            raise ValueError("fixed_chunk_frames must lie within [min_frames, max_frames]")
        if self.min_segment_frames < 2:
            raise ValueError("min_segment_frames must be >= 2")


def split_segments(continuous: Sequence[bool]) -> list[tuple[int, int]]:
    """Split positions ``0..len(continuous)`` into inclusive ``(start, end)`` runs.

    ``continuous[i]`` tells whether positions ``i`` and ``i + 1`` are contiguous.
    """
    segments, start = [], 0
    for i, ok in enumerate(continuous):
        if not ok:
            segments.append((start, i))
            start = i + 1
    segments.append((start, len(continuous)))
    return segments


def fixed_partition(length: int, core: int, min_core: int, max_last: int) -> list[int]:
    """Core lengths for the fixed baseline.

    A trailing remainder shorter than ``min_core`` is merged into the previous
    chunk when the merged chunk stays within ``max_last`` frames; otherwise it
    is kept as a short chunk.
    """
    if length <= core:
        return [length]
    n_full, remainder = divmod(length, core)
    lengths = [core] * n_full
    if remainder:
        if remainder < min_core and lengths[-1] + remainder <= max_last:
            lengths[-1] += remainder
        else:
            lengths.append(remainder)
    return lengths


def _feasible(length: int, min_core: int, max_core: int) -> bool:
    k = math.ceil(length / max_core)
    return k * min_core <= length


def adaptive_partition(
    cut_cost: np.ndarray,
    min_core: int,
    max_core: int,
    target_core: int,
    length_penalty: float,
) -> tuple[list[int], int]:
    """Optimal core lengths for one segment by dynamic programming.

    ``cut_cost[p]`` is the cost of ending a core at position ``p`` (cutting
    between positions ``p`` and ``p + 1``); the segment has
    ``len(cut_cost) + 1`` positions. Minimises::

        sum(cut_cost at the chosen cuts) + length_penalty * sum(((len - target) / target)^2)

    subject to ``min_core <= len <= max_core`` for every core. If the segment
    length admits no such partition, ``min_core`` is relaxed just enough.
    Returns ``(core_lengths, min_core_used)``.
    """
    length = len(cut_cost) + 1
    if length <= min_core:
        return [length], min(length, min_core)
    while not _feasible(length, min_core, max_core):
        min_core -= 1
    cost = np.nan_to_num(np.asarray(cut_cost, dtype=np.float64), nan=1.0)

    best = np.full(length + 1, np.inf)
    back = np.zeros(length + 1, dtype=np.int64)
    best[0] = 0.0
    for end in range(min_core, length + 1):
        boundary = cost[end - 1] if end < length else 0.0
        for size in range(min_core, min(max_core, end) + 1):
            prev = best[end - size]
            if not np.isfinite(prev):
                continue
            candidate = prev + boundary + length_penalty * ((size - target_core) / target_core) ** 2
            if candidate < best[end]:
                best[end], back[end] = candidate, size
    if not np.isfinite(best[length]):  # cannot happen after the feasibility relaxation
        raise RuntimeError("adaptive_partition: no feasible partition found")
    lengths: list[int] = []
    end = length
    while end > 0:
        lengths.append(int(back[end]))
        end -= int(back[end])
    return lengths[::-1], min_core


def _nanmean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size else math.nan


def _nanmax(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    return float(finite.max()) if finite.size else math.nan


def window_cost(score: np.ndarray, start: int, end: int, window: int) -> np.ndarray:
    """Cut cost for a segment spanning positions ``start..end``.

    The cost of cutting after position ``p`` is the mean score over the pairs
    ``p .. p + window - 1`` (clipped to the segment): the frames right after
    the cut form the overlap shared by both chunks, so the whole overlap window
    should be stable, not just one frame pair.
    """
    pairs = score[start:end]  # pair i joins positions i and i + 1
    return np.asarray([_nanmean(pairs[p : p + window]) for p in range(len(pairs))])


def _check_motion_rows(frames: list[FrameRecord], rows: Sequence[dict[str, Any]]) -> None:
    expected = [(a.frame_index, b.frame_index) for a, b in zip(frames[:-1], frames[1:])]
    actual = [(int(r["frame_a"]), int(r["frame_b"])) for r in rows]
    if expected != actual:
        raise ValueError(
            "motion.csv does not match the current frame table (different frames or drop_duplicates setting); "
            "re-run scripts/analyze_motion.py"
        )


def _motion_stats(rows: Sequence[dict[str, Any]]) -> dict[str, float] | None:
    if not rows:
        return None
    col = lambda name: np.asarray([r[name] for r in rows], dtype=np.float64)  # noqa: E731
    return {
        "flow_mag_median_mean": _nanmean(col("flow_mag_median")),
        "flow_mag_median_max": _nanmax(col("flow_mag_median")),
        "flow_mag_std_mean": _nanmean(col("flow_mag_std")),
        "rot_deg_mean": _nanmean(col("rot_deg")),
        "rot_deg_max": _nanmax(col("rot_deg")),
        "parallax_px_mean": _nanmean(col("parallax_px")),
        "lk_survival_mean": _nanmean(col("lk_survival")),
        "inlier_ratio_mean": _nanmean(col("inlier_ratio")),
        "n_matches_mean": _nanmean(col("n_matches")),
        "pose_valid_fraction": _nanmean(col("pose_valid")),
        "sharpness_mean": _nanmean(np.concatenate([col("sharpness_a"), col("sharpness_b")[-1:]])),
    }


def build_chunk_manifest(
    records: list[FrameRecord],
    cfg: ChunkingConfig,
    strategy: str,
    motion_rows: Sequence[dict[str, Any]] | None = None,
    reconstruction_uncertainty: np.ndarray | None = None,
    video_id: str = "unknown",
    frames_dir: str | None = None,
) -> dict[str, Any]:
    """Chunk the extracted frames and return the manifest dictionary.

    ``motion_rows`` (from motion analysis) is required for the adaptive
    strategy. For the fixed baseline it is optional and only used to annotate
    chunks with the same statistics, so both strategies can be compared.
    """
    cfg.validate()
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy {strategy!r}; expected one of {STRATEGIES}")
    if strategy == "adaptive" and motion_rows is None:
        raise ValueError("The adaptive strategy needs motion analysis results")

    frames = usable_frames(records, cfg.drop_duplicates)
    if len(frames) < cfg.min_segment_frames:
        raise ValueError(f"Only {len(frames)} usable frames; nothing to chunk")
    continuous = [is_continuous(records, a, b) for a, b in zip(frames[:-1], frames[1:])]

    score = None
    if motion_rows is not None:
        _check_motion_rows(frames, motion_rows)
        score = compute_boundary_scores(motion_rows, cfg.score, reconstruction_uncertainty)["score"]

    overlap = cfg.overlap_frames
    min_core, max_core = cfg.min_frames - overlap, cfg.max_frames - overlap
    target_core = cfg.target_frames - overlap
    fixed_core = (cfg.fixed_chunk_frames or cfg.target_frames) - overlap
    window = max(overlap, cfg.min_cost_window)

    chunks: list[dict[str, Any]] = []
    segments_out: list[dict[str, Any]] = []
    notes: list[str] = []
    for seg_id, (seg_start, seg_end) in enumerate(split_segments(continuous)):
        seg_len = seg_end - seg_start + 1
        seg_info = {
            "segment_id": seg_id,
            "start_frame": frames[seg_start].frame_index,
            "end_frame": frames[seg_end].frame_index,
            "n_frames": seg_len,
            "status": "chunked",
        }
        segments_out.append(seg_info)
        if seg_len < cfg.min_segment_frames:
            seg_info["status"] = "dropped_too_short"
            continue

        cut_cost = None if score is None else window_cost(score, seg_start, seg_end, window)
        if strategy == "fixed":
            lengths = fixed_partition(seg_len, fixed_core, min_core, cfg.max_frames)
        else:
            assert cut_cost is not None
            lengths, used_min = adaptive_partition(cut_cost, min_core, max_core, target_core, cfg.length_penalty)
            if used_min < min_core and seg_len > min_core:
                notes.append(f"segment {seg_id}: min core length relaxed from {min_core} to {used_min}")

        core_start = seg_start
        seg_chunk_ids: list[int] = []
        for core_len in lengths:
            core_end = core_start + core_len - 1
            end = min(core_end + overlap, seg_end)
            members = frames[core_start : end + 1]
            pair_rows = None if motion_rows is None else motion_rows[core_start:end]
            flags = []
            if len(members) < cfg.min_frames:
                flags.append("below_min_length")
            if len(members) > cfg.max_frames:
                flags.append("above_max_length")
            chunks.append(
                {
                    "chunk_id": f"chunk_{len(chunks) + 1:03d}",
                    "segment_id": seg_id,
                    "start_frame": members[0].frame_index,
                    "end_frame": members[-1].frame_index,
                    "core_start_frame": frames[core_start].frame_index,
                    "core_end_frame": frames[core_end].frame_index,
                    "n_frames": len(members),
                    "frame_indices": [m.frame_index for m in members],
                    "frame_files": [m.filename for m in members],
                    "source_frame_range": [members[0].source_frame_index, members[-1].source_frame_index],
                    "start_timestamp_s": members[0].timestamp_s,
                    "end_timestamp_s": members[-1].timestamp_s,
                    "duration_s": members[-1].timestamp_s - members[0].timestamp_s,
                    "overlap": {"prev": None, "next": None},
                    "boundary_score": {
                        # Cost of the cut before / after this chunk's core; null at segment ends
                        # (forced boundaries) or when no motion analysis is available.
                        "start": None if cut_cost is None or core_start == seg_start
                        else float(cut_cost[core_start - 1 - seg_start]),
                        "end": None if cut_cost is None or core_end == seg_end
                        else float(cut_cost[core_end - seg_start]),
                    },
                    "motion_stats": None if pair_rows is None else _motion_stats(pair_rows),
                    "flags": flags,
                    "_positions": (core_start, end),
                }
            )
            seg_chunk_ids.append(len(chunks) - 1)
            core_start = core_end + 1

        for a, b in zip(seg_chunk_ids[:-1], seg_chunk_ids[1:]):
            shared = sorted(set(chunks[a]["frame_indices"]) & set(chunks[b]["frame_indices"]))
            chunks[a]["overlap"]["next"] = {"chunk_id": chunks[b]["chunk_id"], "n_frames": len(shared), "frame_indices": shared}
            chunks[b]["overlap"]["prev"] = {"chunk_id": chunks[a]["chunk_id"], "n_frames": len(shared), "frame_indices": shared}

    if not chunks:
        raise ValueError("No segment is long enough to form a chunk")
    summary = _summarise(chunks, motion_rows, overlap)
    for chunk in chunks:
        del chunk["_positions"]

    config = asdict(cfg)
    return {
        "schema_version": SCHEMA_VERSION,
        "video_id": video_id,
        "strategy": strategy,
        "created_utc": utc_now_iso(),
        "frames_dir": frames_dir,
        "config": config,
        "config_hash": config_hash({"strategy": strategy, **config}),
        "n_total_frames": len(records),
        "n_usable_frames": len(frames),
        "excluded_frames": {
            "invalid": [r.frame_index for r in records if r.is_invalid],
            "duplicate": [r.frame_index for r in records if r.is_duplicate and cfg.drop_duplicates and not r.is_invalid],
        },
        "motion_analysis_used": motion_rows is not None,
        "reconstruction_uncertainty_used": reconstruction_uncertainty is not None,
        "segments": segments_out,
        "notes": notes,
        "summary": summary,
        "chunks": chunks,
    }


def _summarise(
    chunks: list[dict[str, Any]], motion_rows: Sequence[dict[str, Any]] | None, overlap: int
) -> dict[str, Any]:
    """Aggregate statistics used to compare chunking strategies."""
    n_frames = np.asarray([c["n_frames"] for c in chunks], dtype=np.float64)
    durations = np.asarray([c["duration_s"] for c in chunks], dtype=np.float64)
    cut_costs = np.asarray(
        [c["boundary_score"]["end"] for c in chunks if c["boundary_score"]["end"] is not None], dtype=np.float64
    )
    summary: dict[str, Any] = {
        "n_chunks": len(chunks),
        "n_frames_mean": float(n_frames.mean()),
        "n_frames_min": int(n_frames.min()),
        "n_frames_max": int(n_frames.max()),
        "duration_s_mean": float(durations.mean()),
        "n_internal_cuts": int(cut_costs.size),
        "cut_cost_mean": _nanmean(cut_costs),
        "cut_cost_max": _nanmax(cut_costs),
        "overlap_tracking": None,
    }
    if motion_rows is not None and overlap > 0:
        # Feature-tracking continuity inside the overlaps: pairs between the frames shared with the next chunk.
        rows: list[dict[str, Any]] = []
        for chunk in chunks:
            if chunk["overlap"]["next"] is None:
                continue
            _, end = chunk["_positions"]
            shared = chunk["overlap"]["next"]["n_frames"]
            rows.extend(motion_rows[end - shared : end])  # includes the pair entering the overlap
        if rows:
            col = lambda name: np.asarray([r[name] for r in rows], dtype=np.float64)  # noqa: E731
            summary["overlap_tracking"] = {
                "n_pairs": len(rows),
                "lk_survival_mean": _nanmean(col("lk_survival")),
                "inlier_ratio_mean": _nanmean(col("inlier_ratio")),
                "n_matches_mean": _nanmean(col("n_matches")),
                "rot_deg_mean": _nanmean(col("rot_deg")),
                "pose_valid_fraction": _nanmean(col("pose_valid")),
            }
    return summary


def manifest_dir_name(manifest: dict[str, Any]) -> str:
    """Directory name for a manifest: distinct configurations never overwrite each other."""
    return f"{manifest['strategy']}_{manifest['config_hash']}"


def default_manifest_path(chunks_root: Path, manifest: dict[str, Any]) -> Path:
    return Path(chunks_root) / manifest["video_id"] / manifest_dir_name(manifest) / "chunks.json"
