"""Module B: motion analysis, boundary scoring and temporal chunking."""

from vroom_interlagos.chunking.chunkers import (
    ChunkingConfig,
    adaptive_partition,
    build_chunk_manifest,
    fixed_partition,
    split_segments,
)
from vroom_interlagos.chunking.motion import MotionConfig, analyze_motion, load_motion_table
from vroom_interlagos.chunking.scoring import ScoreConfig, compute_boundary_scores

__all__ = [
    "ChunkingConfig",
    "MotionConfig",
    "ScoreConfig",
    "adaptive_partition",
    "analyze_motion",
    "build_chunk_manifest",
    "compute_boundary_scores",
    "fixed_partition",
    "load_motion_table",
    "split_segments",
]
