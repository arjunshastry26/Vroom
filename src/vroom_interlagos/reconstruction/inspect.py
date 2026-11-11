"""Per-chunk trajectory diagnostics for reconstructed chunks.

Answers one question before any alignment is attempted: does each chunk's
camera path look like a vehicle driving along a track? Distances are reported
relative to the chunk's median scene depth, because every chunk has its own
arbitrary scale.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from vroom_interlagos.reconstruction.artifacts import load_chunk_reconstruction  # noqa: E402


def trajectory_stats(chunk_dir: Path) -> dict[str, Any]:
    """Step statistics of one chunk's camera path."""
    rec = load_chunk_reconstruction(chunk_dir)
    centres = np.asarray(rec.camera_poses[:, :3, 3], dtype=np.float64)
    steps = np.linalg.norm(np.diff(centres, axis=0), axis=1)
    depth_scale = float(np.median(rec.depth))
    median = float(np.median(steps))
    # Heading change between consecutive frames (camera z axis = viewing direction).
    forward = np.asarray(rec.camera_poses[:, :3, 2], dtype=np.float64)
    cosines = np.clip(np.sum(forward[:-1] * forward[1:], axis=1), -1.0, 1.0)
    # Fraction of the motion that goes along the viewing direction: a car drives where it looks.
    direction = np.diff(centres, axis=0) / np.maximum(steps[:, None], 1e-12)
    along = np.sum(direction * forward[:-1], axis=1)
    return {
        "chunk_id": rec.chunk_id,
        "n_frames": len(centres),
        "frame_indices": rec.frame_indices,
        "centres": centres,
        "steps": steps,
        "depth_scale": depth_scale,
        "step_median_rel": median / depth_scale,
        "step_max_rel": float(steps.max()) / depth_scale,
        "max_over_median": float(steps.max()) / (median + 1e-12),
        "largest_step_at": int(np.argmax(steps)),
        "near_static_fraction": float((steps < 0.1 * steps.mean()).mean()),
        "path_length_rel": float(steps.sum()) / depth_scale,
        "heading_change_deg": np.degrees(np.arccos(cosines)),
        "forward_alignment_median": float(np.median(along)),
        # Share of pixels above MonST3R's demo confidence threshold (1.1), sampled sparsely.
        "confident_fraction": float((np.asarray(rec.confidence[:, ::4, ::4]) > 1.1).mean()),
    }


def plot_run(run_dir: Path, manifest: dict[str, Any], out_path: Path) -> list[dict[str, Any]]:
    """Plot every completed chunk of a run: path (top view of its best-fit plane) and step sizes."""
    stats = []
    for chunk in manifest["chunks"]:
        chunk_dir = Path(run_dir) / chunk["chunk_id"]
        if (chunk_dir / "metadata.json").is_file():
            try:
                stats.append(trajectory_stats(chunk_dir))
            except (ValueError, OSError):
                continue
    if not stats:
        raise ValueError(f"No completed chunks found in {run_dir}")

    fig, axes = plt.subplots(len(stats), 2, figsize=(12, 2.9 * len(stats)), squeeze=False)
    for row, s in zip(axes, stats):
        centred = s["centres"] - s["centres"].mean(axis=0)
        _, _, vt = np.linalg.svd(centred, full_matrices=False)  # two principal axes of the path
        xy = centred @ vt[:2].T / s["depth_scale"]
        row[0].plot(xy[:, 0], xy[:, 1], "-", color="0.6", linewidth=1)
        sc = row[0].scatter(xy[:, 0], xy[:, 1], c=np.arange(len(xy)), cmap="viridis", s=18, zorder=3)
        row[0].scatter(*xy[0], marker="s", color="black", s=40, zorder=4, label="start")
        j = s["largest_step_at"]
        row[0].plot(xy[j : j + 2, 0], xy[j : j + 2, 1], color="red", linewidth=2, label="largest step")
        row[0].set_aspect("equal", adjustable="datalim")
        row[0].set_title(f"{s['chunk_id']}: camera path (units of median scene depth)", fontsize=9)
        row[0].legend(fontsize=7, loc="best")
        fig.colorbar(sc, ax=row[0], label="frame in chunk")

        rel = s["steps"] / s["depth_scale"]
        row[1].bar(np.arange(len(rel)), rel, color="tab:blue")
        row[1].axhline(s["step_median_rel"], color="black", linestyle="--", linewidth=1, label="median")
        row[1].set_title(
            f"step per frame pair  |  max/median = {s['max_over_median']:.1f}, "
            f"near-static pairs = {100 * s['near_static_fraction']:.0f}%", fontsize=9)
        row[1].set_xlabel("frame pair in chunk")
        row[1].legend(fontsize=7)
    fig.suptitle(f"{manifest['video_id']} / {manifest['strategy']}: per-chunk camera trajectories (each chunk has its own scale)")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return stats
