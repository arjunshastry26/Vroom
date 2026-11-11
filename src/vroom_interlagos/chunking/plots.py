"""Diagnostic plot: motion statistics over time with chunk boundaries."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")  # headless backend; no display needed
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_STRATEGY_COLOURS = {"fixed": "tab:red", "adaptive": "tab:green"}


def plot_chunk_boundaries(
    motion_rows: Sequence[dict[str, Any]],
    scores: dict[str, np.ndarray],
    manifests: Sequence[dict[str, Any]],
    out_path: Path,
    title: str = "",
) -> Path:
    """Plot flow, rotation, tracking and boundary score against time.

    Vertical lines mark the internal cuts (end of each chunk core) of every
    manifest; the bottom panel shows each chunk's full extent including overlap.
    """
    t = np.asarray([0.5 * (r["t_a"] + r["t_b"]) for r in motion_rows])
    col = lambda name: np.asarray([r[name] for r in motion_rows], dtype=np.float64)  # noqa: E731

    fig, axes = plt.subplots(
        5, 1, figsize=(13, 11), sharex=True, gridspec_kw={"height_ratios": [2, 2, 2, 2, 0.6 + 0.5 * len(manifests)]}
    )
    axes[0].plot(t, col("flow_mag_median"), label="median flow magnitude")
    axes[0].fill_between(t, 0, col("flow_mag_std"), alpha=0.25, label="flow std")
    axes[0].set_ylabel("pixels")
    axes[1].plot(t, col("rot_deg"), color="tab:purple", label="rotation (assumed intrinsics)")
    axes[1].set_ylabel("degrees")
    axes[2].plot(t, col("lk_survival"), color="tab:orange", label="LK track survival")
    axes[2].plot(t, col("inlier_ratio"), color="tab:brown", label="epipolar inlier ratio")
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].set_ylabel("ratio")
    axes[3].plot(t, scores["score"], color="black", label="boundary score (lower = better cut)")
    axes[3].set_ylim(-0.05, 1.05)
    axes[3].set_ylabel("score")

    time_of_frame: dict[int, float] = {}
    for row in motion_rows:
        time_of_frame[int(row["frame_a"])] = row["t_a"]
        time_of_frame[int(row["frame_b"])] = row["t_b"]

    for level, manifest in enumerate(manifests):
        colour = _STRATEGY_COLOURS.get(manifest["strategy"], f"C{level}")
        label = manifest["strategy"]
        for chunk in manifest["chunks"]:
            axes[4].barh(
                level, chunk["end_timestamp_s"] - chunk["start_timestamp_s"], left=chunk["start_timestamp_s"],
                height=0.7, color=colour, alpha=0.45, edgecolor="black", linewidth=0.6,
            )
            if chunk["overlap"]["next"] is None:
                continue
            cut_time = time_of_frame.get(chunk["core_end_frame"])
            if cut_time is None:
                continue
            for ax in axes[:4]:
                ax.axvline(cut_time, color=colour, linestyle="--" if manifest["strategy"] == "fixed" else "-",
                           alpha=0.8, linewidth=1.2, label=f"{label} cut")
                label = "_nolegend_"
    axes[4].set_yticks(range(len(manifests)))
    axes[4].set_yticklabels([m["strategy"] for m in manifests])
    axes[4].set_xlabel("video timestamp (s)")
    axes[4].set_title("chunk extents (including overlap)", fontsize=9)

    for ax in axes[:4]:
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path
