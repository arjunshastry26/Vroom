"""Sequential stitching of aligned chunks into one coordinate frame.

Each chunk ``k`` gets a global similarity ``G_k`` that maps its coordinates
into the frame of the first chunk of its *component*:

    G_first = identity,    G_b = G_a o S_ab    for each accepted edge (a, b)

A rejected or missing edge breaks the chain; the chunks after it start a new
component with its own frame. Nothing is guessed across a break.

This is the *sequential* baseline: every later chunk inherits the errors of
all edges before it. Global pose-graph optimisation (Phase 5) starts from
these edges.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from vroom_interlagos.alignment.pairwise import AlignmentConfig, PairAlignment, _pointmap, align_pair  # noqa: E402
from vroom_interlagos.alignment.sim3 import Sim3, rotation_angle_deg  # noqa: E402
from vroom_interlagos.common import utc_now_iso, write_json  # noqa: E402
from vroom_interlagos.reconstruction.artifacts import METADATA, load_chunk_reconstruction  # noqa: E402

logger = logging.getLogger(__name__)


def load_mask(mask_path: str | Path | None, shape: tuple[int, int]) -> np.ndarray | None:
    """Boolean mask (True = usable pixel) resized to ``shape``; ``None`` if no path is given."""
    if mask_path is None:
        return None
    img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read mask image: {mask_path}")
    if img.shape != shape:
        img = cv2.resize(img, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return img > 0


def _try_load(run_dir: Path, chunk_id: str):
    try:
        return load_chunk_reconstruction(Path(run_dir) / chunk_id)
    except (FileNotFoundError, ValueError):
        return None


def align_run(run_dir: Path, manifest: dict[str, Any], cfg: AlignmentConfig) -> list[PairAlignment]:
    """Align every pair of consecutive chunks that the manifest says overlap."""
    edges: list[PairAlignment] = []
    mask = None
    for chunk in manifest["chunks"]:
        nxt = chunk["overlap"]["next"]
        if nxt is None:
            continue
        rec_a, rec_b = _try_load(run_dir, chunk["chunk_id"]), _try_load(run_dir, nxt["chunk_id"])
        if rec_a is None or rec_b is None:
            edges.append(PairAlignment(chunk["chunk_id"], nxt["chunk_id"], nxt["frame_indices"], accepted=False,
                                       rejection_reason="a chunk reconstruction is missing or failed"))
            continue
        if mask is None and cfg.mask_path:
            mask = load_mask(cfg.mask_path, rec_a.depth.shape[1:])
        edge = align_pair(rec_a, rec_b, cfg, mask)
        edges.append(edge)
        logger.info("%s -> %s: %s  inliers=%d/%d (%.0f%%)  scale=%s  residual=%s  pose check: %s deg, %s x travel%s",
                    edge.chunk_a, edge.chunk_b, "accepted" if edge.accepted else "REJECTED", edge.n_inliers,
                    edge.n_correspondences, 100 * edge.inlier_ratio,
                    "n/a" if edge.sim is None else f"{edge.sim.s:.3f}",
                    "n/a" if edge.rmse_rel is None else f"{edge.rmse_rel:.4f}",
                    "n/a" if edge.pose_rotation_error_deg is None else f"{edge.pose_rotation_error_deg:.2f}",
                    "n/a" if edge.pose_centre_error_rel is None else f"{edge.pose_centre_error_rel:.3f}",
                    "" if edge.accepted else f"  [{edge.rejection_reason}]")
    return edges


def sequential_stitch(chunk_ids: list[str], edges: list[PairAlignment]) -> dict[str, dict[str, Any]]:
    """Chain accepted edges. Returns ``{chunk_id: {"component": int, "sim": Sim3}}``."""
    by_pair = {(e.chunk_a, e.chunk_b): e for e in edges}
    out: dict[str, dict[str, Any]] = {}
    component = 0
    for i, cid in enumerate(chunk_ids):
        edge = by_pair.get((chunk_ids[i - 1], cid)) if i > 0 else None
        if i == 0:
            out[cid] = {"component": component, "sim": Sim3.identity()}
        elif edge is not None and edge.accepted and edge.sim is not None:
            out[cid] = {"component": component, "sim": out[chunk_ids[i - 1]]["sim"].compose(edge.sim)}
        else:
            component += 1
            out[cid] = {"component": component, "sim": Sim3.identity()}
    return out


def stitched_trajectory(
    run_dir: Path, manifest: dict[str, Any], transforms: dict[str, dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Camera poses in the stitched frame.

    Returns ``(trajectory, overlap_checks)``. The trajectory has one entry per
    frame, taken from the chunk whose core contains it. ``overlap_checks``
    compares, for every frame two chunks share, the two stitched estimates of
    that camera.
    """
    trajectory: list[dict[str, Any]] = []
    estimates: dict[int, list[tuple[str, int, np.ndarray]]] = {}
    for chunk in manifest["chunks"]:
        rec = _try_load(run_dir, chunk["chunk_id"])
        if rec is None:
            continue
        info = transforms[chunk["chunk_id"]]
        times = chunk.get("timestamps_s") or [None] * len(chunk["frame_indices"])
        for i, frame in enumerate(rec.frame_indices):
            pose = info["sim"].transform_pose(np.asarray(rec.camera_poses[i], dtype=np.float64))
            estimates.setdefault(frame, []).append((chunk["chunk_id"], info["component"], pose))
            if chunk["core_start_frame"] <= frame <= chunk["core_end_frame"]:
                trajectory.append({"frame_index": frame, "timestamp_s": times[i], "chunk_id": chunk["chunk_id"],
                                   "component": info["component"], "pose": pose})
    trajectory.sort(key=lambda row: row["frame_index"])

    checks: list[dict[str, Any]] = []
    for frame, items in sorted(estimates.items()):
        for (cid_a, comp_a, pose_a), (cid_b, comp_b, pose_b) in zip(items[:-1], items[1:]):
            if comp_a != comp_b:
                continue
            checks.append({
                "frame_index": frame, "chunk_a": cid_a, "chunk_b": cid_b,
                "centre_distance": float(np.linalg.norm(pose_a[:3, 3] - pose_b[:3, 3])),
                "rotation_deg": rotation_angle_deg(pose_a[:3, :3].T @ pose_b[:3, :3]),
            })
    return trajectory, checks


def write_trajectory(path_csv: Path, trajectory: list[dict[str, Any]]) -> None:
    """Write the trajectory as CSV (row-major 3x4 pose) and as ``.npz`` next to it."""
    path_csv.parent.mkdir(parents=True, exist_ok=True)
    names = [f"r{i}{j}" if j < 3 else f"t{i}" for i in range(3) for j in range(4)]
    with path_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["frame_index", "timestamp_s", "chunk_id", "component", *names])
        for row in trajectory:
            writer.writerow([row["frame_index"], "" if row["timestamp_s"] is None else f"{row['timestamp_s']:.6f}",
                             row["chunk_id"], row["component"], *[f"{v:.9g}" for v in row["pose"][:3].ravel()]])
    np.savez(path_csv.with_suffix(".npz"),
             frame_indices=np.asarray([r["frame_index"] for r in trajectory]),
             components=np.asarray([r["component"] for r in trajectory]),
             poses=np.asarray([r["pose"] for r in trajectory]))


def plot_trajectory(trajectory: list[dict[str, Any]], out_path: Path, title: str = "") -> None:
    """Top view of the stitched camera path (best-fit plane), one colour per chunk, one panel per component."""
    components = sorted({r["component"] for r in trajectory})
    fig, axes = plt.subplots(1, len(components), figsize=(7 * len(components), 6.5), squeeze=False)
    for ax, comp in zip(axes[0], components):
        rows = [r for r in trajectory if r["component"] == comp]
        centres = np.asarray([r["pose"][:3, 3] for r in rows])
        centred = centres - centres.mean(axis=0)
        _, _, vt = np.linalg.svd(centred, full_matrices=False)
        xy = centred @ vt[:2].T
        ax.plot(xy[:, 0], xy[:, 1], "-", color="0.75", linewidth=1)
        for k, cid in enumerate(sorted({r["chunk_id"] for r in rows})):
            sel = np.asarray([r["chunk_id"] == cid for r in rows])
            ax.scatter(xy[sel, 0], xy[sel, 1], s=14, color=f"C{k % 10}", label=cid, zorder=3)
        ax.scatter(*xy[0], marker="s", color="black", s=50, zorder=4)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(f"component {comp}: frames {rows[0]['frame_index']}-{rows[-1]['frame_index']} "
                     "(arbitrary scale; square = start)", fontsize=9)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(alpha=0.3)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def export_pointcloud(
    run_dir: Path, manifest: dict[str, Any], transforms: dict[str, dict[str, Any]], out_path: Path,
    cfg: AlignmentConfig, frames_dir: Path | None = None, component: int = 0, max_points: int = 2_000_000,
) -> int:
    """Write the stitched point cloud of one component as a binary PLY. Returns the number of points.

    Only each chunk's core frames are used, so overlap frames are not duplicated.
    Colours come from the frame images when ``frames_dir`` is available.
    """
    points, colours = [], []
    mask = None
    step = max(1, cfg.pixel_stride)
    for chunk in manifest["chunks"]:
        info = transforms[chunk["chunk_id"]]
        rec = _try_load(run_dir, chunk["chunk_id"])
        if rec is None or info["component"] != component:
            continue
        if mask is None and cfg.mask_path:
            mask = load_mask(cfg.mask_path, rec.depth.shape[1:])
        for i, frame in enumerate(rec.frame_indices):
            if not chunk["core_start_frame"] <= frame <= chunk["core_end_frame"]:
                continue
            keep = np.asarray(rec.confidence[i]) > cfg.min_confidence
            if mask is not None:
                keep &= mask
            sel = np.zeros_like(keep)
            sel[::step, ::step] = keep[::step, ::step]
            points.append(info["sim"].apply(_pointmap(rec, i)[sel]).astype(np.float32))
            img = None
            if frames_dir is not None:
                img = cv2.imread(str(Path(frames_dir) / chunk["frame_files"][i]))
            if img is not None and img.shape[:2] != keep.shape:
                img = cv2.resize(img, (keep.shape[1], keep.shape[0]), interpolation=cv2.INTER_AREA)
            colours.append(img[sel][:, ::-1] if img is not None else np.full((int(sel.sum()), 3), 160, np.uint8))
    if not points:
        return 0
    pts, cols = np.concatenate(points), np.concatenate(colours).astype(np.uint8)
    if len(pts) > max_points:
        pick = np.random.default_rng(0).choice(len(pts), max_points, replace=False)
        pts, cols = pts[pick], cols[pick]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vertex = np.empty(len(pts), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
    vertex["x"], vertex["y"], vertex["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    vertex["r"], vertex["g"], vertex["b"] = cols[:, 0], cols[:, 1], cols[:, 2]
    header = ("ply\nformat binary_little_endian 1.0\n" f"element vertex {len(pts)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    with out_path.open("wb") as fh:
        fh.write(header.encode("ascii"))
        fh.write(vertex.tobytes())
    return len(pts)


def run_alignment(
    run_dir: Path, manifest: dict[str, Any], cfg: AlignmentConfig, out_dir: Path, frames_dir: Path | None = None,
    write_pointcloud: bool = True,
) -> dict[str, Any]:
    """Align, stitch sequentially and write all outputs into ``out_dir``. Returns the report."""
    run_dir, out_dir = Path(run_dir), Path(out_dir)
    edges = align_run(run_dir, manifest, cfg)
    chunk_ids = [c["chunk_id"] for c in manifest["chunks"]]
    transforms = sequential_stitch(chunk_ids, edges)
    trajectory, checks = stitched_trajectory(run_dir, manifest, transforms)
    if not trajectory:
        raise ValueError(f"No completed chunk reconstructions found in {run_dir}")

    accepted = [e for e in edges if e.accepted]
    centres = np.asarray([r["pose"][:3, 3] for r in trajectory])
    same = np.asarray([a["component"] == b["component"] for a, b in zip(trajectory[:-1], trajectory[1:])])
    steps = np.linalg.norm(np.diff(centres, axis=0), axis=1)[same] if len(centres) > 1 else np.empty(0)
    median_step = float(np.median(steps)) if steps.size else None

    def mean(values: list[float]) -> float | None:
        return float(np.mean(values)) if values else None

    report = {
        "created_utc": utc_now_iso(),
        "video_id": manifest["video_id"],
        "strategy": manifest["strategy"],
        "run_dir": run_dir.as_posix(),
        "method": "sequential stitching of pairwise Sim(3) alignments",
        "conventions": "S_ab maps chunk b coordinates into chunk a; chunk transforms map chunk coordinates "
                       "into the frame of the first chunk of their component; scale is arbitrary",
        "config": asdict(cfg),
        "summary": {
            "n_chunks": len(chunk_ids),
            "n_edges": len(edges),
            "n_accepted": len(accepted),
            "n_components": len({t["component"] for t in transforms.values()}),
            "inlier_ratio_mean": mean([e.inlier_ratio for e in accepted]),
            "rmse_rel_mean": mean([e.rmse_rel for e in accepted if e.rmse_rel is not None]),
            "pose_rotation_error_deg_mean": mean([e.pose_rotation_error_deg for e in accepted]),
            "pose_centre_error_rel_mean": mean([e.pose_centre_error_rel for e in accepted]),
            "overlap_centre_distance_over_median_step": (
                mean([c["centre_distance"] for c in checks]) / median_step if checks and median_step else None),
            "overlap_rotation_deg_mean": mean([c["rotation_deg"] for c in checks]),
            "n_trajectory_frames": len(trajectory),
        },
        "edges": [e.to_dict() for e in edges],
        "chunk_transforms": {cid: {"component": t["component"], **t["sim"].to_dict()} for cid, t in transforms.items()},
        "overlap_checks": checks,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "alignment.json", report)
    write_trajectory(out_dir / "trajectory_sequential.csv", trajectory)
    plot_trajectory(trajectory, out_dir / "trajectory_sequential.png",
                    title=f"{manifest['video_id']} / {manifest['strategy']}: sequentially stitched camera path")
    if write_pointcloud:
        n = export_pointcloud(run_dir, manifest, transforms, out_dir / "pointcloud_sequential_component0.ply", cfg,
                              frames_dir=frames_dir)
        report["summary"]["pointcloud_points_component0"] = n
        write_json(out_dir / "alignment.json", report)
    return report
