"""Alignment of two reconstructed chunks that share frames.

Adjacent chunks were reconstructed from some of the *same images* (the
overlap). For such a frame, pixel ``(u, v)`` shows the same scene point in
both reconstructions, so the two point maps give dense 3-D to 3-D
correspondences without any feature matching. Those correspondences are
filtered by confidence (and optionally by a mask of camera-fixed regions),
subsampled, and a Sim(3) is estimated with RANSAC, because each chunk has its
own coordinate frame *and* its own scale.

The camera poses of the shared frames are not used for the estimate. They
serve as an independent check: after alignment, both chunks should place the
shared cameras at the same position and orientation.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from vroom_interlagos.alignment.sim3 import Sim3, ransac_sim3, rotation_angle_deg
from vroom_interlagos.reconstruction.artifacts import ChunkReconstruction, unproject_depth

logger = logging.getLogger(__name__)


@dataclass
class RansacConfig:
    iterations: int = 1000
    inlier_threshold_rel: float = 0.05  # residual / distance of the point from its camera
    seed: int = 0


@dataclass
class AcceptConfig:
    min_correspondences: int = 500
    min_inliers: int = 300
    min_inlier_ratio: float = 0.3
    max_scale_ratio: float = 50.0  # reject scale outside [1/x, x]
    max_rmse_rel: float = 0.04  # inlier RMS residual relative to point distance
    max_pose_rotation_deg: float = 10.0  # shared cameras must agree in orientation after alignment
    max_pose_centre_rel: float = 0.5  # and in position, relative to the camera travel inside the overlap


@dataclass
class AlignmentConfig:
    pixel_stride: int = 4  # use every n-th pixel in both directions
    min_confidence: float = 1.1  # MonST3R confidence threshold (its demo default)
    max_points: int = 20000  # correspondences passed to RANSAC
    mask_path: str | None = None  # optional image; non-zero = pixels allowed (exclude car bodywork)
    ransac: RansacConfig = field(default_factory=RansacConfig)
    accept: AcceptConfig = field(default_factory=AcceptConfig)


@dataclass
class PairAlignment:
    """Result for one pair of chunks. ``sim`` maps chunk ``b`` coordinates into chunk ``a``."""

    chunk_a: str
    chunk_b: str
    shared_frames: list[int]
    accepted: bool
    rejection_reason: str | None = None
    sim: Sim3 | None = None
    n_correspondences: int = 0
    n_inliers: int = 0
    inlier_ratio: float = 0.0
    rmse_rel: float | None = None  # RMS of inlier residuals relative to point distance
    pose_rotation_error_deg: float | None = None  # median over shared cameras
    pose_centre_error_rel: float | None = None  # median, relative to camera travel in the overlap
    confidence: float = 0.0  # in [0, 1]; weight for the pose graph

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["sim"] = None if self.sim is None else self.sim.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PairAlignment":
        d = dict(d)
        d["sim"] = None if d.get("sim") is None else Sim3.from_dict(d["sim"])
        return cls(**d)


def _pointmap(rec: ChunkReconstruction, i: int) -> np.ndarray:
    if rec.pointmaps is not None:
        return np.asarray(rec.pointmaps[i], dtype=np.float64)
    return unproject_depth(np.asarray(rec.depth[i]), np.asarray(rec.intrinsics[i]), np.asarray(rec.camera_poses[i]))


def overlap_correspondences(
    rec_a: ChunkReconstruction, rec_b: ChunkReconstruction, cfg: AlignmentConfig, mask: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """3-D correspondences from the frames both chunks contain.

    Returns ``(points_a, points_b, distance_a, shared_frames)`` where
    ``distance_a`` is each point's distance from its camera in chunk ``a``
    (used to normalise residuals).
    """
    index_a = {f: i for i, f in enumerate(rec_a.frame_indices)}
    index_b = {f: i for i, f in enumerate(rec_b.frame_indices)}
    shared = sorted(set(index_a) & set(index_b))
    pts_a, pts_b, dist_a = [], [], []
    step = max(1, cfg.pixel_stride)
    for frame in shared:
        ia, ib = index_a[frame], index_b[frame]
        if rec_a.depth[ia].shape != rec_b.depth[ib].shape:
            raise ValueError(f"Frame {frame}: chunks were reconstructed at different resolutions")
        keep = (np.asarray(rec_a.confidence[ia]) > cfg.min_confidence) & (
            np.asarray(rec_b.confidence[ib]) > cfg.min_confidence)
        if mask is not None:
            keep &= mask
        sel = np.zeros_like(keep)
        sel[::step, ::step] = keep[::step, ::step]
        pa, pb = _pointmap(rec_a, ia)[sel], _pointmap(rec_b, ib)[sel]
        pts_a.append(pa)
        pts_b.append(pb)
        dist_a.append(np.linalg.norm(pa - np.asarray(rec_a.camera_poses[ia][:3, 3], dtype=np.float64), axis=1))
    if not pts_a:
        empty = np.empty((0, 3))
        return empty, empty, np.empty(0), shared
    pts_a, pts_b, dist_a = np.concatenate(pts_a), np.concatenate(pts_b), np.concatenate(dist_a)
    finite = np.isfinite(pts_a).all(axis=1) & np.isfinite(pts_b).all(axis=1) & (dist_a > 0)
    pts_a, pts_b, dist_a = pts_a[finite], pts_b[finite], dist_a[finite]
    if len(pts_a) > cfg.max_points:
        pick = np.random.default_rng(cfg.ransac.seed).choice(len(pts_a), cfg.max_points, replace=False)
        pts_a, pts_b, dist_a = pts_a[pick], pts_b[pick], dist_a[pick]
    return pts_a, pts_b, dist_a, shared


def pose_agreement(
    rec_a: ChunkReconstruction, rec_b: ChunkReconstruction, sim: Sim3, shared: list[int]
) -> tuple[float, float]:
    """How well the shared cameras coincide after alignment.

    Returns ``(median rotation error in degrees, median centre error relative
    to the camera travel inside the overlap in chunk a)``. This check uses only
    the poses, which the point-based estimate did not use.
    """
    index_a = {f: i for i, f in enumerate(rec_a.frame_indices)}
    index_b = {f: i for i, f in enumerate(rec_b.frame_indices)}
    rot, centre, centres_a = [], [], []
    for frame in shared:
        pose_a = np.asarray(rec_a.camera_poses[index_a[frame]], dtype=np.float64)
        pose_b = sim.transform_pose(np.asarray(rec_b.camera_poses[index_b[frame]], dtype=np.float64))
        rot.append(rotation_angle_deg(pose_a[:3, :3].T @ pose_b[:3, :3]))
        centre.append(np.linalg.norm(pose_a[:3, 3] - pose_b[:3, 3]))
        centres_a.append(pose_a[:3, 3])
    centres_a = np.asarray(centres_a)
    travel = float(np.linalg.norm(np.diff(centres_a, axis=0), axis=1).sum()) if len(centres_a) > 1 else 0.0
    rel = float(np.median(centre)) / travel if travel > 1e-12 else float("inf")
    return float(np.median(rot)), rel


def align_pair(
    rec_a: ChunkReconstruction, rec_b: ChunkReconstruction, cfg: AlignmentConfig, mask: np.ndarray | None = None
) -> PairAlignment:
    """Estimate ``S_ab`` (chunk b -> chunk a) and decide whether to trust it.

    Nothing in the chunk reconstructions is modified.
    """
    pts_a, pts_b, dist_a, shared = overlap_correspondences(rec_a, rec_b, cfg, mask)
    result = PairAlignment(chunk_a=rec_a.chunk_id, chunk_b=rec_b.chunk_id, shared_frames=shared, accepted=False,
                           n_correspondences=len(pts_a))
    acc = cfg.accept
    if not shared:
        result.rejection_reason = "no shared frames"
        return result
    if len(pts_a) < acc.min_correspondences:
        result.rejection_reason = f"only {len(pts_a)} confident correspondences (< {acc.min_correspondences})"
        return result

    fit = ransac_sim3(pts_b, pts_a, cfg.ransac.inlier_threshold_rel, residual_scale=dist_a,
                      iterations=cfg.ransac.iterations, seed=cfg.ransac.seed)
    if fit is None:
        result.rejection_reason = "no similarity found (degenerate correspondences)"
        return result
    result.sim = fit.sim
    result.n_inliers = int(fit.inliers.sum())
    result.inlier_ratio = result.n_inliers / len(pts_a)
    result.rmse_rel = float(np.sqrt(np.mean(fit.residuals[fit.inliers] ** 2))) if result.n_inliers else None
    rot_err, centre_err = pose_agreement(rec_a, rec_b, fit.sim, shared)
    result.pose_rotation_error_deg, result.pose_centre_error_rel = rot_err, centre_err
    result.confidence = float(result.inlier_ratio * min(1.0, result.n_inliers / (4 * acc.min_inliers)))

    reasons = []
    if result.n_inliers < acc.min_inliers:
        reasons.append(f"{result.n_inliers} inliers (< {acc.min_inliers})")
    if result.inlier_ratio < acc.min_inlier_ratio:
        reasons.append(f"inlier ratio {result.inlier_ratio:.2f} (< {acc.min_inlier_ratio})")
    if not (1.0 / acc.max_scale_ratio <= fit.sim.s <= acc.max_scale_ratio):
        reasons.append(f"implausible scale {fit.sim.s:.3g}")
    if result.rmse_rel is not None and result.rmse_rel > acc.max_rmse_rel:
        reasons.append(f"residual {result.rmse_rel:.3f} (> {acc.max_rmse_rel})")
    if rot_err > acc.max_pose_rotation_deg:
        reasons.append(f"shared cameras disagree by {rot_err:.1f} deg in orientation")
    if centre_err > acc.max_pose_centre_rel:
        reasons.append(f"shared cameras disagree by {centre_err:.2f} x overlap travel in position")
    result.accepted = not reasons
    result.rejection_reason = "; ".join(reasons) or None
    return result
