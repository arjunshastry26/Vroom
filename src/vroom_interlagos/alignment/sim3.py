"""Similarity transforms (Sim(3)) and their robust estimation.

A similarity is stored as ``(s, R, t)`` and acts on points as::

    X' = s * R @ X + t

Throughout the project ``S_ab`` denotes the similarity that maps coordinates
of chunk ``b`` into the coordinate frame of chunk ``a``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Sim3:
    s: float
    R: np.ndarray  # (3, 3) proper rotation
    t: np.ndarray  # (3,)

    @classmethod
    def identity(cls) -> "Sim3":
        return cls(1.0, np.eye(3), np.zeros(3))

    def apply(self, points: np.ndarray) -> np.ndarray:
        """Transform points of shape (..., 3)."""
        return self.s * (np.asarray(points, dtype=np.float64) @ self.R.T) + self.t

    def compose(self, other: "Sim3") -> "Sim3":
        """``self.compose(other)`` applies ``other`` first, then ``self``."""
        return Sim3(self.s * other.s, self.R @ other.R, self.s * (self.R @ other.t) + self.t)

    def inverse(self) -> "Sim3":
        R_inv = self.R.T
        return Sim3(1.0 / self.s, R_inv, -(R_inv @ self.t) / self.s)

    def matrix(self) -> np.ndarray:
        """4x4 matrix ``[[s R, t], [0, 1]]``."""
        M = np.eye(4)
        M[:3, :3] = self.s * self.R
        M[:3, 3] = self.t
        return M

    def transform_pose(self, pose: np.ndarray) -> np.ndarray:
        """Map a camera-to-chunk pose (4x4, rigid) into the target frame.

        The result is again a rigid camera-to-target pose: the camera centre is
        moved by the similarity and the orientation by its rotation. (The scale
        applies to the scene, i.e. to depths, not to the camera's axes.)
        """
        out = np.eye(4)
        out[:3, :3] = self.R @ pose[:3, :3]
        out[:3, 3] = self.apply(pose[:3, 3])
        return out

    def to_dict(self) -> dict:
        return {"scale": float(self.s), "rotation": self.R.tolist(), "translation": self.t.tolist(),
                "matrix": self.matrix().tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "Sim3":
        return cls(float(d["scale"]), np.asarray(d["rotation"], dtype=np.float64),
                   np.asarray(d["translation"], dtype=np.float64))


def rotation_angle_deg(R: np.ndarray) -> float:
    """Angle of a rotation matrix in degrees."""
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))))


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True) -> Sim3 | None:
    """Least-squares similarity with ``dst ~= s R src + t`` (Umeyama, 1991).

    Returns ``None`` for degenerate input (fewer than 3 points, or points that
    are collinear, so the rotation is not determined).
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n = len(src)
    if n < 3 or src.shape != dst.shape:
        return None
    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    xs, xd = src - mu_s, dst - mu_d
    var_s = float((xs**2).sum() / n)
    if var_s < 1e-18:
        return None
    U, D, Vt = np.linalg.svd(xd.T @ xs / n)
    if D[1] < 1e-9 * D[0]:  # rank < 2: collinear points
        return None
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt
    s = float(np.trace(np.diag(D) @ S) / var_s) if with_scale else 1.0
    if not np.isfinite(s) or s <= 0:
        return None
    return Sim3(s, R, mu_d - s * (R @ mu_s))


@dataclass
class RansacResult:
    sim: Sim3
    inliers: np.ndarray  # boolean mask over the correspondences
    residuals: np.ndarray  # normalised residual of every correspondence under ``sim``
    iterations: int


def ransac_sim3(
    src: np.ndarray,
    dst: np.ndarray,
    threshold: float,
    residual_scale: np.ndarray | None = None,
    iterations: int = 1000,
    seed: int = 0,
    refine_rounds: int = 3,
) -> RansacResult | None:
    """Robust similarity ``dst ~= S(src)`` from 3-D point correspondences.

    A correspondence is an inlier when ``|dst - S(src)| / residual_scale`` is
    below ``threshold``. ``residual_scale`` (one positive value per point,
    default 1) lets the tolerance grow with distance, since depth errors do.
    Minimal samples of 3 points are scored; the best model is refined on its
    inliers. Returns ``None`` if no valid model is found.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n = len(src)
    if n < 3:
        return None
    scale = np.ones(n) if residual_scale is None else np.asarray(residual_scale, dtype=np.float64)
    rng = np.random.default_rng(seed)

    def residuals(sim: Sim3) -> np.ndarray:
        return np.linalg.norm(dst - sim.apply(src), axis=1) / scale

    best_sim, best_count = None, 0
    for _ in range(iterations):
        idx = rng.choice(n, size=3, replace=False)
        sim = umeyama(src[idx], dst[idx])
        if sim is None:
            continue
        count = int((residuals(sim) < threshold).sum())
        if count > best_count:
            best_sim, best_count = sim, count
    if best_sim is None or best_count < 3:
        return None

    sim = best_sim
    inliers = residuals(sim) < threshold
    for _ in range(refine_rounds):  # re-estimate on the consensus set until it is stable
        refined = umeyama(src[inliers], dst[inliers])
        if refined is None:
            break
        new_inliers = residuals(refined) < threshold
        if new_inliers.sum() < 3:
            break
        sim = refined
        if np.array_equal(new_inliers, inliers):
            break
        inliers = new_inliers
    res = residuals(sim)
    return RansacResult(sim=sim, inliers=res < threshold, residuals=res, iterations=iterations)
