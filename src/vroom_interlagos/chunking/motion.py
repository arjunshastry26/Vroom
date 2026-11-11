"""Camera-motion analysis between consecutive extracted frames.

For every consecutive pair of usable frames the following is measured:

* dense optical flow (Farneback): magnitude and spatial variance;
* sparse tracking quality (Shi-Tomasi + pyramidal Lucas-Kanade with a
  forward-backward consistency check);
* feature correspondences (ORB + ratio test) and, from them, the relative
  camera rotation via the essential matrix;
* a rotation-compensated parallax (pixels), a proxy for translation;
* image sharpness (variance of the Laplacian).

Caveats that matter for interpretation:

* Intrinsics of broadcast onboard cameras are unknown, so a pinhole camera
  with an *assumed* horizontal field of view is used. Rotation angles are
  therefore approximate and should be read as a relative signal.
* The translation *magnitude* between two monocular views is unobservable;
  only its direction and the parallax proxy are reported.
* Essential-matrix estimation is degenerate for (near-)pure rotation and for
  planar scenes; ``pose_valid`` and ``inlier_ratio`` expose when the estimate
  should not be trusted. Optical flow alone is never interpreted as camera
  motion.
"""

from __future__ import annotations

import csv
import logging
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np

from vroom_interlagos.common import utc_now_iso, write_json
from vroom_interlagos.preprocessing.frames import FrameRecord, load_frame_table

logger = logging.getLogger(__name__)

MOTION_CSV = "motion.csv"
MOTION_META_JSON = "motion_meta.json"

INT_COLUMNS = ("frame_a", "frame_b", "continuous", "lk_n_init", "lk_n_tracked", "n_matches", "n_inliers", "pose_valid")
COLUMNS = (
    "frame_a", "frame_b", "t_a", "t_b", "dt", "continuous",
    "flow_mag_mean", "flow_mag_median", "flow_mag_std",
    "lk_n_init", "lk_n_tracked", "lk_survival", "lk_fb_error",
    "n_matches", "n_inliers", "inlier_ratio", "pose_valid",
    "rot_deg", "yaw_deg", "parallax_px",
    "sharpness_a", "sharpness_b",
)


@dataclass
class DenseFlowConfig:
    scale: float = 0.5  # flow is computed on a downscaled copy; magnitudes are reported at full size
    pyr_scale: float = 0.5
    levels: int = 4
    winsize: int = 21
    iterations: int = 3
    poly_n: int = 7
    poly_sigma: float = 1.5


@dataclass
class TrackingConfig:
    max_corners: int = 400
    quality_level: float = 0.01
    min_distance: int = 7
    win_size: int = 21
    pyramid_levels: int = 3
    fb_threshold_px: float = 1.5  # forward-backward error above this => track lost


@dataclass
class PoseConfig:
    orb_features: int = 1500
    ratio_test: float = 0.75
    assumed_hfov_deg: float = 80.0  # unknown for broadcast footage; see module docstring
    ransac_threshold_px: float = 1.0
    ransac_confidence: float = 0.999
    min_matches: int = 20
    min_inliers: int = 15


@dataclass
class MotionConfig:
    drop_duplicates: bool = True  # duplicate frames carry no motion information
    mask_path: str | None = None  # optional image, non-zero = pixels to analyse (e.g. exclude car body)
    num_threads: int = 0
    seed: int = 0  # OpenCV RNG seed (RANSAC) for reproducibility
    dense_flow: DenseFlowConfig = field(default_factory=DenseFlowConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    pose: PoseConfig = field(default_factory=PoseConfig)


@dataclass
class PoseEstimate:
    valid: bool
    n_inliers: int = 0
    inlier_ratio: float = math.nan
    rot_deg: float = math.nan
    yaw_deg: float = math.nan
    parallax_px: float = math.nan
    rotation: np.ndarray | None = None
    translation_dir: np.ndarray | None = None


def intrinsics_from_hfov(width: int, height: int, hfov_deg: float) -> np.ndarray:
    """Pinhole intrinsics with square pixels and a centred principal point."""
    focal = 0.5 * width / math.tan(math.radians(hfov_deg) / 2.0)
    return np.array([[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]])


def sharpness(gray: np.ndarray) -> float:
    """Variance of the Laplacian; lower means blurrier."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def dense_flow_stats(
    gray_a: np.ndarray, gray_b: np.ndarray, cfg: DenseFlowConfig, mask: np.ndarray | None = None
) -> dict[str, float]:
    """Mean / median / standard deviation of dense-flow magnitude (full-size pixels)."""
    if cfg.scale != 1.0:
        size = (max(16, int(round(gray_a.shape[1] * cfg.scale))), max(16, int(round(gray_a.shape[0] * cfg.scale))))
        small_a = cv2.resize(gray_a, size, interpolation=cv2.INTER_AREA)
        small_b = cv2.resize(gray_b, size, interpolation=cv2.INTER_AREA)
        small_mask = None if mask is None else cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
    else:
        small_a, small_b, small_mask = gray_a, gray_b, mask
    flow = cv2.calcOpticalFlowFarneback(
        small_a, small_b, None, cfg.pyr_scale, cfg.levels, cfg.winsize, cfg.iterations, cfg.poly_n, cfg.poly_sigma, 0
    )
    mag = np.linalg.norm(flow, axis=2) / cfg.scale
    if small_mask is not None and np.any(small_mask):
        mag = mag[small_mask > 0]
    return {
        "flow_mag_mean": float(mag.mean()),
        "flow_mag_median": float(np.median(mag)),
        "flow_mag_std": float(mag.std()),
    }


def lk_tracking_stats(
    gray_a: np.ndarray, gray_b: np.ndarray, cfg: TrackingConfig, mask: np.ndarray | None = None
) -> dict[str, float]:
    """Fraction of corners that survive forward-backward Lucas-Kanade tracking."""
    corners = cv2.goodFeaturesToTrack(
        gray_a, maxCorners=cfg.max_corners, qualityLevel=cfg.quality_level, minDistance=cfg.min_distance, mask=mask
    )
    n_init = 0 if corners is None else len(corners)
    if n_init == 0:
        return {"lk_n_init": 0, "lk_n_tracked": 0, "lk_survival": 0.0, "lk_fb_error": math.nan}
    params = dict(
        winSize=(cfg.win_size, cfg.win_size),
        maxLevel=cfg.pyramid_levels,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    forward, st_f, _ = cv2.calcOpticalFlowPyrLK(gray_a, gray_b, corners, None, **params)
    backward, st_b, _ = cv2.calcOpticalFlowPyrLK(gray_b, gray_a, forward, None, **params)
    fb_error = np.linalg.norm((corners - backward).reshape(-1, 2), axis=1)
    found = (st_f.ravel() == 1) & (st_b.ravel() == 1)
    good = found & (fb_error < cfg.fb_threshold_px)
    return {
        "lk_n_init": n_init,
        "lk_n_tracked": int(good.sum()),
        "lk_survival": float(good.sum() / n_init),
        "lk_fb_error": float(np.median(fb_error[found])) if found.any() else math.nan,
    }


def orb_correspondences(
    gray_a: np.ndarray, gray_b: np.ndarray, cfg: PoseConfig, mask: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """ORB matches that pass Lowe's ratio test, as two ``(N, 2)`` pixel arrays."""
    orb = cv2.ORB_create(nfeatures=cfg.orb_features)
    kp_a, des_a = orb.detectAndCompute(gray_a, mask)
    kp_b, des_b = orb.detectAndCompute(gray_b, mask)
    empty = np.empty((0, 2), dtype=np.float64)
    if des_a is None or des_b is None or len(kp_a) < 2 or len(kp_b) < 2:
        return empty, empty
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    pts_a, pts_b = [], []
    for pair in matcher.knnMatch(des_a, des_b, k=2):
        if len(pair) == 2 and pair[0].distance < cfg.ratio_test * pair[1].distance:
            pts_a.append(kp_a[pair[0].queryIdx].pt)
            pts_b.append(kp_b[pair[0].trainIdx].pt)
    if not pts_a:
        return empty, empty
    return np.asarray(pts_a, dtype=np.float64), np.asarray(pts_b, dtype=np.float64)


def estimate_relative_pose(pts_a: np.ndarray, pts_b: np.ndarray, K: np.ndarray, cfg: PoseConfig) -> PoseEstimate:
    """Relative pose from correspondences via the essential matrix.

    Convention (OpenCV ``recoverPose``): a point expressed in camera A maps to
    camera B as ``X_b = R @ X_a + t``. ``t`` is a unit vector: its scale is not
    observable from two monocular views.
    """
    n = len(pts_a)
    if n < max(cfg.min_matches, 5):
        return PoseEstimate(valid=False)
    try:
        E, inliers = cv2.findEssentialMat(
            pts_a, pts_b, K, method=cv2.RANSAC, prob=cfg.ransac_confidence, threshold=cfg.ransac_threshold_px
        )
    except cv2.error as exc:
        logger.debug("findEssentialMat failed: %s", exc)
        return PoseEstimate(valid=False)
    if E is None or E.shape != (3, 3) or inliers is None:
        return PoseEstimate(valid=False)
    n_cheirality, R, t, pose_mask = cv2.recoverPose(E, pts_a, pts_b, K, mask=inliers.copy())
    n_inliers = int(inliers.sum())
    ratio = n_inliers / n
    sel = pose_mask.ravel() > 0
    if n_cheirality < cfg.min_inliers or not sel.any():
        return PoseEstimate(valid=False, n_inliers=n_inliers, inlier_ratio=ratio)

    rvec, _ = cv2.Rodrigues(R)
    # Parallax: displacement left after removing the rotation-induced image motion
    # (infinite homography H = K R K^-1). Zero for a purely rotating camera.
    H = K @ R @ np.linalg.inv(K)
    homog = np.hstack([pts_a[sel], np.ones((int(sel.sum()), 1))]) @ H.T
    predicted = homog[:, :2] / homog[:, 2:3]
    parallax = float(np.median(np.linalg.norm(pts_b[sel] - predicted, axis=1)))
    return PoseEstimate(
        valid=True,
        n_inliers=n_inliers,
        inlier_ratio=ratio,
        rot_deg=float(np.degrees(np.linalg.norm(rvec))),
        yaw_deg=float(np.degrees(rvec[1, 0])),  # rotation about the camera's vertical (y) axis
        parallax_px=parallax,
        rotation=R,
        translation_dir=t.ravel(),
    )


def usable_frames(records: list[FrameRecord], drop_duplicates: bool = True) -> list[FrameRecord]:
    """Frames that take part in motion analysis and chunking."""
    return [r for r in records if not r.is_invalid and not (drop_duplicates and r.is_duplicate)]


def is_continuous(records: list[FrameRecord], a: FrameRecord, b: FrameRecord) -> bool:
    """False when a scene cut, timestamp gap or invalid frame separates ``a`` and ``b``."""
    for rec in records[a.frame_index + 1 : b.frame_index + 1]:
        if rec.cut_before or rec.gap_before or rec.is_invalid:
            return False
    return True


def _load_mask(mask_path: str | None, shape: tuple[int, int]) -> np.ndarray | None:
    if mask_path is None:
        return None
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask image: {mask_path}")
    if mask.shape != shape:
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return (mask > 0).astype(np.uint8) * 255


def analyze_pair(
    gray_a: np.ndarray, gray_b: np.ndarray, K: np.ndarray, cfg: MotionConfig, mask: np.ndarray | None = None
) -> dict[str, float]:
    """All pairwise motion measurements for two greyscale frames."""
    row: dict[str, float] = {}
    row.update(dense_flow_stats(gray_a, gray_b, cfg.dense_flow, mask))
    row.update(lk_tracking_stats(gray_a, gray_b, cfg.tracking, mask))
    pts_a, pts_b = orb_correspondences(gray_a, gray_b, cfg.pose, mask)
    pose = estimate_relative_pose(pts_a, pts_b, K, cfg.pose)
    row.update(
        n_matches=len(pts_a),
        n_inliers=pose.n_inliers,
        inlier_ratio=pose.inlier_ratio if len(pts_a) else 0.0,
        pose_valid=int(pose.valid),
        rot_deg=pose.rot_deg,
        yaw_deg=pose.yaw_deg,
        parallax_px=pose.parallax_px,
    )
    return row


def iter_motion_rows(frames_dir: Path, cfg: MotionConfig) -> Iterator[dict[str, Any]]:
    """Yield one measurement row per consecutive pair of usable frames.

    Only two greyscale frames are held in memory at any time.
    """
    frames_dir = Path(frames_dir)
    records = load_frame_table(frames_dir)
    frames = usable_frames(records, cfg.drop_duplicates)
    if len(frames) < 2:
        raise ValueError(f"{frames_dir}: need at least 2 usable frames, found {len(frames)}")
    if cfg.num_threads > 0:
        cv2.setNumThreads(cfg.num_threads)
    cv2.setRNGSeed(cfg.seed)

    def load(rec: FrameRecord) -> np.ndarray:
        img = cv2.imread(str(frames_dir / rec.filename), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not read frame image: {frames_dir / rec.filename}")
        return img

    prev, prev_gray = frames[0], load(frames[0])
    prev_sharp = sharpness(prev_gray)
    mask = _load_mask(cfg.mask_path, prev_gray.shape)
    K = intrinsics_from_hfov(prev_gray.shape[1], prev_gray.shape[0], cfg.pose.assumed_hfov_deg)

    for cur in frames[1:]:
        cur_gray = load(cur)
        cur_sharp = sharpness(cur_gray)
        row: dict[str, Any] = {name: math.nan for name in COLUMNS}
        row.update(
            frame_a=prev.frame_index, frame_b=cur.frame_index,
            t_a=prev.timestamp_s, t_b=cur.timestamp_s, dt=cur.timestamp_s - prev.timestamp_s,
            sharpness_a=prev_sharp, sharpness_b=cur_sharp,
            lk_n_init=0, lk_n_tracked=0, n_matches=0, n_inliers=0, pose_valid=0,
        )
        row["continuous"] = int(is_continuous(records, prev, cur))
        if row["continuous"]:  # measurements across a cut or gap would be meaningless
            row.update(analyze_pair(prev_gray, cur_gray, K, cfg, mask))
        yield row
        prev, prev_gray, prev_sharp = cur, cur_gray, cur_sharp


def analyze_motion(frames_dir: Path, cfg: MotionConfig) -> list[dict[str, Any]]:
    """Run motion analysis and write ``motion.csv`` + ``motion_meta.json`` into ``frames_dir``."""
    frames_dir = Path(frames_dir)
    rows = list(iter_motion_rows(frames_dir, cfg))
    write_motion_table(frames_dir / MOTION_CSV, rows)
    write_json(
        frames_dir / MOTION_META_JSON,
        {
            "created_utc": utc_now_iso(),
            "config": asdict(cfg),
            "n_pairs": len(rows),
            "n_discontinuous_pairs": sum(1 for r in rows if not r["continuous"]),
            "n_pose_valid": sum(int(r["pose_valid"]) for r in rows),
            "opencv": cv2.__version__,
            "units": {"flow_mag_*": "pixels at extracted resolution", "rot_deg": "degrees (assumed intrinsics)"},
        },
    )
    logger.info("Motion analysis: %d pairs -> %s", len(rows), frames_dir / MOTION_CSV)
    return rows


def write_motion_table(path: Path, rows: list[dict[str, Any]]) -> None:
    with Path(path).open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(COLUMNS))
        writer.writeheader()
        for row in rows:
            out = {}
            for name in COLUMNS:
                value = row[name]
                if name in INT_COLUMNS:
                    out[name] = int(value)
                else:
                    out[name] = "" if not np.isfinite(value) else f"{value:.6f}"
            writer.writerow(out)


def load_motion_table(frames_dir: Path) -> list[dict[str, Any]]:
    """Read ``motion.csv``; missing numeric values become NaN."""
    path = Path(frames_dir) / MOTION_CSV
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; run scripts/analyze_motion.py first")
    rows: list[dict[str, Any]] = []
    with path.open("r", newline="", encoding="utf-8") as fh:
        for raw in csv.DictReader(fh):
            row: dict[str, Any] = {}
            for name in COLUMNS:
                text = raw.get(name, "")
                if name in INT_COLUMNS:
                    row[name] = int(text)
                else:
                    row[name] = float(text) if text != "" else math.nan
            rows.append(row)
    return rows
