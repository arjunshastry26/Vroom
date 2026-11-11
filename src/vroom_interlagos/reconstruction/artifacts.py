"""Loading and validation of per-chunk reconstruction artifacts.

The artifact format is defined by ``cloud/monst3r_worker.py`` (see its module
docstring). Validation checks what can be checked without trusting the
worker: checksums, shapes, numerical sanity, that poses are rigid transforms,
and that the stored point maps really are the depth maps unprojected with the
stored intrinsics and poses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from vroom_interlagos.common import read_json, sha256_file

METADATA = "metadata.json"
REQUIRED_ARRAYS = ("camera_poses", "intrinsics", "depth", "confidence")
OPTIONAL_ARRAYS = ("pointmaps", "init_confidence", "dynamic_masks")


@dataclass
class ValidationReport:
    chunk_id: str
    status: str  # "valid" | "invalid" | "failed_in_worker" | "missing"
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == "valid"


@dataclass
class ChunkReconstruction:
    """Arrays of one reconstructed chunk (memory-mapped unless ``mmap=False``)."""

    chunk_id: str
    metadata: dict[str, Any]
    camera_poses: np.ndarray  # (N, 4, 4) camera-to-chunk
    intrinsics: np.ndarray  # (N, 3, 3)
    depth: np.ndarray  # (N, H, W)
    confidence: np.ndarray  # (N, H, W)
    pointmaps: np.ndarray | None = None  # (N, H, W, 3)
    init_confidence: np.ndarray | None = None
    dynamic_masks: np.ndarray | None = None

    @property
    def frame_indices(self) -> list[int]:
        return list(self.metadata["frame_indices"])


def load_chunk_reconstruction(chunk_dir: Path, mmap: bool = True) -> ChunkReconstruction:
    """Load a completed chunk. Large arrays are memory-mapped by default to spare RAM."""
    chunk_dir = Path(chunk_dir)
    meta = read_json(chunk_dir / METADATA)
    if meta.get("status") != "completed":
        raise ValueError(f"{chunk_dir}: chunk status is {meta.get('status')!r}, not 'completed'")
    mode = "r" if mmap else None

    def load(name: str, required: bool) -> np.ndarray | None:
        path = chunk_dir / f"{name}.npy"
        if not path.is_file():
            if required:
                raise FileNotFoundError(path)
            return None
        return np.load(path, mmap_mode=mode)

    return ChunkReconstruction(
        chunk_id=meta["chunk_id"],
        metadata=meta,
        **{name: load(name, True) for name in REQUIRED_ARRAYS},
        **{name: load(name, False) for name in OPTIONAL_ARRAYS},
    )


def unproject_depth(depth: np.ndarray, K: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """Depth map (H, W) -> points (H, W, 3) in the chunk frame.

    Pinhole model with OpenCV axes; ``pose`` is camera-to-chunk.
    """
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    x = (u - K[0, 2]) / K[0, 0] * depth
    y = (v - K[1, 2]) / K[1, 1] * depth
    cam = np.stack([x, y, depth.astype(np.float64)], axis=-1)
    return cam @ pose[:3, :3].T.astype(np.float64) + pose[:3, 3].astype(np.float64)


def validate_chunk_dir(
    chunk_dir: Path,
    manifest_chunk: dict[str, Any] | None = None,
    check_checksums: bool = True,
    pointmap_rtol: float = 1e-3,
) -> ValidationReport:
    """Validate one chunk folder; optionally against its entry in the chunk manifest."""
    chunk_dir = Path(chunk_dir)
    report = ValidationReport(chunk_id=chunk_dir.name, status="invalid")
    meta_path = chunk_dir / METADATA
    if not meta_path.is_file():
        report.status = "missing"
        report.errors.append("metadata.json missing (chunk not processed or interrupted)")
        return report
    try:
        meta = read_json(meta_path)
    except ValueError as exc:
        report.errors.append(f"metadata.json unreadable: {exc}")
        return report
    report.chunk_id = meta.get("chunk_id", chunk_dir.name)
    report.stats.update(runtime_s=meta.get("runtime_s"), peak_gpu_memory_gb=meta.get("peak_gpu_memory_gb"))
    if meta.get("status") != "completed":
        report.status = "failed_in_worker"
        err = meta.get("error") or {}
        report.errors.append(f"worker reported failure: {err.get('type')}: {str(err.get('message'))[:200]}")
        return report

    for key in ("chunk_id", "frame_indices", "frame_files", "model", "reconstruction_config", "files", "gpu"):
        if key not in meta:
            report.errors.append(f"metadata lacks {key!r}")
    if manifest_chunk is not None:
        if meta.get("chunk_id") != manifest_chunk["chunk_id"]:
            report.errors.append(f"chunk_id {meta.get('chunk_id')} does not match manifest {manifest_chunk['chunk_id']}")
        if meta.get("frame_indices") != manifest_chunk["frame_indices"]:
            report.errors.append("frame_indices differ from the chunk manifest")

    files = meta.get("files", {})
    for name in REQUIRED_ARRAYS:
        if f"{name}.npy" not in files:
            report.errors.append(f"{name}.npy not listed in metadata")
    for fname, info in files.items():
        path = chunk_dir / fname
        if not path.is_file():
            report.errors.append(f"{fname} missing")
        elif check_checksums and sha256_file(path) != info.get("sha256"):
            report.errors.append(f"{fname} checksum mismatch (corrupted or partial download)")
    if report.errors:
        return report

    try:
        rec = load_chunk_reconstruction(chunk_dir)
    except (OSError, ValueError) as exc:
        report.errors.append(f"could not load arrays: {exc}")
        return report

    n = len(meta["frame_indices"])
    poses, K, depth, conf = rec.camera_poses, rec.intrinsics, rec.depth, rec.confidence
    if poses.shape != (n, 4, 4):
        report.errors.append(f"camera_poses shape {poses.shape}, expected {(n, 4, 4)}")
    if K.shape != (n, 3, 3):
        report.errors.append(f"intrinsics shape {K.shape}, expected {(n, 3, 3)}")
    if depth.ndim != 3 or depth.shape[0] != n:
        report.errors.append(f"depth shape {depth.shape}, expected ({n}, H, W)")
    if conf.shape != depth.shape:
        report.errors.append(f"confidence shape {conf.shape} differs from depth {depth.shape}")
    if rec.pointmaps is not None and rec.pointmaps.shape != (*depth.shape, 3):
        report.errors.append(f"pointmaps shape {rec.pointmaps.shape}, expected {(*depth.shape, 3)}")
    if rec.init_confidence is not None and rec.init_confidence.shape != depth.shape:
        report.errors.append("init_confidence shape differs from depth")
    if rec.dynamic_masks is not None and rec.dynamic_masks.shape != depth.shape:
        report.errors.append("dynamic_masks shape differs from depth")
    if report.errors:
        return report

    for name, arr in (("camera_poses", poses), ("intrinsics", K), ("depth", depth), ("confidence", conf)):
        if not np.isfinite(arr).all():
            report.errors.append(f"{name} contains NaN or infinite values")
    if report.errors:
        return report

    R = np.asarray(poses[:, :3, :3], dtype=np.float64)
    ortho_err = float(np.abs(R @ R.transpose(0, 2, 1) - np.eye(3)).max())
    dets = np.linalg.det(R)
    if ortho_err > 1e-3 or np.abs(dets - 1.0).max() > 1e-3:
        report.errors.append(f"camera rotations are not proper rotations (orthogonality error {ortho_err:.2e})")
    if not np.allclose(poses[:, 3], [0, 0, 0, 1], atol=1e-5):
        report.errors.append("camera_poses bottom rows are not [0, 0, 0, 1]")
    if (np.asarray(K[:, 0, 0]) <= 0).any() or (np.asarray(K[:, 1, 1]) <= 0).any():
        report.errors.append("non-positive focal length")
    if (np.asarray(depth) <= 0).any():
        report.errors.append("depth has non-positive values")
    if float(np.min(conf)) < 0:
        report.errors.append("confidence has negative values")

    centres = np.asarray(poses[:, :3, 3], dtype=np.float64)
    steps = np.linalg.norm(np.diff(centres, axis=0), axis=1)
    report.stats.update(
        n_frames=n,
        resolution=[int(depth.shape[1]), int(depth.shape[2])],
        focal_px=float(np.median(K[:, 0, 0])),
        depth_median=float(np.median(depth)),
        confidence_median=float(np.median(conf)),
        path_length=float(steps.sum()),
        final_loss=meta.get("final_loss"),
    )
    width = depth.shape[2]
    if not 0.2 * width <= report.stats["focal_px"] <= 20 * width:
        report.warnings.append(
            f"implausible focal length {report.stats['focal_px']:.3g} px for a {width} px wide image "
            "(reconstruction probably degenerate)"
        )
    if steps.size and float(steps.max()) < 1e-6:
        report.warnings.append("camera does not move within the chunk (degenerate trajectory)")
    if steps.size > 2 and float(steps.max()) > 20 * float(np.median(steps) + 1e-12):
        report.warnings.append("one camera step is >20x the median step (possible pose jump)")

    if rec.pointmaps is not None and not report.errors:
        # Point maps must equal the depth unprojected with the stored K and pose (checked on 3 frames).
        # The error is measured relative to each point's distance from the origin, because float32
        # precision scales with the magnitude of the coordinates.
        worst = 0.0
        for i in sorted({0, n // 2, n - 1}):
            expected = unproject_depth(np.asarray(depth[i]), np.asarray(K[i]), np.asarray(poses[i]))
            error = np.linalg.norm(expected - rec.pointmaps[i], axis=-1)
            worst = max(worst, float((error / (np.linalg.norm(expected, axis=-1) + 1e-9)).max()))
        report.stats["pointmap_consistency"] = worst
        if not np.isfinite(worst) or worst > pointmap_rtol:
            report.errors.append(
                f"pointmaps are inconsistent with depth/intrinsics/poses (relative error {worst:.3g})"
            )

    report.status = "invalid" if report.errors else "valid"
    return report


def validate_run(run_dir: Path, manifest: dict[str, Any], check_checksums: bool = True) -> list[ValidationReport]:
    """Validate every chunk of ``manifest`` inside a worker output folder."""
    return [
        validate_chunk_dir(Path(run_dir) / chunk["chunk_id"], chunk, check_checksums=check_checksums)
        for chunk in manifest["chunks"]
    ]
