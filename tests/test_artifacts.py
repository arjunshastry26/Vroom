from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from conftest import make_motion_rows, make_records
from vroom_interlagos.chunking.chunkers import ChunkingConfig, build_chunk_manifest
from vroom_interlagos.common import load_yaml, read_json, sha256_file, write_json
from vroom_interlagos.reconstruction.artifacts import (
    load_chunk_reconstruction, unproject_depth, validate_chunk_dir, validate_run,
)
from vroom_interlagos.reconstruction.package import export_package

ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "cloud" / "monst3r_worker.py"
CFG = ChunkingConfig(min_frames=10, max_frames=30, target_frames=20, overlap_frames=4)


def _load_worker():
    spec = importlib.util.spec_from_file_location("monst3r_worker", WORKER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _manifest(strategy: str = "fixed") -> dict:
    records = make_records(45)
    return build_chunk_manifest(records, CFG, strategy, motion_rows=make_motion_rows(records), video_id="vid")


def write_test_chunk(chunk_dir: Path, chunk: dict, h: int = 12, w: int = 16, seed: int = 0) -> None:
    """Write a geometrically consistent artifact folder in the worker's format (test data, not a reconstruction)."""
    rng = np.random.default_rng(seed)
    n = chunk["n_frames"]
    poses = np.tile(np.eye(4, dtype=np.float32), (n, 1, 1))
    for i in range(n):
        R, _ = cv2.Rodrigues(np.array([0.0, 0.02 * i, 0.0]))
        poses[i, :3, :3] = R
        poses[i, :3, 3] = [0.1 * i, 0.0, 0.5 * i]
    K = np.tile(np.array([[20.0, 0, w / 2], [0, 20.0, h / 2], [0, 0, 1]], dtype=np.float32), (n, 1, 1))
    depth = rng.uniform(2.0, 9.0, (n, h, w)).astype(np.float32)
    arrays = {
        "camera_poses": poses, "intrinsics": K, "depth": depth,
        "confidence": rng.uniform(1.0, 5.0, (n, h, w)).astype(np.float32),
        "pointmaps": np.stack([unproject_depth(depth[i], K[i], poses[i]) for i in range(n)]).astype(np.float32),
    }
    chunk_dir.mkdir(parents=True)
    files = {}
    for name, arr in arrays.items():
        np.save(chunk_dir / f"{name}.npy", arr)
        files[f"{name}.npy"] = {"sha256": sha256_file(chunk_dir / f"{name}.npy"), "shape": list(arr.shape), "dtype": str(arr.dtype)}
    write_json(chunk_dir / "metadata.json", {
        "chunk_id": chunk["chunk_id"], "status": "completed", "frame_indices": chunk["frame_indices"],
        "frame_files": chunk["frame_files"], "model": "test-data", "reconstruction_config": {},
        "reconstruction_config_hash": "testhash", "files": files, "gpu": {"available": False},
        "runtime_s": 1.0, "peak_gpu_memory_gb": None, "final_loss": 0.1, "error": None,
    })


def _rewrite(chunk_dir: Path, name: str, array: np.ndarray) -> None:
    np.save(chunk_dir / f"{name}.npy", array)
    meta = read_json(chunk_dir / "metadata.json")
    meta["files"][f"{name}.npy"]["sha256"] = sha256_file(chunk_dir / f"{name}.npy")
    write_json(chunk_dir / "metadata.json", meta)


def test_valid_chunk_passes_and_loads(tmp_path: Path) -> None:
    chunk = _manifest()["chunks"][0]
    write_test_chunk(tmp_path / chunk["chunk_id"], chunk)
    report = validate_chunk_dir(tmp_path / chunk["chunk_id"], chunk)
    assert report.ok, report.errors
    assert report.stats["n_frames"] == chunk["n_frames"] and report.stats["resolution"] == [12, 16]
    assert report.stats["pointmap_consistency"] < 1e-5 and report.warnings == []
    rec = load_chunk_reconstruction(tmp_path / chunk["chunk_id"])
    assert rec.camera_poses.shape == (chunk["n_frames"], 4, 4) and rec.frame_indices == chunk["frame_indices"]
    assert isinstance(rec.depth, np.memmap) and rec.dynamic_masks is None


def test_validation_detects_problems(tmp_path: Path) -> None:
    manifest = _manifest()
    chunk = manifest["chunks"][0]

    def fresh(name: str) -> Path:
        d = tmp_path / name / chunk["chunk_id"]
        write_test_chunk(d, chunk)
        return d

    assert validate_chunk_dir(tmp_path / "nothing" / "chunk_001").status == "missing"

    d = fresh("corrupt")
    with (d / "depth.npy").open("r+b") as fh:
        fh.seek(200)
        fh.write(b"\x00\x01\x02\x03")
    assert any("checksum" in e for e in validate_chunk_dir(d, chunk).errors)

    d = fresh("nan")
    depth = np.load(d / "depth.npy")
    depth[0, 0, 0] = np.nan
    _rewrite(d, "depth", depth)
    assert any("NaN" in e for e in validate_chunk_dir(d, chunk).errors)

    d = fresh("rotation")
    poses = np.load(d / "camera_poses.npy")
    poses[2, :3, :3] *= 1.3
    _rewrite(d, "camera_poses", poses)
    assert any("proper rotations" in e for e in validate_chunk_dir(d, chunk).errors)

    d = fresh("shape")
    _rewrite(d, "intrinsics", np.load(d / "intrinsics.npy")[:-1])
    assert any("intrinsics shape" in e for e in validate_chunk_dir(d, chunk).errors)

    d = fresh("pointmaps")
    _rewrite(d, "pointmaps", np.load(d / "pointmaps.npy") * 1.05)
    assert any("inconsistent" in e for e in validate_chunk_dir(d, chunk).errors)

    d = fresh("focal")
    K = np.load(d / "intrinsics.npy")
    K[:, 0, 0] = K[:, 1, 1] = 1e-4
    _rewrite(d, "intrinsics", K)
    _rewrite(d, "pointmaps", np.stack([
        unproject_depth(dm, k, pose)
        for dm, k, pose in zip(np.load(d / "depth.npy"), K, np.load(d / "camera_poses.npy"))]).astype(np.float32))
    report = validate_chunk_dir(d, chunk)
    assert report.ok and any("implausible focal" in w for w in report.warnings)

    d = fresh("frames")
    assert any("frame_indices" in e for e in validate_chunk_dir(d, manifest["chunks"][1]).errors)

    d = fresh("failed")
    meta = read_json(d / "metadata.json")
    meta.update(status="failed", error={"type": "out_of_memory", "message": "CUDA out of memory"})
    write_json(d / "metadata.json", meta)
    report = validate_chunk_dir(d, chunk)
    assert report.status == "failed_in_worker" and "out_of_memory" in report.errors[0]
    with pytest.raises(ValueError):
        load_chunk_reconstruction(d)


def test_validate_run_reports_every_chunk(tmp_path: Path) -> None:
    manifest = _manifest()
    for chunk in manifest["chunks"][:-1]:
        write_test_chunk(tmp_path / chunk["chunk_id"], chunk)
    statuses = [r.status for r in validate_run(tmp_path, manifest)]
    assert statuses == ["valid"] * (len(manifest["chunks"]) - 1) + ["missing"]


def test_export_package(tmp_path: Path) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    for i in range(45):
        cv2.imwrite(str(frames / f"frame_{i:06d}.jpg"), np.full((18, 32, 3), i, np.uint8))
    paths = []
    for strategy in ("fixed", "adaptive"):
        path = tmp_path / f"{strategy}.json"
        write_json(path, _manifest(strategy))
        paths.append(path)
    recon_cfg = load_yaml(ROOT / "configs" / "reconstruction.yaml")
    out = tmp_path / "pkg"
    info = export_package(paths, frames, out, WORKER, recon_cfg)

    assert info["n_frames"] == 45 and len(list((out / "frames").glob("*.jpg"))) == 45  # frames stored once
    assert len(info["manifests"]) == 2 and (out / "monst3r_worker.py").is_file()
    for name in info["manifests"]:
        assert read_json(out / "manifests" / f"{name}.json")["chunks"][0]["timestamps_s"][0] == 0.0
    assert read_json(out / "package.json")["frames_sha256"]["frame_000003.jpg"] == sha256_file(frames / "frame_000003.jpg")
    assert (tmp_path / "pkg.zip").stat().st_size > 1000
    with pytest.raises(FileExistsError):
        export_package(paths, frames, out, WORKER, recon_cfg)
    (frames / "frame_000010.jpg").unlink()
    with pytest.raises(FileNotFoundError):
        export_package(paths, frames, tmp_path / "pkg2", WORKER, recon_cfg)


def test_worker_config_and_resume_logic(tmp_path: Path) -> None:
    """Parts of the worker that need neither torch nor MonST3R."""
    worker = _load_worker()
    yaml_cfg = load_yaml(ROOT / "configs" / "reconstruction.yaml")
    cfg = worker.merge_config(worker.DEFAULT_CONFIG, yaml_cfg)
    assert cfg == worker.DEFAULT_CONFIG  # the YAML documents exactly the worker defaults
    assert worker.merge_config(worker.DEFAULT_CONFIG, {"alignment": {"niter": 50}})["alignment"]["niter"] == 50
    assert worker.DEFAULT_CONFIG["alignment"]["niter"] == 300  # defaults are not mutated
    with pytest.raises(ValueError, match="Unknown"):
        worker.merge_config(worker.DEFAULT_CONFIG, {"alignment": {"nitre": 50}})

    chunk = _manifest()["chunks"][0]
    d = tmp_path / chunk["chunk_id"]
    assert not worker.chunk_is_complete(d, chunk, "testhash")
    write_test_chunk(d, chunk)
    assert worker.chunk_is_complete(d, chunk, "testhash")
    assert not worker.chunk_is_complete(d, chunk, "otherhash")  # different settings -> recompute
    assert not worker.chunk_is_complete(d, {**chunk, "frame_indices": chunk["frame_indices"][1:]}, "testhash")
    (d / "depth.npy").write_bytes(b"truncated")
    assert not worker.chunk_is_complete(d, chunk, "testhash")  # partial file -> recompute
    json.loads((d / "metadata.json").read_text())


def test_trajectory_inspection(tmp_path: Path) -> None:
    from vroom_interlagos.reconstruction.inspect import plot_run, trajectory_stats

    manifest = _manifest()
    for chunk in manifest["chunks"]:
        write_test_chunk(tmp_path / chunk["chunk_id"], chunk)
    s = trajectory_stats(tmp_path / "chunk_001")
    # The test path moves 0.1 in x and 0.5 in z per frame, at constant speed.
    assert np.allclose(s["steps"], np.hypot(0.1, 0.5), atol=1e-5)
    assert s["max_over_median"] == pytest.approx(1.0, abs=1e-3) and s["near_static_fraction"] == 0.0
    assert np.allclose(s["heading_change_deg"], np.degrees(0.02), atol=1e-2)
    assert 0.5 < s["forward_alignment_median"] <= 1.0 and 0.9 < s["confident_fraction"] <= 1.0

    # A frozen stretch followed by one jump is reported as such.
    d = tmp_path / "chunk_001"
    poses = np.load(d / "camera_poses.npy")
    poses[:10, :3, 3] = poses[0, :3, 3] + np.arange(10)[:, None] * 1e-4
    _rewrite(d, "camera_poses", poses)
    s = trajectory_stats(d)
    assert s["largest_step_at"] == 9 and s["near_static_fraction"] > 0.4

    stats = plot_run(tmp_path, manifest, tmp_path / "traj.png")
    assert len(stats) == len(manifest["chunks"]) and (tmp_path / "traj.png").stat().st_size > 10_000
