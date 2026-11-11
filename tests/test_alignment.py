from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from conftest import make_motion_rows, make_records, scene_frame, write_scene_run
from vroom_interlagos.alignment.pairwise import AcceptConfig, AlignmentConfig, PairAlignment, align_pair
from vroom_interlagos.alignment.sim3 import Sim3, ransac_sim3, rotation_angle_deg, umeyama
from vroom_interlagos.alignment.stitch import run_alignment, sequential_stitch
from vroom_interlagos.chunking.chunkers import ChunkingConfig, build_chunk_manifest
from vroom_interlagos.common import dataclass_from_dict, load_yaml, read_json
from vroom_interlagos.reconstruction.artifacts import load_chunk_reconstruction

ROOT = Path(__file__).resolve().parents[1]
CHUNK_CFG = ChunkingConfig(min_frames=10, max_frames=30, target_frames=20, overlap_frames=4)
# The constructed scene is tiny (36x64 pixels), so the count thresholds are lowered for the tests.
CFG = AlignmentConfig(pixel_stride=2, accept=AcceptConfig(min_correspondences=200, min_inliers=150))


def _random_sim(rng: np.random.Generator) -> Sim3:
    R, _ = cv2.Rodrigues(rng.normal(0, 1.0, 3))
    return Sim3(float(rng.uniform(0.2, 5.0)), R, rng.normal(0, 10, 3))


def _manifest(n: int = 70, cuts: tuple[int, ...] = ()) -> dict:
    records = make_records(n, cuts=cuts)
    return build_chunk_manifest(records, CHUNK_CFG, "fixed", motion_rows=make_motion_rows(records), video_id="scene")


def _sim_close(a: Sim3, b: Sim3, rot_deg: float, rel: float) -> None:
    assert a.s == pytest.approx(b.s, rel=rel)
    assert rotation_angle_deg(a.R.T @ b.R) < rot_deg
    assert np.linalg.norm(a.t - b.t) < rel * (np.linalg.norm(b.t) + b.s * 40)


# ------------------------------------------------------------------- Sim(3)

def test_sim3_algebra() -> None:
    rng = np.random.default_rng(0)
    A, B = _random_sim(rng), _random_sim(rng)
    pts = rng.normal(size=(50, 3))
    assert np.allclose(A.compose(B).apply(pts), A.apply(B.apply(pts)))
    assert np.allclose(A.inverse().apply(A.apply(pts)), pts)
    assert np.allclose((A.matrix() @ np.c_[pts, np.ones(50)].T).T[:, :3], A.apply(pts))
    assert np.allclose(Sim3.from_dict(A.to_dict()).apply(pts), A.apply(pts))

    pose = np.eye(4)
    pose[:3, :3], _ = cv2.Rodrigues(np.array([0.1, -0.4, 0.2]))
    pose[:3, 3] = [1.0, 2.0, 3.0]
    moved = A.transform_pose(pose)
    assert np.allclose(moved[:3, :3] @ moved[:3, :3].T, np.eye(3), atol=1e-12)  # still a rigid pose
    assert np.allclose(moved[:3, 3], A.apply(pose[:3, 3]))
    # A point seen at depth d along the camera's z axis lands at depth s*d along the moved camera's z axis.
    assert np.allclose(A.apply(pose[:3, 3] + 4.0 * pose[:3, 2]), moved[:3, 3] + A.s * 4.0 * moved[:3, 2])


def test_umeyama_recovers_similarity_and_detects_degeneracy() -> None:
    rng = np.random.default_rng(1)
    S = _random_sim(rng)
    src = rng.normal(size=(200, 3))
    est = umeyama(src, S.apply(src))
    _sim_close(est, S, rot_deg=1e-5, rel=1e-9)
    assert umeyama(src[:2], S.apply(src[:2])) is None
    line = np.outer(np.linspace(0, 1, 20), [1.0, 2.0, 3.0])
    assert umeyama(line, S.apply(line)) is None  # collinear: rotation about the line is undetermined
    rigid = umeyama(src, src @ S.R.T + S.t, with_scale=False)
    assert rigid.s == 1.0 and rotation_angle_deg(rigid.R.T @ S.R) < 1e-5


def test_ransac_sim3_is_robust_to_outliers() -> None:
    rng = np.random.default_rng(2)
    S = _random_sim(rng)
    src = rng.normal(size=(2000, 3)) * 5
    dst = S.apply(src) + rng.normal(0, 0.002 * S.s, (2000, 3))
    outliers = rng.random(2000) < 0.45
    dst[outliers] = rng.normal(size=(int(outliers.sum()), 3)) * 30
    fit = ransac_sim3(src, dst, threshold=0.05 * S.s, iterations=300, seed=0)
    _sim_close(fit.sim, S, rot_deg=0.1, rel=2e-3)
    assert (fit.inliers == ~outliers).mean() > 0.99
    plain = umeyama(src, dst)
    assert rotation_angle_deg(plain.R.T @ S.R) > 1.0  # least squares alone is thrown off
    assert ransac_sim3(src[:2], dst[:2], 0.1) is None


# ----------------------------------------------------------------- pairwise

def test_align_pair_recovers_known_transform(tmp_path: Path) -> None:
    manifest = _manifest()
    truth = write_scene_run(tmp_path, manifest)
    a, b = (load_chunk_reconstruction(tmp_path / cid) for cid in ("chunk_001", "chunk_002"))
    edge = align_pair(a, b, CFG)
    assert edge.accepted, edge.rejection_reason
    expected = truth["chunk_001"].compose(truth["chunk_002"].inverse())  # chunk b -> world -> chunk a
    _sim_close(edge.sim, expected, rot_deg=0.01, rel=1e-4)
    assert edge.shared_frames == manifest["chunks"][0]["overlap"]["next"]["frame_indices"]
    assert edge.inlier_ratio > 0.99 and edge.rmse_rel < 1e-4
    assert edge.pose_rotation_error_deg < 0.01 and edge.pose_centre_error_rel < 1e-3
    assert 0.0 < edge.confidence <= 1.0
    restored = PairAlignment.from_dict(edge.to_dict())
    assert np.allclose(restored.sim.matrix(), edge.sim.matrix()) and restored.accepted


def test_align_pair_with_depth_noise(tmp_path: Path) -> None:
    manifest = _manifest()
    truth = write_scene_run(tmp_path, manifest, depth_noise=0.01)
    a, b = (load_chunk_reconstruction(tmp_path / cid) for cid in ("chunk_002", "chunk_003"))
    edge = align_pair(a, b, CFG)
    assert edge.accepted, edge.rejection_reason
    _sim_close(edge.sim, truth["chunk_002"].compose(truth["chunk_003"].inverse()), rot_deg=0.5, rel=0.01)
    assert 0.005 < edge.rmse_rel < 0.03


def test_align_pair_rejects_unrelated_geometry(tmp_path: Path) -> None:
    manifest = _manifest()
    write_scene_run(tmp_path, manifest)
    a = load_chunk_reconstruction(tmp_path / "chunk_001")
    b = load_chunk_reconstruction(tmp_path / "chunk_002", mmap=False)
    b.pointmaps[:] = np.random.default_rng(0).normal(0, 20, b.pointmaps.shape)
    edge = align_pair(a, b, CFG)
    assert not edge.accepted and edge.rejection_reason
    c = load_chunk_reconstruction(tmp_path / "chunk_003")
    far = align_pair(a, c, CFG)
    assert not far.accepted and far.rejection_reason == "no shared frames"
    low = load_chunk_reconstruction(tmp_path / "chunk_002", mmap=False)
    low.confidence[:] = 1.0
    assert "confident correspondences" in align_pair(a, low, CFG).rejection_reason


def test_camera_fixed_region_is_handled_by_mask(tmp_path: Path) -> None:
    """Pixels fixed to the camera (bodywork) are not world points. A minority is outvoted by RANSAC;
    a mask removes them from the estimate altogether."""
    manifest = _manifest()
    truth = write_scene_run(tmp_path, manifest)
    recs = {cid: load_chunk_reconstruction(tmp_path / cid, mmap=False) for cid in ("chunk_001", "chunk_002")}
    for rec in recs.values():  # bottom 40% of each image: a blob at fixed position in the camera frame
        for i in range(len(rec.frame_indices)):
            pose = rec.camera_poses[i].astype(np.float64)
            _, K, _ = scene_frame(0)
            u, v = np.meshgrid(np.arange(64.0), np.arange(22.0, 36.0))
            cam = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1) * 0.8
            rec.pointmaps[i, 22:] = cam @ pose[:3, :3].T + pose[:3, 3]
    expected = truth["chunk_001"].compose(truth["chunk_002"].inverse())
    unmasked = align_pair(recs["chunk_001"], recs["chunk_002"], CFG)
    assert unmasked.inlier_ratio < 0.7
    mask = np.ones((36, 64), bool)
    mask[22:] = False
    masked = align_pair(recs["chunk_001"], recs["chunk_002"], CFG, mask)
    assert masked.accepted and masked.inlier_ratio > 0.99 and masked.n_correspondences < unmasked.n_correspondences
    _sim_close(masked.sim, expected, rot_deg=0.01, rel=1e-4)


# ---------------------------------------------------------------- stitching

def test_sequential_stitch_chains_and_breaks() -> None:
    def edge(a: str, b: str, ok: bool) -> PairAlignment:
        return PairAlignment(a, b, [0], accepted=ok, sim=Sim3(2.0, np.eye(3), np.array([1.0, 0, 0])))

    ids = ["c1", "c2", "c3", "c4", "c5"]
    out = sequential_stitch(ids, [edge("c1", "c2", True), edge("c2", "c3", True), edge("c3", "c4", False)])
    assert [out[c]["component"] for c in ids] == [0, 0, 0, 1, 2]  # rejected edge and missing edge both break
    assert out["c3"]["sim"].s == 4.0 and np.allclose(out["c3"]["sim"].t, [3.0, 0, 0])
    assert out["c4"]["sim"].s == 1.0


def test_run_alignment_reproduces_world_trajectory(tmp_path: Path) -> None:
    manifest = _manifest()
    truth = write_scene_run(tmp_path / "run", manifest, depth_noise=0.002)
    report = run_alignment(tmp_path / "run", manifest, CFG, tmp_path / "out")
    s = report["summary"]
    assert s["n_accepted"] == s["n_edges"] == len(manifest["chunks"]) - 1 and s["n_components"] == 1
    assert s["n_trajectory_frames"] == 70 and s["overlap_centre_distance_over_median_step"] < 0.05

    data = np.load(tmp_path / "out" / "trajectory_sequential.npz")
    assert list(data["frame_indices"]) == list(range(70))
    first = truth["chunk_001"]  # the stitched frame is chunk 1's frame
    expected = np.asarray([first.apply(scene_frame(i)[0][:3, 3]) for i in range(70)])
    error = np.linalg.norm(data["poses"][:, :3, 3] - expected, axis=1)
    step = np.linalg.norm(np.diff(expected, axis=0), axis=1).mean()
    assert error.max() < 0.1 * step  # small drift after three chained edges
    assert error[-1] > error[5]  # sequential stitching accumulates error along the chain

    saved = read_json(tmp_path / "out" / "alignment.json")
    assert saved["chunk_transforms"]["chunk_001"]["scale"] == 1.0 and len(saved["edges"]) == s["n_edges"]
    assert (tmp_path / "out" / "trajectory_sequential.csv").read_text().count("\n") == 71
    assert (tmp_path / "out" / "trajectory_sequential.png").stat().st_size > 5000
    ply = (tmp_path / "out" / "pointcloud_sequential_component0.ply").read_bytes()
    n = s["pointcloud_points_component0"]
    assert n > 1000 and f"element vertex {n}".encode() in ply[:200]
    assert len(ply) == ply.index(b"end_header\n") + len(b"end_header\n") + 15 * n


def test_run_alignment_with_broken_chain(tmp_path: Path) -> None:
    manifest = _manifest(n=80, cuts=(40,))  # a scene cut: no overlap, hence no edge, across it
    write_scene_run(tmp_path / "run", manifest)
    (tmp_path / "run" / "chunk_002" / "metadata.json").unlink()  # and one chunk that never finished
    report = run_alignment(tmp_path / "run", manifest, CFG, tmp_path / "out", write_pointcloud=False)
    s = report["summary"]
    assert s["n_components"] >= 3 and s["n_accepted"] < s["n_edges"]
    assert any("missing or failed" in (e["rejection_reason"] or "") for e in report["edges"])
    comps = np.load(tmp_path / "out" / "trajectory_sequential.npz")["components"]
    assert len(set(comps.tolist())) >= 2


def test_alignment_yaml_matches_defaults() -> None:
    cfg = dataclass_from_dict(AlignmentConfig, load_yaml(ROOT / "configs" / "alignment.yaml"))
    assert cfg == AlignmentConfig()
