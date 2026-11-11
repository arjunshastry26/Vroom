"""Dry run of cloud/monst3r_worker.py against a real MonST3R checkout.

Skipped unless PyTorch is installed and the environment variable MONST3R_ROOT
points to a MonST3R checkout (cloned with --recursive). A tiny, randomly
initialised network replaces the pretrained weights, so this exercises every
MonST3R call the worker makes (image loading, pair graph, inference, global
alignment, result accessors), the artifact writing and the resume logic on
the CPU in under a minute. The geometry it produces is meaningless; this is a
plumbing test and says nothing about reconstruction quality.
"""

from __future__ import annotations

import importlib.util
import os
from math import inf
from pathlib import Path

import cv2
import numpy as np
import pytest

from vroom_interlagos.common import read_json, write_json
from vroom_interlagos.reconstruction.artifacts import validate_run
from vroom_interlagos.reconstruction.package import export_package
from vroom_interlagos.synthetic import _texture

ROOT = Path(__file__).resolve().parents[1]
MONST3R_ROOT = os.environ.get("MONST3R_ROOT")

pytestmark = pytest.mark.skipif(
    MONST3R_ROOT is None or importlib.util.find_spec("torch") is None,
    reason="needs PyTorch and MONST3R_ROOT pointing to a MonST3R checkout",
)


def _worker():
    spec = importlib.util.spec_from_file_location("monst3r_worker", ROOT / "cloud" / "monst3r_worker.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("batchify", [True, False])  # False = MonST3R's memory-saving mode (the OOM fallback)
def test_worker_dry_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, batchify: bool) -> None:
    monkeypatch.chdir(tmp_path)  # the worker changes directory; keep that contained
    worker = _worker()

    frames = tmp_path / "frames"
    frames.mkdir()
    texture = _texture(np.random.default_rng(0), 900, 288, (1.0, 1.0, 1.0))
    n = 7
    for i in range(n):
        cv2.imwrite(str(frames / f"frame_{i:06d}.jpg"), np.ascontiguousarray(texture[:, i * 20 : i * 20 + 512]))

    def chunk(cid: str, idx: list[int]) -> dict:
        return {"chunk_id": cid, "segment_id": 0, "frame_indices": idx, "n_frames": len(idx),
                "frame_files": [f"frame_{i:06d}.jpg" for i in idx], "timestamps_s": [i / 3 for i in idx],
                "start_timestamp_s": idx[0] / 3, "end_timestamp_s": idx[-1] / 3}

    manifest = {"video_id": "dryrun", "strategy": "fixed", "config_hash": "00000000",
                "chunks": [chunk("chunk_001", [0, 1, 2, 3]), chunk("chunk_002", [3, 4, 5, 6])]}
    write_json(tmp_path / "chunks.json", manifest)
    cfg = worker.merge_config(worker.DEFAULT_CONFIG, {"alignment": {"niter": 15, "batchify": batchify}, "inference": {"window_size": 2}})
    package = tmp_path / "pkg"
    export_package([tmp_path / "chunks.json"], frames, package, ROOT / "cloud" / "monst3r_worker.py", cfg, make_zip=False)

    api = worker.import_monst3r(Path(MONST3R_ROOT))
    import torch

    torch.manual_seed(0)
    model = api["AsymmetricCroCo3DStereo"](
        pos_embed="RoPE100", patch_embed_cls="ManyAR_PatchEmbed", img_size=(512, 512), head_type="linear",
        output_mode="pts3d", depth_mode=("exp", -inf, inf), conf_mode=("exp", 1, inf), landscape_only=False,
        enc_embed_dim=64, enc_depth=2, enc_num_heads=4, dec_embed_dim=64, dec_depth=2, dec_num_heads=4,
    )
    model.worker_model_id = "random-init tiny network (dry run, not a reconstruction)"

    def args(**extra):
        return worker.build_parser().parse_args(
            ["--package", str(package), "--manifest", "fixed_00000000", "--output", str(tmp_path / "out"),
             "--monst3r-root", MONST3R_ROOT, "--allow-cpu", *sum(([f"--{k}", str(v)] for k, v in extra.items()), [])]
        )

    assert worker.run(args(), model=model) == 0
    run_dir = next((tmp_path / "out" / "dryrun").iterdir())
    reports = validate_run(run_dir, manifest)
    assert [r.status for r in reports] == ["valid", "valid"], [r.errors for r in reports]
    assert reports[0].stats["resolution"] == [288, 512] and reports[0].stats["n_frames"] == 4
    assert reports[0].stats["pointmap_consistency"] < 1e-5  # the worker's arrays agree with each other

    meta = read_json(run_dir / "chunk_001" / "metadata.json")
    assert meta["model"].startswith("random-init") and meta["timestamps_s"] == [0.0, 1 / 3, 2 / 3, 1.0]
    assert meta["input_resolution"] == [288, 512] and meta["n_pairs"] > 0 and meta["runtime_s"] > 0
    assert set(meta["files"]) == {"camera_poses.npy", "intrinsics.npy", "depth.npy", "confidence.npy",
                                  "init_confidence.npy", "pointmaps.npy"}
    assert meta["dynamic_masks"].startswith("not computed")

    # Second run: nothing is recomputed.
    before = (run_dir / "chunk_001" / "depth.npy").stat().st_mtime_ns
    assert worker.run(args(), model=model) == 0
    assert (run_dir / "chunk_001" / "depth.npy").stat().st_mtime_ns == before
    summary = read_json(run_dir / "run_summary.json")
    assert all("skipped" in c["note"] for c in summary["chunks"].values())

    # An interrupted chunk (no metadata) is redone; a fresh output folder can be restored via --resume-from.
    (run_dir / "chunk_002" / "metadata.json").unlink()
    assert worker.run(args(), model=model) == 0
    assert read_json(run_dir / "chunk_002" / "metadata.json")["status"] == "completed"
    out2 = tmp_path / "out2"
    resumed = worker.build_parser().parse_args(
        ["--package", str(package), "--manifest", "fixed_00000000", "--output", str(out2),
         "--monst3r-root", MONST3R_ROOT, "--allow-cpu", "--resume-from", str(tmp_path / "out")])
    assert worker.run(resumed, model=model) == 0
    summary2 = read_json(next((out2 / "dryrun").iterdir()) / "run_summary.json")
    assert all("resume" in c["note"] for c in summary2["chunks"].values())
