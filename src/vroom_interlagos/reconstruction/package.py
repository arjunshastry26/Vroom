"""Build the self-contained package that is uploaded to Kaggle / Colab.

    <package>/
      package.json               what is inside, with checksums
      manifests/<name>.json      one or more chunk manifests of the same video
      frames/frame_000123.jpg    only the frames those manifests reference, stored once
      reconstruction.json        MonST3R settings (from configs/reconstruction.yaml)
      monst3r_worker.py          the worker script

Several manifests (e.g. fixed and adaptive) share one ``frames`` folder, so
the frames are uploaded once.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

from vroom_interlagos.common import read_json, sha256_file, utc_now_iso, write_json

logger = logging.getLogger(__name__)


def manifest_name(manifest: dict[str, Any]) -> str:
    return f"{manifest['strategy']}_{manifest['config_hash']}"


def export_package(
    manifest_paths: list[Path],
    frames_dir: Path,
    out_dir: Path,
    worker_script: Path,
    reconstruction_config: dict[str, Any],
    make_zip: bool = True,
) -> dict[str, Any]:
    """Create the upload package and (optionally) ``<out_dir>.zip``. Returns ``package.json``."""
    if not manifest_paths:
        raise ValueError("At least one chunk manifest is required")
    manifests = [read_json(p) for p in manifest_paths]
    video_ids = {m["video_id"] for m in manifests}
    if len(video_ids) != 1:
        raise ValueError(f"All manifests must belong to the same video, got {sorted(video_ids)}")
    names = [manifest_name(m) for m in manifests]
    if len(set(names)) != len(names):
        raise ValueError(f"Duplicate manifests: {names}")
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} already exists; choose another name or delete it")

    needed = sorted({f for m in manifests for c in m["chunks"] for f in c["frame_files"]})
    missing = [f for f in needed if not (frames_dir / f).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} frames referenced by the manifests are missing, e.g. {missing[0]}")
    if not worker_script.is_file():
        raise FileNotFoundError(f"Worker script not found: {worker_script}")

    (out_dir / "frames").mkdir(parents=True)
    (out_dir / "manifests").mkdir()
    frame_hashes: dict[str, str] = {}
    for name in needed:
        shutil.copy2(frames_dir / name, out_dir / "frames" / name)
        frame_hashes[name] = sha256_file(out_dir / "frames" / name)
    for name, manifest in zip(names, manifests):
        write_json(out_dir / "manifests" / f"{name}.json", manifest)
    write_json(out_dir / "reconstruction.json", reconstruction_config)
    shutil.copy2(worker_script, out_dir / "monst3r_worker.py")

    info = {
        "schema_version": 1,
        "video_id": video_ids.pop(),
        "created_utc": utc_now_iso(),
        "manifests": {
            name: {"strategy": m["strategy"], "config_hash": m["config_hash"], "n_chunks": len(m["chunks"]),
                   "max_chunk_frames": max(c["n_frames"] for c in m["chunks"])}
            for name, m in zip(names, manifests)
        },
        "n_frames": len(needed),
        "frames_sha256": frame_hashes,
        "worker_sha256": sha256_file(out_dir / "monst3r_worker.py"),
        "reconstruction_config": reconstruction_config,
    }
    write_json(out_dir / "package.json", info)
    if make_zip:
        archive = shutil.make_archive(str(out_dir), "zip", root_dir=out_dir.parent, base_dir=out_dir.name)
        logger.info("Zip for upload: %s (%.1f MB)", archive, Path(archive).stat().st_size / 2**20)
    return info
