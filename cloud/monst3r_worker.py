"""MonST3R reconstruction worker for Kaggle / Colab.

Reconstructs the chunks of one chunk manifest, one chunk at a time, with the
official MonST3R code (https://github.com/Junyi42/monst3r), and writes one
artifact folder per chunk. This file is self-contained on purpose: it is
copied into the upload package and must not import the local project.

It calls the same MonST3R functions, in the same order, as
``get_reconstructed_scene`` in MonST3R's ``demo.py``:

    load_images -> make_pairs -> inference -> global_aligner
    -> compute_global_alignment (or compute_window_wise_alignment)

and then reads the result through the optimizer's own accessors
(``get_im_poses``, ``get_intrinsics``, ``get_depthmaps``, ``get_pts3d``,
``im_conf``). ``demo.py`` itself is not used because its non-interactive mode
hard-codes the optimisation settings and only writes visualisation-oriented
files.

Usage (inside the MonST3R environment):

    python monst3r_worker.py --package /path/to/package --manifest adaptive_d70e3096 \
        --output /kaggle/working/reconstructions --monst3r-root /kaggle/working/monst3r

Artifact layout, per chunk (``N`` frames of ``H x W`` pixels):

    chunk_001/
      camera_poses.npy     (N, 4, 4) float32  camera-to-world, see conventions below
      intrinsics.npy       (N, 3, 3) float32  pinhole K at the model resolution
      depth.npy            (N, H, W) float32  optimised depth along the camera z axis
      pointmaps.npy        (N, H, W, 3) float32  per-pixel 3D points in the chunk frame
      confidence.npy       (N, H, W) float32  MonST3R confidence (>= 1), after optional cleaning
      init_confidence.npy  (N, H, W) float32  confidence straight from the network
      dynamic_masks.npy    (N, H, W) bool     only when the flow loss is enabled
      metadata.json        written last; a chunk without it is incomplete

Conventions (inherited from DUSt3R / MonST3R):

* Camera axes are OpenCV style: x right, y down, z forward.
* ``camera_poses[i]`` maps homogeneous points from camera ``i`` to the chunk
  frame: ``X_chunk = T_i @ X_cam``.
* Every chunk has its own arbitrary coordinate frame and its own arbitrary
  scale. Nothing here is metric, and two chunks are not comparable until they
  are aligned.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ARTIFACT_SCHEMA_VERSION = 1
METADATA = "metadata.json"
RUN_SUMMARY = "run_summary.json"
SWIN_GRAPHS = ("swin", "swinstride", "swin2stride")

DEFAULT_CONFIG: dict[str, Any] = {
    "model": {"weights": "Junyi42/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt"},
    "image_size": 512,
    "seed": 0,
    "inference": {"batch_size": 8, "scene_graph": "swinstride", "window_size": 5},
    "alignment": {
        "niter": 300,
        "schedule": "linear",
        "lr": 0.01,
        "shared_focal": True,
        "temporal_smoothing_weight": 0.01,
        "translation_weight": 1.0,
        "flow_loss_weight": 0.0,
        "flow_loss_start_iter": 0.1,
        "flow_loss_threshold": 25,
        "batchify": True,
        "window_wise": False,
        "window_size": 100,
        "window_overlap_ratio": 0.5,
    },
    "output": {"save_pointmaps": True, "clean_pointcloud": True},
}


# --------------------------------------------------------------------- helpers

def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, obj: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2)
        fh.write("\n")
    os.replace(tmp, path)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def merge_config(base: dict[str, Any], override: dict[str, Any] | None) -> dict[str, Any]:
    """Recursively overlay ``override`` on ``base``; unknown keys are an error."""
    out = json.loads(json.dumps(base))
    for key, value in (override or {}).items():
        if key not in out:
            raise ValueError(f"Unknown reconstruction config key: {key!r}")
        if isinstance(out[key], dict):
            if not isinstance(value, dict):
                raise ValueError(f"Config key {key!r} must be a mapping")
            out[key] = merge_config(out[key], value)
        else:
            out[key] = value
    return out


def config_hash(cfg: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode("utf-8")).hexdigest()[:8]


def git_commit(repo: Path) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def gpu_info(torch: Any) -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"available": False}
    props = torch.cuda.get_device_properties(0)
    return {
        "available": True,
        "name": props.name,
        "total_memory_gb": round(props.total_memory / 2**30, 2),
        "device_count": torch.cuda.device_count(),
        "cuda_version": torch.version.cuda,
    }


def free_memory(torch: Any) -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ------------------------------------------------------------ MonST3R plumbing

def import_monst3r(monst3r_root: Path) -> dict[str, Any]:
    """Make the MonST3R checkout importable and return the functions the worker uses.

    MonST3R refers to checkpoints and configs with paths relative to its
    repository root, so the working directory is changed to that root.
    """
    monst3r_root = monst3r_root.resolve()
    if not (monst3r_root / "dust3r" / "inference.py").is_file():
        raise FileNotFoundError(f"{monst3r_root} is not a MonST3R checkout (dust3r/inference.py missing)")
    if not (monst3r_root / "croco" / "models").is_dir():
        raise FileNotFoundError("The croco submodule is missing; clone MonST3R with --recursive")
    os.chdir(monst3r_root)
    for path in (monst3r_root / "third_party" / "sam2", monst3r_root):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))  # dust3r.cloud_opt imports sam2 unconditionally

    from dust3r.cloud_opt import GlobalAlignerMode, global_aligner
    from dust3r.image_pairs import make_pairs
    from dust3r.inference import inference
    from dust3r.model import AsymmetricCroCo3DStereo
    from dust3r.utils.image import load_images

    return {
        "GlobalAlignerMode": GlobalAlignerMode,
        "global_aligner": global_aligner,
        "make_pairs": make_pairs,
        "inference": inference,
        "AsymmetricCroCo3DStereo": AsymmetricCroCo3DStereo,
        "load_images": load_images,
    }


def check_flow_checkpoints(monst3r_root: Path, cfg: dict[str, Any]) -> None:
    """The flow loss needs RAFT and SAM2 checkpoints at fixed paths inside the checkout."""
    if cfg["alignment"]["flow_loss_weight"] <= 0:
        return
    needed = [
        monst3r_root / "third_party" / "RAFT" / "models" / "Tartan-C-T-TSKH-spring540x960-M.pth",
        monst3r_root / "third_party" / "sam2" / "checkpoints" / "sam2.1_hiera_large.pt",
    ]
    missing = [str(p) for p in needed if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "alignment.flow_loss_weight > 0 needs these checkpoints (see MonST3R data/download_ckpt.sh): "
            + ", ".join(missing)
        )


def reconstruct_chunk(api: dict[str, Any], model: Any, device: str, files: list[str], cfg: dict[str, Any]) -> dict[str, Any]:
    """Run MonST3R on an ordered list of frame files and return numpy results."""
    import torch

    inf_cfg, al = cfg["inference"], cfg["alignment"]
    imgs = api["load_images"](files, size=cfg["image_size"], verbose=False)
    loaded = [Path(im["instance"]).name for im in imgs]
    if loaded != [Path(f).name for f in files]:
        raise RuntimeError("MonST3R loaded the frames in a different order than the manifest lists them")
    if len(imgs) < 3:
        raise ValueError(f"A chunk needs at least 3 frames for global alignment, got {len(imgs)}")

    graph = inf_cfg["scene_graph"]
    if graph in SWIN_GRAPHS:
        graph = f"{graph}-{inf_cfg['window_size']}-noncyclic"  # as in MonST3R demo.py
    pairs = api["make_pairs"](imgs, scene_graph=graph, prefilter=None, symmetrize=True)
    output = api["inference"](pairs, model, device, batch_size=inf_cfg["batch_size"], verbose=False)
    n_pairs = len(pairs)
    del pairs

    window_wise = bool(al["window_wise"])
    scene = api["global_aligner"](
        output,
        device=device,
        mode=api["GlobalAlignerMode"].PointCloudOptimizer,
        verbose=False,
        shared_focal=al["shared_focal"],
        temporal_smoothing_weight=al["temporal_smoothing_weight"],
        translation_weight=float(al["translation_weight"]),
        flow_loss_weight=al["flow_loss_weight"],
        flow_loss_start_epoch=al["flow_loss_start_iter"],
        flow_loss_thre=al["flow_loss_threshold"],
        use_self_mask=True,
        num_total_iter=al["niter"],
        empty_cache=len(files) > 72,
        batchify=bool(al["batchify"]) and not window_wise,
        window_wise=window_wise,
        window_size=al["window_size"],
        window_overlap_ratio=al["window_overlap_ratio"],
        prev_video_results=None,
    )
    del output
    align = scene.compute_window_wise_alignment if window_wise else scene.compute_global_alignment
    loss = align(init="mst", niter=al["niter"], schedule=al["schedule"], lr=al["lr"])

    with torch.no_grad():
        init_conf = np.stack([c.detach().cpu().numpy() for c in scene.init_conf_maps]).astype(np.float32)
        if cfg["output"]["clean_pointcloud"]:
            scene = scene.clean_pointcloud()  # MonST3R demo default: lowers confidence of occluding floaters
        result: dict[str, Any] = {
            "camera_poses": scene.get_im_poses().detach().cpu().numpy().astype(np.float32),
            "intrinsics": scene.get_intrinsics().detach().cpu().numpy().astype(np.float32),
            "depth": np.stack([d.detach().cpu().numpy() for d in scene.get_depthmaps()]).astype(np.float32),
            "confidence": np.stack([c.detach().cpu().numpy() for c in scene.im_conf]).astype(np.float32),
            "init_confidence": init_conf,
        }
        if cfg["output"]["save_pointmaps"]:
            result["pointmaps"] = np.stack([p.detach().cpu().numpy() for p in scene.get_pts3d()]).astype(np.float32)
        if al["flow_loss_weight"] > 0 and scene.dynamic_masks is not None:
            masks = getattr(scene, "sam2_dynamic_masks", None) or scene.dynamic_masks
            result["dynamic_masks"] = np.stack([np.asarray(m.detach().cpu().numpy(), dtype=bool) for m in masks])

    extras = {
        "n_pairs": n_pairs,
        "final_loss": None if loss is None or not np.isfinite(float(loss)) else float(loss),
        "input_resolution": [int(result["depth"].shape[1]), int(result["depth"].shape[2])],
    }
    del scene
    return {"arrays": result, "extras": extras}


# ----------------------------------------------------------------- resumption

def chunk_is_complete(chunk_dir: Path, chunk: dict[str, Any], cfg_hash: str) -> bool:
    """True if ``chunk_dir`` holds a finished reconstruction of exactly this chunk and config."""
    meta_path = chunk_dir / METADATA
    if not meta_path.is_file():
        return False
    try:
        meta = read_json(meta_path)
    except (OSError, json.JSONDecodeError):
        return False
    if meta.get("status") != "completed" or meta.get("reconstruction_config_hash") != cfg_hash:
        return False
    if meta.get("frame_indices") != chunk["frame_indices"]:
        return False
    for name, info in meta.get("files", {}).items():
        path = chunk_dir / name
        if not path.is_file() or sha256_file(path) != info["sha256"]:
            return False
    return True


def summary_entry(chunk_dir: Path, note: str | None = None) -> dict[str, Any]:
    """Status line for the run summary, read from the chunk's own metadata so that skipped or
    restored chunks keep the runtime and memory figures of the run that produced them."""
    meta = read_json(chunk_dir / METADATA)
    error = meta.get("error")
    entry = {"status": meta["status"], "n_frames": meta.get("n_frames"), "runtime_s": meta.get("runtime_s"),
             "peak_gpu_memory_gb": meta.get("peak_gpu_memory_gb"), "final_loss": meta.get("final_loss"),
             "error": None if not error else error["type"]}
    if note:
        entry["note"] = note
    return entry


def update_summary(out_dir: Path, base: dict[str, Any], statuses: dict[str, dict[str, Any]]) -> None:
    counts: dict[str, int] = {}
    for entry in statuses.values():
        counts[entry["status"]] = counts.get(entry["status"], 0) + 1
    write_json(out_dir / RUN_SUMMARY, {**base, "updated_utc": utc_now(), "counts": counts, "chunks": statuses})


# ------------------------------------------------------------------------ run

def run(args: argparse.Namespace, model: Any = None) -> int:
    """Process the manifest. ``model`` can be injected (used by the dry-run test)."""
    package = Path(args.package).resolve()
    manifest_path = package / "manifests" / f"{args.manifest}.json"
    if not manifest_path.is_file():
        available = sorted(p.stem for p in (package / "manifests").glob("*.json"))
        print(f"ERROR: manifest {args.manifest!r} not found in {package}; available: {available}")
        return 2
    manifest = read_json(manifest_path)
    cfg_file = Path(args.config) if args.config else package / "reconstruction.json"
    cfg = merge_config(DEFAULT_CONFIG, read_json(cfg_file) if cfg_file.is_file() else None)
    cfg_hash = config_hash(cfg)

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and not args.allow_cpu:
        print("ERROR: no CUDA GPU is available. Enable a GPU accelerator for the notebook session "
              "(or pass --allow-cpu to run on the CPU, which is very slow).")
        return 3
    torch.backends.cuda.matmul.allow_tf32 = True  # as in MonST3R demo.py

    monst3r_root = Path(args.monst3r_root).resolve()
    out_dir = Path(args.output).resolve() / manifest["video_id"] / f"{args.manifest}__recon_{cfg_hash}"
    out_dir.mkdir(parents=True, exist_ok=True)
    resume_dir = Path(args.resume_from).resolve() if args.resume_from else None
    if resume_dir is not None:
        nested = resume_dir / manifest["video_id"] / out_dir.name  # accept the artifact root as well
        resume_dir = nested if nested.is_dir() else resume_dir

    chunks = manifest["chunks"]
    if args.chunks:
        wanted = set(args.chunks)
        unknown = wanted - {c["chunk_id"] for c in chunks}
        if unknown:
            print(f"ERROR: unknown chunk ids: {sorted(unknown)}")
            return 2
        chunks = [c for c in chunks if c["chunk_id"] in wanted]
    if args.max_chunks:
        chunks = chunks[: args.max_chunks]

    api = import_monst3r(monst3r_root)  # changes the working directory; all paths above are absolute
    check_flow_checkpoints(monst3r_root, cfg)
    base_info = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "video_id": manifest["video_id"],
        "strategy": manifest["strategy"],
        "manifest_name": args.manifest,
        "manifest_config_hash": manifest["config_hash"],
        "reconstruction_config": cfg,
        "reconstruction_config_hash": cfg_hash,
        "model": cfg["model"]["weights"] if model is None else getattr(model, "worker_model_id", "injected-model"),
        "monst3r_commit": git_commit(monst3r_root),
        "gpu": gpu_info(torch),
        "software": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__},
    }
    print(f"Output folder : {out_dir}")
    print(f"Device        : {device}  {base_info['gpu']}")
    print(f"Chunks to do  : {[c['chunk_id'] for c in chunks]}")

    statuses: dict[str, dict[str, Any]] = {}
    for chunk in chunks:
        cid = chunk["chunk_id"]
        chunk_dir = out_dir / cid
        if chunk_is_complete(chunk_dir, chunk, cfg_hash):
            statuses[cid] = summary_entry(chunk_dir, "already done, skipped")
            print(f"[{cid}] already completed - skipped")
            update_summary(out_dir, base_info, statuses)
            continue
        if resume_dir is not None and chunk_is_complete(resume_dir / cid, chunk, cfg_hash):
            if chunk_dir.exists():
                shutil.rmtree(chunk_dir)
            shutil.copytree(resume_dir / cid, chunk_dir)
            statuses[cid] = summary_entry(chunk_dir, "copied from --resume-from")
            print(f"[{cid}] restored from {resume_dir}")
            update_summary(out_dir, base_info, statuses)
            continue

        if model is None:  # load lazily: nothing to load when every chunk is already done
            print(f"Loading model {cfg['model']['weights']} ...")
            model = api["AsymmetricCroCo3DStereo"].from_pretrained(cfg["model"]["weights"]).to(device)
        model.eval()

        if chunk_dir.exists():
            shutil.rmtree(chunk_dir)  # leftovers of an interrupted or failed attempt
        chunk_dir.mkdir(parents=True)
        files = [str(package / "frames" / name) for name in chunk["frame_files"]]
        meta: dict[str, Any] = {
            **base_info,
            "chunk_id": cid,
            "segment_id": chunk.get("segment_id"),
            "frame_indices": chunk["frame_indices"],
            "frame_files": chunk["frame_files"],
            "timestamps_s": chunk.get("timestamps_s"),
            "start_timestamp_s": chunk["start_timestamp_s"],
            "end_timestamp_s": chunk["end_timestamp_s"],
            "n_frames": len(files),
            "started_utc": utc_now(),
            "conventions": {
                "camera_poses": "camera-to-world (chunk frame), OpenCV axes: x right, y down, z forward",
                "scale": "arbitrary per chunk; not metric",
                "confidence": "MonST3R network confidence, values >= 1; higher is more confident",
            },
        }
        print(f"[{cid}] {len(files)} frames ...", flush=True)
        start = time.time()
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        torch.manual_seed(cfg["seed"])
        np.random.seed(cfg["seed"])
        try:
            missing = [f for f in files if not Path(f).is_file()]
            if missing:
                raise FileNotFoundError(f"{len(missing)} frame files missing from the package, e.g. {missing[0]}")
            res = reconstruct_chunk(api, model, device, files, cfg)
            file_info: dict[str, Any] = {}
            for name, array in res["arrays"].items():
                path = chunk_dir / f"{name}.npy"
                np.save(path, array)
                file_info[path.name] = {"sha256": sha256_file(path), "shape": list(array.shape), "dtype": str(array.dtype)}
            meta.update(res["extras"])
            meta.update(status="completed", files=file_info, error=None,
                        dynamic_masks="saved" if "dynamic_masks" in res["arrays"] else "not computed (flow loss disabled)")
            del res
        except Exception as exc:  # noqa: BLE001 - every failure must be recorded, then the run continues
            oom = isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()
            meta.update(status="failed", files={}, error={
                "type": "out_of_memory" if oom else type(exc).__name__,
                "message": str(exc)[:2000],
                "traceback": traceback.format_exc()[-4000:],
            })
            for leftover in chunk_dir.glob("*.npy"):
                leftover.unlink()
            print(f"[{cid}] FAILED: {meta['error']['type']}: {meta['error']['message'][:300]}")
        meta["runtime_s"] = round(time.time() - start, 2)
        meta["finished_utc"] = utc_now()
        meta["peak_gpu_memory_gb"] = (
            round(torch.cuda.max_memory_allocated() / 2**30, 2) if device == "cuda" else None
        )
        write_json(chunk_dir / METADATA, meta)  # written last: its presence marks the chunk as finished
        statuses[cid] = summary_entry(chunk_dir)
        if meta["status"] == "completed":
            print(f"[{cid}] done in {meta['runtime_s']} s, peak GPU memory {meta['peak_gpu_memory_gb']} GB")
        update_summary(out_dir, base_info, statuses)
        free_memory(torch)

    failed = [cid for cid, s in statuses.items() if s["status"] != "completed"]
    print(f"Finished: {len(statuses) - len(failed)} completed, {len(failed)} failed {failed}")
    print(f"Artifacts in: {out_dir}")
    return 1 if failed else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--package", required=True, help="Folder created by scripts/export_chunks.py")
    parser.add_argument("--manifest", required=True, help="Manifest name inside the package, e.g. adaptive_d70e3096")
    parser.add_argument("--output", required=True, help="Root folder for artifacts")
    parser.add_argument("--monst3r-root", required=True, help="MonST3R checkout (cloned with --recursive)")
    parser.add_argument("--config", default=None, help="Reconstruction config JSON (default: package/reconstruction.json)")
    parser.add_argument("--chunks", nargs="*", default=None, help="Only these chunk ids")
    parser.add_argument("--max-chunks", type=int, default=None, help="Stop after this many chunks (for a first test)")
    parser.add_argument("--resume-from", default=None,
                        help="Folder with artifacts of an earlier session for the same manifest and config; "
                             "completed chunks are copied instead of recomputed")
    parser.add_argument("--allow-cpu", action="store_true", help="Run without a GPU (very slow)")
    return parser


if __name__ == "__main__":
    sys.exit(run(build_parser().parse_args()))
