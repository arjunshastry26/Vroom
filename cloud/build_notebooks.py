"""Generate monst3r_kaggle.ipynb and monst3r_colab.ipynb from one definition.

The two notebooks differ only in paths and in how files get in and out, so
they are built from a shared list of cells. Re-run after editing:

    python cloud/build_notebooks.py
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

# MonST3R commit the worker was written against (inspected 2026-11). Pinned so
# that an upstream change cannot silently alter results or break the worker.
MONST3R_COMMIT = "574cc77ad278bad582f470e5382624e01f8769a7"

PLATFORMS = {
    "kaggle": {
        "title": "Kaggle",
        "work": "/kaggle/working",
        "code": "/tmp/monst3r",
        "intro": """**Before running** (right-hand panel, *Session options* / *Input*):

1. *Accelerator*: **GPU T4 x2** or **GPU P100**.
2. *Internet*: **On** (needs a phone-verified Kaggle account). It is used to clone MonST3R, install a few packages and download the model weights.
3. *Persistence*: **Files only**, if the option is offered, so `/kaggle/working` survives a session restart.
4. *Add Input* -> *Datasets* -> your uploaded package dataset (the zip made by `scripts\\export_chunks.py`).

**Unattended run:** *Save Version* -> *Save & Run All (Commit)* runs the whole notebook in the background with the
GPU. You can close the browser; when it finishes, open the saved version and download `reconstructions.zip` from its Output tab.""",
        "locate": '''import glob, json, os
hits = sorted(glob.glob("/kaggle/input/**/package.json", recursive=True))
assert hits, "No package.json under /kaggle/input. Add your package dataset with 'Add Input'."''',
        "download": '''# The zip appears in the Output panel on the right (refresh it), under /kaggle/working. Download it from there.
# If the session is about to end, use "Save Version" -> "Quick Save" to keep /kaggle/working as notebook output.''',
        "resume": """**If the session disconnects:** start it again and re-run all cells. Chunks whose `metadata.json`
says `completed` are verified by checksum and skipped. If `/kaggle/working` was wiped, add the output of the
earlier run (your downloaded zip re-uploaded as a dataset, or the saved notebook version via *Add Input* ->
*Notebooks*) and set `RESUME_FROM` in the settings cell to its folder under `/kaggle/input`.""",
    },
    "colab": {
        "title": "Google Colab",
        "work": "/content",
        "code": "/content/monst3r",
        "intro": """**Before running:** *Runtime* -> *Change runtime type* -> **T4 GPU**.

The package zip made by `scripts\\export_chunks.py` is uploaded in the "Locate the package" cell.
Colab storage is temporary: download the result zip at the end. Google Drive is optional and not required.""",
        "locate": '''import glob, json, os, zipfile
hits = sorted(glob.glob("/content/**/package.json", recursive=True))
if not hits:
    from google.colab import files
    print("Choose the package zip created by scripts/export_chunks.py")
    for name in files.upload():
        with zipfile.ZipFile(name) as zf:
            zf.extractall("/content/package")
    hits = sorted(glob.glob("/content/**/package.json", recursive=True))
assert hits, "No package.json found after upload."''',
        "download": '''from google.colab import files
files.download(archive)''',
        "resume": """**If the session disconnects:** Colab deletes `/content`. Keep the zip you downloaded. To continue,
upload and unzip it (for example to `/content/previous`), set `RESUME_FROM = "/content/previous"` in the
settings cell and re-run all cells: completed chunks are verified by checksum and copied instead of recomputed.
To be safe on long runs, process a few chunks at a time with `MAX_CHUNKS` and download after each run.""",
    },
}


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


def build(platform: str) -> dict:
    p = PLATFORMS[platform]
    cells = [
        md(f"""# VROOM-Interlagos: MonST3R reconstruction worker ({p['title']})

Reconstructs the chunks of one chunk manifest with the official [MonST3R](https://github.com/Junyi42/monst3r)
code, one chunk at a time, and saves point maps, camera poses, intrinsics, depth and confidence per chunk.

{p['intro']}

MonST3R and its weights are licensed **CC BY-NC-SA 4.0** (non-commercial use only).

**Nothing needs editing.** Use *Run All*: every manifest of every attached package is reconstructed with the same settings,
finished chunks are skipped, and the result is zipped at the end."""),
        md("## 1. Check the environment"),
        code('''
import subprocess, sys, platform
print("Python", platform.python_version())
try:
    import torch
    print("PyTorch", torch.__version__, "| CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print("GPU:", props.name, f"{props.total_memory / 2**30:.1f} GB", "| count:", torch.cuda.device_count())
    else:
        print("NO GPU. Enable a GPU accelerator and restart the session; the worker refuses to run on the CPU.")
except ImportError:
    print("PyTorch is not installed in this image; install it before continuing.")
print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout[:1500])
'''),
        md("## 2. Settings"),
        code(f'''
MANIFESTS = "all"      # "all" = every manifest in the package, or a list such as ["adaptive_d70e3096"]
MAX_CHUNKS = None      # None = every chunk. Set to 1 for a quick first test on a new package.
CHUNKS = None          # or a list such as ["chunk_003", "chunk_004"]
RESUME_FROM = None     # folder with artifacts of an earlier session (see the last section)
OVERRIDES = {{}}         # reconstruction settings to change, e.g. {{"alignment": {{"batchify": False}}}}

DOWNLOAD_FLOW_CHECKPOINTS = False   # only needed when alignment.flow_loss_weight > 0 (uses much more GPU memory)

WORK = "{p['work']}"
MONST3R_ROOT = "{p['code']}"
MONST3R_COMMIT = "{MONST3R_COMMIT}"
OUTPUT = WORK + "/reconstructions"
'''),
        md("## 3. Locate the package"),
        code(p["locate"] + '''
JOBS = []   # (package folder, manifest name); every attached package is processed
for hit in hits:
    package = os.path.dirname(hit)
    info = json.load(open(hit))
    print("Package :", package)
    print("Video   :", info["video_id"], "|", info["n_frames"], "frames")
    for name, m in info["manifests"].items():
        print(f"  manifest {name}  ({m['n_chunks']} chunks, longest {m['max_chunk_frames']} frames)")
    wanted = list(info["manifests"]) if MANIFESTS == "all" else [m for m in MANIFESTS if m in info["manifests"]]
    JOBS += [(package, name) for name in wanted]
assert JOBS, "Nothing to reconstruct: check MANIFESTS in the settings cell."
print("Will reconstruct", len(JOBS), "manifest(s).")
'''),
        md("""## 4. Get MonST3R

Clones the official repository with its `croco` submodule at a pinned commit. PyTorch from the notebook image
is used as is; only the missing pure-Python packages are installed. The optional RoPE CUDA kernels are not
compiled (MonST3R falls back to a slower PyTorch implementation and prints a warning)."""),
        code('''
import os, subprocess
def sh(cmd, cwd=None):
    print("$", cmd)
    result = subprocess.run(cmd, shell=True, cwd=cwd, text=True, capture_output=True)
    print((result.stdout + result.stderr)[-3000:])
    assert result.returncode == 0, f"command failed: {cmd}"

if not os.path.isdir(os.path.join(MONST3R_ROOT, "dust3r")):
    sh(f"git clone --recursive https://github.com/Junyi42/monst3r.git {MONST3R_ROOT}")
sh(f"git checkout {MONST3R_COMMIT} && git submodule update --init --recursive", cwd=MONST3R_ROOT)
sh('pip install -q roma einops evo seaborn imageio hydra-core iopath "huggingface-hub[torch]>=0.22"')
'''),
        md("""### Optional: flow-loss checkpoints

Only run this when `alignment.flow_loss_weight > 0`. These are the downloads from MonST3R's
`data/download_ckpt.sh` (RAFT optical flow and SAM2)."""),
        code('''
if DOWNLOAD_FLOW_CHECKPOINTS:
    sh("pip install -q gdown")
    sh("mkdir -p models && gdown --fuzzy 'https://drive.google.com/file/d/1a0C5FTdhjM4rKrfXiGhec7eq2YM141lu/view?usp=drive_link' -O models/",
       cwd=os.path.join(MONST3R_ROOT, "third_party/RAFT"))
    sh("wget -q https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt -P checkpoints/",
       cwd=os.path.join(MONST3R_ROOT, "third_party/sam2"))
else:
    print("Skipped (flow loss disabled).")
'''),
        md("""## 5. Reconstruct

Each chunk is saved as soon as it finishes, and GPU memory is released before the next one. A chunk that
fails (for example out of memory) is recorded with its error and the run continues with the next chunk."""),
        code('''
import json, os, subprocess, sys
def overlay(base, extra):
    for key, value in extra.items():
        if isinstance(value, dict):
            overlay(base.setdefault(key, {}), value)
        else:
            base[key] = value
    return base

for PACKAGE, manifest_name in JOBS:
    config_path = os.path.join(PACKAGE, "reconstruction.json")
    if OVERRIDES:   # the same overrides are applied to every manifest
        config_path = os.path.join(WORK, "reconstruction_override.json")
        json.dump(overlay(json.load(open(os.path.join(PACKAGE, "reconstruction.json"))), OVERRIDES), open(config_path, "w"), indent=2)
    cmd = [sys.executable, os.path.join(PACKAGE, "monst3r_worker.py"), "--package", PACKAGE, "--manifest", manifest_name,
           "--output", OUTPUT, "--monst3r-root", MONST3R_ROOT, "--config", config_path]
    if MAX_CHUNKS: cmd += ["--max-chunks", str(MAX_CHUNKS)]
    if CHUNKS: cmd += ["--chunks", *CHUNKS]
    if RESUME_FROM: cmd += ["--resume-from", RESUME_FROM]
    print("=" * 20, os.path.basename(PACKAGE), "/", manifest_name, "=" * 20)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        if "FutureWarning" in line or "autocast(" in line:
            continue  # PyTorch deprecation noise from MonST3R
        print(line, end="")
    print("worker exit code:", proc.wait(), "(0 = all requested chunks completed, 1 = some failed)")
'''),
        md("## 6. Results"),
        code('''
import glob, json
for path in sorted(glob.glob(OUTPUT + "/*/*/run_summary.json")):
    summary = json.load(open(path))
    print(path)
    print("  model :", summary["model"], "| MonST3R commit:", summary["monst3r_commit"])
    print("  GPU   :", summary["gpu"])
    print("  counts:", summary["counts"])
    for cid, s in summary["chunks"].items():
        print(f"  {cid}: {s['status']:10s} runtime={s.get('runtime_s')} s  peak GPU={s.get('peak_gpu_memory_gb')} GB  {s.get('error') or s.get('note') or ''}")
# Per-chunk details straight from each chunk's metadata (independent of the last run).
for path in sorted(glob.glob(OUTPUT + "/*/*/chunk_*/metadata.json")):
    m = json.load(open(path))
    print(f"{m['manifest_name']} {m['chunk_id']}: {m['status']}  frames={m['n_frames']}  pairs={m.get('n_pairs')}  "
          f"runtime={m['runtime_s']} s  peak GPU={m['peak_gpu_memory_gb']} GB  final loss={m.get('final_loss')}")
'''),
        md("""### If a chunk failed with `out_of_memory`

Try these in the settings cell, one at a time, then re-run section 5 (completed chunks are skipped):

1. `OVERRIDES = {"alignment": {"batchify": False}}` - MonST3R's slower, memory-saving optimisation.
2. `OVERRIDES = {"alignment": {"batchify": False}, "inference": {"window_size": 3, "batch_size": 4}}` - fewer image pairs.
3. Shorter chunks: lower `max_frames` / `target_frames` in `configs/chunking.yaml` locally, regenerate and re-export.

Changing settings changes the output folder name (`..._recon_<hash>`), so results from different settings never mix."""),
        md("## 7. Download the artifacts"),
        code('''
import shutil
archive = shutil.make_archive(WORK + "/reconstructions", "zip", root_dir=OUTPUT)
print("Created", archive, f"({os.path.getsize(archive) / 2**20:.1f} MB)")
''' + p["download"]),
        md(f"""Unzip it on the laptop into `outputs\\reconstructions\\` and run
`python scripts\\validate_artifacts.py --run outputs\\reconstructions\\<video_id>\\<manifest>__recon_<hash>`.

## Resuming

{p['resume']}"""),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            **({"kaggle": {"accelerator": "gpu", "isInternetEnabled": True, "isGpuEnabled": True}} if platform == "kaggle"
               else {"accelerator": "GPU", "colab": {"provenance": []}}),
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    for platform in PLATFORMS:
        path = HERE / f"monst3r_{platform}.ipynb"
        path.write_text(json.dumps(build(platform), indent=1) + "\n", encoding="utf-8")
        print("wrote", path)


if __name__ == "__main__":
    main()
