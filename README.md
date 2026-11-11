# VROOM-Interlagos

**Motion-aware adaptive chunking and global pose optimization for monocular 3D
reconstruction of Formula 1 circuits.**

A research pipeline, inspired by [VROOM](https://varun-bharadwaj.github.io/vroom/)
([paper](https://arxiv.org/abs/2508.17172)), that reconstructs the Interlagos
circuit and the camera trajectory from onboard footage using
[MonST3R](https://github.com/Junyi42/monst3r). No frontend: computer vision,
optimization and evaluation only.

Hypothesis under test (not yet tested, no claim is made):

> Motion-aware adaptive chunking combined with global pose-graph optimization
> improves trajectory consistency and reduces accumulated drift compared with
> fixed-size chunking and sequential chunk stitching.

## Status

| Phase | Content | State |
| --- | --- | --- |
| 1 | Preprocessing: probe, frame extraction, quality checks, motion analysis, manifests | done |
| 2 | Fixed and motion-aware adaptive chunking, plots | first version done |
| 3 | MonST3R worker, Kaggle / Colab notebooks, package export, artifact validation | written; not yet run on a GPU |
| 4 | Cross-chunk Sim(3) alignment | not started |
| 5 | Global pose-graph optimization, loop closure | not started |
| 6 | Evaluation and ablations | not started |
| 7 | Masking, multi-lap fusion, scale recovery | not started |

Phases 1 and 2 have been run on a synthetic clip and on one real onboard
recording. Phase 3 has been checked against the MonST3R source and with a CPU
dry run using a random network, but never on a GPU with the real weights.
There are no reconstruction results yet.

## What is ours and what is not

* From VROOM (per the project brief): the overall idea of reconstructing an F1
  circuit from onboard video by preprocessing, chunking, per-chunk MonST3R
  reconstruction and stitching, at reduced resolution.
* From MonST3R: the per-chunk reconstruction model (point maps, poses,
  confidence). It will be used unmodified where possible.
* Ours (this repository): the boundary score and dynamic-programming chunker
  (implemented), and later the Sim(3) pose graph with loop closure and the
  evaluation protocol comparing them against fixed chunking and sequential
  stitching.

## Setup (Windows PowerShell)

Python 3.10 or 3.11. No CUDA and no PyTorch are needed locally.

```powershell
py -3.11 -m venv .venv
```

```powershell
.\.venv\Scripts\Activate.ps1
```

```powershell
python -m pip install --upgrade pip
```

```powershell
python -m pip install -r requirements.txt
```

```powershell
python -m pip install -e .
```

If activation is blocked by the execution policy, run this first in the same
window: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`.

FFmpeg is optional but recommended. `ffprobe` provides exact frame timestamps
and richer metadata; without it the pipeline falls back to OpenCV and
`index / fps` timestamps.

```powershell
winget install Gyan.FFmpeg
```

Open a new PowerShell window afterwards so `ffprobe` is on PATH.

## Running the pipeline

Run the tests:

```powershell
python -m pytest -q
```

Create a synthetic 40 s test clip (no real footage needed):

```powershell
python scripts\make_synthetic_clip.py --out data\raw\synthetic_test.mp4
```

Inspect the video and write the dataset manifest:

```powershell
python scripts\prepare_video.py --video data\raw\synthetic_test.mp4 --video-id synthetic_test
```

Extract frames (512x288, 3 fps by default):

```powershell
python scripts\extract_frames.py --manifest data\manifests\synthetic_test.json
```

Analyse motion between consecutive frames:

```powershell
python scripts\analyze_motion.py --frames data\frames\synthetic_test
```

Generate both chunk manifests, the comparison report and the plot:

```powershell
python scripts\generate_chunks.py --frames data\frames\synthetic_test --strategy both
```

Onboard footage needs a mask, because the car's own bodywork fills a large
part of the frame and never moves. Estimate one from the frames (use a clip
window in which the car is moving), check the preview, then set
`motion.mask_path` in `configs\chunking.yaml` to the printed path and re-run
`analyze_motion.py` and `generate_chunks.py`:

```powershell
python scripts\make_static_mask.py --frames data\frames\<video_id>
```

For real footage follow [data/README.md](data/README.md), then run the same
last three commands with your `video_id`. Every script accepts `--help`,
`--log-level` and `--log-file`.

Build the upload package for the cloud worker, then follow
[cloud/README.md](cloud/README.md) to reconstruct on Kaggle or Colab:

```powershell
python scripts\export_chunks.py --video-id <video_id>
```

Validate the artifacts after downloading and unzipping them into
`outputs\reconstructions\`:

```powershell
python scripts\validate_artifacts.py --run outputs\reconstructions\<video_id>\<manifest>__recon_<hash>
```

Outputs:

| Path | Content |
| --- | --- |
| `data/manifests/<id>.json` | dataset manifest: checksum, probe, clip window |
| `data/frames/<id>/` | frames, `frames.csv`, `metadata.json`, `motion.csv`, `motion_meta.json` |
| `data/chunks/<id>/<strategy>_<hash>/chunks.json` | chunk manifest |
| `outputs/reports/<id>/chunking_*.json` | fixed vs adaptive summary |
| `outputs/plots/<id>/chunking_*.png` | boundaries against motion statistics |
| `outputs/cloud_packages/<id>_package(.zip)` | frames, manifests, settings and worker for upload |
| `outputs/reconstructions/<id>/<manifest>__recon_<hash>/chunk_XXX/` | MonST3R artifacts per chunk |

Manifest folders are named by a hash of the configuration, so a new
configuration never overwrites an earlier result. Frame extraction refuses to
replace an existing folder unless `--overwrite` is passed.

## Repository layout

```
configs/      preprocessing.yaml, chunking.yaml, reconstruction.yaml
cloud/        monst3r_worker.py, Kaggle and Colab notebooks, README.md
data/         README.md (dataset guide); footage and derived data are git-ignored
scripts/      prepare_video, extract_frames, make_static_mask, analyze_motion, generate_chunks,
              export_chunks, validate_artifacts, make_synthetic_clip
src/vroom_interlagos/
  preprocessing/   video.py (probe, PTS), frames.py (extraction), quality.py (cuts, duplicates, invalid)
  chunking/        motion.py, scoring.py, chunkers.py, plots.py
  masking/         static_mask.py (static-region mask from temporal variation)
  reconstruction/  package.py (upload package), artifacts.py (loading and validation)
  synthetic.py     test-clip generator
tests/        57 tests (2 need PyTorch and a MonST3R checkout, otherwise skipped)
```

Deviations from the layout in the brief, and why:

* Code lives in an installable package `src/vroom_interlagos/` instead of bare
  `src/<module>/` folders, so imports work the same from scripts, tests and
  later the notebook, without path manipulation.
* Modules, configs and scripts for later phases are not created yet. They will
  be added with real content in their phase.
* `tests/test_motion.py` was added. `scripts/make_synthetic_clip.py` was added.

## Method

### Preprocessing

The video is decoded once, sequentially, one frame in memory. At the source
frame rate each frame gets a 160x90 thumbnail used for hard-cut detection
(HSV-histogram distance, absolute and relative to recent history, plus a pixel
difference check), duplicate detection and black/blank detection. Frames are
sampled on a regular time grid from the original timestamps, resized and
saved. Timestamp gaps and decode failures are recorded as missing frames.
Nothing is dropped at this stage: problem frames are saved and flagged.

### Motion analysis

For each pair of consecutive usable frames (`motion.csv`):

| Signal | How |
| --- | --- |
| optical-flow magnitude and variance | Farneback dense flow |
| tracking quality | Shi-Tomasi corners, Lucas-Kanade, forward-backward check: survival ratio and error |
| correspondence count | ORB with ratio test |
| rotation | essential matrix with RANSAC, `recoverPose`; angle and yaw |
| translation proxy | parallax left after removing the rotation (infinite homography) |
| blur | variance of the Laplacian |

Pose convention: `X_b = R X_a + t`, a point in camera A's frame mapped into
camera B's frame; `t` has unit length.

### Boundary score

For each frame pair, with every component in [0, 1]:

```
score = ( w1*motion_change + w2*flow_variance + w3*rotation_change
        + w4*tracking_instability + w5*blur_penalty + w6*reconstruction_uncertainty ) / sum(w)
```

The brief's `MotionChange` term is split into `motion_change` (temporal change
of flow magnitude) and `flow_variance` so each can be weighted separately.
`reconstruction_uncertainty` is used only when supplied from an earlier
reconstruction; it is never invented, and its weight is dropped otherwise. A
pair whose pose cannot be estimated gets the worst rotation value.

### Chunking

Frames are split into segments at cuts, gaps and invalid frames. Within a
segment each chunk has a core (cores partition the segment) and extends
`overlap_frames` into the next core. `min_frames`, `max_frames` and
`target_frames` are full chunk lengths including overlap.

* **Fixed baseline**: equal cores; a short remainder is merged into the last
  chunk if that stays within `max_frames`.
* **Adaptive**: the cost of a cut is the mean score over the overlap window
  that follows it, since those shared frames are what alignment will rely on.
  Dynamic programming finds the cuts minimising total cut cost plus
  `length_penalty * ((length - target) / target)^2` per chunk under the length
  bounds. This is exact for that objective, which a greedy rule would not be.
  If a segment length admits no valid partition, the minimum is relaxed just
  enough and the manifest says so.

Both write the same manifest: per chunk the id, start/end/core frames, frame
indices and files, source frame range, timestamps, overlap with both
neighbours, motion statistics, boundary score and flags; plus the full
configuration, its hash, segments, excluded frames and a summary.

## Resource use

Measured in a Linux container, not on the target laptop: the 40 s synthetic
clip (120 frames at 512x288) took about 1.5 s to extract and 5 s for motion
analysis on CPU. Memory holds one source frame and two greyscale frames.
`num_threads` in both configs limits OpenCV's CPU threads.

## Known limitations

* **Not yet run on Windows or on real footage.** The code uses `pathlib` and
  no shell-specific features, but the PowerShell commands above are untested
  on an actual Windows machine.
* **The synthetic clip is a 2-D pan.** It tests plumbing only. Its epipolar
  geometry is degenerate, so rotation values on it are meaningless (one pair
  reports about 140 degrees).
* **Adaptive chunking beats fixed on cut cost by construction.** That number
  is not evidence for the hypothesis. Evidence needs reconstruction and
  alignment results from Phases 3 to 6.
* **Camera intrinsics are assumed** (80 degree horizontal field of view, no
  distortion model). Rotation angles are approximate, relative signals.
* **Large displacements.** At 2-4 fps a fast car moves far between frames.
  Farneback flow and Lucas-Kanade underestimate or fail on very large motion;
  on the synthetic corners the measured flow magnitude drops although true
  motion rises. The failure does show up as low track survival and high flow
  variance, which is what the score uses, but flow magnitude alone must not be
  read as speed. Whether 3 fps is adequate for real onboard footage is an open
  question for the first real clip.
* **Static cockpit parts** (halo, bodywork) bias flow towards zero.
  `scripts/make_static_mask.py` estimates a mask from temporal variation. It
  also masks uniform moving regions such as sky, and it does not handle other
  cars or spray. Comparing masking strategies is Phase 7.
* **Boundary scores are normalised per clip** and are not comparable between
  clips.
* **Scene-cut detection** handles hard cuts between visually different shots
  only. See `data/README.md`.
* **Clip start uses sequential decoding** for exact frame indices, so a window
  deep inside a multi-hour file is slow to reach. Trim such files first
  without re-encoding: `ffmpeg -ss 01:02:03 -i in.mp4 -t 60 -c copy out.mp4`.
* **Image stabilisation experiments** from Module A are not implemented.
* Default chunk lengths (15/30/45 frames, overlap 5) are placeholders until
  MonST3R's memory use on a free GPU is measured in the first real run.
* **The cloud worker is unproven on a GPU.** See `cloud/README.md` for what
  was and was not verified, and for the deviations from MonST3R's defaults
  (most importantly, the flow loss is off to fit 16 GB).
* **The car mask is not applied inside MonST3R**, which has no input for
  ignoring pixels.

## Next steps

1. Export the package, run the Kaggle notebook on one chunk, and report the
   runtime, peak GPU memory and any error.
2. Reconstruct both manifests with identical settings, download and validate.
3. Phase 4: cross-chunk Sim(3) alignment from the overlapping frames.
