# Cloud reconstruction worker (MonST3R on Kaggle / Colab)

MonST3R never runs on the laptop. The laptop prepares a package of frames and
chunk manifests; a free GPU notebook reconstructs each chunk; the artifacts
come back as a zip and are validated locally. The two sides share only files.

```
laptop                                   Kaggle / Colab                         laptop
export_chunks.py  ->  <id>_package.zip  ->  monst3r_*.ipynb + monst3r_worker.py  ->  reconstructions.zip  ->  validate_artifacts.py
```

| File | Purpose |
| --- | --- |
| `monst3r_worker.py` | The worker. Self-contained; copied into every package. |
| `monst3r_kaggle.ipynb`, `monst3r_colab.ipynb` | Thin notebooks: set up MonST3R, run the worker, zip the output. |
| `build_notebooks.py` | Generates both notebooks from one definition. |

## What was verified, and what was not

Verified by reading the MonST3R source at commit `574cc77` (June 2025) and by
a CPU dry run of the worker against that checkout with a tiny randomly
initialised network (`tests/test_worker_dryrun.py`):

* every MonST3R function the worker calls exists with the arguments used;
* the artifact files have the documented shapes and pass local validation;
* skipping completed chunks, redoing interrupted ones and `--resume-from` work.

**Not verified: anything that needs a GPU or the pretrained weights.** The
notebooks have not been executed on Kaggle or Colab, the weights have not been
downloaded or loaded, and GPU memory use is unknown. The first real run is
the test. Expect to adjust settings, and report what happens.

## Findings from inspecting MonST3R

* **Licence:** CC BY-NC-SA 4.0 for code and weights. Non-commercial use only;
  fine for a research project, and results must carry attribution.
* **Weights:** `Junyi42/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt` on
  Hugging Face (free, no account needed), also on Google Drive. The worker
  loads them with MonST3R's own `AsymmetricCroCo3DStereo.from_pretrained`.
* **Environment in its README:** Python 3.11, PyTorch with CUDA 12.1, plus
  `requirements.txt`. The notebooks keep the PyTorch already installed in the
  notebook image and add only the missing pure-Python packages. The dry run
  here used Python 3.13, a current PyTorch and NumPy 2 without code changes.
* **Hidden dependencies:** `dust3r/cloud_opt/optimizer.py` imports `sam2` and
  RAFT at import time even when they are not used, and other modules import
  `evo`, `seaborn` and `imageio`. The worker puts MonST3R's bundled
  `third_party/sam2` on the import path; the notebook installs the rest.
* **README inaccuracy:** it shows `python demo.py --input ...`, but the actual
  argument is `--input_dir`. The worker does not use `demo.py`.
* **Memory:** the MonST3R README quotes about 33 GB of GPU memory for 65
  frames at 16:9, or 23 GB in its memory-saving mode, with the flow loss on.
  Free GPUs have about 16 GB. Hence the settings below.
* **Outputs:** MonST3R optimises, per frame, a camera-to-world pose, a focal
  length and a depth map; point maps are the depth maps unprojected with
  those. Confidence comes from the network. Dynamic masks exist only when the
  flow loss is on.

## Deviations from MonST3R's defaults

Set in `configs/reconstruction.yaml`; every one is recorded in each chunk's
`metadata.json` and must be stated with any result.

| Setting | MonST3R demo | Here | Why |
| --- | --- | --- | --- |
| `alignment.flow_loss_weight` | 0.01 | **0** | The flow loss loads RAFT and SAM2 and dominates memory. Without it there are no dynamic masks and no flow-based refinement, which may reduce quality. |
| `inference.batch_size` | 16 | 8 | Lower peak memory during pairwise inference. |
| chunk length | 65+ frames | 15-45 frames | Memory; this is also the research variable. |

Everything else (scene graph `swinstride` with window 5, 300 iterations,
linear schedule, learning rate 0.01, shared focal length, temporal smoothing
0.01, point-cloud cleaning) matches the demo's non-interactive defaults. Mixed
precision is not used because MonST3R's own inference does not use it.

A limitation to keep in mind: MonST3R has no input for "ignore these pixels",
so the static-car mask from `make_static_mask.py` is **not** applied during
reconstruction. The car's bodywork is in every frame and may bias poses.
Handling that is part of the masking experiments (Phase 7).

## Step by step

### 1. On the laptop: build the package

```powershell
python scripts\export_chunks.py --video-id interlagos_lap1_full
```

This writes `outputs\cloud_packages\interlagos_lap1_full_package\` and a zip
next to it, containing both chunk manifests, the frames they reference
(stored once), `reconstruction.json` and the worker. For 190 frames at
512x288 the zip is a few megabytes. To export again after changing something,
pass a new `--name`.

### 2a. Kaggle

1. kaggle.com -> *Datasets* -> *New Dataset* -> upload the zip -> *Create*
   (private is fine). Kaggle unpacks it.
2. *Code* -> *New Notebook* -> *File* -> *Import Notebook* -> upload
   `cloud\monst3r_kaggle.ipynb`.
3. In the right-hand panel: *Add Input* -> your dataset. *Session options*:
   Accelerator **GPU T4 x2** (or P100), Internet **On**, Persistence
   **Files only** if offered. Internet needs a phone-verified account.
4. Run the cells in order. After the "Locate the package" cell prints the
   manifest names, set `MANIFEST` in the settings cell and run on.
5. The first run does one chunk (`MAX_CHUNKS = 1`). Look at the reported
   runtime and peak GPU memory, then set `MAX_CHUNKS = None` and re-run the
   reconstruct cell; the finished chunk is skipped.
6. Run the download cell and fetch `reconstructions.zip` from the Output panel.

Free GPU hours and session lengths on Kaggle are limited and change over
time; check your quota on the site. Nothing here assumes a particular quota.

### 2b. Colab (fallback)

Open `cloud\monst3r_colab.ipynb` in Colab, select a T4 GPU runtime and run the
cells. The package zip is uploaded in the "Locate the package" cell and the
result is downloaded in the last cell. Colab deletes its disk on disconnect,
so process a few chunks at a time with `MAX_CHUNKS` and download in between.

### 3. Back on the laptop: validate

Unzip `reconstructions.zip` into `outputs\reconstructions\`, so that you have
`outputs\reconstructions\<video_id>\<manifest>__recon_<hash>\chunk_001\...`.

```powershell
python scripts\validate_artifacts.py --run outputs\reconstructions\interlagos_lap1_full\adaptive_d70e3096__recon_<hash>
```

It checks checksums, shapes, finite values, that poses are rigid transforms,
that frame indices match the chunk manifest, and that the point maps equal
the depth maps unprojected with the stored intrinsics and poses. It writes
`validation_report.json` into the run folder.

## If a chunk runs out of memory

The chunk is recorded as `failed` with `out_of_memory` and the run continues.
In the notebook's settings cell, try in this order and re-run the reconstruct
cell:

1. `OVERRIDES = {"alignment": {"batchify": False}}`
2. `OVERRIDES = {"alignment": {"batchify": False}, "inference": {"window_size": 3, "batch_size": 4}}`
3. Shorter chunks: lower `max_frames` and `target_frames` in
   `configs\chunking.yaml`, re-run `generate_chunks.py` and `export_chunks.py`.

Different settings write to a different folder (`..._recon_<hash>`), so
results never mix. Use the same settings for the fixed and the adaptive
manifest, otherwise the comparison is not fair.

## Sessions that disconnect

* A chunk is complete only when its `metadata.json` exists; it is written
  last. Re-running the notebook verifies completed chunks by checksum and
  skips them, and redoes anything interrupted.
* If the working folder was wiped, give the artifacts of the earlier session
  to `RESUME_FROM` (Kaggle: add the earlier output as an input; Colab: upload
  and unzip your downloaded zip). Completed chunks are copied, not recomputed.
* The package dataset is uploaded once; resuming never needs a re-upload.

## Artifact format

Per chunk, with `N` frames of `H x W` pixels (512x288 input gives H=288, W=512):

| File | Shape, type | Content |
| --- | --- | --- |
| `camera_poses.npy` | (N, 4, 4) float32 | camera-to-chunk transforms |
| `intrinsics.npy` | (N, 3, 3) float32 | pinhole K in pixels at H x W |
| `depth.npy` | (N, H, W) float32 | depth along the camera z axis |
| `pointmaps.npy` | (N, H, W, 3) float32 | 3D point per pixel, chunk frame |
| `confidence.npy` | (N, H, W) float32 | network confidence (>= 1) after cleaning |
| `init_confidence.npy` | (N, H, W) float32 | confidence before cleaning |
| `dynamic_masks.npy` | (N, H, W) bool | only with the flow loss enabled |
| `metadata.json` | | status, chunk and frame ids, timestamps, model, MonST3R commit, full config, runtime, GPU, peak memory, final loss, SHA-256 of every file |

Conventions: OpenCV camera axes (x right, y down, z forward);
`X_chunk = camera_poses[i] @ X_cam`. Each chunk has its own coordinate frame
and its own arbitrary scale. Nothing is metric.
