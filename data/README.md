# Dataset preparation guide

Nothing in this folder except this guide is committed to git. Footage, frames,
chunks and manifests stay on your machine.

## Footage and licensing

The pipeline never downloads video. You supply the footage, and you are
responsible for having the right to process it (for example F1 TV onboard
footage that your subscription terms allow you to use for private study, or
footage you have explicit permission for). Do not commit or redistribute
footage or frames extracted from it. Record where the footage came from with
`--notes` when creating the manifest.

Recommended starting point: one driver's onboard camera from the 2023
São Paulo GP, a continuous 20-30 s section without broadcast cuts, graphics
transitions or replays. Move to a full lap only once the short clip works.

## Layout

```
data/
  raw/                         your videos (any name)
  manifests/<video_id>.json    written by scripts/prepare_video.py
  frames/<video_id>/           frame_000000.jpg ..., frames.csv, metadata.json, motion.csv
  chunks/<video_id>/<strategy>_<config-hash>/chunks.json
```

## Supported formats

Anything OpenCV's bundled FFmpeg can decode. Tested extensions: `.mp4`, `.mov`,
`.mkv`, `.avi`, `.webm`, `.m4v`, `.ts`. H.264/H.265 in MP4 or MKV is the safe
choice. Variable-frame-rate files are supported when `ffprobe` is installed,
because exact per-frame timestamps are then read from the container.

## Step by step (PowerShell)

1. Copy the video into `data\raw\`.

2. Inspect it and create the dataset manifest:

   ```powershell
   python scripts\prepare_video.py --video data\raw\interlagos_onboard.mp4 --video-id interlagos_2023_test --start 0 --duration 25 --notes "2023 Sao Paulo GP, onboard, driver X, source: ..."
   ```

   This prints resolution, frame rate, frame count and duration, and writes
   `data\manifests\interlagos_2023_test.json` with the SHA-256 of the file, the
   probe results, a timestamp summary and the clip window. `--start` and
   `--duration` select a window without re-encoding or modifying the video.

   The same information straight from FFmpeg, if you want to cross-check:

   ```powershell
   ffprobe -v error -select_streams v:0 -show_entries stream=codec_name,width,height,avg_frame_rate,nb_frames,duration -of default=noprint_wrappers=1 data\raw\interlagos_onboard.mp4
   ```

3. Extract frames:

   ```powershell
   python scripts\extract_frames.py --manifest data\manifests\interlagos_2023_test.json
   ```

## Timestamps

`frames.csv` has one row per extracted frame:

| column | meaning |
| --- | --- |
| `frame_index` | index in the extracted sequence (0-based) |
| `filename` | image file |
| `source_frame_index` | index of the frame in the original video |
| `timestamp_s` | original presentation timestamp (PTS) |
| `clip_time_s` | seconds since the first frame of the video |
| `is_invalid` | black or blank frame |
| `is_duplicate` | identical to the previous extracted frame |
| `cut_before` | a scene cut lies between the previous extracted frame and this one |
| `gap_before` | source frames are missing before this one |

With `ffprobe` on PATH the timestamps are the container's exact PTS
(`"timestamp_source": "ffprobe_pts"` in `metadata.json`). Without it they are
`source_frame_index / fps`, which is only right for constant-frame-rate video.

## Finding cuts, camera changes and missing frames

After extraction, open `data\frames\<video_id>\metadata.json` and look at `scan`:

* `scene_cuts`: hard cuts found at the source frame rate, with timestamps.
* `timestamp_gaps`: places where the PTS jumps by more than `gap_factor` frame
  intervals, meaning frames are missing from the file.
* `decode_failures`: frames the decoder could not produce.
* `n_source_duplicates`: repeated source frames. A large number usually means
  the file was frame-doubled (for example 25 fps content stored at 50 fps).
* `n_invalid_extracted`, `n_duplicate_extracted`: flagged extracted frames.

Chunks never span a cut, a gap or an invalid frame. The cut detector compares
colour histograms, so it finds hard cuts between visually different shots. It
can miss a cut between two similar-looking onboard shots and it does not detect
dissolves or overlaid graphics. Check the first clips by eye: look at the
frames around each reported cut, and skim the sequence for unreported ones. If
needed, tune `scene_cut` in `configs\preprocessing.yaml` and re-extract with
`--overwrite`.

## Reproducibility

A dataset is fully described by the manifest (file checksum plus clip window)
and `configs\preprocessing.yaml`. Both are copied into `metadata.json`, so an
extraction can always be traced back to its inputs. Use a new `--video-id` for
a different clip window rather than overwriting an existing one.
