r"""Inspect a video and create a reproducible dataset manifest.

Example (PowerShell):
    python scripts\prepare_video.py --video data\raw\interlagos_onboard.mp4 --video-id interlagos_2023_test --start 0 --duration 25
"""

from __future__ import annotations

import logging
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

from vroom_interlagos.cli import PROJECT_ROOT, base_parser, init_logging
from vroom_interlagos.common import portable_path, sha256_file, utc_now_iso, write_json
from vroom_interlagos.preprocessing.video import VideoError, probe_video, read_frame_timestamps

logger = logging.getLogger("prepare_video")


def main() -> int:
    parser = base_parser(__doc__.splitlines()[0])
    parser.add_argument("--video", type=Path, required=True, help="Path to the video file")
    parser.add_argument("--video-id", default=None, help="Identifier used for output folders (default: file stem)")
    parser.add_argument("--start", type=float, default=0.0, help="Clip start in seconds from the first frame")
    parser.add_argument("--duration", type=float, default=None, help="Clip duration in seconds (default: to the end)")
    parser.add_argument("--notes", default="", help="Free text: footage source, driver, session, usage permission")
    parser.add_argument("--manifests-dir", default=PROJECT_ROOT / "data" / "manifests", type=Path)
    parser.add_argument("--no-hash", action="store_true", help="Skip the SHA-256 checksum (faster for huge files)")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing manifest with the same id")
    args = parser.parse_args()
    init_logging(args)

    try:
        meta = probe_video(args.video)
    except (FileNotFoundError, VideoError) as exc:
        logger.error("%s", exc)
        return 1
    video_id = args.video_id or args.video.stem
    out_path = args.manifests_dir / f"{video_id}.json"
    if out_path.exists() and not args.overwrite:
        logger.error("%s already exists; pass --overwrite or choose another --video-id", out_path)
        return 1

    timestamps = None
    pts = read_frame_timestamps(args.video)
    if pts is not None and len(pts) > 1:
        steps = np.diff(pts)
        nominal = 1.0 / meta.fps
        timestamps = {
            "n_frames": int(len(pts)),
            "first_s": float(pts[0]),
            "last_s": float(pts[-1]),
            "median_step_s": float(np.median(steps)),
            "max_step_s": float(steps.max()),
            "n_gaps_over_1p5x_nominal": int((steps > 1.5 * nominal).sum()),
            "constant_frame_rate": bool(np.allclose(steps, nominal, rtol=0.05)),
        }

    manifest = {
        "schema_version": 1,
        "video_id": video_id,
        "created_utc": utc_now_iso(),
        "video_path": portable_path(args.video, PROJECT_ROOT),
        "sha256": None if args.no_hash else sha256_file(args.video),
        "video": asdict(meta),
        "timestamps": timestamps,
        "clip": {"start_s": args.start, "duration_s": args.duration},
        "notes": args.notes,
    }
    write_json(out_path, manifest)

    logger.info("Video      : %s", args.video)
    logger.info("Resolution : %dx%d  codec=%s  container=%s", meta.width, meta.height, meta.codec, meta.container)
    logger.info("Frame rate : %.3f fps  frames=%s  duration=%s s", meta.fps, meta.n_frames, meta.duration_s)
    if timestamps:
        logger.info("Timestamps : %s", timestamps)
    else:
        logger.warning("Exact frame timestamps unavailable (ffprobe missing?); index / fps will be used")
    logger.info("Manifest   : %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
