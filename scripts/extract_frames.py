r"""Extract timestamped, resized frames from a video.

Example (PowerShell):
    python scripts\extract_frames.py --manifest data\manifests\interlagos_2023_test.json
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from vroom_interlagos.cli import PROJECT_ROOT, base_parser, init_logging
from vroom_interlagos.common import dataclass_from_dict, load_yaml, read_json
from vroom_interlagos.preprocessing.frames import ClipConfig, PreprocessConfig, extract_frames
from vroom_interlagos.preprocessing.video import VideoError

logger = logging.getLogger("extract_frames")


def main() -> int:
    parser = base_parser(__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", type=Path, help="Dataset manifest written by prepare_video.py")
    source.add_argument("--video", type=Path, help="Video file (skips the manifest; clip window from the config)")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "preprocessing.yaml")
    parser.add_argument("--out", type=Path, default=None, help="Output folder (default: data/frames/<video_id>)")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing extraction")
    args = parser.parse_args()
    init_logging(args)

    try:
        cfg = dataclass_from_dict(PreprocessConfig, load_yaml(args.config))
        if args.overwrite:
            cfg.overwrite = True
        if args.manifest:
            manifest = read_json(args.manifest)
            video_path = Path(manifest["video_path"])
            if not video_path.is_absolute():
                video_path = PROJECT_ROOT / video_path
            video_id = manifest["video_id"]
            cfg.clip = ClipConfig(**manifest["clip"])  # the manifest defines the clip window
        else:
            video_path, video_id = args.video, args.video.stem
        out_dir = args.out or PROJECT_ROOT / "data" / "frames" / video_id
        meta = extract_frames(video_path, out_dir, cfg, video_id=video_id)
    except (FileNotFoundError, FileExistsError, ValueError, VideoError, OSError) as exc:
        logger.error("%s", exc)
        return 1

    scan = meta["scan"]
    logger.info("Frames written   : %d at %dx%d", meta["output"]["n_frames"], cfg.target_width, cfg.target_height)
    logger.info("Achieved rate    : %s fps", meta["output"]["achieved_fps"])
    logger.info("Scene cuts       : %s", [round(c["timestamp_s"], 3) for c in scan["scene_cuts"]])
    logger.info("Timestamp gaps   : %d   decode failures: %d", len(scan["timestamp_gaps"]), len(scan["decode_failures"]))
    logger.info("Invalid/duplicate: %d / %d", scan["n_invalid_extracted"], scan["n_duplicate_extracted"])
    logger.info("Output folder    : %s", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
