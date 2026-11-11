r"""Run every local step for one video with a single command.

    prepare_video -> extract_frames -> make_static_mask -> analyze_motion -> generate_chunks -> export_chunks

Steps whose output already exists are skipped, so the command can be re-run
safely. The result is the zip to upload to Kaggle.

Example (PowerShell):
    python scripts\run_local_pipeline.py --video data\raw\interlagos_onboard.mp4 --video-id interlagos_lap2
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

from vroom_interlagos.cli import PROJECT_ROOT, base_parser, init_logging

logger = logging.getLogger("run_local_pipeline")
SCRIPTS = Path(__file__).resolve().parent


def step(name: str, script: str, args: list[str], skip_if: Path | None = None) -> None:
    if skip_if is not None and skip_if.exists():
        logger.info("[skip] %s (found %s)", name, skip_if)
        return
    logger.info("[run ] %s", name)
    result = subprocess.run([sys.executable, str(SCRIPTS / script), *args])
    if result.returncode != 0:
        raise SystemExit(f"Step '{name}' failed (exit code {result.returncode}); fix the error above and re-run.")


def main() -> int:
    parser = base_parser(__doc__.splitlines()[0])
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--video-id", required=True, help="New identifier for this clip")
    parser.add_argument("--start", type=float, default=0.0, help="Clip start (s)")
    parser.add_argument("--duration", type=float, default=None, help="Clip duration (s); default: to the end")
    parser.add_argument("--mask", type=Path, default=None,
                        help="Reuse an existing mask image (same camera) instead of estimating one")
    parser.add_argument("--no-mask", action="store_true", help="Analyse motion without a mask")
    args = parser.parse_args()
    init_logging(args)

    vid = args.video_id
    data = PROJECT_ROOT / "data"
    frames = data / "frames" / vid
    manifest = data / "manifests" / f"{vid}.json"

    prep = ["--video", str(args.video), "--video-id", vid, "--start", str(args.start)]
    if args.duration is not None:
        prep += ["--duration", str(args.duration)]
    step("inspect video", "prepare_video.py", prep, skip_if=manifest)
    step("extract frames", "extract_frames.py", ["--manifest", str(manifest)], skip_if=frames / "metadata.json")

    motion_args = ["--frames", str(frames)]
    if not args.no_mask:
        mask = args.mask or data / "masks" / vid / "static_mask.png"
        if args.mask is None:
            step("estimate static mask", "make_static_mask.py",
                 ["--frames", str(frames), "--std-threshold", "12", "--close-px", "25", "--fill-holes", "0.12"],
                 skip_if=mask)
        motion_args += ["--mask", str(mask)]
    step("analyse motion", "analyze_motion.py", motion_args, skip_if=frames / "motion.csv")
    step("generate chunks", "generate_chunks.py", ["--frames", str(frames), "--strategy", "both"])
    package = PROJECT_ROOT / "outputs" / "cloud_packages" / f"{vid}_package"
    step("export package", "export_chunks.py", ["--video-id", vid], skip_if=package)

    logger.info("Done. Upload to Kaggle: %s.zip", package)
    if not args.no_mask and args.mask is None:
        logger.info("Check the mask preview before trusting the motion analysis: %s",
                    data / "masks" / vid / "static_mask_preview.jpg")
    return 0


if __name__ == "__main__":
    sys.exit(main())
