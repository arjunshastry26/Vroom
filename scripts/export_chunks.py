r"""Build the upload package for the cloud MonST3R worker from chunk manifests.

Example (PowerShell):
    python scripts\export_chunks.py --video-id interlagos_lap1_full
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from vroom_interlagos.cli import PROJECT_ROOT, base_parser, init_logging
from vroom_interlagos.common import load_yaml, read_json
from vroom_interlagos.reconstruction.package import export_package

logger = logging.getLogger("export_chunks")


def main() -> int:
    parser = base_parser(__doc__.splitlines()[0])
    parser.add_argument("--video-id", default=None, help="Export every chunk manifest of this video")
    parser.add_argument("--manifests", type=Path, nargs="*", default=None, help="Or: explicit chunks.json files")
    parser.add_argument("--frames", type=Path, default=None, help="Frames folder (default: taken from the manifest)")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "reconstruction.yaml")
    parser.add_argument("--name", default=None, help="Package name (default: <video_id>_package)")
    parser.add_argument("--out-root", type=Path, default=PROJECT_ROOT / "outputs" / "cloud_packages")
    parser.add_argument("--no-zip", action="store_true")
    args = parser.parse_args()
    init_logging(args)

    try:
        if args.manifests:
            paths = list(args.manifests)
        elif args.video_id:
            paths = sorted((PROJECT_ROOT / "data" / "chunks" / args.video_id).glob("*/chunks.json"))
            if not paths:
                raise FileNotFoundError(f"No chunk manifests under data/chunks/{args.video_id}; run generate_chunks.py")
        else:
            parser.error("give --video-id or --manifests")
        first = read_json(paths[0])
        frames_dir = args.frames or Path(first["frames_dir"])
        if not frames_dir.is_absolute():
            frames_dir = PROJECT_ROOT / frames_dir
        name = args.name or f"{first['video_id']}_package"
        info = export_package(
            paths, frames_dir, args.out_root / name, PROJECT_ROOT / "cloud" / "monst3r_worker.py",
            load_yaml(args.config), make_zip=not args.no_zip,
        )
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        logger.error("%s", exc)
        return 1

    logger.info("Package  : %s", args.out_root / name)
    logger.info("Frames   : %d", info["n_frames"])
    for mname, m in info["manifests"].items():
        logger.info("Manifest : %s (%d chunks, longest %d frames)", mname, m["n_chunks"], m["max_chunk_frames"])
    logger.info("Next: upload the zip as a Kaggle dataset and follow cloud/README.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
