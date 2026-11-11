r"""Align reconstructed chunks pairwise (Sim(3)) and stitch them sequentially.

Example (PowerShell):
    python scripts\align_chunks.py --run outputs\reconstructions\interlagos_lap1_full\adaptive_d70e3096__recon_bbfd0ab8
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from vroom_interlagos.alignment.pairwise import AlignmentConfig
from vroom_interlagos.alignment.stitch import run_alignment
from vroom_interlagos.cli import PROJECT_ROOT, base_parser, init_logging
from vroom_interlagos.common import dataclass_from_dict, load_yaml, read_json

logger = logging.getLogger("align_chunks")


def main() -> int:
    parser = base_parser(__doc__.splitlines()[0])
    parser.add_argument("--run", type=Path, required=True, help="<video_id>/<manifest>__recon_<hash> folder")
    parser.add_argument("--manifest", type=Path, default=None, help="chunks.json (default: from run_summary.json)")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "alignment.yaml")
    parser.add_argument("--mask", type=Path, default=None, help="Mask image; overrides mask_path in the config")
    parser.add_argument("--out", type=Path, default=None, help="Default: outputs/alignments/<video_id>/<run name>")
    parser.add_argument("--no-pointcloud", action="store_true")
    args = parser.parse_args()
    init_logging(args)

    try:
        cfg = dataclass_from_dict(AlignmentConfig, load_yaml(args.config))
        if args.mask is not None:
            cfg.mask_path = str(args.mask.resolve())
        elif cfg.mask_path and not Path(cfg.mask_path).is_absolute():
            cfg.mask_path = str(PROJECT_ROOT / cfg.mask_path)
        manifest_path = args.manifest
        if manifest_path is None:
            summary = read_json(args.run / "run_summary.json")
            manifest_path = PROJECT_ROOT / "data" / "chunks" / summary["video_id"] / summary["manifest_name"] / "chunks.json"
        manifest = read_json(manifest_path)
        frames_dir = Path(manifest["frames_dir"]) if manifest.get("frames_dir") else None
        if frames_dir is not None and not frames_dir.is_absolute():
            frames_dir = PROJECT_ROOT / frames_dir
        out = args.out or PROJECT_ROOT / "outputs" / "alignments" / manifest["video_id"] / args.run.name
        report = run_alignment(args.run, manifest, cfg, out, frames_dir=frames_dir,
                               write_pointcloud=not args.no_pointcloud)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        logger.error("%s", exc)
        return 1

    s = report["summary"]
    logger.info("Edges accepted : %d of %d  ->  %d connected component(s)", s["n_accepted"], s["n_edges"], s["n_components"])
    logger.info("Accepted edges : mean inlier ratio %s, mean residual %s, pose check %s deg / %s x travel",
                s["inlier_ratio_mean"], s["rmse_rel_mean"], s["pose_rotation_error_deg_mean"], s["pose_centre_error_rel_mean"])
    logger.info("Outputs        : %s", out)
    if s["n_components"] > 1:
        logger.warning("The chain is broken: the trajectory is in %d separate pieces with unrelated frames and scales.",
                       s["n_components"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
