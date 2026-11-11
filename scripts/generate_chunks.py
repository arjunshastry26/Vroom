r"""Generate chunk manifests with the fixed and/or motion-aware adaptive strategy.

Example (PowerShell):
    python scripts\generate_chunks.py --frames data\frames\interlagos_2023_test --strategy both
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from vroom_interlagos.chunking.chunkers import ChunkingConfig, build_chunk_manifest, default_manifest_path
from vroom_interlagos.chunking.motion import MOTION_CSV, load_motion_table
from vroom_interlagos.chunking.plots import plot_chunk_boundaries
from vroom_interlagos.chunking.scoring import compute_boundary_scores
from vroom_interlagos.cli import PROJECT_ROOT, base_parser, init_logging
from vroom_interlagos.common import dataclass_from_dict, load_yaml, portable_path, read_json, write_json
from vroom_interlagos.preprocessing.frames import METADATA_JSON, load_frame_table

logger = logging.getLogger("generate_chunks")


def main() -> int:
    parser = base_parser(__doc__.splitlines()[0])
    parser.add_argument("--frames", type=Path, required=True, help="Folder written by extract_frames.py")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "chunking.yaml")
    parser.add_argument("--strategy", choices=["fixed", "adaptive", "both"], default="both")
    parser.add_argument("--chunks-root", type=Path, default=PROJECT_ROOT / "data" / "chunks")
    parser.add_argument("--outputs-root", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    init_logging(args)

    try:
        cfg = dataclass_from_dict(ChunkingConfig, load_yaml(args.config).get("chunking"))
        records = load_frame_table(args.frames)
        meta_path = args.frames / METADATA_JSON
        video_id = read_json(meta_path)["video_id"] if meta_path.is_file() else args.frames.name
        motion_rows = load_motion_table(args.frames) if (args.frames / MOTION_CSV).is_file() else None
        if motion_rows is None:
            logger.warning("No motion.csv in %s: fixed chunks will carry no motion statistics", args.frames)

        strategies = ["fixed", "adaptive"] if args.strategy == "both" else [args.strategy]
        manifests = []
        for strategy in strategies:
            manifest = build_chunk_manifest(
                records, cfg, strategy, motion_rows=motion_rows, video_id=video_id,
                frames_dir=portable_path(args.frames, PROJECT_ROOT),
            )
            path = default_manifest_path(args.chunks_root, manifest)
            write_json(path, manifest)
            manifests.append(manifest)
            s = manifest["summary"]
            logger.info(
                "%-8s: %d chunks, frames mean/min/max = %.1f/%d/%d, mean cut cost = %s -> %s",
                strategy, s["n_chunks"], s["n_frames_mean"], s["n_frames_min"], s["n_frames_max"],
                "n/a" if s["cut_cost_mean"] != s["cut_cost_mean"] else f"{s['cut_cost_mean']:.3f}", path,
            )
            for note in manifest["notes"]:
                logger.warning("%s: %s", strategy, note)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 1

    tag = "_".join(f"{m['strategy']}-{m['config_hash']}" for m in manifests)
    report = args.outputs_root / "reports" / video_id / f"chunking_{tag}.json"
    write_json(
        report,
        {
            "video_id": video_id,
            "note": "The adaptive strategy minimises cut cost by construction; a lower cut cost is not, by itself, "
                    "evidence of better reconstruction. That requires the Phase 3+ experiments.",
            "strategies": {m["strategy"]: {"config_hash": m["config_hash"], **m["summary"]} for m in manifests},
        },
    )
    logger.info("Comparison report: %s", report)

    if motion_rows is not None and not args.no_plot:
        scores = compute_boundary_scores(motion_rows, cfg.score)
        plot = plot_chunk_boundaries(
            motion_rows, scores, manifests, args.outputs_root / "plots" / video_id / f"chunking_{tag}.png",
            title=f"{video_id}: chunk boundaries vs. motion statistics",
        )
        logger.info("Plot: %s", plot)
    return 0


if __name__ == "__main__":
    sys.exit(main())
