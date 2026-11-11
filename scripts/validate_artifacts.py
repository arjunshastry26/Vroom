r"""Validate reconstruction artifacts downloaded from the cloud worker.

Example (PowerShell):
    python scripts\validate_artifacts.py --run outputs\reconstructions\interlagos_lap1_full\adaptive_d70e3096__recon_1a2b3c4d
"""

from __future__ import annotations

import logging
import sys
from dataclasses import asdict
from pathlib import Path

from vroom_interlagos.cli import PROJECT_ROOT, base_parser, init_logging
from vroom_interlagos.common import read_json, write_json
from vroom_interlagos.reconstruction.artifacts import validate_run

logger = logging.getLogger("validate_artifacts")


def main() -> int:
    parser = base_parser(__doc__.splitlines()[0])
    parser.add_argument("--run", type=Path, required=True,
                        help="Worker output folder: <video_id>/<manifest>__recon_<hash> containing chunk_XXX folders")
    parser.add_argument("--manifest", type=Path, default=None,
                        help="chunks.json to check against (default: looked up in data/chunks from run_summary.json)")
    parser.add_argument("--no-checksums", action="store_true", help="Skip SHA-256 verification (faster)")
    args = parser.parse_args()
    init_logging(args)

    try:
        manifest_path = args.manifest
        if manifest_path is None:
            summary = read_json(args.run / "run_summary.json")
            manifest_path = PROJECT_ROOT / "data" / "chunks" / summary["video_id"] / summary["manifest_name"] / "chunks.json"
        manifest = read_json(manifest_path)
    except (FileNotFoundError, KeyError) as exc:
        logger.error("Could not find the chunk manifest (%s); pass --manifest", exc)
        return 1

    reports = validate_run(args.run, manifest, check_checksums=not args.no_checksums)
    for rep in reports:
        if rep.ok:
            s = rep.stats
            logger.info("%s valid   frames=%s res=%s focal=%.1fpx runtime=%ss peak_gpu=%sGB%s", rep.chunk_id,
                        s.get("n_frames"), s.get("resolution"), s.get("focal_px", float("nan")), s.get("runtime_s"),
                        s.get("peak_gpu_memory_gb"), "".join(f"  WARNING: {w}" for w in rep.warnings))
        else:
            logger.error("%s %s: %s", rep.chunk_id, rep.status, "; ".join(rep.errors))
    counts: dict[str, int] = {}
    for rep in reports:
        counts[rep.status] = counts.get(rep.status, 0) + 1
    out = args.run / "validation_report.json"
    write_json(out, {"manifest": str(manifest_path), "counts": counts, "chunks": [asdict(r) for r in reports]})
    logger.info("Summary: %s  ->  %s", counts, out)
    return 0 if counts.get("valid", 0) == len(reports) else 1


if __name__ == "__main__":
    sys.exit(main())
