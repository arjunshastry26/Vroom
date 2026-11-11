r"""Plot and summarise the camera trajectory of every reconstructed chunk.

Example (PowerShell):
    python scripts\inspect_reconstruction.py --run outputs\reconstructions\interlagos_lap1_full\adaptive_d70e3096__recon_bbfd0ab8
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

from vroom_interlagos.cli import PROJECT_ROOT, base_parser, init_logging
from vroom_interlagos.common import read_json
from vroom_interlagos.reconstruction.inspect import plot_run

logger = logging.getLogger("inspect_reconstruction")


def main() -> int:
    parser = base_parser(__doc__.splitlines()[0])
    parser.add_argument("--run", type=Path, required=True, help="<video_id>/<manifest>__recon_<hash> folder")
    parser.add_argument("--manifest", type=Path, default=None, help="chunks.json (default: from run_summary.json)")
    args = parser.parse_args()
    init_logging(args)
    try:
        summary = read_json(args.run / "run_summary.json")
        manifest_path = args.manifest or (
            PROJECT_ROOT / "data" / "chunks" / summary["video_id"] / summary["manifest_name"] / "chunks.json")
        manifest = read_json(manifest_path)
        out = PROJECT_ROOT / "outputs" / "plots" / manifest["video_id"] / f"trajectories_{args.run.name}.png"
        stats = plot_run(args.run, manifest, out)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        logger.error("%s", exc)
        return 1

    logger.info("Steps are in units of each chunk's median scene depth.")
    logger.info("%-10s %6s %11s %9s %11s %9s %12s %13s %12s %10s", "chunk", "frames", "median step", "max step",
                "max/median", "jump at", "near-static", "median turn", "fwd align", "confident")
    for s in stats:
        logger.info("%-10s %6d %11.4f %9.4f %11.1f %9d %11.0f%% %10.2f deg %12.2f %9.0f%%", s["chunk_id"], s["n_frames"],
                    s["step_median_rel"], s["step_max_rel"], s["max_over_median"],
                    s["frame_indices"][s["largest_step_at"]], 100 * s["near_static_fraction"],
                    float(np.median(s["heading_change_deg"])), s["forward_alignment_median"],
                    100 * s["confident_fraction"])
    logger.info("Plot: %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
