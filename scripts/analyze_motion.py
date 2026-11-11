r"""Measure camera motion between consecutive extracted frames.

Example (PowerShell):
    python scripts\analyze_motion.py --frames data\frames\interlagos_2023_test
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

from vroom_interlagos.chunking.motion import MotionConfig, analyze_motion
from vroom_interlagos.cli import PROJECT_ROOT, base_parser, init_logging
from vroom_interlagos.common import dataclass_from_dict, load_yaml

logger = logging.getLogger("analyze_motion")


def main() -> int:
    parser = base_parser(__doc__.splitlines()[0])
    parser.add_argument("--frames", type=Path, required=True, help="Folder written by extract_frames.py")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "chunking.yaml")
    args = parser.parse_args()
    init_logging(args)

    try:
        cfg = dataclass_from_dict(MotionConfig, load_yaml(args.config).get("motion"))
        rows = analyze_motion(args.frames, cfg)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 1

    cont = [r for r in rows if r["continuous"]]
    if cont:
        logger.info("Pairs: %d (%d continuous, %d with valid pose)", len(rows), len(cont), sum(r["pose_valid"] for r in cont))
        logger.info("Median flow magnitude: %.2f px", float(np.median([r["flow_mag_median"] for r in cont])))
        logger.info("Mean LK survival     : %.3f", float(np.mean([r["lk_survival"] for r in cont])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
