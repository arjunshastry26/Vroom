r"""Create a small synthetic test clip (no real footage needed).

Example (PowerShell):
    python scripts\make_synthetic_clip.py --out data\raw\synthetic_test.mp4
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from vroom_interlagos.cli import PROJECT_ROOT, base_parser, init_logging
from vroom_interlagos.synthetic import SyntheticClipSpec, make_synthetic_clip

logger = logging.getLogger("make_synthetic_clip")


def main() -> int:
    parser = base_parser(__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "data" / "raw" / "synthetic_test.mp4")
    parser.add_argument("--duration", type=float, default=40.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    init_logging(args)
    d = args.duration
    spec = SyntheticClipSpec(
        duration_s=d,
        corners_s=[(0.2 * d, 0.26 * d), (0.5 * d, 0.56 * d), (0.8 * d, 0.86 * d)],
        seed=args.seed,
    )
    truth = make_synthetic_clip(args.out, spec)
    logger.info("Wrote %s (%d frames, corners at %s s)", args.out, truth["n_frames"], truth["corners_s"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
