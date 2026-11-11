r"""Estimate a mask of static image regions (car body, halo, logos) from extracted frames.

Example (PowerShell):
    python scripts\make_static_mask.py --frames data\frames\interlagos_2023_test
"""

from __future__ import annotations

import logging
import sys
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from vroom_interlagos.chunking.motion import usable_frames
from vroom_interlagos.cli import PROJECT_ROOT, base_parser, init_logging
from vroom_interlagos.common import portable_path, sha256_file, utc_now_iso, write_json
from vroom_interlagos.masking.static_mask import StaticMaskConfig, estimate_static_mask
from vroom_interlagos.preprocessing.frames import load_frame_table

logger = logging.getLogger("make_static_mask")


def main() -> int:
    parser = base_parser(__doc__.splitlines()[0])
    parser.add_argument("--frames", type=Path, required=True, help="Folder written by extract_frames.py")
    parser.add_argument("--out-dir", type=Path, default=None, help="Default: data/masks/<frames folder name>")
    parser.add_argument("--std-threshold", type=float, default=StaticMaskConfig.std_threshold,
                        help="Grey-level temporal std below which a pixel is static; raise to mask more")
    parser.add_argument("--open-px", type=int, default=StaticMaskConfig.open_px)
    parser.add_argument("--close-px", type=int, default=StaticMaskConfig.close_px)
    args = parser.parse_args()
    init_logging(args)

    cfg = StaticMaskConfig(std_threshold=args.std_threshold, open_px=args.open_px, close_px=args.close_px)
    out_dir = args.out_dir or PROJECT_ROOT / "data" / "masks" / args.frames.name
    try:
        frames = usable_frames(load_frame_table(args.frames))

        def grays():
            for rec in frames:
                img = cv2.imread(str(args.frames / rec.filename), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    raise FileNotFoundError(f"Could not read {args.frames / rec.filename}")
                yield img

        mask, std = estimate_static_mask(grays(), cfg)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    mask_path = out_dir / "static_mask.png"
    cv2.imwrite(str(mask_path), mask)
    # Preview: a mid-clip frame with the ignored region tinted red.
    preview = cv2.imread(str(args.frames / frames[len(frames) // 2].filename))
    tint = preview.copy()
    tint[mask == 0] = (0, 0, 255)
    cv2.imwrite(str(out_dir / "static_mask_preview.jpg"), cv2.addWeighted(preview, 0.5, tint, 0.5, 0))
    cv2.imwrite(str(out_dir / "temporal_std.png"), np.clip(std * 4, 0, 255).astype(np.uint8))

    masked_fraction = float((mask == 0).mean())
    write_json(
        out_dir / "static_mask.json",
        {
            "created_utc": utc_now_iso(),
            "method": "temporal standard deviation threshold (vroom_interlagos.masking.static_mask)",
            "frames_dir": portable_path(args.frames, PROJECT_ROOT),
            "n_frames_used": len(frames),
            "config": asdict(cfg),
            "masked_fraction": masked_fraction,
            "convention": "255 = analyse, 0 = ignore",
            "sha256": sha256_file(mask_path),
        },
    )
    logger.info("Mask    : %s (%.1f%% of the image ignored)", mask_path, 100 * masked_fraction)
    logger.info("Preview : %s  (red = ignored; check it before use)", out_dir / "static_mask_preview.jpg")
    logger.info("To use it, set motion.mask_path in configs/chunking.yaml to: %s", portable_path(mask_path, PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
