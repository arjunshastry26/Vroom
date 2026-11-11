"""Helpers shared by the command-line scripts."""

from __future__ import annotations

import argparse
from pathlib import Path

from vroom_interlagos.common import setup_logging

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def base_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--log-file", type=Path, default=None, help="Also write the log to this file")
    return parser


def init_logging(args: argparse.Namespace) -> None:
    setup_logging(args.log_level, args.log_file)
