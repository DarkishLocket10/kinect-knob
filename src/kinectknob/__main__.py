"""CLI entry point: python -m kinectknob"""
from __future__ import annotations

import argparse
import sys

from .config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="kinect-knob",
        description="Hand-gesture volume knob for Home Assistant, driven by an Xbox Kinect.",
    )
    parser.add_argument("--config", help="path to config.yaml (env vars override it)")
    parser.add_argument(
        "--backend",
        choices=["auto", "kinect1", "kinect2", "webcam"],
        help="capture backend (overrides config)",
    )
    parser.add_argument("--port", type=int, help="web UI port (overrides config)")
    parser.add_argument(
        "--preview", action="store_true",
        help="show a local OpenCV preview window (development)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.backend:
        cfg.capture.backend = args.backend
    if args.port:
        cfg.web.port = args.port

    from .main import run_app

    sys.exit(run_app(cfg, preview=args.preview))


if __name__ == "__main__":
    main()
