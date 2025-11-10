#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from dash_viewer.app import create_dash_app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the Dash interactive decoding viewer.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("plot_exports"),
        help="Directory containing decoder exports (default: plot_exports).",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host interface for the Dash server (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8050,
        help="Port for the Dash server (default: 8050).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable Dash debug mode.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = create_dash_app(args.data_root)
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
