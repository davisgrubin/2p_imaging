#!/usr/bin/env python3
"""Run suite2p post-processing pipeline (quality control, labeling, traces)."""

from __future__ import annotations

import argparse
import importlib
import logging
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np

LOGGER = logging.getLogger(__name__)

MODULE_PREFIX = "2p_post_process_module_202404.modules"
QualControlDataIO = importlib.import_module(f"{MODULE_PREFIX}.QualControlDataIO")
LabelExcInh = importlib.import_module(f"{MODULE_PREFIX}.LabelExcInh")
DffTraces = importlib.import_module(f"{MODULE_PREFIX}.DffTraces")

DEFAULT_MODES: Dict[str, Dict[str, Iterable[float]]] = {
    "dendrite": {
        "range_skew": (0.0, 2.0),
        "max_connect": 2,
        "range_aspect": (1.2, 5.0),
        "range_compact": (1.06, 5.0),
        "range_footprint": (1.0, 2.0),
        "diameter": 6.0,
    },
    "neuron": {
        "range_skew": (-5.0, 5.0),
        "max_connect": 1,
        "range_aspect": (0.0, 5.0),
        "range_compact": (0.0, 1.06),
        "range_footprint": (1.0, 2.0),
        "diameter": 6.0,
    },
}


def _load_ops(session_path: Path, plane_index: int) -> dict:
    ops_path = session_path / "suite2p" / f"plane{plane_index}" / "ops.npy"
    if not ops_path.exists():
        raise FileNotFoundError(f"ops.npy not found at {ops_path}")
    ops = np.load(ops_path, allow_pickle=True).item()
    ops["save_path0"] = str(session_path)
    return ops


def run_postprocess(
    session_path: Path,
    mode: str = "dendrite",
    plane_index: int = 0,
    correct_pmt: bool = False,
    range_skew: Optional[Iterable[float]] = None,
    max_connect: Optional[int] = None,
    range_aspect: Optional[Iterable[float]] = None,
    range_compact: Optional[Iterable[float]] = None,
    range_footprint: Optional[Iterable[float]] = None,
    diameter: Optional[float] = None,
) -> None:
    if mode not in DEFAULT_MODES:
        raise ValueError(f"Unknown mode '{mode}'. Available: {list(DEFAULT_MODES.keys())}")

    config = DEFAULT_MODES[mode].copy()

    def _maybe_update(key: str, value: Optional[Iterable[float]]):
        if value is not None:
            config[key] = value

    _maybe_update("range_skew", range_skew)
    if max_connect is not None:
        config["max_connect"] = max_connect
    _maybe_update("range_aspect", range_aspect)
    _maybe_update("range_compact", range_compact)
    _maybe_update("range_footprint", range_footprint)
    if diameter is not None:
        config["diameter"] = diameter

    ops = _load_ops(session_path, plane_index)

    LOGGER.info("Running QC on %s", session_path)
    QualControlDataIO.run(
        ops,
        np.asarray(config["range_skew"], dtype=float),
        int(config["max_connect"]),
        np.asarray(config["range_aspect"], dtype=float),
        np.asarray(config["range_compact"], dtype=float),
        np.asarray(config["range_footprint"], dtype=float),
    )

    LOGGER.info("Running anatomical labeling")
    LabelExcInh.run(ops, float(config["diameter"]))

    LOGGER.info("Regenerating ΔF/F traces")
    DffTraces.run(ops, correct_pmt=correct_pmt)


def parse_range(values: Optional[str]) -> Optional[Iterable[float]]:
    if values is None:
        return None
    parts = [float(v) for v in values.split(",")]
    if len(parts) != 2:
        raise ValueError("Ranges must be provided as 'min,max'")
    return parts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run suite2p post-processing (QC, labeling, dF/F)."
    )
    parser.add_argument(
        "session",
        type=Path,
        help="Local path to the session folder that contains suite2p outputs.",
    )
    parser.add_argument(
        "--mode",
        choices=list(DEFAULT_MODES.keys()),
        default="dendrite",
        help="Preset of QC parameters to use.",
    )
    parser.add_argument(
        "--plane-index",
        type=int,
        default=0,
        help="Suite2p plane index to process (default plane0).",
    )
    parser.add_argument(
        "--correct-pmt",
        action="store_true",
        help="Enable PMT correction when regenerating dF/F traces.",
    )
    parser.add_argument("--range-skew", type=parse_range, help="Override skew range, e.g. 0,2")
    parser.add_argument(
        "--max-connect",
        type=int,
        help="Override maximum connectivity threshold.",
    )
    parser.add_argument(
        "--range-aspect",
        type=parse_range,
        help="Override aspect ratio range, e.g. 1.2,5",
    )
    parser.add_argument(
        "--range-compact",
        type=parse_range,
        help="Override compactness range, e.g. 1.06,5",
    )
    parser.add_argument(
        "--range-footprint",
        type=parse_range,
        help="Override footprint range, e.g. 1,2",
    )
    parser.add_argument(
        "--diameter",
        type=float,
        help="Override Cellpose diameter.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    session_path = args.session.expanduser().resolve()
    if not session_path.exists():
        raise FileNotFoundError(f"Session path {session_path} not found")

    run_postprocess(
        session_path=session_path,
        mode=args.mode,
        plane_index=args.plane_index,
        correct_pmt=args.correct_pmt,
        range_skew=args.range_skew,
        max_connect=args.max_connect,
        range_aspect=args.range_aspect,
        range_compact=args.range_compact,
        range_footprint=args.range_footprint,
        diameter=args.diameter,
    )


if __name__ == "__main__":
    main()
