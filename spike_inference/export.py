#!/usr/bin/env python3
"""Utilities for exporting sessions to Colab-friendly formats."""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np

from .pipeline import MissingDependencyError

LOGGER = logging.getLogger(__name__)


def _stage_session(
    get_session_script: Path,
    source_root: Path,
    session_relpath: Path,
    dest_root: Path,
    dest_endpoint: Optional[str],
    force: bool,
) -> Path:
    dest_session_dir = dest_root / session_relpath
    if dest_session_dir.exists() and any(dest_session_dir.iterdir()) and not force:
        LOGGER.info("Using existing local session at %s", dest_session_dir)
        return dest_session_dir

    dest_session_dir.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(get_session_script),
        str((source_root / session_relpath).resolve()),
        str(dest_session_dir.parent.resolve()),
    ]
    if dest_endpoint:
        cmd.append(dest_endpoint)
    LOGGER.info("Running transfer: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return dest_session_dir


def _load_traces(loader_module: str, session_path: Path, plane_index: int) -> np.ndarray:
    module = importlib.import_module(loader_module)
    read_dff = getattr(module, "read_dff", None)
    if read_dff is None:
        raise MissingDependencyError(
            f"Loader module '{loader_module}' must expose a read_dff function."
        )

    read_ops = getattr(module, "read_ops", None)
    if read_ops is not None:
        ops_list = read_ops([str(session_path)])
    else:
        ops_list = _fallback_read_ops(session_path)

    if not ops_list:
        raise RuntimeError(f"No ops.npy metadata found under {session_path}")

    ops = ops_list[plane_index]
    dff = read_dff(ops)
    return np.asarray(dff, dtype=np.float32)


def _fallback_read_ops(session_path: Path) -> list:
    """Load ops.npy files directly when loaders do not expose read_ops."""
    suite2p_dir = session_path / "suite2p"
    if not suite2p_dir.exists():
        raise RuntimeError(f"No suite2p directory found in {session_path}")
    plane_dirs = sorted(
        p for p in suite2p_dir.iterdir() if p.is_dir() and p.name.startswith("plane")
    )
    ops_list = []
    for plane_dir in plane_dirs:
        ops_file = plane_dir / "ops.npy"
        if not ops_file.exists():
            continue
        ops = np.load(ops_file, allow_pickle=True).item()
        ops["save_path0"] = str(session_path)
        ops_list.append(ops)
    if not ops_list:
        raise RuntimeError(f"No ops.npy files found under {suite2p_dir}")
    return ops_list


def export_session(
    source_root: Path,
    session_relpath: Path,
    output_path: Path,
    get_session_script: Path,
    loader_module: str,
    dest_root: Path,
    dest_endpoint: Optional[str],
    overwrite: bool,
    plane_index: int,
    auto_scale: bool,
) -> Path:
    session_dir = _stage_session(
        get_session_script=get_session_script,
        source_root=source_root,
        session_relpath=session_relpath,
        dest_root=dest_root,
        dest_endpoint=dest_endpoint,
        force=overwrite,
    )

    traces = _load_traces(loader_module, session_dir, plane_index)
    metadata = {
        "session": str(session_relpath),
        "loader_module": loader_module,
        "plane_index": plane_index,
        "shape": list(traces.shape),
    }

    stds = np.nanstd(traces, axis=1)
    metadata["median_std"] = float(np.nanmedian(stds))
    metadata["mean_std"] = float(np.nanmean(stds))

    if auto_scale and np.nanmedian(stds) > 2:
        scale = 0.01
        LOGGER.info(
            "Median std %.2f suggests traces are in percent; scaling by %.4f.",
            np.nanmedian(stds),
            scale,
        )
        traces = traces * scale
        metadata["scaled_by"] = scale
    else:
        metadata["scaled_by"] = 1.0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, {"dF_traces": np.asarray(traces, dtype=np.float32)})

    with open(output_path.with_suffix(".json"), "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    LOGGER.info("Saved dF/F matrix to %s with shape %s", output_path, traces.shape)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download an SA11 joystick session and export ΔF/F traces for CASCADE Colab.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "session",
        type=str,
        help="Session path relative to the source root, e.g. SA11/YH18VT/YH18VT_Joystick_20240625",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("/storage/cedar/cedar0/cedarp-fnajafi3-0/2p_imaging"),
        help="Remote data root accessible via get_session.sh",
    )
    parser.add_argument(
        "--dest-root",
        type=Path,
        default=Path.cwd() / "data" / "sessions",
        help="Local directory under which sessions will be stored",
    )
    parser.add_argument(
        "--get-session-script",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "passive_interval_oddball_202412" / "get_session.sh",
        help="Path to the project get_session.sh helper",
    )
    parser.add_argument(
        "--loader-module",
        type=str,
        default="joystick_basic_202304.modules.ReadResults",
        help="Module providing read_ops/read_dff to load ΔF/F traces",
    )
    parser.add_argument(
        "--dest-endpoint",
        type=str,
        default=None,
        help="Globus destination endpoint UUID (if required by get_session.sh)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .npy file path. Defaults to <dest-root>/<session>/dff_traces.npy",
    )
    parser.add_argument(
        "--overwrite-session",
        action="store_true",
        help="Force re-download even if the session already exists locally",
    )
    parser.add_argument(
        "--plane-index",
        type=int,
        default=0,
        help="Index into the list returned by read_ops (useful if multiple planes exist)",
    )
    parser.add_argument(
        "--no-auto-scale",
        action="store_true",
        help="Disable automatic detection of percent-scaled traces",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> Path:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    session_relpath = Path(args.session)
    output_path = (
        args.output
        if args.output is not None
        else args.dest_root / session_relpath / "dff_traces.npy"
    )

    return export_session(
        source_root=args.source_root,
        session_relpath=session_relpath,
        output_path=output_path,
        get_session_script=args.get_session_script,
        loader_module=args.loader_module,
        dest_root=args.dest_root,
        dest_endpoint=args.dest_endpoint,
        overwrite=args.overwrite_session,
        plane_index=args.plane_index,
        auto_scale=not args.no_auto_scale,
    )


if __name__ == "__main__":
    main()
