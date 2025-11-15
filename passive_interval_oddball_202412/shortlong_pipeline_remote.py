#!/usr/bin/env python3
"""
Process ShortLong sessions directly on a host that already has access to the raw
data (e.g., the PACE cluster). This variant skips all Globus/download logic and
simply walks the provided data root, runs post-processing/plotting, and writes
decoder exports to the desired output directory.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import threading
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set

from generate_session_plots import generate_plots_for_session, CELL_SUBSETS  # noqa: E402
from modules import Trialization  # noqa: E402
from modules.ReadResults import read_ops  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class SessionJob:
    subject: str
    session: str


@dataclass(frozen=True)
class PipelineConfig:
    data_root: Path
    output_root: Path
    postprocess_script: Path
    max_workers: int
    session_filter: Optional[Set[str]] = None
    cell_subsets: List[str] = field(default_factory=lambda: ["all", "exc", "vip", "sst"])
    bootstrap_seed: int = 0
    output_format: str = "pdf"
    require_postprocess: bool = True


PLOT_LOCK = threading.Lock()


def setup_logging(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("shortlong_pipeline_remote")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    logger.handlers.clear()

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", "%H:%M:%S"))
    logger.addHandler(console)

    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setLevel(logging.ERROR)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(file_handler)

    return logger


def run_postprocess(session_path: Path, config: PipelineConfig, logger: logging.Logger) -> None:
    if not config.require_postprocess:
        return
    logger.info("Running post-processing on %s", session_path)
    subprocess.run(
        ["bash", str(config.postprocess_script), str(session_path)],
        check=True,
    )


def ensure_neural_trials(session_path: Path, logger: logging.Logger) -> None:
    neural_trials_path = session_path / "neural_trials.h5"
    if neural_trials_path.exists():
        logger.info("neural_trials.h5 already present at %s", neural_trials_path)
        return

    logger.info("Generating neural_trials.h5 for %s", session_path)
    ops = read_ops([str(session_path)])[0]
    Trialization.run(ops)

    if not neural_trials_path.exists():
        raise RuntimeError(f"Trialization did not produce neural_trials.h5 at {neural_trials_path}")


def generate_plots(subject: str, session: str, config: PipelineConfig, logger: logging.Logger) -> None:
    logger.info("Generating plots for %s/%s", subject, session)
    with PLOT_LOCK:
        generate_plots_for_session(
            subject=subject,
            session_name=session,
            data_root=config.data_root,
            output_root=config.output_root,
            subsets=config.cell_subsets,
            plots_only=False,
            output_format=config.output_format,
            bootstrap_seed=config.bootstrap_seed,
        )
    logger.info("Finished plotting for %s/%s", subject, session)


def discover_sessions(
    data_root: Path,
    logger: logging.Logger,
    subjects_filter: Optional[Sequence[str]] = None,
    sessions_filter: Optional[Set[str]] = None,
    name_pattern: str = "ShortLong",
) -> List[SessionJob]:
    jobs: List[SessionJob] = []
    if not data_root.exists():
        logger.error("Data root %s does not exist.", data_root)
        return jobs

    subject_dirs = sorted(p for p in data_root.iterdir() if p.is_dir())
    if subjects_filter:
        allowed = {s.strip() for s in subjects_filter if s.strip()}
        subject_dirs = [p for p in subject_dirs if p.name in allowed]

    for subject_dir in subject_dirs:
        session_dirs = sorted(p for p in subject_dir.iterdir() if p.is_dir())
        for session_dir in session_dirs:
            if name_pattern and name_pattern not in session_dir.name:
                continue
            if sessions_filter and session_dir.name not in sessions_filter:
                continue
            jobs.append(SessionJob(subject=subject_dir.name, session=session_dir.name))

    logger.info("Discovered %d sessions ready for processing", len(jobs))
    return jobs


def process_session(job: SessionJob, config: PipelineConfig, logger: logging.Logger) -> None:
    session_path = config.data_root / job.subject / job.session
    if not session_path.exists():
        raise FileNotFoundError(f"Session folder not found: {session_path}")

    run_postprocess(session_path, config, logger)
    ensure_neural_trials(session_path, logger)
    generate_plots(job.subject, job.session, config, logger)


def handle_future(future: Future[None], job: SessionJob, logger: logging.Logger, failures: List[SessionJob]) -> None:
    try:
        future.result()
    except Exception:
        logger.exception("Processing failed for %s/%s", job.subject, job.session)
        failures.append(job)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process ShortLong sessions directly from a shared data root.")
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Path to the remote/shared data root containing subject/session folders.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path.home() / "Documents",
        help="Directory for generated decoder exports (default: ~/Documents).",
    )
    parser.add_argument(
        "--postprocess-script",
        type=Path,
        default=SCRIPT_DIR / "run_postprocess.sh",
        help="Path to the post-processing helper script.",
    )
    parser.add_argument(
        "--subjects",
        nargs="*",
        help="Optional subject filter (space-separated names).",
    )
    parser.add_argument(
        "--sessions",
        nargs="*",
        help="Optional session filter (space-separated names).",
    )
    parser.add_argument(
        "--name-pattern",
        default="ShortLong",
        help="Substring used to select session folders (default: ShortLong).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum number of concurrent sessions to process.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=SCRIPT_DIR / "pipeline_errors.log",
        help="Path to the error log file.",
    )
    parser.add_argument(
        "--decoder-cell-subsets",
        nargs="+",
        choices=sorted(CELL_SUBSETS.keys()),
        default=["all", "exc", "vip", "sst"],
        help="Neuron subsets to decode (default: all exc vip sst).",
    )
    parser.add_argument(
        "--decoder-bootstrap-seed",
        type=int,
        default=0,
        help="Random seed for neuron bootstrapping (default: 0).",
    )
    parser.add_argument(
        "--decoder-output-format",
        choices=["pdf", "png"],
        default="pdf",
        help="Figure format for decoder plots (default: pdf).",
    )
    parser.add_argument(
        "--skip-postprocess",
        action="store_true",
        help="Assume run_postprocess has already been executed for each session.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logging(args.log_file)
    logger.info("Starting remote ShortLong pipeline with up to %d workers", args.max_workers)

    sessions_filter = {s.strip() for s in args.sessions if s.strip()} if args.sessions else None

    config = PipelineConfig(
        data_root=args.data_root.expanduser().resolve(),
        output_root=args.output_root.expanduser().resolve(),
        postprocess_script=args.postprocess_script.resolve(),
        max_workers=max(1, args.max_workers),
        session_filter=sessions_filter,
        cell_subsets=args.decoder_cell_subsets,
        bootstrap_seed=args.decoder_bootstrap_seed,
        output_format=args.decoder_output_format,
        require_postprocess=not args.skip_postprocess,
    )

    jobs = discover_sessions(config.data_root, logger, args.subjects, sessions_filter, args.name_pattern)
    if not jobs:
        logger.info("No matching sessions found. Nothing to do.")
        return

    failures: List[SessionJob] = []
    with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
        future_map = {executor.submit(process_session, job, config, logger): job for job in jobs}
        for future in as_completed(future_map):
            job = future_map[future]
            handle_future(future, job, logger, failures)

    if failures:
        logger.error("Completed with %d failures.", len(failures))
        for job in failures:
            logger.error(" - %s/%s", job.subject, job.session)
        raise SystemExit(1)

    logger.info("All sessions processed successfully.")


if __name__ == "__main__":
    main()
