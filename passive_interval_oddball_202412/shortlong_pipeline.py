#!/usr/bin/env python3
"""
Orchestrate the passive ShortLong session workflow end-to-end.

Steps per session:
    1. Discover remote sessions containing "ShortLong" in their name.
    2. Transfer each session locally via the existing get_session.sh helper.
    3. Run post-processing, generate neural trial data if missing, and make plots.
    4. Export plots to the user's Documents directory.
    5. Remove the session data from the local results folder after completion.

Up to five sessions are processed concurrently. Any failures are logged to
`pipeline_errors.log`, and processing continues with remaining sessions.
"""

from __future__ import annotations

import argparse
import logging
import os
import posixpath
import re
import shutil
import subprocess
import sys
import threading
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from generate_session_plots import CELL_SUBSETS, generate_plots_for_session  # noqa: E402
from modules import Trialization  # noqa: E402
from modules.ReadResults import read_ops  # noqa: E402

SALVAGE_DIR_NAMES = {
    "suite2p",
    "memmap",
    "cellpose",
    "qc_results",
}

SALVAGE_FILE_SUFFIXES = {
    ".h5",
    ".mat",
    ".npy",
    ".npz",
    ".json",
    ".csv",
    ".tsv",
    ".xlsx",
}


@dataclass(frozen=True)
class SessionJob:
    subject: str
    session: str
    remote_path: str


@dataclass(frozen=True)
class PipelineConfig:
    source_ep: str
    dest_ep: str
    remote_root: str
    local_root: Path
    output_root: Path
    get_script: Path
    postprocess_script: Path
    max_workers: int
    session_filter: Optional[Set[str]] = None
    cell_subsets: List[str] = field(default_factory=lambda: ["all", "exc", "vip", "sst"])
    subsample_neurons: Optional[int] = None
    subsample_seed: int = 0
    output_format: str = "pdf"
    keep_session_data: bool = False
    reuse_existing_sessions: bool = False
    balance_subsets: bool = False

    def command_env(self) -> Dict[str, str]:
        env = {
            "SOURCE_EP": self.source_ep,
            "DEST_EP": self.dest_ep,
        }
        return env


PLOT_LOCK = threading.Lock()


def setup_logging(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("shortlong_pipeline")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    logger.handlers.clear()

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", "%H:%M:%S"))
    logger.addHandler(console)

    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setLevel(logging.ERROR)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(file_handler)

    return logger


def run_command(
    cmd: List[str],
    *,
    cwd: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None,
    capture_output: bool = True,
) -> subprocess.CompletedProcess[str]:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)

    stdout = subprocess.PIPE if capture_output else None
    stderr = subprocess.PIPE if capture_output else None
    result = subprocess.run(
        cmd,
        cwd=str(cwd or SCRIPT_DIR),
        env=merged_env,
        text=True,
        stdout=stdout,
        stderr=stderr,
    )
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)
    return result


def ensure_remote_dir(remote_path: str) -> str:
    remote_path = remote_path.rstrip("/")
    return f"{remote_path}/"


def globus_list_dirs(config: PipelineConfig, remote_path: str) -> List[str]:
    cmd = [
        "globus",
        "ls",
        f"{config.source_ep}:{ensure_remote_dir(remote_path)}",
        "--jmespath",
        "DATA[?type=='dir'].name",
        "--format=UNIX",
    ]
    result = run_command(cmd, env=config.command_env())
    names: List[str] = []
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if "\t" in stripped or "  " in stripped:
            parts = [part for part in stripped.replace("\t", " ").split(" ") if part]
            names.extend(parts)
        else:
            names.append(stripped)
    return names


def discover_shortlong_sessions(
    config: PipelineConfig,
    logger: logging.Logger,
    subjects_filter: Optional[Iterable[str]] = None,
    sessions_filter: Optional[Set[str]] = None,
) -> List[SessionJob]:
    remote_root = config.remote_root.rstrip("/")
    subject_names = globus_list_dirs(config, remote_root)
    if subjects_filter:
        subject_filter_set = {s.strip() for s in subjects_filter if s.strip()}
        subject_names = [name for name in subject_names if name in subject_filter_set]

    jobs: List[SessionJob] = []
    for subject in sorted(subject_names):
        remote_subject = posixpath.join(remote_root, subject)
        try:
            session_names = globus_list_dirs(config, remote_subject)
        except subprocess.CalledProcessError as exc:
            logger.error("Failed to list sessions for %s: %s", subject, exc)
            continue
        shortlong_sessions = [session for session in session_names if "ShortLong" in session]
        for session in shortlong_sessions:
            if sessions_filter and session not in sessions_filter:
                continue
            remote_path = posixpath.join(remote_subject, session)
            jobs.append(SessionJob(subject, session, remote_path))
    logger.info("Discovered %d ShortLong sessions ready for processing", len(jobs))
    return jobs


def download_session(job: SessionJob, config: PipelineConfig, logger: logging.Logger) -> str:
    dest_parent = config.local_root / job.subject
    dest_parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "bash",
        str(config.get_script),
        job.remote_path,
        str(dest_parent),
    ]
    logger.info("Starting Globus transfer for %s/%s", job.subject, job.session)
    result = run_command(cmd, env=config.command_env())

    match = re.search(r"Task ID:\s*([0-9a-fA-F-]+)", result.stdout or "")
    if not match:
        raise RuntimeError(f"Could not determine Globus task ID for {job.subject}/{job.session}")
    task_id = match.group(1)

    logger.info("Waiting for Globus task %s (%s/%s)", task_id, job.subject, job.session)
    run_command(["globus", "task", "wait", task_id], env=config.command_env(), capture_output=False)
    status_result = run_command(
        ["globus", "task", "show", task_id, "--jmespath", "status", "--format=UNIX"],
        env=config.command_env(),
    )
    status = (status_result.stdout or "").strip().lower()
    if status != "succeeded":
        raise RuntimeError(f"Globus task {task_id} ended with status '{status}' for {job.subject}/{job.session}")

    logger.info("Completed transfer for %s/%s (task %s)", job.subject, job.session, task_id)


def run_postprocess(session_path: Path, config: PipelineConfig, logger: logging.Logger) -> None:
    logger.info("Running post-processing on %s", session_path)
    run_command(
        ["bash", str(config.postprocess_script), str(session_path)],
        capture_output=False,
    )


def ensure_neural_trials(session_path: Path, logger: logging.Logger) -> bool:
    neural_trials_path = session_path / "neural_trials.h5"
    if neural_trials_path.exists():
        logger.info("neural_trials.h5 already present at %s", neural_trials_path)
        return False

    logger.info("Generating neural_trials.h5 for %s", session_path)
    ops = read_ops([str(session_path)])[0]
    Trialization.run(ops)

    if not neural_trials_path.exists():
        raise RuntimeError(f"Trialization did not produce neural_trials.h5 at {neural_trials_path}")
    return True


def generate_plots(subject: str, session: str, config: PipelineConfig, logger: logging.Logger) -> None:
    logger.info("Generating plots for %s/%s", subject, session)
    with PLOT_LOCK:
        generate_plots_for_session(
            subject=subject,
            session_name=session,
            data_root=config.local_root,
            output_root=config.output_root,
            subsets=config.cell_subsets,
            subsample_neurons=config.subsample_neurons,
            balance_subsets=config.balance_subsets,
            plots_only=False,
            output_format=config.output_format,
            subsample_seed=config.subsample_seed,
        )
    logger.info("Finished plotting for %s/%s", subject, session)


def cleanup_session(session_path: Path, logger: logging.Logger) -> None:
    if session_path.exists():
        logger.info("Removing %s", session_path)
        shutil.rmtree(session_path)


def salvage_flat_download(local_subject_path: Path, local_session_path: Path, logger: logging.Logger) -> bool:
    """Detect and recover sessions downloaded directly into the subject folder."""
    if not local_subject_path.exists():
        return False
    candidates: List[Path] = []
    for entry in local_subject_path.iterdir():
        if entry.name == local_session_path.name:
            continue
        if entry.is_dir() and entry.name in SALVAGE_DIR_NAMES:
            candidates.append(entry)
        elif entry.is_file() and entry.suffix.lower() in SALVAGE_FILE_SUFFIXES:
            candidates.append(entry)
    if not candidates:
        return False
    logger.warning(
        "Found %d items for %s directly under %s; moving them into the session folder.",
        len(candidates),
        local_session_path.name,
        local_subject_path,
    )
    local_session_path.mkdir(parents=True, exist_ok=True)
    for entry in candidates:
        shutil.move(str(entry), str(local_session_path / entry.name))
    return True


def process_session(job: SessionJob, config: PipelineConfig, logger: logging.Logger) -> None:
    local_subject_path = config.local_root / job.subject
    local_session_path = local_subject_path / job.session
    downloaded = False

    if local_session_path.exists():
        if config.reuse_existing_sessions:
            logger.info(
                "Session %s already present locally; skipping download and post-processing",
                local_session_path,
            )
        else:
            logger.info("Removing existing data before reprocessing %s", local_session_path)
            shutil.rmtree(local_session_path)

    if not local_session_path.exists():
        task_id = download_session(job, config, logger)
        downloaded = True
        if not local_session_path.exists():
            if not salvage_flat_download(local_subject_path, local_session_path, logger) or not local_session_path.exists():
                raise FileNotFoundError(
                    f"Expected session folder {local_session_path} after download (task {task_id})"
                )
            logger.info("Recovered flattened download for %s/%s", job.subject, job.session)

    if not config.reuse_existing_sessions or downloaded:
        run_postprocess(local_session_path, config, logger)
        ensure_neural_trials(local_session_path, logger)

    generate_plots(job.subject, job.session, config, logger)

    if downloaded and not config.keep_session_data:
        cleanup_session(local_session_path, logger)
    elif not downloaded and not config.keep_session_data and not config.reuse_existing_sessions:
        cleanup_session(local_session_path, logger)


def handle_future(future: Future[None], job: SessionJob, logger: logging.Logger, failures: List[SessionJob]) -> None:
    try:
        future.result()
        logger.info("Successfully processed %s/%s", job.subject, job.session)
    except Exception as exc:
        failures.append(job)
        logger.error("Processing failed for %s/%s: %s", job.subject, job.session, exc, exc_info=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the passive ShortLong session pipeline.")
    parser.add_argument(
        "--remote-root",
        default="/storage/cedar/cedar0/cedarp-fnajafi3-0/2p_imaging/processed/passive",
        help="Remote Cedar directory containing session folders.",
    )
    parser.add_argument(
        "--source-endpoint",
        default="6df312ab-ad7c-4bbc-9369-450c82f0cb92",
        help="Globus endpoint UUID for the remote Cedar storage.",
    )
    parser.add_argument(
        "--dest-endpoint",
        default=os.environ.get("DEST_EP"),
        help="Local Globus endpoint UUID. Defaults to DEST_EP environment variable.",
    )
    parser.add_argument(
        "--local-root",
        type=Path,
        default=SCRIPT_DIR / "results",
        help="Local directory where sessions are downloaded (default: ./results).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path.home() / "Documents",
        help="Directory for generated plots (default: ~/Documents).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=5,
        help="Maximum number of concurrent sessions to process.",
    )
    parser.add_argument(
        "--subjects",
        nargs="*",
        help="Optional subject filter (space separated list of subject folders).",
    )
    parser.add_argument(
        "--sessions",
        nargs="*",
        help="Optional session folder filter (space separated list of session names).",
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
        "--decoder-subsample-neurons",
        type=int,
        default=None,
        help="Subsample each subset to this many neurons (default: auto min across subsets).",
    )
    parser.add_argument(
        "--decoder-subsample-seed",
        type=int,
        default=0,
        help="Random seed for neuron subsampling (default: 0).",
    )
    parser.add_argument(
        "--decoder-output-format",
        choices=["pdf", "png"],
        default="pdf",
        help="Figure format for decoder plots (default: pdf).",
    )
    parser.add_argument(
        "--decoder-balance-subsets",
        action="store_true",
        help="Subsample all neuron subsets to the same size (minimum across subsets).",
    )
    parser.add_argument(
        "--reuse-existing-sessions",
        action="store_true",
        help="If a session folder already exists locally, skip download and post-processing steps.",
    )
    parser.add_argument(
        "--keep-session-data",
        action="store_true",
        help="Skip deleting the session folder after processing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.dest_endpoint:
        raise SystemExit("Destination endpoint must be provided via --dest-endpoint or DEST_EP environment variable.")

    logger = setup_logging(args.log_file)
    logger.info("Starting ShortLong pipeline with up to %d concurrent jobs", args.max_workers)

    sessions_filter = {s.strip() for s in args.sessions if s.strip()} if args.sessions else None

    config = PipelineConfig(
        source_ep=args.source_endpoint,
        dest_ep=args.dest_endpoint,
        remote_root=args.remote_root,
        local_root=args.local_root.resolve(),
        output_root=args.output_root.expanduser().resolve(),
        get_script=SCRIPT_DIR / "get_session.sh",
        postprocess_script=SCRIPT_DIR / "run_postprocess.sh",
        max_workers=max(1, args.max_workers),
        session_filter=sessions_filter,
        cell_subsets=args.decoder_cell_subsets,
        subsample_neurons=args.decoder_subsample_neurons,
        subsample_seed=args.decoder_subsample_seed,
        output_format=args.decoder_output_format,
        keep_session_data=args.keep_session_data,
        reuse_existing_sessions=args.reuse_existing_sessions,
        balance_subsets=args.decoder_balance_subsets,
    )

    jobs = discover_shortlong_sessions(config, logger, args.subjects, sessions_filter)
    if not jobs:
        logger.info("No ShortLong sessions found. Nothing to do.")
        return

    failures: List[SessionJob] = []
    with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
        future_map = {executor.submit(process_session, job, config, logger): job for job in jobs}
        for future in as_completed(future_map):
            job = future_map[future]
            handle_future(future, job, logger, failures)

    if failures:
        logger.error("Completed with %d failures. See %s for details.", len(failures), args.log_file)
        for job in failures:
            logger.error(" - %s/%s", job.subject, job.session)
    else:
        logger.info("All sessions processed successfully.")


if __name__ == "__main__":
    main()
