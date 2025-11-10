#!/usr/bin/env python3
"""
Retry transfer-related session failures recorded in pipeline_errors.log.

The script parses the log, identifies sessions that failed during the
Globus/get_session phase, skips any sessions that already have decoder
outputs in the plot directory, and reruns shortlong_pipeline.py with
filters so only the missing sessions are processed.
"""

from __future__ import annotations

import argparse
import re
import sys
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Iterable, List, Optional, Set

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_LOG = SCRIPT_DIR / "pipeline_errors.log"
DEFAULT_PIPELINE = SCRIPT_DIR / "shortlong_pipeline.py"
DEFAULT_OUTPUT = Path.home() / "Documents"

LOG_PATTERN = re.compile(r"Processing failed for (?P<subject>[^/]+)/(?P<session>[^:]+): (?P<message>.+)")
TRANSFER_KEYWORDS = ("globus", "get_session", "task wait")


def parse_failures(log_path: Path) -> DefaultDict[str, Set[str]]:
    failures: DefaultDict[str, Set[str]] = defaultdict(set)
    if not log_path.exists():
        return failures
    with log_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            match = LOG_PATTERN.search(line)
            if not match:
                continue
            message = match.group("message").lower()
            if not any(keyword in message for keyword in TRANSFER_KEYWORDS):
                continue
            subject = match.group("subject").strip()
            session = match.group("session").strip()
            if subject and session:
                failures[subject].add(session)
    return failures


def outputs_exist(output_root: Path, subject: str, session: str) -> bool:
    if not output_root.exists():
        return False
    for region_dir in output_root.iterdir():
        if not region_dir.is_dir():
            continue
        candidate = region_dir / subject / session
        if candidate.is_dir():
            try:
                next(candidate.iterdir())
                return True
            except StopIteration:
                continue
    return False


def build_retry_plan(
    failures: DefaultDict[str, Set[str]],
    output_root: Path,
) -> DefaultDict[str, List[str]]:
    plan: DefaultDict[str, List[str]] = defaultdict(list)
    for subject, sessions in failures.items():
        for session in sorted(sessions):
            if outputs_exist(output_root, subject, session):
                continue
            plan[subject].append(session)
    return plan


def run_pipeline(
    pipeline_path: Path,
    subject: str,
    sessions: Iterable[str],
    dest_endpoint: str | None,
    cell_subsets: List[str],
    subsample_neurons: Optional[int],
    subsample_seed: int,
    output_format: str,
    reuse_existing: bool,
    keep_session_data: bool,
    balance_subsets: bool,
    extra_args: List[str],
) -> None:
    cmd = [sys.executable, str(pipeline_path), "--subjects", subject, "--sessions", *sessions]
    if dest_endpoint:
        cmd.extend(["--dest-endpoint", dest_endpoint])
    if cell_subsets:
        cmd.extend(["--decoder-cell-subsets", *cell_subsets])
    if subsample_neurons is not None:
        cmd.extend(["--decoder-subsample-neurons", str(subsample_neurons)])
    if subsample_seed is not None:
        cmd.extend(["--decoder-subsample-seed", str(subsample_seed)])
    if output_format:
        cmd.extend(["--decoder-output-format", output_format])
    if balance_subsets:
        cmd.append("--decoder-balance-subsets")
    if reuse_existing:
        cmd.append("--reuse-existing-sessions")
    if keep_session_data:
        cmd.append("--keep-session-data")
    cmd.extend(extra_args)
    print(f"Retrying {subject}: {', '.join(sessions)}")
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retry transfer failures from pipeline_errors.log.")
    parser.add_argument("--log-file", type=Path, default=DEFAULT_LOG, help="Path to pipeline error log.")
    parser.add_argument("--pipeline-script", type=Path, default=DEFAULT_PIPELINE, help="Path to shortlong_pipeline.py.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT, help="Plot output directory to check before retrying.")
    parser.add_argument("--dest-endpoint", help="Override DEST_EP for retries (optional).")
    parser.add_argument(
        "--cell-subsets",
        nargs="+",
        default=["all", "exc", "vip", "sst"],
        help="Neuron subsets to decode during retry (default: all exc vip sst).",
    )
    parser.add_argument(
        "--subsample-neurons",
        type=int,
        default=None,
        help="Optional neuron subsample cap passed to the pipeline.",
    )
    parser.add_argument(
        "--subsample-seed",
        type=int,
        default=0,
        help="Random seed for neuron subsampling (default: 0).",
    )
    parser.add_argument(
        "--output-format",
        choices=["pdf", "png"],
        default="pdf",
        help="Figure format for regenerated plots (default: pdf).",
    )
    parser.add_argument(
        "--reuse-existing-sessions",
        action="store_true",
        help="Reuse any existing session folders instead of re-downloading.",
    )
    parser.add_argument(
        "--keep-session-data",
        action="store_true",
        help="Keep session data on disk after retrying.",
    )
    parser.add_argument(
        "--overwrite-plots",
        action="store_true",
        help="Force rerunning retries even if exports already exist.",
    )
    parser.add_argument(
        "--balance-subsets",
        action="store_true",
        help="Request decoder balancing of neuron subset sizes during retry.",
    )
    parser.add_argument(
        "--pipeline-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Extra arguments passed through to the pipeline after the filters.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()
    failures = parse_failures(args.log_file)
    if not failures:
        print("No transfer-related failures found in the log.")
        return

    retry_plan = build_retry_plan(failures, output_root)
    if not retry_plan:
        if args.overwrite_plots:
            retry_plan = failures
        else:
            print("All transfer-related failures already have outputs in the plot directory.")
            return

    pipeline_path = args.pipeline_script.resolve()
    extra_args = [arg for arg in args.pipeline_args if arg]
    for subject, sessions in retry_plan.items():
        run_pipeline(
            pipeline_path,
            subject,
            sessions,
            args.dest_endpoint,
            args.cell_subsets,
            args.subsample_neurons,
            args.subsample_seed,
            args.output_format,
            args.reuse_existing_sessions,
            args.keep_session_data,
            args.balance_subsets,
            extra_args,
        )
    print("Retry attempts completed.")


if __name__ == "__main__":
    main()
