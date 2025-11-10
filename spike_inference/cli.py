#!/usr/bin/env python3
"""CLI entry point for multi-algorithm spike inference."""

from __future__ import annotations

import argparse
import importlib
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

from .pipeline import (
    CascadeBackend,
    ENS2Backend,
    MLSpikeBackend,
    MissingDependencyError,
    SpikeInferencePipeline,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Cascade, ENS², and MLSpike-style spike inference on suite2p outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config-module",
        type=str,
        default="passive_interval_oddball_202412.session_configs",
        help="Python module containing session configuration dictionaries.",
    )
    parser.add_argument(
        "--config-names",
        nargs="+",
        default=["session_config_list_PPC", "session_config_list_V1"],
        help="Attributes within the config module to process.",
    )
    parser.add_argument(
        "--session-names",
        nargs="+",
        default=None,
        help="Explicit session names (or folder/session keys) to process. When provided, "
        "only matching sessions are inferred across all configs.",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(os.environ.get("PASSIVE_SOURCE_ROOT", "")),
        help="Remote source root that hosts session folders.",
    )
    parser.add_argument(
        "--dest-endpoint",
        type=str,
        default=os.environ.get("PASSIVE_DEST_ENDPOINT", None),
        help="Destination Globus endpoint UUID for get_session.sh transfers.",
    )
    parser.add_argument(
        "--get-session-script",
        type=Path,
        default=None,
        help="Path to the project get_session.sh helper. Defaults to config_root/get_session.sh.",
    )
    parser.add_argument(
        "--loader-module",
        type=str,
        default=None,
        help="Module providing read_ops/read_dff helpers for loading ΔF/F traces. "
        "Defaults to '<config-module>.modules.ReadResults'.",
    )
    parser.add_argument(
        "--cascade-root",
        type=Path,
        default=Path(os.environ.get("CASCADE_ROOT", "")),
        help="Path to the HelmchenLab CASCADE repository root.",
    )
    parser.add_argument(
        "--cascade-model",
        type=str,
        default="Universal_30Hz_smoothing100ms",
        help="Cascade pretrained model name to use.",
    )
    parser.add_argument(
        "--cascade-auto-download",
        action="store_true",
        help="Attempt to download the Cascade model if missing locally.",
    )
    parser.add_argument(
        "--ens2-root",
        type=Path,
        default=Path(os.environ.get("ENS2_ROOT", "")),
        help="Path to the ENS² repository root.",
    )
    parser.add_argument(
        "--ens2-checkpoint",
        type=Path,
        default=Path(os.environ.get("ENS2_CHECKPOINT", "")),
        help="Path to a pre-trained ENS² checkpoint (.pt).",
    )
    parser.add_argument(
        "--ens2-neuron-type",
        type=str,
        choices=["Exc", "Inh", "Both"],
        default="Exc",
        help="Neuron cohort for ENS² checkpoint selection.",
    )
    parser.add_argument(
        "--ens2-smoothing-std",
        type=float,
        default=0.025,
        help="Gaussian smoothing std (s) used by ENS².",
    )
    parser.add_argument(
        "--ens2-normalise",
        action="store_true",
        help="Enable ENS² input normalisation.",
    )
    parser.add_argument(
        "--disable-cascade",
        action="store_true",
        help="Skip Cascade inference even if dependencies are available.",
    )
    parser.add_argument(
        "--disable-ens2",
        action="store_true",
        help="Skip ENS² inference.",
    )
    parser.add_argument(
        "--disable-mlspike",
        action="store_true",
        help="Skip the MLSpike-style deconvolution backend.",
    )
    parser.add_argument(
        "--mlspike-tau",
        type=float,
        default=1.5,
        help="Decay constant (s) for the MLSpike-style AR(1) model.",
    )
    parser.add_argument(
        "--mlspike-amplitude",
        type=float,
        default=1.0,
        help="Amplitude scaling for MLSpike-style inference.",
    )
    parser.add_argument(
        "--mlspike-event-height",
        type=float,
        default=0.2,
        help="Event detection threshold for MLSpike-style inference.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Directory where inference outputs are stored. "
        "Defaults to config_root/results/spike_inference.",
    )
    parser.add_argument(
        "--skip-transfer",
        action="store_true",
        help="Assume session data already present locally; do not run get_session.sh.",
    )
    parser.add_argument(
        "--max-neurons",
        type=int,
        default=None,
        help="Optional cap on neurons per session (first N neurons retained).",
    )
    parser.add_argument(
        "--limit-sessions",
        type=int,
        default=None,
        help="Limit the number of sessions processed per config.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def build_backends(args: argparse.Namespace) -> List[object]:
    backends: List[object] = []

    if not args.disable_cascade and args.cascade_root:
        try:
            backends.append(
                CascadeBackend(
                    cascade_root=args.cascade_root,
                    model_name=args.cascade_model,
                    auto_download=args.cascade_auto_download,
                )
            )
        except MissingDependencyError as exc:
            logging.getLogger("spike_inference").warning("Cascade disabled: %s", exc)
    elif not args.disable_cascade:
        logging.getLogger("spike_inference").warning(
            "Cascade backend skipped (no --cascade-root provided)."
        )

    if not args.disable_ens2 and args.ens2_root and args.ens2_checkpoint:
        try:
            backends.append(
                ENS2Backend(
                    ens2_root=args.ens2_root,
                    checkpoint_path=args.ens2_checkpoint,
                    neuron_type=args.ens2_neuron_type,
                    smoothing_std=args.ens2_smoothing_std,
                    normalise=args.ens2_normalise,
                )
            )
        except MissingDependencyError as exc:
            logging.getLogger("spike_inference").warning("ENS² disabled: %s", exc)
    elif not args.disable_ens2:
        logging.getLogger("spike_inference").warning(
            "ENS² backend skipped (ensure --ens2-root and --ens2-checkpoint are set)."
        )

    if not args.disable_mlspike:
        backends.append(
            MLSpikeBackend(
                tau_decay=args.mlspike_tau,
                amplitude=args.mlspike_amplitude,
                event_height=args.mlspike_event_height,
            )
        )

    if not backends:
        raise RuntimeError("No spike inference backends configured; aborting.")
    return backends


def _resolve_defaults(
    args: argparse.Namespace, config_module
) -> Dict[str, Optional[object]]:
    config_root = Path(config_module.__file__).resolve().parent
    get_session_script = (
        args.get_session_script
        if args.get_session_script is not None
        else config_root / "get_session.sh"
    )
    loader_module = (
        args.loader_module
        if args.loader_module is not None
        else f"{args.config_module.rsplit('.', 1)[0]}.modules.ReadResults"
        if "." in args.config_module
        else f"{args.config_module}.modules.ReadResults"
    )
    output_root = args.output_root or (config_root / "results" / "spike_inference")
    return {
        "config_root": config_root,
        "get_session_script": Path(get_session_script),
        "loader_module": loader_module,
        "output_root": Path(output_root),
    }


def main():
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
    )
    logger = logging.getLogger("spike_inference")

    if not args.source_root:
        raise RuntimeError("Provide --source-root pointing to the remote session root.")

    config_module = importlib.import_module(args.config_module)
    defaults = _resolve_defaults(args, config_module)
    get_session_script: Path = defaults["get_session_script"]
    loader_module: str = defaults["loader_module"]  # type: ignore[assignment]
    output_root: Path = defaults["output_root"]
    config_root: Path = defaults["config_root"]  # type: ignore[assignment]

    if not get_session_script.exists():
        raise FileNotFoundError(
            f"get_session.sh script not found at {get_session_script}"
        )

    backends = build_backends(args)
    session_targets = args.session_names

    for config_name in args.config_names:
        if not hasattr(config_module, config_name):
            logger.warning("Config '%s' not found in %s; skipping.", config_name, args.config_module)
            continue
        config = getattr(config_module, config_name)
        subject = config.get("subject_name", config_name)
        region_logger = logging.getLogger(f"spike_inference.{subject}")
        region_output = output_root / subject
        pipeline = SpikeInferencePipeline(
            project_root=config_root,
            session_config_list=config,
            backends=backends,
            get_session_script=get_session_script,
            source_root=args.source_root,
            dest_endpoint=args.dest_endpoint,
            output_root=region_output,
            loader_module=loader_module,
            skip_transfer=args.skip_transfer,
            max_neurons=args.max_neurons,
            logger=region_logger,
            target_sessions=session_targets,
        )
        if session_targets and not pipeline.session_specs:
            logger.warning("No sessions matched --session-names for config %s", config_name)
            continue
        logger.info("Starting spike inference for %s", subject)
        pipeline.run(limit=args.limit_sessions)
        logger.info("Finished %s. Outputs stored in %s", subject, region_output)


if __name__ == "__main__":
    main()
