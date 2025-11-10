#!/usr/bin/env python3
"""Reusable spike inference orchestration for 2p imaging projects."""

from __future__ import annotations

import importlib
import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.signal import find_peaks

LOGGER = logging.getLogger(__name__)


class MissingDependencyError(RuntimeError):
    """Raised when an external spike inference backend is unavailable."""


def _json_default(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value)!r} is not JSON serialisable")


@dataclass(frozen=True)
class SessionSpec:
    """Describes one session within a configuration."""

    session_name: str
    session_folder: str
    label: str
    project_root: Path

    @property
    def local_path(self) -> Path:
        return self.project_root / "results" / self.session_folder / self.session_name

    @property
    def source_relpath(self) -> Path:
        return Path(self.session_folder) / self.session_name

    @property
    def key(self) -> str:
        return f"{self.session_folder}/{self.session_name}"


@dataclass
class SpikeInferenceResult:
    """Container for rate- and event-style outputs."""

    rate: np.ndarray
    spikes: np.ndarray
    events: List[np.ndarray]
    metadata: Dict[str, object] = field(default_factory=dict)


class SpikeInferenceBackend:
    """Abstract base interface for spike inference backends."""

    name: str

    def __init__(self, name: str):
        self.name = name

    def predict(self, dff: np.ndarray, fs_hz: float) -> SpikeInferenceResult:
        raise NotImplementedError


def _ensure_path_on_syspath(path: Path):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _extract_events_from_trace(
    trace: np.ndarray, fs_hz: float, height: float, prominence: Optional[float]
) -> np.ndarray:
    if not np.any(np.isfinite(trace)):
        return np.empty(0, dtype=np.float32)
    peaks, _ = find_peaks(trace, height=height, prominence=prominence)
    return peaks.astype(np.float32) / float(fs_hz)


def _extract_events_from_matrix(
    matrix: np.ndarray, fs_hz: float, height: float, prominence: Optional[float]
) -> List[np.ndarray]:
    return [
        _extract_events_from_trace(row, fs_hz, height=height, prominence=prominence)
        for row in matrix
    ]


class CascadeBackend(SpikeInferenceBackend):
    """Wrapper around the HelmchenLab CASCADE implementation."""

    def __init__(
        self,
        cascade_root: Path,
        model_name: str,
        model_folder: Optional[Path] = None,
        threshold: int = 1,
        padding: float = 0.0,
        event_height: float = 0.1,
        event_prominence: Optional[float] = None,
        auto_download: bool = False,
        verbosity: int = 0,
    ):
        super().__init__("cascade")
        _ensure_path_on_syspath(cascade_root)
        try:
            from cascade2p import cascade as cascade_module
        except ImportError as exc:
            raise MissingDependencyError(
                "Cascade backend unavailable. Provide --cascade-root pointing to a "
                "local CASCADE checkout with dependencies installed."
            ) from exc

        self._cascade = cascade_module
        self.model_name = model_name
        self.model_folder = model_folder or (cascade_root / "Pretrained_models")
        self.threshold = threshold
        self.padding = padding
        self.event_height = event_height
        self.event_prominence = event_prominence
        self.verbosity = verbosity

        if auto_download:
            try:
                self._cascade.download_model(
                    model_name, model_folder=str(self.model_folder), verbose=verbosity
                )
            except Exception as exc:  # pragma: no cover - external side effect
                LOGGER.warning(
                    "Unable to auto-download Cascade model '%s': %s", model_name, exc
                )

    def predict(self, dff: np.ndarray, fs_hz: float) -> SpikeInferenceResult:
        traces = np.asarray(dff, dtype=np.float32)
        predictions = self._cascade.predict(
            model_name=self.model_name,
            traces=traces,
            model_folder=str(self.model_folder),
            threshold=self.threshold,
            padding=self.padding,
            verbosity=self.verbosity,
        )
        rate = np.asarray(predictions, dtype=np.float32)
        spikes = np.clip(rate, a_min=0.0, a_max=None).astype(np.float32)
        events = _extract_events_from_matrix(
            spikes, fs_hz, height=self.event_height, prominence=self.event_prominence
        )
        metadata = {
            "model_name": self.model_name,
            "model_folder": str(self.model_folder),
            "threshold": self.threshold,
            "padding": self.padding,
            "event_height": self.event_height,
            "event_prominence": self.event_prominence,
        }
        return SpikeInferenceResult(rate=rate, spikes=spikes, events=events, metadata=metadata)


class ENS2Backend(SpikeInferenceBackend):
    """Wrapper around the ENS² PyTorch implementation."""

    def __init__(
        self,
        ens2_root: Path,
        checkpoint_path: Path,
        neuron_type: str = "Exc",
        device: Optional[str] = None,
        smoothing_std: float = 0.025,
        normalise: bool = False,
        event_height: float = 0.1,
    ):
        super().__init__("ens2")
        _ensure_path_on_syspath(ens2_root)
        try:
            import torch
        except ImportError as exc:
            raise MissingDependencyError("ENS² backend requires PyTorch.") from exc

        try:
            import ENS2 as ens2_module  # type: ignore
        except ImportError as exc:
            raise MissingDependencyError(
                "Unable to import ENS2. Provide --ens2-root pointing to the ENS² repository."
            ) from exc

        self._torch = torch
        self._ens2_module = ens2_module
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = ens2_module.ENS2()
        self._model.DEVICE = self._device
        self._compile_test_data = ens2_module.compile_test_data
        self._normalise = normalise
        self._smoothing_std = smoothing_std
        self._event_height = event_height
        self._neuron_type = neuron_type

        raw_model = torch.load(checkpoint_path, map_location=torch.device(self._device))
        if hasattr(raw_model, "state_dict"):
            self._state_dict = raw_model.state_dict()
        elif isinstance(raw_model, Mapping) and "state_dict" in raw_model:
            self._state_dict = raw_model["state_dict"]
        else:  # pragma: no cover - alternate checkpoint structures
            self._state_dict = raw_model

    def predict(self, dff: np.ndarray, fs_hz: float) -> SpikeInferenceResult:
        ens2_module = self._ens2_module
        ens2_module.opt.sampling_rate = float(fs_hz)
        ens2_module.opt.smoothing_std = float(self._smoothing_std)
        ens2_module.opt.smoothing = ens2_module.opt.smoothing_std * ens2_module.opt.sampling_rate

        traces = np.asarray(dff, dtype=np.float32)
        trial_time = traces.shape[1] / float(fs_hz)
        compiled = self._compile_test_data(
            traces, trial_time, is_norm=self._normalise, is_denoise=0
        )

        rate_list: List[np.ndarray] = []
        spike_list: List[np.ndarray] = []
        events: List[np.ndarray] = []

        for entry in compiled:
            segments = entry["dff_resampled_segment"]
            _, pd_rate, pd_spike, pd_event = self._model.predict(
                segments, state_dict=self._state_dict
            )
            rate_list.append(np.asarray(pd_rate, dtype=np.float32).squeeze())
            spike_list.append(np.asarray(pd_spike, dtype=np.float32).squeeze())
            events.append(np.asarray(pd_event, dtype=np.float32))

        rate = np.vstack(rate_list)
        spikes = np.vstack(spike_list)
        metadata = {
            "checkpoint_device": self._device,
            "smoothing_std": self._smoothing_std,
            "normalise": self._normalise,
            "event_height": self._event_height,
            "neuron_type": self._neuron_type,
        }
        return SpikeInferenceResult(rate=rate, spikes=spikes, events=events, metadata=metadata)


class MLSpikeBackend(SpikeInferenceBackend):
    """Lightweight proxy mimicking MLSpike behaviour via AR(1) deconvolution."""

    def __init__(
        self,
        tau_decay: float = 1.5,
        amplitude: float = 1.0,
        baseline_percentile: float = 10.0,
        event_height: float = 0.2,
    ):
        super().__init__("mlspike")
        self.tau_decay = tau_decay
        self.amplitude = amplitude
        self.baseline_percentile = baseline_percentile
        self.event_height = event_height

    def predict(self, dff: np.ndarray, fs_hz: float) -> SpikeInferenceResult:
        traces = np.asarray(dff, dtype=np.float32)
        baseline = np.percentile(traces, self.baseline_percentile, axis=1, keepdims=True)
        centred = traces - baseline

        decay = float(np.exp(-1.0 / (fs_hz * self.tau_decay)))
        n_cells, n_time = centred.shape
        spikes = np.zeros((n_cells, n_time), dtype=np.float32)
        rate = np.zeros((n_cells, n_time), dtype=np.float32)

        for cell_idx in range(n_cells):
            c_prev = 0.0
            for t in range(n_time):
                predicted = decay * c_prev
                residual = centred[cell_idx, t] - predicted
                s_t = residual / self.amplitude
                if s_t < 0:
                    s_t = 0.0
                spikes[cell_idx, t] = s_t
                c_curr = predicted + s_t
                rate[cell_idx, t] = c_curr
                c_prev = c_curr

        events = _extract_events_from_matrix(
            spikes, fs_hz, height=self.event_height, prominence=None
        )
        metadata = {
            "tau_decay": self.tau_decay,
            "amplitude": self.amplitude,
            "baseline_percentile": self.baseline_percentile,
            "event_height": self.event_height,
        }
        return SpikeInferenceResult(rate=rate, spikes=spikes, events=events, metadata=metadata)


def _events_to_mask(events: Sequence[np.ndarray], n_time: int, fs_hz: float) -> np.ndarray:
    mask = np.zeros((len(events), n_time), dtype=bool)
    for idx, ev in enumerate(events):
        if ev.size == 0:
            continue
        frame_idx = np.clip(np.round(ev * fs_hz).astype(int), 0, n_time - 1)
        mask[idx, frame_idx] = True
    return mask


def summarise_outputs(outputs: Dict[str, SpikeInferenceResult], fs_hz: float) -> Dict[str, object]:
    if not outputs:
        return {}

    per_algorithm: Dict[str, Dict[str, float]] = {}
    for name, result in outputs.items():
        spikes = result.spikes
        events = result.events
        rate_mean = float(np.nanmean(spikes))
        total_frames = spikes.shape[1]
        duration_s = total_frames / float(fs_hz)
        event_counts = np.array([len(ev) for ev in events], dtype=np.float32)
        per_algorithm[name] = {
            "mean_spike_per_sample": rate_mean,
            "mean_spike_rate_hz": float(rate_mean * fs_hz),
            "median_events_per_neuron": float(np.nanmedian(event_counts)),
            "median_event_rate_hz": float(np.nanmedian(event_counts) / duration_s),
        }

    pairwise_corr: Dict[str, float] = {}
    pairwise_jaccard: Dict[str, float] = {}
    algo_names = sorted(outputs.keys())
    for left, right in combinations(algo_names, 2):
        res_l = outputs[left]
        res_r = outputs[right]
        n_time = min(res_l.rate.shape[1], res_r.rate.shape[1])
        rate_l = res_l.rate[:, :n_time]
        rate_r = res_r.rate[:, :n_time]
        corrs: List[float] = []
        for row_l, row_r in zip(rate_l, rate_r):
            if not (np.any(np.isfinite(row_l)) and np.any(np.isfinite(row_r))):
                continue
            corr_mat = np.corrcoef(row_l, row_r)
            if corr_mat.shape == (2, 2):
                corrs.append(float(corr_mat[0, 1]))
        pairwise_corr[f"{left}__{right}"] = float(np.nanmedian(corrs)) if corrs else float("nan")

        mask_l = _events_to_mask(res_l.events, n_time, fs_hz)
        mask_r = _events_to_mask(res_r.events, n_time, fs_hz)
        if mask_l.shape != mask_r.shape:
            min_cells = min(mask_l.shape[0], mask_r.shape[0])
            mask_l = mask_l[:min_cells]
            mask_r = mask_r[:min_cells]
        unions = mask_l | mask_r
        intersections = mask_l & mask_r
        union_count = unions.sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            jacc = np.divide(
                intersections.sum(axis=1),
                union_count,
                out=np.full_like(union_count, np.nan, dtype=float),
                where=union_count > 0,
            )
        pairwise_jaccard[f"{left}__{right}"] = float(np.nanmedian(jacc))

    return {
        "per_algorithm": per_algorithm,
        "pairwise_correlation": pairwise_corr,
        "pairwise_event_jaccard": pairwise_jaccard,
    }


def _flatten_session_config(
    session_config_list: Mapping[str, object], project_root: Path
) -> List[SessionSpec]:
    flat: List[SessionSpec] = []
    for config in session_config_list.get("list_config", []):  # type: ignore[arg-type]
        folder = config["session_folder"]
        labels = config["list_session_name"]
        for session_name, label in labels.items():
            flat.append(
                SessionSpec(
                    session_name=session_name,
                    session_folder=folder,
                    label=label,
                    project_root=project_root,
                )
            )
    return flat


def _ensure_session_data(
    spec: SessionSpec,
    script_path: Path,
    source_root: Path,
    dest_endpoint: Optional[str],
    skip_transfer: bool,
):
    if spec.local_path.exists() and any(spec.local_path.iterdir()):
        return
    if skip_transfer:
        raise FileNotFoundError(
            f"Session {spec.key} not present locally and transfer disabled."
        )
    args = [str(script_path), str(source_root / spec.source_relpath), str(spec.local_path.parent)]
    if dest_endpoint:
        args.append(dest_endpoint)
    env = os.environ.copy()
    LOGGER.info("Fetching session %s via get_session.sh", spec.key)
    subprocess.run(args, check=True, env=env)


class SpikeInferencePipeline:
    """Coordinate session staging, backend inference, and result summarisation."""

    def __init__(
        self,
        project_root: Path,
        session_config_list: Mapping[str, object],
        backends: Sequence[SpikeInferenceBackend],
        get_session_script: Path,
        source_root: Path,
        dest_endpoint: Optional[str],
        output_root: Path,
        loader_module: str,
        skip_transfer: bool = False,
        max_neurons: Optional[int] = None,
        logger: Optional[logging.Logger] = None,
        target_sessions: Optional[Sequence[str]] = None,
    ):
        self.project_root = project_root
        self.backends = list(backends)
        self.get_session_script = get_session_script
        self.source_root = source_root
        self.dest_endpoint = dest_endpoint
        self.skip_transfer = skip_transfer
        self.output_root = output_root
        self.max_neurons = max_neurons
        self.logger = logger or LOGGER
        self.subject_name = session_config_list.get("subject_name", "sessions")
        self.session_specs = _flatten_session_config(session_config_list, project_root)
        if target_sessions:
            targets = {name.lower() for name in target_sessions}
            self.session_specs = [
                spec
                for spec in self.session_specs
                if spec.session_name.lower() in targets or spec.key.lower() in targets
            ]
        self.loader_module_path = loader_module
        self._loader_module = None

    def _get_loader_module(self):
        if self._loader_module is None:
            self._loader_module = importlib.import_module(self.loader_module_path)
        return self._loader_module

    def _load_dff_and_fs(self, session_path: Path) -> Tuple[np.ndarray, float]:
        loader = self._get_loader_module()
        try:
            read_ops = getattr(loader, "read_ops")
            read_dff = getattr(loader, "read_dff")
        except AttributeError as exc:
            raise AttributeError(
                f"Loader module '{self.loader_module_path}' must expose read_ops and read_dff."
            ) from exc
        ops_list = read_ops([str(session_path)])
        if not ops_list:
            raise RuntimeError(f"No ops.npy metadata found in {session_path}")
        ops = ops_list[0]
        fs = float(ops.get("fs") or ops.get("fr") or ops.get("frame_rate") or 30.0)
        dff = read_dff(ops)
        return np.asarray(dff, dtype=np.float32), fs

    def _save_result(self, path: Path, result: SpikeInferenceResult):
        path.parent.mkdir(parents=True, exist_ok=True)
        events_obj = np.array(
            [np.asarray(ev, dtype=np.float32) for ev in result.events], dtype=object
        )
        metadata_json = json.dumps(result.metadata, default=_json_default)
        np.savez_compressed(
            path,
            rate=result.rate.astype(np.float32),
            spikes=result.spikes.astype(np.float32),
            events=events_obj,
            metadata=np.array(metadata_json),
        )

    def run(self, limit: Optional[int] = None) -> Dict[str, object]:
        aggregate_summary: Dict[str, object] = {}
        processed = 0
        for spec in self.session_specs:
            if limit is not None and processed >= limit:
                break
            try:
                _ensure_session_data(
                    spec,
                    script_path=self.get_session_script,
                    source_root=self.source_root,
                    dest_endpoint=self.dest_endpoint,
                    skip_transfer=self.skip_transfer,
                )
            except FileNotFoundError as exc:
                self.logger.warning("Skipping %s: %s", spec.key, exc)
                continue
            dff, fs_hz = self._load_dff_and_fs(spec.local_path)
            if self.max_neurons and dff.shape[0] > self.max_neurons:
                dff = dff[: self.max_neurons]
            outputs: Dict[str, SpikeInferenceResult] = {}
            for backend in self.backends:
                try:
                    outputs[backend.name] = backend.predict(dff, fs_hz)
                except MissingDependencyError as exc:
                    self.logger.warning(
                        "Skipping backend %s for %s: %s", backend.name, spec.key, exc
                    )
                except Exception as exc:  # pragma: no cover - runtime dependency failures
                    self.logger.exception(
                        "Backend %s failed on %s: %s", backend.name, spec.key, exc
                    )
            if not outputs:
                self.logger.warning("No successful outputs for %s", spec.key)
                continue
            session_dir = self.output_root / spec.session_folder / spec.session_name
            for backend_name, result in outputs.items():
                self._save_result(session_dir / f"{backend_name}.npz", result)
            summary = summarise_outputs(outputs, fs_hz)
            with open(session_dir / "summary.json", "w", encoding="utf-8") as fh:
                json.dump(summary, fh, default=_json_default, indent=2)
            aggregate_summary[spec.key] = summary
            processed += 1
        summary_path = self.output_root / f"{self.subject_name}_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as fh:
            json.dump(aggregate_summary, fh, default=_json_default, indent=2)
        return aggregate_summary


__all__ = [
    "MissingDependencyError",
    "CascadeBackend",
    "ENS2Backend",
    "MLSpikeBackend",
    "SpikeInferencePipeline",
    "summarise_outputs",
]
