from __future__ import annotations

import time
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

TIMECOURSE_FILE = "timecourse_metrics.csv"
WHOLE_FILE = "whole_metrics.csv"
CONFUSION_FILE = "confusion_matrices.csv"

BLOCK_LABELS: Dict[int, str] = {
    0: "Short block",
    1: "Long block",
}

CELL_SUBSET_ORDER = ["all", "exc", "vip", "sst"]
CELL_SUBSET_LABELS: Dict[str, str] = {
    "all": "All neurons",
    "exc": "Excitatory (putative)",
    "vip": "VIP interneurons",
    "sst": "SST interneurons",
}


@dataclass(frozen=True)
class DecoderDataset:
    """Container for loaded decoder exports."""

    timecourse: pd.DataFrame
    metrics: pd.DataFrame
    confusion: pd.DataFrame


def _iter_session_dirs(root: Path) -> Iterable[Tuple[str, str, str, Path]]:
    if not root.exists():
        return
    for region_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for subject_dir in sorted(p for p in region_dir.iterdir() if p.is_dir()):
            for session_dir in sorted(p for p in subject_dir.iterdir() if p.is_dir()):
                yield region_dir.name, subject_dir.name, session_dir.name, session_dir


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - surfaced in UI feedback
        raise RuntimeError(f"Failed to read CSV at {path}") from exc


def normalise_subset_value(value: str) -> str:
    if value is None:
        return "all"
    return str(value).strip().lower()


def normalise_subset_label(value: str) -> str:
    key = normalise_subset_value(value)
    return CELL_SUBSET_LABELS.get(key, value)


def _ensure_metadata(df: pd.DataFrame, **metadata: str) -> pd.DataFrame:
    if df.empty:
        return df
    frame = df.copy()
    for key, value in metadata.items():
        if key in frame.columns:
            frame[key] = frame[key].fillna(value)
        else:
            frame[key] = value
    if "cell_subset" in frame.columns:
        frame["cell_subset"] = frame["cell_subset"].apply(normalise_subset_value)
    else:
        frame["cell_subset"] = "all"
    return frame


def calc_sem(values: Sequence[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return 0.0
    return float(np.std(arr, ddof=1) / math.sqrt(arr.size))


class DataRepository:
    """Load and cache decoder exports for the Dash viewer."""

    def __init__(self, root: Path):
        self.root = Path(root).expanduser()
        self._dataset: Optional[DecoderDataset] = None
        self._last_loaded: Optional[float] = None

    @property
    def last_loaded(self) -> Optional[float]:
        return self._last_loaded

    def reload(self) -> DecoderDataset:
        timecourse_frames: List[pd.DataFrame] = []
        metrics_frames: List[pd.DataFrame] = []
        confusion_frames: List[pd.DataFrame] = []

        for region, subject, session, session_dir in _iter_session_dirs(self.root):
            metadata = {"region": region, "subject": subject, "session": session}
            tc_path = session_dir / TIMECOURSE_FILE
            if tc_path.exists():
                frame = _ensure_metadata(_read_csv(tc_path), **metadata)
                if "target" in frame.columns:
                    frame["target"] = frame["target"].astype(str)
                timecourse_frames.append(frame)
            metrics_path = session_dir / WHOLE_FILE
            if metrics_path.exists():
                frame = _ensure_metadata(_read_csv(metrics_path), **metadata)
                if "target" in frame.columns:
                    frame["target"] = frame["target"].astype(str)
                metrics_frames.append(frame)
            confusion_path = session_dir / CONFUSION_FILE
            if confusion_path.exists():
                frame = _ensure_metadata(_read_csv(confusion_path), **metadata)
                if "target" in frame.columns:
                    frame["target"] = frame["target"].astype(str)
                confusion_frames.append(frame)

        timecourse_df = (
            pd.concat(timecourse_frames, ignore_index=True) if timecourse_frames else pd.DataFrame()
        )
        metrics_df = (
            pd.concat(metrics_frames, ignore_index=True) if metrics_frames else pd.DataFrame()
        )
        confusion_df = (
            pd.concat(confusion_frames, ignore_index=True) if confusion_frames else pd.DataFrame()
        )

        for frame in (timecourse_df, metrics_df, confusion_df):
            if not frame.empty and "cell_subset" in frame.columns:
                frame["cell_subset"] = frame["cell_subset"].apply(normalise_subset_value)

        self._dataset = DecoderDataset(timecourse=timecourse_df, metrics=metrics_df, confusion=confusion_df)
        self._last_loaded = time.time()
        return self._dataset

    def get_dataset(self) -> DecoderDataset:
        if self._dataset is None:
            return self.reload()
        return self._dataset

    def available_regions(self) -> List[str]:
        dataset = self.get_dataset()
        if dataset.timecourse.empty:
            return []
        return sorted(dataset.timecourse["region"].dropna().astype(str).unique())

    def available_targets(self) -> List[str]:
        dataset = self.get_dataset()
        frames = []
        if not dataset.timecourse.empty and "target" in dataset.timecourse.columns:
            frames.append(dataset.timecourse["target"])
        if not dataset.metrics.empty and "target" in dataset.metrics.columns:
            frames.append(dataset.metrics["target"])
        if not frames:
            return []
        combined = pd.concat(frames).dropna().astype(str)
        return sorted(combined.unique())

    def available_subsets(self) -> List[str]:
        dataset = self.get_dataset()
        frames = []
        if not dataset.timecourse.empty and "cell_subset" in dataset.timecourse.columns:
            frames.append(dataset.timecourse["cell_subset"])
        if not dataset.metrics.empty and "cell_subset" in dataset.metrics.columns:
            frames.append(dataset.metrics["cell_subset"])
        if not frames:
            return CELL_SUBSET_ORDER.copy()
        combined = pd.concat(frames).dropna().astype(str)
        candidates = sorted(set(combined.unique()) | set(CELL_SUBSET_ORDER))
        return [value for value in CELL_SUBSET_ORDER if value in candidates] + [
            value for value in candidates if value not in CELL_SUBSET_ORDER
        ]

    def subjects_for_regions(self, regions: Sequence[str]) -> List[str]:
        dataset = self.get_dataset()
        if dataset.timecourse.empty:
            return []
        frame = dataset.timecourse
        if regions:
            frame = frame[frame["region"].isin(regions)]
        return sorted(frame["subject"].dropna().astype(str).unique())

    def sessions_for_selection(self, regions: Sequence[str], subjects: Sequence[str]) -> List[str]:
        dataset = self.get_dataset()
        if dataset.timecourse.empty:
            return []
        frame = dataset.timecourse
        if regions:
            frame = frame[frame["region"].isin(regions)]
        if subjects:
            frame = frame[frame["subject"].isin(subjects)]
        return sorted(frame["session"].dropna().astype(str).unique())


def _apply_filters(
    df: pd.DataFrame,
    *,
    target: Optional[str],
    regions: Sequence[str],
    subjects: Sequence[str],
    sessions: Sequence[str],
    subsets: Sequence[str],
) -> pd.DataFrame:
    if df.empty:
        return df
    frame = df.copy()
    if target and "target" in frame.columns:
        frame = frame[frame["target"] == target]
    if regions and "region" in frame.columns:
        frame = frame[frame["region"].isin(regions)]
    if subjects and "subject" in frame.columns:
        frame = frame[frame["subject"].isin(subjects)]
    if sessions and "session" in frame.columns:
        frame = frame[frame["session"].isin(sessions)]
    if subsets and "cell_subset" in frame.columns:
        frame = frame[frame["cell_subset"].isin(subsets)]
    return frame


def aggregate_timecourse(
    timecourse: pd.DataFrame,
    *,
    group_field: str,
    target: Optional[str],
    regions: Sequence[str],
    subjects: Sequence[str],
    sessions: Sequence[str],
    subsets: Sequence[str],
    block_types: Sequence[int],
) -> pd.DataFrame:
    frame = _apply_filters(
        timecourse,
        target=target,
        regions=regions,
        subjects=subjects,
        sessions=sessions,
        subsets=subsets,
    )
    if frame.empty:
        return frame
    if group_field not in frame.columns:
        return pd.DataFrame()
    if block_types and "block_type" in frame.columns:
        frame = frame[frame["block_type"].isin(block_types)]
    if frame.empty:
        return frame

    session_group = [
        "region",
        "subject",
        "session",
        "cell_subset",
        "target",
        "block_type",
        "time_center_ms",
    ]
    if "cell_subset" not in frame.columns:
        frame = frame.assign(cell_subset="all")
    session_tc = (
        frame.groupby(session_group, as_index=False)
        .agg(
            session_acc_mean=("acc_mean", "mean"),
            session_chance_mean=("chance_mean", "mean"),
        )
    )
    if session_tc.empty:
        return session_tc

    if group_field == "session":
        session_tc = session_tc.rename(
            columns={
                "session_acc_mean": "acc_mean",
                "session_chance_mean": "chance_mean",
            }
        )
        session_tc["acc_sem"] = 0.0
        session_tc["chance_sem"] = 0.0
        session_tc["n_sessions"] = 1
        return session_tc[
            ["session", "block_type", "time_center_ms", "acc_mean", "acc_sem", "chance_mean", "chance_sem", "n_sessions"]
        ]

    records: List[Dict[str, float]] = []
    group_cols = [group_field, "block_type", "time_center_ms"]
    for (group_value, block_type, time_bin), sub in session_tc.groupby(group_cols):
        acc_values = sub["session_acc_mean"].to_numpy(dtype=float)
        chance_values = sub["session_chance_mean"].to_numpy(dtype=float)
        records.append(
            {
                group_field: group_value,
                "block_type": block_type,
                "time_center_ms": time_bin,
                "acc_mean": float(np.mean(acc_values)),
                "acc_sem": calc_sem(acc_values),
                "chance_mean": float(np.mean(chance_values)),
                "chance_sem": calc_sem(chance_values),
                "n_sessions": int(sub["session"].nunique()),
            }
        )
    return pd.DataFrame(records)


def aggregate_counts(
    metrics: pd.DataFrame,
    *,
    group_field: str,
    target: Optional[str],
    regions: Sequence[str],
    subjects: Sequence[str],
    sessions: Sequence[str],
    subsets: Sequence[str],
) -> pd.DataFrame:
    frame = _apply_filters(
        metrics,
        target=target,
        regions=regions,
        subjects=subjects,
        sessions=sessions,
        subsets=subsets,
    )
    if frame.empty:
        return frame
    if group_field not in frame.columns:
        return pd.DataFrame()
    aggregated = (
        frame.groupby(group_field)
        .agg(
            n_sessions=("session", "nunique"),
            n_subjects=("subject", "nunique"),
            n_trials=("n_trials", "sum"),
            n_trials_short=("n_trials_short", "sum"),
            n_trials_long=("n_trials_long", "sum"),
            n_neurons_sampled=("n_neurons_sampled", "sum"),
            n_neurons_total=("n_neurons_total", "sum"),
            n_neuron_types=("cell_subset", lambda col: col.astype(str).nunique()),
        )
        .reset_index()
    )
    return aggregated


def compute_selection_summary(
    metrics: pd.DataFrame,
    *,
    target: Optional[str],
    regions: Sequence[str],
    subjects: Sequence[str],
    sessions: Sequence[str],
    subsets: Sequence[str],
) -> Dict[str, int]:
    frame = _apply_filters(
        metrics,
        target=target,
        regions=regions,
        subjects=subjects,
        sessions=sessions,
        subsets=subsets,
    )
    if frame.empty:
        return {}

    summary: Dict[str, int] = {}
    summary["n_sessions"] = int(frame["session"].nunique()) if "session" in frame.columns else 0
    summary["n_subjects"] = int(frame["subject"].nunique()) if "subject" in frame.columns else 0
    if "cell_subset" in frame.columns:
        summary["n_neuron_types"] = int(frame["cell_subset"].astype(str).nunique())
    dedup_cols = [col for col in ["region", "subject", "session", "cell_subset"] if col in frame.columns]
    unique_rows = frame.drop_duplicates(subset=dedup_cols) if dedup_cols else frame
    if "n_neurons_total" in unique_rows.columns:
        summary["n_neurons_total"] = int(
            pd.to_numeric(unique_rows["n_neurons_total"], errors="coerce").fillna(0).sum()
        )
    return summary
