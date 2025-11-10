#!/usr/bin/env python3
"""
Build region-wide and subject-wide decoder summaries from existing session outputs.

Inputs
------
- Session-level decoder exports under ``~/Documents/<region>/<subject>/<session>/``.
- Each session directory should contain:
    * timecourse_metrics.csv
    * whole_metrics.csv
    * confusion_matrices.csv

Outputs
-------
- Summary figures in ``~/summary_image_change_decoders``:
    * ``{region}/{subset}/{target}_decoder_summary.pdf``
    * ``{subject}/{subset}/{target}_decoder_summary.pdf``
    * ``individual_session_decoders/{region}/{subject}/{subset}/{session}_{target}.pdf``
- Combined CSV exports in ``~/summary_image_decoders/data``.
"""

from __future__ import annotations

import argparse
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union
from itertools import cycle

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.axes_grid1.inset_locator import inset_axes  # noqa: E402
from tqdm import tqdm  # noqa: E402


# ---------- Data structures ----------


@dataclass(frozen=True)
class SessionRecord:
    region: str
    subject: str
    session: str
    path: Path
    has_timecourse: bool
    has_metrics: bool
    has_confusion: bool


# ---------- Utilities ----------


def slugify(text: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("_", "-", ".") else "_" for ch in text)
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe.strip("_") or "unnamed"


SUBSET_LABELS = {
    "all": "All neurons",
    "exc": "Excitatory",
    "vip": "VIP inhibitory",
    "sst": "SST inhibitory",
}

BLOCK_LABELS = {
    0: "Short block",
    1: "Long block",
}


def calc_sem(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size <= 1:
        return 0.0
    return float(arr.std(ddof=1) / math.sqrt(arr.size))


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalize_cell_subset(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if "cell_subset" not in df.columns:
        df = df.copy()
        df["cell_subset"] = "all"
    else:
        df = df.copy()
        df["cell_subset"] = df["cell_subset"].astype(str)
    return df


def counts_from_record(
    record: Union[pd.Series, Dict[str, object], None],
) -> Dict[str, Union[int, float]]:
    if record is None:
        return {}
    if isinstance(record, pd.Series):
        data = record.to_dict()
    elif isinstance(record, dict):
        data = record
    else:
        return {}
    counts: Dict[str, Union[int, float]] = {}
    for key in [
        "n_trials",
        "n_trials_short",
        "n_trials_long",
        "n_neurons_total",
        "n_neurons_sampled",
        "n_sessions",
        "n_subjects",
        "n_mice",
    ]:
        if key in data and not pd.isna(data[key]):
            val = data[key]
            if isinstance(val, (np.integer, np.floating, int, float)):
                counts[key] = int(round(float(val)))
            else:
                counts[key] = val
    return counts


# ---------- Loading helpers ----------


def discover_sessions(
    documents_root: Path,
    region_names: Iterable[str],
    subject_filter: Optional[Iterable[str]] = None,
) -> List[SessionRecord]:
    sessions: List[SessionRecord] = []
    region_names = list(region_names)
    if not region_names:
        return sessions
    for region in tqdm(region_names, desc="Regions", unit="region"):
        region_dir = documents_root / region
        if not region_dir.exists():
            continue
        subject_dirs = [p for p in region_dir.iterdir() if p.is_dir()]
        subject_iter = subject_dirs
        if subject_filter:
            subject_iter = [p for p in subject_dirs if p.name in subject_filter]
        for subject_dir in tqdm(
            sorted(subject_iter), desc=f"Subjects ({region})", unit="subject"
        ):
            session_dirs = [p for p in subject_dir.iterdir() if p.is_dir()]
            for session_dir in tqdm(
                sorted(session_dirs),
                desc=f"Sessions ({subject_dir.name})",
                unit="session",
                leave=False,
            ):
                timecourse_path = session_dir / "timecourse_metrics.csv"
                metrics_path = session_dir / "whole_metrics.csv"
                confusion_path = session_dir / "confusion_matrices.csv"
                sessions.append(
                    SessionRecord(
                        region=region,
                        subject=subject_dir.name,
                        session=session_dir.name,
                        path=session_dir,
                        has_timecourse=timecourse_path.exists(),
                        has_metrics=metrics_path.exists(),
                        has_confusion=confusion_path.exists(),
                    )
                )
    return sessions


def load_csvs(
    sessions: List[SessionRecord],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    timecourse_frames: List[pd.DataFrame] = []
    metrics_frames: List[pd.DataFrame] = []
    confusion_frames: List[pd.DataFrame] = []

    for record in sessions:
        if record.has_timecourse:
            df = pd.read_csv(record.path / "timecourse_metrics.csv")
            timecourse_frames.append(df)
        if record.has_metrics:
            df = pd.read_csv(record.path / "whole_metrics.csv")
            metrics_frames.append(df)
        if record.has_confusion:
            df = pd.read_csv(record.path / "confusion_matrices.csv")
            confusion_frames.append(df)

    timecourse_df = (
        pd.concat(timecourse_frames, ignore_index=True)
        if timecourse_frames
        else pd.DataFrame()
    )
    metrics_df = (
        pd.concat(metrics_frames, ignore_index=True)
        if metrics_frames
        else pd.DataFrame()
    )
    confusion_df = (
        pd.concat(confusion_frames, ignore_index=True)
        if confusion_frames
        else pd.DataFrame()
    )
    return timecourse_df, metrics_df, confusion_df


def write_combined_csvs(
    output_dir: Path,
    timecourse_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    confusion_df: pd.DataFrame,
) -> None:
    ensure_directory(output_dir)
    if not timecourse_df.empty:
        timecourse_df.to_csv(output_dir / "timecourse_metrics.csv", index=False)
    if not metrics_df.empty:
        metrics_df.to_csv(output_dir / "whole_metrics.csv", index=False)
    if not confusion_df.empty:
        confusion_df.to_csv(output_dir / "confusion_matrices.csv", index=False)


# ---------- Aggregation helpers ----------


def prepare_session_timecourses(timecourse_df: pd.DataFrame) -> pd.DataFrame:
    if timecourse_df.empty:
        return pd.DataFrame()
    cols = [
        "region",
        "subject",
        "session",
        "cell_subset",
        "target",
        "block_type",
        "time_center_ms",
    ]
    if "cell_subset" not in timecourse_df.columns:
        timecourse_df = timecourse_df.assign(cell_subset="all")
    aggregations = {
        "acc_mean": "mean",
        "chance_mean": "mean",
    }
    session_tc = (
        timecourse_df.groupby(cols, as_index=False)
        .agg(aggregations)
        .rename(
            columns={
                "acc_mean": "session_acc_mean",
                "chance_mean": "session_chance_mean",
            }
        )
    )
    return session_tc


def aggregate_timecourse_by_level(
    session_tc: pd.DataFrame, level_field: str
) -> pd.DataFrame:
    if session_tc.empty:
        return pd.DataFrame()
    records: List[Dict[str, float]] = []
    group_fields = [level_field, "cell_subset", "target", "time_center_ms"]
    for (level_value, subset, target, time_bin), sub in session_tc.groupby(
        group_fields
    ):
        values = sub["session_acc_mean"].to_numpy(dtype=float)
        chance = sub["session_chance_mean"].to_numpy(dtype=float)
        records.append(
            {
                "level_value": level_value,
                "cell_subset": subset,
                "target": target,
                "time_center_ms": time_bin,
                "mean_acc": float(np.mean(values)),
                "sem_acc": calc_sem(values),
                "mean_chance": float(np.mean(chance)),
                "sem_chance": calc_sem(chance),
                "n_sessions": int(sub["session"].nunique()),
            }
        )
    return pd.DataFrame(records)


def prepare_session_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    if metrics_df.empty:
        return pd.DataFrame()
    if "cell_subset" not in metrics_df.columns:
        metrics_df = metrics_df.assign(cell_subset="all")
    metric_cols = [
        "mean_acc",
        "precision_mean",
        "recall_mean",
        "f1_mean",
        "mean_chance",
    ]
    session_metrics = (
        metrics_df.groupby(
            ["region", "subject", "session", "cell_subset", "target"], as_index=False
        )[metric_cols]
        .mean()
        .rename(columns={col: f"session_{col}" for col in metric_cols})
    )
    return session_metrics


def aggregate_metrics_by_level(
    session_metrics: pd.DataFrame, level_field: str
) -> pd.DataFrame:
    if session_metrics.empty:
        return pd.DataFrame()
    metric_cols = [col for col in session_metrics.columns if col.startswith("session_")]
    records: List[Dict[str, float]] = []
    for (level_value, subset, target), sub in session_metrics.groupby(
        [level_field, "cell_subset", "target"]
    ):
        record: Dict[str, float] = {
            "level_value": level_value,
            "cell_subset": subset,
            "target": target,
            "n_sessions": int(sub["session"].nunique()),
        }
        for col in metric_cols:
            metric_name = col.replace("session_", "")
            values = sub[col].to_numpy(dtype=float)
            record[f"{metric_name}_mean"] = float(np.mean(values))
            record[f"{metric_name}_sem"] = calc_sem(values)
        records.append(record)
    return pd.DataFrame(records)


def aggregate_confusion_by_level(
    confusion_df: pd.DataFrame, level_field: str
) -> pd.DataFrame:
    if confusion_df.empty:
        return pd.DataFrame()
    if "cell_subset" not in confusion_df.columns:
        confusion_df = confusion_df.assign(cell_subset="all")
    return (
        confusion_df.groupby(
            [level_field, "cell_subset", "target", "actual", "predicted"],
            as_index=False,
        )["value"]
        .sum()
        .rename(columns={level_field: "level_value"})
    )


# ---------- Plotting helpers ----------


def _determine_ylim(summary_df: pd.DataFrame) -> Tuple[float, float]:
    if summary_df.empty:
        return (0.0, 1.0)
    candidates = []
    for mean_col, sem_col in [("mean_acc", "sem_acc"), ("mean_chance", "sem_chance")]:
        if mean_col in summary_df:
            mean_vals = summary_df[mean_col].to_numpy(dtype=float)
            sem_vals = (
                summary_df[sem_col].to_numpy(dtype=float)
                if sem_col in summary_df
                else np.zeros_like(mean_vals)
            )
            candidates.append(mean_vals - sem_vals)
            candidates.append(mean_vals + sem_vals)
    if not candidates:
        return (0.0, 1.0)
    concat = np.concatenate(candidates)
    concat = concat[np.isfinite(concat)]
    if concat.size == 0:
        return (0.0, 1.0)
    y_min = float(np.min(concat))
    y_max = float(np.max(concat))
    if np.isclose(y_min, y_max):
        margin = max(0.01, abs(y_min) * 0.1)
    else:
        margin = max(0.01, (y_max - y_min) * 0.1)
    y_min = max(0.0, y_min - margin)
    y_max = min(1.0, y_max + margin)
    if y_max - y_min < 0.05:
        center = 0.5 * (y_min + y_max)
        y_min = max(0.0, center - 0.025)
        y_max = min(1.0, center + 0.025)
    return y_min, y_max


def _format_counts_text(
    counts: Optional[Dict[str, Union[int, float]]],
) -> Optional[str]:
    if not counts:
        return None
    parts: List[str] = []
    trials = counts.get("n_trials")
    if trials is not None:
        short = counts.get("n_trials_short")
        long = counts.get("n_trials_long")
        if short is not None and long is not None:
            parts.append(
                f"Trials: {int(trials)} (Short {int(short)}, Long {int(long)})"
            )
        else:
            parts.append(f"Trials: {int(trials)}")
    neurons = counts.get("n_neurons_sampled")
    if neurons is not None:
        total_neurons = counts.get("n_neurons_total")
        if total_neurons is not None and neurons != total_neurons:
            parts.append(f"Neurons: {int(neurons)} / {int(total_neurons)}")
        else:
            parts.append(f"Neurons: {int(neurons)}")
    sessions = counts.get("n_sessions")
    mice = counts.get("n_mice")
    if sessions is not None or mice is not None:
        info = ", ".join(
            item
            for item in [
                f"{int(sessions)} sessions" if sessions is not None else None,
                f"{int(mice)} mice" if mice is not None else None,
            ]
            if item
        )
        if info:
            parts.append(info)
    return "\n".join(parts) if parts else None


def plot_timecourse(
    ax: plt.Axes,
    timecourse: pd.DataFrame,
    title: str,
    counts: Optional[Dict[str, Union[int, float]]] = None,
) -> None:
    if "block_type" in timecourse.columns:
        tc = timecourse.copy().sort_values(["block_type", "time_center_ms"])
        has_blocks = tc["block_type"].notna().any()
    else:
        tc = timecourse.copy().sort_values("time_center_ms")
        tc["block_type"] = np.nan
        has_blocks = False
    color_cycle = cycle(
        plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2"])
    )
    ylim_source = tc[tc["block_type"].notna()] if has_blocks else tc
    y_min, y_max = _determine_ylim(ylim_source)
    plotted_blocks = set()
    for block, sub in tc.groupby("block_type"):
        label_block = (
            BLOCK_LABELS.get(int(block), f"Block {block}")
            if not np.isnan(block)
            else "Averaged"
        )
        color = next(color_cycle)
        sub = sub.sort_values("time_center_ms")
        times = sub["time_center_ms"].to_numpy()
        mean_acc = sub["mean_acc"].to_numpy() * 100.0
        sem_acc = sub["sem_acc"].to_numpy() * 100.0
        chance = sub["mean_chance"].to_numpy() * 100.0

        ax.plot(times, mean_acc, label=label_block, color=color)
        ax.fill_between(
            times,
            mean_acc - sem_acc,
            mean_acc + sem_acc,
            color=color,
            alpha=0.25,
            linewidth=0,
        )

        if not np.isnan(chance).all():
            if np.allclose(chance, chance[0], equal_nan=True):
                ax.axhline(
                    chance[0], color=color, linestyle="--", linewidth=1, alpha=0.6
                )
            else:
                ax.plot(
                    times, chance, color=color, linestyle="--", linewidth=1, alpha=0.6
                )
        plotted_blocks.add(block)

    if not tc.empty:
        ax.set_xlim(tc["time_center_ms"].min(), tc["time_center_ms"].max())
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title, fontsize=12)
    ax.set_ylim(y_min * 100.0, y_max * 100.0)
    ax.axhline(50.0, color="grey", linestyle=":", linewidth=1)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="lower right", frameon=False, fontsize=9)
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)

    counts_text = _format_counts_text(counts)
    if counts_text:
        ax.text(
            0.02,
            0.98,
            counts_text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
        )


def plot_metrics_box(ax: plt.Axes, metrics: Optional[pd.Series]) -> None:
    if metrics is None:
        return
    lines: List[str] = []

    def fmt(name: str, label: str) -> None:
        mean_key = f"{name}_mean"
        sem_key = f"{name}_sem"
        if mean_key in metrics and not math.isnan(metrics[mean_key]):
            sem_val = metrics.get(sem_key, 0.0)
            lines.append(
                f"{label}: {metrics[mean_key] * 100:.1f}% ± {sem_val * 100:.1f}%"
            )

    fmt("mean_acc", "Accuracy")
    fmt("precision_mean", "Precision")
    fmt("recall_mean", "Recall")
    fmt("f1_mean", "F1 score")
    if "mean_chance_mean" in metrics and not math.isnan(metrics["mean_chance_mean"]):
        lines.append(f"Chance: {metrics['mean_chance_mean'] * 100:.1f}%")

    if lines:
        text = "\n".join(lines)
        ax.text(
            0.98,
            0.02,
            text,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
        )


def plot_confusion(ax: plt.Axes, confusion: pd.DataFrame, title: str) -> None:
    required_cols = {"actual", "predicted", "value"}
    if not required_cols.issubset(confusion.columns):
        ax.text(
            0.5,
            0.5,
            "Missing confusion data",
            ha="center",
            va="center",
            fontsize=12,
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        return

    reduced = (
        confusion.groupby(["actual", "predicted"], as_index=False)["value"].sum().copy()
    )

    pivot = reduced.pivot(index="actual", columns="predicted", values="value").fillna(
        0.0
    )
    actual_labels = pivot.index.tolist()
    predicted_labels = pivot.columns.tolist()
    counts = pivot.to_numpy(dtype=float)
    row_sums = counts.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        normalized = np.divide(
            counts, row_sums, out=np.zeros_like(counts), where=row_sums != 0
        )

    im = ax.imshow(normalized, cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(predicted_labels)))
    ax.set_yticks(np.arange(len(actual_labels)))
    ax.set_xticklabels(predicted_labels, rotation=45, ha="right")
    ax.set_yticklabels(actual_labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title, fontsize=12)

    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            value = counts[i, j]
            pct = normalized[i, j] * 100.0
            text_color = "white" if normalized[i, j] > 0.5 else "black"
            ax.text(
                j,
                i,
                f"{value:.0f}\n({pct:.1f}%)",
                ha="center",
                va="center",
                fontsize=8,
                color=text_color,
            )

    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row-normalized accuracy", fontsize=9)


def make_summary_figure(
    output_path: Path,
    timecourse: pd.DataFrame,
    metrics: Optional[pd.Series],
    confusion: Optional[pd.DataFrame],
    title: str,
    counts: Optional[Dict[str, Union[int, float]]] = None,
    neuron_counts: Optional[Sequence[float]] = None,
) -> None:
    has_confusion = confusion is not None and not confusion.empty
    if has_confusion:
        fig, (ax_tc, ax_cm) = plt.subplots(
            1,
            2,
            figsize=(12, 5),
            gridspec_kw={"width_ratios": [2.0, 1.2]},
            constrained_layout=True,
        )
    else:
        fig, ax_tc = plt.subplots(1, 1, figsize=(8, 5), constrained_layout=True)
        ax_cm = None

    plot_timecourse(ax_tc, timecourse, title="Decoder timecourse", counts=counts)
    if neuron_counts is not None and len(neuron_counts) > 0:
        inset = inset_axes(ax_tc, width="25%", height="35%", loc="upper right")
        parts = inset.violinplot(neuron_counts, vert=True, showmedians=True)
        for partname in ("cbars", "cmins", "cmaxes"):
            part = parts.get(partname)
            if part is not None:
                part.set_edgecolor("black")
        for pc in parts["bodies"]:
            pc.set_facecolor("lightgray")
            pc.set_edgecolor("black")
            pc.set_alpha(0.7)
        inset.set_xticks([])
        inset.set_ylabel("Neurons", fontsize=8)
        inset.tick_params(axis="y", labelsize=7)
        inset.set_title("n", fontsize=8)
    plot_metrics_box(ax_tc, metrics)

    if has_confusion and ax_cm is not None:
        plot_confusion(ax_cm, confusion, title="Combined confusion matrix")

    fig.suptitle(title, fontsize=14)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


# ---------- Figure orchestration ----------


def build_summary_figures(
    summary_root: Path,
    timecourse_level_df: pd.DataFrame,
    metrics_level_df: pd.DataFrame,
    confusion_level_df: pd.DataFrame,
    level_label: str,
    raw_metrics_df: pd.DataFrame,
) -> None:
    if timecourse_level_df.empty:
        return

    for (level_value, subset, target), tc_group in timecourse_level_df.groupby(
        ["level_value", "cell_subset", "target"]
    ):
        subset_slug = slugify(subset)
        subset_label = SUBSET_LABELS.get(subset, subset)
        tc = tc_group.sort_values("time_center_ms")[
            ["time_center_ms", "mean_acc", "sem_acc", "mean_chance", "sem_chance"]
        ]

        metrics_row: Optional[pd.Series] = None
        if not metrics_level_df.empty:
            match = metrics_level_df[
                (metrics_level_df["level_value"] == level_value)
                & (metrics_level_df["cell_subset"] == subset)
                & (metrics_level_df["target"] == target)
            ]
            if not match.empty:
                metrics_row = match.iloc[0]

        counts = counts_from_record(metrics_row) if metrics_row is not None else {}
        neuron_counts: Optional[np.ndarray] = None
        if not raw_metrics_df.empty and "n_neurons_sampled" in raw_metrics_df.columns:
            level_mask = raw_metrics_df[level_label].astype(str) == str(level_value)
            subset_mask = raw_metrics_df["cell_subset"].astype(str) == subset
            target_mask = raw_metrics_df["target"].astype(str) == target
            subset_counts = raw_metrics_df[level_mask & subset_mask & target_mask][
                "n_neurons_sampled"
            ].dropna()
            if not subset_counts.empty:
                neuron_counts = subset_counts.to_numpy(dtype=float)

        confusion: Optional[pd.DataFrame] = None
        if not confusion_level_df.empty:
            conf = confusion_level_df[
                (confusion_level_df["level_value"] == level_value)
                & (confusion_level_df["cell_subset"] == subset)
                & (confusion_level_df["target"] == target)
            ]
            if not conf.empty:
                confusion = conf

        safe_target = slugify(target)
        level_slug = slugify(str(level_value))
        base_dir = ensure_directory(summary_root / level_slug / subset_slug)
        filename = f"{safe_target}_decoder_summary.pdf"
        title = f"{level_value} — {subset_label} — {target.replace('_', ' ')}"
        make_summary_figure(
            base_dir / filename,
            tc,
            metrics_row,
            confusion,
            title,
            counts,
            neuron_counts,
        )


def build_session_figures(
    summary_root: Path,
    session_tc: pd.DataFrame,
    session_metrics: pd.DataFrame,
    confusion_df: pd.DataFrame,
    raw_metrics_df: pd.DataFrame,
) -> None:
    if session_tc.empty:
        return
    base_dir = ensure_directory(summary_root / "individual_session_decoders")

    for (region, subject, session, subset, target), tc_group in session_tc.groupby(
        ["region", "subject", "session", "cell_subset", "target"]
    ):
        tc = (
            tc_group.sort_values("time_center_ms")
            .rename(
                columns={
                    "session_acc_mean": "mean_acc",
                    "session_chance_mean": "mean_chance",
                }
            )
            .assign(sem_acc=0.0, sem_chance=0.0)[
                ["time_center_ms", "mean_acc", "sem_acc", "mean_chance", "sem_chance"]
            ]
        )
        metrics_row: Optional[pd.Series] = None
        if not session_metrics.empty:
            match = session_metrics[
                (session_metrics["region"] == region)
                & (session_metrics["subject"] == subject)
                & (session_metrics["session"] == session)
                & (session_metrics["cell_subset"] == subset)
                & (session_metrics["target"] == target)
            ]
            if not match.empty:
                data = {}
                row = match.iloc[0]
                for col in match.columns:
                    if col.startswith("session_"):
                        metric_name = col.replace("session_", "")
                        data[f"{metric_name}_mean"] = float(row[col])
                        data[f"{metric_name}_sem"] = 0.0
                metrics_row = pd.Series(data)
        confusion: Optional[pd.DataFrame] = None
        if not confusion_df.empty:
            conf = confusion_df[
                (confusion_df["region"] == region)
                & (confusion_df["subject"] == subject)
                & (confusion_df["session"] == session)
                & (confusion_df["cell_subset"] == subset)
                & (confusion_df["target"] == target)
            ]
            if not conf.empty:
                confusion = conf

        safe_target = slugify(target)
        subset_slug = slugify(subset)
        subset_label = SUBSET_LABELS.get(subset, subset)
        output_dir = ensure_directory(base_dir / region / subject / subset_slug)
        filename = f"{session}_{safe_target}.pdf"
        output_path = output_dir / filename
        title = f"{region} — {subject} — {session} — {subset_label} ({target.replace('_', ' ')})"
        counts = (
            counts_from_record(metrics_row)
            if isinstance(metrics_row, pd.Series)
            else {}
        )
        neuron_counts: Optional[np.ndarray] = None
        if not raw_metrics_df.empty and "n_neurons_sampled" in raw_metrics_df.columns:
            mask = (
                (raw_metrics_df["region"] == region)
                & (raw_metrics_df["subject"] == subject)
                & (raw_metrics_df["session"] == session)
                & (raw_metrics_df["cell_subset"] == subset)
                & (raw_metrics_df["target"] == target)
            )
            subset_counts = raw_metrics_df[mask]["n_neurons_sampled"].dropna()
            if not subset_counts.empty:
                neuron_counts = subset_counts.to_numpy(dtype=float)

        make_summary_figure(
            output_path, tc, metrics_row, confusion, title, counts, neuron_counts
        )


def organize_summary_outputs(
    summary_root: Path, regions: Iterable[str], session_timecourses: pd.DataFrame
) -> None:
    summary_root = ensure_directory(summary_root)
    regions = list(regions)
    top_level_summaries = list(summary_root.glob("*_decoder_summary.*"))
    if not top_level_summaries:
        return
    subject_region_pairs = (
        session_timecourses[["subject", "region"]].drop_duplicates()
        if not session_timecourses.empty
        else pd.DataFrame(columns=["subject", "region"])
    )
    subject_to_regions: Dict[str, List[str]] = {}
    for _, row in subject_region_pairs.iterrows():
        subject_to_regions.setdefault(row["subject"], []).append(row["region"])

    for region in regions:
        ensure_directory(summary_root / region)
        for path in list(
            summary_root.glob(f"{region}_all_mice_average_*_decoder_summary.png")
        ):
            destination = summary_root / region / path.name
            if destination.exists():
                destination.unlink()
            shutil.move(str(path), str(destination))

    subject_files = [
        p
        for p in summary_root.glob("*_decoder_summary.png")
        if not any(
            p.name.startswith(f"{region}_all_mice_average") for region in regions
        )
    ]

    for path in subject_files:
        subject = path.name.split("_", 1)[0]
        target_regions = subject_to_regions.get(subject, [])
        if not target_regions:
            continue
        first_region = target_regions[0]
        dest_first = summary_root / first_region / path.name
        if dest_first.exists():
            dest_first.unlink()
        shutil.move(str(path), str(dest_first))
        for region in target_regions[1:]:
            dest = summary_root / region / path.name
            if dest.exists():
                dest.unlink()
            shutil.copy2(dest_first, dest)


# ---------- Main ----------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarise decoder outputs across sessions."
    )
    parser.add_argument(
        "--documents-root",
        type=Path,
        default=Path.home() / "Documents",
        help="Root directory that contains region folders (default: ~/Documents).",
    )
    parser.add_argument(
        "--regions",
        nargs="+",
        default=["PPC", "V1"],
        help="Region folder names to include (default: PPC V1).",
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        help="Optional list of subject IDs to include.",
    )
    parser.add_argument(
        "--summary-root",
        type=Path,
        default=Path.home() / "summary_image_change_decoders",
        help="Output directory for summary figures (default: ~/summary_image_change_decoders).",
    )
    parser.add_argument(
        "--data-output",
        type=Path,
        default=None,
        help="Directory to write combined CSV exports (default: <summary_root>/data).",
    )
    parser.add_argument(
        "--cell-subsets",
        nargs="+",
        default=["all", "exc", "vip", "sst"],
        help="Neuron subsets to include in the summaries (default: all exc vip sst).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_output = args.data_output or (args.summary_root / "data")
    subject_filter_set = (
        {s.strip() for s in args.subjects if s.strip()} if args.subjects else None
    )
    subset_filter = (
        [s.strip().lower() for s in args.cell_subsets] if args.cell_subsets else None
    )

    sessions = discover_sessions(args.documents_root, args.regions, subject_filter_set)
    if not sessions:
        print("No session directories found; nothing to summarise.")
        return

    timecourse_df, metrics_df, confusion_df = load_csvs(sessions)
    timecourse_df = normalize_cell_subset(timecourse_df)
    metrics_df = normalize_cell_subset(metrics_df)
    confusion_df = normalize_cell_subset(confusion_df)

    if subject_filter_set:
        timecourse_df = timecourse_df[timecourse_df["subject"].isin(subject_filter_set)]
        metrics_df = metrics_df[metrics_df["subject"].isin(subject_filter_set)]
        confusion_df = confusion_df[confusion_df["subject"].isin(subject_filter_set)]

    if subset_filter:
        subset_mask_tc = timecourse_df["cell_subset"].str.lower().isin(subset_filter)
        subset_mask_metrics = metrics_df["cell_subset"].str.lower().isin(subset_filter)
        subset_mask_conf = confusion_df["cell_subset"].str.lower().isin(subset_filter)
        timecourse_df = timecourse_df[subset_mask_tc]
        metrics_df = metrics_df[subset_mask_metrics]
        confusion_df = confusion_df[subset_mask_conf]

    if timecourse_df.empty:
        print("No timecourse metrics available; cannot build summaries.")
        return

    write_combined_csvs(data_output, timecourse_df, metrics_df, confusion_df)

    session_timecourses = prepare_session_timecourses(timecourse_df)
    session_metrics = prepare_session_metrics(metrics_df)

    region_timecourses = aggregate_timecourse_by_level(session_timecourses, "region")
    subject_timecourses = aggregate_timecourse_by_level(session_timecourses, "subject")

    region_metrics = aggregate_metrics_by_level(session_metrics, "region")
    subject_metrics = aggregate_metrics_by_level(session_metrics, "subject")

    region_confusion = aggregate_confusion_by_level(confusion_df, "region")
    subject_confusion = aggregate_confusion_by_level(confusion_df, "subject")

    build_summary_figures(
        args.summary_root,
        region_timecourses,
        region_metrics,
        region_confusion,
        "region",
        metrics_df,
    )
    build_summary_figures(
        args.summary_root,
        subject_timecourses,
        subject_metrics,
        subject_confusion,
        "subject",
        metrics_df,
    )
    build_session_figures(
        args.summary_root,
        session_timecourses,
        session_metrics,
        confusion_df,
        metrics_df,
    )
    organize_summary_outputs(args.summary_root, args.regions, session_timecourses)

    print(f"Summary figures written to: {args.summary_root}")
    print(f"Combined CSV exports written to: {data_output}")


if __name__ == "__main__":
    main()
