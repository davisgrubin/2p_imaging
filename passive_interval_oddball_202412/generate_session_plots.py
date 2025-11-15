#!/usr/bin/env python3
"""
Generate decoder plots for a single ImageChange session.

Outputs:
    - Confusion matrices and timecourse plots for change_vs_repeat (binary)
    - Confusion matrices and timecourse plots for orientation (multiclass)
    - Orientation-specific binary decoders (to vs repeat) with per-orientation
      confusion/timecourse plots and an averaged timecourse across orientations.

Figures are written to:
    {output_root}/{region}/{subject}/{session_name}/{target}/*.png
where `target` includes the orientation index for orientation-specific outputs.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from itertools import cycle  # noqa: E402

from modules.ReadResults import read_ops, read_neural_trials, read_masks  # noqa: E402
from modules.Alignment import run_get_stim_response  # noqa: E402
from modules import Trialization  # noqa: E402
from utils import get_neu_trial, get_frame_idx_from_time, pick_trial  # noqa: E402
from modeling.decoding import (  # noqa: E402
    binary_decoder_cv,
    binary_timecourse_decoding,
    multiclass_decoder_cv,
    multiclass_timecourse_decoding,
)

RANDOM_STATE = 0
BIN_WIDTH_MS = 100.0
TIME_WINDOW_MS = (-500.0, 2000.0)
SHUFFLE_BASELINE = True
TRIAL_PARAM = [[2, 3, 4, 5, -2, -3, -4, -5], [0, 1], None, None, [0], [0]]
BLOCK_LABELS = {0: "Short", 1: "Long"}
DEFAULT_OUTPUT_ROOT = Path("plot_exports")
DEGREES_REFERENCE = [0.0, 45.0, 90.0, 135.0]
CELL_SUBSETS: Dict[str, Optional[Sequence[int]]] = {
    "all": None,
    "exc": [-1],
    "vip": [0],
    "sst": [1],
}


@dataclass(frozen=True)
class PlotContext:
    region: str
    subject: str
    session: str
    cell_subset: str
    counts: Dict[str, Union[int, float]]
    output_format: str = "pdf"

    def target_key(self, target: str) -> str:
        subset = slugify(self.cell_subset)
        return f"{target}__{subset}" if subset != "all" else target


def slugify(text: str) -> str:
    import re

    safe = re.sub(r"\s+", "_", str(text))
    safe = re.sub(r"[^0-9A-Za-z_.-]+", "", safe)
    safe = safe.strip("_")
    return safe or "unnamed"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_figure(
    fig: matplotlib.figure.Figure,
    output_root: Path,
    region: str,
    subject: str,
    session: str,
    target: str,
    filename: str,
    output_format: str = "pdf",
) -> Path:
    directory = ensure_dir(
        output_root
        / slugify(region)
        / slugify(subject)
        / slugify(session)
        / slugify(target)
    )
    path = directory / f"{slugify(filename)}.{output_format}"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def append_confusion_rows(
    rows: List[Dict],
    cm: np.ndarray,
    row_labels: List[str],
    col_labels: List[str],
    base_info: Dict,
) -> None:
    cm_arr = np.asarray(cm, dtype=float)
    for i, actual in enumerate(row_labels):
        for j, predicted in enumerate(col_labels):
            entry = {
                **base_info,
                "actual": actual,
                "predicted": predicted,
                "value": float(cm_arr[i, j]),
            }
            rows.append(entry)


def _json_default(obj):
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def minimal_theta_diff(a: float, b: float) -> float:
    if np.isnan(a) or np.isnan(b):
        return np.nan
    diff = abs(a - b) % 180.0
    return diff if diff <= 90.0 else 180.0 - diff


def compute_trial_metadata(stim_labels: np.ndarray) -> pd.DataFrame:
    stim_labels = np.asarray(stim_labels)
    codes = stim_labels[:, 2].astype(int)
    base_codes = sorted({abs(code) for code in codes if abs(code) >= 2})
    if not base_codes:
        return pd.DataFrame(
            columns=[
                "trial_index",
                "label_code",
                "orientation_idx",
                "from_orientation_idx",
                "orientation_deg",
                "from_orientation_deg",
                "theta_change_deg",
                "is_transition",
                "block_type",
                "stim_start_ms",
                "stim_end_ms",
                "pair_label",
            ]
        )
    orientation_map = {code: idx for idx, code in enumerate(base_codes, start=1)}
    if len(base_codes) == len(DEGREES_REFERENCE):
        orientation_deg_lookup = {
            orientation_map[code]: DEGREES_REFERENCE[i]
            for i, code in enumerate(sorted(base_codes))
        }
    else:
        step = 180.0 / max(len(base_codes), 1)
        orientation_deg_lookup = {
            orientation_map[code]: i * step for i, code in enumerate(sorted(base_codes))
        }

    rows = []
    last_ori_idx = None
    last_ori_key = None
    repeat_count = 0
    for trial_idx, code in enumerate(codes):
        block_type = int(stim_labels[trial_idx, 3])
        if code == -1:
            if last_ori_idx is None:
                continue
            ori_idx = last_ori_idx
            ori_key = last_ori_key
            from_idx = last_ori_idx
            from_key = last_ori_key
            is_transition = False
        else:
            ori_key = abs(code)
            if ori_key not in orientation_map:
                continue
            ori_idx = orientation_map[ori_key]
            from_idx = last_ori_idx if last_ori_idx is not None else ori_idx
            from_key = last_ori_key if last_ori_key is not None else ori_key
            is_transition = bool(code < 0)
        ori_deg = float(orientation_deg_lookup.get(ori_idx, np.nan))
        from_deg = float(orientation_deg_lookup.get(from_idx, ori_deg))
        theta_delta = minimal_theta_diff(ori_deg, from_deg)
        if is_transition:
            repeat_count = 0
            pair_label = f"ori{int(from_idx)}->ori{int(ori_idx)}"
        else:
            repeat_count += 1
            pair_label = f"repeat_ori{int(ori_idx)}"
        rows.append(
            {
                "trial_index": int(trial_idx),
                "label_code": int(code),
                "orientation_idx": int(ori_idx),
                "from_orientation_idx": int(from_idx),
                "orientation_deg": ori_deg,
                "from_orientation_deg": from_deg,
                "theta_change_deg": theta_delta,
                "is_transition": is_transition,
                "block_type": block_type,
                "stim_start_ms": float(stim_labels[trial_idx, 0]),
                "stim_end_ms": float(stim_labels[trial_idx, 1]),
                "pair_label": pair_label,
                "repeat_count": repeat_count,
            }
        )
        last_ori_idx = ori_idx
        last_ori_key = ori_key
    return pd.DataFrame(rows)


def prepare_session_dataset(
    neu_session: np.ndarray,
    stim_labels: np.ndarray,
    post_isi: np.ndarray,
    neu_time: np.ndarray,
    selected_indices: Optional[np.ndarray] = None,
) -> Optional[Dict]:
    trial_meta = compute_trial_metadata(stim_labels)
    if trial_meta.empty:
        return None
    if selected_indices is None:
        selected_indices = np.arange(neu_session.shape[0])
    selected_indices = np.asarray(selected_indices)
    idx_map = {orig: pos for pos, orig in enumerate(selected_indices)}
    trial_meta = trial_meta[trial_meta["trial_index"].isin(idx_map)].copy()
    if trial_meta.empty:
        return None
    trial_meta["selected_index"] = trial_meta["trial_index"].map(idx_map)
    trial_meta = (
        trial_meta.dropna(subset=["selected_index"])
        .sort_values("selected_index")
        .reset_index(drop=True)
    )
    trial_meta["selected_index"] = trial_meta["selected_index"].astype(int)
    post_isi = np.asarray(post_isi).reshape(-1)
    neu_time = np.asarray(neu_time)
    features = []
    valid_idx = []
    trial_indices = []
    stim_durations = trial_meta["stim_end_ms"] - trial_meta["stim_start_ms"]
    for row_idx, row in trial_meta.iterrows():
        selected_idx = int(row["selected_index"])
        if selected_idx >= neu_session.shape[0] or selected_idx >= post_isi.size:
            continue
        stim_duration = float(stim_durations.iloc[row_idx])
        isi = float(post_isi[selected_idx])
        if not np.isfinite(isi) or isi <= 0:
            continue
        left = stim_duration
        right = stim_duration + isi
        l_idx, r_idx = get_frame_idx_from_time(neu_time, 0, left, right)
        r_idx = min(r_idx, neu_session.shape[-1])
        if r_idx - l_idx < 2:
            continue
        trace = neu_session[selected_idx, :, l_idx:r_idx]
        if np.all(np.isnan(trace)):
            continue
        feature = np.nanmean(trace, axis=-1)
        if not np.all(np.isfinite(feature)):
            continue
        features.append(feature)
        valid_idx.append(row_idx)
        trial_indices.append(selected_idx)
    if not features:
        return None
    trial_meta_valid = trial_meta.iloc[valid_idx].reset_index(drop=True).copy()
    trial_meta_valid["feature_trial_index"] = trial_indices
    trial_meta_valid["stim_duration_ms"] = (
        trial_meta_valid["stim_end_ms"] - trial_meta_valid["stim_start_ms"]
    )
    trial_meta_valid["post_isi_ms"] = [float(post_isi[idx]) for idx in trial_indices]
    X_isi = np.stack(features)
    trial_neu = neu_session[trial_meta_valid["feature_trial_index"].to_numpy(), :, :]
    return {
        "meta": trial_meta_valid,
        "X_isi": X_isi,
        "trial_neu": trial_neu,
        "neu_time": neu_time,
    }


def _to_builtin(
    value: Union[np.generic, np.ndarray, float, int],
) -> Union[float, int, str]:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return _json_default(value)
    return value


def compute_counts(
    meta: pd.DataFrame,
    total_neurons_available: int,
    sampled_neurons: int,
) -> Dict[str, Union[int, float]]:
    block_counts = meta["block_type"].value_counts().to_dict()
    counts: Dict[str, Union[int, float]] = {
        "n_trials": int(meta.shape[0]),
        "n_neurons_total": int(total_neurons_available),
        "n_neurons_sampled": int(sampled_neurons),
        "n_sessions": 1,
        "n_subjects": 1,
        "n_mice": 1,
    }
    counts["n_trials_short"] = int(block_counts.get(0, 0))
    counts["n_trials_long"] = int(block_counts.get(1, 0))
    return counts


def determine_bootstrap_targets(
    labels: np.ndarray,
    subsets: List[str],
) -> Dict[str, Optional[int]]:
    available_counts: Dict[str, int] = {}
    for name in subsets:
        subset_values = CELL_SUBSETS.get(name)
        mask = (
            np.ones_like(labels, dtype=bool)
            if subset_values is None
            else np.isin(labels, subset_values)
        )
        available_counts[name] = int(mask.sum())

    non_all_counts = [
        count for name, count in available_counts.items() if name != "all" and count > 0
    ]
    target_non_all = max(non_all_counts) if non_all_counts else None

    targets: Dict[str, Optional[int]] = {}
    for name in subsets:
        count = available_counts.get(name, 0)
        if count == 0:
            targets[name] = None
            continue
        if name == "all" or target_non_all is None:
            targets[name] = count
        else:
            targets[name] = target_non_all
    return targets


def subset_dataset_by_neurons(
    dataset: Dict,
    subset_name: str,
    target_neurons: Optional[int],
    rng: np.random.Generator,
) -> Optional[Dict]:
    labels = dataset.get("cell_labels")
    if labels is None:
        return None
    subset_values = CELL_SUBSETS.get(subset_name)
    mask = (
        np.ones_like(labels, dtype=bool)
        if subset_values is None
        else np.isin(labels, subset_values)
    )
    available = np.where(mask)[0]
    available_count = int(available.size)
    if available_count == 0:
        return None

    if target_neurons is None or target_neurons <= available_count:
        selected = np.sort(available)
        bootstrap_multiplier = 1.0
    else:
        sampled = rng.choice(available, size=target_neurons, replace=True)
        selected = np.sort(sampled)
        bootstrap_multiplier = (
            float(target_neurons) / float(available_count) if available_count > 0 else 1.0
        )

    new_dataset = {
        "meta": dataset["meta"],
        "X_isi": dataset["X_isi"][:, selected],
        "trial_neu": dataset["trial_neu"][:, selected, :],
        "neu_time": dataset["neu_time"],
        "cell_labels": dataset["cell_labels"][selected],
        "region": dataset["region"],
        "subject": dataset["subject"],
        "session_name": dataset["session_name"],
    }
    counts = compute_counts(dataset["meta"], available_count, int(selected.size))
    counts["subset_neuron_limit"] = (
        target_neurons if target_neurons is not None else int(selected.size)
    )
    counts["subset_available_neurons"] = available_count
    counts["subset_bootstrap_multiplier"] = bootstrap_multiplier
    new_dataset["counts"] = counts
    new_dataset["subset_name"] = subset_name
    return new_dataset


def infer_region(session_name: str) -> str:
    upper = session_name.upper()
    if "_V1_" in upper:
        return "V1"
    if "_PPC_" in upper:
        return "PPC"
    return "Unknown"


def plot_confusion_matrix(
    cm: np.ndarray, class_labels: List[str], title: str
) -> matplotlib.figure.Figure:
    fig, ax = plt.subplots(figsize=(3.8, 3.4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_labels)))
    ax.set_yticks(range(len(class_labels)))
    ax.set_xticklabels(class_labels, rotation=45, ha="right")
    ax.set_yticklabels(class_labels)
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, int(val), ha="center", va="center", color="black", fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def rms_sem(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    return float(np.sqrt(np.mean(np.square(arr))))


def summarise_timecourse(tc: pd.DataFrame) -> pd.DataFrame:
    return (
        tc.groupby(["block_type", "time_center_ms"])
        .agg(
            acc_mean=("accuracy_mean", "mean"),
            acc_sem=("accuracy_sem", rms_sem),
            chance_mean=("chance_mean", "mean"),
            chance_sem=("chance_sem", rms_sem),
        )
        .reset_index()
    )


def _determine_ylim(summary_df: pd.DataFrame) -> Tuple[float, float]:
    if summary_df.empty:
        return (0.0, 1.0)
    candidates = []
    for mean_col, sem_col in [("acc_mean", "acc_sem"), ("chance_mean", "chance_sem")]:
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
    parts = []
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
        sessions_txt = f"{int(sessions)} sessions" if sessions is not None else None
        mice_txt = f"{int(mice)} mice" if mice is not None else None
        info = ", ".join(item for item in [sessions_txt, mice_txt] if item)
        if info:
            parts.append(info)
    return "\n".join(parts) if parts else None


def plot_timecourse(
    summary_df: pd.DataFrame,
    title: str,
    counts: Optional[Dict[str, Union[int, float]]] = None,
) -> matplotlib.figure.Figure:
    fig, ax = plt.subplots(figsize=(7, 4))
    color_cycle = cycle(plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0"]))
    for block, sub in summary_df.groupby("block_type"):
        color = next(color_cycle)
        label_block = BLOCK_LABELS.get(int(block), f"block {block}")
        sub = sub.sort_values("time_center_ms")
        ax.plot(
            sub["time_center_ms"], sub["acc_mean"], label=f"{label_block}", color=color
        )
        if np.isfinite(sub["acc_sem"]).any():
            ax.fill_between(
                sub["time_center_ms"],
                sub["acc_mean"] - sub["acc_sem"],
                sub["acc_mean"] + sub["acc_sem"],
                color=color,
                alpha=0.2,
            )
        ax.plot(
            sub["time_center_ms"],
            sub["chance_mean"],
            linestyle="--",
            color=color,
            alpha=0.6,
        )
        if np.isfinite(sub["chance_sem"]).any():
            ax.fill_between(
                sub["time_center_ms"],
                sub["chance_mean"] - sub["chance_sem"],
                sub["chance_mean"] + sub["chance_sem"],
                color=color,
                alpha=0.15,
            )
    ax.axhline(0.5, color="grey", linestyle=":", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Time relative to stimulus (ms)")
    ax.set_ylabel("Accuracy")
    ax.legend(loc="lower right")
    y_min, y_max = _determine_ylim(summary_df)
    ax.set_ylim(y_min, y_max)
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
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75),
        )
    fig.tight_layout()
    return fig


def flatten_metrics(
    metrics: Dict,
    context: PlotContext,
    target: str,
) -> Dict:
    payload: Dict[str, object] = {
        "session": context.session,
        "subject": context.subject,
        "region": context.region,
        "target": target,
        "cell_subset": context.cell_subset,
    }
    for key, value in context.counts.items():
        payload[key] = _to_builtin(value)
    for key, value in metrics.items():
        if isinstance(value, (np.ndarray, list, tuple)):
            payload[key] = json.dumps(_json_default(value))
        else:
            payload[key] = _json_default(value)
    return payload


def context_metadata(context: PlotContext, target: str) -> Dict[str, object]:
    payload: Dict[str, object] = {
        "session": context.session,
        "subject": context.subject,
        "region": context.region,
        "target": target,
        "cell_subset": context.cell_subset,
    }
    for key, value in context.counts.items():
        payload[key] = _to_builtin(value)
    return payload


def counts_from_record(
    record: Union[pd.Series, Dict[str, object]],
) -> Dict[str, Union[int, float]]:
    keys = [
        "n_trials",
        "n_trials_short",
        "n_trials_long",
        "n_neurons_total",
        "n_neurons_sampled",
        "n_sessions",
        "n_subjects",
        "n_mice",
    ]
    counts: Dict[str, Union[int, float]] = {}
    for key in keys:
        if key in record and not pd.isna(record[key]):
            counts[key] = _to_builtin(record[key])
    return counts


def parse_confusion_matrix(
    value: Union[str, Sequence[Sequence[float]], np.ndarray],
) -> np.ndarray:
    if isinstance(value, str):
        try:
            data = json.loads(value)
        except json.JSONDecodeError:
            data = value
    else:
        data = value
    return np.asarray(data, dtype=float)


def infer_class_labels(
    row: Union[pd.Series, Dict[str, object]], fallback: Sequence[str]
) -> List[str]:
    labels = row.get("class_labels")
    if labels is None or (isinstance(labels, float) and np.isnan(labels)):
        return list(fallback)
    if isinstance(labels, str):
        try:
            parsed = json.loads(labels)
        except json.JSONDecodeError:
            parsed = labels
        labels = parsed
    if isinstance(labels, (list, tuple, np.ndarray)):
        return [str(label) for label in labels]
    return list(fallback)


def find_decoder_directory(
    output_root: Path, region_guess: str, subject: str, session_name: str
) -> Tuple[Path, str]:
    subject_slug = slugify(subject)
    session_slug = slugify(session_name)
    guess_dir = output_root / slugify(region_guess) / subject_slug / session_slug
    if guess_dir.exists():
        return guess_dir, slugify(region_guess)
    candidates = list(output_root.glob(f"*/{subject_slug}/{session_slug}"))
    if not candidates:
        raise FileNotFoundError(
            f"No decoder exports found for {subject}/{session_name}"
        )
    directory = candidates[0]
    region = directory.parent.parent.name
    return directory, region


def load_session_dataset(subject: str, session_name: str, data_root: Path) -> Dict:
    session_path = data_root / subject / session_name
    if not session_path.exists():
        raise FileNotFoundError(f"Session folder not found: {session_path}")
    ops = read_ops([str(session_path)])[0]
    Trialization.run(ops)
    neural_trials = read_neural_trials(ops, smooth=False)
    labels, *_ = read_masks(ops)
    alignment = run_get_stim_response(
        f"temp_ImageChangeDecoder_{session_name}",
        [neural_trials],
        expected="none",
    )
    neu_time = alignment["neu_time"]
    _, neu_trial_data, neu_labels, _ = get_neu_trial(
        alignment,
        [labels],
        alignment["list_stim_labels"],
        mean_sem=False,
        trial_param=TRIAL_PARAM,
        cate=[-1, 0, 1],
    )
    neu_list, _, _, _, post_isi_list = neu_trial_data
    neu_array = np.asarray(neu_list[0])
    stim_labels = np.asarray(alignment["list_stim_labels"][0])
    post_isi = np.asarray(post_isi_list[0])
    mask = pick_trial(stim_labels, *TRIAL_PARAM, frac=1)
    selection_indices = np.where(mask)[0]
    dataset = prepare_session_dataset(
        neu_array,
        stim_labels,
        post_isi,
        neu_time,
        selection_indices,
    )
    if dataset is None:
        raise RuntimeError("No valid trials after dataset preparation.")
    dataset["subject"] = subject
    dataset["session_name"] = session_name
    dataset["region"] = infer_region(session_name)
    dataset["cell_labels"] = np.asarray(neu_labels, dtype=int)
    return dataset


def run_change_vs_repeat(
    dataset: Dict,
    context: PlotContext,
    output_root: Path,
    whole_rows: List[Dict],
    timecourse_rows: List[Dict],
    confusion_rows: List[Dict],
) -> None:
    meta = dataset["meta"]
    y = meta["is_transition"].astype(int).to_numpy()
    if np.unique(y).size < 2:
        print("Skipping change_vs_repeat: insufficient class diversity.")
        return
    target_base = "change_vs_repeat"
    target_dir = context.target_key(target_base)
    metrics = binary_decoder_cv(
        dataset["X_isi"],
        y,
        n_splits=20,
        random_state=RANDOM_STATE,
        shuffle_baseline=SHUFFLE_BASELINE,
        downsample=True,
    )
    fig_cm = plot_confusion_matrix(
        np.asarray(metrics["confusion_matrix"]),
        ["repeat", "change"],
        f"{context.session} — Change vs Repeat",
    )
    save_figure(
        fig_cm,
        output_root,
        context.region,
        context.subject,
        context.session,
        target_dir,
        "confusion_matrix",
        output_format=context.output_format,
    )
    base_info = context_metadata(context, target_base)
    append_confusion_rows(
        confusion_rows,
        metrics["confusion_matrix"],
        ["repeat", "change"],
        ["repeat", "change"],
        base_info,
    )
    whole_rows.append(flatten_metrics(metrics, context, target_base))

    tc = binary_timecourse_decoding(
        dataset["trial_neu"],
        dataset["neu_time"],
        y,
        block_labels=meta["block_type"].to_numpy(),
        bin_width_ms=BIN_WIDTH_MS,
        window=TIME_WINDOW_MS,
        n_splits=5,
        random_state=RANDOM_STATE,
        shuffle_baseline=SHUFFLE_BASELINE,
        downsample=True,
    )
    if not tc.empty:
        summary = summarise_timecourse(tc)
        fig_tc = plot_timecourse(
            summary,
            f"{context.session} — Change vs Repeat (timecourse)",
            context.counts,
        )
        save_figure(
            fig_tc,
            output_root,
            context.region,
            context.subject,
            context.session,
            target_dir,
            "timecourse",
            output_format=context.output_format,
        )
        summary = summary.assign(
            session=context.session,
            subject=context.subject,
            region=context.region,
            target=target_base,
            cell_subset=context.cell_subset,
        )
        for key, value in context.counts.items():
            summary[key] = _to_builtin(value)
        timecourse_rows.extend(summary.to_dict("records"))


def run_orientation_multiclass(
    dataset: Dict,
    context: PlotContext,
    output_root: Path,
    whole_rows: List[Dict],
    timecourse_rows: List[Dict],
    confusion_rows: List[Dict],
) -> None:
    meta = dataset["meta"]
    y = meta["orientation_idx"].astype(int)
    if y.nunique() < 2:
        print("Skipping orientation decoder: fewer than two orientations present.")
        return
    metrics = multiclass_decoder_cv(
        dataset["X_isi"],
        y.to_numpy(),
        n_splits=20,
        random_state=RANDOM_STATE,
        shuffle_baseline=SHUFFLE_BASELINE,
    )
    target_base = "orientation"
    target_dir = context.target_key(target_base)

    class_labels = metrics.get("class_labels")
    orientation_lookup = (
        meta[["orientation_idx", "orientation_deg"]]
        .drop_duplicates()
        .set_index("orientation_idx")["orientation_deg"]
        .to_dict()
    )
    if class_labels:
        plot_labels = []
        for lbl in class_labels:
            idx = int(float(lbl))
            deg = orientation_lookup.get(idx)
            if deg is None or np.isnan(deg):
                plot_labels.append(f"ori{idx}")
            else:
                plot_labels.append(f"{deg:g}°")
    else:
        unique_idx = sorted(y.unique())
        plot_labels = [f"ori{idx}" for idx in unique_idx]

    fig_cm = plot_confusion_matrix(
        np.asarray(metrics["confusion_matrix"]),
        plot_labels,
        f"{context.session} — Orientation",
    )
    save_figure(
        fig_cm,
        output_root,
        context.region,
        context.subject,
        context.session,
        target_dir,
        "confusion_matrix",
        output_format=context.output_format,
    )
    base_info = context_metadata(context, target_base)
    append_confusion_rows(
        confusion_rows,
        metrics["confusion_matrix"],
        plot_labels,
        plot_labels,
        base_info,
    )
    whole_rows.append(flatten_metrics(metrics, context, target_base))

    tc = multiclass_timecourse_decoding(
        dataset["trial_neu"],
        dataset["neu_time"],
        y.to_numpy(),
        block_labels=meta["block_type"].to_numpy(),
        bin_width_ms=BIN_WIDTH_MS,
        window=TIME_WINDOW_MS,
        n_splits=5,
        random_state=RANDOM_STATE,
        shuffle_baseline=SHUFFLE_BASELINE,
    )
    if not tc.empty:
        summary = summarise_timecourse(tc)
        fig_tc = plot_timecourse(
            summary, f"{context.session} — Orientation (timecourse)", context.counts
        )
        save_figure(
            fig_tc,
            output_root,
            context.region,
            context.subject,
            context.session,
            target_dir,
            "timecourse",
            output_format=context.output_format,
        )
        summary = summary.assign(
            session=context.session,
            subject=context.subject,
            region=context.region,
            target=target_base,
            cell_subset=context.cell_subset,
        )
        for key, value in context.counts.items():
            summary[key] = _to_builtin(value)
        timecourse_rows.extend(summary.to_dict("records"))


def run_to_or_repeat(
    dataset: Dict,
    context: PlotContext,
    output_root: Path,
    whole_rows: List[Dict],
    timecourse_rows: List[Dict],
    confusion_rows: List[Dict],
) -> None:
    meta = dataset["meta"]
    target_base = "to_or_repeat"
    target_dir_base = context.target_key(target_base)

    orientation_degrees = (
        meta[["orientation_idx", "orientation_deg"]]
        .drop_duplicates()
        .set_index("orientation_idx")["orientation_deg"]
        .to_dict()
    )
    aggregated_timecourses: List[pd.DataFrame] = []

    for ori_idx in sorted(meta["orientation_idx"].unique()):
        mask = meta["orientation_idx"] == ori_idx
        if mask.sum() < 4:
            continue
        y = meta.loc[mask, "is_transition"].astype(int).to_numpy()
        X = dataset["X_isi"][mask.to_numpy()]
        if np.unique(y).size < 2 or np.min(np.bincount(y)) < 2:
            continue
        metrics = binary_decoder_cv(
            X,
            y,
            n_splits=20,
            random_state=RANDOM_STATE,
            shuffle_baseline=SHUFFLE_BASELINE,
            downsample=True,
        )
        ori_deg = orientation_degrees.get(int(ori_idx))
        target = f"{target_base}_ori{int(ori_idx)}"
        if ori_deg is not None and np.isfinite(ori_deg):
            target = f"{target_base}_ori{int(ori_idx)}_{int(ori_deg):g}deg"
        target_dir = context.target_key(target)
        fig_cm = plot_confusion_matrix(
            np.asarray(metrics["confusion_matrix"]),
            ["repeat", "to"],
            f"{context.session} — Orientation {int(ori_idx)}",
        )
        save_figure(
            fig_cm,
            output_root,
            context.region,
            context.subject,
            context.session,
            target_dir,
            "confusion_matrix",
            output_format=context.output_format,
        )
        base_info = context_metadata(context, target)
        base_info["orientation_idx"] = int(ori_idx)
        base_info["orientation_deg"] = float(ori_deg) if ori_deg is not None else np.nan
        append_confusion_rows(
            confusion_rows,
            metrics["confusion_matrix"],
            ["repeat", "to"],
            ["repeat", "to"],
            base_info,
        )
        row = flatten_metrics(metrics, context, target)
        row["orientation_idx"] = int(ori_idx)
        row["orientation_deg"] = float(ori_deg) if ori_deg is not None else np.nan
        whole_rows.append(row)

        tc = binary_timecourse_decoding(
            dataset["trial_neu"][mask.to_numpy()],
            dataset["neu_time"],
            y,
            block_labels=meta.loc[mask, "block_type"].to_numpy(),
            bin_width_ms=BIN_WIDTH_MS,
            window=TIME_WINDOW_MS,
            n_splits=5,
            random_state=RANDOM_STATE,
            shuffle_baseline=SHUFFLE_BASELINE,
            downsample=True,
        )
        if not tc.empty:
            tc = tc.assign(orientation_idx=int(ori_idx))
            aggregated_timecourses.append(tc)
            summary = summarise_timecourse(tc)
            fig_tc = plot_timecourse(
                summary,
                f"{context.session} — Orientation {int(ori_idx)} (timecourse)",
                context.counts,
            )
            save_figure(
                fig_tc,
                output_root,
                context.region,
                context.subject,
                context.session,
                target_dir,
                "timecourse",
                output_format=context.output_format,
            )
            summary = summary.assign(
                session=context.session,
                subject=context.subject,
                region=context.region,
                target=target,
                orientation_idx=int(ori_idx),
                orientation_deg=float(ori_deg) if ori_deg is not None else np.nan,
                cell_subset=context.cell_subset,
            )
            for key, value in context.counts.items():
                summary[key] = _to_builtin(value)
            timecourse_rows.extend(summary.to_dict("records"))

    if aggregated_timecourses:
        tc_all = pd.concat(aggregated_timecourses, ignore_index=True)
        summary = summarise_timecourse(tc_all)
        fig_tc = plot_timecourse(
            summary,
            f"{context.session} — To vs Repeat (averaged timecourse)",
            context.counts,
        )
        save_figure(
            fig_tc,
            output_root,
            context.region,
            context.subject,
            context.session,
            target_dir_base,
            "averaged_timecourse",
            output_format=context.output_format,
        )
        summary = summary.assign(
            session=context.session,
            subject=context.subject,
            region=context.region,
            target=target_base,
            cell_subset=context.cell_subset,
        )
        for key, value in context.counts.items():
            summary[key] = _to_builtin(value)
        timecourse_rows.extend(summary.to_dict("records"))


def regenerate_plots_from_exports(
    subject: str,
    session_name: str,
    data_root: Path,
    output_root: Path,
    subsets: List[str],
    output_format: str,
) -> None:
    region_guess = infer_region(session_name)
    decoder_dir, region_slug = find_decoder_directory(
        output_root, region_guess, subject, session_name
    )
    whole_path = decoder_dir / "whole_metrics.csv"
    timecourse_path = decoder_dir / "timecourse_metrics.csv"
    confusion_path = decoder_dir / "confusion_matrices.csv"
    if (
        not whole_path.exists()
        or not timecourse_path.exists()
        or not confusion_path.exists()
    ):
        raise FileNotFoundError(
            f"Missing exported decoder CSVs in {decoder_dir}. Re-run decoders before plotting only."
        )
    whole_df = pd.read_csv(whole_path)
    timecourse_df = pd.read_csv(timecourse_path)
    confusion_df = pd.read_csv(confusion_path)
    if "cell_subset" not in whole_df.columns:
        raise ValueError(
            "Existing decoder exports do not include cell_subset metadata. "
            "Please rerun the decoder pipeline once to regenerate the updated exports."
        )
    region = whole_df["region"].iloc[0] if "region" in whole_df.columns else region_slug
    subject_from_df = (
        whole_df["subject"].iloc[0] if "subject" in whole_df.columns else subject
    )
    print(
        f"Rebuilding plots from exports for {subject_from_df}/{session_name} ({region})"
    )

    for subset in subsets:
        subset_whole = whole_df[whole_df["cell_subset"] == subset]
        subset_timecourse = timecourse_df[timecourse_df["cell_subset"] == subset]

        targets = sorted(
            set(subset_whole["target"].unique())
            | set(subset_timecourse["target"].unique())
        )
        for target in targets:
            meta_row = subset_whole[subset_whole["target"] == target].head(1)
            counts = counts_from_record(meta_row.iloc[0]) if not meta_row.empty else {}
            context = PlotContext(
                region=region,
                subject=subject_from_df,
                session=session_name,
                cell_subset=subset,
                counts=counts,
                output_format=output_format,
            )
            # Confusion matrix
            if not meta_row.empty:
                matrix = parse_confusion_matrix(meta_row.iloc[0]["confusion_matrix"])
                if target == "change_vs_repeat":
                    labels = ["repeat", "change"]
                    title = f"{session_name} — Change vs Repeat"
                    filename = "confusion_matrix"
                elif target == "orientation":
                    labels = infer_class_labels(meta_row.iloc[0], [])
                    if not labels:
                        labels = ["orientation"]
                    title = f"{session_name} — Orientation"
                    filename = "confusion_matrix"
                else:
                    labels = ["repeat", "to"]
                    ori_idx = meta_row.iloc[0].get("orientation_idx")
                    ori_deg = meta_row.iloc[0].get("orientation_deg")
                    if pd.notna(ori_idx):
                        if pd.notna(ori_deg):
                            title = f"{session_name} — Orientation {int(ori_idx)} ({ori_deg:g}°)"
                        else:
                            title = f"{session_name} — Orientation {int(ori_idx)}"
                    else:
                        title = f"{session_name} — To vs Repeat"
                    filename = "confusion_matrix"
                save_figure(
                    plot_confusion_matrix(matrix, labels, title),
                    output_root,
                    context.region,
                    context.subject,
                    context.session,
                    context.target_key(target),
                    filename,
                    output_format=output_format,
                )
            # Timecourse plots
            tc_subset = subset_timecourse[subset_timecourse["target"] == target]
            if not tc_subset.empty:
                counts_tc = counts_from_record(tc_subset.iloc[0]) or counts
                tc_context = PlotContext(
                    region=region,
                    subject=subject_from_df,
                    session=session_name,
                    cell_subset=subset,
                    counts=counts_tc,
                    output_format=output_format,
                )
                ori_idx = (
                    tc_subset.iloc[0].get("orientation_idx")
                    if "orientation_idx" in tc_subset.columns
                    else np.nan
                )
                ori_deg = (
                    tc_subset.iloc[0].get("orientation_deg")
                    if "orientation_deg" in tc_subset.columns
                    else np.nan
                )
                if target == "change_vs_repeat":
                    title = f"{session_name} — Change vs Repeat (timecourse)"
                    filename = "timecourse"
                elif target == "orientation":
                    title = f"{session_name} — Orientation (timecourse)"
                    filename = "timecourse"
                elif pd.notna(ori_idx):
                    if pd.notna(ori_deg):
                        title = (
                            f"{session_name} — Orientation {int(ori_idx)} (timecourse)"
                        )
                    else:
                        title = (
                            f"{session_name} — Orientation {int(ori_idx)} (timecourse)"
                        )
                    filename = "timecourse"
                else:
                    title = f"{session_name} — To vs Repeat (averaged timecourse)"
                    filename = "averaged_timecourse"
                fig_tc = plot_timecourse(tc_subset, title, tc_context.counts)
                save_figure(
                    fig_tc,
                    output_root,
                    tc_context.region,
                    tc_context.subject,
                    tc_context.session,
                    tc_context.target_key(target),
                    filename,
                    output_format=output_format,
                )


def generate_plots_for_session(
    subject: str,
    session_name: str,
    data_root: Path,
    output_root: Path,
    subsets: List[str],
    plots_only: bool,
    output_format: str,
    bootstrap_seed: int,
) -> None:
    output_root = output_root.expanduser()
    if plots_only:
        regenerate_plots_from_exports(
            subject, session_name, data_root, output_root, subsets, output_format
        )
        return

    dataset = load_session_dataset(subject, session_name, data_root)
    print(f"Loaded dataset: {subject}/{session_name} ({dataset['region']})")
    rng = np.random.default_rng(bootstrap_seed)
    bootstrap_targets = determine_bootstrap_targets(
        dataset["cell_labels"],
        subsets,
    )

    whole_rows: List[Dict] = []
    timecourse_rows: List[Dict] = []
    confusion_rows: List[Dict] = []

    for subset in subsets:
        subset_dataset = subset_dataset_by_neurons(
            dataset,
            subset,
            bootstrap_targets.get(subset),
            rng,
        )
        if subset_dataset is None:
            print(f"Skipping subset '{subset}': no neurons available.")
            continue
        context = PlotContext(
            region=dataset["region"],
            subject=dataset["subject"],
            session=session_name,
            cell_subset=subset,
            counts=subset_dataset["counts"],
            output_format=output_format,
        )
        run_change_vs_repeat(
            subset_dataset,
            context,
            output_root,
            whole_rows,
            timecourse_rows,
            confusion_rows,
        )
        run_orientation_multiclass(
            subset_dataset,
            context,
            output_root,
            whole_rows,
            timecourse_rows,
            confusion_rows,
        )
        run_to_or_repeat(
            subset_dataset,
            context,
            output_root,
            whole_rows,
            timecourse_rows,
            confusion_rows,
        )

    region = dataset["region"]
    subject_slug = slugify(dataset["subject"])
    session_slug = slugify(dataset["session_name"])
    decoder_dir = ensure_dir(
        output_root / slugify(region) / subject_slug / session_slug
    )

    if whole_rows:
        pd.DataFrame(whole_rows).to_csv(decoder_dir / "whole_metrics.csv", index=False)
    if timecourse_rows:
        pd.DataFrame(timecourse_rows).to_csv(
            decoder_dir / "timecourse_metrics.csv", index=False
        )
    if confusion_rows:
        pd.DataFrame(confusion_rows).to_csv(
            decoder_dir / "confusion_matrices.csv", index=False
        )

    print("Plots and summaries written to", decoder_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate decoder plots for an ImageChange session."
    )
    parser.add_argument("subject", help="Subject folder name (e.g., YH02VT)")
    parser.add_argument(
        "session_name",
        help="Session folder name (e.g., VTYH02_PPC_20250228_1451ShortLong)",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("results"),
        help="Root directory containing session folders (default: results)",
    )
    parser.add_argument(
        "--plot-dir",
        "--output-root",
        dest="output_root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory to write decoder plots and CSV summaries (default: plot_exports)",
    )
    parser.add_argument(
        "--cell-subsets",
        nargs="+",
        choices=sorted(CELL_SUBSETS.keys()),
        default=["all", "exc", "vip", "sst"],
        help="Neuron subsets to decode separately (default: all exc vip sst).",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=RANDOM_STATE,
        help="Random seed used for neuron bootstrapping (default: 0).",
    )
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Skip decoder inference and regenerate figures from existing CSV exports.",
    )
    parser.add_argument(
        "--output-format",
        choices=["pdf", "png"],
        default="pdf",
        help="Figure file format for saved plots (default: pdf).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_plots_for_session(
        subject=args.subject,
        session_name=args.session_name,
        data_root=args.data_root,
        output_root=args.output_root,
        subsets=args.cell_subsets,
        plots_only=args.plots_only,
        output_format=args.output_format,
        bootstrap_seed=args.bootstrap_seed,
    )


if __name__ == "__main__":
    main()
