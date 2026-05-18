"""Utility functions extracted from the alignment notebooks."""

import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from scipy.stats import sem


def get_early_late_epochs(block_type, early_n=5, late_n=5):
    """Label early and late trials within each block."""
    block_type = np.array(block_type)
    labels = np.zeros_like(block_type, dtype=int)

    diff = np.diff(block_type, prepend=block_type[0], append=block_type[-1])
    block_starts = np.where(diff != 0)[0]
    block_ends = np.where(diff != 0)[0][1:]
    block_ends = np.append(block_ends, len(block_type))

    for start, end in zip(block_starts, block_ends):
        block_id = block_type[start]
        if block_id not in [1, 2]:
            continue
        block_length = end - start
        if block_length < early_n:
            labels[start:end] = 1
        else:
            labels[start:start + early_n] = 1
            if block_length >= early_n + late_n:
                labels[end - late_n:end] = 2
            elif block_length > early_n:
                labels[end - (block_length - early_n):end] = 2
    return labels


def compute_avg_and_sem(data, mask, axis=(0, 1)):
    """Compute mean, SEM, and total trial-neuron count for masked data."""
    masked_data = data[mask, :, :]
    n_trials, n_neurons, _ = masked_data.shape
    avg = np.nanmean(masked_data, axis=axis)
    sem_data = sem(masked_data, axis=axis, nan_policy='omit')
    return avg, sem_data, n_trials * n_neurons


def get_y_limits(avg_data1, sem_data1, avg_data2, sem_data2):
    """Compute shared y-axis limits for two average ± SEM traces."""
    if np.all(np.isnan(avg_data1)) and np.all(np.isnan(avg_data2)):
        return np.nan, np.nan
    y_min1 = np.nanmin(avg_data1 - sem_data1) if not np.all(np.isnan(avg_data1)) else np.nan
    y_max1 = np.nanmax(avg_data1 + sem_data1) if not np.all(np.isnan(avg_data1)) else np.nan
    y_min2 = np.nanmin(avg_data2 - sem_data2) if not np.all(np.isnan(avg_data2)) else np.nan
    y_max2 = np.nanmax(avg_data2 + sem_data2) if not np.all(np.isnan(avg_data2)) else np.nan
    y_min = np.nanmin([y_min1, y_min2])
    y_max = np.nanmax([y_max1, y_max2])
    return y_min, y_max


def process_trial_data(
    neu_seq,
    neu_time,
    stim_seq,
    trial_mask1,
    trial_mask2,
    stim_mask,
    target_state,
    indices,
):
    """Compute average trial responses and aligned mean stimulus times."""
    avg_data1, sem_data1, n_trials_neurons1 = compute_avg_and_sem(neu_seq, trial_mask1)
    avg_data2, sem_data2, n_trials_neurons2 = compute_avg_and_sem(neu_seq, trial_mask2)

    relevant_trials = trial_mask1 | trial_mask2
    if target_state == 'stim_seq':
        stim_idx = indices // 2
        alignment_times = stim_seq[relevant_trials, stim_idx, 0]
        adjusted_stim_seq = stim_seq[relevant_trials, :, :] - alignment_times[:, None, None]
        mean_stim_times = np.nanmean(adjusted_stim_seq, axis=0)
    else:
        mean_stim_times = np.nanmean(stim_seq[relevant_trials & stim_mask, :, :], axis=0)

    return (
        neu_time,
        avg_data1,
        sem_data1,
        n_trials_neurons1,
        avg_data2,
        sem_data2,
        n_trials_neurons2,
        mean_stim_times,
    )


def process_licking_data(licks_per_trial, correctness_per_trial, trial_mask, l_frames, r_frames, frame_rate=30):
    """Bin licking events and return average/SEM lick rates for a condition."""
    dt_ms = 1000 / frame_rate
    time_bins = np.arange(-l_frames, r_frames) * dt_ms

    correct_lick_counts = np.zeros((np.sum(trial_mask), len(time_bins) - 1))
    incorrect_lick_counts = np.zeros((np.sum(trial_mask), len(time_bins) - 1))

    trial_idx = 0
    for trial in range(len(trial_mask)):
        if not trial_mask[trial]:
            continue

        lick_times = licks_per_trial[trial]
        correctness = correctness_per_trial[trial]

        if len(lick_times) == 0:
            trial_idx += 1
            continue

        correct_licks = lick_times[correctness == 1]
        incorrect_licks = lick_times[correctness == 0]

        if len(correct_licks) > 0:
            correct_lick_counts[trial_idx, :], _ = np.histogram(correct_licks, bins=time_bins)

        if len(incorrect_licks) > 0:
            incorrect_lick_counts[trial_idx, :], _ = np.histogram(incorrect_licks, bins=time_bins)

        trial_idx += 1

    dt_sec = dt_ms / 1000
    correct_lick_rates = correct_lick_counts / dt_sec
    incorrect_lick_rates = incorrect_lick_counts / dt_sec

    avg_correct = np.nanmean(correct_lick_rates, axis=0)
    sem_correct = sem(correct_lick_rates, axis=0, nan_policy='omit')
    avg_incorrect = np.nanmean(incorrect_lick_rates, axis=0)
    sem_incorrect = sem(incorrect_lick_rates, axis=0, nan_policy='omit')
    lick_time = time_bins[:-1]

    return (
        lick_time,
        avg_correct,
        sem_correct,
        int(np.sum(trial_mask)),
        avg_incorrect,
        sem_incorrect,
        int(np.sum(trial_mask)),
    )


def update_plot(alignment, conditions, outcomes, mode, data_storage):
    """Update a Plotly figure for the interactive alignment notebooks."""
    if not conditions or not outcomes:
        fig = go.Figure()
        fig.add_annotation(
            text='Please select at least one condition and one outcome.',
            xref='paper', yref='paper',
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20),
        )
        return fig, ''


def plot_rastermap_heatmap(neu_seq, neu_time, ax=None, ax_cb=None, norm_mode='minmax', max_pixel=258):
    """Plot a Rastermap-sorted neural heatmap."""
    import rastermap as rm

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax_cb = fig.add_axes([0.92, 0.15, 0.02, 0.7])

    valid_idx = np.where(~np.all(np.isnan(neu_seq), axis=1))[0]
    data = neu_seq[valid_idx, :].copy()

    model = rm.Rastermap(n_clusters=3, locality=1)
    model.fit(data)
    sorted_idx = model.isort
    data = data[sorted_idx, :]

    nbin = max(1, data.shape[0] // max_pixel)
    data = rm.utils.bin1d(data, bin_size=nbin, axis=0)

    if norm_mode == 'minmax':
        data = (data - np.nanmin(data, axis=1, keepdims=True)) / (
            np.nanmax(data, axis=1, keepdims=True) - np.nanmin(data, axis=1, keepdims=True) + 1e-9
        )
        vmin, vmax = 0, 1
    elif norm_mode == 'none':
        vmin, vmax = np.nanpercentile(data, 1), np.nanpercentile(data, 99)
    else:
        raise ValueError("norm_mode must be 'minmax' or 'none'")

    im = ax.imshow(
        data,
        aspect='auto',
        interpolation='nearest',
        extent=[neu_time[0], neu_time[-1], 1, data.shape[0]],
        cmap='hot',
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_ylabel('Neuron (sorted)')
    ax.set_xlabel('Time (s)')

    if ax_cb is not None:
        cbar = plt.colorbar(im, cax=ax_cb)
        cbar.set_label('dF/F')

    return ax, ax_cb

    try:
        fig = data_storage.get_figure(alignment, conditions, outcomes, mode)
        return fig, ''
    except Exception as exc:
        fig = go.Figure()
        fig.add_annotation(
            text=f'Error generating plot: {str(exc)}',
            xref='paper', yref='paper',
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20),
        )
        return fig, ''
