"""Notebook helper implementations migrated from `Test_pilot/test_nb_lick_analysis.py`.

This module is now self-contained so notebooks do not depend on `Test_pilot/test_nb_*`.
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm
import numpy as np
from scipy.stats import sem
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings("ignore")

def get_roi_label_color(labels, roi_id):
    """
    Get category, colors, and colormap based on neuron label.
    
    Parameters:
    labels (np.ndarray): Array of neuron labels (-1 for excitatory, 0 for unsure, 1 for inhibitory).
    roi_id (int): Index of the neuron.
    
    Returns:
    tuple: (category, color1, color2, cmap)
    """
    if labels[roi_id] == -1:
        cate = 'excitatory'
        color1 = 'grey'
        color2 = 'dodgerblue'
        cmap = LinearSegmentedColormap.from_list('excitatory_cmap', ['white', 'dodgerblue', 'black'])
    elif labels[roi_id] == 0:
        cate = 'unsure'
        color1 = 'grey'
        color2 = 'mediumseagreen'
        cmap = LinearSegmentedColormap.from_list('unsure_cmap', ['white', 'mediumseagreen', 'black'])
    elif labels[roi_id] == 1:
        cate = 'inhibitory'
        color1 = 'grey'
        color2 = 'hotpink'
        cmap = LinearSegmentedColormap.from_list('inhibitory_cmap', ['white', 'hotpink', 'black'])
    return cate, color1, color2, cmap

def apply_colormap(data, cmap):
    """Normalize data and apply colormap."""
    if len(data) > 0:
        vmin, vmax = data.min(), data.max()
        normalized = (data - vmin) / (vmax - vmin + 1e-10)
        return cmap(normalized)
    return np.array([])

def adjust_layout_heatmap(ax):
    """Adjust heatmap layout by removing spines."""
    for spine in ax.spines.values():
        spine.set_visible(False)

def plot_gridspec_subplot(fig, gs, position, time, avg_data, sem_data, title, color='black'):
    """Create a subplot with mean, SEM shading, and custom styling."""
    ax = fig.add_subplot(gs[position])
    ax.plot(time, avg_data, color=color)
    ax.fill_between(time, avg_data - sem_data, avg_data + sem_data, color=color, alpha=0.3)
    ax.axvline(0, color='red', linestyle='--', label='Lick onset')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('dF/F (mean +/- sem)')
    ax.set_title(title)
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return ax

def plot_gridspec_superimposed(fig, gs, position, time, avg_data1, sem_data1, label1, color1, 
                             avg_data2, sem_data2, label2, color2, title):
    """Create a subplot with two superimposed traces."""
    ax = fig.add_subplot(gs[position])
    ax.plot(time, avg_data1, color=color1, label=label1)
    ax.fill_between(time, avg_data1 - sem_data1, avg_data1 + sem_data1, color=color1, alpha=0.3)
    ax.plot(time, avg_data2, color=color2, label=label2)
    ax.fill_between(time, avg_data2 - sem_data2, avg_data2 + sem_data2, color=color2, alpha=0.3)
    ax.axvline(0, color='red', linestyle='--', label='Lick onset')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('dF/F (mean +/- sem)')
    ax.set_title(title)
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return ax

def plot_gridspec_heatmap(fig, gs, position, time, data, labels, title, vmin=None, vmax=None):
    """Create a heatmap subplot with label-specific colormaps."""
    ax = fig.add_subplot(gs[position])
    n_neurons, time_window = data.shape
    heatmap_rgb = np.zeros((n_neurons, time_window, 4))
    
    for i in range(n_neurons):
        _, _, _, cmap = get_roi_label_color(labels, i)
        heatmap_rgb[i] = apply_colormap(data[i], cmap)
    
    im = ax.imshow(heatmap_rgb, interpolation='nearest', aspect='auto',
                   extent=[time[0], time[-1], n_neurons, 0])
    ax.axvline(0, color='red', linestyle='--', label='Lick onset')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Neuron index')
    ax.set_title(title)
    ax.legend()
    adjust_layout_heatmap(ax)
    # fig.colorbar(im, ax=ax, label='dF/F')
    return ax

def plot_sorted_heatmap(fig, gs, position, time, data, labels, title, vmin=None, vmax=None, 
                       sort_interval=None):
    """Plot sorted heatmap using inferno colormap."""
    ax = fig.add_subplot(gs[position])
    n_neurons, time_window = data.shape

    if sort_interval is not None:
        start_t, end_t = sort_interval
        sort_mask = (time >= start_t) & (time <= end_t)
        sort_time = time[sort_mask]
        data_for_sorting = data[:, sort_mask]
        peak_indices = np.argmax(data_for_sorting, axis=1)
        peak_times = sort_time[peak_indices]
    else:
        peak_indices = np.argmax(data, axis=1)
        peak_times = time[peak_indices]

    sorted_indices = np.argsort(peak_times)
    sorted_data = data[sorted_indices]
    cmap = cm.get_cmap('inferno')
    heatmap_rgb = np.array([apply_colormap(row, cmap) for row in sorted_data])

    im = ax.imshow(heatmap_rgb, interpolation='nearest', aspect='auto',
                   extent=[time[0], time[-1], n_neurons, 0])
    ax.axvline(0, color='red', linestyle='--', label='Lick onset')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Neuron index (sorted)')
    ax.set_title(title)
    if ax.get_legend_handles_labels()[0]:
        ax.legend()
    adjust_layout_heatmap(ax)
    # fig.colorbar(im, ax=ax, label='dF/F')
    return ax

def calculate_metrics(neu_seq, filter_licks, labels=None):
    """Calculate average, SEM, and heatmap data for given filter."""
    if labels is None:
        avg = np.nanmean(neu_seq[filter_licks, :, :], axis=(0, 1))
        sem_data = sem(neu_seq[filter_licks, :, :], axis=(0, 1), nan_policy='omit')
        heat = np.nanmean(neu_seq[filter_licks, :, :], axis=0)
        return avg, sem_data, heat
    else:
        avg_all = np.nanmean(neu_seq[filter_licks, :, :], axis=(0, 1))
        sem_all = sem(neu_seq[filter_licks, :, :], axis=(0, 1), nan_policy='omit')
        avg_ex = np.nanmean(neu_seq[filter_licks][:, labels == -1, :], axis=(0, 1))
        sem_ex = sem(neu_seq[filter_licks][:, labels == -1, :], axis=(0, 1), nan_policy='omit')
        avg_inh = np.nanmean(neu_seq[filter_licks][:, labels == 1, :], axis=(0, 1))
        sem_inh = sem(neu_seq[filter_licks][:, labels == 1, :], axis=(0, 1), nan_policy='omit')
        heat = np.nanmean(neu_seq[filter_licks, :, :], axis=0)
        return avg_all, sem_all, avg_ex, sem_ex, avg_inh, sem_inh, heat

def set_uniform_ylim(axes_list):
    """Set uniform y-axis limits across related subplots."""
    ylims = [ax.get_ylim() for ax in axes_list]
    ymin = min(ylim[0] for ylim in ylims)
    ymax = max(ylim[1] for ylim in ylims)
    for ax in axes_list:
        ax.set_ylim(ymin, ymax)

def main(neu_seq, neu_time, direction, correction, lick_type, labels):
    """
    Main function to generate neural activity plots.
    
    Parameters:
    neu_seq (np.ndarray): Neural sequence data
    neu_time (np.ndarray): Time array
    direction (np.ndarray): Direction array (0 for left, 1 for right)
    correction (np.ndarray): Correction array (0 for wrong, 1 for correct)
    lick_type (np.ndarray): Lick type array (1 for first lick, 0 for later)
    labels (np.ndarray): Neuron labels (-1 for excitatory, 0 for unsure, 1 for inhibitory)
    """
    # Calculate metrics for different conditions
    conditions = [
        ('All Licks', np.ones_like(direction, dtype=bool)),
        ('Rewarded First Lick', (lick_type == 1) & (correction == 1)),
        ('Punished First Lick', (lick_type == 1) & (correction == 0)),
        ('Left First Lick', (lick_type == 1) & (direction == 0)),
        ('Right First Lick', (lick_type == 1) & (direction == 1)),
        ('All Left Licks', (direction == 0)),
        ('All Right Licks', (direction == 1))
    ]
    
    metrics = {}
    for name, filter_licks in conditions:
        metrics[name] = calculate_metrics(neu_seq, filter_licks, labels)
    
    # Create figure and GridSpec
    fig = plt.figure(figsize=(34, 72))
    gs = gridspec.GridSpec(14, 6)
    
    # Plotting
    axes_groups = []
    for idx, (name, (avg_all, sem_all, avg_ex, sem_ex, avg_inh, sem_inh, heat)) in enumerate(metrics.items()):
        row = idx * 2
        # Plot individual subplots
        ax1 = plot_gridspec_subplot(
            fig, gs, (row, 0), neu_time, avg_all, sem_all,
            title=f'Average dF/F all neurons around {name}', color='black'
        )
        ax2 = plot_gridspec_subplot(
            fig, gs, (row, 1), neu_time, avg_ex, sem_ex,
            title=f'Average dF/F excitatory neurons around {name}', color='blue'
        )
        ax3 = plot_gridspec_subplot(
            fig, gs, (row, 2), neu_time, avg_inh, sem_inh,
            title=f'Average dF/F inhibitory neurons around {name}', color='red'
        )
        ax4 = plot_gridspec_superimposed(
            fig, gs, (row, 3), neu_time,
            avg_ex, sem_ex, 'Excitatory', 'blue',
            avg_inh, sem_inh, 'Inhibitory', 'red',
            title=f'S superimposed dF/F excitatory and inhibitory neurons around {name}'
        )
        ax5 = plot_gridspec_heatmap(
            fig, gs, (slice(row, row+2), 4), neu_time, heat,
            title=f'All neurons heatmap around {name}', labels=labels
        )
        ax6 = plot_sorted_heatmap(
            fig, gs, (slice(row, row+2), 5), neu_time, heat,
            labels, title=f'Sorted heatmap of all neurons around {name}'
        )
        axes_groups.append([ax1, ax2, ax3, ax4])
    
    # Set uniform y-axis limits for related subplots
    # Uncomment the following lines if you want to set uniform y-limits for each group of axes
    # for axes in axes_groups:
    #     set_uniform_ylim(axes)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Example usage
    l_frames = 30
    r_frames = 30
    neu_seq, neu_time, direction, correction, lick_type = get_lick_response(neural_trials, l_frames, r_frames)
    main(neu_seq, neu_time, direction, correction, lick_type, labels)
