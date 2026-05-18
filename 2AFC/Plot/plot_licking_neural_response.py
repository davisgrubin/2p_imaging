import matplotlib.pyplot as plt
import os
import matplotlib.gridspec as gridspec
from matplotlib import cm
import numpy as np
from scipy.stats import sem
from matplotlib.colors import LinearSegmentedColormap
from Modules.Alignment import get_lick_response
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

def pool_session_data(neural_trials_list, labels_list, l_frames, r_frames):
    """
    Pools neural and behavioral data from multiple experimental sessions into unified arrays.
    This function takes lists of neural trial data and corresponding label arrays from multiple sessions,
    aligns and pads the neural data so that all neurons from all sessions are represented in a single array,
    and concatenates behavioral labels and trial information across sessions. Each session's neurons are
    assigned to a unique slice in the pooled array, with zeros padding non-participating neurons for each trial.
    Parameters:
        neural_trials_list (list of np.ndarray): 
            List of neural trial data arrays, one per session. Each array should have shape 
            (n_trials, n_neurons, n_time), representing the neural activity for each trial.
        labels_list (list of np.ndarray): 
            List of label arrays, one per session. Each array should have shape (n_neurons, ...), 
            containing metadata or identifiers for each neuron.
        l_frames (int): 
            Number of frames to include before the event of interest (e.g., lick onset).
        r_frames (int): 
            Number of frames to include after the event of interest.
        tuple:
            pooled_neu_seq (np.ndarray): 
                Array of pooled neural data with shape (total_trials, total_neurons, n_time), 
                where total_trials is the sum of trials across all sessions and total_neurons is 
                the sum of neurons across all sessions. Neurons not present in a given session are zero-padded.
            neu_time (np.ndarray): 
                Array of time points corresponding to the neural data (shared across sessions).
            pooled_directions (np.ndarray): 
                Concatenated array of trial direction labels from all sessions.
            pooled_corrections (np.ndarray): 
                Concatenated array of correction trial indicators from all sessions.
            pooled_lick_types (np.ndarray): 
                Concatenated array of lick type labels from all sessions.
            pooled_labels (np.ndarray): 
                Concatenated array of neuron labels from all sessions.
    Notes:
        - Assumes that the function `get_lick_response` is available and returns the expected outputs.
        - The function pads neural data so that all neurons from all sessions are represented in the pooled array.
        - Useful for downstream analyses that require all session data to be in a single, aligned format.
    """
    neu_seqs = []
    directions = []
    lick_types = []
    corrections = []

    neuron_counts = [labels.shape[0] for labels in labels_list]
    total_neurons = sum(neuron_counts)

    neuron_offset = 0  # for assigning each session's neurons into the correct padded space

    for i, (neural_trials, session_labels) in enumerate(zip(neural_trials_list, labels_list)):
        neu_seq, neu_time, direction, correction, lick_type = get_lick_response(
            neural_trials, l_frames, r_frames)

        n_trials, n_neurons, n_time = neu_seq.shape
        padded_neu_seq = np.zeros((n_trials, total_neurons, n_time))

        # Place this session’s neurons in the correct slice
        padded_neu_seq[:, neuron_offset:neuron_offset + n_neurons, :] = neu_seq
        neu_seqs.append(padded_neu_seq)

        directions.append(direction)
        corrections.append(correction)
        lick_types.append(lick_type)

        neuron_offset += n_neurons

    # Concatenate along trials
    pooled_neu_seq = np.concatenate(neu_seqs, axis=0)  # (total_trials, total_neurons, time)
    pooled_directions = np.concatenate(directions, axis=0)
    pooled_corrections = np.concatenate(corrections, axis=0)
    pooled_lick_types = np.concatenate(lick_types, axis=0)
    # Concatenate all labels into a single array for pooled data
    pooled_labels = np.concatenate(labels_list, axis=0)

    return pooled_neu_seq, neu_time, pooled_directions, pooled_corrections, pooled_lick_types, pooled_labels


def main(neu_seq, neu_time, direction, correction, lick_type, labels, save_path=None):
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
    if save_path is not None:
        figures_dir = os.path.join(save_path, 'Figures')
        if not os.path.exists(figures_dir):
            os.makedirs(figures_dir)
        save_path = os.path.join(figures_dir, 'neural_response_around_lick.pdf')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_licking_neural_response(neural_trials, labels, save_path=None, pooling=False):
    """
    Wrapper function to extract lick-aligned neural response data and plot neural activity.

    Parameters:
    neural_trials (np.ndarray or list of np.ndarray): Neural trial data for one or multiple sessions.
    labels (np.ndarray or list of np.ndarray): Neuron labels (-1: excitatory, 0: unsure, 1: inhibitory) for one or multiple sessions.
    save_path (str, optional): Directory to save the figure. If None, the figure is not saved.
    pooling (bool, optional): If True, pool data from multiple sessions (expects lists for neural_trials and labels).
    """
    l_frames, r_frames = 30, 30  # Number of frames before and after lick event

    if pooling:
        # Pool data from multiple sessions
        neu_seq, neu_time, direction, correction, lick_type, labels = pool_session_data(neural_trials, labels, l_frames, r_frames)
    else:
        neu_seq, neu_time, direction, correction, lick_type = get_lick_response(neural_trials, l_frames, r_frames)

    main(neu_seq, neu_time, direction, correction, lick_type, labels, save_path=save_path)
