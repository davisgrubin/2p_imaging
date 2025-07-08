import numpy as np
import os
from matplotlib import pyplot as plt
from matplotlib import gridspec
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import sem
from Modules.Alignment import get_perception_response
import warnings
warnings.filterwarnings("ignore")

def get_roi_label_color(labels, roi_id):
    """Get category, colors, and colormap based on neuron label."""
    colormap_dict = {
        -1: ('excitatory', 'grey', 'dodgerblue', 
             LinearSegmentedColormap.from_list('excitatory_cmap', ['white', 'dodgerblue', 'black'])),
        0: ('unsure', 'grey', 'mediumseagreen', 
            LinearSegmentedColormap.from_list('unsure_cmap', ['white', 'mediumseagreen', 'black'])),
        1: ('inhibitory', 'grey', 'hotpink', 
            LinearSegmentedColormap.from_list('inhibitory_cmap', ['white', 'hotpink', 'black']))
    }
    return colormap_dict.get(labels[roi_id], ('unsure', 'grey', 'mediumseagreen', 
                                             LinearSegmentedColormap.from_list('unsure_cmap', ['white', 'mediumseagreen', 'black'])))

def apply_colormap(data, cmap):
    """Normalize data and apply colormap."""
    if data.size > 0:
        vmin, vmax = np.nanmin(data), np.nanmax(data)
        normalized = (data - vmin) / (vmax - vmin + 1e-10)
        return cmap(normalized)
    return np.array([])

def adjust_layout_heatmap(ax):
    """Adjust heatmap layout by removing spines."""
    for spine in ax.spines.values():
        spine.set_visible(False)

def plot_gridspec_subplot(fig, gs, position, time, avg_data, sem_data, title, color='black', 
                        stim1_time=0, stim2_time=None, line_mode='all', stim_style='bar'):
    """Create a subplot with mean, SEM shading, and stimulus onset markers."""
    ax = fig.add_subplot(gs[position])
    ax.plot(time, avg_data, color=color)
    ax.fill_between(time, avg_data - sem_data, avg_data + sem_data, color=color, alpha=0.3)
    
    stim1_time = [float(stim1_time)] if isinstance(stim1_time, (int, float)) else stim1_time
    stim2_time = [float(stim2_time)] if isinstance(stim2_time, (int, float)) else stim2_time if stim2_time is not None else []
    
    if line_mode in ['all', 'reward', 'punish']:
        colors = {'all': 'gray', 'reward': 'green', 'punish': 'red'}
        color = colors[line_mode]
        label_prefix = {'all': 'Stim', 'reward': 'Reward', 'punish': 'Punish'}
        
        for i, t in enumerate(stim1_time):
            if t is not None and not np.isnan(t):
                label = f'{label_prefix[line_mode]} onset' if i == 0 else None
                if stim_style == 'bar':
                    ax.axvspan(t, t + 200, color=color, alpha=0.2, label=label)
                elif stim_style == 'line':
                    ax.axvline(t, color=color, linestyle='--', label=label)
        if line_mode == 'all':
            for i, t in enumerate(stim2_time):
                if t is not None and not np.isnan(t):
                    label = 'Stim2 onset' if i == 0 else None
                    if stim_style == 'bar':
                        ax.axvspan(t, t + 200, color=color, alpha=0.2, label=label)
                    elif stim_style == 'line':
                        ax.axvline(t, color=color, linestyle='--', label=label)
    
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('dF/F (mean +/- sem)')
    ax.set_title(title)
    if ax.get_legend_handles_labels()[0]:
        ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return ax

def plot_gridspec_superimposed(fig, gs, position, time, avg_data1, sem_data1, label1, color1, 
                             avg_data2, sem_data2, label2, color2, title, 
                             stim1_time=0, stim2_time=None, line_mode='all', stim_style='bar'):
    """Create a subplot with two superimposed traces."""
    ax = fig.add_subplot(gs[position])
    ax.plot(time, avg_data1, color=color1, label=label1)
    ax.fill_between(time, avg_data1 - sem_data1, avg_data1 + sem_data1, color=color1, alpha=0.3)
    ax.plot(time, avg_data2, color=color2, label=label2)
    ax.fill_between(time, avg_data2 - sem_data2, avg_data2 + sem_data2, color=color2, alpha=0.3)
    
    stim1_time = [float(stim1_time)] if isinstance(stim1_time, (int, float)) else stim1_time
    stim2_time = [float(stim2_time)] if isinstance(stim2_time, (int, float)) else stim2_time if stim2_time is not None else []
    
    if line_mode in ['all', 'reward', 'punish']:
        colors = {'all': 'gray', 'reward': 'green', 'punish': 'red'}
        color = colors[line_mode]
        label_prefix = {'all': 'Stim', 'reward': 'Reward', 'punish': 'Punish'}
        
        for i, t in enumerate(stim1_time):
            if t is not None and not np.isnan(t):
                label = f'{label_prefix[line_mode]} onset' if i == 0 else None
                if stim_style == 'bar':
                    ax.axvspan(t, t + 200, color=color, alpha=0.2, label=label)
                elif stim_style == 'line':
                    ax.axvline(t, color=color, linestyle='--', label=label)
        if line_mode == 'all':
            for i, t in enumerate(stim2_time):
                if t is not None and not np.isnan(t):
                    label = 'Stim2 onset' if i == 0 else None
                    if stim_style == 'bar':
                        ax.axvspan(t, t + 200, color=color, alpha=0.2, label=label)
                    elif stim_style == 'line':
                        ax.axvline(t, color=color, linestyle='--', label=label)
    
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('dF/F (mean +/- sem)')
    ax.set_title(title)
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return ax

def plot_gridspec_heatmap(fig, gs, position, time, data, labels, title, vmin=None, vmax=None, 
                         stim1_time=0, stim2_time=np.nan, line_mode='all'):
    """Create a heatmap subplot with label-specific colormaps."""
    ax = fig.add_subplot(gs[position])
    n_neurons, time_window = data.shape
    heatmap_rgb = np.zeros((n_neurons, time_window, 4))
    
    unique_labels = np.unique(labels)
    cmap_cache = {label: get_roi_label_color(labels, np.where(labels == label)[0][0])[3] for label in unique_labels}
    
    for i in range(n_neurons):
        heatmap_rgb[i] = apply_colormap(data[i], cmap_cache[labels[i]])
    
    im = ax.imshow(heatmap_rgb, interpolation='nearest', aspect='auto', extent=[time[0], time[-1], n_neurons, 0])
    
    if line_mode in ['all', 'reward', 'punish']:
        colors = {'all': 'red', 'reward': 'red', 'punish': 'purple'}
        labels_dict = {'all': 'Stim1 onset', 'reward': 'Stim1 onset', 'punish': 'Punish onset'}
        if not np.isnan(stim1_time):
            ax.axvline(stim1_time, color=colors[line_mode], linestyle='--', label=labels_dict[line_mode])
        if line_mode == 'all' and not np.isnan(stim2_time):
            ax.axvline(stim2_time, color='blue', linestyle='--', label='Stim2 onset')
    
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Neuron index')
    ax.set_title(title)
    if ax.get_legend_handles_labels()[0]:
        ax.legend()
    adjust_layout_heatmap(ax)
    # fig.colorbar(im, ax=ax, label='dF/F')
    return ax

def plot_sorted_heatmap(fig, gs, position, time, data, labels, title, vmin=None, vmax=None, 
                       stim1_time=0, stim2_time=np.nan, line_mode='all', sort_interval=None):
    """Plot sorted heatmap with label-specific colormaps."""
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
    sorted_labels = labels[sorted_indices]

    cmap = cm.get_cmap('inferno')
    heatmap_rgb = np.array([apply_colormap(row, cmap) for row in sorted_data])

    # heatmap_rgb = np.zeros((n_neurons, time_window, 4))
    # unique_labels = np.unique(labels)
    # cmap_cache = {label: get_roi_label_color(labels, np.where(labels == label)[0][0])[3] for label in unique_labels}

    # for i in range(n_neurons):
    #     heatmap_rgb[i] = apply_colormap(sorted_data[i], cmap_cache[sorted_labels[i]])

    im = ax.imshow(heatmap_rgb, interpolation='nearest', aspect='auto', extent=[time[0], time[-1], n_neurons, 0])

    if line_mode in ['all', 'reward', 'punish']:
        colors = {'all': 'red', 'reward': 'red', 'punish': 'purple'}
        labels_dict = {'all': 'Stim1 onset', 'reward': 'Stim1 onset', 'punish': 'Punish onset'}
        if not np.isnan(stim1_time):
            ax.axvline(stim1_time, color=colors[line_mode], linestyle='--', label=labels_dict[line_mode])
        if line_mode == 'all' and not np.isnan(stim2_time):
            ax.axvline(stim2_time, color='blue', linestyle='--', label='Stim2 onset')

    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Neuron index (sorted)')
    ax.set_title(title)
    if ax.get_legend_handles_labels()[0]:
        ax.legend()
    adjust_layout_heatmap(ax)
    # fig.colorbar(im, ax=ax, label='dF/F')
    return ax

def compute_trial_masks(trial_type, decision):
    """Compute masks for different trial types and decisions."""
    return {
        'short_left': (trial_type == 0) & (decision == 0),
        'short_right': (trial_type == 0) & (decision == 1),
        'long_left': (trial_type == 1) & (decision == 0),
        'long_right': (trial_type == 1) & (decision == 1)
    }

def calculate_metrics(neu_seq, mask, labels):
    """Calculate average, SEM, and heatmap data for given mask."""
    exc_mask = labels == -1
    inh_mask = labels == 1
    return {
        'avg_all': np.nanmean(neu_seq[mask], axis=(0, 1)),
        'sem_all': sem(neu_seq[mask], axis=(0, 1), nan_policy='omit'),
        'avg_exc': np.nanmean(neu_seq[mask][:, exc_mask,:], axis=(0, 1)),
        'sem_exc': sem(neu_seq[mask][:, exc_mask, :], axis=(0, 1), nan_policy='omit'),
        'avg_inh': np.nanmean(neu_seq[mask][:, inh_mask, :], axis=(0, 1)),
        'sem_inh': sem(neu_seq[mask][:, inh_mask, :], axis=(0, 1), nan_policy='omit'),
        'heat': np.nanmean(neu_seq[mask], axis=0)
    }

def set_uniform_ylim(axes_list):
    """Set uniform y-axis limits across related subplots."""
    ylims = [ax.get_ylim() for ax in axes_list]
    ymin = min(ylim[0] for ylim in ylims)
    ymax = max(ylim[1] for ylim in ylims)
    for ax in axes_list:
        ax.set_ylim(ymin, ymax)

def pool_session_data(neural_trials_list, labels_list, state, l_frames, r_frames, indices):
    """
    Pool data from multiple sessions.
    
    Returns:
    tuple: Pooled neu_seq (n_trials_total, n_neurons_total, time), neu_time, trial_type, isi, decision, labels
    """
    neu_seqs = []
    trial_types = []
    block_types = []
    isis = []
    decisions = []
    all_labels = []
    all_outcomes = []

    neuron_counts = [labels.shape[0] for labels in labels_list]
    total_neurons = sum(neuron_counts)

    neuron_offset = 0  # for assigning each session's neurons into the correct padded space

    for i, (neural_trials, session_labels) in enumerate(zip(neural_trials_list, labels_list)):
        neu_seq, neu_time, stim_seq, stim_value, stim_time, led_value, trial_type, block_type, isi, decision, outcome = get_perception_response(
            neural_trials, state, l_frames, r_frames, indices=indices)

        n_trials, n_neurons, n_time = neu_seq.shape
        padded_neu_seq = np.zeros((n_trials, total_neurons, n_time))

        # Place this session’s neurons in the correct slice
        padded_neu_seq[:, neuron_offset:neuron_offset + n_neurons, :] = neu_seq
        neu_seqs.append(padded_neu_seq)

        trial_types.append(trial_type)
        block_types.append(block_type)
        isis.append(isi)
        decisions.append(decision)
        all_labels.append(session_labels)
        all_outcomes.append(outcome)

        neuron_offset += n_neurons

    # Concatenate along trials
    pooled_neu_seq = np.concatenate(neu_seqs, axis=0)  # (total_trials, total_neurons, time)
    pooled_trial_type = np.concatenate(trial_types, axis=0)
    pooled_block_type = np.concatenate(block_types, axis=0)
    pooled_isi = np.concatenate(isis, axis=0)
    pooled_decision = np.concatenate(decisions, axis=0)
    pooled_labels = np.concatenate(all_labels, axis=0)
    pooled_outcomes = np.concatenate(all_outcomes, axis=0)

    return pooled_neu_seq, neu_time, pooled_trial_type, pooled_block_type, pooled_isi, pooled_decision, pooled_labels, pooled_outcomes

def main(neural_trials_list, labels_list, save_path=None):
    """
    Main function to generate neural response plots for pooled session data.
    
    Parameters:
    neural_trials_list (list): List of neural_trials for each session
    labels_list (list): List of neuron labels for each session (-1 for excitatory, 0 for unsure, 1 for inhibitory)
    """
    fig = plt.figure(figsize=(40, 180))
    gs = gridspec.GridSpec(52, 6, wspace=0.4, hspace=0.4)
    
    # First stimulus onset
    l_frames, r_frames = 90, 90
    neu_seq, neu_time, trial_type, block_type, isi, decision, labels, outcomes = pool_session_data(
        neural_trials_list, labels_list, 'stim_seq', l_frames, r_frames, indices=0)
    
    short_trials = trial_type == 0
    long_trials = trial_type == 1
    
    metrics_short = calculate_metrics(neu_seq, short_trials, labels)
    metrics_long = calculate_metrics(neu_seq, long_trials, labels)
    
    sec_stim_short = np.unique(isi[short_trials]) + 200
    sec_stim_time_short = np.unique(neu_time[np.searchsorted(neu_time, sec_stim_short)])
    sec_stim_long = np.unique(isi[long_trials]) + 200
    sec_stim_time_long = np.unique(neu_time[np.searchsorted(neu_time, sec_stim_long)])
    
    axes_group1 = [
        plot_gridspec_subplot(fig, gs, (0, 0), neu_time, metrics_short['avg_all'], metrics_short['sem_all'],
                            title='Average dF/F all neurons around first stimulus onset\nshort trials', color='black',
                            stim1_time=0, stim2_time=sec_stim_time_short, line_mode='all'),
        plot_gridspec_subplot(fig, gs, (0, 1), neu_time, metrics_short['avg_exc'], metrics_short['sem_exc'],
                            title='Average dF/F excitatory neurons around first stimulus onset\nshort trials ', color='blue',
                            stim1_time=0, stim2_time=sec_stim_time_short, line_mode='all'),
        plot_gridspec_subplot(fig, gs, (0, 2), neu_time, metrics_short['avg_inh'], metrics_short['sem_inh'],
                            title='Average dF/F inhibitory neurons around first stimulus onset\nshort trials', color='red',
                            stim1_time=0, stim2_time=sec_stim_time_short, line_mode='all'),
        plot_gridspec_superimposed(fig, gs, (0, 3), neu_time,
                                 metrics_short['avg_exc'], metrics_short['sem_exc'], 'Excitatory', 'blue',
                                 metrics_short['avg_inh'], metrics_short['sem_inh'], 'Inhibitory', 'red',
                                 title='Superimposed dF/F excitatory and inhibitory neurons around first stimulus onset\nshort trials ',
                                 stim1_time=0, stim2_time=sec_stim_time_short, line_mode='all')
    ]
    plot_gridspec_heatmap(fig, gs, (slice(0,2), 4), neu_time, metrics_short['heat'],
                         title='All neurons heatmap around first stimulus onset\nshort trials',
                         labels=labels, stim1_time=0, stim2_time=np.nanmean(sec_stim_time_short), line_mode='all')
    plot_sorted_heatmap(fig, gs, (slice(0,2), 5), neu_time, metrics_short['heat'], labels=labels,
                       title='Sorted heatmap of all neurons around first stimulus onset\nshort trials',
                       stim1_time=0, stim2_time=np.nanmean(sec_stim_time_short), line_mode='all', sort_interval=(-400, 400))
    
    axes_group2 = [
        plot_gridspec_subplot(fig, gs, (2, 0), neu_time, metrics_long['avg_all'], metrics_long['sem_all'],
                            title='Average dF/F all neurons around first stimulus onset\nlong trials', color='black',
                            stim1_time=0, stim2_time=sec_stim_time_long, line_mode='all'),
        plot_gridspec_subplot(fig, gs, (2, 1), neu_time, metrics_long['avg_exc'], metrics_long['sem_exc'],
                            title='Average dF/F excitatory neurons around first stimulus onset\nlong trials', color='blue',
                            stim1_time=0, stim2_time=sec_stim_time_long, line_mode='all'),
        plot_gridspec_subplot(fig, gs, (2, 2), neu_time, metrics_long['avg_inh'], metrics_long['sem_inh'],
                            title='Average dF/F inhibitory neurons around first stimulus onset\nlong trials', color='red',
                            stim1_time=0, stim2_time=sec_stim_time_long, line_mode='all'),
        plot_gridspec_superimposed(fig, gs, (2, 3), neu_time,
                                 metrics_long['avg_exc'], metrics_long['sem_exc'], 'Excitatory', 'blue',
                                 metrics_long['avg_inh'], metrics_long['sem_inh'], 'Inhibitory', 'red',
                                 title='Superimposed dF/F excitatory and inhibitory neurons around first stimulus onset\nlong trials',
                                 stim1_time=0, stim2_time=sec_stim_time_long, line_mode='all')
    ]
    plot_gridspec_heatmap(fig, gs, (slice(2,4), 4), neu_time, metrics_long['heat'],
                         title='All neurons heatmap around first stimulus onset\nlong trials',
                         labels=labels, stim1_time=0, stim2_time=np.nanmean(sec_stim_time_long), line_mode='all')
    plot_sorted_heatmap(fig, gs, (slice(2,4), 5), neu_time, metrics_long['heat'], labels=labels,
                       title='Sorted heatmap of all neurons around first stimulus onset\nlong trials',
                       stim1_time=0, stim2_time=np.nanmean(sec_stim_time_long), line_mode='all', sort_interval=(-400, 400))
    
    # Second stimulus onset
    neu_seq, neu_time, trial_type, block_type, isi, decision, labels, outcomes = pool_session_data(
        neural_trials_list, labels_list, 'stim_seq', l_frames, r_frames, indices=2)
    
    metrics_short = calculate_metrics(neu_seq, short_trials, labels)
    metrics_long = calculate_metrics(neu_seq, long_trials, labels)
    
    first_stim_short = np.unique(isi[short_trials]) + 200
    first_stim_time_short = np.unique(neu_time[np.searchsorted(neu_time, -first_stim_short)])
    first_stim_long = np.unique(isi[long_trials]) + 200
    first_stim_time_long = np.unique(neu_time[np.searchsorted(neu_time, -first_stim_long)])
    
    axes_group3 = [
        plot_gridspec_subplot(fig, gs, (4, 0), neu_time, metrics_short['avg_all'], metrics_short['sem_all'],
                            title='Average dF/F all neurons around second stimulus onset\nshort trials', color='black',
                            stim1_time=first_stim_time_short, stim2_time=0, line_mode='all'),
        plot_gridspec_subplot(fig, gs, (4, 1), neu_time, metrics_short['avg_exc'], metrics_short['sem_exc'],
                            title='Average dF/F excitatory neurons around second stimulus onset\nshort trials', color='blue',
                            stim1_time=first_stim_time_short, stim2_time=0, line_mode='all'),
        plot_gridspec_subplot(fig, gs, (4, 2), neu_time, metrics_short['avg_inh'], metrics_short['sem_inh'],
                            title='Average dF/F inhibitory neurons around second stimulus onset\nshort trials', color='red',
                            stim1_time=first_stim_time_short, stim2_time=0, line_mode='all'),
        plot_gridspec_superimposed(fig, gs, (4, 3), neu_time,
                                 metrics_short['avg_exc'], metrics_short['sem_exc'], 'Excitatory', 'blue',
                                 metrics_short['avg_inh'], metrics_short['sem_inh'], 'Inhibitory', 'red',
                                 title='Superimposed dF/F excitatory and inhibitory neurons around second stimulus onset\nshort trials',
                                 stim1_time=first_stim_time_short, stim2_time=0, line_mode='all')
    ]
    plot_gridspec_heatmap(fig, gs, (slice(4,6), 4), neu_time, metrics_short['heat'],
                         title='All neurons heatmap around second stimulus onset\nshort trials',
                         labels=labels, stim1_time=np.nanmean(first_stim_time_short), stim2_time=0, line_mode='all')
    plot_sorted_heatmap(fig, gs, (slice(4,6), 5), neu_time, metrics_short['heat'], labels=labels,
                       title='Sorted heatmap of all neurons around second stimulus onset\nshort trials',
                       stim1_time=np.nanmean(first_stim_time_short), stim2_time=0, line_mode='all')
    
    axes_group4 = [
        plot_gridspec_subplot(fig, gs, (6, 0), neu_time, metrics_long['avg_all'], metrics_long['sem_all'],
                            title='Average dF/F all neurons around second stimulus onset\nlong trials', color='black',
                            stim1_time=first_stim_time_long, stim2_time=0, line_mode='all'),
        plot_gridspec_subplot(fig, gs, (6, 1), neu_time, metrics_long['avg_exc'], metrics_long['sem_exc'],
                            title='Average dF/F excitatory neurons around second stimulus onset\nlong trials', color='blue',
                            stim1_time=first_stim_time_long, stim2_time=0, line_mode='all'),
        plot_gridspec_subplot(fig, gs, (6, 2), neu_time, metrics_long['avg_inh'], metrics_long['sem_inh'],
                            title='Average dF/F inhibitory neurons around second stimulus onset\nlong trials', color='red',
                            stim1_time=first_stim_time_long, stim2_time=0, line_mode='all'),
        plot_gridspec_superimposed(fig, gs, (6, 3), neu_time,
                                 metrics_long['avg_exc'], metrics_long['sem_exc'], 'Excitatory', 'blue',
                                 metrics_long['avg_inh'], metrics_long['sem_inh'], 'Inhibitory', 'red',
                                 title='Superimposed dF/F excitatory and inhibitory neurons around second stimulus onset\nlong trials',
                                 stim1_time=first_stim_time_long, stim2_time=0, line_mode='all')
    ]
    plot_gridspec_heatmap(fig, gs, (slice(6,8), 4), neu_time, metrics_long['heat'],
                         title='All neurons heatmap around second stimulus onset\nlong trials',
                         labels=labels, stim1_time=np.nanmean(first_stim_time_long), stim2_time=0, line_mode='all')
    plot_sorted_heatmap(fig, gs, (slice(6,8), 5), neu_time, metrics_long['heat'], labels=labels,
                       title='Sorted heatmap of all neurons around second stimulus onset\nlong trials',
                       stim1_time=np.nanmean(first_stim_time_long), stim2_time=0, line_mode='all')
    
    # Reward and punish trials
    l_frames, r_frames = 60, 60
    conditions = [
        ('reward', 'state_reward', 8),
        ('punish', 'state_punish', 16)
    ]
    
    for condition, state, row_start in conditions:
        neu_seq, neu_time, trial_type, block_type, isi, decision, labels, outcomes = pool_session_data(
            neural_trials_list, labels_list, state, l_frames, r_frames, indices=0)
        masks = compute_trial_masks(trial_type, decision)
        
        row_starts = {'short_left': row_start, 'short_right': row_start+2, 'long_left': row_start+4, 'long_right': row_start+6}
        
        for trial_key, row in row_starts.items():
            metrics = calculate_metrics(neu_seq, masks[trial_key], labels)
            axes_group = [
                plot_gridspec_subplot(fig, gs, (row, 0), neu_time, metrics['avg_all'], metrics['sem_all'],
                                    title=f'Average dF/F all neurons around {condition}\n{trial_key.replace("_", " ")}', 
                                    color='black', line_mode=condition, stim_style='line'),
                plot_gridspec_subplot(fig, gs, (row, 1), neu_time, metrics['avg_exc'], metrics['sem_exc'],
                                    title=f'Average dF/F excitatory neurons around {condition}\n{trial_key.replace("_", " ")}', 
                                    color='blue', line_mode=condition, stim_style='line'),
                plot_gridspec_subplot(fig, gs, (row, 2), neu_time, metrics['avg_inh'], metrics['sem_inh'],
                                    title=f'Average dF/F inhibitory neurons around {condition}\n{trial_key.replace("_", " ")} ', 
                                    color='red', line_mode=condition, stim_style='line'),
                plot_gridspec_superimposed(fig, gs, (row, 3), neu_time,
                                         metrics['avg_exc'], metrics['sem_exc'], 'Excitatory', 'blue',
                                         metrics['avg_inh'], metrics['sem_inh'], 'Inhibitory', 'red',
                                         title=f'Superimposed dF/F excitatory and inhibitory neurons around {condition}\n{trial_key.replace("_", " ")}',
                                         line_mode=condition, stim_style='line')
            ]
            plot_gridspec_heatmap(fig, gs, (slice(row, row+2), 4), neu_time, metrics['heat'],
                                 title=f'All neurons heatmap around {condition}\n{trial_key.replace("_", " ")} (pooled)',
                                 labels=labels, line_mode=condition)
            plot_sorted_heatmap(fig, gs, (slice(row, row+2), 5), neu_time, metrics['heat'], labels=labels,
                              title=f'Sorted heatmap of all neurons around {condition}\n{trial_key.replace("_", " ")}')
            set_uniform_ylim(axes_group)
    
    # Servo_In trials
    neu_seq, neu_time, trial_type, block_type, isi, decision, labels, outcomes = pool_session_data(
        neural_trials_list, labels_list, 'state_window_choice', l_frames, r_frames, indices=0)
    
    metrics_short = calculate_metrics(neu_seq, trial_type == 0, labels)
    metrics_long = calculate_metrics(neu_seq, trial_type == 1, labels)
    
    axes_group5 = [
        plot_gridspec_subplot(fig, gs, (24, 0), neu_time, metrics_short['avg_all'], metrics_short['sem_all'],
                            title='Average dF/F all neurons around Servo_In (short trials)', color='black', stim_style='line'),
        plot_gridspec_subplot(fig, gs, (24, 1), neu_time, metrics_short['avg_exc'], metrics_short['sem_exc'],
                            title='Average dF/F excitatory neurons around Servo_In (short trials)', color='blue', stim_style='line'),
        plot_gridspec_subplot(fig, gs, (24, 2), neu_time, metrics_short['avg_inh'], metrics_short['sem_inh'],
                            title='Average dF/F inhibitory neurons around Servo_In (short trials)', color='red', stim_style='line'),
        plot_gridspec_superimposed(fig, gs, (24, 3), neu_time,
                                 metrics_short['avg_exc'], metrics_short['sem_exc'], 'Excitatory', 'blue',
                                 metrics_short['avg_inh'], metrics_short['sem_inh'], 'Inhibitory', 'red',
                                 title='Superimposed dF/F excitatory and inhibitory neurons around Servo_In (short trials)', stim_style='line')
    ]
    plot_gridspec_heatmap(fig, gs, (slice(24, 26), 4), neu_time, metrics_short['heat'],
                         title='All neurons heatmap around Servo_In (short trials)', labels=labels)
    plot_sorted_heatmap(fig, gs, (slice(24, 26), 5), neu_time, metrics_short['heat'], labels=labels,
                       title='Sorted heatmap of all neurons around Servo_In (short trials)')
    
    axes_group6 = [
        plot_gridspec_subplot(fig, gs, (26, 0), neu_time, metrics_long['avg_all'], metrics_long['sem_all'],
                            title='Average dF/F all neurons around Servo_In (long trials)', color='black', stim_style='line'),
        plot_gridspec_subplot(fig, gs, (26, 1), neu_time, metrics_long['avg_exc'], metrics_long['sem_exc'],
                            title='Average dF/F excitatory neurons around Servo_In (long trials)', color='blue', stim_style='line'),
        plot_gridspec_subplot(fig, gs, (26, 2), neu_time, metrics_long['avg_inh'], metrics_long['sem_inh'],
                            title='Average dF/F inhibitory neurons around Servo_In (long trials)', color='red', stim_style='line'),
        plot_gridspec_superimposed(fig, gs, (26, 3), neu_time,
                                 metrics_long['avg_exc'], metrics_long['sem_exc'], 'Excitatory', 'blue',
                                 metrics_long['avg_inh'], metrics_long['sem_inh'], 'Inhibitory', 'red',
                                 title='Superimposed dF/F excitatory and inhibitory neurons around Servo_In (long trials)', stim_style='line')
    ]
    plot_gridspec_heatmap(fig, gs, (slice(26, 28), 4), neu_time, metrics_long['heat'],
                         title='All neurons heatmap around Servo_In (long trials)', labels=labels)
    plot_sorted_heatmap(fig, gs, (slice(26, 28), 5), neu_time, metrics_long['heat'], labels=labels,
                       title='Sorted heatmap of all neurons around Servo_In (long trials')


    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # Block-based analysis - aligned on first stimulus
    l_frames, r_frames = 60, 120
    neu_seq, neu_time, trial_type, block_type, isi, decision, labels, outcomes = pool_session_data(
        neural_trials_list, labels_list, 'stim_seq', l_frames, r_frames, indices=0)
    
    # Define block and trial masks
    short_block_mask = block_type == 1
    long_block_mask = block_type == 2
    short_trial_mask = trial_type == 0
    long_trial_mask = trial_type == 1
    
    # Reward and punish masks
    reward_mask = outcomes == 'reward'  # Assuming rewarded trials have decision == 'reward'
    punish_mask = outcomes == 'punish'  # Assuming punished trials have decision == 'punish'
    
    # Calculate second stimulus timing
    sec_stim_short = np.unique(isi[short_trial_mask]) + 200
    sec_stim_time_short = np.unique(neu_time[np.searchsorted(neu_time, sec_stim_short)])
    sec_stim_long = np.unique(isi[long_trial_mask]) + 200
    sec_stim_time_long = np.unique(neu_time[np.searchsorted(neu_time, sec_stim_long)])
    
    # Short majority block analysis (rows 28-33)
    conditions_short_block = [
        ('all short trials vs all long trials in short block', 
         short_block_mask & short_trial_mask, short_block_mask & long_trial_mask, 28),
        ('rewarded short trials vs rewarded long trials in short block', 
         short_block_mask & short_trial_mask & reward_mask, short_block_mask & long_trial_mask & reward_mask, 30),
        ('punished short trials vs punished long trials in short block', 
         short_block_mask & short_trial_mask & punish_mask, short_block_mask & long_trial_mask & punish_mask, 32)
    ]
    
    for title, mask1, mask2, row in conditions_short_block:
        metrics1 = calculate_metrics(neu_seq, mask1, labels)
        metrics2 = calculate_metrics(neu_seq, mask2, labels)
        
        # Use average of both for display
        # Compute average second stimulus time for use in plot_gridspec_superimposed
        if len(sec_stim_time_short) > 0 and len(sec_stim_time_long) > 0:
            avg_sec_stim_time = np.concatenate([sec_stim_time_short, sec_stim_time_long])
        elif len(sec_stim_time_short) > 0:
            avg_sec_stim_time = sec_stim_time_short
        elif len(sec_stim_time_long) > 0:
            avg_sec_stim_time = sec_stim_time_long
        else:
            avg_sec_stim_time = 0.0
        
        axes_group = [
            plot_gridspec_superimposed(fig, gs, (row, 0), neu_time,
                                     metrics1['avg_all'], metrics1['sem_all'], 'Short trials' if 'short vs' in title else 'Short', 'orange',
                                     metrics2['avg_all'], metrics2['sem_all'], 'Long trials' if 'short vs' in title else 'Long', 'purple',
                                     title=f'All neurons: {title}',
                                     stim1_time=0, stim2_time=avg_sec_stim_time, line_mode='all'),
            plot_gridspec_superimposed(fig, gs, (row, 1), neu_time,
                                     metrics1['avg_exc'], metrics1['sem_exc'], 'Short trials' if 'short vs' in title else 'Short', 'orange',
                                     metrics2['avg_exc'], metrics2['sem_exc'], 'Long trials' if 'short vs' in title else 'Long', 'purple',
                                     title=f'Excitatory neurons: {title}',
                                     stim1_time=0, stim2_time=avg_sec_stim_time, line_mode='all'),
            plot_gridspec_superimposed(fig, gs, (row, 2), neu_time,
                                     metrics1['avg_inh'], metrics1['sem_inh'], 'Short trials' if 'short vs' in title else 'Short', 'orange',
                                     metrics2['avg_inh'], metrics2['sem_inh'], 'Long trials' if 'short vs' in title else 'Long', 'purple',
                                     title=f'Inhibitory neurons: {title}',
                                     stim1_time=0, stim2_time=avg_sec_stim_time, line_mode='all'),
            plot_gridspec_superimposed(fig, gs, (row, 3), neu_time,
                                     metrics1['avg_exc'], metrics1['sem_exc'], 'Exc - Short' if 'short vs' in title else 'Exc - Short', 'blue',
                                     metrics1['avg_inh'], metrics1['sem_inh'], 'Inh - Short' if 'short vs' in title else 'Inh - Short', 'red',
                                     title=f'Exc vs Inh: {title.split(" vs ")[0]}',
                                     stim1_time=0, stim2_time=avg_sec_stim_time, line_mode='all')
        ]
        
        # Heatmaps - combine both conditions for display
        combined_heat = np.vstack([metrics1['heat'], metrics2['heat']])
        combined_labels = np.hstack([labels, labels])
        
        plot_gridspec_heatmap(fig, gs, (slice(row, row+2), 4), neu_time, combined_heat,
                             title=f'Combined heatmap: {title}',
                             labels=combined_labels, stim1_time=0, stim2_time=np.nanmean(avg_sec_stim_time), line_mode='all')
        plot_sorted_heatmap(fig, gs, (slice(row, row+2), 5), neu_time, combined_heat, labels=combined_labels,
                           title=f'Sorted combined heatmap: {title}',
                           stim1_time=0, stim2_time=np.nanmean(avg_sec_stim_time), line_mode='all', sort_interval=(-400, 400))


    # Long majority block analysis (rows 31-33)
    conditions_long_block = [
        ('all short trials vs all long trials in long block', 
         long_block_mask & short_trial_mask, long_block_mask & long_trial_mask, 34),
        ('rewarded short trials vs rewarded long trials in long block', 
         long_block_mask & short_trial_mask & reward_mask, long_block_mask & long_trial_mask & reward_mask, 36),
        ('punished short trials vs punished long trials in long block', 
         long_block_mask & short_trial_mask & punish_mask, long_block_mask & long_trial_mask & punish_mask, 38)
    ]
    
    for title, mask1, mask2, row in conditions_long_block:
        metrics1 = calculate_metrics(neu_seq, mask1, labels)
        metrics2 = calculate_metrics(neu_seq, mask2, labels)
        
        # Determine appropriate second stimulus timing
        if len(sec_stim_time_short) > 0 and len(sec_stim_time_long) > 0:
            avg_sec_stim_time = np.concatenate([sec_stim_time_short, sec_stim_time_long])
        elif len(sec_stim_time_short) > 0:
            avg_sec_stim_time = sec_stim_time_short
        elif len(sec_stim_time_long) > 0:
            avg_sec_stim_time = sec_stim_time_long
        else:
            avg_sec_stim_time = 0.0
        
        axes_group = [
            plot_gridspec_superimposed(fig, gs, (row, 0), neu_time,
                                     metrics1['avg_all'], metrics1['sem_all'], 'Short trials' if 'short vs' in title else 'Short', 'orange',
                                     metrics2['avg_all'], metrics2['sem_all'], 'Long trials' if 'short vs' in title else 'Long', 'purple',
                                     title=f'All neurons: {title}',
                                     stim1_time=0, stim2_time=avg_sec_stim_time, line_mode='all'),
            plot_gridspec_superimposed(fig, gs, (row, 1), neu_time,
                                     metrics1['avg_exc'], metrics1['sem_exc'], 'Short trials' if 'short vs' in title else 'Short', 'orange',
                                     metrics2['avg_exc'], metrics2['sem_exc'], 'Long trials' if 'short vs' in title else 'Long', 'purple',
                                     title=f'Excitatory neurons: {title}',
                                     stim1_time=0, stim2_time=avg_sec_stim_time, line_mode='all'),
            plot_gridspec_superimposed(fig, gs, (row, 2), neu_time,
                                     metrics1['avg_inh'], metrics1['sem_inh'], 'Short trials' if 'short vs' in title else 'Short', 'orange',
                                     metrics2['avg_inh'], metrics2['sem_inh'], 'Long trials' if 'short vs' in title else 'Long', 'purple',
                                     title=f'Inhibitory neurons: {title}',
                                     stim1_time=0, stim2_time=avg_sec_stim_time, line_mode='all'),
            plot_gridspec_superimposed(fig, gs, (row, 3), neu_time,
                                     metrics1['avg_exc'], metrics1['sem_exc'], 'Exc - Short' if 'short vs' in title else 'Exc - Short', 'blue',
                                     metrics1['avg_inh'], metrics1['sem_inh'], 'Inh - Short' if 'short vs' in title else 'Inh - Short', 'red',
                                     title=f'Exc vs Inh: {title.split(" vs ")[0]}',
                                     stim1_time=0, stim2_time=avg_sec_stim_time, line_mode='all')
        ]
        
        # Heatmaps - combine both conditions for display
        combined_heat = np.vstack([metrics1['heat'], metrics2['heat']])
        combined_labels = np.hstack([labels, labels])
        
        plot_gridspec_heatmap(fig, gs, (slice(row, row+2), 4), neu_time, combined_heat,
                             title=f'Combined heatmap: {title}',
                             labels=combined_labels, stim1_time=0, stim2_time=np.nanmean(avg_sec_stim_time), line_mode='all')
        plot_sorted_heatmap(fig, gs, (slice(row, row+2), 5), neu_time, combined_heat, labels=combined_labels,
                           title=f'Sorted combined heatmap: {title}',
                           stim1_time=0, stim2_time=np.nanmean(avg_sec_stim_time), line_mode='all', sort_interval=(-400, 400))
    
    # Block-based analysis - aligned on second stimulus (rows 34-39)
    neu_seq, neu_time, trial_type, block_type, isi, decision, labels, outcomes = pool_session_data(
        neural_trials_list, labels_list, 'stim_seq', l_frames, r_frames, indices=2)
    
    # Calculate first stimulus timing (negative relative to second stimulus)
    first_stim_short = np.unique(isi[short_trial_mask]) + 200
    first_stim_time_short = np.unique(neu_time[np.searchsorted(neu_time, -first_stim_short)])
    first_stim_long = np.unique(isi[long_trial_mask]) + 200
    first_stim_time_long = np.unique(neu_time[np.searchsorted(neu_time, -first_stim_long)])
    
    # Short majority block analysis - aligned on second stimulus (rows 34-36)
    for title, mask1, mask2, row in [
        ('all short trials vs all long trials in short block', 
         short_block_mask & short_trial_mask, short_block_mask & long_trial_mask, 40),
        ('rewarded short trials vs rewarded long trials in short block', 
         short_block_mask & short_trial_mask & reward_mask, short_block_mask & long_trial_mask & reward_mask, 42),
        ('punished short trials vs punished long trials in short block', 
         short_block_mask & short_trial_mask & punish_mask, short_block_mask & long_trial_mask & punish_mask, 44)
    ]:
        metrics1 = calculate_metrics(neu_seq, mask1, labels)
        metrics2 = calculate_metrics(neu_seq, mask2, labels)
        
        # Determine appropriate first stimulus timing
        if len(first_stim_time_short) > 0 and len(first_stim_time_long) > 0:
            avg_first_stim_time = np.concatenate([first_stim_time_short, first_stim_time_long])
        elif len(first_stim_time_short) > 0:
            avg_first_stim_time = first_stim_time_short
        elif len(first_stim_time_long) > 0:
            avg_first_stim_time = first_stim_time_long
        else:
            avg_first_stim_time = 0.0
        
        
        axes_group = [
            plot_gridspec_superimposed(fig, gs, (row, 0), neu_time,
                                     metrics1['avg_all'], metrics1['sem_all'], 'Short trials' if 'short vs' in title else 'Short', 'orange',
                                     metrics2['avg_all'], metrics2['sem_all'], 'Long trials' if 'short vs' in title else 'Long', 'purple',
                                     title=f'All neurons: {title} (2nd stim aligned)',
                                     stim1_time=avg_first_stim_time, stim2_time=0, line_mode='all'),
            plot_gridspec_superimposed(fig, gs, (row, 1), neu_time,
                                     metrics1['avg_exc'], metrics1['sem_exc'], 'Short trials' if 'short vs' in title else 'Short', 'orange',
                                     metrics2['avg_exc'], metrics2['sem_exc'], 'Long trials' if 'short vs' in title else 'Long', 'purple',
                                     title=f'Excitatory neurons: {title} (2nd stim aligned)',
                                     stim1_time=avg_first_stim_time, stim2_time=0, line_mode='all'),
            plot_gridspec_superimposed(fig, gs, (row, 2), neu_time,
                                     metrics1['avg_inh'], metrics1['sem_inh'], 'Short trials' if 'short vs' in title else 'Short', 'orange',
                                     metrics2['avg_inh'], metrics2['sem_inh'], 'Long trials' if 'short vs' in title else 'Long', 'purple',
                                     title=f'Inhibitory neurons: {title} (2nd stim aligned)',
                                     stim1_time=avg_first_stim_time, stim2_time=0, line_mode='all'),
            plot_gridspec_superimposed(fig, gs, (row, 3), neu_time,
                                     metrics1['avg_exc'], metrics1['sem_exc'], 'Exc - Short' if 'short vs' in title else 'Exc - Short', 'blue',
                                     metrics1['avg_inh'], metrics1['sem_inh'], 'Inh - Short' if 'short vs' in title else 'Inh - Short', 'red',
                                     title=f'Exc vs Inh: {title.split(" vs ")[0]} (2nd stim aligned)',
                                     stim1_time=avg_first_stim_time, stim2_time=0, line_mode='all')
        ]
        
        # Heatmaps - combine both conditions for display
        combined_heat = np.vstack([metrics1['heat'], metrics2['heat']])
        combined_labels = np.hstack([labels, labels])
        
        plot_gridspec_heatmap(fig, gs, (slice(row, row+2), 4), neu_time, combined_heat,
                             title=f'Combined heatmap: {title} (2nd stim aligned)',
                             labels=combined_labels, stim1_time=np.nanmean(avg_first_stim_time), stim2_time=0, line_mode='all')
        plot_sorted_heatmap(fig, gs, (slice(row, row+2), 5), neu_time, combined_heat, labels=combined_labels,
                           title=f'Sorted combined heatmap: {title} (2nd stim aligned)',
                           stim1_time=np.nanmean(avg_first_stim_time), stim2_time=0, line_mode='all')
        
    
    # Long majority block analysis - aligned on second stimulus (rows 37-39)
    for title, mask1, mask2, row in [
        ('all short trials vs all long trials in long block', 
         long_block_mask & short_trial_mask, long_block_mask & long_trial_mask, 46),
        ('rewarded short trials vs rewarded long trials in long block', 
         long_block_mask & short_trial_mask & reward_mask, long_block_mask & long_trial_mask & reward_mask, 48),
        ('punished short trials vs punished long trials in long block', 
         long_block_mask & short_trial_mask & punish_mask, long_block_mask & long_trial_mask & punish_mask, 50)
    ]:
        metrics1 = calculate_metrics(neu_seq, mask1, labels)
        metrics2 = calculate_metrics(neu_seq, mask2, labels)
        
        axes_group = [
            plot_gridspec_superimposed(fig, gs, (row, 0), neu_time,
                                     metrics1['avg_all'], metrics1['sem_all'], 'Short trials' if 'short vs' in title else 'Short', 'orange',
                                     metrics2['avg_all'], metrics2['sem_all'], 'Long trials' if 'short vs' in title else 'Long', 'purple',
                                     title=f'All neurons: {title} (2nd stim aligned)',
                                     stim1_time=avg_first_stim_time, stim2_time=0, line_mode='all'),
            plot_gridspec_superimposed(fig, gs, (row, 1), neu_time,
                                     metrics1['avg_exc'], metrics1['sem_exc'], 'Short trials' if 'short vs' in title else 'Short', 'orange',
                                     metrics2['avg_exc'], metrics2['sem_exc'], 'Long trials' if 'short vs' in title else 'Long', 'purple',
                                     title=f'Excitatory neurons: {title} (2nd stim aligned)',
                                     stim1_time=avg_first_stim_time, stim2_time=0, line_mode='all'),
            plot_gridspec_superimposed(fig, gs, (row, 2), neu_time,
                                     metrics1['avg_inh'], metrics1['sem_inh'], 'Short trials' if 'short vs' in title else 'Short', 'orange',
                                     metrics2['avg_inh'], metrics2['sem_inh'], 'Long trials' if 'short vs' in title else 'Long', 'purple',
                                     title=f'Inhibitory neurons: {title} (2nd stim aligned)',
                                     stim1_time=avg_first_stim_time, stim2_time=0, line_mode='all'),
            plot_gridspec_superimposed(fig, gs, (row, 3), neu_time,
                                     metrics1['avg_exc'], metrics1['sem_exc'], 'Exc - Short' if 'short vs' in title else 'Exc - Short', 'blue',
                                     metrics1['avg_inh'], metrics1['sem_inh'], 'Inh - Short' if 'short vs' in title else 'Inh - Short', 'red',
                                     title=f'Exc vs Inh: {title.split(" vs ")[0]} (2nd stim aligned)',
                                     stim1_time=avg_first_stim_time, stim2_time=0, line_mode='all')
        ]
        
        # Heatmaps - combine both conditions for display
        combined_heat = np.vstack([metrics1['heat'], metrics2['heat']])
        combined_labels = np.hstack([labels, labels])
        
        plot_gridspec_heatmap(fig, gs, (slice(row, row+2), 4), neu_time, combined_heat,
                             title=f'Combined heatmap: {title} (2nd stim aligned)',
                             labels=combined_labels, stim1_time=np.nanmean(avg_first_stim_time), stim2_time=0, line_mode='all')
        plot_sorted_heatmap(fig, gs, (slice(row, row+2), 5), neu_time, combined_heat, labels=combined_labels,
                           title=f'Sorted combined heatmap: {title} (2nd stim aligned)',
                           stim1_time=np.nanmean(avg_first_stim_time), stim2_time=0, line_mode='all')
        
        # set_uniform_ylim(axes_group)

    # Set uniform y-axis limits
    # set_uniform_ylim(axes_group1)
    # set_uniform_ylim(axes_group2)
    # set_uniform_ylim(axes_group3)
    # set_uniform_ylim(axes_group4)
    # set_uniform_ylim(axes_group5)
    # set_uniform_ylim(axes_group6)

    plt.tight_layout()
    
    if save_path is not None:
        figures_dir = os.path.join(save_path, 'Figures')
        if not os.path.exists(figures_dir):
            os.makedirs(figures_dir)
        fig_file_path = os.path.join(figures_dir, 'neural_response_around_events.pdf')
        plt.savefig(fig_file_path, dpi=300, bbox_inches='tight')
        plt.close(fig)  # Explicitly close the figure after saving
    else:
        plt.close(fig)
        
def plot_events_neural_response(neural_trials, labels, save_path=None, pooling=False):
   
    """
    Plot neural responses to events with options for pooling data across sessions.
    
    Parameters:
    neural_trials (list): List of neural trials data for each session.
    labels (list): List of neuron labels for each session.
    save_path (str, optional): Path to save the figure. If None, the figure is displayed.
    pooling (bool, optional): Whether to pool data across sessions.
    
    Returns:
    None
    """
    if pooling:
        main(neural_trials, labels, save_path=save_path)
    else:
        # Call the main function without pooling
        main([neural_trials], [labels], save_path=save_path)
