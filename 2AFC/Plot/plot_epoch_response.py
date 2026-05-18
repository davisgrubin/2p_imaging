import numpy as np
import os
from matplotlib import pyplot as plt
from matplotlib import gridspec
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import sem
import warnings
import itertools
warnings.filterwarnings("ignore")

def trim_seq(data, pivots):
    """Trim sequences to the same length based on pivot points.

    This function trims a list of sequences (1D or 3D arrays) to the same length, aligning them
    around specified pivot points. The trimmed length is determined by the shortest available
    segments before and after the pivots across all sequences. For 1D arrays, the function trims
    the sequences directly. For 3D arrays, it trims along the last dimension (time axis).

    Args:
        data (list): List of numpy arrays, either 1D (e.g., time series) or 3D (e.g., multi-channel
            time series with shape [channels, features, time]).
        pivots (numpy.ndarray or list): Array of pivot indices for each sequence, indicating the
            alignment points for trimming.

    Returns:
        list: List of trimmed numpy arrays, all having the same length, aligned around the pivot points.

    Notes:
        - For 1D arrays, each sequence is trimmed to [pivot - len_l_min : pivot + len_r_min].
        - For 3D arrays, trimming is applied along the last dimension (time axis).
        - len_l_min is the minimum distance from the start of any sequence to its pivot.
        - len_r_min is the minimum distance from any pivot to the end of its sequence.
        - The function assumes all sequences in data have compatible shapes (all 1D or all 3D).
    """
    if len(data[0].shape) == 1:
        len_l_min = np.min(pivots)
        len_r_min = np.min([len(data[i]) - pivots[i] for i in range(len(data))])
        data = [data[i][pivots[i] - len_l_min:pivots[i] + len_r_min]
                for i in range(len(data))]
    if len(data[0].shape) == 3:
        len_l_min = np.min(pivots)
        len_r_min = np.min([len(data[i][0, 0, :]) - pivots[i] for i in range(len(data))])
        data = [data[i][:, :, pivots[i] - len_l_min:pivots[i] + len_r_min]
                for i in range(len(data))]
    return data

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
    
    stim1_time = [float(stim1_time)] if isinstance(stim1_time, (int, float, np.floating)) else stim1_time
    stim2_time = [float(stim2_time)] if isinstance(stim2_time, (int, float, np.floating)) and stim2_time is not None else stim2_time if stim2_time is not None else []
    
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
    return ax

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

def get_perception_response(neural_trials, target_state, l_frames, r_frames, indices=0):
    """Extract neural and stimulus responses around specific trial states."""
    exclude_start_trials = 2
    exclude_end_trials = 2
    time = neural_trials['time']
    neu_seq = []
    neu_time = []
    stim_seq = []
    stim_value = []
    stim_time = []
    led_value = []
    trial_type = []
    block_type = []
    isi = []
    decision = []
    outcome = []
    included_ti = []
    for ti in range(len(neural_trials['trial_labels'])):
        t = neural_trials['trial_labels'][target_state][ti].flatten()[indices]
        if (not np.isnan(t) and
            ti >= exclude_start_trials and
            ti < len(neural_trials['trial_labels']) - exclude_end_trials):
            idx = np.searchsorted(neural_trials['time'], t)
            if idx > l_frames and idx < len(neural_trials['time']) - r_frames:
                f = neural_trials['dff'][:, idx - l_frames : idx + r_frames]
                f = np.expand_dims(f, axis=0)
                neu_seq.append(f)
                neu_time.append(neural_trials['time'][idx - l_frames : idx + r_frames] - time[idx])
                vol_t_c = np.searchsorted(neural_trials['vol_time'], neural_trials['time'][idx])
                vol_t_l = np.searchsorted(neural_trials['vol_time'], neural_trials['time'][idx - l_frames])
                vol_t_r = np.searchsorted(neural_trials['vol_time'], neural_trials['time'][idx + r_frames])
                stim_time.append(neural_trials['vol_time'][vol_t_l:vol_t_r] - neural_trials['vol_time'][vol_t_c])
                stim_value.append(neural_trials['vol_stim_vis'][vol_t_l:vol_t_r])
                led_value.append(neural_trials['vol_led'][vol_t_l:vol_t_r])
                stim_seq.append(neural_trials['trial_labels']['stim_seq'][ti].reshape(1, 2, 2) - t)
                trial_type.append(neural_trials['trial_labels']['trial_type'][ti])
                block_type.append(neural_trials['trial_labels']['block_type'][ti])
                isi.append(neural_trials['trial_labels']['isi'][ti])
                decision.append(neural_trials['trial_labels']['lick'][ti][1, 0])
                outcome.append(neural_trials['trial_labels']['outcome'][ti])
                included_ti.append(ti)
    neu_time_zero = [np.argmin(np.abs(nt)) for nt in neu_time]
    neu_time = trim_seq(neu_time, neu_time_zero)
    neu_seq = trim_seq(neu_seq, neu_time_zero)
    stim_time_zero = [np.argmin(np.abs(sv)) for sv in stim_value]
    stim_time = trim_seq(stim_time, stim_time_zero)
    stim_value = trim_seq(stim_value, stim_time_zero)
    led_value = trim_seq(led_value, stim_time_zero)
    neu_seq = np.concatenate(neu_seq, axis=0)
    neu_time = [nt.reshape(1, -1) for nt in neu_time]
    neu_time = np.concatenate(neu_time, axis=0)
    stim_seq = np.concatenate(stim_seq, axis=0)
    stim_value = [sv.reshape(1, -1) for sv in stim_value]
    stim_value = np.concatenate(stim_value, axis=0)
    stim_time = [st.reshape(1, -1) for st in stim_time]
    stim_time = np.concatenate(stim_time, axis=0)
    led_value = [lv.reshape(1, -1) for lv in led_value]
    led_value = np.concatenate(led_value, axis=0)
    trial_type = np.array(trial_type)
    block_type = np.array(block_type)
    isi = np.array(isi)
    decision = np.array(decision)
    outcome = np.array(outcome)
    included_ti = np.array(included_ti)
    neu_time = np.mean(neu_time, axis=0)
    stim_time = np.mean(stim_time, axis=0)
    return [neu_seq, neu_time, stim_seq, stim_value, stim_time, led_value, trial_type, block_type, isi, decision, outcome, included_ti]

def pool_session_data(neural_trials_list, labels_list, state, l_frames, r_frames, indices, epoch_list=None):
    """
    Pool data from multiple sessions.
    
    Returns:
    tuple: Pooled neu_seq (n_trials_total, n_neurons_total, time), neu_time, trial_type, block_type, isi, decision, labels, outcomes, epoch (if epoch_list provided)
    """
    neu_seqs = []
    trial_types = []
    block_types = []
    isis = []
    decisions = []
    all_labels = []
    all_outcomes = []
    all_epochs = [] if epoch_list is not None else None

    neuron_counts = [labels.shape[0] for labels in labels_list]
    total_neurons = sum(neuron_counts)

    neuron_offset = 0

    for i, (neural_trials, session_labels) in enumerate(zip(neural_trials_list, labels_list)):
        neu_seq, neu_time, stim_seq, stim_value, stim_time, led_value, trial_type, block_type, isi, decision, outcome, included_ti = get_perception_response(
            neural_trials, state, l_frames, r_frames, indices=indices)

        n_trials, n_neurons, n_time = neu_seq.shape
        padded_neu_seq = np.zeros((n_trials, total_neurons, n_time))

        padded_neu_seq[:, neuron_offset:neuron_offset + n_neurons, :] = neu_seq
        neu_seqs.append(padded_neu_seq)

        trial_types.append(trial_type)
        block_types.append(block_type)
        isis.append(isi)
        decisions.append(decision)
        all_labels.append(session_labels)
        all_outcomes.append(outcome)

        if epoch_list is not None:
            session_epoch = epoch_list[i]
            all_epochs.append(session_epoch)

        neuron_offset += n_neurons

    pooled_neu_seq = np.concatenate(neu_seqs, axis=0)
    pooled_trial_type = np.concatenate(trial_types, axis=0)
    pooled_block_type = np.concatenate(block_types, axis=0)
    pooled_isi = np.concatenate(isis, axis=0)
    pooled_decision = np.concatenate(decisions, axis=0)
    pooled_labels = np.concatenate(all_labels, axis=0)
    pooled_outcomes = np.concatenate(all_outcomes, axis=0)

    if epoch_list is not None:
        pooled_epoch = np.concatenate(all_epochs, axis=0)
        return pooled_neu_seq, neu_time, pooled_trial_type, pooled_block_type, pooled_isi, pooled_decision, pooled_labels, pooled_outcomes, pooled_epoch
    else:
        return pooled_neu_seq, neu_time, pooled_trial_type, pooled_block_type, pooled_isi, pooled_decision, pooled_labels, pooled_outcomes

def main(neural_trials_list, labels_list, save_path=None):
    """
    Main function to generate neural response plots for pooled session data with epochs.
    """
    # Compute epoch assignments per session, aligned with valid trials for each state
    epoch_list_per_state = {}
    for state in ['stim_seq', 'state_reward', 'state_punish']:
        epoch_list = []
        for neural_trials in neural_trials_list:
            # Get block_type and included trial indices for the specific state
            _, _, _, _, _, _,trial_type, block_type, isi, decision, outcomes, included_ti = get_perception_response(
                neural_trials, state, 1, 1, indices=0 if state == 'stim_seq' else 0)
            epoch = np.zeros(len(neural_trials['trial_labels']), dtype=int)
            for key, group in itertools.groupby(enumerate(block_type), lambda x: x[1]):
                if key in [1, 2]:  # Only process short (1) and long (2) blocks
                    indices = [idx for idx, _ in group]
                    mid = len(indices) // 2
                    epoch[indices[:mid]] = 1
                    epoch[indices[mid:]] = 2
            # Filter epoch to only include trials returned by get_perception_response
            epoch = epoch[included_ti]
            epoch_list.append(epoch)
        epoch_list_per_state[state] = epoch_list

    fig = plt.figure(figsize=(40, 180))
    gs = gridspec.GridSpec(48, 6, wspace=0.4, hspace=0.4)
    
    # Define the 4 base conditions
    base_conditions = [
        ('short trials short block', 0, 1),  # trial_type, block_type for masks
        ('long trials short block', 1, 1),
        ('short trials long block', 0, 2),
        ('long trials long block', 1, 2)
    ]
    
    # Define analysis groups
    analysis_groups = [
        {'state': 'stim_seq', 'indices': 0, 'l_frames': 90, 'r_frames': 120, 'align': 'stim1', 'outcome_filter': None, 'line_mode': 'all', 'stim_style': 'bar', 'row_start': 0},
        {'state': 'stim_seq', 'indices': 2, 'l_frames': 90, 'r_frames': 120, 'align': 'stim2', 'outcome_filter': None, 'line_mode': 'all', 'stim_style': 'bar', 'row_start': 8},
        {'state': 'stim_seq', 'indices': 0, 'l_frames': 90, 'r_frames': 120, 'align': 'stim1', 'outcome_filter': 'reward', 'line_mode': 'all', 'stim_style': 'bar', 'row_start': 16},
        {'state': 'stim_seq', 'indices': 0, 'l_frames': 90, 'r_frames': 120, 'align': 'stim1', 'outcome_filter': 'punish', 'line_mode': 'all', 'stim_style': 'bar', 'row_start': 24},
        {'state': 'state_reward', 'indices': 0, 'l_frames': 60, 'r_frames': 60, 'align': 'reward', 'outcome_filter': 'reward', 'line_mode': 'reward', 'stim_style': 'line', 'row_start': 32},
        {'state': 'state_punish', 'indices': 0, 'l_frames': 60, 'r_frames': 60, 'align': 'punish', 'outcome_filter': 'punish', 'line_mode': 'punish', 'stim_style': 'line', 'row_start': 40}
    ]
    
    for group in analysis_groups:
        # Use the epoch_list corresponding to the state
        epoch_list = epoch_list_per_state[group['state']]
        neu_seq, neu_time, trial_type, block_type, isi, decision, labels, outcomes, epoch = pool_session_data(
            neural_trials_list, labels_list, group['state'], group['l_frames'], group['r_frames'], group['indices'], epoch_list=epoch_list)
        
        row = group['row_start']
        for cond_title, t_type, b_type in base_conditions:
            mask1 = (epoch == 1) & (block_type == b_type) & (trial_type == t_type)
            mask2 = (epoch == 2) & (block_type == b_type) & (trial_type == t_type)
            if group['outcome_filter'] == 'reward':
                mask1 &= (outcomes == 'reward')
                mask2 &= (outcomes == 'reward')
            elif group['outcome_filter'] == 'punish':
                mask1 &= (outcomes == 'punish')
                mask2 &= (outcomes == 'punish')
            
            metrics1 = calculate_metrics(neu_seq, mask1, labels)
            metrics2 = calculate_metrics(neu_seq, mask2, labels)
            
            # Compute stimulus times
            if group['align'] == 'stim1':
                avg_isi = np.nanmean(isi[mask1 | mask2])
                avg_sec_stim = avg_isi + 200
                avg_sec_stim_time = neu_time[np.searchsorted(neu_time, avg_sec_stim, side='left')]
                stim1_time = 0
                stim2_time = avg_sec_stim_time
            elif group['align'] == 'stim2':
                avg_isi = np.nanmean(isi[mask1 | mask2])
                avg_first_stim = - (avg_isi + 200)
                avg_first_stim_time = neu_time[np.searchsorted(neu_time, avg_first_stim, side='left')]
                stim1_time = avg_first_stim_time
                stim2_time = 0
            else:
                stim1_time = 0
                stim2_time = np.nan
            
            title_base = f"{cond_title} {group['outcome_filter'] + ' ' if group['outcome_filter'] else ''}(aligned to {group['align']})"
            
            axes_group = [
                plot_gridspec_superimposed(fig, gs, (row, 0), neu_time,
                                         metrics1['avg_all'], metrics1['sem_all'], 'Epoch1', 'green',
                                         metrics2['avg_all'], metrics2['sem_all'], 'Epoch2', 'blue',
                                         title=f'All neurons: epoch1 vs epoch2 \n {title_base}',
                                         stim1_time=stim1_time, stim2_time=stim2_time, line_mode=group['line_mode'], stim_style=group['stim_style']),
                plot_gridspec_superimposed(fig, gs, (row, 1), neu_time,
                                         metrics1['avg_exc'], metrics1['sem_exc'], 'Epoch1', 'green',
                                         metrics2['avg_exc'], metrics2['sem_exc'], 'Epoch2', 'blue',
                                         title=f'Excitatory neurons: epoch1 vs epoch2 \n {title_base}',
                                         stim1_time=stim1_time, stim2_time=stim2_time, line_mode=group['line_mode'], stim_style=group['stim_style']),
                plot_gridspec_superimposed(fig, gs, (row, 2), neu_time,
                                         metrics1['avg_inh'], metrics1['sem_inh'], 'Epoch1', 'green',
                                         metrics2['avg_inh'], metrics2['sem_inh'], 'Epoch2', 'blue',
                                         title=f'Inhibitory neurons: epoch1 vs epoch2 \n {title_base}',
                                         stim1_time=stim1_time, stim2_time=stim2_time, line_mode=group['line_mode'], stim_style=group['stim_style']),
                plot_gridspec_superimposed(fig, gs, (row, 3), neu_time,
                                         metrics1['avg_exc'], metrics1['sem_exc'], 'Exc - Epoch1', 'dodgerblue',
                                         metrics1['avg_inh'], metrics1['sem_inh'], 'Inh - Epoch1', 'hotpink',
                                         title=f'Exc vs Inh: epoch1 \n {title_base}',
                                         stim1_time=stim1_time, stim2_time=stim2_time, line_mode=group['line_mode'], stim_style=group['stim_style'])
            ]
            
            combined_heat = np.vstack([metrics1['heat'], metrics2['heat']])
            combined_labels = np.concatenate([labels, labels])
            
            plot_gridspec_heatmap(fig, gs, (slice(row, row+2), 4), neu_time, combined_heat,
                                 title=f'Combined heatmap: epoch1 vs epoch2 \n {title_base}',
                                 labels=combined_labels, stim1_time=stim1_time if not np.isnan(stim1_time) else np.nan, stim2_time=stim2_time if not np.isnan(stim2_time) else np.nan, line_mode=group['line_mode'])
            plot_sorted_heatmap(fig, gs, (slice(row, row+2), 5), neu_time, combined_heat, labels=combined_labels,
                               title=f'Sorted combined heatmap: epoch1 vs epoch2 \n {title_base}',
                               stim1_time=stim1_time if not np.isnan(stim1_time) else np.nan, stim2_time=stim2_time if not np.isnan(stim2_time) else np.nan, line_mode=group['line_mode'], sort_interval=(-400, 400))
            
            set_uniform_ylim(axes_group)
            row += 2
    
    plt.tight_layout()
    
    if save_path is not None:
        figures_dir = os.path.join(save_path, 'Figures')
        if not os.path.exists(figures_dir):
            os.makedirs(figures_dir)
        fig_file_path = os.path.join(figures_dir, 'neural_response_epochs.pdf')
        plt.savefig(fig_file_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.close(fig)
        
def plot_events_epoch_neural_response(neural_trials, labels, save_path=None, pooling=False):
    if pooling:
        main(neural_trials, labels, save_path=save_path)
    else:
        main([neural_trials], [labels], save_path=save_path)