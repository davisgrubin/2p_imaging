import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os

def pool_session_data(list_trial_labels):
    """
    Pool data from multiple sessions.
    
    Returns:
    tuple: Pooled neu_seq (n_trials_total, n_neurons_total, time), neu_time, trial_type, isi, decision, labels
    """
    # Initialize lists to pool data across sessions
    pooled_trial_type = []
    pooled_block_type = []
    pooled_correctness = []
    pooled_licks = []
    pooled_trial_start = []

    total_trials = 0

    for trial_labels in list_trial_labels:
        n_trials = len(trial_labels['time_trial_start'])
        total_trials += n_trials

        # Pool trial_type and block_type
        pooled_trial_type.append(trial_labels['trial_type'])
        pooled_block_type.append(trial_labels['block_type'])

        # Pool correctness and licks
        for trial in range(n_trials):
            pooled_licks.append(trial_labels['lick'][trial][0] - trial_labels['stim_seq'][trial][1][0])
            pooled_correctness.append(trial_labels['lick'][trial][2])
            pooled_trial_start.append(trial_labels['time_trial_start'][trial])

    # Concatenate arrays/lists
    pooled_trial_type = np.concatenate(pooled_trial_type, axis=0)
    pooled_block_type = np.concatenate(pooled_block_type, axis=0)
    pooled_correctness = np.array(pooled_correctness, dtype=object)
    pooled_licks = np.array(pooled_licks, dtype=object)
    pooled_trial_start = np.array(pooled_trial_start)

    return pooled_trial_type, pooled_block_type, pooled_correctness, total_trials, pooled_licks, pooled_trial_start

def plot_licking_patterns(n_trials, trial_type, block_type, correctness_per_trial, licks_per_trial, save_path=None):
   
    """
    Plot licking patterns aligned to the second stimulus onset for different trial conditions.
    Parameters:
    - n_trials: int, total number of trials
    - trial_type: array-like, trial types (0 for short, 1 for long)
    - block_type: array-like, block types (1 for standard, 2 for rare)
    - correctness_per_trial: list of arrays, correctness for each trial (1 for correct, 0 for incorrect, NaN for no response)
    - licks_per_trial: list of arrays, lick timestamps for each trial (aligned to second stimulus onset)
    - save_path: str or None, path to save the figure (if None, figure is not saved)
    """
    # Remove NaNs for plotting
    licks_per_trial = [lt[~np.isnan(lt)] for lt in licks_per_trial]

    # Define the 7 conditions
    conditions = [
        {'name': 'All trials', 'mask': np.ones(n_trials, dtype=bool)},
        {'name': 'Short trials\n(trial_type=0)', 'mask': trial_type == 0},
        {'name': 'Long trials\n(trial_type=1)', 'mask': trial_type == 1},
        {'name': 'Standard short\n(block=1, trial=0)', 'mask': (block_type == 1) & (trial_type == 0)},
        {'name': 'Rare short\n(block=2, trial=0)', 'mask': (block_type == 2) & (trial_type == 0)},
        {'name': 'Standard long\n(block=2, trial=1)', 'mask': (block_type == 2) & (trial_type == 1)},
        {'name': 'Rare long\n(block=1, trial=1)', 'mask': (block_type == 1) & (trial_type == 1)}
    ]

    # Create figure with gridspec
    fig = plt.figure(figsize=(28, 12))
    gs = GridSpec(3, 7, figure=fig, hspace=0.4, wspace=0.3)

    # Process each condition (column)
    for col_idx, condition in enumerate(conditions):
        mask = condition['mask']
        indices = np.where(mask)[0]
        
        if len(indices) == 0:
            # Skip if no trials match this condition
            for row in range(3):
                ax = fig.add_subplot(gs[row, col_idx])
                ax.text(0.5, 0.5, 'No trials', ha='center', va='center', fontsize=12)
                ax.set_xticks([])
                ax.set_yticks([])
                if row == 0:
                    ax.set_title(condition['name'], fontsize=10, fontweight='bold')
            continue
        
        # Get licks and correctness for this condition
        cond_licks = [licks_per_trial[i] for i in indices]
        cond_correctness = [correctness_per_trial[i] for i in indices]
        
        # Determine colors for each trial
        colors = []
        for c in cond_correctness:
            if len(c) == 0 or np.all(np.isnan(c)):
                colors.append('gray')
            else:
                mean_corr = np.nanmean(c)
                colors.append('green' if mean_corr >= 0.5 else 'red')
        
        # --- Row 1: All trials raster ---
        ax1 = fig.add_subplot(gs[0, col_idx])
        ax1.eventplot(cond_licks, colors=colors, linelengths=0.8)
        ax1.axvline(0, color='k', linestyle='--', linewidth=1, alpha=0.7)
        ax1.set_ylabel('Trial #', fontsize=9)
        # Remove top and right spines for cleaner look
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        if col_idx == 0:
            ax1.set_ylabel('All trials\nTrial #', fontsize=9, fontweight='bold')
        ax1.set_title(condition['name'], fontsize=10, fontweight='bold')
        ax1.tick_params(labelsize=8)
        
        # --- Row 2: Sorted by correctness (correct then incorrect) ---
        ax2 = fig.add_subplot(gs[1, col_idx])
        
        # Separate correct and incorrect trials
        correct_indices = [i for i, c in enumerate(cond_correctness) 
                        if len(c) > 0 and not np.all(np.isnan(c)) and np.nanmean(c) >= 0.5]
        incorrect_indices = [i for i, c in enumerate(cond_correctness) 
                            if len(c) > 0 and not np.all(np.isnan(c)) and np.nanmean(c) < 0.5]
        
        # Sort: correct first, then incorrect
        sorted_indices = correct_indices + incorrect_indices
        sorted_licks = [cond_licks[i] for i in sorted_indices]
        sorted_colors = ['green'] * len(correct_indices) + ['red'] * len(incorrect_indices)
        
        if len(sorted_licks) > 0:
            ax2.eventplot(sorted_licks, colors=sorted_colors, linelengths=0.8)
            ax2.axvline(0, color='k', linestyle='--', linewidth=1, alpha=0.7)
            # Add horizontal line separating correct/incorrect
            if len(correct_indices) > 0 and len(incorrect_indices) > 0:
                ax2.axhline(len(correct_indices), color='black', linestyle=':', linewidth=1.5, alpha=0.5)
        
        ax2.set_ylabel('Trial # (sorted)', fontsize=9)
        if col_idx == 0:
            ax2.set_ylabel('Sorted\nTrial #', fontsize=9, fontweight='bold')
        ax2.tick_params(labelsize=8)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        
        # --- Row 3: Average lick rate line plot ---
        ax3 = fig.add_subplot(gs[2, col_idx])
        
        # Calculate time bins
        all_lick_times = np.concatenate([lt for lt in cond_licks if len(lt) > 0])
        if len(all_lick_times) > 0:
            time_min, time_max = all_lick_times.min(), all_lick_times.max()
            time_bins = np.linspace(time_min, time_max, 50)
            bin_centers = (time_bins[:-1] + time_bins[1:]) / 2
            
            # Calculate average for correct trials
            if len(correct_indices) > 0:
                correct_licks = [cond_licks[i] for i in correct_indices]
                correct_counts = []
                for lt in correct_licks:
                    if len(lt) > 0:
                        hist, _ = np.histogram(lt, bins=time_bins)
                        correct_counts.append(hist)
                if correct_counts:
                    avg_correct = np.mean(correct_counts, axis=0)
                    ax3.plot(bin_centers, avg_correct, 'g-', linewidth=2, label='Correct', alpha=0.8)
            
            # Calculate average for incorrect trials
            if len(incorrect_indices) > 0:
                incorrect_licks = [cond_licks[i] for i in incorrect_indices]
                incorrect_counts = []
                for lt in incorrect_licks:
                    if len(lt) > 0:
                        hist, _ = np.histogram(lt, bins=time_bins)
                        incorrect_counts.append(hist)
                if incorrect_counts:
                    avg_incorrect = np.mean(incorrect_counts, axis=0)
                    ax3.plot(bin_centers, avg_incorrect, 'r-', linewidth=2, label='Incorrect', alpha=0.8)
        
        ax3.axvline(0, color='k', linestyle='--', linewidth=1, alpha=0.7, label='Stim onset')
        ax3.set_xlabel('Time from stim onset', fontsize=9)
        ax3.set_ylabel('Avg licks', fontsize=9)
        if col_idx == 0:
            ax3.set_ylabel('Average\nLick count', fontsize=9, fontweight='bold')
            ax3.legend(fontsize=8, loc='best')
        ax3.tick_params(labelsize=8)
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        # ax3.grid(True, alpha=0.3)

    plt.suptitle('Lick Analysis by Trial Conditions: Aligned to Second Stimulus Onset', 
                fontsize=14, fontweight='bold', y=0.995)
    
    if save_path is not None:
        figures_dir = os.path.join(save_path, 'Figures')
        if not os.path.exists(figures_dir):
            os.makedirs(figures_dir)
        fig_file_path = os.path.join(figures_dir, 'licking_pattern.pdf')
        plt.savefig(fig_file_path, dpi=300, bbox_inches='tight')
        plt.close(fig)  # Explicitly close the figure after saving
    else:
        plt.show()

def main_plot_licking_patterns(list_trial_labels, save_path=None):
    """
    Main function to pool session data and plot licking patterns.
    
    Parameters:
    - list_trial_labels: list of trial_labels dictionaries from multiple sessions
    - save_path: str or None, path to save the figure (if None, figure is not saved)
    """
    # Pool data across sessions
    trial_type, block_type, correctness_per_trial, n_trials, licks_per_trial, _ = pool_session_data(list_trial_labels)
    
    # Plot licking patterns
    plot_licking_patterns(n_trials, trial_type, block_type, correctness_per_trial, licks_per_trial, save_path=save_path)