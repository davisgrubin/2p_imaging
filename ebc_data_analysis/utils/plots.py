import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def plot_fec_CR_scatter(base_line_avg, CR_interval_avg, CR_stat):
    # Initialize arrays for absolute values, relative changes, and baselines
    cr_amplitudes = []
    cr_relative_changes = []
    baselines = []

    # Calculate baseline, absolute CR amplitudes, and relative changes
    for id in base_line_avg:
        baseline = base_line_avg[id]
        cr_amplitude = abs(CR_interval_avg[id])
        cr_relative_change = CR_interval_avg[id] - baseline

        baselines.append(baseline)
        cr_amplitudes.append(cr_amplitude)
        cr_relative_changes.append(cr_relative_change)

    # Convert lists to numpy arrays
    baselines = np.array(baselines)
    cr_amplitudes = np.array(cr_amplitudes)
    cr_relative_changes = np.array(cr_relative_changes)

    # Separate data based on CR+ and CR-
    baselines_crp = []
    baselines_crn = []
    cr_amplitudes_crp = []
    cr_amplitudes_crn = []
    cr_relative_changes_crp = []
    cr_relative_changes_crn = []

    for id in base_line_avg:
        baseline = base_line_avg[id]
        cr_amplitude = abs(CR_interval_avg[id])
        cr_relative_change = CR_interval_avg[id] - baseline

        if CR_stat[id] == 1:  # CR+
            baselines_crp.append(baseline)
            cr_amplitudes_crp.append(cr_amplitude)
            cr_relative_changes_crp.append(cr_relative_change)
        elif CR_stat[id] == 0:  # CR-
            baselines_crn.append(baseline)
            cr_amplitudes_crn.append(cr_amplitude)
            cr_relative_changes_crn.append(cr_relative_change)

    # Convert lists to numpy arrays
    baselines_crp = np.array(baselines_crp)
    baselines_crn = np.array(baselines_crn)
    cr_amplitudes_crp = np.array(cr_amplitudes_crp)
    cr_amplitudes_crn = np.array(cr_amplitudes_crn)
    cr_relative_changes_crp = np.array(cr_relative_changes_crp)
    cr_relative_changes_crn = np.array(cr_relative_changes_crn)

    # Combine CR+ and CR- data
    all_baselines = np.concatenate([baselines_crp, baselines_crn])
    all_relative_changes = np.concatenate([cr_relative_changes_crp, cr_relative_changes_crn])

    # Create plots
    fig, axes = plt.subplots(3, 2, figsize=(18, 20))

    axes[0, 0].hist(baselines, bins=20, color='lime', edgecolor='black', alpha=0.7)
    axes[0, 0].set_title('Distribution of Baseline Values Across Sessions')
    axes[0, 0].set_xlabel('Baseline Value')
    axes[0, 0].set_ylabel('Frequency')

    axes[0, 1].hist(cr_amplitudes, bins=20, color='green', edgecolor='black', alpha=0.7)
    axes[0, 1].set_title('Distribution of CR Amplitudes Across Sessions')
    axes[0, 1].set_xlabel('CR Value')
    axes[0, 1].set_ylabel('Frequency')

    axes[1, 0].scatter(baselines_crp, cr_amplitudes_crp, color='red', alpha=0.7, label='CR+')
    axes[1, 0].scatter(baselines_crn, cr_amplitudes_crn, color='blue', alpha=0.7, label='CR-')
    axes[1, 0].set_title('CR Amplitude (Absolute) vs. Baseline', fontsize=14)
    axes[1, 0].set_xlabel('Baseline')
    axes[1, 0].set_ylabel('CR Amplitude (Absolute)')

    axes[1, 1].scatter(baselines_crp, cr_relative_changes_crp, color='red', alpha=0.7, label='CR+')
    axes[1, 1].scatter(baselines_crn, cr_relative_changes_crn, color='blue', alpha=0.7, label='CR-')
    axes[1, 1].set_title('CR Size (Relative Change) vs. Baseline', fontsize=14)
    axes[1, 1].set_xlabel('Baseline')
    axes[1, 1].set_ylabel('CR Size (Relative Change)')

    axes[2, 0].hexbin(baselines, cr_amplitudes, gridsize=30, cmap='Blues', mincnt=1)
    cbar = plt.colorbar(axes[2, 0].hexbin(baselines, cr_amplitudes, gridsize=30, cmap='Blues', mincnt=1))
    cbar.set_label('Count')
    axes[2, 0].set_title('Joint Distribution of CR Amplitude and Baseline', fontsize=14)
    axes[2, 0].set_xlabel('Baseline')
    axes[2, 0].set_ylabel('CR Amplitude (Absolute)')

    axes[2, 1].hexbin(all_baselines, all_relative_changes, gridsize=30, cmap='Reds', mincnt=1, alpha=0.7)
    cbar = plt.colorbar(axes[2, 1].hexbin(all_baselines, all_relative_changes, gridsize=30, cmap='Reds', mincnt=1, alpha=0.7))
    cbar.set_label('Count')
    axes[2, 1].set_title('Joint Distribution of CR Size (Relative Change) and Baseline', fontsize=14)
    axes[2, 1].set_xlabel('Baseline')
    axes[2, 1].set_ylabel('CR Size (Relative Change)')

    plt.show()


def plot_fec_averages(short_CRp_fec, short_CRn_fec, long_CRp_fec, long_CRn_fec, fec_time_0, shorts, longs, trials, y_lim,sample_id):
    import numpy as np
    import matplotlib.pyplot as plt

    # Short Trials
    mean1_short = np.mean(short_CRp_fec, axis=0)
    std1_short = np.std(short_CRp_fec, axis=0) / np.sqrt(len(short_CRp_fec))
    mean0_short = np.mean(short_CRn_fec, axis=0)
    std0_short = np.std(short_CRn_fec, axis=0) / np.sqrt(len(short_CRn_fec))
    id_short = shorts[0]

    # Long Trials
    mean1_long = np.mean(long_CRp_fec, axis=0)
    std1_long = np.std(long_CRp_fec, axis=0) / np.sqrt(len(long_CRp_fec))
    mean0_long = np.mean(long_CRn_fec, axis=0)
    std0_long = np.std(long_CRn_fec, axis=0) / np.sqrt(len(long_CRn_fec))
    id_long = longs[0]

    # Create the plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot Short Trials
    axes[0].plot(fec_time_0[sample_id], mean1_short, label="CR+", color="red")
    axes[0].fill_between(fec_time_0[sample_id], mean1_short - std1_short, mean1_short + std1_short, color="red", alpha=0.1)
    axes[0].plot(fec_time_0[sample_id], mean0_short, label="CR-", color="blue")
    axes[0].fill_between(fec_time_0[sample_id], mean0_short - std0_short, mean0_short + std0_short, color="blue", alpha=0.1)
    axes[0].axvspan(0, trials[id_short]["LED"][1] - trials[id_short]["LED"][0], color="gray", alpha=0.5, label="LED")
    axes[0].axvspan(trials[id_short]["AirPuff"][0] - trials[id_short]["LED"][0], trials[id_short]["AirPuff"][1] - trials[id_short]["LED"][0], color="blue", alpha=0.5, label="AirPuff")
    axes[0].set_title("FEC Average for Short Trials")
    axes[0].set_xlabel("Time for LED Onset (ms)")
    axes[0].set_ylabel("FEC")
    axes[0].set_xlim(-200, 550)
    axes[0].set_ylim(y_lim, 1)
    axes[0].legend()

    # Plot Long Trials
    axes[1].plot(fec_time_0[id_long], mean1_long, label="CR+", color="red")
    axes[1].fill_between(fec_time_0[id_long], mean1_long - std1_long, mean1_long + std1_long, color="red", alpha=0.1)
    axes[1].plot(fec_time_0[id_long], mean0_long, label="CR-", color="blue")
    axes[1].fill_between(fec_time_0[id_long], mean0_long - std0_long, mean0_long + std0_long, color="blue", alpha=0.1)
    axes[1].axvspan(0, trials[id_long]["LED"][1] - trials[id_long]["LED"][0], color="gray", alpha=0.5, label="LED")
    axes[1].axvspan(trials[id_long]["AirPuff"][0] - trials[id_long]["LED"][0], trials[id_long]["AirPuff"][1] - trials[id_long]["LED"][0], color="lime", alpha=0.5, label="AirPuff")
    axes[1].set_title("FEC Average for Long Trials")
    axes[1].set_xlabel("Time for LED Onset (ms)")
    axes[1].set_ylabel("FEC")
    axes[1].set_xlim(-200, 750)
    axes[1].set_ylim(y_lim, 1)

    axes[1].legend()

    plt.tight_layout()
    plt.show()

def save_fec_plots_to_pdf(trials, fec_time_0, fec_0, CR_stat, all_id, filename):

    rows_per_page = 7
    cols_per_page = 4
    plots_per_page = rows_per_page * cols_per_page

    with PdfPages(filename) as pdf:
        num_trials = len(trials)
        trial_ids = all_id  # Ensure consistent trial order

        for page_start in range(0, num_trials, plots_per_page):
            fig, axes = plt.subplots(rows_per_page, cols_per_page, figsize=(15, 20), constrained_layout=True)
            axes = axes.flatten()  # Flatten axes for easier indexing

            for i, trial_idx in enumerate(range(page_start, min(page_start + plots_per_page, num_trials))):
                id = trial_ids[trial_idx]
                ax = axes[i]  # Select the correct subplot for this trial

                CR_color = "blue" if CR_stat[id] == 0 else "red"
                block_color = "blue" if trials[id]["trial_type"][()] == 1 else "lime"

                ax.plot(fec_time_0[id], fec_0[id]/100, label=f"FEC of trial {id}", color=CR_color)
                ax.axvspan(0, trials[id]["LED"][1] - trials[id]["LED"][0], 
                           color="gray", alpha=0.5, label="LED")
                ax.axvspan(trials[id]["AirPuff"][0] - trials[id]["LED"][0], 
                           trials[id]["AirPuff"][1] - trials[id]["LED"][0], 
                           color=block_color, alpha=0.5, label="AirPuff")

                ax.set_title(f"Trial {id} FEC", fontsize=10)
                ax.set_xlabel("Time for LED Onset(ms)", fontsize=8)
                ax.set_ylabel("FEC", fontsize=8)
                ax.set_xlim(-50, 550)
                ax.tick_params(axis='both', which='major', labelsize=7)
                ax.legend(fontsize=7)

                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

            for j in range(i + 1, plots_per_page):
                axes[j].axis("off")

            pdf.savefig(fig)
            plt.close(fig)

def save_roi_plots_to_pdf(short_crp_avg_dff, short_crn_avg_dff, short_crp_sem_dff, short_crn_sem_dff, 
                          short_crp_aligned_time, long_crp_avg_dff, long_crn_avg_dff, long_crp_sem_dff, long_crn_sem_dff, 
                          long_crp_aligned_time, trials, pdf_filename):
    x_min, x_max = -300, 550

    with PdfPages(pdf_filename) as pdf:
        for roi in range(len(short_crp_avg_dff)):
            fig, axs = plt.subplots(1, 2, figsize=(15, 6))

            # Short Trials: Calculate y-axis limits dynamically
            id = list(short_crp_aligned_time.keys())[0]
            time_short = short_crp_aligned_time[list(short_crp_aligned_time.keys())[0]]
            mask_short = (time_short >= x_min) & (time_short <= x_max)
            y_min_short = min(
                np.min(short_crp_avg_dff[roi][mask_short] - short_crp_sem_dff[roi][mask_short]),
                np.min(short_crn_avg_dff[roi][mask_short] - short_crn_sem_dff[roi][mask_short])
            )
            y_max_short = max(
                np.max(short_crp_avg_dff[roi][mask_short] + short_crp_sem_dff[roi][mask_short]),
                np.max(short_crn_avg_dff[roi][mask_short] + short_crn_sem_dff[roi][mask_short])
            )

            axs[0].set_xlim(x_min, x_max)
            axs[0].set_ylim(y_min_short, y_max_short)
            axs[0].set_title(f"ROI {roi} - Short Trials", fontsize=14)
            axs[0].set_xlabel("Time for LED Onset(ms)", fontsize=12)
            axs[0].set_ylabel("Average dF/F", fontsize=12)
            axs[0].plot(time_short, short_crp_avg_dff[roi], label="CR+", color="red")
            axs[0].fill_between(time_short, short_crp_avg_dff[roi] - short_crp_sem_dff[roi],
                                short_crp_avg_dff[roi] + short_crp_sem_dff[roi], color="red", alpha=0.1)
            axs[0].plot(time_short, short_crn_avg_dff[roi], label="CR-", color="blue")
            axs[0].fill_between(time_short, short_crn_avg_dff[roi] - short_crn_sem_dff[roi],
                                short_crn_avg_dff[roi] + short_crn_sem_dff[roi], color="blue", alpha=0.1)
            axs[0].axvspan(0, trials[id]["LED"][1] - trials[id]["LED"][0], color="gray", alpha=0.5, label="LED")
            axs[0].axvspan(trials[id]["AirPuff"][0] - trials[id]["LED"][0],
                           trials[id]["AirPuff"][1] - trials[id]["LED"][0], color="blue", alpha=0.5, label="AirPuff")
            axs[0].legend(fontsize=10)

            # Long Trials: Calculate y-axis limits dynamically
            id = list(long_crp_aligned_time.keys())[0]
            time_long = long_crp_aligned_time[list(long_crp_aligned_time.keys())[0]]
            mask_long = (time_long >= x_min) & (time_long <= x_max)
            y_min_long = min(
                np.min(long_crp_avg_dff[roi][mask_long] - long_crp_sem_dff[roi][mask_long]),
                np.min(long_crn_avg_dff[roi][mask_long] - long_crn_sem_dff[roi][mask_long])
            )
            y_max_long = max(
                np.max(long_crp_avg_dff[roi][mask_long] + long_crp_sem_dff[roi][mask_long]),
                np.max(long_crn_avg_dff[roi][mask_long] + long_crn_sem_dff[roi][mask_long])
            )

            axs[1].set_xlim(x_min, x_max)
            axs[1].set_ylim(y_min_long, y_max_long)
            axs[1].set_title(f"ROI {roi} - Long Trials", fontsize=14)
            axs[1].set_xlabel("Time for LED Onset(ms)", fontsize=12)
            axs[1].set_ylabel("Average dF/F", fontsize=12)
            axs[1].plot(time_long, long_crp_avg_dff[roi], label="CR+", color="red")
            axs[1].fill_between(time_long, long_crp_avg_dff[roi] - long_crp_sem_dff[roi],
                                long_crp_avg_dff[roi] + long_crp_sem_dff[roi], color="red", alpha=0.1)
            axs[1].plot(time_long, long_crn_avg_dff[roi], label="CR-", color="blue")
            axs[1].fill_between(time_long, long_crn_avg_dff[roi] - long_crn_sem_dff[roi],
                                long_crn_avg_dff[roi] + long_crn_sem_dff[roi], color="blue", alpha=0.1)
            axs[1].axvspan(0, trials[id]["LED"][1] - trials[id]["LED"][0], color="gray", alpha=0.5, label="LED")
            axs[1].axvspan(trials[id]["AirPuff"][0] - trials[id]["LED"][0],
                           trials[id]["AirPuff"][1] - trials[id]["LED"][0], color="lime", alpha=0.5, label="AirPuff")
            axs[1].legend(fontsize=10)

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        print(f"All plots have been saved to {pdf_filename}")

def plot_trial_averages(trials, aligned_time, crp_avg, crp_sem, crn_avg, crn_sem, title_suffix, pooled=False):
    x_min, x_max = -300, 550

    if title_suffix == "Short":
        suffix_color = "blue"
    if title_suffix == "Long":
        suffix_color = "lime"

    # Extract time and trial id
    time = aligned_time[list(aligned_time.keys())[0]]
    id = list(aligned_time.keys())[0]

    # Mask and filter data
    mask = (time >= x_min) & (time <= x_max)
    crp_filtered = crp_avg[mask]
    crp_sem_filtered = crp_sem[mask]
    crn_filtered = crn_avg[mask]
    crn_sem_filtered = crn_sem[mask]

    # Calculate y-axis limits
    y_min = min(
        np.min(crp_filtered - crp_sem_filtered),
        np.min(crn_filtered - crn_sem_filtered)
    )
    y_max = max(
        np.max(crp_filtered + crp_sem_filtered),
        np.max(crn_filtered + crn_sem_filtered)
    )

    # Create figure
    plt.figure(figsize=(12, 6))
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    trial_type = "Pooled" if pooled else "Grand"
    plt.title(f"{trial_type} Average of {title_suffix} Trials", fontsize=14)
    plt.xlabel("Time for LED Onset (ms)", fontsize=12)
    plt.ylabel("Mean df/f (+/- SEM)", fontsize=12)

    plt.plot(time, crp_avg, label="CR+", color="red")
    plt.fill_between(time, crp_avg - crp_sem, crp_avg + crp_sem, color="red", alpha=0.1)

    plt.plot(time, crn_avg, label="CR-", color="blue")
    plt.fill_between(time, crn_avg - crn_sem, crn_avg + crn_sem, color="blue", alpha=0.1)

    # Add shaded regions for LED and AirPuff
    plt.axvspan(0, trials[id]["LED"][1] - trials[id]["LED"][0], color="gray", alpha=0.5, label="LED")
    plt.axvspan(trials[id]["AirPuff"][0] - trials[id]["LED"][0], trials[id]["AirPuff"][1] - trials[id]["LED"][0], 
                color=suffix_color, alpha=0.5, label="AirPuff")

    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

def plot_trial_averages_sig(trials, aligned_time, crp_avg, crp_sem, crn_avg, crn_sem, title_suffix, event, pooled=False):
    x_min, x_max = -300, 550

    if title_suffix == "Short":
        suffix_color = "blue"
    if title_suffix == "Long":
        suffix_color = "lime"

    # Extract time and trial id
    time = aligned_time[list(aligned_time.keys())[0]]
    id = list(aligned_time.keys())[0]

    # Mask and filter data
    mask = (time >= x_min) & (time <= x_max)
    crp_filtered = crp_avg[mask]
    crp_sem_filtered = crp_sem[mask]
    crn_filtered = crn_avg[mask]
    crn_sem_filtered = crn_sem[mask]

    # Calculate y-axis limits
    y_min = min(
        np.min(crp_filtered - crp_sem_filtered),
        np.min(crn_filtered - crn_sem_filtered)
    )
    y_max = max(
        np.max(crp_filtered + crp_sem_filtered),
        np.max(crn_filtered + crn_sem_filtered)
    )

    # Create figure
    plt.figure(figsize=(12, 6))
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    trial_type = "Pooled" if pooled else "Grand"
    plt.title(f"{trial_type} Average of {title_suffix} Trials for {event} significant ROIs", fontsize=14)
    plt.xlabel("Time for LED Onset (ms)", fontsize=12)
    plt.ylabel("Mean df/f (+/- SEM)", fontsize=12)

    plt.plot(time, crp_avg, label="CR+", color="red")
    plt.fill_between(time, crp_avg - crp_sem, crp_avg + crp_sem, color="red", alpha=0.1)

    plt.plot(time, crn_avg, label="CR-", color="blue")
    plt.fill_between(time, crn_avg - crn_sem, crn_avg + crn_sem, color="blue", alpha=0.1)

    # Add shaded regions for LED and AirPuff
    plt.axvspan(0, trials[id]["LED"][1] - trials[id]["LED"][0], color="gray", alpha=0.5, label="LED")
    plt.axvspan(trials[id]["AirPuff"][0] - trials[id]["LED"][0], trials[id]["AirPuff"][1] - trials[id]["LED"][0], 
                color=suffix_color, alpha=0.5, label="AirPuff")

    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_heatmaps_side_by_side(heat_arrays, aligned_times, titles, trials, main_title, x_label="Time for LED Onset(ms)"):

    fig, axes = plt.subplots(1, 4, figsize=(24, 6), sharey=True, gridspec_kw={"width_ratios": [1, 1, 1, 1.15] , "height_ratios": [1]})
    
    for i, (arr, aligned_time, title) in enumerate(zip(heat_arrays, aligned_times, titles)):
        time_array = aligned_time[list(aligned_time.keys())[0]]
        id = list(aligned_time.keys())[0]
        
        ax = axes[i]
        im = ax.imshow(arr, aspect="auto", cmap="magma", origin="upper",
                       extent=[time_array[0], time_array[-1], 0, arr.shape[0]])
        
        airpuff_color = "lime" if "Long" in title else "blue"
        ax.axvspan(trials[id]["AirPuff"][0] - trials[id]["LED"][0], 
                   trials[id]["AirPuff"][1] - trials[id]["LED"][0], 
                   color=airpuff_color, alpha=0.5, label="AirPuff")
        ax.axvspan(0, trials[id]["LED"][1] - trials[id]["LED"][0], 
                   color="gray", alpha=0.5, label="LED")
        
        ax.set_xlim(-300, 550)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel(x_label, fontsize=10)
        ax.yaxis.set_visible(False)

    cbar = fig.colorbar(im, ax=axes[3], orientation='vertical', label="dF/F intensity")
    fig.suptitle(main_title)
    plt.tight_layout()
    plt.show()