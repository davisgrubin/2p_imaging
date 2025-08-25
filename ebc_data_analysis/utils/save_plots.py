import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

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

                # CR_color = "blue" if CR_stat[id] == 0 elif CR_stat[id] == 1"red"
                if CR_stat[id] == 0:
                    CR_color = 'blue'
                
                if CR_stat[id] == 1:
                    CR_color = 'red'

                if CR_stat[id] == 2:
                    CR_color = 'purple'
                    # breakpoint()

                block_color = "blue" if trials[id]["trial_type"][()] == 1 else "lime"

                ax.plot(fec_time_0[id], fec_0[id], label=f"FEC of trial {id}", color=CR_color)
                ax.axvspan(0, trials[id]["LED"][1] - trials[id]["LED"][0],
                           color="gray", alpha=0.5, label="LED")
                ax.axvspan(trials[id]["AirPuff"][0] - trials[id]["LED"][0],
                           trials[id]["AirPuff"][1] - trials[id]["LED"][0],
                           color=block_color, alpha=0.5, label="AirPuff")

                ax.set_title(f"Trial {id} FEC", fontsize=10)
                ax.set_xlabel("Time for LED Onset(ms)", fontsize=8)
                ax.set_ylabel("FEC", fontsize=8)
                ax.set_xlim(-100, 600)
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
    x_min, x_max = -100, 550

    with PdfPages(pdf_filename) as pdf:
        for roi in range(len(short_crp_avg_dff)):
            fig, axs = plt.subplots(1, 2, figsize=(15, 6))

            # Short Trials: Calculate y-axis limits dynamically
            id = list(short_crp_aligned_time.keys())[0]
            time_short = short_crp_aligned_time[list(short_crp_aligned_time.keys())[0]]
            mask_short = (time_short >= x_min) & (time_short <= x_max)

            time_long = long_crp_aligned_time[list(long_crp_aligned_time.keys())[0]]
            mask_long = (time_long >= x_min) & (time_long <= x_max)


            y_min = min(
                np.min(short_crp_avg_dff[roi][mask_short] - short_crp_sem_dff[roi][mask_short]),
                np.min(short_crn_avg_dff[roi][mask_short] - short_crn_sem_dff[roi][mask_short]),
                np.min(long_crp_avg_dff[roi][mask_long] - long_crp_sem_dff[roi][mask_long]),
                np.min(long_crn_avg_dff[roi][mask_long] - long_crn_sem_dff[roi][mask_long])
            )
            y_max = max(
                np.max(short_crp_avg_dff[roi][mask_short] + short_crp_sem_dff[roi][mask_short]),
                np.max(short_crn_avg_dff[roi][mask_short] + short_crn_sem_dff[roi][mask_short]),
                np.max(long_crp_avg_dff[roi][mask_long] + long_crp_sem_dff[roi][mask_long]),
                np.max(long_crn_avg_dff[roi][mask_long] + long_crn_sem_dff[roi][mask_long])
            )

            axs[0].set_xlim(x_min, x_max)
            axs[0].set_ylim(y_min, y_max)
            axs[0].set_title(f"ROI {roi} - Short Trials", fontsize=14)
            axs[0].set_xlabel("Time for LED Onset(ms)", fontsize=12)
            axs[0].set_ylabel("Mean df/f (+/- SEM) over all trials", fontsize=12)
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


            axs[1].set_xlim(x_min, x_max)
            axs[1].set_ylim(y_min, y_max)
            axs[1].set_title(f"ROI {roi} - Long Trials", fontsize=14)
            axs[1].set_xlabel("Time for LED Onset(ms)", fontsize=12)
            axs[1].set_ylabel("Mean df/f (+/- SEM) over all trials", fontsize=12)
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

def save_roi_plots_to_pdf_sig(short_crp_avg_dff, short_crn_avg_dff, short_crp_sem_dff, short_crn_sem_dff,
                          short_crp_aligned_time, long_crp_avg_dff, long_crn_avg_dff, long_crp_sem_dff, long_crn_sem_dff,
                          long_crp_aligned_time, trials, pdf_filename, ROI_list):
    x_min, x_max = -100, 550

    with PdfPages(pdf_filename) as pdf:
        for roi in ROI_list:
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
            axs[0].set_ylabel("Mean df/f (+/- SEM) over all trials", fontsize=12)
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
            axs[1].set_ylabel("Mean df/f (+/- SEM) over all trials", fontsize=12)
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
