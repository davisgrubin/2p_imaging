import os
from matplotlib.gridspec import GridSpec
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import h5py
from collections import Counter


from utils.indication import intervals
from utils.functions import *
from utils.alignment import *
from utils.indication import *
from plotting.plots import *
from plotting.plots import plot_masks_functions
from utils.save_plots import *

from plotting.plot_values import compute_fec_CR_data, compute_fec_averages
from plotting.plots import plot_histogram, plot_scatter, plot_hexbin, plot_fec_trial


def sig_pdf_name(session_date, sig_event):
    return f"./outputs/{session_date}/significant_ROIs/individual_{sig_event}_sig_roi.pdf"

static_threshold = 0.8 
poor_threshold = 0.004
min_FEC = 0.3

mice = [folder for folder in os.listdir('./data/imaging/') if folder != ".DS_Store"]

for mouse_name in mice:

    if not os.path.exists(f'./data/imaging/{mouse_name}'):
        os.mkdir(f'./data/imaging/{mouse_name}')
        print(f"data folder '{mouse_name}' created.")
    else:
        print(f"data folder '{mouse_name}' already exists.")
    if not os.path.exists(f'./outputs/imaging/{mouse_name}'):
        os.mkdir(f'./outputs/imaging/{mouse_name}')
        print(f"output folder '{mouse_name}' created.")
    else:
        print(f"output folder '{mouse_name}' already exists.")
    session_folder = [folder for folder in os.listdir(f'./data/imaging/{mouse_name}/') if folder != ".DS_Store"]

    for session_date in session_folder:
        # ######
        # if '1101' in session_date:
        #     print('1101')
        #     static_threshold = 0.1
        #     min_FEC = 0
        # if '1104' in session_date:
        #     print('1104')
        #     static_threshold = 0.015
        #     min_FEC = 0.0
        # if '1107' in session_date:
        #     print('1107')
        #     static_threshold = 0.015
        #     min_FEC = 0.55
        # ######
        if session_date[0]=='.' or session_date[0]=='i':
            continue
        if not os.path.exists(f'./outputs/imaging/{mouse_name}/{session_date}'):
            os.mkdir(f'./outputs/imaging/{mouse_name}/{session_date}')
            print(f"output folder '{session_date}' created.")
        else:
            print(f"output folder '{session_date}' already exists.")
        print(mouse_name, session_date)
        # try:
        overal_summary_file = f"./outputs/imaging/{mouse_name}/{session_date}/summary.pdf"
        individual_roi_pdf = f"./outputs/imaging/{mouse_name}/{session_date}/individual_roi.pdf"
        individual_fec_pdf = f"./outputs/imaging/{mouse_name}/{session_date}/individual_fec.pdf"
        sig_summary_file = f"./outputs/imaging/{mouse_name}/{session_date}/sig_summary.pdf"

        # if os.path.exists(f"./data/imaging/{mouse_name}/{session_date}/saved_trials.h5"):
        mask_file = f"./data/imaging/{mouse_name}/{session_date}/masks.h5"
        trials = h5py.File(f"./data/imaging/{mouse_name}/{session_date}/saved_trials.h5")["trial_id"]
    
        init_time, init_index, ending_time, ending_index, led_index, ap_index = aligning_times(trials=trials)
        fec, fec_time_0, _ = fec_zero(trials)
        fec_0 = moving_average(fec , window_size=7)
        # print(fec_0)
        # breakpoint()
        fec_normed = fec_0
        shorts, longs = block_type(trials)
        CR_stat, CR_interval_avg, base_line_avg, cr_interval_idx, bl_interval_idx = CR_stat_indication(trials, fec_normed, fec_time_0, poor_threshold, static_threshold, AP_delay = 3)
        short_CRp_fec, short_CRn_fec, long_CRp_fec, long_CRn_fec = block_and_CR_fec(CR_stat,fec_0, shorts, longs)
        short_CRp_fec_normed, short_CRn_fec_normed, long_CRp_fec_normed, long_CRn_fec_normed = block_and_CR_fec(CR_stat,fec_normed, shorts, longs)
        all_id = sort_numbers_as_strings(shorts + longs)
        event_diff, ap_diff , ending_diff = index_differences(init_index , led_index, ending_index, ap_index)

        # print(CR_stat)
        # breakpoint()
        # print(shorts)
        # breakpoint()

        short_crp_aligned_dff , short_crp_aligned_time = aligned_dff(trials,shorts,CR_stat, 1, init_index, ending_index, shorts[0])
        # print('hello 0')
        # breakpoint()
        short_crn_aligned_dff , short_crn_aligned_time = aligned_dff(trials,shorts,CR_stat, 0, init_index, ending_index, shorts[0])

        # print('hello 1')
        # breakpoint()
        long_crp_aligned_dff , long_crp_aligned_time = aligned_dff(trials,longs,CR_stat, 1, init_index, ending_index, longs[0])
        # print('hello 2')
        # breakpoint()
        long_crn_aligned_dff , long_crn_aligned_time = aligned_dff(trials,longs,CR_stat, 0, init_index, ending_index, longs[0])
        # print(1, short_crp_aligned_dff)
        # print(2, short_crn_aligned_time)

        short_crp_avg_pooled, short_crp_sem_pooled, n_short_crp_pooled = calculate_average_dff_pool(short_crp_aligned_dff)
        short_crn_avg_pooled, short_crn_sem_pooled, n_short_crn_pooled = calculate_average_dff_pool(short_crn_aligned_dff)
        long_crp_avg_pooled,   long_crp_sem_pooled, n_long_crp_pooled = calculate_average_dff_pool(long_crp_aligned_dff)
        long_crn_avg_pooled,   long_crn_sem_pooled, n_long_crn_pooled = calculate_average_dff_pool(long_crn_aligned_dff)

        short_crp_avg_dff, short_crp_sem_dff, n_short_crp_roi = calculate_average_dff_roi(aligned_dff=short_crp_aligned_dff)
        short_crn_avg_dff, short_crn_sem_dff, n_short_crn_roi = calculate_average_dff_roi(aligned_dff=short_crn_aligned_dff)
        long_crp_avg_dff,   long_crp_sem_dff, n_long_crp_roi = calculate_average_dff_roi(aligned_dff=long_crp_aligned_dff)
        long_crn_avg_dff,   long_crn_sem_dff, n_long_crn_roi = calculate_average_dff_roi(aligned_dff=long_crn_aligned_dff)

        short_crp_avg_roi, short_crp_sem_roi = average_over_roi(short_crp_avg_dff)
        short_crn_avg_roi, short_crn_sem_roi = average_over_roi(short_crn_avg_dff)
        long_crp_avg_roi, long_crp_sem_roi =   average_over_roi(long_crp_avg_dff)
        long_crn_avg_roi, long_crn_sem_roi =   average_over_roi(long_crn_avg_dff)

        # the idea here is to use 120 seconds with each time step being 33.6
        interval_window_led = 3
        interval_window_cr = 3
        interval_window_ap = 3
        interval_window_bl = 3

        cr_interval_short_crn, led_interval_short_crn, ap_interval_short_crn, base_line_interval_short_crn = intervals(
            short_crn_aligned_dff, led_index, ap_index, interval_window_led, interval_window_cr, interval_window_ap, interval_window_bl, isi_time=200)

        cr_interval_short_crp, led_interval_short_crp, ap_interval_short_crp, base_line_interval_short_crp = intervals(
            short_crp_aligned_dff, led_index, ap_index, interval_window_led, interval_window_cr, interval_window_ap, interval_window_bl, isi_time=200)

        cr_interval_long_crn, led_interval_long_crn, ap_interval_long_crn, base_line_interval_long_crn = intervals(
            long_crn_aligned_dff, led_index, ap_index, interval_window_led, interval_window_cr, interval_window_ap, interval_window_bl, isi_time=400 )

        cr_interval_long_crp, led_interval_long_crp, ap_interval_long_crp, base_line_interval_long_crp = intervals(
            long_crp_aligned_dff, led_index, ap_index, interval_window_led, interval_window_cr, interval_window_ap, interval_window_bl, isi_time=400)

        trial_types = {
            "Short CRN": {"baseline": interval_averaging(base_line_interval_short_crn),"led": interval_averaging(led_interval_short_crn),"cr": interval_averaging(cr_interval_short_crn),"ap": interval_averaging(ap_interval_short_crn),"color": "blue"},
            "Short CRP": {"baseline": interval_averaging(base_line_interval_short_crp),"led": interval_averaging(led_interval_short_crp),"cr": interval_averaging(cr_interval_short_crp),"ap": interval_averaging(ap_interval_short_crp),"color": "red"},
            "Long CRN": {"baseline": interval_averaging(base_line_interval_long_crn),"led": interval_averaging(led_interval_long_crn),"cr": interval_averaging(cr_interval_long_crn),"ap": interval_averaging(ap_interval_long_crn),"color": "blue"},
            "Long CRP": {"baseline": interval_averaging(base_line_interval_long_crp),"led": interval_averaging(led_interval_long_crp),"cr": interval_averaging(cr_interval_long_crp),"ap": interval_averaging(ap_interval_long_crp),"color": "red"},
        }

        valid_ROIs = {trial_type: {event: [] for event in ["led", "cr", "ap"]} for trial_type in trial_types}
        for trial_type, data in trial_types.items():
            baseline_avg = data["baseline"]
            for event in ["led", "cr", "ap"]:
                event_avg = data[event]
                for roi, event_values in event_avg.items():
                    baseline_value = baseline_avg.get(roi, np.nan)
                    if np.nanmean(event_values) > np.nanmean(baseline_value):
                        valid_ROIs[trial_type][event].append(roi)


        t_stat_short_crn_led, p_value_short_crn_led = ttest_intervals(base_interval=base_line_interval_short_crn, interval_under_test=led_interval_short_crn, roi_list=valid_ROIs["Short CRN"]["led"])
        t_stat_short_crn_ap, p_value_short_crn_ap = ttest_intervals(base_interval=base_line_interval_short_crn, interval_under_test=ap_interval_short_crn, roi_list=valid_ROIs["Short CRN"]["ap"])
        t_stat_short_crn_cr, p_value_short_crn_cr = ttest_intervals(base_interval=base_line_interval_short_crn, interval_under_test=cr_interval_short_crn, roi_list=valid_ROIs["Short CRN"]["cr"])

        t_stat_short_crp_led, p_value_short_crp_led = ttest_intervals(base_interval=base_line_interval_short_crp, interval_under_test=led_interval_short_crp, roi_list=valid_ROIs["Short CRP"]["led"])
        t_stat_short_crp_ap, p_value_short_crp_ap = ttest_intervals(base_interval=base_line_interval_short_crp, interval_under_test=ap_interval_short_crp, roi_list=valid_ROIs["Short CRP"]["ap"])
        t_stat_short_crp_cr, p_value_short_crp_cr = ttest_intervals(base_interval=base_line_interval_short_crp, interval_under_test=cr_interval_short_crp, roi_list=valid_ROIs["Short CRP"]["cr"])

        t_stat_long_crn_led, p_value_long_crn_led = ttest_intervals(base_interval=base_line_interval_long_crn, interval_under_test=led_interval_long_crn, roi_list=valid_ROIs["Long CRN"]["led"])
        t_stat_long_crn_ap, p_value_long_crn_ap = ttest_intervals(base_interval=base_line_interval_long_crn, interval_under_test=ap_interval_long_crn, roi_list=valid_ROIs["Long CRN"]["ap"])
        t_stat_long_crn_cr, p_value_long_crn_cr = ttest_intervals(base_interval=base_line_interval_long_crn, interval_under_test=cr_interval_long_crn, roi_list=valid_ROIs["Long CRN"]["cr"])

        t_stat_long_crp_led, p_value_long_crp_led = ttest_intervals(base_interval=base_line_interval_long_crp, interval_under_test=led_interval_long_crp, roi_list=valid_ROIs["Long CRP"]["led"])
        t_stat_long_crp_ap, p_value_long_crp_ap = ttest_intervals(base_interval=base_line_interval_long_crp, interval_under_test=ap_interval_long_crp, roi_list=valid_ROIs["Long CRP"]["ap"])
        t_stat_long_crp_cr, p_value_long_crp_cr = ttest_intervals(base_interval=base_line_interval_long_crp, interval_under_test=cr_interval_long_crp, roi_list=valid_ROIs["Long CRP"]["cr"])

        t_avg_short_crn_led = calculate_average_ttest(t_stat_short_crn_led)
        t_avg_short_crn_ap = calculate_average_ttest(t_stat_short_crn_ap)
        t_avg_short_crn_cr = calculate_average_ttest(t_stat_short_crn_cr)

        t_avg_short_crp_led = calculate_average_ttest(t_stat_short_crp_led)
        t_avg_short_crp_ap = calculate_average_ttest(t_stat_short_crp_ap)
        t_avg_short_crp_cr = calculate_average_ttest(t_stat_short_crp_cr)

        t_avg_long_crn_led = calculate_average_ttest(t_stat_long_crn_led)
        t_avg_long_crn_ap = calculate_average_ttest(t_stat_long_crn_ap)
        t_avg_long_crn_cr = calculate_average_ttest(t_stat_long_crn_cr)

        t_avg_long_crp_led = calculate_average_ttest(t_stat_long_crp_led)
        t_avg_long_crp_ap = calculate_average_ttest(t_stat_long_crp_ap)
        t_avg_long_crp_cr = calculate_average_ttest(t_stat_long_crp_cr)

        t_stats = {
            "led": [t_avg_short_crn_led, t_avg_short_crp_led, t_avg_long_crn_led, t_avg_long_crp_led],
            "ap": [t_avg_short_crn_ap, t_avg_short_crp_ap, t_avg_long_crn_ap, t_avg_long_crp_ap],
            "cr": [t_avg_short_crn_cr, t_avg_short_crp_cr, t_avg_long_crn_cr, t_avg_long_crp_cr],
        }

        common_rois = {event: Counter(extract_top_rois(t_stats_list)).most_common(7) for event, t_stats_list in t_stats.items()}

        # Extract top ROI IDs
        led_roi = [int(roi) for roi, _ in common_rois["led"]]
        ap_roi = [int(roi) for roi, _ in common_rois["ap"]]
        cr_roi = [int(roi) for roi, _ in common_rois["cr"]]

        sig_rois = {}
        sig_rois["led"] = led_roi
        sig_rois["ap"] = ap_roi
        sig_rois["cr"] = cr_roi
        print(f"sig rois for led:{led_roi}")
        print(f"sig rois for cr:{cr_roi}")

        # print(short_crp_aligned_dff)
        short_crp_avg_led_sig, short_crp_sem_led_sig, short_crp_count_led_sig = calculate_average_sig(short_crp_aligned_dff, roi_indices=led_roi)
        short_crn_avg_led_sig, short_crn_sem_led_sig, short_crn_count_led_sig = calculate_average_sig(short_crn_aligned_dff, roi_indices=led_roi)
        long_crp_avg_led_sig, long_crp_sem_led_sig, long_crp_count_led_sig = calculate_average_sig(long_crp_aligned_dff , roi_indices=led_roi)
        long_crn_avg_led_sig, long_crn_sem_led_sig, long_crn_count_led_sig = calculate_average_sig(long_crn_aligned_dff , roi_indices=led_roi)
        short_crp_avg_ap_sig, short_crp_sem_ap_sig, short_crp_count_ap_sig = calculate_average_sig(short_crp_aligned_dff, roi_indices=ap_roi)
        short_crn_avg_ap_sig, short_crn_sem_ap_sig, short_crn_count_ap_sig = calculate_average_sig(short_crn_aligned_dff, roi_indices=ap_roi)
        long_crp_avg_ap_sig, long_crp_sem_ap_sig, long_crp_count_ap_sig = calculate_average_sig(long_crp_aligned_dff , roi_indices=ap_roi)
        long_crn_avg_ap_sig, long_crn_sem_ap_sig, long_crn_count_ap_sig = calculate_average_sig(long_crn_aligned_dff , roi_indices=ap_roi)
        short_crp_avg_cr_sig, short_crp_sem_cr_sig, short_crp_count_cr_sig = calculate_average_sig(short_crp_aligned_dff, roi_indices=cr_roi)
        short_crn_avg_cr_sig, short_crn_sem_cr_sig, short_crn_count_cr_sig = calculate_average_sig(short_crn_aligned_dff, roi_indices=cr_roi)
        long_crp_avg_cr_sig, long_crp_sem_cr_sig, long_crp_count_cr_sig = calculate_average_sig(long_crp_aligned_dff , roi_indices=cr_roi)
        long_crn_avg_cr_sig, long_crn_sem_cr_sig, long_crn_count_cr_sig = calculate_average_sig(long_crn_aligned_dff , roi_indices=cr_roi)

        try:
            print("plotting sig roi plots")
            save_roi_plots_to_pdf_sig(short_crp_avg_dff, short_crn_avg_dff, short_crp_sem_dff, short_crn_sem_dff,
                                    short_crp_aligned_time, long_crp_avg_dff, long_crn_avg_dff, long_crp_sem_dff, long_crn_sem_dff,
                                    long_crp_aligned_time, trials, pdf_filename = sig_pdf_name(session_date, sig_event="LED"), ROI_list=led_roi)

            save_roi_plots_to_pdf_sig(short_crp_avg_dff, short_crn_avg_dff, short_crp_sem_dff, short_crn_sem_dff,
                                    short_crp_aligned_time, long_crp_avg_dff, long_crn_avg_dff, long_crp_sem_dff, long_crn_sem_dff,
                                    long_crp_aligned_time, trials, pdf_filename = sig_pdf_name(session_date, sig_event="AP"), ROI_list=ap_roi)

            save_roi_plots_to_pdf_sig(short_crp_avg_dff, short_crn_avg_dff, short_crp_sem_dff, short_crn_sem_dff,
                                    short_crp_aligned_time, long_crp_avg_dff, long_crn_avg_dff, long_crp_sem_dff, long_crn_sem_dff,
                                    long_crp_aligned_time, trials, pdf_filename = sig_pdf_name(session_date, sig_event="CR"), ROI_list=cr_roi)
        except:
            print("there is a problem with sig roi plotting")
        # try:
            save_roi_plots_to_pdf(short_crp_avg_dff, short_crn_avg_dff, short_crp_sem_dff, short_crn_sem_dff,
                                short_crp_aligned_time, long_crp_avg_dff, long_crn_avg_dff, long_crp_sem_dff, long_crn_sem_dff,
                                long_crp_aligned_time, trials, pdf_filename = individual_roi_pdf)

            save_fec_plots_to_pdf(trials, fec_time_0, fec_0, CR_stat,all_id, individual_fec_pdf)

            breakpoint()
        # except:
            print("problem with roi dff or trial fec plotting")

        with PdfPages(filename=sig_summary_file) as sig_summary_pdf:
            fig, axs = plt.subplots(5, 3, figsize=(21, 35))

            # Assign specific axes for the first set of plots
            ax0 = axs[0, 0]
            ax1 = axs[0, 1]
            ax2 = axs[1, 0]
            ax3 = axs[1, 1]
            ax4 = axs[2, 0]
            ax5 = axs[2, 1]

            # Plot using `plot_trial_averages_sig` for the first set of axes
            # print(short_crp_avg_led_sig)
            plot_trial_averages_sig(trials, short_crp_aligned_time, short_crp_avg_led_sig, short_crp_sem_led_sig, short_crn_avg_led_sig, short_crn_sem_led_sig, title_suffix="Short", event="LED", pooled=True, ax=ax0)
            plot_trial_averages_sig(trials, long_crp_aligned_time, long_crp_avg_led_sig, long_crp_sem_led_sig, long_crn_avg_led_sig, long_crn_sem_led_sig, title_suffix="Long", event="LED", pooled=True, ax=ax1)
            plot_trial_averages_sig(trials, short_crp_aligned_time, short_crp_avg_ap_sig, short_crp_sem_ap_sig, short_crn_avg_ap_sig, short_crn_sem_ap_sig, title_suffix="Short", event="AirPuff", pooled=True, ax=ax2)
            plot_trial_averages_sig(trials, long_crp_aligned_time, long_crp_avg_ap_sig, long_crp_sem_ap_sig, long_crn_avg_ap_sig, long_crn_sem_ap_sig, title_suffix="Long", event="AP", pooled=True, ax=ax3)
            plot_trial_averages_sig(trials, short_crp_aligned_time, short_crp_avg_cr_sig, short_crp_sem_cr_sig, short_crn_avg_cr_sig, short_crn_sem_cr_sig, title_suffix="Short", event="CR", pooled=True, ax=ax4)
            plot_trial_averages_sig(trials, long_crp_aligned_time, long_crp_avg_cr_sig, long_crp_sem_cr_sig, long_crn_avg_cr_sig, long_crn_sem_cr_sig, title_suffix="Long", event="CR", pooled=True, ax=ax5)

            # fig, axs = plt.subplots(4, 3, figsize=(15, 40))

            # Style adjustments for all axes
            for ax in axs.flat:
                ax.spines['top'].set_visible(False)  # Hide the top spine
                ax.spines['right'].set_visible(False)  # Hide the right spine
                ax.yaxis.set_ticks_position('left')  # Show ticks only on the left
                ax.xaxis.set_ticks_position('bottom')  # Show ticks only on the bottom

            # Hide axes for the correct positions (third column for specific rows)
            for i, ax_row in enumerate(axs):
                for j, ax in enumerate(ax_row):
                    if i not in [3, 4, 5, 6] and j == 2:  # Hide axes in the third column for specific rows
                        ax.axis("off")

            # Additional scatter plots for trial types and events
            for idx, (trial_type, data) in enumerate(trial_types.items()):
                baseline_avg = data["baseline"]
                color_other = "blue" if "CRN" in trial_type else "red"
                for event_idx, event in enumerate(["led", "cr", "ap"]):
                    if idx in [0,1]:
                        idxx = 0
                    else:
                        idxx = 1
                    ax = axs[idxx + 3, event_idx]  # Access the correct axis directly from axs grid
                    event_avg = data[event]
                    baseline_values = []
                    event_values = []
                    colors = []
                    markers = []
                    for roi, event_values_array in event_avg.items():
                        baseline_value = baseline_avg.get(roi, np.nan)
                        event_value_mean = np.nanmean(event_values_array)
                        baseline_value_mean = np.nanmean(baseline_value)
                        baseline_values.append(baseline_value_mean)
                        event_values.append(event_value_mean)
                        
                        if roi in valid_ROIs[trial_type][event] and roi in sig_rois[event]:
                            if color_other == 'blue':
                                colors.append('cyan')
                            else:
                                colors.append('orange')
                            # colors.append(color_other)
                            markers.append('*')
                        else:
                            colors.append(color_other)
                            markers.append('o')

                    event_over = np.nanmean(event_values)
                    baseline_over = np.nanmean(baseline_values)

                    ax.scatter(baseline_over, event_over, marker = '+', s = 400, color=color_other)

                    ax.scatter(baseline_values, event_values, c=colors, alpha=0.7, edgecolor="none")
                    event_title = event
                    if event == "ap":
                        event_title = 'AirPuff'
                    else:
                        evnet_title = event
                    ax.set_title(f"{trial_type} - {event_title} Event")
                    ax.set_xlabel("Baseline Average")
                    ax.set_ylabel(f"Evoked signal for {event.capitalize()} event Average")
                    ax.axline((0, 0), slope=1, color="gray", linestyle="--")  # Line y=x for reference
                    ax.legend(
                        handles=[
                            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='cyan', markersize=8, label="Significant ROI CR-"),
                            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=8, label="Significant ROI CR+"),
                            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label="Non-significant ROI CR-"),
                            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label="Non-significant ROI CR+"),
                            # plt.Line2D([0], [0], marker='+', color='w', markerfacecolor='red', markersize=8, label="Overall average CR+"),
                            # plt.Line2D([0], [0], marker='+', color='w', markerfacecolor='blue', markersize=8, label="Overall average CR-")
                        ],
                        loc="upper left"
                    )
                    ax.grid(False)
            plt.tight_layout()
            sig_summary_pdf.savefig(fig)
            plt.close()
        ######################## overall summary


        data_fec_scatter = compute_fec_CR_data(base_line_avg, CR_interval_avg, CR_stat)
        data_fec_average = compute_fec_averages(short_CRp_fec, short_CRn_fec, long_CRp_fec, long_CRn_fec, fec_time_0, shorts, longs, trials)
        data_fec_average_normed = compute_fec_averages(short_CRp_fec_normed, short_CRn_fec_normed,
                    long_CRp_fec_normed, long_CRn_fec_normed, fec_time_0, shorts, longs, trials)

        baselines = data_fec_scatter['baselines']
        cr_amplitudes = data_fec_scatter['cr_amplitudes']
        cr_relative_changes = data_fec_scatter['cr_relative_changes']
        baselines_crp = data_fec_scatter['baselines_crp']
        baselines_crn = data_fec_scatter['baselines_crn']
        cr_amplitudes_crp = data_fec_scatter['cr_amplitudes_crp']
        cr_amplitudes_crn = data_fec_scatter['cr_amplitudes_crn']
        cr_relative_changes_crp = data_fec_scatter['cr_relative_changes_crp']
        cr_relative_changes_crn = data_fec_scatter['cr_relative_changes_crn']
        all_baselines = data_fec_scatter['all_baselines']
        all_relative_changes = data_fec_scatter['all_relative_changes']

        # Sort and create heatmaps for all datasets
        sorted_avg_short_crp_roi = sort_dff_avg(short_crp_avg_dff, event_diff -3, event_diff + 18)
        sorted_avg_long_crp_roi = sort_dff_avg(long_crp_avg_dff, event_diff-3, event_diff + 18)
        sorted_avg_short_crn_roi = sort_dff_avg(short_crn_avg_dff, event_diff-3, event_diff + 18)
        sorted_avg_long_crn_roi = sort_dff_avg(long_crn_avg_dff, event_diff-3, event_diff + 18)

        # Create heat arrays
        heat_arrays_avg = []
        for sorted_avg_rois, dff in [
            (reversed(sorted_avg_short_crp_roi), short_crp_avg_dff),
            (reversed(sorted_avg_long_crp_roi), long_crp_avg_dff),
            (reversed(sorted_avg_short_crn_roi), short_crn_avg_dff),
            (reversed(sorted_avg_long_crn_roi), long_crn_avg_dff),
        ]:
            heat_array_0 = [dff[roi] for roi in list(sorted_avg_rois)]
            heat_arrays_avg.append(np.vstack(heat_array_0))

        # Create heat arrays
        heat_arrays_avg1 = []
        for sorted_avg_rois, dff in [
            (reversed(sorted_avg_long_crp_roi), short_crp_avg_dff),
            (reversed(sorted_avg_short_crp_roi), long_crp_avg_dff),
            (reversed(sorted_avg_long_crn_roi), short_crn_avg_dff),
            (reversed(sorted_avg_short_crn_roi), long_crn_avg_dff),
        ]:
            heat_array_0 = [dff[roi] for roi in list(sorted_avg_rois)]
            heat_arrays_avg1.append(np.vstack(heat_array_0))

        sorted_max_short_crp_roi = sort_dff_max_index(short_crp_avg_dff, event_diff, event_diff + 12)
        sorted_max_short_crn_roi = sort_dff_max_index(short_crn_avg_dff, event_diff, event_diff + 12)
        sorted_max_long_crp_roi =  sort_dff_max_index(long_crp_avg_dff, event_diff, event_diff + 12)
        sorted_max_long_crn_roi =  sort_dff_max_index(long_crn_avg_dff, event_diff, event_diff + 12)
        print(sorted_max_long_crp_roi)
        # breakpoint()

        # Create heat arrays
        heat_arrays_max = []
        for sorted_max_rois, dff in [
            ((sorted_max_short_crp_roi), short_crp_avg_dff),
            ((sorted_max_long_crp_roi), long_crp_avg_dff),
            ((sorted_max_short_crn_roi), short_crn_avg_dff),
            ((sorted_max_long_crn_roi), long_crn_avg_dff),
        ]:
            heat_array_0 = [dff[roi] for roi in sorted_max_rois]
            heat_arrays_max.append(np.vstack(heat_array_0))

        # Create heat arrays
        heat_arrays_max1 = []
        for sorted_max_rois, dff in [
            ((sorted_max_long_crp_roi), short_crp_avg_dff),
            ((sorted_max_short_crp_roi), long_crp_avg_dff),
            ((sorted_max_long_crn_roi), short_crn_avg_dff),
            ((sorted_max_short_crn_roi), long_crn_avg_dff),
        ]:
            heat_array_0 = [dff[roi] for roi in sorted_max_rois]
            heat_arrays_max1.append(np.vstack(heat_array_0))

        aligned_times = [
            short_crp_aligned_time,
            long_crp_aligned_time,
            short_crn_aligned_time,
            long_crn_aligned_time
        ]

        heatmap_titles_avg = ["Sorted :average CR window short - Short CR+", "Sorted : average CR window long - Long CR+", "Sorted : average CR window short - Short CR-", "Sorted : average CR window long - Long CR-"]
        heatmap_titles_avg1 = ["Sorted :average CR window long - Short CR+", "Sorted : average CR window short - Long CR+", "Sorted : average CR window long - Short CR-", "Sorted : average CR window short - Long CR-"]
        heatmap_titles_max = ["Sorted :max CR window short - Short CR+", "Sorted : max CR window long - Long CR+", "Sorted : max CR window short - Short CR-", "Sorted : max CR window long - Long CR-"]
        heatmap_titles_max1 = ["Sorted :max CR window long - Short CR+", "Sorted : max CR window short - Long CR+", "Sorted : max CR window long - Short CR-", "Sorted : max CR window short - Long CR-"]
        color_maps = ["magma", "magma", "viridis", "viridis"]


        metadata = {
            'Title': f"Overall Summary of {session_date}",  # Set the PDF title here
            'Author': 'Shayan Malekpour',  # Optional: Add author metadata
            'Subject': 'Summary of Analysis',  # Optional: Add subject metadata
            'Keywords': 'FEC, CR, Baseline, Analysis'  # Optional: Add keywords
        }

        with PdfPages(filename=overal_summary_file, metadata=metadata) as summary_pdf:
            y = 15
            x = 6
            # fig, axes = plt.subplots(y, x, figsize=(7*x, 7*y), gridspec_kw={'width_ratios': [1, 1, 0.03], 'height_ratios': [2,1,1,1,1,1,1,1,1,1,1,1,1,1]}, squeeze=False)
            
            spacing = 20  # Number of grid spaces for spacing between plots
            cbar_width = 0.06

            # Create the figure and GridSpec
            fig = plt.figure(figsize=(x * 7, y * 7))
            gs = GridSpec(
                y * 100 + spacing * (y - 1), 
                x * 100 + spacing * (x - 1), 
                figure=fig
            )

            # Define subplots with spacing
            def remove_spines(ax):
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)



            # Subplot definitions
            Ax1 = fig.add_subplot(gs[0:200, 0:200])  # First plot in the first row
            remove_spines(Ax1)

            Ax2 = fig.add_subplot(gs[0:200, 200 + spacing:400 + spacing])  # Second plot in the first row
            remove_spines(Ax2)

            Ax3 = fig.add_subplot(gs[0:200, 400 + spacing: 600 + spacing])
            remove_spines(Ax3)  # Color bar after Ax2

            ax10 = fig.add_subplot(gs[200 + spacing:300 + spacing, 0:100])  # First plot in the second row
            remove_spines(ax10)

            ax11 = fig.add_subplot(gs[200 + spacing:300 + spacing, 100 + spacing:200 + spacing])  # Second plot
            remove_spines(ax11)

            # ax12 = fig.add_subplot(gs[200 + spacing:300 + spacing, 200 + 2 * spacing:int(200 + 2 * spacing + cbar_width * 100)])
            # remove_spines(ax12)  # Color bar after ax11

            ax20 = fig.add_subplot(gs[300 + 2 * spacing:400 + 2 * spacing, 0:100])  # Third row, first column
            remove_spines(ax20)

            ax21 = fig.add_subplot(gs[300 + 2 * spacing:400 + 2 * spacing, 100 + spacing:200 + spacing])  # Third row, second column
            remove_spines(ax21)

            # ax22 = fig.add_subplot(gs[300 + 2 * spacing:400 + 2 * spacing, 200 + 2 * spacing:int(200 + 2 * spacing + cbar_width * 100)])
            # remove_spines(ax22)  # Color bar

            ax30 = fig.add_subplot(gs[400 + 3 * spacing:500 + 3 * spacing, 0:100])  # Fourth row, first column
            remove_spines(ax30)

            ax31 = fig.add_subplot(gs[400 + 3 * spacing:500 + 3 * spacing, 100 + spacing:200 + spacing])  # Fourth row, second column
            remove_spines(ax31)

            # ax32 = fig.add_subplot(gs[400 + 3 * spacing:500 + 3 * spacing, 200 + 2 * spacing:int(200 + 2 * spacing + cbar_width * 100)])
            # remove_spines(ax32)  # Color bar

            ax40 = fig.add_subplot(gs[500 + 4 * spacing:600 + 4 * spacing, 0:100])  # Fifth row, first column
            remove_spines(ax40)

            ax41 = fig.add_subplot(gs[500 + 4 * spacing:600 + 4 * spacing, 100 + spacing:200 + spacing])  # Sixth row, second column
            remove_spines(ax41)

            ax42 = fig.add_subplot(gs[500 + 4 * spacing:600 + 4 * spacing, 200 + 2 * spacing:300 + 2 * spacing])  # Seventh row, se
            remove_spines(ax42)

            ax43 = fig.add_subplot(gs[500 + 4 * spacing:600 + 4 * spacing, 300 + 3 * spacing:400 + 3 * spacing])  # Seventh row, sen
            remove_spines(ax43)

            ax44 = fig.add_subplot(gs[500 + 4 * spacing:600 + 4 * spacing, 400 + 4 * spacing:int(400 + 4 * spacing + cbar_width * 100)])
            remove_spines(ax44)  # Color bar

            ax50 = fig.add_subplot(gs[600 + 5 * spacing:700 + 5 * spacing, 0:100])  # Sixth row, first column
            remove_spines(ax50)

            ax51 = fig.add_subplot(gs[600 + 5 * spacing:700 + 5 * spacing, 100 + spacing:200 + spacing])  # Sixth row, second column
            remove_spines(ax51)
            
            ax52 = fig.add_subplot(gs[600 + 5 * spacing:700 + 5 * spacing, 200 + 2 * spacing:300 + 2 * spacing])  # Seventh row, second column
            remove_spines(ax52)

            ax53 = fig.add_subplot(gs[600 + 5 * spacing:700 + 5 * spacing, 300 + 3 * spacing:400 + 3 * spacing])  # Seventh row, second column
            remove_spines(ax53)
            
            ax54 = fig.add_subplot(gs[600 + 5 * spacing:700 + 5 * spacing, 400 + 4 * spacing:int(400 + 4 * spacing + cbar_width * 100)])
            remove_spines(ax54)  # Color bar

            ax60 = fig.add_subplot(gs[700 + 6 * spacing:800 + 6 * spacing, 0:100])  # Seventh row, first column
            remove_spines(ax60)

            ax61 = fig.add_subplot(gs[700 + 6 * spacing:800 + 6 * spacing, 100 + spacing:200 + spacing])  # Sixth row, second columnmn
            remove_spines(ax61)
            
            ax62 = fig.add_subplot(gs[700 + 6 * spacing:800 + 6 * spacing, 200 + 2 * spacing:300 + 2 * spacing])  # Seventh row, secmn
            remove_spines(ax62)

            ax63 = fig.add_subplot(gs[700 + 6 * spacing:800 + 6 * spacing, 300 + 3 * spacing:400 + 3 * spacing])  # Seventh row, secmn
            remove_spines(ax63)
            
            ax64 = fig.add_subplot(gs[700 + 6 * spacing:800 + 6 * spacing, 400 + 4 * spacing:int(400 + 4 * spacing + cbar_width * 100)])
            remove_spines(ax62)  # Color bar

            ax70 = fig.add_subplot(gs[800 + 7 * spacing:900 + 7 * spacing, 0:100])  # Eighth row, first column
            remove_spines(ax70)

            ax71 = fig.add_subplot(gs[800 + 7 * spacing:900 + 7 * spacing, 100 + spacing:200 + spacing])  # Sixth row, second columnn
            remove_spines(ax71)

            ax72 = fig.add_subplot(gs[800 + 7 * spacing:900 + 7 * spacing, 200 + 2 * spacing:300 + 2 * spacing])  # Seventh row, secn
            remove_spines(ax72)

            ax73 = fig.add_subplot(gs[800 + 7 * spacing:900 + 7 * spacing, 300 + 3 * spacing:400 + 3 * spacing])  # Seventh row, secn
            remove_spines(ax73)

            ax74 = fig.add_subplot(gs[800 + 7 * spacing:900 + 7 * spacing, 400 + 4 * spacing:int(400 + 4 * spacing + cbar_width * 100)])
            remove_spines(ax74)  # Color bar

            ax80 = fig.add_subplot(gs[900 + 8 * spacing:1000 + 8 * spacing, 0:100])
            remove_spines(ax80)

            ax81 = fig.add_subplot(gs[900 + 8 * spacing:1000 + 8 * spacing, 100 + spacing:200 + spacing])
            remove_spines(ax81)

            ax90 = fig.add_subplot(gs[1000 + 9 * spacing:1100 + 9 * spacing, 0:100])
            remove_spines(ax90)

            ax91 = fig.add_subplot(gs[1000 + 9 * spacing:1100 + 9 * spacing, 100 + spacing:200 + spacing])
            remove_spines(ax91)

            ax100 = fig.add_subplot(gs[1100 + 10 * spacing:1200 + 10 * spacing, 0:100])
            remove_spines(ax100)

            ax101 = fig.add_subplot(gs[1100 + 10 * spacing:1200 + 10 * spacing, 100 + spacing:200 + spacing])
            remove_spines(ax101)

            # ax102 = fig.add_subplot(gs[1100 + 10 * spacing:1200 + 10 * spacing, 
            #                           200 + 2 * spacing:int(200 + 2 * spacing + cbar_width * 100)])
            # remove_spines(ax102)

            ax110 = fig.add_subplot(gs[1200 + 11 * spacing:1300 + 11 * spacing, 0:100])
            remove_spines(ax110)

            ax111 = fig.add_subplot(gs[1200 + 11 * spacing:1300 + 11 * spacing, 100 + spacing:200 + spacing])
            remove_spines(ax111)

            ax120 = fig.add_subplot(gs[1300 + 12 * spacing:1400 + 12 * spacing, 0:100])
            remove_spines(ax120)

            ax121 = fig.add_subplot(gs[1300 + 12 * spacing:1400 + 12 * spacing, 100 + spacing:200 + spacing])
            remove_spines(ax121)

            ax130 = fig.add_subplot(gs[1400 + 13 * spacing:1500 + 13 * spacing, 0:100])
            remove_spines(ax130)

            ax131 = fig.add_subplot(gs[1400 + 13 * spacing:1500 + 13 * spacing, 100 + spacing:200 + spacing])
            remove_spines(ax131)        
            plot_masks_functions(mask_file, Ax1, Ax2, Ax3)
            # cbar = plt.colorbar(plt.cm.ScalarMappable(cmap="plasma" ), cax=ax02)
            short_data_normed = data_fec_average_normed["short_trials"]

            #mean1 is positive
            plot_fec_trial(
                ax10,
                short_data_normed["time"],
                short_data_normed["mean1"],
                short_data_normed["std1"],
                short_data_normed["mean0"],
                short_data_normed["std0"],
                short_data_normed["led"],
                short_data_normed["airpuff"],
                y_lim = min_FEC,
                title="Normalized FEC Average for Short Trials",
            )

            # Data for long trials
            long_data_normed = data_fec_average_normed["long_trials"]
            plot_fec_trial(
                ax11,
                long_data_normed["time"],
                long_data_normed["mean1"],
                long_data_normed["std1"],
                long_data_normed["mean0"],
                long_data_normed["std0"],
                long_data_normed["led"],
                long_data_normed["airpuff"],
                y_lim = min_FEC,
                title="Normalized FEC Average for Long Trials"
            )

            plot_trial_averages_side_by_side(
                ax20, ax21,  # Axes for plotting
                n_short_crp_roi, n_short_crn_roi, short_crp_aligned_time, short_crp_avg_roi, short_crp_sem_roi, short_crn_avg_roi, short_crn_sem_roi,  # Short trial data
                n_long_crp_roi, n_long_crn_roi, long_crp_aligned_time, long_crp_avg_roi, long_crp_sem_roi, long_crn_avg_roi, long_crn_sem_roi,  # Long trial data
                trials,  # Trial information
                title_suffix1="Short", title_suffix2="Long"  # Titles for plots
            )

            plot_trial_averages_side_by_side(
                ax30, ax31,
                n_short_crp_pooled, n_short_crn_pooled, short_crp_aligned_time, short_crp_avg_pooled, short_crp_sem_pooled,
                short_crn_avg_pooled, short_crn_sem_pooled,
                n_long_crp_pooled, n_long_crn_pooled, long_crp_aligned_time, long_crp_avg_pooled, long_crp_sem_pooled,
                long_crn_avg_pooled, long_crn_sem_pooled,
                trials, title_suffix1="Short", title_suffix2="Long", pooled=True)

            plot_heatmaps_side_by_side(heat_arrays_avg, aligned_times, heatmap_titles_avg, trials, color_maps=color_maps, axes= [ax40, ax43, ax50, ax53])

            #the weird ones
            plot_heatmaps_side_by_side(heat_arrays_avg1, aligned_times, heatmap_titles_avg1, trials, color_maps=color_maps, axes= [ax41, ax42, ax51, ax52])

            cbar = plt.colorbar(plt.cm.ScalarMappable(cmap="magma"), cax=ax44)
            cbar.set_label("dF/F intensity")

            cbar = plt.colorbar(plt.cm.ScalarMappable(cmap="viridis" ), cax=ax54)
            cbar.set_label("dF/F intensity")

            plot_heatmaps_side_by_side(heat_arrays_max, aligned_times, heatmap_titles_max, trials, color_maps=color_maps, axes= [ax60, ax63, ax70, ax73])

            plot_heatmaps_side_by_side(heat_arrays_max1, aligned_times, heatmap_titles_max1, trials, color_maps=color_maps, axes= [ax61, ax62, ax71, ax72])

            cbar = plt.colorbar(plt.cm.ScalarMappable(cmap="magma" ), cax=ax64)
            cbar.set_label("dF/F intensity")

            cbar = plt.colorbar(plt.cm.ScalarMappable(cmap="viridis" ), cax=ax74)
            cbar.set_label("dF/F intensity")

            # Histogram of baseline4
            plot_histogram(
                ax80, baselines, bins=20, color='lime', edgecolor='black', alpha=0.7,
                title='Distribution of Baseline Values Across Sessions',
                xlabel='Baseline Value', ylabel='Frequency'
            )

            # Histogram of CR amplitudes
            plot_histogram(
                ax81, cr_amplitudes, bins=20, color='green', edgecolor='black', alpha=0.7,
                title='Distribution of CR Amplitudes Across Sessions',
                xlabel='CR Value', ylabel='Frequency'
            )

            # Scatter plot for CR+ and CR-
            plot_scatter(
                ax90, baselines_crp, cr_amplitudes_crp, color='red', alpha=0.7, label='CR+',
                title='CR Amplitude (Absolute) vs. Baseline', xlabel='Baseline', ylabel='CR Amplitude (Absolute)'
            )
            plot_scatter(
                ax90, baselines_crn, cr_amplitudes_crn, color='blue', alpha=0.7, label='CR-',
                title='', xlabel='', ylabel=''  # Title and labels already set
            )

            # Scatter plot for relative change (CR+ and CR-)
            plot_scatter(
                ax91, baselines_crp, cr_relative_changes_crp, color='red', alpha=0.7, label='CR+',
                title='CR Size (Relative Change) vs. Baseline', xlabel='Baseline', ylabel='CR Size (Relative Change)'
            )
            plot_scatter(
                ax91, baselines_crn, cr_relative_changes_crn, color='blue', alpha=0.7, label='CR-',
                title='', xlabel='', ylabel=''  # Title and labels already set
            )

            # Hexbin for CR amplitude vs. baseline
            plot_hexbin(
                ax100, baselines, cr_amplitudes, gridsize=30, cmap='Greens', mincnt=1, alpha=1.0,
                colorbar_label='Count',
                title='Joint Distribution of CR Amplitude and Baseline',
                xlabel='Baseline', ylabel='CR Amplitude (Absolute)'
            )

            # Hexbin for relative change vs. baseline
            plot_hexbin(
                ax101, all_baselines, all_relative_changes, gridsize=30, cmap='Greens', mincnt=1, alpha=0.7,
                colorbar_label='Count',
                title='Joint Distribution of CR Size (Relative Change) and Baseline',
                xlabel='Baseline', ylabel='CR Size (Relative Change)'
            )

            # cbar = plt.colorbar(plt.cm.ScalarMappable(cmap="Greens" ), cax=ax102)
            # cbar.set_label("")

                # Assign specific axes for the first set of plots
            ax0 = ax110
            ax1 = ax111
            ax2 = ax120
            ax3 = ax121
            ax4 = ax130
            ax5 = ax131

            # Plot using `plot_trial_averages_sig` for the first set of axes
            plot_trial_averages_sig(trials, short_crp_aligned_time, short_crp_avg_led_sig, short_crp_sem_led_sig, short_crn_avg_led_sig, short_crn_sem_led_sig, title_suffix="Short", event="LED", pooled=True, ax=ax0)
            plot_trial_averages_sig(trials, long_crp_aligned_time, long_crp_avg_led_sig, long_crp_sem_led_sig, long_crn_avg_led_sig, long_crn_sem_led_sig, title_suffix="Long", event="LED", pooled=True, ax=ax1)
            plot_trial_averages_sig(trials, short_crp_aligned_time, short_crp_avg_ap_sig, short_crp_sem_ap_sig, short_crn_avg_ap_sig, short_crn_sem_ap_sig, title_suffix="Short", event="AirPuff", pooled=True, ax=ax2)
            plot_trial_averages_sig(trials, long_crp_aligned_time, long_crp_avg_ap_sig, long_crp_sem_ap_sig, long_crn_avg_ap_sig, long_crn_sem_ap_sig, title_suffix="Long", event="AP", pooled=True, ax=ax3)
            plot_trial_averages_sig(trials, short_crp_aligned_time, short_crp_avg_cr_sig, short_crp_sem_cr_sig, short_crn_avg_cr_sig, short_crn_sem_cr_sig, title_suffix="Short", event="CR", pooled=True, ax=ax4)
            plot_trial_averages_sig(trials, long_crp_aligned_time, long_crp_avg_cr_sig, long_crp_sem_cr_sig, long_crn_avg_cr_sig, long_crn_sem_cr_sig, title_suffix="Long", event="CR", pooled=True, ax=ax5)


            summary_pdf.savefig(fig, dpi = 400)
            plt.close(fig)


            print("DONE")
        # else:
            # print(f"session {session_date} is missing the saved trials")

        # except Exception as e:
        #     print(f"session {session_date} could not be processed{e}")
