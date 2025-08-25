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
from utils.alignment import pooling, pooling_sig, zscore, pooling_info
from utils.indication import *
from plotting.plots import *
from plotting.plots import plot_masks_functions, plot_mouse_summary
from utils.save_plots import *

from plotting.plot_values import compute_fec_CR_data, compute_fec_averages
from plotting.plots import plot_histogram, plot_scatter, plot_hexbin, plot_fec_trial


def sig_pdf_name(session_date, sig_event):
    return f"./outputs/{session_date}/significant_ROIs/individual_{sig_event}_sig_roi.pdf"

static_threshold = 0.02
min_FEC = 0.3

mice = [folder for folder in os.listdir('./data/imaging/') if folder != ".DS_Store"]

br_file = './outputs/imaging/mouse_br.pdf'
fig_br, ax_br = plt.subplots(nrows=2, ncols=2, figsize=(7*2, 7*2))
fec_short_time = None
fec_long_time = None
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

    mouse_summary_file = f'./outputs/imaging/{mouse_name}/{mouse_name}_2p_summary.pdf'

    #values that will be used to find the session summary
    stacked_fec_short_crp = []
    stacked_fec_short_crn = []
    stacked_fec_long_crp = []
    stacked_fec_long_crn = []

    stacked_fec_short_br = []
    stacked_fec_long_br = []
    # must be getting 

    stacked_dff_short_crp = []
    stacked_dff_short_crn = []
    stacked_dff_long_crp = []
    stacked_dff_long_crn = []

    stacked_short_dff_br = []
    stacked_long_dff_br = []

    stacked_led_short_crp = []
    stacked_led_short_crn = []
    stacked_led_long_crp = []
    stacked_led_long_crn = []

    stacked_cr_short_crp = []
    stacked_cr_short_crn = []
    stacked_cr_long_crp = []
    stacked_cr_long_crn = []

    stacked_ap_short_crp = []
    stacked_ap_short_crn = []
    stacked_ap_long_crp = []
    stacked_ap_long_crn = []
    number_trials_short = 0
    number_trials_long = 0
    number_rois_short = 0
    number_rois_long = 0


    number_of_sessions = 0


    for session_date in session_folder:
        number_of_sessions += 1
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
        fec_normed = fec_0
        shorts, longs = block_type(trials)
        CR_stat, CR_interval_avg, base_line_avg, cr_interval_idx, bl_interval_idx = CR_stat_indication(trials , static_threshold, AP_delay = 3)
        short_CRp_fec, short_CRn_fec, long_CRp_fec, long_CRn_fec = block_and_CR_fec(CR_stat,fec_0, shorts, longs)
        short_CRp_fec_normed, short_CRn_fec_normed, long_CRp_fec_normed, long_CRn_fec_normed = block_and_CR_fec(CR_stat,fec_normed, shorts, longs)
        all_id = sort_numbers_as_strings(shorts + longs)
        event_diff, ap_diff , ending_diff = index_differences(init_index , led_index, ending_index, ap_index)


        short_crp_aligned_dff , short_crp_aligned_time = aligned_dff(trials,shorts,CR_stat, 1, init_index, ending_index, shorts[0])
        short_crn_aligned_dff , short_crn_aligned_time = aligned_dff(trials,shorts,CR_stat, 0, init_index, ending_index, shorts[0])

        long_crp_aligned_dff , long_crp_aligned_time = aligned_dff(trials,longs,CR_stat, 1, init_index, ending_index, longs[0])
        long_crn_aligned_dff , long_crn_aligned_time = aligned_dff(trials,longs,CR_stat, 0, init_index, ending_index, longs[0])

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

        data_fec_average_normed = compute_fec_averages(short_CRp_fec_normed, short_CRn_fec_normed,
                        long_CRp_fec_normed, long_CRn_fec_normed, fec_time_0, shorts, longs, trials)

        short_data_normed = data_fec_average_normed["short_trials"]
        long_data_normed = data_fec_average_normed["long_trials"]

        short_time = short_crp_aligned_time
        long_time = long_crp_aligned_time
        
        fec_short_time = fec_time_0[shorts[0]]
        fec_long_time = fec_time_0[longs[0]]

        #stacking things up
        stacked_fec_short_crp.append(short_data_normed['mean1'])
        stacked_fec_long_crp.append(long_data_normed['mean1'])

        stacked_fec_short_crn.append(short_data_normed['mean0'])
        stacked_fec_long_crn.append(long_data_normed['mean0'])

        #pooling all neurons to find the average for the summary
            #all crp crn pooled and averaged.

        number_of_trials_short_crp = 0
        number_of_trials_short_crn = 0
        number_of_trials_long_crn = 0
        number_of_trials_long_crp = 0
        number_of_rois_short_crp = 0
        number_of_rois_short_crn = 0
        number_of_rois_long_crp = 0
        number_of_rois_long_crn = 0

        stacked_dff_short_crp, number_of_rois_short_crp, number_of_trials_short_crp = pooling_info(stacked_dff_short_crp, short_crp_aligned_dff, led_index)
        stacked_dff_short_crn, number_of_rois_short_crn, number_of_trials_short_crn = pooling_info(stacked_dff_short_crn, short_crn_aligned_dff, led_index)
        stacked_dff_long_crp, number_of_rois_long_crp, number_of_trials_long_crp = pooling_info(stacked_dff_long_crp, long_crp_aligned_dff, led_index)
        stacked_dff_long_crn, number_of_rois_long_crn, number_of_trials_long_crn = pooling_info(stacked_dff_long_crn, long_crn_aligned_dff, led_index)

        number_trials_short += number_of_trials_short_crp + number_of_rois_short_crn
        number_trials_long += number_of_trials_long_crp + number_of_trials_long_crn

        number_rois_short += number_of_rois_short_crp + number_of_rois_short_crn
        number_rois_long += number_of_rois_long_crp + number_of_rois_long_crn

        stacked_short_dff_br = pooling(stacked_short_dff_br, short_crp_aligned_dff, led_index)
        stacked_short_dff_br = pooling(stacked_short_dff_br, short_crn_aligned_dff, led_index)
        stacked_long_dff_br= pooling(stacked_long_dff_br, long_crp_aligned_dff, led_index)
        stacked_long_dff_br= pooling(stacked_long_dff_br, long_crn_aligned_dff, led_index)

        stacked_led_short_crp = pooling_sig(stacked_led_short_crp, short_crp_aligned_dff, led_index, roi_indices=led_roi)
        stacked_led_short_crn = pooling_sig(stacked_led_short_crn, short_crn_aligned_dff, led_index, roi_indices=led_roi)
        stacked_led_long_crp = pooling_sig(stacked_led_long_crp, long_crp_aligned_dff, led_index, roi_indices=led_roi)
        stacked_led_long_crn = pooling_sig(stacked_led_long_crn, long_crn_aligned_dff, led_index, roi_indices=led_roi)


        stacked_ap_short_crp = pooling_sig(stacked_ap_short_crp, short_crp_aligned_dff, led_index, roi_indices=ap_roi)
        stacked_ap_short_crn = pooling_sig(stacked_ap_short_crn, short_crn_aligned_dff, led_index, roi_indices=ap_roi)
        stacked_ap_long_crp = pooling_sig(stacked_ap_long_crp, long_crp_aligned_dff, led_index, roi_indices=ap_roi)
        stacked_ap_long_crn = pooling_sig(stacked_ap_long_crn, long_crn_aligned_dff, led_index, roi_indices=ap_roi)


        stacked_cr_short_crp = pooling_sig(stacked_cr_short_crp, short_crp_aligned_dff, led_index, roi_indices=cr_roi)
        stacked_cr_short_crn = pooling_sig(stacked_cr_short_crn, short_crn_aligned_dff, led_index, roi_indices=cr_roi)
        stacked_cr_long_crp = pooling_sig(stacked_cr_long_crp, long_crp_aligned_dff, led_index, roi_indices=cr_roi)
        stacked_cr_long_crn = pooling_sig(stacked_cr_long_crn, long_crn_aligned_dff, led_index, roi_indices=cr_roi)

    # Create figure and axes
    fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(7*2, 7*5))

    time = np.array([-100 + i * 33.33333 for i in range(22)])
    # Plot each row
    plot_mouse_summary(axes[0, 0], axes[0, 1], stacked_fec_short_crp, stacked_fec_short_crn, stacked_fec_long_crp, stacked_fec_long_crn,
    fec_short_time, fec_long_time, short_title='Average of FEC for Short CR+ trials', long_title='Average of FEC for Long CR+ trials', plot_type='FEC')
    plot_mouse_summary(axes[0 + 1, 0], axes[0 + 1, 1], stacked_dff_short_crp, stacked_dff_short_crn, stacked_dff_long_crp, stacked_dff_long_crn,
     time, time, short_title='Average of df/f signals for Short CR+ trials', long_title='Average of df/f signals for Long CR+ trials', plot_type= 'df/f')
    plot_mouse_summary(axes[1 + 1, 0], axes[1 + 1, 1], stacked_led_short_crp, stacked_led_short_crn, stacked_led_long_crp, stacked_led_long_crn, 
    time, time, short_title='Average of df/f signals for Short CR+ trials (LED significant)', long_title='Average of df/f signals for Long CR+ trials (LED significant)', plot_type='df/f')
    plot_mouse_summary(axes[2 + 1, 0], axes[2 + 1, 1], stacked_ap_short_crp, stacked_ap_short_crn, stacked_ap_long_crp, stacked_ap_long_crn,
    time, time, short_title='Average of df/f signals for Short CR+ trials (Air Puff significant)', long_title='Average of df/f signals for Long CR+ trials (Air Puff significant)', plot_type='df/f')
    plot_mouse_summary(axes[3 + 1, 0], axes[3 + 1, 1], stacked_cr_short_crp, stacked_cr_short_crn, stacked_cr_long_crp, stacked_cr_long_crn,
    time, time, short_title='Average of df/f signals for Short CR+ trials (CR significant)', long_title='Average of df/f signals for Long CR+ trials (CR significant)', plot_type='df/f')

    # Labels for each row
    axes[0, 0].set_ylabel("FEC (+/- SEM)")
    axes[1, 0].set_ylabel("dF/F (+/- SEM)")
    axes[2, 0].set_ylabel("dF/F (+/- SEM)")
    axes[3, 0].set_ylabel("dF/F (+/- SEM)")
    axes[4, 0].set_ylabel("dF/F (+/- SEM)")

    plt.tight_layout()
    # plt.show()
    # fig.suptitle(f"Summary of Mouse Data{mouse_name}")

    # Save as PDF
    with PdfPages(mouse_summary_file) as pdf:
        pdf.savefig(fig)
        plt.close(fig)

    print(f"Saved to {mouse_summary_file}")

    stacked_fec_short_br = stacked_fec_short_crp + stacked_fec_short_crn #br sands for black and red
    stacked_fec_long_br = stacked_fec_long_crp + stacked_fec_long_crn #br sands for black and red

    time = np.array([-100 + i * 33.33333 for i in range(22)])

    avg_dff_short_br = np.mean(stacked_short_dff_br, axis=0)
    #z scoring the avergaed insteasd of doing individual traces
    avg_dff_short_br = zscore(avg_dff_short_br)

    sem_dff_short_br = np.std(stacked_short_dff_br, axis=0) / np.sqrt(len(stacked_short_dff_br))
    #zscoring the sem as well. not sure though.
    sem_dff_short_br = zscore(sem_dff_short_br)

    avg_dff_long_br = np.mean(stacked_long_dff_br, axis=0)
    #z scoring the avergaed insteasd of doing individual traces
    avg_dff_long_br = zscore(avg_dff_long_br)
    sem_dff_long_br = np.std(stacked_long_dff_br, axis=0) / np.sqrt(len(stacked_long_dff_br))
    #zscoring the sem as well. not sure though.
    sem_dff_long_br = zscore(sem_dff_long_br)

    avg_fec_short_br = np.mean(stacked_fec_short_br, axis=0)
    sem_fec_short_br = np.std(stacked_fec_short_br, axis=0) / np.sqrt(len(stacked_fec_short_br))
    avg_fec_long_br = np.mean(stacked_fec_long_br, axis=0)
    sem_fec_long_br = np.std(stacked_fec_long_br, axis=0) / np.sqrt(len(stacked_fec_long_br))

    if 'Control' in mouse_name:
        color = 'black'
        ax_br[0, 0].axvspan(0, 50, color="gray", alpha=0.3, label="LED")
        ax_br[0, 1].axvspan(0, 50, color="gray", alpha=0.3, label="LED")
        ax_br[1, 0].axvspan(0, 50, color="gray", alpha=0.3, label="LED")
        ax_br[1, 1].axvspan(0, 50, color="gray", alpha=0.3, label="LED")

        ax_br[0,0].axvspan(200, 220, color="blue", alpha=0.3, label="Air Puff")
        ax_br[0,1].axvspan(400, 420, color="lime", alpha=0.3, label="Air Puff")
        ax_br[1,0].axvspan(200, 220, color="blue", alpha=0.3, label="Air Puff")
        ax_br[1,1].axvspan(400, 420, color="lime", alpha=0.3, label="Air Puff")
    else:
        color = 'red'


    ax_br[0, 0].plot(fec_short_time, avg_fec_short_br, color = color,
    label = f'{mouse_name} \n {number_of_sessions} sessions \n {number_trials_short} trials \n')
    ax_br[0, 0].fill_between(fec_short_time, avg_fec_short_br - sem_fec_short_br,
    avg_fec_short_br + sem_fec_short_br, alpha=0.2, color=color)

    ax_br[0, 1].plot(fec_long_time, avg_fec_long_br, color = color, 
    label = f'{mouse_name}\n {number_of_sessions} sessions \n {number_trials_long} trials \n')
    ax_br[0, 1].fill_between(fec_long_time, avg_fec_long_br - sem_fec_long_br,
    avg_fec_long_br + sem_fec_long_br, alpha=0.2, color=color)

    ax_br[1, 0].plot(time, avg_dff_short_br, color = color, 
    label = f'{mouse_name} \n pooled {len(stacked_short_dff_br)} traces in total \n {number_of_sessions} sessions \n {number_trials_short} trials \n {number_rois_short} ROIs')
    ax_br[1, 0].fill_between(time, avg_dff_short_br - sem_dff_short_br,
    avg_dff_short_br + sem_dff_short_br, alpha=0.2, color=color)

    ax_br[1, 1].plot(time, avg_dff_long_br, color = color, 
    label = f'{mouse_name} \n pooled {len(stacked_long_dff_br)} traces in total \n {number_of_sessions} sessions \n {number_trials_long} trials \n {number_rois_short} ROIs')
    ax_br[1, 1].fill_between(time, avg_dff_long_br - sem_dff_long_br, 
    avg_dff_long_br + sem_dff_long_br, alpha=0.2, color=color)

    ax_br[0, 0].set_title("FEC")
    ax_br[0, 1].set_title("FEC")
    ax_br[1, 0].set_title("dF/F")
    ax_br[1, 1].set_title("dF/F")

    ax_br[0, 0].set_ylabel("FEC (+/- SEM)")
    ax_br[0, 1].set_ylabel("FEC (+/- SEM)")
    ax_br[1, 0].set_ylabel("dF/F (+/- SEM)")
    ax_br[1, 1].set_ylabel("dF/F (+/- SEM)")

    ax_br[0, 0].set_xlabel("Time(ms)")
    ax_br[0, 1].set_xlabel("Time(ms)")
    ax_br[1, 0].set_xlabel("Time(ms)")
    ax_br[1, 1].set_xlabel("Time(ms)")

    ax_br[0, 0].set_xlim(-100, 600)
    ax_br[0, 1].set_xlim(-100, 600)
    ax_br[1, 0].set_xlim(-100, 600)
    ax_br[1, 1].set_xlim(-100, 600)

    ax_br[0, 0].spines['right'].set_visible(False)
    ax_br[0, 0].spines['top'].set_visible(False)
    ax_br[0, 1].spines['right'].set_visible(False)
    ax_br[0, 1].spines['top'].set_visible(False)
    ax_br[1, 0].spines['right'].set_visible(False)
    ax_br[1, 0].spines['top'].set_visible(False)
    ax_br[1, 1].spines['right'].set_visible(False)
    ax_br[1, 1].spines['top'].set_visible(False)

    

    plt.tight_layout()
    ax_br[0, 0].legend()
    ax_br[0, 1].legend()
    ax_br[1, 0].legend()
    ax_br[1, 1].legend()
# plt.show()

with PdfPages(br_file) as pdf:
    pdf.savefig(fig_br)
    pdf.close()

print(f"PDF successfully saved: {br_file}")



    

        # breakpoint()
        # print(np.mean(stacked_led_long_crp, axis = 0))
        # plt.plot(np.array([-100 + i * 33.33333 for i in range(22)]), np.mean(stacked_led_long_crn, axis = 0))
        # plt.show()
        # plt.plot(np.array([-100 + i * 33.33333 for i in range(22)]), np.mean(stacked_cr_short_crn, axis = 0))
    # plt.show()


        # stacked_avg_dff_short_crp.append()
