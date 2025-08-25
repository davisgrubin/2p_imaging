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
from utils.indication import sort_dff_max_index, sort_dff_avg
from plotting.plots import *
from plotting.plots import plot_masks_functions, plot_mouse_summary
from utils.save_plots import *

from plotting.plot_values import compute_fec_CR_data, compute_fec_averages
from plotting.plots import plot_histogram, plot_scatter, plot_hexbin, plot_fec_trial

def add_colorbar(im, ax):
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label = 'df/f z-scored')
    vmin, vmax = im.get_clim()  # Get min and max values of heatmap
    cbar.set_ticks(np.linspace(vmin, vmax, num=5))  # Show only 5 evenly spaced ticks

def sig_pdf_name(session_date, sig_event):
    return f"./outputs/{session_date}/significant_ROIs/individual_{sig_event}_sig_roi.pdf"

static_threshold = 0.09
min_FEC = 0.3

mice = [folder for folder in os.listdir('./data/imaging/') if folder != ".DS_Store"]

br_file = './outputs/imaging/mouse_br.pdf'
br_p_file = './outputs/imaging/mouse_br_p.pdf'
br_n_file = './outputs/imaging/mouse_br_n.pdf'
fig_br, ax_br = plt.subplots(nrows=4, ncols=4, figsize=(10*4, 10*4))
fig_br_p, ax_br_p = plt.subplots(nrows=4, ncols=4, figsize=(10*4, 10*4))
fig_br_n, ax_br_n = plt.subplots(nrows=4, ncols=4, figsize=(10*4, 10*4))
fec_short_time = None
fec_long_time = None



for mouse_name in mice:

    hm_max_normal = f'./outputs/imaging/{mouse_name}/{mouse_name}_hm_max_normal.pdf'
    hm_avg_normal = f'./outputs/imaging/{mouse_name}/{mouse_name}_hm_avg_normal.pdf'
    hm_max_z = f'./outputs/imaging/{mouse_name}/{mouse_name}_hm_max_zed.pdf'
    hm_avg_z = f'./outputs/imaging/{mouse_name}/{mouse_name}_hm_avg_zed.pdf'

    roi_i_short_br = 0
    roi_i_long_br = 0

    roi_i_short_crp = 0
    roi_i_short_crn = 0
    roi_i_long_crp = 0
    roi_i_long_crn = 0

    short_dff_stack_br = {}
    long_dff_stack_br = {}
    short_crp_dff_stack = {}
    short_crn_dff_stack_z = {}
    short_crn_dff_stack = {}
    short_crp_dff_stack_z = {}
    long_crp_dff_stack = {}
    long_crn_dff_stack_z = {}
    long_crn_dff_stack = {}
    long_crp_dff_stack_z = {}

    if not os.path.exists(f'./data/imaging/{mouse_name}'):
        os.mkdir(f'./data/imaging/{mouse_name}')
        print(f"data folder '{mouse_name}' created.")
    else:
        print(f"data folder '{mouse_name}' already exists.")

    session_folder = [folder for folder in os.listdir(f'./data/imaging/{mouse_name}/') if folder != ".DS_Store"]
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
    number_trials_short_crp = 0
    number_trials_short_crn = 0

    number_trials_long = 0
    number_trials_long_crp = 0
    number_trials_long_crn = 0

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

        short_aligned_dff_br, short_aligned_time_br = aligned_dff_br(trials,shorts, init_index, ending_index, shorts[0])
        long_aligned_dff_br, long_aligned_time_br = aligned_dff_br(trials,longs, init_index, ending_index, longs[0])

        short_crp_avg_pooled, short_crp_sem_pooled, n_short_crp_pooled = calculate_average_dff_pool(short_crp_aligned_dff)
        short_crn_avg_pooled, short_crn_sem_pooled, n_short_crn_pooled = calculate_average_dff_pool(short_crn_aligned_dff)
        long_crp_avg_pooled,   long_crp_sem_pooled, n_long_crp_pooled = calculate_average_dff_pool(long_crp_aligned_dff)
        long_crn_avg_pooled,   long_crn_sem_pooled, n_long_crn_pooled = calculate_average_dff_pool(long_crn_aligned_dff)

        short_crp_avg_dff, short_crp_sem_dff, n_short_crp_roi = calculate_average_dff_roi(aligned_dff=short_crp_aligned_dff)
        short_crn_avg_dff, short_crn_sem_dff, n_short_crn_roi = calculate_average_dff_roi(aligned_dff=short_crn_aligned_dff)

        short_avg_dff_br, short_sem_dff_br, n_short_roi_br = calculate_average_dff_roi(aligned_dff=short_aligned_dff_br)

        long_crp_avg_dff,   long_crp_sem_dff, n_long_crp_roi = calculate_average_dff_roi(aligned_dff=long_crp_aligned_dff)
        long_crn_avg_dff,   long_crn_sem_dff, n_long_crn_roi = calculate_average_dff_roi(aligned_dff=long_crn_aligned_dff)

        long_avg_dff_br,   long_sem_dff_br, n_long_roi_br = calculate_average_dff_roi(aligned_dff=long_aligned_dff_br)

        for roi in short_avg_dff_br:
            short_dff_stack_br[roi_i_short_br] = short_avg_dff_br[roi][event_diff - 3 : event_diff + 18]
            roi_i_short_br += 1

        for roi in long_avg_dff_br:
            long_dff_stack_br[roi_i_long_br] = long_avg_dff_br[roi][event_diff - 3 : event_diff + 18]
            roi_i_long_br += 1


        for roi in short_crp_avg_dff:
            short_crp_dff_stack[roi_i_short_crp] = short_crp_avg_dff[roi][event_diff - 3 : event_diff + 18] 
            # short_crp_dff_stack_z[roi_i_short_crp] = zscore(short_crp_avg_dff[roi][event_diff - 3 : event_diff + 18]) 
            roi_i_short_crp += 1

        for roi in short_crn_avg_dff:
            short_crn_dff_stack[roi_i_short_crn] = short_crn_avg_dff[roi] [event_diff - 3 : event_diff + 18]
            # short_crn_dff_stack_z[roi_i_short_crn] = zscore(short_crn_avg_dff[roi] [event_diff - 3 : event_diff + 18])
            roi_i_short_crn += 1

        for roi in long_crp_avg_dff:
            long_crp_dff_stack[roi_i_long_crp] = long_crp_avg_dff[roi] [event_diff - 3 : event_diff + 18]
            # long_crp_dff_stack_z[roi_i_long_crp] = zscore(long_crp_avg_dff[roi] [event_diff - 3 : event_diff + 18])
            roi_i_long_crp += 1

        for roi in long_crn_avg_dff:
            long_crn_dff_stack[roi_i_long_crn] = long_crn_avg_dff[roi] [event_diff - 3 : event_diff + 18]
            # long_crn_dff_stack_z[roi_i_long_crn] = zscore(long_crn_avg_dff[roi] [event_diff - 3 : event_diff + 18])
            roi_i_long_crn += 1

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

        stacked_dff_short_crp, number_of_rois_short_crp, number_of_trials_short_crp = pooling_info(stacked_dff_short_crp, short_crp_aligned_dff, led_index)
        stacked_dff_short_crn, number_of_rois_short_crn, number_of_trials_short_crn = pooling_info(stacked_dff_short_crn, short_crn_aligned_dff, led_index)
        stacked_dff_long_crp, number_of_rois_long_crp, number_of_trials_long_crp = pooling_info(stacked_dff_long_crp, long_crp_aligned_dff, led_index)
        stacked_dff_long_crn, number_of_rois_long_crn, number_of_trials_long_crn = pooling_info(stacked_dff_long_crn, long_crn_aligned_dff, led_index)

        number_trials_short += number_of_trials_short_crp + number_of_trials_short_crn
        
        number_trials_long += number_of_trials_long_crp + number_of_trials_long_crn

        number_rois_short += number_of_rois_short_crp + number_of_rois_short_crn
        number_rois_long += number_of_rois_long_crp + number_of_rois_long_crn

        number_trials_short_crp += number_of_trials_short_crp
        number_trials_short_crn += number_of_trials_short_crn

        number_trials_long_crp += number_of_trials_long_crp
        number_trials_long_crn += number_of_trials_long_crn

        stacked_short_dff_br = pooling(stacked_short_dff_br, short_crp_aligned_dff, led_index)
        stacked_short_dff_br = pooling(stacked_short_dff_br, short_crn_aligned_dff, led_index)
        stacked_long_dff_br= pooling(stacked_long_dff_br, long_crp_aligned_dff, led_index)
        stacked_long_dff_br= pooling(stacked_long_dff_br, long_crn_aligned_dff, led_index)


    # sort the ROIs according for the heatmap. first according to the max value time.
    sorted_max_short_br  = sort_dff_max_index(short_dff_stack_br, 3, 15)
    sorted_max_long_br  = sort_dff_max_index(long_dff_stack_br, 3, 17)

    sorted_max_short_crp = sort_dff_max_index(short_crp_dff_stack, 3, 9)
    sorted_max_long_crp = sort_dff_max_index(long_crp_dff_stack, 3, 15)
    sorted_max_short_crn = sort_dff_max_index(short_crn_dff_stack, 3, 9)
    sorted_max_long_crn = sort_dff_max_index(long_crn_dff_stack, 3, 15)

    sorted_avg_short_crp = sort_dff_avg(short_crp_dff_stack, 3, 9)
    sorted_avg_long_crp = sort_dff_avg(long_crp_dff_stack, 3, 15)
    sorted_avg_short_crn = sort_dff_avg(short_crn_dff_stack, 3, 9)
    sorted_avg_long_crn = sort_dff_avg(long_crn_dff_stack, 3, 15)

 
    short_dff_stack_sorted_short_max = np.array([short_dff_stack_br[i] for i in reversed(sorted_max_short_br)])
    long_dff_stack_sorted_short_max = np.array([long_dff_stack_br[i] for i in reversed(sorted_max_short_br)])

    short_dff_stack_sorted_long_max = np.array([short_dff_stack_br[i] for i in reversed(sorted_max_long_br)])
    long_dff_stack_sorted_long_max = np.array([long_dff_stack_br[i] for i in reversed(sorted_max_long_br)])

    short_dff_stack_sorted_short_max_crp = np.array([short_crp_dff_stack[i] for i in reversed(sorted_max_short_crp)])
    long_dff_stack_sorted_short_max_crp = np.array([long_crp_dff_stack[i] for i in reversed(sorted_max_short_crp)])
    short_dff_stack_sorted_long_max_crp = np.array([short_crp_dff_stack[i] for i in reversed(sorted_max_long_crp)])
    long_dff_stack_sorted_long_max_crp = np.array([long_crp_dff_stack[i] for i in reversed(sorted_max_long_crp)])
    
    short_dff_stack_sorted_short_max_crn = np.array([short_crn_dff_stack[i] for i in reversed(sorted_max_short_crn)])
    long_dff_stack_sorted_short_max_crn = np.array([long_crn_dff_stack[i] for i in reversed(sorted_max_short_crn)])
    short_dff_stack_sorted_long_max_crn = np.array([short_crn_dff_stack[i] for i in reversed(sorted_max_long_crn)])
    long_dff_stack_sorted_long_max_crn = np.array([long_crn_dff_stack[i] for i in reversed(sorted_max_long_crn)])



    # long_crp_dff_stack_sorted_max = np.array([long_crp_dff_stack[i] for i in sorted_max_long_crp])
    # short_crn_dff_stack_sorted_max = np.array([short_crn_dff_stack[i] for i in sorted_max_short_crn])
    # long_crn_dff_stack_sorted_max = np.array([long_crn_dff_stack[i] for i in sorted_max_long_crn])

    # short_crp_dff_stack_sorted_avg = np.array([short_crp_dff_stack[i] for i in sorted_avg_short_crp])
    # long_crp_dff_stack_sorted_avg = np.array([long_crp_dff_stack[i] for i in sorted_avg_long_crp])
    # short_crn_dff_stack_sorted_avg = np.array([short_crn_dff_stack[i] for i in sorted_avg_short_crn])
    # long_crn_dff_stack_sorted_avg = np.array([long_crn_dff_stack[i] for i in sorted_avg_long_crn])

    
    stacked_fec_short_br = stacked_fec_short_crp + stacked_fec_short_crn #br sands for black and red
    stacked_fec_long_br = stacked_fec_long_crp + stacked_fec_long_crn #br sands for black and red

    time = np.array([-100 + i * 33.33333 for i in range(22)])

    avg_dff_short_br = np.mean(stacked_short_dff_br, axis=0)
    avg_dff_short_crp = np.mean(stacked_dff_short_crp, axis=0)
    avg_dff_short_crn = np.mean(stacked_dff_short_crn, axis=0)

    sem_dff_short_br = np.std(stacked_short_dff_br, axis=0) / np.sqrt(len(stacked_short_dff_br))
    sem_dff_short_crp = np.std(stacked_dff_short_crp, axis=0) / np.sqrt(len(stacked_dff_short_crp))
    sem_dff_short_crn = np.std(stacked_dff_short_crn, axis=0) / np.sqrt(len(stacked_dff_short_crn))

    avg_dff_long_br = np.mean(stacked_long_dff_br, axis=0)
    avg_dff_long_crp = np.mean(stacked_dff_long_crp, axis=0)
    avg_dff_long_crn = np.mean(stacked_dff_long_crn, axis=0)

    sem_dff_long_br = np.std(stacked_long_dff_br, axis=0) / np.sqrt(len(stacked_long_dff_br))
    sem_dff_long_crp = np.std(stacked_dff_long_crp, axis=0) / np.sqrt(len(stacked_dff_long_crp))
    sem_dff_long_crn = np.std(stacked_dff_long_crn, axis=0) / np.sqrt(len(stacked_dff_long_crn))

    avg_fec_short_br = np.mean(stacked_fec_short_br, axis=0)
    sem_fec_short_br = np.std(stacked_fec_short_br, axis=0) / np.sqrt(len(stacked_fec_short_br))
    avg_fec_long_br = np.mean(stacked_fec_long_br, axis=0)
    sem_fec_long_br = np.std(stacked_fec_long_br, axis=0) / np.sqrt(len(stacked_fec_long_br))

    avg_fec_short_crp = np.mean(stacked_fec_short_crp, axis = 0)
    sem_fec_short_crp = np.std(stacked_fec_short_crp, axis=0) / np.sqrt(len(stacked_fec_short_crp))
    avg_fec_long_crp = np.mean(stacked_fec_long_crp, axis=0)
    sem_fec_long_crp = np.std(stacked_fec_long_crp, axis=0) / np.sqrt(len(stacked_fec_long_crp))

    avg_fec_short_crn = np.mean(stacked_fec_short_crn, axis=0)
    sem_fec_short_crn = np.std(stacked_fec_short_crn, axis=0) / np.sqrt(len(stacked_fec_short_crn))
    avg_fec_long_crn = np.mean(stacked_fec_long_crn, axis=0)
    sem_fec_long_crn = np.std(stacked_fec_long_crn, axis=0) / np.sqrt(len(stacked_fec_long_crn))


    if 'Control' in mouse_name:

        color = 'black'
        h = 2
        mouse_type = 'Control'
    else:
        color = 'red'
        h = 3
        mouse_type = 'SD'


    for (ax, fig, file, stacked_short_dff, stacked_long_dff,
         number_short, number_long,
         avg_fec_short, sem_fec_short, avg_fec_long, sem_fec_long,
         avg_dff_short, sem_dff_short, avg_dff_long, sem_dff_long,
         short_stack_short_sorted, short_stack_long_sorted, 
        long_stack_short_sorted, long_stack_long_sorted)  in [

        (ax_br, fig_br, br_file, stacked_short_dff_br, stacked_long_dff_br,
         number_trials_short, number_trials_long, 
         avg_fec_short_br, sem_fec_short_br, avg_fec_long_br, sem_fec_long_br,
         avg_dff_short_br, sem_dff_short_br, avg_dff_long_br, sem_dff_long_br,
         short_dff_stack_sorted_short_max, short_dff_stack_sorted_long_max,
        long_dff_stack_sorted_short_max, long_dff_stack_sorted_long_max), 

        (ax_br_p, fig_br_p, br_p_file, stacked_dff_short_crp, stacked_dff_long_crp,
        number_trials_short_crp, number_trials_long_crp,
         avg_fec_short_crp, sem_fec_short_crp, avg_fec_long_crp, sem_fec_long_crp,
         avg_dff_short_crp, sem_dff_short_crp, avg_dff_long_crp, sem_dff_long_crp,
         short_dff_stack_sorted_short_max_crp, short_dff_stack_sorted_long_max_crp,
        long_dff_stack_sorted_short_max_crp, long_dff_stack_sorted_long_max_crp), 

        (ax_br_n, fig_br_n, br_n_file, stacked_dff_short_crn, stacked_dff_long_crn,
        number_trials_short_crn, number_trials_long_crn,
         avg_fec_short_crn, sem_fec_short_crn, avg_fec_long_crn, sem_fec_long_crn,
         avg_dff_short_crn, sem_dff_short_crn, avg_dff_long_crn, sem_dff_long_crn,
         short_dff_stack_sorted_short_max_crn, short_dff_stack_sorted_long_max_crn,
        long_dff_stack_sorted_short_max_crn, long_dff_stack_sorted_long_max_crn)]:


        vmin = min(short_stack_short_sorted.min(), long_dff_stack_sorted_short_max.min())
        vmax = max(short_stack_short_sorted.max(), long_dff_stack_sorted_short_max.max())

        y_extent = [0, short_stack_short_sorted.shape[0]]  # Full height of data
        im1 = ax[h, 0].imshow(short_stack_short_sorted, aspect='auto', 
                                 extent=[time[0], time[-1], y_extent[0], y_extent[1]],
                                 cmap='viridis', vmin =vmin, vmax = vmax)

        ax[h, 0].set_title(f"Trial-Averaged dF/F: {mouse_type}, Short Trials, Sorted by Peak Time Short")
        ax[h, 0].set_ylabel("Neurons (sorted based on short) for short heatmap")

        y_extent = [0, long_stack_short_sorted.shape[0]]  # Full height of data
        im2 = ax[h, 1].imshow(long_stack_short_sorted, aspect='auto', 
                                 extent=[time[0], time[-1], y_extent[0], y_extent[1]],
                                 cmap='viridis', vmin =vmin, vmax = vmax)

        ax[h, 1].set_title(f"Trial-Averaged dF/F: {mouse_type}, Long Trials, Sorted by Peak Time Short")
        ax[h, 1].set_ylabel("Neurons (sorted based on short) for long heatmap")

        y_extent = [0, short_stack_long_sorted.shape[0]]  # Full height of data
        im3 = ax[h, 2].imshow(short_stack_long_sorted, aspect='auto', 
                                 extent=[time[0], time[-1], y_extent[0], y_extent[1]],
                                 cmap='viridis', vmin =vmin, vmax = vmax)

        ax[h, 2].set_title(f"Trial-Averaged dF/F: {mouse_type}, Short Trials, Sorted by Peak Time Long")
        ax[h, 2].set_ylabel("Neurons (sorted based on long) for short heatmap")

        y_extent = [0, long_stack_long_sorted.shape[0]]  # Full height of data
        im4 = ax[h, 3].imshow(long_stack_long_sorted, aspect='auto', 
                                 extent=[time[0], time[-1], y_extent[0], y_extent[1]],
                                 cmap='viridis', vmin =vmin, vmax = vmax)

        ax[h, 3].set_title(f"Trial-Averaged dF/F: {mouse_type}, Short Trials, Sorted by Peak Time Long")
        ax[h, 2].set_ylabel("Neurons (sorted based on long) for short heatmap")
        # add_colorbar(im1, ax[h, 0])

        add_colorbar(im2, ax[h, 3])
        ax[0, 0].plot(fec_short_time, avg_fec_short, color = color,
        label = f'{mouse_name} \n {number_of_sessions} sessions \n {number_short} trials \n')
        ax[0, 0].fill_between(fec_short_time, avg_fec_short - sem_fec_short,
        avg_fec_short + sem_fec_short, alpha=0.2, color=color)

        ax[0, 1].plot(fec_long_time, avg_fec_long, color = color, 
        label = f'{mouse_name}\n {number_of_sessions} sessions \n {number_long} trials \n')
        ax[0, 1].fill_between(fec_long_time, avg_fec_long - sem_fec_long,
        avg_fec_long + sem_fec_long, alpha=0.2, color=color)

        ax[1, 0].plot(time, avg_dff_short - avg_dff_short[0], color = color, 
        label = f'{mouse_name} \n pooled {len(stacked_short_dff)} traces in total \n {number_of_sessions} sessions \n {number_short} trials \n {number_rois_short} ROIs')
        ax[1, 0].fill_between(time, avg_dff_short - avg_dff_short[0] - sem_dff_short,
        avg_dff_short - avg_dff_short[0] + sem_dff_short, alpha=0.2, color=color)

        ax[1, 1].plot(time, avg_dff_long - avg_dff_long[0], color = color, 
        label = f'{mouse_name} \n pooled {len(stacked_long_dff)} traces in total \n {number_of_sessions} sessions \n {number_long} trials \n {number_rois_short} ROIs')
        ax[1, 1].fill_between(time, avg_dff_long - sem_dff_long - avg_dff_long[0], 
        avg_dff_long + sem_dff_long - avg_dff_long[0], alpha=0.2, color=color)

        for i in range(4):
            for j in range(4):
                if 'Control' in mouse_name:
                    if i<2:
                        if j<2:
                            ax[i, j].axvspan(0, 50, color="gray", alpha=0.3, label="LED")
                            if j == 0:
                                ax[i, j].axvspan(200, 220, color="blue", alpha=0.3, label="Air Puff")
                            if j == 1:
                                ax[i, j].axvspan(400, 420, color="lime", alpha=0.3, label="Air Puff")

                    else:

                        ax[i, j].axvline(0, color='gray', label='LED', linestyle = '--', alpha = 0.7)
                        ax[i, j].axvline(50, color='gray', linestyle = '--', alpha = 0.7)
                        if j == 0 or j == 2:
                            ax[i, j].axvline(200, color='blue', linestyle = '--', alpha = 0.7)
                            ax[i, j].axvline(220, color='blue', label='AirPuff', linestyle = '--', alpha = 0.7)
                        if j ==1 or j == 3:
                            ax[i, j].axvline(400, color='lime', linestyle = '--', alpha = 0.7)
                            ax[i, j].axvline(420, color='lime', label='AirPuff', linestyle = '--', alpha = 0.7)

                if j<2:
                    if i == 0:
                        ax[i, j].set_title("FEC")
                        ax[i, j].set_xlim(-100, 600)
                        ax[i, j].set_ylabel("Eyelid closure (norm.)  (Mean +/- SEM}")

                    if i == 1:
                        ax[i, j].set_title("df/f")
                        ax_br[i ,j].set_ylabel("dF/F (+/- SEM)")

                ax[i, j].spines['right'].set_visible(False)
                ax[i, j].spines['top'].set_visible(False)
                ax[i ,j].set_xlabel("Time from LED onset (ms)")
                ax[i, j].legend()
            
                if i < 2:
                    if j > 1:
                        ax[i, j].axis('off')

        plt.tight_layout()

        with PdfPages(file) as pdf:
            pdf.savefig(fig, dpi = 400)
            pdf.close()

        print(f"PDF successfully saved: {file}")
