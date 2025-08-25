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

static_threshold = 0.02
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

    if not os.path.exists(f'./data/imaging/{mouse_name}'):
        os.mkdir(f'./data/imaging/{mouse_name}')
        print(f"data folder '{mouse_name}' created.")
    else:
        print(f"data folder '{mouse_name}' already exists.")

    session_folder = [folder for folder in os.listdir(f'./data/imaging/{mouse_name}/') if folder != ".DS_Store"]
    #values that will be used to find the session summary
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

        session_br_file = f"./outputs/imaging/{mouse_name}/{mouse_name}{session_date}_dff_summary.pdf"

        number_of_sessions += 1
        if session_date[0]=='.' or session_date[0]=='i':
            continue
        if not os.path.exists(f'./outputs/imaging/{mouse_name}/{session_date}'):
            os.mkdir(f'./outputs/imaging/{mouse_name}/{session_date}')
            print(f"output folder '{session_date}' created.")
        else:
            print(f"output folder '{session_date}' already exists.")
        print(mouse_name, session_date)
        mask_file = f"./data/imaging/{mouse_name}/{session_date}/masks.h5"
        trials = h5py.File(f"./data/imaging/{mouse_name}/{session_date}/saved_trials.h5")["trial_id"]
    
        init_time, init_index, ending_time, ending_index, led_index, ap_index = aligning_times(trials=trials)
        fec, fec_time_0, _ = fec_zero(trials)
        fec_0 = moving_average(fec , window_size=7)
        fec_normed = fec_0
        shorts, longs = block_type(trials)
        CR_stat, CR_interval_avg, base_line_avg, cr_interval_idx, bl_interval_idx = CR_stat_indication(trials, fec_0, fec_time_0 , static_threshold, AP_delay = 3)
        short_CRp_fec, short_CRn_fec, long_CRp_fec, long_CRn_fec = block_and_CR_fec(CR_stat,fec_0, shorts, longs)
        short_CRp_fec_normed, short_CRn_fec_normed, long_CRp_fec_normed, long_CRn_fec_normed = block_and_CR_fec(CR_stat,fec_normed, shorts, longs)
        all_id = sort_numbers_as_strings(shorts + longs)
        event_diff, ap_diff , ending_diff = index_differences(init_index , led_index, ending_index, ap_index)

        number_short_crp = len(short_CRp_fec)
        number_short_crn = len(short_CRn_fec)

        number_long_crp = len(long_CRp_fec)
        number_long_crn = len(long_CRn_fec)

        short_crp_aligned_dff , short_crp_aligned_time = aligned_dff(trials,shorts,CR_stat, 1, init_index, ending_index, shorts[0])
        short_crn_aligned_dff , short_crn_aligned_time = aligned_dff(trials,shorts,CR_stat, 0, init_index, ending_index, shorts[0])

        short_crp_aligned_dff , short_crp_aligned_time = filter_dff(short_crp_aligned_dff, short_crp_aligned_time, led_index)
        short_crn_aligned_dff , short_crn_aligned_time = filter_dff(short_crn_aligned_dff, short_crn_aligned_time, led_index)


        long_crp_aligned_dff , long_crp_aligned_time = aligned_dff(trials,longs,CR_stat, 1, init_index, ending_index, longs[0])
        long_crn_aligned_dff , long_crn_aligned_time = aligned_dff(trials,longs,CR_stat, 0, init_index, ending_index, longs[0])

        long_crp_aligned_dff, long_crp_aligned_time = filter_dff(long_crp_aligned_dff, long_crp_aligned_time, led_index)
        long_crn_aligned_dff, long_crn_aligned_time = filter_dff(long_crn_aligned_dff, long_crn_aligned_time, led_index)

        short_aligned_dff_br, short_aligned_time_br = aligned_dff_br(trials,shorts, init_index, ending_index, shorts[0])
        long_aligned_dff_br, long_aligned_time_br = aligned_dff_br(trials,longs, init_index, ending_index, longs[0])

        short_aligned_dff_br, short_aligned_time_br = filter_dff(short_aligned_dff_br, short_aligned_time_br, led_index)
        long_aligned_dff_br, long_aligned_time_br = filter_dff(long_aligned_dff_br, long_aligned_time_br, led_index)

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

        # average of all roi (grnad average)
        short_crp_avg_roi, short_crp_sem_roi = average_over_roi(short_crp_avg_dff)
        short_crn_avg_roi, short_crn_sem_roi = average_over_roi(short_crn_avg_dff)
        long_crp_avg_roi, long_crp_sem_roi =   average_over_roi(long_crp_avg_dff)
        long_crn_avg_roi, long_crn_sem_roi =   average_over_roi(long_crn_avg_dff)

        short_br_avg_roi, short_br_sem_roi = average_over_roi(short_avg_dff_br)
        long_br_avg_roi, long_br_sem_roi = average_over_roi(long_avg_dff_br)


        data_fec_average_normed = compute_fec_averages(short_CRp_fec_normed, short_CRn_fec_normed,
                        long_CRp_fec_normed, long_CRn_fec_normed, fec_time_0, shorts, longs, trials)

        
        all_data = data_fec_average_normed["all_trials"]

        short_data_normed = data_fec_average_normed["short_trials"]
        long_data_normed = data_fec_average_normed["long_trials"]
        all_trials = data_fec_average_normed["all_trials"]

        short_time = short_crp_aligned_time
        long_time = long_crp_aligned_time
        
        fec_short_time = fec_time_0[shorts[0]]
        fec_long_time = fec_time_0[longs[0]]
        # short_data_normed['mean1']

        all_short_mean = all_data["meanT_short"]
        all_short_sem = all_data["semT_short"]

        all_long_mean = all_data["meanT_long"]
        all_long_sem = all_data["semT_long"]


        #this is where all the plotting and saving must happen
        sorted_max_short_crp = sort_dff_max_index(short_crp_avg_dff, 3, 15)
        sorted_max_short_crn = sort_dff_max_index(short_crn_avg_dff, 3,  15)
        sorted_max_long_crp = sort_dff_max_index(long_crp_avg_dff, 3, 15)
        sorted_max_long_crn = sort_dff_max_index(long_crn_avg_dff, 3,  15)

        sorted_max_short_br  = sort_dff_max_index(short_avg_dff_br, 3, 15)
        sorted_max_long_br  = sort_dff_max_index(long_avg_dff_br, 3, 15)

        roi_short = len(sorted_max_long_br)
        roi_long = len(sorted_max_long_br)


        short_dff_stack_sorted_short_max_crp = np.array([short_crp_avg_dff[i] for i in reversed(sorted_max_short_crp)])
        long_dff_stack_sorted_short_max_crp = np.array([long_crp_avg_dff[i] for i in reversed(sorted_max_short_crp)])
        short_dff_stack_sorted_long_max_crp = np.array([short_crp_avg_dff[i] for i in reversed(sorted_max_long_crp)])
        long_dff_stack_sorted_long_max_crp = np.array([long_crp_avg_dff[i] for i in reversed(sorted_max_long_crp)])

        short_dff_stack_sorted_short_max = np.array([short_avg_dff_br[i] for i in reversed(sorted_max_short_br)])
        long_dff_stack_sorted_short_max = np.array([long_avg_dff_br[i] for i in reversed(sorted_max_short_br)])
       
        short_dff_stack_sorted_short_max_crn = np.array([short_crn_avg_dff[i] for i in reversed(sorted_max_short_crn)])
        long_dff_stack_sorted_short_max_crn = np.array([long_crn_avg_dff[i] for i in reversed(sorted_max_short_crn)])
        short_dff_stack_sorted_long_max_crn = np.array([short_crn_avg_dff[i] for i in reversed(sorted_max_long_crn)])
        long_dff_stack_sorted_long_max_crn = np.array([long_crn_avg_dff[i] for i in reversed(sorted_max_long_crn)])

        short_dff_stack_sorted_long_max = np.array([short_avg_dff_br[i] for i in reversed(sorted_max_long_br)])
        long_dff_stack_sorted_long_max = np.array([long_avg_dff_br[i] for i in reversed(sorted_max_long_br)])

        time = np.array([-100 + i * 33.33333 for i in range(22)])

        x = 4
        y = 10
        spacing = 0
        # Create the figure and GridSpec
        fig = plt.figure(figsize=(x * 7, y * 7))
        gs = GridSpec(
            y + spacing * (y + 1), 
            x + spacing * (x + 1), 
            figure=fig
        )

        # FOV pictures
        ax00 = fig.add_subplot(gs[0:1, 0:1])
        ax01 = fig.add_subplot(gs[0:1, 1:2])
        ax02 = fig.add_subplot(gs[0:1, 2:3])
        # ax03 = fig.add_subplot(gs[0:1, 3:4])

        #FEC
        ax10 = fig.add_subplot(gs[1:2, 0:1])
        ax11 = fig.add_subplot(gs[1:2, 1:2])
        # ax12 = fig.add_subplot(gs[1:2, 2:3])
        # ax13 = fig.add_subplot(gs[1:2, 3:4])

        # overall dff averages
        ax20 = fig.add_subplot(gs[2:3, 0:1])
        ax21 = fig.add_subplot(gs[2:3, 1:2])
        # ax22 = fig.add_subplot(gs[2:3, 2:3])
        # ax23 = fig.add_subplot(gs[2:3, 3:4])

        # crp dff averages
        ax30 = fig.add_subplot(gs[3:4, 0:1])
        ax31 = fig.add_subplot(gs[3:4, 1:2])
        # ax32 = fig.add_subplot(gs[3:4, 2:3])
        # ax33 = fig.add_subplot(gs[3:4, 3:4])

        # crn dff averages
        ax40 = fig.add_subplot(gs[4:5, 0:1])
        ax41 = fig.add_subplot(gs[4:5, 1:2])
        # ax42 = fig.add_subplot(gs[4:5, 2:3])
        # ax43 = fig.add_subplot(gs[4:5, 3:4])

        ax50 = fig.add_subplot(gs[5:6, 0:1])
        ax51 = fig.add_subplot(gs[5:6, 1:2])
        ax52 = fig.add_subplot(gs[5:6, 2:3])
        ax53 = fig.add_subplot(gs[5:6, 3:4])

        ax60 = fig.add_subplot(gs[6:7, 0:1])
        ax61 = fig.add_subplot(gs[6:7, 1:2])
        ax62 = fig.add_subplot(gs[6:7, 2:3])
        ax63 = fig.add_subplot(gs[6:7, 3:4])

        ax70 = fig.add_subplot(gs[7:8, 0:1])
        ax71 = fig.add_subplot(gs[7:8, 1:2])
        ax72 = fig.add_subplot(gs[7:8, 2:3])
        ax73 = fig.add_subplot(gs[7:8, 3:4])


        def remove_spines(ax):
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)

        ax00.axis('off')
        ax01.axis('off')
        ax02.axis('off')

        remove_spines(ax10)
        remove_spines(ax11)
        remove_spines(ax20)
        remove_spines(ax21)
        remove_spines(ax30)
        remove_spines(ax31)
        remove_spines(ax40)
        remove_spines(ax41)

        plot_masks_functions(mask_file, ax00, ax01, ax02)

        ax00.set_title(f'Max Projection, {roi_short} ROIs detected')
        ax01.set_title(f'Mean Projection, {roi_short} ROIs detected')
        ax02.set_title(f'Masks Projection, {roi_short} ROIs detected')

        ax10.plot(short_data_normed['time'], short_data_normed['mean1'], color = 'red',
        label = f'{number_short_crp} trials')
        ax10.fill_between(short_data_normed['time'], short_data_normed['mean1'] -short_data_normed['std1'],
        short_data_normed['mean1'] + short_data_normed['std1'], alpha=0.4, color='red')

        ax11.plot(long_data_normed['time'], long_data_normed['mean1'], color = 'red',
        label = f'{number_long_crp} trials')
        ax11.fill_between(long_data_normed['time'], long_data_normed['mean1'] -long_data_normed['std1'],
        long_data_normed['mean1'] + long_data_normed['std1'], alpha=0.4, color='red')

        ax10.plot(short_data_normed['time'], short_data_normed['mean0'], color = 'blue',
        label = f'{number_short_crn} trials')
        ax10.fill_between(short_data_normed['time'], short_data_normed['mean0'] -short_data_normed['std0'],
        short_data_normed['mean0'] + short_data_normed['std0'], alpha=0.4, color='blue')

        ax11.plot(long_data_normed['time'], long_data_normed['mean0'], color = 'blue',
        label = f'{number_long_crn} trials')
        ax11.fill_between(long_data_normed['time'], long_data_normed['mean0'] -long_data_normed['std0'],
        long_data_normed['mean0'] + long_data_normed['std0'], alpha=0.4, color='blue')

        ax10.plot(long_data_normed['time'], all_short_mean, color = 'purple',
                  label = f'{number_short_crp + number_short_crn} trials')
        ax10.fill_between(long_data_normed['time'], all_short_mean -all_short_sem,
                          all_short_mean + all_short_sem, alpha=0.4, color='purple')

        ax11.plot(long_data_normed['time'], all_long_mean, color = 'purple',
                  label = f'{number_long_crp + number_long_crn} trials')
        ax11.fill_between(long_data_normed['time'], all_long_mean -all_long_sem,
                          all_long_mean + all_long_sem, alpha=0.4, color='purple')


        ax10.set_xlim(-100, 600)
        ax11.set_xlim(-100, 600)

        ax10.axvspan(0, 50, color="gray", alpha=0.3, label="LED")
        ax10.axvspan(200, 220, color="blue", alpha=0.3, label="AirPuff")
        ax10.legend()

        ax10.set_ylabel('FEC (+/- SEM)')
        ax10.set_xlabel("Time from LED Onset")
        ax10.set_title(f'FEC')

        ax11.axvspan(0, 50, color="gray", alpha=0.3, label="LED")
        ax11.axvspan(400, 420, color="lime", alpha=0.3, label="AirPuff")
        ax11.legend()


        ax11.set_ylabel('FEC (+/- SEM)')
        ax11.set_xlabel("Time from LED Onset")
        ax11.set_title(f'FEC')

        for (color, colormap, cr_stat, axdff0, axdff1, axhm0, axhm1, axhm2, axhm3, stacked_short_dff, stacked_long_dff,
             number_short, number_long, number_rois_short, number_rois_long,
             avg_dff_short, sem_dff_short, avg_dff_long, sem_dff_long,
             short_stack_short_sorted, short_stack_long_sorted, 
            long_stack_short_sorted, long_stack_long_sorted)in [

            ('purple', 'Purples', 'CR+ and CR-', ax20, ax21, ax50, ax51, ax52, ax53, short_avg_dff_br, long_avg_dff_br,
             number_short_crp + number_short_crn, number_long_crp + number_long_crn, roi_short, roi_long,
             short_br_avg_roi, short_br_sem_roi, long_br_avg_roi, long_br_sem_roi,
             short_dff_stack_sorted_short_max, short_dff_stack_sorted_long_max,
            long_dff_stack_sorted_short_max, long_dff_stack_sorted_long_max), 

            ('red', 'Reds', 'CR+', ax30, ax31, ax60, ax61, ax62, ax63, short_crp_avg_dff, long_crp_avg_dff,
             number_short_crp , number_long_crp , roi_short, roi_long,
             short_crp_avg_roi, short_crp_sem_roi, long_crp_avg_roi, long_crp_sem_roi,
             short_dff_stack_sorted_short_max_crp, short_dff_stack_sorted_long_max_crp,
            long_dff_stack_sorted_short_max_crp, long_dff_stack_sorted_long_max_crp),

            ('blue', 'Blues' , 'CR-', ax40, ax41, ax70, ax71, ax72, ax73, short_crn_avg_dff, long_crn_avg_dff,
             number_short_crn , number_long_crn, roi_short, roi_long,
             short_crn_avg_roi, short_crn_sem_roi, long_crn_avg_roi, long_crn_sem_roi,
             short_dff_stack_sorted_short_max_crn, short_dff_stack_sorted_long_max_crn,
            long_dff_stack_sorted_short_max_crn, long_dff_stack_sorted_long_max_crn)]:

            axdff0.plot(time, avg_dff_short - avg_dff_short[0], color = color, 
            label = f'\n {number_short} trials \n {number_rois_short} ROIs')
            axdff0.fill_between(time, avg_dff_short - avg_dff_short[0] - sem_dff_short,
            avg_dff_short - avg_dff_short[0] + sem_dff_short, alpha=0.2, color=color)

            ymin0 = np.min(avg_dff_short - avg_dff_short[0] - sem_dff_short)
            ymax0 = np.max(avg_dff_short - avg_dff_short[0] + sem_dff_short)

            axdff0.axvspan(0, 50, color="gray", alpha=0.3, label="LED")
            axdff0.axvspan(200, 220, color="blue", alpha=0.3, label="AirPuff")

            axdff0.legend()
            axdff0.set_xlim(-100, 600)
            axdff0.set_ylabel("Mean df/f (+/- SEM)")
            axdff0.set_xlabel("Time from LED Onset")
            axdff0.set_title(f'df/f {cr_stat}')

            axdff1.plot(time, avg_dff_long - avg_dff_long[0], color = color, 
            label = f'{number_long} trials \n {number_rois_short} ROIs')
            axdff1.fill_between(time, avg_dff_long - sem_dff_long - avg_dff_long[0], 
            avg_dff_long + sem_dff_long - avg_dff_long[0], alpha=0.2, color=color)

            ymin1 = np.min(avg_dff_long - avg_dff_long[0] - sem_dff_long)
            ymax1 = np.max(avg_dff_long - avg_dff_long[0] + sem_dff_long)

            axdff1.axvspan(0, 50, color="gray", alpha=0.3, label="LED")
            axdff1.axvspan(400, 420, color="lime", alpha=0.3, label="AirPuff")

            axdff1.legend()
            axdff1.set_xlim(-100, 600)
            axdff1.set_ylabel("Mean df/f (+/- SEM)")
            axdff1.set_xlabel("Time from LED Onset")
            axdff1.set_title(f'df/f {cr_stat}')

            ymin = min(ymin0, ymin1)
            ymax = max(ymax0, ymax1)

            axdff0.set_ylim(ymin, ymax)
            axdff1.set_ylim(ymin, ymax)


            vmin = min(short_stack_short_sorted.min(), long_dff_stack_sorted_short_max.min())
            vmax = max(short_stack_short_sorted.max(), long_dff_stack_sorted_short_max.max())

            y_extent = [0, short_stack_short_sorted.shape[0]]  # Full height of data
            im1 = axhm0.imshow(short_stack_short_sorted, aspect='auto', 
                 extent=[time[0], time[-1], y_extent[0], y_extent[1]],
                 cmap=colormap, vmin =vmin, vmax = vmax)

            axhm0.set_title(f"Trial-Averaged dF/F: {cr_stat}, Short Trials, Sorted by Peak Time Short")
            axhm0.set_ylabel("Neurons (sorted based on short) for short heatmap")
            axhm0.axvline(0, color='gray', label='LED', linestyle = '--')
            axhm0.axvline(50, color='gray', linestyle = '--')
            axhm0.axvline(200, color='blue', linestyle = '--')
            axhm0.axvline(220, color='blue', linestyle = '--')


            y_extent = [0, long_stack_short_sorted.shape[0]]  # Full height of data
            im2 = axhm1.imshow(long_stack_short_sorted, aspect='auto', 
                 extent=[time[0], time[-1], y_extent[0], y_extent[1]],
                 cmap=colormap, vmin =vmin, vmax = vmax)

            axhm1.set_title(f"Trial-Averaged dF/F: {cr_stat}, Long Trials, Sorted by Peak Time Short")
            axhm1.set_ylabel("Neurons (sorted based on short) for long heatmap")
            axhm1.axvline(0, color='gray', label='LED', linestyle = '--')
            axhm1.axvline(50, color='gray', linestyle = '--')
            axhm1.axvline(400, color='lime', linestyle = '--')
            axhm1.axvline(420, color='lime', linestyle = '--')
            y_extent = [0, short_stack_long_sorted.shape[0]]  # Full height of data
            im3 = axhm2.imshow(short_stack_long_sorted, aspect='auto', 
                 extent=[time[0], time[-1], y_extent[0], y_extent[1]],
                 cmap=colormap, vmin =vmin, vmax = vmax)

            axhm2.set_title(f"Trial-Averaged dF/F: {cr_stat}, Short Trials, Sorted by Peak Time Long")
            axhm2.set_ylabel("Neurons (sorted based on long) for short heatmap")
            axhm2.axvline(0, color='gray', label='LED', linestyle = '--')
            axhm2.axvline(50, color='gray', linestyle = '--')
            axhm2.axvline(200, color='blue', linestyle = '--')
            axhm2.axvline(220, color='blue', linestyle = '--')

            y_extent = [0, long_stack_long_sorted.shape[0]]  # Full height of data
            im4 = axhm3.imshow(long_stack_long_sorted, aspect='auto', 
                 extent=[time[0], time[-1], y_extent[0], y_extent[1]],
                 cmap=colormap, vmin =vmin, vmax = vmax)

            axhm3.set_title(f"Trial-Averaged dF/F: {cr_stat}, Short Trials, Sorted by Peak Time Long")
            axhm3.set_ylabel("Neurons (sorted based on long) for short heatmap")
            axhm3.axvline(0, color='gray', label='LED', linestyle = '--')
            axhm3.axvline(50, color='gray', linestyle = '--')
            axhm3.axvline(400, color='lime', linestyle = '--')
            axhm3.axvline(420, color='lime', linestyle = '--')
            # add_colorbar(im1, ax[h, 0])

            add_colorbar(im2, axhm3)

            plt.tight_layout()

        with PdfPages(session_br_file) as pdf:
            pdf.savefig(fig, dpi = 400)
            pdf.close()

        print(f"PDF successfully saved: {session_br_file}")
