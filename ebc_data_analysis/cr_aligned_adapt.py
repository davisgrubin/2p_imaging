import os
import numpy as np
import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages 
from matplotlib.patches import Polygon
import h5py
from scipy.ndimage import uniform_filter1d
from scipy.spatial import ConvexHull

from utils.alignment import fec_cr_aligned, moving_average, sort_numbers_as_strings, fec_zero, fec_crop
from utils.indication import find_max_with_gradient, CR_stat_indication, CR_FEC, block_type, find_index, cr_onset_calc, cr_loops


beh_folder = "./data/beh"
mice = [folder for folder in os.listdir(beh_folder)] 
for mouse in mice:
    all_sessions = [os.path.splitext(file)[0] for file in os.listdir(os.path.join(beh_folder, mouse,"processed"))if file != ".DS_Store"]
    # print(all_sessions)
    i = 0

    if not os.path.exists(f'./outputs/beh/{mouse}'):
            os.mkdir(f'./outputs/beh/{mouse}')
            print(f"data folder '{mouse}' created.")
    else:
        print(f"data folder '{mouse}' already exists.")
    
    session_folder = [folder for folder in os.listdir(f'./data/beh/{mouse}/')]
    slopes = []
    sig_id_all = {}
    for session_date in all_sessions:
        sig_id_all[session_date] = []
        # try:
        print("processing session" , session_date)
        static_threshold = 0.02
        static_averaging_window_a = 6
        static_averaging_window_v = 10
        moving_avg_window_size = 20
        output_path = f"./outputs/beh/{mouse}/cr_aligned"
        if not os.path.exists(output_path):
            os.mkdir(output_path)
            print("path made")
        adaptation_summary_file = os.path.join(output_path, f"adaptation_summary_{session_date}.pdf")
        print(adaptation_summary_file)

        trials = h5py.File(f"./data/beh/{mouse}/processed/{session_date}.h5")["trial_id"]

        fec, fec_time, trials = fec_zero(trials)
        shorts, longs = block_type(trials)
        CR_stat, CR_interval_avg, base_line_avg, cr_interval_idx, bl_interval_idx  = CR_stat_indication(trials, fec, fec_time , static_threshold = static_threshold, AP_delay = 3)
        all_id = sort_numbers_as_strings(shorts + longs)

        cr_times, cr_indices = cr_loops(trials, fec, fec_time, CR_stat, cr_interval_idx, bl_interval_idx, CR_interval_avg, base_line_avg,)
        fec, fec_time, trials = fec_cr_aligned(trials, cr_times)

        # fec = moving_average(fec , window_size=15)

        # fec, fec_time = fec_crop(fec, fec_time)

        sig_id = []
        sig_id_type = [] #0 for short to long and 1 for long to short
        segments = []
        if trials[list(all_id)[0]]["trial_type"][()] == 1:
            sig_id_type.append(1)
        if trials[list(all_id)[0]]["trial_type"][()] == 2:
            sig_id_type.append(0)
        for i ,id in enumerate(all_id):
            try:
                if trials[id]["trial_type"][()] != trials[str(int(id) + 1)]["trial_type"][()]:
                    sig_id.append(id)
                    sig_id_all[session_date].append(id)
                    if trials[id]["trial_type"][()] == 1:
                        sig_id_type.append(0)
                    if trials[id]["trial_type"][()] == 2:
                        sig_id_type.append(1)
            except Exception as e:
                print(f"Exception {e} happend")
                continue

        if len(sig_id) < 3:
            print(f"the session {session_date} has no change in its trial types")
            continue

        increment = 10

        first_start = all_id.index(sig_id[0])
        segments.append(all_id[:first_start])
        # Iterate through the significant IDs
        for i in range(len(sig_id) - 1):
            start = all_id.index(sig_id[i])
            end = all_id.index(sig_id[i + 1])
            segments.append(all_id[start +1 :end + 1])

        # Handle the last segment (from the last sig_id to the end of all_id)
        last_start = all_id.index(sig_id[-1])
        segments.append(all_id[last_start+1:])

        # for the segments with only the first few trials
        # first_start = all_id.index(sig_id[0])
        # segments.append(all_id[:increment])
        # # Iterate through the significant IDs
        # for i in range(len(sig_id) - 1):
        #     start = all_id.index(sig_id[i])
        #     end = all_id.index(sig_id[i + 1])
        #     segments.append(all_id[start +1 :start + 1 + increment])

        # # Handle the last segment (from the last sig_id to the end of all_id)
        # last_start = all_id.index(sig_id[-1])
        # segments.append(all_id[last_start+1:last_start+ increment + 1])

        baselines_crp, cr_amplitudes_crp, cr_relative_changes_crp, baselines_crn, cr_amplitudes_crn, cr_relative_changes_crn = CR_FEC(base_line_avg, CR_interval_avg, CR_stat)

        # Create a PDF file to save the plots
        pdf_file = adaptation_summary_file
        pdf = PdfPages(pdf_file)

        # Initialize the figure and axes
        fig, axes = plt.subplots(11, len(segments), figsize=(7*len(segments), 7*12), sharex=False, sharey=False)

        # First set of plots (Signal Amplitudes - Row 1)
        for seg_i, segmented_id in enumerate(segments):
            colors = plt.cm.Purples(np.linspace(1, 0.1, len(segmented_id)))[:: -1]

            for i, idx in enumerate(segmented_id):
                label = f'Trial {idx}' if idx == segmented_id[0] or idx == segmented_id[-1] else None
                axes[0, seg_i].plot(fec_time[idx], fec[idx] , label=label , color=colors[i])

            axes[0, seg_i].spines['top'].set_visible(False)
            axes[0, seg_i].spines['right'].set_visible(False)
            if seg_i != 0:
                axes[0, seg_i].set_title(f"Block Number: {seg_i} (Short to Long)" if sig_id_type[seg_i] == 0 else f"Block Number: {seg_i} (Long to Short)")
            
            else:
                axes[0, seg_i].set_title("first block")
            axes[0, seg_i].axvline(0, color='gray', linestyle='--', alpha = 0.5)
            axes[0, seg_i].set_xlabel('Time (ms)')
            axes[0, seg_i].set_xlim(-100, 500)
            # axes[0, seg_i].set_xlim(-20, 20)
            axes[0, seg_i].set_ylabel('FEC over trials')
            axes[0, seg_i].legend()
            print(trials[segmented_id[0]])

        # Add color bar for the first set of plots
        cmap = plt.cm.Purples
        norm = mpl.colors.Normalize()
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=axes[0, -1], orientation='vertical', label='Trial Progression (Earlier to Later)', pad=0.01)

        # Second set of plots (Averaged Signal Amplitudes - Row 2)
        for seg_i, segmented_id in enumerate(segments):
            colors = plt.cm.Reds(np.linspace(1, 0.1, len(segmented_id)))[:: -1]
           
            averaged_fec = 0
          
            avg_fec = []
            for i, idx in enumerate(segmented_id):
                # averaged_fec += fec[idx]
                print(len(fec[id]))
                avg_fec.append(fec[idx])
                if i % static_averaging_window_a == 0 and i != 0:
                    try:
                        avg_fec = np.vstack(avg_fec)
                        print(avg_fec.shape)
                        plotting_value = np.nanmean(avg_fec, axis = 0)
                        avg_fec = []
                        # if max(averaged_fec / static_averaging_window_a) > 60:
                        label = f'Trial {idx}' if idx == segmented_id[0] or idx == segmented_id[-1] else None
                        axes[1, seg_i].plot(fec_time[idx], plotting_value, label=label, color=colors[i])
             
                        averaged_fec = 0
                    except Exception as ex:
                        print(ex)
                        continue
            axes[1, seg_i].axvline(0, color='gray', linestyle='--', alpha = 0.5)
            axes[1, seg_i].spines['top'].set_visible(False)
            axes[1, seg_i].spines['right'].set_visible(False)
            if seg_i != 0:
                axes[1, seg_i].set_title(f"Block Number: {seg_i} (Short to Long)" if sig_id_type[seg_i] == 0 else f"Block Number: {seg_i} (Long to Short)")
            else:
                axes[1, seg_i].set_title("first block")
            axes[1, seg_i].set_xlabel('Time (ms)')
            axes[1, seg_i].set_xlim(-100, 500)
            axes[1, seg_i].set_ylabel(f'FEC (Avg. over {static_averaging_window_a} trials)')

        # Add color bar for the second set of plots
        cmap = plt.cm.Reds
        norm = mpl.colors.Normalize()
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        # fig.colorbar(sm, ax=axes[1, -1], orientation='vertical', label='Trial Progression (Earlier to Later)', pad=0.01)

        # Third set of plots (moving average)
        for seg_i, segmented_id in enumerate(segments):
            colors = plt.cm.Greens(np.linspace(1, 0.1, len(segmented_id)))[:: -1]
        
            for i, idx in enumerate(segmented_id):
                moving_avg_fec = uniform_filter1d(fec[idx], size=moving_avg_window_size)
                label = f'Trial {idx}' if idx == segmented_id[0] or idx == segmented_id[-1] else None
                axes[2, seg_i].plot(fec_time[idx], moving_avg_fec / 100,  color=colors[i], label=label)
            axes[2, seg_i].spines['top'].set_visible(False)
            axes[2, seg_i].spines['right'].set_visible(False)
            if seg_i != 0:
                axes[2, seg_i].set_title(f"Block Number: {seg_i} (Short to Long)" if sig_id_type[seg_i] == 0 else f"Block Number: {seg_i} (Long to Short)")
            else:
                axes[2, seg_i].set_title("first block")

            axes[2, seg_i].axvline(0, color='gray', linestyle='--', alpha = 0.5)
            axes[2, seg_i].set_xlabel('Time (ms)')
            axes[2, seg_i].set_xlim(-100, 500)
            axes[2, seg_i].set_ylabel(f'FEC (Moving Avg. over {moving_avg_window_size} points)')
            axes[2, seg_i].legend()
        pdf.savefig(fig)
        pdf.close()
        # break 
        # except Exception as e:
        #     print(f'prblem with session{session_date} {e}')
