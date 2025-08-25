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
from utils.indication import find_max_with_gradient, CR_stat_indication, CR_FEC, block_type, find_index, cr_onset_calc


beh_folder = "./data/beh"
# early_trials_check = "early"
early_trials_check = 'full'
mice = [folder for folder in os.listdir(beh_folder)] 
for mouse in mice:
    if mouse != 'E4L7':
        continue

    all_sessions = [os.path.splitext(file)[0] for file in os.listdir(os.path.join(beh_folder, mouse,"processed"))if file != ".DS_Store"]
    i = 0
    trial_derivatives0 = {}
    trial_derivatives1 = {}
    peaks0 = {}
    peaks1 = {}
    if not os.path.exists(f'./outputs/beh/{mouse}'):
            os.mkdir(f'./outputs/beh/{mouse}')
            print(f"data folder '{mouse}' created.")
    else:
        print(f"data folder '{mouse}' already exists.")
    
    slopes = []
    sig_id_all = {}
    for i, session_date in enumerate(all_sessions):

        sig_id_all[session_date] = []
        # try:
        print("processing session" , session_date)
        static_threshold = 0.02
        static_averaging_window_a = 6
        static_averaging_window_v = 10
        moving_avg_window_size = 20
        adaptation_summary_file = f"./outputs/beh/{mouse}/adapt_{session_date}.pdf"
        if not os.path.exists(adaptation_summary_file):
            os.mkdir(adaptation_summary_file)

        trials = h5py.File(f"./data/beh/{mouse}/processed/{session_date}.h5")["trial_id"]

        fec, fec_time_0, trials = fec_zero(trials)
        fec_0 = moving_average(fec , window_size=10)
        fec_normed, fec_time_0 = fec_crop(fec_0, fec_time_0)
        shorts, longs = block_type(trials)
        CR_stat, CR_interval_avg, base_line_avg, cr_interval_idx, bl_interval_idx  = CR_stat_indication(trials, fec_normed, fec_time_0 , static_threshold = static_threshold, AP_delay = 3)
        all_id = sort_numbers_as_strings(shorts + longs)

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

        increment = 20

        if early_trials_check == 'early':
            first_start = all_id.index(sig_id[0])
            segments.append(all_id[:increment])
            # Iterate through the significant IDs
            for i in range(len(sig_id) - 1):
                start = all_id.index(sig_id[i])
                end = all_id.index(sig_id[i + 1])
                segments.append(all_id[start +1 :start + 1 + increment])

        else:
            first_start = all_id.index(sig_id[0])
            segments.append(all_id[:first_start])
            # Iterate through the significant IDs
            for i in range(len(sig_id) - 1):
                start = all_id.index(sig_id[i])
                end = all_id.index(sig_id[i + 1])
                segments.append(all_id[start +1 :end + 1])

            # # Handle the last segment (from the last sig_id to the end of all_id)
            # last_start = all_id.index(sig_id[-1])
            # segments.append(all_id[last_start+1:])


        # Handle the last segment (from the last sig_id to the end of all_id)
        last_start = all_id.index(sig_id[-1])
        segments.append(all_id[last_start+1:last_start+ increment + 1])

        baselines_crp, cr_amplitudes_crp, cr_relative_changes_crp, baselines_crn, cr_amplitudes_crn, cr_relative_changes_crn = CR_FEC(base_line_avg, CR_interval_avg, CR_stat)

        # Create a PDF file to save the plots
        pdf_file = adaptation_summary_file
        pdf = PdfPages(pdf_file)

        # Initialize the figure and axes
        fig, axes = plt.subplots(11, len(segments), figsize=(7*len(segments), 7*12), sharex=False, sharey=False)

        # First set of plots (Signal Amplitudes - Row 1)
        for seg_i, segmented_id in enumerate(segments):
            colors = plt.cm.Purples(np.linspace(1, 0.1, len(segmented_id)))[:: -1]
            axes[0, seg_i].axvline(fec_time_0[segmented_id[-1]][cr_interval_idx[segmented_id[-1]][0]], color = 'gray', linestyle='--', label = 'CR interval onset')
            axes[0, seg_i].axvline(fec_time_0[segmented_id[-1]][bl_interval_idx[segmented_id[-1]][0]], color = 'orange', linestyle='--', label = 'Base line interval onset')
            axes[0, seg_i].axvspan(0, trials[segmented_id[-1]]["LED"][1] - trials[segmented_id[-1]]["LED"][0], 
                                   color="gray", alpha=0.5)
            axes[0, seg_i].axvspan(trials[segmented_id[-1]]["AirPuff"][0] - trials[segmented_id[-1]]["LED"][0], 
                                   trials[segmented_id[-1]]["AirPuff"][1] - trials[segmented_id[-1]]["LED"][0], 
                                   color="lime" if sig_id_type[seg_i] == 0 else "blue", alpha=0.5)
            # axes[11, seg_i].axvspan(0, trials[segmented_id[-1]]["LED"][1] - trials[segmented_id[-1]]["LED"][0], 
                                   # color="gray", alpha=0.5)
            # axes[11, seg_i].axvspan(trials[segmented_id[-1]]["AirPuff"][0] - trials[segmented_id[-1]]["LED"][0], 
                                   # trials[segmented_id[-1]]["AirPuff"][1] - trials[segmented_id[-1]]["LED"][0], 
                                   # color="lime" if sig_id_type[seg_i] == 0 else "blue", alpha=0.5)


            for i, idx in enumerate(segmented_id):
                label = f'Trial {idx}' if idx == segmented_id[0] or idx == segmented_id[-1] else None
                axes[0, seg_i].plot(fec_time_0[idx], fec_normed[idx] , label=label , color=colors[i])
                # if seg_i != 0:
                    # if sig_id_type[seg_i] == 0:
                        # axes[11, 3].plot(fec_time_0[idx], fec_normed[idx] , label=label, alpha = 0.2 , color=colors[i])
                    # else:
                        # axes[11, 2].plot(fec_time_0[idx], fec_normed[idx] , label=label, alpha = 0.2, color=colors[i])
                # else:
                    # axes[11, 0].plot(fec_time_0[idx], fec_normed[idx] , label=label, alpha = 0.2, color=colors[i])
            axes[0, seg_i].spines['top'].set_visible(False)
            axes[0, seg_i].spines['right'].set_visible(False)
            if seg_i != 0:
                axes[0, seg_i].set_title(f"Block Number: {seg_i} (Short to Long)" if sig_id_type[seg_i] == 0 else f"Block Number: {seg_i} (Long to Short)")
                # if sig_id_type[seg_i] == 0:
                #     axes[11, 2].plot(fec_time_0[idx], fec_normed[idx] , label=label , color=colors[i])
            else:
                axes[0, seg_i].set_title("first block")
            axes[0, seg_i].set_xlabel('Time (ms)')
            axes[0, seg_i].set_xlim(-100, 500)
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
            axes[1, seg_i].axvline(fec_time_0[segmented_id[-1]][cr_interval_idx[segmented_id[-1]][0]], color = 'gray', linestyle='--', label = 'CR interval onset')
            axes[1, seg_i].axvline(fec_time_0[segmented_id[-1]][bl_interval_idx[segmented_id[-1]][0]], color = 'orange', linestyle='--', label = 'Base line interval onset')
            averaged_fec = 0
            axes[1, seg_i].axvspan(0, trials[segmented_id[-1]]["LED"][1] - trials[segmented_id[-1]]["LED"][0], 
                                   color="gray", alpha=0.5, label="LED")
            axes[1, seg_i].axvspan(trials[segmented_id[-1]]["AirPuff"][0] - trials[segmented_id[-1]]["LED"][0], 
                                   trials[segmented_id[-1]]["AirPuff"][1] - trials[segmented_id[-1]]["LED"][0], 
                                   color="lime" if sig_id_type[seg_i] == 0 else "blue", alpha=0.5, label="AirPuff")
            # axes[12, seg_i].axvspan(0, trials[segmented_id[-1]]["LED"][1] - trials[segmented_id[-1]]["LED"][0], 
            #                        color="gray", alpha=0.5, label="LED")
            # axes[12, seg_i].axvspan(trials[segmented_id[-1]]["AirPuff"][0] - trials[segmented_id[-1]]["LED"][0], 
            #                        trials[segmented_id[-1]]["AirPuff"][1] - trials[segmented_id[-1]]["LED"][0], 
            #                        color="lime" if sig_id_type[seg_i] == 0 else "blue", alpha=0.5, label="AirPuff")

            avg_fec = []
            for i, idx in enumerate(segmented_id):
                # averaged_fec += fec_normed[idx]
                print(len(fec_normed[id]))
                avg_fec.append(fec_normed[idx])
                if i % static_averaging_window_a == 0 and i != 0:
                    avg_fec = np.vstack(avg_fec)
                    print(avg_fec.shape)
                    plotting_value = np.nanmean(avg_fec, axis = 0)
                    avg_fec = []
                    # if max(averaged_fec / static_averaging_window_a) > 60:
                    label = f'Trial {idx}' if idx == segmented_id[0] or idx == segmented_id[-1] else None
                    axes[1, seg_i].plot(fec_time_0[idx], plotting_value, label=label, color=colors[i])
                    # if seg_i != 0:
                    #     if sig_id_type[seg_i] == 0:
                    #         axes[12, 3].plot(fec_time_0[idx], plotting_value, label=label, alpha = 0.2, color=colors[i])
                    #     else:
                    #         axes[12, 2].plot(fec_time_0[idx], plotting_value, label=label, alpha = 0.2, color=colors[i])
                    # else:
                    #     axes[12, 0].plot(fec_time_0[idx], plotting_value, label=label, alpha = 0.2, color=colors[i])
                    averaged_fec = 0
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
            axes[2, seg_i].axvline(fec_time_0[segmented_id[-1]][cr_interval_idx[segmented_id[-1]][0]], color = 'gray', linestyle='--', label = 'CR interval onset')
            axes[2, seg_i].axvline(fec_time_0[segmented_id[-1]][bl_interval_idx[segmented_id[-1]][0]], color = 'orange', linestyle='--', label = 'Base line interval onset')
            axes[2, seg_i].axvspan(0, trials[segmented_id[-1]]["LED"][1] - trials[segmented_id[-1]]["LED"][0], 
                                   color="gray", alpha=0.5)

            axes[2, seg_i].axvspan(trials[segmented_id[-1]]["AirPuff"][0] - trials[segmented_id[-1]]["LED"][0], 
                                   trials[segmented_id[-1]]["AirPuff"][1] - trials[segmented_id[-1]]["LED"][0], 
                                   color="lime" if sig_id_type[seg_i] == 0 else "blue", alpha=0.5)
            for i, idx in enumerate(segmented_id):
                moving_avg_fec = uniform_filter1d(fec_normed[idx], size=moving_avg_window_size)
                label = f'Trial {idx}' if idx == segmented_id[0] or idx == segmented_id[-1] else None
                axes[2, seg_i].plot(fec_time_0[idx], moving_avg_fec / 100,  color=colors[i], label=label)
            axes[2, seg_i].spines['top'].set_visible(False)
            axes[2, seg_i].spines['right'].set_visible(False)
            if seg_i != 0:
                axes[2, seg_i].set_title(f"Block Number: {seg_i} (Short to Long)" if sig_id_type[seg_i] == 0 else f"Block Number: {seg_i} (Long to Short)")
            else:
                axes[2, seg_i].set_title("first block")
            axes[2, seg_i].set_xlabel('Time (ms)')
            axes[2, seg_i].set_xlim(-100, 500)
            axes[2, seg_i].set_ylabel(f'FEC (Moving Avg. over {moving_avg_window_size} points)')
            axes[2, seg_i].legend()

        # Add color bar for the second set of plots
        cmap = plt.cm.Greens
        norm = mpl.colors.Normalize()
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=axes[2, -1], orientation='vertical', label='Trial Progression (Earlier to Later)', pad=0.01)

        # Fourth set of plots (CR Relative Changes - Row 3)
        for seg_i, segmented_id in enumerate(segments):
            for i, idx in enumerate(segmented_id):
                try:
                    if CR_stat[idx] == 1:
                        axes[3, seg_i].scatter(int(idx), cr_relative_changes_crp[idx], color='red')
                        axes[3, seg_i].scatter(int(idx), baselines_crp[idx], color='green', alpha=0.4)
                    if CR_stat[idx] == 0:
                        axes[3, seg_i].scatter(int(idx), cr_relative_changes_crn[idx], color='blue')
                        axes[3, seg_i].scatter(int(idx), baselines_crn[idx], color='green', alpha=0.4)
                except Exception as e:
                    print(f"Exceptin happend {e}")
                    print(idx)
                    continue
            axes[3, seg_i].spines['top'].set_visible(False)
            axes[3, seg_i].spines['right'].set_visible(False)
            if seg_i != 0:
                axes[3, seg_i].set_title(f"Block Number: {seg_i} (Short to Long)" if sig_id_type[seg_i] == 0 else f"Block Number: {seg_i} (Long to Short)")
            else:
                axes[3, seg_i].set_title("first block")
            axes[3, seg_i].axhline(static_threshold, color='gray', linestyle='--', linewidth=0.8)
            axes[3, seg_i].set_ylabel('CR Relative Change')
            axes[3, seg_i].set_xlabel('Trials')
            axes[3, seg_i].legend(
                handles=[
                    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label="CR+ - Baseline amplitude"),
                    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label="CR- - Baseline amplitude"),
                    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=8, alpha=0.4, label="Baseline"),
                    plt.Line2D([0], [0], linestyle='--', markerfacecolor='gray',   label="Static threshold for CR identification")
                ], 
                loc="upper left"
            )

        # Fifth set of plots for the distance of the peak form the Ap onset.
        for seg_i, segmented_id in enumerate(segments):
            peak_time = {}
            peak_value = {}
            peak_distance = {}

            for i, idx in enumerate(segmented_id):
                peaks0[idx] = []
                peaks1[idx] = []
                peak_time[idx], peak_value[idx], _, gradients = find_max_with_gradient(fec_time_0[idx][bl_interval_idx[idx][1]: cr_interval_idx[idx][1]], fec_normed[idx][bl_interval_idx[idx][1]: cr_interval_idx[idx][1]])
                if peak_time[idx]:
                    peak_distance[idx] = trials[segmented_id[-1]]["AirPuff"][1]- trials[segmented_id[-1]]["LED"][0] - peak_time[idx]
                    axes[4, seg_i].plot(i, peak_distance[idx], 'o', color='green', label='Peak Distance' if i == 0 else "")
                else:
                    axes[4, seg_i].axvspan(i - 0.5, i + 0.5, color='lime', alpha=0.1, label='No Peak' if i == 0 else "")


            axes[4, seg_i].spines['top'].set_visible(False)
            axes[4, seg_i].spines['right'].set_visible(False)
            if seg_i != 0:
                axes[4, seg_i].set_title(f"Block Number: {seg_i} (Short to Long)" if sig_id_type[seg_i] == 0 else f"Block Number: {seg_i} (Long to Short)")
            else:
                axes[4, seg_i].set_title("first block")
            axes[4, seg_i].set_xlabel('Trial')
            axes[4, seg_i].set_xlim(-0.5, len(segmented_id) - 0.5)
            axes[4, seg_i].set_ylabel('distance of the peak of the FEC signal to the AirPuff onset. (ms)')

            # Add legend if needed
            axes[4, seg_i].legend(loc='upper right')

        y_min = np.inf
        y_max = -np.inf

        # First pass: Find global y-axis limits
        for seg_i, segmented_id in enumerate(segments):
            avg_derivative = {}

            for i, idx in enumerate(segmented_id):
                avg_derivative[idx] = ((fec_normed[idx][cr_interval_idx[idx][1]] - fec_normed[idx][cr_interval_idx[idx][0]]) / 
                                       (fec_time_0[idx][cr_interval_idx[idx][1]] - fec_time_0[idx][cr_interval_idx[idx][0]]))
                
                y_min = min(y_min, avg_derivative[idx])
                y_max = max(y_max, avg_derivative[idx])
                if seg_i==0:
                    if idx not in trial_derivatives0:
                        trial_derivatives0[idx] = []
                    trial_derivatives0[idx].append(avg_derivative[idx])  # Store value for the current session

                if seg_i==1:
                    if idx not in trial_derivatives1:
                        trial_derivatives1[idx] = []
                    trial_derivatives1[idx].append(avg_derivative[idx])  # Store value for the current session

        # Second pass: Plotting with the same y-axis limits
        for seg_i, segmented_id in enumerate(segments):
            avg_derivative = {}
            points = []  # To store all (x, y) points for this segment

            for i, idx in enumerate(segmented_id):
                avg_derivative[idx] = ((fec_normed[idx][cr_interval_idx[idx][1]] - fec_normed[idx][cr_interval_idx[idx][0]]) / 
                                       (fec_time_0[idx][cr_interval_idx[idx][1]] - fec_time_0[idx][cr_interval_idx[idx][0]]))
                points.append((i, avg_derivative[idx]))
                ### average the slopes over all 
                axes[5, seg_i].plot(i, avg_derivative[idx], 'o', color='green', label='Average derivative in the CR interval window' if i == 0 else "")

            # Create convex hull to outline the points
            if len(points) > 2:  # ConvexHull requires at least 3 points
                points = np.array(points)
                hull = ConvexHull(points)
                hull_vertices = points[hull.vertices]

                # Create a polygon from the hull
                polygon = Polygon(hull_vertices, closed=False, color='lime', alpha=0.1, edgecolor='none')
                axes[5, seg_i].add_patch(polygon)

            for trial_idx0, trial_derivatives_list0 in trial_derivatives0.items():  #ADDED: Iterate through all trials
                avg_derivative0 = np.mean(trial_derivatives_list0)  #ADDED: Calculate average derivative for this trial
                sem_derivative0 = np.std(trial_derivatives_list0) / np.sqrt(len(trial_derivatives_list0))  #ADDED: Calculate SEM
                # if seg_i == 0:
                    # axes[7, seg_i].errorbar(trial_idx0, avg_derivative0, yerr=sem_derivative0, fmt='o', color='green', label='Averaged Trial Derivative' if trial_idx0 == 0 else "")  #ADDED: Plot average point with SEM bars

            for trial_idx1, trial_derivatives_list1 in trial_derivatives1.items():  #ADDED: Iterate through all trials
                avg_derivative1 = np.mean(trial_derivatives_list1)  #ADDED: Calculate average derivative for this trial
                sem_derivative1 = np.std(trial_derivatives_list1) / np.sqrt(len(trial_derivatives_list1))  #ADDED: Calculate SEM
            
                # if seg_i == 1:
                    # axes[7, seg_i].errorbar(trial_idx1, avg_derivative1, yerr=sem_derivative1, fmt='o', color='green', label='Averaged Trial Derivative' if trial_idx1 == 0 else "")  #ADDED: Plot average point with SEM bars

            # Set the same y-axis limits for all plots
            # axes[5, seg_i].set_ylim(y_min, y_max)

            axes[5, seg_i].spines['top'].set_visible(False)
            axes[5, seg_i].spines['right'].set_visible(False)
            if seg_i != 0:
                axes[5, seg_i].set_title(f"Block Number: {seg_i} (Short to Long)" if sig_id_type[seg_i] == 0 else f"Block Number: {seg_i} (Long to Short)")
            else:
                axes[5, seg_i].set_title("first block")
            axes[5, seg_i].set_xlabel('Trial')
            axes[5, seg_i].set_xlim(-0.5, len(segmented_id) - 0.5)
            axes[5, seg_i].set_ylabel('Average gradient of the FEC signal in the CR interval window')

            # Add legend if needed
            axes[5, seg_i].legend(loc='upper right')
            # axes[7, seg_i].set_ylim(y_min, y_max)

            # axes[7, seg_i].spines['top'].set_visible(False)
            # axes[7, seg_i].spines['right'].set_visible(False)
            # axes[7, seg_i].set_title(f"Block Number: {seg_i} (Short to Long)" if sig_id_type[seg_i] == 0 else f"Block Number: {seg_i} (Long to Short)")
            # axes[7, seg_i].set_xlabel('Trial')
            # axes[7, seg_i].set_xlim(-0.5, len(segmented_id) - 0.5)
            # axes[7, seg_i].set_ylabel('Average os FEC slope ine the CR window (+/- SEM) over sessions')

            # # Add legend if needed
            # axes[7, seg_i].legend(loc='upper right')



        cmap = cm.Reds

        # Loop over each segment to create the 2D plot
        for seg_i, segmented_id in enumerate(segments):
            peak_time = {}
            peak_value = {}
            peak_distance = {}
            avg_derivative = {}

            # Store the (distance_of_peak, avg_derivative) points and their trial progression
            points = []
            trial_indices = []  # Renaming to avoid confusion

            for i, idx in enumerate(segmented_id):
                peak_time[idx], peak_value[idx], _, gradients = find_max_with_gradient(fec_time_0[idx][cr_interval_idx[idx][0]: cr_interval_idx[idx][1]], fec_normed[idx][cr_interval_idx[idx][0]: cr_interval_idx[idx][1]])
                
                if peak_time[idx]:
                    peak_distance[idx] = trials[segmented_id[-1]]["AirPuff"][1] - trials[segmented_id[-1]]["LED"][0] - peak_time[idx]
                    avg_derivative[idx] = ((fec_normed[idx][cr_interval_idx[idx][1]] - fec_normed[idx][cr_interval_idx[idx][0]]) / 
                                           (fec_time_0[idx][cr_interval_idx[idx][1]] - fec_time_0[idx][cr_interval_idx[idx][0]]))
                    
                    points.append((peak_distance[idx], avg_derivative[idx]))
                    trial_indices.append(i)  # Storing the trial index for color progression
                else:
                    # Handle cases where no peak is found
                    points.append((np.nan, np.nan))  # Add empty data for points with no peak

            # Convert points and trial_indices to numpy arrays
            points = np.array(points)
            trial_indices = np.array(trial_indices)

            # Remove NaN values from both points and trial_indices
            valid_points = ~np.isnan(points[:, 0]) & ~np.isnan(points[:, 1])  # Find valid points where neither x nor y is NaN
            
            # Keep only valid points and corresponding trial indices
            points = points[valid_points]

            # Normalize trial indices to map them to the colormap
            # norm = plt.Normalize(trial_indices.min(), trial_indices.max())
            colors = cmap(norm(trial_indices))  # Get color based on trial progression

            # Create the 2D plot for this segment
            ax = axes[6, seg_i]  # Assuming axes[6, seg_i] is available for this plot
            sc = ax.scatter(points[:, 0], points[:, 1], c=colors, label=f"Segment {seg_i}", edgecolors='black', alpha=0.7)
            
            # Add labels, title, and colorbar
            ax.set_xlabel('Distance of Peak (ms)')
            ax.set_ylabel('Average Gradient in CR Window')
            if seg_i != 0:
                ax.set_title(f"Block Number: {seg_i} (Short to Long)" if sig_id_type[seg_i] == 0 else f"Block Number: {seg_i} (Long to Short)")
            else:
                ax.set_title("first block")
            
                    # Remove the top and right spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # Add legend if needed
            ax.legend(loc='upper right')
     # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # Empty array for colorbar
        cbar = plt.colorbar(sm, ax=axes[6, -1])
        cbar.set_label('Trial Progression')

        # plotting derivatives #################
        for seg_i, segmented_id in enumerate(segments):
            local_min = []
            local_max = []
            colors = plt.cm.Purples(np.linspace(1, 0.1, len(segmented_id)))[:: -1]
            axes[7, seg_i].axvline(fec_time_0[segmented_id[-1]][cr_interval_idx[segmented_id[-1]][0]], color = 'gray', linestyle='--', label = 'CR interval onset')
            axes[7, seg_i].axvline(fec_time_0[segmented_id[-1]][bl_interval_idx[segmented_id[-1]][0]], color = 'orange', linestyle='--', label = 'Base line interval onset')
            axes[7, seg_i].axvspan(0, trials[segmented_id[-1]]["LED"][1] - trials[segmented_id[-1]]["LED"][0], 
                                   color="gray", alpha=0.5)
            axes[7, seg_i].axvspan(trials[segmented_id[-1]]["AirPuff"][0] - trials[segmented_id[-1]]["LED"][0], 
                                   trials[segmented_id[-1]]["AirPuff"][1] - trials[segmented_id[-1]]["LED"][0], 
                                   color="lime" if sig_id_type[seg_i] == 0 else "blue", alpha=0.5)
            # axes[13, seg_i].axvspan(0, trials[segmented_id[-1]]["LED"][1] - trials[segmented_id[-1]]["LED"][0], 
            #                        color="gray", alpha=0.5)
            # axes[13, seg_i].axvspan(trials[segmented_id[-1]]["AirPuff"][0] - trials[segmented_id[-1]]["LED"][0], 
            #                        trials[segmented_id[-1]]["AirPuff"][1] - trials[segmented_id[-1]]["LED"][0], 
            #                        color="lime" if sig_id_type[seg_i] == 0 else "blue", alpha=0.5)
            # axes[14, seg_i].axvspan(0, trials[segmented_id[-1]]["LED"][1] - trials[segmented_id[-1]]["LED"][0], 
            #                        color="gray", alpha=0.5)
            # axes[14, seg_i].axvspan(trials[segmented_id[-1]]["AirPuff"][0] - trials[segmented_id[-1]]["LED"][0], 
            #                        trials[segmented_id[-1]]["AirPuff"][1] - trials[segmented_id[-1]]["LED"][0], 
            #                        color="lime" if sig_id_type[seg_i] == 0 else "blue", alpha=0.5)
            # axes[14, seg_i].set_xlim(-100, 600)
            # axes[13, seg_i].set_xlim(-100, 600)
            # axes[12, seg_i].set_xlim(-100, 600)
            # # axes[11, seg_i].set_xlim(-100, 600)

            for i, idx in enumerate(segmented_id):
                label = f'Trial {idx}' if idx == segmented_id[0] or idx == segmented_id[-1] else None
                #finding the x limit indices
                ending_idx = find_index(fec_time_0[idx], 500)
                starting_idx = find_index(fec_time_0[idx], -100)
                plotting_value = np.gradient(fec_normed[idx], fec_time_0[idx])[starting_idx: ending_idx]
                local_max.append(np.max(plotting_value))
                local_min.append(np.min(plotting_value))
                axes[7, seg_i].plot(fec_time_0[idx][starting_idx: ending_idx],  plotting_value, label=label , color=colors[i])
                # if seg_i != 0:
                #     if sig_id_type[seg_i] == 0:
                #         axes[13, 3].plot(fec_time_0[idx][starting_idx: ending_idx],  plotting_value, label=label, alpha = 0.2 , color=colors[i])
                #     else:
                #         axes[13, 2].plot(fec_time_0[idx][starting_idx: ending_idx],  plotting_value, label=label, alpha = 0.2 , color=colors[i])
                # else:
                #     axes[13, 0].plot(fec_time_0[idx][starting_idx: ending_idx],  plotting_value, label=label, alpha = 0.2 , color=colors[i])


            axes[7, seg_i].spines['top'].set_visible(False)
            axes[7, seg_i].spines['right'].set_visible(False)
            if seg_i != 0:
                axes[7, seg_i].set_title(f"Block Number: {seg_i} (Short to Long)" if sig_id_type[seg_i] == 0 else f"Block Number: {seg_i} (Long to Short)")
            else:
                axes[7, seg_i].set_title("first block")
            axes[7, seg_i].set_xlabel('Time (ms)')
            axes[7, seg_i].set_xlim(-100, 500)
            axes[7, seg_i].set_ylim(np.min(local_min), np.max(local_max))
            axes[7, seg_i].set_ylabel('FEC over trials')
            axes[7, seg_i].legend()

        cmap = plt.cm.Purples
        norm = mpl.colors.Normalize()
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=axes[7, -1], orientation='vertical', label='Trial Progression (Earlier to Later)', pad=0.01)

        for seg_i, segmented_id in enumerate(segments):
            local_max = []
            local_min = []
            colors = plt.cm.Reds(np.linspace(1, 0.1, len(segmented_id)))[:: -1]
            axes[8, seg_i].axvline(fec_time_0[segmented_id[-1]][cr_interval_idx[segmented_id[-1]][0]], color = 'gray', linestyle='--', label = 'CR interval onset')
            axes[8, seg_i].axvline(fec_time_0[segmented_id[-1]][bl_interval_idx[segmented_id[-1]][0]], color = 'orange', linestyle='--', label = 'Base line interval onset')
            averaged_fec = 0
            axes[8, seg_i].axvspan(0, trials[segmented_id[-1]]["LED"][1] - trials[segmented_id[-1]]["LED"][0], 
                                   color="gray", alpha=0.5, label="LED")
            axes[8, seg_i].axvspan(trials[segmented_id[-1]]["AirPuff"][0] - trials[segmented_id[-1]]["LED"][0], 
                                   trials[segmented_id[-1]]["AirPuff"][1] - trials[segmented_id[-1]]["LED"][0], 
                                   color="lime" if sig_id_type[seg_i] == 0 else "blue", alpha=0.5, label="AirPuff")
            avg_fec = []
            for i, idx in enumerate(segmented_id):
                ending_idx = find_index(fec_time_0[idx], 500)
                starting_idx = find_index(fec_time_0[idx], -100)
                if i % static_averaging_window_v == 0:
                    avg_fec.append(np.gradient(fec_normed[idx][starting_idx: ending_idx], fec_time_0[idx][starting_idx: ending_idx]))
                    plotting_value = np.nanmean(avg_fec, axis = 0)
                    label = f'Trial {idx}' if idx == segmented_id[0] or idx == segmented_id[-1] else None
                    
                    local_max.append(np.max(plotting_value))
                    local_min.append(np.min(plotting_value))
                    axes[8, seg_i].plot(fec_time_0[idx][starting_idx: ending_idx], plotting_value, label=label, color=colors[i])
                    # if seg_i != 0:
                    #     if sig_id_type[seg_i] == 0:
                    #         axes[14, 3].plot(fec_time_0[idx][starting_idx: ending_idx], plotting_value, label=label, alpha = 0.2, color=colors[i])
                    #     else:
                    #         axes[14, 2].plot(fec_time_0[idx][starting_idx: ending_idx], plotting_value, label=label, alpha = 0.2, color=colors[i])
                    # else:
                    #     axes[14, 0].plot(fec_time_0[idx][starting_idx: ending_idx], plotting_value, label=label, alpha = 0.2, color=colors[i])

                    avg_fec = []
                    
            axes[8, seg_i].spines['top'].set_visible(False)
            axes[8, seg_i].spines['right'].set_visible(False)
            if seg_i != 0:
                axes[8, seg_i].set_title(f"Block Number: {seg_i} (Short to Long)" if sig_id_type[seg_i] == 0 else f"Block Number: {seg_i} (Long to Short)")
            else:
                axes[8, seg_i].set_title("first block")
            axes[8, seg_i].set_xlabel('Time (ms)')
            axes[8, seg_i].set_xlim(-100, 500)
            print(seg_i , local_min)
            axes[8, seg_i].set_ylim(np.min(local_min), np.max(local_max))
            axes[8, seg_i].set_ylabel(f'FEC (Avg. over {static_averaging_window_v} trials)')

        cmap = plt.cm.Reds
        norm = mpl.colors.Normalize()
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=axes[8, -1], orientation='vertical', label='Trial Progression (Earlier to Later)', pad=0.01)

        for seg_i, segmented_id in enumerate(segments):
            local_min = []
            local_max = []
            colors = plt.cm.Greens(np.linspace(1, 0.1, len(segmented_id)))[:: -1]
            axes[9, seg_i].axvline(fec_time_0[segmented_id[-1]][cr_interval_idx[segmented_id[-1]][0]], color = 'gray', linestyle='--', label = 'CR interval onset')
            axes[9, seg_i].axvline(fec_time_0[segmented_id[-1]][bl_interval_idx[segmented_id[-1]][0]], color = 'orange', linestyle='--', label = 'Base line interval onset')
            axes[9, seg_i].axvspan(0, trials[segmented_id[-1]]["LED"][1] - trials[segmented_id[-1]]["LED"][0], 
                                   color="gray", alpha=0.5)
            axes[9, seg_i].axvspan(trials[segmented_id[-1]]["AirPuff"][0] - trials[segmented_id[-1]]["LED"][0], 
                                   trials[segmented_id[-1]]["AirPuff"][1] - trials[segmented_id[-1]]["LED"][0], 
                                   color="lime" if sig_id_type[seg_i] == 0 else "blue", alpha=0.5)
            for i, idx in enumerate(segmented_id):
                ending_idx = find_index(fec_time_0[idx], 500)
                starting_idx = find_index(fec_time_0[idx], -100)
                moving_avg_fec = uniform_filter1d(np.gradient(fec_normed[idx][starting_idx: ending_idx], fec_time_0[idx][starting_idx: ending_idx]), size=moving_avg_window_size)/ 100
                label = f'Trial {idx}' if idx == segmented_id[0] or idx == segmented_id[-1] else None
                local_min.append(np.min(moving_avg_fec))
                local_max.append(np.max(moving_avg_fec))
                axes[9, seg_i].plot(fec_time_0[idx][starting_idx: ending_idx], moving_avg_fec ,  color=colors[i], label=label)
            axes[9, seg_i].spines['top'].set_visible(False)
            axes[9, seg_i].spines['right'].set_visible(False)
            if seg_i != 0:
                axes[9, seg_i].set_title(f"block number:{seg_i} (Short to Long)" if sig_id_type[seg_i] == 0 else f"block number:{seg_i} (Long to Short)")
            else:

                axes[9, seg_i].set_title("first block")
            axes[9, seg_i].set_xlabel('Time (ms)')
            axes[9, seg_i].set_xlim(-100, 500)
            axes[9, seg_i].set_ylim(np.min(local_min) , np.max(local_max))
            axes[9, seg_i].set_ylabel(f'FEC (Moving Avg. over {moving_avg_window_size} points)')
            axes[9, seg_i].legend()

        # Add color bar for the second set of plots
        cmap = plt.cm.Greens
        norm = mpl.colors.Normalize()
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=axes[9, -1], orientation='vertical', label='Trial Progression (Earlier to Later)', pad=0.01)

        for seg_i, segmented_id in enumerate(segments):
            for idx in segmented_id:
                airpuff = trials[idx]['AirPuff'][0]- trials[idx]["LED"][0]
                veolcity_threshold_fraction = 0.95
                amp_threshold_fraction = 0.10
                print("cr:", CR_stat[idx])
                cr_idx = cr_onset_calc(
                    fec_normed[idx], fec_time_0[idx], 10, airpuff, CR_stat[idx],
                )
                axes[10, seg_i].scatter(idx, fec_time_0[idx][cr_idx], color='orange')

            axes[10, seg_i].spines['top'].set_visible(False)
            axes[10, seg_i].spines['right'].set_visible(False)
            if seg_i != 0:
                axes[10, seg_i].set_title(f"Block Number: {seg_i} (Short to Long)" if sig_id_type[seg_i] == 0 else f"Block Number: {seg_i} (Long to Short)")
            else:
                axes[10, seg_i].set_title("first block")
            axes[10, seg_i].set_xlabel('Trial')
            axes[10, seg_i].tick_params(axis='x', labelrotation=45, labelsize = 5)
            axes[10, seg_i].set_xlim(-0.5, len(segmented_id) - 0.5)
            axes[10, seg_i].set_ylabel('CR onset distance from Airpuff onset (ms)')
      
        print('Done')
        pdf.savefig(fig)
        pdf.close()
        # # break 
        # except Exception as e:
        #     print(f'prblem with session{session_date} {e}')
