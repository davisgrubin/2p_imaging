import os
from scipy.stats import stats
import numpy as np
from scipy.stats import sem  # Correct import
import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages 
from matplotlib.patches import Polygon
import h5py
from scipy.ndimage import uniform_filter1d
from scipy.spatial import ConvexHull

from utils.alignment import moving_average, sort_numbers_as_strings, fec_zero
from utils.indication import find_max_with_gradient, CR_stat_indication, CR_FEC, block_type, find_index, cr_onset_calc, threshold_time
from plotting.plots import get_color_map_shades

color_scale_short = ['#cce5ff', '#99ccff', '#66b2ff', '#3399ff', '#007fff', '#0066cc', '#0052a3']  # Light to dark blue  
color_scale_long = ['#d9ffb3', '#bfff80', '#a6ff4d', '#8cff1a', '#73e600', '#5cb800', '#459900']  # Light to dark lime

number_of_stacks= 6 #should be the same as the max number os segments
transtion_0 = 5
transtion_1 = 10
increment = 10

beh_folder = "./data/beh"

early_trials_check = None
# early_trials_check = 'early'

transition_focused = None
# transition_focused = 'transition'

late_trials_check = None
# late_trials_check = 'late'

for test_type_check in [1]: #1 for control and 0 for SD



    test_type_text = None
    if test_type_check == 1:
        test_type_text = 'control'
    if test_type_check == 0:
        test_type_text = 'SD'



    mice = [folder for folder in os.listdir(beh_folder) if folder != ".DS_Store"] 
    for mouse in mice:
        if mouse != 'E6LG':
            continue

        all_sessions = [os.path.splitext(file)[0] for file in os.listdir(os.path.join(beh_folder, mouse, "processed")) if file != ".DS_Store"]
        if not os.path.exists(f'./outputs/beh/{mouse}'):
                os.mkdir(f'./outputs/beh/{mouse}')
                print(f"data folder '{mouse}' created.")
        else:
            print(f"data folder '{mouse}' already exists.")

        adaptation_mouse_file = f"./outputs/beh/{mouse}/adapt_summary_{mouse}_{test_type_text}.pdf"
        adaptation_summary_summary = f"./outputs/beh/{mouse}/adaptation_{mouse}_summary_2.pdf"

        session_folder = [folder for folder in os.listdir(f'./data/beh/{mouse}/')]
        sig_type = {}

        stacked_slope = {}
        stacked_peak = {}
        stacked_cr = {}
        stacked_isi = {}

        avg_slope = {}
        avg_peak = {}
        avg_cr = {}
        avg_isi = {}
        sem_isi = {}
        sem_slope = {}
        sem_peak = {}
        sem_cr = {}
        avg_number = {}

        avg_vel0 = []
        avg_vel1 = []
        avg_vel2 = []

        avg_trace0 = []
        avg_trace1 = []
        avg_trace2 = []

        for i in range(number_of_stacks): #number for the number of stacked segments

            stacked_slope[i] = {}
            stacked_peak[i] = {}
            stacked_cr[i] = {}
            stacked_isi[i] = {}
            avg_slope[i] = {}
            avg_peak[i] = {}
            avg_cr[i] = {}
            avg_isi[i] = {}
            sem_isi[i] = {}
            sem_slope[i] = {}
            sem_peak[i] = {}
            sem_cr[i] = {}
            avg_number[i] = {}

        #loop oveer all sesssions to find the max number of segments and exclude the onse that have odd starting types
        sess_slope = {}
        sess_peak = {}
        sess_cr = {}
        for i , session_date in enumerate(all_sessions):

            print(session_date)

            sess_slope[session_date] = {}
            sess_peak[session_date] = {}
            sess_cr[session_date] = {}

            for i in range(number_of_stacks): #number for the number of stacked segments
                sess_slope[session_date][i] = []
                sess_peak[session_date][i] = []
                sess_cr[session_date][i] = []

            trials = h5py.File(f"./data/beh/{mouse}/processed/{session_date}.h5")["trial_id"]
            fec, fec_time_0, trials = fec_zero(trials)
            shorts, longs = block_type(trials)
            all_id = sort_numbers_as_strings(shorts + longs)

            test_type = trials['1']["test_type"][()]
            if test_type == test_type_check:
                # breakpoint()
                continue


            # here we are trying to find the significant ids so we can find where are the transitions.
            sig_id = [] # this is hte significant id that we see the changes of trials types in it.
            sig_id_type = [] #0 for short to long and 1 for long to short
            segments = [] # this includes all the ids of all segments in it.

            if trials[list(all_id)[0]]["trial_type"][()] == 1:
                sig_type[session_date] = 1
                sig_id_type.append(1)
            if trials[list(all_id)[0]]["trial_type"][()] == 2:
                sig_type[session_date] = 0
                sig_id_type.append(0)

            # finding the sig trials and their types
            for i ,id in enumerate(all_id):
                try:
                    next_id = 1

                    while (str(int(id) + next_id)) not in trials:
                        next_id += 1
                        if next_id == len(all_id):
                            break

                    if trials[id]["trial_type"][()] != trials[str(int(id) + next_id)]["trial_type"][()]:
                        sig_id.append(id)
                        if trials[id]["trial_type"][()] == 1:
                            sig_id_type.append(0)
                        if trials[id]["trial_type"][()] == 2:
                            sig_id_type.append(1)

                except Exception as e:
                    print(f"Exception: {e}")
                    continue

            if trials['0']["trial_type"][()] == 2:
                continue

            if transition_focused:
                for i in range(len(sig_id) - 1):
                    segments.append(all_id[all_id.index(sig_id[i])- transtion_0 : all_id.index(sig_id[i]) + transtion_1])
                    # segments.append(all_id[all_id.index(sig_id[i])- transtion_0 : all_id.index(sig_id[i])])



            if late_trials_check:
                for i in range(len(sig_id) - 1):
                    segments.append(all_id[all_id.index(sig_id[i]) : all_id.index(sig_id[i]) + transtion_1])

            if early_trials_check:
                #the first segment
                first_start = all_id.index(sig_id[0])
                segments.append(all_id[:increment])

                # Iterate through the significant IDs
                for i in range(len(sig_id) - 1):
                    start = all_id.index(sig_id[i])
                    end = all_id.index(sig_id[i + 1])
                    segments.append(all_id[start +1 :start + 1 + increment])

                # Handle the last segment (from the last sig_id to the end of all_id)
                last_start = all_id.index(sig_id[-1])
                segments.append(all_id[last_start+1:last_start+1 + increment])

            else:
                #the first segment
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

            for i in range(len(segments)):
                print('*',i)
                for i, id in enumerate(segments[i]):
                    if i > 8 and i < 13:
                        print(i)
                        print(trials[id]["trial_type"][()])
            # breakpoint()

            for seg_i, segment_id in enumerate(segments):
                if seg_i > number_of_stacks-1:
                    # print("break")
                    break
                for i,idx in enumerate(segment_id):
                    stacked_slope[seg_i][idx] = [] 
                    stacked_peak[seg_i][idx] = []
                    stacked_cr[seg_i][idx] = []
                    stacked_isi[seg_i][idx] = []

                print(f'{len(segments)}')
                print(len(stacked_slope[1]))
                # breakpoint()
                


        print(stacked_slope[0])
        # breakpoint()
        j = 0
        session_idx = 0
        total_sessions = len(all_sessions)
        fig, axs = plt.subplots(7, number_of_stacks, figsize=(7*7, 7*7))
        fig1, axs1 = plt.subplots(4, 2, figsize=(7*2, 7*4))

        ##############################################################################
        # appending everything to the corresponding stack
        ##############################################################################

        for i , session_date in enumerate(all_sessions):
            
            session_idx += 1
            j +=1
            # if j>10:
            #     print("idk")
            #     break
            print("processing session" , session_date)
            static_threshold = 0.02
            static_averaging_window_a = 10
            static_averaging_window_v = 10
            moving_avg_window_size = 1
            output_path = f"./outputs/beh/{mouse}"
            if not os.path.exists(output_path):
                os.mkdir(output_path)
            adaptation_summary_file = os.path.join(output_path, f"adaptation_summary_{session_date}.pdf")

            trials = h5py.File(f"./data/beh/{mouse}/processed/{session_date}.h5")["trial_id"]
            fec, fec_time_0, trials = fec_zero(trials)
            fec_0 = moving_average(fec , window_size=10)
            # fec_normed sig_id_type min_max_normalize(fec_0)
            fec_normed = fec_0
            shorts, longs = block_type(trials)
            CR_stat, CR_interval_avg, base_line_avg, cr_interval_idx, bl_interval_idx  = CR_stat_indication(trials , fec_0, fec_time_0, static_threshold = static_threshold, AP_delay = 3)
            all_id = sort_numbers_as_strings(shorts + longs)

            test_type = trials['1']["test_type"][()]
            if test_type == test_type_check:
                print('hello')
                # breakpoint()
                continue


            sig_id = []
            sig_id_type = [] #0 for short to long and 1 for long to short
            segments = []

            if trials[list(all_id)[0]]["trial_type"][()] == 1:
                sig_type[session_date] = 1
                sig_id_type.append(1)
            if trials[list(all_id)[0]]["trial_type"][()] == 2:
                sig_type[session_date] = 0
                sig_id_type.append(0)

            # finding the sig trials and their types
            for i ,id in enumerate(all_id):
                try:
                    next_id = 1

                    while (str(int(id) + next_id)) not in trials:
                        next_id += 1
                        if next_id == len(all_id):
                            break

                    if trials[id]["trial_type"][()] != trials[str(int(id) + next_id)]["trial_type"][()]:
                        sig_id.append(id)
                        if trials[id]["trial_type"][()] == 1:
                            sig_id_type.append(0)
                        if trials[id]["trial_type"][()] == 2:
                            sig_id_type.append(1)

                except Exception as e:
                    print(f"Exception: {e}")
                    continue

            if trials['0']["trial_type"][()] == 2:
                continue

            if len(sig_id) < 2:
                print(f"the session {session_date} has no change in its trial types")
                continue


            if transition_focused:
                for i in range(len(sig_id) - 1):
                    segments.append(all_id[int(sig_id[i])- transtion_0 : all_id.index(sig_id[i]) + transtion_1])
                    # segments.append(all_id[all_id.index(sig_id[i])- transtion_0 : all_id.index(sig_id[i])])

            if late_trials_check:
                for i in range(len(sig_id) - 1):
                    segments.append(all_id[all_id.index(sig_id[i]) : all_id.index(sig_id[i]) + transtion_1])

            if early_trials_check:
                #the first segment
                first_start = all_id.index(sig_id[0])
                segments.append(all_id[:increment])

                # Iterate through the significant IDs
                for i in range(len(sig_id) - 1):
                    start = all_id.index(sig_id[i])
                    end = all_id.index(sig_id[i + 1])
                    segments.append(all_id[start +1 :start + 1 + increment])

                # Handle the last segment (from the last sig_id to the end of all_id)
                last_start = all_id.index(sig_id[-1])
                segments.append(all_id[last_start+1:last_start+1 + increment])
            else:
                # defining segments and adding them to an array

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


            baselines_crp, cr_amplitudes_crp, cr_relative_changes_crp, baselines_crn, cr_amplitudes_crn, cr_relative_changes_crn = CR_FEC(base_line_avg, CR_interval_avg, CR_stat)
            for seg_i, segment_id in enumerate(segments):
                avg_derivative = {}
                derivative = {}
                peak_time = {}
                peak_value = {}
                peak_distance = {}

                velocity_threshold_fraction = 0.95
                amp_threshold_fraction = 0.10

                if seg_i > number_of_stacks-1:
                    print("break")
                    break
                

                avg_fec = []
                averaged_fec = 0
                #calculating the average for all first segments:
                for i, idx in enumerate(segment_id):

                    avg_derivative[idx] = ((fec_normed[idx][cr_interval_idx[idx][1]] - fec_normed[idx][cr_interval_idx[idx][0]]) / 
                                               (fec_time_0[idx][cr_interval_idx[idx][1]] - fec_time_0[idx][cr_interval_idx[idx][0]]))
                                   
                    isi_time = trials[idx]['AirPuff'][0]- trials[idx]["LED"][0]
                    peak_time[idx], peak_value[idx], _ = find_max_with_gradient(fec_time_0[idx][bl_interval_idx[idx][1]: cr_interval_idx[idx][1]], fec_normed[idx][bl_interval_idx[idx][1]: cr_interval_idx[idx][1]])
                    if peak_time[idx] is None:
                        peak_time[idx] = trials[segment_id[-1]]["AirPuff"][1]- trials[segment_id[-1]]["LED"][0]

                    # peak_distance[idx] = trials[segment_id[-1]]["AirPuff"][1]- trials[segment_id[-1]]["LED"][0] - peak_time[idx]
                    peak_distance[idx] = peak_time[idx]

                    airpuff = trials[idx]['AirPuff'][0]- trials[idx]["LED"][0]
                    cr_idx = cr_onset_calc(
                            fec[idx], fec_time_0[idx], 10, airpuff, CR_stat[idx])

                    stacked_slope[seg_i][idx].append(avg_derivative[idx]) 
                    stacked_peak[seg_i][idx].append(peak_distance[idx])
                    if fec_time_0[idx][cr_idx] != 400 and fec_time_0[idx][cr_idx] != 200:
                        stacked_cr[seg_i][idx].append(fec_time_0[idx][cr_idx])

                    stacked_isi[seg_i][idx].append(isi_time)
                    
                    
                    # plotting average derivatives for all trials all segments super imposed (individual)
                    custom_color_map = get_color_map_shades(session_idx, total_sessions, number_of_stacks, colormap= 'rainbow') 

                    if trials[idx]["trial_type"][()] == 1 and seg_i % 2 == 0: #short ones
                        if peak_distance[idx] > 5 and peak_distance[idx] != 0 and peak_distance[idx] < 200:
                            sess_slope[session_date][seg_i].append(avg_derivative[idx])
                            sess_peak[session_date][seg_i].append(peak_distance[idx])
                            sess_cr[session_date][seg_i].append(fec_time_0[idx][cr_idx])

                        if fec_time_0[idx][cr_idx] > 5 and fec_time_0[idx][cr_idx] != 0 and fec_time_0[idx][cr_idx] < 200:
                            sess_slope[session_date][seg_i].append(avg_derivative[idx])
                            sess_peak[session_date][seg_i].append(peak_distance[idx])
                            sess_cr[session_date][seg_i].append(fec_time_0[idx][cr_idx])


                    if trials[idx]["trial_type"][()] == 2 and seg_i % 2 == 1: #long ones
                        if peak_distance[idx] > 5  and peak_distance[idx] !=0 and peak_distance[idx] < 400:
                            sess_slope[session_date][seg_i].append(avg_derivative[idx])
                            sess_peak[session_date][seg_i].append(peak_distance[idx])
                            sess_cr[session_date][seg_i].append(fec_time_0[idx][cr_idx])

                        if fec_time_0[idx][cr_idx] > 5 and fec_time_0[idx][cr_idx] !=0 and fec_time_0[idx][cr_idx] < 400:
                            sess_slope[session_date][seg_i].append(avg_derivative[idx])
                            sess_peak[session_date][seg_i].append(peak_distance[idx])
                            sess_cr[session_date][seg_i].append(fec_time_0[idx][cr_idx])

        y_min_0, y_max_0 = [], []
        y_min_1, y_max_1 = [], []
        y_min_2, y_max_2 = [], []
        # Plot the data for each segment in separate columns
        avg_slope_shorts = []
        avg_slope_longs = []
        avg_peak_shorts = []
        avg_peak_longs = []
        avg_cr_shorts = []
        avg_cr_longs = []
        for seg_i in range(number_of_stacks):

            # 1. Plot Averaged Slopes with SEM
            avg_slope_vals, sem_slope_vals, min_slope, max_slope = [], [], [], []
            for idx in stacked_slope[seg_i]:
                valid_vals = np.array(stacked_slope[seg_i][idx])
                # valid_vals = valid_vals[~np.isnan(valid_vals)]  # Remove NaN values
                
                if len(valid_vals) > 0:
                    avg_slope[seg_i][idx] = np.mean(valid_vals)
                    sem_slope[seg_i][idx] = np.nanstd(valid_vals) / np.sqrt(len(valid_vals))  # SEM
                    avg_slope_vals.append(avg_slope[seg_i][idx])
                    sem_slope_vals.append(sem_slope[seg_i][idx])
                    min_slope.append(avg_slope[seg_i][idx] - sem_slope[seg_i][idx])
                    max_slope.append(avg_slope[seg_i][idx] + sem_slope[seg_i][idx])
            y_min = np.min(min_slope)
            y_max = np.max(max_slope)
            
            axs[0, seg_i].plot(range(len(avg_slope_vals)), avg_slope_vals, label=f'Segment {seg_i}')
            axs[0, seg_i].fill_between(range(len(avg_slope_vals)), np.subtract(avg_slope_vals, sem_slope_vals), np.add(avg_slope_vals, sem_slope_vals), alpha=0.2, color='lime' if seg_i % 2 == 1 else 'blue')
            if seg_i != 0:
                axs[0, seg_i].set_title(f"{seg_i} Long to short" if seg_i % 2 == 1 else f"{seg_i} Short to long")
                if seg_i % 2 == 1:
                    avg_slope_longs.append(np.array(avg_slope_vals[0 : transtion_0 + transtion_1]))
                    axs[5,0].plot(range(len(avg_slope_vals)), avg_slope_vals, color=color_scale_long[seg_i], label=f'Segment {seg_i}')
                else:
                    avg_slope_shorts.append(np.array(avg_slope_vals[0 : transtion_0 + transtion_1]))
                    axs[5,1].plot(range(len(avg_slope_vals)), avg_slope_vals, color=color_scale_short[seg_i], label=f'Segment {seg_i}')
            else:
                axs[0, seg_i].set_title("first block (short to long)")
            axs[0, seg_i].set_xlabel("relative trials")
            axs[0, seg_i].set_ylabel("Average velocity in the CR window")
            # axs[0, seg_i].set_xlim(0, 51)
            # axs[0, seg_i].set_ylim(y_min, y_max)
            y_min_0.append(y_min)
            y_max_0.append(y_max)
            axs[0, seg_i].spines['right'].set_visible(False)
            axs[0, seg_i].spines['top'].set_visible(False)

            # 2. Plot Averaged Peaks with SEM
            avg_peak_vals, sem_peak_vals, min_peak, max_peak = [], [], [], []
            for idx in stacked_peak[seg_i]:
                valid_vals = np.array(stacked_peak[seg_i][idx])
                valid_vals = valid_vals[~np.isnan(valid_vals)]  # Remove NaN values
                
                if len(valid_vals) > 0:
                    avg_peak[seg_i][idx] = np.mean(valid_vals)
                    sem_peak[seg_i][idx] = np.nanstd(valid_vals) / np.sqrt(len(valid_vals))
                    avg_peak_vals.append(avg_peak[seg_i][idx])
                    sem_peak_vals.append(sem_peak[seg_i][idx])
                    min_peak.append(avg_peak[seg_i][idx] - sem_peak[seg_i][idx])
                    max_peak.append(avg_peak[seg_i][idx] + sem_peak[seg_i][idx])

            y_min = np.min(min_peak)
            y_max = np.max(max_peak)
            
            axs[1, seg_i].plot(range(len(avg_peak_vals)), avg_peak_vals,label=f'Segment {seg_i}')
            axs[1, seg_i].fill_between(range(len(avg_peak_vals)), np.subtract(avg_peak_vals, sem_peak_vals), np.add(avg_peak_vals, sem_peak_vals), alpha=0.2, color='lime' if seg_i % 2 == 1 else 'blue')
            if seg_i != 0:
                axs[1, seg_i].set_title(f"{seg_i} Long to short" if seg_i % 2 == 1 else f"{seg_i} Short to long")
                if seg_i % 2 == 1:
                    avg_peak_longs.append(avg_peak_vals[0 : transtion_0 + transtion_1])
                    axs[5, 2].plot(range(len(avg_peak_vals)), avg_peak_vals, color= color_scale_long[seg_i], label=f'Segment {seg_i}')
                else:
                    avg_peak_shorts.append(np.array(avg_peak_vals[0 : transtion_0 + transtion_1]))
                    axs[5, 3].plot(range(len(avg_peak_vals)), avg_peak_vals, color= color_scale_short[seg_i], label=f'Segment {seg_i}')
            else:
                axs[1, seg_i].set_title("first block")
            axs[1, seg_i].set_xlabel("realative trials")
            axs[1, seg_i].set_ylabel("Time from the first presence of peak from LED")
            # axs[1, seg_i].set_xlim(0, 51)
            # axs[1, seg_i].set_ylim(y_min, y_max)
            y_min_1.append(y_min)
            y_max_1.append(y_max)
            axs[1, seg_i].spines['right'].set_visible(False)
            axs[1, seg_i].spines['top'].set_visible(False)

            # 3. Plot Averaged CR with SEM
            avg_cr_vals, sem_cr_vals, min_cr, max_cr = [], [], [], []
            for idx in stacked_cr[seg_i]:
                valid_vals = np.array(stacked_cr[seg_i][idx])
                valid_vals = valid_vals[~np.isnan(valid_vals)]  # Remove NaN values
                
                if len(valid_vals) > 0:
                    avg_cr[seg_i][idx] = np.mean(valid_vals)
                    sem_cr[seg_i][idx] = np.nanstd(valid_vals) / np.sqrt(len(valid_vals))
                    avg_cr_vals.append(avg_cr[seg_i][idx])
                    sem_cr_vals.append(sem_cr[seg_i][idx])
                    min_cr.append(avg_cr[seg_i][idx] - sem_cr[seg_i][idx])
                    max_cr.append(avg_cr[seg_i][idx] + sem_cr[seg_i][idx])

            y_min = np.min(min_cr)
            y_max = np.max(max_cr)

            axs[2, seg_i].plot(range(len(avg_cr_vals)), avg_cr_vals,  label=f'Segment {seg_i}')
            axs[2, seg_i].fill_between(range(len(avg_cr_vals)), np.subtract(avg_cr_vals, sem_cr_vals), np.add(avg_cr_vals, sem_cr_vals), alpha=0.2, color='lime' if seg_i % 2 == 1 else 'blue')
            if seg_i != 0:
                axs[2, seg_i].set_title(f"{seg_i}Long to short" if seg_i % 2 == 1 else f"{seg_i}Short to long")
                if seg_i % 2 == 1:
                    avg_cr_longs.append(avg_cr_vals[0 : transtion_0 + transtion_1])
                    axs[5, 4].plot(range(len(avg_cr_vals)), avg_cr_vals, color= color_scale_long[seg_i], label=f'Segment {seg_i}')
                else:
                    avg_cr_shorts.append(avg_cr_vals[0 : transtion_0 + transtion_1])
                    axs[5, 5].plot(range(len(avg_cr_vals)), avg_cr_vals, color= color_scale_short[seg_i], label=f'Segment {seg_i}')
            else:
                axs[2, seg_i].set_title("first block")


            axs[2, seg_i].set_xlabel("realtive trials")
            # axs[2, seg_i].set_xlim(0, 51)
            # axs[2, seg_i].set_ylim(y_min, y_max)
            y_min_2.append(y_min)
            y_max_2.append(y_max)
            axs[2, seg_i].set_ylabel("Time of CR onset from LED")
            axs[2, seg_i].spines['right'].set_visible(False)
            axs[2, seg_i].spines['top'].set_visible(False)


            # Plot isi time
            avg_isi_vals, sem_isi_vals, min_isi, max_isi = [], [], [], []
            for idx in stacked_isi[seg_i]:
                valid_vals = np.array(stacked_isi[seg_i][idx])
                valid_vals = valid_vals[~np.isnan(valid_vals)]
                
                if len(valid_vals) > 0:
                    avg_isi[seg_i][idx] = np.mean(valid_vals)
                    sem_isi[seg_i][idx] = np.nanstd(valid_vals) / np.sqrt(len(valid_vals))
                    avg_isi_vals.append(avg_isi[seg_i][idx])
                    sem_isi_vals.append(sem_isi[seg_i][idx])
                    min_isi.append(avg_isi[seg_i][idx] - sem_isi[seg_i][idx])
                    max_isi.append(avg_isi[seg_i][idx] + sem_isi[seg_i][idx])

            # Plot isi
            axs[4, seg_i].plot(range(len(avg_isi_vals)), avg_isi_vals, color='black', label=f'Threshold {seg_i}')
            axs[4, seg_i].fill_between(range(len(avg_isi_vals)), np.subtract(avg_isi_vals, sem_isi_vals), np.add(avg_isi_vals, sem_isi_vals), alpha=0.2, color='lime' if seg_i % 2 == 1 else 'blue')
            axs[4, seg_i].set_xlabel("Trials")
            axs[4, seg_i].set_ylabel("mean ISI time (+/- STD)")
            axs[4, seg_i].spines['right'].set_visible(False)
            axs[4, seg_i].spines['top'].set_visible(False)


            # 4. Plot Averaged Number of Data Points (avg_number)
            avg_number_vals = []
            for idx in stacked_slope[seg_i]:  # Assume `avg_number` corresponds to the same index range as slopes
                avg_number[seg_i][idx] = len(stacked_slope[seg_i][idx])  # Average number of data points
                avg_number_vals.append(avg_number[seg_i][idx])

            axs[3, seg_i].plot(range(len(avg_number_vals)), avg_number_vals, 'o',  label=f'Segment {seg_i}')
            if seg_i != 0:
                axs[3, seg_i].set_title(f"Block Number: {seg_i} (Short to Long)" if seg_i % 2 == 1 else f"Block Number: {seg_i} (Long to Short)")
            else:
                axs[3, seg_i].set_title("first block")
            axs[3, seg_i].set_xlabel("relative trials")
            # axs[3, seg_i].set_xlim(0, 51)
            axs[3, seg_i].set_ylabel("Number of sessions went into the averaging for each block")
            axs[3, seg_i].spines['right'].set_visible(False)
            axs[3, seg_i].spines['top'].set_visible(False)

            axs[5,0].set_title("All transitions to short")
            axs[5,1].set_title("All transitions to long")
            axs[5,2].set_title("All transitions to short")
            axs[5,3].set_title("All transitions to long")
            axs[5,4].set_title("All transitions to short")
            axs[5,5].set_title("All transitions to long")

            axs[5,0].set_ylabel("Average velocity")
            axs[5,1].set_ylabel("Average velocity")

            axs[5,2].set_ylabel("Peak Values")
            axs[5,3].set_ylabel("Peak Values")

            axs[5, 4].set_ylabel("CR Values")
            axs[5, 5].set_ylabel("CR Values")

            axs[5, 0].set_xlabel("Trials")
            axs[5, 1].set_xlabel("Trials")
            axs[5, 2].set_xlabel("Trials")
            axs[5, 3].set_xlabel("Trials")
            axs[5, 4].set_xlabel("Trials")
            axs[5, 5].set_xlabel("Trials")

        axs1[0,0].plot(range(transtion_0 + transtion_1), np.mean(avg_slope_shorts, axis=0))
        axs1[0,0].fill_between(
            range(transtion_0 + transtion_1),
            np.mean(avg_slope_shorts, axis=0) - np.std(avg_slope_shorts, axis=0)/len(avg_slope_shorts),
            np.mean(avg_slope_shorts, axis=0) + np.std(avg_slope_shorts, axis=0)/len(avg_slope_shorts),
            color='blue', alpha=0.3
        )

        axs1[0,1].plot(range(transtion_0 + transtion_1), np.mean(avg_slope_longs, axis=0))
        axs1[0,1].fill_between(
            range(transtion_0 + transtion_1),
            np.mean(avg_slope_longs, axis=0) - np.std(avg_slope_longs, axis=0)/len(avg_slope_longs),
            np.mean(avg_slope_longs, axis=0) + np.std(avg_slope_longs, axis=0)/len(avg_slope_longs),
            color='lime', alpha=0.3
        )

        axs1[1,2].plot(range(transtion_0 + transtion_1), np.mean(avg_peak_shorts, axis=0))
        axs1[1,2].fill_between(
            range(transtion_0 + transtion_1),
            np.mean(avg_peak_shorts, axis=0) - np.std(avg_peak_shorts, axis=0)/len(avg_peak_shorts),
            np.mean(avg_peak_shorts, axis=0) + np.std(avg_peak_shorts, axis=0)/len(avg_peak_shorts),
            color='blue', alpha=0.3
        )

        axs1[1,3].plot(range(transtion_0 + transtion_1), np.mean(avg_peak_longs, axis=0))
        axs1[1,3].fill_between(
            range(transtion_0 + transtion_1),
            np.mean(avg_peak_longs, axis=0) - np.std(avg_peak_longs, axis=0)/len(avg_peak_longs),
            np.mean(avg_peak_longs, axis=0) + np.std(avg_peak_longs, axis=0)/len(avg_peak_longs),
            color='lime', alpha=0.3
        )

        axs1[2,4].plot(range(transtion_0 + transtion_1), np.mean(avg_cr_shorts, axis=0))
        axs1[2,4].fill_between(
            range(transtion_0 + transtion_1),
            np.mean(avg_cr_shorts, axis=0) - np.std(avg_cr_shorts, axis=0)/len(avg_cr_shorts),
            np.mean(avg_cr_shorts, axis=0) + np.std(avg_cr_shorts, axis=0)/len(avg_cr_shorts),
            color='blue', alpha=0.3
        )

        axs1[2,5].plot(range(transtion_0 + transtion_1), np.mean(avg_cr_longs, axis=0))
        axs1[2,5].fill_between(
            range(transtion_0 + transtion_1),
            np.mean(avg_cr_longs, axis=0) - np.std(avg_cr_longs, axis=0)/len(avg_cr_longs),
            np.mean(avg_cr_longs, axis=0) + np.std(avg_cr_longs, axis=0)/len(avg_cr_longs),
            color='lime', alpha=0.3
        )

        axs1[0,0].set_ylabel('Average velocity')
        axs1[0,0].set_title('Average Velocity changes short')

        axs1[0,1].set_ylabel('Average velocity')
        axs1[0,1].set_title('Average Velocity changes long')

        axs1[1,2].set_ylabel('Peak Values')
        axs1[1,2].set_title('Average Peak Values short')

        axs1[1,3].set_ylabel('Peak Values')
        axs1[1,3].set_title('Average Peak Values long')

        axs1[2,4].set_ylabel('CR Onset time')
        axs1[2,4].set_title('Average CR onset time short')
        
        axs1[2,5].set_ylabel('CR Onset time')
        axs1[2,5].set_title('Average CR onset time long')

        for i in range(5):
            for j in range(number_of_stacks):
                axs[i, j].set_xlim(0 , transtion_1 + transtion_0)
                axs[i,j].set_xlabel('Trials')
                xticks = axs[i, j].get_xticks()
                axs[i, j].axvline(transtion_0, color = 'gray', linestyle = '--', alpha = 0.7, label = 'Transtion')
                # axs[i, j].set_xticks(xticks - transtion_0)

        for seg_i in range(number_of_stacks):
            axs[0, seg_i].set_ylim(np.min(y_min_0), np.max(y_max_0))
            axs[1, seg_i].set_ylim(np.min(y_min_1), np.max(y_max_1))
            axs[2, seg_i].set_ylim(np.min(y_min_2), np.max(y_max_2))
            axs[5, seg_i].set_title("All transitions to short")
            axs[5, seg_i].spines['right'].set_visible(False)
            axs[5, seg_i].spines['top'].set_visible(False)

        # Adjust layout and save as PDF
        plt.tight_layout()
        try:
            with PdfPages(adaptation_mouse_file) as pdf:
                pdf.savefig(fig)
            print(f"PDF successfully saved: {adaptation_mouse_file}")

            with PdfPages(adaptation_summary_summary) as pdf:
                pdf.savefig(fig1)
            print(f"PDF successfully saved: {adaptation_summary_summary}")

        except Exception as e:
            print(f"Error saving PDF: {e}")
