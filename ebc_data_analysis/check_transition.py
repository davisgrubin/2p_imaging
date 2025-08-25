import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages 
import matplotlib.cm as cm
import h5py

from utils.alignment import sort_numbers_as_strings, fec_zero, fec_crop
from utils.indication import CR_stat_indication

def get_colormap(colormap_name, min_val, max_val, value):
    norm_value = (value - min_val) / (max_val - min_val)
    norm_value = np.clip(norm_value, 0.4, 1)
    colormap = cm.get_cmap(colormap_name)
    return colormap(norm_value)

def isi_type(trial):
    airpuff = trial["AirPuff"][0] - trial["LED"][0]
    if airpuff > (Long_airpuff_on - 10) and airpuff < (Long_airpuff_on + 10):
        trial_type = 2
    elif airpuff > (Short_airpuff_on - 10) and airpuff < (Short_airpuff_on + 10):
        trial_type = 1
    else:
        print(f"FATAL ERROR. The isi duration is not as expected. It is {airpuff}")
        raise ValueError(f"Unexpected ISI duration: {airpuff}")

    return trial_type

plt.rcParams['lines.markersize'] = 2
plt.rcParams['lines.linewidth'] = 0.5

# Expected values
Short_airpuff_on = 200
Short_airpuff_off = 220
Long_airpuff_on = 400
Long_airpuff_off = 420

beh_folder = "./data/beh"

cr_threshold = 0.08
velocity_threshold_fraction = 0.95
amp_threshold_fraction = 0.10
acc_threshold = 1e-6
cr_window_time = 50
number_of_first_block_long = 0
number_of_bad_sessions = 0

number_of_gradient_degrees = 2

positive_singles = 1  # Number of trials right after transition
negative_singles = 1  # Number of trials right before transition

mice = [folder for folder in os.listdir(beh_folder) if folder != ".DS_Store"] 
mice.sort()

for mouse in mice:

    if mouse not in ['E6LG']:
        continue

    for cr_pos in ['all']:
        check_trans_long = f"./outputs/beh/{mouse}/check/check_trans_{mouse}_{cr_pos}_shayan.pdf"

        if not os.path.exists(f'./outputs/beh/{mouse}/check'):
            os.makedirs(f'./outputs/beh/{mouse}/check')
            print(f"check folder '{mouse}' created.")
        else:
            print(f"check folder '{mouse}' already exists.")

        # Setup figure
        fig_long, ax = plt.subplots(2, 2, figsize=(7*2, 2*7), sharex=False, sharey=False)

        for test_type_check in [1]:  # 1 for control and 0 for SD
            if mouse == 'E5LG' and test_type_check == 0:
                continue

            test_type_text = "control" if test_type_check == 1 else "SD"

            all_sessions = [os.path.splitext(file)[0] for file in os.listdir(os.path.join(beh_folder, mouse, "processed")) if file != ".DS_Store"]
            all_sessions.sort()

            # FIX 1: Take last 20 sessions properly
            # all_sessions = all_sessions[-20:] if len(all_sessions) >= 20 else all_sessions
            
            if not os.path.exists(f'./outputs/beh/{mouse}'):
                os.makedirs(f'./outputs/beh/{mouse}')
                print(f"data folder '{mouse}' created.")
            else:
                print(f"data folder '{mouse}' already exists.")

            long_slots = {}
            short_slots = {}
            
            for derivative_degree in range(number_of_gradient_degrees):
                long_slots[derivative_degree] = []
                short_slots[derivative_degree] = []
                # Initialize slots for different trial categories
                for i in range(negative_singles + positive_singles + 4):
                    long_slots[derivative_degree].append([])
                    short_slots[derivative_degree].append([])

            # FIX 2: Initialize time variable outside the loop
            time = None
            sessions_processed = 0

            for sess_i, session_date in enumerate(all_sessions):
                try:
                    print(f'{session_date} being processed')

                    # Load data
                    trials = h5py.File(f"./data/beh/{mouse}/processed/{session_date}.h5")["trial_id"]
                    fec, fec_time, trials = fec_zero(trials)
                    fec, fec_time = fec_crop(fec, fec_time)
                    cr_positives, CR_stat, CR_interval_avg, base_line_avg, cr_interval_idx, bl_interval_idx = CR_stat_indication(
                        trials, fec, fec_time, static_threshold=cr_threshold, AP_delay=0
                    )

                    all_id = []
                    for trial_id in trials:
                        all_id.append(trial_id)

                    if cr_pos == 'all':
                        cr_positives = all_id
                    elif cr_pos == '-':
                        cr_positives = [trial_id for trial_id in all_id if trial_id not in cr_positives]

                    all_id = sort_numbers_as_strings(all_id)
                    
                    # FIX 3: Set time from the first successfully processed session
                    if time is None and len(all_id) > 0:
                        time = fec_time[all_id[-1]]

                    # FIX 4: Check test type properly
                    test_type = trials[all_id[0]]["test_type"][()]
                    if test_type != test_type_check:
                        print(f"Skipping session {session_date}: test_type {test_type} != {test_type_check}")
                        continue

                    # Define blocks
                    blocks = []
                    current_block = {'type': None, 'trials': []}
                    
                    # Determine the starting block type
                    first_type = isi_type(trials[all_id[0]])
                    current_block['type'] = first_type
                    
                    # Group trials into blocks
                    for i, trial_id in enumerate(all_id):
                        trial_type = isi_type(trials[trial_id])
                        
                        # Starting a new block
                        if trial_type != current_block['type']:
                            # Save the completed block
                            if current_block['trials']:  # Only add non-empty blocks
                                blocks.append(current_block)
                            # Start a new block
                            current_block = {'type': trial_type, 'trials': [trial_id]}
                        else:
                            # Add to current block
                            current_block['trials'].append(trial_id)
                    
                    # Add the last block
                    if current_block['trials']:
                        blocks.append(current_block)

                    print(f'Found {len(blocks)} blocks in session {session_date}')
                    
                    # Process each transition (skip the last block as it has no transition)
                    for i in range(len(blocks) - 1):
                        current_block = blocks[i]
                        next_block = blocks[i+1]
                        
                        # Get the trial IDs from each block
                        current_trials = current_block['trials']
                        next_trials = next_block['trials']
                        
                        # Skip if either block doesn't have enough trials
                        if len(current_trials) <= negative_singles or len(next_trials) <= positive_singles:
                            print(f"Skipping transition: insufficient trials in blocks")
                            continue
                            
                        # Process long to short transition
                        if current_block['type'] == 2 and next_block['type'] == 1:
                            print(f"Processing long->short transition")
                            # First half of long block
                            first_half = current_trials[:len(current_trials)//2]
                            for trial_id in first_half:
                                if trial_id in cr_positives:
                                    long_slots[0][0].append(fec[trial_id])
                                    velocity = np.gradient(fec[trial_id], fec_time[trial_id])
                                    long_slots[1][0].append(velocity)
                            
                            # Second half (excluding transition trials)
                            second_half = current_trials[len(current_trials)//2:-negative_singles]
                            for trial_id in second_half:
                                if trial_id in cr_positives:
                                    long_slots[0][1].append(fec[trial_id])
                                    velocity = np.gradient(fec[trial_id], fec_time[trial_id])
                                    long_slots[1][1].append(velocity)
                            
                            # Last few trials before transition
                            for j in range(negative_singles):
                                if j < len(current_trials):
                                    trial_id = current_trials[-j-1]
                                    if trial_id in cr_positives:
                                        long_slots[0][j+2].append(fec[trial_id])
                                        velocity = np.gradient(fec[trial_id], fec_time[trial_id])
                                        long_slots[1][j+2].append(velocity)
                                        print(f'Last long: {trial_id}')
                            
                            # First few trials after transition
                            for j in range(positive_singles):
                                if j < len(next_trials):
                                    trial_id = next_trials[j]
                                    if trial_id in cr_positives:
                                        long_slots[0][negative_singles+j+2].append(fec[trial_id])
                                        velocity = np.gradient(fec[trial_id], fec_time[trial_id])
                                        long_slots[1][negative_singles+j+2].append(velocity)
                                        print(f'First short: {trial_id}')
                            
                            # First half of next block (excluding transition trials)
                            first_half_next = next_trials[positive_singles:len(next_trials)//2]
                            for trial_id in first_half_next:
                                if trial_id in cr_positives:
                                    long_slots[0][negative_singles+positive_singles+2].append(fec[trial_id])
                                    velocity = np.gradient(fec[trial_id], fec_time[trial_id])
                                    long_slots[1][negative_singles+positive_singles+2].append(velocity)
                            
                            # Second half of next block
                            second_half_next = next_trials[len(next_trials)//2:]
                            for trial_id in second_half_next:
                                if trial_id in cr_positives:
                                    long_slots[0][negative_singles+positive_singles+3].append(fec[trial_id])
                                    velocity = np.gradient(fec[trial_id], fec_time[trial_id])
                                    long_slots[1][negative_singles+positive_singles+3].append(velocity)
                        
                        # Process short to long transition
                        elif current_block['type'] == 1 and next_block['type'] == 2:
                            print(f"Processing short->long transition")
                            # First half of short block
                            first_half = current_trials[:len(current_trials)//2]
                            for trial_id in first_half:
                                if trial_id in cr_positives:
                                    short_slots[0][0].append(fec[trial_id])
                                    velocity = np.gradient(fec[trial_id], fec_time[trial_id])
                                    short_slots[1][0].append(velocity)
                            
                            # Second half (excluding transition trials)
                            second_half = current_trials[len(current_trials)//2:-negative_singles]
                            for trial_id in second_half:
                                if trial_id in cr_positives:
                                    short_slots[0][1].append(fec[trial_id])
                                    velocity = np.gradient(fec[trial_id], fec_time[trial_id])
                                    short_slots[1][1].append(velocity)
                            
                            # Last few trials before transition
                            for j in range(negative_singles):
                                if j < len(current_trials):
                                    trial_id = current_trials[-j-1]
                                    if trial_id in cr_positives:
                                        short_slots[0][j+2].append(fec[trial_id])
                                        velocity = np.gradient(fec[trial_id], fec_time[trial_id])
                                        short_slots[1][j+2].append(velocity)
                                        print(f'Last short: {trial_id}')
                            
                            # First few trials after transition
                            for j in range(positive_singles):
                                if j < len(next_trials):
                                    trial_id = next_trials[j]
                                    if trial_id in cr_positives:
                                        short_slots[0][negative_singles+j+2].append(fec[trial_id])
                                        velocity = np.gradient(fec[trial_id], fec_time[trial_id])
                                        short_slots[1][negative_singles+j+2].append(velocity)
                                        print(f'First long: {trial_id}')
                            
                            # First half of next block (excluding transition trials)
                            first_half_next = next_trials[positive_singles:len(next_trials)//2]
                            for trial_id in first_half_next:
                                if trial_id in cr_positives:
                                    short_slots[0][negative_singles+positive_singles+2].append(fec[trial_id])
                                    velocity = np.gradient(fec[trial_id], fec_time[trial_id])
                                    short_slots[1][negative_singles+positive_singles+2].append(velocity)
                            
                            # Second half of next block
                            second_half_next = next_trials[len(next_trials)//2:]
                            for trial_id in second_half_next:
                                if trial_id in cr_positives:
                                    short_slots[0][negative_singles+positive_singles+3].append(fec[trial_id])
                                    velocity = np.gradient(fec[trial_id], fec_time[trial_id])
                                    short_slots[1][negative_singles+positive_singles+3].append(velocity)

                    sessions_processed += 1
                    
                except Exception as e:
                    print(f'Error processing session {session_date}: {e}')
                    continue

            # FIX 5: Check if we have data to plot
            if time is None:
                print(f"No sessions were successfully processed for mouse {mouse}")
                continue
                
            if sessions_processed == 0:
                print(f"No valid sessions found for mouse {mouse}")
                continue

            # Color definitions
            color_pre_2 = 'black'
            color_pre_1 = 'blue'
            color_post_1 = 'orange'
            color_post_2 = 'red'

            # FIX 6: Check if we have data before plotting
            def safe_plot(ax_obj, time_data, data_list, label, color, alpha=1.0):
                if len(data_list) > 0:
                    mean_data = np.nanmean(data_list, axis=0)
                    if len(mean_data) == len(time_data):
                        ax_obj.plot(time_data, mean_data, label=f'{label} n:{len(data_list)}', color=color, alpha=alpha)
                        
                        # Add error bars if there's more than one trial
                        if len(data_list) > 1:
                            sem = np.nanstd(data_list, axis=0) / np.sqrt(len(data_list))
                            ax_obj.fill_between(time_data, mean_data - sem, mean_data + sem, 
                                              color=color, alpha=0.2, edgecolor='none')
                    else:
                        print(f"Warning: Data length mismatch for {label}")
                else:
                    print(f"Warning: No data for {label}")

            # Plot short to long transitions (top row)
            # safe_plot(ax[0, 0], time, short_slots[0][0], '1st half before transition', color_pre_1, 0.3)
            # safe_plot(ax[1, 0], time, short_slots[1][0], '1st half before transition', color_pre_1, 0.3)
            
            safe_plot(ax[0, 0], time, short_slots[0][1], '2nd half before transition', color_pre_2)
            safe_plot(ax[1, 0], time, short_slots[1][1], '2nd half before transition', color_pre_2)

            # for i in range(negative_singles):
            #     color_n = get_colormap('Blues', 0, negative_singles, i)
            #     safe_plot(ax[0, 0], time, short_slots[0][i + 2], f'{i+1} before transition', color_n)
            #     safe_plot(ax[1, 0], time, short_slots[1][i + 2], f'{i+1} before transition', color_n)
            #
            # for i in range(positive_singles):
            #     color_p = get_colormap('Reds', 0, positive_singles, i)
            #     safe_plot(ax[0, 0], time, short_slots[0][negative_singles + i + 2], f'{i+1} after transition', color_p)
            #     safe_plot(ax[1, 0], time, short_slots[1][negative_singles + i + 2], f'{i+1} after transition', color_p)

            safe_plot(ax[0, 0], time, short_slots[0][negative_singles + positive_singles + 2], '1st half after transition', color_post_1)
            safe_plot(ax[1, 0], time, short_slots[1][negative_singles + positive_singles + 2], '1st half after transition', color_post_1)

            safe_plot(ax[0, 0], time, short_slots[0][negative_singles + positive_singles + 3], '2nd half after transition', color_post_2)
            safe_plot(ax[1, 0], time, short_slots[1][negative_singles + positive_singles + 3], '2nd half after transition', color_post_2)

            # Plot long to short transitions (bottom row)
            # safe_plot(ax[0, 1], time, long_slots[0][0], '1st half before transition', color_pre_1, 0.3)
            # safe_plot(ax[1, 1], time, long_slots[1][0], '1st half before transition', color_pre_1, 0.3)
            
            safe_plot(ax[0, 1], time, long_slots[0][1], '2nd half before transition', color_pre_2)
            safe_plot(ax[1, 1], time, long_slots[1][1], '2nd half before transition', color_pre_2)

            # for i in range(negative_singles):
            #     color_n = get_colormap('Blues', 0, negative_singles, i)
            #     safe_plot(ax[0, 1], time, long_slots[0][i + 2], f'{i+1} before transition', color_n)
            #     safe_plot(ax[1, 1], time, long_slots[1][i + 2], f'{i+1} before transition', color_n)
            #
            # for i in range(positive_singles):
            #     color_p = get_colormap('Reds', 0, positive_singles, i)
            #     safe_plot(ax[0, 1], time, long_slots[0][negative_singles + i + 2], f'{i+1} after transition', color_p)
            #     safe_plot(ax[1, 1], time, long_slots[1][negative_singles + i + 2], f'{i+1} after transition', color_p)

            safe_plot(ax[0, 1], time, long_slots[0][negative_singles + positive_singles + 2], '1st half after transition', color_post_1)
            safe_plot(ax[1, 1], time, long_slots[1][negative_singles + positive_singles + 2], '1st half after transition', color_post_1)

            safe_plot(ax[0, 1], time, long_slots[0][negative_singles + positive_singles + 3], '2nd half after transition', color_post_2)
            safe_plot(ax[1, 1], time, long_slots[1][negative_singles + positive_singles + 3], '2nd half after transition', color_post_2)

            # Add stimulus periods and formatting
            for i in range(2):
                for j in range(2):
                    ax[j, i].axvspan(0, 50, color="gray", alpha=0.1, linewidth=0)
                    if i % 2 == 0:
                        ax[j, i].axvspan(Long_airpuff_on, Long_airpuff_off, color="lime", alpha=0.1, linewidth=0)
                        ax[j, i].axvline(Short_airpuff_on, color="blue", alpha=0.2)
                        ax[j, i].axvline(Short_airpuff_off, color="blue", alpha=0.2)
                    else:
                        ax[j, i].axvspan(Short_airpuff_on, Short_airpuff_off, color="blue", alpha=0.1, linewidth=0)
                        ax[j, i].axvline(Long_airpuff_on, color="blue", alpha=0.2)
                        ax[j, i].axvline(Long_airpuff_off, color="blue", alpha=0.2)

                    ax[j, i].spines['top'].set_visible(False)
                    ax[j, i].spines['right'].set_visible(False)
                    ax[j, i].legend(loc='upper left', fontsize=5)
                    # ax[j, i].set_ylim(0, 1)
                    ax[j, i].set_xlabel('Time (ms)')
                    ax[j, i].set_ylabel('FEC (+/- SEM)')

            ax[0, 0].set_title('Short to Long transition')
            ax[1, 0].set_title('Short to Long transition')
            ax[0, 1].set_title('Long to Short transition')
            ax[1, 1].set_title('Long to Short transition')

            plt.tight_layout()

        # Save the figure
        with PdfPages(check_trans_long) as pdf:
            pdf.savefig(fig_long, dpi=400)
            pdf.close()
            
        print(f"Saved plot to {check_trans_long}")
