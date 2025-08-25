import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages 
from matplotlib.gridspec import GridSpec
import h5py

from utils.alignment import sort_numbers_as_strings, fec_zero
from utils.indication import find_max_with_gradient, find_index, cr_onset_calc #, block_type
# from utils.functions import sig_trial_func, isi_type

def sig_trial_func(all_id, trials, transition_0, transition_1):

    sig_trial_ids = []
    slot_ids = []

    if isi_type(trials[all_id[0]]) == 2:
        print('first block is long')
        sig_trial_ids.append([])
        slot_ids.append([])

    for trial_id in all_id:
        try:
            next_id = int(trial_id) + 1

            while str(next_id) not in trials:
                next_id += 1

                if next_id - int(trial_id) >= 20:
                    break

            if trials[trial_id]["trial_type"][()] != trials[str(next_id)]["trial_type"][()] :
                sig_trial_ids.append(trial_id)
                slot_ids.append([str(i) for i in range(int(trial_id) - transition_0 , int(trial_id) + transition_1)])

        except Exception as e:
            print(f"Exception: {e}")
            continue

    return sig_trial_ids, slot_ids


def isi_type(trial):
    airpuff = trial["AirPuff"][0] - trial["LED"][0]
    if airpuff > (Long_airpuff_on - 10) and airpuff < (Long_airpuff_on + 10):
        trial_type = 2
    elif airpuff > (Short_airpuff_on - 10) and airpuff < (Short_airpuff_on + 10):
        trial_type = 1
    else:
        print(f"FATAL ERROR. The isi duration is not as expected. It is {airpuff}")
        ValueError()
        trial_type = None

    return trial_type

def block_type(trials):
    shorts = []
    longs = []
    cntr_300 = 0
    for id in trials:
            
        try:
            if isi_type(trials[id]) == 1:
                shorts.append(id)
            if isi_type(trials[id]) == 2:
                longs.append(id)
            if isi_type(trials[id]) == 3:
                cntr_300 += 1
            elif isi_type(trials[id]) == None:

                if trials[id]["trial_type"][()] == 2:
                    print('long')
                    print(trials[id]["Airpuff"])
                    print(trials[id]["LED"])

                if trials[id]["trial_type"][()] == 1:
                    print('short')
                    print(trials[id]["Airpuff"])
                    print(trials[id]["LED"])

                continue

        except:
            print(f'trial {id} has trial type file problem')
            continue
            # breakpoint()

    return shorts, longs


beh_folder = "./data/beh"

# expected_values
Short_airpuff_on = 200
Short_airpuff_off = 220
Long_airpuff_on = 400
Long_airpuff_off = 420

cr_threshold = 0.08
velocity_threshold_fraction = 0.95
amp_threshold_fraction = 0.10
acc_threshold = 1e-6
cr_window_time = 50
number_of_first_block_long = 0
number_of_bad_sessions = 0

# for the adaptation sanity check. This value shows the number of single trials after the transition that will be used.
number_of_singles = 2

number_of_stacks= 6 #should be the same as the max number os segments
transition_0 = 50
transition_1 = 50
increment = 10

x_grid = number_of_stacks
y_grid = 11

mice = [folder for folder in os.listdir(beh_folder) if folder != ".DS_Store"] 
mice.sort()
for mouse in mice:
    cr_happy = 0
    cr_sad = 0

    # if mouse != 'E5LG':
    #     continue
    if mouse in ['E1VT', 'E2WT', 'E3VT']:
        print('why?')
        continue
        # Long_airpuff_off = 320
        # Long_airpuff_on = 300
    # else:
    #     continue

    # # if mouse == 'E4L7':
    # if mouse != 'E5LG':
    #     continue
    #
    transition_file_0 = f"./outputs/beh/{mouse}/transition_sum_{mouse}.pdf"
    transtion_bar_file = f"./outputs/beh/{mouse}/transition_bar_sum_{mouse}.pdf"

    x_grid_1 = 2
    y_grid_1 = 8
    fig_1 = plt.figure(figsize=(x_grid_1 * 7, y_grid_1 * 7))
    gs_1 = GridSpec(y_grid_1, x_grid_1)
    fig_short, ax_short = plt.subplots(9, 4, figsize=(7*4, 9*7), sharex=False, sharey=False)
    fig_long, ax_long = plt.subplots(9, 4, figsize=(7*4, 9*7), sharex=False, sharey=False)
    fig_0, ax_0 = plt.subplots(11, 2, figsize=(7*2, 11*7), sharex=False, sharey=False)
    fig_1, ax_1 = plt.subplots(10, 2, figsize=(7*2, 8*7), sharex=False, sharey=False)

    y_min_bl_1 = []
    y_max_bl_1 = []

    y_min_cr_1 = []
    y_max_cr_1 = []

    y_min_pk_1 = []
    y_max_pk_1 = []

    y_min_vl_1 = []
    y_max_vl_1 = []

    y_min_n_1 = []
    y_max_n_1 = []

    y_min_cr_val_1 = []
    y_max_cr_val_1 = []

    y_min_cr_200_1 = []
    y_max_cr_200_1 = []

    y_min_max_v_1 = []
    y_max_max_v_1 = []

    y_min_max_t_1 = []
    y_max_max_t_1 = []

    y_min_acc_1 = []
    y_max_acc_1 = []

    y_min_bl_bar = []
    y_max_bl_bar = []

    y_min_cr_bar = []
    y_max_cr_bar = []

    y_min_pk_bar = []
    y_max_pk_bar = []

    y_min_vl_bar = []
    y_max_vl_bar = []

    y_min_n_bar = []
    y_max_n_bar = []

    y_min_cr_val_bar = []
    y_max_cr_val_bar = []

    y_min_cr_200_bar = []
    y_max_cr_200_bar = []

    y_min_max_v_bar = []
    y_max_max_v_bar = []

    y_min_max_t_bar = []
    y_max_max_t_bar = []

    y_min_acc_bar = []
    y_max_acc_bar = []

    for test_type_check in [1]: #1 for control and 0 for SD
        # if mouse == 'E5LG':
        #     test_type_check = 1

        fig = plt.figure(figsize=(x_grid * 7, y_grid * 7))
        gs = GridSpec(y_grid, x_grid)

        # test_type_check = 1 #1 for control and 0 for SD
        test_type_text = None
        if test_type_check == 1:
            test_type_text = 'control'
        if test_type_check == 0:
            test_type_text = 'SD'

        all_sessions = [os.path.splitext(file)[0] for file in os.listdir(os.path.join(beh_folder, mouse, "processed")) if file != ".DS_Store"]
        all_sessions.sort()
        
        if not os.path.exists(f'./outputs/beh/{mouse}'):
                os.mkdir(f'./outputs/beh/{mouse}')
                print(f"data folder '{mouse}' created.")
        else:
            print(f"data folder '{mouse}' already exists.")

        transition_file = f"./outputs/beh/{mouse}/transition_{mouse}_{test_type_text}.pdf"

        session_folder = [folder for folder in os.listdir(f'./data/beh/{mouse}/')]

        bl_slots = {}
        bl_avg = {}
        bl_sem = {}
        
        bl_slots_1 = {}
        bl_avg_1 = {}
        bl_sem_1 = {}

        bl_slots_pre = {}
        bl_slots_post = {}

        bl_slots_short = {}
        bl_avg_short = {}
        bl_sem_short = {}

        bl_slots_long = {}
        bl_avg_long = {}
        bl_sem_long = {}

        cr_slots = {}
        cr_avg = {}
        cr_sem = {}

        cr_slots_1 = {}
        cr_avg_1 = {}
        cr_sem_1 = {}

        cr_slots_pre = {}
        cr_slots_post = {}

        cr_slots_short = {}
        cr_avg_short = {}
        cr_sem_short = {}

        cr_slots_long = {}
        cr_avg_long = {}
        cr_sem_long = {}

        pk_slots = {}
        pk_avg = {}
        pk_sem = {}

        pk_slots_1 = {}
        pk_avg_1 = {}
        pk_sem_1 = {}

        pk_slots_pre = {}
        pk_slots_post = {}

        pk_slots_short = {}
        pk_avg_short = {}
        pk_sem_short = {}

        pk_slots_long = {}
        pk_avg_long = {}
        pk_sem_long = {}

        vl_slots = {}
        vl_avg = {}
        vl_sem = {}

        vl_slots_1 = {}
        vl_avg_1 = {}
        vl_sem_1 = {}

        vl_slots_pre = {}
        vl_slots_post = {}

        vl_slots_short = {}
        vl_avg_short = {}
        vl_sem_short = {}

        vl_slots_long = {}
        vl_avg_long = {}
        vl_sem_long = {}

        ramp_slots = {}
        ramp_avg = {}
        ramp_sem = {}

        ramp_slots_1 = {}
        ramp_avg_1 = {}
        ramp_sem_1 = {}

        ramp_slots_pre = {}
        ramp_slots_post = {}

        ramp_slots_short = {}
        ramp_avg_short = {}
        ramp_sem_short = {}

        ramp_slots_long = {}
        ramp_avg_long = {}
        ramp_sem_long = {}

        isi_slots = {}
        isi_avg = {}
        isi_std = {}

        isi_slots_1 = {}
        isi_avg_1 = {}
        isi_std_1 = {}

        cr_val_slots = {}
        cr_val_avg = {}
        cr_val_sem = {}

        cr_val_slots_1 = {}
        cr_val_avg_1 = {}
        cr_val_sem_1 = {}

        cr_val_slots_pre = {}
        cr_val_slots_post = {}

        cr_val_slots_short = {}
        cr_val_avg_short = {}
        cr_val_sem_short = {}

        cr_val_slots_long = {}
        cr_val_avg_long = {}
        cr_val_sem_long = {}

        cr_200_slots = {}
        cr_200_avg = {}
        cr_200_sem = {}

        cr_200_slots_1 = {}
        cr_200_avg_1 = {}
        cr_200_sem_1 = {}

        cr_200_slots_pre = {}
        cr_200_slots_post = {}

        cr_200_slots_short = {}
        cr_200_avg_short = {}
        cr_200_sem_short = {}

        cr_200_slots_long = {}
        cr_200_avg_long = {}
        cr_200_sem_long = {}

        max_v_slots = {}
        max_v_avg = {}
        max_v_sem = {}

        max_v_slots_1 = {}
        max_v_avg_1 = {}
        max_v_sem_1 = {}

        max_v_slots_pre = {}
        max_v_slots_post = {}

        max_v_slots_short = {}
        max_v_avg_short = {}
        max_v_sem_short = {}

        max_v_slots_long = {}
        max_v_avg_long = {}
        max_v_sem_long = {}

        max_t_slots = {}
        max_t_avg = {}
        max_t_sem = {}

        max_t_slots_1 = {}
        max_t_avg_1 = {}
        max_t_sem_1 = {}

        max_t_slots_pre = {}
        max_t_slots_post = {}

        max_t_slots_short = {}
        max_t_avg_short = {}
        max_t_sem_short = {}

        max_t_slots_long = {}
        max_t_avg_long = {}
        max_t_sem_long = {}

        acc_slots = {}
        acc_avg = {}
        acc_sem = {}

        acc_slots_1 = {}
        acc_avg_1 = {}
        acc_sem_1 = {}

        acc_slots_pre = {}
        acc_slots_post = {}

        acc_slots_short = {}
        acc_avg_short = {}
        acc_sem_short = {}

        acc_slots_long = {}
        acc_avg_long = {}
        acc_sem_long = {}

        n_slots = {}
        n_slots_1 = {}

        bl_pre_1 = {}
        bl_post_1 = {}
        bl_pre_sem_1 = {}
        bl_post_sem_1 = {}

        cr_pre_1 = {}
        cr_post_1 = {}
        cr_pre_sem_1 = {}
        cr_post_sem_1 = {}

        pk_pre_1 = {}
        pk_post_1 = {}
        pk_pre_sem_1 = {}
        pk_post_sem_1 = {}

        vl_pre_1 = {}
        vl_post_1 = {}
        vl_pre_sem_1 = {}
        vl_post_sem_1 = {}

        cr_val_pre_1 = {}
        cr_val_post_1 = {}
        cr_val_pre_sem_1 = {}
        cr_val_post_sem_1 = {}

        cr_200_pre_1 = {}
        cr_200_post_1 = {}
        cr_200_pre_sem_1 = {}
        cr_200_post_sem_1 = {}

        max_v_pre_1 = {}
        max_v_post_1 = {}
        max_v_pre_sem_1 = {}
        max_v_post_sem_1 = {}

        max_t_pre_1 = {}
        max_t_post_1 = {}
        max_t_pre_sem_1 = {}
        max_t_post_sem_1 = {}


        acc_pre_1 = {}
        acc_post_1 = {}
        acc_pre_sem_1 = {}
        acc_post_sem_1 = {}

        y_min_bl = []
        y_max_bl = []

        y_min_cr = []
        y_max_cr = []

        y_min_pk = []
        y_max_pk = []

        y_min_vl = []
        y_max_vl = []

        y_min_isi = []
        y_max_isi = []

        y_min_cr_val = []
        y_max_cr_val = []

        y_min_cr_200 = []
        y_max_cr_200 = []

        y_min_max_v = []
        y_max_max_v = []

        y_min_max_t = []
        y_max_max_t = []

        y_min_acc = []
        y_max_acc = []
        
        n_slots = {}
        n_slots_1 = {}

        y_min_n = []
        y_max_n = []



        
        for i in range(2):

            bl_slots_1[i] = []
            bl_avg_1[i] = []
            bl_sem_1[i] = []

            cr_slots_1[i] = []
            cr_avg_1[i] = []
            cr_sem_1[i] = []

            pk_slots_1[i] = []
            pk_avg_1[i] = []
            pk_sem_1[i] = []

            vl_slots_1[i] = []
            vl_avg_1[i] = []
            vl_sem_1[i] = []

            ramp_slots_1[i] = []
            ramp_avg_1[i] = []
            ramp_sem_1[i] = []

            cr_val_slots_1[i] = []
            cr_val_avg_1[i] = []
            cr_val_sem_1[i] = []

            cr_200_slots_1[i] = []
            cr_200_avg_1[i] = []
            cr_200_sem_1[i] = []

            max_v_slots_1[i] = []
            max_v_avg_1[i] = []
            max_v_sem_1[i] = []

            max_t_slots_1[i] = []
            max_t_avg_1[i] = []
            max_t_sem_1[i] = []

            acc_slots_1[i] = []
            acc_avg_1[i] = []
            acc_sem_1[i] = []

            n_slots_1[i] = []

            bl_slots_pre[i] = []
            bl_slots_post[i] = []

            cr_slots_pre[i] = []
            cr_slots_post[i] = []

            pk_slots_pre[i] = []
            pk_slots_post[i] = []

            vl_slots_pre[i] = []
            vl_slots_post[i] = []

            ramp_slots_pre[i] = []
            ramp_slots_post[i] = []

            cr_val_slots_pre[i] = []
            cr_val_slots_post[i] = []

            cr_200_slots_pre[i] = []
            cr_200_slots_post[i] = []

            max_v_slots_pre[i] = []
            max_v_slots_post[i] = []

            max_t_slots_pre[i] = []
            max_t_slots_post[i] = []

            acc_slots_pre[i] = []
            acc_slots_post[i] = []

            for id in range(transition_0 + transition_1):

                bl_slots_1[i].append([])
                bl_avg_1[i].append([])
                bl_sem_1[i].append([])

                cr_slots_1[i].append([])
                cr_avg_1[i].append([])
                cr_sem_1[i].append([])

                pk_slots_1[i].append([])
                pk_avg_1[i].append([])
                pk_sem_1[i].append([])

                vl_slots_1[i].append([])
                vl_avg_1[i].append([])
                vl_sem_1[i].append([])

                ramp_slots_1[i].append([])
                ramp_avg_1[i].append([])
                ramp_sem_1[i].append([])

                cr_val_slots_1[i].append([])
                cr_val_avg_1[i].append([])
                cr_val_sem_1[i].append([])

                cr_200_slots_1[i].append([])
                cr_200_avg_1[i].append([])
                cr_200_sem_1[i].append([])

                max_v_slots_1[i].append([])
                max_v_avg_1[i].append([])
                max_v_sem_1[i].append([])
                
                max_t_slots_1[i].append([])
                max_t_avg_1[i].append([])
                max_t_sem_1[i].append([])

                acc_slots_1[i].append([])
                acc_avg_1[i].append([])
                acc_sem_1[i].append([])

                n_slots_1[i].append(0)

        for slot_i in range(number_of_stacks):
            bl_slots[slot_i] = []
            bl_avg[slot_i] = []
            bl_sem[slot_i] = []

            cr_slots[slot_i] = []
            cr_avg[slot_i] = []
            cr_sem[slot_i] = []

            pk_slots[slot_i] = []
            pk_avg[slot_i] = []
            pk_sem[slot_i] = []

            vl_slots[slot_i] = []
            vl_avg[slot_i] = []
            vl_sem[slot_i] = []

            ramp_slots[slot_i] = []
            ramp_avg[slot_i] = []
            ramp_sem[slot_i] = []

            isi_slots[slot_i] = []
            isi_avg[slot_i] = []
            isi_std[slot_i] = []

            cr_val_slots[slot_i] = []
            cr_val_avg[slot_i] = []
            cr_val_sem[slot_i] = []

            cr_200_slots[slot_i] = []
            cr_200_avg[slot_i] = []
            cr_200_sem[slot_i] = []

            max_v_slots[slot_i] = []
            max_v_avg[slot_i] = []
            max_v_sem[slot_i] = []

            max_t_slots[slot_i] = []
            max_t_avg[slot_i] = []
            max_t_sem[slot_i] = []

            acc_slots[slot_i] = []
            acc_avg[slot_i] = []
            acc_sem[slot_i] = []

            n_slots[slot_i] = []

            for id in range(transition_0 + transition_1):
                bl_slots[slot_i].append([])
                bl_avg[slot_i].append([])
                bl_sem[slot_i].append([])
                
                cr_slots[slot_i].append([])
                cr_avg[slot_i].append([])
                cr_sem[slot_i].append([])

                pk_slots[slot_i].append([])
                pk_avg[slot_i].append([])
                pk_sem[slot_i].append([])

                vl_slots[slot_i].append([])
                vl_avg[slot_i].append([])
                vl_sem[slot_i].append([])

                ramp_slots[slot_i].append([])
                ramp_avg[slot_i].append([])
                ramp_sem[slot_i].append([])

                isi_slots[slot_i].append([])
                isi_avg[slot_i].append([])
                isi_std[slot_i].append([])

                cr_val_slots[slot_i].append([])
                cr_val_avg[slot_i].append([])
                cr_val_sem[slot_i].append([])

                cr_200_slots[slot_i].append([])
                cr_200_avg[slot_i].append([])
                cr_200_sem[slot_i].append([])

                max_v_slots[slot_i].append([])
                max_v_avg[slot_i].append([])
                max_v_sem[slot_i].append([])

                max_t_slots[slot_i].append([])
                max_t_avg[slot_i].append([])
                max_t_sem[slot_i].append([])

                acc_slots[slot_i].append([])
                acc_avg[slot_i].append([])
                acc_sem[slot_i].append([])

                n_slots[slot_i].append(0) 

        for i , session_date in enumerate(all_sessions):

            # if i > 10:
            #     break

            print(session_date, 'being localized')

            trials = h5py.File(f"./data/beh/{mouse}/processed/{session_date}.h5")["trial_id"]
            fec, fec_time, trials = fec_zero(trials)
            shorts, longs = block_type(trials)
            all_id = sort_numbers_as_strings(shorts + longs)
            if len(all_id) == 0:
               continue

            sig_trial_ids, slot_ids = sig_trial_func(all_id, trials, transition_0, transition_1)

            print('sig',sig_trial_ids)

            print('slots', slot_ids)
            for slot_i in slot_ids:
                print('#############################')
                for slot in slot_i:
                    try:
                        print(slot)
                        print(f'{slot} : {isi_type(trials[slot])}', end=',')
                    except:
                        continue

            if isi_type(trials[all_id[0]]) == 2:
                print('first block is long')
                # print(sig_trial_ids)
                # print(slot_ids)
                number_of_first_block_long += 1
                # continue

            test_type = trials[all_id[0]]["test_type"][()]
            if test_type == 2:
                test_type = 0

            if test_type == test_type_check:
                continue

            for slot_i in range(min(number_of_stacks, len(slot_ids))):
                for i, trial_id in enumerate(slot_ids[slot_i]):

                    if trial_id not in all_id:
                        print('not enough transitions')
                        continue

                    trial = trials[trial_id]
                    airpuff = trial['AirPuff'][0]- trial["LED"][0]

                    if i < transition_0 + 1 and airpuff > 210 and slot_i % 2 == 0:
                        number_of_bad_sessions += 1
                        # breakpoint()
                        continue

                    elif i < transition_0 + 1 and airpuff < 380 and slot_i % 2 == 1:
                        number_of_bad_sessions += 1
                        # breakpoint()
                        continue

                    elif i > transition_0 and airpuff < 380 and slot_i % 2 == 0:
                        number_of_bad_sessions += 1
                        # breakpoint()
                        continue

                    elif i > transition_0 and airpuff > 210 and slot_i % 2 == 1:
                        number_of_bad_sessions += 1
                        # breakpoint()
                        continue

                    # CR stat indication >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                    # fec_index_0 = np.abs(fec_time[trial_id]).argmin()
                    fec_index_led = find_index(fec_time[trial_id], 0.0)
                    fec_index_ap = find_index(fec_time[trial_id] , airpuff)
                    fec_index_cr = find_index(fec_time[trial_id], airpuff - cr_window_time)
                    fec_index_bl = find_index(fec_time[trial_id] , -200)

                    cr_window_avg = np.average(fec[trial_id][fec_index_cr : fec_index_ap])

                    window_size = 20
                    kernel = np.ones(window_size) / window_size
                    smoothed_fec = np.convolve(fec[trial_id], kernel, mode='same')

                    isi_interval = smoothed_fec[fec_index_led : fec_index_cr]
                    isi_interval_avg = np.average(isi_interval)

                    base_line = np.sort(fec[trial_id][fec_index_bl: fec_index_led])
                    base_line_indexes = int(0.3 * len(base_line))
                    base_line_avg = np.average(base_line[:base_line_indexes])

                    if cr_window_avg - base_line_avg > cr_threshold:
                        cr_stat = 1 # CR positive
                    else:
                        if any(value > base_line_avg + cr_threshold for value in isi_interval):
                            cr_stat = 2 # Poor CR
                        else:
                            cr_stat = 0 # No CR
                            
                    # CR stat indication <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

                    bl_fec = fec[trial_id][fec_index_bl: fec_index_led]
                    bl_amp = np.average(bl_fec)

                    cr_slope = ((fec[trial_id][fec_index_cr] - fec[trial_id][fec_index_ap]) / 
                        (fec_time[trial_id][fec_index_cr] - fec_time[trial_id][fec_index_ap]))

                    gradients = np.gradient(
                        fec[trial_id][fec_index_led: fec_index_ap],
                        fec_time[trial_id][fec_index_led: fec_index_ap]
                    )
                    peak_time, peak_value, _, gradients = find_max_with_gradient(
                        fec_time[trial_id][fec_index_led: fec_index_ap], 
                        fec[trial_id][fec_index_led: fec_index_ap],
                        gradients
                    )

                    # ramp metric


                    if peak_time is None:
                        peak_time = airpuff

                    # fec[trial_id] = fec[trial_id] - bl_amp
                    cr_idx = cr_onset_calc(fec[trial_id], fec_time[trial_id], 20, airpuff, cr_stat)

                    cr_value = np.mean(fec[trial_id][fec_index_cr : fec_index_ap])

                    idx_210 = find_index(fec_time[trial_id], 210)
                    idx_170 = find_index(fec_time[trial_id], 170)
                    cr_200 = np.mean(fec[trial_id][idx_170 : idx_210])


                    window_size = 40
                    kernel = np.ones(window_size) / window_size
                    smoothed_grad = np.convolve(gradients, kernel, mode='same')
                    half_kernel = np.ones(window_size // 2) / (window_size // 2)
                    very_smoothed_grad = np.convolve(smoothed_grad, kernel , mode = 'same')

                    acc = np.gradient(very_smoothed_grad, fec_time[trial_id][fec_index_led:fec_index_ap])
                    sign_changes = np.diff(np.sign(acc))
                    num_changes = np.count_nonzero(sign_changes)

                    acc = num_changes

                    if cr_stat == 1:
                        max_v = np.max(gradients)
                    else:
                        max_v = None

                    # max_t = fec_time[trial_id][np.argmax(gradients)] - trial["LED"][0]
                    # max_t = fec_time[trial_id][fec_index_led + np.argmax(smoothed_grad)]
                    max_t = fec_time[trial_id][fec_index_led:fec_index_ap][np.argmax(smoothed_grad)]

                    if cr_idx:
                        cr_happy += 1
                        cr_time = fec_time[trial_id][cr_idx] 
                        # changing the times from having their onset from the LED onset from the LED Offset.
                        cr_slots[slot_i][i].append(cr_time)
                        if peak_value:
                            ramp = (peak_value - fec[trial_id][cr_idx]) / (peak_time - cr_time)
                            ramp_slots[slot_i][i].append(ramp)
                        elif airpuff < 250: #trying to include the short trials that without a goog peak value
                            print('this shoould be short')
                            print(trial['trial_type'][()])
                            breakpoint()
                            ramp = (fec[trial_id][fec_index_ap] - fec[trial_id][cr_idx]) / (fec_time[trial_id][fec_index_ap] - cr_time)
                            ramp_slots[slot_i][i].append(ramp)
                        else:
                            print("what")
                            breakpoint()

                        if slot_i % 2 == 0:
                            ramp_slots_1[0][i].append(ramp)
                            cr_slots_1[0][i].append(cr_time)
                        else:
                            ramp_slots_1[1][i].append(ramp)
                            cr_slots_1[1][i].append(cr_time)
                    else:
                        cr_sad += 1
                        cr_time = None
                        # changing the times from having their onset from the LED onset from the LED Offset.

                    i_2 = slot_i % 2

                    bl_slots[slot_i][i].append(bl_amp)
                    pk_slots[slot_i][i].append(peak_time)
                    vl_slots[slot_i][i].append(cr_slope)
                    isi_slots[slot_i][i].append(airpuff)
                    cr_val_slots[slot_i][i].append(cr_value)
                    cr_200_slots[slot_i][i].append(cr_200)
                    if max_v:
                        max_v_slots[slot_i][i].append(max_v)
                    max_t_slots[slot_i][i].append(max_t)
                    acc_slots[slot_i][i].append(acc)
                    n_slots[slot_i][i] += 1

                    bl_slots_1[i_2][i].append(bl_amp)
                    pk_slots_1[i_2][i].append(peak_time)
                    vl_slots_1[i_2][i].append(cr_slope)
                    cr_val_slots_1[i_2][i].append(cr_value)
                    cr_200_slots_1[i_2][i].append(cr_200)
                    if max_v:
                        max_v_slots_1[i_2][i].append(max_v)
                    max_t_slots_1[i_2][i].append(max_t)
                    acc_slots_1[i_2][i].append(acc)
                    n_slots_1[i_2][i] += 1


                    if i < transition_0:
                        if cr_time:
                            cr_slots_pre[i_2].append(cr_time)
                        bl_slots_pre[i_2].append(bl_amp)
                        pk_slots_pre[i_2].append(peak_time)
                        vl_slots_pre[i_2].append(cr_slope)
                        cr_val_slots_pre[i_2].append(cr_value)
                        cr_200_slots_pre[i_2].append(cr_200)
                        if max_v:
                            max_v_slots_pre[i_2].append(max_v)
                        max_t_slots_pre[i_2].append(max_t)
                        acc_slots_pre[i_2].append(acc)

                    elif i > transition_0:
                        if cr_time:
                            cr_slots_post[i_2].append(cr_time)
                        bl_slots_post[i_2].append(bl_amp)
                        pk_slots_post[i_2].append(peak_time)
                        vl_slots_post[i_2].append(cr_slope)
                        cr_val_slots_post[i_2].append(cr_value)
                        cr_200_slots_post[i_2].append(cr_200)
                        if max_v:
                            max_v_slots_post[i_2].append(max_v)
                        max_t_slots_post[i_2].append(max_t)
                        acc_slots_post[i_2].append(acc)

        for stack_i in range(number_of_stacks):
            for trial_id in range(len(cr_slots[stack_i])):

                bl_avg[stack_i][trial_id] = np.average(bl_slots[stack_i][trial_id])
                bl_sem[stack_i][trial_id] = np.std(bl_slots[stack_i][trial_id]) / np.sqrt(n_slots[stack_i][trial_id])

                cr_avg[stack_i][trial_id] = np.average(cr_slots[stack_i][trial_id])
                cr_sem[stack_i][trial_id] = np.std(cr_slots[stack_i][trial_id]) / np.sqrt(n_slots[stack_i][trial_id])



                pk_avg[stack_i][trial_id] = np.average(pk_slots[stack_i][trial_id])
                pk_sem[stack_i][trial_id] = np.std(pk_slots[stack_i][trial_id]) / np.sqrt(n_slots[stack_i][trial_id])


                vl_avg[stack_i][trial_id] = np.average(vl_slots[stack_i][trial_id])
                vl_sem[stack_i][trial_id] = np.std(vl_slots[stack_i][trial_id]) / np.sqrt(n_slots[stack_i][trial_id])

                ramp_avg[stack_i][trial_id] = np.average(ramp_slots[stack_i][trial_id])
                ramp_sem[stack_i][trial_id] = np.std(ramp_slots[stack_i][trial_id]) / np.sqrt(n_slots[stack_i][trial_id])


                isi_avg[stack_i][trial_id] = np.average(isi_slots[stack_i][trial_id])
                isi_std[stack_i][trial_id] = np.std(isi_slots[stack_i][trial_id])


                cr_val_avg[stack_i][trial_id] = np.average(cr_val_slots[stack_i][trial_id])
                cr_val_sem[stack_i][trial_id] = np.std(cr_val_slots[stack_i][trial_id]) / np.sqrt(n_slots[stack_i][trial_id])

                cr_200_avg[stack_i][trial_id] = np.average(cr_200_slots[stack_i][trial_id])
                cr_200_sem[stack_i][trial_id] = np.std(cr_200_slots[stack_i][trial_id]) / np.sqrt(n_slots[stack_i][trial_id])

                max_v_avg[stack_i][trial_id] = np.average(max_v_slots[stack_i][trial_id])
                max_v_sem[stack_i][trial_id] = np.std(max_v_slots[stack_i][trial_id]) / np.sqrt(n_slots[stack_i][trial_id])

                max_t_avg[stack_i][trial_id] = np.average(max_t_slots[stack_i][trial_id])
                max_t_sem[stack_i][trial_id] = np.std(max_t_slots[stack_i][trial_id]) / np.sqrt(n_slots[stack_i][trial_id])


                acc_avg[stack_i][trial_id] = np.average(acc_slots[stack_i][trial_id])
                acc_sem[stack_i][trial_id] = np.std(acc_slots[stack_i][trial_id]) / np.sqrt(n_slots[stack_i][trial_id])

            y_min_bl.append(min(np.array(bl_avg[stack_i]) - np.array(bl_sem[stack_i])))
            y_max_bl.append(max(np.array(bl_avg[stack_i]) + np.array(bl_sem[stack_i])))
            y_min_cr.append(min(np.array(cr_avg[stack_i]) - np.array(cr_sem[stack_i])))
            y_max_cr.append(max(np.array(cr_avg[stack_i]) + np.array(cr_sem[stack_i])))
            y_min_pk.append(min(np.array(pk_avg[stack_i]) - np.array(pk_sem[stack_i])))
            y_max_pk.append(max(np.array(pk_avg[stack_i]) + np.array(pk_sem[stack_i])))
            y_min_vl.append(min(np.array(vl_avg[stack_i]) - np.array(vl_sem[stack_i])))
            y_max_vl.append(max(np.array(vl_avg[stack_i]) + np.array(vl_sem[stack_i])))
            y_min_isi.append(min(np.array(isi_avg[stack_i]) - np.array(isi_std[stack_i])))
            y_max_isi.append(max(np.array(isi_avg[stack_i]) + np.array(isi_std[stack_i])))
            y_min_cr_val.append(min(np.array(cr_val_avg[stack_i]) - np.array(cr_val_sem[stack_i])))
            y_max_cr_val.append(max(np.array(cr_val_avg[stack_i]) + np.array(cr_val_sem[stack_i])))
            y_min_cr_200.append(min(np.array(cr_200_avg[stack_i]) - np.array(cr_200_sem[stack_i])))
            y_max_cr_200.append(max(np.array(cr_200_avg[stack_i]) + np.array(cr_200_sem[stack_i])))
            y_min_max_v.append(min(np.array(max_v_avg[stack_i]) - np.array(max_v_sem[stack_i])))
            y_max_max_v.append(max(np.array(max_v_avg[stack_i]) + np.array(max_v_sem[stack_i])))
            y_min_max_t.append(min(np.array(max_t_avg[stack_i]) - np.array(max_t_sem[stack_i])))
            y_max_max_t.append(max(np.array(max_t_avg[stack_i]) + np.array(max_t_sem[stack_i])))
            y_min_acc.append(min(np.array(acc_avg[stack_i]) - np.array(acc_sem[stack_i])))
            y_max_acc.append(max(np.array(acc_avg[stack_i]) + np.array(acc_sem[stack_i])))
            y_min_n.append(min(n_slots[stack_i]))
            y_max_n.append(max(n_slots[stack_i]))

        for stack_i in range(number_of_stacks):
            # making the ax for the plots
            ax_cr = fig.add_subplot(gs[ 0:1 , stack_i])
            ax_pk = fig.add_subplot(gs[ 1:2 , stack_i])
            ax_max_t = fig.add_subplot(gs[ 2:3 , stack_i])
            ax_bl = fig.add_subplot(gs[ 3:4 , stack_i])
            ax_cr_val = fig.add_subplot(gs[ 4:5 , stack_i])
            ax_cr_200 = fig.add_subplot(gs[ 5:6 , stack_i])
            ax_vl = fig.add_subplot(gs[ 6:7 , stack_i])
            ax_max_v = fig.add_subplot(gs[ 7:8 , stack_i])
            ax_acc = fig.add_subplot(gs[ 8:9 , stack_i])
            ax_isi = fig.add_subplot(gs[ 9:10 , stack_i])
            ax_n = fig.add_subplot(gs[ 10:11 , stack_i])

            ax_bl.spines['top'].set_visible(False)
            ax_bl.spines['right'].set_visible(False)

            ax_cr.spines['top'].set_visible(False)
            ax_cr.spines['right'].set_visible(False)

            ax_pk.spines['top'].set_visible(False)
            ax_pk.spines['right'].set_visible(False)

            ax_vl.spines['top'].set_visible(False)
            ax_vl.spines['right'].set_visible(False)

            ax_isi.spines['top'].set_visible(False)
            ax_isi.spines['right'].set_visible(False)

            ax_cr_val.spines['top'].set_visible(False)
            ax_cr_val.spines['right'].set_visible(False)

            ax_cr_200.spines['top'].set_visible(False)
            ax_cr_200.spines['right'].set_visible(False)

            ax_max_v.spines['top'].set_visible(False)
            ax_max_v.spines['right'].set_visible(False)

            ax_max_t.spines['top'].set_visible(False)
            ax_max_t.spines['right'].set_visible(False)

            ax_acc.spines['top'].set_visible(False)
            ax_acc.spines['right'].set_visible(False)

            ax_n.spines['top'].set_visible(False)
            ax_n.spines['right'].set_visible(False)

            ax_bl.set_xlabel('Trials')
            ax_cr.set_xlabel('Trials')
            ax_pk.set_xlabel('Trials')
            ax_vl.set_xlabel('Trials')
            ax_isi.set_xlabel('Trials')
            ax_cr_val.set_xlabel('Trials')
            ax_cr_200.set_xlabel('Trials')
            ax_max_v.set_xlabel('Trials')
            ax_max_t.set_xlabel('Trials')
            ax_acc.set_xlabel('Trials')
            ax_n.set_xlabel('Trials')
            
            ax_cr.set_ylabel('CR Time (ms)')
            ax_pk.set_ylabel('Peak Time (ms)')
            ax_max_t.set_ylabel('Max velocity time')
            ax_bl.set_ylabel('Average base line amplitude')
            ax_cr_val.set_ylabel('Average FEC value in CR window')
            ax_cr_200.set_ylabel('Average FEC value in t = 200 ms')
            ax_vl.set_ylabel('Averge velocity in the CR window')
            ax_max_v.set_ylabel('Max velocity in isi')
            ax_acc.set_ylabel('Sign changes during the isi')
            ax_isi.set_ylabel('ISI (ms)')
            ax_n.set_ylabel('Number of sessions for each trial')

            x_axis = list(range(- transition_0 - 1, transition_1 - 1))

            ax_bl.axvline(0, color = 'gray', linestyle = '--', alpha = 0.7, label = 'Transition')
            ax_bl.plot(x_axis, bl_avg[stack_i])
            ax_bl.fill_between(x_axis, np.array(bl_avg[stack_i]) - np.array(bl_sem[stack_i]), np.array(bl_avg[stack_i]) + np.array(bl_sem[stack_i]) ,alpha=0.2, color='lime' if stack_i % 2 == 1 else 'blue')
            

            ax_cr.axvline(0, color = 'gray', linestyle = '--', alpha = 0.7, label = 'Transition')
            ax_cr.plot(x_axis, cr_avg[stack_i])
            ax_cr.fill_between(x_axis, np.array(cr_avg[stack_i]) - np.array(cr_sem[stack_i]), np.array(cr_avg[stack_i]) + np.array(cr_sem[stack_i]) ,alpha=0.2, color='lime' if stack_i % 2 == 1 else 'blue')
            

            ax_pk.axvline(0, color = 'gray', linestyle = '--', alpha = 0.7, label = 'Transtion')
            ax_pk.plot(x_axis, pk_avg[stack_i])
            ax_pk.fill_between(x_axis, np.array(pk_avg[stack_i]) - np.array(pk_sem[stack_i]), np.array(pk_avg[stack_i]) + np.array(pk_sem[stack_i]) ,alpha=0.2, color='lime' if stack_i % 2 == 1 else 'blue')
            

            ax_vl.axvline(0, color = 'gray', linestyle = '--', alpha = 0.7, label = 'Transtion')
            ax_vl.plot(x_axis, vl_avg[stack_i])
            ax_vl.fill_between(x_axis, np.array(vl_avg[stack_i]) - np.array(vl_sem[stack_i]), np.array(vl_avg[stack_i]) + np.array(vl_sem[stack_i]) ,alpha=0.2, color='lime' if stack_i % 2 == 1 else 'blue')


            ax_isi.axvline(0, color = 'gray', linestyle = '--', alpha = 0.7, label = 'Transtion')
            ax_isi.plot(x_axis, isi_avg[stack_i])
            ax_isi.fill_between(x_axis, np.array(isi_avg[stack_i]) - np.array(isi_std[stack_i]), np.array(isi_avg[stack_i]) + np.array(isi_std[stack_i]) ,alpha=0.2, color='lime' if stack_i % 2 == 1 else 'blue')


            ax_cr_val.axvline(0, color = 'gray', linestyle = '--', alpha = 0.7, label = 'Transtion')
            ax_cr_val.plot(x_axis, cr_val_avg[stack_i])
            ax_cr_val.fill_between(x_axis, np.array(cr_val_avg[stack_i]) - np.array(cr_val_sem[stack_i]), np.array(cr_val_avg[stack_i]) + np.array(cr_val_sem[stack_i]) ,alpha=0.2, color='lime' if stack_i % 2 == 1 else 'blue')

            ax_cr_200.axvline(0, color = 'gray', linestyle = '--', alpha = 0.7, label = 'Transtion')
            ax_cr_200.plot(x_axis, cr_200_avg[stack_i])
            ax_cr_200.fill_between(x_axis, np.array(cr_200_avg[stack_i]) - np.array(cr_200_sem[stack_i]), np.array(cr_200_avg[stack_i]) + np.array(cr_200_sem[stack_i]) ,alpha=0.2, color='lime' if stack_i % 2 == 1 else 'blue')


            ax_max_v.axvline(0, color = 'gray', linestyle = '--', alpha = 0.7, label = 'Transtion')
            ax_max_v.plot(x_axis, max_v_avg[stack_i])
            ax_max_v.fill_between(x_axis, np.array(max_v_avg[stack_i]) - np.array(max_v_sem[stack_i]), np.array(max_v_avg[stack_i]) + np.array(max_v_sem[stack_i]) ,alpha=0.2, color='lime' if stack_i % 2 == 1 else 'blue')

            ax_max_t.axvline(0, color = 'gray', linestyle = '--', alpha = 0.7, label = 'Transtion')
            ax_max_t.plot(x_axis, max_t_avg[stack_i])
            ax_max_t.fill_between(x_axis, np.array(max_t_avg[stack_i]) - np.array(max_t_sem[stack_i]), np.array(max_t_avg[stack_i]) + np.array(max_t_sem[stack_i]) ,alpha=0.2, color='lime' if stack_i % 2 == 1 else 'blue')


            ax_acc.axvline(0, color = 'gray', linestyle = '--', alpha = 0.7, label = 'Transtion')
            ax_acc.plot(x_axis, acc_avg[stack_i])
            ax_acc.fill_between(x_axis, np.array(acc_avg[stack_i]) - np.array(acc_sem[stack_i]), np.array(acc_avg[stack_i]) + np.array(acc_sem[stack_i]) ,alpha=0.2, color='lime' if stack_i % 2 == 1 else 'blue')


            ax_n.axvline(0, color = 'gray', linestyle = '--', alpha = 0.7, label = 'Transtion')
            ax_n.plot(x_axis, n_slots[stack_i], linestyle = '--')
            


            ax_cr.set_title(f'{stack_i} short to long' if stack_i % 2 == 0 else f'{stack_i} long to short' )
            try:
                ax_bl.set_ylim(min(y_min_bl), max(y_max_bl))
                ax_bl.legend()
                ax_cr.set_ylim(min(y_min_cr), max(y_max_cr))
                ax_cr.legend()
                ax_pk.set_ylim(min(y_min_pk), max(y_max_pk))
                ax_pk.legend()
                ax_vl.set_ylim(min(y_min_vl), max(y_max_vl))
                ax_vl.legend()
                ax_isi.set_ylim(min(y_min_isi) - 10, max(y_max_isi) + 10)
                ax_isi.legend()
                ax_n.set_ylim(min(y_min_n) - 1, max(y_max_n) + 1)
                ax_n.legend()
                ax_cr_val.set_ylim(min(y_min_cr_val), max(y_max_cr_val))
                ax_cr_val.legend()
                ax_cr_200.set_ylim(min(y_min_cr_200), max(y_max_cr_200))
                ax_cr_200.legend()
                ax_max_v.set_ylim(min(y_min_max_v), max(y_max_max_v))
                ax_max_v.legend()
                ax_max_t.set_ylim(min(y_min_max_t), max(y_max_max_t))
                ax_max_t.legend()
                ax_acc.set_ylim(min(y_min_acc), max(y_max_acc))
                ax_acc.legend()
            except Exception as e:
                print(e)


            
        with PdfPages(transition_file) as pdf:
            pdf.savefig(fig, dpi = 400)
            pdf.close()


        for i in range(2):
            for trial_id in range(len(bl_slots_1[i])):

                bl_avg_1[i][trial_id] = np.nanmean(bl_slots_1[i][trial_id])
                bl_sem_1[i][trial_id] = np.nanstd(bl_slots_1[i][trial_id]) / np.sqrt(n_slots_1[i][trial_id])

                cr_avg_1[i][trial_id] = np.nanmean(cr_slots_1[i][trial_id])
                cr_sem_1[i][trial_id] = np.nanmean(cr_slots_1[i][trial_id]) / np.sqrt(n_slots_1[i][trial_id])

                pk_avg_1[i][trial_id] = np.nanmean(pk_slots_1[i][trial_id])
                pk_sem_1[i][trial_id] = np.nanmean(pk_slots_1[i][trial_id]) / np.sqrt(n_slots_1[i][trial_id])

                vl_avg_1[i][trial_id] = np.nanmean(vl_slots_1[i][trial_id])
                vl_sem_1[i][trial_id] = np.nanstd(vl_slots_1[i][trial_id]) / np.sqrt(n_slots_1[i][trial_id])

                ramp_avg_1[i][trial_id] = np.nanmean(ramp_slots_1[i][trial_id])
                ramp_sem_1[i][trial_id] = np.nanstd(ramp_slots_1[i][trial_id]) / np.sqrt(n_slots_1[i][trial_id])

                cr_val_avg_1[i][trial_id] = np.nanmean(cr_val_slots_1[i][trial_id])
                cr_val_sem_1[i][trial_id] = np.nanstd(cr_val_slots_1[i][trial_id]) / np.sqrt(n_slots_1[i][trial_id])

                cr_200_avg_1[i][trial_id] = np.nanmean(cr_200_slots_1[i][trial_id])
                cr_200_sem_1[i][trial_id] = np.nanstd(cr_200_slots_1[i][trial_id]) / np.sqrt(n_slots_1[i][trial_id])

                max_v_avg_1[i][trial_id] = np.nanmean(max_v_slots_1[i][trial_id])
                max_v_sem_1[i][trial_id] = np.nanmean(max_v_slots_1[i][trial_id]) / np.sqrt(n_slots_1[i][trial_id])

                max_t_avg_1[i][trial_id] = np.nanmean(max_t_slots_1[i][trial_id])
                max_t_sem_1[i][trial_id] = np.nanmean(max_t_slots_1[i][trial_id]) / np.sqrt(n_slots_1[i][trial_id])

                acc_avg_1[i][trial_id] = np.nanmean(acc_slots_1[i][trial_id])
                acc_sem_1[i][trial_id] = np.nanmean(acc_slots_1[i][trial_id]) / np.sqrt(n_slots_1[i][trial_id])

                 
            bl_pre_1[i] = np.average(bl_slots_pre[i])
            bl_post_1[i] = np.average(bl_slots_post[i])
            bl_pre_sem_1[i] = np.std(bl_slots_pre[i]) / np.sqrt(len(bl_slots_pre[i]))
            bl_post_sem_1[i] = np.std(bl_slots_post[i]) / np.sqrt(len(bl_slots_post[i]))

            cr_pre_1[i] = np.average(cr_slots_pre[i])
            cr_post_1[i] = np.average(cr_slots_post[i])
            cr_pre_sem_1[i] = np.std(cr_slots_pre[i]) / np.sqrt(len(cr_slots_pre[i]))
            cr_post_sem_1[i] = np.std(cr_slots_post[i]) / np.sqrt(len(cr_slots_post[i]))

            pk_pre_1[i] = np.average(pk_slots_pre[i])
            pk_post_1[i] = np.average(pk_slots_post[i])
            pk_pre_sem_1[i] = np.std(pk_slots_pre[i]) / np.sqrt(len(pk_slots_pre[i]))
            pk_post_sem_1[i] = np.std(pk_slots_post[i]) / np.sqrt(len(pk_slots_post[i]))

            vl_pre_1[i] = np.average(vl_slots_pre[i])
            vl_post_1[i] = np.average(vl_slots_post[i])
            vl_pre_sem_1[i] = np.std(vl_slots_pre[i]) / np.sqrt(len(vl_slots_pre[i]))
            vl_post_sem_1[i] = np.std(vl_slots_post[i]) / np.sqrt(len(vl_slots_post[i]))

            # ramp_pre_1[i] = np.average(ramp_slots_pre[i])
            # ramp_post_1[i] = np.average(ramp_slots_post[i])
            # ramp_pre_sem_1[i] = np.std(ramp_slots_pre[i]) / np.sqrt(len(ramp_slots_pre[i]))
            # ramp_post_sem_1[i] = np.std(ramp_slots_post[i]) / np.sqrt(len(ramp_slots_post[i]))

            cr_val_pre_1[i] = np.average(cr_val_slots_pre[i])
            cr_val_post_1[i] = np.average(cr_val_slots_post[i])
            cr_val_pre_sem_1[i] = np.std(cr_val_slots_pre[i]) / np.sqrt(len(cr_val_slots_pre[i]))
            cr_val_post_sem_1[i] = np.std(cr_val_slots_post[i]) / np.sqrt(len(cr_val_slots_post[i]))

            cr_200_pre_1[i] = np.average(cr_200_slots_pre[i])
            cr_200_post_1[i] = np.average(cr_200_slots_post[i])
            cr_200_pre_sem_1[i] = np.std(cr_200_slots_pre[i]) / np.sqrt(len(cr_200_slots_pre[i]))
            cr_200_post_sem_1[i] = np.std(cr_200_slots_post[i]) / np.sqrt(len(cr_200_slots_post[i]))

            max_v_pre_1[i] = np.average(max_v_slots_pre[i])
            max_v_post_1[i] = np.average(max_v_slots_post[i])
            max_v_pre_sem_1[i] = np.std(max_v_slots_pre[i]) / np.sqrt(len(max_v_slots_pre[i]))
            max_v_post_sem_1[i] = np.std(max_v_slots_post[i]) / np.sqrt(len(max_v_slots_post[i]))

            max_t_pre_1[i] = np.average(max_t_slots_pre[i])
            max_t_post_1[i] = np.average(max_t_slots_post[i])
            max_t_pre_sem_1[i] = np.std(max_t_slots_pre[i]) / np.sqrt(len(max_t_slots_pre[i]))
            max_t_post_sem_1[i] = np.std(max_t_slots_post[i]) / np.sqrt(len(max_t_slots_post[i]))

            acc_pre_1[i] = np.average(acc_slots_pre[i])
            acc_post_1[i] = np.average(acc_slots_post[i])
            acc_pre_sem_1[i] = np.std(acc_slots_pre[i]) / np.sqrt(len(acc_slots_pre[i]))
            acc_post_sem_1[i] = np.std(acc_slots_post[i]) / np.sqrt(len(acc_slots_post[i]))

            x_axis = list(range(- transition_0 - 1, transition_1 - 1))

            y_min_bl_1.append(min(np.array(bl_avg_1[i]) - np.array(bl_sem_1[i])))
            y_max_bl_1.append(max(np.array(bl_avg_1[i]) + np.array(bl_sem_1[i])))

            y_min_cr_1.append(min(np.array(cr_avg_1[i]) - np.array(cr_sem_1[i])))
            y_max_cr_1.append(max(np.array(cr_avg_1[i]) + np.array(cr_sem_1[i])))

            y_min_pk_1.append(min(np.array(pk_avg_1[i]) - np.array(pk_sem_1[i])))
            y_max_pk_1.append(max(np.array(pk_avg_1[i]) + np.array(pk_sem_1[i])))

            y_min_vl_1.append(min(np.array(vl_avg_1[i]) - np.array(vl_sem_1[i])))
            y_max_vl_1.append(max(np.array(vl_avg_1[i]) + np.array(vl_sem_1[i])))

            y_min_cr_val_1.append(min(np.array(cr_val_avg_1[i]) - np.array(cr_val_sem_1[i])))
            y_max_cr_val_1.append(max(np.array(cr_val_avg_1[i]) + np.array(cr_val_sem_1[i])))

            y_min_cr_200_1.append(min(np.array(cr_200_avg_1[i]) - np.array(cr_200_sem_1[i])))
            y_max_cr_200_1.append(max(np.array(cr_200_avg_1[i]) + np.array(cr_200_sem_1[i])))

            y_min_max_v_1.append(min(np.array(max_v_avg_1[i]) - np.array(max_v_sem_1[i])))
            y_max_max_v_1.append(max(np.array(max_v_avg_1[i]) + np.array(max_v_sem_1[i])))

            y_min_max_t_1.append(min(np.array(max_t_avg_1[i]) - np.array(max_t_sem_1[i])))
            y_max_max_t_1.append(max(np.array(max_t_avg_1[i]) + np.array(max_t_sem_1[i])))

            y_min_acc_1.append(min(np.array(acc_avg_1[i]) - np.array(acc_sem_1[i])))
            y_max_acc_1.append(max(np.array(acc_avg_1[i]) + np.array(acc_sem_1[i])))

            y_min_bl_bar.append(min(bl_post_1[i] - 1.5 * (bl_post_sem_1[i]), bl_pre_1[i] - 1.5 * (bl_pre_sem_1[i])))
            y_max_bl_bar.append(max(bl_post_1[i] + 1.5 * (bl_post_sem_1[i]), bl_pre_1[i] + 1.5 * (bl_pre_sem_1[i])))

            y_min_cr_bar.append(min(cr_post_1[i] - 1.5 * (cr_post_sem_1[i]), cr_pre_1[i] - 1.5 * (cr_pre_sem_1[i])))
            y_max_cr_bar.append(max(cr_post_1[i] + 1.5 * (cr_post_sem_1[i]), cr_pre_1[i] + 1.5 * (cr_pre_sem_1[i])))

            y_min_pk_bar.append(min(pk_post_1[i] - 1.5 * (pk_post_sem_1[i]), pk_pre_1[i] - 1.5 * (pk_pre_sem_1[i])))
            y_max_pk_bar.append(max(pk_post_1[i] + 1.5 * (pk_post_sem_1[i]), pk_pre_1[i] + 1.5 * (pk_pre_sem_1[i])))

            y_min_vl_bar.append(min(vl_post_1[i] - 1.5 * (vl_post_sem_1[i]), vl_pre_1[i] - 1.5 * (vl_pre_sem_1[i])))
            y_max_vl_bar.append(max(vl_post_1[i] + 1.5 * (vl_post_sem_1[i]), vl_pre_1[i] + 1.5 * (vl_pre_sem_1[i])))

            y_min_cr_val_bar.append(min(cr_val_post_1[i] - 1.5 * (cr_val_post_sem_1[i]), cr_val_pre_1[i] - 1.5 * (cr_val_pre_sem_1[i])))
            y_max_cr_val_bar.append(max(cr_val_post_1[i] + 1.5 * (cr_val_post_sem_1[i]), cr_val_pre_1[i] + 1.5 * (cr_val_pre_sem_1[i])))

            y_min_cr_200_bar.append(min(cr_200_post_1[i] - 1.5 * (cr_200_post_sem_1[i]), cr_200_pre_1[i] - 1.5 * (cr_200_pre_sem_1[i])))
            y_max_cr_200_bar.append(max(cr_200_post_1[i] + 1.5 * (cr_200_post_sem_1[i]), cr_200_pre_1[i] + 1.5 * (cr_200_pre_sem_1[i])))

            y_min_max_v_bar.append(min(max_v_post_1[i] - 1.5 * (max_v_post_sem_1[i]), max_v_pre_1[i] - 1.5 * (max_v_pre_sem_1[i])))
            y_max_max_v_bar.append(max(max_v_post_1[i] + 1.5 * (max_v_post_sem_1[i]), max_v_pre_1[i] + 1.5 * (max_v_pre_sem_1[i])))

            y_min_max_t_bar.append(min(max_t_post_1[i] - 1.5 * (max_t_post_sem_1[i]), max_t_pre_1[i] - 1.5 * (max_t_pre_sem_1[i])))
            y_max_max_t_bar.append(max(max_t_post_1[i] + 1.5 * (max_t_post_sem_1[i]), max_t_pre_1[i] + 1.5 * (max_t_pre_sem_1[i])))

            y_min_acc_bar.append(min(acc_post_1[i] - 1.5 * (acc_post_sem_1[i]), acc_pre_1[i] - 1.5 * (acc_pre_sem_1[i])))
            y_max_acc_bar.append(max(acc_post_1[i] + 1.5 * (acc_post_sem_1[i]), acc_pre_1[i] + 1.5 * (acc_pre_sem_1[i])))


            color_type = 'black' if test_type_check == 1 else 'red'
            label_type = 'Control' if test_type_check == 1 else 'SD'
            color_block = 'lime' if i % 2 == 1 else 'blue'
            x_pre = -0.9 if test_type_check == 1 else -1
            x_post = 0.9 if test_type_check != 1 else 1

            ax_0[0, i].plot(x_axis, cr_avg_1[i], color = color_type, label = label_type)
            ax_0[0, i].fill_between(x_axis, np.array(cr_avg_1[i]) - np.array(cr_sem_1[i]), np.array(cr_avg_1[i]) + np.array(cr_sem_1[i]) ,alpha=0.2, color = color_type)
            ax_0[0, i].axvline(0, color = color_block, linestyle = '--', alpha = 0.7, label = 'Transtion' if test_type_check == 1 else '')

            ax_0[0, i].scatter(x_axis[transition_0 - 1: transition_0 + 4], cr_avg_1[i][transition_0 - 1: transition_0 + 4], marker = '*', color = color_type)

            ax_0[1, i].plot(x_axis, pk_avg_1[i], color = color_type, label = label_type)
            ax_0[1, i].fill_between(x_axis, np.array(pk_avg_1[i]) - np.array(pk_sem_1[i]), np.array(pk_avg_1[i]) + np.array(pk_sem_1[i]) ,alpha=0.2, color = color_type)
            ax_0[1, i].axvline(0, color = color_block, linestyle = '--', alpha = 0.7, label = 'Transtion' if test_type_check == 1 else '')

            ax_0[1, i].scatter(x_axis[transition_0 - 1: transition_0 + 4], pk_avg_1[i][transition_0 - 1: transition_0 + 4], marker = '*', color = color_type)

            ax_0[2, i].axvline(0, color = color_block, linestyle = '--', alpha = 0.7, label = 'Transtion' if test_type_check == 1 else '')
            ax_0[2, i].plot(x_axis, max_t_avg_1[i], color = color_type, label = label_type)
            ax_0[2, i].fill_between(x_axis, np.array(max_t_avg_1[i]) - np.array(max_t_sem_1[i]), np.array(max_t_avg_1[i]) + np.array(max_t_sem_1[i]) ,alpha=0.2, color = color_type)

            ax_0[2, i].scatter(x_axis[transition_0 - 1: transition_0 + 4], max_t_avg_1[i][transition_0 - 1: transition_0 + 4], marker = '*', color = color_type)

            ax_0[3, i].axvline(0, color = color_block, linestyle = '--', alpha = 0.7, label = 'Transtion' if test_type_check == 1 else '')
            ax_0[3, i].plot(x_axis, bl_avg_1[i], color = color_type, label = label_type)
            ax_0[3, i].fill_between(x_axis, np.array(bl_avg_1[i]) - np.array(bl_sem_1[i]), np.array(bl_avg_1[i]) + np.array(bl_sem_1[i]) ,alpha=0.2, color = color_type)

            ax_0[3, i].scatter(x_axis[transition_0 - 1: transition_0 + 4], bl_avg_1[i][transition_0 - 1: transition_0 + 4], marker = '*', color = color_type)

            ax_0[4, i].axvline(0, color = color_block, linestyle = '--', alpha = 0.7, label = 'Transtion' if test_type_check == 1 else '')
            ax_0[4, i].plot(x_axis, cr_val_avg_1[i], color = color_type, label = label_type) 
            ax_0[4, i].fill_between(x_axis, np.array(cr_val_avg_1[i]) - np.array(cr_val_sem_1[i]), np.array(cr_val_avg_1[i]) + np.array(cr_val_sem_1[i]) ,alpha=0.2, color = color_type)

            ax_0[4, i].scatter(x_axis[transition_0 - 1: transition_0 + 4], cr_val_avg_1[i][transition_0 - 1: transition_0 + 4], marker = '*', color = color_type)

            ax_0[5, i].axvline(0, color = color_block, linestyle = '--', alpha = 0.7, label = 'Transtion' if test_type_check == 1 else '')
            ax_0[5, i].plot(x_axis, cr_200_avg_1[i], color = color_type, label = label_type)
            ax_0[5, i].fill_between(x_axis, np.array(cr_200_avg_1[i]) - np.array(cr_200_sem_1[i]), np.array(cr_200_avg_1[i]) + np.array(cr_200_sem_1[i]) ,alpha=0.2, color = color_type)

            ax_0[5, i].scatter(x_axis[transition_0 - 1: transition_0 + 4], cr_200_avg_1[i][transition_0 - 1: transition_0 + 4], marker = '*', color = color_type)


            ax_0[6, i].axvline(0, color = color_block, linestyle = '--', alpha = 0.7, label = 'Transtion' if test_type_check == 1 else '')
            ax_0[6, i].plot(x_axis, vl_avg_1[i], color = color_type, label = label_type)
            ax_0[6, i].fill_between(x_axis, np.array(vl_avg_1[i]) - np.array(vl_sem_1[i]), np.array(vl_avg_1[i]) + np.array(vl_sem_1[i]) ,alpha=0.2, color = color_type)

            ax_0[6, i].scatter(x_axis[transition_0 - 1: transition_0 + 4], vl_avg_1[i][transition_0 - 1: transition_0 + 4], marker = '*', color = color_type)

            ax_0[7, i].axvline(0, color = color_block, linestyle = '--', alpha = 0.7, label = 'Transtion' if test_type_check == 1 else '')
            ax_0[7, i].plot(x_axis, ramp_avg_1[i], color = color_type, label = label_type)
            ax_0[7, i].fill_between(x_axis, np.array(ramp_avg_1[i]) - np.array(ramp_sem_1[i]), np.array(ramp_avg_1[i]) + np.array(ramp_sem_1[i]) ,alpha=0.2, color = color_type)

            ax_0[7, i].scatter(x_axis[transition_0 - 1: transition_0 + 4], ramp_avg_1[i][transition_0 - 1: transition_0 + 4], marker = '*', color = color_type)

            ax_0[8, i].axvline(0, color = color_block, linestyle = '--', alpha = 0.7, label = 'Transtion' if test_type_check == 1 else '')
            ax_0[8, i].plot(x_axis, max_v_avg_1[i], color = color_type, label = label_type)
            ax_0[8, i].fill_between(x_axis, np.array(max_v_avg_1[i]) - np.array(max_v_sem_1[i]), np.array(max_v_avg_1[i]) + np.array(max_v_sem_1[i]) ,alpha=0.2, color = color_type)

            ax_0[8, i].scatter(x_axis[transition_0 - 1: transition_0 + 4], max_v_avg_1[i][transition_0 - 1: transition_0 + 4], marker = '*', color = color_type)

            ax_0[9, i].axvline(0, color = color_block, linestyle = '--', alpha = 0.7, label = 'Transtion' if test_type_check == 1 else '')
            ax_0[9, i].plot(x_axis, acc_avg_1[i], color = color_type, label = label_type)
            ax_0[9, i].fill_between(x_axis, np.array(acc_avg_1[i]) - np.array(acc_sem_1[i]), np.array(acc_avg_1[i]) + np.array(acc_sem_1[i]) ,alpha=0.2, color = color_type)

            ax_0[9, i].scatter(x_axis[transition_0 - 1: transition_0 + 4], acc_avg_1[i][transition_0 - 1: transition_0 + 4], marker = '*', color = color_type)

            # ax_1_n.axvline(0, color = 'gray', linestyle = '--', alpha = 0.7, label = 'Transtion' if test_type_check == 1 else '')
            # ax_1_n.plot(x_axis, n_slots_1[i], linestyle = '--', color = 'black' if test_type_check == 1 else 'red')
            ax_0[10, i].axvline(0, color = color_block, linestyle = '--', alpha = 0.7, label = 'Transtion' if test_type_check == 1 else '')
            ax_0[10, i].plot(x_axis, n_slots_1[i], linestyle = '--', color = color_type, label = label_type)

            ax_1[0, i].axvline(0, color=color_block, linestyle='--', linewidth=2, alpha=0.8, label='Transition' if test_type_check == 1 else '')
            ax_1[0, i].errorbar(x_pre, cr_pre_1[i], fmt='o', markersize = 4, capsize = 5, color=color_type, ecolor=color_type, label=label_type, yerr=cr_pre_sem_1[i])
            ax_1[0, i].errorbar(x_post, cr_post_1[i], fmt='o', markersize = 4, capsize = 5,color=color_type, ecolor=color_type, yerr=cr_post_sem_1[i])
            ax_1[0, i].plot([x_pre, x_post], [cr_pre_1[i], cr_post_1[i]], color=color_type, linestyle = "--")

            ax_1[1, i].axvline(0, color=color_block, linestyle='--', linewidth=2, alpha=0.8, label='Transition' if test_type_check == 1 else '')
            ax_1[1, i].errorbar(x_pre, pk_pre_1[i], fmt='o', markersize = 4, capsize = 5, color=color_type, ecolor=color_type, label=label_type, yerr=pk_pre_sem_1[i])
            ax_1[1, i].errorbar(x_post, pk_post_1[i], fmt='o', markersize = 4, capsize = 5,color=color_type, ecolor=color_type, yerr=pk_post_sem_1[i])
            ax_1[1, i].plot([x_pre, x_post], [pk_pre_1[i], pk_post_1[i]], color=color_type, linestyle = "--")

            ax_1[2, i].axvline(0, color=color_block, linestyle='--', linewidth=2, alpha=0.8, label='Transition' if test_type_check == 1 else '')
            ax_1[2, i].errorbar(x_pre, max_t_pre_1[i], fmt='o', markersize = 4, capsize = 5, color=color_type, ecolor=color_type, label=label_type, yerr=max_t_pre_sem_1[i])
            ax_1[2, i].errorbar(x_post, max_t_post_1[i], fmt='o', markersize = 4, capsize = 5,color=color_type, ecolor=color_type, yerr=max_t_post_sem_1[i])
            ax_1[2, i].plot([x_pre, x_post], [max_t_pre_1[i], max_t_post_1[i]], color=color_type, linestyle = "--")

            ax_1[3, i].axvline(0, color=color_block, linestyle='--', linewidth=2, alpha=0.8, label='Transition' if test_type_check == 1 else '')
            ax_1[3, i].errorbar(x_pre, bl_pre_1[i], fmt='o', markersize = 4, capsize = 5, color=color_type, ecolor=color_type, label=label_type, yerr=bl_pre_sem_1[i])
            ax_1[3, i].errorbar(x_post, bl_post_1[i], fmt='o', markersize = 4, capsize = 5,color=color_type, ecolor=color_type, yerr=bl_post_sem_1[i])
            ax_1[3, i].plot([x_pre, x_post], [bl_pre_1[i], bl_post_1[i]], color=color_type, linestyle = "--")

            ax_1[4, i].axvline(0, color=color_block, linestyle='--', linewidth=2, alpha=0.8, label='Transition' if test_type_check == 1 else '')
            ax_1[4, i].errorbar(x_pre, cr_val_pre_1[i], fmt='o', markersize = 4, capsize = 5, color=color_type, ecolor=color_type, label=label_type, yerr=cr_val_pre_sem_1[i])
            ax_1[4, i].errorbar(x_post, cr_val_post_1[i], fmt='o', markersize = 4, capsize = 5,color=color_type, ecolor=color_type, yerr=cr_val_post_sem_1[i])
            ax_1[4, i].plot([x_pre, x_post], [cr_val_pre_1[i], cr_val_post_1[i]], color=color_type, linestyle = "--")

            ax_1[5, i].axvline(0, color=color_block, linestyle='--', linewidth=2, alpha=0.8, label='Transition' if test_type_check == 1 else '')
            ax_1[5, i].errorbar(x_pre, cr_200_pre_1[i], fmt='o', markersize = 4, capsize = 5, color=color_type, ecolor=color_type, label=label_type, yerr=cr_200_pre_sem_1[i])
            ax_1[5, i].errorbar(x_post, cr_200_post_1[i], fmt='o', markersize = 4, capsize = 5,color=color_type, ecolor=color_type, yerr=cr_200_post_sem_1[i])
            ax_1[5, i].plot([x_pre, x_post], [cr_200_pre_1[i], cr_200_post_1[i]], color=color_type, linestyle = "--")

            ax_1[6, i].axvline(0, color=color_block, linestyle='--', linewidth=2, alpha=0.8, label='Transition' if test_type_check == 1 else '')
            ax_1[6, i].errorbar(x_pre, vl_pre_1[i], fmt='o', markersize = 4, capsize = 5, color=color_type, ecolor=color_type, label=label_type, yerr=vl_pre_sem_1[i])
            ax_1[6, i].errorbar(x_post, vl_post_1[i], fmt='o', markersize = 4, capsize = 5,color=color_type, ecolor=color_type, yerr=vl_post_sem_1[i])
            ax_1[6, i].plot([x_pre, x_post], [vl_pre_1[i], vl_post_1[i]], color=color_type, linestyle = "--")


            ax_1[7, i].axvline(0, color=color_block, linestyle='--', linewidth=2, alpha=0.8, label='Transition' if test_type_check == 1 else '')
            ax_1[7, i].errorbar(x_pre, max_v_pre_1[i], fmt='o', markersize = 4, capsize = 5, color=color_type, ecolor=color_type, label=label_type, yerr=max_v_pre_sem_1[i])
            ax_1[7, i].errorbar(x_post, max_v_post_1[i], fmt='o', markersize = 4, capsize = 5,color=color_type, ecolor=color_type, yerr=max_v_post_sem_1[i])
            ax_1[7, i].plot([x_pre, x_post], [max_v_pre_1[i], max_v_post_1[i]], color=color_type, linestyle = "--")


            ax_1[8, i].axvline(0, color=color_block, linestyle='--', linewidth=2, alpha=0.8, label='Transition' if test_type_check == 1 else '')
            ax_1[8, i].errorbar(x_pre, acc_pre_1[i], fmt='o', markersize = 4, capsize = 5, color=color_type, ecolor=color_type, label=label_type, yerr=acc_pre_sem_1[i])
            ax_1[8, i].errorbar(x_post, acc_post_1[i], fmt='o', markersize = 4, capsize = 5,color=color_type, ecolor=color_type, yerr=acc_post_sem_1[i])
            ax_1[8, i].plot([x_pre, x_post], [acc_pre_1[i], acc_post_1[i]], color=color_type, linestyle = "--")

            y_min_n_1.append(min(n_slots_1[i]))
            y_max_n_1.append(max(n_slots_1[i]))

            # ax_1_cr.set_title(f'{i} short to long' if i % 2 == 0 else f'{i} long to short' )
            # ax_1_cr.legend()
            # ax_1_pk.legend()
            # ax_1_vl.legend()

            ax_0[0, i].set_title(f'{i} short to long' if i % 2 == 0 else f'{i} long to short' )
            ax_0[0, i].legend()
            ax_0[0, i].set_ylabel('CR onset time(ms)')
            ax_0[1, i].legend()
            ax_0[1, i].set_ylabel('Peak time (ms)')
            ax_0[2, i].legend()
            ax_0[2, i].set_ylabel('Max velocity time')
            ax_0[3, i].legend()
            ax_0[3, i].set_ylabel('Average base line amplitude')

            ax_0[4, i].legend()
            ax_0[4, i].set_ylabel('Average FEC in the CR window')

            ax_0[5, i].legend()
            ax_0[5, i].set_ylabel('Average FEC in the 200 ms window')

            ax_0[6, i].set_ylabel('Average Velocity in the CR window (ms)')
            ax_0[6, i].legend()

            ax_0[7, i].set_ylabel('Average velocity from Cr onset to Peak')
            ax_0[7, i].legend()

            ax_0[8, i].set_ylabel('Max velocity in isi')
            ax_0[8, i].legend()
            ax_0[9, i].set_ylabel('Number of sign changes durin the isi')
            ax_0[9, i].legend()
            ax_0[10, i].set_ylabel('Number of sessions')
            ax_0[10, i].legend()

            ax_1[0, i].set_title(f'{i} short to long' if i % 2 == 0 else f'{i} long to short' )
            ax_1[0, i].legend()
            ax_1[0, i].set_ylabel('CR onset time(ms)')
            ax_1[1, i].legend()
            ax_1[1, i].set_ylabel('Peak time (ms)')
            ax_1[2, i].legend()
            ax_1[2, i].set_ylabel('Max velocity time')
            ax_1[3, i].legend()
            ax_1[3, i].set_ylabel('Average base line amplitude')
            ax_1[4, i].legend()
            ax_1[4, i].set_ylabel('Average FEC in the CR window')

            ax_1[5, i].legend()
            ax_1[5, i].set_ylabel('Average FEC in the 200 ms window')
            ax_1[6, i].set_ylabel('Average Velocity in the CR window (ms)')
            ax_1[6, i].legend()
            ax_1[7, i].set_ylabel('Max velocity in isi')
            ax_1[7, i].legend()
            ax_1[8, i].set_ylabel('Number of sign changes durin the isi')
            ax_1[8, i].legend()
            ax_1[9, i].set_ylabel('Number of sessions')
            ax_1[9, i].legend()

            for ax_0_1 in ax_0:
                for ax_0_2 in ax_0_1:
                    ax_0_2.spines['top'].set_visible(False)
                    ax_0_2.spines['right'].set_visible(False)
                    ax_0_2.set_xlabel('Trial')

            for j in range(len(ax_1)):
                for i in [0, 1]:
                    ax_1[j, i].set_xticks([-0.9, 0, 0.9])

                    if i % 2 == 0:
                        ax_1[j, i].set_xticklabels(['short', 'Transition', 'long'])
                    else:
                        ax_1[j, i].set_xticklabels(['long', 'Transition', 'short'])

                    ax_1[j, i].spines['top'].set_visible(False)
                    ax_1[j, i].spines['right'].set_visible(False)

            
    for i in range(2):
        try:
            ax_0[0, i].set_ylim(min(y_min_cr_1), max(y_max_cr_1))
            ax_0[1, i].set_ylim(min(y_min_pk_1), max(y_max_pk_1))
            ax_0[2, i].set_ylim(min(y_min_max_t_1), max(y_max_max_t_1))
            ax_0[3, i].set_ylim(min(y_min_bl_1), max(y_max_bl_1))
            ax_0[4, i].set_ylim(min(y_min_cr_val_1), max(y_max_cr_val_1))
            ax_0[5, i].set_ylim(min(y_min_cr_200_1), max(y_max_cr_200_1))
            ax_0[6, i].set_ylim(min(y_min_vl_1), max(y_max_vl_1))
            ax_0[8, i].set_ylim(min(y_min_max_v_1), max(y_max_max_v_1))
            ax_0[9, i].set_ylim(min(y_min_acc_1), max(y_max_acc_1))
            ax_0[10, i].set_ylim(0, max(y_max_n_1) + 1)

            ax_1[0, i].set_ylim(min(y_min_cr_bar), max(y_max_cr_bar))
            ax_1[1, i].set_ylim(min(y_min_pk_bar), max(y_max_pk_bar))
            ax_1[2, i].set_ylim(min(y_min_max_t_bar), max(y_max_max_t_bar))
            ax_1[3, i].set_ylim(min(y_min_bl_bar), max(y_max_bl_bar))
            ax_1[4, i].set_ylim(min(y_min_cr_val_bar), max(y_max_cr_val_bar))
            ax_1[5, i].set_ylim(min(y_min_cr_200_bar), max(y_max_cr_200_bar))
            ax_1[6, i].set_ylim(min(y_min_vl_bar), max(y_max_vl_bar))
            ax_1[7, i].set_ylim(min(y_min_max_v_bar), max(y_max_max_v_bar))
            ax_1[8, i].set_ylim(min(y_min_acc_bar), max(y_max_acc_bar))
        except:
            print('no')

    with PdfPages(transition_file_0) as pdf:
        pdf.savefig(fig_0, dpi = 400)
        pdf.close()

    with PdfPages(transtion_bar_file) as pdf:
        pdf.savefig(fig_1, dpi = 400)
        pdf.close()

    plt.close()
    print('number of first block long', number_of_first_block_long)
    print('number of bad trials' , number_of_bad_sessions)
    print('happy', cr_happy)
    print('sad', cr_sad)
    # print(f'number of all sessions: {len(all_sessions)}')
