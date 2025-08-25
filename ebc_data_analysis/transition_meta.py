import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages 
import h5py

from utils.alignment import sort_numbers_as_strings, fec_zero
from utils.indication import find_max_with_gradient, block_type, find_index, cr_onset_calc
from utils.functions import sig_trial_func, color_list


beh_folder = "./data/beh"

cr_threshold = 0.02
velocity_threshold_fraction = 0.95
amp_threshold_fraction = 0.10
cr_window_time = 50
number_of_first_block_long = 0
number_of_bad_sessions = 0

number_of_stacks= 6 #should be the same as the max number os segments
transition_0 = 20
transition_1 = 30
slice_size = 18



mice = [folder for folder in os.listdir(beh_folder) if folder != ".DS_Store"] 
mice.sort()
for mouse in mice:
    x_grid = number_of_stacks
    y_grid = 6
    # if mouse != 'E6LG':
    #     continue

    for test_type_check in [1, 0]: #1 for control and 0 for SD
            
        if mouse == 'E5LG':
            test_type_check = 1

        all_sessions = [os.path.splitext(file)[0] for file in os.listdir(os.path.join(beh_folder, mouse, "processed")) if file != ".DS_Store"]
        all_sessions.sort()
        number_of_session_slices = len(all_sessions) // slice_size

        fig, ax = plt.subplots(y_grid, x_grid, figsize=(7 * x_grid, 7 * y_grid), sharex=False, sharey=False)

        y_grid = number_of_session_slices

        fig_cr, ax_cr = plt.subplots(y_grid, x_grid, figsize=(7 * x_grid, 7 * y_grid), sharex=False, sharey=False)
        fig_pk, ax_pk = plt.subplots(y_grid, x_grid, figsize=(7 * x_grid, 7 * y_grid), sharex=False, sharey=False)
        fig_vl, ax_vl = plt.subplots(y_grid, x_grid, figsize=(7 * x_grid, 7 * y_grid), sharex=False, sharey=False)
        fig_cr_val, ax_cr_val = plt.subplots(y_grid, x_grid, figsize=(7 * x_grid, 7 * y_grid), sharex=False, sharey=False)
        fig_max_v, ax_max_v = plt.subplots(y_grid, x_grid, figsize=(7 * x_grid, 7 * y_grid), sharex=False, sharey=False)
        fig_acc, ax_acc = plt.subplots(y_grid, x_grid, figsize=(7 * x_grid, 7 * y_grid), sharex=False, sharey=False)

        # test_type_check = 1 #1 for control and 0 for SD
        test_type_text = None
        if test_type_check == 1:
            test_type_text = 'control'
        if test_type_check == 0:
            test_type_text = 'SD'


        
        if not os.path.exists(f'./outputs/beh/{mouse}/meta'):
                os.mkdir(f'./outputs/beh/{mouse}/meta')
                print(f"data folder '{mouse}' created.")
        else:
            print(f"data folder '{mouse}' already exists.")

        meta_transition_file = f"./outputs/beh/{mouse}/meta/meta_superimposed_{test_type_text}_{slice_size}.pdf"
        meta_transition_file_cr = f"./outputs/beh/{mouse}/meta/meta_cr_onset_{test_type_text}_{slice_size}.pdf"
        meta_transition_file_pk = f"./outputs/beh/{mouse}/meta/meta_peak_{test_type_text}_{slice_size}.pdf"
        meta_transition_file_vl = f"./outputs/beh/{mouse}/meta/meta_velocity_{test_type_text}_{slice_size}.pdf"
        meta_transition_file_cr_val = f"./outputs/beh/{mouse}/meta/meta_cr_value_{test_type_text}_{slice_size}.pdf"
        meta_transition_file_max_v = f"./outputs/beh/{mouse}/meta/meta_max_velocity_{test_type_text}_{slice_size}.pdf"
        meta_transition_file_acc = f"./outputs/beh/{mouse}/meta/meta_acceleration_{test_type_text}_{slice_size}.pdf"

        y_min_cr = []
        y_max_cr = []

        y_min_pk = []
        y_max_pk = []

        y_min_vl = []
        y_max_vl = []

        y_min_cr_val = []
        y_max_cr_val = []

        y_min_max_v = []
        y_max_max_v = []

        y_min_acc = []
        y_max_acc = []
        
        n_slots = {}

        y_min_n = []
        y_max_n = []


        for slice_i in range(number_of_session_slices):
            cr_slots = {}
            cr_avg = {}
            cr_sem = {}

            pk_slots = {}
            pk_avg = {}
            pk_sem = {}

            vl_slots = {}
            vl_avg = {}
            vl_sem = {}

            cr_val_slots = {}
            cr_val_avg = {}
            cr_val_sem = {}

            max_v_slots = {}
            max_v_avg = {}
            max_v_sem = {}

            acc_slots = {}
            acc_avg = {}
            acc_sem = {}

            
            for slot_i in range(number_of_stacks):
                cr_slots[slot_i] = []
                cr_avg[slot_i] = []
                cr_sem[slot_i] = []

                pk_slots[slot_i] = []
                pk_avg[slot_i] = []
                pk_sem[slot_i] = []

                vl_slots[slot_i] = []
                vl_avg[slot_i] = []
                vl_sem[slot_i] = []

                cr_val_slots[slot_i] = []
                cr_val_avg[slot_i] = []
                cr_val_sem[slot_i] = []

                max_v_slots[slot_i] = []
                max_v_avg[slot_i] = []
                max_v_sem[slot_i] = []

                acc_slots[slot_i] = []
                acc_avg[slot_i] = []
                acc_sem[slot_i] = []

                n_slots[slot_i] = []

                for id in range(transition_0 + transition_1):
                    cr_slots[slot_i].append([])
                    cr_avg[slot_i].append([])
                    cr_sem[slot_i].append([])

                    pk_slots[slot_i].append([])
                    pk_avg[slot_i].append([])
                    pk_sem[slot_i].append([])

                    vl_slots[slot_i].append([])
                    vl_avg[slot_i].append([])
                    vl_sem[slot_i].append([])

                    cr_val_slots[slot_i].append([])
                    cr_val_avg[slot_i].append([])
                    cr_val_sem[slot_i].append([])

                    max_v_slots[slot_i].append([])
                    max_v_avg[slot_i].append([])
                    max_v_sem[slot_i].append([])

                    acc_slots[slot_i].append([])
                    acc_avg[slot_i].append([])
                    acc_sem[slot_i].append([])

                    n_slots[slot_i].append(0) 


            i_start = slice_i * slice_size
            for i , session_date in enumerate(all_sessions[i_start: i_start + slice_size]):

                print(session_date, 'being localized')

                trials = h5py.File(f"./data/beh/{mouse}/processed/{session_date}.h5")["trial_id"]
                fec, fec_time, trials = fec_zero(trials)
                shorts, longs = block_type(trials)
                all_id = sort_numbers_as_strings(shorts + longs)

                sig_trial_ids, slot_ids = sig_trial_func(all_id, trials, transition_0, transition_1)

                # if trials[all_id[0]]["trial_type"][()] == 2:
                #     print('first block is long')
                #     number_of_first_block_long += 1
                #     continue

                test_type = trials[all_id[0]]["test_type"][()]
                if test_type == test_type_check:
                    # breakpoint()
                    continue

                for slot_i in range(min(number_of_stacks, len(slot_ids))):
                    print(slot_i)
                    for i, trial_id in enumerate(slot_ids[slot_i]):

                        if trial_id not in all_id:
                            print('not enough transitions')
                            continue

                        trial = trials[trial_id]
                        airpuff = trial['AirPuff'][0]- trial["LED"][0]

                        if i < transition_0 + 1 and airpuff > 210 and slot_i % 2 == 0:
                            number_of_bad_sessions += 1
                            continue

                        elif i < transition_0 + 1 and airpuff < 380 and slot_i % 2 == 1:
                            number_of_bad_sessions += 1
                            continue

                        elif i > transition_0 and airpuff < 380 and slot_i % 2 == 0:
                            number_of_bad_sessions += 1
                            continue

                        elif i > transition_0 and airpuff > 210 and slot_i % 2 == 1:
                            number_of_bad_sessions += 1
                            continue

                        # CR stat indication >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                        fec_index_0 = np.abs(fec_time[trial_id]).argmin()
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

                        peak_time, peak_value, _, gradients = find_max_with_gradient(
                            fec_time[trial_id][fec_index_led: fec_index_ap], 
                            fec[trial_id][fec_index_led: fec_index_ap])

                        if peak_time is None:
                            peak_time = airpuff

                        

                        cr_idx = cr_onset_calc(
                            fec[trial_id], fec_time[trial_id], 10, airpuff, cr_stat)

                        cr_value = np.mean(fec[trial_id][fec_index_cr : fec_index_ap]) - bl_amp

                        acc = np.mean(np.gradient(gradients, fec_time[trial_id][fec_index_led : fec_index_ap]))
                        max_v = np.max(gradients)
                        if cr_idx:
                            cr_time = fec_time[trial_id][cr_idx]
                            cr_slots[slot_i][i].append(cr_time)
                        pk_slots[slot_i][i].append(peak_time)
                        vl_slots[slot_i][i].append(cr_slope)
                        cr_val_slots[slot_i][i].append(cr_value)
                        max_v_slots[slot_i][i].append(max_v)
                        acc_slots[slot_i][i].append(acc)
                        n_slots[slot_i][i] += 1
                        
            for stack_i in range(number_of_stacks):
                for trial_id in range(len(cr_slots[stack_i])):

                    cr_avg[stack_i][trial_id] = np.average(cr_slots[stack_i][trial_id])
                    cr_sem[stack_i][trial_id] = np.std(cr_slots[stack_i][trial_id]) / np.sqrt(n_slots[stack_i][trial_id])

                    pk_avg[stack_i][trial_id] = np.average(pk_slots[stack_i][trial_id])
                    pk_sem[stack_i][trial_id] = np.std(pk_slots[stack_i][trial_id]) / np.sqrt(n_slots[stack_i][trial_id])

                    vl_avg[stack_i][trial_id] = np.average(vl_slots[stack_i][trial_id])
                    vl_sem[stack_i][trial_id] = np.std(vl_slots[stack_i][trial_id]) / np.sqrt(n_slots[stack_i][trial_id])

                    cr_val_avg[stack_i][trial_id] = np.average(cr_val_slots[stack_i][trial_id])
                    cr_val_sem[stack_i][trial_id] = np.std(cr_val_slots[stack_i][trial_id]) / np.sqrt(n_slots[stack_i][trial_id])

                    max_v_avg[stack_i][trial_id] = np.average(max_v_slots[stack_i][trial_id])
                    max_v_sem[stack_i][trial_id] = np.std(max_v_slots[stack_i][trial_id]) / np.sqrt(n_slots[stack_i][trial_id])


                    acc_avg[stack_i][trial_id] = np.average(acc_slots[stack_i][trial_id])
                    acc_sem[stack_i][trial_id] = np.std(acc_slots[stack_i][trial_id]) / np.sqrt(n_slots[stack_i][trial_id])

                y_min_cr.append(min(np.array(cr_avg[stack_i]) - np.array(cr_sem[stack_i])))
                y_max_cr.append(max(np.array(cr_avg[stack_i]) + np.array(cr_sem[stack_i])))
                y_min_pk.append(min(np.array(pk_avg[stack_i]) - np.array(pk_sem[stack_i])))
                y_max_pk.append(max(np.array(pk_avg[stack_i]) + np.array(pk_sem[stack_i])))
                y_min_vl.append(min(np.array(vl_avg[stack_i]) - np.array(vl_sem[stack_i])))
                y_max_vl.append(max(np.array(vl_avg[stack_i]) + np.array(vl_sem[stack_i])))
                y_min_cr_val.append(min(np.array(cr_val_avg[stack_i]) - np.array(cr_val_sem[stack_i])))
                y_max_cr_val.append(max(np.array(cr_val_avg[stack_i]) + np.array(cr_val_sem[stack_i])))
                y_min_max_v.append(min(np.array(max_v_avg[stack_i]) - np.array(max_v_sem[stack_i])))
                y_max_max_v.append(max(np.array(max_v_avg[stack_i]) + np.array(max_v_sem[stack_i])))
                y_min_acc.append(min(np.array(acc_avg[stack_i]) - np.array(acc_sem[stack_i])))
                y_max_acc.append(max(np.array(acc_avg[stack_i]) + np.array(acc_sem[stack_i])))
                y_min_n.append(min(n_slots[stack_i]))
                y_max_n.append(max(n_slots[stack_i]))

            for stack_i in range(number_of_stacks):

                for i in range(6):
                    ax[i, stack_i].spines['top'].set_visible(False)
                    ax[i, stack_i].spines['right'].set_visible(False)
                    ax[i, stack_i].set_xlabel('Trials')
                    ax[i, stack_i].axvline(0, color = 'gray', linestyle = '--', alpha = 0.7)

                ax_cr[slice_i, stack_i].spines['top'].set_visible(False)
                ax_cr[slice_i, stack_i].spines['right'].set_visible(False)

                ax_pk[slice_i, stack_i].spines['top'].set_visible(False)
                ax_pk[slice_i, stack_i].spines['right'].set_visible(False)

                ax_vl[slice_i, stack_i].spines['top'].set_visible(False)
                ax_vl[slice_i, stack_i].spines['right'].set_visible(False)

                ax_cr_val[slice_i, stack_i].spines['top'].set_visible(False)
                ax_cr_val[slice_i, stack_i].spines['right'].set_visible(False)

                ax_max_v[slice_i, stack_i].spines['top'].set_visible(False)
                ax_max_v[slice_i, stack_i].spines['right'].set_visible(False)

                ax_acc[slice_i, stack_i].spines['top'].set_visible(False)
                ax_acc[slice_i, stack_i].spines['right'].set_visible(False)

                ax_cr[slice_i, stack_i].set_xlabel('Trials')
                ax_pk[slice_i, stack_i].set_xlabel('Trials')
                ax_vl[slice_i, stack_i].set_xlabel('Trials')
                ax_cr_val[slice_i, stack_i].set_xlabel('Trials')
                ax_max_v[slice_i, stack_i].set_xlabel('Trials')
                ax_acc[slice_i, stack_i].set_xlabel('Trials')

                ax_cr[slice_i, stack_i].set_ylabel('CR Time (ms)')
                ax_pk[slice_i, stack_i].set_ylabel('Peak Time (ms)')
                ax_vl[slice_i, stack_i].set_ylabel('Averge velocity in the CR window')
                ax_cr_val[slice_i, stack_i].set_ylabel('Average FEC value in CR window (Baseline subtracted)')
                ax_max_v[slice_i, stack_i].set_ylabel('Max velocity in isi')
                ax_acc[slice_i, stack_i].set_ylabel('Average acceleration during isi (ms^2)')

                ax[0, stack_i].set_ylabel('CR Time (ms)')
                ax[1, stack_i].set_ylabel('Peak Time (ms)')
                ax[2, stack_i].set_ylabel('Averge velocity in the CR window')
                ax[3, stack_i].set_ylabel('Average FEC value in CR window (Baseline subtracted)')
                ax[4, stack_i].set_ylabel('Max velocity in isi')
                ax[5, stack_i].set_ylabel('Average acceleration during isi (ms^2)')

                x_axis = list(range(- transition_0 - 1, transition_1 - 1))

                ax_cr[slice_i, stack_i].axvline(0, color = 'gray', linestyle = '--', alpha = 0.7, label = 'Transition')
                ax_cr[slice_i, stack_i].plot(x_axis, cr_avg[stack_i])
                ax[0, stack_i].plot(x_axis, cr_avg[stack_i], color = color_list(slice_i, stack_i, number_of_session_slices), alpha = 0.4, label = f'{i_start} to {i_start + slice_size}')
                ax_cr[slice_i, stack_i].fill_between(x_axis, np.array(cr_avg[stack_i]) - np.array(cr_sem[stack_i]), np.array(cr_avg[stack_i]) + np.array(cr_sem[stack_i]) ,alpha=0.2, color='lime' if stack_i % 2 == 1 else 'blue')
                
                ax_pk[slice_i, stack_i].axvline(0, color = 'gray', linestyle = '--', alpha = 0.7, label = 'Transtion')
                ax_pk[slice_i, stack_i].plot(x_axis, pk_avg[stack_i])
                ax[1, stack_i].plot(x_axis, pk_avg[stack_i], color = color_list(slice_i, stack_i, number_of_session_slices), alpha = 0.4, label = f'{i_start} to {i_start + slice_size}')
                ax_pk[slice_i, stack_i].fill_between(x_axis, np.array(pk_avg[stack_i]) - np.array(pk_sem[stack_i]), np.array(pk_avg[stack_i]) + np.array(pk_sem[stack_i]) ,alpha=0.2, color='lime' if stack_i % 2 == 1 else 'blue')
                
                ax_vl[slice_i, stack_i].axvline(0, color = 'gray', linestyle = '--', alpha = 0.7, label = 'Transtion')
                ax_vl[slice_i, stack_i].plot(x_axis, vl_avg[stack_i])
                ax[2, stack_i].plot(x_axis, vl_avg[stack_i], color = color_list(slice_i, stack_i, number_of_session_slices), alpha = 0.4, label = f'{i_start} to {i_start + slice_size}')
                ax_vl[slice_i, stack_i].fill_between(x_axis, np.array(vl_avg[stack_i]) - np.array(vl_sem[stack_i]), np.array(vl_avg[stack_i]) + np.array(vl_sem[stack_i]) ,alpha=0.2, color='lime' if stack_i % 2 == 1 else 'blue')

                ax_cr_val[slice_i, stack_i].axvline(0, color = 'gray', linestyle = '--', alpha = 0.7, label = 'Transtion')
                ax_cr_val[slice_i, stack_i].plot(x_axis, cr_val_avg[stack_i])
                ax[3, stack_i].plot(x_axis, cr_val_avg[stack_i], color = color_list(slice_i, stack_i, number_of_session_slices), alpha = 0.4, label = f'{i_start} to {i_start + slice_size}')
                ax_cr_val[slice_i, stack_i].fill_between(x_axis, np.array(cr_val_avg[stack_i]) - np.array(cr_val_sem[stack_i]), np.array(cr_val_avg[stack_i]) + np.array(cr_val_sem[stack_i]) ,alpha=0.2, color='lime' if stack_i % 2 == 1 else 'blue')

                ax_max_v[slice_i, stack_i].axvline(0, color = 'gray', linestyle = '--', alpha = 0.7, label = 'Transtion')
                ax_max_v[slice_i, stack_i].plot(x_axis, max_v_avg[stack_i])
                ax[4, stack_i].plot(x_axis, max_v_avg[stack_i], color = color_list(slice_i, stack_i, number_of_session_slices), alpha = 0.4, label = f'{i_start} to {i_start + slice_size}')
                ax_max_v[slice_i, stack_i].fill_between(x_axis, np.array(max_v_avg[stack_i]) - np.array(max_v_sem[stack_i]), np.array(max_v_avg[stack_i]) + np.array(max_v_sem[stack_i]) ,alpha=0.2, color='lime' if stack_i % 2 == 1 else 'blue')

                ax_acc[slice_i, stack_i].axvline(0, color = 'gray', linestyle = '--', alpha = 0.7, label = 'Transtion')
                ax_acc[slice_i, stack_i].plot(x_axis, acc_avg[stack_i])
                ax[5, stack_i].plot(x_axis, acc_avg[stack_i], color = color_list(slice_i, stack_i, number_of_session_slices), alpha = 0.4, label = f'{i_start} to {i_start + slice_size}')
                ax_acc[slice_i, stack_i].fill_between(x_axis, np.array(acc_avg[stack_i]) - np.array(acc_sem[stack_i]), np.array(acc_avg[stack_i]) + np.array(acc_sem[stack_i]) ,alpha=0.2, color='lime' if stack_i % 2 == 1 else 'blue')

                # ax[6, stack_i].plot(x_axis, n_slots[stack_i], linestyle = '--', color = color_list(slice_i, stack_i, number_of_session_slices), alpha = 0.4, label = f'sessions {i_start} to {i_start + slice_size}')

                ax_cr[slice_i, stack_i].set_title(f'{stack_i} short to long' if stack_i % 2 == 0 else f'{stack_i} long to short' )
                ax_cr[slice_i, stack_i].legend()
                ax_pk[slice_i, stack_i].legend()
                ax_vl[slice_i, stack_i].legend()
                ax_cr_val[slice_i, stack_i].legend()
                ax_max_v[slice_i, stack_i].legend()
                ax_acc[slice_i, stack_i].legend()
                ax[0, stack_i].set_title(f'{stack_i} short to long' if stack_i % 2 == 0 else f'{stack_i} long to short' )
                ax[0, stack_i].legend()
                ax[1, stack_i].legend()
                ax[2, stack_i].legend()
                ax[3, stack_i].legend()
                ax[4, stack_i].legend()
                ax[5, stack_i].legend()

        for slice_i in range(number_of_session_slices):
            for stack_i in range(number_of_stacks):

                try:
                    ax_cr[slice_i, stack_i].set_ylim(min(y_min_cr), max(y_max_cr))
                    ax_pk[slice_i, stack_i].set_ylim(min(y_min_pk), max(y_max_pk))
                    ax_vl[slice_i, stack_i].set_ylim(min(y_min_vl), max(y_max_vl))
                    ax_cr_val[slice_i, stack_i].set_ylim(min(y_min_cr_val), max(y_max_cr_val))
                    ax_max_v[slice_i, stack_i].set_ylim(min(y_min_max_v), max(y_max_max_v))
                    ax_acc[slice_i, stack_i].set_ylim(min(y_min_acc), max(y_max_acc))

                    # ax[0, stack_i].set_ylim(min(y_min_cr), max(y_max_cr))
                    # ax[1, stack_i].set_ylim(min(y_min_pk), max(y_max_pk))
                    # ax[2, stack_i].set_ylim(min(y_min_vl), max(y_max_vl))
                    # ax[3, stack_i].set_ylim(min(y_min_cr_val), max(y_max_cr_val))
                    # ax[4, stack_i].set_ylim(min(y_min_max_v), max(y_max_max_v))
                    # ax[5, stack_i].set_ylim(min(y_min_acc), max(y_max_acc))

                except:
                    print('idk')

        with PdfPages(meta_transition_file) as pdf:
            pdf.savefig(fig, dpi = 400)
            pdf.close()

        with PdfPages(meta_transition_file_cr) as pdf:
            pdf.savefig(fig_cr, dpi = 400)
            pdf.close()

        with PdfPages(meta_transition_file_pk) as pdf:
            pdf.savefig(fig_pk, dpi = 400)
            pdf.close()

        with PdfPages(meta_transition_file_vl) as pdf:
            pdf.savefig(fig_vl, dpi = 400)
            pdf.close()

        with PdfPages(meta_transition_file_acc) as pdf:
            pdf.savefig(fig_acc, dpi = 400)
            pdf.close()

        with PdfPages(meta_transition_file_max_v) as pdf:
            pdf.savefig(fig_max_v, dpi = 400)
            pdf.close()

        with PdfPages(meta_transition_file_cr_val) as pdf:
            pdf.savefig(fig_cr_val, dpi = 400)
            pdf.close()

    plt.close()
    print('number of first block long', number_of_first_block_long)
    print('number of bad trials' , number_of_bad_sessions)

