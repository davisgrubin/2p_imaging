#!/usr/bin/env python3

import os
import h5py
import numpy as np


def read_neural_trials(ops):
    print('Reading neural trials from saved files')
    f = h5py.File(os.path.join(ops['save_path0'], 'neural_trials.h5'),'r')
    neural_trials = dict()
    for trial in f['trial_id'].keys():
        neural_trials[trial] = dict()
        for data in f['trial_id'][trial].keys():
            neural_trials[trial][data] = np.array(f['trial_id'][trial][data])
    f.close()
    return neural_trials


def get_labels(neural_trials, first_epoch = 6, last_epoch = 10):
    """
    takes neural_trials created in trialization; and the number of trials consider first and last epoch.
    will add:
        outcome
        opoch number
        post trial otcome
        inter push delay
        the change of delay with next trial
    to each trial and return the complete version of neural_trials
    """
    print('creating labels')
    all_dt = []
    for trials in neural_trials.keys():
        # epoch label
        if 'trial_pos_in_block' in neural_trials[trials].keys():
            if neural_trials[trials]['trial_pos_in_block'] < first_epoch:
                neural_trials[trials]['block_epoch'] = 0
            elif neural_trials[trials]['trial_block_len'] - neural_trials[trials]['trial_pos_in_block'] < last_epoch:
                neural_trials[trials]['block_epoch'] = 2
            else:
                neural_trials[trials]['block_epoch'] = 1
        else: 
            neural_trials[trials]['block_epoch'] = 1
        # outcome label 
        neural_trials[trials]['trial_outcome'] = get_trial_outcome(neural_trials, trials)
        # post trial outcome label
        if 'post_trial' in neural_trials[trials].keys():
            new_trial = str(neural_trials[trials]['post_trial'])
            if new_trial in neural_trials.keys():
                neural_trials[trials]['post_trial_outcome'] = get_trial_outcome(neural_trials, new_trial)
            else:
                neural_trials[trials]['post_trial_outcome'] = np.nan
        else:
             neural_trials[trials]['post_trial_outcome'] = np.nan
        # the delay deference of current trial and next trial
        neural_trials[trials]['trial_delay_delta'] = get_trial_delta_delay(neural_trials, trials) 
        if not np.isnan(neural_trials[trials]['trial_delay_delta']):
            all_dt.append(neural_trials[trials]['trial_delay_delta'])
    decrease_lim = np.percentile(np.array(all_dt), 30)
    increase_lim = np.percentile(np.array(all_dt), 70)
    #print(all_dt)
    for trials in neural_trials.keys():
        if np.isnan(neural_trials[trials]['trial_delay_delta']):
            neural_trials[trials]['delay_change_label'] = -1
        else:
            if neural_trials[trials]['trial_delay_delta'] < decrease_lim and neural_trials[trials]['trial_delay_delta'] < 0:
                neural_trials[trials]['delay_change_label'] = 1
                print(neural_trials[trials]['trial_delay_delta'])
            elif neural_trials[trials]['trial_delay_delta'] > increase_lim and neural_trials[trials]['trial_delay_delta'] > 0:
                neural_trials[trials]['delay_change_label'] = 2
                print(neural_trials[trials]['trial_delay_delta'])
            else:
                neural_trials[trials]['delay_change_label'] = 0
        
    return neural_trials, decrease_lim, increase_lim, all_dt


def get_trial_outcome(neural_trials, trials):
    """
    takes neural_trials andthe trial string.
    returns outcome label for that trial:
        Reward = 0
        DidNotPress1 = 1
        DidNotPress2 = 2
        LatePress2 = 3
        EarlyPress2 = 4
        probe = 5
        other outcomes = -1
    """
    if neural_trials[trials]['trial_probe'] == 1:
        trial_outcome = 5
    elif not np.isnan(neural_trials[trials]['trial_reward'][0]):
        trial_outcome = 0
    elif not np.isnan(neural_trials[trials]['trial_no1stpush'][0]):
        trial_outcome = 1
    elif not np.isnan(neural_trials[trials]['trial_no2ndpush'][0])and (np.isnan(neural_trials[trials]['trial_push2'][0])):
        trial_outcome = 2
    elif (not np.isnan(neural_trials[trials]['trial_no2ndpush'][0])) and (not np.isnan(neural_trials[trials]['trial_push2'][0])):
        #print('late_press1')
        trial_outcome = 3
    elif (not np.isnan(neural_trials[trials]['trial_late2ndpush'][0])):
        #print('late_press2')
        trial_outcome = 3
    elif not np.isnan(neural_trials[trials]['trial_early2ndpush'][0]):
        trial_outcome = 4
    else:
        trial_outcome = -1
    return trial_outcome


def get_trial_delta_delay(neural_trials, trials):
    trial_outcome = np.nan
    if 'post_trial' in neural_trials[trials].keys():
        new_trial = str(neural_trials[trials]['post_trial'])
        if new_trial in neural_trials.keys():
            if not np.isnan(neural_trials[trials]['trial_press_delay'][0]):
                if not np.isnan(neural_trials[new_trial]['trial_press_delay'][0]):
                    if neural_trials[trials]['trial_press_delay'][0] > 0 and neural_trials[new_trial]['trial_press_delay'][0] > 0:
                        change_delay = neural_trials[new_trial]['trial_press_delay'][0] - neural_trials[trials]['trial_press_delay'][0]
                        if change_delay < -80:
                            trial_outcome = -1
                        elif change_delay > 80:
                            trial_outcome = 1
                        else: 
                            trial_outcome = 0
    return trial_outcome