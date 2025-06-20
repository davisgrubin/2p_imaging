# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 09:19:15 2025

@author: saminnaji3
"""
import os
import numpy as np
import h5py
from modules.ReadResults import read_ROI_label
from modules import Trialization

all_keys = ['dff', 'stim', 'time', 'trial_ST', 'trial_delay', 'trial_early2ndpush', 'trial_iti', 'trial_js_pos', 'trial_js_time',
            'trial_lick', 'trial_no1stpush', 'trial_no2ndpush', 'trial_punish', 'trial_push1', 'trial_push2', 'trial_retract1', 
            'trial_retract1_init', 'trial_retract2', 'trial_reward', 'trial_types', 'trial_vis1', 'trial_vis2', 'trial_wait2',
            'vol_stim', 'vol_time', 'trial_press_delay']

def read_neural_trials(ops):
    # read h5 file.
    f = h5py.File(
        os.path.join(ops['save_path0'], 'neural_trials.h5'),
        'r')
    neural_trials = dict()
    for trial in f['trial_id'].keys():
        neural_trials[trial] = dict()
        for data in f['trial_id'][trial].keys():
            neural_trials[trial][data] = np.array(f['trial_id'][trial][data])
    f.close()
    return neural_trials

def read_trials(file_names, trailization_on = 0):
    
    mega_session_data = dict()
    file_names.sort(key=lambda x: x[-8:])
    session_id = 0
    for f in range(len(file_names)):
        fname = file_names[f]
        ops = np.load(
            os.path.join(fname, 'suite2p', 'plane0','ops.npy'),
            allow_pickle=True).item()
        ops['save_path0'] = os.path.join(fname)
        if trailization_on:
            Trialization.run(ops)
            print('Trialization done')
        good_roi, bad_roi = read_ROI_label(ops)
        neural_trials = read_neural_trials(ops)
        for trials in neural_trials.keys():
            trial_id = fname[-8:] + '_' + trials
            mega_session_data[trial_id] = dict()
            mega_session_data[trial_id]['session_id'] = session_id
            if 'post_trial' in neural_trials[trials].keys():
                if str(neural_trials[trials]['post_trial']) in neural_trials.keys():
                    mega_session_data[trial_id]['post_trial'] = fname[-8:] + '_' + str(neural_trials[trials]['post_trial'])
                else:
                    print('last trial of ' + fname[-8:] + ': ' + str(neural_trials[trials]['post_trial']))
                       
            if 'pre_trial' in neural_trials[trials].keys():
                if str(neural_trials[trials]['pre_trial']) in neural_trials.keys():
                    mega_session_data[trial_id]['pre_trial'] = fname[-8:] + '_' + str(neural_trials[trials]['pre_trial'])
            for key_id in all_keys:
                if key_id == 'dff':
                    mega_session_data[trial_id][key_id] = neural_trials[trials][key_id][good_roi, :]
                else:
                    mega_session_data[trial_id][key_id] = neural_trials[trials][key_id]
                    
            mega_session_data[trial_id]['block_epoch'] = neural_trials[trials]['block_epoch']
            mega_session_data[trial_id]['date'] = fname[-8:]
        session_id = session_id + 1
                
    return mega_session_data