# -*- coding: utf-8 -*-
"""
Created on Mon May 26 09:47:58 2025

@author: saminnaji3
"""

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
    partition = 4
    # resort delay settings.
    start = np.min(np.array([t for t in neural_trials.keys()]).astype('int32'))
    end   = np.max(np.array([t for t in neural_trials.keys()]).astype('int32'))
    trial_idx = np.arange(start, end+1)
    trial_delay = np.array([neural_trials[str(t)]['trial_delay'] for t in trial_idx])
    # mark short and long delay trials short:0 long:1.
    trial_type = np.zeros_like(trial_delay)
    trial_type[trial_delay>300] = 1
    # mark epoch trials unvalid:-1 early:1 late:0
    block_change = np.diff(trial_type, prepend=0)
    block_change[block_change!=0] = 1
    block_change[0] = 1
    block_change[-1] = 1
    block_change = np.where(block_change==1)[0]
    #print(block_change)
    block_epoch = np.zeros_like(trial_type)
    for start, end in zip(block_change[:-1], block_change[1:]):
        #tran = start + (end - start) // partition
        tran = start + 5
        tran2 = end-5
        block_epoch[start:tran] = 1
        block_epoch[tran2:end] = 2
    block_epoch[:block_change[1]] = -1
    # write into neural trials.
    for i in range(len(trial_idx)):
        neural_trials[str(trial_idx[i])]['trial_type'] = trial_type[i]
        neural_trials[str(trial_idx[i])]['block_epoch'] = block_epoch[i]
    return neural_trials


def get_trial_outcome(neural_trials, trials):
    if not np.isnan(neural_trials[trials]['trial_reward'][0]):
        trial_outcome = 0
    elif not np.isnan(neural_trials[trials]['trial_no1stpush'][0]):
        trial_outcome = 1
    elif not np.isnan(neural_trials[trials]['trial_no2ndpush'][0])and (np.isnan(neural_trials[trials]['trial_push2'][0])):
        trial_outcome = 2
    elif (not np.isnan(neural_trials[trials]['trial_no2ndpush'][0])) and (not np.isnan(neural_trials[trials]['trial_push2'][0])):
        #print('late_press')
        trial_outcome = 3
    elif not np.isnan(neural_trials[trials]['trial_early2ndpush'][0]):
        trial_outcome = 4
    else:
        trial_outcome = -1
    return trial_outcome


def get_data(file_names, state,l_frames = 0, r_frames = 30,end_align = 0, trailization_on = 0):
    
    
    mega_session_data = dict()
    file_names.sort(key=lambda x: x[-8:])
    session_id = 0
    
    state_info = []
    data_date = []
    outcome = []
    next_outcome = []
    type_label = []
    
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
        
        state_info_sess = []
        data_date.append(fname[-8:])
        outcome_sess = []
        next_outcome_sess = []
        type_sess = []
        good_trials = 0
        all_trials = 0
        print(fname[-8:])
        for trials in neural_trials.keys():
            #print(trials)
            all_trials = all_trials + 1
            # read trial data.
            fluo = neural_trials[trials]['dff']
            #print(fluo.shape)
            time = neural_trials[trials]['time']
            trial_vis = neural_trials[trials][state]
            # trial_delay = neural_trials[trials]['trial_delay']
            trial_type = neural_trials[trials]['trial_types']
            block_epoch = neural_trials[trials]['block_epoch']
            #trial_epoch = neural_trials[trials]['block_epoch']
            trial_outcome = get_trial_outcome(neural_trials, trials)
            if 'post_trial' in neural_trials[trials].keys():
                new_trials = str(neural_trials[trials]['post_trial'])
                if new_trials in neural_trials.keys():
                    next_trial_outcome = get_trial_outcome(neural_trials, new_trials)
                else:
                    next_trial_outcome = -1
            else:
                next_trial_outcome = -1
            
            if not np.isnan(trial_vis[end_align]):
                idx = np.argmin(np.abs(time - trial_vis[end_align]))
                if idx > l_frames and idx < len(time)-r_frames:
                    good_trials = good_trials + 1
                    #print(1)
                    # signal response.
                    f = fluo[good_roi, idx-l_frames : idx+r_frames]
                    state_info_sess.append(np.nanmean(f, axis=1))
                    outcome_sess.append(trial_outcome)
                    #post_outcome.append(trial_post_outcome)
                    type_sess.append(trial_type)
                    next_outcome_sess.append(next_trial_outcome)
            
            
            
        state_info.append(state_info_sess)
        outcome.append(outcome_sess)
        type_label.append(type_sess)
        next_outcome.append(next_outcome_sess)
        #print('good_trials', good_trials)
        #print('all_trials', all_trials)
            
        session_id = session_id + 1
                
    return state_info, outcome,type_label,next_outcome, data_date

def get_data2(file_names, state,state_end, trailization_on = 0):
    
    end_align = 0
    mega_session_data = dict()
    file_names.sort(key=lambda x: x[-8:])
    session_id = 0
    
    state_info = []
    data_date = []
    outcome = []
    type_label = []
    next_outcome = []
    
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
        
        state_info_sess = []
        data_date.append(fname[-8:])
        outcome_sess = []
        type_sess = []
        next_outcome_sess = []
        good_trials = 0
        all_trials = 0
        print(fname[-8:])
        for trials in neural_trials.keys():
            
            all_trials = all_trials + 1
            # read trial data.
            fluo = neural_trials[trials]['dff']
            #print(fluo.shape)
            time = neural_trials[trials]['time']
            trial_vis = neural_trials[trials][state]
            trial_vis_end = neural_trials[trials][state_end]
            # trial_delay = neural_trials[trials]['trial_delay']
            trial_type = neural_trials[trials]['trial_types']
            #trial_epoch = neural_trials[trials]['block_epoch']
            trial_outcome = get_trial_outcome(neural_trials, trials)
            if 'post_trial' in neural_trials[trials].keys():
                new_trials = str(neural_trials[trials]['post_trial'])
                if new_trials in neural_trials.keys():
                    next_trial_outcome = get_trial_outcome(neural_trials, new_trials)
                else:
                    next_trial_outcome = -1
            else:
                next_trial_outcome = -1
            
            
            if (not np.isnan(trial_vis[0])) and (not np.isnan(trial_vis_end[0])):
                idx = np.argmin(np.abs(time - trial_vis[0]))
                idx_end = np.argmin(np.abs(time - trial_vis_end[0]))
                if idx <idx_end:
                    good_trials = good_trials + 1
                    #print(1)
                    # signal response.
                    f = fluo[good_roi, idx : idx_end]
                    state_info_sess.append(np.nanmean(f, axis=1))
                    outcome_sess.append(trial_outcome)
                    #post_outcome.append(trial_post_outcome)
                    type_sess.append(trial_type)
                    next_outcome_sess.append(next_trial_outcome)
            
            
            
            
        state_info.append(state_info_sess)
        outcome.append(outcome_sess)
        type_label.append(type_sess)
        next_outcome.append(next_outcome_sess)
        #print('good_trials', good_trials)
        #print('all_trials', all_trials)
            
        session_id = session_id + 1
                
    return state_info, outcome,type_label,next_outcome, data_date


def get_data_epoch(file_names, state,l_frames = 0, r_frames = 30,end_align = 0, trailization_on = 0):
    
    
    mega_session_data = dict()
    file_names.sort(key=lambda x: x[-8:])
    session_id = 0
    
    first_state_info = []
    data_date = []
    first_outcome = []
    first_next_outcome = []
    first_type_label = []
    
    rest_state_info = []
    rest_outcome = []
    rest_next_outcome = []
    rest_type_label = []
    print('processing')
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
        
        data_date.append(fname[-8:])
        state_info_sess1 = []
        outcome_sess1 = []
        next_outcome_sess1 = []
        type_sess1 = []
        state_info_sess2 = []
        outcome_sess2 = []
        next_outcome_sess2 = []
        type_sess2 = []
        
        good_trials = 0
        all_trials = 0
        #print(fname[-8:])
        for trials in neural_trials.keys():
            #print(trials)
            all_trials = all_trials + 1
            # read trial data.
            fluo = neural_trials[trials]['dff']
            #print(fluo.shape)
            time = neural_trials[trials]['time']
            trial_vis = neural_trials[trials][state]
            # trial_delay = neural_trials[trials]['trial_delay']
            trial_type = neural_trials[trials]['trial_types']
            block_epoch = neural_trials[trials]['block_epoch']
            #trial_epoch = neural_trials[trials]['block_epoch']
            trial_outcome = get_trial_outcome(neural_trials, trials)
            if 'post_trial' in neural_trials[trials].keys():
                new_trials = str(neural_trials[trials]['post_trial'])
                if new_trials in neural_trials.keys():
                    next_trial_outcome = get_trial_outcome(neural_trials, new_trials)
                else:
                    next_trial_outcome = -1
            else:
                next_trial_outcome = -1
            
            if not np.isnan(trial_vis[end_align]):
                idx = np.argmin(np.abs(time - trial_vis[end_align]))
                if idx > l_frames and idx < len(time)-r_frames:
                    good_trials = good_trials + 1
                    #print(1)
                    # signal response.
                    f = fluo[good_roi, idx-l_frames : idx+r_frames]
                    if block_epoch == 2:
                        state_info_sess2.append(np.nanmean(f, axis=1))
                        outcome_sess2.append(trial_outcome)
                        #post_outcome.append(trial_post_outcome)
                        type_sess2.append(trial_type)
                        next_outcome_sess2.append(next_trial_outcome)
                    elif block_epoch == 1:
                        state_info_sess1.append(np.nanmean(f, axis=1))
                        outcome_sess1.append(trial_outcome)
                        #post_outcome.append(trial_post_outcome)
                        type_sess1.append(trial_type)
                        next_outcome_sess1.append(next_trial_outcome)
            
            
            
        first_state_info.append(state_info_sess1)
        first_outcome.append(outcome_sess1)
        first_type_label.append(type_sess1)
        first_next_outcome.append(next_outcome_sess1)
        
        rest_state_info.append(state_info_sess2)
        rest_outcome.append(outcome_sess2)
        rest_type_label.append(type_sess2)
        rest_next_outcome.append(next_outcome_sess2)
        #print('good_trials', good_trials)
        #print('all_trials', all_trials)
            
        session_id = session_id + 1
                
    return first_state_info, first_outcome, first_type_label, first_next_outcome, rest_state_info, rest_outcome, rest_type_label, rest_next_outcome, data_date