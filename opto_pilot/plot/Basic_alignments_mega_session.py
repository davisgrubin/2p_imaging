# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 11:18:12 2025

@author: saminnaji3
"""
import numpy as np
from scipy.interpolate import interp1d


# get trial outcomes.
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

def get_post_trial_outcome(neural_trials, trials):
    if 'post_trial' in neural_trials[trials].keys():
        new_trial = neural_trials[trials]['post_trial']
        if not np.isnan(neural_trials[new_trial]['trial_reward'][0]):
            trial_outcome = 0
        elif not np.isnan(neural_trials[new_trial]['trial_no1stpush'][0]):
            trial_outcome = 1
        elif not np.isnan(neural_trials[new_trial]['trial_no2ndpush'][0])and (np.isnan(neural_trials[trials]['trial_push2'][0])):
            trial_outcome = 2
        elif (not np.isnan(neural_trials[new_trial]['trial_no2ndpush'][0])) and (not np.isnan(neural_trials[trials]['trial_push2'][0])):
            #print('late_press')
            trial_outcome = 3
        elif not np.isnan(neural_trials[new_trial]['trial_early2ndpush'][0]):
            trial_outcome = 4
        else:
            trial_outcome = -1
    else:
        trial_outcome = np.nan
    return trial_outcome


def get_trial_delay_dt(neural_trials, trials):
    if 'post_trial' in neural_trials[trials].keys():
        new_trial = neural_trials[trials]['post_trial']
        if not np.isnan(neural_trials[trials]['trial_press_delay'][0]):
            if not np.isnan(neural_trials[new_trial]['trial_press_delay'][0]):
                if neural_trials[trials]['trial_press_delay'][0] > 0 and neural_trials[new_trial]['trial_press_delay'][0] > 0:
                    change_delay = neural_trials[new_trial]['trial_press_delay'][0] - neural_trials[trials]['trial_press_delay'][0]
                    if change_delay < -120:
                        trial_outcome = -1
                    elif change_delay > 120:
                        trial_outcome = 1
                    else: 
                        trial_outcome = 0
                else:
                    trial_outcome = np.nan
            else:
                trial_outcome = np.nan
        else:
            trial_outcome = np.nan
    else:
        trial_outcome = np.nan
    return trial_outcome
# cut sequence into the same length as the shortest one given pivots.
def trim_seq(
        data,
        pivots,
        ):
    if len(data[0].shape) == 1:
        len_l_min = np.min(pivots)
        len_r_min = np.min([len(data[i])-pivots[i] for i in range(len(data))])
        data = [data[i][pivots[i]-len_l_min:pivots[i]+len_r_min]
                for i in range(len(data))]
    if len(data[0].shape) == 3:
        len_l_min = np.min(pivots)
        len_r_min = np.min([len(data[i][0,0,:])-pivots[i] for i in range(len(data))])
        #print(np.max(pivots-len_l_min),len_r_min)
        data = [data[i][:, :, pivots[i]-len_l_min:pivots[i]+len_r_min]
                for i in range(len(data))]
    return data


# pad sequence with time stamps to the longest length with nan.
def pad_seq(align_data, align_time):
    pad_time = np.arange(
        np.min([np.min(t) for t in align_time]),
        np.max([np.max(t) for t in align_time]) + 1)
    pad_data = []
    for data, time in zip(align_data, align_time):
        aligned_seq = np.full_like(pad_time, np.nan, dtype=float)
        idx = np.searchsorted(pad_time, time)
        aligned_seq[idx] = data
        pad_data.append(aligned_seq)
    return pad_data, pad_time
        
        
# align collected sequences.
def align_neu_seq_utils(neu_seq, neu_time):
    # correct neuron time stamps centering at perturbation.
    #print(len(neu_seq), neu_seq[10].shape)
    neu_time_zero = [np.argmin(np.abs(nt)) for nt in neu_time]
    #print(len(neu_time_zero))
    neu_time = trim_seq(neu_time, neu_time_zero)
    neu_seq = trim_seq(neu_seq, neu_time_zero)
    #print(len(neu_seq), neu_seq[10].shape)
    #print(np.array(neu_seq).shape)
    # concatenate results.
    neu_time  = [nt.reshape(1,-1) for nt in neu_time]
    #neu_seq   = np.concatenate(neu_seq, axis=0)
    neu_time  = np.concatenate(neu_time, axis=0)
    # get mean time stamps.
    neu_time  = np.mean(neu_time, axis=0)
    return neu_seq, neu_time

# making a Number_of_ROIs*time_step array for neu_seq
def mega_neu_seq(neu_seq, session_id, outcome, post_outcome, delay_dt, delay):
    mega_neu_seq = []
    ROI_id = []
    outcome_label = []
    post_outcome_label = []
    delay_dt_label = []
    delay_label = []
    stack_ROI = 0
    for trial in range(len(neu_seq)):
        mega_neu_seq.append(np.squeeze(neu_seq[trial], axis=0))
        ROI_id.append(np.arange(0, neu_seq[trial].shape[1])+stack_ROI)
        outcome_label.append(np.ones(neu_seq[trial].shape[1])*outcome[trial])
        post_outcome_label.append(np.ones(neu_seq[trial].shape[1])*post_outcome[trial])
        delay_dt_label.append(np.ones(neu_seq[trial].shape[1])*delay_dt[trial])
        delay_label.append(np.ones(neu_seq[trial].shape[1])*delay[trial])
        if trial < len(neu_seq) - 1:
            if not session_id[trial] == session_id[trial+1]:
                #print(session_id[trial] , session_id[trial+1] , neu_seq[trial-1].shape[1], neu_seq[trial].shape[1])
                stack_ROI = stack_ROI + neu_seq[trial].shape[1]
                #print(stack_ROI)
    result_array = np.vstack(mega_neu_seq)
    ROI_id = np.concatenate(ROI_id)
    outcome_label = np.concatenate(outcome_label)
    post_outcome_label = np.concatenate(post_outcome_label)
    delay_dt_label = np.concatenate(delay_dt_label)
    delay_label = np.concatenate(delay_label)
    return result_array, ROI_id, outcome_label, post_outcome_label,delay_dt_label, delay_label

# align joystick trajectory at given state.
def get_js_pos(neural_trials, state):
    interval = 1
    js_time = [neural_trials[t]['trial_js_time']
                    for t in neural_trials.keys()]
    js_pos = [neural_trials[t]['trial_js_pos']
                    for t in neural_trials.keys()]
    
    trial_type = np.array([neural_trials[t]['trial_types']
              for t in neural_trials.keys()])
    epoch = []
    # epoch = np.array([neural_trials[t]['block_epoch']
    #           for t in neural_trials.keys()])
    outcome = np.array([get_trial_outcome(neural_trials, t)
               for t in neural_trials.keys()])
    delay_dt = np.array([get_trial_delay_dt(neural_trials, t)
               for t in neural_trials.keys()])
    inter_time = []
    inter_pos = []
    for (pos, time) in zip(js_pos, js_time):
        interpolator = interp1d(time, pos, bounds_error=False)
        new_time = np.arange(np.min(time), np.max(time), interval)
        new_pos = interpolator(new_time)
        inter_time.append(new_time)
        inter_pos.append(new_pos)
    if np.size(neural_trials[next(iter(neural_trials))][state]) == 1:
        time_state = [
            neural_trials[t][state] - neural_trials[t]['vol_time'][0]
            for t in neural_trials.keys()]
    if np.size(neural_trials[next(iter(neural_trials))][state]) == 2:
        time_state = [
            neural_trials[t][state][0] - neural_trials[t]['vol_time'][0]
            for t in neural_trials.keys()]

    trial_type = np.array([trial_type[i]
                   for i in range(len(inter_time))
                   if not np.isnan(time_state)[i]])
    # epoch = np.array([epoch[i]
    #                for i in range(len(inter_time))
    #                if not np.isnan(time_state)[i]])
    outcome = np.array([outcome[i]
                   for i in range(len(inter_time))
                   if not np.isnan(time_state)[i]])
    
    delay_dt = np.array([delay_dt[i]
                   for i in range(len(inter_time))
                   if not np.isnan(time_state)[i]])

    zero_state = [np.argmin(np.abs(inter_time[i] - time_state[i]))
                  for i in range(len(inter_time))
                  if not np.isnan(time_state)[i]]
    align_data = [inter_pos[i] - inter_pos[i][0]
                  for i in range(len(inter_pos))
                  if not np.isnan(time_state)[i]]
    align_time = [inter_time[i]
                  for i in range(len(inter_time))
                  if not np.isnan(time_state)[i]]
    align_time = [align_time[i] - align_time[i][zero_state[i]]
                  for i in range(len(align_time))]
    if len(align_data) > 0:
        align_data, align_time = pad_seq(align_data, align_time)
        align_data = np.array(align_data)
    else:
        align_data = np.array([[np.nan]])
        align_time = np.array([np.nan])
        trial_type = np.array([np.nan])
        epoch = np.array([np.nan])
        outcome = np.array([np.nan])
        delay_dt = np.array([np.nan])
    return [align_data, align_time, trial_type, epoch, outcome, delay_dt]


# extract response around given stimulus.
def get_stim_response(neural_trials, state,
        l_frames, r_frames, end_align = 0):
    # initialize list.
    neu_seq  = []
    session_id = []
    neu_time = []
    stim_seq = []
    delay = []
    epoch = []
    outcome = []
    post_outcome = []
    delay_dt = []
    delay = []
    # loop over trials.
    for trials in neural_trials.keys():
        # read trial data.
        fluo = neural_trials[trials]['dff']
        #print(fluo.shape)
        time = neural_trials[trials]['time']
        trial_vis = neural_trials[trials][state]
        # trial_delay = neural_trials[trials]['trial_delay']
        trial_type = neural_trials[trials]['trial_types']
        #trial_epoch = neural_trials[trials]['block_epoch']
        trial_outcome = get_trial_outcome(neural_trials, trials)
        trial_post_outcome = get_post_trial_outcome(neural_trials, trials)
        trial_delay_dt = get_trial_delay_dt(neural_trials, trials)
        
        
        if not np.isnan(trial_vis[end_align]):
            idx = np.argmin(np.abs(time - trial_vis[end_align]))
            if idx > l_frames and idx < len(time)-r_frames:
                #print(1)
                # signal response.
                f = fluo[:, idx-l_frames : idx+r_frames]
                f = np.expand_dims(f, axis=0)
                neu_seq.append(f)
                # signal time stamps.
                t = time[idx-l_frames : idx+r_frames] - time[idx]
                neu_time.append(t)
                # visual stimulus timestamps.
                stim_seq.append(np.array(trial_vis).reshape(1,-1) - trial_vis[end_align])
                # outcome.
                outcome.append(trial_outcome)
                post_outcome.append(trial_post_outcome)
                delay_dt.append(trial_delay_dt)
                # delay.
                # delay.append(trial_delay)
                delay.append(trial_type)
                #epoch.append(trial_epoch)
                session_id.append(neural_trials[trials]['session_id'])
                
    if len(neu_seq) > 0:
        neu_seq, neu_time = align_neu_seq_utils(neu_seq, neu_time)
        outcome = np.array(outcome)
        post_outcome = np.array(post_outcome)
        delay_dt = np.array(delay_dt)
        stim_seq = np.median(np.concatenate(stim_seq),axis=0)
        [neu_seq, ROI_id, outcome_label, post_outcome_label, delay_dt_label, delay_label] = mega_neu_seq(neu_seq, session_id, outcome, post_outcome,delay_dt, delay)
        delay = np.array(delay)
        epoch = np.array(epoch)
    else:
        neu_seq = np.array([[[np.nan]]])
        neu_time = np.array([np.nan])
        outcome = np.array([np.nan])
        post_outcome = np.array([np.nan])
        delay_dt = np.array([np.nan])
        stim_seq = np.array([np.nan, np.nan])
        delay = np.array(np.nan)
        epoch = np.array(np.nan)
    return [neu_seq, neu_time, ROI_id, outcome_label, post_outcome_label,delay_dt_label, outcome, post_outcome, delay_dt, stim_seq, delay, delay_label, epoch]


# extract response around outcome.
def get_outcome_response(
        neural_trials, state,
        l_frames, r_frames
        ):
    # initialize list.
    neu_seq  = []
    neu_time = []
    outcome_seq = []
    delay = []
    epoch = []
    outcome = []
    # loop over trials.
    for trials in neural_trials.keys():
        # read trial data.
        fluo = neural_trials[trials]['dff']
        time = neural_trials[trials]['time']
        time_outcome = neural_trials[trials][state]

        # trial_delay = neural_trials[trials]['trial_delay']

        trial_type = neural_trials[trials]['trial_type']
        trial_epoch = neural_trials[trials]['block_epoch']
        trial_outcome = get_trial_outcome(neural_trials, trials)
        
        if not np.isnan(time_outcome[0]):
            idx = np.argmin(np.abs(time - time_outcome[0]))
            if idx > l_frames and idx < len(time)-r_frames:
                # signal response.
                f = fluo[:, idx-l_frames : idx+r_frames]
                f = np.expand_dims(f, axis=0)
                neu_seq.append(f)
                # signal time stamps.
                t = time[idx-l_frames : idx+r_frames] - time[idx]
                neu_time.append(t)
                # outcome timestamps.
                outcome_seq.append(np.array(time_outcome).reshape(1,-1) - time_outcome[0])
                # label.
                outcome.append(trial_outcome)
                # delay.
                # delay.append(trial_delay)

                delay.append(trial_type)
                epoch.append(trial_epoch)

    if len(neu_seq) > 0:
        neu_seq, neu_time = align_neu_seq_utils(neu_seq, neu_time)
        outcome = np.array(outcome)
        outcome_seq = np.median(np.concatenate(outcome_seq),axis=0)
        delay = np.array(delay)


        epoch = np.array(epoch)

    else:
        neu_seq = np.array([[[np.nan]]])
        neu_time = np.array([np.nan])
        outcome = np.array([np.nan])
        outcome_seq = np.array([np.nan, np.nan])
        delay = np.array(np.nan)
        epoch = np.array(np.nan)
        
    return [neu_seq, neu_time, outcome_seq, outcome, delay, epoch]

# extract pushing response around given state.