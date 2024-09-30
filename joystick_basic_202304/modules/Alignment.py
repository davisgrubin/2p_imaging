#!/usr/bin/env python3

import numpy as np
from scipy.interpolate import interp1d


# get trial outcomes.
def get_trial_outcome(neural_trials, trials):
    if not np.isnan(neural_trials[trials]['trial_reward'][0]):
        trial_outcome = 0
    elif not np.isnan(neural_trials[trials]['trial_no1stpush'][0]):
        trial_outcome = 1
    elif not np.isnan(neural_trials[trials]['trial_no2ndpush'][0]):
        trial_outcome = 2
    elif not np.isnan(neural_trials[trials]['trial_early2ndpush'][0]):
        trial_outcome = 3
    else:
        trial_outcome = -1
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
        
        
# align coolected sequences.
def align_neu_seq_utils(neu_seq, neu_time):
    # correct neuron time stamps centering at perturbation.
    neu_time_zero = [np.argmin(np.abs(nt)) for nt in neu_time]
    neu_time = trim_seq(neu_time, neu_time_zero)
    neu_seq = trim_seq(neu_seq, neu_time_zero)
    # concatenate results.
    neu_time  = [nt.reshape(1,-1) for nt in neu_time]
    neu_seq   = np.concatenate(neu_seq, axis=0)
    neu_time  = np.concatenate(neu_time, axis=0)
    # get mean time stamps.
    neu_time  = np.mean(neu_time, axis=0)
    return neu_seq, neu_time


# align joystick trajectory at given state.
def get_js_pos(neural_trials, state):
    interval = 1
    js_time = [neural_trials[t]['trial_js_time']
                    for t in neural_trials.keys()]
    js_pos = [neural_trials[t]['trial_js_pos']
                    for t in neural_trials.keys()]
    
    trial_type = np.array([neural_trials[t]['trial_type']
              for t in neural_trials.keys()])
    epoch = np.array([neural_trials[t]['block_epoch']
              for t in neural_trials.keys()])
    outcome = np.array([get_trial_outcome(neural_trials, t)
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
    epoch = np.array([epoch[i]
                   for i in range(len(inter_time))
                   if not np.isnan(time_state)[i]])
    outcome = np.array([outcome[i]
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
    return [align_data, align_time, trial_type, epoch, outcome]


# extract response around given stimulus.
def get_stim_response(
        neural_trials, state,
        l_frames, r_frames):
    # initialize list.
    neu_seq  = []
    neu_time = []
    stim_seq = []
    delay = []
    epoch = []
    outcome = []
    delay = []
    # loop over trials.
    for trials in neural_trials.keys():
        # read trial data.
        fluo = neural_trials[trials]['dff']
        time = neural_trials[trials]['time']
        trial_vis = neural_trials[trials][state]
        # trial_delay = neural_trials[trials]['trial_delay']
        trial_type = neural_trials[trials]['trial_type']
        trial_epoch = neural_trials[trials]['block_epoch']
        trial_outcome = get_trial_outcome(neural_trials, trials)
        
        if not np.isnan(trial_vis[0]):
            idx = np.argmin(np.abs(time - trial_vis[0]))
            if idx > l_frames and idx < len(time)-r_frames:
                # signal response.
                f = fluo[:, idx-l_frames : idx+r_frames]
                f = np.expand_dims(f, axis=0)
                neu_seq.append(f)
                # signal time stamps.
                t = time[idx-l_frames : idx+r_frames] - time[idx]
                neu_time.append(t)
                # visual stimulus timestamps.
                stim_seq.append(np.array(trial_vis).reshape(1,-1) - trial_vis[0])
                # outcome.
                outcome.append(trial_outcome)
                # delay.
                # delay.append(trial_delay)
                delay.append(trial_type)
                epoch.append(trial_epoch)
                
    if len(neu_seq) > 0:
        neu_seq, neu_time = align_neu_seq_utils(neu_seq, neu_time)
        outcome = np.array(outcome)
        stim_seq = np.median(np.concatenate(stim_seq),axis=0)
        delay = np.array(delay)
        epoch = np.array(epoch)
    else:
        neu_seq = np.array([[[np.nan]]])
        neu_time = np.array([np.nan])
        outcome = np.array([np.nan])
        stim_seq = np.array([np.nan, np.nan])
        delay = np.array(np.nan)
        epoch = np.array(np.nan)
    return [neu_seq, neu_time, outcome, stim_seq, delay, epoch]


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
def get_motor_response(
        neural_trials, state,
        l_frames, r_frames):
    # initialize list.
    neu_seq  = []
    neu_time = []
    delay = []
    epoch = []
    outcome = []
    delay = []
    # loop over trials.
    for trials in neural_trials.keys():
        # read trial data.
        fluo = neural_trials[trials]['dff']
        time = neural_trials[trials]['time']
        time_state = neural_trials[trials][state]
        # trial_delay = neural_trials[trials]['trial_delay']
        trial_type = neural_trials[trials]['trial_type']
        trial_epoch = neural_trials[trials]['block_epoch']
        trial_outcome = get_trial_outcome(neural_trials, trials)
        if np.isnan(np.sum(time_state)):
            continue
        for i in range(np.size(time_state)):
            idx = np.argmin(np.abs(time - time_state[i]))
            if idx > l_frames and idx < len(time)-r_frames:
                # signal response.
                f = fluo[:, idx-l_frames : idx+r_frames]
                f = np.expand_dims(f, axis=0)
                neu_seq.append(f)
                # signal time stamps.
                t = time[idx-l_frames : idx+r_frames] - time[idx]
                neu_time.append(t)
                # outcome.
                outcome.append(trial_outcome)
                # delay.
                delay.append(trial_type)
                epoch.append(trial_epoch)
    neu_seq, neu_time = align_neu_seq_utils(neu_seq, neu_time)
    outcome = np.array(outcome)
    delay = np.array(delay)
    epoch = np.array(epoch)
    return [neu_seq, neu_time, outcome, delay, epoch]


# extract licking response.
def get_lick_response(
        neural_trials,
        l_frames, r_frames):
    # initialize list.
    neu_seq  = []
    neu_time = []
    lick_label = []
    # loop over trials.
    for trials in neural_trials.keys():
        # read trial data.
        fluo = neural_trials[trials]['dff']
        time = neural_trials[trials]['time']
        lick = neural_trials[trials]['trial_lick']
        for i in range(lick.shape[1]):
            idx = np.argmin(np.abs(time - lick[0,i]))
            if idx > l_frames and idx < len(time)-r_frames:
                # signal response.
                f = fluo[:, idx-l_frames : idx+r_frames]
                f = np.expand_dims(f, axis=0)
                neu_seq.append(f)
                # signal time stamps.
                t = time[idx-l_frames : idx+r_frames] - time[idx]
                neu_time.append(t)
                # label.
                lick_label.append(lick[1,i])
    neu_seq, neu_time = align_neu_seq_utils(neu_seq, neu_time)
    lick_label = np.array(lick_label)
    return [neu_seq, neu_time, lick_label]


# extract spontaneous response during iti.
def get_iti_response(
        neural_trials,
        l_frames, r_frames):
    # initialize list.
    neu_seq  = []
    neu_time = []
    delay = []
    epoch = []
    outcome = []

    # loop over trials.
    for trials in neural_trials.keys():
        # read trial data.
        fluo = neural_trials[trials]['dff']
        time = neural_trials[trials]['time']
        time_state = neural_trials[trials]['trial_iti']

        # trial_delay = neural_trials[trials]['trial_delay']

        trial_type = neural_trials[trials]['trial_type']
        trial_epoch = neural_trials[trials]['block_epoch']
        trial_outcome = get_trial_outcome(neural_trials, trials)

        if np.isnan(np.sum(time_state)):
            continue
        for i in range(np.size(time_state)):
            idx = np.argmin(np.abs(time - time_state[i]))
            if idx > l_frames and idx < len(time)-r_frames:
                # signal response.
                f = fluo[:, idx-l_frames : idx+r_frames]
                f = np.expand_dims(f, axis=0)
                neu_seq.append(f)
                # signal time stamps.
                t = time[idx-l_frames : idx+r_frames] - time[idx]
                neu_time.append(t)
                # label.
                outcome.append(trial_outcome)
                # delay.
                delay.append(trial_type)
                epoch.append(trial_epoch)
    neu_seq, neu_time = align_neu_seq_utils(neu_seq, neu_time)
    outcome = np.array(outcome)
    delay = np.array(delay)
    epoch = np.array(epoch)
    return [neu_seq, neu_time, outcome, delay, epoch]