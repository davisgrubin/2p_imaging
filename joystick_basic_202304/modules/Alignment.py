#!/usr/bin/env python3

import numpy as np
from scipy.interpolate import interp1d


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
    js_time = [neural_trials[str(i)]['trial_js_time']
                    for i in range(len(neural_trials))]
    js_pos = [neural_trials[str(i)]['trial_js_pos']
                    for i in range(len(neural_trials))]
    reward = np.array([not np.isnan(neural_trials[str(i)]['trial_reward'][0])
              for i in range(len(neural_trials))])
    punish = np.array([not np.isnan(neural_trials[str(i)]['trial_punish'][0])
              for i in range(len(neural_trials))])
    trial_types = np.array([neural_trials[str(i)]['trial_types']
              for i in range(len(neural_trials))])
    inter_time = []
    inter_pos = []
    for (pos, time) in zip(js_pos, js_time):
        interpolator = interp1d(time, pos, bounds_error=False)
        new_time = np.arange(np.min(time), np.max(time), interval)
        new_pos = interpolator(new_time)
        inter_time.append(new_time)
        inter_pos.append(new_pos)
    if np.size(neural_trials[str(0)][state]) == 1:
        time_state = [
            neural_trials[str(i)][state] - neural_trials[str(i)]['vol_time'][0]
            for i in range(len(neural_trials))]
    if np.size(neural_trials[str(0)][state]) == 2:
        time_state = [
            neural_trials[str(i)][state][0] - neural_trials[str(i)]['vol_time'][0]
            for i in range(len(neural_trials))]
    trial_types = [trial_types[i]
                   for i in range(len(inter_time))
                   if not np.isnan(time_state)[i]]
    reward = np.array([reward[i] * 1
              for i in range(len(inter_time))
              if not np.isnan(time_state)[i]])
    punish = np.array([punish[i] * -1
              for i in range(len(inter_time))
              if not np.isnan(time_state)[i]])
    outcome = reward + punish
    zero_state = [np.argmin(np.abs(inter_time[i] - time_state[i]))
                  for i in range(len(inter_time))
                  if not np.isnan(time_state)[i]]
    align_data = [inter_pos[i]
                  for i in range(len(inter_pos))
                  if not np.isnan(time_state)[i]]
    align_time = [inter_time[i]
                  for i in range(len(inter_time))
                  if not np.isnan(time_state)[i]]
    align_time = [align_time[i] - align_time[i][zero_state[i]]
                  for i in range(len(align_time))]
    if len(align_data) > 0:
        align_data = np.array(trim_seq(align_data, zero_state))
        align_time = np.array(trim_seq(align_time, zero_state))[0,:]
    return [align_data, align_time, trial_types, outcome]


# extract response around given stimulus.
def get_stim_response(
        neural_trials, state,
        l_frames, r_frames
        ):
    # initialize list.
    neu_seq  = []
    neu_time = []
    stim_seq = []
    outcome = []
    # loop over trials.
    for trials in neural_trials.keys():
        # read trial data.
        fluo = neural_trials[trials]['dff']
        time = neural_trials[trials]['time']
        trial_vis = neural_trials[str(trials)][state]
        trial_punish = neural_trials[str(trials)]['trial_punish']
        trial_reward = neural_trials[str(trials)]['trial_reward']
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
                if not np.isnan(trial_punish[0]):
                    trial_outcome = -1
                if not np.isnan(trial_reward[0]):
                    trial_outcome = 1
                outcome.append(trial_outcome)
    if len(neu_seq) > 0:
        neu_seq, neu_time = align_neu_seq_utils(neu_seq, neu_time)
        outcome = np.array(outcome)
        stim_seq = np.median(np.concatenate(stim_seq),axis=0)
    return [neu_seq, neu_time, outcome, stim_seq]


# extract response around outcome.
def get_outcome_response(
        neural_trials, state,
        l_frames, r_frames
        ):
    # initialize list.
    neu_seq  = []
    neu_time = []
    outcome_seq = []
    # loop over trials.
    for trials in neural_trials.keys():
        # read trial data.
        fluo = neural_trials[trials]['dff']
        time = neural_trials[trials]['time']
        time_outcome = neural_trials[str(trials)][state]
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
    neu_seq, neu_time = align_neu_seq_utils(neu_seq, neu_time)
    outcome_seq = np.median(np.concatenate(outcome_seq),axis=0)
    return [neu_seq, neu_time, outcome_seq]


# extract response around given state.
def get_motor_response(
        neural_trials, state,
        l_frames, r_frames):
    # initialize list.
    neu_seq  = []
    neu_time = []
    outcome = []
    # loop over trials.
    for trials in neural_trials.keys():
        # read trial data.
        fluo = neural_trials[trials]['dff']
        time = neural_trials[trials]['time']
        time_state = neural_trials[trials][state]
        trial_punish = neural_trials[str(trials)]['trial_punish']
        trial_reward = neural_trials[str(trials)]['trial_reward']
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
                if not np.isnan(trial_punish[0]):
                    trial_outcome = -1
                if not np.isnan(trial_reward[0]):
                    trial_outcome = 1
                outcome.append(trial_outcome)
    neu_seq, neu_time = align_neu_seq_utils(neu_seq, neu_time)
    outcome = np.array(outcome)
    return [neu_seq, neu_time, outcome]
