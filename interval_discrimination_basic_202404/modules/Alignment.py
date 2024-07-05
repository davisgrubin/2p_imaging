#!/usr/bin/env python3

import numpy as np

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

# extract response around stimulus with outcome.
def get_stim_response(
        neural_trials,
        l_frames, r_frames
        ):
    # initialize list.
    neu_seq  = []
    neu_time = []
    stim_seq = []
    trial_types = []
    trial_isi_pre = []
    trial_isi_post = []
    outcome = []
    stim_idx = []
    # loop over trials.
    for trials in neural_trials.keys():
        # read trial data.
        fluo = neural_trials[trials]['dff']
        time = neural_trials[trials]['time']
        time_stim_seq = neural_trials[trials]['trial_stim_seq']
        trial_punish = neural_trials[str(trials)]['trial_punish']
        trial_reward = neural_trials[str(trials)]['trial_reward']
        if not np.isnan(time_stim_seq[0,0]):
            for i in range(time_stim_seq.shape[1]):
                idx = np.argmin(np.abs(time - time_stim_seq[0,i]))
                if idx > l_frames and idx < len(time)-r_frames:
                    # signal response.
                    f = fluo[:, idx-l_frames : idx+r_frames]
                    f = np.expand_dims(f, axis=0)
                    neu_seq.append(f)
                    # signal time stamps.
                    t = time[idx-l_frames : idx+r_frames] - time[idx]
                    neu_time.append(t)
                    # stim timestamps.
                    stim_seq.append(np.array([0, time_stim_seq[1,i]-time_stim_seq[0,i]]))
                    # trial type.
                    trial_types.append(neural_trials[trials]['trial_types'])
                    # pre perturbation isi.
                    trial_isi_pre.append(np.mean(time_stim_seq[0,1:3] - time_stim_seq[1,0:2]))
                    # post perturbation isi.
                    if time_stim_seq.shape[1]<4:
                        stim_isi = np.nan
                    else:
                        stim_isi = np.mean(time_stim_seq[0,3:] - time_stim_seq[1,2:-1])
                    trial_isi_post.append(stim_isi)
                    # outcome.
                    if not np.isnan(trial_punish[0]):
                        trial_outcome = -1
                    elif not np.isnan(trial_reward[0]):
                        trial_outcome = 1
                    else:
                        trial_outcome = 0
                    outcome.append(trial_outcome)
                    # stimulus id.
                    stim_idx.append(i)
    neu_seq, neu_time = align_neu_seq_utils(neu_seq, neu_time)
    stim_seq = np.mean(stim_seq, axis=0)
    trial_types = np.array(trial_types).reshape(-1)
    trial_isi_pre = np.array(trial_isi_pre).reshape(-1)
    trial_isi_post = np.array(trial_isi_post).reshape(-1)
    stim_idx = np.array(stim_idx).reshape(-1)
    return [neu_seq, neu_time, stim_seq, trial_types, trial_isi_pre, trial_isi_post, outcome, stim_idx]

# get subset of stimulus response given stimulus type.
def get_stim_response_mode(neu_cate, trial_types, trial_isi, stim_idx, mode):
    if mode == 'onset':
        idx = stim_idx==0
    if mode == 'pre_all':
        idx = stim_idx<=2
    if mode == 'post_first':
        idx = stim_idx==3
    if mode == 'pre_last':
        idx = stim_idx==2
    if mode == 'post_all':
        idx = stim_idx>=3
    neu_cate = neu_cate[idx,:,:]
    trial_types = trial_types[idx]
    trial_isi = trial_isi[idx]
    return neu_cate, trial_types, trial_isi

# extract response around all licking.
def get_lick_response(
        neural_trials, lick_state,
        l_frames, r_frames):
    # initialize list.
    neu_seq_lick  = []
    neu_time_lick = []
    lick_direc = []
    # loop over trials.
    for trials in neural_trials.keys():
        # read trial data.
        fluo = neural_trials[trials]['dff']
        time = neural_trials[trials]['time']
        time_licking = neural_trials[trials][lick_state]
        for lick_idx in range(time_licking.shape[1]):
            if not np.isnan(time_licking[0,lick_idx]):
                idx = np.argmin(np.abs(time - time_licking[0,lick_idx]))
                if idx > l_frames and idx < len(time)-r_frames:
                    # signal response.
                    f = fluo[:, idx-l_frames : idx+r_frames]
                    f = np.expand_dims(f, axis=0)
                    neu_seq_lick.append(f)
                    # signal time stamps.
                    t = time[idx-l_frames : idx+r_frames] - time[idx]
                    neu_time_lick.append(t)
                    # licking direction 0-left 1-right.
                    lick_direc.append(time_licking[1,lick_idx])
    neu_seq_lick, neu_time_lick = align_neu_seq_utils(neu_seq_lick, neu_time_lick)
    lick_direc = np.array(lick_direc)
    return [neu_seq_lick, neu_time_lick, lick_direc]

# extract response around outcome.
def get_outcome_response(
        neural_trials, state,
        l_frames, r_frames
        ):
    # initialize list.
    neu_seq  = []
    neu_time = []
    outcome_seq = []
    lick_direc = []
    # loop over trials.
    for trials in neural_trials.keys():
        # read trial data.
        fluo = neural_trials[trials]['dff']
        time = neural_trials[trials]['time']
        time_outcome = neural_trials[str(trials)][state]
        time_decision = neural_trials[trials]['trial_decision']
        # compute stimulus start point in ms.
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
                # get outcome timestamps
                outcome_seq.append(time_outcome - time_outcome[0])
                # licking direction 0-left 1-right.
                lick_direc.append(time_decision[1,0])
    neu_seq, neu_time = align_neu_seq_utils(neu_seq, neu_time)
    outcome_seq = np.mean(outcome_seq, axis=0)
    lick_direc = np.array(lick_direc)
    return [neu_seq, neu_time, outcome_seq, lick_direc]
