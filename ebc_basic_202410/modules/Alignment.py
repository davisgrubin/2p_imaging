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


# extract response around given stimulus.
def get_stim_response(
        neural_trials, state, trial_types,
        l_frames, r_frames):
    # initialize list.
    neu_seq  = []
    neu_time = []
    stim_seq = []

    # loop over trials.
    for trials in neural_trials.keys():
        # read trial data.
        fluo = neural_trials[trials]['dff']
        time = neural_trials[trials]['time']
        trial_vis = neural_trials[trials][state]
        trial_type = neural_trials[trials]['trial_type']
        if trial_type ==trial_types:
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
                    stim_seq.append(trial_vis[1] - trial_vis[0])
                    
                
    if len(neu_seq) > 0:
        neu_seq, neu_time = align_neu_seq_utils(neu_seq, neu_time)
        stim_seq = np.median(stim_seq,axis=0)
        
    else:
        neu_seq = np.array([[[np.nan]]])
        neu_time = np.array([np.nan])
        stim_seq = np.array([np.nan, np.nan])
        
    return [neu_seq, neu_time, stim_seq]




# extract spontaneous response during iti.
def get_iti_response(
        neural_trials,
        l_frames, r_frames):
    # initialize list.
    neu_seq  = []
    neu_time = []
    

    # loop over trials.
    for trials in neural_trials.keys():
        # read trial data.
        fluo = neural_trials[trials]['dff']
        time = neural_trials[trials]['time']
        time_state = neural_trials[trials]['trial_iti']


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
                
    neu_seq, neu_time = align_neu_seq_utils(neu_seq, neu_time)
    
    return [neu_seq, neu_time]