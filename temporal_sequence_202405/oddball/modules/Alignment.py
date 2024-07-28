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


# extract response around stimulus.
def get_stim_response(
        neural_trials,
        l_frames, r_frames,
        ):
    stim_labels = neural_trials['stim_labels']
    dff = neural_trials['dff']
    time = neural_trials['time']
    vol_stim = neural_trials['vol_stim']
    vol_time = neural_trials['vol_time']
    # initialize list.
    neu_seq  = []
    neu_time = []
    stim_seq  = []
    stim_value = []
    stim_time = []
    # loop over stimulus.
    for stim_id in range(1, stim_labels.shape[0]-1):
        idx = np.argmin(np.abs(time - stim_labels[stim_id,0]))
        if idx > l_frames and idx < len(time)-r_frames:
            # signal response.
            f = dff[:, idx-l_frames : idx+r_frames]
            f = np.expand_dims(f, axis=0)
            neu_seq.append(f)
            # signal time stamps.
            t = time[idx-l_frames : idx+r_frames] - time[idx]
            neu_time.append(t)
            # voltage.
            vol_t_c = np.argmin(np.abs(vol_time - time[idx]))
            vol_t_l = np.argmin(np.abs(vol_time - time[idx-l_frames]))
            vol_t_r = np.argmin(np.abs(vol_time - time[idx+r_frames]))
            stim_value.append(vol_stim[vol_t_l:vol_t_r])
            stim_time.append(vol_time[vol_t_l:vol_t_r] - vol_time[vol_t_c])
            # stimulus.
            stim_seq.append(np.array(
                [[stim_labels[stim_id-1,0]-stim_labels[stim_id,0],
                 stim_labels[stim_id-1,1]-stim_labels[stim_id,0]],
                 [0,
                  stim_labels[stim_id,1]-stim_labels[stim_id,0]],
                 [stim_labels[stim_id+1,0]-stim_labels[stim_id,0],
                  stim_labels[stim_id+1,1]-stim_labels[stim_id,0]]]
                ).reshape(1,3,2))
    # correct neural data centering at zero.
    neu_time_zero = [np.argmin(np.abs(nt)) for nt in neu_time]
    neu_time = trim_seq(neu_time, neu_time_zero)
    neu_seq = trim_seq(neu_seq, neu_time_zero)
    # correct voltage data centering at zero.
    stim_time_zero = [np.argmin(np.abs(sv)) for sv in stim_value]
    stim_time = trim_seq(stim_time, stim_time_zero)
    stim_value = trim_seq(stim_value, stim_time_zero)
    # concatenate results.
    neu_seq    = np.concatenate(neu_seq, axis=0)
    neu_time   = [nt.reshape(1,-1) for nt in neu_time]
    neu_time   = np.concatenate(neu_time, axis=0)
    stim_seq   = np.concatenate(stim_seq, axis=0)
    stim_value = [sv.reshape(1,-1) for sv in stim_value]
    stim_value = np.concatenate(stim_value, axis=0)
    stim_time  = [st.reshape(1,-1) for st in stim_time]
    stim_time  = np.concatenate(stim_time, axis=0)
    # get mean time stamps.
    neu_time  = np.mean(neu_time, axis=0)
    stim_time = np.mean(stim_time, axis=0)
    return [neu_seq, neu_time, stim_seq, stim_value, stim_time]