#!/usr/bin/env python3

import os
import gc
import numpy as np
from tqdm import tqdm

# create a numpy memmap from numpy array.
def create_memmap(data, file_name, ni):
    file_path = os.path.join('results', 'temp', 'alignment_memmap', file_name+'_'+str(ni).zfill(5)+'.mmap')
    memmap_arr = np.memmap(file_path, dtype=data.dtype, mode='w+', shape=data.shape)
    memmap_arr[:] = data[...]
    return memmap_arr

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
        expected,
        n_stim,
        ):
    # initialization.
    stim_labels = neural_trials['stim_labels']
    dff         = neural_trials['dff']
    time        = neural_trials['time']
    vol_stim    = neural_trials['vol_stim_vis']
    vol_led     = neural_trials['vol_led']
    vol_time    = neural_trials['vol_time']
    neu_seq    = []
    neu_time   = []
    stim_seq   = []
    stim_value = []
    stim_time  = []
    led_value  = []
    pre_isi    = []
    # loop over stimulus.
    for stim_id in tqdm(range(n_stim, stim_labels.shape[0]-n_stim)):
        # find alignment offset.
        if expected=='none':
            t = stim_labels[stim_id,0]
        elif expected=='local':
            t = stim_labels[stim_id,1] + stim_labels[stim_id,0]-stim_labels[stim_id-1,1]
        else:
            raise ValueError('epected can only be none or local')
        # get stimulus timing.
        idx = np.searchsorted(time, t)
        if idx > l_frames and idx < len(time)-r_frames:
            # signal response.
            f = dff[:, idx-l_frames : idx+r_frames]
            f = np.expand_dims(f, axis=0)
            neu_seq.append(f)
            # signal time stamps.
            t = time[idx-l_frames : idx+r_frames] - time[idx]
            neu_time.append(t)
            # voltage.
            vol_t_c = np.searchsorted(vol_time, time[idx])
            vol_t_l = np.searchsorted(vol_time, time[idx-l_frames])
            vol_t_r = np.searchsorted(vol_time, time[idx+r_frames])
            stim_time.append(vol_time[vol_t_l:vol_t_r] - vol_time[vol_t_c])
            stim_value.append(vol_stim[vol_t_l:vol_t_r])
            led_value.append(vol_led[vol_t_l:vol_t_r])
            # stimulus timing.
            stim_seq.append(np.array(
                [[stim_labels[stim_id+i,0]-stim_labels[stim_id,0],
                 stim_labels[stim_id+i,1]-stim_labels[stim_id,0]]
                for i in np.arange(-n_stim,n_stim+1)]).reshape(1,2*n_stim+1,2))
            # preceeding isi.
            pre_isi.append(np.array([stim_labels[stim_id,0]-stim_labels[stim_id-1,1]]))
    # correct stimulus labels.
    stim_labels = neural_trials['stim_labels'][n_stim:-n_stim,:]
    # correct neural data centering at zero.
    neu_time_zero = [np.argmin(np.abs(nt)) for nt in neu_time]
    neu_time = trim_seq(neu_time, neu_time_zero)
    neu_seq = trim_seq(neu_seq, neu_time_zero)
    # correct voltage data centering at zero.
    stim_time_zero = [np.argmin(np.abs(sv)) for sv in stim_value]
    stim_time = trim_seq(stim_time, stim_time_zero)
    stim_value = trim_seq(stim_value, stim_time_zero)
    led_value = trim_seq(led_value, stim_time_zero)
    # concatenate results.
    neu_seq    = np.concatenate(neu_seq, axis=0)
    neu_time   = [nt.reshape(1,-1) for nt in neu_time]
    neu_time   = np.concatenate(neu_time, axis=0)
    stim_seq   = np.concatenate(stim_seq, axis=0)
    stim_value = [sv.reshape(1,-1) for sv in stim_value]
    stim_value = np.concatenate(stim_value, axis=0)
    stim_time  = [st.reshape(1,-1) for st in stim_time]
    stim_time  = np.concatenate(stim_time, axis=0)
    led_value  = [lv.reshape(1,-1) for lv in led_value]
    led_value  = np.concatenate(led_value, axis=0)
    pre_isi    = np.concatenate(pre_isi, axis=0)
    # get mean time stamps.
    neu_time  = np.mean(neu_time, axis=0)
    stim_time = np.mean(stim_time, axis=0)
    # combine results.
    return [stim_labels, neu_seq, neu_time, stim_seq, stim_value, stim_time, led_value, pre_isi]

# run alignment for all sessions
def run_get_stim_response(
        list_neural_trials,
        l_frames, r_frames,
        expected='None',
        n_stim=5,
        ):
    if not os.path.exists(os.path.join('results', 'temp', 'alignment_memmap')):
        os.makedirs(os.path.join('results', 'temp', 'alignment_memmap'))
    # run alignment for each session.
    list_stim_labels = []
    list_neu_seq     = []
    list_neu_time    = []
    list_stim_seq    = []
    list_stim_value  = []
    list_stim_time   = []
    list_led_value   = []
    list_pre_isi     = []
    for ni in range(len(list_neural_trials)):
        [stim_labels, neu_seq, neu_time,
         stim_seq, stim_value, stim_time,
         led_value, pre_isi
         ] = get_stim_response(
             list_neural_trials[ni],
             l_frames, r_frames,
             expected, n_stim)
        stim_labels = create_memmap(stim_labels, 'stim_labels', ni)
        neu_seq     = create_memmap(neu_seq,     'neu_seq',     ni)
        neu_time    = create_memmap(neu_time,    'neu_time',    ni)
        stim_seq    = create_memmap(stim_seq,    'stim_seq',    ni)
        stim_value  = create_memmap(stim_value,  'stim_value',  ni)
        stim_time   = create_memmap(stim_time,   'stim_time',   ni)
        led_value   = create_memmap(led_value,   'led_value',   ni)
        pre_isi     = create_memmap(pre_isi,     'pre_isi',     ni)
        list_stim_labels.append(stim_labels)
        list_neu_seq.append(neu_seq)
        list_neu_time.append(neu_time)
        list_stim_seq.append(stim_seq)
        list_stim_value.append(stim_value)
        list_stim_time.append(stim_time)
        list_led_value.append(led_value)
        list_pre_isi.append(pre_isi)
        del stim_labels
        del neu_seq
        del neu_time
        del stim_seq
        del stim_value
        del stim_time
        del led_value
        del pre_isi
        gc.collect()
    # combine neu_time.
    neu_time = np.nanmean(np.concatenate([nt.reshape(1,-1) for nt in list_neu_time]),axis=0)
    # combine stim_time.
    st_min  = int(np.nanmin(np.concatenate(list_stim_time)))
    st_max  = int(np.nanmax(np.concatenate(list_stim_time)))
    st_rate = np.nanmean(np.concatenate([np.diff(st) for st in list_stim_time]))
    stim_time = np.arange(st_min, st_max, st_rate)
    # interpolate stim_value on the common stim_time.
    list_stim_value = [
        np.apply_along_axis(
            lambda row: np.interp(stim_time, list_stim_time[i], row), 
            axis=1, 
            arr=list_stim_value[i])
        for i in range(len(list_stim_value))]
    # interpolate led_value on the common led_time.
    list_led_value = [
        np.apply_along_axis(
            lambda row: np.interp(stim_time, list_stim_time[i], row), 
            axis=1, 
            arr=list_led_value[i])
        for i in range(len(list_led_value))]
    # combine results.
    alignment = {
        'list_stim_labels': list_stim_labels,
        'list_neu_seq':     list_neu_seq,
        'neu_time':         neu_time,
        'list_stim_seq':    list_stim_seq,
        'list_stim_value':  list_stim_value,
        'stim_time':        stim_time,
        'list_led_value':   list_led_value,
        'list_pre_isi':     list_pre_isi
        }
    return alignment

