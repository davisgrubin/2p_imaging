#!/usr/bin/env python3

import os
import gc
import numpy as np
from tqdm import tqdm

# create a numpy memmap from numpy array.
def create_memmap(temp_folder, data, file_name, ni):
    file_path = os.path.join('results', temp_folder, 'alignment_memmap', file_name+'_'+str(ni).zfill(5)+'.mmap')
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
        expected,
        n_stim,
        ):
    l_frames = 250
    r_frames = 250
    # initialization.
    stim_labels = neural_trials['stim_labels']
    dff         = neural_trials['dff']
    time        = neural_trials['time']
    pupil       = neural_trials['camera_pupil']
    neu_seq      = []
    neu_time     = []
    stim_seq     = []
    camera_pupil = []
    pre_isi      = []
    post_isi     = []
    # loop over stimulus.
    for stim_id in tqdm(range(n_stim, stim_labels.shape[0]-n_stim), desc='trials'):
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
            # stimulus timing.
            stim_seq.append(np.array(
                [[stim_labels[stim_id+i,0]-stim_labels[stim_id,0],
                 stim_labels[stim_id+i,1]-stim_labels[stim_id,0]]
                for i in np.arange(-n_stim,n_stim+1)]).reshape(1,2*n_stim+1,2))
            # camera for dlc.
            camera_pupil.append(pupil[idx-l_frames : idx+r_frames])
            # interval around.
            pre_isi.append(np.array([stim_labels[stim_id,0]-stim_labels[stim_id-1,1]]))
            post_isi.append(np.array([stim_labels[stim_id+1,0]-stim_labels[stim_id,1]]))
    # correct stimulus labels.
    stim_labels = neural_trials['stim_labels'][n_stim:-n_stim,:]
    # correct neural data centering at zero.
    neu_time_zero = [np.argmin(np.abs(nt)) for nt in neu_time]
    neu_time = trim_seq(neu_time, neu_time_zero)
    neu_seq = trim_seq(neu_seq, neu_time_zero)
    # correct camera data centering at zero.
    camera_pupil = trim_seq(camera_pupil, neu_time_zero)
    # concatenate results.
    neu_seq  = np.concatenate(neu_seq, axis=0)
    neu_time = [nt.reshape(1,-1) for nt in neu_time]
    neu_time = np.concatenate(neu_time, axis=0)
    stim_seq   = np.concatenate(stim_seq, axis=0)
    camera_pupil = [cp.reshape(1,-1) for cp in camera_pupil]
    camera_pupil = np.concatenate(camera_pupil, axis=0)
    pre_isi  = np.concatenate(pre_isi, axis=0)
    post_isi = np.concatenate(post_isi, axis=0)
    # get mean time stamps.
    neu_time  = np.mean(neu_time, axis=0)
    # combine results.
    return [stim_labels, neu_seq, neu_time, stim_seq, camera_pupil, pre_isi, post_isi]

# run alignment for all sessions
def run_get_stim_response(
        temp_folder,
        list_neural_trials,
        expected='None',
        n_stim=5,
        ):
    # run alignment for each session.
    list_stim_labels  = []
    list_neu_seq      = []
    list_neu_time     = []
    list_stim_seq     = []
    list_camera_pupil = []
    list_pre_isi      = []
    list_post_isi     = []
    for ni in range(len(list_neural_trials)):
        print(f'Aligning trials for session {ni+1}/{len(list_neural_trials)}')
        [stim_labels, neu_seq, neu_time,
         stim_seq,
         camera_pupil,
         pre_isi, post_isi
         ] = get_stim_response(
             list_neural_trials[ni],
             expected, n_stim)
        stim_labels  = create_memmap(temp_folder, stim_labels,  'stim_labels',  ni)
        neu_seq      = create_memmap(temp_folder, neu_seq,      'neu_seq',      ni)
        neu_time     = create_memmap(temp_folder, neu_time,     'neu_time',     ni)
        stim_seq     = create_memmap(temp_folder, stim_seq,     'stim_seq',     ni)
        camera_pupil = create_memmap(temp_folder, camera_pupil, 'camera_pupil', ni)
        pre_isi      = create_memmap(temp_folder, pre_isi,      'pre_isi',      ni)
        post_isi     = create_memmap(temp_folder, post_isi,     'post_isi',     ni)
        list_stim_labels.append(stim_labels)
        list_neu_seq.append(neu_seq)
        list_neu_time.append(neu_time)
        list_stim_seq.append(stim_seq)
        list_camera_pupil.append(camera_pupil)
        list_pre_isi.append(pre_isi)
        list_post_isi.append(post_isi)
        del stim_labels
        del neu_seq
        del neu_time
        del stim_seq
        del camera_pupil
        del pre_isi
        del post_isi
        gc.collect()
    # combine neu_time.
    neu_time = np.nanmean(np.concatenate([nt.reshape(1,-1) for nt in list_neu_time]),axis=0)
    # combine results.
    alignment = {
        'list_stim_labels':  list_stim_labels,
        'list_neu_seq':      list_neu_seq,
        'neu_time':          neu_time,
        'list_stim_seq':     list_stim_seq,
        'list_camera_pupil': list_camera_pupil,
        'list_pre_isi':      list_pre_isi,
        'list_post_isi':     list_post_isi,
        }
    return alignment

