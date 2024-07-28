#!/usr/bin/env python3

import os
import h5py
import numpy as np
from scipy.stats import ks_2samp

from modules.Alignment import get_stim_response
from modules.ReadResults import read_neural_trials
from plot.utils import get_frame_idx_from_time
from plot.utils import exclude_post_odd_stim
from plot.utils import get_odd_stim_prepost_idx


# label ROIs responsiveness comparing baseline and early window.
def test_win(neu_seq, neu_time, win_base, win_early):
    p_thres = 0.05
    responsive = np.zeros(neu_seq.shape[1])
    for i in range(neu_seq.shape[1]):
        l_base, r_base = get_frame_idx_from_time(neu_time, 0, win_base[0], win_base[1])
        l_early, r_early = get_frame_idx_from_time(neu_time, 0, win_early[0], win_early[1])
        neu_base = neu_seq[:,i,l_base:r_base].reshape(-1)
        neu_early = neu_seq[:,i,l_early:r_early].reshape(-1)
        pvalue = ks_2samp(neu_base, neu_early)[1]
        responsive[i] = 1 if pvalue < p_thres else 0
    responsive = responsive.astype('bool')
    return responsive

# label ROIs responsiveness comparing pre and post events in early window.
def test_prepost(neu_seq_pre, neu_seq_post, neu_time, win_early):
    p_thres = 0.05
    responsive = np.zeros(neu_seq_pre.shape[1])
    for i in range(neu_seq_pre.shape[1]):
        l_early, r_early = get_frame_idx_from_time(neu_time, 0, win_early[0], win_early[1])
        neu_pre_early = neu_seq_pre[:,i,l_early:r_early].reshape(-1)
        neu_post_early = neu_seq_post[:,i,l_early:r_early].reshape(-1)
        pvalue = ks_2samp(neu_pre_early, neu_post_early)[1]
        responsive[i] = 1 if pvalue < p_thres else 0
    responsive = responsive.astype('bool')
    return responsive

# find responsive neurons to normal stimulus.
def get_r_normal(neu_seq, neu_time, stim_labels):
    win_base = [-150,0]
    win_early = [0,250]
    idx = exclude_post_odd_stim(stim_labels)[:,2]>0
    r_normal = test_win(neu_seq[idx,:,:], neu_time, win_base, win_early)
    return r_normal

# find responsive neurons to change stimulus.
def get_r_change(neu_seq, neu_time, stim_labels):
    win_early = [0,250]
    idx_post = stim_labels[:,2]<-1
    idx_pre = np.zeros_like(idx_post)
    idx_pre[:-1] = idx_post[1:]
    r_change = test_prepost(
        neu_seq[idx_pre,:,:], neu_seq[idx_post,:,:],
        neu_time, win_early)
    return r_change

# find responsive neurons to change stimulus.
def get_r_oddball(neu_seq, neu_time, stim_labels):
    win_early = [0,250]
    [idx_pre_short,  idx_pre_long,
     idx_post_short, idx_post_long] = get_odd_stim_prepost_idx(
        stim_labels)
    r_odd_short = test_prepost(
        neu_seq[idx_pre_short,:,:], neu_seq[idx_post_short,:,:],
        neu_time, win_early)
    r_odd_long = test_prepost(
        neu_seq[idx_pre_long,:,:], neu_seq[idx_post_long,:,:],
        neu_time, win_early)
    r_odd = r_odd_short + r_odd_long
    return r_odd

# save significance label results.
def save_significance(
        ops,
        r_normal, r_change, r_oddball
        ):
    f = h5py.File(
        os.path.join(ops['save_path0'], 'significance.h5'),
        'w')
    grp = f.create_group('significance')
    grp['r_normal'] = r_normal
    grp['r_change'] = r_change
    grp['r_oddball'] = r_oddball
    f.close()
    

def run(ops):
    print('===============================================')
    print('============== significance test ==============')
    print('===============================================')
    print('Aligning neural population response')
    neural_trials = read_neural_trials(ops)
    stim_labels = neural_trials['stim_labels'][1:-1,:]
    [neu_seq, neu_time, _, _, _] = get_stim_response(neural_trials, 200, 200)
    print('Running statistics test')
    r_normal  = get_r_normal(neu_seq, neu_time, stim_labels)
    r_change  = get_r_change(neu_seq, neu_time, stim_labels)
    r_oddball = get_r_oddball(neu_seq, neu_time, stim_labels)
    print('{}/{} ROIs responsive to normal'.format(np.sum(r_normal), len(r_normal)))
    print('{}/{} ROIs responsive to change'.format(np.sum(r_change), len(r_change)))
    print('{}/{} ROIs responsive to oddball'.format(np.sum(r_oddball), len(r_oddball)))
    save_significance(ops, r_normal, r_change, r_oddball)


    
    