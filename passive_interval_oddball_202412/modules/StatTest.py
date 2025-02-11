#!/usr/bin/env python3

import os
import h5py
import numpy as np
from scipy.stats import ks_2samp

from modules.Alignment import get_stim_response
from modules.ReadResults import read_neural_trials
from utils import get_odd_stim_prepost_idx
from utils import exclude_odd_stim
from utils import pick_trial

# compute indice with givn time window for df/f.
def get_frame_idx_from_time(timestamps, l_time, r_time):
    l_idx = np.argmin(np.abs(timestamps-l_time))
    r_idx = np.argmin(np.abs(timestamps-r_time))
    return l_idx, r_idx

# label ROIs responsiveness comparing baseline and early window.
def test_win(neu_seq, neu_time, win_base, win_early):
    p_thres = 0.05
    responsive = np.zeros(neu_seq.shape[1])
    for i in range(neu_seq.shape[1]):
        l_base,  r_base  = get_frame_idx_from_time(neu_time, win_base[0],  win_base[1])
        l_early, r_early = get_frame_idx_from_time(neu_time, win_early[0], win_early[1])
        neu_base  = np.mean(neu_seq[:,i,l_base:r_base], axis=1)
        neu_early = np.mean(neu_seq[:,i,l_early:r_early], axis=1)
        pvalue = ks_2samp(neu_base, neu_early)[1]
        responsive[i] = 1 if pvalue < p_thres else 0
    responsive = responsive.astype('bool')
    return responsive

# label ROIs responsiveness comparing pre and post evoked window.
def test_prepost(neu_seq_pre, neu_seq_post, neu_time, win_evoke):
    p_thres = 0.05
    responsive = np.zeros(neu_seq_pre.shape[1])
    for i in range(neu_seq_pre.shape[1]):
        l_evoke, r_evoke = get_frame_idx_from_time(neu_time, win_evoke[0], win_evoke[1])
        neu_pre = np.mean(neu_seq_pre[:,i,l_evoke:r_evoke], axis=1)
        neu_post = np.mean(neu_seq_post[:,i,l_evoke:r_evoke], axis=1)
        pvalue = ks_2samp(neu_pre, neu_post)[1]
        responsive[i] = 1 if pvalue < p_thres else 0
    responsive = responsive.astype('bool')
    return responsive

# pick trials and compute responsiveness for standard.
def stat_test_standard(neu_seq, neu_time, stim_labels):
    win_base  = [-200,0]
    win_early = [0,400]
    labels = exclude_odd_stim(stim_labels)
    idx = pick_trial(labels, [2,3,4,5], None, None, None, None, [0])
    responsive = test_win(
        neu_seq[idx,:,:], neu_time, win_base, win_early)
    return responsive

# pick trials and compute responsiveness for change.
def stat_test_change(neu_seq, neu_time, stim_labels):
    win_evoke = [-300,300]
    idx_post = pick_trial(stim_labels, [-2,-3,-4,-5], None, None, None, None, [0])
    idx_pre = np.diff(idx_post, append=0)
    idx_pre[idx_pre==-1] = 0
    idx_pre = idx_pre.astype('bool')
    responsive = test_prepost(
        neu_seq[idx_pre,:,:], neu_seq[idx_post,:,:], neu_time, win_evoke)
    return responsive

# pick trials and compute responsiveness for oddball.
def stat_test_oddball(neu_seq, neu_time, stim_labels):
    win_evoke = [-300,300]
    [idx_pre_short, _,
     idx_post_short, _] = get_odd_stim_prepost_idx(stim_labels)
    responsive = test_prepost(
        neu_seq[idx_pre_short,:,:], neu_seq[idx_post_short,:,:], neu_time, win_evoke)
    return responsive

# save significance label results.
def save_significance(
        ops,
        r_standard, r_change, r_oddball
        ):
    h5_path = os.path.join(ops['save_path0'], 'significance.h5')
    if os.path.exists(h5_path):
        os.remove(h5_path)
    f = h5py.File(h5_path, 'w')
    grp = f.create_group('significance')
    grp['r_standard']  = r_standard
    grp['r_change']  = r_change
    grp['r_oddball'] = r_oddball
    f.close()

def run(ops):
    print('Aligning neural population response')
    neural_trials = read_neural_trials(ops)
    stim_labels = neural_trials['stim_labels']
    [stim_labels, neu_seq, neu_time, _, _, _, _, _] = get_stim_response(
            neural_trials, 100, 200, 'none', 2)
    print('Running statistics test')
    r_standard = stat_test_standard(neu_seq, neu_time, stim_labels)
    r_change   = stat_test_standard(neu_seq, neu_time, stim_labels)
    r_oddball  = stat_test_standard(neu_seq, neu_time, stim_labels)
    print('{}/{} ROIs responsive to standard'.format(np.sum(r_standard), len(r_standard)))
    print('{}/{} ROIs responsive to change'.format(np.sum(r_change), len(r_change)))
    print('{}/{} ROIs responsive to oddball'.format(np.sum(r_oddball), len(r_oddball)))
    save_significance(ops, r_standard, r_change, r_oddball)

