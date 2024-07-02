#!/usr/bin/env python3

import os
import h5py
import numpy as np
from scipy.stats import ks_2samp

from modules.Alignment import get_stim_response
from modules.Alignment import get_outcome_response
from modules.Alignment import get_motor_response
from modules.ReadResults import read_neural_trials


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
        l_base, r_base = get_frame_idx_from_time(neu_time, win_base[0], win_base[1])
        l_early, r_early = get_frame_idx_from_time(neu_time, win_early[0], win_early[1])
        neu_base = neu_seq[:,i,l_base:r_base].reshape(-1)
        neu_early = neu_seq[:,i,l_early:r_early].reshape(-1)
        pvalue = ks_2samp(neu_base, neu_early)[1]
        responsive[i] = 1 if pvalue < p_thres else 0
    responsive = responsive.astype('bool')
    return responsive


# save significance label results.
def save_significance(
        ops,
        r_vis1, r_press1, r_retract1,
        r_vis2, r_press2,
        r_reward, r_punish, r_lick
        ):
    f = h5py.File(
        os.path.join(ops['save_path0'], 'significance.h5'),
        'w')
    grp = f.create_group('significance')
    grp['r_vis1']     = r_vis1
    grp['r_press1']   = r_press1
    grp['r_retract1'] = r_retract1
    grp['r_vis2']     = r_vis2
    grp['r_press2']   = r_press2
    grp['r_reward']   = r_reward
    grp['r_punish']   = r_punish
    grp['r_lick']     = r_lick
    f.close()
    

def run(ops):
    print('===============================================')
    print('============== significance test ==============')
    print('===============================================')
    print('Aligning neural population response')
    neural_trials = read_neural_trials(ops)
    [neu_seq_vis1, neu_time_vis1, _, _] = get_stim_response(
            neural_trials, 'trial_vis1', 5, 75)
    [neu_seq_press1, neu_time_press1, _] = get_motor_response(
        neural_trials, 'trial_press1', 30, 50)
    [neu_seq_retract1, neu_time_retract1, _] = get_motor_response(
        neural_trials, 'trial_retract1', 30, 50)
    [neu_seq_vis2, neu_time_vis2, _, _] = get_stim_response(
            neural_trials, 'trial_vis2', 30, 50)
    [neu_seq_press2, neu_time_press2, _] = get_motor_response(
        neural_trials, 'trial_press2', 30, 50)
    [neu_seq_reward, neu_time_reward, _] = get_outcome_response(
            neural_trials, 'trial_reward', 30, 50)
    [neu_seq_punish, neu_time_punish, _] = get_outcome_response(
            neural_trials, 'trial_punish', 30, 50)
    [neu_seq_lick, neu_time_lick, _] = get_motor_response(
        neural_trials, 'trial_lick', 30, 50)
    print('Running statistics test')
    r_vis1     = test_win(neu_seq_vis1,     neu_time_vis1,     [-100,0], [0,250])
    r_press1   = test_win(neu_seq_press1,   neu_time_press1,   [-100,0], [0,250])
    r_retract1 = test_win(neu_seq_retract1, neu_time_retract1, [-100,0], [0,250])
    r_vis2     = test_win(neu_seq_vis2,     neu_time_vis2,     [-100,0], [0,250])
    r_press2   = test_win(neu_seq_press2,   neu_time_press2,   [-100,0], [0,250])
    r_reward   = test_win(neu_seq_reward,   neu_time_reward,   [-500,0], [0,500])
    r_punish   = test_win(neu_seq_punish,   neu_time_punish,   [-500,0], [0,500])
    r_lick     = test_win(neu_seq_lick,     neu_time_lick,     [-300,-50], [-50,50])
    print('{}/{} ROIs responsive to vis1'.format(np.sum(r_vis1), len(r_vis1)))
    print('{}/{} ROIs responsive to press1'.format(np.sum(r_press1), len(r_press1)))
    print('{}/{} ROIs responsive to retract1'.format(np.sum(r_retract1), len(r_retract1)))
    print('{}/{} ROIs responsive to vis2'.format(np.sum(r_vis2), len(r_vis2)))
    print('{}/{} ROIs responsive to press2'.format(np.sum(r_press2), len(r_press2)))
    print('{}/{} ROIs responsive to reward'.format(np.sum(r_reward), len(r_reward)))
    print('{}/{} ROIs responsive to punish'.format(np.sum(r_punish), len(r_punish)))
    print('{}/{} ROIs responsive to lick'.format(np.sum(r_lick), len(r_lick)))
    save_significance(
        ops,
        r_vis1, r_press1, r_retract1,
        r_vis2, r_press2,
        r_reward, r_punish, r_lick)
    
    
    
    