#!/usr/bin/env python3

import os
import h5py
import numpy as np
from scipy.stats import ks_2samp

from modules.Alignment import get_stim_response
from modules.Alignment import get_stim_response_mode
from modules.Alignment import get_lick_response
from modules.Alignment import get_outcome_response
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

# label ROIs responsiveness comparing response on lick directions.
def test_sides_lick(neu_seq, neu_time, lick_direc, win_evoked):
    p_thres = 0.05
    responsive = np.zeros(neu_seq.shape[1])
    for i in range(neu_seq.shape[1]):
        l_evoked, r_evoked = get_frame_idx_from_time(
            neu_time, win_evoked[0], win_evoked[1])
        neu_left = neu_seq[lick_direc==0, i, l_evoked:r_evoked].reshape(-1)
        neu_right = neu_seq[lick_direc==1, i, l_evoked:r_evoked].reshape(-1)
        pvalue = ks_2samp(neu_left, neu_right)[1]
        responsive[i] = 1 if pvalue < p_thres else 0
    responsive = responsive.astype('bool')
    return responsive

# label ROIs responsiveness comparing response on trial types.
def test_sides_trial(neu_seq, neu_time, trial_types, win_evoked):
    p_thres = 0.05
    responsive = np.zeros(neu_seq.shape[1])
    for i in range(neu_seq.shape[1]):
        l_evoked, r_evoked = get_frame_idx_from_time(
            neu_time, win_evoked[0], win_evoked[1])
        neu_left = neu_seq[trial_types==1, i, l_evoked:r_evoked].reshape(-1)
        neu_right = neu_seq[trial_types==2, i, l_evoked:r_evoked].reshape(-1)
        pvalue = ks_2samp(neu_left, neu_right)[1]
        responsive[i] = 1 if pvalue < p_thres else 0
    responsive = responsive.astype('bool')
    return responsive

# save significance label results.
def save_significance(
        ops,
        r_stim_all, r_stim_onset, r_stim_pre,
        r_stim_post_first, r_stim_post_all,
        r_reward, r_punish,
        r_lick_all, r_lick_reaction, r_lick_decision
        ):
    f = h5py.File(
        os.path.join(ops['save_path0'], 'significance.h5'),
        'w')
    grp = f.create_group('significance')
    grp['r_stim_all']   = r_stim_all
    grp['r_stim_onset'] = r_stim_onset
    grp['r_stim_pre'] = r_stim_pre
    grp['r_stim_post_first'] = r_stim_post_first
    grp['r_stim_post_all'] = r_stim_post_all
    grp['r_reward'] = r_reward
    grp['r_punish'] = r_punish
    grp['r_lick_all']      = r_lick_all
    grp['r_lick_reaction'] = r_lick_reaction
    grp['r_lick_decision'] = r_lick_decision
    f.close()


def run(ops):
    print('===============================================')
    print('============== significance test ==============')
    print('===============================================')
    print('Aligning neural population response')
    neural_trials = read_neural_trials(ops)
    [neu_seq_stim_all, neu_time_stim,
     stim_seq, trial_types,
     trial_isi_pre, trial_isi_post,
     outcome, stim_idx] = get_stim_response(neural_trials, 30, 50)
    [neu_seq_stim_onset, _, _] = get_stim_response_mode(
        neu_seq_stim_all, trial_types, trial_isi_pre, stim_idx, 'onset')
    [neu_seq_stim_pre, _, _] = get_stim_response_mode(
        neu_seq_stim_all, trial_types, trial_isi_pre, stim_idx, 'pre_all')
    [neu_cate_post_first, trial_types_post_first, _] = get_stim_response_mode(
        neu_seq_stim_all, trial_types, trial_isi_post, stim_idx, 'post_first')
    [neu_cate_post_all, trial_types_post_all, _] = get_stim_response_mode(
        neu_seq_stim_all, trial_types, trial_isi_post, stim_idx, 'post_all')
    [neu_seq_reward, neu_time_reward, _, lick_direc_reward] = get_outcome_response(
         neural_trials, 'trial_reward', 30, 50)
    [neu_seq_punish, neu_time_punish, _, lick_direc_punish] = get_outcome_response(
         neural_trials, 'trial_punish', 30, 50)
    [neu_seq_lick_all, neu_time_lick_all, lick_direc_all] = get_lick_response(
        neural_trials, 'trial_lick', 30, 50)
    [neu_seq_lick_reaction, neu_time_lick_reaction, lick_direc_reaction] = get_lick_response(
        neural_trials, 'trial_reaction', 30, 50)
    [neu_seq_lick_decision, neu_time_lick_decision, lick_direc_decision] = get_lick_response(
        neural_trials, 'trial_decision', 30, 50)
    print('Running statistics test')
    r_stim_all   = test_win(neu_seq_stim_all,   neu_time_stim, [-200,0], [0,250])
    r_stim_onset = test_win(neu_seq_stim_onset, neu_time_stim, [-200,0], [0,250])
    r_stim_pre   = test_win(neu_seq_stim_pre,   neu_time_stim, [-200,0], [0,250])
    r_stim_post_first = test_sides_trial(neu_cate_post_first, neu_time_stim, trial_types_post_first, [0,250])
    r_stim_post_all = test_sides_trial(neu_cate_post_all, neu_time_stim, trial_types_post_all, [0,250])
    r_reward = test_sides_lick(neu_seq_reward, neu_time_reward, lick_direc_reward, [-200,200])
    r_punish = test_sides_lick(neu_seq_punish, neu_time_punish, lick_direc_punish, [-200,200])
    r_lick_all      = test_sides_lick(neu_seq_lick_all, neu_time_lick_all, lick_direc_all, [0,200])
    r_lick_reaction = test_sides_lick(neu_seq_lick_reaction, neu_time_lick_reaction, lick_direc_reaction, [0,300])
    r_lick_decision = test_sides_lick(neu_seq_lick_decision, neu_time_lick_decision, lick_direc_decision, [0,300])
    print('{}/{} ROIs responsive to all stim'.format(np.sum(r_stim_all), len(r_stim_all)))
    print('{}/{} ROIs responsive to stim onset'.format(np.sum(r_stim_onset), len(r_stim_onset)))
    print('{}/{} ROIs responsive to pre pert stim'.format(np.sum(r_stim_pre), len(r_stim_pre)))
    print('{}/{} ROIs responsive to 1st post pert stim'.format(np.sum(r_stim_post_first), len(r_stim_post_first)))
    print('{}/{} ROIs responsive to all post pert stim'.format(np.sum(r_stim_post_all), len(r_stim_post_all)))
    print('{}/{} ROIs responsive to reward'.format(np.sum(r_reward), len(r_reward)))
    print('{}/{} ROIs responsive to punish'.format(np.sum(r_punish), len(r_punish)))
    print('{}/{} ROIs responsive to all lick'.format(np.sum(r_lick_all), len(r_lick_all)))
    print('{}/{} ROIs responsive to reaction lick'.format(np.sum(r_lick_reaction), len(r_lick_reaction)))
    print('{}/{} ROIs responsive to decision lick'.format(np.sum(r_lick_decision), len(r_lick_decision)))
    save_significance(
        ops,
        r_stim_all, r_stim_onset, r_stim_pre,
        r_stim_post_first, r_stim_post_all,
        r_reward, r_punish,
        r_lick_all, r_lick_reaction, r_lick_decision)
    
    
    
    