#!/usr/bin/env python3

import os
import h5py
import numpy as np
import scipy.io as sio


# read raw_voltages.h5.
def read_raw_voltages(ops):
    try:
        f = h5py.File(
            os.path.join(ops['save_path0'], 'raw_voltages.h5'),
            'r')
        vol_time = np.array(f['raw']['vol_time'])
        vol_start_bin = np.array(f['raw']['vol_start_bin'])
        vol_stim_bin = np.array(f['raw']['vol_stim_bin'])
        vol_img_bin = np.array(f['raw']['vol_img_bin'])
        f.close()
        return [vol_time, vol_start_bin, vol_stim_bin, vol_img_bin]
    except:
        raise ValueError('Fail to read voltage data')


# read masks.
def read_masks(ops):
    f = h5py.File(os.path.join(ops['save_path0'], 'masks.h5'),'r')
    labels = np.array(f['labels'])
    masks = np.array(f['masks_func'])
    mean_func = np.array(f['mean_func'])
    max_func = np.array(f['max_func'])
    mean_anat = np.array(f['mean_anat']) if ops['nchannels'] == 2 else None
    masks_anat = np.array(f['masks_anat']) if ops['nchannels'] == 2 else None
    f.close()
    return [labels, masks, mean_func, max_func, mean_anat, masks_anat]


# read motion correction offsets.
def read_move_offset(ops):
    f = h5py.File(os.path.join(ops['save_path0'], 'move_offset.h5'), 'r')
    xoff = np.array(f['xoff'])
    yoff = np.array(f['yoff'])
    f.close()
    return [xoff, yoff]


# read dff traces.
def read_dff(ops):
    f = h5py.File(os.path.join(ops['save_path0'], 'dff.h5'), 'r')
    dff = np.array(f['dff'])
    f.close()
    return dff


# read trailized neural traces with stimulus alignment.
def read_neural_trials(ops):
    f = h5py.File(
        os.path.join(ops['save_path0'], 'neural_trials.h5'),
        'r')
    neural_trials = dict()
    for trial in f['trial_id'].keys():
        neural_trials[trial] = dict()
        for data in f['trial_id'][trial].keys():
            neural_trials[trial][data] = np.array(f['trial_id'][trial][data])
    f.close()
    return neural_trials


# read significance test label results.
def read_significance(ops):
    f = h5py.File(
        os.path.join(ops['save_path0'], 'significance.h5'),
        'r')    
    significance = {}
    significance['r_stim_all']   = np.array(f['significance']['r_stim_all'])
    significance['r_stim_onset'] = np.array(f['significance']['r_stim_onset'])
    significance['r_stim_pre']   = np.array(f['significance']['r_stim_pre'])
    significance['r_stim_post_first'] = np.array(f['significance']['r_stim_post_first'])
    significance['r_stim_post_all']   = np.array(f['significance']['r_stim_post_all'])
    significance['r_reward'] = np.array(f['significance']['r_reward'])
    significance['r_punish'] = np.array(f['significance']['r_punish'])
    significance['r_lick_all']      = np.array(f['significance']['r_lick_all'])
    significance['r_lick_reaction'] = np.array(f['significance']['r_lick_reaction'])
    significance['r_lick_decision'] = np.array(f['significance']['r_lick_decision'])
    return significance


# read bpod session data.
def read_bpod_mat_data(ops):
    def _check_keys(d):
        for key in d:
            if isinstance(d[key], sio.matlab.mat_struct):
                d[key] = _todict(d[key])
        return d
    def _todict(matobj):
        d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, sio.matlab.mat_struct):
                d[strg] = _todict(elem)
            elif isinstance(elem, np.ndarray):
                d[strg] = _tolist(elem)
            else:
                d[strg] = elem
        return d
    def _tolist(ndarray):
        elem_list = []
        for sub_elem in ndarray:
            if isinstance(sub_elem, sio.matlab.mat_struct):
                elem_list.append(_todict(sub_elem))
            elif isinstance(sub_elem, np.ndarray):
                elem_list.append(_tolist(sub_elem))
            else:
                elem_list.append(sub_elem)
        return elem_list
    raw = sio.loadmat(
        os.path.join(ops['save_path0'], 'bpod_session_data.mat'),
        struct_as_record=False, squeeze_me=True)
    raw = _check_keys(raw)['SessionData']
    trial_types = np.array(raw['TrialTypes'])
    # loop over one session for extracting data.
    trial_reward = []
    trial_punish = []
    trial_wrinit = []
    trial_notcho = []
    trial_stim_start = []
    trial_stim_seq = []
    trial_lick = []
    trial_reaction = []
    trial_decision = []
    trial_isi_pre = []
    trial_isi_post = []
    for i in range(raw['nTrials']):
        trial_states = raw['RawEvents']['Trial'][i]['States']
        trial_events = raw['RawEvents']['Trial'][i]['Events']
        # empirical isi.
        trial_isi_pre.append(np.mean(raw['ProcessedSessionData'][i]['trial_isi']['PreISI']))
        trial_isi_post.append(np.mean(raw['ProcessedSessionData'][i]['trial_isi']['PostISI']))
        # stimulus intervals.
        if ('VisStimTrigger' in trial_states.keys() and
            not np.isnan(trial_states['VisStimTrigger'][0])):
            # stimulus start.
            trial_stim_start.append(np.array([1000*np.array(trial_states['VisStimTrigger'][1])]))
            # stimulus sequence.
            if ('BNC1High' in trial_events.keys() and
                'BNC1Low' in trial_events.keys() and
                np.array(trial_events['BNC1High']).reshape(-1).shape[0]==np.array(trial_events['BNC1Low']).reshape(-1).shape[0] and
                np.array(trial_events['BNC1High']).reshape(-1).shape[0] >= 3
                ):
                stim_seq = 1000*np.array([trial_events['BNC1High'], trial_events['BNC1Low']])
            else:
                stim_seq = np.array([[np.nan], [np.nan]])
            trial_stim_seq.append(stim_seq)
        else:
            trial_stim_start.append(np.array([np.nan]))
            trial_stim_seq.append([[np.nan], [np.nan]])
        # lick events.
        if ('VisStimTrigger' in trial_states.keys() and
            not np.isnan(trial_states['VisStimTrigger'][1])):
            licking_events = []
            direction = []
            correctness = []
            if 'Port1In' in trial_events.keys():
                lick_left = np.array(trial_events['Port1In']).reshape(-1)
                licking_events.append(lick_left)
                direction.append(np.zeros_like(lick_left))
                if trial_types[i] == 1:
                    correctness.append(np.ones_like(lick_left))
                else:
                    correctness.append(np.zeros_like(lick_left))
            if 'Port3In' in trial_events.keys():
                lick_right = np.array(trial_events['Port3In']).reshape(-1)
                licking_events.append(lick_right)
                direction.append(np.ones_like(lick_right))
                if trial_types[i] == 2:
                    correctness.append(np.ones_like(lick_right))
                else:
                    correctness.append(np.zeros_like(lick_right))
            if len(licking_events) > 0:
                licking_events = np.concatenate(licking_events).reshape(1,-1)
                correctness = np.concatenate(correctness).reshape(1,-1)
                direction = np.concatenate(direction).reshape(1,-1)
                lick = np.concatenate([1000*licking_events, direction, correctness])
                # all licking events.
                trial_lick.append(lick)
                # reaction licking.
                reaction_idx = np.where(lick[0]>1000*trial_states['VisStimTrigger'][1])[0]
                if len(reaction_idx)>0:
                    lick_reaction = lick.copy()[:, reaction_idx[0]].reshape(3,1)
                    trial_reaction.append(lick_reaction)
                else:
                    trial_reaction.append(np.array([[np.nan], [np.nan], [np.nan]]))
                # effective licking to outcome.
                decision_idx = np.where(lick[0]>1000*trial_states['WindowChoice'][1])[0]
                if len(decision_idx)>0:
                    lick_decision = lick.copy()[:, decision_idx[0]].reshape(3,1)
                    trial_decision.append(lick_decision)
                else:
                    trial_decision.append(np.array([[np.nan], [np.nan], [np.nan]]))
            else:
                trial_lick.append(np.array([[np.nan], [np.nan], [np.nan]]))
                trial_reaction.append(np.array([[np.nan], [np.nan], [np.nan]]))
                trial_decision.append(np.array([[np.nan], [np.nan], [np.nan]]))
        else:
            trial_lick.append(np.array([[np.nan], [np.nan], [np.nan]]))
            trial_reaction.append(np.array([[np.nan], [np.nan], [np.nan]]))
            trial_decision.append(np.array([[np.nan], [np.nan], [np.nan]]))
        # reward.
        if ('Reward' in trial_states.keys() and not np.isnan(trial_states['Reward'][0])):
            trial_reward.append(1000*np.array(trial_states['Reward']).reshape(-1))
        else:
            trial_reward.append(np.array([np.nan, np.nan]))
        # punish.
        if ('Punish' in trial_states.keys() and not np.isnan(trial_states['Punish'][0])):
            trial_punish.append(1000*np.array(trial_states['Punish']).reshape(-1))
        else:
            trial_punish.append(np.array([np.nan, np.nan]))
        # wrong init.
        if ('WrongInitiation' in trial_states.keys() and not np.isnan(trial_states['WrongInitiation'][0])):
            trial_wrinit.append(1000*np.array(trial_states['WrongInitiation']).reshape(-1))
        else:
            trial_wrinit.append(np.array([np.nan, np.nan]))
        # did not choose.
        if ('DidNotChoose' in trial_states.keys() and not np.isnan(trial_states['DidNotChoose'][0])):
            trial_notcho.append(1000*np.array(trial_states['DidNotChoose']).reshape(-1))
        else:
            trial_notcho.append(np.array([np.nan, np.nan]))
    bpod_sess_data = {
        'trial_types'      : np.array(raw['TrialTypes']),
        'trial_isi_pre'    : trial_isi_pre,
        'trial_isi_post'   : trial_isi_post,
        'trial_stim_start' : trial_stim_start,
        'trial_stim_seq'   : trial_stim_seq,
        'trial_lick'       : trial_lick,
        'trial_reaction'   : trial_reaction,
        'trial_decision'   : trial_decision,
        'trial_reward'     : trial_reward,
        'trial_punish'     : trial_punish,
        'trial_wrinit'     : trial_wrinit,
        'trial_notcho'     : trial_notcho,
        }
    return bpod_sess_data
