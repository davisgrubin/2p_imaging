#!/usr/bin/env python3

import os
import h5py
import numpy as np
import scipy.io as sio
from scipy.signal import find_peaks
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

# read raw_voltages.h5.


def read_raw_voltages(ops):
    f = h5py.File(
        os.path.join(ops['save_path0'], 'raw_voltages.h5'),
        'r')
    try:
        vol_time = np.array(f['raw']['vol_time'])
        vol_start = np.array(f['raw']['vol_start'])
        vol_stim_vis = np.array(f['raw']['vol_stim_vis'])
        vol_hifi = np.array(f['raw']['vol_hifi'])
        vol_img = np.array(f['raw']['vol_img'])
        vol_stim_aud = np.array(f['raw']['vol_stim_aud'])
        vol_flir = np.array(f['raw']['vol_flir'])
        vol_pmt = np.array(f['raw']['vol_pmt'])
        vol_led = np.array(f['raw']['vol_led'])
    except:
        vol_time = np.array(f['raw']['vol_time'])
        vol_start = np.array(f['raw']['vol_start_bin'])
        vol_stim_vis = np.array(f['raw']['vol_stim_bin'])
        vol_img = np.array(f['raw']['vol_img_bin'])
        vol_hifi = np.zeros_like(vol_time)
        vol_stim_aud = np.zeros_like(vol_time)
        vol_flir = np.zeros_like(vol_time)
        vol_pmt = np.zeros_like(vol_time)
        vol_led = np.zeros_like(vol_time)
    f.close()

    return [vol_time, vol_start, vol_stim_vis, vol_img, 
            vol_hifi, vol_stim_aud, vol_flir,
            vol_pmt, vol_led]


# read masks.
def read_masks(ops):
    f = h5py.File(os.path.join(ops['save_path0'], 'masks.h5'), 'r')
    labels = np.array(f['labels'])
    masks = np.array(f['masks_func'])
    mean_func = np.array(f['mean_func'])
    max_func = np.array(f['max_func'])
    mean_anat = np.array(f['mean_anat']) if ops['nchannels'] == 2 else None
    masks_anat = np.array(f['masks_anat']) if ops['nchannels'] == 2 else None
    masks_anat_corrected = np.array(
        f['masks_anat_corrected']) if ops['nchannels'] == 2 else None
    f.close()
    return [labels, masks, mean_func, max_func, mean_anat, masks_anat, masks_anat_corrected]


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
def read_neural_trials(ops, cate_delay):
    # read h5 file.
    f = h5py.File(
        os.path.join(ops['save_path0'], 'neural_trials.h5'),
        'r')
    neural_trials = dict()
    for trial in f['trial_id'].keys():
        neural_trials[trial] = dict()
        for data in f['trial_id'][trial].keys():
            neural_trials[trial][data] = np.array(f['trial_id'][trial][data])
    f.close()
    partition = 4
    # resort delay settings.
    start = np.min(np.array([t for t in neural_trials.keys()]).astype('int32'))
    end   = np.max(np.array([t for t in neural_trials.keys()]).astype('int32'))
    trial_idx = np.arange(start, end+1)
    trial_delay = np.array([neural_trials[str(t)]['trial_delay'] for t in trial_idx])
    # mark short and long delay trials short:0 long:1.
    trial_type = np.zeros_like(trial_delay)
    trial_type[trial_delay>cate_delay] = 1
    # mark epoch trials unvalid:-1 early:1 late:0
    block_change = np.diff(trial_type, prepend=0)
    block_change[block_change!=0] = 1
    block_change[0] = 1
    block_change[-1] = 1
    block_change = np.where(block_change==1)[0]
    block_epoch = np.zeros_like(trial_type)
    for start, end in zip(block_change[:-1], block_change[1:]):
        tran = start + (end - start) // partition
        block_epoch[start:tran] = 1
    block_epoch[:block_change[1]] = -1
    # write into neural trials.
    for i in range(len(trial_idx)):
        neural_trials[str(trial_idx[i])]['trial_type'] = trial_type[i]
        neural_trials[str(trial_idx[i])]['block_epoch'] = block_epoch[i]
    return neural_trials


# read significance test label results.
def read_significance(ops):
    f = h5py.File(
        os.path.join(ops['save_path0'], 'significance.h5'),
        'r')
    significance = {}


    significance['r_vis1']     = np.array(f['significance']['r_vis1'])
    significance['r_push1']    = np.array(f['significance']['r_push1'])
    significance['r_retract1'] = np.array(f['significance']['r_retract1'])
    significance['r_vis2']     = np.array(f['significance']['r_vis2'])
    significance['r_push2']    = np.array(f['significance']['r_push2'])
    significance['r_retract2'] = np.array(f['significance']['r_retract2'])
    significance['r_wait']     = np.array(f['significance']['r_wait'])
    significance['r_reward']   = np.array(f['significance']['r_reward'])
    significance['r_punish']   = np.array(f['significance']['r_punish'])
    significance['r_lick']     = np.array(f['significance']['r_lick'])

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


    def main():
        raw = sio.loadmat(
            os.path.join(ops['save_path0'], 'bpod_session_data.mat'),
            struct_as_record=False, squeeze_me=True)
        raw = _check_keys(raw)['SessionData']
        
        trial_type = []
        trial_delay = []
        trial_AirPuff = []
        trial_LED = []
        trial_ITI = []
        
        
        for i in range(raw['nTrials']):
            trial_states = raw['RawEvents']['Trial'][i]['States']
            trial_Data = raw['RawEvents']['Trial'][i]['Data']
            

            trial_LED.append(1000*np.array(trial_states['LED_Onset']).reshape(-1))
            trial_AirPuff.append(1000*np.array(trial_states['AirPuff']).reshape(-1))
            trial_ITI.append(1000*np.array(trial_states['ITI']).reshape(-1))
            
            if trial_Data['BlockType'] == 'short':
                trial_type.append(1)
            else:
                trial_type.append(2)
                
            trial_delay.append(1000*np.array(trial_Data['AirPuff_OnsetDelay']))
            

                
        bpod_sess_data = {

            'trial_types'        : trial_type,
            'trial_delay'        : trial_delay,
            'trial_AirPuff'      : trial_AirPuff,
            'trial_LED'          : trial_LED,
            'trial_ITI'          : trial_ITI,
            }

        return bpod_sess_data
    bpod_sess_data = main()
    
    return bpod_sess_data
