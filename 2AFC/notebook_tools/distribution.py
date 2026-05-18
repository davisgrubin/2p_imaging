"""Notebook helper implementations migrated from `Test_pilot/test_nb_distribution.py`.

This module is now self-contained so notebooks do not depend on `Test_pilot/test_nb_*`.
"""

import glob
import os
import re

import h5py
import numpy as np
import scipy.io as sio

def _distribution_bpod_mat_path(session_name):
    mouse_name = session_name.split('_')[0]
    session_date = re.search(r'_(\d{8})_', session_name).group(1)
    shared_dir = os.path.join(
        '/storage/project/r-fnajafi3-0/shared/single_interval_discrimination/session_data',
        mouse_name)
    matches = sorted(glob.glob(os.path.join(shared_dir, f'*{session_date}*.mat')))
    if len(matches) == 0:
        raise FileNotFoundError(f'No Bpod mat found for {session_name}')
    return matches[0]

def _distribution_rising_edges(vol_time, vol_bin):
    return vol_time[np.where(np.diff(vol_bin.astype(np.int8), prepend=0) == 1)[0]]

def _distribution_first_vis_times(session_name, session_data_path):
    with h5py.File(os.path.join(session_data_path, 'raw_voltages.h5'), 'r') as f:
        vol_time = np.array(f['raw']['vol_time'])
        vol_stim_vis = np.array(f['raw']['vol_stim_vis'])
        vol_start = np.array(f['raw']['vol_start'])
    start_times = _distribution_rising_edges(vol_time, vol_start)
    stim_times = _distribution_rising_edges(vol_time, vol_stim_vis)
    raw = sio.loadmat(
        _distribution_bpod_mat_path(session_name),
        struct_as_record=False, squeeze_me=True)['SessionData']
    bpod_isis = []
    for trial in np.array(raw.RawEvents.Trial).reshape(-1):
        if not hasattr(trial.Events, 'BNC1High'):
            continue
        pulse_times = 1000*np.array(trial.Events.BNC1High).reshape(-1)
        if len(pulse_times) < 2:
            continue
        bpod_isis.append(pulse_times[1] - pulse_times[0])
    bpod_isis = np.array(bpod_isis)
    candidates = []
    for start_idx in [0, 1]:
        stim_first = stim_times[start_idx::2]
        stim_second = stim_times[start_idx+1::2]
        n_pairs = min(len(stim_first), len(stim_second), len(bpod_isis), len(start_times))
        if n_pairs == 0:
            continue
        score = np.median(np.abs((stim_second[:n_pairs] - stim_first[:n_pairs]) - bpod_isis[:n_pairs]))
        candidates.append((score, stim_first[:n_pairs], start_times[:n_pairs]))
    if len(candidates) == 0:
        raise ValueError(f'Unable to pair Input1 pulses for {session_name}')
    _, stim_first, start_times = min(candidates, key=lambda x: x[0])
    return stim_first, start_times
