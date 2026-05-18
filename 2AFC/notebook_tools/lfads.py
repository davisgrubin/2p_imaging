"""LFADS data-preparation helpers extracted from `test_LFAD.ipynb`."""

import os

import h5py
import numpy as np
from scipy.interpolate import interp1d

from .alignment import get_perception_response


def prepare_data_for_lfads(
    neural_trials,
    target_state,
    l_frames,
    r_frames,
    output_path='lfads_data.h5',
    run_name='dataset_01',
):
    """Prepare aligned continuous dF/F data and metadata for LFADS-style models."""
    print(f'[{run_name}] Extracting aligned data for state: {target_state}...')
    extracted_data = get_perception_response(neural_trials, target_state, l_frames, r_frames)

    neu_seq = extracted_data[0]
    neu_time = extracted_data[1]
    stim_seq = extracted_data[2]
    stim_value = extracted_data[3]
    stim_time = extracted_data[4]
    decision = extracted_data[9]
    outcome = extracted_data[10]
    trial_type = extracted_data[6]

    print('Transposing data to (Trials, Time, Neurons) for LFADS compatibility...')
    neu_seq = np.transpose(neu_seq, (0, 2, 1))

    n_trials, n_time, _ = neu_seq.shape
    lfads_inputs = np.zeros((n_trials, n_time, 1))

    print('Interpolating stimulus data...')
    for i in range(n_trials):
        f_interp = interp1d(stim_time, stim_value[i], kind='linear', fill_value='extrapolate')
        lfads_inputs[i, :, 0] = f_interp(neu_time)

    indices = np.arange(n_trials)
    np.random.shuffle(indices)
    split_idx = int(0.8 * n_trials)
    train_idxs = indices[:split_idx]
    valid_idxs = indices[split_idx:]

    def split_var(arr):
        if arr is None:
            return None, None
        return arr[train_idxs], arr[valid_idxs]

    train_data, valid_data = split_var(neu_seq)
    train_ext, valid_ext = split_var(lfads_inputs)
    train_dict = {
        'inds': train_idxs,
        'stim_seq': split_var(stim_seq)[0],
        'decision': split_var(decision)[0],
        'outcome': split_var(outcome)[0],
        'trial_type': split_var(trial_type)[0],
    }
    valid_dict = {
        'inds': valid_idxs,
        'stim_seq': split_var(stim_seq)[1],
        'decision': split_var(decision)[1],
        'outcome': split_var(outcome)[1],
        'trial_type': split_var(trial_type)[1],
    }

    def h5_compatible(data):
        if data is None:
            return data
        if data.dtype.kind == 'U':
            return data.astype('S')
        return data

    print(f'Saving data to {output_path}...')
    with h5py.File(output_path, 'w') as hf:
        hf.create_dataset('train_data', data=train_data)
        hf.create_dataset('valid_data', data=valid_data)
        hf.create_dataset('train_ext_input', data=train_ext)
        hf.create_dataset('valid_ext_input', data=valid_ext)

        g_train = hf.create_group('train_meta')
        for key, value in train_dict.items():
            g_train.create_dataset(key, data=h5_compatible(value))

        g_valid = hf.create_group('valid_meta')
        for key, value in valid_dict.items():
            g_valid.create_dataset(key, data=h5_compatible(value))

        hf.create_dataset('dt', data=np.mean(np.diff(neu_time)))
        hf.create_dataset('neu_time_vec', data=neu_time)

    print('Process Complete.')
    print(f'Train Data Shape: {train_data.shape} (Trials, Time, Neurons)')
    print(f'Valid Data Shape: {valid_data.shape}')
    print(f'Output saved to: {os.path.abspath(output_path)}')
    return train_data, valid_data

