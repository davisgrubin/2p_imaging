#!/usr/bin/env python3

import numpy as np
from tqdm import tqdm
from scipy.interpolate import interp1d
from sklearn.metrics import explained_variance_score

#%% generalized linear model

# interpolate input trace to the same length as neural response.
def interp_factor_in(input_value, input_time, neu_time):
    model = interp1d(input_time, input_value, bounds_error=False, fill_value=0)
    return model(neu_time)

# linear convolution to obtain neural response from one factor and kernel.
def get_factor_dff_neu(factor_target, kernel, l_idx, r_idx):
    padded = np.pad(factor_target, (r_idx, l_idx-1), mode='constant')
    return np.convolve(padded, kernel, mode='valid')

# zero out stimulus that is not modeling target.
def filter_stimulus(factor_in, stim_label, target_labels):
    factor_target = factor_in.copy()
    # compute differences to detect onsets and offsets.
    diff = np.diff(np.concatenate(([0], factor_in, [0])))
    idx_up = np.where(diff == 1)[0]
    idx_down = np.where(diff == -1)[0]
    # iterate over stimulus segments to zero out non-targets.
    for i, (start, end) in enumerate(zip(idx_up, idx_down)):
        if stim_label[i] not in target_labels:
            factor_target[start:end] = 0
    return factor_target

# construct the input design matrix for glm.
def construct_design_matrix(factor_target, l_idx, r_idx):
    total_window = l_idx + r_idx + 2
    padded = np.pad(factor_target, (r_idx+1, l_idx+1), mode='constant')
    windows = np.lib.stride_tricks.sliding_window_view(padded, window_shape=total_window)
    return windows[:len(factor_target), :]

# fit regression model.
def run_glm_multi_sess(
        list_dff, list_neu_time,
        list_input_time, list_input_value, list_stim_labels,
        l_idx, r_idx, alpha=1106):
    target_labels = [2, 3, 4, 5, -2, -3, -4, -5, -1]
    # pre-interpolate input data per session.
    list_factor_in = [
        interp_factor_in(iv, it, nt)
        for iv, it, nt in zip(list_input_value, list_input_time, list_neu_time)]
    kernel_all = []
    exp_var_all = []
    for si in range(len(list_dff)):
        print(f'Fitting GLM for session {si+1}/{len(list_dff)}')
        stim_label = list_stim_labels[si][:, 2]
        session_factor_in = list_factor_in[si]
        # find indices where the stimulus is active.
        stim_indices = np.where(session_factor_in == 1)[0]
        stim_l_idx = stim_indices[0] - 1
        stim_r_idx = stim_indices[-1] + 1
        # extract the relevant part of the stimulus and filter it.
        factor_in_session = session_factor_in[stim_l_idx:stim_r_idx]
        factor_target = filter_stimulus(factor_in_session, stim_label, target_labels)
        # precompute the design matrix for the current session.
        design_matrix = construct_design_matrix(factor_target, l_idx, r_idx)
        total_window = l_idx + r_idx + 2
        # precompute the pseudo inverse for ridge regression:
        A = design_matrix.T @ design_matrix + alpha * np.eye(total_window)
        M = np.linalg.solve(A, design_matrix.T)
        n_neurons = list_dff[si].shape[0]
        kernel_sess = np.zeros((n_neurons, total_window - 2))
        exp_var_sess = np.zeros(n_neurons)
        # fit the ridge regression for each neuron.
        for ni in tqdm(range(n_neurons), desc="Neurons"):
            dff_neu = list_dff[si][ni, stim_l_idx:stim_r_idx]
            # ridge solution for this neuron.
            beta = M @ dff_neu
            # flip and trim the coefficient vector to obtain the kernel.
            kernel_fit = np.flip(beta)[1:-1]
            # compute the predicted neural response using the kernel.
            dff_neu_fit = get_factor_dff_neu(factor_target, kernel_fit, l_idx, r_idx)
            kernel_sess[ni,:] = kernel_fit
            exp_var_sess[ni] = explained_variance_score(dff_neu, dff_neu_fit)
        # collect results.
        kernel_all.append(kernel_sess)
        exp_var_all.append(exp_var_sess)
    kernel_all = np.concatenate(kernel_all, axis=0)
    exp_var_all = np.concatenate(exp_var_all)
    return kernel_all, exp_var_all
