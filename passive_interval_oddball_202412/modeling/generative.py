#!/usr/bin/env python3

import numpy as np
from tqdm import tqdm
from scipy.interpolate import interp1d
from sklearn.metrics import explained_variance_score

from sklearn.decomposition import PCA
from scipy.linalg import orthogonal_procrustes

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
    if stim_label[-1] == -1:
        idx_up = idx_up[:-1]
        idx_down = idx_down[:-1]
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
        l_idx, r_idx):
    alpha = 52025
    target_labels = [2, 3, 4, 5, -2, -3, -4, -5, -1]
    # interpolate inputs per session.
    list_factor_in = [
        interp_factor_in(iv, it, nt)
        for iv, it, nt in zip(list_input_value, list_input_time, list_neu_time)
    ]
    # pre-allocate result arrays.
    n_neurons = np.nansum([dff.shape[0] for dff in list_dff])
    n_times = l_idx + r_idx
    kernel_all = np.zeros([n_neurons, n_times])
    #exp_var_all = np.zeros([n_neurons])
    # loop sessions and fill slices.
    write_idx = 0
    for si, (dff_sess, factor_in_sess) in enumerate(zip(list_dff, list_factor_in)):
        print(f'Fitting GLM for session {si+1}/{len(list_dff)}')
        stim_label = list_stim_labels[si][:, 2]
        # find stimulus window in this session.
        stim_inds = np.where(factor_in_sess == 1)[0]
        left  = stim_inds[0] - 5
        right = stim_inds[-1] + 5
        # crop and filter to get design targets.
        factor_in = factor_in_sess[left:right]
        f_target  = filter_stimulus(factor_in, stim_label, target_labels)
        # build design matrix and precompute ridge pseudo-inverse.
        X = construct_design_matrix(f_target, l_idx, r_idx)
        TW = l_idx + r_idx + 2
        A = X.T @ X + alpha * np.eye(TW)
        M = np.linalg.solve(A, X.T)
        n_neurons = dff_sess.shape[0]
        # for each neuron solve and write into the big arrays.
        for ni in tqdm(range(n_neurons), desc='neurons'):
            y    = dff_sess[ni, left:right]
            beta = M @ y
            kern = np.flip(beta)[1:-1]  # length = kernel_length
            #y_hat = get_factor_dff_neu(f_target, kern, l_idx, r_idx)
            kernel_all[write_idx + ni, :] = kern
            #exp_var_all[write_idx + ni]   = explained_variance_score(y, y_hat)
        write_idx += n_neurons
    #return kernel_all, exp_var_all
    return kernel_all

# retrieve glm kernels for category.
def get_glm_cate(glm, list_labels, cate):
    idx = np.concatenate([np.in1d(list_labels[i],cate)
           for i in range(len(list_labels))])
    kernel_all = glm['kernel_all'][idx,:]
    return kernel_all

    
