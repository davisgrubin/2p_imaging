#!/usr/bin/env python3

import numpy as np
from tqdm import tqdm
from scipy.interpolate import interp1d
from sklearn.linear_model import Ridge
from sklearn.metrics import explained_variance_score

#%% generalized linear model

# interpolate input trace to the same length as neural response.
def interp_factor_in(input_value, input_time, neu_time):
    model = interp1d(input_time, input_value, bounds_error=False)
    factor_in = model(neu_time)
    factor_in = np.nan_to_num(factor_in)
    return factor_in

# linear convolution to obtain neural response from one factor and kernel.
def get_factor_dff_neu(factor_target, kernel, l_idx, r_idx):
    ft = np.pad(factor_target, (r_idx, l_idx-1), mode='constant')
    factor_dff_neu = np.convolve(ft, kernel, mode='valid')
    return factor_dff_neu

# zero out stimulus that is not modeling target.
def filter_stimulus(factor_in, stim_label, target_labels):
    factor_target = factor_in.copy()
    # pad with 0 at both ends and compute differences
    diff = np.diff(np.concatenate(([0], factor_in, [0])))
    # edges mark onsets and offsets.
    idx_up = np.where(diff == 1)[0]
    idx_down = np.where(diff == -1)[0]
    # filtering stimulus.
    for i, (start, end) in enumerate(zip(idx_up, idx_down)):
        if stim_label[i] not in target_labels:
            factor_target[start:end] = 0
    return factor_target

# construct the input design matrix for glm.
def construct_design_matrix(factor_target, l_idx, r_idx):
    total_window = l_idx + r_idx + 2
    # pad with zeros at the beginning and end.
    padded = np.pad(factor_target, (r_idx+1, l_idx+1), mode='constant')
    # construct sliding window view.
    windows = np.lib.stride_tricks.sliding_window_view(padded, window_shape=total_window)
    factor_design_matrix = windows[:len(factor_target), :]
    return factor_design_matrix

# fit regression model.
def fit_ridge_regression(factor_target, l_idx, r_idx, dff_neu):
    factor_design_matrix = construct_design_matrix(factor_target, l_idx, r_idx)
    model = Ridge(alpha=1106)
    model.fit(factor_design_matrix, dff_neu)
    kernel_fit = np.flip(model.coef_)[1:-1]
    dff_neu_fit = get_factor_dff_neu(factor_target, kernel_fit, l_idx, r_idx)
    return kernel_fit, dff_neu_fit

# compute explain variance.
def get_exp_var(dff_neu, dff_neu_fit):
    ev = explained_variance_score(dff_neu, dff_neu_fit)
    return ev

# compute dropout score based on explained score.
def get_dropout_score(neu_seq, input_seq, w):
    0

# run glm for multiple sessions.
def run_glm_multi_sess(
        list_dff, list_neu_time,
        list_input_time, list_input_value, list_stim_labels,
        l_idx, r_idx
        ):
    # interpolate input data.
    list_factor_in = [
        interp_factor_in(iv, it, nt)
        for iv,it,nt in zip(list_input_value, list_input_time, list_neu_time)]
    # fit glm kernel across sessions.
    kernel_all = []
    exp_var_all = []
    for si in range(len(list_dff)):
        print(f'Fitting GLM for {si}/{len(list_dff)} session')
        stim_label = list_stim_labels[si][:,2]
        # get indice with stimulus.
        stim_l_idx = np.where(list_factor_in[si]==1)[0][0]-1
        stim_r_idx = np.where(list_factor_in[si]==1)[0][-1]+1
        factor_in = list_factor_in[si][stim_l_idx:stim_r_idx].copy()
        # initialize session results.
        kernel_sess = np.zeros((list_dff[si].shape[0], l_idx+r_idx))
        exp_var_sess = np.zeros((list_dff[si].shape[0]))
        # fit glm kernel for each neuron in a session.
        for ni in tqdm(range(list_dff[si].shape[0])):
            dff_neu = list_dff[si][ni,stim_l_idx:stim_r_idx]
            # fit full model.
            target_labels = [2,3,4,5,-2,-3,-4,-5,-1]
            factor_target = filter_stimulus(factor_in, stim_label, target_labels)
            kernel_fit, dff_neu_fit = fit_ridge_regression(
                factor_target, l_idx, r_idx, dff_neu)
            kernel_sess[ni,:] = kernel_fit
            exp_var_sess[ni] = get_exp_var(dff_neu, dff_neu_fit)
        # collect session results.
        kernel_all.append(kernel_sess)
        exp_var_all.append(exp_var_sess)
    kernel_all = np.concatenate(kernel_all, axis=0)
    exp_var_all = np.concatenate(exp_var_all)
    return kernel_all, exp_var_all
