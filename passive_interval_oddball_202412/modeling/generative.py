#!/usr/bin/env python3

import numpy as np
from scipy.interpolate import interp1d
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge

# interpolate input trace to the same length as neural response.
def interp_input_seq(input_value, input_time, neu_time):
    # initialize input sequence [n_trials*n_times].
    input_seq = np.zeros((input_value.shape[0], len(neu_time)))
    # inerpolation for each trial.
    for i in range(input_value.shape[0]):
        model = interp1d(input_time, input_value[i,:], bounds_error=False)
        input_seq[i,:] = model(neu_time)
    input_seq = np.nan_to_num(input_seq)
    return input_seq
def filter_stimulus(factor_in, stim_label, target_labels):
    # Copy to avoid modifying the original
    factor_target = factor_in.copy()
    # Pad with 0 at both ends and compute differences
    diff = np.diff(np.concatenate(([0], factor_in, [0])))
    # Rising edges mark onsets; falling edges mark offsets
    onsets = np.where(diff == 1)[0]
    offsets = np.where(diff == -1)[0]
    # Iterate over blocks: if label not in target, zero out that block
    for i, (start, end) in enumerate(zip(onsets, offsets)):
        if stim_label[i] not in target_labels:
            factor_target[start:end] = 0
    return factor_target


def construct_design_matrix(factor_target, l_idx, r_idx):
    n = len(factor_target)
    total_window = l_idx + r_idx
    # Pad factor_target with zeros at the beginning and end
    padded = np.pad(factor_target, (l_idx, r_idx), mode='constant')
    # sliding_window_view returns a view of shape (len(padded)-total_window+1, total_window)
    windows = np.lib.stride_tricks.sliding_window_view(padded, window_shape=total_window)
    # Take the first n windows so that the design matrix has shape (n, total_window)
    factor_design_matrix = windows[:n, :]
    return factor_design_matrix


def fit_ridge_regression(X, dff_neu):
    # Fit ridge regression to recover the kernel
    model = Ridge(alpha=1.0)
    model.fit(X, dff_neu)
    kernel_fit = model.coef_
    dff_neu_fit = model.predict(X)
    return kernel_fit, dff_neu_fit



# compute coding score based on reconstruction r2.
def get_coding_score(neu_seq, input_seq, w):
    # build the design matrix.
    x = build_design_matrix(input_seq, w.shape[1])
    # reshape neural response.
    y = neu_seq.transpose(1, 0, 2).reshape(neu_seq.shape[1], -1).T
    # compute prediction.
    pred = x @ w.T
    # calculate r2 scores for each neuron.
    r2 = np.array([r2_score(y[:,i], pred[:,i]) for i in range(y.shape[1])])
    return r2

# compute explain variance.
def get_explain_variance(neu_seq, input_seq, w):
    e = 1e-10
    # build the design matrix.
    x = build_design_matrix(input_seq, w.shape[1])
    # reshape neural response.
    y = neu_seq.transpose(1, 0, 2).reshape(neu_seq.shape[1], -1).T
    # compute prediction.
    pred = x @ w.T
    # calculate explain varaiance for each neuron.
    exvar = np.array([np.nanvar(pred[:,i]) / (np.nanvar(y[:,i])+e) for i in range(y.shape[1])])
    return exvar




