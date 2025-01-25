#!/usr/bin/env python3

import numpy as np
from scipy.interpolate import interp1d
from sklearn.metrics import r2_score

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

# build a toeplitz-like design matrix from input sequence [n_trials*n_times].
def build_design_matrix(input_seq, len_kernel):
    # pad the input sequence to allow shifting
    input_padded = np.pad(input_seq, ((0, 0), (len_kernel - 1, 0)), mode='constant')
    # use strides to efficiently create shifted views
    input_seq = np.lib.stride_tricks.sliding_window_view(
        input_padded, window_shape=len_kernel, axis=1)
    # reshape to [n_trials*n_times, len_kernel].
    input_seq = input_seq.reshape(-1, len_kernel)
    return input_seq

# fit general linear model to find kernal function.
def fit_glm(neu_seq, input_seq, len_kernel):
    # build the design matrix.
    x = build_design_matrix(input_seq, len_kernel)
    # reshape neural data for all neurons [n_trials*n_times, n_neurons].
    n_trials, n_neurons, n_times = neu_seq.shape
    y = neu_seq.transpose(1, 0, 2).reshape(neu_seq.shape[1], -1).T
    # compute pseudo inverse for analytic solution [n_neurons, len_kernel].
    w = (np.linalg.pinv(x.T @ x) @ x.T @ y).T
    return w

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




