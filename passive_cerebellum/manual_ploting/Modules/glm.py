# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 09:11:55 2026

@author: saminnaji3
"""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import h5py
import os

def save_weights(ops, weights):
    f = h5py.File(os.path.join(ops['save_path0'], 'session_data', 'linear_reg_weight.h5'), 'w')
    f['weights'] = weights
    f.close()
    
def read_weights(ops):
    f = h5py.File(os.path.join(ops['save_path0'], 'session_data', 'linear_reg_weight.h5'), 'r')
    weights = np.array(f['weights'])
    f.close()
    return weights

def save_cluster_labels(ops, labels, name = 'cluster_labels'):
    f = h5py.File(os.path.join(ops['save_path0'], 'session_data', name+ '.h5'), 'w')
    f['labels'] = labels
    f.close()
    
def read_cluster_labels(ops, name = 'cluster_labels'):
    f = h5py.File(os.path.join(ops['save_path0'], 'session_data', name+ '.h5'), 'r')
    labels = np.array(f['labels'])
    f.close()
    return labels

def make_stim_array(input_array, pre = 0, post = 0, duration = 0):
    stim = np.where(input_array == 1)[0]
    stim_end = np.where(input_array == -1)[0]
    output_array = np.zeros_like(input_array)
    for i, idx in enumerate(stim[:-1]):
        output_array[idx-pre:idx] = 1
        end_stim = np.where(stim_end > idx)[0]
        if len(end_stim) > 0:
            if stim[i+1]  > stim_end[end_stim[0]]:
                final_end = stim_end[end_stim[0]]
            else:
                final_end = idx
        else:
            final_end = idx
        if duration:
            output_array[idx:final_end] = 1
        output_array[final_end:final_end+post] = 1
        
    return output_array
        
def create_glm_input(neural_data):
    
    pre_motor = make_stim_array(neural_data['push1'], pre = 3, post = 0, duration = 0)
    pre_motor = pre_motor + make_stim_array(neural_data['push2'], pre = 3, post = 0, duration = 0)
    
    pred_reward = make_stim_array(neural_data['reward_delay'], pre = 0, post = 0, duration = 1)
    
    reward = make_stim_array(neural_data['reward'], pre = 0, post = 6, duration = 0)
    
    lick = make_stim_array(neural_data['lick'], pre = 3, post = 0)
    
    visual = make_stim_array(neural_data['vis1'], pre = 0, post = 0, duration = 1)
    visual = visual + make_stim_array(neural_data['vis2'], pre = 0, post = 0, duration = 1)
    
    auditory = make_stim_array(neural_data['retract1'], pre = 0, post = 3, duration = 0)
    auditory = auditory + make_stim_array(neural_data['retract2'], pre = 0, post = 3, duration = 0)
    
    motor = make_stim_array(neural_data['push1'], pre = 0, post = 6, duration = 0)
    motor = motor + make_stim_array(neural_data['push2'], pre = 0, post = 6, duration = 0)
    
    all_stim_arrays = np.array([pre_motor, motor, lick, visual, auditory, pred_reward, reward])
    
    print('the shape of glm input:', all_stim_arrays.shape)
    
    return all_stim_arrays


def compute_glm_weights(features, neural, ops, alpha=1):
    """
    Fit GLM for one session.
    
    Parameters
    ----------
    features : array (n_features × n_timepoints)
    neural   : array (n_neurons × n_timepoints)
    alpha    : Ridge regularization
    
    Returns
    -------
    weights  : array (n_neurons × n_features)
    """
    
    # normalize stimulus features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features.T).T
    
    n_neurons = neural.shape[0]
    n_features = features.shape[0]
    
    weights = np.zeros((n_neurons, n_features))
    
    regressor = Ridge(alpha=alpha)
    
    for n in range(n_neurons):
        regressor.fit(features_scaled.T, neural[n])
        weights[n] = regressor.coef_
        
    save_weights(ops, weights)
    
    return weights


def clustering(session_weights, n_clusters):

    # ensure list input
    if not isinstance(session_weights, list):
        session_weights = [session_weights]

    n_sessions = len(session_weights)
    n_neurons_per_session = [w.shape[0] for w in session_weights]

    # combine all neurons
    all_weights = np.concatenate(session_weights, axis=0)

    # normalize weights
    scaler = StandardScaler()
    all_weights_norm = scaler.fit_transform(all_weights)

    # run clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(all_weights_norm)

    cluster_labels = kmeans.labels_

    # determine dominant feature per cluster
    centers = kmeans.cluster_centers_
    cluster_feature = np.argmax(np.abs(centers), axis=1)

    # convert cluster labels → feature labels
    feature_labels = cluster_feature[cluster_labels]

    # split labels per session
    if n_sessions == 1:
        labels_per_session = [feature_labels]
    else:
        split_idx = np.cumsum(n_neurons_per_session[:-1])
        labels_per_session = list(np.split(feature_labels, split_idx))
        
    print(n_sessions)

    return labels_per_session