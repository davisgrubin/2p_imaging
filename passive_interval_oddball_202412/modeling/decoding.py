#!/usr/bin/env python3

import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

from modeling.utils import get_frame_idx_from_time
from modeling.utils import get_mean_sem

# sample neuron population decoding by sliding window.
def neu_pop_sample_decoding_slide_win(
        neu_x, neu_y, neu_time,
        win_decode, win_sample, win_step,
        ):
    neu_pct = 0.2
    sample_time = 10
    start_idx, end_idx = get_frame_idx_from_time(neu_time, 0, win_decode[0], win_decode[1])
    l_idx, r_idx = get_frame_idx_from_time(neu_time, 0, 0, win_sample)
    n_sample = r_idx - l_idx
    # run decoding.
    n_neu = np.max([1, int(neu_x.shape[1] * neu_pct)])
    llh_time = []
    llh = np.zeros([sample_time,(end_idx - start_idx + win_step - 1) // win_step])
    y_onehot = OneHotEncoder().fit_transform(neu_y.reshape(-1,1)).toarray()
    for ti in tqdm(range(start_idx, end_idx, win_step)):
        for si in range(sample_time):
            x = neu_x[:,np.random.choice(neu_x.shape[1], size=n_neu, replace=False), ti-n_sample:ti].copy()
            x = np.mean(x, axis=2)
            y = neu_y.copy()
            # fit model.
            model = LogisticRegression().fit(x, y)
            # test model.
            llh[si,(ti-start_idx)*win_step] = - log_loss(y_onehot, model.predict_proba(x))
        llh_time.append(ti)
    llh_time = np.array(llh_time)
    llh_mean, llh_sem = get_mean_sem(llh)
    return llh_time, llh_mean, llh_sem

# single trial decoding by sliding window.
def multi_sess_decoding_slide_win(
        neu_x, neu_y, neu_time,
        win_decode, win_sample, win_step,
        ):
    mode = 'spatial'
    n_sess = len(neu_x)
    start_idx, end_idx = get_frame_idx_from_time(neu_time, 0, win_decode[0], win_decode[1])
    l_idx, r_idx = get_frame_idx_from_time(neu_time, 0, 0, win_sample)
    n_sample = r_idx - l_idx
    # run decoding.
    acc_time   = []
    acc_model  = []
    acc_chance = []
    for i in tqdm(range(start_idx, end_idx, win_step)):
        results_model = []
        results_chance = []
        for s in range(n_sess):
            x = neu_x[s][:,:,i-n_sample:i].copy()
            y = neu_y[s].copy()
            am, ac = decoding_spatial_temporal(x, y, mode)
            results_model.append(am)
            results_chance.append(ac)
        acc_time.append(i)
        acc_model.append(np.array(results_model).reshape(-1,1))
        acc_chance.append(np.array(results_chance).reshape(-1,1))
    acc_model = np.concatenate(acc_model, axis=1)
    acc_chance = np.concatenate(acc_chance, axis=1)
    acc_time = neu_time[np.array(acc_time)]
    acc_model_mean, acc_model_sem = get_mean_sem(acc_model)
    acc_chance_mean, acc_chance_sem = get_mean_sem(acc_chance)
    return acc_time, acc_model_mean, acc_model_sem, acc_chance_mean, acc_chance_sem

# run spatial-temporal model for single trial decoding.
def decoding_spatial_temporal(x, y, mode):
    test_size = 0.2
    # split train/val/test sets.
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size)
    # define models.
    if mode == 'temporal':
        x_train = np.mean(x_train, axis=1)
        x_test  = np.mean(x_test,  axis=1)
    if mode == 'spatial':
        x_train = np.mean(x_train, axis=2)
        x_test  = np.mean(x_test,  axis=2)
    # fit model.
    model = SVC(kernel='linear')
    model.fit(x_train, y_train)
    # test model.
    acc_test = accuracy_score(
        y_test, model.predict(x_test))
    acc_shuffle = accuracy_score(
        y_test, model.predict(np.random.permutation(x_test)))
    return acc_test, acc_shuffle
