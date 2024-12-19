#!/usr/bin/env python3

import numpy as np
from tqdm import tqdm

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# single trial decoding by sampling subset of neurons. 
def multi_sess_decoding_num_neu(
        neu_x, neu_y,
        num_step, n_decode,
        mode
        ):
    n_sess = len(neu_x)
    # define sampling numbers.
    max_num = np.nanmax([neu_x[i].shape[1] for i in range(n_sess)])
    sampling_nums = np.arange(num_step, ((max_num//num_step)+1)*num_step, num_step)
    # run decoding.
    acc_model  = []
    acc_chance = []
    for n_neu in tqdm(sampling_nums):
        results_model = []
        results_chance = []
        for s in range(n_sess):
            # not enough neurons.
            if n_neu > neu_x[s].shape[1]:
                results_model.append(np.nan)
                results_chance.append(np.nan)
            # random sampling n_decode times.
            else:
                for _ in range(n_decode):
                    sub_idx = np.random.choice(neu_x[s].shape[1], n_neu, replace=False)
                    x = neu_x[s][:,sub_idx].copy()
                    y = neu_y[s].copy()
                    am, ac = decoding_spatial_temporal(x, y, mode)
                    results_model.append(am)
                    results_chance.append(ac)
        acc_model.append(np.array(results_model).reshape(-1,1))
        acc_chance.append(np.array(results_chance).reshape(-1,1))
    return sampling_nums, acc_model, acc_chance

# single trial decoding by sliding window.
def multi_sess_decoding_slide_win(
        neu_x, neu_y,
        start_idx, end_idx, win_step,
        n_decode, num_frames,
        ):
    mode = 'spatial'
    n_sess = len(neu_x)
    # run decoding.
    acc_model  = []
    acc_chance = []
    for i in tqdm(range(start_idx, end_idx, win_step)):
        results_model = []
        results_chance = []
        for s in range(n_sess):
            x = neu_x[s][:,:,i-num_frames:i].copy()
            y = neu_y[s].copy()
            am, ac = decoding_spatial_temporal(x, y, mode)
            results_model.append(am)
            results_chance.append(ac)
        acc_model.append(np.array(results_model).reshape(-1,1))
        acc_chance.append(np.array(results_chance).reshape(-1,1))
    return acc_model, acc_chance

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
