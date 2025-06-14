#!/usr/bin/env python3

import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedShuffleSplit

from modeling.utils import get_frame_idx_from_time
from modeling.utils import get_mean_sem

# fit a line and report goodness.
def fit_linear_r2(x1,x2):
    model = LinearRegression().fit(x1.reshape(-1, 1), x2.reshape(-1, 1))
    a = model.coef_[0]
    b = model.intercept_
    r2 = model.score(x1.reshape(-1, 1), x2.reshape(-1, 1))
    return a, b, r2

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
    llh_model = np.zeros([sample_time,(end_idx - start_idx + win_step - 1) // win_step])
    y_onehot = OneHotEncoder().fit_transform(neu_y.reshape(-1,1)).toarray()
    for ti in tqdm(range(start_idx, end_idx, win_step), desc='moving window'):
        for si in range(sample_time):
            x = neu_x[:,np.random.choice(neu_x.shape[1], size=n_neu, replace=False), ti-n_sample:ti].copy()
            x = np.mean(x, axis=2)
            y = neu_y.copy()
            # fit model.
            model = LogisticRegression().fit(x, y)
            # test model.
            llh_model[si,(ti-start_idx)*win_step] = - log_loss(y_onehot, model.predict_proba(x))
        llh_time.append(ti)
    llh_time = np.array(llh_time)
    llh_mean, llh_sem = get_mean_sem(llh_model)
    llh_chance = - log_loss(y_onehot, np.ones_like(y_onehot)/y_onehot.shape[1])
    llh_chance = np.array([llh_chance]).reshape(-1)
    return llh_time, llh_mean, llh_sem, llh_chance

# run validation for single trial decoding.
def decoding_evaluation(x, y):
    n_splits = 5
    test_size = 0.5
    results_model = []
    results_chance = []
    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size)
    for train_idx, test_idx in sss.split(x, y):
        # split sets.
        x_train, x_test = x[train_idx], x[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        # fit model.
        model = SVC(kernel='linear', probability=True)
        model.fit(x_train, y_train)
        # test model.
        results_model.append(model.score(x_test, y_test))
        results_chance.append(model.score(x_test, np.random.permutation(y_test)))
    return results_model, results_chance

# single trial decoding by sliding window.
def multi_sess_decoding_slide_win(
        neu_x, neu_time,
        win_decode, win_sample, win_step,
        ):
    n_sess = len(neu_x[0])
    start_idx, end_idx = get_frame_idx_from_time(neu_time, 0, win_decode[0], win_decode[1])
    l_idx, r_idx = get_frame_idx_from_time(neu_time, 0, 0, win_sample)
    n_sample = r_idx - l_idx
    # run decoding.
    acc_time   = []
    acc_model  = []
    acc_chance = []
    print('Running decoding with slide window')
    for ti in tqdm(range(start_idx, end_idx, win_step), desc='time'):
        results_model = []
        results_chance = []
        # decoding each session.
        for si in range(n_sess):
            if neu_x[0][si].shape[0] >= 2 and neu_x[0][si].shape[1] >=1 :
                # average within sliding window.
                x = [np.nanmean(neu_x[ci][si][:,:,ti-n_sample:ti], axis=2) for ci in range(len(neu_x))]
                x = np.concatenate(x, axis=0)
                # create corresponding labels.
                y = [np.ones(neu_x[ci][si].shape[0])*ci for ci in range(len(neu_x))]
                y = np.concatenate(y, axis=0)
                # run decoding.
                rm, rc = decoding_evaluation(x, y)
                results_model.append(rm)
                results_chance.append(rc)   
        acc_time.append(ti)
        acc_model.append(np.array(results_model).reshape(-1,1))
        acc_chance.append(np.array(results_chance).reshape(-1,1))
    acc_model = np.concatenate(acc_model, axis=1)
    acc_chance = np.concatenate(acc_chance, axis=1)
    acc_time = neu_time[np.array(acc_time)]
    acc_model_mean, acc_model_sem = get_mean_sem(acc_model)
    acc_chance_mean, acc_chance_sem = get_mean_sem(acc_chance)
    return acc_time, acc_model_mean, acc_model_sem, acc_chance_mean, acc_chance_sem

# decoding time collapse.
def multi_sess_decoding_time(
        neu_x, neu_time, win_decode
        ):
    n_sess = len(neu_x[0])
    start_idx, end_idx = get_frame_idx_from_time(neu_time, 0, win_decode[0], win_decode[1])
    # run decoding.
    results_model = []
    # run decoding for each condition.
    for ci in range(len(neu_x)):
        rm_vi = []
        for si in range(n_sess):
            if neu_x[0][si].shape[0] >= 2 and neu_x[0][si].shape[1] >=1 :
                # take data within range.
                x = [neu_x[ci][si][:,:,start_idx:end_idx] for ci in range(len(neu_x))]
                y = neu_time[start_idx:end_idx]
                sss = StratifiedShuffleSplit(n_splits=20, test_size=0.5)
                # create input data.
                x_ci = x[ci].copy().transpose(0, 2, 1).reshape(-1, x[ci].shape[1])
                y_ci = np.tile(y, x[ci].shape[0])
                for train_idx, test_idx in sss.split(x_ci, y_ci):
                    # split sets.
                    x_train, x_test = x_ci[train_idx], x_ci[test_idx]
                    y_train, y_test = y_ci[train_idx], y_ci[test_idx]
                    # fit model.
                    model = LinearRegression()
                    model.fit(x_train, y_train)
                    # test model.
                    rm_vi.append(r2_score(model.predict(x_test), y_test))
        rm_vi = np.array(rm_vi).reshape(-1,1)
        results_model.append(rm_vi)
    results_model = np.concatenate(results_model, axis=1)
    model_mean, model_sem = get_mean_sem(results_model)

    

            
            

