#!/usr/bin/env python3

import numpy as np
from tqdm import tqdm
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.metrics import confusion_matrix
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import StratifiedShuffleSplit

from modeling.utils import norm01
from modeling.utils import get_frame_idx_from_time
from modeling.utils import get_mean_sem

# fit a line and report goodness.
def fit_poly_line(x, y, order):
    idx = ~np.isnan(y)
    coeffs = np.polyfit(x[idx], y[idx], order)
    y_pred = np.polyval(coeffs, x)
    mape = mean_absolute_percentage_error(y[idx], y_pred[idx])
    return y_pred, mape

# run validation for single trial decoding.
def decoding_evaluation(x, y):
    n_splits = 5
    test_size = 0.1
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
        win_decode, win_sample,
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
    for ti in tqdm(range(start_idx, end_idx), desc='time'):
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

# decoding time collapse and evaluate consufion matrix.
def decoding_time_confusion(neu_x, neu_time, bin_times):
    n_splits = 2
    test_size = 0.2
    bin_l_idx, bin_r_idx = get_frame_idx_from_time(neu_time, 0, 0, bin_times)
    bin_len = bin_r_idx - bin_l_idx
    # trim remainder.
    t = neu_time[:(neu_x.shape[2]//bin_len)*bin_len]
    t = np.nanmin(t.reshape(-1, bin_len), axis=1)
    x = neu_x[:,:,:(neu_x.shape[2]//bin_len)*bin_len]
    x = np.nanmean(x.reshape(x.shape[0], x.shape[1], -1, bin_len), axis=3)
    y = np.tile(np.arange(x.shape[2]), (x.shape[0], 1))
    # normalize data.
    for ni in range(x.shape[1]):
        x[:,ni,:] = norm01(x[:,ni,:].reshape(-1)).reshape(x.shape[0],x.shape[2])
    # run model.
    print('Running time decoding')
    x, y = shuffle(x, y)
    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size)
    acc_mat = np.zeros([n_splits, len(t), len(t)])
    acc_all = np.zeros(n_splits)
    for ti, (train_idx, test_idx) in tqdm(enumerate(sss.split(x, y)), desc='test'):
        # split sets.
        x_train, x_test = x[train_idx], x[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        # reshape data.
        x_train = np.transpose(x_train, [1,0,2]).reshape(x.shape[1],-1).T
        y_train = y_train.reshape(-1)
        x_test = np.transpose(x_test, [1,0,2]).reshape(x.shape[1],-1).T
        y_test = y_test.reshape(-1)
        # fit model
        model = SVC(kernel='linear')
        model.fit(x_train, y_train)
        # evaluate pairwise class.
        cm = confusion_matrix(y_test, model.predict(x_test), labels=np.arange(len(t)))
        acc = (np.diagonal(cm)[:, None] + np.diagonal(cm)[None, :]) / (np.sum(cm,axis=1)[:, None] + np.sum(cm,axis=1)[None, :] + 1e-10)
        np.fill_diagonal(acc, 0)
        acc_mat[ti,:,:] = acc
        # evaluate overall performance.
        acc_all[ti] = accuracy_score(y_test, model.predict(x_test))
    acc_mat = np.nanmean(acc_mat, axis=0)
    return t, acc_mat, acc_all

# regression from neural activity to time.
def regression_time_frac(neu_x, neu_time, bin_times, fracs):
    n_splits = 1
    n_sampling = 2
    test_size = 0.3
    bin_l_idx, bin_r_idx = get_frame_idx_from_time(neu_time, 0, 0, bin_times)
    bin_len = bin_r_idx - bin_l_idx
    # trim remainder.
    t = neu_time[:(neu_x.shape[2]//bin_len)*bin_len]
    t = np.nanmin(t.reshape(-1, bin_len), axis=1)
    x = neu_x[:,:,:(neu_x.shape[2]//bin_len)*bin_len]
    x = np.nanmean(x.reshape(x.shape[0], x.shape[1], -1, bin_len), axis=3)
    y = np.tile((np.arange(x.shape[2])+1)/x.shape[2], (x.shape[0], 1))
    # normalize data.
    for ni in range(x.shape[1]):
        x[:,ni,:] = norm01(x[:,ni,:].reshape(-1)).reshape(x.shape[0],x.shape[2])
    # create results wrt fraction of features.
    r2_all = np.zeros([n_sampling, len(fracs), n_splits])
    # run model.
    x, y = shuffle(x, y)
    print('Running decoding with fraction of features')
    for si in tqdm(range(n_sampling), desc='sampling'):
        for fi in tqdm(range(len(fracs)), desc='frac'):
            # get fraction of features.
            sub_idx = np.random.choice(x.shape[1], int(x.shape[1]*fracs[fi]), replace=False)
            x_sub = x[:,sub_idx,:].copy()
            # run cross validation.
            sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size)
            for ti, (train_idx, test_idx) in enumerate(sss.split(x, y)):
                # split sets.
                x_train, x_test = x_sub[train_idx], x_sub[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                # reshape data.
                x_train = np.transpose(x_train, (0, 2, 1)).reshape(-1, x_train.shape[1])
                y_train = y_train.reshape(-1)
                # fit model.
                model = SVR(kernel='rbf')
                model.fit(x_train, y_train)
                # test model.
                y_pred = model.predict(np.transpose(x_test,(0, 2, 1)).reshape(-1, x_test.shape[1]))
                y_pred = y_pred.reshape(y_test.shape)
                r2_all[si,fi,ti] = np.nanmean([r2_score(y_test[ti,:], y_pred[ti,:]) for ti in range(y_test.shape[0])])
    # average across folds.
    r2_all = np.nanmean(r2_all, axis=2)
    return r2_all





        