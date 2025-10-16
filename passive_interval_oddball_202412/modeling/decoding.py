#!/usr/bin/env python3

import numpy as np
from tqdm import tqdm
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.svm import SVR
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support

import pandas as pd

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
    n_splits = 25
    test_size = 0.1
    results_model = []
    results_chance = []
    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size)
    for train_idx, test_idx in sss.split(x, y):
        # split sets.
        x_train, x_test = x[train_idx], x[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        # fit model.
        model = SVC(kernel="linear", probability=True)
        model.fit(x_train, y_train)
        # test model.
        results_model.append(model.score(x_test, y_test))
        results_chance.append(model.score(x_test, np.random.permutation(y_test)))
    return results_model, results_chance


# single trial decoding by sliding window.
def multi_sess_decoding_slide_win(
    neu_x,
    neu_time,
    win_decode,
    win_sample,
):
    n_sess = len(neu_x[0])
    start_idx, end_idx = get_frame_idx_from_time(
        neu_time, 0, win_decode[0], win_decode[1]
    )
    l_idx, r_idx = get_frame_idx_from_time(neu_time, 0, 0, win_sample)
    n_sample = r_idx - l_idx
    # run decoding.
    acc_time = []
    acc_model = []
    acc_chance = []
    print("Running decoding with slide window")
    for ti in tqdm(range(start_idx, end_idx), desc="time"):
        results_model = []
        results_chance = []
        # decoding each session.
        for si in range(n_sess):
            if neu_x[0][si].shape[0] >= 2 and neu_x[0][si].shape[1] >= 1:
                # average within sliding window.
                x = [
                    np.nanmean(neu_x[ci][si][:, :, ti - n_sample : ti], axis=2)
                    for ci in range(len(neu_x))
                ]
                x = np.concatenate(x, axis=0)
                # create corresponding labels.
                y = [np.ones(neu_x[ci][si].shape[0]) * ci for ci in range(len(neu_x))]
                y = np.concatenate(y, axis=0)
                # run decoding.
                rm, rc = decoding_evaluation(x, y)
                results_model.append(rm)
                results_chance.append(rc)
        acc_time.append(ti)
        acc_model.append(np.array(results_model).reshape(-1, 1))
        acc_chance.append(np.array(results_chance).reshape(-1, 1))
    acc_model = np.concatenate(acc_model, axis=1)
    acc_chance = np.concatenate(acc_chance, axis=1)
    acc_time = neu_time[np.array(acc_time)]
    acc_model_mean, acc_model_sem = get_mean_sem(acc_model)
    acc_chance_mean, acc_chance_sem = get_mean_sem(acc_chance)
    return acc_time, acc_model_mean, acc_model_sem, acc_chance_mean, acc_chance_sem


# decoding time collapse and evaluate confusion matrix.
def decoding_time_confusion(neu_x, neu_time, bin_times):
    n_splits = 50
    test_size = 0.2
    bin_l_idx, bin_r_idx = get_frame_idx_from_time(neu_time, 0, 0, bin_times)
    bin_len = bin_r_idx - bin_l_idx
    # trim remainder.
    t = neu_time[: (neu_x.shape[2] // bin_len) * bin_len]
    t = np.nanmin(t.reshape(-1, bin_len), axis=1)
    x = neu_x[:, :, : (neu_x.shape[2] // bin_len) * bin_len]
    x = np.nanmean(x.reshape(x.shape[0], x.shape[1], -1, bin_len), axis=3)
    y = np.tile(np.arange(x.shape[2]), (x.shape[0], 1))
    # normalize data.
    for ni in range(x.shape[1]):
        x[:, ni, :] = norm01(x[:, ni, :].reshape(-1)).reshape(x.shape[0], x.shape[2])
    # run model.
    print("Running pairwise time decoding")
    x, y = shuffle(x, y)
    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size)
    acc_model = np.zeros([n_splits, len(t), len(t)])
    acc_shuffle = np.zeros([n_splits, len(t), len(t)])
    for ti, (train_idx, test_idx) in tqdm(enumerate(sss.split(x, y)), desc="test"):
        # split sets.
        x_train, x_test = x[train_idx], x[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        # reshape data.
        x_train = np.transpose(x_train, [1, 0, 2]).reshape(x.shape[1], -1).T
        y_train = y_train.reshape(-1)
        x_test = np.transpose(x_test, [1, 0, 2]).reshape(x.shape[1], -1).T
        y_test = y_test.reshape(-1)
        # evaluate pairwise class.
        n_classes = len(t)
        a_model = np.zeros((n_classes, n_classes))
        a_shuffle = np.zeros((n_classes, n_classes))
        # precompute per-class indices for efficiency
        train_idx_per_cls = [np.where(y_train == k)[0] for k in range(n_classes)]
        test_idx_per_cls = [np.where(y_test == k)[0] for k in range(n_classes)]
        for i in range(n_classes):
            ti_idx_tr = train_idx_per_cls[i]
            ti_idx_te = test_idx_per_cls[i]
            if ti_idx_tr.size == 0 or ti_idx_te.size == 0:
                continue
            for j in range(i + 1, n_classes):
                tj_idx_tr = train_idx_per_cls[j]
                tj_idx_te = test_idx_per_cls[j]
                if tj_idx_tr.size == 0 or tj_idx_te.size == 0:
                    continue
                tr_idx = np.concatenate([ti_idx_tr, tj_idx_tr])
                te_idx = np.concatenate([ti_idx_te, tj_idx_te])
                # normal model
                model = LinearSVC()
                model.fit(x_train[tr_idx], y_train[tr_idx])
                y_pred = model.predict(x_test[te_idx])
                a = np.mean(y_pred == y_test[te_idx])
                a_model[i, j] = a_model[j, i] = a
                # shuffled labels
                y_train_shuf = np.random.permutation(y_train[tr_idx])
                model.fit(x_train[tr_idx], y_train_shuf)
                y_pred_shuf = model.predict(x_test[te_idx])
                a_shuf = np.mean(y_pred_shuf == y_test[te_idx])
                a_shuffle[i, j] = a_shuffle[j, i] = a_shuf
        np.fill_diagonal(a_model, 0)
        np.fill_diagonal(a_shuffle, 0)
        acc_model[ti, :, :] = a_model
        acc_shuffle[ti, :, :] = a_shuffle
    return t, acc_model, acc_shuffle


# -------------------------------------------------------------------------
# New binary decoding helpers (simple ISI and time-binned)
# -------------------------------------------------------------------------


def _mean_sem(arr):
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        return np.nan, np.nan
    if arr.size == 1:
        return float(arr[0]), np.nan
    return float(arr.mean()), float(arr.std(ddof=1) / np.sqrt(arr.size))


def _downsample_binary(X, y, rng):
    counts = np.bincount(y, minlength=2)
    maj = int(np.argmax(counts))
    minc = 1 - maj
    if counts[minc] == 0 or counts[maj] <= counts[minc]:
        return X, y
    maj_idx = np.where(y == maj)[0]
    keep = rng.choice(maj_idx, counts[minc], replace=False)
    min_idx = np.where(y == minc)[0]
    sel = np.concatenate([keep, min_idx])
    return X[sel], y[sel]


def binary_decoder_cv(
    X,
    y,
    n_splits=5,
    random_state=0,
    shuffle_baseline=False,
    downsample=True,
    class_weight="balanced",
    progress_desc=None,
):
    X = np.asarray(X)
    y = np.asarray(y).astype(int)
    if X.ndim != 2:
        raise ValueError("X must be 2D (trials × features)")
    if np.unique(y).size < 2:
        return None

    counts = np.bincount(y)
    min_class = counts[counts > 0].min()
    if min_class < 2:
        return None
    n_splits = min(n_splits, int(min_class))
    if n_splits < 2:
        return None

    skf = StratifiedShuffleSplit(n_splits=n_splits, random_state=random_state)

    accs, precisions, recalls, f1s = [], [], [], []
    chances = []
    conf_total = np.zeros((2, 2), dtype=int)

    iterator = skf.split(X, y)
    if progress_desc is not None:
        iterator = tqdm(iterator, total=n_splits, desc=progress_desc, leave=False)
    for fold_idx, (train_idx, test_idx) in enumerate(iterator):
        X_train, X_test = X[train_idx].copy(), X[test_idx]
        y_train, y_test = y[train_idx].copy(), y[test_idx]

        if downsample:
            rng = np.random.default_rng(random_state + fold_idx)
            X_train, y_train = _downsample_binary(X_train, y_train, rng)

        model = make_pipeline(
            LinearSVC(C=1.0, max_iter=5000, class_weight=class_weight),
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accs.append(float(np.mean(y_pred == y_test)))
        p, r, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="binary", labels=[0, 1], zero_division=0
        )
        precisions.append(float(p))
        recalls.append(float(r))
        f1s.append(float(f1))
        conf_total += confusion_matrix(y_test, y_pred, labels=[0, 1])

        if shuffle_baseline:
            rng = np.random.default_rng(random_state + 1000 + fold_idx)
            y_train_perm = rng.permutation(y_train)
            model_shuf = make_pipeline(
                LinearSVC(C=1.0, max_iter=5000, class_weight=class_weight),
            )
            model_shuf.fit(X_train, y_train_perm)
            y_pred_shuf = model_shuf.predict(X_test)
            chances.append(float(np.mean(y_pred_shuf == y_test)))
        else:
            chances.append(np.nan)

    acc_mean, acc_sem = _mean_sem(accs)
    prec_mean, prec_sem = _mean_sem(precisions)
    rec_mean, rec_sem = _mean_sem(recalls)
    f1_mean, f1_sem = _mean_sem(f1s)

    if shuffle_baseline and np.isfinite(chances).any():
        mean_chance = float(np.nanmean(chances))
        valid = np.isfinite(chances)
        sem_chance = (
            float(np.nanstd(np.asarray(chances)[valid], ddof=1) / np.sqrt(valid.sum()))
            if valid.sum() > 1
            else np.nan
        )
    else:
        mean_chance = np.nan
        sem_chance = np.nan

    return {
        "fold_accuracies": np.asarray(accs),
        "fold_precision": np.asarray(precisions),
        "fold_recall": np.asarray(recalls),
        "fold_f1": np.asarray(f1s),
        "fold_chance": np.asarray(chances, dtype=float),
        "mean_acc": acc_mean,
        "sem_acc": acc_sem,
        "precision_mean": prec_mean,
        "precision_sem": prec_sem,
        "recall_mean": rec_mean,
        "recall_sem": rec_sem,
        "f1_mean": f1_mean,
        "f1_sem": f1_sem,
        "mean_chance": mean_chance,
        "sem_chance": sem_chance,
        "confusion_matrix": conf_total,
        "n_splits": len(accs),
    }


def binary_timecourse_decoding(
    trial_neu,
    neu_time,
    y,
    block_labels=None,
    bin_width_ms=100.0,
    window=(-500.0, 2000.0),
    n_splits=5,
    random_state=0,
    shuffle_baseline=False,
    downsample=True,
):
    trial_neu = np.asarray(trial_neu)
    neu_time = np.asarray(neu_time)
    y = np.asarray(y).astype(int)

    if trial_neu.ndim != 3:
        raise ValueError("trial_neu must be (trials × neurons × time)")
    if np.unique(y).size < 2:
        return pd.DataFrame()

    if block_labels is None:
        block_labels = np.zeros(trial_neu.shape[0], dtype=int)
    else:
        block_labels = np.asarray(block_labels)

    bin_edges = np.arange(window[0], window[1], bin_width_ms)
    records = []

    for block in np.unique(block_labels):
        block_mask = block_labels == block
        if block_mask.sum() < max(4, n_splits):
            continue
        y_block = y[block_mask]
        if np.unique(y_block).size < 2:
            continue
        neu_block = trial_neu[block_mask]

        for left in bin_edges:
            right = left + bin_width_ms
            l_idx, r_idx = get_frame_idx_from_time(neu_time, 0, left, right)
            r_idx = min(r_idx, neu_block.shape[-1])
            if r_idx - l_idx < 2:
                continue

            window_data = neu_block[:, :, l_idx:r_idx]
            if np.all(np.isnan(window_data)):
                continue

            X = np.nanmean(window_data, axis=-1)
            if np.any(~np.isfinite(X)):
                continue

            metrics = binary_decoder_cv(
                X,
                y_block,
                n_splits=n_splits,
                random_state=random_state,
                shuffle_baseline=shuffle_baseline,
                downsample=downsample,
            )
            if metrics is None:
                continue

            records.append(
                {
                    "block_type": int(block),
                    "time_start_ms": float(left),
                    "time_center_ms": float(left + bin_width_ms / 2.0),
                    "time_stop_ms": float(right),
                    "accuracy_mean": metrics["mean_acc"],
                    "accuracy_sem": metrics["sem_acc"],
                    "chance_mean": metrics["mean_chance"],
                    "chance_sem": metrics["sem_chance"],
                    "precision_mean": metrics["precision_mean"],
                    "precision_sem": metrics["precision_sem"],
                    "recall_mean": metrics["recall_mean"],
                    "recall_sem": metrics["recall_sem"],
                    "f1_mean": metrics["f1_mean"],
                    "f1_sem": metrics["f1_sem"],
                    "n_splits": metrics["n_splits"],
                }
            )

    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)


def multiclass_decoder_cv(
    X,
    y,
    n_splits=20,
    random_state=0,
    shuffle_baseline=False,
    class_weight="balanced",
    progress_desc=None,
):
    X = np.asarray(X)
    y = np.asarray(y)
    if X.ndim != 2:
        raise ValueError("X must be 2D (trials × features)")
    classes, counts = np.unique(y, return_counts=True)
    if classes.size < 2:
        return None
    min_count = counts.min()
    if min_count < 2:
        return None
    n_splits = min(n_splits, int(min_count))
    if n_splits < 2:
        return None

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    accs, precisions, recalls, f1s = [], [], [], []
    chances = []
    conf_total = np.zeros((classes.size, classes.size), dtype=int)

    iterator = skf.split(X, y)
    if progress_desc is not None:
        iterator = tqdm(iterator, total=n_splits, desc=progress_desc, leave=False)
    for fold_idx, (train_idx, test_idx) in enumerate(iterator):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = make_pipeline(
            StandardScaler(with_mean=True),
            LinearSVC(C=1.0, max_iter=5000, class_weight=class_weight),
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accs.append(float(np.mean(y_pred == y_test)))
        p, r, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="macro", zero_division=0
        )
        precisions.append(float(p))
        recalls.append(float(r))
        f1s.append(float(f1))
        conf_total += confusion_matrix(y_test, y_pred, labels=classes)

        if shuffle_baseline:
            rng = np.random.default_rng(random_state + 5000 + fold_idx)
            y_perm = rng.permutation(y_train)
            model_perm = make_pipeline(
                StandardScaler(with_mean=True),
                LinearSVC(C=1.0, max_iter=5000, class_weight=class_weight),
            )
            model_perm.fit(X_train, y_perm)
            y_perm_pred = model_perm.predict(X_test)
            chances.append(float(np.mean(y_perm_pred == y_test)))
        else:
            chances.append(np.nan)

    acc_mean, acc_sem = _mean_sem(accs)
    prec_mean, prec_sem = _mean_sem(precisions)
    rec_mean, rec_sem = _mean_sem(recalls)
    f1_mean, f1_sem = _mean_sem(f1s)

    if shuffle_baseline and np.isfinite(chances).any():
        mean_chance = float(np.nanmean(chances))
        valid = np.isfinite(chances)
        sem_chance = (
            float(np.nanstd(np.asarray(chances)[valid], ddof=1) / np.sqrt(valid.sum()))
            if valid.sum() > 1
            else np.nan
        )
    else:
        mean_chance = np.nan
        sem_chance = np.nan

    return {
        "classes": classes.tolist(),
        "fold_accuracies": np.asarray(accs),
        "fold_precision": np.asarray(precisions),
        "fold_recall": np.asarray(recalls),
        "fold_f1": np.asarray(f1s),
        "fold_chance": np.asarray(chances, dtype=float),
        "mean_acc": acc_mean,
        "sem_acc": acc_sem,
        "precision_mean": prec_mean,
        "precision_sem": prec_sem,
        "recall_mean": rec_mean,
        "recall_sem": rec_sem,
        "f1_mean": f1_mean,
        "f1_sem": f1_sem,
        "mean_chance": mean_chance,
        "sem_chance": sem_chance,
        "confusion_matrix": conf_total,
        "n_splits": len(accs),
    }


def multiclass_timecourse_decoding(
    trial_neu,
    neu_time,
    y,
    block_labels=None,
    bin_width_ms=100.0,
    window=(-500.0, 2000.0),
    n_splits=20,
    random_state=0,
    shuffle_baseline=False,
):
    trial_neu = np.asarray(trial_neu)
    neu_time = np.asarray(neu_time)
    y = np.asarray(y)

    if trial_neu.ndim != 3:
        raise ValueError("trial_neu must be (trials × neurons × time)")
    if np.unique(y).size < 2:
        return pd.DataFrame()

    if block_labels is None:
        block_labels = np.zeros(trial_neu.shape[0], dtype=int)
    else:
        block_labels = np.asarray(block_labels)

    bin_edges = np.arange(window[0], window[1], bin_width_ms)
    records = []

    for block in np.unique(block_labels):
        block_mask = block_labels == block
        if block_mask.sum() < max(4, n_splits):
            continue
        y_block = y[block_mask]
        if np.unique(y_block).size < 2:
            continue
        neu_block = trial_neu[block_mask]

        for left in bin_edges:
            right = left + bin_width_ms
            l_idx, r_idx = get_frame_idx_from_time(neu_time, 0, left, right)
            r_idx = min(r_idx, neu_block.shape[-1])
            if r_idx - l_idx < 2:
                continue

            window_data = neu_block[:, :, l_idx:r_idx]
            if np.all(np.isnan(window_data)):
                continue

            X = np.nanmean(window_data, axis=-1)
            if np.any(~np.isfinite(X)):
                continue

            metrics = multiclass_decoder_cv(
                X,
                y_block,
                n_splits=n_splits,
                random_state=random_state,
                shuffle_baseline=shuffle_baseline,
            )
            if metrics is None:
                continue

            records.append(
                {
                    "block_type": int(block),
                    "time_start_ms": float(left),
                    "time_center_ms": float(left + bin_width_ms / 2.0),
                    "time_stop_ms": float(right),
                    "accuracy_mean": metrics["mean_acc"],
                    "accuracy_sem": metrics["sem_acc"],
                    "chance_mean": metrics["mean_chance"],
                    "chance_sem": metrics["sem_chance"],
                    "n_splits": metrics["n_splits"],
                }
            )

    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)


# decoding single time point from the rest.
def decoding_time_single(neu_x, neu_time, bin_times):
    n_splits = 50
    test_size = 0.2
    bin_l_idx, bin_r_idx = get_frame_idx_from_time(neu_time, 0, 0, bin_times)
    bin_len = bin_r_idx - bin_l_idx
    # trim remainder.
    t = neu_time[: (neu_x.shape[2] // bin_len) * bin_len]
    t = np.nanmin(t.reshape(-1, bin_len), axis=1)
    x = neu_x[:, :, : (neu_x.shape[2] // bin_len) * bin_len]
    x = np.nanmean(x.reshape(x.shape[0], x.shape[1], -1, bin_len), axis=3)
    y = np.tile(np.arange(x.shape[2]), (x.shape[0], 1))
    # normalize data.
    for ni in range(x.shape[1]):
        x[:, ni, :] = norm01(x[:, ni, :].reshape(-1)).reshape(x.shape[0], x.shape[2])
    # run model.
    print("Running single time decoding")
    x, y = shuffle(x, y)
    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size)
    acc_model = np.zeros([n_splits, len(t)])
    acc_shuffle = np.zeros([n_splits, len(t)])
    for ti, (train_idx, test_idx) in tqdm(enumerate(sss.split(x, y)), desc="test"):
        # split sets.
        x_train, x_test = x[train_idx], x[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        # reshape data.
        x_train = np.transpose(x_train, [1, 0, 2]).reshape(x.shape[1], -1).T
        y_train = y_train.reshape(-1)
        x_test = np.transpose(x_test, [1, 0, 2]).reshape(x.shape[1], -1).T
        y_test = y_test.reshape(-1)
        # evaluate one-vs-rest decoding.
        n_classes = len(t)
        a_model = np.zeros(n_classes)
        a_shuffle = np.zeros(n_classes)
        for ci in range(n_classes):
            y_train_bin = (y_train == ci).astype(int)
            y_test_bin = (y_test == ci).astype(int)
            if len(np.unique(y_train_bin)) < 2 or len(np.unique(y_test_bin)) < 2:
                continue
            # normal model
            model = LogisticRegression(
                solver="liblinear", max_iter=200, class_weight="balanced"
            )
            model.fit(x_train, y_train_bin)
            y_pred = model.predict(x_test)
            a_model[ci] = np.mean(y_pred == y_test_bin)
            # shuffled labels
            y_train_shuf = np.random.permutation(y_train_bin)
            model.fit(x_train, y_train_shuf)
            y_pred_shuf = model.predict(x_test)
            a_shuffle[ci] = np.mean(y_pred_shuf == y_test_bin)
        acc_model[ti, :] = a_model
        acc_shuffle[ti, :] = a_shuffle
    return t, acc_model, acc_shuffle


# regression from neural activity to time.
def regression_time_frac(neu_x, neu_time, bin_times, fracs):
    n_splits = 50
    n_sampling = 25
    test_size = 0.3
    bin_l_idx, bin_r_idx = get_frame_idx_from_time(neu_time, 0, 0, bin_times)
    bin_len = bin_r_idx - bin_l_idx
    # trim remainder.
    t = neu_time[: (neu_x.shape[2] // bin_len) * bin_len]
    t = np.nanmin(t.reshape(-1, bin_len), axis=1)
    x = neu_x[:, :, : (neu_x.shape[2] // bin_len) * bin_len]
    x = np.nanmean(x.reshape(x.shape[0], x.shape[1], -1, bin_len), axis=3)
    y = np.tile((np.arange(x.shape[2]) + 1) / x.shape[2], (x.shape[0], 1))
    # normalize data.
    for ni in range(x.shape[1]):
        x[:, ni, :] = norm01(x[:, ni, :].reshape(-1)).reshape(x.shape[0], x.shape[2])
    # create results wrt fraction of features.
    r2_all = np.zeros([n_sampling, len(fracs), n_splits])
    # run model.
    x, y = shuffle(x, y)
    print("Running decoding with fraction of features")
    for si in tqdm(range(n_sampling), desc="sampling"):
        for fi in tqdm(range(len(fracs)), desc="frac"):
            # get fraction of features.
            sub_idx = np.random.choice(
                x.shape[1], int(x.shape[1] * fracs[fi]), replace=False
            )
            x_sub = x[:, sub_idx, :].copy()
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
                model = SVR(kernel="rbf")
                model.fit(x_train, y_train)
                # test model.
                y_pred = model.predict(
                    np.transpose(x_test, (0, 2, 1)).reshape(-1, x_test.shape[1])
                )
                y_pred = y_pred.reshape(y_test.shape)
                r2_all[si, fi, ti] = np.nanmean(
                    [
                        r2_score(y_test[ti, :], y_pred[ti, :])
                        for ti in range(y_test.shape[0])
                    ]
                )
    # average across folds.
    r2_all = np.nanmean(r2_all, axis=2)
    return r2_all
