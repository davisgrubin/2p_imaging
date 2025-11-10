#!/usr/bin/env python3

import time
import functools
import tracemalloc
import numpy as np
import pandas as pd
import rastermap as rm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mtick
from datetime import datetime
# typing helpers
from typing import Tuple

from scipy.stats import mannwhitneyu
from scipy.stats import levene
from scipy.spatial.distance import pdist
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from scipy.signal import savgol_filter
from scipy.signal import find_peaks
from scipy.cluster.hierarchy import dendrogram

from modeling.clustering import get_mean_sem_cluster
from modeling.clustering import clustering_neu_response_mode
from modeling.clustering import remap_cluster_id
from modeling.generative import run_glm_multi_sess
from modeling.quantifications import fit_trf_model

#%% general data processing

# monitor memory usage.
def show_resource_usage(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(f'running {func.__name__}')
        tracemalloc.start()
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        print(f'--- current time: {datetime.now().strftime("%Y.%m.%d %H:%M:%S")}')
        print(f'--- time cost: {elapsed:.2f}s')
        print(f'--- current memory: {current/1024/1024:.2f} MB')
        print(f'--- memory peak: {peak/1024/1024:.2f} MB')
        return result
    return wrapper

# rescale voltage recordings.
def rescale(data, upper, lower):
    data = data.copy()
    data = ( data - np.nanmin(data) ) / (np.nanmax(data) - np.nanmin(data))
    data = data * (upper - lower) + lower
    return data

# normalization into [0,1].
def norm01(data):
    return (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data) + 1e-5)

# compute the scale parameters when normalizing data into [0,1].
def get_norm01_params(data):
    x_scale = 1 / (np.nanmax(data) - np.nanmin(data))
    x_offset = - np.nanmin(data) / (np.nanmax(data) - np.nanmin(data))
    x_min = np.nanmin(data)
    x_max = np.nanmax(data)
    return x_scale, x_offset, x_min, x_max

# bin data and return index.
def get_bin_idx(data, bin_win, bin_num):
    bin_size = (bin_win[1] - bin_win[0]) / bin_num
    bins = np.arange(bin_win[0], bin_win[1] + bin_size, bin_size)
    bin_center = bins[:-1] + bin_size / 2
    bin_idx = np.digitize(data, bins) - 1
    return bins, bin_center, bin_idx

# compute mean and sem for average dF/F within given time window.
def get_mean_sem_win(neu_seq, neu_time, c_time, l_time, r_time, mode):
    pct = 50
    l_idx, r_idx = get_frame_idx_from_time(
        neu_time, c_time, l_time, r_time)
    neu_win = neu_seq[:, l_idx:r_idx].copy()
    # mean values in the window.
    if mode == 'mean':
        neu = np.nanmean(neu_win, axis=1)
    # values lower than percentile in the window.
    if mode == 'lower':
        neu = np.array([np.nanmean(ns[ns < np.percentile(ns, pct)]) for ns in neu_win])
    # values higher than percentile in the window.
    if mode == 'higher':
        neu = np.array([np.nanmean(ns[ns > np.percentile(ns, pct)]) for ns in neu_win])
    # compute mean and sem.
    neu_mean = np.nanmean(neu)
    std = np.nanstd(neu)
    count = np.nansum(~np.isnan(np.nanmean(neu_win, axis=1)), axis=0)
    neu_sem = std / np.sqrt(count)
    return neu, neu_mean, neu_sem

# compute mean and sem for 3d array data.
def get_mean_sem(data, win_baseline=None):
    # compute mean.
    m = np.nanmean(data.reshape(-1, data.shape[-1]), axis=0)
    # compute sem.
    std = np.nanstd(data.reshape(-1, data.shape[-1]), axis=0)
    count = np.nansum(~np.isnan(data.reshape(-1, data.shape[-1])), axis=0)
    s = std / np.sqrt(count)
    return m, s

# compute peak time within evaluation window.
def get_peak_time(neu, neu_time, win_peak):
    l_idx, r_idx = get_frame_idx_from_time(neu_time, 0, win_peak[0], win_peak[1])
    peak_time = neu_time[l_idx:r_idx][find_peaks(neu[l_idx:r_idx],distance=1106)[0][0]]
    return peak_time

#%% retreating neural data

# find trials based on stim_labels.
def pick_trial(
        stim_labels,
        img_seq_label,
        standard_types,
        fix_jitter_types,
        oddball_types,
        random_types,
        opto_types,
        frac):
    idx1 = np.isin(stim_labels[:,2], img_seq_label)    if img_seq_label    else np.ones_like(stim_labels[:,2])
    idx2 = np.isin(stim_labels[:,3], standard_types)   if standard_types   else np.ones_like(stim_labels[:,3])
    idx3 = np.isin(stim_labels[:,4], fix_jitter_types) if fix_jitter_types else np.ones_like(stim_labels[:,4])
    idx4 = np.isin(stim_labels[:,5], oddball_types)    if oddball_types    else np.ones_like(stim_labels[:,5])
    idx5 = np.isin(stim_labels[:,6], random_types)     if random_types     else np.ones_like(stim_labels[:,6])
    idx6 = np.isin(stim_labels[:,7], opto_types)       if opto_types       else np.ones_like(stim_labels[:,7])
    idx_fraction = np.random.permutation(np.repeat([1,0], [int(stim_labels.shape[0]*frac), stim_labels.shape[0]-int(stim_labels.shape[0]*frac)]))
    idx = (idx1*idx2*idx3*idx4*idx5*idx6*idx_fraction).astype('bool')
    return idx

# average across subsampling trials.
def sub_sampling_trial(neu_seq, samping_size=3, sampling_times=252):
    # compute number of samples.
    n_samples = int(samping_size*neu_seq.shape[0])+1 if samping_size<1 else samping_size
    # average across ramdom subset of neurons.
    neu_seq_sub = np.zeros([sampling_times, neu_seq.shape[1], neu_seq.shape[2]])
    for ti in range(sampling_times):
        sub_idx = np.random.choice(neu_seq.shape[0], n_samples, replace=False)
        neu_seq_sub[ti,:,:] = np.nanmean(neu_seq[sub_idx,:,:], axis=0)
    return neu_seq_sub

# get trial results across sessions.
def get_multi_sess_neu_trial(
        list_stim_labels,
        neu_cate,
        alignment,
        trial_idx=None,
        trial_param=None,
        mean_sem=True,
        sub_sampling=False,
        frac=1,
    ):
    neu = []
    stim_seq = []
    camera_pupil = []
    pre_isi = []
    post_isi = []
    # pull everything out of the alignment dict once.
    list_stim_seq     = alignment['list_stim_seq']
    list_camera_pupil = alignment['list_camera_pupil']
    list_pre_isi      = alignment['list_pre_isi']
    list_post_isi     = alignment['list_post_isi']
    # decide mode: 1 = only trial_param, 2 = only trial_idx, 3 = both.
    mode = (trial_param is not None) + 2 * (trial_idx is not None)
    if trial_param is not None:
        p0, p1, p2, p3, p4, p5 = trial_param
    # one pass over sessions.
    for i, labels in enumerate(list_stim_labels):
        if mode == 1:
            idx = pick_trial(labels, p0, p1, p2, p3, p4, p5, frac)
        elif mode == 2:
            idx = trial_idx[i]
        elif mode == 3:
            idx = trial_idx[i] * pick_trial(labels, p0, p1, p2, p3, p4, p5, frac)
        else:
            # no trial_param or trial_idx → nothing to do.
            continue
        neu.append(neu_cate[i][idx,...])
        stim_seq.append(list_stim_seq[i][idx,...] )
        camera_pupil.append(list_camera_pupil[i][idx,...])
        pre_isi.append(list_pre_isi[i][idx])
        post_isi.append(list_post_isi[i][idx])
    # total trials & neurons.
    n_trials  = sum(n.shape[0] for n in neu)
    n_neurons = sum(n.shape[1] for n in neu)
    if mean_sem:
        if sub_sampling:
            neu = np.concatenate([sub_sampling_trial(n) for n in neu], axis=1)
            stim_seq = np.nanmean(np.concatenate(stim_seq,     axis=0), axis=0)
            camera_pupil = np.nanmean(np.concatenate(camera_pupil, axis=0), axis=0)
            return [neu, stim_seq, camera_pupil], [n_trials, n_neurons]
        else:
            # per-session means & stds, then concatenate.
            means = [np.nanmean(n, axis=0) for n in neu]
            stds  = [np.nanstd(n,  axis=0) for n in neu]
            mean  = np.concatenate(means, axis=0)
            std   = np.concatenate(stds,  axis=0)
            # collapse sequences & pupil across all trials.
            stim_seq     = np.nanmean(np.concatenate(stim_seq,     axis=0), axis=0)
            camera_pupil = np.nanmean(np.concatenate(camera_pupil, axis=0), axis=0)
            return [mean, std, stim_seq, camera_pupil], [n_trials, n_neurons]
    else:
        return [neu, stim_seq, camera_pupil, pre_isi, post_isi], [n_trials, n_neurons]

# find neuron category and trial data.
def get_neu_trial(
        alignment, list_labels, list_stim_labels,
        trial_idx=None, trial_param=None, mean_sem=True, frac=1, sub_sampling=False,
        cate=None, roi_id=None,
        ):
    if cate != None:
        idx = np.concatenate(
            [np.in1d(list_labels[i],cate)
             for i in range(len(list_stim_labels))])
        if np.sum(idx) == 0:
            raise ValueError(f'No available neurons in given category {cate}')
        list_neu_seq = alignment['list_neu_seq']
        colors = get_roi_label_color(cate=cate)
        neu_labels = np.concatenate([
            list_labels[i][np.in1d(list_labels[i],cate)]
            for i in range(len(list_stim_labels))])
        masks = [np.isin(labels, cate) for labels in list_labels]
        neu_cate = [neu_seq[:, mask, :] for neu_seq, mask in zip(list_neu_seq, masks)]
    if roi_id != None:
        colors = get_roi_label_color(list_labels, roi_id=roi_id)
        neu_labels = None
        neu_cate = [np.expand_dims(list_neu_seq[0][:,roi_id,:], axis=1)]
    neu_trial, neu_num = get_multi_sess_neu_trial(
        list_stim_labels, neu_cate, alignment,
        trial_idx=trial_idx, trial_param=trial_param, mean_sem=mean_sem, sub_sampling=sub_sampling, frac=frac)
    return colors, neu_trial, neu_labels, neu_num

# compute indice with givn time window for dF/F.
def get_frame_idx_from_time(timestamps, c_time, l_time, r_time):
    l_idx = np.searchsorted(timestamps, c_time+l_time)
    r_idx = np.searchsorted(timestamps, c_time+r_time)
    return l_idx, r_idx

# find expected standard fix interval.
def get_expect_interval(stim_labels):
    idx_short = (stim_labels[:,2]>1)*(stim_labels[:,3]==0)
    expect_short = stim_labels[1:,0] - stim_labels[:-1,1]
    expect_short = np.median(expect_short[idx_short[:-1]])
    idx_long = (stim_labels[:,2]>1)*(stim_labels[:,3]==1)
    expect_long = stim_labels[1:,0] - stim_labels[:-1,1]
    expect_long = np.median(expect_long[idx_long[:-1]])
    return expect_short, expect_long

# get index for the first trial of block transition.
def get_block_1st_idx(stim_labels, target, prepost='post'):
    if prepost == 'pre':
        diff = np.diff(stim_labels[:,target], prepend=stim_labels[:,target][0])
    if prepost == 'post':
        diff = np.diff(stim_labels[:,target], prepend=stim_labels[:,target][:2])
        diff = diff[:-1]
    idx_0 = (diff==-1) * (stim_labels[:,target]==0)
    idx_1 = (diff==1) * (stim_labels[:,target]==1)
    return idx_0, idx_1

# find index for epoch.
def get_block_epochs_idx(stim_labels, epoch_len, block_combine=True):
    total_length = len(stim_labels)
    # identify start and end indices for each continuous block
    label_changes = np.diff(stim_labels)
    change_indices = np.where(label_changes != 0)[0]
    block_starts = np.concatenate(([0], change_indices + 1))
    block_ends = np.concatenate((change_indices, [total_length - 1]))
    # dictionaries to collect epoch masks and count epochs per block for each label.
    blocks_dict = {0: [], 1: []}
    blocks_num_epochs = {0: [], 1: []}
    # process each block.
    for start, end in zip(block_starts, block_ends):
        current_label = stim_labels[start]
        block_length = end - start + 1
        num_epochs = block_length // epoch_len
        # build epoch masks for the current block.
        epoch_masks = []
        for i in range(num_epochs):
            mask = np.zeros(total_length, dtype=bool)
            epoch_start = start + i * epoch_len
            epoch_end = epoch_start + epoch_len
            mask[epoch_start:epoch_end] = True
            epoch_masks.append(mask)
        # stack masks.
        block_epochs = np.stack(epoch_masks)
        blocks_dict[current_label].append(block_epochs)
        blocks_num_epochs[current_label].append(num_epochs)
    # trim each block to the minimum number of epochs and stack.
    result = []
    for label in [0, 1]:
        if blocks_dict[label]:
            min_epochs = min(blocks_num_epochs[label])
            trimmed_blocks = [block[:min_epochs] for block in blocks_dict[label]]
            result_array = np.stack(trimmed_blocks)
            # exclude the first block.
            result_array[0,:,:] = False
            result.append(result_array)
        else:
            result.append(np.empty((0, 0, total_length), dtype=bool))
    if block_combine:
        epoch_0s = np.sum(result[0], axis=0).astype('bool')
        epoch_1s = np.sum(result[1], axis=0).astype('bool')
    else:
        epoch_0s = result[0]
        epoch_1s = result[1]
    return epoch_0s, epoch_1s

# find index around transition.
def get_block_transition_idx(stim_labels, trials_around):
    n_trials = len(stim_labels)
    # compute the difference to find transitions.
    diff = np.diff(stim_labels, prepend=stim_labels[0])
    trans_0to1 = np.where(diff == 1)[0]
    trans_1to0 = np.where(diff == -1)[0]
    # only keep transitions that allow a full window.
    valid_0to1 = trans_0to1[(trans_0to1-trials_around>=0) & (trans_0to1+trials_around<n_trials)][1:]
    valid_1to0 = trans_1to0[(trans_1to0-trials_around>=0) & (trans_1to0+trials_around<n_trials)][1:]
    # preallocate boolean masks.
    trans_0to1 = np.zeros((len(valid_0to1), n_trials), dtype=bool)
    trans_1to0 = np.zeros((len(valid_1to0), n_trials), dtype=bool)
    # mark the indices corresponding to the window:
    for i, t in enumerate(valid_0to1):
        trans_0to1[i, t-trials_around:t+trials_around] = True
    for i, t in enumerate(valid_1to0):
        trans_1to0[i, t-trials_around:t+trials_around] = True
    return trans_0to1, trans_1to0

# get index for pre/post short/long oddball.
def get_odd_stim_prepost_idx(stim_labels):
    idx_pre_short = (stim_labels[:,2]==-1) * (stim_labels[:,5]==0) * (stim_labels[:,6]==0)
    idx_pre_long  = (stim_labels[:,2]==-1) * (stim_labels[:,5]==1) * (stim_labels[:,6]==0)
    idx_post_short = np.zeros_like(idx_pre_short)
    idx_post_short[1:] = idx_pre_short[:-1]
    idx_post_long = np.zeros_like(idx_pre_long)
    idx_post_long[1:] = idx_pre_long[:-1]
    idx_pre_short[-1] = False
    idx_pre_long[-1] = False
    return idx_pre_short, idx_pre_long, idx_post_short, idx_post_long

# get index for pre/post image change.
def get_change_prepost_idx(stim_labels):
    idx_pre = np.diff(stim_labels[:,2]<-1, append=0)
    idx_pre[idx_pre==-1] = 0
    idx_pre = idx_pre.astype('bool')
    idx_post = stim_labels[:,2]<-1
    return idx_pre, idx_post

# get index to split concatenated labels for categories.
def get_split_idx(list_labels, cate):
    split_idx = [len(list_labels[i][np.in1d(list_labels[i],cate)])
        for i in range(len(list_labels))]
    split_idx = np.cumsum(split_idx)[:-1]
    return split_idx

# mark stim around image change and oddball as outlier.
def exclude_odd_stim(stim_labels):
    n = 2
    neg = np.where(stim_labels[:,2] < 0)[0]
    stim_labels_mark = stim_labels.copy()
    for i in neg:
        stim_labels_mark[
            np.max([0, i-n]):np.min([stim_labels.shape[0], i+n+1]),2] = -np.abs(
                stim_labels_mark[np.max([0, i-n]):np.min([stim_labels.shape[0], i+n+1]),2])
    return stim_labels_mark

#%% detailed processing

# compute modulation index.
def get_modulation_index(v_pre, v_post):
    m = (v_post - v_pre) / (np.abs(v_pre) + np.abs(v_post) + 1e-8)
    return m

# get isi based binning average neural response.
def get_isi_bin_neu(
        neu_seq, stim_seq, camera_pupil, isi,
        bin_win, bin_num,
        mean_sem=True
        ):
    # define bins.
    bin_size = (bin_win[1] - bin_win[0]) / bin_num
    bins = np.arange(bin_win[0], bin_win[1] + bin_size, bin_size)
    bin_center = bins[:-1] + bin_size / 2
    list_bin_idx = [np.digitize(i, bins) - 1 for i in isi]
    # compute binned results.
    bin_neu_seq_trial = []
    bin_neu_seq = []
    bin_neu_mean = np.zeros((len(bin_center), neu_seq[0].shape[2]))
    bin_neu_sem = np.zeros((len(bin_center), neu_seq[0].shape[2]))
    bin_stim_seq = np.zeros((len(bin_center), stim_seq[0].shape[1], 2))
    bin_camera_pupil = np.zeros((len(bin_center), camera_pupil[0].shape[1]))
    for i in range(len(bin_center)):
        # get binned neural traces.
        neu_trial = [n[bi==i,:,:] for n, bi in zip(neu_seq, list_bin_idx)]
        neu = [np.nanmean(n, axis=0) for n in neu_trial]
        neu = np.concatenate(neu, axis=0)
        # get binned stimulus timing.
        s_seq = [s[bi==i,:,:] for s, bi in zip(stim_seq, list_bin_idx)]
        s_seq = np.concatenate(s_seq, axis=0)
        s_seq = np.nanmean(s_seq, axis=0)
        # get pupil area.
        c_pupil = [p[bi==i,:] for p, bi in zip(camera_pupil, list_bin_idx)]
        c_pupil = np.concatenate(c_pupil, axis=0)
        c_pupil = np.nanmean(c_pupil, axis=0)
        # collect results.
        bin_neu_seq_trial.append(neu_trial)
        bin_neu_seq.append(neu)
        bin_neu_mean[i,:] = get_mean_sem(neu)[0]
        bin_neu_sem[i,:] = get_mean_sem(neu)[1]
        bin_stim_seq[i,:,:] = s_seq
        bin_camera_pupil[i,:] = c_pupil
    bin_center = bin_center.astype('int32')
    return [bins, bin_center, bin_neu_seq_trial, bin_neu_seq,
            bin_neu_mean, bin_neu_sem, bin_stim_seq, bin_camera_pupil]

# compute synchrnization across time.
def get_neu_sync(neu, win_width):
    sync = []
    for t in range(win_width, neu.shape[1]):
        window_data = neu[:, t-win_width:t]
        # normalization.
        norms = np.linalg.norm(window_data, axis=1, keepdims=True)
        normalized_data = window_data / (norms + 1e-10)
        # pairwise absolute cosine distances.
        cosine_distances = pdist(normalized_data, metric='cosine')
        abs_cosine_distances = np.abs(cosine_distances)
        # convert to overall similarity.
        cos_sim = 1 - abs_cosine_distances
        s = (np.nansum(cos_sim) - neu.shape[0]) / (neu.shape[0]**2 - neu.shape[0])
        sync.append(s)
    return sync

# find expected isi.
def get_expect_time(stim_labels):
    idx_short = (stim_labels[:,2]>1)*(stim_labels[:,3]==0)
    expect_short = stim_labels[1:,0] - stim_labels[:-1,1]
    expect_short = np.median(expect_short[idx_short[:-1]])
    idx_long = (stim_labels[:,2]>1)*(stim_labels[:,3]==1)
    expect_long = stim_labels[1:,0] - stim_labels[:-1,1]
    expect_long = np.median(expect_long[idx_long[:-1]])
    return expect_short, expect_long

# compute calcium transient event timing and average.
def get_ca_transient(dff, time_img):
    pct = 95
    win_peak = 25
    timescale = 1.0
    l_frames = int(20*timescale)
    r_frames = int(150*timescale)
    # calculate the area under this window as a threshold.
    thres = np.percentile(dff, pct) * win_peak
    # find the window larger than baseline.
    def detect_spikes_win(dff_traces):
        # use convolution to calculate the sum of sliding windows.
        sliding_window_sum = np.convolve(dff_traces, np.ones(win_peak), mode='same')
        # compare against the threshold.
        ca_tran_win = (sliding_window_sum > thres).astype(int)
        return ca_tran_win
    # remove outliers.
    def win_thres(ca_tran_win):
        k = 0.5
        # reset start and end.
        if ca_tran_win[0] == 1:
            ca_tran_win[:np.where(ca_tran_win==0)[0][0]] = 0
        if ca_tran_win[-1] == 1:
            ca_tran_win[np.where(np.diff(ca_tran_win, append=0)==1)[0][-1]:] = 0
        ca_tran_diff = np.diff(ca_tran_win, append=0)
        # compute window width.
        win_wid = np.where(ca_tran_diff==-1)[0] - np.where(ca_tran_diff==1)[0]
        # find windows with outlier width.
        lower = np.nanmean(win_wid)-k*np.nanstd(win_wid)
        outlier_idx = (win_wid < lower)
        # reset outlier.
        for i in range(len(win_wid)):
            if outlier_idx[i]:
                ca_tran_win[np.where(ca_tran_diff==1)[0][i]:np.where(ca_tran_diff==-1)[0][i]+1] = 0
        return ca_tran_win
    # only keep the start of a transient.
    def get_tran_time(ca_tran_win):
        ca_tran_idx = (np.diff(ca_tran_win, append=0)==1).astype('int32')
        return ca_tran_idx
    # extract dff traces around calcium transient.
    def get_ca_mean(ca_tran):
        dff_ca_neu = []
        dff_ca_time = []
        for i in range(dff.shape[0]):
            ca_event_time = np.where(ca_tran[i,:]==1)[0]
            # find dff.
            cn = [dff[i,t-l_frames:t+r_frames].reshape(1,-1)
                  for t in ca_event_time
                  if t > l_frames and t < len(time_img)-r_frames]
            cn = np.concatenate(cn, axis=0) if len(cn) > 0 else np.array([np.nan])
            # find time.
            ct = [time_img[t-l_frames:t+r_frames].reshape(1,-1)-time_img[t]
                  for t in ca_event_time
                  if t > l_frames and t < len(time_img)-r_frames]
            ct = np.concatenate(ct, axis=0) if len(ct) > 0 else np.array([np.nan])
            ct = np.nanmean(ct, axis=0) if not np.isnan(np.mean(ct)) else np.array([np.nan])
            # collect.
            dff_ca_neu.append(cn)
            dff_ca_time.append(ct)
        dff_ca_time = np.concatenate([t.reshape(1,-1) for t in dff_ca_time], axis=0)
        dff_ca_time = np.nanmean(dff_ca_time, axis=0)
        return dff_ca_neu, dff_ca_time
    # compute transient events.
    ca_tran = np.zeros_like(dff)
    for i in range(dff.shape[0]):
        ca_tran_win = detect_spikes_win(dff[i,:])
        ca_tran_win = win_thres(ca_tran_win)
        ca_tran[i,:] = get_tran_time(ca_tran_win)
    # compute average.
    dff_ca_neu, dff_ca_time = get_ca_mean(ca_tran)
    # get the total number of events for each neurons.
    n_ca = np.nansum(ca_tran, axis=1)
    return n_ca, dff_ca_neu, dff_ca_time

# compute calcium transient event timing and average with multiple sessions.
def get_ca_transient_multi_sess(list_neural_trials):
    list_dff = [nt['dff'] for nt in list_neural_trials] 
    list_time = [nt['time'] for nt in list_neural_trials]
    list_n_ca = []
    list_dff_ca_neu = []
    list_dff_ca_time = []
    for i in range(len(list_dff)):
        n_ca, dff_ca_neu, dff_ca_time = get_ca_transient(
            list_dff[i], list_time[i])
        list_n_ca.append(n_ca)
        list_dff_ca_neu += dff_ca_neu
        list_dff_ca_time.append(dff_ca_time.reshape(1,-1))
    list_n_ca = np.concatenate(list_n_ca)
    list_dff_ca_time = np.nanmean((np.concatenate(list_dff_ca_time, axis=0)), axis=0)
    return list_n_ca, list_dff_ca_neu, list_dff_ca_time

# strech data to target time stamps for temporal scaling for 2d data.
def get_temporal_scaling_data(data, t_org, t_target):
    # map to target time stamps.
    t_mapped = (t_target - t_target[0]) / (t_target[-1] - t_target[0]) * (t_org[-1] - t_org[0]) + t_org[0]
    # find the target inserted index.
    idx_right = np.searchsorted(t_org, t_mapped)
    idx_right = np.clip(idx_right, 1, len(t_org)-1)
    idx_left = idx_right - 1
    # compute the fractional distance between the two original time stamps.
    t_left = t_org[idx_left]
    t_right = t_org[idx_right]
    weights = (t_mapped - t_left) / (t_right - t_left)
    # lienar interpolation for each row.
    left_vals = data[:, idx_left]
    right_vals = data[:, idx_right]
    data_scaled = left_vals + weights * (right_vals - left_vals)
    return data_scaled

# compute scaled data for single trial response across sessions.
def get_temporal_scaling_trial_multi_sess(neu_seq, stim_seq, neu_time, target_isi):
    # compute mean time stamps.
    stim_seq_target = np.nanmean(np.concatenate(stim_seq, axis=0),axis=0)
    c_idx = int(stim_seq_target.shape[0]/2)
    # target isi before.
    target_l_pre, target_r_pre = get_frame_idx_from_time(
        neu_time, 0,
        -target_isi,
        stim_seq_target[c_idx,0])
    # target stimulus.
    target_l_stim, target_r_stim = get_frame_idx_from_time(
        neu_time, 0,
        stim_seq_target[c_idx,0],
        stim_seq_target[c_idx,1])
    # target isi after.
    target_l_post, target_r_post = get_frame_idx_from_time(
        neu_time, 0,
        stim_seq_target[c_idx,1],
        stim_seq_target[c_idx,1]+target_isi)
    # compute trial wise temporal scaling.
    scale_neu_seq = []
    for si in range(len(neu_seq)):
        scale_ns = np.zeros((neu_seq[si].shape[0], neu_seq[si].shape[1], target_r_post-target_l_pre))
        for ti in range(neu_seq[si].shape[0]):
            # trial isi before.
            l_pre, r_pre = get_frame_idx_from_time(
                neu_time, 0,
                stim_seq[si][ti,c_idx-1,1],
                stim_seq[si][ti,c_idx,0])
            # trial stimulus.
            l_stim, r_stim = get_frame_idx_from_time(
                neu_time, 0,
                stim_seq[si][ti,c_idx,0],
                stim_seq[si][ti,c_idx,1])
            # trial isi after.
            l_post, r_post = get_frame_idx_from_time(
                neu_time, 0,
                stim_seq[si][ti,c_idx,1],
                stim_seq[si][ti,c_idx+1,0])
            # compute scaled data.
            scale_ns[ti,:,:] = np.concatenate([
                get_temporal_scaling_data(
                    neu_seq[si][ti,:,l_pre:r_pre],
                    neu_time[l_pre:r_pre],
                    neu_time[target_l_pre:target_r_pre]),
                get_temporal_scaling_data(
                    neu_seq[si][ti,:,l_stim:r_stim],
                    neu_time[l_stim:r_stim],
                    neu_time[target_l_stim:target_r_stim]),
                get_temporal_scaling_data(
                    neu_seq[si][ti,:,l_post:r_post],
                    neu_time[l_post:r_post],
                    neu_time[target_l_post:target_r_post]),
                ], axis=1)
        # collect single session results.
        scale_neu_seq.append(scale_ns)
    return scale_neu_seq

# run statistics test and get significance level.
def get_stat_test(data_1, data_2, stat_test):
    p_thres = np.array([5e-2, 5e-4, 5e-6])
    if stat_test == 'mean':
        _, p = mannwhitneyu(data_1, data_2)
    if stat_test == 'var':
        _, p = levene(data_1, data_2)
    r = np.sum(p < p_thres)
    return p, r

# statistics test for response in evaluation windows.
def get_win_mag_quant_stat_test(neu_seq_1, neu_seq_2, neu_time, c_time, win_eval, stat_test):
    baseline_correction = False
    mode = ['lower', 'mean', 'mean', 'mean']
    # get window data.
    neu_1 = [get_mean_sem_win(
        neu_seq_1.reshape(-1, neu_seq_1.shape[-1]),
        neu_time, c_time, win_eval[i][0], win_eval[i][1], mode=mode[i])
        for i in range(4)]
    neu_2 = [get_mean_sem_win(
        neu_seq_2.reshape(-1, neu_seq_2.shape[-1]),
        neu_time, c_time, win_eval[i][0], win_eval[i][1], mode=mode[i])
        for i in range(4)]
    # baseline correction.
    if baseline_correction:
        neu_1 = [neu_1[i][0] - neu_1[0][1] for i in [1,2,3]]
        neu_2 = [neu_2[i][0] - neu_2[0][1] for i in [1,2,3]]
    else:
        neu_1 = [neu_1[i][0] for i in [1,2,3]]
        neu_2 = [neu_2[i][0] for i in [1,2,3]]
    # run statistics test.
    p = np.array([get_stat_test(n1, n2, stat_test)[0] for n1, n2 in zip(neu_1, neu_2)])
    r = np.array([get_stat_test(n1, n2, stat_test)[1] for n1, n2 in zip(neu_1, neu_2)])
    return p, r

# compute correlation between rows.
def get_row_corr(x, y):
    xm = x
    ym = y
    num = np.sum(xm * ym, axis=1)
    den = np.sqrt(np.sum(xm*xm, axis=1) * np.sum(ym*ym, axis=1))
    c = num / den
    return c

# average across subsampling neurons.
def sub_sampling_neuron(neu_seq):
    samping_size = 0.2
    sampling_times = 50
    # compute number of samples.
    n_samples = int(samping_size*neu_seq.shape[0])+1
    # average across ramdom subset of neurons.
    neu_seq_sub = np.zeros([sampling_times, neu_seq.shape[1]])
    for qi in range(sampling_times):
        sub_idx = np.random.choice(neu_seq.shape[0], n_samples, replace=False)
        neu_seq_sub[qi,:] = np.nanmean(neu_seq[sub_idx,:], axis=0)
    return neu_seq_sub


#%% plotting

# get color from label.
def get_roi_label_color(labels=None, cate=None, roi_id=None):
    if (roi_id != None and labels[roi_id] == -1) or (cate == [-1]):
        color0 = 'dimgrey'
        color1 = 'deepskyblue'
        color2 = 'royalblue'
        cmap = mcolors.LinearSegmentedColormap.from_list(
            'yicong_will_love_you_forever',
            ['lemonchiffon', 'royalblue', 'black'])
    if (roi_id != None and labels[roi_id] == 1) or (cate == [1]):
        color0 = 'dimgrey'
        color1 = 'chocolate'
        color2 = 'crimson'
        cmap = mcolors.LinearSegmentedColormap.from_list(
            'yicong_will_love_you_forever',
            ['lemonchiffon', 'crimson', 'black'])
    if (roi_id != None and labels[roi_id] == 2) or (cate == [2]):
        color0 = 'dimgrey'
        color1 = 'mediumseagreen'
        color2 = 'forestgreen'
        cmap = mcolors.LinearSegmentedColormap.from_list(
            'yicong_will_love_you_forever',
            ['lemonchiffon', 'forestgreen', 'black'])
    if len(cate) > 1:
        color0 = 'dimgrey'
        color1 = 'hotpink'
        color2 = 'darkviolet'
        cmap = mcolors.LinearSegmentedColormap.from_list(
            'yicong_will_love_you_forever',
            ['lemonchiffon', 'violet', 'black'])
    return color0, color1, color2, cmap

# return colors from dim to dark with a base color.
def get_cmap_color(n_colors, base_color=None, cmap=None, return_cmap=False):
    c_margin = 0.05
    if base_color != None:
        cmap = mcolors.LinearSegmentedColormap.from_list(
            None, base_color)
    if cmap != None:
        pass
    colors = cmap(np.linspace(c_margin, 1 - c_margin, n_colors))
    colors = [mcolors.to_hex(color, keep_alpha=True) for color in colors]
    if return_cmap:
        return cmap, colors
    else:
        return colors

# sort each row in a heatmap.
def sort_heatmap_neuron(neu_seq_sort, sort_method):
    win_conv = 9
    n_clusters = 3
    locality = 1
    # smooth values.
    smoothed = np.array(
        [np.convolve(row, np.ones(win_conv)/win_conv, mode='same')
         for row in neu_seq_sort])
    if sort_method == 'rastermap':
        # fit model.
        model = rm.Rastermap(n_clusters=n_clusters, locality=locality)
        model.fit(smoothed)
        # get ordering.
        sorted_idx = model.isort
    if sort_method == 'peak_timing':
        # get peak timing.
        peak_time = np.argmax(smoothed, axis=1).reshape(-1)
        # combine and sort.
        sorted_idx = peak_time.argsort()
    if sort_method == 'shuffle':
        sorted_idx = np.random.permutation(np.arange(smoothed.shape[0]))
    return sorted_idx

# apply colormap.
def apply_colormap(data, norm_mode, data_share):
    hm_cmap = plt.cm.hot
    pct = 1
    # no valid data found.
    if len(data) == 0:
        hm_data = np.zeros((0, data.shape[1], 3))
        hm_norm = None
        return hm_data, hm_norm, hm_cmap
    else:
        # binary heatmap.
        if norm_mode == 'binary':
            hm_norm = mcolors.Normalize(vmin=0, vmax=1)
            cs = get_cmap_color(3, cmap=hm_cmap)
            hm_cmap = mcolors.LinearSegmentedColormap.from_list(None, ['#FCFCFC', cs[1]])
        else:
            # no normalization.
            if norm_mode == 'none':
                hm_norm = mcolors.Normalize(vmin=np.nanpercentile(data, pct), vmax=np.nanpercentile(data, 100-pct))
                data = np.clip(data, np.nanpercentile(data, pct), np.nanpercentile(data, 100-pct))
                data = norm01(data)
            # normalized into [0,1].
            elif norm_mode == 'minmax':
                for i in range(data.shape[0]):
                    data[i,:] = norm01(data[i,:])
                hm_norm = mcolors.Normalize(vmin=0, vmax=1)
            # share the global scale.
            elif norm_mode == 'share':
                hm_norm = mcolors.Normalize(vmin=np.nanpercentile(data_share, pct), vmax=np.nanpercentile(data_share, 100-pct))
                data = np.clip(data, np.nanpercentile(data_share, pct), np.nanpercentile(data_share, 100-pct))
                data = norm01(data)
            # handle errors.
            else:
                raise ValueError('norm_mode can only be [binary, none, minmax, share].')
        hm_cmap.set_bad(color='white')
        hm_data = hm_cmap(data)
        hm_data = hm_data[..., :3]
    return hm_data, hm_norm, hm_cmap

# hide all axis.
def hide_all_axis(ax):
    ax.tick_params(tick1On=False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

# generate random rotation matrix for 3d dynamics.
def get_random_rotate_mat_3d():
    u1, u2, u3 = np.random.random(3)
    s1, s2 = np.sqrt(1.0 - u1), np.sqrt(u1)
    th1, th2 = 2.0*np.pi*u2, 2.0*np.pi*u3
    x = s1 * np.sin(th1)
    y = s1 * np.cos(th1)
    z = s2 * np.sin(th2)
    w = s2 * np.cos(th2)
    rm = np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
        [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)]])
    return rm

# adjust layout for isi example epoch.
def adjust_layout_isi_example_epoch(ax, trial_win, bin_win):
    ax.tick_params(tick1On=False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel('trial #')
    ax.set_ylabel('interval (ms)')
    ax.set_xlim(trial_win)
    ax.set_ylim(bin_win)
    ax.set_yticks(500*np.arange((bin_win[0]+50)/500, (bin_win[1]-50)/500+1).astype('int32'))
    ax.set_yticklabels(
        500*np.arange((bin_win[0]+50)/500, (bin_win[1]-50)/500+1).astype('int32'))

# adjust layout for grand average neural traces.
def adjust_layout_neu(ax):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_ylabel('dF/F (z-scored)')
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
    ax.yaxis.set_major_locator(mtick.MaxNLocator(nbins=1))

# adjust layout for grand average neural traces for clustering.
def adjust_layout_cluster_neu(ax, n_clusters, xlim):
    ax.tick_params(axis='y', tick1On=False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlim(xlim)
    ax.set_xticks([0])
    ax.set_yticks([])
    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f'))
    ax.xaxis.set_major_locator(mtick.MaxNLocator(nbins=3))

# adjust layout for scatter comparison.
def adjust_layout_scatter(ax, upper, lower):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
    ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
    ax.xaxis.set_major_locator(mtick.MaxNLocator(nbins=2))
    ax.yaxis.set_major_locator(mtick.MaxNLocator(nbins=2))

# adjust layout for heatmap.
def adjust_layout_heatmap(ax):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

# adjust layout for 2d latent dynamics.
def adjust_layout_2d_latent(ax):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
    ax.xaxis.set_major_locator(mtick.MaxNLocator(nbins=2))
    ax.yaxis.set_major_locator(mtick.MaxNLocator(nbins=2))
    
# adjust layout for 3d latent dynamics.
def adjust_layout_3d_latent(ax):
    ax.grid(False)
    ax.view_init(elev=15, azim=15)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

# add legend into subplots.
def add_legend(ax, colors, labels, n_trials, n_neurons, n_sessions, loc, dim=2):
    if dim == 2:
        plot_args = [[],[]]
    if dim == 3:
        plot_args = [[],[],[]]
    handles = []
    if colors != None and labels != None:
        handles += [
            ax.plot(*plot_args, lw=0, color=colors[i], label=labels[i])[0]
            for i in range(len(labels))]
    if n_trials != None and n_neurons != None:
        handles += [
            ax.plot(*plot_args, lw=0, color='black', label=r'$n_{trial}=$'+str(n_trials))[0],
            ax.plot(*plot_args, lw=0, color='black', label=r'$n_{neuron}=$'+str(n_neurons))[0],
            ax.plot(*plot_args, lw=0, color='black', label=r'$n_{session}=$'+str(n_sessions))[0]]
    ax.legend(
        loc=loc,
        handles=handles,
        labelcolor='linecolor',
        frameon=False,
        framealpha=0)

# add heatmap colorbar.
def add_heatmap_colorbar(ax, cmap, norm, label, yticklabels=None):
    if ax is not None:
        # Hide the axis if needed
        hide_all_axis(ax)
        
        # Create an inset axis for the colorbar
        cax = ax.inset_axes([1.05, 0.1, 0.05, 0.8], transform=ax.transAxes)
        
        # Set default normalization if not provided
        if norm is None:
            norm = mcolors.Normalize(vmin=0, vmax=1)
        
        # Create the colorbar
        cbar = ax.figure.colorbar(
            plt.cm.ScalarMappable(cmap=cmap, norm=norm),
            cax=cax
        )
        
        # Customize the colorbar appearance
        cbar.outline.set_linewidth(0.25)
        cbar.ax.set_ylabel(label, rotation=90, labelpad=10)
        cbar.ax.tick_params(axis='y')
        cbar.ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
        cbar.ax.yaxis.set_major_locator(
            mtick.FixedLocator([norm.vmin + 0.2 * (norm.vmax - norm.vmin),
                                norm.vmax - 0.2 * (norm.vmax - norm.vmin)])
        )
        
        # Set custom y-tick labels if provided
        if yticklabels is not None:
            cbar.ax.set_yticklabels(yticklabels)

#%% basic plotting utils class

class utils_basic:

    def __init__(self):
        self.min_num_trial = 5
        self.cluster_up_cmap = plt.cm.spring
        self.cluster_dn_cmap = plt.cm.winter
        self.latent_cmap = plt.cm.nipy_spectral
        self.random_bin_cmap = plt.cm.gist_ncar
        self.cross_day_cmap = plt.cm.gnuplot_r
        self.stat_sym = ['n.s.', '*', '**', '***']
        self.heatmap_sort_frac = 0.5

    @show_resource_usage
    def run_glm(self,):
        # define kernel window.
        kernel_win = [-1500,1500]
        l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, kernel_win[0], kernel_win[1])
        kernel_time = self.alignment['neu_time'][l_idx:r_idx]
        l_idx = np.searchsorted(self.alignment['neu_time'], 0) - l_idx
        r_idx = r_idx - np.searchsorted(self.alignment['neu_time'], 0)
        # collect data.
        list_dff = [nt['dff'] for nt in self.list_neural_trials]
        list_neu_time = [nt['time'] for nt in self.list_neural_trials]
        list_input_time  = [nt['vol_time'] for nt in self.list_neural_trials]
        list_input_value = [nt['vol_stim_vis'] for nt in self.list_neural_trials]
        list_stim_labels = [nt['stim_labels'] for nt in self.list_neural_trials]
        # fit glm model.
        kernel_all = run_glm_multi_sess(
            list_dff, list_neu_time,
            list_input_time, list_input_value, list_stim_labels,
            l_idx, r_idx)
        glm = {
            'kernel_time': kernel_time,
            'kernel_all': kernel_all,
            }
        return glm
    
    def run_clustering(self, n_up, n_dn):
        0.1
        0.3
        0.8
        r2_thres = 0.3
        # get data within range.
        z_idx = get_frame_idx_from_time(self.glm['kernel_time'], 0, -100, 0)[0]
        neu_seq_l = self.glm['kernel_all'][:,:z_idx]
        neu_time_l = self.glm['kernel_time'][:z_idx]
        neu_seq_r = self.glm['kernel_all'][:,z_idx:]
        neu_time_r = self.glm['kernel_time'][z_idx:]
        # fit response model.
        [trf_param_up, pred_up, r2_all_up,
         trf_param_dn, pred_dn, r2_all_dn] = fit_trf_model(
             neu_seq_l, neu_time_l, neu_seq_r, neu_time_r)
        # initilize clustering.
        self.n_clusters = n_up + n_dn
        cluster_id = np.ones(self.glm['kernel_all'].shape[0]) * -1
        # get good ramp index.
        idx_up = (r2_all_up > r2_thres) * (r2_all_up > r2_all_dn)
        idx_dn = (r2_all_dn > r2_thres) * (r2_all_dn > r2_all_up)
        # run clustering.
        cluster_id_up = clustering_neu_response_mode(trf_param_up[idx_up,3].reshape(-1,1), n_up, 'kmeans')
        cluster_id_dn = clustering_neu_response_mode(trf_param_dn[idx_dn,3].reshape(-1,1), n_dn, 'kmeans')
        # relabel based on temporal receptive field.
        sorted_up = np.argsort([np.nanmean(trf_param_up[idx_up,3][cluster_id_up==ci]) for ci in range(n_up)])
        sorted_dn = np.argsort([np.nanmean(trf_param_dn[idx_dn,3][cluster_id_dn==ci]) for ci in range(n_dn)])
        map_up = {val: i for i, val in enumerate(sorted_up)}
        map_dn = {val: i for i, val in enumerate(sorted_dn)}
        cluster_id_up = np.vectorize(map_up.get)(cluster_id_up)
        cluster_id_dn = np.vectorize(map_dn.get)(cluster_id_dn)
        cluster_id[idx_up] = cluster_id_up
        cluster_id[idx_dn] = cluster_id_dn+n_up
        return cluster_id
    
    def plot_cluster_type_percentage(self, ax):
        #fig, ax = plt.subplots(1, 1, figsize=(3, 3))
        colors = ['hotpink', 'coral', 'cornflowerblue', 'mediumseagreen']
        lbls = ['stim-act-fast','stim-act-slow', 'stim-inh-fast', 'stim-inh-slow']
        #num = [1374, 455+1821, 572+420, 1875+1068+593]
        num = [548+923, 734+475, 265+301, 250+701+734]
        ax.pie(
            num,
            labels=[f'{lbls[i]}' for i in range(4)],
            colors=colors,
            autopct='%1.1f%%',
            wedgeprops={'linewidth': 1, 'edgecolor':'white', 'width':0.2})

    def plot_mean_sem(self, ax, t, m, s, c, l=None, a=1.0):
        ax.plot(t, m, color=c, label=l, alpha=a)
        ax.fill_between(t, m - s, m + s, color=c, alpha=0.25, edgecolor='none')
        ax.set_xlim([np.min(t), np.max(t)])
    
    def plot_half_violin(self, ax, data, x, color, side):
        p = ax.violinplot(data[~np.isnan(data)], positions=[x], widths=1, showextrema=False)
        v = p['bodies'][0].get_paths()[0].vertices
        p['bodies'][0].set(facecolor=color, alpha=0.3)
        p['bodies'][0].set_edgecolor(color)
        p['bodies'][0].set_linewidth(1)
        m = v[:,0].mean()
        v[:,0] = np.minimum(v[:,0], m) if side.lower().startswith('l') else np.maximum(v[:,0], m)

    def plot_vol(self, ax, st, sv, c, u, l):
        v = np.mean(sv, axis=0)
        v = rescale(v, u, l)
        ax.plot(st, v, color=c, lw=0.5, linestyle=':')
    
    def plot_pupil(self, ax, nt, cp, c, u, l):
        cp = rescale(cp, u, l)
        ax.plot(nt, cp, color=c, lw=0.5, linestyle=':')
    
    def plot_dist(self, ax, data, c, cumulative):
        bins = 25
        # raw counts
        counts, bin_edges = np.histogram(data, bins=bins)
        # fraction of samples in each bin.
        total = counts.sum()
        fractions = counts / total  
        # bin centers
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        # cumulative.
        if cumulative:
            fractions = np.cumsum(fractions)
        # plot line.
        ax.plot(bin_centers, fractions, color=c)
    
    def plot_scatter(self, ax, q1, q2, c):
        # subsampling if lengths differ.
        if len(q1) > len(q2):
            q1 = q1[np.random.choice(len(q1), size=len(q2), replace=False)]
        if len(q1) < len(q2):
            q2 = q2[np.random.choice(len(q2), size=len(q1), replace=False)]          
        # find bounds.
        upper = np.nanmax([q1, q2])
        lower = np.nanmin([q1, q2])
        # plot scatter.
        ax.scatter(q1, q2, color=c, s=1, alpha=0.5)
        # plot unit line.
        ax.plot(
            np.linspace(lower,upper,1106),
            np.linspace(lower,upper,1106),
            lw=1, linestyle=':', color='#2C2C2C')
        # plot mean value.
        ax.scatter(
            np.nanmean(q1), np.nanmean(q2),
            color='black', marker='x', s=10)
        # plot statistics test.
        r_m = get_stat_test(q1, q2, 'mean')[1]
        r_v = get_stat_test(q1, q2, 'var')[1]
        ax.text(
            upper-0.05*(upper-lower), upper-0.4*(upper-lower), self.stat_sym[r_m],
            ha='center', va='center')
        ax.text(
            upper-0.05*(upper-lower), upper-0.6*(upper-lower), self.stat_sym[r_v],
            ha='center', va='center')
        # adjust layouts.
        adjust_layout_scatter(ax, upper, lower)
    
    def plot_glm_kernel(self, ax, kernel_all, cluster_id, color, height):
        gap = 0.05
        # define layouts.
        ax.axis('off')
        ax_lbl = ax.inset_axes([0, 0, 0.4, height], transform=ax.transAxes)
        ax_glm = ax.inset_axes([0.5, 0, 0.5, height], transform=ax.transAxes)
        # get cluster average.
        glm_mean, glm_sem = get_mean_sem_cluster(kernel_all, self.n_clusters, cluster_id)
        norm_params = [get_norm01_params(glm_mean[ci,:]) for ci in range(self.n_clusters)]
        # plot results.
        ax_glm.axvline(0, color=color, lw=1, linestyle='--')
        self.plot_cluster_mean_sem(
            ax_glm, glm_mean, glm_sem, self.glm['kernel_time'],
            norm_params, None, None, [color]*self.n_clusters,
            [np.nanmin(self.glm['kernel_time']), np.nanmax(self.glm['kernel_time'])])
        # adjust layouts.
        ax_lbl.spines['right'].set_visible(False)
        ax_lbl.spines['top'].set_visible(False)
        ax_lbl.spines['bottom'].set_visible(False)
        ax_lbl.set_ylim([-0.2, self.n_clusters+0.1])
        ax_lbl.set_xticks([])
        ax_lbl.set_yticks([(1-2*gap)+gap+self.n_clusters-ci-1.5 for ci in range(self.n_clusters)])
        ax_lbl.set_yticklabels(np.arange(self.n_clusters))
        ax_lbl.set_ylabel('Cluster ID')
        ax_glm.set_xlabel('time since stim (ms)')
        
    def plot_heatmap_neuron(
            self, ax_hm, ax_cb, neu_seq, neu_time, neu_seq_sort,
            sort_method='rastermap',
            norm_mode=None, neu_seq_share=None,
            cbar_label=None,
            ):
        n_yticks = 2
        max_pixel = 258
        if len(neu_seq) > 0:
            # exclude pure nan row.
            neu_idx = np.where(~np.all(np.isnan(neu_seq), axis=1))[0]
            neu_seq = neu_seq[neu_idx,:].copy()
            neu_seq_sort = neu_seq_sort[neu_idx,:].copy()
            # sort heatmap.
            sorted_idx = sort_heatmap_neuron(neu_seq_sort, sort_method=sort_method)
            data = neu_seq[sorted_idx,:].copy()
            n_neurons = data.shape[0]
            # reduce pixels.
            nbin = int(neu_seq.shape[0] / max_pixel)
            data = rm.utils.bin1d(data, bin_size=nbin, axis=0)
            # compute share scale if give.
            if neu_seq_share != None:
                data_share = np.concatenate(neu_seq_share, axis=0)
            else:
                data_share = np.nan
            # prepare heatmap.
            hm_data, hm_norm, hm_cmap = apply_colormap(data, norm_mode, data_share)
            # plot heatmap.
            ax_hm.imshow(
                hm_data,
                extent=[neu_time[0], neu_time[-1], 1, hm_data.shape[0]],
                interpolation='nearest', aspect='auto')
            # adjust layouts.
            adjust_layout_heatmap(ax_hm)
            ax_hm.set_ylabel('neuron id (sorted)')
            ax_hm.tick_params(axis='y', labelrotation=90)
            ax_hm.set_yticks((((np.arange(n_yticks)+0.5)/n_yticks)*data.shape[0]).astype('int32'))
            ax_hm.set_yticklabels((((np.arange(n_yticks)+0.5)/n_yticks)*n_neurons).astype('int32'))
            # add colorbar.
            add_heatmap_colorbar(ax_cb, hm_cmap, hm_norm, 'dF/F')
    
    def plot_heatmap_trial(
            self, ax_hm, ax_cb, neu_seq, neu_time,
            norm_mode=None, neu_seq_share=None,
            cbar_label=None,
            ):
        n_yticks = 2
        max_pixel = 258
        if len(neu_seq) > 0:
            # exclude pure nan row.
            neu_idx = np.where(~np.all(np.isnan(neu_seq), axis=1))[0]
            neu_seq = neu_seq[neu_idx,:].copy()
            # reduce pixels.
            nbin = int(neu_seq.shape[0] / max_pixel)
            data = rm.utils.bin1d(neu_seq, bin_size=nbin, axis=0)
            # compute share scale if give.
            if neu_seq_share != None:
                data_share = np.concatenate(neu_seq_share, axis=0)
                data_share = rm.utils.bin1d(data_share, bin_size=nbin, axis=0)
            else:
                data_share = np.nan
            # prepare heatmap.
            hm_data, hm_norm, hm_cmap = apply_colormap(data, norm_mode, data_share)
            # plot heatmap.
            ax_hm.imshow(
                hm_data,
                extent=[neu_time[0], neu_time[-1], 1, hm_data.shape[0]],
                interpolation='nearest', aspect='auto')
            # adjust layouts.
            adjust_layout_heatmap(ax_hm)
            ax_hm.set_yticks((((np.arange(n_yticks)+0.5)/n_yticks)*data.shape[0]).astype('int32'))
            ax_hm.set_yticklabels((((np.arange(n_yticks)+0.5)/n_yticks)*neu_seq.shape[0]).astype('int32'))
            # add colorbar.
            add_heatmap_colorbar(ax_cb, hm_cmap, hm_norm, 'dF/F')

    def plot_win_mag_quant_win_eval(
            self, ax, win_eval, color, xlim, baseline=True
            ):
        if baseline:
            ax.plot(win_eval[0], [1,1], color=color, linestyle=':', marker='|')
        for i in range(len(win_eval)-1):
            ax.plot(win_eval[i+1], [1,1], color=color, marker='|')
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xlim(xlim)
        ax.set_ylim([0,1.1])
        ax.set_yticks([])
                
    def plot_win_mag_quant(
            self, ax, neu_seq, neu_time,
            win_eval, color, c_time, offset,
            ):
        mode = ['lower', 'mean', 'mean', 'mean']
        # compute response within window.
        quant = [get_mean_sem_win(
            neu_seq.reshape(-1, neu_seq.shape[-1]),
            neu_time, c_time, win_eval[i][0], win_eval[i][1], mode=mode[i])
            for i in range(4)]
        m = np.array([quant[i][1] for i in range(4)])
        s = np.array([quant[i][2] for i in range(4)])
        # plot errorbar.
        for i in [1,2,3]:
            ax.errorbar(
                i + offset,
                m[i]-m[0], s[i],
                color=color,
                capsize=2, marker='o', linestyle='none',
                markeredgecolor='white', markeredgewidth=0.1)
        # adjust layouts.
        ax.tick_params(tick1On=False)
        ax.tick_params(axis='x', labelrotation=90)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_ylabel('evoked magnitude')
        ax.set_xticks([1,2,3])
        ax.set_xticklabels(['early', 'late', 'post'])
        ax.set_xlim([0.5, 3.5])

    def plot_multi_sess_decoding_num_neu(
            self, ax,
            sampling_nums, acc_model, acc_chance,
            color1, color2
            ):
        # compute mean and sem.
        acc_mean_model  = np.array([get_mean_sem(a)[0] for a in acc_model]).reshape(-1)
        acc_sem_model   = np.array([get_mean_sem(a)[1] for a in acc_model]).reshape(-1)
        acc_mean_chance = np.array([get_mean_sem(a)[0] for a in acc_chance]).reshape(-1)
        acc_sem_chance  = np.array([get_mean_sem(a)[1] for a in acc_chance]).reshape(-1)
        # plot results.
        self.plot_mean_sem(ax, sampling_nums, acc_mean_model,  acc_sem_model,  color2, 'model')
        self.plot_mean_sem(ax, sampling_nums, acc_mean_chance, acc_sem_chance, color1, 'chance')
        ax.tick_params(tick1On=False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xlabel('number of sampled neurons')
        add_legend(ax, [color2,color1], ['model','chance'], 'upper left')

    def plot_multi_sess_decoding_slide_win(
            self, ax,
            eval_time, acc_model, acc_chance,
            color1, color2
            ):
        # compute mean and sem.
        acc_mean_model  = np.array([get_mean_sem(a)[0] for a in acc_model]).reshape(-1)
        acc_sem_model   = np.array([get_mean_sem(a)[1] for a in acc_model]).reshape(-1)
        acc_mean_chance = np.array([get_mean_sem(a)[0] for a in acc_chance]).reshape(-1)
        acc_sem_chance  = np.array([get_mean_sem(a)[1] for a in acc_chance]).reshape(-1)
        # compute bounds.
        upper = np.nanmax([acc_mean_model,acc_mean_chance]) + np.nanmax([acc_sem_model,acc_sem_chance])
        lower = np.nanmin([acc_mean_model,acc_mean_chance]) - np.nanmax([acc_sem_model,acc_sem_chance])
        # plot results.
        self.plot_mean_sem(ax, eval_time, acc_mean_model,  acc_sem_model,  color2, 'model')
        self.plot_mean_sem(ax, eval_time, acc_mean_chance, acc_sem_chance, color1, 'chance')
        ax.tick_params(tick1On=False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
        add_legend(ax, [color2,color1], ['model','chance'], 'upper left')

    def plot_cluster_neu_fraction_in_cluster(self, ax, cluster_id, color):
        bar_width = 0.2
        # get fraction in each cluster.
        num = np.array([np.nansum(cluster_id==i) for i in range(self.n_clusters)])
        fraction = num / (np.nansum(num) + 1e-5)
        # plot bars.
        ax.axis('off')
        ax = ax.inset_axes([0, 0, 0.6, 1], transform=ax.transAxes)
        for ci in range(self.n_clusters):
            ax.barh(
                self.n_clusters-ci-1+0.5, fraction[ci],
                left=0,
                edgecolor='white',
                height=bar_width,
                color=color)
            ax.text(0.01, self.n_clusters-ci-1+0.7,
                    f'{fraction[ci]:.2f}, n={num[ci]}',
                    ha='left', va='center', color='#2C2C2C')
        # adjust layouts.
        ax.tick_params(tick1On=False)
        ax.tick_params(axis='y', labelrotation=90)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim([0,np.nanmax(fraction)])
        ax.set_ylim([-0.2, self.n_clusters+0.1])
        ax.set_xlabel('fraction of neurons')

    def plot_cluster_cate_fraction_in_cluster(self, ax, cluster_id, neu_labels, label_names, color=None):
        bar_width = 0.5
        cate_eval = [int(k) for k in label_names.keys()]
        if color != None:
            cate_color = [color]*len(cate_eval) 
        else:
            cate_color = [get_roi_label_color(cate=[c])[2] for c in cate_eval]
        # get fraction in each category.
        fraction = np.zeros((len(cate_eval), self.n_clusters))
        for i in range(len(cate_eval)):
            for j in range(self.n_clusters):
                nc = np.nansum((cluster_id==j)*(neu_labels==cate_eval[i]))
                nt = np.nansum(neu_labels==cate_eval[i])
                fraction[i,j] = nc / (nt + 1e-5)
        # plot bars.
        ax.axis('off')
        axs = [ax.inset_axes([0, ci/self.n_clusters, 0.8, 0.8/self.n_clusters], transform=ax.transAxes)
               for ci in range(self.n_clusters)]
        axs.reverse()
        for ci in range(self.n_clusters):
            for i in range(len(cate_eval)):
                if fraction[i,ci] != 0:
                    axs[ci].bar(
                        i, fraction[i,ci],
                        edgecolor='white',
                        width=bar_width,
                        color=cate_color[i])
                axs[ci].text(
                    i, fraction[i,ci],
                    f' {fraction[i,ci]:.2f}',
                    ha='center', va='bottom', color='#2C2C2C')
        # adjust layouts.
        for ci in range(self.n_clusters):
            axs[ci].tick_params(tick1On=False)
            axs[ci].spines['left'].set_visible(False)
            axs[ci].spines['right'].set_visible(False)
            axs[ci].spines['top'].set_visible(False)
            axs[ci].set_xticks([])
            axs[ci].set_yticks([])
            axs[ci].set_xlim([-0.5,2.5])
            axs[ci].set_ylim([0, np.nanmax(fraction)*1.25])
        axs[self.n_clusters-1].tick_params(axis='x', labelrotation=90)
        axs[self.n_clusters-1].set_xticks([0,1,2])
        axs[self.n_clusters-1].set_xticklabels([v for v in label_names.values()])
        axs[self.n_clusters-1].set_xlabel('fraction of cell-type')

    def plot_cluster_mean_sem(
            self, ax, neu_mean, neu_sem, neu_time,
            norm_params, stim_seq, c_stim, c_neu, xlim,
            scale_bar=True
            ):
        gap = 0.05
        l_nan_margin = 5
        r_nan_margin = 5
        len_scale_y = 0.5
        # set margin values to nan.
        l_idx, r_idx = get_frame_idx_from_time(neu_time, 0, xlim[0], xlim[1])
        nm = neu_mean.copy()
        ns = neu_sem.copy()
        nm[:,:l_idx+l_nan_margin] = np.nan
        nm[:,r_idx-r_nan_margin-1:] = np.nan
        ns[:,:l_idx+l_nan_margin] = np.nan
        ns[:,r_idx-r_nan_margin-1:] = np.nan
        # plot stimulus.
        if not stim_seq is None:
            for si in range(stim_seq.shape[0]):
                if stim_seq[si,0] >= xlim[0] and stim_seq[si,1] <= xlim[1]:
                    ax.fill_between(
                        stim_seq[si,:],
                        0, nm.shape[0],
                        color=c_stim[si], edgecolor='none', alpha=0.25, step='mid')
        # plot cluster average.
        for ci in range(self.n_clusters):
            a, b, c, d = norm_params[ci]
            # add y=0 line.
            #ax.hlines(ci+b, xlim[0]*0.99, xlim[1]*0.99, linestyle=':', color='#2C2C2C', alpha=0.2)
            # plot neural traces.
            self.plot_mean_sem(
                ax, neu_time,
                (a*nm[ci,:]+b)*(1-2*gap)+gap+self.n_clusters-ci-1, np.abs(a)*ns[ci,:],
                c_neu[ci], None)
            # plot y scalebar.
            if scale_bar:
                y_start = ci + 0.5 - len_scale_y/2
                ax.vlines(xlim[0]*0.99, y_start, y_start+len_scale_y, color='#2C2C2C')
                ax.text(xlim[0]*0.99, ci+0.5,
                    '{:.3f}'.format(len_scale_y*(d-c)),
                    va='center', ha='right', rotation=90, color='#2C2C2C')
        # plot x scalebar.
        if scale_bar:
            ax.set_ylim([-0.2, neu_mean.shape[0]+0.1])
        # adjust layouts.
        adjust_layout_cluster_neu(ax, nm.shape[0], xlim)
    
    def plot_cluster_box(self, ax, cluster_id, means, sems, colors):
        # plot results for each class.                 
        for ci in range(self.n_clusters):
            if np.sum(cluster_id==ci) > 0:
                ax.errorbar(
                    means[ci], self.n_clusters-ci-1,
                    xerr=sems[ci],
                    color=colors[ci],
                    capsize=2, marker='o', linestyle='none',
                    markeredgecolor='white', markeredgewidth=0.1)
        # adjust layouts.
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_yticks([])
        
    def plot_cluster_win_mag_quant(
            self, axs, day_cluster_id, neu_seq, neu_time,
            win_eval, color, c_time, offset, average_axis=0
            ):
        baseline_correction = False
        mode = ['lower', 'mean', 'mean', 'mean']
        # plot results for each class.
        for ci in range(self.n_clusters):
            # average.
            neu_ci = np.concatenate(
                [np.nanmean(neu[:,dci==ci,:],axis=average_axis)
                 for neu,dci in zip(neu_seq,day_cluster_id)], axis=0)
            # compute response within window.
            quant = [get_mean_sem_win(
                neu_ci.reshape(-1, neu_ci.shape[-1]),
                neu_time, c_time, win_eval[i][0], win_eval[i][1], mode=mode[i])
                for i in range(4)]
            m = np.array([quant[i][1] for i in range(4)])
            s = np.array([quant[i][2] for i in range(4)])
            # plot errorbar.
            for i in [1,2,3]:
                mi = m[i]-m[0] if baseline_correction else m[i]
                axs[ci].errorbar(
                    i + offset,
                    mi, s[i],
                    color=color,
                    capsize=2, marker='o', linestyle='none',
                    markeredgecolor='white', markeredgewidth=0.1)
        # adjust layouts.
        for ci in range(self.n_clusters):
            axs[ci].tick_params(axis='x', labelrotation=90, tick1On=False)
            axs[ci].tick_params(axis='y')
            axs[ci].spines['right'].set_visible(False)
            axs[ci].spines['top'].set_visible(False)
            axs[ci].set_xlim([0.5, 3.5])
            axs[ci].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
            axs[ci].set_xticks([])
        axs[self.n_clusters-1].set_xticks([1,2,3])
        axs[self.n_clusters-1].set_xticklabels(['early', 'late', 'post'])
        axs[self.n_clusters-1].set_xlabel('quantification window \n across trials')
    
    def plot_cluster_win_mag_quant_stat(
            self, axs, day_cluster_id, neu_seq_1, neu_seq_2, neu_time,
            win_eval, c_time, average_axis, stat_test
            ):
        # plot results for each class.
        for ci in range(self.n_clusters):
            # average across neurons within cluster.
            neu_1 = np.concatenate(
                [np.nanmean(ns1[:,dci==ci,:],axis=average_axis)
                 for ns1,dci in zip(neu_seq_1,day_cluster_id)], axis=0)
            neu_2 = np.concatenate(
                [np.nanmean(ns2[:,dci==ci,:],axis=average_axis)
                 for ns2,dci in zip(neu_seq_2,day_cluster_id)], axis=0)
            # get significance level.
            r_m = get_win_mag_quant_stat_test(neu_1, neu_2, neu_time, c_time, win_eval, stat_test)[1]
            # plot results.
            for i in range(3):
                axs[ci].text(i+1, 0, self.stat_sym[r_m[i]], ha='center', va='center')
        # adjust layouts.
        for ci in range(self.n_clusters):
            axs[ci].axis('off')
            axs[ci].set_xlim([0.5, 3.5])
    
    def plot_cluster_win_mag_scatter(
            self, axs, day_cluster_id, neu_seq_1, neu_seq_2, neu_time,
            win_eval_1, win_eval_2, color, c_time,
            average_axis=0, baseline_correction=False
            ):
        mode = ['lower', 'mean']
        # plot results for each class.
        for ci in range(self.n_clusters):
            # organize results.
            neu_ci_1 = np.concatenate(
                [np.nanmean(neu[:,dci==ci,:],axis=average_axis)
                 for neu,dci in zip(neu_seq_1,day_cluster_id)], axis=0)
            neu_ci_2 = np.concatenate(
                [np.nanmean(neu[:,dci==ci,:],axis=average_axis)
                 for neu,dci in zip(neu_seq_2,day_cluster_id)], axis=0)
            # compute response within window.
            quant_1 = [get_mean_sem_win(
                neu_ci_1.reshape(-1, neu_ci_1.shape[-1]),
                neu_time, c_time, win_eval_1[i][0], win_eval_1[i][1], mode=mode[i])
                for i in range(2)]
            quant_2 = [get_mean_sem_win(
                neu_ci_2.reshape(-1, neu_ci_2.shape[-1]),
                neu_time, c_time, win_eval_2[i][0], win_eval_2[i][1], mode=mode[i])
                for i in range(2)]
            # get single neuron response with baseline correction.
            if baseline_correction:
                q1 = quant_1[1][0] - quant_1[0][1]
                q2 = quant_2[1][0] - quant_2[0][1]
            else:
                q1 = quant_1[1][0]
                q2 = quant_2[1][0]
            # plot results.
            self.plot_scatter(axs[ci], q1, q2, color)
                    
    def plot_cluster_win_mag_dist_compare(
            self, axs, day_cluster_id, neu_seq_1, neu_seq_2, neu_time,
            win_eval, c_neu, c_time, cumulative
            ):
        len_scale = 0.5
        baseline_correction = False
        mode = ['lower', 'mean']
        # plot results for each class.
        for ci in range(self.n_clusters):
            # organize results.
            neu_ci_1 = np.concatenate(
                [neu[:,dci==ci,:].reshape(-1, neu.shape[2])
                 for neu,dci in zip(neu_seq_1,day_cluster_id)], axis=0)
            neu_ci_2 = np.concatenate(
                [neu[:,dci==ci,:].reshape(-1, neu.shape[2])
                 for neu,dci in zip(neu_seq_2,day_cluster_id)], axis=0)
            # compute response within window.
            quant_1 = [get_mean_sem_win(
                neu_ci_1.reshape(-1, neu_ci_1.shape[-1]),
                neu_time, c_time, win_eval[i][0], win_eval[i][1], mode=mode[i])
                for i in range(2)]
            quant_2 = [get_mean_sem_win(
                neu_ci_2.reshape(-1, neu_ci_2.shape[-1]),
                neu_time, c_time, win_eval[i][0], win_eval[i][1], mode=mode[i])
                for i in range(2)]
            # get single neuron response with baseline correction.
            if baseline_correction:
                q1 = quant_1[1][0] - quant_1[0][1]
                q2 = quant_2[1][0] - quant_2[0][1]
            else:
                q1 = quant_1[1][0]
                q2 = quant_2[1][0]
            # plot empirical distribution.
            d1 = self.plot_dist(axs[ci], q1, c_neu[0], cumulative)
            d2 = self.plot_dist(axs[ci], q2, c_neu[1], cumulative)
            # find bounds.
            xu = np.nanmax(np.concatenate([q1, q2]))
            xl = np.nanmin(np.concatenate([q1, q2]))
            yu = np.nanmax(np.concatenate([d1, d2]))
            xlim = [xl - 0.1*(xu-xl), xu - 0.4*(xu-xl)]
            # plot statistics test.
            r = get_stat_test(q1, q2, 'mean')[1]
            axs[ci].text(
                xl-0.05*(xu-xl), yu*(1-len_scale-0.2), self.stat_sym[r],
                ha='center', va='center')
            # plot scale bar.
            axs[ci].vlines(
                xl-0.05*(xu-xl),
                yu*(1-len_scale), yu,
                color='#2C2C2C')
            axs[ci].text(xl-0.09*(xu-xl), yu*(1-len_scale/2),
                '{:.2f}'.format(len_scale*yu),
                va='center', ha='center', rotation=90, color='#2C2C2C')
            # adjust layouts.
            axs[ci].tick_params(axis='x')
            axs[ci].tick_params(axis='y', tick1On=False)
            axs[ci].xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
            axs[ci].xaxis.set_major_locator(mtick.MaxNLocator(nbins=4))
            axs[ci].spines['left'].set_visible(False)
            axs[ci].spines['right'].set_visible(False)
            axs[ci].spines['top'].set_visible(False) 
            axs[ci].set_yticks([])
            axs[ci].set_xlim(xlim)
            axs[ci].set_ylim([0, yu*1.1])
    
    def plot_cluster_pred_mod_index_compare(
            self, axs, day_cluster_id, neu_seq_1, neu_seq_2, neu_time,
            win_eval_1, win_eval_2, color1, color2, c_time,
            average_axis=0, baseline_correction=False
            ):
        mode = ['lower', 'mean', 'mean']
        # plot results for each class.
        for ci in range(self.n_clusters):
            # organize results.
            neu_ci_1 = np.concatenate(
                [np.nanmean(neu[:,dci==ci,:],axis=average_axis)
                 for neu,dci in zip(neu_seq_1,day_cluster_id)], axis=0)
            neu_ci_2 = np.concatenate(
                [np.nanmean(neu[:,dci==ci,:],axis=average_axis)
                 for neu,dci in zip(neu_seq_2,day_cluster_id)], axis=0)
            # compute response within window.
            quant_1 = [get_mean_sem_win(
                neu_ci_1.reshape(-1, neu_ci_1.shape[-1]),
                neu_time, c_time, win_eval_1[i][0], win_eval_1[i][1], mode=mode[i])
                for i in range(3)]
            quant_2 = [get_mean_sem_win(
                neu_ci_2.reshape(-1, neu_ci_2.shape[-1]),
                neu_time, c_time, win_eval_2[i][0], win_eval_2[i][1], mode=mode[i])
                for i in range(3)]
            # get single neuron response with baseline correction.
            if baseline_correction:
                q1 = [quant_1[1][0] - quant_1[0][1], quant_1[2][0] - quant_1[0][1]]
                q2 = [quant_2[1][0] - quant_2[0][1], quant_2[2][0] - quant_2[0][1]]
            else:
                q1 = [quant_1[1][0], quant_1[2][0]]
                q2 = [quant_2[1][0], quant_2[2][0]]
            # compute modulation index.
            mod1 = get_modulation_index(q1[1], q1[0])
            mod2 = get_modulation_index(q2[1], q2[0])
            m1, s1 = get_mean_sem(mod1.reshape(-1,1))
            m2, s2 = get_mean_sem(mod2.reshape(-1,1))
            # plot errorbar.
            self.plot_half_violin(axs[ci], mod1, 0, color1, 'left')
            self.plot_half_violin(axs[ci], mod2, 1, color2, 'right')
            axs[ci].errorbar(
                0, m1, s1,
                color='black',
                capsize=2, marker='o', linestyle='none',
                markeredgecolor='white', markeredgewidth=0.1)
            axs[ci].errorbar(
                1, m2, s2,
                color='black',
                capsize=2, marker='o', linestyle='none',
                markeredgecolor='white', markeredgewidth=0.1)
            # plot statistics test.
            r_m = get_stat_test(mod1, mod2, 'mean')[1]
            axs[ci].text(0.5, 0.8, self.stat_sym[r_m], ha='center', va='center')
            # adjust layouts.
            axs[ci].tick_params(axis='x', tick1On=False)
            axs[ci].yaxis.set_major_locator(mtick.MaxNLocator(nbins=2))
            axs[ci].spines['right'].set_visible(False)
            axs[ci].spines['top'].set_visible(False) 
            axs[ci].set_xlim([-1,2])
            axs[ci].set_ylim([-1.2, 1.2])
            axs[ci].set_xticks([0,1])
                
    def plot_cluster_heatmap_trial(
            self, axs_hm, axs_cb,
            neu_x, neu_time, norm_mode,
            ):
        for ci in range(self.n_clusters):
            neu_h, norm, cmap = apply_colormap(
                neu_x[ci], norm_mode=norm_mode, data_share=neu_x)
            # plot heatmap.
            axs_hm[ci].imshow(
                neu_h,
                extent=[neu_time[0], neu_time[-1], 1, neu_h.shape[0]],
                interpolation='nearest', aspect='auto')
            # adjust layouts.
            adjust_layout_heatmap(axs_hm[ci])
            add_heatmap_colorbar(axs_cb[ci], cmap, norm, 'dF/F')
        
    def plot_cluster_heatmap(self, ax, neu_seq, neu_time, cluster_id, norm_mode):
        gap = 5
        win_conv = 5
        win_sort = [-500, 500]
        # create heatmap for all clusters.
        data = []
        yticks = []
        yticklabels = []
        for ci in range(self.n_clusters):
            # get data within cluster.
            idx = cluster_id==(self.n_clusters-ci-1)
            if np.sum(idx) > 0:
                neu = neu_seq[idx,:].copy()
                # smooth the values in the sorting window.
                l_idx, r_idx = get_frame_idx_from_time(neu_time, 0, win_sort[0], win_sort[1])
                smoothed_mean = np.array(
                    [np.convolve(row, np.ones(win_conv)/win_conv, mode='same')
                     for row in neu[:,l_idx:r_idx]])
                sort_idx_neu = np.argmax(smoothed_mean, axis=1).reshape(-1).argsort()
                # rearrange the matrix.
                neu = neu[sort_idx_neu,:]
                # add into full heatmap.
                data.append(neu)
                data.append(np.nan*np.ones([gap,neu.shape[1]]))
                yticks.append(neu.shape[0]+gap)
                yticklabels.append(ci)
            else: pass
        # plot heatmap.
        data = np.concatenate(data,axis=0)
        hm_data, hm_norm, hm_cmap = apply_colormap(data, norm_mode, data)
        ax.imshow(
            hm_data,
            extent=[neu_time[0], neu_time[-1], 1, hm_data.shape[0]],
            interpolation='nearest', aspect='auto',
            origin='lower')
        # adjust layouts.
        adjust_layout_heatmap(ax)
        ax.set_ylabel('neuron id (clustered)')
        ax.set_yticks(np.cumsum(yticks)-gap)
        ax.set_yticklabels([f'#{str(ci).zfill(2)}' for ci in yticklabels])
    
    def plot_dendrogram(self, ax, kernel_all, cmap):
        p=5
        # run hierarchical clustering.
        model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
        model.fit(kernel_all)
        # create the counts of samples under each node.
        counts = np.zeros(model.children_.shape[0])
        n_samples = len(model.labels_)
        for i, merge in enumerate(model.children_):
            current_count = 0
            for child_idx in merge:
                if child_idx < n_samples:
                    current_count += 1
                else:
                    current_count += counts[child_idx - n_samples]
            counts[i] = current_count
        # create linkage matrix.
        linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)
        # create colors.
        dists = linkage_matrix[:, 2]
        norm = mcolors.Normalize(vmin=dists.min(), vmax=dists.max())
        color_lookup = {
            i + n_samples: mcolors.rgb2hex(cmap(norm(dist)))
            for i, dist in enumerate(dists)}
        # plot dendrogram.
        dendrogram(
            linkage_matrix,
            truncate_mode='level',
            p=p,
            ax=ax,
            color_threshold=0,
            link_color_func=lambda idx: color_lookup.get(idx, '#000000'))
        # adjust layouts.
        ax.tick_params(tick1On=False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xlabel('number of points in node')
        ax.set_yticks([])

    def plot_3d_latent_dynamics(self, ax, neu_z, stim_seq, neu_time, end_color='black', c_stim=None, cmap=None, add_stim=True, add_mark=None):
        resampling = 5
        cmap = self.latent_cmap if cmap==None else cmap
        # interpolation.
        t = np.linspace(neu_time[0], neu_time[-1], resampling*len(neu_time))
        z = np.zeros([3, len(t)])
        for ri in range(3):
            z[ri,:] = np.interp(t, neu_time, neu_z[ri,:])
        z = np.apply_along_axis(savgol_filter, 1, z, window_length=resampling*2, polyorder=3)
        # plot dynamics.
        c_neu = get_cmap_color(z.shape[1], cmap=cmap)
        for ti in range(z.shape[1]-1):
            ax.plot(z[0,ti:ti+2]-z[0,0], z[1,ti:ti+2]-z[1,0], z[2,ti:ti+2]-z[2,0], color=c_neu[ti], lw=0.5)
        # end point.
        ax.scatter(z[0,0]-z[0,0], z[1,0]-z[1,0], z[2,0]-z[2,0], color='black', marker='x', lw=1)
        ax.scatter(z[0,-1]-z[0,0], z[1,-1]-z[1,0], z[2,-1]-z[2,0], color=end_color, marker='o', lw=1)
        # stimulus.
        if add_stim:
            for j in range(stim_seq.shape[0]):
                idx = get_frame_idx_from_time(t, 0, stim_seq[j,0], 0)[0]
                if idx > 0 and idx < len(t):
                    ax.scatter(z[0,idx]-z[0,0], z[1,idx]-z[1,0], z[2,idx]-z[2,0], color=c_stim[j], marker='s', lw=0.25)
        # add cretical mark point.
        if add_mark != None:
            for tm, c_mark in add_mark:
                idx = get_frame_idx_from_time(t, 0, tm, 0)[0]
                if idx > 0 and idx < len(t):
                    ax.scatter(z[0,idx]-z[0,0], z[1,idx]-z[1,0], z[2,idx]-z[2,0], color=c_mark, marker='o', lw=2)
    
    def plot_dis_mat(self, ax_hm, ax_cb, d, annotate=False):
        cmap = plt.cm.hot
        # mask upper half.
        mask = np.tril(np.ones_like(d, dtype=bool), k=0)
        masked_mat = np.where(mask, d, np.nan)
        # plot matrix.
        ax_hm.matshow(masked_mat, interpolation='nearest', cmap=cmap)
        # annotate.
        if annotate:
            for i in range(d.shape[0]):
                for j in range(d.shape[1]):
                    if not np.isnan(masked_mat[i, j]):
                        ax_hm.text(j, i, f'{masked_mat[i, j]:.2f}',
                                    ha='center', va='center', color='grey')
        # adjust layout.
        ax_hm.tick_params(tick1On=True, bottom=False, top=False, labelbottom=True, labeltop=False)
        ax_hm.spines[:].set_visible(False)
        ax_hm.set_xticks(np.arange(d.shape[0]))
        ax_hm.set_yticks(np.arange(d.shape[0]))
    
    def plot_time_decode_confusion_matrix(self, ax_hm, acc_mat, t_range, ax_cb=None, vmax=None):
        cmap = plt.cm.magma
        vmax = 1 if vmax == None else vmax
        # plot matrix.
        ax_hm.matshow(
            acc_mat[::-1,:],
            extent=[t_range[0], t_range[-1], t_range[0], t_range[-1]],
            interpolation='nearest', cmap=cmap, vmin=0.45, vmax=vmax)
        # adjust layout.
        ax_hm.spines['right'].set_visible(False)
        ax_hm.spines['top'].set_visible(False)
        ax_hm.tick_params(tick1On=True, bottom=False, top=False, labelbottom=True, labeltop=False)
        # add colorbar.
        if ax_cb != None:
            norm = mcolors.Normalize(vmin=0.45, vmax=vmax if vmax==None else vmax)
            add_heatmap_colorbar(ax_cb, cmap, norm, 'decoding accuracy')


def _prepare_time_axis(time_vector_ms: np.ndarray, window_ms: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    """Return mask and cropped time axis for the requested window."""
    time_vector_ms = np.asarray(time_vector_ms)
    mask = (time_vector_ms >= window_ms[0]) & (time_vector_ms <= window_ms[1])
    if not mask.any():
        raise ValueError(
            f"Window {window_ms} ms contains no samples. Available range: {time_vector_ms.min():.1f}–{time_vector_ms.max():.1f} ms."
        )
    return mask, time_vector_ms[mask]


def plot_top_orientation_neurons(summary_df: pd.DataFrame,
                                 Y_trials: np.ndarray,
                                 trial_df: pd.DataFrame,
                                 time_vector_ms: np.ndarray,
                                 orientation_labels: list[str],
                                 top_n: int = 3,
                                 window_ms: tuple[float, float] = (-1000.0, 1000.0)) -> None:
    """Plot the mean peri-stimulus traces for the most orientation-tuned neurons."""
    if summary_df.empty:
        raise ValueError('summary_df is empty; run the parameter extraction cell first.')
    if Y_trials is None:
        raise ValueError('Y_trials is None; ensure the loader produced trial-wise responses.')

    mask, time_axis = _prepare_time_axis(time_vector_ms, window_ms)
    top_units = summary_df.sort_values('orientation_strength', ascending=False).head(top_n)

    for _, row in top_units.iterrows():
        neuron_idx = int(row['cell'])
        fig, ax = plt.subplots(figsize=(6.5, 4))
        for ori_idx, label in enumerate(orientation_labels, start=1):
            trials_sel = trial_df.index[trial_df['theta_idx'] == ori_idx]
            if len(trials_sel) == 0:
                continue
            traces = Y_trials[trials_sel][:, mask, neuron_idx]
            trace_mean = np.nanmean(traces, axis=0)
            ax.plot(time_axis, trace_mean, label=label)
        ax.axvline(0, color='k', linestyle='--', linewidth=1)
        ax.set_title(f'Orientation example neuron {neuron_idx}')
        ax.set_xlabel('Time relative to onset (ms)')
        ax.set_ylabel('Response (ΔF/F)')
        ax.legend(loc='upper right', fontsize=8)
        plt.tight_layout()
        plt.show()


def plot_top_transition_neurons(summary_df: pd.DataFrame,
                                 Y_trials: np.ndarray,
                                 trial_df: pd.DataFrame,
                                 time_vector_ms: np.ndarray,
                                 top_n: int = 3,
                                 window_ms: tuple[float, float] = (-1000.0, 1000.0)) -> None:
    """Plot mean traces for neurons that prefer orientation changes vs repeats."""
    if summary_df.empty:
        raise ValueError('summary_df is empty; run the parameter extraction cell first.')
    if Y_trials is None:
        raise ValueError('Y_trials is None; ensure the loader produced trial-wise responses.')

    mask, time_axis = _prepare_time_axis(time_vector_ms, window_ms)
    top_units = summary_df.sort_values('change_any_strength', ascending=False).head(top_n)

    repeat_idx = trial_df.index[trial_df['is_transition'] == False]
    change_idx = trial_df.index[trial_df['is_transition'] == True]

    for _, row in top_units.iterrows():
        neuron_idx = int(row['cell'])
        fig, ax = plt.subplots(figsize=(6.5, 4))
        if len(repeat_idx):
            repeat_traces = Y_trials[repeat_idx][:, mask, neuron_idx]
            ax.plot(time_axis, np.nanmean(repeat_traces, axis=0), label='repeat', color='tab:blue')
        if len(change_idx):
            change_traces = Y_trials[change_idx][:, mask, neuron_idx]
            ax.plot(time_axis, np.nanmean(change_traces, axis=0), label='transition', color='tab:orange')
        ax.axvline(0, color='k', linestyle='--', linewidth=1)
        ax.set_title(f'Change example neuron {neuron_idx}')
        ax.set_xlabel('Time relative to onset (ms)')
        ax.set_ylabel('Response (ΔF/F)')
        ax.legend(loc='upper right')
        plt.tight_layout()
        plt.show()




def plot_top_change_specific_neurons(summary_df: pd.DataFrame,
                                     Y_trials: np.ndarray,
                                     trial_df: pd.DataFrame,
                                     time_vector_ms: np.ndarray,
                                     top_n: int = 3,
                                     window_ms: tuple[float, float] = (-1000.0, 1000.0),
                                     include_other_pairs: bool = True,
                                     plot_average: bool = False) -> None:
    """Plot mean traces for neurons that prefer a specific orientation transition.

    Parameters
    ----------
    include_other_pairs : bool
        Overlay traces for all other orientation-change pairs (faint grey lines).
    plot_average : bool
        If True, also overlay the average trace across all change trials.
    """
    if summary_df.empty:
        raise ValueError('summary_df is empty; run the parameter extraction cell first.')
    if 'top_change_pair_strength' not in summary_df.columns:
        raise ValueError('summary_df must contain "top_change_pair_strength" and "top_change_pair" columns.')
    if Y_trials is None:
        raise ValueError('Y_trials is None; ensure the loader produced trial-wise responses.')

    mask, time_axis = _prepare_time_axis(time_vector_ms, window_ms)
    top_units = (summary_df.sort_values('top_change_pair_strength', ascending=False)
                             .head(top_n))

    change_pairs_all = trial_df['pair_label'].astype(str)
    transition_mask = change_pairs_all.str.contains('->')
    all_transition_idx = trial_df.index[transition_mask]

    for _, row in top_units.iterrows():
        pair_label = row.get('top_change_pair')
        strength = float(row.get('top_change_pair_strength', 0.0))
        if not pair_label or pair_label == 'N/A' or strength <= 0:
            continue
        trials_sel = trial_df.index[trial_df['pair_label'] == pair_label]
        if len(trials_sel) == 0:
            continue

        neuron_idx = int(row['cell'])
        traces = Y_trials[trials_sel][:, mask, neuron_idx]
        trace_mean = np.nanmean(traces, axis=0)

        fig, ax = plt.subplots(figsize=(6.5, 4))

        if include_other_pairs:
            other_pairs = sorted({p for p in change_pairs_all.unique() if p != pair_label and '->' in p})
            added_label = False
            for other in other_pairs:
                idx_sel = trial_df.index[trial_df['pair_label'] == other]
                if len(idx_sel) == 0:
                    continue
                other_trace = np.nanmean(Y_trials[idx_sel][:, mask, neuron_idx], axis=0)
                label = 'other transitions' if not added_label else None
                ax.plot(time_axis, other_trace, color='lightgrey', alpha=0.6, linewidth=1.0, label=label)
                added_label = True

        ax.plot(time_axis, trace_mean, label=pair_label, color='tab:purple', linewidth=2)

        if plot_average and len(all_transition_idx) > 0:
            avg_trace = np.nanmean(Y_trials[all_transition_idx][:, mask, neuron_idx], axis=0)
            ax.plot(time_axis, avg_trace, color='black', linestyle='--', linewidth=1.5, label='all transitions')

        ax.axvline(0, color='k', linestyle='--', linewidth=1)
        ax.set_title(f'Change-specific neuron {neuron_idx} ({pair_label})')
        ax.set_xlabel('Time relative to onset (ms)')
        ax.set_ylabel('Response (ΔF/F)')
        ax.legend(loc='upper right')
        plt.tight_layout()
        plt.show()
cat_order = ['orientation-pref', 'change-pref', 'change-specific', 'change-from', 'change-to', 'mixed']
cat_colors = {
    'orientation-pref': '#d62728',
    'change-pref': '#1f77b4',
    'change-specific': '#2ca02c',
    'change-from': '#9467bd',
    'change-to': '#8c564b',
    'mixed': '#7f7f7f',
}

def plot_transition_rastermaps(classification_df: pd.DataFrame,
                               Y_trials: np.ndarray,
                               trial_df: pd.DataFrame,
                               time_vector_ms: np.ndarray,
                               orientation_code_lookup: dict,
                               orientation_labels: list[str],
                               change_label: str,
                               top_k: int = 50,
                               window_ms: tuple[float, float] = (-1000.0, 1000.0),
                               cmap: str = 'magma') -> None:
    """Plot a neuron-by-time rastermap for a specific change pair, sorted by model category."""

    if classification_df.empty:
        raise ValueError('classification_df is empty; run the classification cell first.')
    if Y_trials is None:
        raise ValueError('Y_trials is None; ensure trial-wise responses are available.')

    mask, time_axis = _prepare_time_axis(time_vector_ms, window_ms)

    trials_sel = trial_df.index[trial_df['pair_label'] == change_label]
    if len(trials_sel) == 0:
        raise ValueError(f'No trials found for change_label="{change_label}". Available: {sorted(trial_df["pair_label"].unique())}')

    # Average across selected trials
    data = np.nanmean(Y_trials[trials_sel][:, mask, :], axis=0).T  # neurons × time

    cat_order = ['orientation-pref', 'change-pref', 'change-specific', 'change-from', 'change-to', 'mixed']
    cat_colors = {
        'orientation-pref': '#d62728',
        'change-pref': '#1f77b4',
        'change-specific': '#2ca02c',
        'change-from': '#9467bd',
        'change-to': '#8c564b',
        'mixed': '#7f7f7f',
    }

    info = classification_df[['cell', 'response_class']].copy()
    info['response_class'] = info['response_class'].astype(str)
    info.loc[~info['response_class'].isin(cat_colors), 'response_class'] = 'mixed'
    info['sort_key'] = info['response_class'].map({cls: idx for idx, cls in enumerate(cat_order)})
    info.sort_values(['sort_key', 'cell'], inplace=True)

    if top_k is not None:
        info = info.head(top_k)

    neurons = info['cell'].astype(int).to_numpy()
    classes = info['response_class'].to_list()

    data_sel = data[neurons]
    fig, (ax, ax_class) = plt.subplots(1, 2, figsize=(7.0, 5.0), gridspec_kw={'width_ratios': [20, 1]}, sharey=True)

    im = ax.imshow(data_sel, aspect='auto', cmap=cmap,
                   extent=[time_axis[0], time_axis[-1], 0, len(neurons)],
                   origin='lower')
    ax.axvline(0, color='white', linestyle='--', linewidth=1)
    ax.set_xlabel('Time relative to onset (ms)')
    ax.set_ylabel('Neuron (sorted by class)')
    ax.set_title(f'Raster: {change_label} (n_trials={len(trials_sel)})')

    # category color bar
    color_rgba = np.array([mcolors.to_rgba(cat_colors.get(cls, '#7f7f7f')) for cls in classes])
    ax_class.imshow(color_rgba[:, None, :], aspect='auto', origin='lower')
    ax_class.set_xticks([])
    ax_class.set_yticks([])
    ax_class.set_title('Class', fontsize=10)

    handles = [plt.Line2D([0], [0], color=cat_colors[c], linewidth=6) for c in cat_order]
    labels = cat_order
    ax.legend(handles, labels, title='Response class', bbox_to_anchor=(1.02, 1), loc='upper left')

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('ΔF/F')

    plt.tight_layout()
    plt.show()
