#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import to_hex

#%% general data processing

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
    a = 1 / (np.nanmax(data) - np.nanmin(data))
    b = - np.nanmin(data) / (np.nanmax(data) - np.nanmin(data))
    return a,b

# bin data and return index.
def get_bin_idx(data, bin_win, bin_num):
    bin_size = (bin_win[1] - bin_win[0]) / bin_num
    bins = np.arange(bin_win[0], bin_win[1] + bin_size, bin_size)
    bin_center = bins[:-1] + bin_size / 2
    bin_idx = np.digitize(data, bins) - 1
    return bins, bin_center, bin_idx

# compute baseline within given time window.
def get_base_mean_win(neu_seq, neu_time, c_time, win_base):
    pct = 25
    l_idx, r_idx = get_frame_idx_from_time(
        neu_time, c_time, win_base[0], win_base[1])
    neu = neu_seq[:, l_idx:r_idx].copy().reshape(-1)
    neu = neu[neu<np.nanpercentile(neu, pct)]
    mean_base = np.nanmean(neu)
    return mean_base

# compute mean and sem across trials for mean df/f within given time window.
def get_mean_sem_win(neu_seq, neu_time, c_time, l_time, r_time):
    l_idx, r_idx = get_frame_idx_from_time(
        neu_time, c_time, l_time, r_time)
    neu_win_mean = np.nanmean(neu_seq[:, l_idx:r_idx], axis=1)
    neu_mean = np.nanmean(neu_win_mean)
    std = np.nanstd(neu_win_mean, axis=0)
    count = np.nansum(~np.isnan(neu_win_mean), axis=0)
    neu_sem = std / np.sqrt(count)
    return neu_mean, neu_sem

# compute mean and sem for 3d array data
def get_mean_sem(data, zero_start=False):
    m = np.nanmean(data.reshape(-1, data.shape[-1]), axis=0)
    m = m-m[0] if zero_start else m
    std = np.nanstd(data.reshape(-1, data.shape[-1]), axis=0)
    count = np.nansum(~np.isnan(data.reshape(-1, data.shape[-1])), axis=0)
    s = std / np.sqrt(count)
    return m, s

#%% retreating neural data

# find trials based on stim_labels.
def pick_trial(
        stim_labels,
        img_seq_label,
        normal_types,
        fix_jitter_types,
        oddball_types,
        random_types,
        opto_types):
    idx1 = np.isin(stim_labels[:,2], img_seq_label)    if img_seq_label    else np.ones_like(stim_labels[:,2])
    idx2 = np.isin(stim_labels[:,3], normal_types)     if normal_types     else np.ones_like(stim_labels[:,3])
    idx3 = np.isin(stim_labels[:,4], fix_jitter_types) if fix_jitter_types else np.ones_like(stim_labels[:,4])
    idx4 = np.isin(stim_labels[:,5], oddball_types)    if oddball_types    else np.ones_like(stim_labels[:,5])
    idx5 = np.isin(stim_labels[:,6], random_types)     if random_types     else np.ones_like(stim_labels[:,6])
    idx6 = np.isin(stim_labels[:,7], opto_types)       if opto_types       else np.ones_like(stim_labels[:,7])
    idx = (idx1*idx2*idx3*idx4*idx5*idx6).astype('bool')
    return idx

# for multi session settings find trials based on stim_labels and trial avergae.
def get_multi_sess_neu_trial_average(
        list_stim_labels,
        neu_cate,
        alignment,
        trial_idx=None,
        trial_param=None,
        mean_sem=True,
        ):
    neu = []
    stim_seq = []
    stim_value = []
    pre_isi = []
    # use stim_labels to find trials.
    if trial_param != None and trial_idx == None:
        for i in range(len(neu_cate)):
            idx = pick_trial(
                list_stim_labels[i],
                trial_param[0],
                trial_param[1],
                trial_param[2],
                trial_param[3],
                trial_param[4],
                trial_param[5])
            neu.append(neu_cate[i][idx,:,:])
            stim_seq.append(alignment['list_stim_seq'][i][idx,:,:])
            stim_value.append(alignment['list_stim_value'][i][idx,:])
            pre_isi.append(alignment['list_pre_isi'][i][idx])
    # use given idx to find trials.
    if trial_param == None and trial_idx != None:
        for i in range(len(neu_cate)):
            neu.append(neu_cate[i][trial_idx[i],:,:])
            stim_seq.append(alignment['list_stim_seq'][i][trial_idx[i],:,:])
            stim_value.append(alignment['list_stim_value'][i][trial_idx[i],:])
            pre_isi.append(alignment['list_pre_isi'][i][trial_idx[i]])
    # use both.
    if trial_param != None and trial_idx != None:
        for i in range(len(neu_cate)):
            idx = pick_trial(
                list_stim_labels[i],
                trial_param[0],
                trial_param[1],
                trial_param[2],
                trial_param[3],
                trial_param[4])
            neu.append(neu_cate[i][trial_idx[i]*idx,:,:])
            stim_seq.append(alignment['list_stim_seq'][i][trial_idx[i]*idx,:,:])
            stim_value.append(alignment['list_stim_value'][i][trial_idx[i]*idx,:])
            pre_isi.append(alignment['list_pre_isi'][i][trial_idx[i]*idx])
    # compute trial average and concatenate.
    if mean_sem:
        mean = [np.nanmean(n, axis=0) for n in neu]
        sem  = [np.nanstd(n, axis=0)/np.sqrt(np.sum(~np.isnan(n), axis=0)) for n in neu]
        mean = np.concatenate(mean, axis=0)
        sem  = np.concatenate(sem, axis=0)
        stim_seq   = np.mean(np.concatenate(stim_seq, axis=0),axis=0)
        stim_value = np.mean(np.concatenate(stim_value, axis=0),axis=0)
        return mean, sem, stim_seq, stim_value, None
    # return single trial response.
    else:
        return neu, stim_seq, stim_value, pre_isi

# find neuron category and trial data.
def get_neu_trial(
        alignment, list_labels, list_significance, list_stim_labels,
        trial_idx=None, trial_param=None, mean_sem=True,
        cate=None, roi_id=None,
        ):
    if cate != None:
        colors = get_roi_label_color([cate], 0)
        neu_cate = [
            alignment['list_neu_seq'][i][:,(list_labels[i]==cate)*list_significance[i]['r_standard'],:]
            for i in range(len(list_stim_labels))]
    if roi_id != None:
        colors = get_roi_label_color(list_labels, roi_id)
        neu_cate = [np.expand_dims(alignment['list_neu_seq'][0][:,roi_id,:], axis=1)]
    neu_trial = get_multi_sess_neu_trial_average(
        list_stim_labels, neu_cate, alignment,
        trial_idx=trial_idx, trial_param=trial_param, mean_sem=mean_sem)
    return colors, neu_trial

# compute indice with givn time window for df/f.
def get_frame_idx_from_time(timestamps, c_time, l_time, r_time):
    l_idx = np.searchsorted(timestamps, c_time+l_time)
    r_idx = np.searchsorted(timestamps, c_time+r_time)
    return l_idx, r_idx

# get subsequence index with given start and end.
def get_sub_time_idx(time, start, end):
    idx = np.where((time >= start) &(time <= end))[0]
    return idx

# find index for each epoch.
def get_epoch_idx(stim_labels):
    num_early_trials = 50
    switch_idx = np.where(np.diff(stim_labels[:,3], prepend=1-stim_labels[:,3][0])!=0)[0]
    epoch_early = np.zeros_like(stim_labels[:,3], dtype='bool')
    for start in switch_idx:
        epoch_early[start:start+num_early_trials] = True
    epoch_late = ~epoch_early
    return epoch_early, epoch_late

# get index for short/long or pre/post.
def get_odd_stim_prepost_idx(stim_labels):
    idx_pre_short = (stim_labels[:,2]==-1) * (stim_labels[:,5]==0)
    idx_pre_long  = (stim_labels[:,2]==-1) * (stim_labels[:,5]==1)
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

# get isi based binning average neural response.
def get_isi_bin_neu(
        neu_seq, stim_seq, stim_value, pre_isi,
        bin_win, bin_num,
        mean_sem=True
        ):
    # define bins.
    bin_size = (bin_win[1] - bin_win[0]) / bin_num
    bins = np.arange(bin_win[0], bin_win[1] + bin_size, bin_size)
    bin_center = bins[:-1] + bin_size / 2
    list_bin_idx = [np.digitize(i, bins) - 1 for i in pre_isi]
    # compute binned results.
    bin_neu_seq = []
    bin_neu_mean = np.zeros((len(bin_center), neu_seq[0].shape[2]))
    bin_neu_sem = np.zeros((len(bin_center), neu_seq[0].shape[2]))
    bin_stim_seq = np.zeros((len(bin_center), stim_seq[0].shape[1], 2))
    bin_stim_value = np.zeros((len(bin_center), stim_value[0].shape[1]))
    for i in range(len(bin_center)):
        # get binned neural traces.
        neu = [n[bi==i,:,:] for n, bi in zip(neu_seq, list_bin_idx)]
        neu = [np.nanmean(n, axis=0) for n in neu]
        neu = np.concatenate(neu, axis=0)
        # get binned stimulus timing.
        s_seq = [s[bi==i,:,:] for s, bi in zip(stim_seq, list_bin_idx)]
        s_seq = np.concatenate(s_seq, axis=0)
        s_seq = np.nanmean(s_seq, axis=0)
        # get binned stimulus values.
        s_value = [s[bi==i,:] for s, bi in zip(stim_value, list_bin_idx)]
        s_value = np.concatenate(s_value, axis=0)
        s_value = np.nanmean(s_value, axis=0)
        # collect results.
        bin_neu_seq.append(neu)
        bin_neu_mean[i,:] = get_mean_sem(neu)[0]
        bin_neu_sem[i,:] = get_mean_sem(neu)[1]
        bin_stim_seq[i,:,:] = s_seq
        bin_stim_value[i,:] = s_value
    bin_center = bin_center.astype('int32')
    return bins, bin_center, bin_neu_seq, bin_neu_mean, bin_neu_sem, bin_stim_seq, bin_stim_value

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

# compute calcium transient event timing.
def get_ca_transient(dff):
    pct = 95
    win_peak = 25
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
    # main.
    ca_tran = np.zeros_like(dff)
    for i in range(dff.shape[0]):
        ca_tran_win = detect_spikes_win(dff[i,:])
        ca_tran_win = win_thres(ca_tran_win)
        ca_tran[i,:] = get_tran_time(ca_tran_win)
    return ca_tran

    
#%% plotting

# get ROI color from label.
def get_roi_label_color(labels, roi_id):
    if labels[roi_id] == -1:
        color0 = 'grey'
        color1 = 'deepskyblue'
        color2 = 'royalblue'
        cmap = LinearSegmentedColormap.from_list(
            'yicong_will_love_you_forever',
            ['white', 'royalblue', 'black'])
    if labels[roi_id] == 0:
        color0 = 'grey'
        color1 = 'mediumseagreen'
        color2 = 'forestgreen'
        cmap = LinearSegmentedColormap.from_list(
            'yicong_will_love_you_forever',
            ['white', 'forestgreen', 'black'])
    if labels[roi_id] == 1:
        color0 = 'grey'
        color1 = 'hotpink'
        color2 = 'crimson'
        cmap = LinearSegmentedColormap.from_list(
            'yicong_will_love_you_forever',
            ['white', 'crimson', 'black'])
    return color0, color1, color2, cmap

# return colors from dim to dark with a base color.
def get_cmap_color(n_colors, base_color=None, cmap=None):
    if base_color != None:
        cmap = LinearSegmentedColormap.from_list(
            None, ['#FCFCFC', base_color, '#2C2C2C'])
    if cmap != None:
        pass
    colors = cmap((np.arange(n_colors)+1)/n_colors)
    colors = [to_hex(color, keep_alpha=True) for color in colors]
    return colors

# normalize and apply colormap.
def apply_colormap(data, cmap):
    if data.shape[1] == 0:
        return np.zeros((0, data.shape[1], 3))
    for i in range(data.shape[0]):
        data[i,:] = norm01(data[i,:])
    data_heatmap = cmap(data)
    data_heatmap = data_heatmap[..., :3]
    return data_heatmap

# adjust layout for grand average neural traces.
def adjust_layout_neu(ax):
    ax.tick_params(axis='y', tick1On=False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_ylabel('df/f (z-scored)')

# adjust layout for heatmap.
def adjust_layout_heatmap(ax):
    ax.tick_params(tick1On=False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_yticks([])

# adjust layout for 3d latent dynamics.
def adjust_layout_3d_latent(ax, neu_z, cmap, neu_time, cbar_label):
    scale = 0.9
    ax.grid(False)
    ax.view_init(elev=30, azim=30)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlim([np.min(neu_z[0,:])*scale, np.max(neu_z[0,:])*scale])
    ax.set_ylim([np.min(neu_z[1,:])*scale, np.max(neu_z[1,:])*scale])
    ax.set_zlim([np.min(neu_z[2,:])*scale, np.max(neu_z[2,:])*scale])
    ax.set_xlabel('latent 1')
    ax.set_ylabel('latent 2')
    ax.set_zlabel('latent 3')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    cbar = ax.figure.colorbar(
        plt.cm.ScalarMappable(cmap=cmap), ax=ax, ticks=[0.2,0.5,0.8],
        shrink=0.5, aspect=25)
    cbar.outline.set_visible(False)
    cbar.ax.set_ylabel(cbar_label, rotation=-90, va='bottom')
    cbar.ax.set_yticklabels(
        [int(neu_time[int(len(neu_time)*0.2)]),
         int(neu_time[int(len(neu_time)*0.5)]),
         int(neu_time[int(len(neu_time)*0.8)])], rotation=-90)

# add legend into subplots.
def add_legend(ax, colors, labels, loc, dim=2):
    if dim == 2:
        handles = [
            ax.plot([], lw=0, color=colors[i], label=labels[i])[0]
            for i in range(len(labels))]
    if dim == 3:
        handles = [
            ax.plot([],[],[], lw=0, color=colors[i], label=labels[i])[0]
            for i in range(len(labels))]
    ax.legend(
        loc=loc,
        handles=handles,
        labelcolor='linecolor',
        frameon=False,
        framealpha=0)

# add number of trials and neurons into subplots.
def add_number(ax, n_neuron, n_trials, loc, dim=2):
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    if dim == 2:
        handles = [
            ax.plot([], lw=0, color='black', label=r'$n_{neuron}=$'+str(n_neuron))[0],
            ax.plot([], lw=0, color='black', label=r'$n_{trial}=$'+str(n_trials))[0]]
    if dim == 3:
        handles = [
            ax.plot([],[],[], lw=0, color='black', label=r'$n_{neuron}=$'+str(n_neuron))[0],
            ax.plot([],[],[], lw=0, color='black', label=r'$n_{trial}=$'+str(n_trials))[0]]
    ax.legend(
        loc=loc,
        handles=handles,
        labelcolor='linecolor',
        frameon=False,
        framealpha=0)

#%% basic plotting utils class

class utils_basic:

    def __init__(self):
        self.min_num_trial = 5

    def plot_mean_sem(self, ax, t, m, s, c, l, a=1.0):
        ax.plot(t, m, color=c, label=l, alpha=a)
        ax.fill_between(t, m - s, m + s, color=c, alpha=0.2)
        ax.set_xlim([np.min(t), np.max(t)])

    def plot_vol(self, ax, st, sv, c, u, l):
        v = np.mean(sv, axis=0)
        v = rescale(v, u, l)
        ax.plot(st, v, color=c, lw=0.5, linestyle=':')

    def plot_heatmap_neuron(
            self, ax, neu_seq, neu_time, neu_seq_sort,
            win_sort, labels, s, colorbar=False):
        win_conv = 5
        if len(neu_seq) > 0:
            _, _, _, cmap_exc = get_roi_label_color([-1], 0)
            _, _, _, cmap_inh = get_roi_label_color([1], 0)
            zero = np.searchsorted(neu_time, 0)
            # exclude nan.
            neu_idx = ~np.isnan(np.sum(neu_seq,axis=1))
            neu_seq = neu_seq[neu_idx,:].copy()
            neu_seq_sort = neu_seq_sort[neu_idx,:].copy()
            labels = labels[neu_idx].copy()
            s = s[neu_idx].copy()
            # smooth the values in the sorting window.
            l_idx, r_idx = get_frame_idx_from_time(neu_time, 0, win_sort[0], win_sort[1])
            smoothed_mean = np.array(
                [np.convolve(row, np.ones(win_conv)/win_conv, mode='same')
                 for row in neu_seq_sort[s,l_idx:r_idx]])
            sort_idx_neu = np.argmax(smoothed_mean, axis=1).reshape(-1).argsort()
            # rearrange the matrix.
            mean = neu_seq[s,:][sort_idx_neu,:].copy()
            # plot heatmap.
            heatmap_exc = apply_colormap(mean[labels[s]==-1,:], cmap_exc)
            heatmap_inh = apply_colormap(mean[labels[s]==1,:], cmap_inh)
            neu_h = np.concatenate([heatmap_exc, heatmap_inh], axis=0)
            ax.imshow(
                neu_h,
                extent=[neu_time[0], neu_time[-1], 1, neu_h.shape[0]],
                interpolation='nearest', aspect='auto')
            adjust_layout_heatmap(ax)
            ax.set_ylabel('neuron id (sorted)')
            ax.axvline(zero, color='black', lw=1, label='stim', linestyle='--')
            # add coloarbar.
            if colorbar:
                if heatmap_exc.shape[0] != 0:
                    cbar_exc = ax.figure.colorbar(
                        plt.cm.ScalarMappable(cmap=cmap_exc), ax=ax, ticks=[0.2,0.8], aspect=100)
                    cbar_exc.ax.set_ylabel('excitory', rotation=-90, va="bottom")
                    cbar_exc.ax.set_yticklabels(['0.2', '0.8'], rotation=-90)
                if heatmap_inh.shape[0] != 0:
                    cbar_inh = ax.figure.colorbar(
                        plt.cm.ScalarMappable(cmap=cmap_inh), ax=ax, ticks=[0.2,0.8], aspect=100)
                    cbar_inh.ax.set_ylabel('inhibitory', rotation=-90, va="bottom")
                    cbar_inh.ax.set_yticklabels(['0.2', '0.8'], rotation=-90)

    def plot_win_mag_box(self, ax, neu_seq, neu_time, win_base, color, c_time, offset):
        win_early = [0,250]
        win_late  = [250,500]
        baseline = get_base_mean_win(
            neu_seq.reshape(-1, neu_seq.shape[-1]), neu_time, c_time, win_base)
        [mean_late, sem_late] = get_mean_sem_win(
            neu_seq.reshape(-1, neu_seq.shape[-1]),
            neu_time, c_time, win_late[0], win_late[1])
        [mean_early, sem_early] = get_mean_sem_win(
            neu_seq.reshape(-1, neu_seq.shape[-1]),
            neu_time, c_time, win_early[0], win_early[1])
        mean_early -= baseline
        mean_late  -= baseline
        [mean_base, sem_base] = get_mean_sem_win(
            neu_seq.reshape(-1, neu_seq.shape[-1]),
            neu_time, c_time, win_base[0], win_base[1])
        ax.errorbar(
            0 + offset,
            mean_base, sem_base,
            color=color,
            capsize=2, marker='o', linestyle='none',
            markeredgecolor='white', markeredgewidth=0.1)
        ax.errorbar(
            1 + offset,
            mean_early, sem_early,
            color=color,
            capsize=2, marker='o', linestyle='none',
            markeredgecolor='white', markeredgewidth=0.1)
        ax.errorbar(
            2 + offset,
            mean_late, sem_late,
            color=color,
            capsize=2, marker='o', linestyle='none',
            markeredgecolor='white', markeredgewidth=0.1)
        ax.tick_params(tick1On=False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_ylabel(
            'response magnitude df/f (mean$\pm$sem) \n baseline [{},{}] ms'.format(
                win_base[0], win_base[1]))
        ax.set_xticks([0,1,2])
        ax.set_xticklabels([
            'baseline \n [{},{}] ms'.format(win_base[0], win_base[1]),
            'early \n [{},{}] ms'.format(win_early[0], win_early[1]),
            'late \n [{},{}] ms'.format(win_late[0], win_late[1])])
        ax.set_xlim([-0.5, 3.5])

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

    def plot_cluster_mean_sem(self, ax, neu_mean, neu_sem, norm_params, stim_seq, c, colors):
        for i in range(stim_seq.shape[0]):
            ax.fill_between(
                stim_seq[i,:],
                0-0.1*neu_mean.shape[0], neu_mean.shape[0]+0.1*neu_mean.shape[0],
                color=c, alpha=0.15, step='mid')
        for i in range(neu_mean.shape[0]):
            a, b = norm_params[i]
            self.plot_mean_sem(
                ax, self.alignment['neu_time'],
                (a*neu_mean[i,:]+b)+neu_mean.shape[0]-i-1, np.abs(a)*neu_sem[i,:],
                colors[i], None)
        ax.tick_params(axis='y', tick1On=False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_yticks([])
        ax.set_xlabel('time since stim (ms)')
        ax.set_ylabel('df/f (z-scored)')
        ax.set_ylim([-0.1, neu_mean.shape[0]+0.1])

    def plot_heatmap_trials(self, ax, neu_seq, neu_time, cmap, norm=True):
        if not np.isnan(np.sum(neu_seq)) and len(neu_seq)>0:
            if len(neu_seq.shape) == 3:
                mean = np.mean(neu_seq, axis=1)
            else:
                mean = neu_seq
            if norm:
                for i in range(mean.shape[0]):
                    mean[i,:] = norm01(mean[i,:])
            zero = np.searchsorted(neu_time, 0)
            img = ax.imshow(
                mean, interpolation='nearest', aspect='auto', cmap=cmap)
            adjust_layout_heatmap(ax)
            ax.set_ylabel('trial id')
            ax.axvline(zero, color='black', lw=1, linestyle='--')
            ax.set_xticks([0, zero, len(neu_time)])
            ax.set_xticklabels([int(neu_time[0]), 0, int(neu_time[-1])])
            ax.set_yticks([0, int(mean.shape[0]/3), int(mean.shape[0]*2/3), int(mean.shape[0])])
            ax.set_yticklabels([0, int(mean.shape[0]/3), int(mean.shape[0]*2/3), int(mean.shape[0])])
            cbar = ax.figure.colorbar(img, ax=ax, ticks=[0.2,0.8], aspect=100)
            cbar.ax.set_ylabel('normalized response', rotation=-90, va="bottom")
            cbar.ax.set_yticklabels(['0.2', '0.8'])
