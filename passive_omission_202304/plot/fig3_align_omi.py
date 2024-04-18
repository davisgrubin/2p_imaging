#!/usr/bin/env python3

import numpy as np
from scipy.stats import mode
from scipy.stats import sem
from matplotlib.colors import LinearSegmentedColormap


#%% params


l_frames = 100
r_frames = 100


#%% utils


# normalization into [0,1].

def norm01(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-5)


# rescale voltage recordings.

def rescale(data, upper, lower):
    data = data.copy()
    data = ( data - np.min(data) ) / (np.max(data) - np.min(data))
    data = data * (upper - lower) + lower
    return data


# cut sequence into the same length as the shortest one given pivots.

def trim_seq(
        data,
        pivots,
        ):
    if len(data[0].shape) == 1:
        len_l_min = np.min(pivots)
        len_r_min = np.min([len(data[i])-pivots[i] for i in range(len(data))])
        data = [data[i][pivots[i]-len_l_min:pivots[i]+len_r_min]
                for i in range(len(data))]
    if len(data[0].shape) == 3:
        len_l_min = np.min(pivots)
        len_r_min = np.min([len(data[i][0,0,:])-pivots[i] for i in range(len(data))])
        data = [data[i][:, :, pivots[i]-len_l_min:pivots[i]+len_r_min]
                for i in range(len(data))]
    return data


# compute omission start point.

def get_omi_time(vol_stim, vol_time):
    stim_neg = vol_stim.copy() / np.max(vol_stim)
    stim_neg[stim_neg>0] = 0
    diff_stim_neg = np.diff(stim_neg, prepend=0)
    omi_time = vol_time[diff_stim_neg == -1]
    return omi_time

# compute expected grating time.

def get_expect_stim_time(stim_vol_fix, fix_stim_time):
    diff_stim = np.diff(stim_vol_fix, prepend=0)
    up = np.where(diff_stim==1)[0][1:]
    down = np.where(diff_stim==-1)[0][:-1]
    isi = np.median(fix_stim_time[up] - fix_stim_time[down])
    expect_stim_time = fix_stim_time[diff_stim==-1]
    expect_stim_time = expect_stim_time[expect_stim_time<0][-1] + isi
    return expect_stim_time


# compute isi before omission

def get_isi_pre_omi(vol_stim, vol_time, omi_time):
    diff_stim = np.diff(vol_stim.copy(), prepend=0)
    edge_time = vol_time[diff_stim!=0]
    isi_pre_omi = []
    for t in omi_time:
        omi_edge_idx = np.where(edge_time==t)[0]
        isi = edge_time[omi_edge_idx-2] - edge_time[omi_edge_idx-3]
        isi_pre_omi.append(isi)
    return isi_pre_omi


# bin the data with timestamps.

def get_bin_stat(data, time, time_range, bin_size):
    bins = np.arange(time_range[0], time_range[1] + bin_size, bin_size)
    bins = bins - bin_size / 2
    bin_indices = np.digitize(time, bins) - 1
    bin_mean = []
    bin_sem = []
    for i in range(len(bins)-1):
        bin_values = data[bin_indices == i]
        m = np.mean(bin_values) if len(bin_values) > 0 else np.nan
        s = sem(bin_values) if len(bin_values) > 0 else np.nan
        bin_mean.append(m)
        bin_sem.append(s)
    bin_mean = np.array(bin_mean)
    bin_sem = np.array(bin_sem)
    return bins, bin_mean, bin_sem


# extract response around omissions.

def get_omi_response(
        neural_trials,
        trial_idx,
        ):
    # initialize list.
    neu_seq  = []
    neu_time = []
    stim_vol  = []
    stim_time = []
    omi_isi = []
    # loop over trials.
    for trials in trial_idx:
        # read trial data.
        fluo = neural_trials[str(trials)]['dff']
        time = neural_trials[str(trials)]['time']
        vol_stim = neural_trials[str(trials)]['vol_stim']
        vol_time = neural_trials[str(trials)]['vol_time']
        # compute stimulus start point in ms.
        omi_time = get_omi_time(vol_stim, vol_time)
        isi_pre_omi = get_isi_pre_omi(vol_stim, vol_time, omi_time)
        for i in range(len(omi_time)):
            idx = np.argmin(np.abs(time - omi_time[i]))
            if idx > l_frames and idx < len(time)-r_frames:
                # isi before omission.
                omi_isi.append(isi_pre_omi[i])
                # signal response.
                f = fluo[:, idx-l_frames : idx+r_frames]
                f = np.expand_dims(f, axis=0)
                neu_seq.append(f)
                # signal time stamps.
                t = time[idx-l_frames : idx+r_frames] - time[idx]
                neu_time.append(t)
                # voltage recordings.
                vidx = np.where(
                    (vol_time > time[idx-l_frames]) &
                    (vol_time < time[idx+r_frames]))[0]
                stim_vol.append(np.abs(vol_stim[vidx]) / np.max(vol_stim))
                # voltage time stamps.
                stim_time.append(vol_time[vidx] - time[idx])
    # correct voltage recordings centering at perturbation.
    stim_time_zero = [np.argmin(np.abs(st)) for st in stim_time]
    stim_time = trim_seq(stim_time, stim_time_zero)
    stim_vol = trim_seq(stim_vol, stim_time_zero)
    # correct neuron time stamps centering at perturbation.
    neu_time_zero = [np.argmin(np.abs(nt)) for nt in neu_time]
    neu_time = trim_seq(neu_time, neu_time_zero)
    neu_seq = trim_seq(neu_seq, neu_time_zero)
    # concatenate results.
    omi_isi = np.array(omi_isi).reshape(-1)
    neu_time  = [nt.reshape(1,-1) for nt in neu_time]
    stim_time = [st.reshape(1,-1) for st in stim_time]
    stim_vol  = [sv.reshape(1,-1) for sv in stim_vol]
    neu_seq   = np.concatenate(neu_seq, axis=0)
    neu_time  = np.concatenate(neu_time, axis=0)
    stim_vol  = np.concatenate(stim_vol, axis=0)
    stim_time = np.concatenate(stim_time, axis=0)
    # get mean time stamps.
    neu_time  = np.mean(neu_time, axis=0)
    stim_time = np.mean(stim_time, axis=0)
    # get mode and mean stimulus.
    stim_vol_fix, _ = mode(stim_vol, axis=0)
    stim_vol_jitter = np.mean(stim_vol, axis=0)
    return [neu_seq, neu_time,
            stim_vol_fix, stim_vol_jitter, stim_time,
            omi_isi]


# run computation to separate fix and jitter response.

def get_fix_jitter_response(
        vol_stim_bin, vol_time,
        neural_trials, jitter_flag,
        ):
    # find trial id for jitter and fix.
    fix_idx = np.where(jitter_flag==0)[0]
    jitter_idx = np.where(jitter_flag==1)[0]
    # fix data.
    [fix_neu_seq,  fix_neu_time,
     stim_vol_fix, _,
     fix_stim_time,
     omi_isi_fix] = get_omi_response(
        neural_trials, fix_idx)
    # jitter data.
    [jitter_neu_seq,  jitter_neu_time,
     _, stim_vol_jitter,
     jitter_stim_time,
     omi_isi_jitter] = get_omi_response(
        neural_trials, jitter_idx)
    return [fix_neu_seq, fix_neu_time,
            stim_vol_fix, fix_stim_time,
            jitter_neu_seq, jitter_neu_time,
            stim_vol_jitter, jitter_stim_time,
            omi_isi_fix, omi_isi_jitter]


# adjust layout for omission average.

def adjust_layout_mean(ax):
    ax.axvline(0, color='red', lw=2, label='omission', linestyle='--')
    ax.tick_params(axis='y', tick1On=False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel('time since first grating after omission (ms)')
    ax.set_ylabel('df/f (z-scored)')
    ax.legend(loc='upper right')


# adjust layout for heatmap.

def adjust_layout_heatmap(ax):
    ax.tick_params(tick1On=False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_yticks([])
    ax.set_xlabel('time since first grating after omission (ms)')
    ax.set_ylabel('neuron id')


#%% ppc


# mean response of excitory.

def plot_ppc_exc_omi_mean(
        ax,
        vol_stim_bin, vol_time, neural_trials,
        jitter_flag, labels
        ):
    [fix_neu_seq, fix_neu_time,
     stim_vol_fix, fix_stim_time,
     jitter_neu_seq, jitter_neu_time,
     stim_vol_jitter, jitter_stim_time,
     _, _] = get_fix_jitter_response(
         vol_stim_bin, vol_time, neural_trials, jitter_flag)
    fix_all  = fix_neu_seq[:, labels==-1, :].reshape(-1, l_frames+r_frames)
    fix_mean = np.mean(fix_all, axis=0)
    fix_sem  = sem(fix_all, axis=0)
    jitter_all  = jitter_neu_seq[:, labels==-1, :].reshape(-1, l_frames+r_frames)
    jitter_mean = np.mean(jitter_all, axis=0)
    jitter_sem  = sem(jitter_all, axis=0)
    upper = np.max(fix_mean) + np.max(fix_sem)
    lower = np.min(fix_mean) - np.max(fix_sem)
    ax.fill_between(
        fix_stim_time,
        lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
        where=(stim_vol_fix==1),
        color='silver', step='mid', label='fix stim')
    ax.plot(
        fix_stim_time,
        rescale(stim_vol_jitter, upper, lower),
        color='grey',
        linestyle=':',
        label='jitter stim')
    ax.plot(
        fix_neu_time,
        fix_mean,
        color='mediumseagreen',
        label='excitory_fix')
    ax.plot(
        jitter_neu_time,
        jitter_mean,
        color='turquoise',
        label='excitory_jitter')
    ax.fill_between(
        fix_neu_time,
        fix_mean - fix_sem,
        fix_mean + fix_sem,
        color='mediumseagreen',
        alpha=0.2)
    ax.fill_between(
        jitter_neu_time,
        jitter_mean - jitter_sem,
        jitter_mean + jitter_sem,
        color='turquoise',
        alpha=0.2)
    ax.axvline(
        get_expect_stim_time(stim_vol_fix, fix_stim_time),
        color='black', lw=2, label='expectation', linestyle='--')
    adjust_layout_mean(ax)
    ax.set_xlim([np.min(fix_stim_time), np.max(fix_stim_time)])
    ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
    ax.set_title(
        'omission average trace of {} excitory neurons'.format(
        np.sum(labels==-1)))


# mean response of inhibitory.

def plot_ppc_inh_omi_mean(
        ax,
        vol_stim_bin, vol_time, neural_trials,
        jitter_flag, labels
        ):
    [fix_neu_seq, fix_neu_time,
     stim_vol_fix, fix_stim_time,
     jitter_neu_seq, jitter_neu_time,
     stim_vol_jitter, jitter_stim_time,
     _, _] = get_fix_jitter_response(
         vol_stim_bin, vol_time, neural_trials, jitter_flag)
    fix_all  = fix_neu_seq[:, labels==1, :].reshape(-1, l_frames+r_frames)
    fix_mean = np.mean(fix_all, axis=0)
    fix_sem  = sem(fix_all, axis=0)
    jitter_all  = jitter_neu_seq[:, labels==1, :].reshape(-1, l_frames+r_frames)
    jitter_mean = np.mean(jitter_all, axis=0)
    jitter_sem  = sem(jitter_all, axis=0)
    upper = np.max(fix_mean) + np.max(fix_sem)
    lower = np.min(fix_mean) - np.max(fix_sem)
    ax.fill_between(
        fix_stim_time,
        lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
        where=(stim_vol_fix==1),
        color='silver', step='mid', label='fix stim')
    ax.plot(
        fix_stim_time,
        rescale(stim_vol_jitter, upper, lower),
        color='grey',
        linestyle=':',
        label='jitter stim')
    ax.plot(
        fix_neu_time,
        fix_mean,
        color='brown',
        label='inhitory_fix')
    ax.plot(
        jitter_neu_time,
        jitter_mean,
        color='coral',
        label='inhitory_jitter')
    ax.fill_between(
        fix_neu_time,
        fix_mean - fix_sem,
        fix_mean + fix_sem,
        color='brown',
        alpha=0.2)
    ax.fill_between(
        jitter_neu_time,
        jitter_mean - jitter_sem,
        jitter_mean + jitter_sem,
        color='coral',
        alpha=0.2)
    ax.axvline(
        get_expect_stim_time(stim_vol_fix, fix_stim_time),
        color='black', lw=2, label='expectation', linestyle='--')
    adjust_layout_mean(ax)
    ax.set_xlim([np.min(fix_stim_time), np.max(fix_stim_time)])
    ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
    ax.set_title(
        'omission average trace of {} inhitory neurons'.format(
        np.sum(labels==1)))


# plot response heat map around omission for fix.

def plot_ppc_omi_fix_heatmap(
        ax,
        vol_stim_bin, vol_time, neural_trials,
        jitter_flag, labels
        ):
    [fix_neu_seq, fix_neu_time,
     stim_vol_fix, fix_stim_time,
     jitter_neu_seq, jitter_neu_time,
     stim_vol_jitter, jitter_stim_time,
     _, _] = get_fix_jitter_response(
         vol_stim_bin, vol_time, neural_trials, jitter_flag)
    mean = np.mean(fix_neu_seq, axis=0)
    for i in range(mean.shape[0]):
        mean[i,:] = norm01(mean[i,:])
    zero = np.where(fix_neu_time==0)[0][0]
    expect = get_expect_stim_time(stim_vol_fix, fix_stim_time)
    expect = np.argmin(np.abs(fix_neu_time-expect))
    sort_idx_fix = mean[:, zero].reshape(-1).argsort()
    sort_fix = mean[sort_idx_fix,:]
    cmap = LinearSegmentedColormap.from_list(
        'fix', ['white','dodgerblue', 'black'])
    im_fix = ax.imshow(sort_fix, cmap=cmap)
    adjust_layout_heatmap(ax)
    ax.axvline(zero, color='red', lw=1, label='omission', linestyle='--')
    ax.axvline(
        expect, color='black', lw=1,
        label='expect grating', linestyle='--')
    ax.set_xticks([0, zero, len(fix_neu_time)])
    ax.set_xticklabels([int(fix_neu_time[0]), 0, int(fix_neu_time[-1])])
    cbar = ax.figure.colorbar(im_fix, ax=ax, pad=-0.05, ticks=[0.2,0.8])
    cbar.ax.set_ylabel('normalized response', rotation=-90, va="bottom")
    cbar.ax.set_yticklabels(['0.2', '0.8'])
    ax.set_title('omission response heatmap on fix')


# plot response heat map around omission for jitter.

def plot_ppc_omi_jitter_heatmap(
        ax,
        vol_stim_bin, vol_time, neural_trials,
        jitter_flag, labels
        ):
    [fix_neu_seq, fix_neu_time,
     stim_vol_fix, fix_stim_time,
     jitter_neu_seq, jitter_neu_time,
     stim_vol_jitter, jitter_stim_time,
     _, _] = get_fix_jitter_response(
         vol_stim_bin, vol_time, neural_trials, jitter_flag)
    mean = np.mean(jitter_neu_seq, axis=0)
    for i in range(mean.shape[0]):
        mean[i,:] = norm01(mean[i,:])
    zero = np.where(fix_neu_time==0)[0][0]
    expect = get_expect_stim_time(stim_vol_fix, fix_stim_time)
    expect = np.argmin(np.abs(fix_neu_time-expect))
    sort_idx_fix = mean[:, zero].reshape(-1).argsort()
    sort_fix = mean[sort_idx_fix,:]
    cmap = LinearSegmentedColormap.from_list(
        'jitter', ['white','violet', 'black'])
    im_fix = ax.imshow(sort_fix, cmap=cmap)
    adjust_layout_heatmap(ax)
    ax.axvline(zero, color='red', lw=1, label='omission', linestyle='--')
    ax.axvline(
        expect, color='black', lw=1,
        label='expect grating', linestyle='--')
    ax.set_xticks([0, zero, len(fix_neu_time)])
    ax.set_xticklabels([int(fix_neu_time[0]), 0, int(fix_neu_time[-1])])
    cbar = ax.figure.colorbar(im_fix, ax=ax, pad=-0.05, ticks=[0.2,0.8])
    cbar.ax.set_ylabel('normalized response', rotation=-90, va="bottom")
    cbar.ax.set_yticklabels(['0.2', '0.8'])
    ax.set_title('omission response heatmap on jitter')


# omission response magnitude and isi before omission.

def plot_ppc_omi_isi(
        ax,
        vol_stim_bin, vol_time, neural_trials,
        jitter_flag, labels
        ):
    frames = 1
    time_range = [200,1000]
    bin_size = 100
    offset = 2
    [_, _, _, _, jitter_neu_seq, jitter_neu_time,
     _, _, _, omi_isi_jitter] = get_fix_jitter_response(
         vol_stim_bin, vol_time, neural_trials, jitter_flag)
    zero = np.where(jitter_neu_time==0)[0][0]
    mag_exc = jitter_neu_seq.copy()[:,labels==-1, zero:zero+frames]
    mag_exc = np.mean(mag_exc,axis=2).reshape(-1)
    mag_inh = jitter_neu_seq.copy()[:,labels==1, zero:zero+frames]
    mag_inh = np.mean(mag_inh,axis=2).reshape(-1)
    isi_exc = np.tile(omi_isi_jitter, np.sum(labels==-1))
    isi_inh = np.tile(omi_isi_jitter, np.sum(labels==1))
    bins, bin_mean_exc, bin_sem_exc = get_bin_stat(
        mag_exc, isi_exc, time_range, bin_size)
    bins, bin_mean_inh, bin_sem_inh = get_bin_stat(
        mag_inh, isi_inh, time_range, bin_size)
    ax.errorbar(
        bins[:-1] + (bins[1]-bins[0]) / 2 - offset,
        bin_mean_exc,
        bin_sem_exc,
        color='mediumaquamarine',
        capsize=2,
        marker='o',
        markeredgecolor='white',
        markeredgewidth=1,
        label='excitory')
    ax.errorbar(
        bins[:-1] + (bins[1]-bins[0]) / 2 + offset,
        bin_mean_inh,
        bin_sem_inh,
        color='lightcoral',
        capsize=2,
        marker='o',
        markeredgecolor='white',
        markeredgewidth=1,
        label='inhibitory')
    ax.tick_params(tick1On=False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_title('omission response and isi beforee omission')
    ax.set_xlabel('isi before omission (ms)')
    ax.set_ylabel('df/f (mean$\pm$sem)')
    ax.yaxis.grid(True)
    ax.legend(loc='upper left')


#%% crbl

