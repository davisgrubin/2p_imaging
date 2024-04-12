#!/usr/bin/env python3

import numpy as np
from scipy.stats import mode
from scipy.stats import sem
from sklearn.mixture import GaussianMixture
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


# compute duration of sequence.

def frame_dur(stim, time):
    diff_stim = np.diff(stim, prepend=0)
    idx_up   = np.where(diff_stim == 1)[0]
    idx_down = np.where(diff_stim == -1)[0]
    dur_high = time[idx_down] - time[idx_up]
    dur_low  = time[idx_up[1:]] - time[idx_down[:-1]]
    return [idx_up, idx_down, dur_high, dur_low]


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

def get_omi_idx(stim, time):
    [grat_start, grat_end, _, isi] = frame_dur(stim, time)
    idx = np.where(isi >= 950)[0]
    omi_first_stim_after = grat_start[idx+1]
    #expected_grating = grat_end[idx] + int(500/np.median(np.diff(time, prepend=0))) + 1
    return omi_first_stim_after


# extract response around omissions.

def get_omi_response(
        vol_stim_bin, vol_time,
        neural_trials,
        trial_idx,
        ):
    # initialize list.
    neu_seq  = []
    neu_time = []
    stim_vol  = []
    stim_time = []
    # loop over trials.
    for trials in trial_idx:
        # read trial data.
        fluo = neural_trials[str(trials)]['dff']
        stim = neural_trials[str(trials)]['stim']
        time = neural_trials[str(trials)]['time']
        # compute stimulus start point.
        omi_first_stim_after = get_omi_idx(stim, time)
        for idx in omi_first_stim_after:
            if idx > l_frames and idx < len(time)-r_frames:
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
                stim_vol.append(vol_stim_bin[vidx])
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
    return [neu_seq, neu_time, stim_vol_fix, stim_vol_jitter, stim_time]


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
     fix_stim_time] = get_omi_response(
        vol_stim_bin, vol_time, neural_trials, fix_idx)
    # jitter data.
    [jitter_neu_seq,  jitter_neu_time,
     _, stim_vol_jitter,
     jitter_stim_time] = get_omi_response(
        vol_stim_bin, vol_time, neural_trials, jitter_idx)
    return [fix_neu_seq, fix_neu_time,
            stim_vol_fix, fix_stim_time,
            jitter_neu_seq, jitter_neu_time,
            stim_vol_jitter, jitter_stim_time]


# adjust layout for omission average.

def adjust_layout_mean(ax):
    ax.axvline(0, color='red', lw=2, label='omission', linestyle='--')
    ax.tick_params(axis='y', tick1On=False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel('time since first grating after omission (ms)')
    ax.set_ylabel('response')
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
     stim_vol_jitter, jitter_stim_time
     ] = get_fix_jitter_response(
         vol_stim_bin, vol_time, neural_trials, jitter_flag)
    fix_all  = fix_neu_seq[:, labels==-1, :].reshape(-1, l_frames+r_frames)
    fix_mean = np.mean(fix_all, axis=0)
    fix_sem  = sem(fix_all, axis=0)
    jitter_all  = jitter_neu_seq[:, labels==-1, :].reshape(-1, l_frames+r_frames)
    jitter_mean = np.mean(jitter_all, axis=0)
    jitter_sem  = sem(jitter_all, axis=0)
    upper = np.max(fix_mean) + np.max(fix_sem)
    lower = np.min(fix_mean) - np.max(fix_sem)
    ax.plot(
        fix_stim_time,
        rescale(stim_vol_fix, upper, lower),
        color='grey',
        label='fix stim')
    ax.plot(
        fix_stim_time,
        rescale(stim_vol_jitter, upper, lower),
        color='grey',
        linestyle=':',
        label='jitter stim')
    ax.plot(
        fix_neu_time,
        fix_mean,
        color='seagreen',
        label='excitory_fix')
    ax.plot(
        jitter_neu_time,
        jitter_mean,
        color='mediumspringgreen',
        label='excitory_jitter')
    ax.fill_between(
        fix_neu_time,
        fix_mean - fix_sem,
        fix_mean + fix_sem,
        color='seagreen',
        alpha=0.2)
    ax.fill_between(
        jitter_neu_time,
        jitter_mean - jitter_sem,
        jitter_mean + jitter_sem,
        color='mediumspringgreen',
        alpha=0.2)
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
     stim_vol_jitter, jitter_stim_time
     ] = get_fix_jitter_response(
         vol_stim_bin, vol_time, neural_trials, jitter_flag)
    fix_all  = fix_neu_seq[:, labels==1, :].reshape(-1, l_frames+r_frames)
    fix_mean = np.mean(fix_all, axis=0)
    fix_sem  = sem(fix_all, axis=0)
    jitter_all  = jitter_neu_seq[:, labels==1, :].reshape(-1, l_frames+r_frames)
    jitter_mean = np.mean(jitter_all, axis=0)
    jitter_sem  = sem(jitter_all, axis=0)
    upper = np.max(fix_mean) + np.max(fix_sem)
    lower = np.min(fix_mean) - np.max(fix_sem)
    ax.plot(
        fix_stim_time,
        rescale(stim_vol_fix, upper, lower),
        color='grey',
        label='fix stim')
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
     stim_vol_jitter, jitter_stim_time
     ] = get_fix_jitter_response(
         vol_stim_bin, vol_time, neural_trials, jitter_flag)
    mean = np.mean(fix_neu_seq, axis=0)
    for i in range(mean.shape[0]):
        mean[i,:] = norm01(mean[i,:])
    zero = np.where(fix_neu_time==0)[0][0]
    sort_idx_fix = mean[:, zero].reshape(-1).argsort()
    sort_fix = mean[sort_idx_fix,:]
    cmap = LinearSegmentedColormap.from_list(
        'fix', ['white','dodgerblue', 'black'])
    im_fix = ax.imshow(sort_fix, cmap=cmap)
    adjust_layout_heatmap(ax)
    ax.axvline(zero, color='red', lw=1, label='omission', linestyle='--')
    ax.set_xticks([0, zero, len(fix_neu_time)])
    ax.set_xticklabels([int(fix_neu_time[0]), 0, int(fix_neu_time[-1])])
    cbar = ax.figure.colorbar(im_fix, ax=ax, pad=0, ticks=[0.2,0.8])
    cbar.ax.set_ylabel('normalized response', rotation=-90, va="bottom")
    cbar.ax.set_yticklabels(['Low', 'High'])
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
     stim_vol_jitter, jitter_stim_time
     ] = get_fix_jitter_response(
         vol_stim_bin, vol_time, neural_trials, jitter_flag)
    mean = np.mean(jitter_neu_seq, axis=0)
    for i in range(mean.shape[0]):
        mean[i,:] = norm01(mean[i,:])
    zero = np.where(fix_neu_time==0)[0][0]
    sort_idx_fix = mean[:, zero].reshape(-1).argsort()
    sort_fix = mean[sort_idx_fix,:]
    cmap = LinearSegmentedColormap.from_list(
        'jitter', ['white','violet', 'black'])
    im_fix = ax.imshow(sort_fix, cmap=cmap)
    adjust_layout_heatmap(ax)
    ax.axvline(zero, color='red', lw=1, label='omission', linestyle='--')
    ax.set_xticks([0, zero, len(fix_neu_time)])
    ax.set_xticklabels([int(fix_neu_time[0]), 0, int(fix_neu_time[-1])])
    cbar = ax.figure.colorbar(im_fix, ax=ax, pad=0, ticks=[0.2,0.8])
    cbar.ax.set_ylabel('normalized response', rotation=-90, va="bottom")
    cbar.ax.set_yticklabels(['Low', 'High'])
    ax.set_title('omission response heatmap on jitter')


# plot response correlation heat map around omission for fix.


#%% crbl

