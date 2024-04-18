#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode
from scipy.stats import sem

from postprocess.ReadResults import read_masks
from postprocess.ReadResults import read_raw_voltages
from postprocess.ReadResults import read_dff
from postprocess.ReadResults import read_neural_trials


#%% params


l_frames = 50
r_frames = 50


#%% utils


# rescale voltage recordings.

def rescale(data, upper, lower):
    data = data.copy()
    data = ( data - np.min(data) ) / (np.max(data) - np.min(data))
    data = data * (upper - lower) + lower
    return data


# get saved results.

def read_data(
        ops
        ):
    [labels, _, _, _, _] = read_masks(ops)
    [vol_time, _, vol_stim_bin, vol_img_bin] = read_raw_voltages(ops)
    dff = read_dff(ops)
    neural_trials = read_neural_trials(ops)
    jitter_flag = np.load(
        os.path.join(ops['save_path0'], 'jitter_flag.npy'),
        allow_pickle=True)
    return [labels,
            vol_img_bin, vol_stim_bin, vol_time,
            dff, neural_trials, jitter_flag]


# extract response around stimulus.

def get_stim_response(
        vol_stim_bin,
        vol_time,
        neural_trials,
        trial_idx,
        ):
    neu_seq  = []
    neu_time = []
    stim_vol  = []
    stim_time = []
    for trials in trial_idx:
        # read trial data.
        fluo = neural_trials[str(trials)]['dff']
        stim = neural_trials[str(trials)]['stim']
        time = neural_trials[str(trials)]['time']
        stim[stim!=0] = 1
        # compute stimulus start point.
        grat_start = get_grating_start(stim)
        for idx in grat_start:
            if idx > l_frames and idx < len(time)-r_frames:
                # reshape for concatenate into matrix.
                stim = stim.reshape(1,-1)
                time = time.reshape(1,-1)
                # signal response.
                f = fluo[:, idx-l_frames : idx+r_frames]
                f = np.expand_dims(f, axis=0)
                neu_seq.append(f)
                # signal time stamps.
                t = time[:, idx-l_frames : idx+r_frames] - time[:, idx]
                neu_time.append(t)
                # voltage recordings.
                vidx = np.where(
                    (vol_time > time[:,idx-l_frames]) &
                    (vol_time < time[:,idx+r_frames]))[0]
                sv = vol_stim_bin[vidx].reshape(1,-1)
                stim_vol.append(sv)
                # voltage time stamps.
                st = vol_time[vidx].reshape(1,-1) - time[:, idx]
                stim_time.append(st)
    # correct voltage recordings to the same length.
    min_len = np.min([sv.shape[1] for sv in stim_vol])
    stim_vol = [sv[:,:min_len] for sv in stim_vol]
    stim_time = [st[:,:min_len] for st in stim_time]
    # concatenate results.
    neu_seq   = np.concatenate(neu_seq, axis=0)
    neu_time  = np.concatenate(neu_time, axis=0)
    stim_vol  = np.concatenate(stim_vol, axis=0)
    stim_time = np.concatenate(stim_time, axis=0)
    # get mean time stamps.
    neu_time  = np.mean(neu_time, axis=0)
    stim_time = np.mean(stim_time, axis=0)
    # get mode stimulus.
    stim_vol_fix, _ = mode(stim_vol, axis=0)
    stim_vol_jitter = np.mean(stim_vol, axis=0)
    # scale stimulus sequence.
    return [neu_seq, neu_time, stim_vol_fix, stim_vol_jitter, stim_time]


# find grating start.

def get_grating_start(stim):
    diff_stim = np.diff(stim, prepend=0)
    grat_start = np.where(diff_stim == 1)[0]
    return grat_start


# run computation to separate fix and jitter response.

def get_fix_jitter_response(
        vol_stim_bin, vol_time,
        neural_trials, jitter_flag,
        ):
    # find trial id for jitter and fix.
    fix_idx = np.where(jitter_flag==0)[0]
    jitter_idx = np.where(jitter_flag==1)[0]
    # fix data
    [fix_neu_seq, fix_neu_time,
     stim_vol_fix, _,
     fix_stim_time] = get_stim_response(
        vol_stim_bin, vol_time,
        neural_trials, fix_idx)
    # jitter data
    [jitter_neu_seq, jitter_neu_time,
     _, stim_vol_jitter,
     jitter_stim_time] = get_stim_response(
        vol_stim_bin, vol_time,
        neural_trials, jitter_idx)
    return [fix_neu_seq, fix_neu_time,
            stim_vol_fix, fix_stim_time,
            jitter_neu_seq, jitter_neu_time,
            stim_vol_jitter, jitter_stim_time]


# adjust layout for grating average.

def adjust_layout(ax):
    ax.tick_params(axis='y', tick1On=False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel('time since center grating start (ms)')
    ax.set_ylabel('df/f (z-scored)')
    ax.legend(loc='upper right')


#%% ppc


# mean response of excitory.

def plot_ppc_exc_mean(
        ax,
        vol_stim_bin, vol_time,
        neural_trials, jitter_flag, labels,
        ):
    [fix_neu_seq, fix_neu_time,
     stim_vol_fix, fix_stim_time,
     jitter_neu_seq, jitter_neu_time,
     stim_vol_jitter, jitter_stim_time] = get_fix_jitter_response(
         vol_stim_bin, vol_time,
         neural_trials, jitter_flag)
    fix_all = fix_neu_seq[:, labels==-1, :].reshape(-1, l_frames+r_frames)
    fix_mean = np.mean(fix_all, axis=0)
    fix_sem = sem(fix_all, axis=0)
    jitter_all = jitter_neu_seq[:, labels==-1, :].reshape(-1, l_frames+r_frames)
    jitter_mean = np.mean(jitter_all, axis=0)
    jitter_sem = sem(jitter_all, axis=0)
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
        color='grey', linestyle=':', label='jitter stim')
    ax.plot(
        fix_neu_time,
        fix_mean,
        color='mediumseagreen', label='excitory_fix')
    ax.plot(
        jitter_neu_time,
        jitter_mean,
        color='turquoise', label='excitory_jitter')
    ax.fill_between(
        fix_neu_time,
        fix_mean - fix_sem,
        fix_mean + fix_sem,
        color='mediumseagreen', alpha=0.2)
    ax.fill_between(
        jitter_neu_time,
        jitter_mean - jitter_sem,
        jitter_mean + jitter_sem,
        color='turquoise', alpha=0.2)
    adjust_layout(ax)
    ax.set_xlim([np.min(fix_stim_time), np.max(fix_stim_time)])
    ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
    ax.set_title(
        'grating average trace of {} excitory neurons'.format(
        np.sum(labels==-1)))


# mean response of excitory.

def plot_ppc_inh_mean(
        ax,
        vol_stim_bin, vol_time,
        neural_trials, jitter_flag, labels,
        ):
    [fix_neu_seq, fix_neu_time,
     stim_vol_fix, fix_stim_time,
     jitter_neu_seq, jitter_neu_time,
     stim_vol_jitter, jitter_stim_time] = get_fix_jitter_response(
         vol_stim_bin, vol_time,
         neural_trials, jitter_flag)
    fix_all = fix_neu_seq[:, labels==1, :].reshape(-1, l_frames+r_frames)
    fix_mean = np.mean(fix_all, axis=0)
    fix_sem = sem(fix_all, axis=0)
    jitter_all = jitter_neu_seq[:, labels==1, :].reshape(-1, l_frames+r_frames)
    jitter_mean = np.mean(jitter_all, axis=0)
    jitter_sem = sem(jitter_all, axis=0)
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
        color='grey', linestyle=':', label='jitter stim')
    ax.plot(
        fix_neu_time,
        fix_mean,
        color='brown', label='inhibitory_fix')
    ax.plot(
        jitter_neu_time,
        jitter_mean,
        color='coral', label='inhibitory_jitter')
    ax.fill_between(
        fix_neu_time,
        fix_mean - fix_sem,
        fix_mean + fix_sem,
        color='brown', alpha=0.2)
    ax.fill_between(
        jitter_neu_time,
        jitter_mean - jitter_sem,
        jitter_mean + jitter_sem,
        color='coral', alpha=0.2)
    adjust_layout(ax)
    ax.set_xlim([np.min(fix_stim_time), np.max(fix_stim_time)])
    ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
    ax.set_title(
        'grating average trace of {} inhibitory neurons'.format(
        np.sum(labels==1)))



#%% crbl
