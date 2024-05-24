#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode
from scipy.stats import sem
from matplotlib.colors import LinearSegmentedColormap

from modules.ReadResults import read_masks
from modules.ReadResults import read_raw_voltages
from modules.ReadResults import read_dff
from modules.ReadResults import read_neural_trials


#%% params


l_frames = 50
r_frames = 50
# fig, ax = plt.subplots(1, 1, figsize=(6, 6))


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


# find grating orientation response.

def get_stim_type_response(
        vol_stim_bin,
        vol_time,
        neural_trials,
        trial_idx,
        keep_orien=False,
        ):
    neu_seq  = [[],[],[],[]]
    neu_time = [[],[],[],[]]
    stim_vol  = [[],[],[],[]]
    stim_time = [[],[],[],[]]
    for trials in trial_idx:
        # read trial data.
        fluo = neural_trials[str(trials)]['dff']
        stim = neural_trials[str(trials)]['stim']
        time = neural_trials[str(trials)]['time']
        if not keep_orien:
            stim[stim!=0] = 1
        else:
            stim = np.abs(stim)
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
                # signal time stamps.
                t = time[:, idx-l_frames : idx+r_frames] - time[:, idx]
                # voltage recordings.
                vidx = np.where(
                    (vol_time > time[:,idx-l_frames]) &
                    (vol_time < time[:,idx+r_frames]))[0]
                sv = vol_stim_bin[vidx].reshape(1,-1)
                # voltage time stamps.
                st = vol_time[vidx].reshape(1,-1) - time[:, idx]
                # collect data for different timulus types.
                neu_seq[int(np.unique(stim)[1]-1)].append(f)
                neu_time[int(np.unique(stim)[1]-1)].append(t)
                stim_vol[int(np.unique(stim)[1]-1)].append(sv)
                stim_time[int(np.unique(stim)[1]-1)].append(st)
    for i in range(len(neu_seq)):
        # correct voltage recordings to the same length.
        if len(stim_vol[i]) > 0:
            min_len = np.min([sv.shape[1] for sv in stim_vol[i]])
            stim_vol[i] = [sv[:,:min_len] for sv in stim_vol[i] if len(sv)>0]
            stim_time[i] = [st[:,:min_len] for st in stim_time[i] if len(st)>0]
            # concatenate results.
            neu_seq[i]   = np.concatenate(neu_seq[i], axis=0)
            neu_time[i]  = np.concatenate(neu_time[i], axis=0)
            stim_vol[i]  = np.concatenate(stim_vol[i], axis=0)
            stim_time[i] = np.concatenate(stim_time[i], axis=0)
            # get mean time stamps.
            neu_time[i]  = np.mean(neu_time[i], axis=0)
            stim_time[i] = np.mean(stim_time[i], axis=0)
    stim_vol_fix = [mode(sv, axis=0)[0] for sv in stim_vol]
    stim_vol_jitter = [np.mean(stim_vol[0], axis=0) for sv in stim_vol]
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
        keep_orien=False
        ):
    # find trial id for jitter and fix.
    fix_idx = np.where(jitter_flag==0)[0]
    jitter_idx = np.where(jitter_flag==1)[0]
    # fix data
    [fix_neu_seq, fix_neu_time,
     stim_vol_fix, _,
     fix_stim_time] = get_stim_type_response(
        vol_stim_bin, vol_time,
        neural_trials, fix_idx,
        keep_orien)
    # jitter data
    [jitter_neu_seq, jitter_neu_time,
     _, stim_vol_jitter,
     jitter_stim_time] = get_stim_type_response(
        vol_stim_bin, vol_time,
        neural_trials, jitter_idx,
        keep_orien)
    return [fix_neu_seq, fix_neu_time,
            stim_vol_fix, fix_stim_time,
            jitter_neu_seq, jitter_neu_time,
            stim_vol_jitter, jitter_stim_time]


# get ROI color from label.

def get_roi_label_color(labels, roi_id):
    if labels[roi_id] == -1:
        cate = 'excitory'
        color1 = 'mediumseagreen'
        color2 = 'turquoise'
    if labels[roi_id] == 0:
        cate = 'unsure'
        color1 = 'violet'
        color2 = 'dodgerblue'
    if labels[roi_id] == 1:
        cate = 'inhibitory'
        color1 = 'brown'
        color2 = 'coral'
    return cate, color1, color2  


# adjust layout for grating average.

def adjust_layout(ax):
    ax.tick_params(axis='y', tick1On=False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel('time since center grating start (ms)')
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

def plot_ppc_exc_grating_mean(
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

def plot_ppc_inh_grating_mean(
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
    
    
# ROI mean response to different grating orientations.

def plot_ppc_roi_grating_orien_mean(
        stim_axs, roi_id,
        vol_stim_bin, vol_time,
        neural_trials, jitter_flag, labels,
        ):
    [fix_neu_seq, fix_neu_time,
     stim_vol_fix, fix_stim_time,
     jitter_neu_seq, jitter_neu_time,
     stim_vol_jitter, jitter_stim_time] = get_fix_jitter_response(
         vol_stim_bin, vol_time,
         neural_trials, jitter_flag,
         keep_orien=True)
    fig, stim_axs = plt.subplots(1, 4, figsize=(8, 2))
    for i in range(len(stim_axs)):
        if len(fix_neu_seq[i])>0:
            fix_all = fix_neu_seq[i][:, roi_id, :].reshape(-1, l_frames+r_frames)
            fix_mean = np.mean(fix_all, axis=0)
            fix_sem = sem(fix_all, axis=0)
            jitter_all = jitter_neu_seq[i][:, roi_id, :].reshape(-1, l_frames+r_frames)
            jitter_mean = np.mean(jitter_all, axis=0)
            jitter_sem = sem(jitter_all, axis=0)
            upper = np.max([fix_mean, jitter_mean]) + np.max([fix_sem, jitter_sem])
            lower = np.min([fix_mean, jitter_mean]) - np.max([fix_sem, jitter_sem])
            _, color1, color2 = get_roi_label_color(labels, roi_id)
            stim_axs[i].fill_between(
                fix_stim_time[i],
                lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
                where=(stim_vol_fix[i]!=0),
                color='silver', step='mid', label='fix stim')
            stim_axs[i].plot(
                fix_stim_time[i],
                rescale(stim_vol_jitter[i], upper, lower),
                color='grey', linestyle=':', label='jitter stim')
            stim_axs[i].plot(
                fix_neu_time[i],
                fix_mean,
                color=color1, label='dff on fix')
            stim_axs[i].plot(
                jitter_neu_time[i],
                jitter_mean,
                color=color2, label='dff on jitter')
            stim_axs[i].fill_between(
                fix_neu_time[i],
                fix_mean - fix_sem,
                fix_mean + fix_sem,
                color=color1, alpha=0.2)
            stim_axs[i].fill_between(
                jitter_neu_time[i],
                jitter_mean - jitter_sem,
                jitter_mean + jitter_sem,
                color=color2, alpha=0.2)
            stim_axs[i].set_xlim([np.min(fix_stim_time[i]), np.max(fix_stim_time[i])])
            stim_axs[i].set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
        adjust_layout(stim_axs[i])
        stim_axs[i].set_title('ROI # {} mean response to grating type # {}'.format(
            str(roi_id).zfill(4), int(i+1)))
    

    
    
#%% crbl

# mean response.

def plot_crbl_grating_mean(
        ax,
        vol_stim_bin, vol_time,
        neural_trials, jitter_flag,
        ):
    [fix_neu_seq, fix_neu_time,
     stim_vol_fix, fix_stim_time,
     jitter_neu_seq, jitter_neu_time,
     stim_vol_jitter, jitter_stim_time] = get_fix_jitter_response(
         vol_stim_bin, vol_time,
         neural_trials, jitter_flag)
    fix_all = fix_neu_seq.reshape(-1, l_frames+r_frames)
    fix_mean = np.mean(fix_all, axis=0)
    fix_sem = sem(fix_all, axis=0)
    jitter_all = jitter_neu_seq.reshape(-1, l_frames+r_frames)
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
        color='mediumseagreen', label='fix')
    ax.plot(
        jitter_neu_time,
        jitter_mean,
        color='turquoise', label='jitter')
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
    ax.set_title('grating average trace of {} ROIs'.format(fix_neu_seq.shape[1]))


# plot response heat map around grating for fix.

def plot_crbl_grating_fix_heatmap(
        ax,
        vol_stim_bin, vol_time, neural_trials,
        jitter_flag
        ):
    [fix_neu_seq, fix_neu_time,
     stim_vol_fix, fix_stim_time,
     jitter_neu_seq, jitter_neu_time,
     stim_vol_jitter, jitter_stim_time] = get_fix_jitter_response(
         vol_stim_bin, vol_time, neural_trials, jitter_flag)
    mean = np.mean(fix_neu_seq, axis=0)
    for i in range(mean.shape[0]):
        mean[i,:] = norm01(mean[i,:])
    zero = np.where(fix_neu_time==0)[0][0]
    sort_idx_fix = mean[:, zero].reshape(-1).argsort()
    sort_fix = mean[sort_idx_fix,:]
    cmap = LinearSegmentedColormap.from_list(
        'fix', ['white','dodgerblue', 'black'])
    im_fix = ax.imshow(
        sort_fix, interpolation='nearest', aspect='auto', cmap=cmap)
    adjust_layout_heatmap(ax)
    ax.axvline(zero, color='red', lw=1, label='omission', linestyle='--')
    ax.set_xticks([0, zero, len(fix_neu_time)])
    ax.set_xticklabels([int(fix_neu_time[0]), 0, int(fix_neu_time[-1])])
    cbar = ax.figure.colorbar(im_fix, ax=ax, ticks=[0.2,0.8])
    cbar.ax.set_ylabel('normalized response', rotation=-90, va="bottom")
    cbar.ax.set_yticklabels(['0.2', '0.8'])
    ax.set_title('grating response heatmap on fix')


# plot response heat map around grating for jitter.

def plot_crbl_grating_jitter_heatmap(
        ax,
        vol_stim_bin, vol_time, neural_trials,
        jitter_flag
        ):
    [fix_neu_seq, fix_neu_time,
     stim_vol_fix, fix_stim_time,
     jitter_neu_seq, jitter_neu_time,
     stim_vol_jitter, jitter_stim_time] = get_fix_jitter_response(
         vol_stim_bin, vol_time, neural_trials, jitter_flag)
    mean = np.mean(jitter_neu_seq, axis=0)
    for i in range(mean.shape[0]):
        mean[i,:] = norm01(mean[i,:])
    zero = np.where(fix_neu_time==0)[0][0]
    sort_idx_fix = mean[:, zero].reshape(-1).argsort()
    sort_fix = mean[sort_idx_fix,:]
    cmap = LinearSegmentedColormap.from_list(
        'jitter', ['white','violet', 'black'])
    im_fix = ax.imshow(
        sort_fix, interpolation='nearest', aspect='auto', cmap=cmap)
    adjust_layout_heatmap(ax)
    ax.axvline(zero, color='red', lw=1, label='omission', linestyle='--')
    ax.set_xticks([0, zero, len(fix_neu_time)])
    ax.set_xticklabels([int(fix_neu_time[0]), 0, int(fix_neu_time[-1])])
    cbar = ax.figure.colorbar(im_fix, ax=ax, ticks=[0.2,0.8])
    cbar.ax.set_ylabel('normalized response', rotation=-90, va="bottom")
    cbar.ax.set_yticklabels(['0.2', '0.8'])
    ax.set_title('grating response heatmap on jitter')
