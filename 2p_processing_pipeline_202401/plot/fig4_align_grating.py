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


# z score normalization.

def normz(data):
    return (data - np.mean(data)) / (np.std(data) + 1e-5)


# get saved results.

def read_data(
        ops
        ):
    [labels, _, _, _, _, _] = read_masks(ops)
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
        l_frames,
        r_frames,
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


# main function for plot

def plot_fig4(
        ops,
        l_frames = 30,
        r_frames = 30,
        ):

    try:
        print('plotting fig4 grating aligned traces')

        [labels,
         vol_img_bin, vol_stim_bin, vol_time,
         dff, neural_trials, jitter_flag] = read_data(ops)

        # find trial id for jitter and fix.
        fix_idx = np.where(jitter_flag==0)[0]
        jitter_idx = np.where(jitter_flag==1)[0]

        # fix data
        [fix_neu_seq,  fix_neu_time,
         stim_vol_fix, _,
         fix_stim_time] = get_stim_response(
            vol_stim_bin, vol_time,
            neural_trials, fix_idx,
            l_frames, r_frames)
        # jitter data
        [jitter_neu_seq,  jitter_neu_time,
         _, stim_vol_jitter,
         jitter_stim_time] = get_stim_response(
            vol_stim_bin, vol_time,
            neural_trials, jitter_idx,
            l_frames, r_frames)

        # plot signals.
        color_fix = ['dodgerblue', 'coral']
        color_jitter = ['violet', 'brown']
        fluo_label = ['excitory', 'inhibitory']
        num_subplots = fix_neu_seq.shape[1] + 2
        fig, axs = plt.subplots(num_subplots, 1, figsize=(20, 10))
        plt.subplots_adjust(hspace=0.8)

        # mean response of excitory.
        axs[0].plot(
            fix_stim_time,
            (stim_vol_fix - 0.5),
            color='grey',
            label='fix stim')
        axs[0].plot(
            fix_stim_time,
            (stim_vol_jitter - 0.5),
            color='grey',
            linestyle=':',
            label='jitter stim')
        axs[0].plot(
            fix_neu_time,
            np.mean(np.mean(normz(
                fix_neu_seq[:, labels==0, :]), axis=0), axis=0),
            color=color_fix[0],
            label='excitory_fix')
        axs[0].plot(
            jitter_neu_time,
            np.mean(np.mean(normz(
                jitter_neu_seq[:, labels==0, :]), axis=0), axis=0),
            color=color_jitter[0],
            label='excitory_jitter')
        axs[0].set_title(
            'grating average trace of {} excitory neurons'.format(
            np.sum(labels==0)))

        # mean response of inhibitory.
        axs[1].plot(
            fix_stim_time,
            (stim_vol_fix - 0.5),
            color='grey',
            label='fix stim')
        axs[1].plot(
            fix_stim_time,
            (stim_vol_jitter - 0.5),
            color='grey',
            linestyle=':',
            label='jitter stim')
        axs[1].plot(
            fix_neu_time,
            np.mean(np.mean(normz(
                fix_neu_seq[:, labels==1, :]), axis=0), axis=0),
            color=color_fix[1],
            label='inhibitory_fix')
        axs[1].plot(
            jitter_neu_time,
            np.mean(np.mean(normz(
                jitter_neu_seq[:, labels==1, :]), axis=0), axis=0),
            color=color_jitter[1],
            label='inhibitory_jitter')
        axs[1].set_title(
            'grating average trace of {} inhibitory neurons'.format(
            np.sum(labels==1)))

        # individual neuron response.
        for i in range(fix_neu_seq.shape[1]):
            fix_mean = np.mean(normz(fix_neu_seq[:,i,:]), axis=0)
            fix_sem = sem(normz(fix_neu_seq[:,i,:]), axis=0)
            jitter_mean = np.mean(normz(jitter_neu_seq[:,i,:]), axis=0)
            jitter_sem = sem(normz(jitter_neu_seq[:,i,:]), axis=0)
            axs[i+2].plot(
                fix_stim_time,
                (stim_vol_fix - 0.5) * 2,
                color='grey',
                label='fix stim')
            axs[i+2].plot(
                fix_stim_time,
                (stim_vol_jitter - 0.5) * 2,
                color='grey',
                linestyle=':',
                label='jitter stim')
            axs[i+2].plot(
                fix_neu_time,
                fix_mean,
                color=color_fix[labels[i]],
                label=fluo_label[labels[i]]+'_fix')
            axs[i+2].fill_between(
                fix_neu_time,
                fix_mean - fix_sem,
                fix_mean + fix_sem,
                color=color_fix[labels[i]],
                alpha=0.2)
            axs[i+2].plot(
                jitter_neu_time,
                jitter_mean,
                color=color_jitter[labels[i]],
                label=fluo_label[labels[i]]+'_jitter')
            axs[i+2].fill_between(
                jitter_neu_time,
                jitter_mean - jitter_sem,
                jitter_mean + jitter_sem,
                color=color_jitter[labels[i]],
                alpha=0.2)
            axs[i+2].set_title(
                'grating average trace of neuron # '+ str(i).zfill(3))

        # adjust layout.
        for i in range(num_subplots):
            axs[i].tick_params(axis='y', tick1On=False)
            axs[i].spines['right'].set_visible(False)
            axs[i].spines['top'].set_visible(False)
            axs[i].set_xlabel('time since center grating start / ms')
            axs[i].set_ylabel('response')
            axs[i].set_xlim([np.min(fix_stim_time), np.max(fix_stim_time)])
            axs[i].set_ylim([-1,1])
            axs[i].legend(loc='upper left')
        axs[0].set_ylim([-0.5,0.5])
        axs[1].set_ylim([-0.5,0.5])
        fig.set_size_inches(8, num_subplots*4)
        fig.tight_layout()

        # save figure
        fig.savefig(os.path.join(
            ops['save_path0'], 'figures', 'fig4_align_grating.pdf'),
            dpi=300)
        plt.close()

    except:
        print('plotting fig4 failed')
