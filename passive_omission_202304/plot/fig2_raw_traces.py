#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from postprocess.ReadResults import read_masks
from postprocess.ReadResults import read_raw_voltages
from postprocess.ReadResults import read_dff


#%% utils


# rescale voltage recordings.

def rescale(data, upper, lower):
    data = data.copy()
    data = ( data - np.min(data) ) / (np.max(data) - np.min(data))
    data = data * (upper - lower) + lower
    return data


# read and process traces

def read_data(
        ops
        ):
    [vol_time, _, vol_stim_bin, vol_img_bin] = read_raw_voltages(ops)
    [labels, _, _, _, _] = read_masks(ops)
    dff = read_dff(ops)
    time_img = get_img_time(vol_time, vol_img_bin)
    return [labels, dff, time_img, vol_stim_bin, vol_time]


# find imaging trigger time stamps

def get_img_time(
        vol_time,
        vol_img_bin
        ):
    diff_vol = np.diff(vol_img_bin, append=0)
    idx_up = np.where(diff_vol == 1)[0]+1
    img_time = vol_time[idx_up]
    return img_time


# get subsequence index with given start and end.

def get_sub_time_idx(
        time,
        start,
        end
        ):
    idx = np.where((time >= start) &(time <= end))[0]
    return idx


# main function for plot

def plot_fig2(
        ops,
        max_ms = 120000,
        ):
    if not os.path.exists(os.path.join(
            ops['save_path0'], 'figures', 'fig2_raw_traces')):
        os.makedirs(os.path.join(
            ops['save_path0'], 'figures', 'fig2_raw_traces'))
    try:
        print('plotting fig2 raw traces')

        [labels, dff, time_img, vol_stim_bin, vol_time] = read_data(ops)
        mean_fluo_0 = np.mean(dff[labels==-1, :], axis=0)
        mean_fluo_1 = np.mean(dff[labels==1, :], axis=0)

        # plot figs.
        if np.max(vol_time) < max_ms:
            num_figs = 1
        else:
            num_figs = int(np.max(vol_time)/max_ms)
        num_subplots = dff.shape[0] + 2
        for f in tqdm(range(num_figs)):

            # find sequence start and end timestamps.
            start = f * max_ms
            end   = (f+1) * max_ms

            # get subplot range.
            sub_vol_time_idx = get_sub_time_idx(vol_time, start, end)
            sub_time_img_idx = get_sub_time_idx(time_img, start, end)
            sub_vol_time     = vol_time[sub_vol_time_idx]
            sub_time_img     = time_img[sub_time_img_idx]
            sub_dff          = dff[:, sub_time_img_idx]
            sub_mean_fluo_0  = mean_fluo_0[sub_time_img_idx]
            sub_mean_fluo_1  = mean_fluo_1[sub_time_img_idx]
            sub_vol_stim_bin = vol_stim_bin[sub_vol_time_idx]

            # create new figure.
            fig, axs = plt.subplots(num_subplots, 1, figsize=(24, 16))
            plt.subplots_adjust(hspace=0.6)

            # plot mean excitory fluo on functional only.
            upper = np.max(sub_mean_fluo_0)
            lower = np.min(sub_mean_fluo_0)
            axs[0].plot(
                sub_vol_time,
                rescale(sub_vol_stim_bin, upper, lower),
                color='grey',
                label='stimulus',
                lw=0.5)
            axs[0].plot(
                sub_time_img,
                sub_mean_fluo_0,
                color='dodgerblue',
                label='mean of excitory',
                lw=0.5)
            axs[0].set_title(
                'mean trace of {} excitory neurons'.format(
                    np.sum(labels==0)))
            axs[0].set_ylim([lower - 0.1*(upper-lower),
                             upper + 0.1*(upper-lower)])

            # plot mean inhibitory fluo on functional and anatomical channels.
            if ops['nchannels'] == 2:
                upper = np.max(sub_mean_fluo_1)
                lower = np.min(sub_mean_fluo_1)
                axs[1].plot(
                    sub_vol_time,
                    rescale(sub_vol_stim_bin, upper, lower),
                    color='grey',
                    label='stimulus',
                    lw=0.5)
                axs[1].plot(
                    sub_time_img,
                    sub_mean_fluo_1,
                    color='dodgerblue',
                    label='mean of inhibitory',
                    lw=0.5)
                axs[1].set_title(
                    'mean trace of {} inhibitory neurons'.format(
                        np.sum(labels==1)))
                axs[1].set_ylim([lower - 0.1*(upper-lower),
                                 upper + 0.1*(upper-lower)])


            # plot individual traces.
            fluo_color = ['seagreen', 'dodgerblue', 'coral']
            fluo_label = ['excitory', 'unsure', 'inhibitory']
            for i in range(dff.shape[0]):
                upper = np.max(sub_dff[i,:])
                lower = np.min(sub_dff[i,:])
                roi_label = int(labels[i]+1)
                axs[i+2].plot(
                    sub_vol_time,
                    rescale(sub_vol_stim_bin, upper, lower),
                    color='grey',
                    label='stimulus',
                    lw=0.5)
                axs[i+2].plot(
                    sub_time_img,
                    sub_dff[i,:],
                    color=fluo_color[roi_label],
                    label=fluo_label[roi_label],
                    lw=0.5)
                axs[i+2].set_title('raw trace of ROI # '+ str(i).zfill(3))
                axs[i+2].set_ylim([lower - 0.1*(upper-lower),
                                   upper + 0.1*(upper-lower)])

            # adjust layout.
            for i in range(num_subplots):
                axs[i].tick_params(tick1On=False)
                axs[i].spines['left'].set_visible(False)
                axs[i].spines['right'].set_visible(False)
                axs[i].spines['top'].set_visible(False)
                axs[i].spines['bottom'].set_visible(False)
                axs[i].set_xlabel('time / ms')
                axs[i].set_xlim([start, end])
                axs[i].set_xticks(f * max_ms + np.arange(0,max_ms/5000+1) * 5000)
                axs[i].legend(loc='upper left')
            fig.set_size_inches(max_ms/4000, num_subplots*2)
            fig.tight_layout()

            # save figure.
            fig.savefig(os.path.join(
                ops['save_path0'], 'figures', 'fig2_raw_traces',
                str(f).zfill(3)+'.pdf'),
                dpi=300)
            plt.close()

    except:
        print('plotting fig2 failed')
