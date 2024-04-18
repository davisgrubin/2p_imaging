#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem

from postprocess.ReadResults import read_masks
from postprocess.ReadResults import read_raw_voltages
from postprocess.ReadResults import read_dff
from postprocess.ReadResults import read_neural_trials


#%% ppc

def plot_example_spikes()























# get saved results.

def read_data(ops):
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


#%% plot roi


# find a patch contains roi.

def get_roi_img(
        ops,
        mask,
        size = 128
        ):
    masks = mask['ch'+str(ops['functional_chan'])]['masks']
    input_img = mask['ch'+str(ops['functional_chan'])]['input_img']
    func_img = []
    roi_mask = []
    for i in np.unique(masks)[1:]:
        rows, cols = np.where(masks == i)
        min_row, max_row = rows.min(), rows.max()
        min_col, max_col = cols.min(), cols.max()
        center_row = (min_row + max_row) // 2
        center_col = (min_col + max_col) // 2
        r = int(max(0, min(masks.shape[0] - size, center_row - size/2)))
        c = int(max(0, min(masks.shape[1] - size, center_col - size/2)))
        func_img.append(input_img[r:r+size, c:c+size])
        roi_mask.append((masks[r:r+size, c:c+size]==i)*1)
    return func_img, roi_mask


# automatical adjustment of contrast.

def adjust_contrast(org_img, lower_percentile=75, upper_percentile=99):
    lower = np.percentile(org_img, lower_percentile)
    upper = np.percentile(org_img, upper_percentile)
    img = np.clip((org_img - lower) * 255 / (upper - lower), 0, 255)
    img = img.astype(np.uint8)
    return img


# main function

def plot_roi(
        axs,
        ops, mask,
        col = 0,
        ):

    func_img, roi_mask = get_roi_img(
        ops, mask)

    for i in range(len(roi_mask)):
        # background image is the functional reference frame.
        img = np.zeros((func_img[i].shape[0], func_img[i].shape[1], 3))
        img[:,:,1] = func_img[i]
        img = adjust_contrast(img)
        # add roi.
        img[:,:,0] = roi_mask[i].astype('bool') * 255
        # combine.
        axs[i+1,col].imshow(img)

    # adjust layout
    for i in range(axs.shape[0]):
        axs[i,col].set_title('ROI')
        axs[i,col].tick_params(tick1On=False)
        axs[i,col].spines['left'].set_visible(False)
        axs[i,col].spines['right'].set_visible(False)
        axs[i,col].spines['top'].set_visible(False)
        axs[i,col].spines['bottom'].set_visible(False)
        axs[i,col].set_xticks([])
        axs[i,col].set_yticks([])


#%% plot average fluorescence signal


# extract dff around spikes.

def get_stim_response(
        dff,
        spikes,
        l_frames,
        r_frames,
        percentile = 0.05,
        ):
    all_tri_dff = []
    # check neuron one by one.
    for neu_idx in range(dff.shape[0]):
        neu_tri_dff = []
        # find the largest num_spikes spikes.
        neu_spike_time = np.nonzero(spikes[neu_idx,:])[0]
        neu_spike_amp = spikes[neu_idx,neu_spike_time]
        if len(neu_spike_amp) > 0:
            num_spikes = int(len(neu_spike_amp)*percentile)
            neu_spike_time = neu_spike_time[np.argsort(-neu_spike_amp)[:num_spikes]]
        # find spikes.
        if len(neu_spike_time) > 0:
            for neu_t in neu_spike_time:
                if (neu_t - l_frames > 0 and
                    neu_t + r_frames < dff.shape[1]):
                    f = dff[neu_idx,neu_t - l_frames:neu_t + r_frames]
                    f = f.reshape(1, -1)
                    f = f / (spikes[neu_idx,neu_t]+1e-5)
                    neu_tri_dff.append(f)
            if len(neu_tri_dff) > 0:
                neu_tri_dff = np.concatenate(neu_tri_dff, axis=0)
            else:
                neu_tri_dff = np.zeros((1, l_frames+r_frames))
        else:
            neu_tri_dff = np.zeros((1, l_frames+r_frames))
        all_tri_dff.append(neu_tri_dff)
    return all_tri_dff


# main function

def plot_dff(
        axs,
        ops, dff, spikes, label,
        l_frames, r_frames,
        col = 1,
        ):

    # get response.

    all_tri_dff = get_stim_response(
        dff, spikes, l_frames, r_frames)
    frame = np.arange(-l_frames, r_frames)

    # individual neuron response.
    color = ['seagreen', 'coral']
    for i in range(dff.shape[0]):
        # individual triggered traces.
        if np.sum(all_tri_dff[i]) != 0:
            dff_mean = np.mean(all_tri_dff[i], axis=0)
            dff_sem = sem(all_tri_dff[i], axis=0)
            # aveerage for one neuron.
            axs[i+1,col].plot(
                frame,
                dff_mean,
                color=color[label[i]],
                label='mean trace')
            axs[i+1,col].fill_between(
                frame,
                dff_mean - dff_sem,
                dff_mean + dff_sem,
                color=color[label[i]],
                alpha=0.2)
            axs[i+1,col].set_title(
                'spike trigger average of neuron # '+ str(i).zfill(3))

    # mean response.
    tri_dff = np.concatenate(all_tri_dff, axis=0)
    tri_dff = tri_dff[
        np.where(np.sum(tri_dff, axis=1)!=0)[0], :]
    dff_mean = np.mean(tri_dff, axis=0)
    dff_sem = sem(tri_dff, axis=0)
    axs[0,col].plot(
        frame,
        dff_mean,
        color='coral',
        label='mean trace')
    axs[0,col].fill_between(
        frame,
        dff_mean - dff_sem,
        dff_mean + dff_sem,
        color='coral',
        alpha=0.2)
    axs[0,col].set_title(
        'mean spike trigger average across {} neurons'.format(
            dff.shape[0]))

    # adjust layout.
    for i in range(axs.shape[0]):
        axs[i,col].tick_params(axis='y', tick1On=False)
        axs[i,col].spines['left'].set_visible(False)
        axs[i,col].spines['right'].set_visible(False)
        axs[i,col].spines['top'].set_visible(False)
        axs[i,col].set_xlabel('frame')
        axs[i,col].set_ylabel('response')


#%% plot firing rate distribution


# find imaging trigger time stamps.

def get_img_time(
        time_vol,
        vol_img_bin
        ):
    diff_vol = np.diff(vol_img_bin, append=0)
    idx_up = np.where(diff_vol == 1)[0]+1
    img_time = time_vol[idx_up]
    return img_time


# get inter spike interval.

def get_spike_interval(
        spikes,
        time_img
        ):
    intervals = []
    for i in range(spikes.shape[0]):
        idx = np.where(spikes[i,:]!=0)[0]
        if len(idx) > 0:
            intervals.append(np.diff(time_img[idx])/1000)
        else:
            intervals.append(np.nan)
    return intervals


# main function for firing rate distribution.

def plot_dist(
        axs,
        spikes,
        vol_time, vol_img_bin,
        col=2
        ):
    time_img = get_img_time(vol_time, vol_img_bin)
    intervals = get_spike_interval(spikes, time_img)

    for i in range(spikes.shape[0]):
        if not np.isnan(np.sum(intervals[i])):
            axs[i+1,col].hist(intervals[i], bins=100, range=[0,5])

    # adjust layout.
    for i in range(axs.shape[0]):
        axs[i,col].set_title('firing interval distribution')
        axs[i,col].tick_params(axis='y', tick1On=False)
        axs[i,col].spines['left'].set_visible(False)
        axs[i,col].spines['right'].set_visible(False)
        axs[i,col].spines['top'].set_visible(False)
        axs[i,col].set_xlim(0,5)
        axs[i,col].set_xlabel('time / sec')
        axs[i,col].set_ylabel('response')


#%% main

def plot_fig3(
        ops,
        l_frames = 10,
        r_frames = 30,
        ):

    try:
        print('plotting fig3 spike trigger fluorescence average')

        [mask, label,
         dff, spikes,
         vol_img_bin, vol_stim_bin, vol_time] = read_data(ops)

        # create canvas.
        num_subplots = dff.shape[0] + 1
        fig, axs = plt.subplots(num_subplots, 3, figsize=(20, 10))
        plt.subplots_adjust(hspace=0.2)

        # plot roi and reference frame.
        plot_roi(
            axs,
            ops, mask)

        # plot average fluorescence signal.
        plot_dff(
            axs,
            ops, dff, spikes, label,
            l_frames, r_frames)

        # plot firing rate.
        plot_dist(
            axs,
            spikes,
            vol_time, vol_img_bin)

        # adjust layout.
        fig.set_size_inches(10, num_subplots*3)
        fig.tight_layout()

        # save figure
        fig.savefig(os.path.join(
            ops['save_path0'], 'figures', 'fig3_spike_trigger_average.pdf'),
            dpi=300)
        plt.close()

    except:
        print('plotting fig3 failed')

