#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt

from modules import PostProcess


# read and process stimulus.

def read_data(
        ops
        ):
    [_, _, raw_voltages, neural_trials] = PostProcess.run(ops)
    stim_vol = raw_voltages['vol_stim_bin']
    time_vol = np.arange(0, len(stim_vol))
    stim_align = []
    time_align = []
    for trial in range(len(neural_trials)-1):
        stim_align.append(neural_trials[str(trial)]['stim'])
        time_align.append(neural_trials[str(trial)]['time'])
    stim_align = np.concatenate(stim_align)
    time_align = np.concatenate(time_align)
    return [stim_vol, time_vol, stim_align, time_align]


# compute duration of sequence.

def frame_dur(stim, time):
    diff_stim = np.diff(stim, prepend=0)
    idx_up   = np.where(diff_stim == 1)[0]
    idx_down = np.where(diff_stim == -1)[0]
    dur_high = time[idx_down] - time[idx_up]
    dur_low  = time[idx_up[1:]] - time[idx_down[:-1]]
    return [dur_high, dur_low]


# main function for plot

def plot_fig2(
        ops,
        bins=100,
        ):
    try:
        print('plotting fig2 stimulus interval distribution')

        # read stimulus sequence.
        [stim_vol, time_vol, stim_align, time_align] = read_data(ops)

        # get durations
        [dur_high_vol, dur_low_vol] = frame_dur(stim_vol, time_vol)
        [dur_high_align, dur_low_align] = frame_dur(stim_align, time_align)

        # plot figs.
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        # voltage.
        axs[0].hist(dur_high_vol,
                    bins=bins, range=[0, 2000], align='left',
                    color='dodgerblue', label='flashing')
        axs[0].hist(dur_low_vol,  bins=bins, range=[0, 2000], align='left',
                    color='coral', label='grey')
        axs[0].set_title('interval distribution from voltage recordings')
        # alignment.
        axs[1].hist(dur_high_align, bins=bins, range=[0, 2000], align='left',
                    color='dodgerblue', label='flashing')
        axs[1].hist(dur_low_align,  bins=bins, range=[0, 2000], align='left',
                    color='coral', label='grey')
        axs[1].set_title('interval distribution from aligned sequence')
        # adjust layout.
        for i in range(2):
            axs[i].spines['left'].set_visible(False)
            axs[i].spines['right'].set_visible(False)
            axs[i].spines['top'].set_visible(False)
            axs[i].yaxis.grid(True)
            axs[i].set_xlabel('time / ms')
            axs[i].set_xlim([0, 2000])
            axs[i].set_xticks(np.arange(0, 21)*100)
            axs[i].set_xticklabels(np.arange(0, 21)*100, rotation='vertical')
        handles, labels = axs[i].get_legend_handles_labels()
        fig.legend(handles[-2:], labels[-2:], loc='upper right')
        fig.suptitle('Inter stimulus interval distribution')
        fig.tight_layout()
        # save figure.
        fig.savefig(os.path.join(
            ops['save_path0'], 'figures',
            'fig2_stim_distribution.pdf'),
            dpi=300)
        plt.close()

    except:
        print('plotting fig2 failed')
