#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt


# plot motion correction offsets.

def plot_motion_offset_hist(ax, xoff, yoff):
    range_min = np.min(np.unique(xoff).tolist() + np.unique(yoff).tolist())
    range_max = np.max(np.unique(xoff).tolist() + np.unique(yoff).tolist())
    center = np.arange(range_min,range_max+1)
    width = 0.25
    for c in center:
        ax.bar(
            x=c-width/2,
            height=np.sum(xoff[xoff!=0]==c)/len(xoff),
            width=width, color='dodgerblue', label='x')
        ax.bar(
            x=c+width/2,
            height=np.sum(yoff[yoff!=0]==c)/len(yoff),
            width=width, color='springgreen', label='y')
    ax.tick_params(axis='x', tick1On=False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.grid(axis='y')
    ax.set_xticks(center)
    ax.set_xlabel('offset pixels')
    ax.set_ylabel('percentage of frames')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:2], labels[:2], loc='upper right')
    ax.set_title('motion correction offset distribution')


# plot inhibitory/excitory labels.

def plot_inh_exc_label_pc(ax, labels):
    exc = np.sum(labels==-1)
    uns = np.sum(labels==0)
    inh = np.sum(labels==1)
    ax.pie(
        [exc, uns, inh],
        labels=['{} excitory'.format(exc),
                '{} unsure'.format(uns),
                '{} inhibitory'.format(inh)],
        colors=['mediumseagreen', 'royalblue', 'coral'],
        autopct='%1.1f%%',
        wedgeprops={'linewidth': 1, 'edgecolor':'white'})
    ax.set_title('percentage of neuron labels')


# isi distribution

def plot_isi_distribution(ax, vol_stim_bin, vol_time):
    max_time = 2500
    diff_stim = np.diff(vol_stim_bin, prepend=0)
    idx_up   = np.where(diff_stim == 1)[0]
    idx_down = np.where(diff_stim == -1)[0]
    dur_low  = vol_time[idx_up[1:]] - vol_time[idx_down[:-1]]
    ax.hist(dur_low,
        bins=100, range=[0, max_time], align='left', color='coral', density=True)
    ax.set_title('interval distribution')
    ax.tick_params(axis='y', tick1On=False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.grid(True)
    ax.set_xlabel('time / ms')
    ax.set_xlim([0, max_time])
    ax.set_xticks(500*np.arange(0, max_time/500+1).astype('int32'))
    ax.set_xticklabels(np.arange(0, max_time/500+1)*500, rotation='vertical')








# fig, ax = plt.subplots(1, 1, figsize=(6, 6))
