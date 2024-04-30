#!/usr/bin/env python3

import numpy as np


# plot motion correction offsets.

def plot_motion_offset_hist(ax, xoff, yoff):
    range_min = np.min(np.unique(xoff).tolist() + np.unique(yoff).tolist())
    range_max = np.max(np.unique(xoff).tolist() + np.unique(yoff).tolist())
    center = np.arange(range_min,range_max+1)
    width = 0.25
    for c in center:
        ax.bar(
            x=c-width/2,
            height=np.sum(xoff[xoff!=0]==c),
            width=width, color='#A4CB9E', label='x')
        ax.bar(
            x=c+width/2,
            height=np.sum(yoff[yoff!=0]==c),
            width=width, color='#F9C08A', label='y')
    ax.tick_params(tick1On=False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.grid(axis='y')
    ax.set_xticks(center)
    ax.set_xlabel('offset pixels')
    ax.set_ylabel('frames')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:2], labels[:2], loc='upper right')
    ax.set_title('motion correction offset distribution for {} frames'.format(
        len(xoff)))


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
        colors=['#A4CB9E', '#9DB4CE', '#EDA1A4'],
        autopct='%1.1f%%',
        wedgeprops={'linewidth': 1, 'edgecolor':'white'})
    ax.set_title('percentage of neuron labels')


# isi distribution

def plot_isi_distribution(ax, neural_trials):
    vol_stim = [neural_trials[trials]['vol_stim'] for trials in neural_trials.keys()]
    vol_time = [neural_trials[trials]['vol_time'] for trials in neural_trials.keys()]
    diff_stim = [np.diff(stim.copy(), prepend=0) for stim in vol_stim]
    isi = []
    for i in range(len(diff_stim)):
        if len(np.where(diff_stim[i] == 1)[0]) == 2:
            isi.append(vol_time[i][diff_stim[i]==1][1] -
                       vol_time[i][diff_stim[i]==-1][0])
    ax.hist(isi,
        bins=100, range=[0, np.max(isi)], align='left',
        color='#F4CE91', density=True)
    ax.set_title('interval distribution for double stimulus trial')
    ax.tick_params(axis='y', tick1On=False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.grid(True)
    ax.set_xlabel('time (ms)')
    ax.set_xlim([0, np.max(isi)])
    ax.set_xticks(500*np.arange(0, np.max(isi)/500+1).astype('int32'))
    ax.set_xticklabels(
        500*np.arange(0, np.max(isi)/500+1).astype('int32'),
        rotation='vertical')



# fig, ax = plt.subplots(1, 1, figsize=(6, 6))
