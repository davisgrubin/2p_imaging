#!/usr/bin/env python3

import numpy as np

# fig, ax = plt.subplots(1, 1, figsize=(6, 6))

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
            width=width, color='#A4CB9E', label='x')
        ax.bar(
            x=c+width/2,
            height=np.sum(yoff[yoff!=0]==c)/len(yoff),
            width=width, color='#F9C08A', label='y')
    ax.tick_params(tick1On=False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
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
        colors=['#A4CB9E', '#9DB4CE', '#EDA1A4'],
        autopct='%1.1f%%',
        wedgeprops={'linewidth': 1, 'edgecolor':'white'})
    ax.set_title('percentage of neuron labels')


# isi distribution

def plot_isi_distribution(ax, neural_trials):
    vol_stim = np.concatenate([neural_trials[str(i)]['vol_stim']
                for i in range(len(neural_trials))])
    vol_time = np.concatenate([neural_trials[str(i)]['vol_time']
                for i in range(len(neural_trials))])
    max_time = 1000
    vol_stim[vol_stim!=0] = 1
    diff_stim = np.diff(vol_stim, prepend=0)
    idx_up   = np.where(diff_stim == 1)[0]
    idx_down = np.where(diff_stim == -1)[0]
    fix_isi  = vol_time[idx_up[1:]] - vol_time[idx_down[:-1]]
    ax.hist(fix_isi, label='fix',
        bins=200, range=[0, max_time], align='left',
        color='#F4CE91', density=True)
    ax.set_title('interval distribution')
    ax.tick_params(tick1On=False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel('time (ms)')
    ax.set_ylabel('percentage of trials')
    ax.set_xlim([0, max_time])
    ax.set_xticks(500*np.arange(0, max_time/500+1).astype('int32'))
    ax.set_xticklabels(
        500*np.arange(0, max_time/500+1).astype('int32'),
        rotation='vertical')

