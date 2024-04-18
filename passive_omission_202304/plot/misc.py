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
            height=np.sum(xoff[xoff!=0]==c)/len(xoff),
            width=width, color='#A4CB9E', label='x')
        ax.bar(
            x=c+width/2,
            height=np.sum(yoff[yoff!=0]==c)/len(yoff),
            width=width, color='#F9C08A', label='y')
    ax.tick_params(tick1On=False)
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
        colors=['#A4CB9E', '#9DB4CE', '#EDA1A4'],
        autopct='%1.1f%%',
        wedgeprops={'linewidth': 1, 'edgecolor':'white'})
    ax.set_title('percentage of neuron labels')


# isi distribution

def plot_isi_distribution(ax, neural_trials, jitter_flag):
    fix_stim = np.concatenate([neural_trials[str(i)]['vol_stim']
                for i in range(len(jitter_flag)) if jitter_flag[i] == 0])
    fix_time = np.concatenate([neural_trials[str(i)]['vol_time']
                for i in range(len(jitter_flag)) if jitter_flag[i] == 0])
    jitter_stim = np.concatenate([neural_trials[str(i)]['vol_stim']
                   for i in range(len(jitter_flag)) if jitter_flag[i] == 1])
    jitter_time = np.concatenate([neural_trials[str(i)]['vol_time']
                   for i in range(len(jitter_flag)) if jitter_flag[i] == 1])
    max_time = 2500
    fix_stim[fix_stim!=0] = 1
    diff_stim = np.diff(fix_stim, prepend=0)
    idx_up   = np.where(diff_stim == 1)[0]
    idx_down = np.where(diff_stim == -1)[0]
    fix_isi  = fix_time[idx_up[1:]] - fix_time[idx_down[:-1]]
    jitter_stim[jitter_stim!=0] = 1
    diff_stim  = np.diff(jitter_stim, prepend=0)
    idx_up     = np.where(diff_stim == 1)[0]
    idx_down   = np.where(diff_stim == -1)[0]
    jitter_isi = jitter_time[idx_up[1:]] - jitter_time[idx_down[:-1]]
    ax.hist(fix_isi, label='fix',
        bins=100, range=[0, max_time], align='left',
        color='#F4CE91', density=True)
    ax.hist(jitter_isi, label='jitter',
        bins=100, range=[0, max_time], align='left',
        color='#6CBFBF', density=True)
    ax.set_title('interval distribution')
    ax.tick_params(axis='y', tick1On=False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.grid(True)
    ax.set_xlabel('time (ms)')
    ax.set_xlim([0, max_time])
    ax.set_xticks(500*np.arange(0, max_time/500+1).astype('int32'))
    ax.set_xticklabels(
        500*np.arange(0, max_time/500+1).astype('int32'),
        rotation='vertical')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:2], labels[:2], loc='upper right')


# omission distribution.

def plot_omi_distribution(ax, neural_trials, jitter_flag):
    max_time = 15000
    fix_stim = np.concatenate([neural_trials[str(i)]['vol_stim']
                for i in range(len(jitter_flag)) if jitter_flag[i] == 0])
    fix_time = np.concatenate([neural_trials[str(i)]['vol_time']
                for i in range(len(jitter_flag)) if jitter_flag[i] == 0])
    jitter_stim = np.concatenate([neural_trials[str(i)]['vol_stim']
                   for i in range(len(jitter_flag)) if jitter_flag[i] == 1])
    jitter_time = np.concatenate([neural_trials[str(i)]['vol_time']
                   for i in range(len(jitter_flag)) if jitter_flag[i] == 1])
    fix_stim[fix_stim>0] = 0
    diff_fix_stim = np.diff(fix_stim, prepend=0)
    omi_time_fix = fix_time[diff_fix_stim<0]
    omi_interval_fix = np.diff(omi_time_fix)
    jitter_stim[jitter_stim>0] = 0
    diff_jitter_stim = np.diff(jitter_stim, prepend=0)
    omi_time_jitter = jitter_time[diff_jitter_stim<0]
    omi_interval_jitter = np.diff(omi_time_jitter)
    ax.hist(omi_interval_fix, label='fix',
        bins=150, range=[0, max_time], align='left',
        color='#F4CE91', density=True)
    ax.hist(omi_interval_jitter, label='jitter',
        bins=150, range=[0, max_time], align='left',
        color='#6CBFBF', density=True)
    ax.set_title('omission interval distribution')
    ax.tick_params(axis='y', tick1On=False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.grid(True)
    ax.set_xlabel('time (ms)')
    ax.set_xlim([0, max_time])
    ax.set_xticks(5000*np.arange(0, max_time/5000+1).astype('int32'))
    ax.set_xticklabels(
        5000*np.arange(0, max_time/5000+1).astype('int32'),
        rotation='vertical')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:2], labels[:2], loc='upper right')







# fig, ax = plt.subplots(1, 1, figsize=(6, 6))
