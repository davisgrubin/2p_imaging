#!/usr/bin/env python3

import numpy as np

# fig, ax = plt.subplots(1, 1, figsize=(6, 6))

# motion correction offsets.
def plot_motion_offset_hist(ax, xoff, yoff):
    center = np.arange(-5,6)
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

# inhibitory/excitory labels.
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
    stim_labels = neural_trials['stim_labels']
    isi = stim_labels[1:,0] - stim_labels[:-1,1]
    ax.hist(isi[(stim_labels[:-1,2]!=-1)*(stim_labels[:-1,3]==0)],
        bins=200, range=[0, np.max(isi)], align='left',
        color='#9DB4CE', density=True, label='normal (short)')
    ax.hist(isi[(stim_labels[:-1,2]!=-1)*(stim_labels[:-1,3]==1)],
        bins=200, range=[0, np.max(isi)], align='right',
        color='dodgerblue', density=True, label='normal (long)')
    ax.hist(isi[(stim_labels[:-1,2]==-1)*(stim_labels[:-1,3]==0)],
        bins=200, range=[0, np.max(isi)], align='left',
        color='#F9C08A', density=True, label='oddball (short)')
    ax.hist(isi[(stim_labels[:-1,2]==-1)*(stim_labels[:-1,3]==1)],
        bins=200, range=[0, np.max(isi)], align='right',
        color='coral', density=True, label='oddball (long)')
    ax.set_title('stimulus interval distribution')
    ax.tick_params(tick1On=False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel('time (ms)')
    ax.set_ylabel('percentage (ms)')
    ax.set_xlim([0, np.max(isi)])
    ax.set_xticks(500*np.arange(0, np.max(isi)/500+1).astype('int32'))
    ax.set_xticklabels(
        500*np.arange(0, np.max(isi)/500+1).astype('int32'),
        rotation='vertical')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:4], labels[:4], loc='upper left')

# stimulus type distribution
def plot_stim_type(ax, neural_trials):
    stim_labels = neural_trials['stim_labels']
    num_omi = np.sum(stim_labels[:,2]==-1)
    num_change = np.sum(stim_labels[:,2]<-1)
    num_img1 = np.sum(stim_labels[:,2]==2)
    num_img2 = np.sum(stim_labels[:,2]==3)
    num_img3 = np.sum(stim_labels[:,2]==4)
    num_img4 = np.sum(stim_labels[:,2]==5)
    ax.pie(
        [num_omi, num_change, num_img1, num_img2, num_img3, num_img4],
        labels=['{} omi'.format(num_omi),
                '{} change'.format(num_change),
                '{} img#1'.format(num_img1),
                '{} img#2'.format(num_img2),
                '{} img#3'.format(num_img3),
                '{} img#4'.format(num_img4)],
        colors=['#989A9C', '#B3D8D5', '#A4CB9E', '#9DB4CE', '#EDA1A4', '#F9C08A'],
        autopct='%1.1f%%',
        wedgeprops={'linewidth': 1, 'edgecolor':'white'})
    ax.set_title('percentage of stimulus types')

# normal type distribution
def plot_normal_type(ax, neural_trials):
    stim_labels = neural_trials['stim_labels']
    num_short = np.sum(stim_labels[:,3]==0)
    num_long = np.sum(stim_labels[:,3]==1)
    ax.pie(
        [num_short, num_long],
        labels=['{} short'.format(num_short),
                '{} long'.format(num_long)],
        colors=['#9DB4CE', '#EDA1A4'],
        autopct='%1.1f%%',
        wedgeprops={'linewidth': 1, 'edgecolor':'white'})
    ax.set_title('percentage of normal types')
    
# fix jitter type distribution
def plot_fix_jitter_type(ax, neural_trials):
    stim_labels = neural_trials['stim_labels']
    num_fix = np.sum(stim_labels[:,4]==0)
    num_jitter = np.sum(stim_labels[:,4]==1)
    ax.pie(
        [num_fix, num_jitter],
        labels=['{} fix'.format(num_fix),
                '{} jitter'.format(num_jitter)],
        colors=['#9DB4CE', '#EDA1A4'],
        autopct='%1.1f%%',
        wedgeprops={'linewidth': 1, 'edgecolor':'white'})
    ax.set_title('percentage of fix/jitter types')

# oddball type distribution
def plot_oddball_type(ax, neural_trials):
    stim_labels = neural_trials['stim_labels']
    num_short = np.sum(stim_labels[:,5]==0)
    num_long = np.sum(stim_labels[:,5]==1)
    ax.pie(
        [num_short, num_long],
        labels=['{} short'.format(num_short),
                '{} long'.format(num_long)],
        colors=['#9DB4CE', '#EDA1A4'],
        autopct='%1.1f%%',
        wedgeprops={'linewidth': 1, 'edgecolor':'white'})
    ax.set_title('percentage of oddball types')

# oddball interval distribution
def plot_oddball_distribution(ax, neural_trials):
    stim_labels = neural_trials['stim_labels']
    interval = np.diff(np.where(stim_labels[:,2]==-1)[0], append=0)
    ax.hist(interval,
        bins=np.max(interval), range=[0, np.max(interval)], align='left',
        color='#F4CE91', density=True)
    ax.set_title('oddball interval distribution')
    ax.tick_params(tick1On=False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel('images')
    ax.set_ylabel('percentage')
    ax.set_xlim([0, np.max(interval)])
    ax.set_xticks(4*np.arange(0, np.max(interval)/4+1).astype('int32'))
    ax.set_xticklabels(
        4*np.arange(0, np.max(interval)/4+1).astype('int32'),
        rotation='vertical')
