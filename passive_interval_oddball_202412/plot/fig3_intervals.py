#!/usr/bin/env python3

import numpy as np
from utils import add_legend

# fig, ax = plt.subplots(1, 1, figsize=(6, 6)) colors=['#A4CB9E', '#F9C08A']

# standard type fraction.
def plot_standard_type(ax, list_neural_trials):
    stim_labels = np.concatenate([nt['stim_labels'] for nt in list_neural_trials], axis=0)
    num_short = np.sum((stim_labels[:,2]!=-1)*(stim_labels[:,3]==0)*(stim_labels[:,6]==0))
    num_long = np.sum((stim_labels[:,2]!=-1)*(stim_labels[:,3]==1)*(stim_labels[:,6]==0))
    if num_short > 0 or num_long > 0:
        ax.pie(
            [num_short, num_long],
            labels=['{} short'.format(num_short),
                    '{} long'.format(num_long)],
            colors=['mediumseagreen','coral'],
            autopct='%1.1f%%',
            wedgeprops={'linewidth': 1, 'edgecolor':'white', 'width':0.2})
        ax.set_title('fraction of standard types')

# fix jitter type fraction.
def plot_fix_jitter_type(ax, list_neural_trials):
    stim_labels = np.concatenate([nt['stim_labels'] for nt in list_neural_trials], axis=0)
    num_fix = np.sum((stim_labels[:,2]!=-1)*(stim_labels[:,4]==0)*(stim_labels[:,6]==0))
    num_jitter = np.sum((stim_labels[:,2]!=-1)*(stim_labels[:,4]==1)*(stim_labels[:,6]==0))
    if num_fix > 0 or num_jitter > 0:
        ax.pie(
            [num_fix, num_jitter],
            labels=['{} fix'.format(num_fix),
                    '{} jitter'.format(num_jitter)],
            colors=['mediumseagreen','coral'],
            autopct='%1.1f%%',
            wedgeprops={'linewidth': 1, 'edgecolor':'white', 'width':0.2})
        ax.set_title('fraction of fix/jitter types')

# oddball type fraction.
def plot_oddball_type(ax, list_neural_trials):
    stim_labels = np.concatenate([nt['stim_labels'] for nt in list_neural_trials], axis=0)
    num_short = np.sum((stim_labels[:,2]==-1)*(stim_labels[:,5]==0)*(stim_labels[:,6]==0))
    num_long = np.sum((stim_labels[:,2]==-1)*(stim_labels[:,5]==1)*(stim_labels[:,6]==0))
    if num_short > 0 or num_long > 0:
        ax.pie(
            [num_short, num_long],
            labels=['{} short'.format(num_short),
                    '{} long'.format(num_long)],
            colors=['mediumseagreen','coral'],
            autopct='%1.1f%%',
            wedgeprops={'linewidth': 1, 'edgecolor':'white', 'width':0.2})
        ax.set_title('fraction of oddball types')

# random type fraction.
def plot_random_type(ax, list_neural_trials):
    stim_labels = np.concatenate([nt['stim_labels'] for nt in list_neural_trials], axis=0)
    num_design = np.sum(stim_labels[:,6]==0)
    num_random = np.sum(stim_labels[:,6]==1)
    if num_design > 0 or num_random > 0:
        ax.pie(
            [num_random, num_design],
            labels=['{} random'.format(num_random),
                    '{} design'.format(num_design)],
            colors=['mediumseagreen','coral'],
            autopct='%1.1f%%',
            wedgeprops={'linewidth': 1, 'edgecolor':'white', 'width':0.2})
        ax.set_title('fraction of random types')

# standard isi distribution.
def plot_standard_isi_distribution(ax, list_neural_trials):
    resolution = 10
    isi_range = [0,6000]
    stim_labels = np.concatenate([nt['stim_labels'] for nt in list_neural_trials], axis=0)
    isi = stim_labels[1:,0] - stim_labels[:-1,1]
    data = isi[(stim_labels[:-1,2]!=-1)*(stim_labels[:-1,3]==0)*(stim_labels[:-1,4]==0)*(stim_labels[:-1,6]==0)]
    ax.hist(data,
        bins=int((isi_range[1]-isi_range[0])/resolution),
        weights=np.ones(len(data))/len(data),
        range=isi_range, align='mid',
        color='mediumseagreen')
    data = isi[(stim_labels[:-1,2]!=-1)*(stim_labels[:-1,3]==1)*(stim_labels[:-1,4]==0)*(stim_labels[:-1,6]==0)]
    ax.hist(data,
        bins=int((isi_range[1]-isi_range[0])/resolution),
        weights=np.ones(len(data))/len(data),
        range=isi_range, align='mid',
        color='coral', density=True)
    ax.set_title('standard interval distribution')
    ax.tick_params(axis='y', tick1On=False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel('interval (ms)')
    ax.set_ylabel('fraction')
    ax.set_xlim(isi_range)
    ax.set_xticks(500*np.arange(isi_range[0], isi_range[1]/500+1).astype('int32'))
    ax.set_xticklabels(
        500*np.arange(isi_range[0], isi_range[1]/500+1).astype('int32'),
        rotation='vertical')
    add_legend(ax, ['mediumseagreen','coral'], ['short standard', 'long standard'], None, None, None, 'upper right')

# jitter isi distribution.
def plot_jitter_isi_distribution(ax, list_neural_trials):
    resolution = 10
    isi_range = [0,6000]
    stim_labels = np.concatenate([nt['stim_labels'] for nt in list_neural_trials], axis=0)
    isi = stim_labels[1:,0] - stim_labels[:-1,1]
    data = isi[(stim_labels[:-1,2]!=-1)*(stim_labels[:-1,4]==1)*(stim_labels[:-1,6]==0)]
    ax.hist(data,
        bins=int((isi_range[1]-isi_range[0])/resolution),
        weights=np.ones(len(data))/len(data),
        range=isi_range, align='mid',
        color='coral', density=True)
    ax.set_title('jitter interval distribution')
    ax.tick_params(axis='y', tick1On=False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel('interval (ms)')
    ax.set_ylabel('fraction')
    ax.set_xlim(isi_range)
    ax.set_xticks(500*np.arange(isi_range[0], isi_range[1]/500+1).astype('int32'))
    ax.set_xticklabels(
        500*np.arange(isi_range[0], isi_range[1]/500+1).astype('int32'),
        rotation='vertical')

# oddball isi distribution.
def plot_oddball_isi_distribution(ax, list_neural_trials):
    resolution = 10
    isi_range = [0,6000]
    stim_labels = np.concatenate([nt['stim_labels'] for nt in list_neural_trials], axis=0)
    isi = stim_labels[1:,0] - stim_labels[:-1,1]
    data = isi[(stim_labels[:-1,2]==-1)*(stim_labels[:-1,5]==0)*(stim_labels[:-1,6]==0)]
    ax.hist(data,
        bins=int((isi_range[1]-isi_range[0])/resolution),
        weights=np.ones(len(data))/len(data),
        range=isi_range, align='mid',
        color='mediumseagreen', density=True)
    data = isi[(stim_labels[:-1,2]==-1)*(stim_labels[:-1,5]==1)]
    ax.hist(data,
        bins=int((isi_range[1]-isi_range[0])/resolution),
        weights=np.ones(len(data))/len(data),
        range=isi_range, align='mid',
        color='coral', density=True)
    ax.set_title('oddball interval distribution')
    ax.tick_params(axis='y', tick1On=False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel('interval (ms)')
    ax.set_ylabel('fraction')
    ax.set_xlim(isi_range)
    ax.set_xticks(500*np.arange(isi_range[0], isi_range[1]/500+1).astype('int32'))
    ax.set_xticklabels(
        500*np.arange(isi_range[0], isi_range[1]/500+1).astype('int32'),
        rotation='vertical')
    add_legend(ax, ['mediumseagreen', 'coral'], ['short oddball', 'long oddball'], None, None, None, 'upper right')

# random isi distribution.
def plot_random_isi_distribution(ax, list_neural_trials):
    resolution = 10
    isi_range = [0,4000]
    stim_labels = np.concatenate([nt['stim_labels'] for nt in list_neural_trials], axis=0)
    isi = stim_labels[1:,0] - stim_labels[:-1,1]
    data = isi[stim_labels[:-1,6]==1]
    ax.hist(data,
        bins=int((isi_range[1]-isi_range[0])/resolution),
        weights=np.ones(len(data))/len(data),
        range=isi_range, align='mid',
        color='mediumseagreen', density=True)
    ax.set_title('random interval distribution')
    ax.tick_params(axis='y', tick1On=False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel('interval (ms)')
    ax.set_ylabel('fraction')
    ax.set_xlim(isi_range)
    ax.set_xticks(500*np.arange(isi_range[0], isi_range[1]/500+1).astype('int32'))
    ax.set_xticklabels(
        500*np.arange(isi_range[0], isi_range[1]/500+1).astype('int32'),
        rotation='vertical')

# stimulus trial structure.
def plot_stim_type(ax, list_neural_trials):
    stim_labels = np.concatenate([nt['stim_labels'] for nt in list_neural_trials], axis=0)
    num_random = np.sum(stim_labels[:,6]==1)
    num_odd = np.sum((stim_labels[:,2]==-1)*(stim_labels[:,6]==0))
    num_change = np.sum((stim_labels[:,2]<-1)*(stim_labels[:,6]==0))
    num_img1 = np.sum((stim_labels[:,2]==2)*(stim_labels[:,6]==0))
    num_img2 = np.sum((stim_labels[:,2]==3)*(stim_labels[:,6]==0))
    num_img3 = np.sum((stim_labels[:,2]==4)*(stim_labels[:,6]==0))
    num_img4 = np.sum((stim_labels[:,2]==5)*(stim_labels[:,6]==0))
    ax.pie(
        [num_random, num_odd, num_change, num_img1, num_img2, num_img3, num_img4],
        labels=['{} random'.format(num_random),
                '{} odd'.format(num_odd),
                '{} change'.format(num_change),
                '{} img#1'.format(num_img1),
                '{} img#2'.format(num_img2),
                '{} img#3'.format(num_img3),
                '{} img#4'.format(num_img4)],
        colors=['silver', 'black', 'yellow', 'cornflowerblue', 'mediumseagreen', 'hotpink', 'coral'],
        autopct='%1.1f%%',
        wedgeprops={'linewidth': 1, 'edgecolor':'white', 'width':0.2})
    ax.set_title('fraction of {} stimulus types'.format(
        num_odd+num_change+num_img1+num_img2+num_img3+num_img4))

# trial labels.
def plot_stim_label(ax, list_neural_trials):
    isi_range = [0,3000]
    colors = ['black', 'cornflowerblue', 'mediumseagreen', 'hotpink', 'coral']
    labels = ['oddball', 'img#1', 'img#2', 'img#3', 'img#4']
    stim_labels = list_neural_trials[-1]['stim_labels']
    img_seq_label = np.abs(stim_labels[:-1,2]).astype('int32')
    isi = stim_labels[1:,0] - stim_labels[:-1,1]
    ax.scatter(np.arange(len(isi)), isi, c=[colors[l-1] for l in img_seq_label])
    ax.tick_params(tick1On=False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.grid(True, linewidth=0.25)
    ax.set_xlabel('trial #')
    ax.set_ylabel('interval (ms)')
    ax.set_ylim(isi_range)
    ax.set_yticks(500*np.arange(isi_range[0], isi_range[1]/500+1).astype('int32'))
    ax.set_yticklabels(
        500*np.arange(isi_range[0], isi_range[1]/500+1).astype('int32'))
    ax.set_title('single trial interval distribution')
    add_legend(ax, colors, labels, None, None, None, 'upper right')
    