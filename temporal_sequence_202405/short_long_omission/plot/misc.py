#!/usr/bin/env python3

import numpy as np
from plot.utils import get_roi_label_color

# fig, ax = plt.subplots(1, 1, figsize=(6, 6))

# used session names.
def plot_sess_name(ax, list_session_data_path):
    names = [p.split('/')[-1] for p in list_session_data_path]
    for i, name in enumerate(names):
        ax.text(0, len(names)-i-1, name, ha='left', va='center', fontsize=12)
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(-10, len(names))
    ax.set_title('used sessions for alignment')

# motion correction offsets.
def plot_motion_offset_hist(ax, list_move_offset):
    xoff = np.concatenate([l[0] for l in list_move_offset])
    yoff = np.concatenate([l[1] for l in list_move_offset])
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
def plot_inh_exc_label_pc(ax, list_labels):
    labels = np.concatenate(list_labels)
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
    ax.set_title('percentage of {} neuron labels'.format(len(labels)))

# significant neurons ratio.
def plot_significance(ax, list_significance, list_labels):
    labels = np.concatenate(list_labels)
    significance = {}
    significance['r_normal'] = np.concatenate(
        [list_significance[i]['r_normal'] for i in range(len(list_significance))])
    significance['r_change'] = np.concatenate(
        [list_significance[i]['r_change'] for i in range(len(list_significance))])
    significance['r_oddball'] = np.concatenate(
        [list_significance[i]['r_oddball'] for i in range(len(list_significance))])
    width = 0.1
    sig = ['normal', 'change', 'oddball']
    _, _, c_exc, _ = get_roi_label_color([-1], 0)
    _, _, c_inh, _ = get_roi_label_color([1], 0)
    for i in range(len(sig)):
        r0_exc = np.sum((significance['r_'+sig[i]]==0)*(labels==-1))
        r1_exc = np.sum((significance['r_'+sig[i]]==1)*(labels==-1))
        r0_inh = np.sum((significance['r_'+sig[i]]==0)*(labels==1))
        r1_inh = np.sum((significance['r_'+sig[i]]==1)*(labels==1))
        ax.bar(
            i-width/2, r1_exc/(r0_exc+r1_exc),
            bottom=0,
            width=width, color=c_exc)
        ax.bar(
            i+width/2, r1_inh/(r0_inh+r1_inh),
            bottom=0,
            width=width, color=c_inh)
    ax.set_title('percentage of neuron with significant window response')
    ax.tick_params(tick1On=False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylabel('percentage')
    ax.set_xlim([-1,len(sig)+1])
    ax.set_ylim([0,1])
    ax.set_xticks(np.arange(len(sig)))
    ax.set_xticklabels(sig, rotation='vertical')
    ax.plot([], color=c_exc, label='exc')
    ax.plot([], color=c_inh, label='inh')
    ax.legend(loc='upper right')

# roi significance label.
def plot_roi_significance(ax, significance, roi_id):
    labels = ['normal', 'change', 'oddball']
    respon = [significance['r_normal'][roi_id],
              significance['r_change'][roi_id],
              significance['r_oddball'][roi_id]]
    for i in range(len(labels)):
        if respon[i]:
            ax.bar(i, 1, bottom=0, edgecolor='white', width=0.25, color='#F9C08A')
    ax.set_title('window response significance test label')
    ax.tick_params(tick1On=False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_yticks([])
    ax.set_xlim([-1,len(labels)+1])
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation='vertical')

    
# isi distribution
def plot_isi_distribution(ax, list_neural_trials):
    stim_labels = np.concatenate([nt['stim_labels'] for nt in list_neural_trials], axis=0)
    isi = stim_labels[1:,0] - stim_labels[:-1,1]
    ax.hist(isi[(stim_labels[:-1,2]!=-1)*(stim_labels[:-1,3]==0)],
        bins=200, range=[0, np.nanmax(isi)], align='left',
        color='#9DB4CE', density=True, label='normal (short)')
    ax.hist(isi[(stim_labels[:-1,2]!=-1)*(stim_labels[:-1,3]==1)],
        bins=200, range=[0, np.nanmax(isi)], align='right',
        color='dodgerblue', density=True, label='normal (long)')
    ax.hist(isi[(stim_labels[:-1,2]==-1)*(stim_labels[:-1,3]==0)],
        bins=200, range=[0, np.nanmax(isi)], align='left',
        color='#F9C08A', density=True, label='oddball (short)')
    ax.hist(isi[(stim_labels[:-1,2]==-1)*(stim_labels[:-1,3]==1)],
        bins=200, range=[0, np.nanmax(isi)], align='right',
        color='coral', density=True, label='oddball (long)')
    ax.set_title('stimulus interval distribution')
    ax.tick_params(tick1On=False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel('time (ms)')
    ax.set_ylabel('percentage')
    ax.set_xlim([0, np.nanmax(isi)])
    ax.set_xticks(500*np.arange(0, np.nanmax(isi)/500+1).astype('int32'))
    ax.set_xticklabels(
        500*np.arange(0, np.max(isi)/500+1).astype('int32'),
        rotation='vertical')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:4], labels[:4], loc='upper left')

# stimulus type distribution
def plot_stim_type(ax, list_neural_trials):
    stim_labels = np.concatenate([nt['stim_labels'] for nt in list_neural_trials], axis=0)
    num_odd = np.sum(stim_labels[:,2]==-1)
    num_change = np.sum(stim_labels[:,2]<-1)
    num_img1 = np.sum(stim_labels[:,2]==2)
    num_img2 = np.sum(stim_labels[:,2]==3)
    num_img3 = np.sum(stim_labels[:,2]==4)
    num_img4 = np.sum(stim_labels[:,2]==5)
    ax.pie(
        [num_odd, num_change, num_img1, num_img2, num_img3, num_img4],
        labels=['{} odd'.format(num_odd),
                '{} change'.format(num_change),
                '{} img#1'.format(num_img1),
                '{} img#2'.format(num_img2),
                '{} img#3'.format(num_img3),
                '{} img#4'.format(num_img4)],
        colors=['#989A9C', '#B3D8D5', '#A4CB9E', '#9DB4CE', '#EDA1A4', '#F9C08A'],
        autopct='%1.1f%%',
        wedgeprops={'linewidth': 1, 'edgecolor':'white'})
    ax.set_title('percentage of {} stimulus types'.format(
        num_odd+num_change+num_img1+num_img2+num_img3+num_img4))

# normal type distribution
def plot_normal_type(ax, list_neural_trials):
    stim_labels = np.concatenate([nt['stim_labels'] for nt in list_neural_trials], axis=0)
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
def plot_fix_jitter_type(ax, list_neural_trials):
    stim_labels = np.concatenate([nt['stim_labels'] for nt in list_neural_trials], axis=0)
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
def plot_oddball_type(ax, list_neural_trials):
    stim_labels = np.concatenate([nt['stim_labels'] for nt in list_neural_trials], axis=0)
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
