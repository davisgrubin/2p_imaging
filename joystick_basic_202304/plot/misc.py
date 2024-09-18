#!/usr/bin/env python3

import numpy as np
from plot.utils import get_roi_label_color

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
    ax.set_ylabel('fraction of frames')
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
    ax.set_title('fraction of neuron labels')

# significant neurons ratio.
def plot_significance(ax, significance, labels):
    width = 0.2
    sig = ['vis1', 'push1', 'retract1',
           'vis2', 'push2', 'retract2',
           'reward', 'punish', 'lick']
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
    ax.set_title('fraction of neuron with significant window response')
    ax.tick_params(tick1On=False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylabel('fraction')
    ax.set_xlim([-1,len(sig)+1])
    ax.set_ylim([0,1])
    ax.set_xticks(np.arange(len(sig)))
    ax.set_xticklabels(sig, rotation='vertical')
    ax.plot([], color=c_exc, label='exc')
    ax.plot([], color=c_inh, label='inh')
    ax.legend(loc='upper right')

# roi significance label.
def plot_roi_significance(ax, significance, roi_id):
    labels = ['vis1', 'push1', 'retract1',
              'vis2', 'push2', 'retract2', 
              'reward', 'punish', 'lick']
    respon = [significance['r_vis1'][roi_id],
              significance['r_push1'][roi_id],
              significance['r_retract1'][roi_id],
              significance['r_vis2'][roi_id],
              significance['r_push2'][roi_id],
              significance['r_retract2'][roi_id],
              significance['r_reward'][roi_id],
              significance['r_punish'][roi_id],
              significance['r_lick'][roi_id]]
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
    
    
    
    