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

# significant neurons ratio.
def plot_significance(ax, significance):
    labels = ['vis1', 'press1', 'retract1',
              'vis2', 'press2',
              'reward', 'punish', 'lick']
    respon = [
        [np.sum(significance['r_vis1']==0), np.sum(significance['r_vis1']==1)],
        [np.sum(significance['r_press1']==0), np.sum(significance['r_press1']==1)],
        [np.sum(significance['r_retract1']==0), np.sum(significance['r_retract1']==1)],
        [np.sum(significance['r_vis2']==0), np.sum(significance['r_vis2']==1)],
        [np.sum(significance['r_press2']==0), np.sum(significance['r_press2']==1)],
        [np.sum(significance['r_reward']==0), np.sum(significance['r_reward']==1)],
        [np.sum(significance['r_punish']==0), np.sum(significance['r_punish']==1)],
        [np.sum(significance['r_lick']==0), np.sum(significance['r_lick']==1)]]
    for i in range(len(labels)):
        ax.bar(
            i, respon[i][0]/(respon[i][0]+respon[i][1]),
            bottom=0,
            edgecolor='white', width=0.25, color='#989A9C')
        ax.bar(
            i, respon[i][1]/(respon[i][0]+respon[i][1]),
            bottom=respon[i][0]/(respon[i][0]+respon[i][1]),
            edgecolor='white', width=0.25, color='#F9C08A')
    ax.set_title('percentage of neuron with significant window response')
    ax.tick_params(tick1On=False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.grid(True)
    ax.set_ylabel('percentage')
    ax.set_xlim([-1,len(labels)+1])
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation='vertical')
    ax.plot([], color='#F9C08A', label='Y')
    ax.plot([], color='#989A9C', label='N')
    ax.legend(loc='upper right')


# roi significance label.
def plot_roi_significance(ax, significance, roi_id):
    labels = ['vis1', 'press1', 'retract1',
              'vis2', 'press2',
              'reward', 'punish', 'lick']
    respon = [significance['r_vis1'][roi_id],
              significance['r_press1'][roi_id],
              significance['r_retract1'][roi_id],
              significance['r_vis2'][roi_id],
              significance['r_press2'][roi_id],
              significance['r_reward'][roi_id],
              significance['r_punish'][roi_id],
              significance['r_lick'][roi_id]]
    for i in range(len(labels)):
        if respon[i]:
            ax.bar(i, 1, bottom=0, edgecolor='white', width=0.25, color='#F9C08A')
        else:
            ax.bar(i, 1, bottom=0, edgecolor='white', width=0.25, color='#989A9C')
    ax.set_title('window response significance test label')
    ax.tick_params(tick1On=False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_yticks([])
    ax.set_xlim([-1,len(labels)+1])
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation='vertical')
    ax.plot([], color='#F9C08A', label='Y')
    ax.plot([], color='#989A9C', label='N')
    ax.legend(loc='upper right')