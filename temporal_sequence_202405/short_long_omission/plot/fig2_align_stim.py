#!/usr/bin/env python3

import numpy as np

from modules.Alignment import get_stim_response
from plot.utils import exclude_post_odd_stim
from plot.utils import get_mean_sem
from plot.utils import get_roi_label_color
from plot.utils import get_epoch_idx
from plot.utils import pick_trial
from plot.utils import adjust_layout_neu
from plot.utils import utils
    
# fig, ax = plt.subplots(1, 1, figsize=(6, 6))

class plotter_utils(utils):

    def __init__(
            self,
            neural_trials, labels, significance
            ):
        self.l_frames = 50
        self.r_frames = 100
        self.stim_labels = neural_trials['stim_labels'][1:-1,:]
        self.stim_labels = exclude_post_odd_stim(self.stim_labels)
        self.labels = labels
        [self.epoch_short, self.epoch_long] = get_epoch_idx(self.stim_labels)
        [self.neu_seq, self.neu_time,
         self.stim_seq, self.stim_value, self.stim_time, _, _] = get_stim_response(
                neural_trials, self.l_frames, self.r_frames)
        self.significance = significance
        
    def plot_normal(self, ax, normal, cate=None, roi_id=None):
        colors = ['royalblue', 'mediumseagreen', 'violet', 'coral']
        lbl = ['img#1', 'img#2', 'img#3', 'img#4']
        if cate != None:
            neu_cate = self.neu_seq[:,(self.labels==cate)*self.significance['r_normal'],:]
        if roi_id != None:
            neu_cate = np.expand_dims(self.neu_seq[:,roi_id,:], axis=1)
        neu_mean = []
        neu_sem = []
        for i in [2,3,4,5]:
            idx = pick_trial(self.stim_labels, [i], [normal], None, None, [0])
            m, s = get_mean_sem(neu_cate[idx,:,:])
            neu_mean.append(m)
            neu_sem.append(s)
        upper = np.max(neu_mean) + np.max(neu_sem)
        lower = np.min(neu_mean) - np.max(neu_sem)
        stim = np.mean(self.stim_seq[idx,1,:],axis=0)
        ax.fill_between(
            stim,
            lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
            color='gold', alpha=0.15, step='mid', label='stim')
        self.plot_vol(
            ax, self.stim_time, self.stim_value[idx,:],
            'gold', upper, lower)
        for i in range(4):
            self.plot_mean_sem(
                ax, self.neu_time, neu_mean[i], neu_sem[i],
                colors[i], lbl[i])
        adjust_layout_neu(ax)
        ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
        ax.set_xlabel('time since stim (ms)')
    
    def plot_normal_box(self, ax, normal, cate=None, roi_id=None):
        win_base = [-1000,0]
        colors = ['royalblue', 'mediumseagreen', 'violet', 'coral', 'grey']
        lbl = ['img#1', 'img#2', 'img#3', 'img#4', 'all']
        offsets = [0.1, 0.2, 0.3, 0.4, 0.5]
        if cate != None:
            neu_cate = self.neu_seq[:,(self.labels==cate)*self.significance['r_normal'],:]
        if roi_id != None:
            neu_cate = np.expand_dims(self.neu_seq[:,roi_id,:], axis=1)
        neu = []
        for i in [2,3,4,5]:
            idx = pick_trial(self.stim_labels, [i], [normal], None, None, [0])
            neu.append(neu_cate[idx,:,:].copy())
        idx = pick_trial(self.stim_labels, [2,3,4,5], [normal], None, None, [0])
        neu.append(neu_cate[idx,:,:].copy())
        for i in range(5):
            self.plot_win_mag_box(
                ax, neu[i], self.neu_time, win_base, colors[i], 0, offsets[i])
            ax.plot([], color=colors[i], label=lbl[i])
        ax.legend(loc='upper right')
    
    def plot_normal_epoch(self, ax, epoch, normal, colors, cate=None, roi_id=None):
        lbl = ['ep11', 'ep12', 'ep21', 'ep22']
        if cate != None:
            neu_cate = self.neu_seq[:,(self.labels==cate)*self.significance['r_normal'],:]
        if roi_id != None:
            neu_cate = np.expand_dims(self.neu_seq[:,roi_id,:], axis=1)
        neu_mean = []
        neu_sem = []
        for i in range(4):
            idx = epoch[i] * pick_trial(self.stim_labels, [2,3,4,5], [normal], None, None, [0])
            m, s = get_mean_sem(neu_cate[idx,:,:])
            neu_mean.append(m)
            neu_sem.append(s)
        upper = np.max(neu_mean) + np.max(neu_sem)
        lower = np.min(neu_mean) - np.max(neu_sem)
        idx = pick_trial(self.stim_labels, [2,3,4,5], [normal], None, None, [0])
        stim_seq = np.mean(self.stim_seq[idx,1,:],axis=0)
        ax.fill_between(
            stim_seq,
            lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
            color='gold', alpha=0.15, step='mid', label='stim')
        self.plot_vol(
            ax, self.stim_time, self.stim_value[idx,:],
            'gold', upper, lower)
        for i in range(4):
            self.plot_mean_sem(
                ax, self.neu_time, neu_mean[i], neu_sem[i],
                colors[i], lbl[i])
        adjust_layout_neu(ax)
        ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
        ax.set_xlabel('time since stim (ms)')
    
    def plot_normal_epoch_box(self, ax, epoch, normal, colors, cate=None, roi_id=None):
        win_base = [-1000,0]
        lbl = ['ep11', 'ep12', 'ep21', 'ep22']
        offsets = [0.1, 0.2, 0.3, 0.4]
        if cate != None:
            neu_cate = self.neu_seq[:,(self.labels==cate)*self.significance['r_normal'],:]
        if roi_id != None:
            neu_cate = np.expand_dims(self.neu_seq[:,roi_id,:], axis=1)
        neu = []
        for i in range(4):
            idx = epoch[i] * pick_trial(self.stim_labels, [2,3,4,5], [normal], None, None, [0])
            neu.append(neu_cate[idx,:,:].copy())
        for i in range(4):
            self.plot_win_mag_box(
                ax, neu[i], self.neu_time, win_base, colors[i], 0, offsets[i])
            ax.plot([], color=colors[i], label=lbl[i])
        ax.legend(loc='upper right')
        
    def plot_context(self, ax, img_id, cate=None, roi_id=None):
        if cate != None:
            _, color1, color2, _ = get_roi_label_color([cate], 0)
            neu_cate = self.neu_seq[:,(self.labels==cate)*self.significance['r_normal'],:]
        if roi_id != None:
            _, color1, color2, _ = get_roi_label_color(self.labels, roi_id)
            neu_cate = np.expand_dims(self.neu_seq[:,roi_id,:], axis=1)
        idx_short = pick_trial(self.stim_labels, img_id, [0], None, None, [0])
        idx_long  = pick_trial(self.stim_labels, img_id, [1], None, None, [0])
        mean_short, sem_short = get_mean_sem(neu_cate[idx_short,:,:])
        mean_long,  sem_long  = get_mean_sem(neu_cate[idx_long,:,:])
        upper = np.max([mean_short, mean_long]) + np.max([sem_short, sem_long])
        lower = np.min([mean_short, mean_long]) - np.max([sem_short, sem_long])
        for i,c in zip([idx_short, idx_long], [color1, color2]):
            ax.fill_between(
                np.mean(self.stim_seq[i,1,:],axis=0),
                lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
                color=c, alpha=0.15, step='mid')
            ax.fill_between(
                np.mean(self.stim_seq[i,2,:],axis=0),
                lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
                color=c, alpha=0.15, step='mid')
            self.plot_vol(
                ax, self.stim_time, self.stim_value[i,:],
                c, upper, lower)
        self.plot_mean_sem(ax, self.neu_time, mean_short, sem_short, color1, 'short')
        self.plot_mean_sem(ax, self.neu_time, mean_long, sem_long, color2, 'long')
        adjust_layout_neu(ax)
        ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
        ax.set_xlabel('time since stim (ms)')
    
    def plot_context_box(self, ax, cate=None, roi_id=None):
        win_base = [-1500,-300]
        lbl = ['short', 'long']
        offsets = [0.1, 0.2]
        if cate != None:
            _, color1, color2, _ = get_roi_label_color([cate], 0)
            neu_cate = self.neu_seq[:,(self.labels==cate)*self.significance['r_normal'],:]
        if roi_id != None:
            _, color1, color2, _ = get_roi_label_color(self.labels, roi_id)
            neu_cate = np.expand_dims(self.neu_seq[:,roi_id,:], axis=1)
        colors = [color1, color2]
        idx_short = pick_trial(self.stim_labels, [2,3,4,5], [0], None, None, [0])
        idx_long  = pick_trial(self.stim_labels, [2,3,4,5], [1], None, None, [0])
        neu = []
        neu.append(neu_cate[idx_short,:,:].copy())
        neu.append(neu_cate[idx_long,:,:].copy())
        for i in range(2):
            self.plot_win_mag_box(
                    ax, neu[i], self.neu_time, win_base, colors[i], 0, offsets[0])
            ax.plot([], color=colors[i], label=lbl[i])
        ax.legend(loc='upper right')

    def plot_change_prepost(self, ax, cate=None, roi_id=None):
        idx_pre = np.diff(self.stim_labels[:,2]<-1, append=0)
        idx_pre[idx_pre==-1] = 0
        idx_pre = idx_pre.astype('bool')
        idx_post = self.stim_labels[:,2]<-1
        if cate != None:
            neu_cate = self.neu_seq[:,(self.labels==cate)*self.significance['r_change'],:]
            _, color1, color2, _ = get_roi_label_color([cate], 0)
        if roi_id != None:
            neu_cate = np.expand_dims(self.neu_seq[:,roi_id,:], axis=1)
            _, color1, color2, _ = get_roi_label_color(self.labels, roi_id) 
        neu_mean_pre,  neu_sem_pre  = get_mean_sem(neu_cate[idx_pre,:,:])
        neu_mean_post, neu_sem_post = get_mean_sem(neu_cate[idx_post,:,:])
        stim_seq = np.mean(self.stim_seq[self.stim_labels[:,2]>0,1,:],axis=0)
        upper = np.max([neu_mean_pre, neu_mean_post]) + np.max([neu_sem_pre, neu_sem_post])
        lower = np.min([neu_mean_pre, neu_mean_post]) - np.max([neu_sem_pre, neu_sem_post])
        ax.fill_between(
            stim_seq,
            lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
            color='gold', alpha=0.25, step='mid', label='stim')
        self.plot_vol(
            ax, self.stim_time, self.stim_value[idx_post,:],
            'black', upper, lower)
        self.plot_mean_sem(ax, self.neu_time, neu_mean_pre, neu_sem_pre, color1, 'pre')
        self.plot_mean_sem(ax, self.neu_time, neu_mean_post, neu_sem_post, color2, 'post')
        adjust_layout_neu(ax)
        ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
        ax.set_xlabel('time since stim (ms)')
    
    def plot_change_prepost_box(self, ax, cate=None, roi_id=None):
        win_base = [-1000,0]
        lbl = ['pre', 'post']
        offsets = [0.2, 0.4]
        idx_pre = np.diff(self.stim_labels[:,2]<-1, append=0)
        idx_pre[idx_pre==-1] = 0
        idx_pre = idx_pre.astype('bool')
        idx_post = self.stim_labels[:,2]<-1
        if cate != None:
            neu_cate = self.neu_seq[:,(self.labels==cate)*self.significance['r_change'],:]
            _, color1, color2, _ = get_roi_label_color([cate], 0)
        if roi_id != None:
            neu_cate = np.expand_dims(self.neu_seq[:,roi_id,:], axis=1)
            _, color1, color2, _ = get_roi_label_color(self.labels, roi_id) 
        colors = [color1, color2]
        neu = [neu_cate[idx_pre,:,:], neu_cate[idx_post,:,:]]
        for i in range(2):
            self.plot_win_mag_box(
                ax, neu[i], self.neu_time, win_base, colors[i], 0, offsets[i])
            ax.plot([], color=colors[i], label=lbl[i])
        ax.legend(loc='upper right')
    
    # roi response to normal stimulus.
    def roi_normal(self, ax, roi_id):
        self.plot_normal(ax, roi_id=roi_id)
        ax.set_title('response to normal stimulus')

    # roi response to normal stimulus quantification.
    def roi_normal_box(self, ax, roi_id):
        self.plot_normal_box(ax, roi_id=roi_id)
        ax.set_title('response to normal stimulus')
        
    # roi response to normal stimulus single trial heatmap.
    def roi_normal_heatmap_trials(self, ax, roi_id):
        neu_cate = self.neu_seq[:,roi_id,:]
        neu_cate = neu_cate[self.stim_labels[:,2]>0,:]
        _, _, _, cmap = get_roi_label_color(self.labels, roi_id)
        self.plot_heatmap_trials(ax, neu_cate, cmap)
        ax.set_title('single trial response to normal stimulus')
    
    # roi response to image change in short block.
    def roi_change_short(self, ax, roi_id):
        self.plot_change(ax, 0, roi_id=roi_id)
        ax.set_title('response to image change in short block')
    
    # roi response to image change in short block.
    def roi_change_long(self, ax, roi_id):
        self.plot_change(ax, 1, roi_id=roi_id)
        ax.set_title('response to image change in long block')
    
    # roi response to image change quantification.
    def roi_change_box(self, ax, roi_id):
        self.plot_change_box(ax, roi_id=roi_id)
        ax.set_title('response to image change')
    
    # roi response to image change single trial heatmap.
    def roi_change_heatmap_trials(self, ax, roi_id):
        neu_cate = self.neu_seq[:,roi_id,:]
        neu_cate = neu_cate[self.stim_labels[:,2]<-1,:]
        _, _, _, cmap = get_roi_label_color(self.labels, roi_id)
        self.plot_heatmap_trials(ax, neu_cate, cmap)
        ax.set_title('response to image change')
    
    # roi response to short long context for all normal image.
    def roi_context_all(self, ax, roi_id):
        self.plot_context(ax, 5, roi_id=roi_id)
        ax.set_title('response to all normal image')
    
    # roi response to all stimulus average across trial for short.
    def roi_context_all_short_heatmap_trial(self, ax, roi_id):
        idx = (self.stim_labels[:,2]>0) * (self.stim_labels[:,3]==0)
        neu_cate = self.neu_seq[idx,:,:]
        neu_cate = neu_cate[:,roi_id,:]
        _, _, _, cmap = get_roi_label_color(self.labels, roi_id)
        self.plot_heatmap_trials(ax, neu_cate, cmap)
        ax.set_title('single trial response to normal stim (short)')

    # roi response to all stimulus average across trial for long.
    def roi_context_all_long_heatmap_trial(self, ax, roi_id):
        idx = (self.stim_labels[:,2]>0) * (self.stim_labels[:,3]==1)
        neu_cate = self.neu_seq[idx,:,:]
        neu_cate = neu_cate[:,roi_id,:]
        _, _, _, cmap = get_roi_label_color(self.labels, roi_id)
        self.plot_heatmap_trials(ax, neu_cate, cmap)
        ax.set_title('single trial response to normal stim (long)')
    
    # roi response to short long context for individual image.
    def roi_context_individual(self, axs, roi_id):
        titles = [
            'response to normal img#1',
            'response to normal img#2',
            'response to normal img#3',
            'response to normal img#4']
        for i in range(4):
            self.plot_context(axs[i], i+1, roi_id=roi_id)
            axs[i].set_title(titles[i])
            
    # roi response to short long context quantification
    def roi_context_box(self, ax, roi_id):
        self.plot_context_box(ax, roi_id=roi_id)
        ax.set_title('response to different image')
        
        
        
# fig, ax = plt.subplots(1, 1, figsize=(6, 6))
# colors = ['#989A9C', '#A4CB9E', '#9DB4CE', '#EDA1A4', '#F9C08A']
class plotter_VIPTD_G8_align_stim(plotter_utils):
    def __init__(self, neural_trials, labels, significance):
        super().__init__(neural_trials, labels, significance)
    
    def all_normal_exc(self, axs):
        self.normal_short_exc(axs[0])
        self.normal_short_exc_box(axs[1])
        self.normal_epoch_short_exc(axs[2])
        self.normal_epoch_short_exc_box(axs[3])
        self.normal_long_exc(axs[4])
        self.normal_long_exc_box(axs[5])
        self.normal_epoch_long_exc(axs[6])
        self.normal_epoch_long_exc_box(axs[7])
    
    def all_normal_inh(self, axs):
        self.normal_short_inh(axs[0])
        self.normal_short_inh_box(axs[1])
        self.normal_epoch_short_inh(axs[2])
        self.normal_epoch_short_inh_box(axs[3])
        self.normal_long_inh(axs[4])
        self.normal_long_inh_box(axs[5])
        self.normal_epoch_long_inh(axs[6])
        self.normal_epoch_long_inh_box(axs[7])
    
    def all_normal_heatmap(self, axs):
        self.normal_short_short_heatmap_neuron(axs[0])
        self.normal_short_long_heatmap_neuron(axs[1])
        self.normal_long_short_heatmap_neuron(axs[2])
        self.normal_long_long_heatmap_neuron(axs[3])
    
    def all_context_exc(self, axs):
        self.context_exc(axs[0])
        self.context_exc_box(axs[1])
    
    def all_context_inh(self, axs):
        self.context_inh(axs[0])
        self.context_inh_box(axs[1])
    
    def all_change_exc(self, axs):
        self.change_prepost_exc(axs[0])
        self.change_prepost_exc_box(axs[1])
    
    def all_change_inh(self, axs):
        self.change_prepost_inh(axs[0])
        self.change_prepost_inh_box(axs[1])
        
    # excitory response to normal stimulus for short.
    def normal_short_exc(self, ax):
        self.plot_normal(ax, 0, cate=-1)
        ax.set_title('excitory response to normal (short)')
        
    # inhibitory response to normal stimulus for short.
    def normal_short_inh(self, ax):
        self.plot_normal(ax, 0, cate=1)
        ax.set_title('inhibitory response to normal (short)')

    # excitory response to normal stimulus for long.
    def normal_long_exc(self, ax):
        self.plot_normal(ax, 1, cate=-1)
        ax.set_title('excitory response to normal (long)')
        
    # inhibitory response to normal stimulus for long.
    def normal_long_inh(self, ax):
        self.plot_normal(ax, 1, cate=1)
        ax.set_title('inhibitory response to normal (long)')
        
    # excitory response to normal stimulus for short quantification.
    def normal_short_exc_box(self, ax):
        self.plot_normal_box(ax, 0, cate=-1)
        ax.set_title('excitory response to normal (short)')
        
    # inhibitory response to normal stimulus for short quantification.
    def normal_short_inh_box(self, ax):
        self.plot_normal_box(ax, 0, cate=1)
        ax.set_title('inhibitory response to normal (short)')

    # excitory response to normal stimulus for long quantification.
    def normal_long_exc_box(self, ax):
        self.plot_normal_box(ax, 1, cate=-1)
        ax.set_title('excitory response to normal (long)')
        
    # inhibitory response to normal stimulus for long quantification.
    def normal_long_inh_box(self, ax):
        self.plot_normal_box(ax, 1, cate=1)
        ax.set_title('inhibitory response to normal (long)')

    # excitory response to normal stimulus with epoch for short.
    def normal_epoch_short_exc(self, ax):
        colors = ['#969696', '#6BAED6', '#2171B5', '#504099']
        self.plot_normal_epoch(ax, self.epoch_short, 0, colors, cate=-1)
        ax.set_title('excitory response to normal with epoch (short)')
    
    # inhibitory response to normal stimulus with epoch for short.
    def normal_epoch_short_inh(self, ax):
        colors = ['#969696', '#FE7BBF', '#974EC3', '#504099']
        self.plot_normal_epoch(ax, self.epoch_short, 0, colors, cate=1)
        ax.set_title('inhibitory response to normal with epoch (short)')
    
    # excitory response to normal stimulus with epoch for long.
    def normal_epoch_long_exc(self, ax):
        colors = ['#969696', '#6BAED6', '#2171B5', '#504099']
        self.plot_normal_epoch(ax, self.epoch_long, 1, colors, cate=-1)
        ax.set_title('excitory response to normal with epoch (long)')
    
    # inhibitory response to normal stimulus with epoch for long.
    def normal_epoch_long_inh(self, ax):
        colors = ['#969696', '#FE7BBF', '#974EC3', '#504099']
        self.plot_normal_epoch(ax, self.epoch_long, 1, colors, cate=1)
        ax.set_title('inhibitory response to normal with epoch (long)')

    # excitory response to normal stimulus with epoch for short quantification.
    def normal_epoch_short_exc_box(self, ax):
        colors = ['#969696', '#6BAED6', '#2171B5', '#504099']
        self.plot_normal_epoch_box(ax, self.epoch_short, 0, colors, cate=-1)
        ax.set_title('excitory response to normal with epoch (short)')
    
    # inhibitory response to normal stimulus with epoch for short quantification.
    def normal_epoch_short_inh_box(self, ax):
        colors = ['#969696', '#FE7BBF', '#974EC3', '#504099']
        self.plot_normal_epoch_box(ax, self.epoch_short, 0, colors, cate=1)
        ax.set_title('inhibitory response to normal with epoch (short)')
    
    # excitory response to normal stimulus with epoch for long quantification.
    def normal_epoch_long_exc_box(self, ax):
        colors = ['#969696', '#6BAED6', '#2171B5', '#504099']
        self.plot_normal_epoch_box(ax, self.epoch_long, 1, colors, cate=-1)
        ax.set_title('excitory response to normal with epoch (long)')
    
    # inhibitory response to normal stimulus with epoch for long quantification.
    def normal_epoch_long_inh_box(self, ax):
        colors = ['#969696', '#FE7BBF', '#974EC3', '#504099']
        self.plot_normal_epoch_box(ax, self.epoch_long, 1, colors, cate=1)
        ax.set_title('inhibitory response to normal with epoch (long)')
    
    # response to normal stimulus heatmap average across trials for short sorted from short.
    def normal_short_short_heatmap_neuron(self, ax):
        idx_short = pick_trial(self.stim_labels, [2,3,4,5], [0], None, None, [0])
        neu_short = self.neu_seq[idx_short,:,:].copy()
        self.plot_heatmap_neuron(
            ax, neu_short, self.neu_time, neu_short,
            self.significance['r_normal'])
        ax.set_xlabel('time since stim (ms)')
        ax.set_title('response to normal (short sorted by short)')

    # response to normal stimulus heatmap average across trials for short sorted from short.
    def normal_short_long_heatmap_neuron(self, ax):
        idx_short = pick_trial(self.stim_labels, [2,3,4,5], [0], None, None, [0])
        idx_long  = pick_trial(self.stim_labels, [2,3,4,5], [1], None, None, [0])
        neu_short = self.neu_seq[idx_short,:,:].copy()
        neu_long  = self.neu_seq[idx_long,:,:].copy()
        self.plot_heatmap_neuron(
            ax, neu_short, self.neu_time, neu_long,
            self.significance['r_normal'])
        ax.set_xlabel('time since stim (ms)')
        ax.set_title('response to normal (short sorted by long)')
        
    # response to normal stimulus heatmap average across trials for short sorted from short.
    def normal_long_short_heatmap_neuron(self, ax):
        idx_short = pick_trial(self.stim_labels, [2,3,4,5], [0], None, None, [0])
        idx_long  = pick_trial(self.stim_labels, [2,3,4,5], [1], None, None, [0])
        neu_short = self.neu_seq[idx_short,:,:].copy()
        neu_long  = self.neu_seq[idx_long,:,:].copy()
        self.plot_heatmap_neuron(
            ax, neu_long, self.neu_time, neu_short,
            self.significance['r_normal'])
        ax.set_xlabel('time since stim (ms)')
        ax.set_title('response to normal (long sorted by short)')
        
    # response to normal stimulus heatmap average across trials for long sorted from long.
    def normal_long_long_heatmap_neuron(self, ax):
        idx_long  = pick_trial(self.stim_labels, [2,3,4,5], [1], None, None, [0])
        neu_long  = self.neu_seq[idx_long,:,:].copy()
        self.plot_heatmap_neuron(
            ax, neu_long, self.neu_time, neu_long,
            self.significance['r_normal'])
        ax.set_xlabel('time since stim (ms)')
        ax.set_title('response to normal (long sorted by long)')
    
    # excitory response to short long context.
    def context_exc(self, axs):
        titles = [
            'excitory response to all image',
            'excitory response to img#1',
            'excitory response to img#2',
            'excitory response to img#3',
            'excitory response to img#4']
        img_id = [[2,3,4,5], [2], [3], [4], [5]]
        for i in range(5):
            self.plot_context(axs[i], img_id[i], cate=-1)
            axs[i].set_title(titles[i])
    
    # inhibitory response to short long context.
    def context_inh(self, axs):
        titles = [
            'inhibitory response to all image',
            'inhibitory response to img#1',
            'inhibitory response to img#2',
            'inhibitory response to img#3',
            'inhibitory response to img#4']
        img_id = [[2,3,4,5], [2], [3], [4], [5]]
        for i in range(5):
            self.plot_context(axs[i], img_id[i], cate=1)
            axs[i].set_title(titles[i])
    
    # excitory response to short long context quantification.
    def context_exc_box(self, ax):
        self.plot_context_box(ax, cate=-1)
        ax.set_title('excitory response to all image')
    
    # inhibitory response to short long context quantification.
    def context_inh_box(self, ax):
        self.plot_context_box(ax, cate=1)
        ax.set_title('inhibitory response to all image')
        
    # excitory response to image change for all images.
    def change_prepost_exc(self, ax):
        self.plot_change_prepost(ax, cate=-1)
        ax.set_title('excitory response to pre & post change')
    
    # inhibitory response to image change for all images.
    def change_prepost_inh(self, ax):
        self.plot_change_prepost(ax, cate=1)
        ax.set_title('inhibitory response to pre & post change')
    
    # excitoryresponse to image change for all images quantification.
    def change_prepost_exc_box(self, ax):
        self.plot_change_prepost_box(ax, cate=-1)
        ax.set_title('excitory response to pre & post change')
    
    # inhibitory response to image change for all images quantification.
    def change_prepost_inh_box(self, ax):
        self.plot_change_prepost_box(ax, cate=1)
        ax.set_title('inhibitory response to pre & post change')

class plotter_L7G8_align_stim(plotter_utils):
    def __init__(self, neural_trials, labels):
        super().__init__(neural_trials, labels)
        self.labels = -1*np.ones_like(labels)
        
    # response to all stimulus.
    def normal(self, ax):
        self.plot_normal(ax, cate=-1)
        ax.set_title('response to normal stimulus')
    
    # response to all stimulus quantification.
    def normal_box(self, ax):
        self.plot_normal_box(ax, cate=-1)
        ax.set_title('response to normal stimulus')
    
    # response to image change in short block.
    def change_short(self, ax):
        self.plot_change(ax, 0, cate=-1)
        ax.set_title('response to image change (short)')
    
    # response to image change in short block.
    def change_long(self, ax):
        self.plot_change(ax, 1, cate=-1)
        ax.set_title('response to image change (long)')
    
    # response to image change quantification.
    def change_box(self, ax):
        self.plot_change_box(ax, cate=-1)
        ax.set_title('response to image change')

    # response to image change average across trial.
    def change_heatmap_neuron(self, ax):
        neu_seq = self.neu_seq[self.stim_labels[:,2]<-1,:,:]
        self.plot_heatmap_neuron(ax, neu_seq, self.neu_time, neu_seq)
        ax.set_title('mean response to image change')
        
    # response to short long context.
    def context(self, axs):
        titles = [
            'response to img#1',
            'response to img#2',
            'response to img#3',
            'response to img#4',
            'response to all image']
        for i in range(5):
            self.plot_context(axs[i], i+1, cate=-1)
            axs[i].set_title(titles[i])
    
    # response to short long context quantification
    def context_box(self, ax):
        self.plot_context_box(ax, cate=-1)
        ax.set_title('response to different image')
        
    # response to all stimulus average across trial for short.
    def context_all_short_heatmap_neuron(self, ax):
        idx = (self.stim_labels[:,2]>0) * (self.stim_labels[:,3]==0)
        neu_seq = self.neu_seq[idx,:,:]
        self.plot_heatmap_neuron(ax, neu_seq, self.neu_time, neu_seq)
        ax.set_title('mean response to normal stimulus (short)')
    
    # response to all stimulus average across trial for long.
    def context_all_long_heatmap_neuron(self, ax):
        idx = (self.stim_labels[:,2]>0) * (self.stim_labels[:,3]==1)
        neu_seq = self.neu_seq[idx,:,:]
        self.plot_heatmap_neuron(ax, neu_seq, self.neu_time, neu_seq)
        ax.set_title('mean response to normal stimulus (long)')