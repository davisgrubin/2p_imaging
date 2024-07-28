#!/usr/bin/env python3

import numpy as np

from modules.Alignment import get_stim_response
from plot.utils import get_mean_sem
from plot.utils import get_roi_label_color
from plot.utils import get_epoch_idx
from plot.utils import get_odd_stim_prepost_idx
from plot.utils import adjust_layout_neu
from plot.utils import utils


class plotter_utils(utils):
    
    def __init__(
            self,
            neural_trials, labels, significance
            ):
        self.l_frames = 200
        self.r_frames = 200
        self.cut_frames = 100
        self.labels = labels
        self.stim_labels = neural_trials['stim_labels'][1:-1,:]
        self.epoch_idx = get_epoch_idx(self.stim_labels)
        [self.neu_seq, self.neu_time, self.stim_seq,
         self.stim_value, self.stim_time] = get_stim_response(
                neural_trials, self.l_frames, self.r_frames)
        self.significance = significance
        [self.idx_pre_short,  self.idx_pre_long,
         self.idx_post_short, self.idx_post_long] = get_odd_stim_prepost_idx(
             self.stim_labels)
        
    def plot_odd_pre(self, ax, s, cate=None, roi_id=None):
        if cate != None:
            neu_cate = self.neu_seq[:,self.labels==cate,:]
            _, color1, color2, _ = get_roi_label_color([cate], 0)
        if roi_id != None:
            neu_cate = np.expand_dims(self.neu_seq[:,roi_id,:], axis=1)
            _, color1, color2, _ = get_roi_label_color(self.labels, roi_id)
        mean_short, sem_short = get_mean_sem(neu_cate[self.idx_pre_short,:,:])
        mean_long,  sem_long  = get_mean_sem(neu_cate[self.idx_pre_long,:,:])
        stim_seq_short = np.mean(self.stim_seq[self.idx_pre_short,:,:], axis=0)
        stim_seq_long  = np.mean(self.stim_seq[self.idx_pre_long,:,:], axis=0)
        stim_value_short = self.stim_value[self.idx_pre_short,:]
        stim_value_long  = self.stim_value[self.idx_pre_long,:]
        upper = np.max([mean_short, mean_long]) + np.max([sem_short, sem_long])
        lower = np.min([mean_short, mean_long]) - np.max([sem_short, sem_long])
        ax.fill_between(
            (stim_seq_short[1,:] + stim_seq_long[1,:])/2,
            lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
            color='gold', alpha=0.15, step='mid', label='pre')
        ax.fill_between(
            stim_seq_short[2,:],
            lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
            color=color1, alpha=0.15, step='mid', label='post (short)')
        ax.fill_between(
            stim_seq_long[2,:],
            lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
            color=color2, alpha=0.15, step='mid', label='post (long)')
        self.plot_vol(ax, self.stim_time, stim_value_short, color1, upper, lower)
        self.plot_vol(ax, self.stim_time, stim_value_long,  color2, upper, lower)
        self.plot_mean_sem(ax, self.neu_time, mean_short, sem_short, color1, 'short')
        self.plot_mean_sem(ax, self.neu_time, mean_long,  sem_long,  color2, 'long')
        adjust_layout_neu(ax)
        ax.set_xlim([self.neu_time[np.argmin(np.abs(self.neu_time))-self.cut_frames],
                     self.neu_time[-1]])
        ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
        ax.set_xlabel('time since pre oddball stim (ms)')
        ax.legend(loc='upper right')
    
    def plot_odd_post(self, ax, s, cate=None, roi_id=None):
        if cate != None:
            neu_cate = self.neu_seq[:,self.labels==cate,:]
            _, color1, color2, _ = get_roi_label_color([cate], 0)
        if roi_id != None:
            neu_cate = np.expand_dims(self.neu_seq[:,roi_id,:], axis=1)
            _, color1, color2, _ = get_roi_label_color(self.labels, roi_id)
        mean_short, sem_short = get_mean_sem(neu_cate[self.idx_post_short,:,:])
        mean_long,  sem_long  = get_mean_sem(neu_cate[self.idx_post_long,:,:])
        stim_seq_short = np.mean(self.stim_seq[self.idx_post_short,:,:], axis=0)
        stim_seq_long  = np.mean(self.stim_seq[self.idx_post_long,:,:], axis=0)
        stim_value_short = self.stim_value[self.idx_post_short,:]
        stim_value_long  = self.stim_value[self.idx_post_long,:]
        upper = np.max([mean_short, mean_long]) + np.max([sem_short, sem_long])
        lower = np.min([mean_short, mean_long]) - np.max([sem_short, sem_long])
        ax.fill_between(
            (stim_seq_short[1,:] + stim_seq_long[1,:])/2,
            lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
            color='gold', alpha=0.15, step='mid', label='post')
        ax.fill_between(
            stim_seq_short[0,:],
            lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
            color=color1, alpha=0.15, step='mid', label='pre (short)')
        ax.fill_between(
            stim_seq_long[0,:],
            lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
            color=color2, alpha=0.15, step='mid', label='pre (long)')
        self.plot_vol(ax, self.stim_time, stim_value_short, color1, upper, lower)
        self.plot_vol(ax, self.stim_time, stim_value_long,  color2, upper, lower)
        self.plot_mean_sem(ax, self.neu_time, mean_short, sem_short, color1, 'short')
        self.plot_mean_sem(ax, self.neu_time, mean_long,  sem_long,  color2, 'long')
        adjust_layout_neu(ax)
        ax.set_xlim([self.neu_time[0],
                     self.neu_time[np.argmin(np.abs(self.neu_time))+self.cut_frames]])
        ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
        ax.set_xlabel('time since post oddball stim (ms)')
        ax.legend(loc='upper left')
    
    def plot_odd_post_box(self, ax, s, cate=None, roi_id=None):
        win_base = [2000,4000]
        lbl = ['short', 'long']
        offsets = [0.2, 0.4]
        if cate != None:
            neu_cate = self.neu_seq[:,self.labels==cate,:]
            _, color1, color2, _ = get_roi_label_color([cate], 0)
        if roi_id != None:
            neu_cate = np.expand_dims(self.neu_seq[:,roi_id,:], axis=1)
            _, color1, color2, _ = get_roi_label_color(self.labels, roi_id)
        colors = [color1, color2]
        neu = [neu_cate[self.idx_post_short,:,:],
               neu_cate[self.idx_post_long,:,:]]
        for i in range(2):
            self.plot_win_mag_box(
                ax, neu[i], self.neu_time, win_base, colors[i], 0, offsets[i])
            ax.plot([], color=colors[i], label=lbl[i])
        ax.legend(loc='upper right')
    
    def plot_odd_post_epoch(self, ax, idx_post_shortlong, colors, s, cate=None, roi_id=None):
        lbl = ['ep11', 'ep12', 'ep21', 'ep22']
        if cate != None:
            neu_cate = self.neu_seq[:,self.labels==cate,:]
            _, color1, color2, _ = get_roi_label_color([cate], 0)
        if roi_id != None:
            neu_cate = np.expand_dims(self.neu_seq[:,roi_id,:], axis=1)
        neu_mean = []
        neu_sem = []
        for i in range(4):
            idx = self.epoch_idx[i] * idx_post_shortlong
            m, s = get_mean_sem(neu_cate[idx,:,:])
            neu_mean.append(m)
            neu_sem.append(s)
        stim_seq = np.mean(self.stim_seq[idx_post_shortlong,:,:],axis=0)
        upper = np.max(neu_mean) + np.max(neu_sem)
        lower = np.min(neu_mean) - np.max(neu_sem)
        ax.fill_between(
            stim_seq[1,:],
            lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
            color='gold', alpha=0.15, step='mid', label='post')
        ax.fill_between(
            stim_seq[0,:],
            lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
            color=color1, alpha=0.15, step='mid', label='pre')
        self.plot_vol(
            ax, self.stim_time, self.stim_value[idx_post_shortlong,:],
            'black', upper, lower)
        for i in range(4):
            self.plot_mean_sem(
                ax, self.neu_time, neu_mean[i], neu_sem[i],
                colors[i], lbl[i])
        adjust_layout_neu(ax)
        ax.set_xlim([self.neu_time[0],
                     self.neu_time[np.argmin(np.abs(self.neu_time))+self.cut_frames]])
        ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
        ax.set_xlabel('time since post oddball stim (ms)')
        ax.legend(loc='upper left')
    
    def plot_odd_post_epoch_box(self, ax, idx_post_shortlong, colors, s, cate=None, roi_id=None):
        win_base = [2000,4000]
        lbl = ['ep11', 'ep12', 'ep21', 'ep22']
        offsets = [0.1, 0.2, 0.3, 0.4]
        if cate != None:
            neu_cate = self.neu_seq[:,self.labels==cate,:]
        if roi_id != None:
            neu_cate = np.expand_dims(self.neu_seq[:,roi_id,:], axis=1)
        neu = []
        for i in range(4):
            idx = self.epoch_idx[i] * idx_post_shortlong
            neu.append(neu_cate[idx,:,:].copy())
        for i in range(4):
            self.plot_win_mag_box(
                ax, neu[i], self.neu_time, win_base, colors[i], 0, offsets[i])
            ax.plot([], color=colors[i], label=lbl[i])
        ax.legend(loc='upper right')
        

class plotter_VIPTD_G8_align_odd(plotter_utils):
    def __init__(self, neural_trials, labels, significance):
        super().__init__(neural_trials, labels, significance)
        
    # excitory response to pre oddball stimulus.
    def odd_pre_exc(self, ax):
        self.plot_odd_pre(ax, self.significance['r_oddball'], cate=-1)
        ax.set_title('exitory response to pre oddball stim')
    
    # inhibitory response to pre oddball stimulus.
    def odd_pre_inh(self, ax):
        self.plot_odd_pre(ax, self.significance['r_oddball'], cate=1)
        ax.set_title('inhibitory response to pre oddball stim')
    
    # response to pre short oddball stimulus heatmap average across trials.
    def odd_pre_short_heatmap_neuron(self, ax):
        l = np.argmin(np.abs(self.neu_time))-self.cut_frames
        r = -1
        neu_short = self.neu_seq[self.idx_pre_short,:,:].copy()
        self.plot_heatmap_neuron(
            ax, neu_short[:,:,l:r], self.neu_time[l:r], neu_short[:,:,l:r])
        ax.set_xlabel('time since pre oddball stim (ms)')
        ax.set_title('response to pre short oddball stim')
    
    # response to pre long oddball stimulus heatmap average across trials sorted from short.
    def odd_pre_short_long_heatmap_neuron(self, ax):
        l = np.argmin(np.abs(self.neu_time))-self.cut_frames
        r = -1
        neu_short = self.neu_seq[self.idx_pre_short,:,:].copy()
        neu_long = self.neu_seq[self.idx_pre_long,:,:].copy()
        self.plot_heatmap_neuron(
            ax, neu_long[:,:,l:r], self.neu_time[l:r], neu_short[:,:,l:r])
        ax.set_xlabel('time since pre oddball stim (ms)')
        ax.set_title('response to pre long oddball stim (sorted from short)')
    
    # response to pre long oddball stimulus heatmap average across trials.
    def odd_pre_long_heatmap_neuron(self, ax):
        l = np.argmin(np.abs(self.neu_time))-self.cut_frames
        r = -1
        neu_long = self.neu_seq[self.idx_pre_long,:,:].copy()
        self.plot_heatmap_neuron(
            ax, neu_long[:,:,l:r], self.neu_time[l:r], neu_long[:,:,l:r])
        ax.set_xlabel('time since pre oddball stim (ms)')
        ax.set_title('response to pre long oddball stim')

    # excitory response to post oddball stimulus.
    def odd_post_exc(self, ax):
        self.plot_odd_post(ax, self.significance['r_oddball'], cate=-1)
        ax.set_title('exitory response to post oddball stim')
    
    # inhibitory response to post oddball stimulus.
    def odd_post_inh(self, ax):
        self.plot_odd_post(ax, self.significance['r_oddball'], cate=1)
        ax.set_title('inhibitory response to post oddball stim')

    # excitory response to post oddball stimulus quantification.
    def odd_post_exc_box(self, ax):
        self.plot_odd_post_box(ax, self.significance['r_oddball'], cate=-1)
        ax.set_title('exitory response to post oddball stim')
    
    # inhibitory response to post oddball stimulus quantification.
    def odd_post_inh_box(self, ax):
        self.plot_odd_post_box(ax, self.significance['r_oddball'], cate=1)
        ax.set_title('inhibitory response to post oddball stim')
        
    # response to post short oddball stimulus heatmap average across trials.
    def odd_post_short_heatmap_neuron(self, ax):
        l = 0
        r = np.argmin(np.abs(self.neu_time))+self.cut_frames
        neu_short = self.neu_seq[self.idx_post_short,:,:].copy()
        self.plot_heatmap_neuron(
            ax, neu_short[:,:,l:r], self.neu_time[l:r], neu_short[:,:,l:r])
        ax.set_xlabel('time since post oddball stim (ms)')
        ax.set_title('response to post short oddball stim')
    
    # response to post long oddball stimulus heatmap average across trials sorted from short.
    def odd_post_short_long_heatmap_neuron(self, ax):
        l = 0
        r = np.argmin(np.abs(self.neu_time))+self.cut_frames
        neu_short = self.neu_seq[self.idx_post_short,:,:].copy()
        neu_long = self.neu_seq[self.idx_post_long,:,:].copy()
        self.plot_heatmap_neuron(
            ax, neu_long[:,:,l:r], self.neu_time[l:r], neu_short[:,:,l:r])
        ax.set_xlabel('time since post oddball stim (ms)')
        ax.set_title('response to post long oddball stim (sorted from short)')
    
    # response to post long oddball stimulus heatmap average across trials.
    def odd_post_long_heatmap_neuron(self, ax):
        l = 0
        r = np.argmin(np.abs(self.neu_time))+self.cut_frames
        neu_long = self.neu_seq[self.idx_post_long,:,:].copy()
        self.plot_heatmap_neuron(
            ax, neu_long[:,:,l:r], self.neu_time[l:r], neu_long[:,:,l:r])
        ax.set_xlabel('time since post oddball stim (ms)')
        ax.set_title('response to post long oddball stim')
    
    # excitory response to post short oddball stimulus with epoch.
    def odd_post_exc_short_epoch(self, ax):
        colors = ['#969696', '#6BAED6', '#2171B5', '#504099']
        self.plot_odd_post_epoch(
            ax, self.idx_post_short, colors, self.significance['r_oddball'], cate=-1)
        ax.set_title('exitory response to post short oddball stim')
    
    # inhibitory response to post short oddball stimulus with epoch.
    def odd_post_inh_short_epoch(self, ax):
        colors = ['#969696', '#FE7BBF', '#974EC3', '#504099']
        self.plot_odd_post_epoch(
            ax, self.idx_post_short, colors, self.significance['r_oddball'], cate=1)
        ax.set_title('inhibitory response to post short oddball stim')

    # excitory response to post short oddball stimulus with epoch quantification.
    def odd_post_exc_short_epoch_box(self, ax):
        colors = ['#969696', '#6BAED6', '#2171B5', '#504099']
        self.plot_odd_post_epoch_box(
            ax, self.idx_post_short, colors, self.significance['r_oddball'], cate=-1)
        ax.set_title('exitory response to post short oddball stim')
    
    # inhibitory response to post short oddball stimulus with epoch quantification.
    def odd_post_inh_short_epoch_box(self, ax):
        colors = ['#969696', '#FE7BBF', '#974EC3', '#504099']
        self.plot_odd_post_epoch_box(
            ax, self.idx_post_short, colors, self.significance['r_oddball'], cate=1)
        ax.set_title('inhibitory response to post short oddball stim')
        
    # excitory response to post long oddball stimulus with epoch.
    def odd_post_exc_long_epoch(self, ax):
        colors = ['#969696', '#6BAED6', '#2171B5', '#504099']
        self.plot_odd_post_epoch(
            ax, self.idx_post_long, colors, self.significance['r_oddball'], cate=-1)
        ax.set_title('exitory response to post long oddball stim')
    
    # inhibitory response to post long oddball stimulus with epoch.
    def odd_post_inh_long_epoch(self, ax):
        colors = ['#969696', '#FE7BBF', '#974EC3', '#504099']
        self.plot_odd_post_epoch(
            ax, self.idx_post_long, colors, self.significance['r_oddball'], cate=1)
        ax.set_title('inhibitory response to post long oddball stim')

    # excitory response to post long oddball stimulus with epoch quantification.
    def odd_post_exc_long_epoch_box(self, ax):
        colors = ['#969696', '#6BAED6', '#2171B5', '#504099']
        self.plot_odd_post_epoch_box(
            ax, self.idx_post_long, colors, self.significance['r_oddball'], cate=-1)
        ax.set_title('exitory response to post long oddball stim')
    
    # inhibitory response to post long oddball stimulus with epoch quantification.
    def odd_post_inh_long_epoch_box(self, ax):
        colors = ['#969696', '#FE7BBF', '#974EC3', '#504099']
        self.plot_odd_post_epoch_box(
            ax, self.idx_post_long, colors, self.significance['r_oddball'], cate=1)
        ax.set_title('inhibitory response to post long oddball stim')