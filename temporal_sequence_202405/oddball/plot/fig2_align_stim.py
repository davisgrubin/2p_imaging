#!/usr/bin/env python3

import numpy as np

from modules.Alignment import get_stim_response
from plot.utils import exclude_post_odd_stim
from plot.utils import get_mean_sem
from plot.utils import get_roi_label_color
from plot.utils import get_epoch_idx
from plot.utils import adjust_layout_neu
from plot.utils import utils


class plotter_utils(utils):

    def __init__(
            self,
            neural_trials, labels, significance
            ):
        self.l_frames = 30
        self.r_frames = 50
        self.stim_labels = exclude_post_odd_stim(neural_trials['stim_labels'][1:-1,:])
        self.labels = labels
        self.epoch_idx = get_epoch_idx(self.stim_labels)
        [self.neu_seq, self.neu_time, self.stim_seq,
         self.stim_value, self.stim_time] = get_stim_response(
                neural_trials, self.l_frames, self.r_frames)
        self.significance = significance
        
    def plot_normal(self, ax, s, cate=None, roi_id=None):
        colors = ['royalblue', 'mediumseagreen', 'violet', 'coral', 'grey']
        lbl = ['img#1', 'img#2', 'img#3', 'img#4', 'all']
        if cate != None:
            neu_cate = self.neu_seq[:,self.labels==cate,:]
        if roi_id != None:
            neu_cate = np.expand_dims(self.neu_seq[:,roi_id,:], axis=1)
        neu_mean = []
        neu_sem = []
        for i in [2,3,4,5]:
            m, s = get_mean_sem(neu_cate[self.stim_labels[:,2]==i,:,:])
            neu_mean.append(m)
            neu_sem.append(s)
        m, s = get_mean_sem(neu_cate[self.stim_labels[:,2]>0,:,:])
        neu_mean.append(m)
        neu_sem.append(s)
        stim_seq = np.mean(self.stim_seq[self.stim_labels[:,2]>0,1,:],axis=0)
        upper = np.max(neu_mean) + np.max(neu_sem)
        lower = np.min(neu_mean) - np.max(neu_sem)
        ax.fill_between(
            stim_seq,
            lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
            color='gold', alpha=0.25, step='mid', label='stim')
        self.plot_vol(
            ax, self.stim_time, self.stim_value[self.stim_labels[:,2]>0,:],
            'black', upper, lower)
        for i in range(5):
            self.plot_mean_sem(
                ax, self.neu_time, neu_mean[i], neu_sem[i],
                colors[i], lbl[i])
        adjust_layout_neu(ax)
        ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
        ax.set_xlabel('time since stim (ms)')
    
    def plot_normal_box(self, ax, s, cate=None, roi_id=None):
        win_base = [-1000,0]
        colors = ['royalblue', 'mediumseagreen', 'violet', 'coral', 'grey']
        lbl = ['img#1', 'img#2', 'img#3', 'img#4', 'all']
        offsets = [0.1, 0.2, 0.3, 0.4, 0.5]
        if cate != None:
            neu_cate = self.neu_seq[:,self.labels==cate,:]
        if roi_id != None:
            neu_cate = np.expand_dims(self.neu_seq[:,roi_id,:], axis=1)
        neu = []
        for i in [2,3,4,5]:
            neu.append(neu_cate[self.stim_labels[:,2]==i,:,:].copy())
        neu.append(neu_cate[self.stim_labels[:,2]>0,:,:].copy())
        for i in range(5):
            self.plot_win_mag_box(
                ax, neu[i], self.neu_time, win_base, colors[i], 0, offsets[i])
            ax.plot([], color=colors[i], label=lbl[i])
        ax.legend(loc='upper right')
    
    def plot_normal_epoch(self, ax, colors, s, cate=None, roi_id=None):
        lbl = ['ep11', 'ep12', 'ep21', 'ep22']
        if cate != None:
            neu_cate = self.neu_seq[:,self.labels==cate,:]
        if roi_id != None:
            neu_cate = np.expand_dims(self.neu_seq[:,roi_id,:], axis=1)
        neu_mean = []
        neu_sem = []
        for i in range(4):
            idx = self.epoch_idx[i] * self.stim_labels[:,2]>0
            m, s = get_mean_sem(neu_cate[idx,:,:])
            neu_mean.append(m)
            neu_sem.append(s)
        stim_seq = np.mean(self.stim_seq[self.stim_labels[:,2]>0,1,:],axis=0)
        upper = np.max(neu_mean) + np.max(neu_sem)
        lower = np.min(neu_mean) - np.max(neu_sem)
        ax.fill_between(
            stim_seq,
            lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
            color='gold', alpha=0.25, step='mid', label='stim')
        self.plot_vol(
            ax, self.stim_time, self.stim_value[self.stim_labels[:,2]>0,:],
            'black', upper, lower)
        for i in range(4):
            self.plot_mean_sem(
                ax, self.neu_time, neu_mean[i], neu_sem[i],
                colors[i], lbl[i])
        adjust_layout_neu(ax)
        ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
        ax.set_xlabel('time since stim (ms)')
    
    def plot_normal_epoch_box(self, ax, colors, s, cate=None, roi_id=None):
        win_base = [-1000,0]
        lbl = ['ep11', 'ep12', 'ep21', 'ep22']
        offsets = [0.1, 0.2, 0.3, 0.4]
        if cate != None:
            neu_cate = self.neu_seq[:,self.labels==cate,:]
        if roi_id != None:
            neu_cate = np.expand_dims(self.neu_seq[:,roi_id,:], axis=1)
        neu = []
        for i in range(4):
            idx = self.epoch_idx[i] * self.stim_labels[:,2]>0
            neu.append(neu_cate[idx,:,:].copy())
        for i in range(4):
            self.plot_win_mag_box(
                ax, neu[i], self.neu_time, win_base, colors[i], 0, offsets[i])
            ax.plot([], color=colors[i], label=lbl[i])
        ax.legend(loc='upper right')

    def plot_change_img(self, ax, s, cate=None, roi_id=None):
        colors = ['royalblue', 'mediumseagreen', 'violet', 'coral']
        lbl = ['img#1', 'img#2', 'img#3', 'img#4']
        idx_pre = np.diff(self.stim_labels[:,2]<-1, append=0)
        idx_pre[idx_pre==-1] = 0
        idx_pre = idx_pre.astype('bool')
        idx_post = self.stim_labels[:,2]<-1
        if cate != None:
            neu_cate = self.neu_seq[:,self.labels==cate,:]
            _, color1, color2, _ = get_roi_label_color([cate], 0)
        if roi_id != None:
            neu_cate = np.expand_dims(self.neu_seq[:,roi_id,:], axis=1)
            _, color1, color2, _ = get_roi_label_color(self.labels, roi_id)
        neu_mean = []
        neu_sem = []
        for i in [2,3,4,5]:
            i_post = idx_post * (self.stim_labels[:,2]==-i)
            m_post, s_post = get_mean_sem(neu_cate[i_post,:,:])
            neu_mean.append(m_post)
            neu_sem.append(s_post)
        upper = np.max(neu_mean) + np.max(neu_sem)
        lower = np.min(neu_mean) - np.max(neu_sem)
        stim_seq = np.mean(self.stim_seq[self.stim_labels[:,2]>0,1,:],axis=0)
        ax.fill_between(
            stim_seq,
            lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
            color='gold', alpha=0.25, step='mid', label='stim')
        self.plot_vol(
            ax, self.stim_time, self.stim_value[idx_post,:],
            'black', upper, lower)
        for i in range(4):
            self.plot_mean_sem(ax, self.neu_time, neu_mean[i], neu_sem[i], colors[i], lbl[i])
        adjust_layout_neu(ax)
        ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
        ax.set_xlabel('time since stim (ms)')
    
    def plot_change_img_box(self, ax, s, cate=None, roi_id=None):
        win_base = [-1000,0]
        offsets = [0.1, 0.2, 0.3, 0.4]
        colors = ['royalblue', 'mediumseagreen', 'violet', 'coral']
        lbl = ['img#1', 'img#2', 'img#3', 'img#4']
        idx_pre = np.diff(self.stim_labels[:,2]<-1, append=0)
        idx_pre[idx_pre==-1] = 0
        idx_pre = idx_pre.astype('bool')
        idx_post = self.stim_labels[:,2]<-1
        if cate != None:
            neu_cate = self.neu_seq[:,self.labels==cate,:]
            _, color1, color2, _ = get_roi_label_color([cate], 0)
        if roi_id != None:
            neu_cate = np.expand_dims(self.neu_seq[:,roi_id,:], axis=1)
            _, color1, color2, _ = get_roi_label_color(self.labels, roi_id)
        neu = []
        for i in [2,3,4,5]:
            i_post = idx_post * (self.stim_labels[:,2]==-i)
            neu.append(neu_cate[i_post,:,:])
        for i in range(4):
            self.plot_win_mag_box(
                ax, neu[i], self.neu_time, win_base, colors[i], 0, offsets[i])
            ax.plot([], color=colors[i], label=lbl[i])
        ax.legend(loc='upper right')
        
    def plot_change_prepost(self, ax, s, cate=None, roi_id=None):
        idx_pre = np.diff(self.stim_labels[:,2]<-1, append=0)
        idx_pre[idx_pre==-1] = 0
        idx_pre = idx_pre.astype('bool')
        idx_post = self.stim_labels[:,2]<-1
        if cate != None:
            neu_cate = self.neu_seq[:,self.labels==cate,:]
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
    
    def plot_change_prepost_box(self, ax, s, cate=None, roi_id=None):
        win_base = [-1000,0]
        lbl = ['pre', 'post']
        offsets = [0.2, 0.4]
        idx_pre = np.diff(self.stim_labels[:,2]<-1, append=0)
        idx_pre[idx_pre==-1] = 0
        idx_pre = idx_pre.astype('bool')
        idx_post = self.stim_labels[:,2]<-1
        if cate != None:
            neu_cate = self.neu_seq[:,self.labels==cate,:]
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

        
# fig, ax = plt.subplots(1, 1, figsize=(6, 6))
# colors = ['#989A9C', '#A4CB9E', '#9DB4CE', '#EDA1A4', '#F9C08A']
class plotter_VIPTD_G8_align_stim(plotter_utils):
    def __init__(self, neural_trials, labels, significance):
        super().__init__(neural_trials, labels, significance)
        
    # excitory response to normal stimulus.
    def normal_exc(self, ax):
        self.plot_normal(ax, self.significance['r_normal'], cate=-1)
        ax.set_title('excitory response to normal stim')
        
    # inhibitory response to normal stimulus.
    def normal_inh(self, ax):
        self.plot_normal(ax, self.significance['r_normal'], cate=1)
        ax.set_title('inhibitory response to normal stim')

    # excitory response to normal stimulus quantification.
    def normal_exc_box(self, ax):
        self.plot_normal_box(ax, self.significance['r_normal'], cate=-1)
        ax.set_title('excitory response to normal stim')
        
    # inhibitory response to normal stimulus quantification.
    def normal_inh_box(self, ax):
        self.plot_normal_box(ax, self.significance['r_normal'], cate=1)
        ax.set_title('inhibitory response to normal stim')

    # excitory response to normal stimulus with epoch.
    def normal_epoch_exc(self, ax):
        colors = ['#969696', '#6BAED6', '#2171B5', '#504099']
        self.plot_normal_epoch(ax, colors, self.significance['r_normal'], cate=-1)
        ax.set_title('excitory response to normal stimulus with epoch')
    
    # inhibitory response to normal stimulus with epoch.
    def normal_epoch_inh(self, ax):
        colors = ['#969696', '#FE7BBF', '#974EC3', '#504099']
        self.plot_normal_epoch(ax, colors, self.significance['r_normal'], cate=1)
        ax.set_title('inhibitory response to normal stimulus with epoch')    

    # excitory response to normal stimulus with quantification.
    def normal_epoch_exc_box(self, ax):
        colors = ['#969696', '#6BAED6', '#2171B5', '#504099']
        self.plot_normal_epoch_box(ax, colors, self.significance['r_normal'], cate=-1)
        ax.set_title('excitory response to normal stimulus with epoch')
    
    # inhibitory response to normal stimulus with epoch quantification.
    def normal_epoch_inh_box(self, ax):
        colors = ['#969696', '#FE7BBF', '#974EC3', '#504099']
        self.plot_normal_epoch_box(ax, colors, self.significance['r_normal'], cate=1)
        ax.set_title('inhibitory response to normal stimulus with epoch')   

    # response to normal stimulus heatmap average across trials.
    def normal_heatmap_neuron(self, ax):
        self.plot_heatmap_neuron(ax, self.neu_seq, self.neu_time, self.neu_seq)
        ax.set_xlabel('time since stim (ms)')
        ax.set_title('response to normal stim (all)')

    # excitory mean response to image change for different images.
    def change_img_exc(self, ax):
        self.plot_change_img(ax, self.significance['r_change'], cate=-1)
        ax.set_title('excitory response to image change (post)')
    
    # inhibitory mean response to image change for different images.
    def change_img_inh(self, ax):
        self.plot_change_img(ax, self.significance['r_change'], cate=1)
        ax.set_title('inhibitory response to image change (post)')
        
    # excitory mean response to image change for different images quantification.
    def change_img_exc_box(self, ax):
        self.plot_change_img_box(ax, self.significance['r_change'], cate=-1)
        ax.set_title('excitory response to image change (post)')
    
    # inhibitory mean response to image change for different images quantification.
    def change_img_inh_box(self, ax):
        self.plot_change_img_box(ax, self.significance['r_change'], cate=1)
        ax.set_title('inhibitory response to image change (post)')
        
    # excitory mean response to image change for all images.
    def change_prepost_exc(self, ax):
        self.plot_change_prepost(ax, self.significance['r_change'], cate=-1)
        ax.set_title('excitory response to pre & post img change')
    
    # inhibitory mean response to image change for all images.
    def change_prepost_inh(self, ax):
        self.plot_change_prepost(ax, self.significance['r_change'], cate=1)
        ax.set_title('inhibitory response to pre & post img change')
    
    # excitory mean response to image change for all images quantification.
    def change_prepost_exc_box(self, ax):
        self.plot_change_prepost_box(ax, self.significance['r_change'], cate=-1)
        ax.set_title('excitory response to pre & post img change')
    
    # inhibitory mean response to image change for all images quantification.
    def change_prepost_inh_box(self, ax):
        self.plot_change_prepost_box(ax, self.significance['r_change'], cate=1)
        ax.set_title('inhibitory response to pre & post img change')
    
    # mean response to image change heatmap average across trial.
    def change_heatmap_neuron(self, ax):
        neu_seq = self.neu_seq[self.stim_labels[:,2]<-1,:,:]
        self.plot_heatmap_neuron(ax, neu_seq, self.neu_time, neu_seq)
        ax.set_title('response to image change (all post)')
            