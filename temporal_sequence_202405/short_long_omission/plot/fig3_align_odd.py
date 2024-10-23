#!/usr/bin/env python3

import numpy as np

from modules.Alignment import get_stim_response
from plot.utils import get_mean_sem
from plot.utils import get_roi_label_color
from plot.utils import get_epoch_idx
from plot.utils import get_epoch_response
from plot.utils import get_odd_stim_prepost_idx
from plot.utils import get_expect_time
from plot.utils import pick_trial
from plot.utils import adjust_layout_neu
from plot.utils import utils


class plotter_utils(utils):
    
    def __init__(
            self,
            neural_trials, labels, significance
            ):
        self.l_frames = 200
        self.r_frames = 200
        self.cut_frames = 50
        self.stim_labels = neural_trials['stim_labels'][1:-1,:]
        self.labels = labels
        [self.epoch_short, self.epoch_long] = get_epoch_idx(self.stim_labels)
        [self.neu_seq, self.neu_time,
         self.stim_seq, self.stim_value, self.stim_time, _, self.pre_isi] = get_stim_response(
            neural_trials, self.l_frames, self.r_frames)
        [self.expect_short, self.expect_long] = get_expect_time(
            self.stim_labels)
        [self.idx_pre_short, self.idx_pre_long,
         self.idx_post_short, self.idx_post_long] = get_odd_stim_prepost_idx(
            self.stim_labels)
        self.significance = significance
        
    def plot_odd_normal_pre(self, ax, cate=None, roi_id=None):
        if cate != None:
            neu_cate = self.neu_seq[:,(self.labels==cate)*self.significance['r_oddball'],:]
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
    
    def plot_odd_normal_post(self, ax, cate=None, roi_id=None):
        if cate != None:
            neu_cate = self.neu_seq[:,(self.labels==cate)*self.significance['r_oddball'],:]
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
    
    def plot_odd_epoch_post(self, ax, epoch, idx, colors, cate=None, roi_id=None):
        lbl = ['ep11', 'ep12', 'ep21', 'ep22']
        if cate != None:
            neu_cate = self.neu_seq[:,(self.labels==cate)*self.significance['r_oddball'],:]
        if roi_id != None:
            neu_cate = np.expand_dims(self.neu_seq[:,roi_id,:], axis=1)
        [stim_seq_epoch, expect_epoch, neu_epoch] = get_epoch_response(
            neu_cate, self.stim_seq, self.stim_labels, epoch, idx)
        neu_mean = []
        neu_sem = []
        for i in range(4):
            m, s = get_mean_sem(neu_epoch[i])
            neu_mean.append(m)
            neu_sem.append(s)
        upper = np.max(neu_mean) + np.max(neu_sem)
        lower = np.min(neu_mean) - np.max(neu_sem)
        ax.fill_between(
            stim_seq_epoch[0][1,:],
            lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
            color='gold', alpha=0.15, step='mid')
        for i in range(4):
            '''
            ax.axvline(
                -expect_epoch[i]+stim_seq_epoch[i][1,1],
                color=colors[i],
                lw=1, linestyle='--')
            '''
            ax.fill_between(
                stim_seq_epoch[i][0,:],
                lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
                color=colors[i], alpha=0.15, step='mid')
            self.plot_mean_sem(
                ax, self.neu_time, neu_mean[i], neu_sem[i],
                colors[i], lbl[i])
        ax.plot([], lw=1, color='grey', linestyle='--', label='omi')
        adjust_layout_neu(ax)
        ax.set_xlim([self.neu_time[0],
                     self.neu_time[np.argmin(np.abs(self.neu_time))+self.cut_frames]])
        ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
        ax.set_xlabel('time since post omi stim (ms)')
        ax.legend(loc='upper left')

    def plot_odd_normal_shortlong(self, ax, idx_pre, idx_post, cate=None, roi_id=None):
        if cate != None:
            neu_cate = self.neu_seq[:,(self.labels==cate)*self.significance['r_oddball'],:]
            _, color1, color2, _ = get_roi_label_color([cate], 0)
        if roi_id != None:
            neu_cate = np.expand_dims(self.neu_seq[:,roi_id,:], axis=1)
            _, color1, color2, _ = get_roi_label_color(self.labels, roi_id)
        stim_seq_pre = np.mean(self.stim_seq[idx_pre,:,:], axis=0)
        neu_pre  = neu_cate[idx_pre,:,:]
        neu_post = neu_cate[idx_post,:,:]
        mean_pre  = np.mean(neu_pre.reshape(-1, self.l_frames+self.r_frames), axis=0)
        mean_post = np.mean(neu_post.reshape(-1, self.l_frames+self.r_frames), axis=0)
        sem_pre  = sem(neu_pre.reshape(-1, self.l_frames+self.r_frames), axis=0)
        sem_post = sem(neu_post.reshape(-1, self.l_frames+self.r_frames), axis=0)
        upper = np.max([mean_pre, mean_post]) + np.max([sem_pre, sem_post])
        lower = np.min([mean_pre, mean_post]) - np.max([sem_pre, sem_post])
        ax.fill_between(
            stim_seq_pre[1,:],
            lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
            color='gold', alpha=0.15, step='mid', label='stim')
        self.plot_mean_sem(ax, self.neu_time, mean_pre, sem_pre, color1, 'pre')
        self.plot_mean_sem(ax, self.neu_time, mean_post, sem_post, color2, 'post')
        adjust_layout_neu(ax)
        ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
        ax.set_xlabel('time since stim (ms)')
    
    def plot_omi_normal_box(self, ax, cate=None, roi_id=None):
        lbl = ['pre omi stim','omi','post omi stim']
        color_short = ['black', 'grey', 'silver']
        color_long = ['darkgreen', 'mediumseagreen', 'mediumspringgreen']
        if cate != None:
            neu_cate = self.neu_seq[:,self.labels==cate,:]
        if roi_id != None:
            neu_cate = np.expand_dims(self.neu_seq[:,roi_id,:], axis=1)
        idx_short = (self.stim_labels[:,2]==-1) * (self.stim_labels[:,3]==0)
        idx_long  = (self.stim_labels[:,2]==-1) * (self.stim_labels[:,3]==1)
        neu_short = neu_cate[idx_short,:,:]
        neu_long  = neu_cate[idx_long,:,:]
        stim_seq_short = np.mean(self.stim_seq[idx_short,:,:], axis=0)
        stim_seq_long = np.mean(self.stim_seq[idx_long,:,:], axis=0)
        self.plot_win_mean_sem_box(
            ax, [neu_short]*3, color_short, lbl,
            [0, self.expect_short, stim_seq_short[1,0]], 0.04)
        self.plot_win_mean_sem_box(
            ax, [neu_long]*3, color_long, lbl,
            [0, self.expect_long, stim_seq_long[1,0]], -0.04)
        stages = ['baseline', 'early', 'late']
        for i in range(3):
            ax.plot([], c=color_short[i], label=stages[i]+' short')
            ax.plot([], c=color_long[i], label=stages[i]+' long')
        ax.legend(loc='upper right')
    
    def plot_omi_post_isi(
            self, ax,
            idx_post, bins, isi_labels, colors,
            cate=None, roi_id=None
            ):
        if cate != None:
            neu_cate = self.neu_seq[:,self.labels==cate,:]
            _, _, color, _ = get_roi_label_color([cate], 0)
        if roi_id != None:
            neu_cate = np.expand_dims(self.neu_seq[:,roi_id,:], axis=1)
            _, _, color, _ = get_roi_label_color(self.labels, roi_id)
        bin_idx = np.digitize(self.pre_isi, bins)-1
        stim_seq_post = np.mean(self.stim_seq[idx_post,:,:], axis=0)
        m = []
        s = []
        for i in range(len(isi_labels)):
            if len(np.where(bin_idx==i)[0])>0:
                bin_idx_post = np.zeros_like(bin_idx, dtype=bool)
                bin_idx_post[1:] = (bin_idx==i)[:-1]
                isi_idx = bin_idx_post*idx_post
                neu = neu_cate[isi_idx,:,:].reshape(-1, self.l_frames+self.r_frames)
                self.plot_mean_sem(
                    ax, self.neu_time, np.mean(neu, axis=0), sem(neu, axis=0),
                    colors[i], isi_labels[i])
                m.append(np.mean(neu, axis=0))
                s.append(sem(neu, axis=0))
        m = np.concatenate(m)
        s = np.concatenate(s)
        m = m[~np.isnan(m)]
        s = s[~np.isnan(s)]
        upper = np.max(m) + np.max(s)
        lower = np.min(m) - np.max(s)
        ax.fill_between(
            stim_seq_post[1,:],
            lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
            color='grey', alpha=0.15, step='mid', label='post')
        adjust_layout_neu(ax)
        ax.set_xlabel('time since post omi stim (ms)')
        ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
    
    def plot_omi_post_isi_box(self, ax, idx_post, bins, isi_labels, cate=None, roi_id=None):
        colors = ['black', 'grey', 'silver']
        if cate != None:
            neu_cate = self.neu_seq[:,self.labels==cate,:]
            _, _, color, _ = get_roi_label_color([cate], 0)
        if roi_id != None:
            neu_cate = np.expand_dims(self.neu_seq[:,roi_id,:], axis=1)
            _, _, color, _ = get_roi_label_color(self.labels, roi_id)
        bin_idx = np.digitize(self.pre_isi, bins)-1
        neu = []
        for i in range(len(isi_labels)):
            if len(np.where(bin_idx==i)[0])>0:
                bin_idx_post = np.zeros_like(bin_idx, dtype=bool)
                bin_idx_post[1:] = (bin_idx==i)[:-1]
                isi_idx = bin_idx_post*idx_post
                neu.append(neu_cate[isi_idx,:,:].reshape(
                    -1, self.l_frames+self.r_frames))
        self.plot_win_mean_sem_box(ax, neu, colors, isi_labels, [0]*4, 0.04)
                
    def plot_omi_isi_box(self, ax, idx, isi, c_time, offset, lbl, cate=None, roi_id=None):
        l_time = 0
        r_time = 300
        time_range = [200,2000]
        bin_size = 200
        l_idx, r_idx = get_frame_idx_from_time(
            self.neu_time, c_time, l_time, r_time)
        if cate != None:
            neu_cate = self.neu_seq[:,self.labels==cate,:]
            _, _, color, _ = get_roi_label_color([cate], 0)
        if roi_id != None:
            neu_cate = np.expand_dims(self.neu_seq[:,roi_id,:], axis=1)
            _, _, color, _ = get_roi_label_color(self.labels, roi_id)
        neu_cate = neu_cate[idx,:,:]
        mag = np.mean(neu_cate[:,:, l_idx:r_idx], axis=2).reshape(-1)
        bins, bin_mean, bin_sem = get_bin_stat(mag, isi, time_range, bin_size)
        ax.errorbar(
            bins[:-1] + (bins[1]-bins[0]) / 2 - offset,
            bin_mean,
            bin_sem,
            color=color, capsize=2, marker='o',
            markeredgecolor='white', markeredgewidth=1, label=lbl)
        ax.tick_params(tick1On=False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xlabel('preceeding isi (ms)')
        ax.set_ylabel('df/f at omission with {}ms window (mean$\pm$sem)'.format(
            r_time-l_time))
        ax.legend(loc='upper right')
        
    def plot_omi_context(self, ax, img_id, cate=None, roi_id=None):
        if cate != None:
            neu_cate = self.neu_seq[:,self.labels==cate,:]
            _, color1, color2, _ = get_roi_label_color([cate], 0)
        if roi_id != None:
            neu_cate = np.expand_dims(self.neu_seq[:,roi_id,:], axis=1)
            _, color1, color2, _ = get_roi_label_color(self.labels, roi_id) 
        n_s = neu_cate[self.idx_post_short*(self.stim_labels[:,2]==img_id),:,:].reshape(
            -1, self.l_frames+self.r_frames)
        n_l = neu_cate[self.idx_post_long*(self.stim_labels[:,2]==img_id),:,:].reshape(
            -1, self.l_frames+self.r_frames)
        mean_short = np.mean(n_s, axis=0)
        mean_long  = np.mean(n_l, axis=0)
        sem_short  = sem(n_s, axis=0)
        sem_long   = sem(n_l, axis=0)
        upper = np.max([mean_short, mean_long]) + np.max([sem_short, sem_long])
        lower = np.min([mean_short, mean_long]) - np.max([sem_short, sem_long])
        stim_seq_short = np.mean(self.stim_seq[self.idx_post_short,:,:], axis=0)
        stim_seq_long = np.mean(self.stim_seq[self.idx_post_long,:,:], axis=0)
        ax.fill_between(
            stim_seq_short[1,:],
            lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
            color='gold', alpha=0.15, step='mid', label='post')
        ax.axvline(
            -self.expect_short+stim_seq_short[1,1],
            color=color1, label='omi (short)',
            lw=1, linestyle='--')
        ax.axvline(
            -self.expect_long+stim_seq_short[1,1],
            color=color2, label='omi (long)',
            lw=1, linestyle='--')
        ax.fill_between(
            stim_seq_short[0,:],
            lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
            color=color1, alpha=0.15, step='mid', label='pre (short)')
        ax.fill_between(
            stim_seq_long[0,:],
            lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
            color=color2, alpha=0.15, step='mid', label='pre (long)')
        self.plot_mean_sem(ax, self.neu_time, mean_short, sem_short, color1, 'short')
        self.plot_mean_sem(ax, self.neu_time, mean_long, sem_long, color2, 'long')
        adjust_layout_neu(ax)
        ax.set_xlim([self.neu_time[0],
                     self.neu_time[np.argmin(np.abs(self.neu_time))+self.cut_frames]])
        ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
        ax.set_xlabel('time since post omi stim (ms)')

    def plot_omi_context_box(self, ax, cate=None, roi_id=None):
        if cate != None:
            neu_cate = self.neu_seq[:,self.labels==cate,:]
        if roi_id != None:
            neu_cate = np.expand_dims(self.neu_seq[:,roi_id,:], axis=1)
        lbl = ['img#1', 'img#2', 'img#3', 'img#4']
        color_short = ['black', 'grey', 'silver']
        color_long = ['darkgreen', 'mediumseagreen', 'mediumspringgreen']
        neu_short = []
        neu_long = []
        for img_id in [2,3,4,5]:
            n_s = neu_cate[self.idx_post_short*(self.stim_labels[:,2]==img_id),:,:].reshape(
                -1, self.l_frames+self.r_frames)
            n_l = neu_cate[self.idx_post_long*(self.stim_labels[:,2]==img_id),:,:].reshape(
                -1, self.l_frames+self.r_frames)
            neu_short.append(n_s)
            neu_long.append(n_l)
        self.plot_win_mean_sem_box(ax, neu_short, color_short, lbl, [0]*4, 0.04)
        self.plot_win_mean_sem_box(ax, neu_long, color_long, lbl, [0]*4, -0.04)
        stages = ['baseline', 'early', 'late']
        for i in range(3):
            ax.plot([], c=color_short[i], label=stages[i]+' short')
            ax.plot([], c=color_long[i], label=stages[i]+' long')
        ax.legend(loc='upper right')

    
    
    def plot_omi_epoch_post_box(self, ax, cate=None, roi_id=None):
        if cate != None:
            neu_cate = self.neu_seq[:,self.labels==cate,:]
        if roi_id != None:
            neu_cate = np.expand_dims(self.neu_seq[:,roi_id,:], axis=1)
        lbl = ['ep11', 'ep12', 'ep21', 'ep22']
        color_short = ['black', 'grey', 'silver']
        color_long = ['darkgreen', 'mediumseagreen', 'mediumspringgreen']
        [_, _, neu_short] = get_epoch_response(
            neu_cate, self.stim_seq, self.stim_labels,
            self.epoch_short, self.idx_post_short)
        [_, _, neu_long] = get_epoch_response(
            neu_cate, self.stim_seq, self.stim_labels,
            self.epoch_long, self.idx_post_long)
        neu_short = [n_s.reshape(-1, self.l_frames+self.r_frames) for n_s in neu_short]
        neu_long = [n_l.reshape(-1, self.l_frames+self.r_frames) for n_l in neu_long]
        self.plot_win_mean_sem_box(ax, neu_short, color_short, lbl, [0]*4, 0.04)
        self.plot_win_mean_sem_box(ax, neu_long, color_long, lbl, [0]*4, -0.04)
        stages = ['baseline', 'early', 'late']
        for i in range(3):
            ax.plot([], c=color_short[i], label=stages[i]+' short')
            ax.plot([], c=color_long[i], label=stages[i]+' long')
        ax.legend(loc='upper right')

    # roi omission response comparison between pre and post for short. 
    def roi_omi_normal_short(self, ax, roi_id):
        self.plot_omi_normal_shortlong(
            ax, self.idx_pre_short, self.idx_post_short, roi_id=roi_id)
        ax.set_title('mean stim response around omi comparison (short)')
    
    # roi omission response comparison between pre and post for long. 
    def roi_omi_normal_long(self, ax, roi_id):
        self.plot_omi_normal_shortlong(
            ax, self.idx_pre_long, self.idx_post_long, roi_id=roi_id)
        ax.set_title('mean stim response around omi comparison (long)')
    
    # roi response quantification.
    def roi_omi_normal_box(self, ax, roi_id):
        self.plot_omi_normal_box(ax, roi_id=roi_id)
        ax.set_title('mean response comparison')
    
    # roi omission response and isi before omission for short.
    def roi_omi_isi_short(self, ax, roi_id):
        isi = np.tile(self.pre_isi[self.idx_pre_short], 1)
        self.plot_omi_isi_box(ax, self.idx_pre_short, isi, self.expect_short, 0, roi_id=roi_id)
        ax.set_title('omission response and preceeding isi (short)')
    
    # roi omission response and isi before omission for long.
    def roi_omi_isi_long(self, ax, roi_id):
        isi = np.tile(self.pre_isi[self.idx_pre_long], 1)
        self.plot_omi_isi_box(ax, self.idx_pre_long, isi, self.expect_long, 0, roi_id=roi_id)
        ax.set_title('omission response and preceeding isi (long)')
    
    # roi post omission stimulus response and isi before omission for short.
    def roi_omi_isi_post_short(self, ax, roi_id):
        idx = np.diff(self.idx_post_short, append=0)
        idx[idx==-1] = 0
        idx = idx.astype('bool')
        isi = np.tile(self.pre_isi[idx], 1)
        self.plot_omi_isi_box(ax, self.idx_post_short, isi, 0, 0, roi_id=roi_id)
        ax.set_title('post omi stim response and preceeding isi (short)')
    
    # roi post omission stimulus response and isi before omission for long.
    def roi_omi_isi_post_long(self, ax, roi_id):
        idx = np.diff(self.idx_post_long, append=0)
        idx[idx==-1] = 0
        idx = idx.astype('bool')
        isi = np.tile(self.pre_isi[idx], 1)
        self.plot_omi_isi_box(ax, self.idx_post_long, isi, 0, 0, roi_id=roi_id)
        ax.set_title('post omi stim response and preceeding isi (long)')
    
    # response to short long context.
    def roi_omi_context(self, axs, roi_id):
        titles = [
            'post omi stim response around img#1',
            'post omi stim response around img#2',
            'post omi stim response around img#3',
            'post omi stim response around img#4']
        for i in range(4):
            self.plot_omi_context(axs[i], i+2, roi_id=roi_id)
            axs[i].set_title(titles[i])
    
    # roi response to short long context quantification.
    def roi_omi_context_box(self, ax, roi_id):
        self.plot_omi_context_box(ax, roi_id=roi_id)
        ax.set_title('post omi stim response around different image')
        
        
class plotter_VIPTD_G8_align_odd(plotter_utils):
    def __init__(self, neural_trials, labels, significance):
        super().__init__(neural_trials, labels, significance)
    
    def all_odd_normal_exc(self, axs):
        self.odd_normal_pre_exc(axs[0])
        self.odd_normal_post_exc(axs[1])
        self.odd_epoch_post_exc_short(axs[3])
        self.odd_epoch_post_exc_long(axs[4])
    
    def all_odd_normal_inh(self, axs):
        self.odd_normal_pre_inh(axs[0])
        self.odd_normal_post_inh(axs[1])
        self.odd_epoch_post_inh_short(axs[3])
        self.odd_epoch_post_inh_long(axs[4])
        
    # excitory response to pre omission stimulus.
    def odd_normal_pre_exc(self, ax):
        self.plot_odd_normal_pre(ax, cate=-1)
        ax.set_title('exitory response to pre omi stim')
    
    # inhibitory response to pre omission stimulus.
    def odd_normal_pre_inh(self, ax):
        self.plot_odd_normal_pre(ax, cate=1)
        ax.set_title('inhibitory response to pre omi stim')
        
    # excitory response to post omission stimulus.
    def odd_normal_post_exc(self, ax):
        self.plot_odd_normal_post(ax, cate=-1)
        ax.set_title('excitory response to post omi stim')
    
    # inhibitory response to post omission stimulus.
    def odd_normal_post_inh(self, ax):
        self.plot_odd_normal_post(ax, cate=1)
        ax.set_title('inhibitory response to post omi stim')
    
    # response to post omission stimulus heatmap average across trial for short sorted by short.
    def odd_normal_post_short_short_heatmap_neuron(self, ax):
        idx_short = self.idx_post_short * pick_trial(self.stim_labels, None, [0], None, None, [0])
        neu_short = self.neu_seq[idx_short,:,:].copy()
        self.plot_heatmap_neuron(ax, neu_short, self.neu_time, neu_short)
        ax.set_xlabel('time since post omi stim (ms)')
        ax.set_title('response to post omi stim (short sorted by short)')
    
    # response to post omission stimulus heatmap average across trial for short sorted by long.
    def odd_normal_post_short_long_heatmap_neuron(self, ax):
        idx_short = self.idx_post_short * pick_trial(self.stim_labels, None, [0], None, None, [0])
        idx_long  = self.idx_post_long  * pick_trial(self.stim_labels, None, [1], None, None, [0])
        neu_short = self.neu_seq[idx_short,:,:].copy()
        neu_long  = self.neu_seq[idx_long,:,:].copy()
        self.plot_heatmap_neuron(ax, neu_short, self.neu_time, neu_long)
        ax.set_xlabel('time since post omi stim (ms)')
        ax.set_title('response to post omi stim (short sorted by long)')
    
    # response to post omission stimulus heatmap average across trial for long sorted by short.
    def odd_normal_post_long_short_heatmap_neuron(self, ax):
        idx_short = self.idx_post_short * pick_trial(self.stim_labels, None, [0], None, None, [0])
        idx_long  = self.idx_post_long  * pick_trial(self.stim_labels, None, [1], None, None, [0])
        neu_short = self.neu_seq[idx_short,:,:].copy()
        neu_long  = self.neu_seq[idx_long,:,:].copy()
        self.plot_heatmap_neuron(ax, neu_long, self.neu_time, neu_short)
        ax.set_xlabel('time since post omi stim (ms)')
        ax.set_title('response to post omi stim (long sorted by short)')
    
    # response to post omission stimulus heatmap average across trial for long sorted by long.
    def odd_normal_post_long_long_heatmap_neuron(self, ax):
        idx_long  = self.idx_post_long  * pick_trial(self.stim_labels, None, [1], None, None, [0])
        neu_long  = self.neu_seq[idx_long,:,:].copy()
        self.plot_heatmap_neuron(ax, neu_long, self.neu_time, neu_long)
        ax.set_xlabel('time since post omi stim (ms)')
        ax.set_title('response to post omi stim (long sorted by long)')
    
    # response to post omission stimulus average across trial for short.
    def omi_normal_post_short_heatmap_neuron(self, ax):
        idx_post = np.diff(self.stim_labels[:,2]==-1, prepend=0)
        idx_post[idx_post==1] = 0
        idx_post[idx_post==-1] = 1
        idx_post = idx_post.astype('bool')
        idx = idx_post * (self.stim_labels[:,3]==0)
        neu_seq = self.neu_seq[idx,:,:]
        self.plot_heatmap_neuron(ax, neu_seq)
        ax.set_title('mean response to post omi stim (short)')
    
    # response to post omission stimulus average across trial for long.
    def omi_normal_post_long_heatmap_neuron(self, ax):
        idx_post = np.diff(self.stim_labels[:,2]==-1, prepend=0)
        idx_post[idx_post==1] = 0
        idx_post[idx_post==-1] = 1
        idx_post = idx_post.astype('bool')
        idx = idx_post * (self.stim_labels[:,3]==1)
        neu_seq = self.neu_seq[idx,:,:]
        self.plot_heatmap_neuron(ax, neu_seq)
        ax.set_title('mean response to post omi stim (long)')

    # excitory response to post omi stim across epoch for short.
    def odd_epoch_post_exc_short(self, ax):
        colors = ['#969696', '#6BAED6', '#2171B5', '#504099']
        self.plot_odd_epoch_post(ax, self.epoch_short, self.idx_post_short, colors, cate=-1)
        ax.set_title(
            'excitory response across epoch (short)')
    
    # excitory response to post omi stim across epoch for long.
    def odd_epoch_post_exc_long(self, ax):
        colors = ['#969696', '#6BAED6', '#2171B5', '#504099']
        self.plot_odd_epoch_post(ax, self.epoch_long, self.idx_post_long, colors, cate=-1)
        ax.set_title('excitory response across epoch (long)')
    
    # inhibitory response to post omi stim across epoch for short.
    def odd_epoch_post_inh_short(self, ax):
        colors = ['#969696', '#FE7BBF', '#974EC3', '#504099']
        self.plot_odd_epoch_post(ax, self.epoch_short, self.idx_post_short, colors, cate=1)
        ax.set_title('inhibitory response across epoch (short)')
    
    # inhibitory response to post omi stim across epoch for long.
    def odd_epoch_post_inh_long(self, ax):
        colors = ['#969696', '#FE7BBF', '#974EC3', '#504099']
        self.plot_odd_epoch_post(ax, self.epoch_long, self.idx_post_long, colors, cate=1)
        ax.set_title('inhibitory response across epoch (long)')
        
        
        
    # excitory omission response comparison between pre and post for short. 
    def odd_normal_exc_short(self, ax):
        self.plot_odd_normal_shortlong(
            ax, self.idx_pre_short, self.idx_post_short, cate=-1)
        ax.set_title('mean stim response around omi comparison (short)')
    
    # excitory omission response comparison between pre and post for long. 
    def odd_normal_exc_long(self, ax):
        self.plot_odd_normal_shortlong(
            ax, self.idx_pre_long, self.idx_post_long, cate=-1)
        ax.set_title('mean stim response around omi comparison (long)')
    
    # inhibitory omission response comparison between pre and post for short. 
    def odd_normal_inh_short(self, ax):
        self.plot_odd_normal_shortlong(
            ax, self.idx_pre_short, self.idx_post_short, cate=1)
        ax.set_title('mean stim response around omi comparison (short)')
    
    # inhibitory omission response comparison between pre and post for long. 
    def odd_normal_inh_long(self, ax):
        self.plot_odd_normal_shortlong(
            ax, self.idx_pre_long, self.idx_post_long, cate=1)
        ax.set_title('mean stim response around omi comparison (long)')
        
    # excitory response quantification.
    def omi_normal_exc_box(self, ax):
        self.plot_omi_normal_box(ax, cate=-1)
        ax.set_title('excitory response comparison')
    
    # inhibitory response quantification.
    def omi_normal_inh_box(self, ax):
        self.plot_omi_normal_box(ax, cate=1)
        ax.set_title('inhibitory response comparison')
    



    
    # excitory response to post omi stim across epoch quantification.
    def omi_epoch_post_exc_box(self, ax):
        self.plot_omi_epoch_post_box(ax, cate=-1)
        ax.set_title('excitory response on post omi stim across epoch')
    
    # inhibitory response to post omi stim across epoch quantification.
    def omi_epoch_post_inh_box(self, ax):
        self.plot_omi_epoch_post_box(ax, cate=1)
        ax.set_title('inhibitory response on post omi stim across epoch')
    
    # excitory response to post omi stim single trial heatmap for short.
    def omi_post_exc_short_heatmap_trials(self, ax):
        neu_cate = self.neu_seq[:,self.labels==-1,:]
        neu_cate = neu_cate[self.idx_post_short,:,:]
        _, _, _, cmap = get_roi_label_color([-1], 0)
        self.plot_heatmap_trials(ax, neu_cate, cmap)
        ax.set_xlabel('time since post omi stim (ms)')
        ax.set_title('excitory single trial response to post omi stim (short)')
    
    # excitory response to post omi stim single trial heatmap for short.
    def omi_post_exc_long_heatmap_trials(self, ax):
        neu_cate = self.neu_seq[:,self.labels==-1,:]
        neu_cate = neu_cate[self.idx_post_long,:,:]
        _, _, _, cmap = get_roi_label_color([-1], 0)
        self.plot_heatmap_trials(ax, neu_cate, cmap)
        ax.set_xlabel('time since post omi stim (ms)')
        ax.set_title('excitory single trial response to post omi stim (long)')
        
    # inhibitory response to post omi stim single trial heatmap for short.
    def omi_post_inh_short_heatmap_trials(self, ax):
        neu_cate = self.neu_seq[:,self.labels==1,:]
        neu_cate = neu_cate[self.idx_post_short,:,:]
        _, _, _, cmap = get_roi_label_color([1], 0)
        self.plot_heatmap_trials(ax, neu_cate, cmap)
        ax.set_xlabel('time since post omi stim (ms)')
        ax.set_title('inhibitory single trial response to post omi stim (short)')
    
    # inhibitory response to post omi stim single trial heatmap for short.
    def omi_post_inh_long_heatmap_trials(self, ax):
        neu_cate = self.neu_seq[:,self.labels==1,:]
        neu_cate = neu_cate[self.idx_post_long,:,:]
        _, _, _, cmap = get_roi_label_color([1], 0)
        self.plot_heatmap_trials(ax, neu_cate, cmap)
        ax.set_xlabel('time since post omi stim (ms)')
        ax.set_title('inhibitory single trial response to post omi stim (long)')
        
    # omission response and isi before omission for short.
    def omi_isi_short(self, ax):
        isi_exc = np.tile(self.pre_isi[self.idx_pre_short], np.sum(self.labels==-1))
        isi_inh = np.tile(self.pre_isi[self.idx_pre_short], np.sum(self.labels==1))
        self.plot_omi_isi_box(ax, self.idx_pre_short, isi_exc, self.expect_short, -0.02,
                          'excitory', cate=-1)
        self.plot_omi_isi_box(ax, self.idx_pre_short, isi_inh, self.expect_short, 0.02,
                          'inhibitory', cate=1)
        ax.set_title('omission response and preceeding isi (short)')
        
    # omission response and isi before omission for long.
    def omi_isi_long(self, ax):
        isi_exc = np.tile(self.pre_isi[self.idx_pre_long], np.sum(self.labels==-1))
        isi_inh = np.tile(self.pre_isi[self.idx_pre_long], np.sum(self.labels==1))
        self.plot_omi_isi_box(ax, self.idx_pre_long, isi_exc, self.expect_long, -0.02,
                          'excitory', cate=-1)
        self.plot_omi_isi_box(ax, self.idx_pre_long, isi_inh, self.expect_long, 0.02,
                          'inhibitory', cate=1)
        ax.set_title('omission response and preceeding isi (long)')

    # excitory response to short long context.
    def omi_context_exc(self, axs):
        titles = [
            'excitory post omi stim response around img#1',
            'excitory post omi stim response around img#2',
            'excitory post omi stim response around img#3',
            'excitory post omi stim response around img#4']
        for i in range(4):
            self.plot_omi_context(axs[i], i+2, cate=-1)
            axs[i].set_title(titles[i])
    
    # inhibitory response to short long context.
    def omi_context_inh(self, axs):
        titles = [
            'inhibitory post omi stim response around img#1',
            'inhibitory post omi stim response around img#2',
            'inhibitory post omi stim response around img#3',
            'inhibitory post omi stim response around img#4']
        for i in range(4):
            self.plot_omi_context(axs[i], i+2, cate=1)
            axs[i].set_title(titles[i])
    
    # excitory response to short long context quantification.
    def omi_context_exc_box(self, ax):
        self.plot_omi_context_box(ax, cate=-1)
        ax.set_title('excitory post omi stim response around different image')
    
    # inhibitory response to short long context quantification.
    def omi_context_inh_box(self, ax):
        self.plot_omi_context_box(ax, cate=1)
        ax.set_title('inhibitory post omi stim response around different image')
    
    # excitory response to post omi stim with proceeding isi for short.
    def omi_post_isi_exc_short(self, ax):
        bins = [0, 600, 750, 900, 1500]
        isi_labels = ['0-400', '400-800', '800-1200', '1200-1600']
        colors = ['#969696', '#6BAED6', '#2171B5', '#504099']
        self.plot_omi_post_isi(
            ax, self.idx_post_short,
            bins, isi_labels, colors, cate=-1)
        ax.set_title('excitory post omi stim response and proceeding isi (short)')
    
    # excitory response to post omi stim with proceeding isi for short quantification.
    def omi_post_isi_exc_short_box(self, ax):
        bins = [0, 600, 750, 900, 1500]
        isi_labels = ['0-400', '400-800', '800-1200', '1200-1600']
        self.plot_omi_post_isi_box(ax, self.idx_post_short, bins, isi_labels, cate=-1)
        ax.set_title('excitory post omi stim response and proceeding isi (short)')
            
    # excitory response to post omi stim with proceeding isi for long.
    def omi_post_isi_exc_long(self, ax):
        bins = [0, 600, 1200, 1800, 2400]
        isi_labels = ['0-600', '600-1200', '1200-1800', '1800-2400']
        colors = ['#969696', '#6BAED6', '#2171B5', '#504099']
        self.plot_omi_post_isi(
            ax, self.idx_post_long,
            bins, isi_labels, colors, cate=-1)
        ax.set_title('excitory post omi stim response and proceeding isi (long)')
    
    # excitory response to post omi stim with proceeding isi for long quantification.
    def omi_post_isi_exc_long_box(self, ax):
        bins = [0, 600, 1200, 1800, 2400]
        isi_labels = ['0-600', '600-1200', '1200-1800', '1800-2400']
        self.plot_omi_post_isi_box(ax, self.idx_post_long, bins, isi_labels, cate=-1)
        ax.set_title('excitory post omi stim response and proceeding isi (long)')
    
    # inhibitory response to post omi stim with proceeding isi for short.
    def omi_post_isi_inh_short(self, ax):
        bins = [0, 600, 750, 900, 1500]
        isi_labels = ['0-400', '400-800', '800-1200', '1200-1600']
        colors = ['#969696', '#FE7BBF', '#974EC3', '#504099']
        self.plot_omi_post_isi(
            ax, self.idx_post_short,
            bins, isi_labels, colors, cate=1)
        ax.set_title('inhibitory post omi stim response and proceeding isi (short)')
    
    # inhibitory response to post omi stim with proceeding isi for short quantification.
    def omi_post_isi_inh_short_box(self, ax):
        bins = [0, 600, 750, 900, 1500]
        isi_labels = ['0-400', '400-800', '800-1200', '1200-1600']
        self.plot_omi_post_isi_box(ax, self.idx_post_short, bins, isi_labels, cate=1)
        ax.set_title('inhibitory post omi stim response and proceeding isi (short)')
            
    # inhibitory response to post omi stim with proceeding isi for long.
    def omi_post_isi_inh_long(self, ax):
        bins = [0, 600, 1200, 1800, 2400]
        isi_labels = ['0-600', '600-1200', '1200-1800', '1800-2400']
        colors = ['#969696', '#FE7BBF', '#974EC3', '#504099']
        self.plot_omi_post_isi(
            ax, self.idx_post_long,
            bins, isi_labels, colors, cate=1)
        ax.set_title('inhibitory post omi stim response and proceeding isi (long)')
    
    # inhibitory response to post omi stim with proceeding isi for long quantification.
    def omi_post_isi_inh_long_box(self, ax):
        bins = [0, 600, 1200, 1800, 2400]
        isi_labels = ['0-600', '600-1200', '1200-1800', '1800-2400']
        self.plot_omi_post_isi_box(ax, self.idx_post_long, bins, isi_labels, cate=1)
        ax.set_title('inhibitory post omi stim response and proceeding isi (long)')


class plotter_L7G8_align_odd(plotter_utils):
    def __init__(self, neural_trials, labels, significance):
        super().__init__(neural_trials, labels, significance)
    
    def all_odd_normal(self, axs):
        self.odd_normal_pre(axs[0])
        self.odd_normal_post(axs[1])
        self.odd_epoch_post_short(axs[3])
        self.odd_epoch_post_long(axs[4])
    
    # response to pre omission stimulus.
    def odd_normal_pre(self, ax):
        self.plot_odd_normal_pre(ax, cate=-1)
        ax.set_title('exitory response to pre omi stim')
    
    # response to post omission stimulus.
    def odd_normal_post(self, ax):
        self.plot_odd_normal_post(ax, cate=-1)
        ax.set_title('response to post omi stim')

    # response to post omission stimulus heatmap average across trial for short sorted by short.
    def odd_normal_post_short_short_heatmap_neuron(self, ax):
        idx_short = self.idx_post_short * pick_trial(self.stim_labels, None, [0], None, None, [0])
        neu_short = self.neu_seq[idx_short,:,:].copy()
        self.plot_heatmap_neuron(ax, neu_short, self.neu_time, neu_short)
        ax.set_xlabel('time since post omi stim (ms)')
        ax.set_title('response to post omi stim (short sorted by short)')
    
    # response to post omission stimulus heatmap average across trial for short sorted by long.
    def odd_normal_post_short_long_heatmap_neuron(self, ax):
        idx_short = self.idx_post_short * pick_trial(self.stim_labels, None, [0], None, None, [0])
        idx_long  = self.idx_post_long  * pick_trial(self.stim_labels, None, [1], None, None, [0])
        neu_short = self.neu_seq[idx_short,:,:].copy()
        neu_long  = self.neu_seq[idx_long,:,:].copy()
        self.plot_heatmap_neuron(ax, neu_short, self.neu_time, neu_long)
        ax.set_xlabel('time since post omi stim (ms)')
        ax.set_title('response to post omi stim (short sorted by long)')
    
    # response to post omission stimulus heatmap average across trial for long sorted by short.
    def odd_normal_post_long_short_heatmap_neuron(self, ax):
        idx_short = self.idx_post_short * pick_trial(self.stim_labels, None, [0], None, None, [0])
        idx_long  = self.idx_post_long  * pick_trial(self.stim_labels, None, [1], None, None, [0])
        neu_short = self.neu_seq[idx_short,:,:].copy()
        neu_long  = self.neu_seq[idx_long,:,:].copy()
        self.plot_heatmap_neuron(ax, neu_long, self.neu_time, neu_short)
        ax.set_xlabel('time since post omi stim (ms)')
        ax.set_title('response to post omi stim (long sorted by short)')
    
    # response to post omission stimulus heatmap average across trial for long sorted by long.
    def odd_normal_post_long_long_heatmap_neuron(self, ax):
        idx_long  = self.idx_post_long  * pick_trial(self.stim_labels, None, [1], None, None, [0])
        neu_long  = self.neu_seq[idx_long,:,:].copy()
        self.plot_heatmap_neuron(ax, neu_long, self.neu_time, neu_long)
        ax.set_xlabel('time since post omi stim (ms)')
        ax.set_title('response to post omi stim (long sorted by long)')
    
    # response to post omission stimulus average across trial for short.
    def omi_normal_post_short_heatmap_neuron(self, ax):
        idx_post = np.diff(self.stim_labels[:,2]==-1, prepend=0)
        idx_post[idx_post==1] = 0
        idx_post[idx_post==-1] = 1
        idx_post = idx_post.astype('bool')
        idx = idx_post * (self.stim_labels[:,3]==0)
        neu_seq = self.neu_seq[idx,:,:]
        self.plot_heatmap_neuron(ax, neu_seq)
        ax.set_title('mean response to post omi stim (short)')
    
    # response to post omission stimulus average across trial for long.
    def omi_normal_post_long_heatmap_neuron(self, ax):
        idx_post = np.diff(self.stim_labels[:,2]==-1, prepend=0)
        idx_post[idx_post==1] = 0
        idx_post[idx_post==-1] = 1
        idx_post = idx_post.astype('bool')
        idx = idx_post * (self.stim_labels[:,3]==1)
        neu_seq = self.neu_seq[idx,:,:]
        self.plot_heatmap_neuron(ax, neu_seq)
        ax.set_title('mean response to post omi stim (long)')

    # response to post omi stim across epoch for short.
    def odd_epoch_post_short(self, ax):
        colors = ['#969696', '#6BAED6', '#2171B5', '#504099']
        self.plot_odd_epoch_post(ax, self.epoch_short, self.idx_post_short, colors, cate=-1)
        ax.set_title(
            'response across epoch (short)')
    
    # response to post omi stim across epoch for long.
    def odd_epoch_post_long(self, ax):
        colors = ['#969696', '#6BAED6', '#2171B5', '#504099']
        self.plot_odd_epoch_post(ax, self.epoch_long, self.idx_post_long, colors, cate=-1)
        ax.set_title('response across epoch (long)')
    
    # omission response comparison between pre and post for short. 
    def odd_normal_short(self, ax):
        self.plot_odd_normal_shortlong(
            ax, self.idx_pre_short, self.idx_post_short, cate=-1)
        ax.set_title('mean stim response around omi comparison (short)')
    
    # omission response comparison between pre and post for long. 
    def odd_normal_long(self, ax):
        self.plot_odd_normal_shortlong(
            ax, self.idx_pre_long, self.idx_post_long, cate=-1)
        ax.set_title('mean stim response around omi comparison (long)')
    
    # response quantification.
    def omi_normal_box(self, ax):
        self.plot_omi_normal_box(ax, cate=-1)
        ax.set_title('response comparison')
    
    # response to post omi stim across epoch quantification.
    def omi_epoch_post_box(self, ax):
        self.plot_omi_epoch_post_box(ax, cate=-1)
        ax.set_title('response on post omi stim across epoch')
    
    # response to post omi stim single trial heatmap for short.
    def omi_post_short_heatmap_trials(self, ax):
        neu_cate = self.neu_seq[:,self.labels==-1,:]
        neu_cate = neu_cate[self.idx_post_short,:,:]
        _, _, _, cmap = get_roi_label_color([-1], 0)
        self.plot_heatmap_trials(ax, neu_cate, cmap)
        ax.set_xlabel('time since post omi stim (ms)')
        ax.set_title('single trial response to post omi stim (short)')
    
    # response to post omi stim single trial heatmap for short.
    def omi_post_long_heatmap_trials(self, ax):
        neu_cate = self.neu_seq[:,self.labels==-1,:]
        neu_cate = neu_cate[self.idx_post_long,:,:]
        _, _, _, cmap = get_roi_label_color([-1], 0)
        self.plot_heatmap_trials(ax, neu_cate, cmap)
        ax.set_xlabel('time since post omi stim (ms)')
        ax.set_title('single trial response to post omi stim (long)')
        
    # response to short long context.
    def omi_context(self, axs):
        titles = [
            'post omi stim response around img#1',
            'post omi stim response around img#2',
            'post omi stim response around img#3',
            'post omi stim response around img#4']
        for i in range(4):
            self.plot_omi_context(axs[i], i+2, cate=-1)
            axs[i].set_title(titles[i])
    
    # response to short long context quantification.
    def omi_context_box(self, ax):
        self.plot_omi_context_box(ax, cate=-1)
        ax.set_title('post omi stim response around different image')
    
    # response to post omi stim with proceeding isi for short.
    def omi_post_isi_short(self, ax):
        bins = [0, 600, 750, 900, 1500]
        isi_labels = ['0-400', '400-800', '800-1200', '1200-1600']
        colors = ['#969696', '#6BAED6', '#2171B5', '#504099']
        self.plot_omi_post_isi(
            ax, self.idx_post_short,
            bins, isi_labels, colors, cate=-1)
        ax.set_title('post omi stim response and proceeding isi (short)')
    
    # response to post omi stim with proceeding isi for short quantification.
    def omi_post_isi_short_box(self, ax):
        bins = [0, 600, 750, 900, 1500]
        isi_labels = ['0-400', '400-800', '800-1200', '1200-1600']
        self.plot_omi_post_isi_box(ax, self.idx_post_short, bins, isi_labels, cate=-1)
        ax.set_title('post omi stim response and proceeding isi (short)')
            
    # response to post omi stim with proceeding isi for long.
    def omi_post_isi_long(self, ax):
        bins = [0, 600, 1200, 1800, 2400]
        isi_labels = ['0-600', '600-1200', '1200-1800', '1800-2400']
        colors = ['#969696', '#6BAED6', '#2171B5', '#504099']
        self.plot_omi_post_isi(
            ax, self.idx_post_long,
            bins, isi_labels, colors, cate=-1)
        ax.set_title('post omi stim response and proceeding isi (long)')
    
    # response to post omi stim with proceeding isi for long quantification.
    def omi_post_isi_long_box(self, ax):
        bins = [0, 600, 1200, 1800, 2400]
        isi_labels = ['0-600', '600-1200', '1200-1800', '1800-2400']
        self.plot_omi_post_isi_box(ax, self.idx_post_long, bins, isi_labels, cate=-1)
        ax.set_title('post omi stim response and proceeding isi (long)')