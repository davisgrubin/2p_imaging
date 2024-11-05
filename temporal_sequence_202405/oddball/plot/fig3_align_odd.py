#!/usr/bin/env python3

import numpy as np

from modules.Alignment import run_get_stim_response
from plot.utils import get_mean_sem
from plot.utils import get_multi_sess_neu_trial_average
from plot.utils import get_roi_label_color
from plot.utils import get_epoch_idx
from plot.utils import get_odd_stim_prepost_idx
from plot.utils import get_expect_time
from plot.utils import pick_trial
from plot.utils import adjust_layout_neu
from plot.utils import utils


class plotter_utils(utils):

    def __init__(
            self,
            list_neural_trials, list_labels, list_significance
            ):
        timescale = 1.0
        self.n_sess = len(list_neural_trials)
        self.l_frames = int(250*timescale)
        self.r_frames = int(250*timescale)
        self.cut_frames = int(100*timescale)
        self.list_stim_labels = [
            nt['stim_labels'][1:-1,:] for nt in list_neural_trials]
        self.list_labels = list_labels
        [self.list_neu_seq, self.list_neu_time,
         self.list_stim_seq, self.list_stim_value, self.list_stim_time,
         _, self.list_pre_isi] = run_get_stim_response(
                list_neural_trials, self.l_frames, self.r_frames)
        self.list_epoch = [
            get_epoch_idx(sl) for sl in self.list_stim_labels]
        self.epoch_short = [[self.list_epoch[n][0][e] for n in range(self.n_sess)] for e in range(4)]
        self.epoch_long  = [[self.list_epoch[n][1][e] for n in range(self.n_sess)] for e in range(4)]
        self.list_odd_idx = [
            get_odd_stim_prepost_idx(sl) for sl in self.list_stim_labels]
        self.list_significance = list_significance

    def plot_odd_pre(self, ax, cate=None, roi_id=None):
        idx_cut = np.searchsorted(self.list_neu_time[0], 0)-self.cut_frames
        if cate != None:
            neu_cate = [
                self.list_neu_seq[i][:,(self.list_labels[i]==cate)*self.list_significance[i]['r_oddball'],:]
                for i in range(self.n_sess)]
            _, color1, color2, _ = get_roi_label_color([cate], 0)
        if roi_id != None:
            neu_cate = [np.expand_dims(self.list_neu_seq[0][:,roi_id,:], axis=1)]
            _, color1, color2, _ = get_roi_label_color(self.labels, roi_id)
        neu_short, _ = get_multi_sess_neu_trial_average(
            None, neu_cate, trial_idx=[l[0] for l in self.list_odd_idx])
        neu_long, _ = get_multi_sess_neu_trial_average(
            None, neu_cate, trial_idx=[l[1] for l in self.list_odd_idx])
        mean_short, sem_short = get_mean_sem(neu_short[:,idx_cut:])
        mean_long,  sem_long  = get_mean_sem(neu_long[:,idx_cut:])
        stim_seq_short = np.mean(self.list_stim_seq[0][self.list_odd_idx[0][0],:,:], axis=0)
        stim_seq_long  = np.mean(self.list_stim_seq[0][self.list_odd_idx[0][1],:,:], axis=0)
        stim_value_short = self.list_stim_value[0][self.list_odd_idx[0][0],:]
        stim_value_long  = self.list_stim_value[0][self.list_odd_idx[0][1],:]
        upper = np.nanmax([mean_short, mean_long]) + np.nanmax([sem_short, sem_long])
        lower = np.nanmin([mean_short, mean_long]) - np.nanmax([sem_short, sem_long])
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
        self.plot_vol(ax, self.list_stim_time[0], stim_value_short, color1, upper, lower)
        self.plot_vol(ax, self.list_stim_time[0], stim_value_long,  color2, upper, lower)
        self.plot_mean_sem(ax, self.list_neu_time[0][idx_cut:], mean_short, sem_short, color1, 'short')
        self.plot_mean_sem(ax, self.list_neu_time[0][idx_cut:], mean_long,  sem_long,  color2, 'long')
        adjust_layout_neu(ax)
        ax.set_xlim([self.list_neu_time[0][idx_cut], self.list_neu_time[0][-1]])
        ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
        ax.set_xlabel('time since pre oddball stim (ms)')
        ax.legend(loc='upper right')

    def plot_odd_post(self, ax, cate=None, roi_id=None):
        idx_cut = np.searchsorted(self.list_neu_time[0], 0)+self.cut_frames
        if cate != None:
            neu_cate = [
                self.list_neu_seq[i][:,(self.list_labels[i]==cate)*self.list_significance[i]['r_oddball'],:]
                for i in range(self.n_sess)]
            _, color1, color2, _ = get_roi_label_color([cate], 0)
        if roi_id != None:
            neu_cate = [np.expand_dims(self.list_neu_seq[0][:,roi_id,:], axis=1)]
            _, color1, color2, _ = get_roi_label_color(self.labels, roi_id)
        neu_short, _ = get_multi_sess_neu_trial_average(
            None, neu_cate, trial_idx=[l[2] for l in self.list_odd_idx])
        neu_long, _ = get_multi_sess_neu_trial_average(
            None, neu_cate, trial_idx=[l[3] for l in self.list_odd_idx])
        mean_short, sem_short = get_mean_sem(neu_short[:,:idx_cut])
        mean_long,  sem_long  = get_mean_sem(neu_long[:,:idx_cut])
        stim_seq_short = np.mean(self.list_stim_seq[0][self.list_odd_idx[0][2],:,:], axis=0)
        stim_seq_long  = np.mean(self.list_stim_seq[0][self.list_odd_idx[0][3],:,:], axis=0)
        stim_value_short = self.list_stim_value[0][self.list_odd_idx[0][2],:]
        stim_value_long  = self.list_stim_value[0][self.list_odd_idx[0][3],:]
        upper = np.nanmax([mean_short, mean_long]) + np.nanmax([sem_short, sem_long])
        lower = np.nanmin([mean_short, mean_long]) - np.nanmax([sem_short, sem_long])
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
        self.plot_vol(ax, self.list_stim_time[0], stim_value_short, color1, upper, lower)
        self.plot_vol(ax, self.list_stim_time[0], stim_value_long,  color2, upper, lower)
        self.plot_mean_sem(ax, self.list_neu_time[0][:idx_cut], mean_short, sem_short, color1, 'short')
        self.plot_mean_sem(ax, self.list_neu_time[0][:idx_cut], mean_long,  sem_long,  color2, 'long')
        adjust_layout_neu(ax)
        ax.set_xlim([self.list_neu_time[0][0], self.list_neu_time[0][idx_cut]])
        ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
        ax.set_xlabel('time since post oddball stim (ms)')
        ax.legend(loc='upper left')

    def plot_odd_post_box(self, ax, cate=None, roi_id=None):
        win_base = [-1800,-200]
        lbl = ['short', 'long']
        offsets = [0.0, 0.1]
        if cate != None:
            neu_cate = [
                self.list_neu_seq[i][:,(self.list_labels[i]==cate)*self.list_significance[i]['r_oddball'],:]
                for i in range(self.n_sess)]
            _, color1, color2, _ = get_roi_label_color([cate], 0)
        if roi_id != None:
            neu_cate = [np.expand_dims(self.list_neu_seq[0][:,roi_id,:], axis=1)]
            _, color1, color2, _ = get_roi_label_color(self.labels, roi_id)
        colors = [color1, color2]
        neu = [
            get_multi_sess_neu_trial_average(
                None, neu_cate, trial_idx=[l[2] for l in self.list_odd_idx])[0],
            get_multi_sess_neu_trial_average(
                None, neu_cate, trial_idx=[l[3] for l in self.list_odd_idx])[0]]
        for i in range(2):
            self.plot_win_mag_box(
                    ax, neu[i], self.list_neu_time[0], win_base, colors[i], 0, offsets[i])
            ax.plot([], color=colors[i], label=lbl[i])
        ax.legend(loc='upper right')

    def plot_odd_normal_prepost(self, ax, normal, color, cate=None, roi_id=None):
        idx_cut_l = np.searchsorted(self.list_neu_time[0], 0)-self.cut_frames
        idx_cut_r = np.searchsorted(self.list_neu_time[0], 0)+self.cut_frames
        if cate != None:
            neu_cate = [
                self.list_neu_seq[i][:,(self.list_labels[i]==cate)*self.list_significance[i]['r_oddball'],:]
                for i in range(self.n_sess)]
        if roi_id != None:
            neu_cate = [np.expand_dims(self.list_neu_seq[0][:,roi_id,:], axis=1)]
        neu_pre, _ = get_multi_sess_neu_trial_average(
            None, neu_cate, trial_idx=[l[normal] for l in self.list_odd_idx])
        neu_post, _ = get_multi_sess_neu_trial_average(
            None, neu_cate, trial_idx=[l[normal+2] for l in self.list_odd_idx])
        mean_pre,  sem_pre  = get_mean_sem(neu_pre[:,idx_cut_l:idx_cut_r])
        mean_post, sem_post = get_mean_sem(neu_post[:,idx_cut_l:idx_cut_r])
        stim_seq = np.mean(self.list_stim_seq[0][self.list_odd_idx[0][normal],:,:], axis=0)
        stim_value = self.list_stim_value[0][self.list_odd_idx[0][normal],:]
        upper = np.nanmax([mean_pre, mean_post]) + np.nanmax([sem_pre, sem_post])
        lower = np.nanmin([mean_pre, mean_post]) - np.nanmax([sem_pre, sem_post])
        ax.fill_between(
            (stim_seq[1,:] + stim_seq[1,:])/2,
            lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
            color='gold', alpha=0.15, step='mid', label='stim')
        self.plot_vol(ax, self.list_stim_time[0], stim_value, 'gold', upper, lower)
        self.plot_mean_sem(ax, self.list_neu_time[0][idx_cut_l:idx_cut_r], mean_pre,  sem_pre,  color, 'pre',  0.5)
        self.plot_mean_sem(ax, self.list_neu_time[0][idx_cut_l:idx_cut_r], mean_post, sem_post, color, 'post', 1.0)
        adjust_layout_neu(ax)
        ax.set_xlim([self.list_neu_time[0][idx_cut_l], self.list_neu_time[0][idx_cut_r]])
        ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
        ax.set_xlabel('time since stim (ms)')
        ax.legend(loc='upper left')

    def plot_odd_context_post(self, ax, img_id, cate=None, roi_id=None):
        idx_cut = np.searchsorted(self.list_neu_time[0], 0)+self.cut_frames
        if cate != None:
            neu_cate = [
                self.list_neu_seq[i][:,(self.list_labels[i]==cate)*self.list_significance[i]['r_oddball'],:]
                for i in range(self.n_sess)]
            _, color1, color2, _ = get_roi_label_color([cate], 0)
        if roi_id != None:
            neu_cate = [np.expand_dims(self.list_neu_seq[0][:,roi_id,:], axis=1)]
            _, color1, color2, _ = get_roi_label_color(self.labels, roi_id)
        neu_short, _ = get_multi_sess_neu_trial_average(
            self.list_stim_labels, neu_cate,
            trial_idx=[l[2] for l in self.list_odd_idx],
            trial_param=[img_id, [0], None, None, [0]])
        neu_long, _ = get_multi_sess_neu_trial_average(
            self.list_stim_labels, neu_cate,
            trial_idx=[l[3] for l in self.list_odd_idx],
            trial_param=[img_id, [1], None, None, [0]])
        mean_short, sem_short = get_mean_sem(neu_short[:,:idx_cut])
        mean_long,  sem_long  = get_mean_sem(neu_long[:,:idx_cut])
        stim_seq_short = np.mean(self.list_stim_seq[0][self.list_odd_idx[0][2],:,:], axis=0)
        stim_seq_long  = np.mean(self.list_stim_seq[0][self.list_odd_idx[0][3],:,:], axis=0)
        stim_value_short = self.list_stim_value[0][self.list_odd_idx[0][2],:]
        stim_value_long  = self.list_stim_value[0][self.list_odd_idx[0][3],:]
        upper = np.nanmax([mean_short, mean_long]) + np.nanmax([sem_short, sem_long])
        lower = np.nanmin([mean_short, mean_long]) - np.nanmax([sem_short, sem_long])
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
        ax.axvline(
            -self.expect[0],
            color=color1, lw=1, linestyle='--')
        ax.axvline(
            -self.expect[1],
            color=color2, lw=1, linestyle='--')
        self.plot_vol(ax, self.list_stim_time[0], stim_value_short, color1, upper, lower)
        self.plot_vol(ax, self.list_stim_time[0], stim_value_long,  color2, upper, lower)
        self.plot_mean_sem(ax, self.list_neu_time[0][:idx_cut], mean_short, sem_short, color1, 'short')
        self.plot_mean_sem(ax, self.list_neu_time[0][:idx_cut], mean_long,  sem_long,  color2, 'long')
        adjust_layout_neu(ax)
        ax.set_xlim([self.list_neu_time[0][0], self.list_neu_time[0][idx_cut]])
        ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
        ax.set_xlabel('time since post oddball stim (ms)')
        ax.legend(loc='upper left')
    
    def plot_odd_normal_context_post_box(self, ax, normal, cate=None, roi_id=None):
        win_base = [-1800,-200]
        colors = ['royalblue', 'mediumseagreen', 'violet', 'coral']
        lbl = ['img#1', 'img#2', 'img#3', 'img#4']
        offsets = [0.0, 0.1, 0.2, 0.3]
        if cate != None:
            neu_cate = [
                self.list_neu_seq[i][:,(self.list_labels[i]==cate)*self.list_significance[i]['r_oddball'],:]
                for i in range(self.n_sess)]
        if roi_id != None:
            neu_cate = [np.expand_dims(self.list_neu_seq[0][:,roi_id,:], axis=1)]
        neu = []
        for i in [2,3,4,5]:
            neu.append(get_multi_sess_neu_trial_average(
                self.list_stim_labels, neu_cate,
                trial_idx=[l[normal+2] for l in self.list_odd_idx],
                trial_param=[[i], [normal], None, None, [0]])[0])
        for i in range(4):
            self.plot_win_mag_box(
                ax, neu[i], self.list_neu_time[0], win_base, colors[i], 0, offsets[i])
            ax.plot([], color=colors[i], label=lbl[i])
        ax.legend(loc='upper right')

    # roi oddball response comparison between pre and post for short.
    def roi_omi_normal_short(self, ax, roi_id):
        self.plot_omi_normal_shortlong(
            ax, self.idx_pre_short, self.idx_post_short, roi_id=roi_id)
        ax.set_title('mean stim response around omi comparison (short)')

    # roi oddball response comparison between pre and post for long.
    def roi_omi_normal_long(self, ax, roi_id):
        self.plot_omi_normal_shortlong(
            ax, self.idx_pre_long, self.idx_post_long, roi_id=roi_id)
        ax.set_title('mean stim response around omi comparison (long)')

    # roi response quantification.
    def roi_omi_normal_box(self, ax, roi_id):
        self.plot_omi_normal_box(ax, roi_id=roi_id)
        ax.set_title('mean response comparison')

    # roi oddball response and isi before oddball for short.
    def roi_omi_isi_short(self, ax, roi_id):
        isi = np.tile(self.pre_isi[self.idx_pre_short], 1)
        self.plot_omi_isi_box(ax, self.idx_pre_short, isi, self.expect_short, 0, roi_id=roi_id)
        ax.set_title('oddball response and preceeding isi (short)')

    # roi oddball response and isi before oddball for long.
    def roi_omi_isi_long(self, ax, roi_id):
        isi = np.tile(self.pre_isi[self.idx_pre_long], 1)
        self.plot_omi_isi_box(ax, self.idx_pre_long, isi, self.expect_long, 0, roi_id=roi_id)
        ax.set_title('oddball response and preceeding isi (long)')

    # roi post oddball stimulus response and isi before oddball for short.
    def roi_omi_isi_post_short(self, ax, roi_id):
        idx = np.diff(self.idx_post_short, append=0)
        idx[idx==-1] = 0
        idx = idx.astype('bool')
        isi = np.tile(self.pre_isi[idx], 1)
        self.plot_omi_isi_box(ax, self.idx_post_short, isi, 0, 0, roi_id=roi_id)
        ax.set_title('post omi stim response and preceeding isi (short)')

    # roi post oddball stimulus response and isi before oddball for long.
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

    def odd_normal_exc(self, axs):
        _, color1, color2, _ = get_roi_label_color([-1], 0)
        # excitatory response to pre oddball stimulus.
        self.plot_odd_pre(axs[0], cate=-1)
        axs[0].set_title('excitatory response to pre odd stim')
        # excitatory response to post oddball stimulus.
        self.plot_odd_post(axs[1], cate=-1)
        axs[1].set_title('excitatory response to post odd stim')
        # excitatory response to post oddball stimulus quantification.
        self.plot_odd_post_box(axs[2], cate=-1)
        axs[2].set_title('excitatory response to post odd stim')
        # excitatory response to pre/post oddball stimulus (short).
        self.plot_odd_normal_prepost(axs[3], 0, color1, cate=-1)
        axs[3].set_title('excitatory comparison on pre/post odd stim (short)')
        # excitatory response to pre/post oddball stimulus (long).
        self.plot_odd_normal_prepost(axs[4], 1, color2, cate=-1)
        axs[4].set_title('excitatory comparison on pre/post odd stim (long)')

    def odd_normal_inh(self, axs):
        _, color1, color2, _ = get_roi_label_color([1], 0)
        # inhibitory response to pre oddball stimulus.
        self.plot_odd_pre(axs[0], cate=1)
        axs[0].set_title('inhibitory response to pre odd stim')
        # inhibitory response to post oddball stimulus.
        self.plot_odd_post(axs[1], cate=1)
        axs[1].set_title('inhibitory response to post odd stim')
        # inhibitory response to post oddball stimulus quantification.
        self.plot_odd_post_box(axs[2], cate=1)
        axs[2].set_title('inhibitory response to post odd stim')
        # inhibitory response to pre/post oddball stimulus (short).
        self.plot_odd_normal_prepost(axs[3], 0, color1, cate=1)
        axs[3].set_title('inhibitory comparison on pre/post odd stim (short)')
        # inhibitory response to pre/post oddball stimulus (long).
        self.plot_odd_normal_prepost(axs[4], 1, color2, cate=1)
        axs[4].set_title('inhibitory comparison on pre/post odd stim (long)')
    
    def odd_normal_post_heatmap_neuron(self, axs):
        sig = np.concatenate([self.list_significance[n]['r_oddball'] for n in range(self.n_sess)])
        labels = np.concatenate(self.list_labels)
        neu_short, _ = get_multi_sess_neu_trial_average(
            None, self.list_neu_seq, trial_idx=[l[2] for l in self.list_odd_idx])
        neu_long, _ = get_multi_sess_neu_trial_average(
            None, self.list_neu_seq, trial_idx=[l[3] for l in self.list_odd_idx])
        for i in range(4):
            axs[i].set_xlabel('time since post odd stim (ms)')
        # response to post oddball stimulus heatmap average across trials for short sorted by short.
        self.plot_heatmap_neuron(
            axs[0], neu_short, self.list_neu_time[0], neu_short,
            labels, sig)
        axs[0].set_title('response to post odd stim (short sorted by short)')
        # response to post oddball stimulus heatmap average across trials for short sorted by long.
        self.plot_heatmap_neuron(
            axs[1], neu_short, self.list_neu_time[0], neu_long,
            labels, sig)
        axs[1].set_title('response to post odd stim (short sorted by long)')
        # response to post oddball stimulus heatmap average across trials for long sorted by short.
        self.plot_heatmap_neuron(
            axs[2], neu_long, self.list_neu_time[0], neu_short,
            labels, sig)
        axs[2].set_title('response to post odd stim (long sorted by short)')
        # response to post oddball stimulus heatmap average across trials for long sorted by long.
        self.plot_heatmap_neuron(
            axs[3], neu_long, self.list_neu_time[0], neu_long,
            labels, sig)
        axs[3].set_title('response to post odd stim (long sorted by long)')

    def odd_context_exc(self, axs):
        # excitatory response to stimulus specific stimulus.
        titles = [
            'excitatory response to post odd stim (img#1)',
            'excitatory response to post odd stim (img#2)',
            'excitatory response to post odd stim (img#3)',
            'excitatory response to post odd stim (img#4)']
        img_id = [[2], [3], [4], [5]]
        for i in range(4):
            self.plot_odd_context_post(axs[i], img_id[i], cate=-1)
            axs[i].set_title(titles[i])
        # excitatory response to stimulus specific stimulus quantification (short).
        self.plot_odd_normal_context_post_box(axs[4], 0, cate=-1)
        axs[4].set_title('excitatory response to post odd stim (short)')
        # excitatory response to stimulus specific stimulus quantification (long).
        self.plot_odd_normal_context_post_box(axs[5], 1, cate=-1)
        axs[5].set_title('excitatory response to post odd stim (long)')
        
    def odd_context_inh(self, axs):
        # inhibitory response to stimulus specific stimulus.
        titles = [
            'inhibitory response to post odd stim (img#1)',
            'inhibitory response to post odd stim (img#2)',
            'inhibitory response to post odd stim (img#3)',
            'inhibitory response to post odd stim (img#4)']
        img_id = [[2], [3], [4], [5]]
        for i in range(4):
            self.plot_odd_context_post(axs[i], img_id[i], cate=1)
            axs[i].set_title(titles[i])
        # inhibitory response to stimulus specific stimulus quantification (short).
        self.plot_odd_normal_context_post_box(axs[4], 0, cate=1)
        axs[4].set_title('inhibitory response to post odd stim (short)')
        # inhibitory response to stimulus specific stimulus quantification (long).
        self.plot_odd_normal_context_post_box(axs[5], 1, cate=1)
        axs[5].set_title('inhibitory response to post odd stim (long)')
