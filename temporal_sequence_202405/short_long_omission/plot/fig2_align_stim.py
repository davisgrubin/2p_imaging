#!/usr/bin/env python3

import numpy as np
from sklearn.cluster import KMeans

from modules.Alignment import run_get_stim_response
from plot.utils import exclude_post_odd_stim
from plot.utils import get_mean_sem
from plot.utils import get_frame_idx_from_time
from plot.utils import get_multi_sess_neu_trial_average
from plot.utils import get_roi_label_color
from plot.utils import get_epoch_idx
from plot.utils import get_expect_time
from plot.utils import adjust_layout_neu
from plot.utils import utils

# fig, ax = plt.subplots(1, 1, figsize=(6, 6))

class plotter_utils(utils):

    def __init__(
            self,
            list_neural_trials, list_labels, list_significance
            ):
        super().__init__()
        timescale = 1.0
        self.n_sess = len(list_neural_trials)
        self.l_frames = int(80*timescale)
        self.r_frames = int(100*timescale)
        self.list_stim_labels = [
            nt['stim_labels'][1:-1,:] for nt in list_neural_trials]
        self.list_stim_labels = [
            exclude_post_odd_stim(sl) for sl in self.list_stim_labels]
        self.list_labels = list_labels
        self.expect = np.array([
            np.mean([get_expect_time(sl)[0] for sl in self.list_stim_labels]),
            np.mean([get_expect_time(sl)[1] for sl in self.list_stim_labels])])
        self.epoch_early = [get_epoch_idx(sl)[0] for sl in self.list_stim_labels]
        self.epoch_late  = [get_epoch_idx(sl)[1] for sl in self.list_stim_labels]
        self.alignment = run_get_stim_response(
                list_neural_trials, self.l_frames, self.r_frames, expected='none')
        self.alignment_local = run_get_stim_response(
                list_neural_trials, self.l_frames, self.r_frames, expected='local')
        self.list_significance = list_significance

    def plot_normal(self, ax, normal, fix_jitter, cate=None, roi_id=None):
        if cate != None:
            _, color1, color2, _ = get_roi_label_color([cate], 0)
            neu_cate = [
                self.alignment['list_neu_seq'][i][:,(self.list_labels[i]==cate)*self.list_significance[i]['r_normal'],:]
                for i in range(self.n_sess)]
        if roi_id != None:
            _, color1, color2, _ = get_roi_label_color(self.labels, roi_id)
            neu_cate = [np.expand_dims(self.alignment['list_neu_seq'][0][:,roi_id,:], axis=1)]
        # collect data.
        neu_mean = []
        neu_sem = []
        if 0 in normal:
            neu_short, _, stim_seq_short, stim_value_short, _ = get_multi_sess_neu_trial_average(
                self.list_stim_labels, neu_cate, self.alignment['list_stim_seq'],
                self.alignment['list_stim_value'], self.alignment['list_pre_isi'],
                trial_param=[[2,3,4,5], [0], [fix_jitter], None, [0]])
            mean_short, sem_short = get_mean_sem(neu_short)
            neu_mean.append(mean_short)
            neu_sem.append(sem_short)
        if 1 in normal:
            neu_long, _, stim_seq_long, stim_value_long, _ = get_multi_sess_neu_trial_average(
                self.list_stim_labels, neu_cate, self.alignment['list_stim_seq'],
                self.alignment['list_stim_value'], self.alignment['list_pre_isi'],
                trial_param=[[2,3,4,5], [1], [fix_jitter], None, [0]])
            mean_long, sem_long = get_mean_sem(neu_long)
            neu_mean.append(mean_long)
            neu_sem.append(sem_long)
        # compute bounds.
        upper = np.nanmax(neu_mean) + np.nanmax(neu_sem)
        lower = np.nanmin(neu_mean) - np.nanmax(neu_sem)
        # plot voltages.
        if 0 in normal:
            for i in range(3):
                ax.fill_between(
                    stim_seq_short[i,:],
                    lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
                    color=color1, alpha=0.15, step='mid')
            self.plot_vol(ax, self.alignment['stim_time'], stim_value_short.reshape(1,-1), color1, upper, lower)
        if 1 in normal:
            for i in range(3):
                ax.fill_between(
                    stim_seq_long[i,:],
                    lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
                    color=color2, alpha=0.15, step='mid')
            self.plot_vol(ax, self.alignment['stim_time'], stim_value_long.reshape(1,-1), color2, upper, lower)
        # plot neural traces.
        if 0 in normal:
            self.plot_mean_sem(ax, self.alignment['neu_time'], mean_short, sem_short, color1, 'short')
        if 1 in normal:
            self.plot_mean_sem(ax, self.alignment['neu_time'], mean_long,  sem_long,  color2, 'long')
        if not np.isnan(upper+lower):
            adjust_layout_neu(ax)
            ax.set_xlim([np.nanmin(self.alignment['neu_time']), np.nanmax(self.alignment['neu_time'])])
            ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
            ax.set_xlabel('time since stim (ms)')
    
    def plot_normal_box(self, ax, fix_jitter, cate=None, roi_id=None):
        win_base = [-1500,0]
        lbl = ['short', 'long']
        offsets = [0.0, 0.1]
        if cate != None:
            _, color1, color2, _ = get_roi_label_color([cate], 0)
            neu_cate = [
                self.alignment['list_neu_seq'][i][:,(self.list_labels[i]==cate)*self.list_significance[i]['r_normal'],:]
                for i in range(self.n_sess)]
        if roi_id != None:
            _, color1, color2, _ = get_roi_label_color(self.labels, roi_id)
            neu_cate = [np.expand_dims(self.alignment['list_neu_seq'][0][:,roi_id,:], axis=1)]
        colors = [color1, color2]
        neu = [
            get_multi_sess_neu_trial_average(
                self.list_stim_labels, neu_cate, self.alignment['list_stim_seq'],
                self.alignment['list_stim_value'], self.alignment['list_pre_isi'],
                trial_param=[[2,3,4,5], [0], [fix_jitter], None, [0]])[0],
            get_multi_sess_neu_trial_average(
                self.list_stim_labels, neu_cate, self.alignment['list_stim_seq'],
                self.alignment['list_stim_value'], self.alignment['list_pre_isi'],
                trial_param=[[2,3,4,5], [1], [fix_jitter], None, [0]])[0]]
        for i in range(2):
            self.plot_win_mag_box(
                    ax, neu[i], self.alignment['neu_time'], win_base, colors[i], 0, offsets[i])
            ax.plot([], color=colors[i], label=lbl[i])
        ax.legend(loc='upper right')

    def plot_normal_global(self, ax, normal, cate=None, roi_id=None):
        if cate != None:
            color0, color1, color2, _ = get_roi_label_color([cate], 0)
            neu_cate = [
                self.alignment['list_neu_seq'][i][:,(self.list_labels[i]==cate)*self.list_significance[i]['r_normal'],:]
                for i in range(self.n_sess)]
        if roi_id != None:
            color0, color1, color2, _ = get_roi_label_color([cate], 0)
            neu_cate = [np.expand_dims(self.alignment['list_neu_seq'][0][:,roi_id,:], axis=1)]
        colors = [color0, [color1, color2][normal]]
        # collect single trial data.
        neu_seq, stim_seq, stim_value, pre_isi = get_multi_sess_neu_trial_average(
            self.list_stim_labels, neu_cate, self.alignment['list_stim_seq'],
            self.alignment['list_stim_value'], self.alignment['list_pre_isi'],
            trial_param=[[2,3,4,5], [normal], [1], None, [0]],
            mean_sem=False)
        # compute bin indices.
        bin_idx_neg = [isi<self.expect[normal] for isi in pre_isi]
        bin_idx_pos = [isi>self.expect[normal] for isi in pre_isi]
        # collect data.
        neu_mean = []
        neu_sem = []
        stim_mean = []
        for bin_idx in [bin_idx_neg, bin_idx_pos]:
            neu = np.concatenate(
                [np.nanmean(neu_seq[n][bin_idx[n],:,:], axis=0)
                 for n in range(self.n_sess)], axis=0)
            m, s = get_mean_sem(neu)
            neu_mean.append(m)
            neu_sem.append(s)
            stim_mean.append(np.concatenate([stim_value[n][bin_idx[n],:] for n in range(self.n_sess)]))
        # compute bounds.
        upper = np.nanmax(neu_mean) + np.nanmax(neu_sem)
        lower = np.nanmin(neu_mean) - np.nanmax(neu_sem)
        # plot voltages.
        stim_seq = np.nanmean(np.concatenate(stim_seq, axis=0), axis=0)
        ax.fill_between(
            stim_seq[1,:],
            lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
            color=[color1, color2][normal], alpha=0.15, step='mid')
        self.plot_vol(ax, self.alignment['stim_time'], stim_mean[0], colors[0], upper, lower)
        self.plot_vol(ax, self.alignment['stim_time'], stim_mean[1], colors[1], upper, lower)
        # plot neural traces.
        self.plot_mean_sem(
            ax, self.alignment['neu_time'],
            neu_mean[0], neu_sem[0], colors[0],
            '< {} ms'.format(int(self.expect[normal])))
        self.plot_mean_sem(
            ax, self.alignment['neu_time'],
            neu_mean[1], neu_sem[1], colors[1],
            '> {} ms'.format(int(self.expect[normal])))
        if not np.isnan(upper+lower):
            adjust_layout_neu(ax)
            ax.set_xlabel('time since stim (ms)')
            ax.set_xlim([np.nanmin(self.alignment['neu_time']), np.nanmax(self.alignment['neu_time'])])
            ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
    
    def plot_normal_overlap(self, ax, cate=None, roi_id=None):
        if cate != None:
            color0, color1, color2, _ = get_roi_label_color([cate], 0)
            neu_cate = [
                self.alignment['list_neu_seq'][i][:,(self.list_labels[i]==cate)*self.list_significance[i]['r_normal'],:]
                for i in range(self.n_sess)]
        if roi_id != None:
            color0, color1, color2, _ = get_roi_label_color(self.labels, roi_id)
            neu_cate = [np.expand_dims(self.alignment['list_neu_seq'][0][:,roi_id,:], axis=1)]
        neu_mean = []
        neu_sem = []
        stim_mean = []
        # get short trials.
        neu_seq_short, stim_seq_short, stim_value_short, pre_isi_short = get_multi_sess_neu_trial_average(
            self.list_stim_labels, neu_cate, self.alignment['list_stim_seq'],
            self.alignment['list_stim_value'], self.alignment['list_pre_isi'],
            trial_param=[[2,3,4,5], [0], [1], None, [0]],
            mean_sem=False)
        bin_idx_short = [isi>self.expect[0] for isi in pre_isi_short]
        neu = np.concatenate(
            [np.nanmean(neu_seq_short[n][bin_idx_short[n],:,:], axis=0)
             for n in range(self.n_sess)], axis=0)
        m, s = get_mean_sem(neu)
        neu_mean.append(m)
        neu_sem.append(s)
        stim_mean.append(np.concatenate([stim_value_short[n][bin_idx_short[n],:] for n in range(self.n_sess)]))
        stim_seq_short = np.nanmean(np.concatenate(stim_seq_short, axis=0), axis=0)
        # get long trials.
        neu_seq_long, stim_seq_long, stim_value_long, pre_isi_long = get_multi_sess_neu_trial_average(
            self.list_stim_labels, neu_cate, self.alignment['list_stim_seq'],
            self.alignment['list_stim_value'], self.alignment['list_pre_isi'],
            trial_param=[[2,3,4,5], [1], [1], None, [0]],
            mean_sem=False)
        bin_idx_long = [isi<self.expect[1] for isi in pre_isi_long]
        neu = np.concatenate(
            [np.nanmean(neu_seq_long[n][bin_idx_long[n],:,:], axis=0)
             for n in range(self.n_sess)], axis=0)
        m, s = get_mean_sem(neu)
        neu_mean.append(m)
        neu_sem.append(s)
        stim_mean.append(np.concatenate([stim_value_long[n][bin_idx_long[n],:] for n in range(self.n_sess)]))
        stim_seq_long = np.nanmean(np.concatenate(stim_seq_long, axis=0), axis=0)
        # compute bounds.
        upper = np.nanmax(neu_mean) + np.nanmax(neu_sem)
        lower = np.nanmin(neu_mean) - np.nanmax(neu_sem)
        # plot voltages.
        ax.fill_between(
            stim_seq_short[1,:],
            lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
            color=color1, alpha=0.15, step='mid')
        ax.fill_between(
            stim_seq_long[1,:],
            lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
            color=color2, alpha=0.15, step='mid')
        self.plot_vol(ax, self.alignment['stim_time'], stim_mean[0], color1, upper, lower)
        self.plot_vol(ax, self.alignment['stim_time'], stim_mean[1], color2, upper, lower)
        # plot neural traces.
        self.plot_mean_sem(ax, self.alignment['neu_time'], neu_mean[0], neu_sem[0], color1, 'short')
        self.plot_mean_sem(ax, self.alignment['neu_time'], neu_mean[1], neu_sem[1], color2, 'long')
        if not np.isnan(upper+lower):
            adjust_layout_neu(ax)
            ax.set_xlabel('time since stim (ms)')
            ax.set_xlim([np.nanmin(self.alignment['neu_time']), np.nanmax(self.alignment['neu_time'])])
            ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
            ax.legend(loc='upper right')
    
    def plot_normal_local(self, ax, normal, cate=None, roi_id=None):
        if cate != None:
            color0, color1, color2, _ = get_roi_label_color([cate], 0)
            neu_cate = [
                self.alignment_local['list_neu_seq'][i][:,(self.list_labels[i]==cate)*self.list_significance[i]['r_normal'],:]
                for i in range(self.n_sess)]
        if roi_id != None:
            color0, color1, color2, _ = get_roi_label_color([cate], 0)
            neu_cate = [np.expand_dims(self.alignment_local['list_neu_seq'][0][:,roi_id,:], axis=1)]
        colors = [color0, [color1, color2][normal]]
        # collect single trial data.
        neu_seq, stim_seq, stim_value, pre_isi = get_multi_sess_neu_trial_average(
            self.list_stim_labels, neu_cate, self.alignment_local['list_stim_seq'],
            self.alignment_local['list_stim_value'], self.alignment_local['list_pre_isi'],
            trial_param=[[2,3,4,5], [normal], [1], None, [0]],
            mean_sem=False)
        # compute bin indices.
        bin_idx_neg = [isi<self.expect[normal] for isi in pre_isi]
        bin_idx_pos = [isi>self.expect[normal] for isi in pre_isi]
        # collect data.
        neu_mean = []
        neu_sem = []
        stim_mean = []
        for bin_idx in [bin_idx_neg, bin_idx_pos]:
            neu = np.concatenate(
                [np.nanmean(neu_seq[n][bin_idx[n],:,:], axis=0)
                 for n in range(self.n_sess)], axis=0)
            m, s = get_mean_sem(neu)
            neu_mean.append(m)
            neu_sem.append(s)
            stim_mean.append(np.concatenate([stim_value[n][bin_idx[n],:] for n in range(self.n_sess)]))
        # compute bounds.
        upper = np.nanmax(neu_mean) + np.nanmax(neu_sem)
        lower = np.nanmin(neu_mean) - np.nanmax(neu_sem)
        # plot voltages.
        self.plot_vol(ax, self.alignment_local['stim_time'], stim_mean[0], colors[0], upper, lower)
        self.plot_vol(ax, self.alignment_local['stim_time'], stim_mean[1], colors[1], upper, lower)
        ax.axvline(0, color='gold', lw=1, linestyle='--')
        # plot neural traces.
        self.plot_mean_sem(
            ax, self.alignment_local['neu_time'],
            neu_mean[0], neu_sem[0], colors[0],
            '< {} ms'.format(int(self.expect[normal])))
        self.plot_mean_sem(
            ax, self.alignment_local['neu_time'],
            neu_mean[1], neu_sem[1], colors[1],
            '> {} ms'.format(int(self.expect[normal])))
        if not np.isnan(upper+lower):
            adjust_layout_neu(ax)
            ax.set_xlabel('time since expected stim (ms)')
            ax.set_xlim([np.nanmin(self.alignment['neu_time']), np.nanmax(self.alignment['neu_time'])])
            ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
            ax.legend(loc='upper right')
        
    def plot_normal_epoch(self, ax, normal, fix_jitter, cate=None, roi_id=None):
        if cate != None:
            color0, color1, color2, _ = get_roi_label_color([cate], 0)
            neu_cate = [
                self.alignment['list_neu_seq'][i][:,(self.list_labels[i]==cate)*self.list_significance[i]['r_normal'],:]
                for i in range(self.n_sess)]
        if roi_id != None:
            color0, color1, color2, _ = get_roi_label_color(self.labels, roi_id)
            neu_cate = [np.expand_dims(self.alignment['list_neu_seq'][0][:,roi_id,:], axis=1)]
        colors = [color0, [color1, color2][normal]]
        # collect data.
        neu_early, _, stim_seq_early, stim_value_early, _ = get_multi_sess_neu_trial_average(
            self.list_stim_labels, neu_cate, self.alignment['list_stim_seq'],
            self.alignment['list_stim_value'], self.alignment['list_pre_isi'],
            trial_idx=self.epoch_early,
            trial_param=[[2,3,4,5], [normal], [fix_jitter], None, [0]])
        neu_late, _, stim_seq_late, stim_value_late, _ = get_multi_sess_neu_trial_average(
            self.list_stim_labels, neu_cate, self.alignment['list_stim_seq'],
            self.alignment['list_stim_value'], self.alignment['list_pre_isi'],
            trial_idx=self.epoch_late,
            trial_param=[[2,3,4,5], [normal], [fix_jitter], None, [0]])
        mean_early, sem_early = get_mean_sem(neu_early)
        mean_late,  sem_late  = get_mean_sem(neu_late)
        # compute bounds.
        upper = np.nanmax([mean_early, mean_late]) + np.nanmax([sem_early, sem_late])
        lower = np.nanmin([mean_early, mean_late]) - np.nanmax([sem_early, sem_late])
        # plot voltages.
        stim_seq = (stim_seq_early + stim_seq_late) / 2
        stim_value = (stim_value_early + stim_value_late) / 2
        for i in range(3):
            ax.fill_between(
                stim_seq[i,:],
                lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
                color=colors[1], alpha=0.15, step='mid')
        self.plot_vol(
            ax, self.alignment['stim_time'],
            stim_value.reshape(1,-1),
            colors[1], upper, lower)
        # plot neural traces.
        self.plot_mean_sem(ax, self.alignment['neu_time'], mean_early, sem_early, colors[0], 'ep1')
        self.plot_mean_sem(ax, self.alignment['neu_time'], mean_late,  sem_late,  colors[1], 'ep2')
        if not np.isnan(upper+lower):
            adjust_layout_neu(ax)
            ax.set_xlim([np.nanmin(self.alignment['neu_time']), np.nanmax(self.alignment['neu_time'])])
            ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
            ax.set_xlabel('time since stim (ms)')
    
    def plot_normal_cluster(self, axs, normal, fix_jitter, cate):
        n_clusters = 5
        win_sort = [-500,500]
        l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, win_sort[0], win_sort[1])
        _, color1, color2, cmap = get_roi_label_color([cate], 0)
        neu_cate = [
            self.alignment['list_neu_seq'][i][:,(self.list_labels[i]==cate)*self.list_significance[i]['r_normal'],:]
            for i in range(self.n_sess)]
        # collect data.
        neu, _, stim_seq, _, _ = get_multi_sess_neu_trial_average(
            self.list_stim_labels, neu_cate, self.alignment['list_stim_seq'],
            self.alignment['list_stim_value'], self.alignment['list_pre_isi'],
            trial_param=[[2,3,4,5], [normal], [fix_jitter], None, [0]])
        neu = neu[~np.isnan(np.sum(neu,axis=1)),:]
        # fit model.
        corr_matrix = np.corrcoef(neu)
        model = KMeans(n_clusters)
        cluster_id = model.fit_predict(corr_matrix)
        # collect neural traces based on clusters and sorted by peak timing.
        neu_mean = np.zeros((n_clusters, len(self.alignment['neu_time'])))
        neu_sem  = np.zeros((n_clusters, len(self.alignment['neu_time'])))
        for i in range(n_clusters):
            neu_mean[i,:], neu_sem[i,:] = get_mean_sem(neu[np.where(cluster_id==i)[0], :])
        sort_idx_neu = np.argmax(neu_mean[:,l_idx:r_idx], axis=1).reshape(-1).argsort()
        neu_mean = neu_mean[sort_idx_neu,:]
        neu_sem  = neu_sem[sort_idx_neu,:]
        # plot sorted correlation matrix.
        sorted_indices = np.argsort(cluster_id)
        sorted_corr_matrix = corr_matrix[sorted_indices, :][:, sorted_indices]
        axs[0].imshow(sorted_corr_matrix, cmap=cmap)
        axs[0].set_xlabel('neuron id')
        axs[0].set_ylabel('neuron id')
        axs[0].spines['left'].set_visible(False)
        axs[0].spines['right'].set_visible(False)
        axs[0].spines['top'].set_visible(False)
        axs[0].spines['bottom'].set_visible(False)
        axs[0].set_xticks([])
        axs[0].set_yticks([])
        # plot stimulus.
        upper = n_clusters
        lower = 0
        for i in range(3):
            axs[1].fill_between(
                stim_seq[i,:],
                lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
                color=color1, alpha=0.15, step='mid')
        # plot neural traces averaged within clusters.
        for i in range(n_clusters):
            a = 1 / (np.nanmax(neu_mean[i,:]) - np.nanmin(neu_mean[i,:]))
            b = - np.nanmin(neu_mean[i,:]) / (np.nanmax(neu_mean[i,:]) - np.nanmin(neu_mean[i,:]))
            self.plot_mean_sem(
                axs[1], self.alignment['neu_time'],
                (a*neu_mean[i,:]+b)+n_clusters-i-1, np.abs(a)*neu_sem[i,:],
                color2, None)
        axs[1].tick_params(axis='y', tick1On=False)
        axs[1].spines['left'].set_visible(False)
        axs[1].spines['right'].set_visible(False)
        axs[1].spines['top'].set_visible(False)
        axs[1].set_yticks([])
        axs[1].set_xlabel('time since stim (ms)')
        axs[1].set_ylabel('df/f (z-scored)')
        axs[1].set_ylim([-0.1, n_clusters+0.1])
        

# colors = ['#989A9C', '#A4CB9E', '#9DB4CE', '#EDA1A4', '#F9C08A']
class plotter_VIPTD_G8_align_stim(plotter_utils):
    def __init__(self, neural_trials, labels, significance):
        super().__init__(neural_trials, labels, significance)

    def normal_exc(self, axs):
        
        self.plot_normal(axs[0], [0], 0, cate=-1)
        axs[0].set_title('response to normal \n excitatory (short, fix)')
        
        self.plot_normal(axs[1], [1], 0, cate=-1)
        axs[1].set_title('response to normal \n excitatory (long, fix)')
        
        self.plot_normal(axs[2], [0,1], 0, cate=-1)
        axs[2].set_title('response to normal \n excitatory (fix)')
        
        self.plot_normal_box(axs[3], 0, cate=-1)
        axs[3].set_title('response to normal \n excitatory (fix)')
        
        self.plot_normal(axs[4], [0], 1, cate=-1)
        axs[4].set_title('response to normal \n excitatory (short, jitter)')
        
        self.plot_normal(axs[5], [1], 1, cate=-1)
        axs[5].set_title('response to normal \n excitatory (long, jitter)')
        
        self.plot_normal(axs[6], [0,1], 1, cate=-1)
        axs[6].set_title('response to normal \n excitatory (jitter)')
        
        self.plot_normal_box(axs[7], 1, cate=-1)
        axs[7].set_title('response to normal \n excitatory (jitter)')

    def normal_inh(self, axs):
        
        self.plot_normal(axs[0], [0], 0, cate=1)
        axs[0].set_title('response to normal \n inhibitory (short, fix)')
        
        self.plot_normal(axs[1], [1], 0, cate=1)
        axs[1].set_title('response to normal \n inhibitory (long, fix)')
        
        self.plot_normal(axs[2], [0,1], 0, cate=1)
        axs[2].set_title('response to normal \n inhibitory (fix)')
        
        self.plot_normal_box(axs[3], 0, cate=1)
        axs[3].set_title('response to normal \n inhibitory (fix)')
        
        self.plot_normal(axs[4], [0], 1, cate=1)
        axs[4].set_title('response to normal \n inhibitory (short, jitter)')
        
        self.plot_normal(axs[5], [1], 1, cate=1)
        axs[5].set_title('response to normal \n inhibitory (long, jitter)')
        
        self.plot_normal(axs[6], [0,1], 1, cate=1)
        axs[6].set_title('response to normal \n inhibitory (jitter)')
        
        self.plot_normal_box(axs[7], 1, cate=1)
        axs[7].set_title('response to normal \n inhibitory (jitter)')

    def normal_heatmap(self, axs):
        win_sort = [-500, 500]
        labels = np.concatenate(self.list_labels)
        sig = np.concatenate([self.list_significance[n]['r_normal'] for n in range(self.n_sess)])
        neu_short_fix, _, _, _, _ = get_multi_sess_neu_trial_average(
            self.list_stim_labels, self.alignment['list_neu_seq'], self.alignment['list_stim_seq'],
            self.alignment['list_stim_value'], self.alignment['list_pre_isi'],
            trial_param=[[2,3,4,5], [0], [0], None, [0]])
        neu_long_fix, _, _, _, _ = get_multi_sess_neu_trial_average(
            self.list_stim_labels, self.alignment['list_neu_seq'], self.alignment['list_stim_seq'],
            self.alignment['list_stim_value'], self.alignment['list_pre_isi'],
            trial_param=[[2,3,4,5], [1], [0], None, [0]])
        neu_short_jitter, _, _, _, _ = get_multi_sess_neu_trial_average(
            self.list_stim_labels, self.alignment['list_neu_seq'], self.alignment['list_stim_seq'],
            self.alignment['list_stim_value'], self.alignment['list_pre_isi'],
            trial_param=[[2,3,4,5], [0], [1], None, [0]])
        neu_long_jitter, _, _, _, _ = get_multi_sess_neu_trial_average(
            self.list_stim_labels, self.alignment['list_neu_seq'], self.alignment['list_stim_seq'],
            self.alignment['list_stim_value'], self.alignment['list_pre_isi'],
            trial_param=[[2,3,4,5], [1], [1], None, [0]])
        for i in range(8):
            axs[i].set_xlabel('time since stim (ms)')

        if np.sum(~np.isnan(neu_short_fix)) > 0:
            self.plot_heatmap_neuron(
                axs[0], neu_short_fix, self.alignment['neu_time'], neu_short_fix,
                win_sort, labels, sig)
            axs[0].set_title('response to normal \n (short sorted by short, fix)')

        if np.sum(~np.isnan(neu_long_fix)) > 0:
            self.plot_heatmap_neuron(
                axs[1], neu_long_fix, self.alignment['neu_time'], neu_short_fix,
                win_sort, labels, sig)
            axs[1].set_title('response to normal \n (long sorted by short, fix)')

        if np.sum(~np.isnan(neu_short_fix)) > 0:
            self.plot_heatmap_neuron(
                axs[2], neu_short_fix, self.alignment['neu_time'], neu_long_fix,
                win_sort, labels, sig)
            axs[2].set_title('response to normal \n (short sorted by long, fix)')

        if np.sum(~np.isnan(neu_long_fix)) > 0:
            self.plot_heatmap_neuron(
                axs[3], neu_long_fix, self.alignment['neu_time'], neu_long_fix,
                win_sort, labels, sig)
            axs[3].set_title('response to normal \n (long sorted by long, fix)')

        if np.sum(~np.isnan(neu_short_jitter)) > 0:
            self.plot_heatmap_neuron(
                axs[4], neu_short_jitter, self.alignment['neu_time'], neu_short_jitter,
                win_sort, labels, sig)
            axs[4].set_title('response to normal \n (short sorted by short, jitter)')

        if np.sum(~np.isnan(neu_long_jitter)) > 0:
            self.plot_heatmap_neuron(
                axs[5], neu_long_jitter, self.alignment['neu_time'], neu_short_jitter,
                win_sort, labels, sig)
            axs[5].set_title('response to normal \n (long sorted by short, jitter)')

        if np.sum(~np.isnan(neu_short_jitter)) > 0:
            self.plot_heatmap_neuron(
                axs[6], neu_short_jitter, self.alignment['neu_time'], neu_long_jitter,
                win_sort, labels, sig)
            axs[6].set_title('response to normal \n (short sorted by long, jitter)')

        if np.sum(~np.isnan(neu_long_jitter)) > 0:
            self.plot_heatmap_neuron(
                axs[7], neu_long_jitter, self.alignment['neu_time'], neu_long_jitter,
                win_sort, labels, sig)
            axs[7].set_title('response to normal \n (long sorted by long, jitter)')

    def normal_isi_exc(self, axs):
        
        self.plot_normal_global(axs[0], 0, cate=-1)
        axs[0].set_title('response to normal with preceeding ISI \n excitatory (global, short)')
        
        self.plot_normal_global(axs[1], 1, cate=-1)
        axs[1].set_title('response to normal with preceeding ISI \n excitatory (global, long)')
        
        self.plot_normal_overlap(axs[2], cate=-1)
        axs[2].set_title('response to normal with preceeding ISI \n excitatory (global, {}-{} ms overlap)'.format(
            int(self.expect[0]), int(self.expect[1])))
        
        self.plot_normal_local(axs[3], 0, cate=-1)
        axs[3].set_title('response to normal with preceeding ISI \n excitatory (local, short)')
        
        self.plot_normal_local(axs[4], 1, cate=-1)
        axs[4].set_title('response to normal with preceeding ISI \n excitatory (local, long)')

    def normal_isi_inh(self, axs):
        
        self.plot_normal_global(axs[0], 0, cate=1)
        axs[0].set_title('response to normal with preceeding ISI \n inhibitory (global, short)')
        
        self.plot_normal_global(axs[1], 1, cate=1)
        axs[1].set_title('response to normal with preceeding ISI \n inhibitory (global, long)')
        
        self.plot_normal_overlap(axs[2], cate=1)
        axs[2].set_title('response to normal with preceeding ISI \n inhibitory (global, {}-{} ms overlap)'.format(
            int(self.expect[0]), int(self.expect[1])))
        
        self.plot_normal_local(axs[3], 0, cate=1)
        axs[3].set_title('response to normal with preceeding ISI \n inhibitory (local, short)')
        
        self.plot_normal_local(axs[4], 1, cate=1)
        axs[4].set_title('response to normal with preceeding ISI \n inhibitory (local, long)')

    def normal_epoch_exc(self, axs):
        
        self.plot_normal_epoch(axs[0], 0, 0, cate=-1)
        axs[0].set_title('response to normal \n excitatory (short, fix)')
        
        self.plot_normal_epoch(axs[1], 1, 0, cate=-1)
        axs[1].set_title('response to normal \n excitatory (long, fix)')
        
        self.plot_normal_epoch(axs[2], 0, 1, cate=-1)
        axs[2].set_title('response to normal \n excitatory (short, jitter)')
        
        self.plot_normal_epoch(axs[3], 1, 1, cate=-1)
        axs[3].set_title('response to normal \n excitatory (long, jitter)')

    def normal_epoch_inh(self, axs):
        
        self.plot_normal_epoch(axs[0], 0, 0, cate=1)
        axs[0].set_title('response to normal \n inhibitory (short, fix)')
        
        self.plot_normal_epoch(axs[1], 1, 0, cate=1)
        axs[1].set_title('response to normal \n inhibitory (long, fix)')
        
        self.plot_normal_epoch(axs[2], 0, 1, cate=1)
        axs[2].set_title('response to normal \n inhibitory (short, jitter)')
        
        self.plot_normal_epoch(axs[3], 1, 1, cate=1)
        axs[3].set_title('response to normal \n inhibitory (long, jitter)')

    def normal_cluster(self, axs):
        
        self.plot_normal_cluster([axs[0], axs[1]], 0, 0, cate=-1)
        axs[0].set_title('sorted correlation matrix \n (short, fix) excitatory')
        axs[1].set_title('clustering response \n (short, fix) excitatory')
        
        self.plot_normal_cluster([axs[2], axs[3]], 1, 0, cate=-1)
        axs[2].set_title('sorted correlation matrix \n (long, fix) excitatory')
        axs[4].set_title('clustering response \n (long, fix) excitatory')
        
        self.plot_normal_cluster([axs[4], axs[5]], 0, 0, cate=1)
        axs[4].set_title('sorted correlation matrix \n (short, fix) inhibitory')
        axs[5].set_title('clustering response \n (short, fix) inhibitory')
        
        self.plot_normal_cluster([axs[6], axs[7]], 1, 0, cate=1)
        axs[6].set_title('sorted correlation matrix \n (long, fix) inhibitory')
        axs[7].set_title('clustering response \n (long, fix) inhibitory')


