#!/usr/bin/env python3

import numpy as np
from sklearn.manifold import TSNE
from scipy.signal import cwt, ricker
from scipy.interpolate import interp1d
from matplotlib.colors import LinearSegmentedColormap

from modules.Alignment import run_get_stim_response
from modeling.decoding import multi_sess_decoding_num_neu
from modeling.decoding import multi_sess_decoding_slide_win
from modeling.clustering import clustering_neu_response_mode
from plot.utils import get_mean_sem
from plot.utils import get_frame_idx_from_time
from plot.utils import get_multi_sess_neu_trial_average
from plot.utils import get_roi_label_color
from plot.utils import get_epoch_idx
from plot.utils import get_odd_stim_prepost_idx
from plot.utils import get_expect_time
from plot.utils import get_neu_sync
from plot.utils import adjust_layout_neu
from plot.utils import adjust_layout_spectral
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
        self.l_frames = int(100*timescale)
        self.r_frames = int(200*timescale)
        self.cut_frames = int(60*timescale)
        self.list_stim_labels = [
            nt['stim_labels'][1:-1,:] for nt in list_neural_trials]
        self.list_labels = list_labels
        self.alignment = run_get_stim_response(
                list_neural_trials, self.l_frames, self.r_frames, expected='none')
        self.expect = np.array([
            np.mean([get_expect_time(sl)[0] for sl in self.list_stim_labels]),
            np.mean([get_expect_time(sl)[1] for sl in self.list_stim_labels])])
        self.epoch_early = [get_epoch_idx(sl)[0] for sl in self.list_stim_labels]
        self.epoch_late  = [get_epoch_idx(sl)[1] for sl in self.list_stim_labels]
        self.list_odd_idx = [
            get_odd_stim_prepost_idx(sl) for sl in self.list_stim_labels]
        self.list_significance = list_significance

    def plot_pre_odd_isi_distribution(self, ax):
        isi_short = [self.alignment['list_pre_isi'][n][self.list_odd_idx[n][0]] for n in range(self.n_sess)]
        isi_long  = [self.alignment['list_pre_isi'][n][self.list_odd_idx[n][1]] for n in range(self.n_sess)]
        isi_short = np.concatenate(isi_short)
        isi_long  = np.concatenate(isi_long)
        isi_max = np.nanmax(np.concatenate([isi_short, isi_long]))
        ax.hist(isi_short,
            bins=100, range=[0, isi_max], align='left',
            color='#9DB4CE', density=True)
        ax.hist(isi_long,
            bins=100, range=[0, isi_max], align='right',
            color='dodgerblue', density=True)
        ax.set_title('pre oddball isi distribution')
        ax.tick_params(tick1On=False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xlabel('time (ms)')
        ax.set_ylabel('percentage')
        ax.set_xlim([0, isi_max])
        ax.set_xticks(500*np.arange(0, isi_max/500+1).astype('int32'))
        ax.set_xticklabels(
            500*np.arange(0, isi_max/500+1).astype('int32'),
            rotation='vertical')
        ax.plot([], color='#9DB4CE', label='short')
        ax.plot([], color='dodgerblue', label='long')
        ax.legend(loc='upper left')

    def plot_odd_normal_pre(self, ax, normal, cate=None, roi_id=None):
        if cate != None:
            neu_cate = [
                self.alignment['list_neu_seq'][i][:,(self.list_labels[i]==cate)*self.list_significance[i]['r_oddball'],:]
                for i in range(self.n_sess)]
            color0, color1, color2, _ = get_roi_label_color([cate], 0)
        if roi_id != None:
            neu_cate = [np.expand_dims(self.alignment['list_neu_seq'][0][:,roi_id,:], axis=1)]
            color0, color1, color2, _ = get_roi_label_color(self.labels, roi_id)
        neu_mean = []
        neu_sem = []
        # collect data.
        if 0 in normal:
            neu_short, _, stim_seq_short, stim_value_short, _ = get_multi_sess_neu_trial_average(
                self.list_stim_labels, neu_cate, self.alignment,
                trial_idx=[l[0] for l in self.list_odd_idx])
            mean_short, sem_short = get_mean_sem(neu_short)
            neu_mean.append(mean_short)
            neu_sem.append(sem_short)
        if 1 in normal:
            neu_long, _, stim_seq_long, stim_value_long, _ = get_multi_sess_neu_trial_average(
                self.list_stim_labels, neu_cate, self.alignment,
                trial_idx=[l[1] for l in self.list_odd_idx])
            mean_long, sem_long = get_mean_sem(neu_long)
            neu_mean.append(mean_long)
            neu_sem.append(sem_long)
        # find bounds.
        upper = np.nanmax(neu_mean) + np.nanmax(neu_sem)
        lower = np.nanmin(neu_mean) - np.nanmax(neu_sem)
        # plot stimulus.
        if 0 in normal:
            for i in range(3):
                ax.fill_between(
                    stim_seq_short[i,:],
                    lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
                    color=color1, alpha=0.15, step='mid')
            self.plot_vol(
                ax, self.alignment['stim_time'],
                stim_value_short.reshape(1,-1), color1, upper, lower)
        if 1 in normal:
            for i in range(3):
                ax.fill_between(
                    stim_seq_long[i,:],
                    lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
                    color=color2, alpha=0.15, step='mid')
            self.plot_vol(
                ax, self.alignment['stim_time'],
                stim_value_long.reshape(1,-1), color2, upper, lower)
        # plot neural traces.
        if 0 in normal:
            self.plot_mean_sem(
                ax, self.alignment['neu_time'],
                mean_short, sem_short, color1, 'short')
        if 1 in normal:
            self.plot_mean_sem(
                ax, self.alignment['neu_time'],
                mean_long, sem_long, color2, 'long')
        adjust_layout_neu(ax)
        ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
        ax.set_xlabel('time since pre oddball stim (ms)')
        ax.legend(loc='upper left')

    def plot_odd_normal_spectral(self, ax, normal, cate=None):
        max_freq = 64
        win_eval = [-3000,2000]
        l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, win_eval[0], win_eval[1])
        neu_cate = [
            self.alignment['list_neu_seq'][i][:,(self.list_labels[i]==cate)*self.list_significance[i]['r_oddball'],:]
            for i in range(self.n_sess)]
        _, _, _, cmap = get_roi_label_color([cate], 0)
        # collect data.
        neu_x, _, _, _, _ = get_multi_sess_neu_trial_average(
            self.list_stim_labels, neu_cate, self.alignment,
            trial_idx=[l[normal+2] for l in self.list_odd_idx])
        # wavelet decomposition.
        neu_cwt = [
            cwt(neu_x[i, :], ricker, np.arange(1, max_freq-1))
            for i in range(neu_x.shape[0])]
        neu_cwt = np.nanmean(neu_cwt, axis=0)
        # plot spectrogram.
        ax.imshow(
            neu_cwt,
            extent=[self.alignment['neu_time'][0], self.alignment['neu_time'][-1], 1, max_freq-1],
            aspect='auto', origin='lower', cmap=cmap)
        adjust_layout_spectral(ax)
        ax.set_xlabel('time since post oddball stim (ms)')
        ax.axvline(0, color='black', lw=1, label='stim', linestyle='--')
        ax.set_xlim([win_eval[0], win_eval[1]])
    
    def plot_odd_prepost(self, ax, cate=None):
        color0, color1, color2, _ = get_roi_label_color([cate], 0)
        colors = [color0, color1, color2]
        lbl = ['pre', 'short', 'long']
        idx_cut_l = np.searchsorted(self.alignment['neu_time'], 0)-self.cut_frames
        idx_cut_r = np.searchsorted(self.alignment['neu_time'], 0)+self.cut_frames
        neu_cate = [
            self.alignment['list_neu_seq'][i][:,(self.list_labels[i]==cate)*self.list_significance[i]['r_normal'],:]
            for i in range(self.n_sess)]
        # collect data.
        neu_pre, _, stim_seq, _, _ = get_multi_sess_neu_trial_average(
            self.list_stim_labels, neu_cate, self.alignment,
            trial_param=[[-1], [0], [0], None, [0]])
        neu_short, _, _, _, _ = get_multi_sess_neu_trial_average(
            self.list_stim_labels, neu_cate, self.alignment,
            trial_idx=[l[2] for l in self.list_odd_idx])
        neu_long, _, _, _, _ = get_multi_sess_neu_trial_average(
            self.list_stim_labels, neu_cate, self.alignment,
            trial_idx=[l[3] for l in self.list_odd_idx])
        neu_mean = []
        neu_sem = []
        for neu in [neu_pre, neu_short, neu_long]:
            m, s = get_mean_sem(neu[:,idx_cut_l:idx_cut_r])
            neu_mean.append(m)
            neu_sem.append(s)
        # compute bounds.
        upper = np.nanmax(neu_mean) + np.nanmax(neu_sem)
        lower = np.nanmin(neu_mean) - np.nanmax(neu_sem)
        # plot stimulus.
        ax.fill_between(
            stim_seq[1,:],
            lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
            color=color0, alpha=0.15, step='mid')
        # plot neural traces.
        for i in range(3):
            self.plot_mean_sem(
                ax, self.alignment['neu_time'][idx_cut_l:idx_cut_r],
                neu_mean[i], neu_sem[i], colors[i], lbl[i])
        adjust_layout_neu(ax)
        ax.set_xlim([self.alignment['neu_time'][idx_cut_l], self.alignment['neu_time'][idx_cut_r]])
        ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
        ax.set_xlabel('time since stim (ms)')

    def plot_odd_post_box(self, ax, cate=None):
        win_base = [-1500,0]
        offsets = [0.0, 0.1, 0.2]
        color0, color1, color2, _ = get_roi_label_color([cate], 0)
        colors = [color0, color1, color2]
        lbl = ['pre', 'short', 'long']
        neu_cate = [
            self.alignment['list_neu_seq'][i][:,(self.list_labels[i]==cate)*self.list_significance[i]['r_normal'],:]
            for i in range(self.n_sess)]
        # collect data.
        neu_pre, _, _, _, _ = get_multi_sess_neu_trial_average(
            self.list_stim_labels, neu_cate, self.alignment,
            trial_param=[[-1], [0], [0], None, [0]])
        neu_short, _, _, _, _ = get_multi_sess_neu_trial_average(
            self.list_stim_labels, neu_cate, self.alignment,
            trial_idx=[l[2] for l in self.list_odd_idx])
        neu_long, _, _, _, _ = get_multi_sess_neu_trial_average(
            self.list_stim_labels, neu_cate, self.alignment,
            trial_idx=[l[3] for l in self.list_odd_idx])
        # plot errorbar.
        neu_x = [neu_pre, neu_short, neu_long]
        for i in range(3):
            self.plot_win_mag_box(
                ax, neu_x[i], self.alignment['neu_time'], win_base,
                colors[i], 0, offsets[i])
            ax.plot([], color=colors[i], label=lbl[i])
        ax.legend(loc='upper right')
        
    def plot_odd_normal_cluster(self, axs, normal, fix_jitter, cate):
        n_clusters = 4
        max_clusters = 32
        win_sort = [-500,3000]
        colors=['#A4CB9E', '#9DB4CE', '#EDA1A4', '#F9C08A']
        metric_lbl = ['silhouette', 'calinski_harabasz', 'davies_bouldin', 'inertia']
        l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, win_sort[0], win_sort[1])
        _, color1, color2, cmap = get_roi_label_color([cate], 0)
        neu_cate = [
            self.alignment['list_neu_seq'][i][:,(self.list_labels[i]==cate)*self.list_significance[i]['r_normal'],:]
            for i in range(self.n_sess)]
        # collect data.       
        neu, _, stim_seq, _, _ = get_multi_sess_neu_trial_average(
            self.list_stim_labels, neu_cate, self.alignment,
            trial_idx=[l[normal] for l in self.list_odd_idx])
        neu = neu[~np.isnan(np.sum(neu,axis=1)),:]
        # fit model.
        [metrics, sorted_neu_corr, neu_mean, neu_sem, cluster_corr] = clustering_neu_response_mode(
            neu, n_clusters, max_clusters, l_idx, r_idx)
        # plot sorted correlation matrix.
        axs[0].imshow(sorted_neu_corr, cmap=cmap)
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
                color=[color1, color2][normal], alpha=0.15, step='mid')
        # plot neural traces averaged within clusters.
        for i in range(n_clusters):
            a = 1 / (np.nanmax(neu_mean[i,:]) - np.nanmin(neu_mean[i,:]))
            b = - np.nanmin(neu_mean[i,:]) / (np.nanmax(neu_mean[i,:]) - np.nanmin(neu_mean[i,:]))
            self.plot_mean_sem(
                axs[1], self.alignment['neu_time'],
                (a*neu_mean[i,:]+b)+n_clusters-i-1, np.abs(a)*neu_sem[i,:],
                [color1, color2][normal], None)
        axs[1].tick_params(axis='y', tick1On=False)
        axs[1].spines['left'].set_visible(False)
        axs[1].spines['right'].set_visible(False)
        axs[1].spines['top'].set_visible(False)
        axs[1].set_yticks([])
        axs[1].set_xlabel('time since pre oddball stim (ms)')
        axs[1].set_ylabel('df/f (z-scored)')
        axs[1].set_ylim([-0.1, n_clusters+0.1])
        # plot clustering metrics.
        for i in range(4):
            axs[2].plot(metrics['n_clusters'], metrics[metric_lbl[i]], color=colors[i], label=metric_lbl[i])
        axs[2].tick_params(axis='y', tick1On=False)
        axs[2].spines['right'].set_visible(False)
        axs[2].spines['top'].set_visible(False)
        axs[2].set_xlabel('n_clusters')
        axs[2].set_ylabel('normalized value')
        axs[2].set_xlim(2, max_clusters+2)
        axs[2].legend(loc='upper right')
        # plot cross cluster correlations.
        mask = np.tril(np.ones_like(cluster_corr, dtype=bool), k=0)
        masked_corr = np.where(mask, cluster_corr, np.nan)
        axs[3].matshow(masked_corr, interpolation='nearest', cmap=cmap)
        axs[3].tick_params(bottom=False, top=False, labelbottom=True, labeltop=False)
        axs[3].tick_params(tick1On=False)
        axs[3].spines['left'].set_visible(False)
        axs[3].spines['right'].set_visible(False)
        axs[3].spines['top'].set_visible(False)
        axs[3].spines['bottom'].set_visible(False)
        axs[3].set_xticks(np.arange(n_clusters))
        axs[3].set_yticks(np.arange(n_clusters))
        axs[2].set_xlabel('cluster id')
        axs[2].set_ylabel('cluster id')
        for i in range(cluster_corr.shape[0]):
            for j in range(cluster_corr.shape[1]):
                if not np.isnan(masked_corr[i, j]):
                    axs[3].text(j, i, f'{masked_corr[i, j]:.2f}',
                                ha='center', va='center', color='grey')
            
    def plot_odd_decode_num_neu(self, ax, cate=None):
        num_step = 10
        n_decode = 1
        win_eval = [-500,500]
        l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, win_eval[0], win_eval[1])
        color0, color1, color2, _ = get_roi_label_color([cate], 0)
        neu_cate = [
            self.alignment['list_neu_seq'][i][:,(self.list_labels[i]==cate)*self.list_significance[i]['r_normal'],:]
            for i in range(self.n_sess)]
        # collect data.
        neu_pre, _, _, _ = get_multi_sess_neu_trial_average(
            self.list_stim_labels, neu_cate, self.alignment,
            trial_param=[[-1], [0], [0], None, [0]],
            mean_sem=False)
        neu_short, _, _, _ = get_multi_sess_neu_trial_average(
            self.list_stim_labels, neu_cate, self.alignment,
            trial_idx=[l[2] for l in self.list_odd_idx],
            mean_sem=False)
        neu_long, _, _, _ = get_multi_sess_neu_trial_average(
            self.list_stim_labels, neu_cate, self.alignment,
            trial_idx=[l[3] for l in self.list_odd_idx],
            mean_sem=False)
        # extract response projection for each neuron.
        neu_x = [neu_pre, neu_short, neu_long]
        # organize data.
        neu_y = [np.concatenate([
            np.ones(neu_x[i][s].shape[0])*i
            for i in range(3)], axis=0) for s in range(self.n_sess)]
        neu_x = [np.concatenate([
            neu_x[i][s][:,:,l_idx:r_idx]
            for i in range(3)], axis=0) for s in range(self.n_sess)]
        # run decoding.
        sampling_nums, acc_model, acc_chance = multi_sess_decoding_num_neu(
            neu_x, neu_y, num_step, n_decode)
        # plot results.
        self.plot_multi_sess_decoding_num_neu(
            ax, sampling_nums, acc_model, acc_chance, color0, color2)
        ax.set_xlim([sampling_nums[0]-num_step, sampling_nums[-1]+num_step])
        ax.set_ylabel('ACC \n window [{},{}] ms'.format(win_eval[0], win_eval[1]))
    
    def plot_odd_decode_slide_win(self, ax, cate=None):
        win_step = 3
        n_decode = 1
        win_width = 300
        l_idx, r_idx = get_frame_idx_from_time(
            self.alignment['neu_time'], 0, 0, win_width)
        num_frames = r_idx - l_idx
        win_range = [-2500, 2500]
        start_idx, end_idx = get_frame_idx_from_time(
            self.alignment['neu_time'], 0, win_range[0], win_range[1])
        color0, color1, color2, _ = get_roi_label_color([cate], 0)
        neu_cate = [
            self.alignment['list_neu_seq'][i][:,(self.list_labels[i]==cate)*self.list_significance[i]['r_normal'],:]
            for i in range(self.n_sess)]
        # collect data.
        neu_pre, _, _, _ = get_multi_sess_neu_trial_average(
            self.list_stim_labels, neu_cate, self.alignment,
            trial_param=[[-1], [0], [0], None, [0]],
            mean_sem=False)
        neu_short, _, _, _ = get_multi_sess_neu_trial_average(
            self.list_stim_labels, neu_cate, self.alignment,
            trial_idx=[l[2] for l in self.list_odd_idx],
            mean_sem=False)
        neu_long, _, _, _ = get_multi_sess_neu_trial_average(
            self.list_stim_labels, neu_cate, self.alignment,
            trial_idx=[l[3] for l in self.list_odd_idx],
            mean_sem=False)
        neu_x = [neu_pre, neu_short, neu_long]
        _, _, stim_seq, _, _ = get_multi_sess_neu_trial_average(
            self.list_stim_labels, neu_cate, self.alignment,
            trial_param=[[-1], [0], [0], None, [0]])
        # organize data.
        neu_y = [np.concatenate([
            np.ones(neu_x[i][s].shape[0])*i
            for i in range(3)], axis=0) for s in range(self.n_sess)]
        neu_x = [np.concatenate([
            neu_x[i][s]
            for i in range(3)], axis=0) for s in range(self.n_sess)]
        # run decoding.
        acc_model, acc_chance = multi_sess_decoding_slide_win(
            neu_x, neu_y, start_idx, end_idx, win_step, n_decode, num_frames)
        # compute bounds.
        upper = np.nanmax([acc_model, acc_chance])
        lower = np.nanmin([acc_model, acc_chance])
        # plot stimulus.
        ax.fill_between(
            stim_seq[1,:],
            lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
            color=color0, alpha=0.15, step='mid')
        # plot results.
        eval_time = self.alignment['neu_time'][
            np.arange(start_idx, end_idx, win_step)]
        self.plot_multi_sess_decoding_slide_win(
            ax, eval_time, acc_model, acc_chance, color0, color2)
        ax.set_xlabel('time since stim (ms)')
        ax.set_ylabel('ACC')
        ax.set_xlim([eval_time[0], eval_time[-1]])
    
    def plot_odd_sync(self, ax, cate=None):
        win_width = 200
        l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, 0, win_width)
        win_width = r_idx - l_idx
        neu_cate = [
            self.alignment['list_neu_seq'][i][:,(self.list_labels[i]==cate)*self.list_significance[i]['r_oddball'],:]
            for i in range(self.n_sess)]
        color0, color1, color2, _ = get_roi_label_color([cate], 0)
        # collect data.
        neu_short, _, stim_seq_short, stim_value_short, _ = get_multi_sess_neu_trial_average(
            self.list_stim_labels, neu_cate, self.alignment,
            trial_idx=[l[0] for l in self.list_odd_idx])
        neu_long, _, stim_seq_long, stim_value_long, _ = get_multi_sess_neu_trial_average(
            self.list_stim_labels, neu_cate, self.alignment,
            trial_idx=[l[1] for l in self.list_odd_idx])
        # compute synchronization.
        sync_short = get_neu_sync(neu_short, win_width)
        sync_long  = get_neu_sync(neu_long, win_width)
        # find bounds.
        upper = np.nanmax([sync_short, sync_long])
        lower = np.nanmin([sync_short, sync_long])
        # plot stimulus.
        for i in range(3):
            ax.fill_between(
                stim_seq_short[i,:],
                lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
                color=color1, alpha=0.15, step='mid')
        self.plot_vol(
            ax, self.alignment['stim_time'],
            stim_value_short.reshape(1,-1), color1, upper, lower)
        for i in range(3):
            ax.fill_between(
                stim_seq_long[i,:],
                lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
                color=color2, alpha=0.15, step='mid')
        self.plot_vol(
            ax, self.alignment['stim_time'],
            stim_value_long.reshape(1,-1), color2, upper, lower)
        # plot synchronization.
        ax.plot(self.alignment['neu_time'][win_width:], sync_short, color=color1, label='short')
        ax.plot(self.alignment['neu_time'][win_width:], sync_long,  color=color2, label='long')
        ax.tick_params(axis='y', tick1On=False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_ylabel('sync level')
        ax.legend(loc='upper right')
        ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
        ax.set_xlabel('time since pre oddball stim (ms)')
        ax.legend(loc='upper left')
    
    def plot_odd_stim_corr(self, ax, cate=None):
        color0, color1, color2, _ = get_roi_label_color([cate], 0)
        colors = [color0, color1, color2]
        lbl = ['pre', 'short', 'long']
        pos = [0, 1, 2]
        win_eval = [-500,500]
        l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, win_eval[0], win_eval[1])
        neu_cate = [
            self.alignment['list_neu_seq'][i][:,(self.list_labels[i]==cate)*self.list_significance[i]['r_normal'],:]
            for i in range(self.n_sess)]
        # collect data.
        neu_pre, _, _, stim_pre, _ = get_multi_sess_neu_trial_average(
            self.list_stim_labels, neu_cate, self.alignment,
            trial_param=[[-1], [0], [0], None, [0]])
        neu_short, _, _, stim_short, _ = get_multi_sess_neu_trial_average(
            self.list_stim_labels, neu_cate, self.alignment,
            trial_idx=[l[2] for l in self.list_odd_idx])
        neu_long, _, _, stim_long, _ = get_multi_sess_neu_trial_average(
            self.list_stim_labels, neu_cate, self.alignment,
            trial_idx=[l[3] for l in self.list_odd_idx])
        neu_x = [neu_pre[:,l_idx:r_idx], neu_short[:,l_idx:r_idx], neu_long[:,l_idx:r_idx]]
        # stimulus traces interpolation.
        interp   = interp1d(self.alignment['stim_time'], stim_pre, bounds_error=False)
        stim_pre = interp(self.alignment['neu_time'])
        #stim_pre = np.tile(stim_pre.reshape(1,-1), (neu_pre.shape[0],1))
        interp     = interp1d(self.alignment['stim_time'], stim_short, bounds_error=False)
        stim_short = interp(self.alignment['neu_time'])
        #stim_short = np.tile(stim_short.reshape(1,-1), (neu_pre.shape[0],1))
        interp    = interp1d(self.alignment['stim_time'], stim_long, bounds_error=False)
        stim_long = interp(self.alignment['neu_time'])
        #stim_long = np.tile(stim_long.reshape(1,-1), (neu_pre.shape[0],1))
        neu_y = [stim_pre[l_idx:r_idx], stim_short[l_idx:r_idx], stim_long[l_idx:r_idx]]
        # compute correlation.
        corr = []
        for i in range(3):
            sim = np.dot(neu_x[i], neu_y[i]/np.linalg.norm(neu_y[i])) / np.linalg.norm(neu_x[i], axis=1)
            corr.append(np.abs(sim))
        m = [get_mean_sem(c.reshape(-1,1))[0] for c in corr]
        s = [get_mean_sem(c.reshape(-1,1))[1] for c in corr]
        # plot errorbar.
        for i in range(3):
            ax.errorbar(
                pos[i],
                m[i], s[i],
                color=colors[i],
                capsize=2, marker='o', linestyle='none',
                markeredgecolor='white', markeredgewidth=0.1)
        ax.tick_params(tick1On=False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xlabel('stimulus type')
        ax.set_ylabel('correlation absolute (mean$\pm$sem) \n window [{},{}] ms'.format(win_eval[0], win_eval[1]))
        ax.set_xticks([0,1,2])
        ax.set_xticklabels(lbl)
        ax.set_xlim([-0.5, 2.5])
        


    def plot_odd_latent(self, ax, cate=None):
        d_latent = 2
        _, color1, color2, _ = get_roi_label_color([cate], 0)
        colors = [color1, color2]
        lbl = ['short', 'long']
        neu_cate = [
            self.alignment['list_neu_seq'][i][:,(self.list_labels[i]==cate)*self.list_significance[i]['r_normal'],:]
            for i in range(self.n_sess)]
        # collect data.
        neu_short, _, stim_seq_short, _, _ = get_multi_sess_neu_trial_average(
            self.list_stim_labels, neu_cate, self.alignment,
            trial_idx=[l[0] for l in self.list_odd_idx])
        neu_long, _, stim_seq_long, _, _ = get_multi_sess_neu_trial_average(
            self.list_stim_labels, neu_cate, self.alignment,
            trial_idx=[l[1] for l in self.list_odd_idx])
        neu_x = [np.expand_dims(neu_short,axis=2), np.expand_dims(neu_long,axis=2)]
        neu_x = np.concatenate(neu_x, axis=2)
        stim_seq = [stim_seq_short, stim_seq_long]
        # fit model.
        model = TSNE(n_components=d_latent)
        neu_z = model.fit_transform(
            neu_x.reshape(neu_x.shape[0],-1).T
            ).T.reshape(d_latent, -1, 2)
        # plot dynamics.
        for i in range(2):
            # trajaectory.
            cmap = LinearSegmentedColormap.from_list(lbl[i], ['white', colors[i]])
            for t in range(neu_z.shape[1]-1):
                c = cmap(np.linspace(0, 1, neu_z.shape[1]))
                ax.plot(neu_z[0,t:t+2,i], neu_z[1,t:t+2,i], color=c[t,:])
            # end point.
            ax.scatter(neu_z[0,-1,i], neu_z[1,-1,i], color=colors[i], label=lbl[i])
            # oddball isi.
            l_idx, r_idx = get_frame_idx_from_time(
                self.alignment['neu_time'], 0, stim_seq[i][1,1], stim_seq[i][2,0])
            ax.plot(neu_z[0,l_idx:r_idx,i], neu_z[1,l_idx:r_idx,i], linestyle=':', color='black')
        ax.plot([], linestyle=':', color='black', label='oddball ISI')
        ax.tick_params(tick1On=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel('latent 1')
        ax.set_ylabel('latent 2')
        ax.legend(loc='upper right')

class plotter_VIPTD_G8_align_odd(plotter_utils):
    def __init__(self, neural_trials, labels, significance):
        super().__init__(neural_trials, labels, significance)

    def odd_normal_exc(self, axs):

        self.plot_odd_normal_pre(axs[0], [0], cate=-1)
        axs[0].set_title('response to oddball \n excitatory (short)')
        
        self.plot_odd_normal_pre(axs[1], [1], cate=-1)
        axs[1].set_title('response to oddball \n excitatory (long)')
        
        self.plot_odd_normal_pre(axs[2], [0,1], cate=-1)
        axs[2].set_title('response to oddball \n excitatory')
        
        self.plot_odd_normal_spectral(axs[3], 0, cate=-1)
        axs[3].set_title('response spectrogram to oddball \n excitatory (short)')
        
        self.plot_odd_normal_spectral(axs[4], 1, cate=-1)
        axs[4].set_title('response spectrogram to oddball \n excitatory (long)')
        
        self.plot_odd_prepost(axs[5], cate=-1)
        axs[5].set_title('response to oddball stim \n excitatory')

        self.plot_odd_post_box(axs[6], cate=-1)
        axs[6].set_title('response to oddball stim \n excitatory')
        
        self.plot_odd_decode_num_neu(axs[7], cate=-1)
        axs[7].set_title('single trial decoding accuracy for pre/short/long \n with number of neurons \n excitatory')
        
        self.plot_odd_decode_slide_win(axs[8], cate=-1)
        axs[8].set_title('single trial decoding accuracy for pre/short/long \n with sliding window \n excitatory')
        
    def odd_normal_inh(self, axs):
        
        self.plot_odd_normal_pre(axs[0], [0], cate=1)
        axs[0].set_title('response to oddball \n inhibitory (short)')
        
        self.plot_odd_normal_pre(axs[1], [1], cate=1)
        axs[1].set_title('response to oddball \n inhibitory (long)')
        
        self.plot_odd_normal_pre(axs[2], [0,1], cate=1)
        axs[2].set_title('response to oddball \n inhibitory')
        
        self.plot_odd_normal_spectral(axs[3], 0, cate=1)
        axs[3].set_title('response spectrogram to oddball \n inhibitory (short)')
        
        self.plot_odd_normal_spectral(axs[4], 1, cate=1)
        axs[4].set_title('response spectrogram to oddball \n inhibitory (long)')
        
        self.plot_odd_prepost(axs[5], cate=1)
        axs[5].set_title('response to oddball stim \n inhibitory')
        
        self.plot_odd_post_box(axs[6], cate=1)
        axs[6].set_title('response to oddball stim \n inhibitory')
        
        self.plot_odd_decode_num_neu(axs[7], cate=1)
        axs[7].set_title('single trial decoding accuracy for pre/short/long \n with number of neurons \n inhibitory')
        
        self.plot_odd_decode_slide_win(axs[8], cate=1)
        axs[8].set_title('single trial decoding accuracy for pre/short/long \n with sliding window \n inhibitory')

    def odd_normal_heatmap(self, axs):
        win_sort = [-500, 1500]
        labels = np.concatenate(self.list_labels)
        sig = np.concatenate([self.list_significance[n]['r_oddball'] for n in range(self.n_sess)])
        neu_short, _, stim_seq_short, _, _ = get_multi_sess_neu_trial_average(
            self.list_stim_labels, self.alignment['list_neu_seq'], self.alignment,
            trial_idx=[l[0] for l in self.list_odd_idx])
        neu_long, _, stim_seq_long, _, _ = get_multi_sess_neu_trial_average(
            self.list_stim_labels, self.alignment['list_neu_seq'], self.alignment,
            trial_idx=[l[1] for l in self.list_odd_idx])
        for i in range(4):
            axs[i].set_xlabel('time since pre oddball stim (ms)')
            
        self.plot_heatmap_neuron(axs[0], neu_short, self.alignment['neu_time'], neu_short, win_sort, labels, sig)
        axs[0].set_title('response to oddball \n (short sorted by short)')
        
        self.plot_heatmap_neuron(axs[1], neu_long, self.alignment['neu_time'], neu_short, win_sort, labels, sig)
        axs[1].set_title('response to oddball \n (long sorted by short)')
        
        self.plot_heatmap_neuron(axs[2], neu_short, self.alignment['neu_time'], neu_long, win_sort, labels, sig)
        axs[2].set_title('response to oddball \n (short sorted by long)')
        
        self.plot_heatmap_neuron(axs[3], neu_long, self.alignment['neu_time'], neu_long, win_sort, labels, sig)
        axs[3].set_title('response to oddball \n (long sorted by long)')
    
    def odd_normal_pop_exc(self, axs):
        
        self.plot_odd_sync(axs[0], cate=-1)
        axs[0].set_title('synchronization level to oddball \n excitatory')

        self.plot_odd_stim_corr(axs[1], cate=-1)
        axs[1].set_title('correlation with stimulus \n excitatory')
        
        self.plot_odd_latent(axs[2], cate=-1)
        axs[2].set_title('latent dynamics to oddball \n excitatory')

    def odd_normal_pop_inh(self, axs):
        
        self.plot_odd_sync(axs[0], cate=1)
        axs[0].set_title('synchronization level to oddball \n inhibitory')
        
        self.plot_odd_stim_corr(axs[1], cate=1)
        axs[1].set_title('correlation with stimulus \n inhibitory')
        
        self.plot_odd_latent(axs[2], cate=1)
        axs[2].set_title('latent dynamics to oddball \n inhibitory')
        
    def odd_normal_mode(self, axs):
        
        self.plot_odd_normal_cluster([axs[0], axs[1], axs[2], axs[3]], 0, 0, cate=-1)
        axs[0].set_title('sorted correlation matrix \n (short) excitatory')
        axs[1].set_title('clustering response \n (short) excitatory')
        axs[2].set_title('clustering evaluation \n (short) excitatory')
        axs[3].set_title('cross cluster correlation \n (short) excitatory')
        
        self.plot_odd_normal_cluster([axs[4], axs[5], axs[6], axs[7]], 1, 0, cate=-1)
        axs[4].set_title('sorted correlation matrix \n (long) excitatory')
        axs[5].set_title('clustering response \n (long) excitatory')
        axs[6].set_title('clustering evaluation \n (long) excitatory')
        axs[7].set_title('cross cluster correlation \n (long) excitatory')
        
        self.plot_odd_normal_cluster([axs[8], axs[9], axs[10], axs[11]], 0, 0, cate=1)
        axs[8].set_title('sorted correlation matrix \n (short) inhibitory')
        axs[9].set_title('clustering response \n (short) inhibitory')
        axs[10].set_title('clustering evaluation \n (short) inhibitory')
        axs[11].set_title('cross cluster correlation \n (short) inhibitory')
        
        self.plot_odd_normal_cluster([axs[12], axs[13], axs[14], axs[15]], 1, 0, cate=1)
        axs[12].set_title('sorted correlation matrix \n (long) inhibitory')
        axs[13].set_title('clustering response \n (long) inhibitory')
        axs[14].set_title('clustering evaluation \n (long) inhibitory')
        axs[15].set_title('cross cluster correlation \n (long) inhibitory')
        