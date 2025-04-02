#!/usr/bin/env python3

import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from modules.Alignment import run_get_stim_response
from modeling.clustering import clustering_neu_response_mode
from modeling.clustering import get_mean_sem_cluster
from modeling.clustering import get_bin_mean_sem_cluster
from modeling.clustering import feature_categorization
from modeling.generative import run_glm_multi_sess
from modeling.quantifications import get_all_metrics
from utils import show_memory_usage
from utils import get_norm01_params
from utils import get_odd_stim_prepost_idx
from utils import exclude_odd_stim
from utils import get_mean_sem
from utils import get_neu_trial
from utils import get_frame_idx_from_time
from utils import get_isi_bin_neu
from utils import get_expect_interval
from utils import get_block_1st_idx
from utils import get_neu_sync
from utils import run_wilcoxon_trial
from utils import get_cmap_color
from utils import adjust_layout_neu
from utils import adjust_layout_3d_latent
from utils import add_legend
from utils import utils_basic

# fig, ax = plt.subplots(1, 1, figsize=(6, 6))
# fig, axs = plt.subplots(1, 4, figsize=(30, 6))
# fig, ax = plt.subplots(1, 1, figsize=(6, 6), subplot_kw={"projection": "3d"})

class plotter_utils(utils_basic):

    def __init__(
            self,
            list_neural_trials, list_labels, list_significance
            ):
        super().__init__()
        timescale = 1.0
        self.n_sess = len(list_neural_trials)
        self.l_frames = int(200*timescale)
        self.r_frames = int(250*timescale)
        self.list_labels = list_labels
        self.list_neural_trials = list_neural_trials
        self.alignment = run_get_stim_response(
                list_neural_trials, self.l_frames, self.r_frames, expected='none')
        self.list_stim_labels = self.alignment['list_stim_labels']
        self.list_odd_idx = [
            get_odd_stim_prepost_idx(sl) for sl in self.list_stim_labels]
        self.expect = np.array(np.mean([get_expect_interval(sl)[0] for sl in self.list_stim_labels]))
        self.list_block_start = [get_block_1st_idx(sl, 3) for sl in self.list_stim_labels]
        self.list_significance = list_significance
        self.bin_win = [450,2550]
        self.bin_num = 2
        self.n_clusters = 10                                                                             
        self.max_clusters = 10
        self.d_latent = 3
    
    def run_clustering(self, cate):
        n_latents = 25
        win_eval = [-3000,2000]
        l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, win_eval[0], win_eval[1])
        # fix standard.
        _, [neu_1, _, _, _], _, _ = get_neu_trial(
            self.alignment, self.list_labels, self.list_significance,
            [exclude_odd_stim(sl) for sl in self.list_stim_labels],
            trial_param=[[2,3,4,5], [0], [0], None, [0], [0]],
            cate=cate, roi_id=None)
        # jitter standard.
        _, [neu_2, _, _, _], _, _ = get_neu_trial(
            self.alignment, self.list_labels, self.list_significance,
            [exclude_odd_stim(sl) for sl in self.list_stim_labels],
            trial_param=[[2,3,4,5], [0], [1], None, [0], [0]],
            cate=cate, roi_id=None)
        # fix short oddball.
        _, [neu_3, _, _, _], _, _ = get_neu_trial(
            self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
            trial_idx=[l[2] for l in self.list_odd_idx],
            trial_param=[None, None, [0], None, [0], [0]],
            cate=cate, roi_id=None)
        # jitter short oddball.
        _, [neu_4, _, _, _], _, _ = get_neu_trial(
            self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
            trial_idx=[l[2] for l in self.list_odd_idx],
            trial_param=[None, None, [1], None, [0], [0]],
            cate=cate, roi_id=None)
        # fix long oddball.
        _, [neu_5, _, _, _], _, _ = get_neu_trial(
            self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
            trial_idx=[l[3] for l in self.list_odd_idx],
            trial_param=[None, None, [0], None, [0], [0]],
            cate=cate, roi_id=None)
        # jitter long oddball.
        _, [neu_6, _, _, _], _, _ = get_neu_trial(
            self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
            trial_idx=[l[3] for l in self.list_odd_idx],
            trial_param=[None, None, [1], None, [0], [0]],
            cate=cate, roi_id=None)
        # collect data.
        neu_all = [neu[:,l_idx:r_idx] for neu in [neu_1, neu_2, neu_3, neu_4, neu_5, neu_6]]
        # get correlation matrix.
        cluster_corr = [np.corrcoef(neu) for neu in neu_all]
        cluster_corr = np.concatenate(cluster_corr, axis=1)
        # extract features.
        model = PCA(n_components=n_latents)
        neu_x = model.fit_transform(cluster_corr)
        metrics, cluster_id = clustering_neu_response_mode(
            neu_x, self.n_clusters, self.max_clusters)
        return metrics, cluster_id
    
    def run_features_categorization(self, cate):
        win_eval = [-1500,1500]
        l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, win_eval[0], win_eval[1])
        # collect data.
        _, [neu_seq_mean, neu_seq_sem, stim_seq, stim_value], _, [n_trials, n_neurons] = get_neu_trial(
            self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
            trial_param=[[2,3,4,5], [0], [0], None, [0], [0]],
            cate=cate, roi_id=None)
        neu_seq_mean = neu_seq_mean[:,l_idx:r_idx]
        neu_seq_sem = neu_seq_sem[:,l_idx:r_idx]
        neu_time = self.alignment['neu_time'][l_idx:r_idx]
        results_all = feature_categorization(neu_seq_mean, neu_seq_sem, neu_time)
        return results_all
    
    def run_glm(self, cate):
        # define kernel window.
        kernel_win = [-500,3000]
        l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, kernel_win[0], kernel_win[1])
        kernel_time = self.alignment['neu_time'][l_idx:r_idx+1]
        l_idx = np.searchsorted(self.alignment['neu_time'], 0) - l_idx
        r_idx = r_idx - np.searchsorted(self.alignment['neu_time'], 0)
        # collect data.
        list_dff = [
            self.list_neural_trials[i]['dff'][
                np.in1d(self.list_labels[i],cate)*self.list_significance[i]['r_standard'],:]
            for i in range(self.n_sess)]
        list_neu_time = [nt['time'] for nt in self.list_neural_trials]
        list_input_time  = [nt['vol_time'] for nt in self.list_neural_trials]
        list_input_value = [nt['vol_stim_vis'] for nt in self.list_neural_trials]
        list_stim_labels = [nt['stim_labels'] for nt in self.list_neural_trials]
        # fit glm model.
        kernel_all, exp_var_all = run_glm_multi_sess(
            list_dff, list_neu_time,
            list_input_time, list_input_value, list_stim_labels,
            l_idx, r_idx)
        return kernel_time, kernel_all, exp_var_all

    def plot_oddball_fix(self, ax, oddball, cate=None, roi_id=None):
        xlim = [-2000,5000]
        # collect data.
        neu_mean = []
        neu_sem = []
        n_trials = 0
        if 0 in oddball:
            [[color0, color1, color2, _],
             [neu_short, _, stim_seq_short, stim_value_short], _,
             [nt, n_neurons]] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_idx=[l[0] for l in self.list_odd_idx],
                trial_param=[None, None, [0], None, [0], [0]],
                cate=cate, roi_id=None)
            mean_short, sem_short = get_mean_sem(neu_short)
            neu_mean.append(mean_short)
            neu_sem.append(sem_short)
            n_trials += nt
        if 1 in oddball:
            [[color0, color1, color2, _],
             [neu_long, _, stim_seq_long, stim_value_long], _,
             [nt, n_neurons]] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_idx=[l[1] for l in self.list_odd_idx],
                trial_param=[None, None, [0], None,[0], [0]],
                cate=cate, roi_id=None)
            mean_long, sem_long = get_mean_sem(neu_long)
            neu_mean.append(mean_long)
            neu_sem.append(sem_long)
            n_trials += nt
        # find bounds.
        upper = np.nanmax(neu_mean) + np.nanmax(neu_sem)
        lower = np.nanmin(neu_mean) - np.nanmax(neu_sem)
        # plot stimulus.
        if 0 in oddball:
            c_idx = int(stim_seq_short.shape[0]/2)
            c_stim = [color1] * stim_seq_short.shape[-2]
            c_stim[c_idx] = color1
            ax.axvline(
                stim_seq_short[c_idx,1]+self.expect,
                color=color1, lw=1, linestyle='--')
            for i in range(stim_seq_short.shape[0]):
                ax.fill_between(
                    stim_seq_short[i,:],
                    lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
                    color=c_stim[i], edgecolor='none', alpha=0.25, step='mid')
            self.plot_vol(
                ax, self.alignment['stim_time'],
                stim_value_short.reshape(1,-1), color1, upper, lower)
        if 1 in oddball:
            c_idx = int(stim_seq_long.shape[0]/2)
            c_stim = [color2] * stim_seq_long.shape[-2]
            c_stim[c_idx] = color2
            ax.axvline(
                stim_seq_long[c_idx,1]+self.expect,
                color=color2, lw=1, linestyle='--')
            for i in range(stim_seq_long.shape[0]):
                ax.fill_between(
                    stim_seq_long[i,:],
                    lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
                    color=c_stim[i], edgecolor='none', alpha=0.25, step='mid')
            self.plot_vol(
                ax, self.alignment['stim_time'],
                stim_value_long.reshape(1,-1), color2, upper, lower)
        # plot neural traces.
        if 0 in oddball:
            self.plot_mean_sem(
                ax, self.alignment['neu_time'],
                mean_short, sem_short, color1, None)
        if 1 in oddball:
            self.plot_mean_sem(
                ax, self.alignment['neu_time'],
                mean_long, sem_long, color2, None)
        # adjust layout.
        adjust_layout_neu(ax)
        ax.set_xlim(xlim)
        ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
        ax.set_xlabel('time since stim before oddball interval (ms)')
        add_legend(ax, [color1,color2], ['short oddball','long oddball'],
                   n_trials, n_neurons, self.n_sess, 'upper right')

    def plot_oddball_fix_sync(self, ax, oddball, cate=None, roi_id=None):
        xlim = [-2000,5000]
        win_width = 200
        l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, 0, win_width)
        win_width = r_idx - l_idx
        # collect data.
        neu = []
        n_trials = 0
        if 0 in oddball:
            [[color0, color1, color2, _],
             [neu_short, _, stim_seq_short, stim_value_short], _,
             [nt, n_neurons]] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_idx=[l[0] for l in self.list_odd_idx],
                trial_param=[None, None, [0], None, [0], [0]],
                cate=cate, roi_id=None)
            sync_short = get_neu_sync(neu_short, win_width)
            neu.append(sync_short)
            n_trials += nt
        if 1 in oddball:
            [[color0, color1, color2, _],
             [neu_long, _, stim_seq_long, stim_value_long], _,
             [nt, n_neurons]] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_idx=[l[1] for l in self.list_odd_idx],
                trial_param=[None, None, [0], None,[0], [0]],
                cate=cate, roi_id=None)
            sync_long = get_neu_sync(neu_long, win_width)
            neu.append(sync_long)
            n_trials += nt
        # find bounds.
        upper = np.nanmax(neu)
        lower = np.nanmin(neu)
        # plot stimulus.
        if 0 in oddball:
            c_idx = int(stim_seq_short.shape[0]/2)
            c_stim = [color1] * stim_seq_short.shape[-2]
            c_stim[c_idx] = color1
            ax.axvline(
                stim_seq_short[c_idx,1]+self.expect,
                color=color1, lw=1, linestyle='--')
            for i in range(stim_seq_short.shape[0]):
                ax.fill_between(
                    stim_seq_short[i,:],
                    lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
                    color=c_stim[i], edgecolor='none', alpha=0.25, step='mid')
            self.plot_vol(
                ax, self.alignment['stim_time'],
                stim_value_short.reshape(1,-1), color1, upper, lower)
        if 1 in oddball:
            c_idx = int(stim_seq_long.shape[0]/2)
            c_stim = [color2] * stim_seq_long.shape[-2]
            c_stim[c_idx] = color2
            ax.axvline(
                stim_seq_long[c_idx,1]+self.expect,
                color=color2, lw=1, linestyle='--')
            for i in range(stim_seq_long.shape[0]):
                ax.fill_between(
                    stim_seq_long[i,:],
                    lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
                    color=c_stim[i], edgecolor='none', alpha=0.25, step='mid')
            self.plot_vol(
                ax, self.alignment['stim_time'],
                stim_value_long.reshape(1,-1), color2, upper, lower)
        # plot neural traces.
        if 0 in oddball:
            ax.plot(self.alignment['neu_time'][win_width:], sync_short, color=color1)
        if 1 in oddball:
            ax.plot(self.alignment['neu_time'][win_width:], sync_long,  color=color2)
        # adjust layout.
        adjust_layout_neu(ax)
        ax.set_xlim(xlim)
        ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
        ax.set_xlabel('time since stim before oddball interval (ms)')
        add_legend(ax, [color1,color2], ['short oddball','long oddball'],
                   n_trials, n_neurons, self.n_sess, 'upper right')
    
    def plot_oddball_fix_latent(self, ax, oddball, cate=None):
        # collect data.
        [[color0, color1, color2, _],
         [neu_x, _, stim_seq, stim_value], _,
         [n_trials, n_neurons]] = get_neu_trial(
            self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
            trial_idx=[l[oddball] for l in self.list_odd_idx],
            trial_param=[None, None, [0], None, [0], [0]],
            cate=cate, roi_id=None)
        cmap, c_neu = get_cmap_color(
            neu_x.shape[1], base_color=[color1, color2][oddball],
            return_cmap=True)
        c_idx = int(stim_seq.shape[0]/2)
        c_stim = [[color1, color2][oddball]] * stim_seq.shape[-2]
        c_stim[c_idx] = [color1, color2][1-oddball]
        # fit model.
        model = TSNE(n_components=self.d_latent)
        neu_z = model.fit_transform(
            neu_x.reshape(neu_x.shape[0],-1).T
            ).T.reshape(self.d_latent, -1)
        # plot dynamics.
        self.plot_3d_latent_dynamics(ax, neu_z, stim_seq, c_neu, c_stim)
        # adjust layout.
        adjust_layout_3d_latent(ax, neu_z, cmap, self.alignment['neu_time'], 'time since stim (ms)')
        add_legend(ax, [color1,color2], ['short oddball','long oddball'],
                   n_trials, n_neurons, self.n_sess, 'upper right', dim=3)
    
    def plot_oddball_jitter(self, ax, oddball, cate=None, roi_id=None):
        xlim = [-3000,3500]
        # collect data.
        [[color0, color1, color2, _],
         [neu_seq, stim_seq, stim_value, pre_isi], _,
         [n_trials, n_neurons]] = get_neu_trial(
            self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
            trial_idx=[l[oddball] for l in self.list_odd_idx],
            trial_param=[None, None, [1], None, [0], [0]],
            mean_sem=False,
            cate=cate, roi_id=roi_id)
        colors = get_cmap_color(self.bin_num, base_color=[color1, color2][oddball])
        # bin data based on isi.
        [bins, bin_center, _, _, bin_neu_mean, bin_neu_sem, bin_stim_seq, bin_stim_value] = get_isi_bin_neu(
            neu_seq, stim_seq, stim_value, pre_isi, self.bin_win, self.bin_num)
        # compute bounds.
        upper = np.nanmax(bin_neu_mean) + np.nanmax(bin_neu_sem)
        lower = np.nanmin(bin_neu_mean) - np.nanmax(bin_neu_sem)
        # only keep stimulus after bins.
        c_idx = int(bin_stim_seq.shape[1]/2)
        stim_seq = bin_stim_seq[:,c_idx-1:c_idx+2,:]
        stim_seq[:,1:,:] = np.nanmean(stim_seq[:,1:,:], axis=0)
        # plot stimulus.
        for i in range(self.bin_num):
            c_stim = [colors[i], [color1, color2][oddball], color0]
            for j in range(stim_seq.shape[1]):
                ax.fill_between(
                    stim_seq[i,j,:],
                    lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
                    color=c_stim[j], edgecolor='none', alpha=0.25, step='mid')
            self.plot_vol(ax, self.alignment['stim_time'], bin_stim_value[i,:].reshape(1,-1), colors[i], upper, lower)
        # plot neural traces.
        for i in range(self.bin_num):
            self.plot_mean_sem(ax, self.alignment['neu_time'], bin_neu_mean[i], bin_neu_sem[i], colors[i], None)
        # adjust layout.
        adjust_layout_neu(ax)
        ax.set_xlim(xlim)
        ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
        ax.set_xlabel('time since stim (ms)')
        lbl = ['[{},{}] ms'.format(int(bins[i]),int(bins[i+1])) for i in range(self.bin_num)]
        add_legend(
            ax, [color1, color2]+colors,
            ['pre oddball interval stim (short)', 'pre oddball interval stim (long)']+lbl,
            n_trials, n_neurons, self.n_sess, 'upper left')
    
    def plot_oddball_jitter_box(self, ax, oddball, cate=None, roi_id=None):
        win_base = [-self.bin_win[1],0]
        offsets = np.arange(self.bin_num)/20
        # collect data.
        [[color0, color1, color2, _],
         [neu_seq, stim_seq, stim_value, pre_isi], _,
         [n_trials, n_neurons]] = get_neu_trial(
            self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
            trial_idx=[l[oddball] for l in self.list_odd_idx],
            trial_param=[None, None, [1], None, [0], [0]],
            mean_sem=False,
            cate=cate, roi_id=roi_id)
        colors = get_cmap_color(self.bin_num, base_color=[color1, color2][oddball])
        # bin data based on isi.
        [bins, bin_center, _, bin_neu_seq, _, _, bin_stim_seq, _] = get_isi_bin_neu(
            neu_seq, stim_seq, stim_value, pre_isi, self.bin_win, self.bin_num)
        # only keep stimulus after bins.
        c_idx = int(bin_stim_seq.shape[1]/2)
        stim_seq = bin_stim_seq[:,c_idx-1:c_idx+2,:]
        c_time = np.nanmean(stim_seq[:,1:,:], axis=0)[1,0]
        # plot errorbar.
        for i in range(self.bin_num):
            self.plot_win_mag_box(
                ax, bin_neu_seq[i], self.alignment['neu_time'], win_base,
                colors[i], c_time, offsets[i])
        # adjust layout.
        lbl = ['[{},{}] ms'.format(int(bins[i]),int(bins[i+1])) for i in range(self.bin_num)]
        add_legend(ax, colors, lbl, n_trials, n_neurons, self.n_sess, 'upper right')
    
    def plot_oddball_share(self, ax, oddball, mode, cate=None, roi_id=None):
        isi_win = 100
        xlim = [-3000,4000]
        # collect data.
        n_trials = 0
        # fix.
        [[color0, color1, color2, _],
         [neu_fix, _, stim_seq_fix, stim_value_fix], _,
         [nt, n_neurons]] = get_neu_trial(
            self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
            trial_idx=[l[oddball] for l in self.list_odd_idx],
            trial_param=[None, None, [0], None, [0], [0]],
            cate=cate, roi_id=None)
        n_trials += nt
        mean_fix, sem_fix = get_mean_sem(neu_fix)
        # jitter.
        if mode == 'common':
            _, [neu_jitter, _, stim_value_jitter, pre_isi], _, _ = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_idx=[l[oddball] for l in self.list_odd_idx],
                trial_param=[None, None, [1], None, [0], [0]],
                mean_sem=False,
                cate=cate, roi_id=roi_id)
            idx = [(isi>np.nanmean(isi)-isi_win)*(isi<np.nanmean(isi)+isi_win) for isi in pre_isi]
            n_trials += np.sum(np.concatenate(idx))
            neu_jitter = [neu_jitter[i][idx[i],:,:] for i in range(self.n_sess)]
            neu_jitter = np.concatenate([np.nanmean(n, axis=0) for n in neu_jitter], axis=0)
            mean_jitter, sem_jitter = get_mean_sem(neu_jitter)
            stim_value_jitter = [stim_value_jitter[i][idx[i],:] for i in range(self.n_sess)]
            stim_value_jitter = np.nanmean(np.concatenate(stim_value_jitter,axis=0), axis=0)
            lbl = ['fix', r'jitter within $\pm{}$ ms'.format(isi_win)]
        if mode == 'all':
            _, [neu_jitter, _, stim_seq_jitter, stim_value_jitter], _, [nt, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_idx=[l[oddball] for l in self.list_odd_idx],
                trial_param=[None, None, [1], None, [0], [0]],
                cate=cate, roi_id=None)
            n_trials += nt
            mean_jitter, sem_jitter = get_mean_sem(neu_jitter)
            lbl = ['fix', 'jitter']
        # find bounds.
        upper = np.nanmax([mean_fix, mean_jitter]) + np.nanmax([sem_fix, sem_jitter])
        lower = np.nanmin([mean_fix, mean_jitter]) - np.nanmax([sem_fix, sem_jitter])
        # plot stimulus.
        c_fix = [color1, color2][oddball]
        c_idx = int(stim_seq_fix.shape[0]/2)
        c_stim = [c_fix] * stim_seq_fix.shape[-2]
        c_stim[c_idx] = c_fix
        ax.axvline(
            stim_seq_fix[c_idx,1]+self.expect,
            color=c_fix, lw=1, linestyle='--')
        for i in range(stim_seq_fix.shape[0]):
            ax.fill_between(
                stim_seq_fix[i,:],
                lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
                color=c_stim[i], edgecolor='none', alpha=0.25, step='mid')
        self.plot_vol(
            ax, self.alignment['stim_time'],
            stim_value_fix.reshape(1,-1), c_fix, upper, lower)
        self.plot_vol(
            ax, self.alignment['stim_time'],
            stim_value_jitter.reshape(1,-1), color0, upper, lower)
        # plot neural traces.
        self.plot_mean_sem(ax, self.alignment['neu_time'], mean_fix, sem_fix, c_fix, None)
        self.plot_mean_sem(ax, self.alignment['neu_time'], mean_jitter, sem_jitter, color0, None)
        # adjust layout.
        adjust_layout_neu(ax)
        ax.set_xlim(xlim)
        ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
        ax.set_xlabel('time since stim before oddball interval (ms)')
        add_legend(ax, [c_fix,color0], lbl, n_trials, n_neurons, self.n_sess, 'upper left')
        
    def plot_cluster(self, axs, cate=None):
        metrics, cluster_id = self.run_clustering(cate)
        colors = get_cmap_color(self.n_clusters, cmap=self.cluster_cmap)
        lbl = ['cluster #'+str(i) for i in range(self.n_clusters)]
        cmap_3d, _ = get_cmap_color(2, base_color='#2C2C2C', return_cmap=True)
        # plot basic clustering results.
        def plot_info(axs):
            # collect data.
            [color0, color1, color2, cmap], [neu_seq, _, _, _], [neu_labels, _], _ = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_param=[[2,3,4,5], [0], None, None, [0], [0]],
                cate=cate, roi_id=None)
            # plot info.
            self.plot_cluster_ca_transient(
                axs[0], colors, cluster_id, cate)
            self.plot_cluster_fraction(
                axs[1], colors, cluster_id)
            self.plot_cluster_cluster_fraction_in_cate(
                axs[2], colors, cluster_id, neu_labels, self.label_names)
            self.plot_cluster_cate_fraction_in_cluster(
                axs[3], cluster_id, neu_labels, self.label_names)
        # plot bin normalized interval tracking for all clusters.
        def plot_interval_norm(ax):
            # collect data.
            [_, [neu_seq, stim_seq, stim_value, pre_isi], _, [n_trials, n_neurons]] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_param=[None, None, None, None, None, [0]],
                mean_sem=False,
                cate=cate, roi_id=None)
            # bin data based on isi.
            [bins, _, _, bin_neu_seq, _, _, bin_stim_seq, _] = get_isi_bin_neu(
                neu_seq, stim_seq, stim_value, pre_isi, self.bin_win, self.bin_num)
            # plot results.
            lbl = ['[{},{}] ms'.format(int(bins[i]),int(bins[i+1])) for i in range(self.bin_num)]
            lbl+= ['cluster #'+str(i) for i in range(self.n_clusters)]
            c_all = get_cmap_color(self.bin_num, base_color='#2C2C2C') + colors
            self.plot_cluster_interval_norm(
                ax, cluster_id, bin_neu_seq, bin_stim_seq, colors)
            add_legend(ax, c_all, lbl, n_trials, n_neurons, self.n_sess, 'upper right')
        # plot fix standard response.
        def plot_standard_fix(ax):
            xlim = [-4000, 3000]
            # collect data.
            [color0, _, _, _], [neu_seq, _, stim_seq, stim_value], _, [n_trials, n_neurons] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance,
                [exclude_odd_stim(sl) for sl in self.list_stim_labels],
                trial_param=[[2,3,4,5], [0], [0], None, [0], [0]],
                cate=cate, roi_id=None)
            c_stim = [color0] * stim_seq.shape[-2]
            # get response within cluster.
            neu_mean, neu_sem = get_mean_sem_cluster(neu_seq, cluster_id)
            # plot results.
            norm_params = [get_norm01_params(neu_mean[i,:]) for i in range(neu_mean.shape[0])]
            self.plot_cluster_mean_sem(
                ax, neu_mean, neu_sem, self.alignment['neu_time'],
                norm_params, stim_seq, c_stim, colors, xlim)
            # adjust layout.
            ax.set_xlim(xlim)
            ax.set_xlabel('time since stim (ms)')
            add_legend(ax, [c_stim[0]]+colors, ['stim']+lbl, n_trials, n_neurons, self.n_sess, 'upper left')
        # plot fix oddball response.
        def plot_oddball_fix(ax, oddball):
            xlim = [-2500,5000]
            # collect data.
            [[color0, color1, color2, cmap],
             [neu_seq_fix, _, stim_seq_fix, stim_value_fix], _,
             [n_trials, n_neurons]] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_idx=[l[oddball] for l in self.list_odd_idx],
                trial_param=[None, None, [0], None, [0], [0]],
                cate=cate, roi_id=None)
            [[color0, color1, color2, cmap],
             [neu_seq_jitter, _, stim_seq_jitter, stim_value_jitter], _,
             [n_trials, n_neurons]] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_idx=[l[oddball] for l in self.list_odd_idx],
                trial_param=[None, None, [1], None, [0], [0]],
                cate=cate, roi_id=None)
            # get response within cluster.
            neu_mean_fix, neu_sem_fix = get_mean_sem_cluster(neu_seq_fix, cluster_id)
            neu_mean_jitter, neu_sem_jitter = get_mean_sem_cluster(neu_seq_jitter, cluster_id)
            # plot jitter.
            c_stim = ['white'] * stim_seq_fix.shape[-2]
            c_neu = ['#2C2C2C'] * len(colors)
            norm_params = [get_norm01_params(neu_mean_jitter[i,:]) for i in range(neu_mean_jitter.shape[0])]
            self.plot_cluster_mean_sem(
                ax, neu_mean_jitter, neu_sem_jitter, self.alignment['neu_time'],
                norm_params, stim_seq_jitter, c_stim, c_neu, xlim)
            # plot fix.
            c_idx = int(stim_seq_fix.shape[0]/2)
            c_stim = [color0] * stim_seq_fix.shape[-2]
            c_stim[c_idx] = [color1, color2][oddball]
            norm_params = [get_norm01_params(neu_mean_fix[i,:]) for i in range(neu_mean_fix.shape[0])]
            self.plot_cluster_mean_sem(
                ax, neu_mean_fix, neu_sem_fix, self.alignment['neu_time'],
                norm_params, stim_seq_fix, c_stim, colors, xlim)
            # adjust layout.
            ax.set_xlim(xlim)
            ax.set_xlabel('time since stim before oddball interval (ms)')
            add_legend(
                ax, [color1, color2]+colors+['#2C2C2C'],
                ['pre oddball interval stim (short)', 'pre oddball interval stim (long)']+lbl+['jitter'],
                n_trials, n_neurons, self.n_sess, 'upper left')
        # plot jitter standard response.
        def plot_standard_jitter(ax):
            xlim = [-4000, 3000]
            # collect data.
            _, [neu_seq, stim_seq, stim_value, pre_isi], _, [n_trials, n_neurons] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance,
                [exclude_odd_stim(sl) for sl in self.list_stim_labels],
                trial_param=[[2,3,4,5], [0], [1], None, [0], [0]],
                mean_sem=False,
                cate=cate, roi_id=None)
            # bin data based on isi.
            [bins, bin_center, _, bin_neu_seq, _, _, bin_stim_seq, _] = get_isi_bin_neu(
                neu_seq, stim_seq, stim_value, pre_isi, self.bin_win, self.bin_num)
            cluster_bin_neu_mean, cluster_bin_neu_sem = get_bin_mean_sem_cluster(bin_neu_seq, cluster_id)
            norm_params = [get_norm01_params(cluster_bin_neu_mean[:,i,:]) for i in range(self.n_clusters)]
            # get line colors for each cluster.
            cs = [get_cmap_color(self.bin_num, base_color=c) for c in colors]
            # convert to colors for each bin.
            cs = [[cs[i][j] for i in range(self.n_clusters)] for j in range(self.bin_num)]
            # get overall colors.
            cs_all = get_cmap_color(self.bin_num, base_color='silver')
            lbl = ['[{},{}] ms'.format(int(bins[i]),int(bins[i+1])) for i in range(self.bin_num)]
            # only keep 2 stimulus.
            c_idx = int(bin_stim_seq.shape[1]/2)
            stim_seq = bin_stim_seq[:,c_idx-1:c_idx+1,:]
            # plot results.
            for i in range(self.bin_num):
                self.plot_cluster_mean_sem(
                    ax, cluster_bin_neu_mean[i,:,:], cluster_bin_neu_sem[i,:,:], self.alignment['neu_time'],
                    norm_params, stim_seq[i,:,:], [cs_all[i]]*stim_seq.shape[-2], cs[i], xlim)
            # adjust layout.
            ax.set_xlim(xlim)
            ax.set_xlabel('time since stim (ms)')
            add_legend(ax, cs_all, lbl, n_trials, n_neurons, self.n_sess, 'upper right')
        # plot jitter oddball response.
        def plot_oddball_jitter(ax, oddball):
            xlim = [-2500,5000]
            # collect data.
            [[_, color1, color2, _],
             [neu_seq, stim_seq, stim_value, pre_isi], _,
             [n_trials, n_neurons]] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_idx=[l[oddball] for l in self.list_odd_idx],
                trial_param=[None, None, [1], None, [0], [0]],
                mean_sem=False,
                cate=cate, roi_id=None)
            # bin data based on isi.
            [bins, bin_center, _, bin_neu_seq, _, _, bin_stim_seq, _] = get_isi_bin_neu(
                neu_seq, stim_seq, stim_value, pre_isi, self.bin_win, self.bin_num)
            cluster_bin_neu_mean, cluster_bin_neu_sem = get_bin_mean_sem_cluster(bin_neu_seq, cluster_id)
            norm_params = [get_norm01_params(cluster_bin_neu_mean[:,i,:]) for i in range(self.n_clusters)]
            # get line colors for each cluster.
            cs = [get_cmap_color(self.bin_num, base_color=c) for c in colors]
            # convert to colors for each bin.
            cs = [[cs[i][j] for i in range(self.n_clusters)] for j in range(self.bin_num)]
            # get overall colors.
            cs_all = get_cmap_color(self.bin_num, base_color='silver')
            lbl = ['[{},{}] ms'.format(int(bins[i]),int(bins[i+1])) for i in range(self.bin_num)]
            # only keep stimulus after bins.
            c_idx = int(bin_stim_seq.shape[1]/2)
            stim_seq = bin_stim_seq[:,c_idx-1:c_idx+3,:]
            stim_seq[:,1:,:] = np.nanmean(stim_seq[:,1:,:], axis=0)
            # plot results.
            for i in range(self.bin_num):
                c_stim = [cs_all[i]]*stim_seq.shape[-2]
                c_stim[1] = [color1, color2][oddball]
                self.plot_cluster_mean_sem(
                    ax, cluster_bin_neu_mean[i,:,:], cluster_bin_neu_sem[i,:,:], self.alignment['neu_time'],
                    norm_params, stim_seq[i,:,:], c_stim, cs[i], xlim)
            # adjust layout.
            ax.set_xlim(xlim)
            ax.set_xlabel('time since stim (ms)')
            add_legend(
                ax, [color1, color2]+cs_all,
                ['pre oddball interval stim (short)', 'pre oddball interval stim (long)']+lbl,
                n_trials, n_neurons, self.n_sess, 'upper left')
        # plot fix standard response heatmap.
        def plot_heatmap_standard_fix(ax):
            xlim = [-4000, 3000]
            l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, xlim[0], xlim[1])
            # collect data.
            _, [neu_seq, _, _, _], _, _ = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance,
                [exclude_odd_stim(sl) for sl in self.list_stim_labels],
                trial_param=[[2,3,4,5], [0], [0], None, [0], [0]],
                cate=cate, roi_id=None)
            neu_seq = neu_seq[:,l_idx:r_idx]
            neu_time = self.alignment['neu_time'][l_idx:r_idx]
            # plot heatmap.
            self.plot_cluster_heatmap(ax, neu_seq, neu_time, neu_seq, cluster_id, colors)
        # plot fix oddball response heatmap.
        def plot_heatmap_oddball_fix(ax, oddball):
            xlim = [-2500,5000]
            l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, xlim[0], xlim[1])
            # collect data.
            _, [neu_seq, _, _, _], _, _ = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_idx=[l[oddball] for l in self.list_odd_idx],
                trial_param=[None, None, [0], None, [0], [0]],
                cate=cate, roi_id=None)
            neu_seq = neu_seq[:,l_idx:r_idx]
            neu_time = self.alignment['neu_time'][l_idx:r_idx]
            # plot heatmap.
            self.plot_cluster_heatmap(ax, neu_seq, neu_time, neu_seq, cluster_id, colors)
        # plot jitter standard response heatmap.
        def plot_heatmap_standard_jitter(ax):
            xlim = [-4000, 3000]
            l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, xlim[0], xlim[1])
            # collect data.
            _, [neu_seq, _, _, _], _, _ = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance,
                [exclude_odd_stim(sl) for sl in self.list_stim_labels],
                trial_param=[[2,3,4,5], [0], [1], None, [0], [0]],
                cate=cate, roi_id=None)
            neu_seq = neu_seq[:,l_idx:r_idx]
            neu_time = self.alignment['neu_time'][l_idx:r_idx]
            # plot heatmap.
            self.plot_cluster_heatmap(ax, neu_seq, neu_time, neu_seq, cluster_id, colors)
        # plot jitter oddball response heatmap.
        def plot_heatmap_oddball_jitter(ax, oddball):
            xlim = [-2500,5000]
            l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, xlim[0], xlim[1])
            # collect data.
            _, [neu_seq, _, _, _], _, _ = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_idx=[l[oddball] for l in self.list_odd_idx],
                trial_param=[None, None, [1], None, [0], [0]],
                cate=cate, roi_id=None)
            neu_seq = neu_seq[:,l_idx:r_idx]
            neu_time = self.alignment['neu_time'][l_idx:r_idx]
            # plot heatmap.
            self.plot_cluster_heatmap(ax, neu_seq, neu_time, neu_seq, cluster_id, colors)
        # plot all.
        try: plot_info(axs[:5])
        except: pass
        try: plot_interval_norm(axs[5])
        except: pass
        try: plot_standard_fix(axs[6])
        except: pass
        try: plot_oddball_fix(axs[7], 0)
        except: pass
        try: plot_oddball_fix(axs[8], 1)
        except: pass
        try: plot_standard_jitter(axs[9])
        except: pass
        try: plot_oddball_jitter(axs[10],0)
        except: pass
        try: plot_oddball_jitter(axs[11],1)
        except: pass
        try: plot_heatmap_standard_fix(axs[12])
        except: pass
        try: plot_heatmap_oddball_fix(axs[13], 0)
        except: pass
        try: plot_heatmap_oddball_fix(axs[14], 1)
        except: pass
        try: plot_heatmap_standard_jitter(axs[15])
        except: pass
        try: plot_heatmap_oddball_jitter(axs[16], 0)
        except: pass
        try: plot_heatmap_oddball_jitter(axs[17], 1)
        except: pass
    
    def plot_categorization_features(self, axs):
        cate = [-1,1,2]
        kernel_time, kernel_all, exp_var_all = self.run_glm(cate)
        _, cluster_id = clustering_neu_response_mode(kernel_all, self.n_clusters, None)
        #results_all = self.run_features_categorization(cate)
        #cluster_id = results_all['cluster_id'].to_numpy()
        lbl = ['cluster #'+str(ci) for ci in range(self.n_clusters)]
        colors = get_cmap_color(len(lbl), cmap=self.cluster_cmap)
        # collect data.
        [color0, _, _, _], _, [neu_labels, neu_sig], _ = get_neu_trial(
            self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
            trial_param=[[2,3,4,5], [0], [0], None, [0], [0]],
            cate=cate, roi_id=None)
        # plot basic statistics.
        def plot_info(axs):
            #self.plot_cluster_ca_transient(
            #    axs[0], colors, cluster_id, cate)
            self.plot_cluster_fraction(
                axs[1], colors, cluster_id)
            self.plot_cluster_cluster_fraction_in_cate(
                axs[2], colors, cluster_id, neu_labels, self.label_names)
            self.plot_cluster_cate_fraction_in_cluster(
                axs[3], cluster_id, neu_labels, self.label_names)
        # compare respons on fix and jitter.
        def plot_oddball_fix_jitter(axs, oddball):
            xlim = [-2500, 4000]
            l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, xlim[0], xlim[1])
            # collect data.
            neu_time = self.alignment['neu_time'][l_idx:r_idx]
            # fix oddball.
            [_, [neu_seq_fix, _, stim_seq_fix, stim_value_fix],
             [neu_labels, neu_sig], [n_trials_fix, n_neurons]] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_idx=[l[oddball] for l in self.list_odd_idx],
                trial_param=[None, None, [0], None, [0], [0]],
                cate=cate, roi_id=None)
            neu_seq_fix = neu_seq_fix[:,l_idx:r_idx]
            # jitter oddball.
            [_, [neu_seq_jitter, _, stim_seq_jitter, stim_value_jitter],
             [neu_labels, neu_sig], [n_trials_jitter, n_neurons]] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_idx=[l[oddball] for l in self.list_odd_idx],
                trial_param=[None, None, [1], None, [0], [0]],
                cate=cate, roi_id=None)
            neu_seq_jitter = neu_seq_jitter[:,l_idx:r_idx]
            n_trials = n_trials_fix + n_trials_jitter
            # plot results for each class.
            for ci in range(len(lbl)):
                neu_mean = []
                neu_sem = []
                # get class data.
                mean_fix, sem_fix = get_mean_sem(neu_seq_fix[cluster_id==ci,:])
                neu_mean.append(mean_fix)
                neu_sem.append(sem_fix)
                mean_jitter, sem_jitter = get_mean_sem(neu_seq_jitter[cluster_id==ci,:])
                neu_mean.append(mean_jitter)
                neu_sem.append(sem_jitter)
                # find bounds.
                upper = np.nanmax(neu_mean) + np.nanmax(neu_sem)
                lower = np.nanmin(neu_mean) - np.nanmax(neu_sem)
                # find class colors.
                cs = get_cmap_color(7, base_color=colors[ci])
                cs = [cs[1], cs[5]]
                # plot stimulus.
                for i in range(stim_seq_fix.shape[0]):
                    axs[ci].fill_between(
                        stim_seq_fix[i,:],
                        lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
                        color=color0, edgecolor='none', alpha=0.25, step='mid')
                # plot neural traces.
                self.plot_mean_sem(
                    axs[ci], neu_time,
                    mean_fix, sem_fix, cs[0], None)
                self.plot_mean_sem(
                    axs[ci], neu_time,
                    mean_jitter, sem_jitter, cs[1], None)
                # adjust layouts.
                adjust_layout_neu(axs[ci])
                axs[ci].set_xlim(xlim)
                axs[ci].set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
                axs[ci].set_xlabel('time since pre oddball stim (ms)')
                add_legend(axs[ci], cs, ['fix','jitter'],
                           n_trials, n_neurons, self.n_sess, 'upper right')
        # significance test between fix and jitter.
        def plot_sig_test_fix_jitter(ax, oddball):
            thres = 0.05
            time_eval = 500
            # collect data.
            [_, [neu_seq_fix_trial, stim_seq_fix, _, _],
             [neu_labels, neu_sig], [n_trials_fix, n_neurons]] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_idx=[l[oddball] for l in self.list_odd_idx],
                trial_param=[None, None, [0], None, [0], [0]],
                mean_sem=False,
                cate=cate, roi_id=None)
            [_, [neu_seq_jitter_trial, stim_seq_jitter, _, _],
             [neu_labels, neu_sig], [n_trials_jitter, n_neurons]] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_idx=[l[oddball] for l in self.list_odd_idx],
                trial_param=[None, None, [1], None, [0], [0]],
                mean_sem=False,
                cate=cate, roi_id=None)
            stim_seq_fix = np.mean(np.concatenate(stim_seq_fix, axis=0),axis=0)
            stim_seq_jitter = np.mean(np.concatenate(stim_seq_jitter, axis=0),axis=0)
            n_trials = n_trials_fix + n_trials_jitter
            # find time range to evaluate.
            c_idx = int(stim_seq_fix.shape[0]/2)
            if oddball == 0:
                l_time = stim_seq_fix[c_idx+1, 0]
                r_time = stim_seq_fix[c_idx+1, 0] + time_eval
            if oddball == 1:
                l_time = stim_seq_fix[c_idx, 1] + self.expect
                r_time = stim_seq_fix[c_idx+1, 0]
            l_idx, r_idx = get_frame_idx_from_time(
                self.alignment['neu_time'], 0, l_time, r_time)
            # wilcoxon test results.
            neu_seq_fix_trial = [ns[:,:,l_idx:r_idx] for ns in neu_seq_fix_trial]
            neu_seq_jitter_trial = [ns[:,:,l_idx:r_idx] for ns in neu_seq_jitter_trial]
            p_all, _ = run_wilcoxon_trial(neu_seq_fix_trial, neu_seq_jitter_trial, thres)
            # plot results for each class.
            for ci in range(len(lbl)):
                m,s = get_mean_sem(np.nanmin(p_all[cluster_id==ci],axis=1).reshape(-1,1))
                ax.errorbar(
                    ci, m, s,
                    color=colors[ci],
                    capsize=2, marker='o', linestyle='none',
                    markeredgecolor='white', markeredgewidth=0.1)
            # adjust layout.
            ax.tick_params(tick1On=False)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.set_xlabel('cluster id')
            ax.set_ylabel('minimum p value across time (mean$\pm$sem)')
            ax.set_xticks(np.arange(len(lbl)))
            ax.set_xticklabels(lbl, rotation='vertical')
            ax.set_xlim([-0.5,len(lbl)+2])
            add_legend(ax, colors, lbl, n_trials, n_neurons, self.n_sess, 'upper right')
        # quantification comparison between fix and jitter.
        def plot_oddball_fix_jitter_box(axs, oddball):
            list_target_metric = ['response_latency', 'peak_amp', 'ramp_slope', 'decay_rate']
            time_eval = 500
            # collect data.
            [_, [neu_seq_fix, _, stim_seq_fix, stim_value_fix],
             [neu_labels, neu_sig], [n_trials_fix, n_neurons]] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_idx=[l[oddball] for l in self.list_odd_idx],
                trial_param=[None, None, [0], None, [0], [0]],
                cate=cate, roi_id=None)
            [_, [neu_seq_jitter, _, stim_seq_jitter, stim_value_jitter],
             [neu_labels, neu_sig], [n_trials_jitter, n_neurons]] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_idx=[l[oddball] for l in self.list_odd_idx],
                trial_param=[None, None, [1], None, [0], [0]],
                cate=cate, roi_id=None)
            # find time range to evaluate.
            c_idx = int(stim_seq_fix.shape[0]/2)
            list_win_eval = [[stim_seq_fix[c_idx, 1], stim_seq_fix[c_idx+1, 0] + time_eval],
                             [stim_seq_fix[c_idx, 1], stim_seq_fix[c_idx+1, 0] + time_eval]]
            # get quantification results.
            list_metrics = get_all_metrics(
                [neu_seq_fix, neu_seq_jitter], self.alignment['neu_time'], list_win_eval)
            # plot results for each class.
            for mi in range(len(list_target_metric)):
                m = [lm[list_target_metric[mi]] for lm in list_metrics]
                self.plot_cluster_metric_box(
                    axs[mi], m, list_target_metric[mi], cluster_id, colors)
            
        # comparison between short and long preceeding interval.
        def plot_oddball_local_isi(axs, oddball):
            xlim = [-2500, 4000]
            l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, xlim[0], xlim[1])
            # collect data.
            neu_time = self.alignment['neu_time'][l_idx:r_idx]
            _, [neu_seq, stim_seq, stim_value, pre_isi], _, [n_trials, n_neurons] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_idx=[l[oddball] for l in self.list_odd_idx],
                trial_param=[None, None, [1], None, [0], [0]],
                mean_sem=False,
                cate=cate, roi_id=None)
            # bin data based on isi.
            [bins, bin_center, _, bin_neu_seq, _, _, bin_stim_seq, bin_stim_value] = get_isi_bin_neu(
                neu_seq, stim_seq, stim_value, pre_isi, self.bin_win, self.bin_num)
            cluster_bin_neu_mean, cluster_bin_neu_sem = get_bin_mean_sem_cluster(bin_neu_seq, cluster_id)
            cluster_bin_neu_mean = cluster_bin_neu_mean[:,:,l_idx:r_idx]
            cluster_bin_neu_sem  = cluster_bin_neu_sem[:,:,l_idx:r_idx]
            # plot results for each class.
            for ci in range(len(lbl)):
                # find bounds.
                upper = np.nanmax(cluster_bin_neu_mean[:,ci,:]) + np.nanmax(cluster_bin_neu_sem[:,ci,:])
                lower = np.nanmin(cluster_bin_neu_mean[:,ci,:]) - np.nanmax(cluster_bin_neu_sem[:,ci,:])
                # find class colors.
                cs = get_cmap_color(3, base_color=colors[ci])
                color1, color2 = cs[0], cs[-1]
                # only keep stimulus after bins.
                c_idx = int(bin_stim_seq.shape[1]/2)
                stim_seq = bin_stim_seq[:,c_idx-1:c_idx+2,:]
                stim_seq[:,1:,:] = np.nanmean(stim_seq[:,1:,:], axis=0)
                # plot stimulus.
                for i in range(stim_seq[0,:,:].shape[0]):
                    axs[ci].fill_between(
                        stim_seq[0,i,:],
                        lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
                        color=color1, edgecolor='none', alpha=0.25, step='mid')
                    axs[ci].fill_between(
                        stim_seq[1,i,:],
                        lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
                        color=color2, edgecolor='none', alpha=0.25, step='mid')
                self.plot_vol(
                    axs[ci], self.alignment['stim_time'],
                    bin_stim_value[0,:].reshape(1,-1), color1, upper, lower)
                self.plot_vol(
                    axs[ci], self.alignment['stim_time'],
                    bin_stim_value[1,:].reshape(1,-1), color2, upper, lower)
                # plot neural traces.
                self.plot_mean_sem(
                    axs[ci], neu_time,
                    cluster_bin_neu_mean[0,ci,:], cluster_bin_neu_sem[0,ci,:], color1, None)
                self.plot_mean_sem(
                    axs[ci], neu_time,
                    cluster_bin_neu_mean[1,ci,:], cluster_bin_neu_sem[1,ci,:], color2, None)
                # adjust layouts.
                adjust_layout_neu(axs[ci])
                axs[ci].set_xlim(xlim)
                axs[ci].set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
                axs[ci].set_xlabel('time since pre oddball stim (ms)')
                add_legend(axs[ci], [color1,color2],
                           ['[{},{}] ms'.format(int(bins[i]),int(bins[i+1])) for i in range(self.bin_num)],
                           n_trials, n_neurons, self.n_sess, 'upper right')
        # significance test between local isi.
        def plot_sig_test_local_isi(ax, oddball):
            thres = 0.05
            time_eval = 500
            # collect data.
            _, [neu_seq, stim_seq, stim_value, pre_isi], _, [n_trials, n_neurons] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_idx=[l[oddball] for l in self.list_odd_idx],
                trial_param=[None, None, [1], None, [0], [0]],
                mean_sem=False,
                cate=cate, roi_id=None)
            # bin data based on isi.
            [_, _, bin_neu_seq_trial, _, _, _, _, _] = get_isi_bin_neu(
                neu_seq, stim_seq, stim_value, pre_isi, self.bin_win, self.bin_num)
            stim_seq = np.mean(np.concatenate(stim_seq, axis=0),axis=0)
            # find time range to evaluate.
            c_idx = int(stim_seq.shape[0]/2)
            if oddball == 0:
                l_time = stim_seq[c_idx+1, 0]
                r_time = stim_seq[c_idx+1, 0] + time_eval
            if oddball == 1:
                l_time = stim_seq[c_idx, 1] + self.expect
                r_time = stim_seq[c_idx+1, 0]
            l_idx, r_idx = get_frame_idx_from_time(
                self.alignment['neu_time'], 0, l_time, r_time)
            # wilcoxon test results.
            neu_seq_short_bin_trial = [ns[:,:,l_idx:r_idx] for ns in bin_neu_seq_trial[0]]
            neu_seq_long_bin_trial = [ns[:,:,l_idx:r_idx] for ns in bin_neu_seq_trial[1]]
            p_all, _ = run_wilcoxon_trial(neu_seq_short_bin_trial, neu_seq_long_bin_trial, thres)
            # plot results for each class.
            for ci in range(len(lbl)):
                m,s = get_mean_sem(p_all[cluster_id==ci].reshape(-1,1))
                ax.errorbar(
                    ci, m, s,
                    color=colors[ci],
                    capsize=2, marker='o', linestyle='none',
                    markeredgecolor='white', markeredgewidth=0.1)
            # adjust layout.
            ax.tick_params(tick1On=False)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.set_xlabel('cluster id')
            ax.set_ylabel('minimum p value across time (mean$\pm$sem)')
            ax.set_xticks(np.arange(len(lbl)))
            ax.set_xticklabels(lbl, rotation='vertical')
            ax.set_xlim([-0.5,len(lbl)+2])
            add_legend(ax, colors, lbl, n_trials, n_neurons, self.n_sess, 'upper right')
        # quantification comparison between local isi.
        def plot_oddball_local_isi_box(axs, oddball):
            list_target_metric = ['response_latency', 'peak_amp', 'ramp_slope', 'decay_rate']
            time_eval = 500
            # collect data.
            _, [neu_seq, stim_seq, stim_value, pre_isi], _, [n_trials, n_neurons] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_idx=[l[oddball] for l in self.list_odd_idx],
                trial_param=[None, None, [1], None, [0], [0]],
                mean_sem=False,
                cate=cate, roi_id=None)
            # bin data based on isi.
            [_, _, _, bin_neu_seq, _, _, bin_stim_seq, _] = get_isi_bin_neu(
                neu_seq, stim_seq, stim_value, pre_isi, self.bin_win, self.bin_num)
            # find time range to evaluate.
            c_idx = int(bin_stim_seq.shape[1]/2)
            win_eval = [bin_stim_seq[0,c_idx, 1], bin_stim_seq[0,c_idx+1, 0] + time_eval]
            # get quantification results.
            list_metrics = get_all_metrics(
                bin_neu_seq, self.alignment['neu_time'], win_eval)
            # plot results for each class.
            for mi in range(len(list_target_metric)):
                m = [lm[list_target_metric[mi]] for lm in list_metrics]
                self.plot_cluster_metric_box(
                    axs[mi], m, list_target_metric[mi], cluster_id, colors)
        # plot all.
        try: plot_info(axs[0])
        except: pass
        try: plot_oddball_fix_jitter(axs[1], 0)
        except: pass
        try: plot_sig_test_fix_jitter(axs[2], 0)
        except: pass
        try: plot_oddball_fix_jitter_box(axs[3], 0)
        except: pass
        try: plot_oddball_fix_jitter(axs[4], 1)
        except: pass
        try: plot_sig_test_fix_jitter(axs[5], 1)
        except: pass
        try: plot_oddball_fix_jitter_box(axs[6], 1)
        except: pass
        try: plot_oddball_local_isi(axs[7], 0)
        except: pass
        try: plot_sig_test_local_isi(axs[8], 0)
        except: pass
        try: plot_oddball_local_isi_box(axs[9], 0)
        except: pass
        try: plot_oddball_local_isi(axs[10], 1)
        except: pass
        try: plot_sig_test_local_isi(axs[11], 1)
        except: pass
        try: plot_oddball_local_isi_box(axs[12], 1)
        except: pass
    
    def plot_sorted_heatmaps_fix_jitter(self, axs, norm_mode, oddball):
        thres = 0.05
        cate = [-1,1]
        win_sort = [-500, 500]
        xlim = [-2500, 5000]
        l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, xlim[0], xlim[1])
        # collect data.
        neu_x = []
        stim_x = []
        neu_time = self.alignment['neu_time'][l_idx:r_idx]
        # fix oddball.
        _, [neu_seq_fix, _, stim_seq_fix, _], [neu_labels, neu_sig], _ = get_neu_trial(
            self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
            trial_idx=[l[oddball] for l in self.list_odd_idx],
            trial_param=[None, None, [0], None, [0], [0]],
            cate=cate, roi_id=None)
        neu_seq_fix = neu_seq_fix[:,l_idx:r_idx]
        neu_x.append(neu_seq_fix)
        stim_x.append(stim_seq_fix)
        # jitter oddball.
        _, [neu_seq_jitter, _, stim_seq_jitter, _], _, _ = get_neu_trial(
            self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
            trial_idx=[l[oddball] for l in self.list_odd_idx],
            trial_param=[None, None, [1], None, [0], [0]],
            cate=cate, roi_id=None)
        neu_seq_jitter = neu_seq_jitter[:,l_idx:r_idx]
        neu_x.append(neu_seq_jitter)
        stim_x.append(stim_seq_jitter)
        # difference between fix and jitter.
        neu_x.append(neu_seq_fix-neu_seq_jitter)
        stim_x.append(stim_seq_jitter)
        # wilcoxon test results.
        _, [neu_seq_fix_trial, _, _, _], _, _ = get_neu_trial(
            self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
            trial_idx=[l[oddball] for l in self.list_odd_idx],
            trial_param=[None, None, [0], None, [0], [0]],
            mean_sem=False,
            cate=cate, roi_id=None)
        _, [neu_seq_jitter_trial, _, _, _], _, _ = get_neu_trial(
            self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
            trial_idx=[l[oddball] for l in self.list_odd_idx],
            trial_param=[None, None, [1], None, [0], [0]],
            mean_sem=False,
            cate=cate, roi_id=None)
        neu_seq_fix_trial = [ns[:,:,l_idx:r_idx] for ns in neu_seq_fix_trial]
        neu_seq_jitter_trial = [ns[:,:,l_idx:r_idx] for ns in neu_seq_jitter_trial]
        _, p_bin = run_wilcoxon_trial(neu_seq_fix_trial, neu_seq_jitter_trial, thres)
        neu_x.append(p_bin)
        stim_x.append(stim_seq_jitter)
        # plot heatmaps.
        for bi in range(len(neu_x)):
            self.plot_heatmap_neuron(
                axs[bi], neu_x[bi], neu_time, neu_x[0], win_sort, neu_labels, neu_sig,
                norm_mode=norm_mode if bi!=3 else 'binary',
                neu_seq_share=neu_x,
                neu_labels_share=[neu_labels]*len(neu_x),
                neu_sig_share=[neu_sig]*len(neu_x))
            axs[bi].set_xlabel('time since pre oddball stim (ms) \n sorting window [{},{}] ms'.format(
                win_sort[0], win_sort[1]))
        # add stimulus line.
        for bi in [0]:
            c_idx = int(stim_seq_fix.shape[0]/2)
            xlines = [neu_time[np.searchsorted(neu_time, t)]
                      for t in [stim_x[bi][c_idx+i,0]
                                for i in [-2,-1,0,1,2]]]
            for xl in xlines:
                if xl>neu_time[0] and xl<neu_time[-1]:
                    axs[bi].axvline(xl, color='black', lw=1, linestyle='--')
        for bi in [1,2,3]:
            c_idx = int(stim_seq_fix.shape[0]/2)
            xlines = [neu_time[np.searchsorted(neu_time, t)]
                      for t in [stim_x[bi][c_idx+i,0]
                                for i in [0,1]]]
            for xl in xlines:
                if xl>neu_time[0] and xl<neu_time[-1]:
                    axs[bi].axvline(xl, color='black', lw=1, linestyle='--')
    
    def plot_sorted_heatmaps_local_isi(self, axs, norm_mode, oddball):
        thres = 0.05
        cate = [-1,1]
        win_sort = [-500, 500]
        xlim = [-2500, 5000]
        l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, xlim[0], xlim[1])
        # collect data.
        neu_x = []
        stim_x = []
        neu_time = self.alignment['neu_time'][l_idx:r_idx]
        # oddball with bins.
        _, [neu_seq, stim_seq, stim_value, pre_isi], [neu_labels, neu_sig], _ = get_neu_trial(
            self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
            trial_idx=[l[oddball] for l in self.list_odd_idx],
            trial_param=[None, None, [1], None, [0], [0]],
            mean_sem=False,
            cate=cate, roi_id=None)
        neu_seq = [ns[:,:,l_idx:r_idx] for ns in neu_seq]
        [_, _, bin_neu_seq_trial, bin_neu_seq, _, _, bin_stim_seq, _] = get_isi_bin_neu(
            neu_seq, stim_seq, stim_value, pre_isi, self.bin_win, self.bin_num)
        neu_x += bin_neu_seq
        stim_x += [bin_stim_seq[bi,:,:] for bi in range(self.bin_num)]
        # difference between short and long bins.
        neu_x.append(bin_neu_seq[0][:,:]-bin_neu_seq[1][:,:])
        stim_x.append(bin_stim_seq[0,:,:])
        # wilcoxon test results.
        _, p_bin = run_wilcoxon_trial(bin_neu_seq_trial[0], bin_neu_seq_trial[1], thres)
        neu_x.append(p_bin)
        stim_x.append(bin_stim_seq[0,:,:])
        # plot heatmaps.
        for bi in range(len(neu_x)):
            self.plot_heatmap_neuron(
                axs[bi], neu_x[bi], neu_time, neu_x[0], win_sort, neu_labels, neu_sig,
                norm_mode=norm_mode if bi!=3 else 'binary',
                neu_seq_share=neu_x,
                neu_labels_share=[neu_labels]*len(neu_x),
                neu_sig_share=[neu_sig]*len(neu_x))
            axs[bi].set_xlabel('time since pre oddball stim (ms) \n sorting window [{},{}] ms'.format(
                win_sort[0], win_sort[1]))
        # add stimulus line.
        for bi in [0,1,2,3]:
            c_idx = int(bin_stim_seq[0,:,:].shape[0]/2)
            xlines = [neu_time[np.searchsorted(neu_time, t)]
                      for t in [stim_x[bi][c_idx+i,0]
                                for i in [0,1]]]
            for xl in xlines:
                if xl>neu_time[0] and xl<neu_time[-1]:
                    axs[bi].axvline(xl, color='black', lw=1, linestyle='--')
        
    def plot_cluster_latents(self, axs, cate=None):
        _, cluster_id = self.run_clustering(cate)
        self.plot_cluster_bin_3d_latents(axs, cluster_id, cate=cate)
    
    def plot_glm(self, axs, cate):
        kernel_time, kernel_all, exp_var_all = self.run_glm(cate)
        _, cluster_id = clustering_neu_response_mode(kernel_all, self.n_clusters, None)
        # collect data.
        colors = get_cmap_color(self.n_clusters, cmap=self.cluster_cmap)
        [[color0, _, _, _],
         [_, _, stim_seq, _],
         [neu_labels, neu_sig],
         [n_trials, n_neurons]] = get_neu_trial(
            self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
            trial_param=[None, None, None, None, None, [0]],
            cate=cate, roi_id=None)
        # plot bin normalized interval tracking for all clusters.
        def plot_interval_norm(ax):
            # collect data.
            [_, [neu_seq, stim_seq, stim_value, pre_isi], _, [n_trials, n_neurons]] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_param=[None, None, None, None, None, [0]],
                mean_sem=False,
                cate=cate, roi_id=None)
            # bin data based on isi.
            [bins, _, _, bin_neu_seq, _, _, bin_stim_seq, _] = get_isi_bin_neu(
                neu_seq, stim_seq, stim_value, pre_isi, self.bin_win, self.bin_num)
            # plot results.
            lbl = ['[{},{}] ms'.format(int(bins[i]),int(bins[i+1])) for i in range(self.bin_num)]
            lbl+= ['cluster #'+str(i) for i in range(self.n_clusters)]
            c_all = get_cmap_color(self.bin_num, base_color='#2C2C2C') + colors
            self.plot_cluster_interval_norm(
                ax, cluster_id, bin_neu_seq, bin_stim_seq, colors)
            add_legend(ax, c_all, lbl, n_trials, n_neurons, self.n_sess, 'upper right')
        # plot all.
        try: self.plot_glm_heatmap(
            axs[0], kernel_time, kernel_all, stim_seq,
            neu_labels, neu_sig)
        except: pass
        try: self.plot_glm_kernel_cluster(
            axs[1], kernel_time, kernel_all, cluster_id,
            stim_seq, color0, colors)
        except: pass
        try: plot_interval_norm(axs[2])
        except: pass
    
# colors = ['#989A9C', '#A4CB9E', '#9DB4CE', '#EDA1A4', '#F9C08A']
class plotter_main(plotter_utils):
    def __init__(self, neural_trials, labels, significance, label_names):
        super().__init__(neural_trials, labels, significance)
        self.label_names = label_names
    
    def oddball_fix(self, axs_all):
        for cate, axs in zip([[-1],[1]], axs_all):
            label_name = self.label_names[str(cate[0])] if len(cate)==1 else 'all'
            try:
            
                self.plot_oddball_fix(axs[0], [0], cate=cate)
                axs[0].set_title(f'response to oddball interval \n (fix, short oddball) {label_name}')
                
                self.plot_oddball_fix(axs[1], [1], cate=cate)
                axs[1].set_title(f'response to oddball interval \n (fix, long oddball) {label_name}')
                
                self.plot_oddball_fix(axs[2], [0,1], cate=cate)
                axs[2].set_title(f'response to oddball interval \n (fix) {label_name}')
                
                self.plot_oddball_fix_sync(axs[3], [0], cate=cate)
                axs[3].set_title(f'synchronization to oddball interval \n (fix, short oddball) {label_name}')
                
                self.plot_oddball_fix_sync(axs[4], [1], cate=cate)
                axs[4].set_title(f'synchronization to oddball interval \n (fix, long oddball) {label_name}')
                
                self.plot_oddball_fix_sync(axs[5], [0,1], cate=cate)
                axs[5].set_title(f'synchronization to oddball interval \n (fix) {label_name}')
                
                self.plot_oddball_fix_latent(axs[6], 0, cate=cate)
                axs[6].set_title(f'latent dynamics to oddball interval \n (fix, short oddball) {label_name}')
                
                self.plot_oddball_fix_latent(axs[7], 1, cate=cate)
                axs[7].set_title(f'latent dynamics to oddball interval \n (fix, long oddball) {label_name}')
                
            except: pass

    def oddball_jitter(self, axs_all):
        for cate, axs in zip([[-1],[1]], axs_all):
            label_name = self.label_names[str(cate[0])] if len(cate)==1 else 'all'
            try:
            
                self.plot_oddball_jitter(axs[0], 0, cate=cate)
                axs[0].set_title(f'response to oddball interval \n (jitter, short oddball) {label_name}')
                
                self.plot_oddball_jitter_box(axs[1], 0, cate=cate)
                axs[1].set_title(f'response to post oddball interval stim \n (jitter, short oddball) {label_name}')
                
                self.plot_oddball_jitter(axs[2], 1, cate=cate)
                axs[2].set_title(f'response to oddball interval \n (jitter, long oddball) {label_name}')
                
                self.plot_oddball_jitter_box(axs[3], 0, cate=cate)
                axs[3].set_title(f'response to post oddball interval stim \n (jitter, long oddball) {label_name}')
                
                self.plot_oddball_share(axs[4], 0, 'common', cate=cate)
                axs[4].set_title(f'response to oddball interval across blocks \n (common preceding interval, short oddball) {label_name}')
                
                self.plot_oddball_share(axs[5], 1, 'common', cate=cate)
                axs[5].set_title(f'response to oddball interval across blocks \n (common preceding interval, long oddball) {label_name}')
                
                self.plot_oddball_share(axs[6], 0, 'all', cate=cate)
                axs[6].set_title(f'response to oddball interval across blocks \n (all preceding interval, short oddball) {label_name}')
                
                self.plot_oddball_share(axs[7], 1, 'all', cate=cate)
                axs[7].set_title(f'response to oddball interval across blocks \n (all preceding interval, long oddball) {label_name}')
                
            except: pass

    def cluster(self, axs_all):
        for cate, axs in zip([[-1],[1],[-1,1]], axs_all):
            label_name = self.label_names[str(cate[0])] if len(cate)==1 else 'all'
            try:
                
                self.plot_cluster(axs, cate=cate)
                axs[ 0].set_title(f'sorted correlation matrix \n (short) {label_name}')
                axs[ 1].set_title(f'clustering evaluation metrics \n (short) {label_name}')
                axs[ 2].set_title(f'cluster calcium transient \n (short) {label_name}')
                axs[ 3].set_title(f'cluster fraction \n {label_name}')
                axs[ 4].set_title(f'cluster fraction for subtypes \n {label_name}')
                axs[ 5].set_title(f'time normalized reseponse to preceeding interval with bins \n {label_name}')
                axs[ 6].set_title(f'response to standard interval \n (fix) {label_name}')
                axs[ 7].set_title(f'response to oddball interval \n (fix, short oddball) {label_name}')
                axs[ 8].set_title(f'response to oddball interval \n (fix, long oddball) {label_name}')
                axs[ 9].set_title(f'response to standard interval \n (jitter) {label_name}')
                axs[10].set_title(f'response to oddball interval \n (jitter, short oddball) {label_name}')
                axs[11].set_title(f'response to oddball interval \n (jitter, long oddball) {label_name}')
                axs[12].set_title(f'response to standard interval \n (fix) {label_name}')
                axs[13].set_title(f'response to oddball interval \n (fix, short oddball) {label_name}')
                axs[14].set_title(f'response to oddball interval \n (fix, long oddball) {label_name}')
                axs[15].set_title(f'response to standard interval \n (jitter) {label_name}')
                axs[16].set_title(f'response to oddball interval \n (jitter, short oddball) {label_name}')
                axs[17].set_title(f'response to oddball interval \n (jitter, long oddball) {label_name}')
    
            except: pass
        
    def categorization_features(self, axs_all):
        for cate, axs in zip([-1,1,2], axs_all):
            try:
            
                self.plot_categorization_features(axs)
                for i in range(len(axs[[1]])):
                    axs[1][i].set_title(f'response to oddball interval comparison for cluster #{i} \n (between fix and jitter, short oddball)')
                for i in range(len(axs[[2]])):
                    axs[2][i].set_title(f'response to oddball interval comparison for cluster #{i} \n (between fix and jitter, long oddball)')
                axs[3][0].set_title('significance test across trials \n (between fix and jitter, short oddball)')
                axs[3][1].set_title('significance test across trials \n (between fix and jitter, long oddball)')
                for i in range(len(axs[[4]])):
                    axs[4][i].set_title(f'response to oddball interval comparison for cluster #{i} \n (between short and long preceeding isi, short oddball)')
                for i in range(len(axs[[5]])):
                    axs[5][i].set_title(f'response to oddball interval comparison for cluster #{i} \n (between short and long preceeding isi, long oddball)')
                axs[6][0].set_title('significance test across trials \n (between short and long preceeding isi, short oddball)')
                axs[6][1].set_title('significance test across trials \n (between short and long preceeding isi, long oddball)')
        
            except: pass

    def sorted_heatmaps_fix_jitter(self, axs_all):
        for norm_mode, axs in zip(['none', 'minmax', 'share'], axs_all):
            try:
                
                self.plot_sorted_heatmaps_fix_jitter(axs[0], norm_mode, 0)
                axs[0][0].set_title(f'response to short oddball interval \n (fix, normalized with {norm_mode})')
                axs[0][1].set_title(f'response to short oddball interval \n (jitter, normalized with {norm_mode})')
                axs[0][2].set_title(f'response to short oddball interval \n (fix minors jitter, normalized with {norm_mode})')
                axs[0][3].set_title('significance test across trials')

                self.plot_sorted_heatmaps_fix_jitter(axs[1], norm_mode, 1)
                axs[1][0].set_title(f'response to long oddball interval \n (fix, normalized with {norm_mode})')
                axs[1][1].set_title(f'response to long oddball interval \n (jitter, normalized with {norm_mode})')
                axs[1][2].set_title(f'response to long oddball interval \n (fix minors jitter, normalized with {norm_mode})')
                axs[1][3].set_title('significance test across trials')

            except: pass
    
    def sorted_heatmaps_local_isi(self, axs_all):
        for norm_mode, axs in zip(['none', 'minmax', 'share'], axs_all):
            try:
                
                self.plot_sorted_heatmaps_local_isi(axs[0], norm_mode, 0)
                axs[0][0].set_title(f'response to short oddball interval \n (jitter with short preceeding interval, normalized with {norm_mode})')
                axs[0][1].set_title(f'response to short oddball interval \n (jitter with long preceeding interval, normalized with {norm_mode})')
                axs[0][2].set_title(f'response to short oddball interval \n (short minors long preceeding interval, normalized with {norm_mode})')
                axs[0][3].set_title('significance test across trials')

                self.plot_sorted_heatmaps_local_isi(axs[1], norm_mode, 1)
                axs[1][0].set_title(f'response to long oddball interval \n (jitter with short preceeding interval, normalized with {norm_mode})')
                axs[1][1].set_title(f'response to long oddball interval \n (jitter with long preceeding interval, normalized with {norm_mode})')
                axs[1][2].set_title(f'response to long oddball interval \n (short minors long preceeding interval, normalized with {norm_mode})')
                axs[1][3].set_title('significance test across trials')

            except: pass
    
    def glm(self, axs_all):
        for cate, axs in zip([[-1],[1],[-1,1]], axs_all):
            label_name = self.label_names[str(cate[0])] if len(cate)==1 else 'all'
            try:
                
                self.plot_glm(axs, cate=cate)
                axs[0].set_title(f'GLM kernel weights heatmap \n {label_name}')
                axs[1].set_title(f'clustered GLM kernel weights \n {label_name}')
                axs[2].set_title(f'clustered GLM kernel weights \n {label_name}')
                axs[3].set_title(f'time normalized reseponse to preceeding interval with bins \n {label_name}')
                
            except: pass
    
