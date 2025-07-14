#!/usr/bin/env python3

import numpy as np
import matplotlib.ticker as mtick
from sklearn.decomposition import PCA

from modules.Alignment import run_get_stim_response
from modeling.clustering import get_mean_sem_cluster
from modeling.clustering import get_bin_mean_sem_cluster
from modeling.clustering import get_cluster_cate
from modeling.decoding import neu_pop_sample_decoding_slide_win
from modeling.decoding import multi_sess_decoding_time_eclapse
from modeling.decoding import multi_sess_decoding_slide_win
from modeling.generative import get_glm_cate
from modeling.quantifications import run_quantification
from utils import show_resource_usage
from utils import get_norm01_params
from utils import get_odd_stim_prepost_idx
from utils import get_mean_sem
from utils import get_mean_sem_win
from utils import get_neu_trial
from utils import get_frame_idx_from_time
from utils import get_isi_bin_neu
from utils import get_expect_interval
from utils import get_split_idx
from utils import get_block_1st_idx
from utils import get_cmap_color
from utils import hide_all_axis
from utils import adjust_layout_neu
from utils import adjust_layout_2d_latent
from utils import adjust_layout_3d_latent
from utils import add_legend
from utils import utils_basic

# fig, ax = plt.subplots(1, 1, figsize=(5, 5))
# fig, ax = plt.subplots(1, 1, figsize=(3, 9))
# fig, axs = plt.subplots(1, 6, figsize=(18, 3))
# fig, ax = plt.subplots(1, 1, figsize=(6, 6), subplot_kw={"projection": "3d"})
# cluster_id = run_clustering()

class plotter_utils(utils_basic):

    def __init__(
            self,
            list_neural_trials, list_labels, list_significance,
            temp_folder, cate_list
            ):
        super().__init__()
        self.cate_list = cate_list
        self.n_sess = len(list_neural_trials)
        self.list_labels = list_labels
        self.list_neural_trials = list_neural_trials
        self.alignment = run_get_stim_response(temp_folder, list_neural_trials, expected='none')
        self.list_stim_labels = self.alignment['list_stim_labels']
        self.list_odd_idx = [
            get_odd_stim_prepost_idx(sl) for sl in self.list_stim_labels]
        self.expect = np.array(np.mean([get_expect_interval(sl)[0] for sl in self.list_stim_labels]))
        self.list_block_start = [get_block_1st_idx(sl, 3) for sl in self.list_stim_labels]
        self.list_significance = list_significance
        self.bin_win = [450,2550]
        self.bin_num = 2
        self.n_clusters = 9
        self.max_clusters = 10
        self.d_latent = 3
        self.glm = self.run_glm()
        self.cluster_id = self.run_clustering()

    def get_neu_seq_trial_fix_jitter(self, jitter_trial_mode, oddball, cate, isi_win):
        # jitter oddball.  
        [_, [neu_seq_jitter, _, _, pre_isi, _], _, _] = get_neu_trial(
            self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
            trial_idx=[l[oddball] for l in self.list_odd_idx],
            trial_param=[None, None, [1], None, [0], [0]],
            mean_sem=False,
            cate=cate, roi_id=None)
        # fix oddball.
        [_, [neu_seq_fix, _, _, _, _], _, _] = get_neu_trial(
            self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
            trial_idx=[l[oddball] for l in self.list_odd_idx],
            trial_param=[None, None, [0], None, [0], [0]],
            mean_sem=False,
            cate=cate, roi_id=None)
        if jitter_trial_mode == 'global':
            idx_jitter = [isi>0 for isi in pre_isi]
        if jitter_trial_mode == 'similar':
            idx_jitter = [(isi>np.nanmean(isi)-isi_win)*(isi<np.nanmean(isi)+isi_win) for isi in pre_isi]
        if jitter_trial_mode == 'lower':
            idx_jitter = [(isi<np.nanmean(isi)-isi_win) for isi in pre_isi]
        if jitter_trial_mode == 'higher':
            idx_jitter = [(isi>np.nanmean(isi)+isi_win) for isi in pre_isi]
        neu_seq_jitter = [neu_seq_jitter[i][idx_jitter[i],:,:] for i in range(self.n_sess)]
        return neu_seq_fix, neu_seq_jitter

    def plot_cluster_oddball_fix_all(self, axs, cate):
        color0 = 'dimgrey'
        color1 = 'hotpink'
        color2 = 'darkviolet'
        xlim = [-2500, 4000]
        kernel_all = get_glm_cate(self.glm, self.list_labels, self.list_significance, cate)
        cluster_id, neu_labels = get_cluster_cate(self.cluster_id, self.list_labels, self.list_significance, cate)
        @show_resource_usage
        def plot_glm_kernel(ax):
            ax.axis('off')
            ax = ax.inset_axes([0, 0, 0.5, 1], transform=ax.transAxes)
            # collect data.
            [_, [_, _, stim_seq, _], _, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_param=[[2,3,4,5], None, [0], None, [0], [0]],
                cate=cate, roi_id=None)
            c_idx = stim_seq.shape[0]//2
            # get cluster average.
            glm_mean, glm_sem = get_mean_sem_cluster(kernel_all, self.n_clusters, cluster_id)
            # plot results.
            norm_params = [get_norm01_params(glm_mean[ci,:]) for ci in range(self.n_clusters)]
            self.plot_cluster_mean_sem(
                ax, glm_mean, glm_sem, self.glm['kernel_time'],
                norm_params, stim_seq[c_idx,:].reshape(1,-1),
                [color0], [color0]*self.n_clusters,
                [np.nanmin(self.glm['kernel_time']), np.nanmax(self.glm['kernel_time'])])
            # adjust layouts.
            ax.set_xlabel('time since stim (ms)')
        @show_resource_usage
        def plot_standard_fix(ax):
            # collect data.
            [_, [neu_seq, _, stim_seq, _],
             [neu_labels, neu_sig], _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_param=[[2,3,4,5], None, [0], None, [0], [0]],
                cate=cate, roi_id=None)
            # get response within cluster at each bin.
            neu_mean, neu_sem = get_mean_sem_cluster(neu_seq, self.n_clusters, cluster_id)
            norm_params = [get_norm01_params(neu_mean[i,:]) for i in range(self.n_clusters)]
            # plot results.
            self.plot_cluster_mean_sem(
                ax, neu_mean, neu_sem,
                self.alignment['neu_time'], norm_params,
                stim_seq,
                [color0]*stim_seq.shape[0], [color0]*self.n_clusters, xlim)
            # adjust layouts.
            ax.set_xlabel('time since stim (ms)')
        @show_resource_usage
        def plot_oddball_fix(ax, oddball):
            # collect data.
            [_, [neu_seq, _, stim_seq, _], _, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_idx=[l[oddball] for l in self.list_odd_idx],
                trial_param=[None, None, [0], None, [0], [0]],
                cate=cate, roi_id=None)
            # get response within cluster at each bin.
            neu_mean, neu_sem = get_mean_sem_cluster(neu_seq, self.n_clusters, cluster_id)
            norm_params = [get_norm01_params(neu_mean[i,:]) for i in range(self.n_clusters)]
            # plot results.
            c_idx = stim_seq.shape[0]//2
            if oddball == 0:
                ax.axvline(stim_seq[c_idx+1,0], color='gold', lw=1, linestyle='--')
            if oddball == 1:
                ax.axvline(stim_seq[c_idx,1]+self.expect, color='gold', lw=1, linestyle='--')
            self.plot_cluster_mean_sem(
                ax, neu_mean, neu_sem,
                self.alignment['neu_time'], norm_params,
                stim_seq,
                [color0]*stim_seq.shape[0], [[color1,color2][oddball]]*self.n_clusters, xlim)
            # adjust layouts.
            ax.set_xlabel('time since pre oddball stim (ms)')
        @show_resource_usage
        def plot_oddball_fix_quant(ax):
            win = 300
            # collect data.
            [_, [neu_seq_standard, _, stim_seq_standard, _], _, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_param=[[2,3,4,5], None, [0], None, [0], [0]],
                cate=cate, roi_id=None)
            [_, [neu_seq_short, _, stim_seq_short, _], _, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_idx=[l[0] for l in self.list_odd_idx],
                trial_param=[None, None, [0], None, [0], [0]],
                cate=cate, roi_id=None)
            [_, [neu_seq_long, _, stim_seq_long, _], _, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_idx=[l[1] for l in self.list_odd_idx],
                trial_param=[None, None, [0], None, [0], [0]],
                cate=cate, roi_id=None)
            c_idx = stim_seq_standard.shape[0]//2
            # define layouts.
            ax.axis('off')
            ax = ax.inset_axes([0, 0, 1, 0.95], transform=ax.transAxes)
            axs = [ax.inset_axes([0.3, ci/self.n_clusters, 0.4, 0.7/self.n_clusters], transform=ax.transAxes)
                      for ci in range(self.n_clusters)]
            axs.reverse()
            # plot results for each class.
            for ci in range(self.n_clusters):
                if np.sum(cluster_id==ci) > 0:
                    quant_standard = get_mean_sem_win(
                        neu_seq_standard[cluster_id==ci,:],
                        self.alignment['neu_time'], 0, 0, win, mode='mean')
                    quant_short = get_mean_sem_win(
                        neu_seq_short[cluster_id==ci,:],
                        self.alignment['neu_time'], 0, stim_seq_short[c_idx+1,0], stim_seq_short[c_idx+1,0]+win, mode='mean')
                    quant_long = get_mean_sem_win(
                        neu_seq_long[cluster_id==ci,:],
                        self.alignment['neu_time'], 0, stim_seq_long[c_idx+1,0], stim_seq_long[c_idx+1,0]+win, mode='mean')
                    axs[ci].errorbar(
                        1, quant_standard[1], quant_standard[2], None,
                        color=color0,
                        capsize=2, marker='o', linestyle='none',
                        markeredgecolor='white', markeredgewidth=0.1)
                    axs[ci].errorbar(
                        2, quant_short[1], quant_short[2], None,
                        color=color1,
                        capsize=2, marker='o', linestyle='none',
                        markeredgecolor='white', markeredgewidth=0.1)
                    axs[ci].errorbar(
                        3, quant_long[1], quant_long[2], None,
                        color=color2,
                        capsize=2, marker='o', linestyle='none',
                        markeredgecolor='white', markeredgewidth=0.1)
                    # adjust layouts.
                    axs[ci].tick_params(direction='in')
                    axs[ci].spines['right'].set_visible(False)
                    axs[ci].spines['top'].set_visible(False)
                    axs[ci].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
                    axs[ci].yaxis.set_major_locator(mtick.MaxNLocator(nbins=2))
                    axs[ci].set_xlim([0.5, 3.5])
                    axs[ci].set_xticks([])
            hide_all_axis(ax)
            ax.set_ylabel(f'response magnitude in {win} ms window since stimulus onset after interval \n')
            axs[self.n_clusters-1].tick_params(axis='x', labelrotation=90)
            axs[self.n_clusters-1].set_xticks([1,2,3])
            axs[self.n_clusters-1].set_xticklabels(['standard', 'short oddball', 'long oddball'])
        @show_resource_usage
        def plot_neu_fraction(ax):
            # plot results.
            self.plot_cluster_neu_fraction_in_cluster(ax, cluster_id, color0)
        @show_resource_usage
        def plot_cate_fraction(ax):
            # plot results.
            self.plot_cluster_cate_fraction_in_cluster(ax, cluster_id, neu_labels, self.label_names, color0)
        @show_resource_usage
        def plot_legend(ax):
            [[color0, color1, color2, _], _, _,
             [n_trials, n_neurons]] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_param=[[-1], None, [0], None, [0], [0]],
                cate=cate, roi_id=None)
            cs = [color0, color1, color2, 'gold']
            lbl = ['standard', 'short oddball', 'long oddball', 'unexpected event']
            add_legend(ax, cs, lbl, n_trials, n_neurons, self.n_sess, 'upper right')
            ax.axis('off')
        # plot all.
        try: plot_glm_kernel(axs[0])
        except Exception as e: print(e)
        try: plot_standard_fix(axs[1])
        except Exception as e: print(e)
        try: plot_oddball_fix(axs[2], 0)
        except Exception as e: print(e)
        try: plot_oddball_fix(axs[3], 1)
        except Exception as e: print(e)
        try: plot_oddball_fix_quant(axs[4])
        except Exception as e: print(e)
        try: plot_neu_fraction(axs[5])
        except Exception as e: print(e)
        try: plot_cate_fraction(axs[6])
        except Exception as e: print(e)
        try: plot_legend(axs[7])
        except Exception as e: print(e)
    
    def plot_cluster_oddball_jitter_global_all(self, axs, jitter_trial_mode, cate):
        win_lbl = ['early', 'late', 'post']
        isi_win = 250
        xlim = [-2500, 4000]
        l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, xlim[0], xlim[1])
        kernel_all = get_glm_cate(self.glm, self.list_labels, self.list_significance, cate)
        cluster_id, neu_labels = get_cluster_cate(self.cluster_id, self.list_labels, self.list_significance, cate)
        split_idx = get_split_idx(self.list_labels, self.list_significance, cate)
        day_cluster_id = np.split(cluster_id, split_idx)
        @show_resource_usage
        def plot_glm_kernel(ax):
            # collect data.
            [[color0, _, _, _], [_, _, stim_seq, _], _, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_param=[[2,3,4,5], None, [0], None, [0], [0]],
                cate=cate, roi_id=None)
            c_idx = stim_seq.shape[0]//2
            # get cluster average.
            glm_mean, glm_sem = get_mean_sem_cluster(kernel_all, self.n_clusters, cluster_id)
            # plot results.
            ax.axis('off')
            ax = ax.inset_axes([0, 0, 0.5, 0.95], transform=ax.transAxes)
            norm_params = [get_norm01_params(glm_mean[i,:]) for i in range(self.n_clusters)]
            self.plot_cluster_mean_sem(
                ax, glm_mean, glm_sem, self.glm['kernel_time'],
                norm_params, stim_seq[c_idx,:].reshape(1,-1),
                [color0], [color0]*self.n_clusters,
                [np.nanmin(self.glm['kernel_time']), np.nanmax(self.glm['kernel_time'])])
            # adjust layouts.
            ax.set_xlabel('time since stim (ms)')
        @show_resource_usage
        def plot_random_bin(ax, mode):
            # collect data.
            [[color0, _, color2, _],
             [neu_seq, stim_seq, camera_pupil, pre_isi, post_isi],
             [neu_labels, _],
             [n_trials, n_neurons]] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_param=[[2,3,4,5], None, None, None, [1], [0]],
                mean_sem=False,
                cate=cate, roi_id=None)
            colors = get_cmap_color(self.bin_num, base_color=color2)
            # bin data based on isi.
            if mode == 'pre':
                xl = [-3500, 1500]
                isi = pre_isi
                isi_idx_offset = -1
            if mode == 'post':
                xl = [-1000, 4000]
                isi = post_isi
                isi_idx_offset = 1
            [bins, bin_center, _, bin_neu_seq, _, _, bin_stim_seq, _] = get_isi_bin_neu(
                neu_seq, stim_seq, camera_pupil, isi, self.bin_win, self.bin_num)
            c_idx = bin_stim_seq.shape[1]//2
            # get response within cluster at each bin.
            cluster_bin_neu_mean, cluster_bin_neu_sem = get_bin_mean_sem_cluster(bin_neu_seq, self.n_clusters, cluster_id)
            norm_params = [get_norm01_params(cluster_bin_neu_mean[:,i,:]) for i in range(self.n_clusters)]
            # plot results.
            ax.axis('off')
            ax = ax.inset_axes([0, 0, 1, 0.95], transform=ax.transAxes)
            ax.fill_between(
                np.nanmean(bin_stim_seq, axis=0)[c_idx,:],
                0, self.n_clusters,
                color=color0, edgecolor='none', alpha=0.25, step='mid')
            for bi in range(self.bin_num):
                self.plot_cluster_mean_sem(
                    ax, cluster_bin_neu_mean[bi,:,:], cluster_bin_neu_sem[bi,:,:],
                    self.alignment['neu_time'], norm_params,
                    bin_stim_seq[bi, c_idx+isi_idx_offset, :].reshape(1,-1),
                    [colors[bi]], [colors[bi]]*self.n_clusters, xl)
            # adjust layouts.
            ax.set_xlabel('time since stim (ms)')
        @show_resource_usage
        def plot_oddball_jitter(ax, oddball):
            # collect data.
            [[color0, color1, color2, _],
             [_, _, stim_seq, _], _, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_idx=[l[oddball] for l in self.list_odd_idx],
                trial_param=[None, None, [0], None, [0], [0]],
                cate=cate, roi_id=None)
            c_idx = stim_seq.shape[0]//2
            neu_seq_fix, neu_seq_jitter = self.get_neu_seq_trial_fix_jitter(jitter_trial_mode, oddball, cate, isi_win)
            # get response within cluster.
            neu_seq_jitter = np.concatenate([np.nanmean(neu, axis=0) for neu in neu_seq_jitter], axis=0)
            neu_seq_fix = np.concatenate([np.nanmean(neu, axis=0) for neu in neu_seq_fix], axis=0)
            neu_fix_mean, neu_fix_sem = get_mean_sem_cluster(neu_seq_fix, self.n_clusters, cluster_id)
            neu_jitter_mean, neu_jitter_sem = get_mean_sem_cluster(neu_seq_jitter, self.n_clusters, cluster_id)
            norm_params = [get_norm01_params(
                np.concatenate([neu_fix_mean[ci,:], neu_jitter_mean[ci,:]]))
                for ci in range(self.n_clusters)]
            # plot results.
            ax.axis('off')
            ax = ax.inset_axes([0, 0, 1, 0.95], transform=ax.transAxes)
            if oddball == 0:
                ax.axvline(stim_seq[c_idx+1,0], color='gold', lw=1, linestyle='--')
            if oddball == 1:
                ax.axvline(stim_seq[c_idx,1]+self.expect, color='gold', lw=1, linestyle='--')
            self.plot_cluster_mean_sem(
                ax, neu_fix_mean, neu_fix_sem,
                self.alignment['neu_time'], norm_params,
                stim_seq,
                [color0]*stim_seq.shape[0], [color0]*self.n_clusters, xlim)
            self.plot_cluster_mean_sem(
                ax, neu_jitter_mean, neu_jitter_sem,
                self.alignment['neu_time'], norm_params,
                None, None, [[color1,color2][oddball]]*self.n_clusters, xlim)
            # adjust layouts.
            ax.set_xlabel('time since pre oddball stim (ms)')
        @show_resource_usage
        def plot_oddball_jitter_variability(ax, oddball):
            average_axis = 0
            # collect data.
            [[color0, color1, color2, _],
             [_, _, stim_seq, _], _, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_idx=[l[oddball] for l in self.list_odd_idx],
                trial_param=[None, None, [0], None, [0], [0]],
                cate=cate, roi_id=None)
            c_idx = stim_seq.shape[0]//2
            neu_seq_fix, neu_seq_jitter = self.get_neu_seq_trial_fix_jitter(jitter_trial_mode, oddball, cate, isi_win)
            # compute variance for each class.
            e_fix = np.zeros([self.n_clusters, len(self.alignment['neu_time'])]) * np.nan
            e_jitter = np.zeros([self.n_clusters, len(self.alignment['neu_time'])]) * np.nan
            for ci in range(self.n_clusters):
                if np.sum(cluster_id==ci) > 0:
                    # average.
                    neu_ci_fix = np.concatenate(
                        [np.nanmean(neu[:,dci==ci,:],axis=average_axis)
                         for neu,dci in zip(neu_seq_fix,day_cluster_id)], axis=0)
                    neu_ci_jitter = np.concatenate(
                        [np.nanmean(neu[:,dci==ci,:],axis=average_axis)
                         for neu,dci in zip(neu_seq_jitter,day_cluster_id)], axis=0)
                    # get variance trace.
                    e_fix[ci,:] = np.nanstd(neu_ci_fix, axis=0)
                    e_jitter[ci,:] = np.nanstd(neu_ci_jitter, axis=0)
            norm_params = [get_norm01_params(
                np.concatenate([e_fix[ci,:], e_jitter[ci,:]]))
                for ci in range(self.n_clusters)]
            # plot results.
            ax.axis('off')
            ax = ax.inset_axes([0, 0, 1, 0.95], transform=ax.transAxes)
            if oddball == 0:
                ax.axvline(stim_seq[c_idx+1,0], color='gold', lw=1, linestyle='--')
            if oddball == 1:
                ax.axvline(stim_seq[c_idx,1]+self.expect, color='gold', lw=1, linestyle='--')
            self.plot_cluster_mean_sem(
                ax, e_fix, np.zeros_like(e_fix),
                self.alignment['neu_time'], norm_params,
                stim_seq,
                [color0]*stim_seq.shape[0], [color0]*self.n_clusters, xlim)
            self.plot_cluster_mean_sem(
                ax, e_jitter, np.zeros_like(e_jitter),
                self.alignment['neu_time'], norm_params,
                None, None, [[color1,color2][oddball]]*self.n_clusters, xlim)
            # adjust layouts.
            ax.set_xlabel('time since pre oddball stim (ms)')
        @show_resource_usage
        def plot_win_mag_quant_stat(ax, oddball):
            average_axis = 1
            offset = [0, 0.1]
            # collect data.
            neu_seq_fix, neu_seq_jitter = self.get_neu_seq_trial_fix_jitter(jitter_trial_mode, oddball, cate, isi_win)
            [[color0, color1, color2, _],
             [_, _, stim_seq, _], _, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_idx=[l[oddball] for l in self.list_odd_idx],
                trial_param=[None, None, [0], None, [0], [0]],
                cate=cate, roi_id=None)
            c_idx = stim_seq.shape[0]//2
            # define evaluation windows.
            if oddball == 0:
                win_eval = [[-2500, 0],
                            [stim_seq[c_idx+1,0],stim_seq[c_idx+1,0]+250],
                            [stim_seq[c_idx+1,0]+300,stim_seq[c_idx+1,0]+550],
                            [stim_seq[c_idx+1,0]+600,stim_seq[c_idx+1,0]+850]]
            if oddball == 1:
                win_eval = [[-2500, 0],
                            [stim_seq[c_idx,1]+self.expect, stim_seq[c_idx,1]+self.expect+250],
                            [stim_seq[c_idx+1,0]-250,stim_seq[c_idx+1,0]],
                            [stim_seq[c_idx+1,1],stim_seq[c_idx+1,1]+250]]
            # define layout.
            ax.axis('off')
            ax0 = ax.inset_axes([0, 0.97, 1, 0.03], transform=ax.transAxes)
            ax1 = ax.inset_axes([0, 0, 1, 0.95], transform=ax.transAxes)
            ax_box = [ax1.inset_axes([0.3, ci/self.n_clusters, 0.5, 0.65/self.n_clusters], transform=ax1.transAxes)
                      for ci in range(self.n_clusters)]
            axs_stat_m = [ax1.inset_axes([0.3, (0.8+ci)/self.n_clusters, 0.5, 0.1/self.n_clusters], transform=ax1.transAxes)
                   for ci in range(self.n_clusters)]
            axs_stat_v = [ax1.inset_axes([0.3, (0.7+ci)/self.n_clusters, 0.5, 0.1/self.n_clusters], transform=ax1.transAxes)
                   for ci in range(self.n_clusters)]
            ax_box.reverse()
            axs_stat_m.reverse()
            axs_stat_v.reverse()
            # plot evaluation window.
            if oddball == 0:
                ax0.axvline(stim_seq[c_idx+1,0], color='gold', lw=1, linestyle='--')
            if oddball == 1:
                ax0.axvline(stim_seq[c_idx,1]+self.expect, color='gold', lw=1, linestyle='--')
            for i in range(stim_seq.shape[0]):
                ax0.fill_between(
                    stim_seq[i,:], 0, 1,
                    color=color0, edgecolor='none', alpha=0.25, step='mid')
            self.plot_win_mag_quant_win_eval(ax0, win_eval, color0, xlim)
            # plot errorbar.             
            self.plot_cluster_win_mag_quant(
                ax_box, day_cluster_id, neu_seq_fix, self.alignment['neu_time'],
                win_eval, color0, 0, offset[0], average_axis)
            self.plot_cluster_win_mag_quant(
                ax_box, day_cluster_id, neu_seq_jitter, self.alignment['neu_time'],
                win_eval, [color1,color2][oddball], 0, offset[1], average_axis)
            # plot statistics test.
            self.plot_cluster_win_mag_quant_stat(
                axs_stat_m, day_cluster_id, neu_seq_fix, neu_seq_jitter, self.alignment['neu_time'],
                win_eval, 0, average_axis, 'mean')
            self.plot_cluster_win_mag_quant_stat(
                axs_stat_v, day_cluster_id, neu_seq_fix, neu_seq_jitter, self.alignment['neu_time'],
                win_eval, 0, average_axis, 'var')
            # adjust layouts.
            hide_all_axis(ax1)
            ax0.set_xticks([])
            ax1.set_ylabel('evoked magnitude \n (sem across trials)')
        @show_resource_usage
        def plot_win_mag_scatter(ax, oddball, wi):
            average_axis = 0
            # collect data.
            neu_seq_fix, neu_seq_jitter = self.get_neu_seq_trial_fix_jitter(jitter_trial_mode, oddball, cate, isi_win)
            [[color0, color1, color2, _],
             [_, _, stim_seq, _], _, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_idx=[l[oddball] for l in self.list_odd_idx],
                trial_param=[None, None, [0], None, [0], [0]],
                cate=cate, roi_id=None)
            c_idx = stim_seq.shape[0]//2
            # define evaluation windows.
            if oddball == 0:
                win_eval = [[-2500, 0],
                            [stim_seq[c_idx+1,0],stim_seq[c_idx+1,0]+250],
                            [stim_seq[c_idx+1,0]+300,stim_seq[c_idx+1,0]+550],
                            [stim_seq[c_idx+1,0]+600,stim_seq[c_idx+1,0]+850]]
            if oddball == 1:
                win_eval = [[-2500, 0],
                            [stim_seq[c_idx,1]+self.expect, stim_seq[c_idx,1]+self.expect+250],
                            [stim_seq[c_idx+1,0]-250,stim_seq[c_idx+1,0]],
                            [stim_seq[c_idx+1,1],stim_seq[c_idx+1,1]+250]]
            win_eval = [win_eval[0], win_eval[wi]]
            # define layouts.
            ax.axis('off')
            ax = ax.inset_axes([0, 0, 1, 0.95], transform=ax.transAxes)
            axs = [ax.inset_axes([0.3, ci/self.n_clusters, 0.5, 0.7/self.n_clusters], transform=ax.transAxes)
                      for ci in range(self.n_clusters)]
            axs.reverse()
            # plot results.
            self.plot_cluster_win_mag_scatter(
                axs, day_cluster_id, neu_seq_fix, neu_seq_jitter, self.alignment['neu_time'],
                win_eval, win_eval, [color1,color2][oddball], 0, average_axis)
            # adjust layouts.
            for ci in range(self.n_clusters):
                axs[ci].set_ylabel('jitter')
            axs[self.n_clusters-1].set_xlabel('fix')
            hide_all_axis(ax)
            ax.set_ylabel(f'response comparison in the {win_lbl[wi]} window \n (sem across neurons)')
        @show_resource_usage
        def plot_win_mag_dist(ax, oddball, wi, cumulative):
            # collect data.
            neu_seq_fix, neu_seq_jitter = self.get_neu_seq_trial_fix_jitter(jitter_trial_mode, oddball, cate, isi_win)
            [[color0, color1, color2, _],
             [_, _, stim_seq, _], _, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_idx=[l[oddball] for l in self.list_odd_idx],
                trial_param=[None, None, [0], None, [0], [0]],
                cate=cate, roi_id=None)
            c_idx = stim_seq.shape[0]//2
            # define evaluation windows.
            if oddball == 0:
                win_eval = [[-2500, 0],
                            [stim_seq[c_idx+1,0],stim_seq[c_idx+1,0]+250],
                            [stim_seq[c_idx+1,0]+300,stim_seq[c_idx+1,0]+550],
                            [stim_seq[c_idx+1,0]+600,stim_seq[c_idx+1,0]+850]]
            if oddball == 1:
                win_eval = [[-2500, 0],
                            [stim_seq[c_idx,1]+self.expect, stim_seq[c_idx,1]+self.expect+250],
                            [stim_seq[c_idx+1,0]-250,stim_seq[c_idx+1,0]],
                            [stim_seq[c_idx+1,1],stim_seq[c_idx+1,1]+250]]
            win_eval = [win_eval[0], win_eval[wi]]
            # define layouts.
            ax.axis('off')
            ax = ax.inset_axes([0, 0, 1, 0.95], transform=ax.transAxes)
            h = ax.get_window_extent().height
            axs = [ax.inset_axes([0.1, ci/self.n_clusters, 0.9, h/self.n_clusters*0.7/h],
                                 transform=ax.transAxes) for ci in range(self.n_clusters)]
            axs.reverse()
            # plot results.
            self.plot_cluster_win_mag_dist_compare(
                axs, day_cluster_id, neu_seq_fix, neu_seq_jitter, self.alignment['neu_time'],
                win_eval, [color0, [color1,color2][oddball]], 0, cumulative)
            # adjust layouts.
            hide_all_axis(ax)
            ax.set_ylabel(f'response comparison in the {win_lbl[wi]} window \n (sem across neurons)')
        @show_resource_usage
        def plot_block_win_decode(ax, oddball):
            win_eval_baseline = [-2500, 0]
            win_decode = [-2500, 4000]
            win_sample = 200
            # collect data.
            neu_seq_fix, neu_seq_jitter = self.get_neu_seq_trial_fix_jitter(jitter_trial_mode, oddball, cate, isi_win)
            [[color0, color1, color2, _],
             [_, _, stim_seq, _], _, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_idx=[l[oddball] for l in self.list_odd_idx],
                trial_param=[None, None, [0], None, [0], [0]],
                cate=cate, roi_id=None)
            results_all = []
            # run decoding for each class.
            for ci in range(self.n_clusters):
                if np.sum(cluster_id==ci) > 0:
                    # baseline correction.
                    neu_ci_fix = [np.nanmean(neu[:,dci==ci,:],axis=0) for neu,dci in zip(neu_seq_fix,day_cluster_id)]
                    neu_ci_jitter = [np.nanmean(neu[:,dci==ci,:],axis=0) for neu,dci in zip(neu_seq_jitter,day_cluster_id)]
                    baseline_fix = [get_mean_sem_win(
                        nc.reshape(-1, nc.shape[-1]),
                        self.alignment['neu_time'], 0, win_eval_baseline[0], win_eval_baseline[1], mode='lower')[1]
                        for nc in neu_ci_fix]
                    baseline_jitter = [get_mean_sem_win(
                        nc.reshape(-1, nc.shape[-1]),
                        self.alignment['neu_time'], 0, win_eval_baseline[0], win_eval_baseline[1], mode='lower')[1]
                        for nc in neu_ci_jitter]
                    # construct input data.
                    neu_x = [
                        [nsf[:,dci==ci,:]-bl for nsf,dci,bl in zip(neu_seq_fix, day_cluster_id, baseline_fix)],
                        [nsj[:,dci==ci,:]-bl for nsj,dci,bl in zip(neu_seq_jitter, day_cluster_id, baseline_jitter)]]
                    # run decoding.
                    r = multi_sess_decoding_slide_win(
                         neu_x, self.alignment['neu_time'],
                         win_decode, win_sample)
                    results_all.append(r)
                else:
                    results_all.append([np.nan]*5)
            # split results.
            decode_model_mean  = [results_all[ci][1] for ci in range(self.n_clusters)]
            decode_model_sem   = [results_all[ci][2] for ci in range(self.n_clusters)]
            decode_chance_mean = [results_all[ci][3] for ci in range(self.n_clusters)]
            decode_chance_sem  = [results_all[ci][4] for ci in range(self.n_clusters)]
            # define layouts.
            ax.axis('off')
            ax = ax.inset_axes([0, 0, 1, 0.95], transform=ax.transAxes)
            axs = [ax.inset_axes([0.5, ci/self.n_clusters, 1, 0.8/self.n_clusters], transform=ax.transAxes)
                      for ci in range(self.n_clusters)]
            axs.reverse()
            # plot results for each class.
            for ci in range(self.n_clusters):
                if np.sum(cluster_id==ci) > 0:
                    # find bounds.
                    upper = np.nanmax([decode_model_mean[ci], decode_chance_mean[ci]])
                    lower = np.nanmin([decode_model_mean[ci], decode_chance_mean[ci]])
                    # plot stimulus.
                    c_idx = stim_seq.shape[0]//2
                    if oddball == 0:
                        axs[ci].axvline(stim_seq[c_idx+1,0], color='gold', lw=1, linestyle='--')
                    if oddball == 1:
                        axs[ci].axvline(stim_seq[c_idx,1]+self.expect, color='gold', lw=1, linestyle='--')
                    for i in range(stim_seq.shape[0]):
                        axs[ci].fill_between(
                            stim_seq[i,:],
                            lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
                            color=color0, edgecolor='none', alpha=0.25, step='mid')
                    # plot decoding results.
                    self.plot_mean_sem(
                        axs[ci],
                        results_all[ci][0], decode_model_mean[ci], decode_model_sem[ci],
                        [color1,color2][oddball], None)
                    self.plot_mean_sem(
                        axs[ci],
                        results_all[ci][0], decode_chance_mean[ci], decode_chance_sem[ci],
                        color0, None)
                    # adjust layouts.
                    axs[ci].set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
            for ci in range(self.n_clusters):
                axs[ci].tick_params(tick1On=False)
                axs[ci].spines['right'].set_visible(False)
                axs[ci].spines['top'].set_visible(False)
                axs[ci].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
                if ci != self.n_clusters-1:
                    axs[ci].set_xticks([])
            axs[self.n_clusters-1].set_xlabel('time since pre oddball stim (ms)')
            ax.set_ylabel('block decoding accuracy')
            hide_all_axis(ax)
        @show_resource_usage
        def plot_oddball_time_eclapse(ax, oddball):
            offset = [0, 0.1]
            # collect data.
            [[color0, color1, color2, _],
             [_, _, stim_seq, _], _, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_idx=[l[oddball] for l in self.list_odd_idx],
                trial_param=[None, None, [0], None, [0], [0]],
                cate=cate, roi_id=None)
            c_idx = stim_seq.shape[0]//2
            neu_seq_fix, neu_seq_jitter = self.get_neu_seq_trial_fix_jitter(jitter_trial_mode, oddball, cate, isi_win)
            colors_fix = get_cmap_color(2, base_color=color0)
            colors_jitter = get_cmap_color(2, base_color=[color1,color2][oddball])
            # define evaluation windows.
            if oddball == 0:
                win_decode = [stim_seq[c_idx+1,1], stim_seq[c_idx+1,1]+500]
            if oddball == 1:
                win_decode = [stim_seq[c_idx,1]+self.expect, stim_seq[c_idx+1,0]]
            # define layout.
            ax.axis('off')
            ax0 = ax.inset_axes([0, 0.97, 1, 0.03], transform=ax.transAxes)
            ax1 = ax.inset_axes([0, 0, 1, 0.95], transform=ax.transAxes)
            axs = [ax1.inset_axes([0.5, ci/self.n_clusters, 0.5, 0.8/self.n_clusters], transform=ax1.transAxes)
                      for ci in range(self.n_clusters)]
            axs.reverse()
            # plot evaluation window.
            if oddball == 0:
                ax0.axvline(stim_seq[c_idx+1,0], color='gold', lw=1, linestyle='--')
            if oddball == 1:
                ax0.axvline(stim_seq[c_idx,1]+self.expect, color='gold', lw=1, linestyle='--')
            for i in range(stim_seq.shape[0]):
                ax0.fill_between(
                    stim_seq[i,:], 0, 1,
                    color=color0, edgecolor='none', alpha=0.25, step='mid')
            self.plot_win_mag_quant_win_eval(ax0, [None, win_decode], color0, xlim, False)
            # plot results for each class.
            for ci in range(self.n_clusters):
                if np.sum(cluster_id==ci) > 0:
                    neu_x_fix = [ns[:,dci==ci,:] for ns,dci in zip(neu_seq_fix, day_cluster_id)]
                    neu_x_jitter = [ns[:,dci==ci,:] for ns,dci in zip(neu_seq_jitter, day_cluster_id)]
                    # run decoding.
                    results_model_fix, results_chance_fix = multi_sess_decoding_time_eclapse(neu_x_fix, self.alignment['neu_time'], win_decode)
                    results_model_jitter, results_chance_jitter = multi_sess_decoding_time_eclapse(neu_x_jitter, self.alignment['neu_time'], win_decode)
                    # plot decoding results.
                    m, s = get_mean_sem(results_model_fix.reshape(-1,1))
                    axs[ci].errorbar(
                        1+offset[0],
                        m, s,
                        color=colors_fix[1],
                        capsize=2, marker='o', linestyle='none',
                        markeredgecolor='white', markeredgewidth=0.1)
                    m, s = get_mean_sem(results_chance_fix.reshape(-1,1))
                    axs[ci].errorbar(
                        1+offset[1],
                        m, s,
                        color=colors_fix[0],
                        capsize=2, marker='o', linestyle='none',
                        markeredgecolor='white', markeredgewidth=0.1)
                    m, s = get_mean_sem(results_model_jitter.reshape(-1,1))
                    axs[ci].errorbar(
                        2+offset[0],
                        m, s,
                        color=colors_jitter[1],
                        capsize=2, marker='o', linestyle='none',
                        markeredgecolor='white', markeredgewidth=0.1)
                    m, s = get_mean_sem(results_chance_jitter.reshape(-1,1))
                    axs[ci].errorbar(
                        2+offset[1],
                        m, s,
                        color=colors_jitter[0],
                        capsize=2, marker='o', linestyle='none',
                        markeredgecolor='white', markeredgewidth=0.1)
                    # adjust layouts.
                    axs[ci].tick_params(axis='x', tick1On=False, labelrotation=90)
                    axs[ci].tick_params(axis='y', direction='in')
                    axs[ci].spines['right'].set_visible(False)
                    axs[ci].spines['top'].set_visible(False)
                    axs[ci].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
                    axs[ci].yaxis.set_major_locator(mtick.MaxNLocator(nbins=2))
                    axs[ci].set_xlim([0.5, 2.5])
                    axs[ci].set_xticks([])
            axs[self.n_clusters-1].set_xticks([1,2])
            axs[self.n_clusters-1].set_xticklabels(['fix', 'jitter'])
            hide_all_axis(ax1)
            ax0.set_xticks([])
            ax1.set_ylabel('regression $R^2$')
        @show_resource_usage
        def plot_neu_fraction(ax):
            ax.axis('off')
            ax = ax.inset_axes([0, 0, 1, 0.95], transform=ax.transAxes)
            # collect data.
            [[_, _, color2, _], _, _, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_param=[None, None, [0], None, [0], [0]],
                cate=cate, roi_id=None)
            # plot results.
            self.plot_cluster_neu_fraction_in_cluster(ax, cluster_id, color2)
        @show_resource_usage
        def plot_cate_fraction(ax):
            ax.axis('off')
            ax = ax.inset_axes([0, 0, 1, 0.95], transform=ax.transAxes)
            self.plot_cluster_cate_fraction_in_cluster(ax, cluster_id, neu_labels, self.label_names)
        @show_resource_usage
        def plot_legend(ax):
            [[color0, color1, color2, _], _, _,
             [n_trials, n_neurons]] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_param=[[-1], None, [0], None, [0], [0]],
                cate=cate, roi_id=None)
            cs = [color0, color1, color2, 'gold']
            lbl = ['fix', 'jitter (short oddball)', 'jitter (long oddball)', 'unexpected event']
            add_legend(ax, cs, lbl, n_trials, n_neurons, self.n_sess, 'upper right')
            ax.axis('off')
        # plot all.
        try: plot_glm_kernel(axs[0])
        except Exception as e: print(e)
        try: plot_random_bin(axs[1], 'pre')
        except Exception as e: print(e)
        try: plot_random_bin(axs[2], 'post')
        except Exception as e: print(e)
        try: plot_oddball_jitter(axs[3], 0)
        except Exception as e: print(e)
        try: plot_oddball_jitter_variability(axs[4], 0)
        except Exception as e: print(e)
        try: plot_win_mag_quant_stat(axs[5], 0)
        except Exception as e: print(e)
        try: plot_win_mag_scatter(axs[6], 0, 2)
        except Exception as e: print(e)
        try: plot_win_mag_dist(axs[7], 0, 2,False)
        except Exception as e: print(e)
        try: plot_win_mag_dist(axs[8], 0, 2,True)
        except Exception as e: print(e)
        try: plot_block_win_decode(axs[9], 0)
        except Exception as e: print(e)
        try: plot_oddball_time_eclapse(axs[10], 0)
        except Exception as e: print(e)
        try: plot_oddball_jitter(axs[11], 1)
        except Exception as e: print(e)
        try: plot_oddball_jitter_variability(axs[12], 1)
        except Exception as e: print(e)
        try: plot_win_mag_quant_stat(axs[13], 1)
        except Exception as e: print(e)
        try: plot_win_mag_scatter(axs[14], 1, 2)
        except Exception as e: print(e)
        try: plot_win_mag_dist(axs[15], 1, 2,False)
        except Exception as e: print(e)
        try: plot_win_mag_dist(axs[16], 1, 2,True)
        except Exception as e: print(e)
        try: plot_block_win_decode(axs[17], 1)
        except Exception as e: print(e)
        try: plot_oddball_time_eclapse(axs[18], 1)
        except Exception as e: print(e)
        try: plot_neu_fraction(axs[19])
        except Exception as e: print(e)
        try: plot_cate_fraction(axs[20])
        except Exception as e: print(e)
        try: plot_legend(axs[21])
        except Exception as e: print(e)
    
    def plot_cluster_oddball_fix_heatmap_all(self, axs, cate):
        xlim = [-2500, 4000]
        l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, xlim[0], xlim[1])
        kernel_all = get_glm_cate(self.glm, self.list_labels, self.list_significance, cate)
        cluster_id, neu_labels = get_cluster_cate(self.cluster_id, self.list_labels, self.list_significance, cate)
        @show_resource_usage
        def plot_cluster_features(ax):
            # fit model.
            features = PCA(n_components=2).fit_transform(kernel_all)
            # plot results.
            ax.scatter(features[:,0], features[:,1], c=cluster_id, cmap='hsv')
            # adjust layouts.
            ax.tick_params(tick1On=False)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.set_xlabel('latent 1')
            ax.set_ylabel('latent 2')
        @show_resource_usage
        def plot_hierarchical_dendrogram(ax):
            # collect data.
            [_, _, _, cmap], _, _, _ = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_param=[[2,3,4,5], None, [0], None, [0], [0]],
                cate=cate, roi_id=None)
            # plot results.
            self.plot_dendrogram(ax, kernel_all, cmap)
        @show_resource_usage
        def plot_glm_kernel(ax):
            ax.axis('off')
            ax = ax.inset_axes([0, 0, 0.5, 1], transform=ax.transAxes)
            # collect data.
            [[_, _, _, cmap], [_, _, stim_seq, _], _, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_param=[[2,3,4,5], None, [0], None, [0], [0]],
                cate=cate, roi_id=None)
            c_idx = stim_seq.shape[0]//2
            # plot results.
            self.plot_cluster_heatmap(ax, kernel_all, self.glm['kernel_time'], cluster_id, 'minmax', cmap)
            # adjust layouts.
            ax.set_xlabel('time since stim (ms)')
            ax.axvline(stim_seq[c_idx,0], color='black', lw=1, linestyle='--')
        @show_resource_usage
        def plot_standard_fix(ax):
            # collect data.
            [[_, _, _, cmap],
             [neu_seq, _, stim_seq, _], _, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_param=[[2,3,4,5], None, [0], None, [0], [0]],
                cate=cate, roi_id=None)
            c_idx = stim_seq.shape[0]//2
            neu_seq = neu_seq[:,l_idx:r_idx]
            neu_time = self.alignment['neu_time'][l_idx:r_idx]
            # plot results.
            self.plot_cluster_heatmap(ax, neu_seq, neu_time, cluster_id, 'minmax', cmap)
            # add stimulus line.
            for bi in range(3):
                xlines = [self.alignment['neu_time'][np.searchsorted(self.alignment['neu_time'], t)]
                          for t in [stim_seq[c_idx+i,0]
                                    for i in [-2,-1,0,1,2]]]
                for xl in xlines:
                    if xl>neu_time[0] and xl<neu_time[-1]:
                        ax.axvline(xl, color='black', lw=1, linestyle='--')
            # adjust layouts.
            ax.set_xlabel('time since stim (ms)')
        @show_resource_usage
        def plot_oddball_fix(ax, oddball):
            # collect data.
            [[_, _, _, cmap],
             [neu_seq, _, stim_seq, _], _, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_idx=[l[oddball] for l in self.list_odd_idx],
                trial_param=[None, None, [0], None, [0], [0]],
                cate=cate, roi_id=None)
            c_idx = stim_seq.shape[0]//2
            neu_seq = neu_seq[:,l_idx:r_idx]
            neu_time = self.alignment['neu_time'][l_idx:r_idx]
            # plot results.
            self.plot_cluster_heatmap(ax, neu_seq, neu_time, cluster_id, 'minmax', cmap)
            # add stimulus line.
            for bi in range(3):
                xlines = [self.alignment['neu_time'][np.searchsorted(self.alignment['neu_time'], t)]
                          for t in [stim_seq[c_idx+i,0]
                                    for i in [-2,-1,0,1,2]]]
                for xl in xlines:
                    if xl>neu_time[0] and xl<neu_time[-1]:
                        ax.axvline(xl, color='black', lw=1, linestyle='--')
            if oddball == 0:
                ax.axvline(stim_seq[c_idx+1,0], color='gold', lw=1, linestyle='--')
            if oddball == 1:
                ax.axvline(stim_seq[c_idx,1]+self.expect, color='gold', lw=1, linestyle='--')
            # adjust layouts.
            ax.set_xlabel('time since pre oddball stim (ms)')
        # plot all.
        try: plot_cluster_features(axs[0])
        except Exception as e: print(e)
        try: plot_hierarchical_dendrogram(axs[1])
        except Exception as e: print(e)
        try: plot_glm_kernel(axs[2])
        except Exception as e: print(e)
        try: plot_standard_fix(axs[3])
        except Exception as e: print(e)
        try: plot_oddball_fix(axs[4], 0)
        except Exception as e: print(e)
        try: plot_oddball_fix(axs[5], 1)
        except Exception as e: print(e)
        
    def plot_sorted_heatmaps_fix_all(self, axs, norm_mode, cate):
        win_sort = [-500, 500]
        xlim = [-2500, 5000]
        l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, xlim[0], xlim[1])
        # collect data.
        neu_x = []
        stim_x = []
        neu_time = self.alignment['neu_time'][l_idx:r_idx]
        # standard.
        [_, _, _, cmap], [neu_seq, _, stim_seq, _], _, _ = get_neu_trial(
            self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
            trial_param=[[2,3,4,5], None, [0], None, [0], [0]],
            cate=cate, roi_id=None)
        neu_seq = neu_seq[:,l_idx:r_idx]
        neu_x.append(neu_seq)
        stim_x.append(stim_seq)
        # short oddball.
        _, [neu_seq, _, stim_seq, _], _, _ = get_neu_trial(
            self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
            trial_idx=[l[0] for l in self.list_odd_idx],
            trial_param=[None, None, [0], None, [0], [0]],
            cate=cate, roi_id=None)
        neu_seq = neu_seq[:,l_idx:r_idx]
        neu_x.append(neu_seq)
        stim_x.append(stim_seq)
        # long oddball.
        _, [neu_seq, _, stim_seq, _], _, _ = get_neu_trial(
            self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
            trial_idx=[l[1] for l in self.list_odd_idx],
            trial_param=[None, None, [0], None, [0], [0]],
            cate=cate, roi_id=None)
        neu_seq = neu_seq[:,l_idx:r_idx]
        neu_x.append(neu_seq)
        stim_x.append(stim_seq)
        # plot heatmaps.
        for bi in range(3):
            self.plot_heatmap_neuron(
                axs[bi], neu_x[bi], neu_time, neu_x[0], cmap, win_sort,
                norm_mode=norm_mode,
                neu_seq_share=neu_x,
                cbar_label='dF/F')
            axs[bi].set_xlabel('time since pre oddball stim (ms) \n sorting window [{},{}] ms'.format(
                win_sort[0], win_sort[1]))
        # add stimulus line.
        c_idx = stim_x[0].shape[0]//2
        for bi in range(3):
            xlines = [neu_time[np.searchsorted(neu_time, t)]
                      for t in [stim_x[bi][c_idx+i,0]
                                for i in [-2,-1,0,1,2]]]
            for xl in xlines:
                if xl>neu_time[0] and xl<neu_time[-1]:
                    axs[bi].axvline(xl, color='black', lw=1, linestyle='--')
        # adjust layouts.
        axs[0].set_xlabel('time since stim (ms) \n sorting window [{},{}] ms'.format(
            win_sort[0], win_sort[1]))
        axs[1].set_xlabel('time since pre oddball stim (ms) \n sorting window [{},{}] ms'.format(
            win_sort[0], win_sort[1]))
        axs[2].set_xlabel('time since pre oddball stim (ms) \n sorting window [{},{}] ms'.format(
            win_sort[0], win_sort[1]))
        axs[1].axvline(stim_x[1][c_idx+1,0], color='gold', lw=1, linestyle='--')
        axs[2].axvline(stim_x[2][c_idx,1]+self.expect, color='gold', lw=1, linestyle='--')

    def plot_oddball_latent_fix_all(self, ax, oddball, cate):
        # collect data.
        xlim = [-2500, 6000]
        l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, xlim[0], xlim[1])
        [[color0, color1, color2, _],
         [neu_x, _, stim_seq, _], _,
         [n_trials, n_neurons]] = get_neu_trial(
            self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
            trial_idx=[l[oddball] for l in self.list_odd_idx],
            trial_param=[None, None, [0], None, [0], [0]],
            cate=cate, roi_id=None)
        neu_x = neu_x[:,l_idx:r_idx]
        neu_time = self.alignment['neu_time'][l_idx:r_idx]
        cmap, c_neu = get_cmap_color(
            neu_x.shape[1], base_color=[color1, color2][oddball],
            return_cmap=True)
        c_idx = stim_seq.shape[0]//2
        c_stim = [color0] * stim_seq.shape[-2]
        # fit model.
        model = PCA(n_components=self.d_latent)
        neu_z = model.fit_transform(
            neu_x.reshape(neu_x.shape[0],-1).T
            ).T.reshape(self.d_latent, -1)
        # define layouts.
        ax.axis('off')
        ax0 = ax.inset_axes([0,   0.8, 0.7, 0.2], transform=ax.transAxes)
        ax1 = ax.inset_axes([0,   0,   0.7, 0.7], transform=ax.transAxes, projection='3d')
        ax2 = ax.inset_axes([0.8, 0,   0.2,   1], transform=ax.transAxes)
        # plot mean trace.
        neu_mean, neu_sem = get_mean_sem(neu_x)
        c_neu = get_cmap_color(neu_mean.shape[0], cmap=self.latent_cmap)
        # find bounds.
        upper = np.nanmax(neu_mean) + np.nanmax(neu_sem)
        lower = np.nanmin(neu_mean) - np.nanmax(neu_sem)
        # plot stimulus.
        for si in range(stim_seq.shape[0]):
            ax0.fill_between(
                stim_seq[si,:],
                lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
                color=color0, edgecolor='none', alpha=0.25, step='mid')
        if oddball == 0:
            ax0.axvline(stim_seq[c_idx+1,0], color='gold', lw=1, linestyle='--')
        if oddball == 1:
            ax0.axvline(stim_seq[c_idx,1]+self.expect, color='gold', lw=1, linestyle='--')
        # plot neural traces.
        self.plot_mean_sem(ax0, neu_time, neu_mean, neu_sem, color0, None)
        for t in range(neu_mean.shape[0]-1):
            ax0.plot(neu_time[t:t+2], neu_mean[t:t+2], color=c_neu[t])
        # adjust layouts.
        adjust_layout_neu(ax0)
        ax0.set_xlim(xlim)
        ax0.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
        ax0.set_xlabel('time since pre oddball stim (ms)')
        # plot 3d dynamics.
        self.plot_3d_latent_dynamics(ax1, neu_z, stim_seq, neu_time, c_stim)
        # mark unexpected event.
        idx_unexpect = get_frame_idx_from_time(
            neu_time, 0, stim_seq[c_idx+1,0], stim_seq[c_idx,1]+self.expect)[oddball]
        ax1.scatter(neu_z[0,idx_unexpect], neu_z[1,idx_unexpect], neu_z[2,idx_unexpect], color='gold', marker='o', lw=5)
        # adjust layouts.
        adjust_layout_3d_latent(ax1, neu_z, None, neu_time, 'time since pre oddball stim (ms)')
        
# colors = ['#989A9C', '#A4CB9E', '#9DB4CE', '#EDA1A4', '#F9C08A']
class plotter_main(plotter_utils):
    def __init__(self, neural_trials, labels, significance, label_names, temp_folder, cate_list):
        super().__init__(neural_trials, labels, significance, temp_folder, cate_list)
        self.label_names = label_names

    def cluster_oddball_fix_all(self, axs_all):
        for cate, axs in zip(self.cate_list, axs_all):
            try:
                label_name = self.label_names[str(cate[0])] if len(cate)==1 else 'all'
                print(f'plotting results for {label_name}')
                
                self.plot_cluster_oddball_fix_all(axs, cate=cate)
                axs[0].set_title(f'GLM kernel course \n {label_name}')
                axs[1].set_title(f'reponse to standard interval \n {label_name}')
                axs[2].set_title(f'reponse to short oddball interval \n {label_name}')
                axs[3].set_title(f'reponse to long oddball interval \n {label_name}')
                axs[4].set_title(f'reponse to stimulus after interval \n {label_name}')
                axs[5].set_title(f'fraction of neurons in cluster \n {label_name}')
                axs[6].set_title(f'fraction of cell-types in cluster \n {label_name}')
                axs[7].set_title(f'legend \n {label_name}')

            except Exception as e: print(e)
            
    def cluster_oddball_jitter_similar_all(self, axs_all):
        for cate, axs in zip(self.cate_list, axs_all):
            try:
                label_name = self.label_names[str(cate[0])] if len(cate)==1 else 'all'
                print(f'plotting results for {label_name}')
                
                self.plot_cluster_oddball_jitter_global_all(axs, 'similar', cate=cate)
                axs[ 0].set_title(f'GLM kernel course \n {label_name}')
                axs[ 1].set_title(f'reponse to binned (pre) random \n {label_name}')
                axs[ 2].set_title(f'reponse to binned (post) random \n {label_name}')
                axs[ 3].set_title(f'reponse to short oddball \n {label_name}')
                axs[ 4].set_title(f'response SD to short oddball \n {label_name}')
                axs[ 5].set_title(f'reponse to short oddball \n {label_name}')
                axs[ 6].set_title(f'comparison for short oddball \n {label_name}')
                axs[ 7].set_title(f'response distribution for short oddball \n {label_name}')
                axs[ 8].set_title(f'response distribution for short oddball \n {label_name}')
                axs[ 9].set_title(f'decoding in short oddball \n {label_name}')
                axs[10].set_title(f'reponse to long oddball \n {label_name}')
                axs[11].set_title(f'response SD to long oddball \n {label_name}')
                axs[12].set_title(f'reponse to long oddball \n {label_name}')
                axs[13].set_title(f'comparison for long oddball \n {label_name}')
                axs[14].set_title(f'response distribution for long oddball \n {label_name}')
                axs[15].set_title(f'response distribution for long oddball \n {label_name}')
                axs[16].set_title(f'decoding in long oddball \n {label_name}')
                axs[17].set_title(f'fraction of neurons in cluster \n {label_name}')
                axs[18].set_title(f'fraction of cell-types in cluster \n {label_name}')
                axs[19].set_title(f'legend \n {label_name}')
                
            except Exception as e: print(e)
    
    def cluster_oddball_jitter_lower_all(self, axs_all):
        for cate, axs in zip(self.cate_list, axs_all):
            try:
                label_name = self.label_names[str(cate[0])] if len(cate)==1 else 'all'
                print(f'plotting results for {label_name}')
                
                self.plot_cluster_oddball_jitter_global_all(axs, 'lower', cate=cate)
                axs[ 0].set_title(f'GLM kernel course \n {label_name}')
                axs[ 1].set_title(f'reponse to binned (pre) random \n {label_name}')
                axs[ 2].set_title(f'reponse to binned (post) random \n {label_name}')
                axs[ 3].set_title(f'reponse to short oddball \n {label_name}')
                axs[ 4].set_title(f'response SD to short oddball \n {label_name}')
                axs[ 5].set_title(f'reponse to short oddball \n {label_name}')
                axs[ 6].set_title(f'comparison for short oddball \n {label_name}')
                axs[ 7].set_title(f'response distribution for short oddball \n {label_name}')
                axs[ 8].set_title(f'response distribution for short oddball \n {label_name}')
                axs[ 9].set_title(f'decoding in short oddball \n {label_name}')
                axs[10].set_title(f'reponse to long oddball \n {label_name}')
                axs[11].set_title(f'response SD to long oddball \n {label_name}')
                axs[12].set_title(f'reponse to long oddball \n {label_name}')
                axs[13].set_title(f'comparison for long oddball \n {label_name}')
                axs[14].set_title(f'response distribution for long oddball \n {label_name}')
                axs[15].set_title(f'response distribution for long oddball \n {label_name}')
                axs[16].set_title(f'decoding in long oddball \n {label_name}')
                axs[17].set_title(f'fraction of neurons in cluster \n {label_name}')
                axs[18].set_title(f'fraction of cell-types in cluster \n {label_name}')
                axs[19].set_title(f'legend \n {label_name}')
                
            except Exception as e: print(e)
            
    def cluster_oddball_jitter_higher_all(self, axs_all):
        for cate, axs in zip(self.cate_list, axs_all):
            try:
                label_name = self.label_names[str(cate[0])] if len(cate)==1 else 'all'
                print(f'plotting results for {label_name}')
                
                self.plot_cluster_oddball_jitter_global_all(axs, 'higher', cate=cate)
                axs[ 0].set_title(f'GLM kernel course \n {label_name}')
                axs[ 1].set_title(f'reponse to binned (pre) random \n {label_name}')
                axs[ 2].set_title(f'reponse to binned (post) random \n {label_name}')
                axs[ 3].set_title(f'reponse to short oddball \n {label_name}')
                axs[ 4].set_title(f'response SD to short oddball \n {label_name}')
                axs[ 5].set_title(f'reponse to short oddball \n {label_name}')
                axs[ 6].set_title(f'comparison for short oddball \n {label_name}')
                axs[ 7].set_title(f'response distribution for short oddball \n {label_name}')
                axs[ 8].set_title(f'response distribution for short oddball \n {label_name}')
                axs[ 9].set_title(f'decoding in short oddball \n {label_name}')
                axs[10].set_title(f'reponse to long oddball \n {label_name}')
                axs[11].set_title(f'response SD to long oddball \n {label_name}')
                axs[12].set_title(f'reponse to long oddball \n {label_name}')
                axs[13].set_title(f'comparison for long oddball \n {label_name}')
                axs[14].set_title(f'response distribution for long oddball \n {label_name}')
                axs[15].set_title(f'response distribution for long oddball \n {label_name}')
                axs[16].set_title(f'decoding in long oddball \n {label_name}')
                axs[17].set_title(f'fraction of neurons in cluster \n {label_name}')
                axs[18].set_title(f'fraction of cell-types in cluster \n {label_name}')
                axs[19].set_title(f'legend \n {label_name}')
                
            except Exception as e: print(e)
            
    def cluster_oddball_jitter_global_all(self, axs_all):
        for cate, axs in zip(self.cate_list, axs_all):
            try:
                label_name = self.label_names[str(cate[0])] if len(cate)==1 else 'all'
                print(f'plotting results for {label_name}')
                
                self.plot_cluster_oddball_jitter_global_all(axs, 'global', cate=cate)
                axs[ 0].set_title(f'GLM kernel course \n {label_name}')
                axs[ 1].set_title(f'reponse to binned (pre) random \n {label_name}')
                axs[ 2].set_title(f'reponse to binned (post) random \n {label_name}')
                axs[ 3].set_title(f'reponse to short oddball \n {label_name}')
                axs[ 4].set_title(f'response SD to short oddball \n {label_name}')
                axs[ 5].set_title(f'reponse to short oddball \n {label_name}')
                axs[ 6].set_title(f'comparison for short oddball \n {label_name}')
                axs[ 7].set_title(f'response distribution for short oddball \n {label_name}')
                axs[ 8].set_title(f'response distribution for short oddball \n {label_name}')
                axs[ 9].set_title(f'decoding in short oddball \n {label_name}')
                axs[10].set_title(f'time decoding in short oddball \n {label_name}')
                axs[11].set_title(f'reponse to long oddball \n {label_name}')
                axs[12].set_title(f'response SD to long oddball \n {label_name}')
                axs[13].set_title(f'reponse to long oddball \n {label_name}')
                axs[14].set_title(f'comparison for long oddball \n {label_name}')
                axs[15].set_title(f'response distribution for long oddball \n {label_name}')
                axs[16].set_title(f'response distribution for long oddball \n {label_name}')
                axs[17].set_title(f'decoding in long oddball \n {label_name}')
                axs[18].set_title(f'time decoding in long oddball \n {label_name}')
                axs[19].set_title(f'fraction of neurons in cluster \n {label_name}')
                axs[20].set_title(f'fraction of cell-types in cluster \n {label_name}')
                axs[21].set_title(f'legend \n {label_name}')
                
            except Exception as e: print(e)
    
    def cluster_oddball_fix_heatmap_all(self, axs_all):
        for cate, axs in zip(self.cate_list, axs_all):
            try:
                label_name = self.label_names[str(cate[0])] if len(cate)==1 else 'all'
                print(f'plotting results for {label_name}')
                
                self.plot_cluster_oddball_fix_heatmap_all(axs, cate=cate)
                axs[0].set_title(f'clustered latent features \n {label_name}')
                axs[1].set_title(f'cluster dendrogram \n {label_name}')
                axs[2].set_title(f'GLM kernel course \n {label_name}')
                axs[3].set_title(f'reponse to standard interval \n {label_name}')
                axs[4].set_title(f'reponse to short oddball interval \n {label_name}')
                axs[5].set_title(f'reponse to long oddball interval \n {label_name}')

            except Exception as e: print(e)
        
    def sorted_heatmaps_fix_all(self, axs_all):
        for cate, axs in zip(self.cate_list, axs_all):
            try:
                label_name = self.label_names[str(cate[0])] if len(cate)==1 else 'all'
                print(f'plotting results for {label_name}')
                
                for axss, norm_mode in zip([axs[0:3], axs[3:6], axs[6:9]], ['none', 'minmax', 'share']):
                    self.plot_sorted_heatmaps_fix_all(axss, norm_mode, cate=cate)
                    axss[0].set_title(f'response to standard interval \n {label_name} (normalized with {norm_mode})')
                    axss[1].set_title(f'response to short oddball interval \n {label_name} (normalized with {norm_mode})')
                    axss[2].set_title(f'response to long oddball interval \n {label_name} (normalized with {norm_mode})')

            except Exception as e: print(e)
    
    def oddball_latent_fix_all(self, axs_all):
        for cate, axs in zip(self.cate_list, axs_all):
            try:
                label_name = self.label_names[str(cate[0])] if len(cate)==1 else 'all'
                print(f'plotting results for {label_name}')
                
                self.plot_oddball_latent_fix_all(axs[0], 0, cate=cate)
                self.plot_oddball_latent_fix_all(axs[1], 1, cate=cate)

            except Exception as e: print(e)
