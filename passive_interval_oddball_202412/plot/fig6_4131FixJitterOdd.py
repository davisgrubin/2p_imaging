#!/usr/bin/env python3

import traceback
import numpy as np
import matplotlib.ticker as mtick
from sklearn.decomposition import PCA

from modules.Alignment import run_get_stim_response
from modeling.clustering import get_mean_sem_cluster
from modeling.clustering import get_bin_mean_sem_cluster
from modeling.clustering import get_cluster_cate
from modeling.decoding import multi_sess_decoding_slide_win
from modeling.decoding import fit_poly_line
from modeling.decoding import decoding_time_confusion
from modeling.generative import get_glm_cate
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
from utils import get_random_rotate_mat_3d
from utils import adjust_layout_isi_example_epoch
from utils import adjust_layout_3d_latent
from utils import add_legend
from utils import add_heatmap_colorbar
from utils import utils_basic

# fig, ax = plt.subplots(1, 1, figsize=(3, 6))
# fig, ax = plt.subplots(1, 1, figsize=(3, 3))
# fig, axs = plt.subplots(1, 3, figsize=(9, 6))
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
        self.d_latent = 3
        self.glm = self.run_glm()
        self.n_up = 2
        self.n_dn = 3
        self.cluster_id = self.run_clustering(self.n_up, self.n_dn)

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
        neu_seq_jitter = [neu_seq_jitter[si][idx_jitter[si],:,:] for si in range(self.n_sess)]
        return neu_seq_fix, neu_seq_jitter

    def plot_neuron_fraction(self, ax):
        try:
            colors = ['cornflowerblue', 'violet', 'mediumseagreen']
            cluster_id, neu_labels = get_cluster_cate(self.cluster_id, self.list_labels, self.list_significance, [-1,1,2])
            exc = np.sum(neu_labels==-1)
            vip = np.sum(neu_labels==1)
            sst = np.sum(neu_labels==2)
            ax.pie(
                [exc, vip, sst],
                labels=['{} Exc'.format(exc),
                        '{} VIP'.format(vip),
                        '{} SST'.format(sst)],
                colors=colors,
                autopct='%1.1f%%',
                wedgeprops={'linewidth': 1, 'edgecolor':'white', 'width':0.2})
            ax.set_title('fraction of {} neuron labels'.format(len(neu_labels)))
        except: traceback.print_exc()
        
    def plot_isi_seting(self, ax):
        gap = 25
        ax.hlines(0.5, 500+gap, 2500-gap, color='royalblue')
        ax.vlines([500+gap, 2500-gap], 0, 0.5, color='royalblue')
        ax.vlines(1500, 0, 1, color='deeppink')
        ax.vlines(500, 0, 0.6, color='black')
        ax.vlines(2500, 0, 0.6, color='black')
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xlim([450,2550])
        ax.set_ylim([0, 1.05])
        ax.set_xticks([500,1500,2500])
        ax.set_yticks([])
        ax.set_xticklabels([500,1500,2500])
    
    def plot_isi_example_epoch(self, ax):
        trial_win = [1000,1500]
        # get isi and trial labels.
        stim_labels = self.list_neural_trials[0]['stim_labels'][trial_win[0]:trial_win[1],:]
        isi = stim_labels[1:,0] - stim_labels[:-1,1]
        img_seq_label    = stim_labels[:-1,2]
        fix_jitter_types = stim_labels[:-1,4]
        # plot trials.
        color_map = np.array(['deeppink', 'royalblue', 'black'], dtype=object)
        idx = np.where(img_seq_label == -1, 2, fix_jitter_types)
        colors = color_map[idx]
        ax.scatter(np.arange(trial_win[0], trial_win[1]-1), isi, c=colors, s=5)
        # adjust layouts.
        adjust_layout_isi_example_epoch(ax, trial_win, self.bin_win)
        
    def plot_cluster_oddball_fix_all(self, axs, cate):
        color0 = 'dimgrey'
        color1 = 'hotpink'
        color2 = 'darkviolet'
        xlim = [-2500, 4000]
        cluster_id, neu_labels = get_cluster_cate(self.cluster_id, self.list_labels, self.list_significance, cate)
        @show_resource_usage
        def plot_glm_kernel(ax):
            kernel_all = get_glm_cate(self.glm, self.list_labels, self.list_significance, cate)
            self.plot_glm_kernel(ax, kernel_all, cluster_id, color0, 1)
        @show_resource_usage
        def plot_standard_fix(ax):
            # collect data.
            [_, [neu_seq, _, stim_seq, _],
             [neu_labels, neu_sig], _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_param=[[2,3,4,5], None, [0], None, [0], [0]],
                cate=cate, roi_id=None)
            # get response within cluster.
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
        except: traceback.print_exc()
        try: plot_standard_fix(axs[1])
        except: traceback.print_exc()
        try: plot_oddball_fix(axs[2], 0)
        except: traceback.print_exc()
        try: plot_oddball_fix(axs[3], 1)
        except: traceback.print_exc()
        try: plot_oddball_fix_quant(axs[4])
        except: traceback.print_exc()
        try: plot_neu_fraction(axs[5])
        except: traceback.print_exc()
        try: plot_cate_fraction(axs[6])
        except: traceback.print_exc()
        try: plot_legend(axs[7])
        except: traceback.print_exc()
    
    def plot_cluster_oddball_jitter_global_all(self, axs, jitter_trial_mode, cate):
        color0 = 'dimgrey'
        color1 = 'deeppink'
        color2 = 'royalblue'
        color_model = 'plum'
        win_lbl = ['early', 'late', 'post']
        isi_win = 250
        xlim = [-2500, 4000]
        l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, xlim[0], xlim[1])
        cluster_id, neu_labels = get_cluster_cate(self.cluster_id, self.list_labels, self.list_significance, cate)
        split_idx = get_split_idx(self.list_labels, self.list_significance, cate)
        day_cluster_id = np.split(cluster_id, split_idx)
        @show_resource_usage
        def plot_oddball_jitter(ax, oddball):
            # collect data.
            [_, [_, _, stim_seq, _], _, _] = get_neu_trial(
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
                [color0]*stim_seq.shape[0], [color1]*self.n_clusters, xlim)
            self.plot_cluster_mean_sem(
                ax, neu_jitter_mean, neu_jitter_sem,
                self.alignment['neu_time'], norm_params,
                None, None, [color2]*self.n_clusters, xlim)
            # adjust layouts.
            ax.set_xlabel('time since pre oddball stim (ms)')
        @show_resource_usage
        def plot_pred_mod_index_box(ax, oddball, pe):
            # collect data.
            neu_seq_1, neu_seq_2 = self.get_neu_seq_trial_fix_jitter(jitter_trial_mode, oddball, cate, isi_win)
            [_, [_, _, stim_seq, _], _, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_idx=[l[oddball] for l in self.list_odd_idx],
                trial_param=[None, None, [0], None, [0], [0]],
                cate=cate, roi_id=None)
            c_idx = stim_seq.shape[0]//2
            # define evaluation windows.
            if pe == 'pos':
                win_eval = [[-2500, 0],
                            [stim_seq[c_idx,0], stim_seq[c_idx,0]+300],
                            [stim_seq[c_idx+1,0], stim_seq[c_idx+1,0]+300]]
            if pe == 'neg':
                win_eval = [[-2500, 0],
                            [stim_seq[c_idx,1]+self.expect-300, stim_seq[c_idx,1]+self.expect],
                            [stim_seq[c_idx,1]+self.expect, stim_seq[c_idx,1]+self.expect+300]]
            # define layouts.
            ax0 = ax.inset_axes([0, 0.97, 0.5, 0.03], transform=ax.transAxes)
            ax1 = ax.inset_axes([0, 0, 0.5, 0.95], transform=ax.transAxes)
            axs = [ax1.inset_axes([0, ci/self.n_clusters, 1, 0.8/self.n_clusters], transform=ax1.transAxes)
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
            self.plot_win_mag_quant_win_eval(ax0, win_eval, color0, xlim, False)
            # plot results.
            self.plot_cluster_pred_mod_index_compare(
                axs, day_cluster_id, neu_seq_1, neu_seq_2, self.alignment['neu_time'],
                win_eval, win_eval, color1, color2, 0, average_axis=1)
            # adjust layouts.
            for ci in range(self.n_clusters):
                if ci != self.n_clusters-1:
                    axs[ci].set_xticks([])
                    axs[ci].set_yticklabels([])
            axs[self.n_clusters-1].set_xticklabels(['fix', 'jitter'])
            ax.set_title(pe+' PE')
            hide_all_axis(ax)
            hide_all_axis(ax0)
            hide_all_axis(ax1)
        @show_resource_usage
        def plot_win_mag_scatter(ax, oddball, wi):
            average_axis = 1
            # collect data.
            neu_seq_fix, neu_seq_jitter = self.get_neu_seq_trial_fix_jitter(jitter_trial_mode, oddball, cate, isi_win)
            [_, [_, _, stim_seq, _], _, _] = get_neu_trial(
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
                win_eval, win_eval, color0, 0, average_axis)
            # adjust layouts.
            for ci in range(self.n_clusters):
                axs[ci].set_ylabel('jitter')
            axs[self.n_clusters-1].set_xlabel('fix')
            hide_all_axis(ax)
            ax.set_ylabel(f'response comparison in the {win_lbl[wi-1]} window')
        @show_resource_usage
        def plot_win_mag_dist_var(ax, oddball, wi, cumulative):
            # collect data.
            neu_seq_1, neu_seq_2 = self.get_neu_seq_trial_fix_jitter(jitter_trial_mode, oddball, cate, isi_win)
            [_, [_, _, stim_seq, _], _, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_idx=[l[oddball] for l in self.list_odd_idx],
                trial_param=[None, None, [0], None, [0], [0]],
                cate=cate, roi_id=None)
            c_idx = stim_seq.shape[0]//2
            # define evaluation windows.
            if oddball == 0:
                win_eval = [[-2500, 0], [stim_seq[c_idx+1,0],stim_seq[c_idx+1,0]+300]]
            if oddball == 1:
                win_eval = [[-2500, 0], [stim_seq[c_idx,1]+self.expect, stim_seq[c_idx,1]+self.expect+300]]
            # define layouts.
            ax.axis('off')
            ax = ax.inset_axes([0, 0, 1, 0.95], transform=ax.transAxes)
            axs = [ax.inset_axes([0.1, ci/self.n_clusters, 0.5, 1/self.n_clusters*0.7],
                                 transform=ax.transAxes) for ci in range(self.n_clusters)]
            axs.reverse()
            # plot results.
            self.plot_cluster_win_mag_dist_compare(
                axs, day_cluster_id, neu_seq_1, neu_seq_2, self.alignment['neu_time'],
                win_eval, [color1, color2], 0, cumulative)
            # adjust layouts.
            hide_all_axis(ax)
            ax.set_ylabel(f'response comparison in the {win_lbl[wi]} window')
        @show_resource_usage
        def plot_block_win_decode(ax, oddball):
            num_sess_decode = 5
            win_sample = 200
            # collect data.
            neu_seq_fix, neu_seq_jitter = self.get_neu_seq_trial_fix_jitter(jitter_trial_mode, oddball, cate, isi_win)
            [_, [_, _, stim_seq, _], _, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_idx=[l[oddball] for l in self.list_odd_idx],
                trial_param=[None, None, [0], None, [0], [0]],
                cate=cate, roi_id=None)
            c_idx = stim_seq.shape[0]//2
            sub_idx = np.random.choice(self.n_sess, num_sess_decode, replace=False)
            neu_seq_fix = np.array(neu_seq_fix, dtype='object')[sub_idx].tolist()
            neu_seq_jitter = np.array(neu_seq_jitter, dtype='object')[sub_idx].tolist()
            # run decoding for each class.
            results_all = []
            win_decode = [-500, stim_seq[c_idx+1,1]+500]
            for ci in range(self.n_clusters):
                if np.sum(cluster_id==ci) > 0:
                    # construct input data.
                    neu_x = [
                        [nsf[:,dci==ci,:] for nsf,dci in zip(neu_seq_fix, day_cluster_id)],
                        [nsj[:,dci==ci,:] for nsj,dci in zip(neu_seq_jitter, day_cluster_id)]]
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
            axs = [ax.inset_axes([0.2, ci/self.n_clusters, 0.5, 0.8/self.n_clusters], transform=ax.transAxes)
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
                    for si in [0,1]:
                        axs[ci].fill_between(
                            stim_seq[c_idx+si,:],
                            lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
                            color=color0, edgecolor='none', alpha=0.25, step='mid')
                    # plot decoding results.
                    self.plot_mean_sem(
                        axs[ci],
                        results_all[ci][0], decode_chance_mean[ci], decode_chance_sem[ci],
                        color0, None)
                    self.plot_mean_sem(
                        axs[ci],
                        results_all[ci][0], decode_model_mean[ci], decode_model_sem[ci],
                        color_model, None)
                    # adjust layouts.
                    axs[ci].set_xlim([-500,3000])
                    axs[ci].set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
            for ci in range(self.n_clusters):           
                axs[ci].spines['right'].set_visible(False)
                axs[ci].spines['top'].set_visible(False)
                axs[ci].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
                axs[ci].xaxis.set_major_locator(mtick.MaxNLocator(nbins=2))
                axs[ci].yaxis.set_major_locator(mtick.MaxNLocator(nbins=2))
                if ci != self.n_clusters-1:
                    axs[ci].set_xticks([])
            axs[self.n_clusters-1].set_xlabel('time since pre oddball stim (ms)')
            ax.set_ylabel('block decoding accuracy')
            hide_all_axis(ax)
        @show_resource_usage
        def plot_oddball_time_decode(ax, oddball):
            bin_times = 50
            # collect data.
            [_, [neu_seq_0, stim_seq, _], _, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_idx=[l[oddball] for l in self.list_odd_idx],
                trial_param=[None, None, [0], None, [0], [0]],
                sub_sampling=True,
                cate=cate, roi_id=None)
            [_, [neu_seq_1, _, _], _, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_idx=[l[oddball] for l in self.list_odd_idx],
                trial_param=[None, None, [1], None, [0], [0]],
                sub_sampling=True,
                cate=cate, roi_id=None)
            c_idx = stim_seq.shape[0]//2
            # get data within range.
            bin_l_idx, bin_r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, 0, bin_times)
            bin_len = bin_r_idx - bin_l_idx
            if oddball == 0:
                tlim = [stim_seq[c_idx+1,0], stim_seq[c_idx+1,1]+500]
            if oddball == 1:
                tlim = [stim_seq[c_idx,1]+self.expect, stim_seq[c_idx+1,0]]                
            l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, tlim[0], tlim[1])
            t_range = self.alignment['neu_time'][l_idx:r_idx]
            t_range = t_range[:(len(t_range)//bin_len)*bin_len]
            t_range = np.nanmin(t_range.reshape(-1, bin_len), axis=1)
            l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, 0, stim_seq[c_idx+1, 0])
            neu_time = self.alignment['neu_time'][l_idx:r_idx]
            neu_x_0 = neu_seq_0[:,:,l_idx:r_idx]
            neu_x_1 = neu_seq_1[:,:,l_idx:r_idx]
            # get decoding matrix.
            acc_all_0_ci = []
            acc_all_1_ci = []
            for ci in range(self.n_clusters):
                if np.sum(cluster_id==ci) > 0:
                    _, _, acc_all_0 = decoding_time_confusion(neu_x_0[:,cluster_id==ci,:], neu_time, bin_times)
                    _, _, acc_all_1 = decoding_time_confusion(neu_x_1[:,cluster_id==ci,:], neu_time, bin_times)
                    acc_all_0_ci.append(acc_all_0)
                    acc_all_1_ci.append(acc_all_1)
                else:
                    acc_all_0_ci.append(np.array([np.nan]))
                    acc_all_1_ci.append(np.array([np.nan]))
            # define layouts.
            ax.axis('off')
            ax = ax.inset_axes([0, 0, 1, 0.95], transform=ax.transAxes)
            axs = [ax.inset_axes([0.3, ci/self.n_clusters, 0.5, 0.7/self.n_clusters], transform=ax.transAxes)
                      for ci in range(self.n_clusters)]
            axs.reverse()
            # plot results for each class.
            for ci in range(self.n_clusters):
                if np.sum(cluster_id==ci) > 0:
                    self.plot_raincloud_compare(axs[ci], acc_all_0_ci[ci], acc_all_1_ci[ci], color1, color2)
                # adjust layouts.
                axs[ci].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
                axs[ci].yaxis.set_major_locator(mtick.MaxNLocator(nbins=2))
                if ci != self.n_clusters-1:
                    axs[ci].set_xticks([])
            axs[self.n_clusters-1].set_xticklabels(['fix', 'jitter'])
            axs[self.n_clusters-1].tick_params(axis='x', labelrotation=90)
            ax.set_ylabel('decoding accuracy')
            hide_all_axis(ax)
        @show_resource_usage
        def plot_neu_fraction(ax):
            ax.axis('off')
            ax = ax.inset_axes([0, 0, 1, 0.95], transform=ax.transAxes)
            # plot results.
            self.plot_cluster_neu_fraction_in_cluster(ax, cluster_id, color0)
        @show_resource_usage
        def plot_cate_fraction(ax):
            ax.axis('off')
            ax = ax.inset_axes([0, 0, 1, 0.95], transform=ax.transAxes)
            self.plot_cluster_cate_fraction_in_cluster(ax, cluster_id, neu_labels, self.label_names, color0)
        @show_resource_usage
        def plot_legend(ax):
            [_, _, _, [n_trials, n_neurons]] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_param=[[-1], None, [0], None, [0], [0]],
                cate=cate, roi_id=None)
            cs = [color1, color2, 'gold', color0, color_model]
            lbl = ['fix', 'jitter', 'unexpected event', 'shuffle','model']
            add_legend(ax, cs, lbl, n_trials, n_neurons, self.n_sess, 'upper right')
            ax.axis('off')
        @show_resource_usage
        def plot_block_win_decode_all(ax, oddball):
            color_model = 'plum'
            win_sample = 200
            # collect data.
            neu_seq_fix, neu_seq_jitter = self.get_neu_seq_trial_fix_jitter(jitter_trial_mode, oddball, cate, isi_win)
            [_, [_, _, stim_seq, _], _, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_idx=[l[oddball] for l in self.list_odd_idx],
                trial_param=[None, None, [0], None, [0], [0]],
                cate=cate, roi_id=None)
            c_idx = stim_seq.shape[0]//2
            # run decoding for each class.
            win_decode = [-500, stim_seq[c_idx+1,1]+500]
            # construct input data.
            neu_x = [neu_seq_fix, neu_seq_jitter]
            # run decoding.
            acc_time, acc_model_mean, acc_model_sem, acc_chance_mean, acc_chance_sem = multi_sess_decoding_slide_win(
                neu_x, self.alignment['neu_time'],
                win_decode, win_sample)
            # define layouts.
            ax.axis('off')
            ax = ax.inset_axes([0, 0, 0.8, 0.95], transform=ax.transAxes)
            # plot results.
            upper = 0.9
            lower = 0.3
            # plot stimulus.
            c_idx = stim_seq.shape[0]//2
            if oddball == 0:
                ax.axvline(stim_seq[c_idx+1,0], color='gold', lw=1, linestyle='--')
            if oddball == 1:
                ax.axvline(stim_seq[c_idx,1]+self.expect, color='gold', lw=1, linestyle='--')
            for si in [0,1]:
                ax.fill_between(
                    stim_seq[c_idx+si,:],
                    lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
                    color=color0, edgecolor='none', alpha=0.25, step='mid')
            # plot decoding results.
            self.plot_mean_sem(ax, acc_time, acc_chance_mean, acc_chance_sem, color0, None)
            self.plot_mean_sem(ax, acc_time, acc_model_mean, acc_model_sem, color_model, None)
            # adjust layouts.
            ax.set_xlim([-500,3500])
            ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])        
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
            ax.xaxis.set_major_locator(mtick.MaxNLocator(nbins=2))
            ax.yaxis.set_major_locator(mtick.MaxNLocator(nbins=2))
            ax.set_xlabel('time since pre oddball stim (ms)')
            ax.set_ylabel('block decoding accuracy')
        # plot all.
        try: plot_oddball_jitter(axs[0], 0)
        except: traceback.print_exc()
        try: plot_pred_mod_index_box(axs[1], 0, 'pos')
        except: traceback.print_exc()
        try: plot_win_mag_scatter(axs[2], 0, 2)
        except: traceback.print_exc()
        try: plot_win_mag_dist_var(axs[3], 0, 2, False)
        except: traceback.print_exc()
        #try: plot_block_win_decode(axs[4], 0)
        #except: traceback.print_exc()
        #try: plot_oddball_time_decode(axs[5], 0)
        #except: traceback.print_exc()
        try: plot_oddball_jitter(axs[6], 1)
        except: traceback.print_exc()
        try: plot_pred_mod_index_box(axs[7], 1, 'pos')
        except: traceback.print_exc()
        try: plot_pred_mod_index_box(axs[8], 1, 'neg')
        except: traceback.print_exc()
        try: plot_win_mag_scatter(axs[9], 1, 2)
        except: traceback.print_exc()
        try: plot_win_mag_dist_var(axs[10], 1, 2, False)
        except: traceback.print_exc()
        #try: plot_block_win_decode(axs[11], 1)
        #except: traceback.print_exc()
        #try: plot_oddball_time_decode(axs[12], 1)
        #except: traceback.print_exc()
        try: plot_neu_fraction(axs[13])
        except: traceback.print_exc()
        try: plot_cate_fraction(axs[14])
        except: traceback.print_exc()
        try: plot_legend(axs[15])
        except: traceback.print_exc()
        try: plot_block_win_decode_all(axs[16], 0)
        except: traceback.print_exc()
        try: plot_block_win_decode_all(axs[17], 1)
        except: traceback.print_exc()
    
    def plot_cluster_oddball_jitter_local_all(self, axs, jitter_trial_mode, cate):
        color0 = 'dimgrey'
        color1 = 'mediumseagreen'
        color2 = 'coral'
        cs = [color1, color2]
        win_lbl = ['early', 'late', 'post']
        xlim = [-3000, 4000]
        l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, xlim[0], xlim[1])
        cluster_id, neu_labels = get_cluster_cate(self.cluster_id, self.list_labels, self.list_significance, cate)
        split_idx = get_split_idx(self.list_labels, self.list_significance, cate)
        day_cluster_id = np.split(cluster_id, split_idx)
        @show_resource_usage
        def plot_oddball_jitter(ax, oddball):
            # collect data.
            [_, [neu_seq, stim_seq, camera_pupil, pre_isi, _], _, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_idx=[l[oddball] for l in self.list_odd_idx],
                trial_param=[None, None, [1], None, [0], [0]],
                mean_sem=False,
                cate=cate, roi_id=None)
            # bin data based on isi.
            [bins, bin_center, _, bin_neu_seq, _, _, bin_stim_seq, bin_camera_pupil] = get_isi_bin_neu(
                neu_seq, stim_seq, camera_pupil, pre_isi, self.bin_win, self.bin_num)
            stim_seq = np.nanmean(np.concatenate(stim_seq, axis=0), axis=0)
            c_idx = stim_seq.shape[0]//2
            cluster_bin_neu_mean, cluster_bin_neu_sem = get_bin_mean_sem_cluster(bin_neu_seq, self.n_clusters, cluster_id)
            norm_params = [get_norm01_params(cluster_bin_neu_mean[:,ci,:]) for ci in range(self.n_clusters)]
            # define layouts.
            ax.axis('off')
            ax = ax.inset_axes([0, 0, 1, 0.95], transform=ax.transAxes)
            # plot results.
            for si in [0,1]:
                ax.fill_between(
                    stim_seq[c_idx+si,:],
                    0, self.n_clusters,
                    color=color0, edgecolor='none', alpha=0.25, step='mid')
            if oddball == 0:
                ax.axvline(stim_seq[c_idx+1,0], color='gold', lw=1, linestyle='--')
            if oddball == 1:
                ax.axvline(stim_seq[c_idx,1]+self.expect, color='gold', lw=1, linestyle='--')
            for bi in range(self.bin_num):
                ax.axvline(bin_stim_seq[bi,c_idx-1,0], color=cs[bi], lw=1, linestyle='--')
            for bi in range(self.bin_num):
                self.plot_cluster_mean_sem(
                    ax, cluster_bin_neu_mean[bi,:,:], cluster_bin_neu_sem[bi,:,:],
                    self.alignment['neu_time'], norm_params,
                    None, None, [cs[bi]]*self.n_clusters, xlim)
            # adjust layouts.
            ax.set_xlabel('time since stim (ms)')
        @show_resource_usage
        def plot_win_mag_quant_stat(ax, oddball):
            average_axis = 1
            offset = [0, 0.1]
            # collect data.
            [_, [neu_seq, stim_seq, camera_pupil, pre_isi, _], _, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_idx=[l[oddball] for l in self.list_odd_idx],
                trial_param=[None, None, [1], None, [0], [0]],
                mean_sem=False,
                cate=cate, roi_id=None)
            # bin data based on isi.
            [_, _, bin_neu_seq_trial, _, _, _, _, _] = get_isi_bin_neu(
                neu_seq, stim_seq, camera_pupil, pre_isi, self.bin_win, self.bin_num)
            stim_seq = np.nanmean(np.concatenate(stim_seq, axis=0), axis=0)
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
            for bi in range(self.bin_num):
                self.plot_cluster_win_mag_quant(
                    ax_box, day_cluster_id, bin_neu_seq_trial[bi], self.alignment['neu_time'],
                    win_eval, cs[bi], 0, offset[1], average_axis)
            # plot statistics test.
            self.plot_cluster_win_mag_quant_stat(
                axs_stat_m, day_cluster_id, bin_neu_seq_trial[0], bin_neu_seq_trial[-1], self.alignment['neu_time'],
                win_eval, 0, average_axis, 'mean')
            self.plot_cluster_win_mag_quant_stat(
                axs_stat_v, day_cluster_id, bin_neu_seq_trial[0], bin_neu_seq_trial[-1], self.alignment['neu_time'],
                win_eval, 0, average_axis, 'var')
            # adjust layouts.
            hide_all_axis(ax1)
            ax0.set_xticks([])
            ax1.set_ylabel('evoked magnitude \n')
        @show_resource_usage
        def plot_win_mag_scatter(ax, oddball, wi):
            average_axis = 1
            # collect data.
            [_, [neu_seq, stim_seq, camera_pupil, pre_isi, _], _, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_idx=[l[oddball] for l in self.list_odd_idx],
                trial_param=[None, None, [1], None, [0], [0]],
                mean_sem=False,
                cate=cate, roi_id=None)
            # bin data based on isi.
            [_, _, bin_neu_seq_trial, _, _, _, _, _] = get_isi_bin_neu(
                neu_seq, stim_seq, camera_pupil, pre_isi, self.bin_win, self.bin_num)
            stim_seq = np.nanmean(np.concatenate(stim_seq, axis=0), axis=0)
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
                axs, day_cluster_id, bin_neu_seq_trial[0], bin_neu_seq_trial[-1], self.alignment['neu_time'],
                win_eval, win_eval, color0, 0, average_axis)
            # adjust layouts.
            for ci in range(self.n_clusters):
                axs[ci].set_ylabel('jitter')
            axs[self.n_clusters-1].set_xlabel('fix')
            hide_all_axis(ax)
            ax.set_ylabel(f'response comparison in the {win_lbl[wi-1]} window \n')
        @show_resource_usage
        def plot_win_mag_dist(ax, oddball, wi, cumulative):
            # collect data.
            [_, [neu_seq, stim_seq, camera_pupil, pre_isi, _], _, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_idx=[l[oddball] for l in self.list_odd_idx],
                trial_param=[None, None, [1], None, [0], [0]],
                mean_sem=False,
                cate=cate, roi_id=None)
            # bin data based on isi.
            [_, _, bin_neu_seq_trial, _, _, _, _, _] = get_isi_bin_neu(
                neu_seq, stim_seq, camera_pupil, pre_isi, self.bin_win, self.bin_num)
            stim_seq = np.nanmean(np.concatenate(stim_seq, axis=0), axis=0)
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
            axs = [ax.inset_axes([0.1, ci/self.n_clusters, 0.5, 1/self.n_clusters*0.7],
                                 transform=ax.transAxes) for ci in range(self.n_clusters)]
            axs.reverse()
            # plot results.
            self.plot_cluster_win_mag_dist_compare(
                axs, day_cluster_id, bin_neu_seq_trial[0], bin_neu_seq_trial[-1], self.alignment['neu_time'],
                win_eval, [cs[0], cs[-1]], 0, cumulative)
            # adjust layouts.
            hide_all_axis(ax)
            ax.set_ylabel(f'response comparison in the {win_lbl[wi-1]} window')
        @show_resource_usage
        def plot_interval_scaling(ax, oddball):
            post_color = 'lightcoral'
            gap = 2
            bin_num = 50
            win_eval = [[-2500,0], [0, 400]]
            order = 2
            # collect data.
            [_, [neu_seq, stim_seq, camera_pupil, pre_isi, _], _, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_idx=[l[oddball] for l in self.list_odd_idx],
                trial_param=[None, None, [1], None, [0], [0]],
                mean_sem=False,
                cate=cate, roi_id=None)
            # bin data based on isi.
            [_, bin_center, _, bin_neu_seq, _, _, _, _] = get_isi_bin_neu(
                neu_seq, stim_seq, camera_pupil, pre_isi, self.bin_win, bin_num)
            bin_center = bin_center[gap:-gap]
            stim_seq = np.nanmean(np.concatenate(stim_seq, axis=0), axis=0)
            c_idx = stim_seq.shape[0]//2
            # get response within cluster.
            bin_win_baseline = [np.array([get_mean_sem_win(
                bns[cluster_id==ci,:], self.alignment['neu_time'], 0,
                win_eval[0][0], win_eval[0][1], 'lower')[1]
                for bns in bin_neu_seq])[gap:-gap] for ci in range(self.n_clusters)]
            bin_win_evoked_pre = [np.array([get_mean_sem_win(
                bns[cluster_id==ci,:], self.alignment['neu_time'], 0,
                stim_seq[c_idx,0]+win_eval[1][0], stim_seq[c_idx,0]+win_eval[1][1], 'mean')[1]
                for bns in bin_neu_seq])[gap:-gap] for ci in range(self.n_clusters)]
            bin_win_evoked_post = [np.array([get_mean_sem_win(
                bns[cluster_id==ci,:], self.alignment['neu_time'], 0,
                stim_seq[c_idx+1,0]+win_eval[1][0], stim_seq[c_idx+1,0]+win_eval[1][1], 'mean')[1]
                for bns in bin_neu_seq])[gap:-gap] for ci in range(self.n_clusters)]
            bin_win_pre  = [bwe - bwb for bwe, bwb in zip (bin_win_evoked_pre, bin_win_baseline)]
            bin_win_post = [bwe - bwb for bwe, bwb in zip (bin_win_evoked_post, bin_win_baseline)]
            # define layouts.
            ax.axis('off')
            ax = ax.inset_axes([0, 0, 1, 1], transform=ax.transAxes)
            axs = [ax.inset_axes([0.1, ci/self.n_clusters, 0.4, 0.7/self.n_clusters], transform=ax.transAxes)
                      for ci in range(self.n_clusters)]
            axs.reverse()
            # plot results for each class.
            for ci in range(self.n_clusters):
                if np.sum(cluster_id==ci) > 0:
                    # find bounds.
                    upper = np.nanmax([bin_win_pre[ci], bin_win_post[ci]])
                    lower = np.nanmin([bin_win_pre[ci], bin_win_post[ci]])
                    # plot scatter.
                    axs[ci].scatter(bin_center, bin_win_pre[ci], color=color0, s=2, alpha=0.5)
                    axs[ci].scatter(bin_center, bin_win_post[ci], color=post_color, s=2, alpha=0.5)
                    # fit line.
                    y_pred_0, _ = fit_poly_line(bin_center, bin_win_pre[ci], order)
                    y_pred_1, _ = fit_poly_line(bin_center, bin_win_post[ci], order)
                    # plot line.
                    axs[ci].plot(bin_center, y_pred_0, color=color0, lw=2)
                    axs[ci].plot(bin_center, y_pred_1, color=post_color, lw=2)
                    # adjust layouts.
                    axs[ci].spines['right'].set_visible(False)
                    axs[ci].spines['top'].set_visible(False)
                    axs[ci].set_xlim(self.bin_win)
                    axs[ci].set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
                    axs[ci].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
                    axs[ci].xaxis.set_major_locator(mtick.MaxNLocator(nbins=2))
                    axs[ci].yaxis.set_major_locator(mtick.MaxNLocator(nbins=2))
                    if ci != self.n_clusters-1:
                        axs[ci].set_xticklabels([])
            axs[self.n_clusters-1].set_xlabel('preceding interval (ms)')
            ax.set_ylabel(f'evoked magnitude in the {win_eval[1]} window')
            hide_all_axis(ax)
            add_legend(ax, [color0, post_color], ['pre','post'], None, None, None, 'upper right')
        @show_resource_usage
        def plot_block_win_decode(ax, oddball):
            color_model = 'plum'
            win_eval_baseline = [-2500, 0]
            win_decode = [-2500, 4000]
            win_sample = 400
            # collect data.
            [_, [neu_seq, stim_seq, camera_pupil, pre_isi, _], _, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_idx=[l[oddball] for l in self.list_odd_idx],
                trial_param=[None, None, [1], None, [0], [0]],
                mean_sem=False,
                cate=cate, roi_id=None)
            # bin data based on isi.
            [_, _, bin_neu_seq_trial, _, _, _, bin_stim_seq, _] = get_isi_bin_neu(
                neu_seq, stim_seq, camera_pupil, pre_isi, self.bin_win, self.bin_num)
            stim_seq = np.nanmean(np.concatenate(stim_seq, axis=0), axis=0)
            c_idx = stim_seq.shape[0]//2
            # run decoding for each class.
            results_all = []
            for ci in range(self.n_clusters):
                if np.sum(cluster_id==ci) > 0:
                    # baseline correction.
                    neu_ci_s = [np.nanmean(neu[:,dci==ci,:],axis=0) for neu,dci in zip(bin_neu_seq_trial[0],day_cluster_id)]
                    neu_ci_l = [np.nanmean(neu[:,dci==ci,:],axis=0) for neu,dci in zip(bin_neu_seq_trial[-1],day_cluster_id)]
                    baseline_s = [get_mean_sem_win(
                        nc.reshape(-1, nc.shape[-1]),
                        self.alignment['neu_time'], 0, win_eval_baseline[0], win_eval_baseline[1], mode='lower')[1]
                        for nc in neu_ci_s]
                    baseline_l = [get_mean_sem_win(
                        nc.reshape(-1, nc.shape[-1]),
                        self.alignment['neu_time'], 0, win_eval_baseline[0], win_eval_baseline[1], mode='lower')[1]
                        for nc in neu_ci_l]
                    # construct input data.
                    neu_x = [
                        [ns[:,dci==ci,:]-bl for ns,dci,bl in zip(bin_neu_seq_trial[0], day_cluster_id, baseline_s)],
                        [ns[:,dci==ci,:]-bl for ns,dci,bl in zip(bin_neu_seq_trial[-1], day_cluster_id, baseline_l)]]
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
            axs = [ax.inset_axes([0, ci/self.n_clusters, 0.5, 0.8/self.n_clusters], transform=ax.transAxes)
                      for ci in range(self.n_clusters)]
            axs.reverse()
            # plot results for each class.
            for ci in range(self.n_clusters):
                if np.sum(cluster_id==ci) > 0:
                    # find bounds.
                    upper = np.nanmax([decode_model_mean[ci], decode_chance_mean[ci]])
                    lower = np.nanmin([decode_model_mean[ci], decode_chance_mean[ci]])
                    # plot stimulus.
                    for si in [0,1]:
                        axs[ci].fill_between(
                            stim_seq[c_idx+si,:],
                            0, self.n_clusters,
                            color=color0, edgecolor='none', alpha=0.25, step='mid')
                    if oddball == 0:
                        axs[ci].axvline(stim_seq[c_idx+1,0], color='gold', lw=1, linestyle='--')
                    if oddball == 1:
                        axs[ci].axvline(stim_seq[c_idx,1]+self.expect, color='gold', lw=1, linestyle='--')
                    for bi in range(self.bin_num):
                        axs[ci].axvline(bin_stim_seq[bi,c_idx-1,0], color=cs[bi], lw=1, linestyle='--')
                    # plot decoding results.
                    self.plot_mean_sem(
                        axs[ci],
                        results_all[ci][0], decode_chance_mean[ci], decode_chance_sem[ci],
                        color0, None)
                    self.plot_mean_sem(
                        axs[ci],
                        results_all[ci][0], decode_model_mean[ci], decode_model_sem[ci],
                        color_model, None)
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
            ax.set_ylabel('bin decoding accuracy')
            hide_all_axis(ax)
            add_legend(ax, [color0, color_model], ['shuffle','model'], None, None, None, 'upper right')
        @show_resource_usage
        def plot_neu_fraction(ax):
            ax.axis('off')
            ax = ax.inset_axes([0, 0, 1, 0.95], transform=ax.transAxes)
            # plot results.
            self.plot_cluster_neu_fraction_in_cluster(ax, cluster_id, color0)
        @show_resource_usage
        def plot_cate_fraction(ax):
            ax.axis('off')
            ax = ax.inset_axes([0, 0, 1, 0.95], transform=ax.transAxes)
            self.plot_cluster_cate_fraction_in_cluster(ax, cluster_id, neu_labels, self.label_names, color0)
        @show_resource_usage
        def plot_legend(ax):
            [_, _, _, [n_trials, n_neurons]] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_param=[[-1], None, [0], None, [0], [0]],
                cate=cate, roi_id=None)
            cs = [color1, color2, 'gold']
            lbl = ['short preceding ISI', 'long preceding ISI', 'unexpected event']
            add_legend(ax, cs, lbl, n_trials, n_neurons, self.n_sess, 'upper right')
            ax.axis('off')
        # plot all.
        try: plot_oddball_jitter(axs[0], 0)
        except: traceback.print_exc()
        try: plot_win_mag_quant_stat(axs[1], 0)
        except: traceback.print_exc()
        try: plot_win_mag_scatter(axs[2], 0, 2)
        except: traceback.print_exc()
        try: plot_win_mag_dist(axs[3], 0, 2, False)
        except: traceback.print_exc()
        try: plot_interval_scaling(axs[4], 0)
        except: traceback.print_exc()
        try: plot_block_win_decode(axs[5], 0)
        except: traceback.print_exc()
        try: plot_oddball_jitter(axs[6], 1)
        except: traceback.print_exc()
        try: plot_win_mag_quant_stat(axs[7], 1)
        except: traceback.print_exc()
        try: plot_win_mag_scatter(axs[8], 1, 2)
        except: traceback.print_exc()
        try: plot_win_mag_dist(axs[9], 1, 2, False)
        except: traceback.print_exc()
        try: plot_interval_scaling(axs[10], 1)
        except: traceback.print_exc()
        try: plot_block_win_decode(axs[11], 1)
        except: traceback.print_exc()
        try: plot_neu_fraction(axs[12])
        except: traceback.print_exc()
        try: plot_cate_fraction(axs[13])
        except: traceback.print_exc()
        try: plot_legend(axs[14])
        except: traceback.print_exc()
           
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
            self.plot_cluster_heatmap(ax, kernel_all, self.glm['kernel_time'], cluster_id, 'minmax')
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
            self.plot_cluster_heatmap(ax, neu_seq, neu_time, cluster_id, 'minmax')
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
            self.plot_cluster_heatmap(ax, neu_seq, neu_time, cluster_id, 'minmax')
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
        except: traceback.print_exc()
        try: plot_hierarchical_dendrogram(axs[1])
        except: traceback.print_exc()
        try: plot_glm_kernel(axs[2])
        except: traceback.print_exc()
        try: plot_standard_fix(axs[3])
        except: traceback.print_exc()
        try: plot_oddball_fix(axs[4], 0)
        except: traceback.print_exc()
        try: plot_oddball_fix(axs[5], 1)
        except: traceback.print_exc()
        
    def plot_sorted_heatmaps_fix_all(self, axs, norm_mode, cate):
        xlim = [-2500, 5000]
        l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, xlim[0], xlim[1])
        # collect data.
        neu_x = []
        stim_x = []
        neu_time = self.alignment['neu_time'][l_idx:r_idx]
        # standard.
        _, [neu_seq, _, stim_seq, _], _, _ = get_neu_trial(
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
        # define layouts.
        axs_hm = [axs[ai].inset_axes([0, 0, 0.7, 1], transform=axs[ai].transAxes) for ai in range(3)]
        axs_cb = [axs[ai].inset_axes([0.8, 0, 0.1, 1], transform=axs[ai].transAxes) for ai in range(3)]
        # plot heatmaps.
        for ai in range(3):
            self.plot_heatmap_neuron(
                axs_hm[ai], axs_cb[ai], neu_x[ai], neu_time, neu_x[0],
                norm_mode=norm_mode,
                neu_seq_share=neu_x)
            hide_all_axis(axs[ai])
        # add stimulus line.
        c_idx = stim_x[0].shape[0]//2
        for ai in range(3):
            xlines = [neu_time[np.searchsorted(neu_time, t)]
                      for t in [stim_x[ai][c_idx+i,0]
                                for i in [-2,-1,0,1,2]]]
            for xl in xlines:
                if xl>neu_time[0] and xl<neu_time[-1]:
                    axs_hm[ai].axvline(xl, color='black', lw=1, linestyle='--')
        axs_hm[1].axvline(stim_x[1][c_idx+1,0], color='gold', lw=1, linestyle='--')
        axs_hm[2].axvline(stim_x[2][c_idx,1]+self.expect, color='gold', lw=1, linestyle='--')
        # adjust layouts.
        axs_hm[0].set_xlabel('time since stim (ms)')
        axs_hm[1].set_xlabel('time since pre oddball stim (ms)')
        axs_hm[2].set_xlabel('time since pre oddball stim (ms)')
        
    def plot_latent_all(self, axs, cate):
        xlim = [-2500, 4000]
        color0 = 'dimgrey'
        def plot_jitter_global_oddball(axs, oddball, fix_jitter):
            base_color1 = ['chocolate', 'crimson']
            base_color2 = ['deepskyblue', 'royalblue']
            # collect data.
            [_, [_, _, stim_seq, _], _, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_idx=[l[oddball] for l in self.list_odd_idx],
                trial_param=[None, None, [0], None, [0], [0]],
                cate=cate, roi_id=None)
            c_idx = stim_seq.shape[0]//2
            neu_seq_0, neu_seq_1 = self.get_neu_seq_trial_fix_jitter('global', oddball, cate, None)
            neu_seq_0 = np.concatenate([np.nanmean(neu, axis=0) for neu in neu_seq_0], axis=0)
            neu_seq_1 = np.concatenate([np.nanmean(neu, axis=0) for neu in neu_seq_1], axis=0)
            c_stim = [color0] * stim_seq.shape[-2]
            # fit model.
            neu_x = np.concatenate([neu_seq_0, neu_seq_1], axis=1)
            model = PCA(n_components=self.d_latent)
            model.fit(neu_x.reshape(neu_x.shape[0],-1).T)
            neu_z_0 = model.transform(neu_seq_0.reshape(neu_seq_0.shape[0],-1).T).T.reshape(self.d_latent, -1)
            neu_z_1 = model.transform(neu_seq_1.reshape(neu_seq_1.shape[0],-1).T).T.reshape(self.d_latent, -1)
            # get data within range.
            l_idx, r_idx   = get_frame_idx_from_time(self.alignment['neu_time'], 0, stim_seq[c_idx,0]-500, stim_seq[c_idx+1,1]+500)
            neu_time = self.alignment['neu_time'][l_idx:r_idx]
            neu_seq_0 = neu_seq_0[:,l_idx:r_idx]
            neu_seq_1 = neu_seq_1[:,l_idx:r_idx]
            neu_z_0 = neu_z_0[:,l_idx:r_idx]
            neu_z_1 = neu_z_1[:,l_idx:r_idx]
            cmap_0, c_neu_0 = get_cmap_color(neu_seq_0.shape[1], base_color=['lemonchiffon']+base_color1+['black'], return_cmap=True)
            cmap_1, c_neu_1 = get_cmap_color(neu_seq_1.shape[1], base_color=['lemonchiffon']+base_color2+['black'], return_cmap=True)
            # random rotate dynamics.
            for ai in range(len(axs)):
                rm = get_random_rotate_mat_3d()
                axs[ai].axis('off')
                ax0 = axs[ai].inset_axes([0, 0, 0.6, 0.6], transform=axs[ai].transAxes, projection='3d')
                ax10 = axs[ai].inset_axes([0, 0.6, 0.3, 0.08], transform=axs[ai].transAxes)
                ax11 = axs[ai].inset_axes([0, 0.8, 0.3, 0.08], transform=axs[ai].transAxes)
                ax20 = axs[ai].inset_axes([0.6, 0, 0.1, 0.4], transform=axs[ai].transAxes)
                ax21 = axs[ai].inset_axes([0.8, 0, 0.1, 0.4], transform=axs[ai].transAxes)
                # fix.
                if 0 in fix_jitter:
                    neu_mean_0, neu_sem_0 = get_mean_sem(neu_seq_0)
                    upper = np.nanmax(neu_mean_0) + np.nanmax(neu_sem_0)
                    lower = np.nanmin(neu_mean_0) - np.nanmax(neu_sem_0)
                    for si in [-1,0,1]:
                        ax10.fill_between(
                            stim_seq[c_idx+si, :],
                            lower - 0.1 * (upper - lower), upper + 0.1 * (upper - lower),
                            color=color0, edgecolor='none', alpha=0.25, step='mid')
                    if oddball == 0:
                        ax10.axvline(stim_seq[c_idx + 1, 0], color='gold', lw=1, linestyle='--')
                    if oddball == 1:
                        ax10.axvline(stim_seq[c_idx, 1] + self.expect, color='gold', lw=1, linestyle='--')
                    for t in range(neu_mean_0.shape[0] - 1):
                        ax10.plot(neu_time[t:t + 2], neu_mean_0[t:t + 2], color=c_neu_0[t])
                    ax10.scatter(neu_time[0], neu_mean_0[0], color='black', marker='x', lw=2)
                    ax10.scatter(neu_time[-1], neu_mean_0[-1], color='black', marker='o', lw=2)
                    t_unexpect = [stim_seq[c_idx + 1, 0], stim_seq[c_idx, 1] + self.expect][oddball]
                    self.plot_3d_latent_dynamics(
                        ax0, np.matmul(rm, neu_z_0), stim_seq, neu_time,
                        end_color='black', c_stim=c_stim, cmap=cmap_0, add_mark=[(t_unexpect, 'gold')])
                    hide_all_axis(ax10)
                    ax10.set_xlim(xlim)
                    ax10.set_ylim([lower - 0.1 * (upper - lower), upper + 0.1 * (upper - lower)])
                    add_heatmap_colorbar(
                        ax20, cmap_0, None, 'time since pre oddball stim (ms)',
                        [int(neu_time[0] + 0.2 * (neu_time[-1] - neu_time[0])),
                         int(neu_time[-1] - 0.2 * (neu_time[-1] - neu_time[0]))])
                # jitter.
                if 1 in fix_jitter:
                    neu_mean_1, neu_sem_1 = get_mean_sem(neu_seq_1)
                    upper = np.nanmax(neu_mean_1) + np.nanmax(neu_sem_1)
                    lower = np.nanmin(neu_mean_1) - np.nanmax(neu_sem_1)
                    for si in range(stim_seq.shape[0]):
                        ax11.fill_between(
                            stim_seq[si, :],
                            lower - 0.1 * (upper - lower), upper + 0.1 * (upper - lower),
                            color=color0, edgecolor='none', alpha=0.25, step='mid')
                    if oddball == 0:
                        ax11.axvline(stim_seq[c_idx + 1, 0], color='gold', lw=1, linestyle='--')
                    if oddball == 1:
                        ax11.axvline(stim_seq[c_idx, 1] + self.expect, color='gold', lw=1, linestyle='--')
                    for t in range(neu_mean_1.shape[0] - 1):
                        ax11.plot(neu_time[t:t + 2], neu_mean_1[t:t + 2], color=c_neu_1[t])
                    ax11.scatter(neu_time[0], neu_mean_1[0], color='black', marker='x', lw=2)
                    ax11.scatter(neu_time[-1], neu_mean_1[-1], color='black', marker='o', lw=2)
                    t_unexpect = [stim_seq[c_idx + 1, 0], stim_seq[c_idx, 1] + self.expect][oddball]
                    self.plot_3d_latent_dynamics(
                        ax0, np.matmul(rm, neu_z_1), stim_seq, neu_time,
                        end_color='black', c_stim=c_stim, cmap=cmap_1, add_mark=[(t_unexpect, 'gold')])
                    hide_all_axis(ax11)
                    ax11.set_xlim(xlim)
                    ax11.set_ylim([lower - 0.1 * (upper - lower), upper + 0.1 * (upper - lower)])
                    add_heatmap_colorbar(
                        ax21, cmap_1, None, 'time since pre oddball stim (ms)',
                        [int(neu_time[0] + 0.2 * (neu_time[-1] - neu_time[0])),
                         int(neu_time[-1] - 0.2 * (neu_time[-1] - neu_time[0]))])
                adjust_layout_3d_latent(ax0)
        def plot_jitter_local_oddball(axs, oddball):
            base_color1 = ['coral', 'crimson']
            base_color2 = ['mediumseagreen', 'forestgreen']
            # collect data.
            [_, [neu_seq, stim_seq, camera_pupil, pre_isi, _], _, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_idx=[l[oddball] for l in self.list_odd_idx],
                trial_param=[None, None, [1], None, [0], [0]],
                mean_sem=False,
                cate=cate, roi_id=None)
            # bin data based on isi.
            [bins, bin_center, _, bin_neu_seq, _, _, bin_stim_seq, bin_camera_pupil] = get_isi_bin_neu(
                neu_seq, stim_seq, camera_pupil, pre_isi, self.bin_win, self.bin_num)
            stim_seq = np.nanmean(np.concatenate(stim_seq, axis=0), axis=0)
            c_idx = stim_seq.shape[0]//2
            c_stim = [color0] * stim_seq.shape[-2]
            # fit model.
            neu_x = np.concatenate(bin_neu_seq, axis=1)
            model = PCA(n_components=self.d_latent)
            model.fit(neu_x.reshape(neu_x.shape[0],-1).T)
            neu_z_0 = model.transform(bin_neu_seq[0].reshape(bin_neu_seq[0].shape[0],-1).T).T.reshape(self.d_latent, -1)
            neu_z_1 = model.transform(bin_neu_seq[1].reshape(bin_neu_seq[1].shape[0],-1).T).T.reshape(self.d_latent, -1)
            # get data within range.
            l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, stim_seq[c_idx,0]-500, stim_seq[c_idx+1,1]+500)
            neu_time = self.alignment['neu_time'][l_idx:r_idx]
            neu_seq_0 = bin_neu_seq[0][:,l_idx:r_idx]
            neu_seq_1 = bin_neu_seq[1][:,l_idx:r_idx]
            neu_z_0 = neu_z_0[:,l_idx:r_idx]
            neu_z_1 = neu_z_1[:,l_idx:r_idx]
            cmap_0, c_neu_0 = get_cmap_color(neu_seq_0.shape[1], base_color=['lemonchiffon']+base_color1+['black'], return_cmap=True)
            cmap_1, c_neu_1 = get_cmap_color(neu_seq_1.shape[1], base_color=['lemonchiffon']+base_color2+['black'], return_cmap=True)
            # random rotate dynamics.
            for ai in range(len(axs)):
                # get random matrix.
                rm = get_random_rotate_mat_3d()
                # define layouts.
                axs[ai].axis('off')
                ax0 = axs[ai].inset_axes([0,   0,   0.6, 0.6], transform=axs[ai].transAxes, projection='3d')
                ax10 = axs[ai].inset_axes([0,  0.6, 0.3, 0.08], transform=axs[ai].transAxes)
                ax11 = axs[ai].inset_axes([0,  0.8, 0.3, 0.08], transform=axs[ai].transAxes)
                ax20 = axs[ai].inset_axes([0.6, 0,   0.1, 0.4], transform=axs[ai].transAxes)
                ax21 = axs[ai].inset_axes([0.8, 0,   0.1, 0.4], transform=axs[ai].transAxes)
                # plot mean trace.
                neu_mean_0, neu_sem_0 = get_mean_sem(neu_seq_0)
                neu_mean_1, neu_sem_1 = get_mean_sem(neu_seq_1)
                # find bounds.
                upper = np.nanmax([neu_mean_0, neu_mean_1]) + np.nanmax([neu_sem_0, neu_sem_1])
                lower = np.nanmin([neu_mean_0, neu_mean_1]) - np.nanmax([neu_sem_0, neu_sem_1])
                # plot stimulus.
                for si in [0, 1]:
                    ax10.fill_between(
                        stim_seq[c_idx+si,:],
                        lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
                        color=color0, edgecolor='none', alpha=0.25, step='mid')
                    ax11.fill_between(
                        stim_seq[c_idx+si,:],
                        lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
                        color=color0, edgecolor='none', alpha=0.25, step='mid')
                if oddball == 0:
                    ax10.axvline(stim_seq[c_idx+1,0], color='gold', lw=1, linestyle='--')
                    ax11.axvline(stim_seq[c_idx+1,0], color='gold', lw=1, linestyle='--')
                if oddball == 1:
                    ax10.axvline(stim_seq[c_idx,1]+self.expect, color='gold', lw=1, linestyle='--')
                    ax11.axvline(stim_seq[c_idx,1]+self.expect, color='gold', lw=1, linestyle='--')
                # plot neural traces.
                for t in range(neu_mean_0.shape[0]-1):
                    ax10.plot(neu_time[t:t+2], neu_mean_0[t:t+2], color=c_neu_0[t])
                    ax11.plot(neu_time[t:t+2], neu_mean_1[t:t+2], color=c_neu_1[t])
                ax10.scatter(neu_time[0], neu_mean_0[0], color='black', marker='x', lw=2)
                ax11.scatter(neu_time[0], neu_mean_1[0], color='black', marker='x', lw=2)
                ax10.scatter(neu_time[-1], neu_mean_0[-1], color='black', marker='o', lw=2)
                ax11.scatter(neu_time[-1], neu_mean_1[-1], color='black', marker='o', lw=2)
                # plot 3d dynamics.
                t_unexpect = [stim_seq[c_idx+1,0], stim_seq[c_idx,1]+self.expect][oddball]
                self.plot_3d_latent_dynamics(ax0, np.matmul(rm, neu_z_0), stim_seq, neu_time, c_stim, cmap=cmap_0, add_mark=[(t_unexpect, 'gold')])
                self.plot_3d_latent_dynamics(ax0, np.matmul(rm, neu_z_1), stim_seq, neu_time, c_stim, cmap=cmap_1, add_mark=[(t_unexpect, 'gold')])
                # adjust layouts.
                for axi in [ax10, ax11]:
                    hide_all_axis(axi)
                    axi.set_xlim(xlim)
                    axi.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
                adjust_layout_3d_latent(ax0)
                # add colorbar.
                yticklabels = [int(neu_time[0]+0.2*(neu_time[-1]-neu_time[0])), int(neu_time[-1]-0.2*(neu_time[-1]-neu_time[0]))]
                add_heatmap_colorbar(ax20, cmap_0, None, 'time since pre oddball stim (ms)', yticklabels)
                add_heatmap_colorbar(ax21, cmap_1, None, 'time since pre oddball stim (ms)', yticklabels)
        # plot all.
        try: plot_jitter_global_oddball(axs[0], 0, [0])
        except: traceback.print_exc()
        try: plot_jitter_global_oddball(axs[1], 0, [1])
        except: traceback.print_exc()
        try: plot_jitter_global_oddball(axs[2], 0, [0,1])
        except: traceback.print_exc()
        try: plot_jitter_global_oddball(axs[3], 1, [0])
        except: traceback.print_exc()
        try: plot_jitter_global_oddball(axs[4], 1, [1])
        except: traceback.print_exc()
        try: plot_jitter_global_oddball(axs[5], 1, [0,1])
        except: traceback.print_exc()
        
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

            except: traceback.print_exc()
            
    def cluster_oddball_jitter_global_all(self, axs_all):
        for cate, axs in zip(self.cate_list, axs_all):
            try:
                label_name = self.label_names[str(cate[0])] if len(cate)==1 else 'all'
                print(f'plotting results for {label_name}')
                
                self.plot_cluster_oddball_jitter_global_all(axs, 'global', cate=cate)

            except: traceback.print_exc()
    
    def cluster_oddball_jitter_local_all(self, axs_all):
        for cate, axs in zip(self.cate_list, axs_all):
            try:
                label_name = self.label_names[str(cate[0])] if len(cate)==1 else 'all'
                print(f'plotting results for {label_name}')
                
                self.plot_cluster_oddball_jitter_local_all(axs, 'global', cate=cate)
                
            except: traceback.print_exc()
    
    def cluster_oddball_fix_heatmap_all(self, axs_all):
        for cate, axs in zip(self.cate_list, axs_all):
            try:
                label_name = self.label_names[str(cate[0])] if len(cate)==1 else 'all'
                print(f'plotting results for {label_name}')
                
                self.plot_cluster_oddball_fix_heatmap_all(axs, cate=cate)

            except: traceback.print_exc()
        
    def sorted_heatmaps_fix_all(self, axs_all):
        for cate, axs in zip(self.cate_list, axs_all):
            try:
                label_name = self.label_names[str(cate[0])] if len(cate)==1 else 'all'
                print(f'plotting results for {label_name}')
                
                for axss, norm_mode in zip([axs[0:3], axs[3:6], axs[6:9]], ['none', 'minmax', 'share']):
                    self.plot_sorted_heatmaps_fix_all(axss, norm_mode, cate=cate)

            except: traceback.print_exc()
    
    def latent_all(self, axs_all):
        for cate, axs in zip(self.cate_list, axs_all):
            try:
                label_name = self.label_names[str(cate[0])] if len(cate)==1 else 'all'
                print(f'plotting results for {label_name}')
                
                self.plot_latent_all(axs, cate=cate)

            except: traceback.print_exc()
