#!/usr/bin/env python3

import traceback
import numpy as np
import matplotlib.ticker as mtick
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

from modules.Alignment import run_get_stim_response
from modeling.clustering import get_mean_sem_cluster
from modeling.clustering import get_cluster_cate
from modeling.decoding import decoding_time_confusion
from modeling.decoding import regression_time_frac
from modeling.generative import get_glm_cate
from modeling.quantifications import fit_trf_model

from utils import show_resource_usage
from utils import get_norm01_params
from utils import get_odd_stim_prepost_idx
from utils import get_mean_sem_win
from utils import get_mean_sem
from utils import get_neu_trial
from utils import get_frame_idx_from_time
from utils import get_split_idx
from utils import get_expect_interval
from utils import get_block_1st_idx
from utils import get_block_transition_idx
from utils import exclude_odd_stim
from utils import get_temporal_scaling_data
from utils import get_cmap_color
from utils import hide_all_axis
from utils import get_random_rotate_mat_3d
from utils import adjust_layout_isi_example_epoch
from utils import adjust_layout_neu
from utils import adjust_layout_3d_latent
from utils import add_legend
from utils import add_heatmap_colorbar
from utils import utils_basic

# fig, axs = plt.subplots(1, 2, figsize=(6, 6))
# fig, ax = plt.subplots(1, 1, figsize=(2, 20))
# fig, ax = plt.subplots(1, 1, figsize=(3, 6))
# fig, axs = plt.subplots(1, 8, figsize=(24, 3))
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
        self.expect = np.nanmin(np.array([get_expect_interval(sl) for sl in self.list_stim_labels]),axis=0)
        self.list_block_start = [get_block_1st_idx(sl, 3) for sl in self.list_stim_labels]
        self.list_significance = list_significance
        self.bin_win = [450,2550]
        self.bin_num = 2
        self.d_latent = 3
        self.glm = self.run_glm()
        self.n_up = 2
        self.n_dn = 3
        self.cluster_id = self.run_clustering(self.n_up, self.n_dn)

    def plot_isi_seting(self, ax):
        ax.vlines(1000, 0, 1, color='dodgerblue')
        ax.vlines(2000, 0, 1, color='springgreen')
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xlim([450,2550])
        ax.set_ylim([0, 1.05])
        ax.set_xticks([1000,2000])
        ax.set_yticks([])
        ax.set_xticklabels([1000,2000])

    def plot_isi_example_epoch(self, ax):
        trial_win = [1000,1500]
        # get isi and trial labels.
        stim_labels = self.list_neural_trials[0]['stim_labels'][trial_win[0]:trial_win[1],:]
        isi = stim_labels[1:,0] - stim_labels[:-1,1]
        img_seq_label  = stim_labels[:-1,2]
        standard_types = stim_labels[:-1,3]
        # plot trials.
        colors = np.array(['dodgerblue', 'springgreen', 'black'], dtype=object)
        colors = colors[np.where(img_seq_label == -1, 2, standard_types)]
        ax.scatter(np.arange(trial_win[0], trial_win[1]-1), isi, c=colors, s=5)
        # adjust layouts.
        adjust_layout_isi_example_epoch(ax, trial_win, self.bin_win)
        
    def plot_cluster_all(self, axs, cate):
        color0 = 'dimgrey'
        color1 = 'dodgerblue'
        color2 = 'springgreen'
        cluster_id, neu_labels = get_cluster_cate(self.cluster_id, self.list_labels, self.list_significance, cate)
        # collect data.
        [_, [neu_seq_0, _, stim_seq_0, _], _, _] = get_neu_trial(
            self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
            trial_param=[[2,3,4,5], [0], None, None, [0], [0]],
            cate=cate, roi_id=None)
        [_, [neu_seq_1, _, stim_seq_1, _], _, _] = get_neu_trial(
            self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
            trial_param=[[2,3,4,5], [1], None, None, [0], [0]],
            cate=cate, roi_id=None)
        c_idx = stim_seq_0.shape[0]//2
        @show_resource_usage
        def plot_glm_kernel(ax):
            kernel_all = get_glm_cate(self.glm, self.list_labels, self.list_significance, cate)
            self.plot_glm_kernel(ax, kernel_all, cluster_id, color0, 0.9)
        @show_resource_usage
        def plot_standard_heatmap(ax, norm_mode):
            xlim = [-2000,3000]
            l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, xlim[0], xlim[1])
            # collect data.
            neu_time = self.alignment['neu_time'][l_idx:r_idx]
            neu_ci_0 = [neu_seq_0[cluster_id==ci,l_idx:r_idx] for ci in range(self.n_clusters)]
            neu_ci_1 = [neu_seq_1[cluster_id==ci,l_idx:r_idx] for ci in range(self.n_clusters)]
            # define layouts.
            ax0 = ax.inset_axes([0, 0, 0.4, 0.9], transform=ax.transAxes)
            ax1 = ax.inset_axes([0.6, 0, 0.4, 0.9], transform=ax.transAxes)
            axs_hm_0 = [ax0.inset_axes([0.2, ci/self.n_clusters, 0.5, 0.75/self.n_clusters], transform=ax0.transAxes)
                      for ci in range(self.n_clusters)]
            axs_cb_0 = [ax0.inset_axes([0.8, ci/self.n_clusters, 0.1, 0.75/self.n_clusters], transform=ax0.transAxes)
                      for ci in range(self.n_clusters)]
            axs_hm_1 = [ax1.inset_axes([0.2, ci/self.n_clusters, 0.5, 0.75/self.n_clusters], transform=ax1.transAxes)
                      for ci in range(self.n_clusters)]
            axs_cb_1 = [ax1.inset_axes([0.8, ci/self.n_clusters, 0.1, 0.75/self.n_clusters], transform=ax1.transAxes)
                      for ci in range(self.n_clusters)]
            axs_hm_0.reverse()
            axs_cb_0.reverse()
            axs_hm_1.reverse()
            axs_cb_1.reverse()
            # plot results for each class.
            for ci in range(self.n_clusters):
                if np.sum(cluster_id==ci) > 0:
                    self.plot_heatmap_neuron(axs_hm_0[ci], axs_cb_0[ci], neu_ci_0[ci], neu_time, neu_ci_0[ci],
                                             norm_mode='minmax', sort_method='rastermap')
                    self.plot_heatmap_neuron(axs_hm_1[ci], axs_cb_1[ci], neu_ci_1[ci], neu_time, neu_ci_0[ci],
                                             norm_mode='minmax', sort_method='rastermap')
                    # add stimulus line.
                    xlines_0 = [self.alignment['neu_time'][np.searchsorted(self.alignment['neu_time'], t)]
                              for t in [stim_seq_0[c_idx+i,0] for i in [-2,-1,0,1,2]]]
                    xlines_1 = [self.alignment['neu_time'][np.searchsorted(self.alignment['neu_time'], t)]
                              for t in [stim_seq_1[c_idx+i,0] for i in [-2,-1,0,1,2]]]
                    for xl in xlines_0:
                        if xl>neu_time[0] and xl<neu_time[-1]:
                            axs_hm_0[ci].axvline(xl, color='black', lw=1, linestyle='--')
                    for xl in xlines_1:
                        if xl>neu_time[0] and xl<neu_time[-1]:
                            axs_hm_1[ci].axvline(xl, color='black', lw=1, linestyle='--')
                # adjust layouts.
                axs_hm_0[ci].tick_params(axis='y', labelrotation=0)
                axs_hm_1[ci].tick_params(axis='y', labelrotation=0)
                axs_hm_0[ci].set_ylabel(None)
                axs_hm_1[ci].set_ylabel(None)
                if ci != self.n_clusters-1:
                    axs_hm_0[ci].set_xticks([])
                    axs_hm_1[ci].set_xticks([])
            ax.set_xlabel('time since stim (ms)')
            ax.set_ylabel('neuron id')
            hide_all_axis(ax)
            hide_all_axis(ax0)
            hide_all_axis(ax1)
        @show_resource_usage
        def plot_standard(ax, standard, scaled):
            xlim = [-2500,3000]
            # get response within cluster.
            neu_mean_0, neu_sem_0 = get_mean_sem_cluster(neu_seq_0, self.n_clusters, cluster_id)
            neu_mean_1, neu_sem_1 = get_mean_sem_cluster(neu_seq_1, self.n_clusters, cluster_id)
            if scaled:
                norm_params = [get_norm01_params(np.concatenate([neu_mean_0[ci,:], neu_mean_1[ci,:]])) for ci in range(self.n_clusters)]
            else:
                norm_params = [get_norm01_params(np.concatenate([neu_mean_0, neu_mean_1])) for ci in range(self.n_clusters)]
            # define layouts.
            ax.axis('off')
            ax = ax.inset_axes([0, 0, 1, 0.9], transform=ax.transAxes)
            # plot results.
            if standard == 0 :
                self.plot_cluster_mean_sem(
                    ax, neu_mean_0, neu_sem_0,
                    self.alignment['neu_time'], norm_params,
                    stim_seq_0, [color0]*stim_seq_0.shape[0], [color1]*self.n_clusters, xlim)
            if standard == 1 :
                self.plot_cluster_mean_sem(
                    ax, neu_mean_1, neu_sem_1,
                    self.alignment['neu_time'], norm_params,
                    stim_seq_1, [color0]*stim_seq_1.shape[0], [color2]*self.n_clusters, xlim)
            if standard == 'both':
                xlim_0 = [stim_seq_0[c_idx-1,1], stim_seq_0[c_idx+1,0]]
                xlim_1 = [stim_seq_1[c_idx-1,1], stim_seq_1[c_idx+1,0]]
                l_0, r_0 = get_frame_idx_from_time(self.alignment['neu_time'], 0, xlim_0[0], xlim_0[1])
                l_1, r_1 = get_frame_idx_from_time(self.alignment['neu_time'], 0, xlim_1[0], xlim_1[1])
                neu_mean_0 = neu_mean_0[:,l_0:r_0]
                neu_mean_1 = neu_mean_1[:,l_1:r_1]
                neu_sem_0 = neu_sem_0[:,l_0:r_0]
                neu_sem_1 = neu_sem_1[:,l_1:r_1]
                self.plot_cluster_mean_sem(
                    ax, neu_mean_0, neu_sem_0,
                    self.alignment['neu_time'][l_0:r_0], norm_params,
                    stim_seq_0[c_idx,:].reshape(1,2), [color0]*stim_seq_0.shape[0], [color1]*self.n_clusters, xlim, False)
                self.plot_cluster_mean_sem(
                    ax, neu_mean_1, neu_sem_1,
                    self.alignment['neu_time'][l_1:r_1], norm_params,
                    stim_seq_1[c_idx,:].reshape(1,2), [color0]*stim_seq_1.shape[0], [color2]*self.n_clusters, xlim)
            # adjust layouts.
            ax.set_xlabel('time since stim (ms)')
            ax.set_xlim(xlim)
        @show_resource_usage
        def plot_standard_scale(ax, scaled):
            target_isi = 1500
            t_onset, t_offset = get_frame_idx_from_time(
                self.alignment['neu_time'], 0, stim_seq_0[c_idx,0], stim_seq_0[c_idx,1])
            l_0, r_0 = get_frame_idx_from_time(
                self.alignment['neu_time'], 0, stim_seq_0[c_idx-1,1], stim_seq_0[c_idx+1,0])
            l_1, r_1 = get_frame_idx_from_time(
                self.alignment['neu_time'], 0, stim_seq_1[c_idx-1,1], stim_seq_1[c_idx+1,0])
            l_target, r_target = get_frame_idx_from_time(
                self.alignment['neu_time'], 0, -target_isi, stim_seq_0[c_idx,1]+target_isi)
            # scale data.
            neu_seq_0_scale = np.concatenate([
                get_temporal_scaling_data(
                    neu_seq_0[:,l_0:t_onset],
                    self.alignment['neu_time'][l_0:t_onset],
                    self.alignment['neu_time'][l_target:t_onset]),
                np.nanmean(neu_seq_0[:,t_onset:t_offset],axis=1).reshape(-1,1),
                get_temporal_scaling_data(
                    neu_seq_0[:,t_offset:r_0],
                    self.alignment['neu_time'][t_offset:r_0],
                    self.alignment['neu_time'][t_offset:r_target])],
                axis=1)
            neu_seq_1_scale = np.concatenate([
                get_temporal_scaling_data(
                    neu_seq_1[:,l_1:t_onset],
                    self.alignment['neu_time'][l_1:t_onset],
                    self.alignment['neu_time'][l_target:t_onset]),
                np.nanmean(neu_seq_1[:,t_onset:t_offset],axis=1).reshape(-1,1),
                get_temporal_scaling_data(
                    neu_seq_1[:,t_offset:r_1],
                    self.alignment['neu_time'][t_offset:r_1],
                    self.alignment['neu_time'][t_offset:r_target])],
                axis=1)
            neu_time = np.concatenate([
                self.alignment['neu_time'][l_target:t_onset],
                np.nanmean(self.alignment['neu_time'][t_onset:t_offset]).reshape(-1),
                self.alignment['neu_time'][t_offset:r_target],
                ])
            # get response within cluster.
            neu_mean_0, neu_sem_0 = get_mean_sem_cluster(neu_seq_0_scale, self.n_clusters, cluster_id)
            neu_mean_1, neu_sem_1 = get_mean_sem_cluster(neu_seq_1_scale, self.n_clusters, cluster_id)
            if scaled:
                norm_params = [get_norm01_params(np.concatenate([neu_mean_0,neu_mean_1], axis=1)[ci,:]) for ci in range(self.n_clusters)]
            else:
                norm_params = [get_norm01_params(np.concatenate([neu_mean_0,neu_mean_1], axis=1)) for ci in range(self.n_clusters)]
            # define layouts.
            ax.axis('off')
            ax = ax.inset_axes([0, 0, 0.75, 0.9], transform=ax.transAxes)
            # plot results.
            ax.axvline(0, color=color0, lw=1, linestyle='-')
            self.plot_cluster_mean_sem(
                ax, neu_mean_0, neu_sem_0,
                neu_time, norm_params,
                None, None, [color1]*self.n_clusters, [neu_time[0], neu_time[-1]])
            self.plot_cluster_mean_sem(
                ax, neu_mean_1, neu_sem_1,
                neu_time, norm_params,
                None, None, [color2]*self.n_clusters, [neu_time[0], neu_time[-1]])
            # adjust layouts.
            ax.set_xlabel('time since stim (ms)')
        @show_resource_usage
        def plot_neu_fraction(ax):
            # define layouts.
            ax.axis('off')
            ax = ax.inset_axes([0, 0, 1, 0.9], transform=ax.transAxes)
            self.plot_cluster_neu_fraction_in_cluster(ax, cluster_id, color0)
        @show_resource_usage
        def plot_fraction(ax):
            # define layouts.
            ax.axis('off')
            ax = ax.inset_axes([0, 0, 1, 0.9], transform=ax.transAxes)
            self.plot_cluster_cate_fraction_in_cluster(ax, cluster_id, neu_labels, self.label_names, color0)
        @show_resource_usage
        def plot_legend(ax):
            [_, [neu_seq, stim_seq, camera_pupil, pre_isi, post_isi],
             [neu_labels, _],
             [n_trials, n_neurons]] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_param=[[2,3,4,5], None, None, None, [0], [0]],
                mean_sem=False,
                cate=cate, roi_id=None)
            cs = [color1, color2]
            lbl= ['short', 'long']
            add_legend(ax, cs, lbl, n_trials, n_neurons, self.n_sess, 'upper right')
            ax.axis('off')
        try: plot_glm_kernel(axs[0])
        except: traceback.print_exc()()
        try: plot_neu_fraction(axs[1])
        except: traceback.print_exc()()
        try: plot_fraction(axs[2])
        except: traceback.print_exc()()
        try: plot_legend(axs[3])
        except: traceback.print_exc()()
        try: plot_standard_heatmap(axs[4], 'none')
        except: traceback.print_exc()()
        try: plot_standard_heatmap(axs[5], 'minmax')
        except: traceback.print_exc()()
        try: plot_standard_heatmap(axs[6], 'share')
        except: traceback.print_exc()()
        try: plot_standard(axs[7], 0, False)
        except: traceback.print_exc()()
        try: plot_standard(axs[8], 0, True)
        except: traceback.print_exc()()
        try: plot_standard(axs[9], 1, False)
        except: traceback.print_exc()()
        try: plot_standard(axs[10], 1, True)
        except: traceback.print_exc()()
        try: plot_standard(axs[11], 'both', False)
        except: traceback.print_exc()()
        try: plot_standard(axs[12], 'both', True)
        except: traceback.print_exc()()
        try: plot_standard_scale(axs[13], False)
        except: traceback.print_exc()()
        try: plot_standard_scale(axs[14], True)
        except: traceback.print_exc()()
    
    def plot_cluster_heatmap_all(self, axs, cate):
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
            # define layouts.
            ax.axis('off')
            ax_hm = ax.inset_axes([0, 0, 0.6, 1], transform=ax.transAxes)
            ax_cb = ax.inset_axes([0.8, 0, 0.1, 1], transform=ax.transAxes)
            # collect data.
            [[_, _, _, cmap], [_, _, stim_seq, _], _, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_param=[[2,3,4,5], None, [0], None, [0], [0]],
                cate=cate, roi_id=None)
            c_idx = stim_seq.shape[0]//2
            # plot results.
            self.plot_heatmap_neuron(ax_hm, ax_cb, kernel_all, self.glm['kernel_time'], kernel_all, norm_mode='minmax')
            # adjust layouts.
            ax.set_xlabel('time since stim (ms)')
            ax.axvline(stim_seq[c_idx,0], color='black', lw=1, linestyle='--')
        @show_resource_usage
        def plot_standard(ax, standard, group):
            xlim = [-2500,3000]
            l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, xlim[0], xlim[1])
            # collect data.
            _, [neu_seq, _, stim_seq, _], _, _ = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_param=[[2,3,4,5], [standard], None, None, [0], [0]],
                cate=cate, roi_id=None)
            c_idx = stim_seq.shape[0]//2
            neu_seq = neu_seq[:,l_idx:r_idx]
            neu_time = self.alignment['neu_time'][l_idx:r_idx]
            # define layouts.
            ax_hm = ax.inset_axes([0, 0, 0.7, 1], transform=ax.transAxes)
            ax_cb = ax.inset_axes([0.8, 0, 0.1, 1], transform=ax.transAxes)
            # plot results. 
            if group:
                self.plot_cluster_heatmap(ax_hm, neu_seq, neu_time, cluster_id, 'none')
            else:
                self.plot_heatmap_neuron(
                    ax_hm, ax_cb, neu_seq, neu_time, neu_seq, norm_mode='none')
            # add stimulus line.
            for bi in range(3):
                xlines = [self.alignment['neu_time'][np.searchsorted(self.alignment['neu_time'], t)]
                          for t in [stim_seq[c_idx+i,0]
                                    for i in [-2,-1,0,1,2]]]
                for xl in xlines:
                    if xl>neu_time[0] and xl<neu_time[-1]:
                        ax_hm.axvline(xl, color='black', lw=1, linestyle='--')
            # adjust layouts.
            ax_hm.set_xlabel('time since stim (ms)')
            hide_all_axis(ax)
        # plot all.
        try: plot_cluster_features(axs[0])
        except: traceback.print_exc()()
        try: plot_hierarchical_dendrogram(axs[1])
        except: traceback.print_exc()()
        try: plot_glm_kernel(axs[2])
        except: traceback.print_exc()()
        try: plot_standard(axs[3], 0, True)
        except: traceback.print_exc()()
        try: plot_standard(axs[4], 1, True)
        except: traceback.print_exc()()
        try: plot_standard(axs[5], 0, False)
        except: traceback.print_exc()()
        try: plot_standard(axs[6], 1, False)
        except: traceback.print_exc()()

    def plot_cluster_adapt_all(self, axs, cate=None):
        color0 = 'dimgrey'
        color1 = 'dodgerblue'
        color2 = 'springgreen'
        trials_around = 40
        cluster_id, neu_labels = get_cluster_cate(self.cluster_id, self.list_labels, self.list_significance, cate)
        split_idx = get_split_idx(self.list_labels, self.list_significance, cate)
        day_cluster_id = np.split(cluster_id, split_idx)
        # collect data.
        [_, [_, _, stim_seq, _], _,
         [n_trials, n_neurons]] = get_neu_trial(
            self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
            trial_param=[None, None, None, None, [0], [0]],
            cate=cate, roi_id=None)
        c_idx = stim_seq.shape[0]//2
        # get transition trials indice.
        list_trans_0to1 = [get_block_transition_idx(sl[:,3], trials_around)[0] for sl in self.list_stim_labels]
        list_trans_1to0 = [get_block_transition_idx(sl[:,3], trials_around)[1] for sl in self.list_stim_labels]
        list_trans_0to1 = [np.nansum(ti, axis=0).astype('bool') for ti in list_trans_0to1]
        list_trans_1to0 = [np.nansum(ti, axis=0).astype('bool') for ti in list_trans_1to0]
        @show_resource_usage
        def plot_tansition_trial_heatmap(ax, norm_mode):
            xlim = [0,2500]
            trials_eval = 10
            gap_margin = 5
            # collect data.
            [_, [neu_trans_0to1, _, _, _, _], _, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_param=[None, None, None, None, [0], [0]],
                trial_idx=list_trans_0to1,
                mean_sem=False,
                cate=cate, roi_id=None)
            [_, [neu_trans_1to0, _, _, _, _], _, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_param=[None, None, None, None, [0], [0]],
                trial_idx=list_trans_1to0,
                mean_sem=False,
                cate=cate, roi_id=None)
            # reshape into [n_transition*n_trials*n_neurons*n_times]
            neu_trans_0to1 = [neu.reshape([-1, 2*trials_around, neu.shape[1], neu.shape[2]]) for neu in neu_trans_0to1]
            neu_trans_1to0 = [neu.reshape([-1, 2*trials_around, neu.shape[1], neu.shape[2]]) for neu in neu_trans_1to0]
            n_transition = np.array([nt.shape[0] for nt in neu_trans_0to1])
            if np.max(n_transition) != np.min(n_transition):
                neu_trans_0to1 = [nt[:np.min(n_transition),:,:,:] for nt in neu_trans_0to1]
            n_transition = np.array([nt.shape[0] for nt in neu_trans_1to0])
            if np.max(n_transition) != np.min(n_transition):
                neu_trans_1to0 = [nt[:np.min(n_transition),:,:,:] for nt in neu_trans_1to0]
            # get less trials around transition.
            neu_trans_0to1 = [nt[:, trials_around-trials_eval:trials_around+trials_eval, :, :] for nt in neu_trans_0to1]
            neu_trans_1to0 = [nt[:, trials_around-trials_eval:trials_around+trials_eval, :, :] for nt in neu_trans_1to0]
            # get average across transition [n_trials*n_neurons*n_times]
            neu_trans_0to1 = np.concatenate(neu_trans_0to1, axis=2)
            neu_trans_1to0 = np.concatenate(neu_trans_1to0, axis=2)
            neu_0to1 = np.nanmean(neu_trans_0to1, axis=0)
            neu_1to0 = np.nanmean(neu_trans_1to0, axis=0)
            # find stimulus timing.
            z, _ = get_frame_idx_from_time(self.alignment['neu_time'], 0, 0, 0)
            e1, e2 = get_frame_idx_from_time(
                self.alignment['neu_time'], 0,
                stim_seq[c_idx,1]+self.expect[0],
                stim_seq[c_idx,1]+self.expect[1])
            neu_time = self.alignment['neu_time'][z:e2]
            # extract transition for each class.
            neu_x = []
            for ci in range(self.n_clusters):
                neu = np.full((4*trials_eval+gap_margin, e2-z), np.nan)
                neu[:trials_eval, :e1-z] = np.nanmean(neu_0to1[0:trials_eval, cluster_id==ci, z:e1], axis=1)
                neu[trials_eval:2*trials_eval, :e2-z] = np.nanmean(neu_0to1[trials_eval:2*trials_eval, cluster_id==ci, z:e2], axis=1)
                neu[2*trials_eval+gap_margin:3*trials_eval+gap_margin, :e2-z] = np.nanmean(neu_1to0[:trials_eval, cluster_id==ci, z:e2], axis=1)
                neu[3*trials_eval+gap_margin:, :e1-z] = np.nanmean(neu_1to0[trials_eval:2*trials_eval, cluster_id==ci, z:e1], axis=1)
                neu_x.append(neu)
            # define layouts.
            ax0 = ax.inset_axes([0.1, 0, 0.5, 0.95], transform=ax.transAxes)
            ax1 = ax.inset_axes([0.7, 0, 0.1, 0.95], transform=ax.transAxes)
            axs_hm = [ax0.inset_axes([0, 0.05+ci/self.n_clusters, 1, 0.8/self.n_clusters], transform=ax0.transAxes)
                      for ci in range(self.n_clusters)]
            axs_cb = [ax1.inset_axes([0, 0.05+ci/self.n_clusters, 1, 0.8/self.n_clusters], transform=ax1.transAxes)
                      for ci in range(self.n_clusters)]
            axs_hm.reverse()
            axs_cb.reverse()
            # plot results for each class.
            self.plot_cluster_heatmap_trial(axs_hm, axs_cb, neu_x, neu_time, norm_mode)
            # adjust layouts.
            for ci in range(self.n_clusters):
                axs_hm[ci].axhline(trials_eval, color='red', lw=1, linestyle='--')
                axs_hm[ci].axhline(3*trials_eval+gap_margin, color='red', lw=1, linestyle='--')
                axs_hm[ci].set_yticks([trials_eval, 3*trials_eval+gap_margin])
                axs_hm[ci].set_yticklabels(['L\u2192S', 'S\u2192L'])
                axs_hm[ci].set_xlim(xlim)
                axs_hm[ci].set_ylim([0, 4*trials_eval+gap_margin])
                if ci != self.n_clusters-1:
                    axs_hm[ci].set_xticklabels([])
            axs_hm[self.n_clusters-1].set_xlabel('time since stim (ms)')
            ax.set_title(f'sorted with {norm_mode}')
            hide_all_axis(ax)
            hide_all_axis(ax0)
            hide_all_axis(ax1)
        @show_resource_usage
        def plot_tansition(ax):
            xlim = [-7500, 7000]
            l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, xlim[0], xlim[1])
            # collect data.
            [_, [neu_trans_0to1, _, stim_seq_0to1, _], _, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_idx=[l[1] for l in self.list_block_start],
                cate=cate, roi_id=None)
            [_, [neu_trans_1to0, _, stim_seq_1to0, _], _, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_idx=[l[0] for l in self.list_block_start],
                cate=cate, roi_id=None)
            neu_time = self.alignment['neu_time'][l_idx:r_idx]
            # define layouts.
            ax0 = ax.inset_axes([0.1, 0, 0.4, 0.95], transform=ax.transAxes)
            ax1 = ax.inset_axes([0.6, 0, 0.4, 0.95], transform=ax.transAxes)
            axs0 = [ax0.inset_axes([0, 0.05+ci/self.n_clusters, 1, 0.8/self.n_clusters], transform=ax0.transAxes)
                    for ci in range(self.n_clusters)]
            axs1 = [ax1.inset_axes([0, 0.05+ci/self.n_clusters, 1, 0.8/self.n_clusters], transform=ax1.transAxes)
                    for ci in range(self.n_clusters)]
            axs0.reverse()
            axs1.reverse()
            # plot results for each class.
            for ci in range(self.n_clusters):
                if np.sum(cluster_id==ci) > 0:
                    neu_mean_0to1, neu_sem_0to1 = get_mean_sem(neu_trans_0to1[cluster_id==ci,l_idx:r_idx])
                    neu_mean_1to0, neu_sem_1to0 = get_mean_sem(neu_trans_1to0[cluster_id==ci,l_idx:r_idx])
                    # find bounds.
                    upper = np.nanmax([neu_mean_0to1, neu_mean_1to0])
                    lower = np.nanmin([neu_mean_0to1, neu_mean_1to0])
                    # plot stimulus.
                    for si in range(stim_seq.shape[0]):
                        axs0[ci].fill_between(
                            stim_seq_0to1[si,:],
                            lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
                            color=color0, edgecolor='none', alpha=0.25, step='mid')
                        axs1[ci].fill_between(
                            stim_seq_1to0[si,:],
                            lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
                            color=color0, edgecolor='none', alpha=0.25, step='mid')
                    axs0[ci].axvline(-self.expect[0], color='gold', lw=1, linestyle='--')
                    axs0[ci].axvline(0, color='red', lw=1, linestyle='--')
                    axs1[ci].axvline(0, color='red', lw=1, linestyle='--')
                    # plot neural traces.
                    z_idx_0 = get_frame_idx_from_time(neu_time, 0, stim_seq_1to0[c_idx-1,1], 0)[0]
                    z_idx_1 = get_frame_idx_from_time(neu_time, 0, stim_seq_0to1[c_idx-1,1], 0)[0]
                    self.plot_mean_sem(
                        axs0[ci], neu_time[:z_idx_1],
                        neu_mean_0to1[:z_idx_1], neu_sem_0to1[:z_idx_1], color1, None)
                    self.plot_mean_sem(
                        axs0[ci], neu_time[z_idx_1:],
                        neu_mean_0to1[z_idx_1:], neu_sem_0to1[z_idx_1:], color2, None)
                    self.plot_mean_sem(
                        axs1[ci], neu_time[:z_idx_0],
                        neu_mean_1to0[:z_idx_0], neu_sem_1to0[:z_idx_0], color2, None)
                    self.plot_mean_sem(
                        axs1[ci], neu_time[z_idx_0:],
                        neu_mean_1to0[z_idx_0:], neu_sem_1to0[z_idx_0:], color1, None)
                    # adjust layouts.
                    adjust_layout_neu(axs0[ci])
                    adjust_layout_neu(axs1[ci])
                    axs0[ci].set_xlim(xlim)
                    axs1[ci].set_xlim(xlim)
                    axs0[ci].set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
                    axs1[ci].set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
                    axs0[ci].set_ylabel(None)
                    axs1[ci].set_ylabel(None)
                    if ci != self.n_clusters-1:
                        axs0[ci].set_xticklabels([])
                        axs1[ci].set_xticklabels([])
                    axs1[ci].set_yticklabels([])
            ax.set_xlabel('time since block transition (ms)')
            ax.set_ylabel('dF/F (z-scored)')
            ax0.set_title('S\u2192L adaptation')
            ax1.set_title('L\u2192S adaptation')
            hide_all_axis(ax)
            hide_all_axis(ax0)
            hide_all_axis(ax1)
        @show_resource_usage
        def plot_trial_quant(ax):
            win = 250
            trials_eval = 5
            # collect data.
            [_, [neu_trans_0to1, _, stim_seq_0to1, _], _,
             [n_trials, n_neurons]] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_idx=[l[1] for l in self.list_block_start],
                cate=cate, roi_id=None)
            [_, [neu_trans_1to0, _, stim_seq_1to0, _], _,
             [n_trials, n_neurons]] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_idx=[l[0] for l in self.list_block_start],
                cate=cate, roi_id=None)
            # compute response within window.
            quant_0to1 = [[get_mean_sem_win(
                neu_trans_0to1[cluster_id==ci,:],
                self.alignment['neu_time'], 0, stim_seq_0to1[c_idx+si,0], stim_seq_0to1[c_idx+si,0]+win, 'mean')
                for si in np.arange(-trials_eval, trials_eval)]
                for ci in range(self.n_clusters)]
            quant_1to0 = [[get_mean_sem_win(
                neu_trans_1to0[cluster_id==ci,:],
                self.alignment['neu_time'], 0, stim_seq_1to0[c_idx+si,0], stim_seq_1to0[c_idx+si,0]+win, 'mean')
                for si in np.arange(-trials_eval, trials_eval)]
                for ci in range(self.n_clusters)]
            # collect results.
            m_early_0 = [np.array([quant_1to0[ci][si+trials_eval][1] for si in range(trials_eval)]) for ci in range(self.n_clusters)]
            m_early_1 = [np.array([quant_0to1[ci][si+trials_eval][1] for si in range(trials_eval)]) for ci in range(self.n_clusters)]
            m_late_0 = [np.array([quant_0to1[ci][si][1] for si in range(trials_eval)]) for ci in range(self.n_clusters)]
            m_late_1 = [np.array([quant_1to0[ci][si][1] for si in range(trials_eval)]) for ci in range(self.n_clusters)]
            s_early_0 = [np.array([quant_1to0[ci][si+trials_eval][2] for si in range(trials_eval)]) for ci in range(self.n_clusters)]
            s_early_1 = [np.array([quant_0to1[ci][si+trials_eval][2] for si in range(trials_eval)]) for ci in range(self.n_clusters)]
            s_late_0 = [np.array([quant_0to1[ci][si][2] for si in range(trials_eval)]) for ci in range(self.n_clusters)]
            s_late_1 = [np.array([quant_1to0[ci][si][2] for si in range(trials_eval)]) for ci in range(self.n_clusters)]
            # define layouts.
            ax0 = ax.inset_axes([0.1, 0, 0.4, 0.95], transform=ax.transAxes)
            ax1 = ax.inset_axes([0.6, 0, 0.4, 0.95], transform=ax.transAxes)
            axs_01 = [ax0.inset_axes([0, ci/self.n_clusters, 0.3, 0.7/self.n_clusters], transform=ax0.transAxes)
                      for ci in range(self.n_clusters)]
            axs_02 = [ax0.inset_axes([0.35, ci/self.n_clusters, 0.3, 0.7/self.n_clusters], transform=ax0.transAxes)
                      for ci in range(self.n_clusters)]
            axs_03 = [ax0.inset_axes([0.7, ci/self.n_clusters, 0.3, 0.7/self.n_clusters], transform=ax0.transAxes)
                      for ci in range(self.n_clusters)]
            axs_11 = [ax1.inset_axes([0, ci/self.n_clusters, 0.3, 0.7/self.n_clusters], transform=ax1.transAxes)
                      for ci in range(self.n_clusters)]
            axs_12 = [ax1.inset_axes([0.35, ci/self.n_clusters, 0.3, 0.7/self.n_clusters], transform=ax1.transAxes)
                      for ci in range(self.n_clusters)]
            axs_13 = [ax1.inset_axes([0.7, ci/self.n_clusters, 0.3, 0.7/self.n_clusters], transform=ax1.transAxes)
                      for ci in range(self.n_clusters)]
            axs_01.reverse()
            axs_02.reverse()
            axs_03.reverse()
            axs_11.reverse()
            axs_12.reverse()
            axs_13.reverse()
            # plot results for each class.
            for ci in range(self.n_clusters):
                if np.sum(cluster_id==ci) > 0:
                    # find bounds.
                    upper = np.nanmax(np.concatenate([m_early_0[ci], m_early_1[ci], m_late_0[ci], m_late_1[ci]]))
                    lower = np.nanmin(np.concatenate([m_early_0[ci], m_early_1[ci], m_late_0[ci], m_late_1[ci]]))
                    # early short.
                    for ti in np.arange(trials_eval):
                        axs_12[ci].errorbar(
                            ti, m_early_0[ci][ti], s_early_0[ci][ti], None,
                            color=color1,
                            capsize=2, marker='o', linestyle='none',
                            markeredgecolor='white', markeredgewidth=0.1)
                        axs_12[ci].axvline(0, color='red', lw=1, linestyle='--')
                    # early long.
                    for ti in np.arange(trials_eval):
                        axs_02[ci].errorbar(
                            ti, m_early_1[ci][ti], s_early_1[ci][ti], None,
                            color=color2,
                            capsize=2, marker='o', linestyle='none',
                            markeredgecolor='white', markeredgewidth=0.1)
                        axs_02[ci].axvline(0, color='red', lw=1, linestyle='--')
                    # late short.
                    for ti in np.arange(trials_eval):
                        for axi in [axs_01[ci], axs_13[ci]]:
                            axi.errorbar(
                                ti-trials_eval, m_late_0[ci][ti], s_late_0[ci][ti], None,
                                color=color1 if ti < trials_eval else color2,
                                capsize=2, marker='o', linestyle='none',
                                markeredgecolor='white', markeredgewidth=0.1)
                    # late long.
                    for ti in np.arange(trials_eval):
                        for axi in [axs_03[ci], axs_11[ci]]:
                            axi.errorbar(
                                ti-trials_eval, m_late_1[ci][ti], s_late_1[ci][ti], None,
                                color=color2,
                                capsize=2, marker='o', linestyle='none',
                                markeredgecolor='white', markeredgewidth=0.1)
                    # adjust layouts.
                    axs_01[ci].yaxis.set_major_locator(mtick.MaxNLocator(nbins=2))
                    axs_01[ci].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
                    axs_11[ci].set_yticks([])
                    for axi in [axs_02, axs_12, axs_03, axs_13]:
                        axi[ci].set_yticks([])
                        axi[ci].spines['left'].set_visible(False)
                    for axi in [axs_01, axs_11, axs_02, axs_12, axs_03, axs_13]:
                        axi[ci].set_xticks([])
                        axi[ci].spines['right'].set_visible(False)
                        axi[ci].spines['top'].set_visible(False) 
                        axi[ci].set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
                    for axi in [axs_01, axs_11, axs_03, axs_13]:
                        axi[ci].set_xlim(-trials_eval-1, 0)
                    for axi in [axs_02, axs_12]:
                        axi[ci].set_xlim(-1, trials_eval)
                for axi in [axs_01, axs_11, axs_03, axs_13]:
                    axi[self.n_clusters-1].tick_params(axis='x', labelrotation=90)
                    axi[self.n_clusters-1].set_xticks(np.arange(-trials_eval-1, 0, 2)+1)
                    axi[self.n_clusters-1].set_xticklabels(np.arange(-trials_eval-1, 0, 2)+1)
                for axi in [axs_02, axs_12]:
                    axi[self.n_clusters-1].tick_params(axis='x', labelrotation=90)
                    axi[self.n_clusters-1].set_xticks(np.arange(0, trials_eval, 2))
                    axi[self.n_clusters-1].set_xticklabels(np.arange(0, trials_eval, 2))
                    axi[self.n_clusters-1].set_xlabel('early')
                for axi in [axs_01, axs_11]:
                    axi[self.n_clusters-1].set_xlabel('pre')
                for axi in [axs_03, axs_13]:
                    axi[self.n_clusters-1].set_xlabel('late')
            ax.set_ylabel('dF/F (z-scored)')
            ax0.set_title('S\u2192L adaptation')
            ax1.set_title('L\u2192S adaptation')
            hide_all_axis(ax)
            hide_all_axis(ax0)
            hide_all_axis(ax1)
        @show_resource_usage
        def plot_transition_trial(ax):
            xlim = [-500,2500]
            colors_0 = ['springgreen', 'red', 'dodgerblue', 'black']
            colors_1 = ['dodgerblue', 'red', 'springgreen', 'black']
            # collect data.
            [_, [neu_trans_0to1, _, stim_seq_0to1, _], _, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_idx=[l[1] for l in self.list_block_start],
                cate=cate, roi_id=None)
            [_, [neu_trans_1to0, _, stim_seq_1to0, _], _, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_idx=[l[0] for l in self.list_block_start],
                cate=cate, roi_id=None)
            l_idx_0, r_idx_0 = get_frame_idx_from_time(self.alignment['neu_time'], 0, stim_seq_1to0[c_idx,0], stim_seq_1to0[c_idx+1,0])
            l_idx_1, r_idx_1 = get_frame_idx_from_time(self.alignment['neu_time'], 0, stim_seq_0to1[c_idx,0], stim_seq_0to1[c_idx+1,0])
            len_0 = r_idx_0-l_idx_0
            len_1 = r_idx_1-l_idx_1
            neu_time_0 = [self.alignment['neu_time'][l_idx_1:r_idx_1],
                          self.alignment['neu_time'][l_idx_0:r_idx_0],
                          self.alignment['neu_time'][l_idx_0:r_idx_0],
                          self.alignment['neu_time'][l_idx_0:r_idx_0]]
            neu_time_1 = [self.alignment['neu_time'][l_idx_0:r_idx_0],
                          self.alignment['neu_time'][l_idx_1:r_idx_1],
                          self.alignment['neu_time'][l_idx_1:r_idx_1],
                          self.alignment['neu_time'][l_idx_1:r_idx_1]]
            idx_0 = [get_frame_idx_from_time(self.alignment['neu_time'], 0, stim_seq_1to0[c_idx-2,0], 0)[0],
                     get_frame_idx_from_time(self.alignment['neu_time'], 0, stim_seq_1to0[c_idx-1,0], 0)[0],
                     get_frame_idx_from_time(self.alignment['neu_time'], 0, stim_seq_1to0[c_idx,0],   0)[0],
                     get_frame_idx_from_time(self.alignment['neu_time'], 0, stim_seq_0to1[c_idx-2,0], 0)[0]]
            idx_1 = [get_frame_idx_from_time(self.alignment['neu_time'], 0, stim_seq_0to1[c_idx-2,0], 0)[0],
                     get_frame_idx_from_time(self.alignment['neu_time'], 0, stim_seq_0to1[c_idx-1,0], 0)[0],
                     get_frame_idx_from_time(self.alignment['neu_time'], 0, stim_seq_0to1[c_idx,0],   0)[0],
                     get_frame_idx_from_time(self.alignment['neu_time'], 0, stim_seq_1to0[c_idx-2,0], 0)[0]]
            # define layouts.
            ax.axis('off')
            axg = ax.inset_axes([0.5, 0, 0.5, 1], transform=ax.transAxes)
            ax = ax.inset_axes([0, 0, 0.5, 1], transform=ax.transAxes)
            ax0 = ax.inset_axes([0.1, 0, 0.4, 0.95], transform=ax.transAxes)
            ax1 = ax.inset_axes([0.6, 0, 0.4, 0.95], transform=ax.transAxes)
            axs0 = [ax0.inset_axes([0, 0.05+ci/self.n_clusters, 1, 0.8/self.n_clusters], transform=ax0.transAxes)
                    for ci in range(self.n_clusters)]
            axs1 = [ax1.inset_axes([0, 0.05+ci/self.n_clusters, 1, 0.8/self.n_clusters], transform=ax1.transAxes)
                    for ci in range(self.n_clusters)]
            axs0.reverse()
            axs1.reverse()
            # plot results for each class.
            for ci in range(self.n_clusters):
                if np.sum(cluster_id==ci) > 0:
                    neu_mean_0to1, neu_sem_0to1 = get_mean_sem(neu_trans_0to1[cluster_id==ci,:])
                    neu_mean_1to0, neu_sem_1to0 = get_mean_sem(neu_trans_1to0[cluster_id==ci,:])
                    # find bounds.
                    upper = np.nanmax([neu_mean_0to1, neu_mean_1to0])
                    lower = np.nanmin([neu_mean_0to1, neu_mean_1to0])
                    # get trials.
                    neu_mean_0 = [neu_mean_1to0[idx_0[0]:idx_0[0]+len_1],
                                  neu_mean_1to0[idx_0[1]:idx_0[1]+len_0],
                                  neu_mean_1to0[idx_0[2]:idx_0[2]+len_0],
                                  neu_mean_0to1[idx_0[3]:idx_0[3]+len_0]]
                    neu_mean_1 = [neu_mean_0to1[idx_1[0]:idx_1[0]+len_0],
                                  neu_mean_0to1[idx_1[1]:idx_1[1]+len_1],
                                  neu_mean_0to1[idx_1[2]:idx_1[2]+len_1],
                                  neu_mean_1to0[idx_1[3]:idx_1[3]+len_1]]
                    neu_sem_0 = [neu_sem_1to0[idx_0[0]:idx_0[0]+len_1],
                                 neu_sem_1to0[idx_0[1]:idx_0[1]+len_0],
                                 neu_sem_1to0[idx_0[2]:idx_0[2]+len_0],
                                 neu_sem_0to1[idx_0[3]:idx_0[3]+len_0]]
                    neu_sem_1 = [neu_sem_0to1[idx_1[0]:idx_1[0]+len_0],
                                 neu_sem_0to1[idx_1[1]:idx_1[1]+len_1],
                                 neu_sem_0to1[idx_1[2]:idx_1[2]+len_1],
                                 neu_sem_1to0[idx_1[3]:idx_1[3]+len_1]]
                    # plot stimulus.
                    axs0[ci].fill_between(
                        stim_seq_0to1[c_idx,:],
                        lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
                        color=color0, edgecolor='none', alpha=0.25, step='mid')
                    axs1[ci].fill_between(
                        stim_seq_1to0[c_idx,:],
                        lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
                        color=color0, edgecolor='none', alpha=0.25, step='mid')
                    # plot neural traces.
                    for ti in range(4):
                        self.plot_mean_sem(
                            axs0[ci], neu_time_0[ti],
                            neu_mean_0[ti], neu_sem_0[ti], colors_0[ti], None)
                        self.plot_mean_sem(
                            axs1[ci], neu_time_1[ti],
                            neu_mean_1[ti], neu_sem_1[ti], colors_1[ti], None)
                    # adjust layouts.
                    adjust_layout_neu(axs0[ci])
                    adjust_layout_neu(axs1[ci])
                    axs0[ci].set_xlim(xlim)
                    axs1[ci].set_xlim(xlim)
                    axs0[ci].set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
                    axs1[ci].set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
                    axs0[ci].set_ylabel(None)
                    axs1[ci].set_ylabel(None)
                    if ci != self.n_clusters-1:
                        axs0[ci].set_xticklabels([])
                        axs1[ci].set_xticklabels([])
                    axs1[ci].set_yticklabels([])
            ax.set_xlabel('time since stim (ms)')
            ax.set_ylabel('dF/F (z-scored)')
            ax0.set_title('L\u2192S')
            ax1.set_title('S\u2192L')
            hide_all_axis(axg)
            hide_all_axis(ax)
            hide_all_axis(ax0)
            hide_all_axis(ax1)
            lbl = ['trial -1 (L)', 'trial 0 (S)', 'trial 1 (S)', 'trial N (S)'] + ['trial -1 (S)', 'trial 0 (L)', 'trial 1 (L)', 'trial N (L)']
            add_legend(
                axg, colors_0+colors_1, lbl, n_trials, n_neurons, self.n_sess, 'upper right')
        @show_resource_usage
        def plot_trial_corr(ax):
            trials_eval = 40
            # collect data.
            [_, [neu_trans_0to1, _, _, _, _], _, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_param=[None, None, None, None, [0], [0]],
                trial_idx=list_trans_0to1,
                mean_sem=False,
                cate=cate, roi_id=None)
            [_, [neu_trans_1to0, _, _, _, _], _, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_param=[None, None, None, None, [0], [0]],
                trial_idx=list_trans_1to0,
                mean_sem=False,
                cate=cate, roi_id=None)
            [_, [_, _, stim_seq_0to1, _], _, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_idx=[l[1] for l in self.list_block_start],
                cate=cate, roi_id=None)
            [_, [_, _, stim_seq_1to0, _], _, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_idx=[l[0] for l in self.list_block_start],
                cate=cate, roi_id=None)
            c_idx = stim_seq_1to0.shape[0]//2
            l_idx_0, r_idx_0 = get_frame_idx_from_time(self.alignment['neu_time'], 0, stim_seq_1to0[c_idx,1], stim_seq_1to0[c_idx+1,0])
            l_idx_1, r_idx_1 = get_frame_idx_from_time(self.alignment['neu_time'], 0, stim_seq_0to1[c_idx,1], stim_seq_0to1[c_idx+1,0])
            # reshape into [n_transition*n_trials*n_neurons*n_times]
            neu_trans_0to1 = [neu.reshape([-1, 2*trials_around, neu.shape[1], neu.shape[2]]) for neu in neu_trans_0to1]
            neu_trans_1to0 = [neu.reshape([-1, 2*trials_around, neu.shape[1], neu.shape[2]]) for neu in neu_trans_1to0]
            n_transition = np.array([nt.shape[0] for nt in neu_trans_0to1])
            if np.max(n_transition) != np.min(n_transition):
                neu_trans_0to1 = [nt[:np.min(n_transition),:,:,:] for nt in neu_trans_0to1]
            n_transition = np.array([nt.shape[0] for nt in neu_trans_1to0])
            if np.max(n_transition) != np.min(n_transition):
                neu_trans_1to0 = [nt[:np.min(n_transition),:,:,:] for nt in neu_trans_1to0]
            # get less trials around transition.
            neu_trans_0to1 = [nt[:, trials_around:trials_around+trials_eval, :, :] for nt in neu_trans_0to1]
            neu_trans_1to0 = [nt[:, trials_around:trials_around+trials_eval, :, :] for nt in neu_trans_1to0]
            # get average across transition [n_trials*n_neurons*n_times]
            neu_trans_0to1 = np.concatenate(neu_trans_0to1, axis=2)
            neu_trans_1to0 = np.concatenate(neu_trans_1to0, axis=2)
            neu_0to1 = np.nanmean(neu_trans_0to1, axis=0)
            neu_1to0 = np.nanmean(neu_trans_1to0, axis=0)
            # get response within cluster.
            neu_0to1 = [get_mean_sem_cluster(nt, self.n_clusters, cluster_id)[0] for nt in neu_0to1]
            neu_1to0 = [get_mean_sem_cluster(nt, self.n_clusters, cluster_id)[0] for nt in neu_1to0]
            neu_0to1 = np.concatenate(np.expand_dims(neu_0to1, 0), axis=0)
            neu_1to0 = np.concatenate(np.expand_dims(neu_1to0, 0), axis=0)
            # get data within range.
            neu_seq_0 = neu_1to0[:,:,l_idx_0:r_idx_0]
            neu_seq_1 = neu_0to1[:,:,l_idx_1:r_idx_1]            
            # compute trial correlation.
            d_0 = [cosine_similarity(neu_seq_0[:,ci,:]) for ci in range(self.n_clusters)]
            d_1 = [cosine_similarity(neu_seq_1[:,ci,:]) for ci in range(self.n_clusters)]
            # define layouts.
            ax0 = ax.inset_axes([0.1, 0, 0.4, 0.95], transform=ax.transAxes)
            ax1 = ax.inset_axes([0.6, 0, 0.4, 0.95], transform=ax.transAxes)
            axs_hm_0 = [ax0.inset_axes([0, 0.05+ci/self.n_clusters, 0.8, 0.8/self.n_clusters], transform=ax0.transAxes)
                        for ci in range(self.n_clusters)]
            axs_hm_1 = [ax1.inset_axes([0, 0.05+ci/self.n_clusters, 0.8, 0.8/self.n_clusters], transform=ax1.transAxes)
                        for ci in range(self.n_clusters)]
            axs_cb_0 = [ax0.inset_axes([0.8, 0.05+ci/self.n_clusters, 0.2, 0.8/self.n_clusters], transform=ax0.transAxes)
                        for ci in range(self.n_clusters)]
            axs_cb_1 = [ax1.inset_axes([0.8, 0.05+ci/self.n_clusters, 0.2, 0.8/self.n_clusters], transform=ax1.transAxes)
                        for ci in range(self.n_clusters)]
            axs_hm_0.reverse()
            axs_hm_1.reverse()
            axs_cb_0.reverse()
            axs_cb_1.reverse()
            # plot results for each class.
            for ci in range(self.n_clusters):
                if np.sum(cluster_id==ci) > 0:
                    self.plot_dis_mat(axs_hm_0[ci], axs_cb_0[ci], d_0[ci])
                    self.plot_dis_mat(axs_hm_1[ci], axs_cb_1[ci], d_1[ci])
                # adjust layouts.
                axs_hm_0[ci].set_xticks([])
                axs_hm_0[ci].set_yticks([0, int(trials_eval/2)])
                axs_hm_1[ci].set_xticks([])
                axs_hm_1[ci].set_yticks([])
                hide_all_axis(axs_cb_0[ci])
                hide_all_axis(axs_cb_1[ci])
            axs_hm_0[self.n_clusters-1].set_xticks([0, int(trials_eval/2)])
            axs_hm_1[self.n_clusters-1].set_xticks([0, int(trials_eval/2)])
            hide_all_axis(ax)
            hide_all_axis(ax0)
            hide_all_axis(ax1)
        @show_resource_usage
        def plot_win_mag_scatter_epoch(ax):
            win = 250
            average_axis = 0
            # collect data.
            [_, [neu_trans_0to1, stim_seq_0to1, _, _, _], _, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_idx=[l[1] for l in self.list_block_start],
                mean_sem=False, cate=cate, roi_id=None)
            [_, [neu_trans_1to0, stim_seq_1to0, _, _, _], _, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_idx=[l[0] for l in self.list_block_start],
                mean_sem=False, cate=cate, roi_id=None)
            stim_seq_0to1 = np.nanmean(np.concatenate(stim_seq_0to1, axis=0),axis=0)
            stim_seq_1to0 = np.nanmean(np.concatenate(stim_seq_1to0, axis=0),axis=0)
            # compute response within window.
            win_eval_short_first = [[stim_seq_1to0[c_idx-1,1], stim_seq_1to0[c_idx,0]],   [stim_seq_1to0[c_idx,0],   stim_seq_1to0[c_idx,0]+win]]
            win_eval_short_last  = [[stim_seq_0to1[c_idx-2,1], stim_seq_0to1[c_idx-1,0]], [stim_seq_0to1[c_idx-1,0], stim_seq_0to1[c_idx-1,0]+win]]
            win_eval_long_first  = [[stim_seq_0to1[c_idx-1,1], stim_seq_0to1[c_idx,0]],   [stim_seq_0to1[c_idx,0],   stim_seq_0to1[c_idx,0]+win]]
            win_eval_long_last   = [[stim_seq_1to0[c_idx-2,1], stim_seq_1to0[c_idx-1,0]], [stim_seq_1to0[c_idx-1,0], stim_seq_1to0[c_idx-1,0]+win]]
            # define layouts.
            ax.axis('off')
            ax = ax.inset_axes([0, 0, 1, 0.85], transform=ax.transAxes)
            ax0 = ax.inset_axes([0.1, 0, 0.4, 1], transform=ax.transAxes)
            ax1 = ax.inset_axes([0.6, 0, 0.4, 1], transform=ax.transAxes)
            axs0 = [ax0.inset_axes([0, 0.1+ci/self.n_clusters, 1, 0.8/self.n_clusters], transform=ax0.transAxes)
                    for ci in range(self.n_clusters)]
            axs1 = [ax1.inset_axes([0, 0.1+ci/self.n_clusters, 1, 0.8/self.n_clusters], transform=ax1.transAxes)
                    for ci in range(self.n_clusters)]
            axs0.reverse()
            axs1.reverse()
            # plot results.
            self.plot_cluster_win_mag_scatter(
                axs0, day_cluster_id, neu_trans_1to0, neu_trans_0to1, self.alignment['neu_time'],
                win_eval_short_first, win_eval_short_last,
                color1, 0, average_axis, True)
            self.plot_cluster_win_mag_scatter(
                axs1, day_cluster_id, neu_trans_1to0, neu_trans_1to0, self.alignment['neu_time'],
                win_eval_long_first, win_eval_long_last,
                color2, 0, average_axis, True)
            # adjust layouts.
            ax.set_xlabel('first')
            ax.set_ylabel('last')
            ax0.set_title('short')
            ax1.set_title('long')
            hide_all_axis(ax)
            hide_all_axis(ax0)
            hide_all_axis(ax1)
        @show_resource_usage
        def plot_legend(ax):
            lbl = ['short', 'long']
            cs = [color1, color2]
            add_legend(ax, cs, lbl, n_trials, n_neurons, self.n_sess, 'upper right')
            ax.axis('off')
        # plot all.
        try: plot_tansition_trial_heatmap(axs[0], 'none')
        except: traceback.print_exc()()
        try: plot_tansition_trial_heatmap(axs[1], 'minmax')
        except: traceback.print_exc()()
        try: plot_tansition_trial_heatmap(axs[2], 'share')
        except: traceback.print_exc()()
        try: plot_tansition(axs[3])
        except: traceback.print_exc()()
        try: plot_trial_quant(axs[4])
        except: traceback.print_exc()()
        try: plot_transition_trial(axs[5])
        except: traceback.print_exc()()
        try: plot_trial_corr(axs[6])
        except: traceback.print_exc()()
        try: plot_win_mag_scatter_epoch(axs[7])
        except: traceback.print_exc()()
        try: plot_legend(axs[8])
        except: traceback.print_exc()()
    
    def plot_sorted_heatmaps_all(self, axs, norm_mode, cate):
        xlim = [-2000,3000]
        l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, xlim[0], xlim[1])
        # collect data.
        neu_x = []
        stim_x = []
        neu_time = self.alignment['neu_time'][l_idx:r_idx]
        # short.
        _, [neu_seq, _, stim_seq, _], _, _ = get_neu_trial(
            self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
            trial_param=[[2,3,4,5], [0], None, None, [0], [0]],
            cate=cate, roi_id=None)
        neu_seq = neu_seq[:,l_idx:r_idx]
        neu_x.append(neu_seq)
        stim_x.append(stim_seq)
        # long.
        _, [neu_seq, _, stim_seq, _], _, _ = get_neu_trial(
            self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
            trial_param=[[2,3,4,5], [1], None, None, [0], [0]],
            cate=cate, roi_id=None)
        neu_seq = neu_seq[:,l_idx:r_idx]
        neu_x.append(neu_seq)
        stim_x.append(stim_seq)
        # define layouts.
        axs_hm = [axs[ai].inset_axes([0, 0, 0.8, 1], transform=axs[ai].transAxes) for ai in range(2)]
        axs_cb = [axs[ai].inset_axes([0.9, 0, 0.1, 1], transform=axs[ai].transAxes) for ai in range(2)]
        # plot heatmaps.
        for ai in range(2):
            self.plot_heatmap_neuron(
                axs_hm[ai], axs_cb[ai], neu_x[ai], neu_time, neu_x[0],
                norm_mode=norm_mode,
                neu_seq_share=neu_x)
            axs_hm[ai].set_xlabel('time since stim (ms)')
            hide_all_axis(axs[ai])
        # add stimulus line.
        c_idx = stim_x[0].shape[0]//2
        for ai in range(2):
            xlines = [self.alignment['neu_time'][np.searchsorted(self.alignment['neu_time'], t)]
                      for t in [stim_x[ai][c_idx+i,0]
                                for i in [-2,-1,0,1,2]]]
            for xl in xlines:
                if xl>neu_time[0] and xl<neu_time[-1]:
                    axs_hm[ai].axvline(xl, color='black', lw=1, linestyle='--')
    def plot_latent_all(self, axs, cate):
        color0 = 'dimgrey'
        base_color1 = ['dodgerblue', 'darkviolet']
        base_color2 = ['greenyellow', 'mediumseagreen']
        cluster_id, neu_labels = get_cluster_cate(self.cluster_id, self.list_labels, self.list_significance, cate)
        # collect data.
        [_, [_, _, stim_seq, _], _,
         [n_trials, n_neurons]] = get_neu_trial(
            self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
            trial_param=[None, None, None, None, [0], [0]],
            cate=cate, roi_id=None)
        c_idx = stim_seq.shape[0]//2
        def plot_standard(axs):
            xlim = [-4500,5000]
            # collect data.
            [_, [neu_seq_0, _, stim_seq_0, _], _, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_param=[[2,3,4,5], [0], None, None, [0], [0]],
                cate=cate, roi_id=None)
            [_, [neu_seq_1, _, stim_seq_1, _], _, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_param=[[2,3,4,5], [1], None, None, [0], [0]],
                cate=cate, roi_id=None)
            c_stim = [color0] * stim_seq_0.shape[-2]
            neu_x = [neu_seq_0, neu_seq_1]
            # fit model.
            model = PCA(n_components=self.d_latent)
            model.fit(np.concatenate(neu_x,axis=1).reshape(-1,2*len(self.alignment['neu_time'])).T)
            neu_z = [model.transform(neu_x[s].reshape(neu_x[s].shape[0],-1).T).T.reshape(self.d_latent, -1) for s in [0,1]]
            # get data within range.
            l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, stim_seq_0[c_idx-2,0], stim_seq_0[c_idx+2,1])
            neu_time_0 = self.alignment['neu_time'][l_idx:r_idx]
            neu_x_0 = neu_x[0][:,l_idx:r_idx]
            neu_z_0 = neu_z[0][:,l_idx:r_idx]
            l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, stim_seq_1[c_idx-2,0], stim_seq_1[c_idx+2,1])
            neu_time_1 = self.alignment['neu_time'][l_idx:r_idx]
            neu_x_1 = neu_x[1][:,l_idx:r_idx]
            neu_z_1 = neu_z[1][:,l_idx:r_idx]
            cmap_0, c_neu_0 = get_cmap_color(len(neu_time_0), base_color=['lemonchiffon']+base_color1+['black'], return_cmap=True)
            cmap_1, c_neu_1 = get_cmap_color(len(neu_time_1), base_color=['lemonchiffon']+base_color2+['black'], return_cmap=True)
            # random rotate dynamics.
            for ai in range(len(axs)):
                # get random matrix.
                rm = get_random_rotate_mat_3d()
                # define layouts.
                axs[ai].axis('off')
                ax0 = axs[ai].inset_axes([0,   0,   0.6, 0.6], transform=axs[ai].transAxes, projection='3d')
                ax10 = axs[ai].inset_axes([0,  0.6, 0.6, 0.08], transform=axs[ai].transAxes)
                ax11 = axs[ai].inset_axes([0,  0.8, 0.6, 0.08], transform=axs[ai].transAxes)
                ax20 = axs[ai].inset_axes([0.6, 0,   0.1, 0.4], transform=axs[ai].transAxes)
                ax21 = axs[ai].inset_axes([0.8, 0,   0.1, 0.4], transform=axs[ai].transAxes)
                # plot mean trace.
                neu_mean_0, neu_sem_0 = get_mean_sem(neu_x_0)
                neu_mean_1, neu_sem_1 = get_mean_sem(neu_x_1)
                # find bounds.
                upper = np.nanmax(np.concatenate([neu_mean_0, neu_mean_1])) + np.nanmax(np.concatenate([neu_sem_0, neu_sem_1]))
                lower = np.nanmin(np.concatenate([neu_mean_0, neu_mean_1])) - np.nanmax(np.concatenate([neu_sem_0, neu_sem_1]))
                # plot stimulus.
                for si in range(stim_seq.shape[0]):
                    ax10.fill_between(
                        stim_seq_0[si,:],
                        lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
                        color=color0, edgecolor='none', alpha=0.25, step='mid')
                    ax11.fill_between(
                        stim_seq_1[si,:],
                        lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
                        color=color0, edgecolor='none', alpha=0.25, step='mid')
                # plot neural traces.
                for t in range(len(neu_time_0)-1):
                    ax10.plot(neu_time_0[t:t+2], neu_mean_0[t:t+2], color=c_neu_0[t])
                for t in range(len(neu_time_1)-1):
                    ax11.plot(neu_time_1[t:t+2], neu_mean_1[t:t+2], color=c_neu_1[t])
                ax10.scatter(neu_time_0[0], neu_mean_0[0], color='black', marker='x', lw=1)
                ax11.scatter(neu_time_1[0], neu_mean_1[0], color='black', marker='x', lw=1)
                ax10.scatter(neu_time_0[-1], neu_mean_0[-1], color='black', marker='o', lw=1)
                ax11.scatter(neu_time_1[-1], neu_mean_1[-1], color='black', marker='o', lw=1)
                # plot 3d dynamics.
                self.plot_3d_latent_dynamics(ax0, np.matmul(rm,neu_z_0), stim_seq, neu_time_0, c_stim, cmap=cmap_0)
                self.plot_3d_latent_dynamics(ax0, np.matmul(rm,neu_z_1), stim_seq, neu_time_1, c_stim, cmap=cmap_1)
                # adjust layouts.
                for axi in [ax10, ax11]:
                    hide_all_axis(axi)
                    axi.set_xlim(xlim)
                    axi.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
                adjust_layout_3d_latent(ax0)
                # add colorbar.
                yticklabels = [int(neu_time_0[0]+0.2*(neu_time_0[-1]-neu_time_0[0])), int(neu_time_0[-1]-0.2*(neu_time_0[-1]-neu_time_0[0]))]
                add_heatmap_colorbar(ax20, cmap_0, None, 'time since stim (S)', yticklabels)
                yticklabels = [int(neu_time_1[0]+0.2*(neu_time_1[-1]-neu_time_1[0])), int(neu_time_1[-1]-0.2*(neu_time_1[-1]-neu_time_1[0]))]
                add_heatmap_colorbar(ax21, cmap_1, None, 'time since stim (L)', yticklabels)
        # plot all.
        try: plot_standard(axs[0])
        except: traceback.print_exc()()
    
    def plot_decode_all(self, axs, standard, cate):
        r2_thres = 0.3
        color0 = 'dimgrey'
        color1 = 'mediumseagreen'
        color2 = 'coral'
        cluster_id, neu_labels = get_cluster_cate(self.cluster_id, self.list_labels, self.list_significance, cate)
        # collect data.
        [_, [neu_seq, _, stim_seq, _], _, _] = get_neu_trial(
            self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
            trial_param=[[2,3,4,5], [standard], None, None, [0], [0]],
            cate=cate, roi_id=None)
        c_idx = stim_seq.shape[0]//2
        # get data within range.
        l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, stim_seq[c_idx-1,1], 0)
        neu_seq_l = neu_seq[:,l_idx:r_idx]
        neu_time_l = self.alignment['neu_time'][l_idx:r_idx]
        l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, 0, stim_seq[c_idx+1,0])
        neu_seq_r = neu_seq[:,l_idx:r_idx]
        neu_time_r = self.alignment['neu_time'][l_idx:r_idx]
        # fit response model.
        [trf_param_up, pred_up, r2_all_up,
         trf_param_dn, pred_dn, r2_all_dn] = fit_trf_model(
             neu_seq_l, neu_time_l, neu_seq_r, neu_time_r)
        idx_up = (r2_all_up>r2_thres)
        idx_dn = (r2_all_dn>r2_thres)
        @show_resource_usage
        def plot_standard_time_decode(axs):
            bin_times = 50
            tlim = [0, 2500]
            # collect data.
            [_, [neu_seq, stim_seq, _], _, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_param=[[2,3,4,5], [standard], None, None, [0], [0]],
                sub_sampling=True,
                cate=cate, roi_id=None)
            c_idx = stim_seq.shape[0]//2
            # get data within range.
            bin_l_idx, bin_r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, 0, bin_times)
            bin_len = bin_r_idx - bin_l_idx
            l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, tlim[0], tlim[1])
            t_range = self.alignment['neu_time'][l_idx:r_idx]
            t_range = t_range[:(len(t_range)//bin_len)*bin_len]
            t_range = np.nanmin(t_range.reshape(-1, bin_len), axis=1)
            l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, 0, stim_seq[c_idx+1, 0])
            neu_time = self.alignment['neu_time'][l_idx:r_idx]
            neu_x = neu_seq[:,:,l_idx:r_idx]
            # get decoding matrix.
            t_ci = []
            acc_mat_ci = []
            for ci in range(self.n_clusters):
                if np.sum(cluster_id==ci) > 0:
                    t, acc_mat, _ = decoding_time_confusion(neu_x[:,cluster_id==ci,:], neu_time, bin_times)
                    acc_mat_ci.append(acc_mat)
                    t_ci.append(t)
                else:
                    acc_mat_ci.append(np.array([np.nan]))
                    t_ci.append(np.array([np.nan]))
            neu_time = t_ci[np.where(np.array([np.mean(t) for t in t_ci]))[0][0]]
            # get decoding accuracy for adjacent time bin.
            acc_adj = np.full([self.n_clusters, len(t_range)], np.nan)
            for ci in range(self.n_clusters):
                if np.sum(cluster_id==ci) > 0:
                    acc_adj[ci,1:len(neu_time)] = np.diag(acc_mat_ci[ci], k=1)
            # define layouts.
            axs[0].axis('off')
            axs[1].axis('off')
            axs[2].axis('off')
            ax1 = axs[0].inset_axes([0, 0, 1, 0.9], transform=axs[0].transAxes)
            ax2 = axs[1].inset_axes([0, 0, 0.5, 0.9], transform=axs[1].transAxes)
            ax_lns = axs[2].inset_axes([0, 0, 0.2, 0.5], transform=axs[2].transAxes)
            ax_lg = axs[2].inset_axes([0, 0.5, 0.5, 0.5], transform=axs[2].transAxes)
            axs_hm = [ax1.inset_axes([0.4, ci/self.n_clusters, 0.6, 0.8/self.n_clusters], transform=ax1.transAxes)
                      for ci in range(self.n_clusters)]
            axs_ln = [ax2.inset_axes([0.3, ci/self.n_clusters, 0.7, 0.8/self.n_clusters], transform=ax2.transAxes)
                      for ci in range(self.n_clusters)]
            axs_hm.reverse()
            axs_ln.reverse()
            # plot results for each class.
            for ci in range(self.n_clusters):
                if np.sum(cluster_id==ci) > 0:
                    # populate decoding matrix.
                    t_range_mat = np.full([len(t_range), len(t_range)], np.nan)
                    t_range_mat[:len(neu_time),:len(neu_time)] = acc_mat_ci[ci]
                    # plot decoding matrix.
                    self.plot_time_decode_confusion_matrix(axs_hm[ci], t_range_mat, t_range)
                    # plot adjacent decoding accuracy.
                    axs_ln[ci].plot(t_range, acc_adj[ci,:], color=color0)
                    # plot adjacent decoding accuracy.
                    c = color1 if ci<self.n_up else color2
                    ax_lns.plot(t_range, acc_adj[ci,:], color=c, alpha=0.5)
            # adjust layouts.
            for ci in range(self.n_clusters):
                axs_hm[ci].xaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f'))
                axs_hm[ci].xaxis.set_major_locator(mtick.MaxNLocator(nbins=2))
                axs_hm[ci].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f'))
                axs_hm[ci].yaxis.set_major_locator(mtick.MaxNLocator(nbins=2))
                axs_hm[ci].set_xlim(tlim)
                axs_hm[ci].set_ylim(tlim)
                axs_ln[ci].spines['right'].set_visible(False)
                axs_ln[ci].spines['top'].set_visible(False)
                axs_ln[ci].xaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f'))
                axs_ln[ci].xaxis.set_major_locator(mtick.MaxNLocator(nbins=2))
                axs_ln[ci].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
                axs_ln[ci].yaxis.set_major_locator(mtick.MaxNLocator(nbins=2))
                axs_ln[ci].set_xlim(tlim)
                axs_ln[ci].set_ylim([np.nanmin(acc_adj),np.nanmax(acc_adj)])
                if ci != self.n_clusters-1:
                    axs_hm[ci].set_xticklabels([])
                    axs_ln[ci].set_xticklabels([])
            ax_lns.spines['right'].set_visible(False)
            ax_lns.spines['top'].set_visible(False)
            ax_lns.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f'))
            ax_lns.xaxis.set_major_locator(mtick.MaxNLocator(nbins=2))
            ax_lns.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
            ax_lns.yaxis.set_major_locator(mtick.MaxNLocator(nbins=2))
            ax_lns.set_xlim(tlim)
            ax_lns.set_ylim([np.nanmin(acc_adj),np.nanmax(acc_adj)])
            axs_hm[self.n_clusters-1].tick_params(axis='x', labelrotation=90)
            axs_ln[self.n_clusters-1].tick_params(axis='x', labelrotation=90)
            axs_ln[self.n_clusters-1].set_xlabel('time since stim (ms)')
            ax1.set_ylabel('time since stim (ms)')
            ax2.set_ylabel('decoding accuracy for adjacent time bins')
            hide_all_axis(ax1)
            hide_all_axis(ax2)
            hide_all_axis(ax_lg)
        @show_resource_usage
        def plot_standard_time_decode_all(ax, target):
            bin_times = 50
            tlim = [0, 2500]
            # collect data.
            [_, [neu_seq, stim_seq, _], _, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_param=[[2,3,4,5], [standard], None, None, [0], [0]],
                sub_sampling=True,
                cate=cate, roi_id=None)
            c_idx = stim_seq.shape[0]//2
            # get data within range.
            bin_l_idx, bin_r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, 0, bin_times)
            bin_len = bin_r_idx - bin_l_idx
            l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, tlim[0], tlim[1])
            t_range = self.alignment['neu_time'][l_idx:r_idx]
            t_range = t_range[:(len(t_range)//bin_len)*bin_len]
            t_range = np.nanmin(t_range.reshape(-1, bin_len), axis=1)
            l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, 0, stim_seq[c_idx+1, 0])
            neu_time = self.alignment['neu_time'][l_idx:r_idx]
            if target == 'all':
                neu_x = neu_seq[:,np.logical_or(idx_up, idx_dn),l_idx:r_idx]
            if target == 'up':
                neu_x = neu_seq[:,idx_up,l_idx:r_idx]
            if target == 'down':
                neu_x = neu_seq[:,idx_dn,l_idx:r_idx]
            # get decoding matrix.
            neu_time, acc_mat, _ = decoding_time_confusion(neu_x, neu_time, bin_times)
            # define layouts.
            ax.axis('off')
            ax_ln = ax.inset_axes([0, 0.9, 0.8, 0.1], transform=ax.transAxes)
            ax_hm = ax.inset_axes([0, 0, 0.8, 0.8], transform=ax.transAxes)
            ax_cb = ax.inset_axes([0.9, 0, 0.1, 0.8], transform=ax.transAxes)
            # plot stimulus.
            for si in [0,1]:
                ax_ln.fill_between(stim_seq[c_idx+si,:], 0, 1, color=color0, edgecolor='none', alpha=0.25, step='mid')
            ax_ln.set_yticks([])
            ax_ln.set_xlim([t_range[0],t_range[-1]])
            ax_ln.set_ylim([0,1])
            ax_ln.spines['left'].set_visible(False)
            ax_ln.spines['right'].set_visible(False)
            ax_ln.spines['top'].set_visible(False)
            # populate decoding matrix.
            t_range_mat = np.full([len(t_range), len(t_range)], np.nan)
            t_range_mat[:len(neu_time),:len(neu_time)] = acc_mat
            # plot decoding matrix.
            self.plot_time_decode_confusion_matrix(ax_hm, t_range_mat, t_range, ax_cb)
            # adjust layouts.
            ax_hm.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f'))
            ax_hm.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f'))
            ax_hm.xaxis.set_major_locator(mtick.MaxNLocator(nbins=3))
            ax_hm.yaxis.set_major_locator(mtick.MaxNLocator(nbins=3))
            ax_hm.set_ylabel('time since stim (ms)')
            hide_all_axis(ax)
            hide_all_axis(ax_cb)
        @show_resource_usage
        def plot_standard_ramp_params(ax, param_idx):
            param_lbl = [
                r'$b$ (baseline)',
                r'$a$ (amplitude)',
                r'$m$ (latency)',
                r'$r$ (1/locality)',
                r'$s$ (timescale)']
            # get parameters.
            q1 = trf_param_up[r2_all_up>r2_thres,param_idx].copy()
            q2 = trf_param_dn[r2_all_dn>r2_thres,param_idx].copy()
            # define layouts.
            ax.axis('off')
            ax1 = ax.inset_axes([0, 0, 1, 1], transform=ax.transAxes)
            ax2 = ax.inset_axes([0.7, 0.4, 0.3, 0.3], transform=ax.transAxes)
            # plot distribution.
            d1 = self.plot_dist(ax1, q1, color1, False)
            d2 = self.plot_dist(ax1, q2, color2, False)
            # find bounds.
            xu = np.nanmax(np.concatenate([q1, q2]))
            xl = np.nanmin(np.concatenate([q1, q2]))
            yu = np.nanmax(np.concatenate([d1, d2]))
            xlim = [xl - 0.1*(xu-xl), xu - 0.3*(xu-xl)]
            # adjust layouts.
            ax1.tick_params(axis='x')
            ax1.tick_params(axis='y', tick1On=False)
            ax1.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
            ax1.xaxis.set_major_locator(mtick.MaxNLocator(nbins=4))
            ax1.spines['left'].set_visible(False)
            ax1.spines['right'].set_visible(False)
            ax1.spines['top'].set_visible(False) 
            ax1.set_yticks([])
            ax1.set_xlim(xlim)
            ax1.set_ylim([0, yu*1.1])
            ax1.set_xlabel(param_lbl[param_idx])
            add_legend(ax, [color1, color2], ['ramp-up', 'ramp-down'], None, None, None, 'upper right')
            # plot distribution.
            d1 = self.plot_dist(ax2, q1, color1, True)
            d2 = self.plot_dist(ax2, q2, color2, True)
            # find bounds.
            xu = np.nanmax(np.concatenate([q1, q2]))
            xl = np.nanmin(np.concatenate([q1, q2]))
            yu = np.nanmax(np.concatenate([d1, d2]))
            # adjust layouts.
            ax2.tick_params(tick1On=False)
            ax2.spines['right'].set_visible(False)
            ax2.spines['top'].set_visible(False) 
            ax2.set_xticks([])
            ax2.set_yticks([])
            ax2.set_xlim([xl, xu])
            ax2.set_ylim([0, yu])
        @show_resource_usage
        def plot_standard_ramp_params_all(ax, param_idx):
            param_lbl = [
                r'$b$ (baseline)',
                r'$a$ (amplitude)',
                r'$m$ (latency)',
                r'$r$ (1/locality)',
                r'$s$ (timescale)']
            # get parameters.
            q_all = []
            for ci in range(self.n_clusters):
                if np.sum(cluster_id==ci) > 0:
                    ru = np.nanmean(r2_all_up[cluster_id==ci])
                    rd = np.nanmean(r2_all_dn[cluster_id==ci])
                    if ru > rd:
                        q = trf_param_up[(cluster_id==ci)*idx_up,param_idx].copy()
                    else:
                        q = trf_param_dn[(cluster_id==ci)*idx_dn,param_idx].copy()
                    q_all.append(q)
                else:
                    q_all.append(np.array([np.nan]))
            # define layouts.
            ax.axis('off')
            ax1 = ax.inset_axes([0, 0, 1, 1], transform=ax.transAxes)
            ax2 = ax.inset_axes([0.6, 0.6, 0.4, 0.4], transform=ax.transAxes)
            # plot results for each class.
            for ci in range(self.n_clusters):
                if np.sum(cluster_id==ci) > 0:
                    # plot distribution.
                    c = color1 if ci<self.n_up else color2
                    self.plot_dist(ax1, q_all[ci], c, False, alpha=0.5)
                    self.plot_dist(ax2, q_all[ci], c, True, alpha=0.5)
            # find bounds.
            xu = np.nanmax(np.concatenate(q_all))
            xl = np.nanmin(np.concatenate(q_all))
            xlim = [xl, xu - 0.3*(xu-xl)]
            # adjust layouts.
            ax1.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
            ax1.xaxis.set_major_locator(mtick.MaxNLocator(nbins=2))
            ax1.spines['right'].set_visible(False)
            ax1.spines['top'].set_visible(False) 
            ax1.set_xlim(xlim)
            ax1.set_xlabel(param_lbl[param_idx])
            ax2.tick_params(tick1On=False)
            ax2.spines['right'].set_visible(False)
            ax2.spines['top'].set_visible(False) 
        def plot_standard_time_regress_drop_neu_all(ax):
            frac_step = 0.1
            max_frac = 0.5
            bin_times = 50
            tlim = [0, 2500]
            # collect data.
            [_, [neu_seq, stim_seq, _], _, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_param=[[2,3,4,5], [standard], None, None, [0], [0]],
                sub_sampling=True,
                cate=cate, roi_id=None)
            c_idx = stim_seq.shape[0]//2
            # get data within range.
            bin_l_idx, bin_r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, 0, bin_times)
            bin_len = bin_r_idx - bin_l_idx
            l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, tlim[0], tlim[1])
            t_range = self.alignment['neu_time'][l_idx:r_idx]
            t_range = t_range[:(len(t_range)//bin_len)*bin_len]
            t_range = np.nanmin(t_range.reshape(-1, bin_len), axis=1)
            l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, 0, stim_seq[c_idx+1, 0])
            neu_time = self.alignment['neu_time'][l_idx:r_idx]
            # select neurons.
            neu_x  = neu_seq[:,:,l_idx:r_idx]
            neu_up = neu_seq[:,idx_up,l_idx:r_idx]
            neu_dn = neu_seq[:,idx_dn,l_idx:r_idx]
            # fit model.
            fracs = np.arange(0,max_frac,frac_step)+frac_step
            r2_all = regression_time_frac(neu_x,  neu_time, bin_times, fracs)
            r2_up  = regression_time_frac(neu_up, neu_time, bin_times, fracs)
            r2_dn  = regression_time_frac(neu_dn, neu_time, bin_times, fracs)
            # plot results.
            m_all, s_all = get_mean_sem(r2_all)
            m_up,  s_up  = get_mean_sem(r2_up)
            m_dn,  s_dn  = get_mean_sem(r2_dn)
            self.plot_mean_sem(ax, fracs, m_all, s_all, color0, None)
            self.plot_mean_sem(ax, fracs, m_up,  s_up,  color1, None)
            self.plot_mean_sem(ax, fracs, m_dn,  s_dn,  color2, None)
            # adjust layouts.
            adjust_layout_neu(ax)
            ax.set_xlim([0,max_frac])
            ax.set_ylim([0,1])
            ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
            ax.xaxis.set_major_locator(mtick.MaxNLocator(nbins=3))
            ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
            ax.yaxis.set_major_locator(mtick.MaxNLocator(nbins=2))
            ax.set_xlabel('fraction of neurons')
            ax.set_ylabel('decoding accuracy')
            add_legend(ax, [color0, color1, color2], ['all', 'ramp-up', 'ramp-down'], None, None, None, 'upper left')
        @show_resource_usage
        def plot_example_ramp_fit(ax):
            n_example = 5
            p_up = pred_up[np.argsort(np.nan_to_num(r2_all_up, nan=-np.inf))[-n_example:][::-1]]
            p_dn = pred_dn[np.argsort(np.nan_to_num(r2_all_dn, nan=-np.inf))[-n_example:][::-1]]
            # define layouts.
            ax1 = ax.inset_axes([0.2, 0.1, 0.3, 0.5], transform=ax.transAxes)
            ax2 = ax.inset_axes([0.6, 0.1, 0.3, 0.5], transform=ax.transAxes)
            # plot results.
            for ni in range(n_example):
                ax1.plot(neu_time_l, p_up[ni,:], color=color1, alpha=0.3)
                ax2.plot(neu_time_r, p_dn[ni,:], color=color2, alpha=0.3)
            # adjust layouts.
            for axi in [ax1, ax2]:
                axi.spines['top'].set_visible(False)
                axi.spines['right'].set_visible(False)
                axi.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f'))
                axi.xaxis.set_major_locator(mtick.MaxNLocator(nbins=2))
            ax1.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
            ax1.yaxis.set_major_locator(mtick.MaxNLocator(nbins=2))
            ax2.spines['left'].set_visible(False)
            ax2.set_yticks([])
            ax1.set_ylabel('model value')
            ax.set_xlabel('time since stim (ms)')
            hide_all_axis(ax)
        @show_resource_usage
        def plot_ramp_heatmap(ax, target):
            xlim = [-2000,3000]
            l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, xlim[0], xlim[1])
            # collect data.
            [_, [neu_seq, stim_seq, _], _, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_param=[[2,3,4,5], [standard], None, None, [0], [0]],
                sub_sampling=True,
                cate=cate, roi_id=None)
            c_idx = stim_seq.shape[0]//2
            neu_time = self.alignment['neu_time'][l_idx:r_idx]
            if target == 'up':
                neu_x = np.nanmean(neu_seq[:,idx_up,l_idx:r_idx], axis=0)
            if target == 'down':
                neu_x = np.nanmean(neu_seq[:,idx_dn,l_idx:r_idx], axis=0)
            # define layouts.
            axs_hm = ax.inset_axes([0, 0, 0.8, 1], transform=ax.transAxes)
            axs_cb = ax.inset_axes([0.9, 0, 0.1, 1], transform=ax.transAxes)
            # plot heatmaps.
            self.plot_heatmap_neuron(
                axs_hm, axs_cb, neu_x, neu_time, neu_x, norm_mode='minmax')
            axs_hm.set_xlabel('time since stim (ms)')
            hide_all_axis(ax)
            # add stimulus line.
            for ai in range(2):
                xlines = [self.alignment['neu_time'][np.searchsorted(self.alignment['neu_time'], t)]
                          for t in [stim_seq[c_idx+i,0]
                                    for i in [-2,-1,0,1,2]]]
                for xl in xlines:
                    if xl>neu_time[0] and xl<neu_time[-1]:
                        axs_hm.axvline(xl, color='black', lw=1, linestyle='--')
        # plot all.
        try: plot_standard_time_decode([axs[0], axs[1], axs[2]])
        except: traceback.print_exc()()
        try: plot_standard_time_decode_all(axs[10], 'all')
        except: traceback.print_exc()()
        try: plot_standard_time_decode_all(axs[11], 'up')
        except: traceback.print_exc()()
        try: plot_standard_time_decode_all(axs[12], 'down')
        except: traceback.print_exc()()
        try: plot_standard_ramp_params(axs[13], 0)
        except: traceback.print_exc()()
        try: plot_standard_ramp_params(axs[14], 1)
        except: traceback.print_exc()()
        try: plot_standard_ramp_params(axs[15], 2)
        except: traceback.print_exc()()
        try: plot_standard_ramp_params(axs[16], 3)
        except: traceback.print_exc()()
        try: plot_standard_ramp_params(axs[17], 4)
        except: traceback.print_exc()()
        try: plot_standard_time_regress_drop_neu_all(axs[18])
        except: traceback.print_exc()()
        try: plot_example_ramp_fit(axs[19])
        except: traceback.print_exc()()
        
        try: plot_standard_ramp_params_all(axs[15], 0)
        except: traceback.print_exc()()
        try: plot_standard_ramp_params_all(axs[16], 1)
        except: traceback.print_exc()()
        try: plot_standard_ramp_params_all(axs[17], 2)
        except: traceback.print_exc()()
        try: plot_standard_ramp_params_all(axs[18], 3)
        except: traceback.print_exc()()
        try: plot_standard_ramp_params_all(axs[19], 4)
        except: traceback.print_exc()()
        try: plot_ramp_heatmap(axs[20], 'up')
        except: traceback.print_exc()()
        try: plot_ramp_heatmap(axs[21], 'down')
        except: traceback.print_exc()()

# colors = ['#989A9C', '#A4CB9E', '#9DB4CE', '#EDA1A4', '#F9C08A']
class plotter_main(plotter_utils):
    def __init__(self, neural_trials, labels, significance, label_names, temp_folder, cate_list):
        super().__init__(neural_trials, labels, significance, temp_folder, cate_list)
        self.label_names = label_names
    
    def cluster_all(self, axs_all):
        for cate, axs in zip(self.cate_list, axs_all):
            try:
                label_name = self.label_names[str(cate[0])] if len(cate)==1 else 'all'
                print(f'plotting results for {label_name}')

                self.plot_cluster_all(axs, cate=cate)

            except: traceback.print_exc()()

    def cluster_heatmap_all(self, axs_all):
        for cate, axs in zip(self.cate_list, axs_all):
            try:
                label_name = self.label_names[str(cate[0])] if len(cate)==1 else 'all'
                print(f'plotting results for {label_name}')

                self.plot_cluster_heatmap_all(axs, cate=cate)

            except: traceback.print_exc()()

    def cluster_adapt_all(self, axs_all):
        for cate, axs in zip(self.cate_list, axs_all):
            try:
                label_name = self.label_names[str(cate[0])] if len(cate)==1 else 'all'
                print(f'plotting results for {label_name}')

                self.plot_cluster_adapt_all(axs, cate=cate)
                
            except: traceback.print_exc()()
    
    def sorted_heatmaps_all(self, axs_all):
        for cate, axs in zip(self.cate_list, axs_all):
            try:
                label_name = self.label_names[str(cate[0])] if len(cate)==1 else 'all'
                print(f'plotting results for {label_name}')
                
                for axss, norm_mode in zip([axs[0:2], axs[2:4], axs[4:8]], ['none', 'minmax', 'share']):
                    self.plot_sorted_heatmaps_all(axss, norm_mode, cate=cate)
                    
            except: traceback.print_exc()()
    
    def latent_all(self, axs_all):
        for cate, axs in zip(self.cate_list, axs_all):
            try:
                label_name = self.label_names[str(cate[0])] if len(cate)==1 else 'all'
                print(f'plotting results for {label_name}')

                self.plot_latent_all(axs, cate=cate)              

            except: traceback.print_exc()()
    
    def decode_all(self, axs_all):
        for cate, axs in zip(self.cate_list, axs_all):
            try:
                label_name = self.label_names[str(cate[0])] if len(cate)==1 else 'all'
                print(f'plotting results for {label_name}')

                #self.plot_decode_all(axs[0], 0, cate=cate)
                self.plot_decode_all(axs[1], 1, cate=cate)              

            except: traceback.print_exc()()
