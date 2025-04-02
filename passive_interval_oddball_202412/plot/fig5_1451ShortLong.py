#!/usr/bin/env python3

import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from modules.Alignment import run_get_stim_response
from modeling.clustering import clustering_neu_response_mode
from modeling.clustering import get_bin_mean_sem_cluster
from modeling.clustering import get_mean_sem_cluster
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
from utils import get_block_1st_idx
from utils import get_block_epochs_idx
from utils import get_block_transition_idx
from utils import get_isi_bin_neu
from utils import get_expect_interval
from utils import get_neu_sync
from utils import get_temporal_scaling_trial_multi_sess
from utils import get_temporal_scaling_data
from utils import run_wilcoxon_trial
from utils import get_cmap_color
from utils import adjust_layout_neu
from utils import adjust_layout_3d_latent
from utils import add_legend
from utils import utils_basic

# fig, ax = plt.subplots(1, 1, figsize=(6, 6))
# fig, axs = plt.subplots(1, 7, figsize=(28, 4))
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
        self.r_frames = int(500*timescale)
        self.list_labels = list_labels
        self.list_neural_trials = list_neural_trials
        self.alignment = run_get_stim_response(
                list_neural_trials, self.l_frames, self.r_frames, expected='none')
        self.list_stim_labels = self.alignment['list_stim_labels']
        self.list_odd_idx = [
            get_odd_stim_prepost_idx(sl) for sl in self.list_stim_labels]
        self.expect = np.nanmin(np.array([get_expect_interval(sl) for sl in self.list_stim_labels]),axis=0)
        self.list_block_start = [get_block_1st_idx(sl, 3) for sl in self.list_stim_labels]
        self.list_significance = list_significance
        self.bin_win = [450,2550]
        self.bin_num = 2
        self.n_clusters = 15
        self.max_clusters = 10
        self.d_latent = 3

    def run_clustering(self, cate):
        n_latents = 25
        cmap_3d, _ = get_cmap_color(2, base_color='#2C2C2C', return_cmap=True)
        # construct features for clustering.
        win_eval = [-2500,2500]
        l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, win_eval[0], win_eval[1])
        # short standard.
        _, [neu_1, _, _, _], _, _ = get_neu_trial(
            self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
            trial_param=[[2,3,4,5], [0], None, None, [0], [0]],
            cate=cate, roi_id=None)
        # long standard.
        _, [neu_2, _, _, _], _, _ = get_neu_trial(
            self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
            trial_param=[[2,3,4,5], [1], None, None, [0], [0]],
            cate=cate, roi_id=None)
        # short oddball.
        _, [neu_3, _, _, _], _, _ = get_neu_trial(
            self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
            trial_idx=[l[0] for l in self.list_odd_idx],
            cate=cate, roi_id=None)
        # long oddball.
        _, [neu_4, _, _, _], _, _ = get_neu_trial(
            self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
            trial_idx=[l[1] for l in self.list_odd_idx],
            cate=cate, roi_id=None)
        # transition from short to long.
        _, [neu_5, _, _, _], _, _ = get_neu_trial(
            self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
            trial_idx=[l[1] for l in self.list_block_start],
            cate=cate, roi_id=None)
        # transition from long to short.
        _, [neu_6, _, _, _], _, _ = get_neu_trial(
            self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
            trial_idx=[l[0] for l in self.list_block_start],
            cate=cate, roi_id=None)
        # collect data.
        neu_all = [neu[:,l_idx:r_idx] for neu in [neu_1, neu_2, neu_3, neu_4, neu_5, neu_6]]
        # get correlation matrix.
        cluster_corr = [np.corrcoef(neu) for neu in neu_all]
        cluster_corr = np.concatenate(cluster_corr, axis=1)
        # extract features.
        model = PCA(n_components=n_latents)
        neu_x = model.fit_transform(cluster_corr)
        # run clustering on extracted correlation.
        metrics, cluster_id = clustering_neu_response_mode(
            neu_x, self.n_clusters, self.max_clusters)
        return metrics, cluster_id

    def run_features_categorization(self, cate):
        win_eval = [-1000,1000]
        l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, win_eval[0], win_eval[1])
        # collect data.
        _, [neu_seq_mean, neu_seq_sem, stim_seq, stim_value], _, [n_trials, n_neurons] = get_neu_trial(
            self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
            trial_param=[[2,3,4,5], None, None, None, [0], [0]],
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
    
    def plot_standard(self, ax, standard, cate=None, roi_id=None):
        xlim = [-3000,3000]
        neu_mean = []
        neu_sem = []
        n_trials = 0
        # collect data.
        if 0 in standard:
            [[color0, color1, color2, _],
             [neu_seq_short, _, stim_seq_short, stim_value_short], _,
             [n_trials_short, n_neurons]] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance,
                [exclude_odd_stim(sl) for sl in self.list_stim_labels],
                trial_param=[[2,3,4,5], [0], None, None, [0], [0]],
                cate=cate, roi_id=roi_id)
            mean_short, sem_short = get_mean_sem(neu_seq_short)
            neu_mean.append(mean_short)
            neu_sem.append(sem_short)
            n_trials += n_trials_short
        if 1 in standard:
            [[color0, color1, color2, _],
             [neu_seq_long, _, stim_seq_long, stim_value_long], _,
             [n_trials_long, n_neurons]] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance,
                [exclude_odd_stim(sl) for sl in self.list_stim_labels],
                trial_param=[[2,3,4,5], [1], None, None, [0], [0]],
                cate=cate, roi_id=roi_id)
            mean_long, sem_long = get_mean_sem(neu_seq_long)
            neu_mean.append(mean_long)
            neu_sem.append(sem_long)
            n_trials += n_trials_long
        # find bounds.
        upper = np.nanmax(neu_mean) + np.nanmax(neu_sem)
        lower = np.nanmin(neu_mean) - np.nanmax(neu_sem)
        # plot stimulus.
        if 0 in standard:
            for i in range(stim_seq_short.shape[0]):
                ax.fill_between(
                    stim_seq_short[i,:],
                    lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
                    color=color1, edgecolor='none', alpha=0.25, step='mid')
            self.plot_vol(
                ax, self.alignment['stim_time'],
                stim_value_short.reshape(1,-1), color1, upper, lower)
        if 1 in standard:
            for i in range(stim_seq_long.shape[0]):
                ax.fill_between(
                    stim_seq_long[i,:],
                    lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
                    color=color2, edgecolor='none', alpha=0.25, step='mid')
            self.plot_vol(
                ax, self.alignment['stim_time'],
                stim_value_long.reshape(1,-1), color2, upper, lower)
        # plot neural traces.
        if 0 in standard:
            self.plot_mean_sem(
                ax, self.alignment['neu_time'],
                mean_short, sem_short, color1, 'short')
        if 1 in standard:
            self.plot_mean_sem(
                ax, self.alignment['neu_time'],
                mean_long, sem_long, color2, 'long')
        adjust_layout_neu(ax)
        ax.set_xlim(xlim)
        ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
        ax.set_xlabel('time since stim (ms)')
        add_legend(ax, [color1,color2], ['short','long'],
                   n_trials, n_neurons, self.n_sess, 'upper right')

    def plot_standard_box(self, ax, cate=None, roi_id=None):
        win_base = [-self.bin_win[1],0]
        offsets = [0.0, 0.1]
        # collect data.
        [[color0, color1, color2, _],
         [neu_seq_short, _, stim_seq_short, stim_value_short], _,
         [n_trials_short, n_neurons]] = get_neu_trial(
            self.alignment, self.list_labels, self.list_significance,
            [exclude_odd_stim(sl) for sl in self.list_stim_labels],
            trial_param=[[2,3,4,5], [0], None, None, [0], [0]],
            cate=cate, roi_id=roi_id)
        [[color0, color1, color2, _],
         [neu_seq_long, _, stim_seq_long, stim_value_long], _,
         [n_trials_long, n_neurons]] = get_neu_trial(
            self.alignment, self.list_labels, self.list_significance,
            [exclude_odd_stim(sl) for sl in self.list_stim_labels],
            trial_param=[[2,3,4,5], [1], None, None, [0], [0]],
            cate=cate, roi_id=roi_id)
        # plot errorbar.
        self.plot_win_mag_box(ax, neu_seq_short, self.alignment['neu_time'], win_base, color1, 0, offsets[0])
        self.plot_win_mag_box(ax, neu_seq_long,  self.alignment['neu_time'], win_base, color2, 0, offsets[1])
        # adjust layout.
        add_legend(ax, [color1,color2], ['short','long'],
                   n_trials_short+n_trials_long, n_neurons, self.n_sess, 'upper right')

    def plot_standard_latent(self, ax, standard, cate=None):
        win_eval = [-3000, 3000]
        l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, win_eval[0], win_eval[1])
        # collect data.
        [[color0, color1, color2, _],
         [neu_x, _, stim_seq, _], _,
         [n_trials, n_neurons]] = get_neu_trial(
            self.alignment, self.list_labels, self.list_significance,
            [exclude_odd_stim(sl) for sl in self.list_stim_labels],
            trial_param=[[2,3,4,5], [standard], None, None, [0], [0]],
            cate=cate, roi_id=None)
        neu_x = neu_x[:, l_idx:r_idx]
        c_stim = [[color1, color2][standard]] * stim_seq.shape[-2]
        cmap, c_neu = get_cmap_color(
            neu_x.shape[1], base_color=[color1, color2][standard],
            return_cmap=True)
        # fit model.
        model = TSNE(n_components=self.d_latent)
        neu_z = model.fit_transform(
            neu_x.reshape(neu_x.shape[0],-1).T
            ).T.reshape(self.d_latent, -1)
        # plot dynamics.
        self.plot_3d_latent_dynamics(ax, neu_z, stim_seq, c_neu, c_stim)
        # adjust layout.
        adjust_layout_3d_latent(ax, neu_z, cmap, self.alignment['neu_time'], 'time since stim (ms)')
        add_legend(ax, [color0,color1,color2], ['pre interval stim','short','long'],
                   n_trials, n_neurons, self.n_sess, 'upper right', dim=3)

    def plot_oddball(self, ax, standard, cate=None, roi_id=None):
        xlim = [-3000,3000]
        # collect data.
        [[_, color1, color2, _],
         [neu_seq, _, stim_seq, stim_value], _,
         [n_trials, n_neurons]] = get_neu_trial(
            self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
            trial_idx=[l[1-standard] for l in self.list_odd_idx],
            cate=cate, roi_id=roi_id)
        neu_mean, neu_sem = get_mean_sem(neu_seq)
        c_idx = int(stim_seq.shape[0]/2)
        c_stim = [[color1, color2][standard]] * stim_seq.shape[-2]
        c_stim[c_idx] = [color1, color2][1-standard]
        # find bounds.
        upper = np.nanmax(neu_mean) + np.nanmax(neu_sem)
        lower = np.nanmin(neu_mean) - np.nanmax(neu_sem)
        # plot stimulus.
        for i in range(stim_seq.shape[0]):
            ax.fill_between(
                stim_seq[i,:],
                lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
                color=c_stim[i], edgecolor='none', alpha=0.25, step='mid')
        self.plot_vol(
            ax, self.alignment['stim_time'],
            stim_value.reshape(1,-1), [color1, color2][standard], upper, lower)
        ax.axvline(
            stim_seq[c_idx,1]+self.expect[standard],
            color=[color1, color2][standard], lw=1, linestyle='--')
        # plot neural traces.
        self.plot_mean_sem(
            ax, self.alignment['neu_time'],
            neu_mean, neu_sem, [color1, color2][standard], None)
        # adjust layout.
        adjust_layout_neu(ax)
        ax.set_xlim(xlim)
        ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
        ax.set_xlabel('time since stim after oddball interval (ms)')
        add_legend(ax, [color1,color2], ['post short interval stim','post long interval stim'],
                   n_trials, n_neurons, self.n_sess, 'upper right')

    def plot_oddball_sync(self, ax, standard, cate=None, roi_id=None):
        xlim = [-3000,3000]
        win_width = 200
        l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, 0, win_width)
        win_width = r_idx - l_idx
        # collect data.
        [[_, color1, color2, _],
         [neu_seq, _, stim_seq, stim_value], _,
         [n_trials, n_neurons]] = get_neu_trial(
            self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
            trial_idx=[l[1-standard] for l in self.list_odd_idx],
            cate=cate, roi_id=None)
        sync = get_neu_sync(neu_seq, win_width)
        c_idx = int(stim_seq.shape[0]/2)
        c_stim = [[color1, color2][standard]] * stim_seq.shape[-2]
        c_stim[c_idx] = [color1, color2][1-standard]
        # find bounds.
        upper = np.nanmax(sync)
        lower = np.nanmin(sync)
        # plot stimulus.
        for i in range(stim_seq.shape[0]):
            ax.fill_between(
                stim_seq[i,:],
                lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
                color=c_stim[i], edgecolor='none', alpha=0.25, step='mid')
        self.plot_vol(
            ax, self.alignment['stim_time'],
            stim_value.reshape(1,-1), [color1, color2][standard], upper, lower)
        ax.axvline(
            stim_seq[c_idx,1]+self.expect[standard],
            color=[color1, color2][standard], lw=1, linestyle='--')
        # plot neural traces.
        ax.plot(self.alignment['neu_time'][win_width:], sync, color=[color1, color2][standard])
        # adjust layout.
        adjust_layout_neu(ax)
        ax.set_xlim(xlim)
        ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
        ax.set_xlabel('time since stim after oddball interval (ms)')
        ax.set_ylabel('synchronization metric')
        add_legend(ax, [color1,color2], ['pre short interval stim','pre long interval stim'],
                   n_trials, n_neurons, self.n_sess, 'upper right')

    def plot_oddball_latent(self, ax, standard, cate=None):
        win_eval = [-5000, 5000]
        l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, win_eval[0], win_eval[1])
        # collect data.
        [[color0, color1, color2, _],
         [neu_x, _, stim_seq, _], _,
         [n_trials, n_neurons]] = get_neu_trial(
            self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
            trial_idx=[l[1-standard] for l in self.list_odd_idx],
            cate=cate, roi_id=None)
        neu_x = neu_x[:, l_idx:r_idx]
        cmap, c_neu = get_cmap_color(
            neu_x.shape[1], base_color=[color1, color2][standard],
            return_cmap=True)
        c_idx = int(stim_seq.shape[0]/2)
        c_stim = [[color1, color2][standard]] * stim_seq.shape[-2]
        c_stim[c_idx] = [color1, color2][1-standard]
        # fit model.
        model = TSNE(n_components=self.d_latent)
        neu_z = model.fit_transform(
            neu_x.reshape(neu_x.shape[0],-1).T
            ).T.reshape(self.d_latent, -1)
        # plot dynamics.
        self.plot_3d_latent_dynamics(ax, neu_z, stim_seq, c_neu, c_stim)
        # adjust layout.
        adjust_layout_3d_latent(ax, neu_z, cmap, self.alignment['neu_time'], 'time since stim (ms)')
        add_legend(ax, [color1,color2], ['post short interval stim','post long interval stim'],
                   n_trials, n_neurons, self.n_sess, 'upper right', dim=3)

    def plot_block_transition(self, ax, standard, cate=None):
        xlim = [-6000,15000]
        # collect data.
        [[_, color1, color2, _],
         [neu_seq, _, stim_seq, stim_value], _,
         [n_trials, n_neurons]] = get_neu_trial(
            self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
            trial_idx=[l[1-standard] for l in self.list_block_start],
            cate=cate, roi_id=None)
        neu_mean, neu_sem = get_mean_sem(neu_seq)
        c_idx = int(stim_seq.shape[0]/2)
        c_stim = [[color1, color2][standard]] * (c_idx)
        c_stim+= [[color1, color2][1-standard]] * (stim_seq.shape[0]-c_idx)
        # find bounds.
        upper = np.nanmax(neu_mean) + np.nanmax(neu_sem)
        lower = np.nanmin(neu_mean) - np.nanmax(neu_sem)
        # plot stimulus.
        for i in range(stim_seq.shape[0]):
            ax.fill_between(
                stim_seq[i,:],
                lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
                color=c_stim[i], edgecolor='none', alpha=0.25, step='mid')
        self.plot_vol(
            ax, self.alignment['stim_time'],
            stim_value.reshape(1,-1), color1, upper, lower)
        # plot neural traces.
        self.plot_mean_sem(
            ax, self.alignment['neu_time'],
            neu_mean, neu_sem, [color1, color2][standard], None)
        # adjust layout.
        adjust_layout_neu(ax)
        ax.set_xlim(xlim)
        ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
        ax.set_xlabel('time since stim after interval block change (ms)')
        add_legend(
            ax, [color1, color2],
            ['post interval stim (short)', 'post interval stim (long)'],
            n_trials, n_neurons, self.n_sess, 'upper left')

    def plot_block_transition_latent(self, ax, standard, cate=None):
        win_eval = [-6000,15000]
        l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, win_eval[0], win_eval[1])
        # collect data.
        [[color0, color1, color2, _],
         [neu_x, _, stim_seq, _], _,
         [n_trials, n_neurons]] = get_neu_trial(
            self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
            trial_idx=[l[1-standard] for l in self.list_block_start],
            cate=cate, roi_id=None)
        neu_x = neu_x[:, l_idx:r_idx]
        cmap, c_neu = get_cmap_color(
            neu_x.shape[1], base_color=[color1, color2][standard],
            return_cmap=True)
        c_idx = int(stim_seq.shape[0]/2)
        c_stim = [[color1, color2][standard]] * (c_idx+1)
        c_stim+= [[color1, color2][1-standard]] * (stim_seq.shape[0]-c_idx-1)
        # fit model.
        model = TSNE(n_components=self.d_latent)
        neu_z = model.fit_transform(
            neu_x.reshape(neu_x.shape[0],-1).T
            ).T.reshape(self.d_latent, -1)
        # plot dynamics.
        self.plot_3d_latent_dynamics(ax, neu_z, stim_seq, c_neu, c_stim)
        # adjust layout.
        adjust_layout_3d_latent(ax, neu_z, cmap, self.alignment['neu_time'], 'time since stim (ms)')
        add_legend(ax, [color1,color2], ['post short interval stim','post long interval stim'],
                   n_trials, n_neurons, self.n_sess, 'upper right', dim=3)

    def plot_cluster(self, axs, cate=None):
        metrics, cluster_id = self.run_clustering(cate)
        colors = get_cmap_color(self.n_clusters, cmap=self.cluster_cmap)
        lbl = ['cluster #'+str(i) for i in range(self.n_clusters)]
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
        # plot standard response.
        def plot_standard(ax, standard):
            xlim = [-3000, 3000]
            # collect data.
            [[color0, color1, color2, _],
             [neu_seq, _, stim_seq, stim_value], _,
             [n_trials, n_neurons]] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_param=[[2,3,4,5], [standard], None, None, [0], [0]],
                cate=cate, roi_id=None)
            c_stim = [[color1, color2][standard]] * stim_seq.shape[-2]
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
        # plot oddball response.
        def plot_oddball(ax, standard):
            xlim = [-3500, 5500]
            # collect data.
            [[_, color1, color2, _],
             [neu_seq, _, stim_seq, stim_value], _,
             [n_trials, n_neurons]] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_idx=[l[3-standard] for l in self.list_odd_idx],
                cate=cate, roi_id=None)
            c_idx = int(stim_seq.shape[0]/2)
            c_stim = [[color1, color2][standard]] * stim_seq.shape[-2]
            c_stim[c_idx] = [color1, color2][1-standard]
            # get response within cluster.
            neu_mean, neu_sem = get_mean_sem_cluster(neu_seq, cluster_id)
            # plot results.
            norm_params = [get_norm01_params(neu_mean[i,:]) for i in range(neu_mean.shape[0])]
            self.plot_cluster_mean_sem(
                ax, neu_mean, neu_sem, self.alignment['neu_time'],
                norm_params, stim_seq, c_stim, colors, xlim)
            # adjust layout.
            ax.set_xlim(xlim)
            ax.set_xlabel('time since stim after oddball interval (ms)')
            add_legend(
                ax, [color1, color2]+colors,
                ['post interval stim (short)', 'post interval stim (long)']+lbl,
                n_trials, n_neurons, self.n_sess, 'upper left')
        # plot transition response.
        def plot_block_transition(ax, standard):
            xlim = [-6000,15000]
            # collect data.
            [[color0, color1, color2, _],
             [neu_seq, _, stim_seq, stim_value], _,
             [n_trials, n_neurons]] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_idx=[l[1-standard] for l in self.list_block_start],
                cate=cate, roi_id=None)
            c_idx = int(stim_seq.shape[0]/2)
            c_stim = [[color1, color2][standard]] * (c_idx)
            c_stim+= [[color1, color2][1-standard]] * (stim_seq.shape[0]-c_idx)
            # get response within cluster.
            neu_mean, neu_sem = get_mean_sem_cluster(neu_seq, cluster_id)
            # plot results.
            norm_params = [get_norm01_params(neu_mean[i,:]) for i in range(neu_mean.shape[0])]
            self.plot_cluster_mean_sem(
                ax, neu_mean, neu_sem, self.alignment['neu_time'],
                norm_params, stim_seq, c_stim, colors, xlim)
            # adjust layout.
            ax.set_xlim(xlim)
            ax.set_xlabel('time since stim after interval block change (ms)')
            add_legend(
                ax, [color1, color2]+colors,
                ['post interval stim (short)', 'post interval stim (long)']+lbl,
                n_trials, n_neurons, self.n_sess, 'upper left')
        # plot standard response heatmap.
        def plot_heatmap_standard(ax, standard):
            xlim = [-3000, 3000]
            l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, xlim[0], xlim[1])
            # collect data.
            _, [neu_seq, _, _, _], _, _ = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_param=[[2,3,4,5], [standard], None, None, [0], [0]],
                cate=cate, roi_id=None)
            neu_seq = neu_seq[:,l_idx:r_idx]
            neu_time = self.alignment['neu_time'][l_idx:r_idx]
            # plot heatmap.
            self.plot_cluster_heatmap(ax, neu_seq, neu_time, neu_seq, cluster_id, colors)
        # plot oddball response heatmap.
        def plot_heatmap_oddball(ax, standard):
            xlim = [-3500, 5500]
            l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, xlim[0], xlim[1])
            # collect data.
            [[_, color1, color2, _],
             [neu_seq, _, stim_seq, stim_value], _,
             [n_trials, n_neurons]] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_idx=[l[3-standard] for l in self.list_odd_idx],
                cate=cate, roi_id=None)
            neu_seq = neu_seq[:,l_idx:r_idx]
            neu_time = self.alignment['neu_time'][l_idx:r_idx]
            # plot heatmap.
            self.plot_cluster_heatmap(ax, neu_seq, neu_time, neu_seq, cluster_id, colors)
        # plot transition response.
        def plot_heatmap_block_transition(ax, standard):
            xlim = [-6000,15000]
            l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, xlim[0], xlim[1])
            # collect data.
            _, [neu_seq, _, stim_seq, stim_value], _, [n_trials, n_neurons] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_idx=[l[1-standard] for l in self.list_block_start],
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
        try: plot_standard(axs[6], 0)
        except: pass
        try: plot_standard(axs[7], 1)
        except: pass
        try: plot_oddball(axs[8], 0)
        except: pass
        try: plot_oddball(axs[9], 1)
        except: pass
        try: plot_block_transition(axs[10], 0)
        except: pass
        try: plot_block_transition(axs[11], 1)
        except: pass
        try: plot_heatmap_standard(axs[12], 0)
        except: pass
        try: plot_heatmap_standard(axs[13], 1)
        except: pass
        try: plot_heatmap_oddball(axs[14], 0)
        except: pass
        try: plot_heatmap_oddball(axs[15], 1)
        except: pass
        try: plot_heatmap_block_transition(axs[16], 0)
        except: pass
        try: plot_heatmap_block_transition(axs[17], 1)
        except: pass

    def plot_categorization_features(self, axs):
        list_target_metric = ['response_latency', 'peak_amp', 'ramp_slope', 'decay_rate']
        bin_num = 2
        cate = [-1,1,2]
        n_latents = 15
        kernel_time, kernel_all, exp_var_all = self.run_glm(cate)
        model = PCA(n_components=n_latents if n_latents < kernel_all.shape[1] else kernel_all.shape[1])
        neu_x = model.fit_transform(kernel_all)
        _, cluster_id = clustering_neu_response_mode(neu_x, self.n_clusters, None)
        #results_all = self.run_features_categorization(cate)
        #cluster_id = results_all['cluster_id'].to_numpy()
        lbl = ['cluster #'+str(ci) for ci in range(self.n_clusters)]
        colors = get_cmap_color(len(lbl), cmap=self.cluster_cmap)
        # collect data.
        [[color0, _, _, _], [neu_seq_short, _, stim_seq_short, stim_value_short],
         [neu_labels, neu_sig], [n_trials_short, n_neurons]] = get_neu_trial(
            self.alignment, self.list_labels, self.list_significance,
            [exclude_odd_stim(sl) for sl in self.list_stim_labels],
            trial_param=[[2,3,4,5], [0], None, None, [0], [0]],
            cate=cate, roi_id=None)
        [[color0, _, _, _], [neu_seq_long, _, stim_seq_long, stim_value_long],
         [neu_labels, neu_sig], [n_trials_long, n_neurons]] = get_neu_trial(
            self.alignment, self.list_labels, self.list_significance,
            [exclude_odd_stim(sl) for sl in self.list_stim_labels],
            trial_param=[[2,3,4,5], [1], None, None, [0], [0]],
            cate=cate, roi_id=None)
        n_trials = n_trials_short + n_trials_long
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
        # compare respons on short and long standard.
        def plot_standard(axs):
            xlim = [-3000,3000]
            # plot results for each class.
            for ci in range(len(lbl)):
                neu_mean = []
                neu_sem = []
                # get class data.
                mean_short, sem_short = get_mean_sem(neu_seq_short[cluster_id==ci,:])
                neu_mean.append(mean_short)
                neu_sem.append(sem_short)
                mean_long, sem_long = get_mean_sem(neu_seq_long[cluster_id==ci,:])
                neu_mean.append(mean_long)
                neu_sem.append(sem_long)
                # find bounds.
                upper = np.nanmax(neu_mean) + np.nanmax(neu_sem)
                lower = np.nanmin(neu_mean) - np.nanmax(neu_sem)
                # find class colors.
                cs = get_cmap_color(3, base_color=colors[ci])
                color1, color2 = cs[0], cs[-1]
                # plot stimulus.
                for i in range(stim_seq_short.shape[0]):
                    axs[ci].fill_between(
                        stim_seq_short[i,:],
                        lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
                        color=color1, edgecolor='none', alpha=0.25, step='mid')
                self.plot_vol(
                    axs[ci], self.alignment['stim_time'],
                    stim_value_short.reshape(1,-1), color1, upper, lower)
                for i in range(stim_seq_long.shape[0]):
                    axs[ci].fill_between(
                        stim_seq_long[i,:],
                        lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
                        color=color2, edgecolor='none', alpha=0.25, step='mid')
                self.plot_vol(
                    axs[ci], self.alignment['stim_time'],
                    stim_value_long.reshape(1,-1), color2, upper, lower)
                # plot neural traces.
                self.plot_mean_sem(
                    axs[ci], self.alignment['neu_time'],
                    mean_short, sem_short, color1, 'short')
                self.plot_mean_sem(
                    axs[ci], self.alignment['neu_time'],
                    mean_long, sem_long, color2, 'long')
                # adjust layouts.
                adjust_layout_neu(axs[ci])
                axs[ci].set_xlim(xlim)
                axs[ci].set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
                axs[ci].set_xlabel('time since stim (ms)')
                axs[ci].set_title(f'response comparison for cluster #{ci}')
                add_legend(axs[ci], [color1,color2], ['short','long'],
                           n_trials, n_neurons, self.n_sess, 'upper right')
        # quantification comparison between short and long.
        @show_memory_usage
        def plot_standard_box(axs):
            # find time range to evaluate.
            c_idx = int(stim_seq_long.shape[0]/2)
            list_win_eval = [[0, stim_seq_short[c_idx+1,1]],
                             [0, stim_seq_long[c_idx+1,1]]]
            # get quantification results.
            list_metrics = get_all_metrics(
                [neu_seq_short, neu_seq_long], self.alignment['neu_time'], list_win_eval)
            # plot results for each class.
            for mi in range(len(list_target_metric)):
                m = [lm[list_target_metric[mi]] for lm in list_metrics]
                self.plot_cluster_metric_box(
                    axs[mi], m, list_target_metric[mi], cluster_id, colors)
        # compare temporal scaling.
        @show_memory_usage
        def plot_temporal_scaling(axs):
            target_isi = 1500
            # collect data.
            [[color0, _, _, _], [neu_seq, stim_seq, stim_value, pre_isi],
             [neu_labels, neu_sig], [n_trials, n_neurons]] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_param=[[2,3,4,5], None, None, None, [0], [0]],
                mean_sem=False,
                cate=cate, roi_id=None)
            c_idx = int(stim_seq[0][0,:,:].shape[0]/2)
            # compute mean time stamps.
            stim_seq_target = np.nanmean(np.concatenate(stim_seq, axis=0),axis=0)
            neu_seq = get_temporal_scaling_trial_multi_sess(
                neu_seq, stim_seq, self.alignment['neu_time'], target_isi)
            # bin data based on isi.
            [_, _, _, neu_seq, _, _, _, _] = get_isi_bin_neu(
                neu_seq, stim_seq, stim_value, pre_isi, self.bin_win, bin_num)
            cluster_bin_neu_mean, cluster_bin_neu_sem = get_bin_mean_sem_cluster(neu_seq, cluster_id)
            # plot results for each class.
            l_idx, r_idx = get_frame_idx_from_time(
                self.alignment['neu_time'], 0, -target_isi, stim_seq_target[c_idx,1]+target_isi)
            neu_time = self.alignment['neu_time'][l_idx:r_idx]
            for ci in range(len(lbl)):
                # find bounds.
                upper = np.nanmax(cluster_bin_neu_mean[:,ci,:]) + np.nanmax(cluster_bin_neu_sem[:,ci,:])
                lower = np.nanmin(cluster_bin_neu_mean[:,ci,:]) - np.nanmax(cluster_bin_neu_sem[:,ci,:])
                # find class colors.
                cs = get_cmap_color(bin_num+2, base_color=colors[ci])
                cs = [cs[0], cs[1]]
                # plot stimulus.
                for si in range(stim_seq_target.shape[0]):
                    axs[ci].fill_between(
                        stim_seq_target[si,:],
                        lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
                        color=color0, edgecolor='none', alpha=0.25, step='mid')
                # plot neural traces.
                for bi in range(bin_num):
                    self.plot_mean_sem(
                        axs[ci], neu_time,
                        cluster_bin_neu_mean[bi,ci,:], cluster_bin_neu_sem[bi,ci,:], cs[bi], None)
                # adjust layouts.
                adjust_layout_neu(axs[ci])
                axs[ci].set_xlim([-target_isi-stim_seq_target[c_idx,1], 2*stim_seq_target[c_idx,1]+target_isi])
                axs[ci].set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
                axs[ci].set_xlabel('time since stim (ms)')
                axs[ci].set_title(f'response comparison with scaling for cluster #{ci}')
                add_legend(axs[ci], cs, ['short','long'], None, None, None, 'upper right')
        # quantification comparison between short and long with temporal scaling.
        @show_memory_usage
        def plot_temporal_scaling_box(axs):
            target_isi = 1500
            # collect data.
            [[color0, _, _, _], [neu_seq, stim_seq, stim_value, pre_isi],
             [neu_labels, neu_sig], [n_trials, n_neurons]] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_param=[[2,3,4,5], None, None, None, [0], [0]],
                mean_sem=False,
                cate=cate, roi_id=None)
            # compute mean time stamps.
            stim_seq_target = np.nanmean(np.concatenate(stim_seq, axis=0),axis=0)
            neu_seq = get_temporal_scaling_trial_multi_sess(
                neu_seq, stim_seq, self.alignment['neu_time'], target_isi)
            # bin data based on isi.
            [_, _, _, neu_seq, _, _, _, _] = get_isi_bin_neu(
                neu_seq, stim_seq, stim_value, pre_isi, self.bin_win, bin_num)
            # find time range to evaluate.
            c_idx = int(stim_seq_target.shape[0]/2)
            list_win_eval = [[0, stim_seq_target[c_idx+1,1]],
                             [0, stim_seq_target[c_idx+1,1]]]
            l_idx, r_idx = get_frame_idx_from_time(
                self.alignment['neu_time'], 0, -target_isi, stim_seq_target[c_idx,1]+target_isi)
            neu_time = self.alignment['neu_time'][l_idx:r_idx]
            # get quantification results.
            list_metrics = get_all_metrics(
                neu_seq, neu_time, list_win_eval)
            # plot results for each class.
            for mi in range(len(list_target_metric)):
                m = [lm[list_target_metric[mi]] for lm in list_metrics]
                self.plot_cluster_metric_box(
                    axs[mi], m, list_target_metric[mi], cluster_id, colors)
        # block adaptation with epoches.
        def plot_block_adapatation(axs, standard):
            n_epoch = 2
            xlim = [-1500,3000]
            # find epoch indices.
            epoch_len = 2
            list_epoch_idx = [
                get_block_epochs_idx(sl[:,3], epoch_len)[standard]
                for sl in self.list_stim_labels]
            # collect data.
            _, [odd_neu_seq, _, odd_stim_seq, odd_stim_value], _, _ = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_idx=[l[standard] for l in self.list_odd_idx],
                cate=cate, roi_id=None)
            epoch_neu_trial = [get_neu_trial(
                self.alignment, self.list_labels, self.list_significance,
                [exclude_odd_stim(sl) for sl in self.list_stim_labels],
                trial_idx=[epoch_idx[ei,:] for epoch_idx in list_epoch_idx],
                trial_param=[[2,3,4,5], [standard], None, None, [0], [0]],
                cate=cate, roi_id=None)[1]
                for ei in range(n_epoch)]
            epoch_neu_seq = [ent[0] for ent in epoch_neu_trial]
            epoch_stim_seq = epoch_neu_trial[-1][2]
            epoch_stim_value = epoch_neu_trial[-1][3]
            # plot results for each class.
            for ci in range(len(lbl)):
                neu_mean = []
                neu_sem = []
                # find class colors.
                cn = get_cmap_color(n_epoch+1, base_color=colors[ci])
                cn = [color0] + cn[:n_epoch]
                # get data for oddball response.
                m, s = get_mean_sem(odd_neu_seq[cluster_id==ci,:])
                neu_mean.append(m)
                neu_sem.append(s)
                # get class data for each epoch.
                for ei in range(n_epoch):
                    m, s = get_mean_sem(epoch_neu_seq[ei][cluster_id==ci,:])
                    neu_mean.append(m)
                    neu_sem.append(s)
                # find bounds.
                upper = np.nanmax(neu_mean) + np.nanmax(neu_sem)
                lower = np.nanmin(neu_mean) - np.nanmax(neu_sem)
                # plot stimulus.
                for i in range(epoch_stim_seq.shape[0]):
                    axs[ci].fill_between(
                        epoch_stim_seq[i,:],
                        lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
                        color=color0, edgecolor='none', alpha=0.25, step='mid')
                self.plot_vol(
                    axs[ci], self.alignment['stim_time'],
                    epoch_stim_value.reshape(1,-1), color0, upper, lower)
                # plot neural traces.
                for ei in range(n_epoch+1):
                    self.plot_mean_sem(
                        axs[ci], self.alignment['neu_time'],
                        neu_mean[ei], neu_sem[ei], cn[ei], None)
                # adjust layouts.
                adjust_layout_neu(axs[ci])
                axs[ci].set_xlim(xlim)
                axs[ci].set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
                axs[ci].set_xlabel('time since stim (ms)')
                add_legend(axs[ci], cn,
                           ['oddball']+[f'epoch #{ei}' for ei in range(n_epoch)],
                           n_trials, n_neurons, self.n_sess, 'upper right')
        # quantification comparison between epochs.
        def plot_block_adapatation_box(axs, standard):
            n_epoch = 2
            # find epoch indices.
            epoch_len = 2
            list_epoch_idx = [
                get_block_epochs_idx(sl[:,3], epoch_len)[standard]
                for sl in self.list_stim_labels]
            # collect data.
            _, [odd_neu_seq, _, odd_stim_seq, odd_stim_value], _, _ = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_idx=[l[standard] for l in self.list_odd_idx],
                cate=cate, roi_id=None)
            epoch_neu_trial = [get_neu_trial(
                self.alignment, self.list_labels, self.list_significance,
                [exclude_odd_stim(sl) for sl in self.list_stim_labels],
                trial_idx=[epoch_idx[ei,:] for epoch_idx in list_epoch_idx],
                trial_param=[[2,3,4,5], [standard], None, None, [0], [0]],
                cate=cate, roi_id=None)[1]
                for ei in range(n_epoch)]
            epoch_neu_seq = [ent[0] for ent in epoch_neu_trial]
            epoch_stim_seq = epoch_neu_trial[-1][2]
            # find time range to evaluate.
            c_idx = int(epoch_stim_seq.shape[0]/2)
            list_win_eval = [[0, epoch_stim_seq[c_idx+1,1]],
                             [0, epoch_stim_seq[c_idx+1,1]]]
            # get quantification results.
            list_metrics = get_all_metrics(
                epoch_neu_seq, self.alignment['neu_time'], list_win_eval)
            # plot results for each class.
            for mi in range(len(list_target_metric)):
                m = [lm[list_target_metric[mi]] for lm in list_metrics]
                self.plot_cluster_metric_box(
                    axs[mi], m, list_target_metric[mi], cluster_id, colors)
        # single trial population block adaptation.
        @show_memory_usage
        def plot_block_adapatation_trial_heatmap(axs, norm_mode):
            gap_margin = 10
            trials_around = 15
            xlim = [-1000,2500]
            l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, xlim[0], xlim[1])
            # get transition trials indice.
            list_trans_0to1 = [get_block_transition_idx(sl[:,3], trials_around)[0] for sl in self.list_stim_labels]
            list_trans_1to0 = [get_block_transition_idx(sl[:,3], trials_around)[1] for sl in self.list_stim_labels]
            # collect data.
            neu_time = self.alignment['neu_time'][l_idx:r_idx]
            _, [_, _, stim_seq, _], _, _ = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_param=[None, None, None, None, [0], [0]],
                cate=cate, roi_id=None)
            # preallocate arrays for 0to1 and 1to0 transitions
            neu_trans_0to1 = []
            for ti in range(list_trans_0to1[0].shape[0]):
                trial_idx = [t01[ti, :] for t01 in list_trans_0to1]
                trial_data = get_neu_trial(
                    self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                    trial_param=[None, None, None, None, [0], [0]],
                    trial_idx=trial_idx,
                    mean_sem=False,
                    cate=cate, roi_id=None)[1][0]
                neu_trans_0to1.append(np.concatenate(trial_data, axis=1))
            neu_trans_0to1 = np.stack(neu_trans_0to1, axis=0)
            neu_trans_0to1 = np.nanmean(neu_trans_0to1, axis=0)
            neu_trans_1to0 = []
            for ti in range(list_trans_1to0[0].shape[0]):
                trial_idx = [t10[ti, :] for t10 in list_trans_1to0]
                trial_data = get_neu_trial(
                    self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                    trial_param=[None, None, None, None, [0], [0]],
                    trial_idx=trial_idx,
                    mean_sem=False,
                    cate=cate, roi_id=None)[1][0]
                neu_trans_1to0.append(np.concatenate(trial_data, axis=1))
            neu_trans_1to0 = np.stack(neu_trans_1to0, axis=0)
            neu_trans_1to0 = np.nanmean(neu_trans_1to0, axis=0)
            # extract transition for each class.
            neu_x = []
            zero_gap = np.full((gap_margin, r_idx - l_idx), np.nan)
            for ci in range(len(lbl)):
                cluster_mask = cluster_id == ci
                left = np.nanmean(neu_trans_0to1[:, cluster_mask, l_idx:r_idx], axis=1)
                right = np.nanmean(neu_trans_1to0[:, cluster_mask, l_idx:r_idx], axis=1)
                neu = np.concatenate([left, zero_gap, right], axis=0)
                neu_x.append(neu)
            # plot results for each class.
            for ci in range(len(lbl)):
                cmap, _ = get_cmap_color(1, base_color=colors[ci], return_cmap=True)
                self.plot_heatmap_trial(
                    axs[ci], neu_x[ci], neu_time, cmap, norm_mode, neu_x)
                axs[ci].set_xlabel('time since stim (ms)')
                axs[ci].set_ylabel('trial since transition')
            # add stimulus line.
            n_trials = neu_trans_1to0.shape[0]
            total = 2 * n_trials + gap_margin
            for ci in range(len(lbl)):
                c_idx = stim_seq.shape[0] // 2
                z, _ = get_frame_idx_from_time(neu_time, 0, 0, 0)
                e1, e2 = get_frame_idx_from_time(
                    neu_time, 0,
                    stim_seq[c_idx,1]+self.expect[0],
                    stim_seq[c_idx,1]+self.expect[1])
                # stimulus before interval.
                axs[ci].axvline(neu_time[z], 0/total, n_trials/total, color='black', lw=1, linestyle='--')
                axs[ci].axvline(neu_time[z], (n_trials+gap_margin)/total, 1, color='black', lw=1, linestyle='--')
                # stimulus after short interval.
                axs[ci].axvline(neu_time[e1], 0, (n_trials/2)/total, color='black', lw=1, linestyle='--')
                axs[ci].axvline(neu_time[e1], (n_trials+gap_margin+n_trials/2)/total, 1, color='black', lw=1, linestyle='--')
                # stimulus after long interval.
                axs[ci].axvline(neu_time[e2], (n_trials/2)/total, n_trials/total, color='black', lw=1, linestyle='--')
                axs[ci].axvline(neu_time[e2], (n_trials+gap_margin)/total, (n_trials+gap_margin+n_trials/2)/total, color='black', lw=1, linestyle='--')
                # intervals before and after transition.
                axs[ci].axvline(neu_time[z], (n_trials/2)/total, (n_trials/2+1)/total, color='red', lw=3)
                axs[ci].axvline(neu_time[z], (n_trials+gap_margin+n_trials/2-1)/total, (n_trials+gap_margin+n_trials/2)/total, color='red', lw=3)
            # adjust layouts.
            for ci in range(len(lbl)):
                axs[ci].set_yticks([
                    0, int(n_trials/2), n_trials,
                    n_trials + gap_margin, int(n_trials/2+n_trials + gap_margin), total])
                axs[ci].set_yticklabels([
                    int(n_trials/2), 0, -int(n_trials/2), int(n_trials/2), 0, -int(n_trials/2)])
                axs[ci].set_ylim([0, total])
        # single trial population block adaptation.
        @show_memory_usage
        def plot_block_adapatation_trial_box(axs):
            trials_around = 5
            # get transition trials indice.
            list_trans_0to1 = [get_block_transition_idx(sl[:,3], trials_around)[0] for sl in self.list_stim_labels]
            list_trans_1to0 = [get_block_transition_idx(sl[:,3], trials_around)[1] for sl in self.list_stim_labels]
            # collect data.
            _, [_, _, stim_seq, _], _, _ = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_param=[None, None, None, None, [0], [0]],
                cate=cate, roi_id=None)
            list_neu_seq_trial_0to1 = [get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_param=[None, None, None, None, [0], [0]],
                trial_idx=[t01[ti,:] for t01 in list_trans_0to1],
                mean_sem=False,
                cate=cate, roi_id=None)[1][0]
                for ti in range(list_trans_0to1[0].shape[0])]
            list_neu_seq_trial_1to0 = [get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_param=[None, None, None, None, [0], [0]],
                trial_idx=[t10[ti,:] for t10 in list_trans_1to0],
                mean_sem=False,
                cate=cate, roi_id=None)[1][0]
                for ti in range(list_trans_1to0[0].shape[0])]
            # compute block average.
            neu_trans_0to1 = [np.concatenate(nst, axis=1) for nst in list_neu_seq_trial_0to1]
            neu_trans_0to1 = np.concatenate([np.expand_dims(nst, axis=0) for nst in neu_trans_0to1], axis=0)
            neu_trans_0to1 = np.nanmean(neu_trans_0to1, axis=0)
            neu_trans_1to0 = [np.concatenate(nst, axis=1) for nst in list_neu_seq_trial_1to0]
            neu_trans_1to0 = np.concatenate([np.expand_dims(nst, axis=0) for nst in neu_trans_1to0], axis=0)
            neu_trans_1to0 = np.nanmean(neu_trans_1to0, axis=0)
            # find time range to evaluate.
            c_idx = int(stim_seq_short.shape[0]/2)
            list_win_eval_0to1 = [[0, stim_seq_short[c_idx+1,1]]]*trials_around + [[0, stim_seq_long[c_idx+1,1]]]*trials_around
            list_win_eval_1to0 = [[0, stim_seq_long[c_idx+1,1]]]*trials_around + [[0, stim_seq_short[c_idx+1,1]]]*trials_around
            # get quantification results.
            list_metrics_0to1 = get_all_metrics(
                [neu_trans_0to1[ni,:,:] for ni in range(trials_around*2)],
                self.alignment['neu_time'], list_win_eval_0to1)
            list_metrics_1to0 = get_all_metrics(
                [neu_trans_1to0[ni,:,:] for ni in range(trials_around*2)],
                self.alignment['neu_time'], list_win_eval_1to0)
            # plot results for each class.
            for mi in range(len(list_target_metric)):
                metrics_0to1 = [lm[list_target_metric[mi]].reshape(-1,1) for lm in list_metrics_0to1]
                metrics_1to0 = [lm[list_target_metric[mi]].reshape(-1,1) for lm in list_metrics_1to0]
                metrics_0to1 = np.concatenate(metrics_0to1, axis=1)
                metrics_1to0 = np.concatenate(metrics_1to0, axis=1)
                t = np.arange(-trials_around,trials_around)
                for ci in range(len(lbl)):
                    self.plot_cluster_metric_line(
                        axs[0][mi], metrics_0to1, t,
                        list_target_metric[mi], cluster_id, colors)
                    self.plot_cluster_metric_line(
                        axs[1][mi], metrics_1to0, t,
                        list_target_metric[mi], cluster_id, colors)
                    # adjust layout.
                    axs[0][mi].set_xlabel('trial since short to long transition')
                    axs[1][mi].set_xlabel('trial since long to short transition')
        # plot all.
        try: plot_info(axs[0])
        except: pass
        try: plot_standard(axs[1])
        except: pass
        try: plot_standard_box(axs[2])
        except: pass
        try: plot_temporal_scaling(axs[3])
        except: pass
        try: plot_temporal_scaling_box(axs[4])
        except: pass
        try: plot_block_adapatation(axs[5], 0)
        except: pass
        try: plot_block_adapatation_box(axs[6], 1)
        except: pass
        try: plot_block_adapatation(axs[7], 0)
        except: pass
        try: plot_block_adapatation_box(axs[8], 1)
        except: pass
        try: plot_block_adapatation_trial_heatmap(axs[9], 'none')
        except: pass
        try: plot_block_adapatation_trial_heatmap(axs[10], 'minmax')
        except: pass
        try: plot_block_adapatation_trial_heatmap(axs[11], 'share')
        except: pass
        try: plot_block_adapatation_trial_box([axs[12],axs[13]])
        except: pass

    def plot_sorted_heatmaps_standard(self, axs, norm_mode, standard):
        thres = 0.05
        cate = [-1,1]
        win_sort = [-500, 500]
        # collect data.
        neu_x = []
        stim_x = []
        # short standard.
        [_, [neu_seq_short, _, stim_seq_short, _], [neu_labels, neu_sig],
         [n_trials_short, n_neurons]] = get_neu_trial(
            self.alignment, self.list_labels, self.list_significance,
            [exclude_odd_stim(sl) for sl in self.list_stim_labels],
            trial_param=[[2,3,4,5], [0], None, None, [0], [0]],
            cate=cate, roi_id=None)
        # long standard.
        [_, [neu_seq_long, _, stim_seq_long, _], [neu_labels, neu_sig],
         [n_trials_long, n_neurons]] = get_neu_trial(
            self.alignment, self.list_labels, self.list_significance,
            [exclude_odd_stim(sl) for sl in self.list_stim_labels],
            trial_param=[[2,3,4,5], [1], None, None, [0], [0]],
            cate=cate, roi_id=None)
        # get indice for only 2 intervals.
        c_idx = int(stim_seq_short.shape[0]/2)
        l_short, r_short = get_frame_idx_from_time(
            self.alignment['neu_time'], 0, stim_seq_short[c_idx-1,1], stim_seq_short[c_idx+1,0])
        l_long, r_long = get_frame_idx_from_time(
            self.alignment['neu_time'], 0, stim_seq_long[c_idx-1,1], stim_seq_long[c_idx+1,0])
        # cut and pad data.
        neu_seq_long = neu_seq_long[:,l_long:r_long]
        neu_seq_short = np.concatenate([
            np.nan*np.zeros((neu_seq_short.shape[0],l_short-l_long)),
            neu_seq_short[:,l_short:r_short],
            np.nan*np.zeros((neu_seq_short.shape[0],r_long-r_short))],
            axis=1)
        neu_time = self.alignment['neu_time'][l_long:r_long]
        # organize data.
        if standard == 0:
            neu_x += [neu_seq_short, neu_seq_long]
            stim_x += [stim_seq_short, stim_seq_long]
        if standard == 1:
            neu_x += [neu_seq_long, neu_seq_short]
            stim_x += [stim_seq_long, stim_seq_short]
        # difference between short and long.
        neu_x.append(neu_x[0]-neu_x[1])
        stim_x.append(stim_seq_short)
        # wilcoxon test results.
        [_, [neu_seq_short, stim_seq_short, _, _], _, _] = get_neu_trial(
            self.alignment, self.list_labels, self.list_significance,
            [exclude_odd_stim(sl) for sl in self.list_stim_labels],
            trial_param=[[2,3,4,5], [0], None, None, [0], [0]],
            mean_sem=False,
            cate=cate, roi_id=None)
        [_, [neu_seq_long, stim_seq_long, _, _], _, _] = get_neu_trial(
            self.alignment, self.list_labels, self.list_significance,
            [exclude_odd_stim(sl) for sl in self.list_stim_labels],
            trial_param=[[2,3,4,5], [1], None, None, [0], [0]],
            mean_sem=False,
            cate=cate, roi_id=None)
        neu_seq_short = [ns[:,:,l_short:r_short] for ns in neu_seq_short]
        neu_seq_long = [ns[:,:,l_short:r_short] for ns in neu_seq_long]
        if standard == 0:
            _, p_bin = run_wilcoxon_trial(neu_seq_short, neu_seq_long, thres)
        if standard == 1:
            _, p_bin = run_wilcoxon_trial(neu_seq_long, neu_seq_short, thres)
        p_bin = np.concatenate([
            np.nan*np.zeros((p_bin.shape[0],l_short-l_long)),
            p_bin,
            np.nan*np.zeros((p_bin.shape[0],r_long-r_short))],
            axis=1)
        neu_x.append(p_bin)
        stim_x.append(stim_x[-1])
        # plot heatmaps.
        for bi in range(len(neu_x)):
            self.plot_heatmap_neuron(
                axs[bi], neu_x[bi], neu_time, neu_x[0], win_sort, neu_labels, neu_sig,
                norm_mode=norm_mode if bi!=3 else 'binary',
                neu_seq_share=neu_x,
                neu_labels_share=[neu_labels]*len(neu_x),
                neu_sig_share=[neu_sig]*len(neu_x))
            axs[bi].set_xlabel('time since stim (ms) \n sorting window [{},{}] ms'.format(
                win_sort[0], win_sort[1]))
        # add stimulus line.
        for bi in [0,1,2,3]:
            c_idx = int(stim_x[0].shape[0]/2)
            xlines = [neu_time[np.searchsorted(neu_time, t)]
                      for t in [stim_x[bi][c_idx+i,0]
                                for i in [0]]]
            for xl in xlines:
                if xl>neu_time[0] and xl<neu_time[-1]:
                    axs[bi].axvline(xl, color='black', lw=1, linestyle='--')

    def plot_temporal_scaling(self, axs, norm_mode):
        cate = [-1,1]
        win_sort = [-500, 500]
        # collect data.
        neu_x = []
        stim_x = []
        # short standard.
        [_, [neu_seq_short, _, stim_seq_short, _], [neu_labels, neu_sig],
         [n_trials_short, n_neurons]] = get_neu_trial(
            self.alignment, self.list_labels, self.list_significance,
            [exclude_odd_stim(sl) for sl in self.list_stim_labels],
            trial_param=[[2,3,4,5], [0], None, None, [0], [0]],
            cate=cate, roi_id=None)
        # long standard.
        [_, [neu_seq_long, _, stim_seq_long, _], [neu_labels, neu_sig],
         [n_trials_long, n_neurons]] = get_neu_trial(
            self.alignment, self.list_labels, self.list_significance,
            [exclude_odd_stim(sl) for sl in self.list_stim_labels],
            trial_param=[[2,3,4,5], [1], None, None, [0], [0]],
            cate=cate, roi_id=None)
        # get indice for only 2 intervals.
        c_idx = int(stim_seq_short.shape[0]/2)
        t_onset, t_offset = get_frame_idx_from_time(
            self.alignment['neu_time'], 0, stim_seq_short[c_idx,0], stim_seq_short[c_idx,1])
        l_short, r_short = get_frame_idx_from_time(
            self.alignment['neu_time'], 0, stim_seq_short[c_idx-1,1], stim_seq_short[c_idx+1,0])
        l_long, r_long = get_frame_idx_from_time(
            self.alignment['neu_time'], 0, stim_seq_long[c_idx-1,1], stim_seq_long[c_idx+1,0])
        # scale data.
        neu_seq_short_scale = np.concatenate([
            get_temporal_scaling_data(
                neu_seq_short[:,l_short:t_onset],
                self.alignment['neu_time'][l_short:t_onset],
                self.alignment['neu_time'][l_long:t_onset]),
            neu_seq_short[:,t_onset:t_offset],
            get_temporal_scaling_data(
                neu_seq_short[:,t_offset:r_short],
                self.alignment['neu_time'][t_offset:r_short],
                self.alignment['neu_time'][t_offset:r_long])],
            axis=1)
        neu_seq_short = np.concatenate([
            np.nan*np.zeros((neu_seq_short.shape[0],l_short-l_long)),
            neu_seq_short[:,l_short:r_short],
            np.nan*np.zeros((neu_seq_short.shape[0],r_long-r_short))],
            axis=1)
        neu_seq_long = neu_seq_long[:,l_long:r_long]
        neu_time = self.alignment['neu_time'][l_long:r_long]
        # organize data.
        neu_x += [neu_seq_short, neu_seq_short_scale, neu_seq_long]
        stim_x += [stim_seq_short, stim_seq_short, stim_seq_long]
        # plot heatmaps.
        for bi in range(len(neu_x)):
            self.plot_heatmap_neuron(
                axs[bi], neu_x[bi], neu_time, neu_x[0], win_sort, neu_labels, neu_sig,
                norm_mode=norm_mode if bi!=3 else 'binary',
                neu_seq_share=neu_x,
                neu_labels_share=[neu_labels]*len(neu_x),
                neu_sig_share=[neu_sig]*len(neu_x))
            axs[bi].set_xlabel('time since stim (ms) \n sorting window [{},{}] ms'.format(
                win_sort[0], win_sort[1]))
        # add stimulus line.
        for bi in [0,1,2]:
            c_idx = int(stim_x[0].shape[0]/2)
            xlines = [neu_time[np.searchsorted(neu_time, t)]
                      for t in [stim_x[bi][c_idx+i,0]
                                for i in [0]]]
            for xl in xlines:
                if xl>neu_time[0] and xl<neu_time[-1]:
                    axs[bi].axvline(xl, color='black', lw=1, linestyle='--')

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

    def standard(self, axs_all):
        for cate, axs in zip([[-1],[1]], axs_all):
            label_name = self.label_names[str(cate[0])] if len(cate)==1 else 'all'
            try:

                self.plot_standard(axs[0], [0], cate=cate)
                axs[0].set_title(f'response to standard stim \n (short) {label_name}')

                self.plot_standard(axs[1], [1], cate=cate)
                axs[1].set_title(f'response to standard stim \n (long) {label_name}')

                self.plot_standard(axs[2], [0,1], cate=cate)
                axs[2].set_title(f'response to standard stim \n {label_name}')

                self.plot_standard_box(axs[3], cate=cate)
                axs[3].set_title(f'response to standard stim \n {label_name}')

                self.plot_standard_latent(axs[4], 0, cate=cate)
                axs[4].set_title(f'latent dynamics to standard stim \n (short) {label_name}')

                self.plot_standard_latent(axs[5], 1, cate=cate)
                axs[5].set_title(f'latent dynamics to standard stim \n (long) {label_name}')

            except: pass

    def oddball(self, axs_all):
        for cate, axs in zip([[-1],[1]], axs_all):
            label_name = self.label_names[str(cate[0])] if len(cate)==1 else 'all'
            try:

                self.plot_oddball(axs[0], 0, cate=cate)
                axs[0].set_title(f'response to oddball interval \n (short standard) {label_name}')

                self.plot_oddball(axs[1], 1, cate=cate)
                axs[1].set_title(f'response to oddball interval \n (long standard) {label_name}')

                self.plot_oddball_sync(axs[2], 0, cate=cate)
                axs[2].set_title(f'synchronization to oddball interval \n (short standard) {label_name}')

                self.plot_oddball_sync(axs[3], 1, cate=cate)
                axs[3].set_title(f'synchronization to oddball interval \n (long standard) {label_name}')

                self.plot_oddball_latent(axs[4], 0, cate=cate)
                axs[4].set_title(f'latent dynamics to oddball interval \n (short standard) {label_name}')

                self.plot_oddball_latent(axs[5], 1, cate=cate)
                axs[5].set_title(f'latent dynamics to oddball interval \n (long standard) {label_name}')

            except: pass

    def block(self, axs_all):
        for cate, axs in zip([[-1],[1]], axs_all):
            label_name = self.label_names[str(cate[0])] if len(cate)==1 else 'all'
            try:

                self.plot_block_transition(axs[0], 0, cate=cate)
                axs[0].set_title(f'response to block transition \n (short to long) {label_name}')

                self.plot_block_transition(axs[1], 1, cate=cate)
                axs[1].set_title(f'response to block transition \n (long to short) {label_name}')

                self.plot_block_transition_latent(axs[2], 0, cate=cate)
                axs[2].set_title(f'latent dynamics to block transition \n (short to long) {label_name}')

                self.plot_block_transition_latent(axs[3], 1, cate=cate)
                axs[3].set_title(f'latent dynamics to block transition \n (long to short) {label_name}')

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
                axs[ 6].set_title(f'response to standard interval \n (short) {label_name}')
                axs[ 7].set_title(f'response to standard interval \n (long) {label_name}')
                axs[ 8].set_title(f'response to oddball interval \n (short standard) {label_name}')
                axs[ 9].set_title(f'response to oddball interval \n (long standard) {label_name}')
                axs[10].set_title(f'response to block transition \n (short to long) {label_name}')
                axs[11].set_title(f'response to block transition \n (long to short) {label_name}')
                axs[12].set_title(f'response to standard interval \n (short) {label_name}')
                axs[13].set_title(f'response to standard interval \n (long) {label_name}')
                axs[14].set_title(f'response to oddball interval \n (short standard) {label_name}')
                axs[15].set_title(f'response to oddball interval \n (long standard) {label_name}')
                axs[16].set_title(f'response to block transition \n (short to long) {label_name}')
                axs[17].set_title(f'response to block transition \n (long to short) {label_name}')

            except: pass

    def categorization_features(self, axs_all):
        for cate, axs in zip([-1,1,2], axs_all):
            try:

                self.plot_categorization_features(axs)
                for ci in range(len(axs[3])):
                    axs[3][ci].set_title(f'response comparison with epoches and oddball interval for cluster #{ci} \n (short standard and short oddball)')
                for ci in range(len(axs[4])):
                    axs[4][ci].set_title(f'response comparison with epoches and oddball interval for cluster #{ci} \n (long standard and long oddball)')

            except: pass

    def sorted_heatmaps_standard(self, axs_all):
        for norm_mode, axs in zip(['none', 'minmax', 'share'], axs_all):
            try:

                self.plot_sorted_heatmaps_standard(axs[0], norm_mode, 0)
                axs[0][0].set_title(f'response to short standard interval \n (normalized with {norm_mode})')
                axs[0][1].set_title(f'response to long standard interval \n (normalized with {norm_mode})')
                axs[0][2].set_title(f'response to standard interval \n (short minors long, normalized with {norm_mode})')
                axs[0][3].set_title('significance test across trials')

                self.plot_sorted_heatmaps_standard(axs[1], norm_mode, 1)
                axs[1][0].set_title(f'response to long standard interval \n (normalized with {norm_mode})')
                axs[1][1].set_title(f'response to short standard interval \n (normalized with {norm_mode})')
                axs[1][2].set_title(f'response to standard interval \n (long minors short, normalized with {norm_mode})')
                axs[1][3].set_title('significance test across trials')

            except: pass

    def temporal_scaling(self, axs_all):
        for norm_mode, axs in zip(['none', 'minmax', 'share'], axs_all):
            try:

                self.plot_temporal_scaling(axs, norm_mode)
                axs[0].set_title(f'response to short standard interval \n (normalized with {norm_mode})')
                axs[1].set_title(f'response to short standard interval \n (scaled, normalized with {norm_mode})')
                axs[2].set_title(f'response to long standard interval \n (normalized with {norm_mode})')

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
