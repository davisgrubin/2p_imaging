#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from modules.Alignment import run_get_stim_response
from modeling.clustering import clustering_neu_response_mode
from modeling.clustering import get_mean_sem_cluster
from modeling.generative import run_glm_multi_sess
from utils import get_norm01_params
from utils import get_odd_stim_prepost_idx
from utils import exclude_odd_stim
from utils import get_mean_sem
from utils import get_neu_trial
from utils import get_frame_idx_from_time
from utils import get_block_1st_idx
from utils import get_isi_bin_neu
from utils import get_expect_interval
from utils import get_neu_sync
from utils import get_cmap_color
from utils import adjust_layout_neu
from utils import adjust_layout_3d_latent
from utils import add_legend
from utils import utils_basic

# fig, ax = plt.subplots(1, 1, figsize=(6, 6))
# fig, axs = plt.subplots(1, 5, figsize=(30, 6))
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
        self.expect = np.array([get_expect_interval(sl) for sl in self.list_stim_labels]).reshape(-1)
        self.list_block_start = [get_block_1st_idx(sl, 3) for sl in self.list_stim_labels]
        self.list_significance = list_significance
        self.bin_win = [450,2550]
        self.bin_num = 2
        self.n_clusters = 5
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

    def run_glm(self, cate):
        # define kernel window.
        kernel_win = [-500,1500]
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
        colors = get_cmap_color(self.n_clusters, cmap=plt.cm.nipy_spectral)
        lbl = ['cluster #'+str(i) for i in range(self.n_clusters)]
        # plot basic clustering results.
        def plot_info(axs):
            # collect data.
            [color0, color1, color2, cmap], [neu_seq, _, _, _], [neu_labels, _], _ = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_param=[[2,3,4,5], [0], None, None, [0], [0]],
                cate=cate, roi_id=None)
            # plot info.
            self.plot_cluster_info(
                axs, colors, cmap, neu_seq,
                self.n_clusters, self.max_clusters,
                metrics, cluster_id, neu_labels, self.label_names, cate)
        # plot bin normalized interval tracking for all clusters.
        def plot_interval_norm(ax):
            # collect data.
            [_, [neu_seq, stim_seq, stim_value, pre_isi], _, [n_trials, n_neurons]] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_param=[None, None, None, None, None, [0]],
                mean_sem=False,
                cate=cate, roi_id=None)
            # bin data based on isi.
            [bins, _, bin_neu_seq, _, _, bin_stim_seq, _] = get_isi_bin_neu(
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
    
    def plot_cluster_latents(self, axs, cate=None):
        _, cluster_id = self.run_clustering(cate)
        self.plot_cluster_bin_3d_latents(axs, cluster_id, cate=cate)

    def plot_sorted_heatmaps(self, axs, standard):
        cate = [-1,1]
        win_sort = [-500, 500]
        xlim = [-3000, 4000]
        l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, xlim[0], xlim[1])
        # collect data.
        neu_x = []
        stim_x = []
        neu_time = self.alignment['neu_time'][l_idx:r_idx]
        # standard.
        _, [neu_seq, _, stim_seq, _], [neu_labels, neu_sig], _ = get_neu_trial(
            self.alignment, self.list_labels, self.list_significance,
            [exclude_odd_stim(sl) for sl in self.list_stim_labels],
            trial_param=[[2,3,4,5], [standard], None, None, [0], [0]],
            cate=cate, roi_id=None)
        neu_seq = neu_seq[:,l_idx:r_idx]
        neu_x.append(neu_seq)
        stim_x.append(stim_seq)
        # oddballs.
        _, [neu_seq, _, stim_seq, _], _, _ = get_neu_trial(
            self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
            trial_idx=[l[1-standard] for l in self.list_odd_idx],
            cate=cate, roi_id=None)
        neu_seq = neu_seq[:,l_idx:r_idx]
        neu_x.append(neu_seq)
        stim_x.append(stim_seq)
        # opposite standard.
        _, [neu_seq, _, stim_seq, _], _, _ = get_neu_trial(
            self.alignment, self.list_labels, self.list_significance,
            [exclude_odd_stim(sl) for sl in self.list_stim_labels],
            trial_param=[[2,3,4,5], [1-standard], None, None, [0], [0]],
            cate=cate, roi_id=None)
        neu_seq = neu_seq[:,l_idx:r_idx]
        neu_x.append(neu_seq)
        stim_x.append(stim_seq)
        # opposite oddballs.
        _, [neu_seq, _, stim_seq, _], _, _ = get_neu_trial(
            self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
            trial_idx=[l[standard] for l in self.list_odd_idx],
            cate=cate, roi_id=None)
        neu_seq = neu_seq[:,l_idx:r_idx]
        neu_x.append(neu_seq)
        stim_x.append(stim_seq)
        # plot heatmaps.
        for bi in range(len(neu_x)):
            self.plot_heatmap_neuron(
                axs[bi], neu_x[bi], neu_time, neu_x[0], win_sort, neu_labels, neu_sig)
            axs[bi].set_xlabel('time since stim (ms)')
            # add stimulus line.
            c_idx = int(stim_seq.shape[0]/2)
            xlines = [self.alignment['neu_time'][np.searchsorted(self.alignment['neu_time'], t)]
                      for t in [stim_x[bi][c_idx+i,0]
                                for i in [-2,-1,0,1,2]]]
            for xl in xlines:
                if xl>neu_time[0] and xl<neu_time[-1]:
                    axs[bi].axvline(xl, color='black', lw=1, linestyle='--')
                    
    def plot_glm(self, axs, cate):
        kernel_time, kernel_all, exp_var_all = self.run_glm(cate)
        _, cluster_id = clustering_neu_response_mode(
            kernel_all, self.n_clusters, self.max_clusters)
        # collect data.
        colors = get_cmap_color(self.n_clusters, cmap=plt.cm.nipy_spectral)
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
            [bins, _, bin_neu_seq, _, _, bin_stim_seq, _] = get_isi_bin_neu(
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

    def cluster_latents(self, axs_all):
        for cate, axs in zip([[-1,1]], axs_all):
            label_name = self.label_names[str(cate[0])] if len(cate)==1 else 'all'
            try:
                
                self.plot_cluster_latents(axs, cate=cate)
                for i in range(self.n_clusters):
                    axs[i].set_title(f'latent dynamics with binned interval for cluster # {i} \n {label_name}')

            except: pass

    def sorted_heatmaps(self, axs_all):
        for cate, axs in zip([[-1,1]], axs_all):
            self.plot_sorted_heatmaps(axs[0], 0)
            axs[0][0].set_title('response to standard interval \n (short, sorted by short standard)')
            axs[0][1].set_title('response to odball interval \n (long, sorted by short standard)')
            axs[0][2].set_title('response to standard interval \n (long, sorted by short standard)')
            axs[0][3].set_title('response to odball interval \n (short, sorted by short standard)')
            
            self.plot_sorted_heatmaps(axs[1], 1)
            axs[1][0].set_title('response to standard interval \n (long, sorted by long standard)')
            axs[1][1].set_title('response to odball interval \n (short, sorted by long standard)')
            axs[1][2].set_title('response to standard interval \n (short, sorted by long standard)')
            axs[1][3].set_title('response to odball interval \n (long, sorted by long standard)')

            try:

                0
            except: pass
    
    def glm(self, axs_all):
        for cate, axs in zip([[-1,1]], axs_all):
            label_name = self.label_names[str(cate[0])] if len(cate)==1 else 'all'
            try:
                
                self.plot_glm(axs, cate=cate)
                axs[0].set_title(f'GLM kernel weights heatmap \n {label_name}')
                axs[1].set_title(f'clustered GLM kernel weights \n {label_name}')
                axs[2].set_title(f'clustered GLM kernel weights \n {label_name}')
                axs[3].set_title(f'time normalized reseponse to preceeding interval with bins \n {label_name}')
                
            except: pass