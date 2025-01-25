#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from modules.Alignment import run_get_stim_response
from modeling.clustering import clustering_neu_response_mode
from modeling.clustering import get_mean_sem_cluster
from utils import get_norm01_params
from utils import get_odd_stim_prepost_idx
from utils import exclude_odd_stim
from utils import get_mean_sem
from utils import get_multi_sess_neu_trial_average
from utils import get_neu_trial
from utils import get_frame_idx_from_time
from utils import get_block_1st_idx
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
        self.r_frames = int(150*timescale)
        self.list_labels = list_labels
        self.alignment = run_get_stim_response(
                list_neural_trials, self.l_frames, self.r_frames, expected='none')
        self.list_stim_labels = self.alignment['list_stim_labels']
        self.list_odd_idx = [
            get_odd_stim_prepost_idx(sl) for sl in self.list_stim_labels]
        self.list_block_start = [get_block_1st_idx(sl, 3) for sl in self.list_stim_labels]
        self.list_significance = list_significance
        self.bin_win = [500,2500]
        self.bin_num = 4
        self.n_clusters = 6
        self.max_clusters = 20
        self.d_latent = 3
        self.xlim = [-4000,3000]

    def plot_standard(self, ax, standard, cate=None, roi_id=None):
        neu_mean = []
        neu_sem = []
        n_trials = 0
        # collect data.
        if 0 in standard:
            [[color0, color1, color2, _],
             [neu_seq_short, _, stim_seq_short, stim_value_short],
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
             [neu_seq_long, _, stim_seq_long, stim_value_long],
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
                    color=color1, alpha=0.15, step='mid')
            self.plot_vol(
                ax, self.alignment['stim_time'],
                stim_value_short.reshape(1,-1), color1, upper, lower)
        if 1 in standard:
            for i in range(stim_seq_long.shape[0]):
                ax.fill_between(
                    stim_seq_long[i,:],
                    lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
                    color=color2, alpha=0.15, step='mid')
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
        ax.set_xlim(self.xlim)
        ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
        ax.set_xlabel('time since stim (ms)')
        add_legend(ax, [color1,color2], ['short','long'],
                   n_trials, n_neurons, 'upper right')
    
    def plot_standard_box(self, ax, cate=None, roi_id=None):
        win_base = [-self.bin_win[1],0]
        offsets = [0.0, 0.1]
        # collect data.
        [[color0, color1, color2, _],
         [neu_seq_short, _, stim_seq_short, stim_value_short],
         [n_trials_short, n_neurons]] = get_neu_trial(
            self.alignment, self.list_labels, self.list_significance,
            [exclude_odd_stim(sl) for sl in self.list_stim_labels],
            trial_param=[[2,3,4,5], [0], None, None, [0], [0]],
            cate=cate, roi_id=roi_id)
        [[color0, color1, color2, _],
         [neu_seq_long, _, stim_seq_long, stim_value_long],
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
                   n_trials_short+n_trials_long, n_neurons, 'upper right')

    def plot_standard_latent(self, ax, standard, cate=None):
        # collect data.
        [[color0, color1, color2, _],
         [neu_x, _, stim_seq, _],
         [n_trials, n_neurons]] = get_neu_trial(
            self.alignment, self.list_labels, self.list_significance,
            [exclude_odd_stim(sl) for sl in self.list_stim_labels],
            trial_param=[[2,3,4,5], [standard], None, None, [0], [0]],
            cate=cate, roi_id=None)
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
        add_legend(ax, [color0,color1,color2], ['post interval stim','short','long'],
                   n_trials, n_neurons, 'upper right', dim=3)
    
    def plot_oddball(self, ax, standard, cate=None, roi_id=None):
        # collect data.
        [_, color1, color2, _], [neu_seq, _, stim_seq, stim_value], [n_trials, n_neurons] = get_neu_trial(
            self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
            trial_idx=[l[3-standard] for l in self.list_odd_idx],
            cate=cate, roi_id=None)
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
                color=c_stim[i], alpha=0.15, step='mid')
        self.plot_vol(
            ax, self.alignment['stim_time'],
            stim_value.reshape(1,-1), [color1, color2][standard], upper, lower)
        # plot neural traces.
        self.plot_mean_sem(
            ax, self.alignment['neu_time'],
            neu_mean, neu_sem, [color1, color2][standard], None)
        # adjust layout.
        adjust_layout_neu(ax)
        ax.set_xlim(self.xlim)
        ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
        ax.set_xlabel('time since stim after oddball interval (ms)')
        add_legend(ax, [color1,color2], ['post short interval stim','post long interval stim'],
                   n_trials, n_neurons, 'upper right')
    
    def plot_oddball_latent(self, ax, standard, cate=None):
        # collect data.
        [[color0, color1, color2, _],
         [neu_x, _, stim_seq, _],
         [n_trials, n_neurons]] = get_neu_trial(
            self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
            trial_idx=[l[3-standard] for l in self.list_odd_idx],
            cate=cate, roi_id=None)
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
                   n_trials, n_neurons, 'upper right', dim=3)
    
    def plot_block_transition(self, ax, standard, cate=None):
        # collect data.
        [_, color1, color2, _], [neu_seq, _, stim_seq, stim_value], [n_trials, n_neurons] = get_neu_trial(
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
                color=c_stim[i], alpha=0.15, step='mid')
        self.plot_vol(
            ax, self.alignment['stim_time'],
            stim_value.reshape(1,-1), color1, upper, lower)
        # plot neural traces.
        self.plot_mean_sem(
            ax, self.alignment['neu_time'],
            neu_mean, neu_sem, [color1, color2][standard], None)
        # adjust layout.
        adjust_layout_neu(ax)
        ax.set_xlim(self.xlim)
        ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
        ax.set_xlabel('time since stim after interval block change (ms)')
        add_legend(
            ax, [color1, color2],
            ['post interval stim (short)', 'post interval stim (long)'],
            n_trials, n_neurons, 'upper left')
    
    def plot_block_transition_latent(self, ax, standard, cate=None):
        # collect data.
        [[color0, color1, color2, _],
         [neu_x, _, stim_seq, _],
         [n_trials, n_neurons]] = get_neu_trial(
            self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
            trial_idx=[l[1-standard] for l in self.list_block_start],
            cate=cate, roi_id=None)
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
                   n_trials, n_neurons, 'upper right', dim=3)
    
    def plot_oddball_sync(self, ax, standard, cate=None, roi_id=None):
        win_width = 200
        l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, 0, win_width)
        win_width = r_idx - l_idx
        # collect data.
        [_, color1, color2, _], [neu_seq, _, stim_seq, stim_value], [n_trials, n_neurons] = get_neu_trial(
            self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
            trial_idx=[l[3-standard] for l in self.list_odd_idx],
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
                color=c_stim[i], alpha=0.15, step='mid')
        self.plot_vol(
            ax, self.alignment['stim_time'],
            stim_value.reshape(1,-1), [color1, color2][standard], upper, lower)
        # plot neural traces.
        ax.plot(self.alignment['neu_time'][win_width:], sync, color=[color1, color2][standard])
        # adjust layout.
        adjust_layout_neu(ax)
        ax.set_xlim(self.xlim)
        ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
        ax.set_xlabel('time since stim after oddball interval (ms)')
        ax.set_ylabel('synchronization metric')
        add_legend(ax, [color1,color2], ['post short interval stim','post long interval stim'],
                   n_trials, n_neurons, 'upper right')
        
    def plot_cluster(self, axs, standard, cate=None):
        n_latents = 25
        # collect data.
        [color0, color1, color2, cmap], [neu_seq, _, _, _], _ = get_neu_trial(
            self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
            trial_param=[[2,3,4,5], [standard], None, None, [0], [0]],
            cate=cate, roi_id=None)
        colors = get_cmap_color(self.n_clusters, cmap=plt.cm.nipy_spectral_r)
        lbl = ['cluster #'+str(i) for i in range(self.n_clusters)]
        cmap_3d, _ = get_cmap_color(2, base_color='#2C2C2C', return_cmap=True)
        # construct features for clustering.
        def prepare_data():
            # get correlation matrix.
            cluster_corr = np.corrcoef(neu_seq)
            # extract features.
            model = PCA(n_components=n_latents)
            neu_x = model.fit_transform(cluster_corr)
            return neu_x
        neu_x = prepare_data()
        # run clustering on extracted correlation.
        def run_clustering():
            metrics, cluster_id = clustering_neu_response_mode(
                neu_x, self.n_clusters, self.max_clusters)
            return metrics, cluster_id
        metrics, cluster_id = run_clustering()
        # plot basic clustering results.
        def plot_info(axs):
            self.plot_cluster_info(
                axs, colors, cmap, neu_seq, neu_x,
                self.n_clusters, self.max_clusters,
                metrics, cluster_id)
        plot_info(axs[:5])
        # plot standard response.
        def plot_standard(ax):
            # collect data.
            _, [neu_seq, _, stim_seq, stim_value], [n_trials, n_neurons] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_param=[[2,3,4,5], [standard], None, None, [0], [0]],
                cate=cate, roi_id=None)
            c_stim = [[color1, color2][standard]] * stim_seq.shape[-2]
            # get response within cluster.
            neu_mean, neu_sem = get_mean_sem_cluster(neu_seq, cluster_id)
            # plot results.
            norm_params = [get_norm01_params(neu_mean[i,:]) for i in range(neu_mean.shape[0])]
            self.plot_cluster_mean_sem(
                ax, neu_mean, neu_sem, norm_params, stim_seq, c_stim, colors)
            # adjust layout.
            ax.set_xlim(self.xlim)
            ax.set_xlabel('time since stim (ms)')
            add_legend(ax, [c_stim[0]]+colors, ['stim']+lbl, n_trials, n_neurons, 'upper left')
        plot_standard(axs[5])
        # plot oddball response.
        def plot_oddball(ax):
            # collect data.
            _, [neu_seq, _, stim_seq, stim_value], [n_trials, n_neurons] = get_neu_trial(
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
                ax, neu_mean, neu_sem, norm_params, stim_seq, c_stim, colors)
            # adjust layout.
            ax.set_xlim(self.xlim)
            ax.set_xlabel('time since stim after oddball interval (ms)')
            add_legend(
                ax, [color1, color2]+colors,
                ['post interval stim (short)', 'post interval stim (long)']+lbl,
                n_trials, n_neurons, 'upper left')
        plot_oddball(axs[6])
        # plot transition response.
        def plot_block_transition(ax):
            # collect data.
            _, [neu_seq, _, stim_seq, stim_value], [n_trials, n_neurons] = get_neu_trial(
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
                ax, neu_mean, neu_sem, norm_params, stim_seq, c_stim, colors)
            # adjust layout.
            ax.set_xlim(self.xlim)
            ax.set_xlabel('time since stim after interval block change (ms)')
            add_legend(
                ax, [color1, color2]+colors,
                ['post interval stim (short)', 'post interval stim (long)']+lbl,
                n_trials, n_neurons, 'upper left')
        plot_block_transition(axs[7])
        # plot opposite standard response.
        def plot_oppo_standard(ax):
            # collect data.
            _, [neu_seq, _, stim_seq, stim_value], [n_trials, n_neurons] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_param=[[2,3,4,5], [1-standard], None, None, [0], [0]],
                cate=cate, roi_id=None)
            c_stim = [[color1, color2][1-standard]] * stim_seq.shape[-2]
            # get response within cluster.
            neu_mean, neu_sem = get_mean_sem_cluster(neu_seq, cluster_id)
            # plot results.
            norm_params = [get_norm01_params(neu_mean[i,:]) for i in range(neu_mean.shape[0])]
            self.plot_cluster_mean_sem(
                ax, neu_mean, neu_sem, norm_params, stim_seq, c_stim, colors)
            # adjust layout.
            ax.set_xlim(self.xlim)
            ax.set_xlabel('time since stim (ms)')
            add_legend(ax, [c_stim[0]]+colors, ['stim']+lbl, n_trials, n_neurons, 'upper left')
        plot_oppo_standard(axs[8])
        # plot latent dyanmics of standard response.
        def plot_latent_standard(ax):
            # collect data.
            _, [neu_seq, _, stim_seq, stim_value], [n_trials, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_param=[[2,3,4,5], [standard], None, None, [0], [0]],
                cate=cate, roi_id=None)
            c_stim = [[color1, color2][standard]] * stim_seq.shape[-2]
            for i in range(self.n_clusters):
                c_neu = get_cmap_color(
                    neu_seq.shape[1], base_color=colors[i])
                # get neural data within cluster.
                neu_x = neu_seq[cluster_id==i].copy()
                # fit model.
                model = TSNE(n_components=self.d_latent)
                neu_z = model.fit_transform(
                    neu_x.reshape(neu_x.shape[0],-1).T
                    ).T.reshape(self.d_latent, -1)
                # plot dynamics.
                self.plot_3d_latent_dynamics(ax, neu_z, stim_seq, c_neu, c_stim)
            # adjust layout.
            adjust_layout_3d_latent(
                ax, neu_z, cmap_3d, self.alignment['neu_time'],
                'time since stim (ms)')
            add_legend(ax, [c_stim[0]]+colors, ['stim']+lbl, n_trials, neu_x.shape[0], 'upper left', dim=3)
        plot_latent_standard(axs[9])
        # plot latent dyanmics of oddball response.
        def plot_latent_oddball(ax):
            # collect data.
            _, [neu_seq, _, stim_seq, stim_value], [n_trials, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_idx=[l[3-standard] for l in self.list_odd_idx],
                cate=cate, roi_id=None)
            c_idx = int(stim_seq.shape[0]/2)
            c_stim = [[color1, color2][standard]] * stim_seq.shape[-2]
            c_stim[c_idx] = [color1, color2][1-standard]
            for i in range(self.n_clusters):
                c_neu = get_cmap_color(
                    neu_seq.shape[1], base_color=colors[i])
                # get neural data within cluster.
                neu_x = neu_seq[cluster_id==i].copy()
                # fit model.
                model = TSNE(n_components=self.d_latent)
                neu_z = model.fit_transform(
                    neu_x.reshape(neu_x.shape[0],-1).T
                    ).T.reshape(self.d_latent, -1)
                # plot dynamics.
                self.plot_3d_latent_dynamics(ax, neu_z, stim_seq, c_neu, c_stim)
            # adjust layout.
            adjust_layout_3d_latent(
                ax, neu_z, cmap_3d, self.alignment['neu_time'],
                'time since stim after oddball interval (ms)')
            add_legend(ax, [c_stim[0]]+colors, ['stim']+lbl, n_trials, neu_x.shape[0], 'upper left', dim=3)
        plot_latent_oddball(axs[10])
        # plot latent dyanmics of transition response.
        def plot_latent_block_transition(ax):
            # collect data.
            _, [neu_seq, _, stim_seq, stim_value], [n_trials, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_idx=[l[1-standard] for l in self.list_block_start],
                cate=cate, roi_id=None)
            c_idx = int(stim_seq.shape[0]/2)
            c_stim = [[color1, color2][standard]] * (c_idx+1)
            c_stim+= [[color1, color2][1-standard]] * (stim_seq.shape[0]-c_idx-1)
            for i in range(self.n_clusters):
                c_neu = get_cmap_color(
                    neu_seq.shape[1], base_color=colors[i])
                # get neural data within cluster.
                neu_x = neu_seq[cluster_id==i].copy()
                # fit model.
                model = TSNE(n_components=self.d_latent)
                neu_z = model.fit_transform(
                    neu_x.reshape(neu_x.shape[0],-1).T
                    ).T.reshape(self.d_latent, -1)
                # plot dynamics.
                self.plot_3d_latent_dynamics(ax, neu_z, stim_seq, c_neu, c_stim)
            # adjust layout.
            adjust_layout_3d_latent(
                ax, neu_z, cmap_3d, self.alignment['neu_time'],
                'time since stim after interval block change (ms)')
            add_legend(ax, [c_stim[0]]+colors, ['stim']+lbl, n_trials, neu_x.shape[0], 'upper left', dim=3)
        plot_latent_block_transition(axs[11])
        # plot latent dyanmics of opposite standard response.
        def plot_latent_oppo_standard(ax):
            # collect data.
            _, [neu_seq, _, stim_seq, stim_value], [n_trials, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_param=[[2,3,4,5], [1-standard], None, None, [0], [0]],
                cate=cate, roi_id=None)
            c_stim = [[color1, color2][1-standard]] * stim_seq.shape[-2]
            for i in range(self.n_clusters):
                c_neu = get_cmap_color(
                    neu_seq.shape[1], base_color=colors[i])
                # get neural data within cluster.
                neu_x = neu_seq[cluster_id==i].copy()
                # fit model.
                model = TSNE(n_components=self.d_latent)
                neu_z = model.fit_transform(
                    neu_x.reshape(neu_x.shape[0],-1).T
                    ).T.reshape(self.d_latent, -1)
                # plot dynamics.
                self.plot_3d_latent_dynamics(ax, neu_z, stim_seq, c_neu, c_stim)
            # adjust layout.
            adjust_layout_3d_latent(
                ax, neu_z, cmap_3d, self.alignment['neu_time'],
                'time since stim (ms)')
            add_legend(ax, [c_stim[0]]+colors, ['stim']+lbl, n_trials, neu_x.shape[0], 'upper left', dim=3)
        plot_latent_oppo_standard(axs[12])
        
# colors = ['#989A9C', '#A4CB9E', '#9DB4CE', '#EDA1A4', '#F9C08A']
class plotter_main(plotter_utils):
    def __init__(self, neural_trials, labels, significance, label_names):
        super().__init__(neural_trials, labels, significance)
        self.label_names = label_names
    
    def standard_exc(self, axs):
        cate = -1
        label_name = self.label_names[str(cate)]
        
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
        
    
    def standard_inh(self, axs):
        cate = 1
        label_name = self.label_names[str(cate)]
        
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
    

    def standard_heatmap(self, axs):
        try:
            win_sort = [-500, 1500]
            labels = np.concatenate(self.list_labels)
            sig = np.concatenate([self.list_significance[n]['r_oddball'] for n in range(self.n_sess)])
            [neu_short, _, _, _], _ = get_multi_sess_neu_trial_average(
                [exclude_odd_stim(sl) for sl in self.list_stim_labels],
                self.alignment['list_neu_seq'], self.alignment,
                trial_param=[[2,3,4,5], [0], None, None, [0], [0]])
            [neu_long, _, _, _], _ = get_multi_sess_neu_trial_average(
                [exclude_odd_stim(sl) for sl in self.list_stim_labels],
                self.alignment['list_neu_seq'], self.alignment,
                trial_param=[[2,3,4,5], [1], None, None, [0], [0]])
            for i in range(4):
                axs[i].set_xlabel('time since stim (ms)')
            
            self.plot_heatmap_neuron(axs[0], neu_short, self.alignment['neu_time'], neu_short, win_sort, labels, sig)
            axs[0].set_title('response to standard stim \n (short sorted by short)')

            self.plot_heatmap_neuron(axs[1], neu_long,  self.alignment['neu_time'], neu_short, win_sort, labels, sig)
            axs[1].set_title('response to standard stim \n (long sorted by short)')

            self.plot_heatmap_neuron(axs[2], neu_short, self.alignment['neu_time'], neu_long,  win_sort, labels, sig)
            axs[2].set_title('response to standard stim \n (short sorted by long)')

            self.plot_heatmap_neuron(axs[3], neu_long,  self.alignment['neu_time'], neu_long,  win_sort, labels, sig)
            axs[3].set_title('response to standard stim \n (long sorted by long)')
                
        except: pass
    
    def oddball_exc(self, axs):
        cate = -1
        label_name = self.label_names[str(cate)]
        try:
            
            self.plot_oddball(axs[0], 0, cate=cate)
            axs[0].set_title(f'response to oddball interval \n (short standard) {label_name}')
            
            self.plot_oddball(axs[1], 1, cate=cate)
            axs[1].set_title(f'response to oddball interval \n (long standard) {label_name}')
            
            self.plot_block_transition(axs[2], 0, cate=cate)
            axs[2].set_title(f'response to block transition \n (short to long) {label_name}')
            
            self.plot_block_transition(axs[3], 1, cate=cate)
            axs[3].set_title(f'response to block transition \n (long to short) {label_name}')
            
            self.plot_oddball_latent(axs[4], 0, cate=cate)
            axs[4].set_title(f'latent dynamics to oddball interval \n (short standard) {label_name}')
            
            self.plot_oddball_latent(axs[5], 1, cate=cate)
            axs[5].set_title(f'latent dynamics to oddball interval \n (long standard) {label_name}')
            
            self.plot_block_transition_latent(axs[6], 0, cate=cate)
            axs[6].set_title(f'latent dynamics to block transition \n (short to long) {label_name}')
            
            self.plot_block_transition_latent(axs[7], 1, cate=cate)
            axs[7].set_title(f'latent dynamics to block transition \n (long to short) {label_name}')
            
            self.plot_oddball_sync(axs[8], 0, cate=cate)
            axs[8].set_title(f'synchronization to oddball interval \n (short standard) {label_name}')
            
            self.plot_oddball_sync(axs[9], 1, cate=cate)
            axs[9].set_title(f'synchronization to oddball interval \n (long standard) {label_name}')

        except: pass

    def oddball_inh(self, axs):
        cate = 1
        label_name = self.label_names[str(cate)]
        try:
            
            self.plot_oddball(axs[0], 0, cate=cate)
            axs[0].set_title(f'response to oddball interval \n (short standard) {label_name}')
            
            self.plot_oddball(axs[1], 1, cate=cate)
            axs[1].set_title(f'response to oddball interval \n (long standard) {label_name}')
            
            self.plot_block_transition(axs[2], 0, cate=cate)
            axs[2].set_title(f'response to block transition \n (short to long) {label_name}')
            
            self.plot_block_transition(axs[3], 1, cate=cate)
            axs[3].set_title(f'response to block transition \n (long to short) {label_name}')
            
            self.plot_oddball_latent(axs[4], 0, cate=cate)
            axs[4].set_title(f'latent dynamics to oddball interval \n (short standard) {label_name}')
            
            self.plot_oddball_latent(axs[5], 1, cate=cate)
            axs[5].set_title(f'latent dynamics to oddball interval \n (long standard) {label_name}')
            
            self.plot_block_transition_latent(axs[6], 0, cate=cate)
            axs[6].set_title(f'latent dynamics to block transition \n (short to long) {label_name}')
            
            self.plot_block_transition_latent(axs[7], 1, cate=cate)
            axs[7].set_title(f'latent dynamics to block transition \n (long to short) {label_name}')
            
            self.plot_oddball_sync(axs[8], 0, cate=cate)
            axs[8].set_title(f'synchronization to oddball interval \n (short standard) {label_name}')
            
            self.plot_oddball_sync(axs[9], 1, cate=cate)
            axs[9].set_title(f'synchronization to oddball interval \n (long standard) {label_name}')

        except: pass

    def cluster_exc(self, axs):
        cate = -1
        label_name = self.label_names[str(cate)]
        try:
            self.plot_cluster(axs[0], 0, cate=cate)
            axs[0][ 0].set_title(f'sorted correlation matrix \n (short) {label_name}')
            axs[0][ 1].set_title(f'cross cluster correlation \n (short) {label_name}')
            axs[0][ 2].set_title(f'clustering evaluation metrics \n (short) {label_name}')
            axs[0][ 3].set_title(f'cluster fraction \n (short) {label_name}')
            axs[0][ 4].set_title(f'hierarchical clustering dendrogram \n (short) {label_name}')
            axs[0][ 5].set_title(f'response to standard interval \n (short) {label_name}')
            axs[0][ 6].set_title(f'response to oddball interval \n (short) {label_name}')
            axs[0][ 7].set_title(f'response to block transition \n (short to long) {label_name}')
            axs[0][ 8].set_title(f'response to standard interval \n (long) {label_name}')
            axs[0][ 9].set_title(f'latent dynamics to standard interval \n (short) {label_name}')
            axs[0][10].set_title(f'latent dynamics to oddball interval \n (short) {label_name}')
            axs[0][11].set_title(f'latent dynamics to block transition \n (short to long) {label_name}')
            axs[0][12].set_title(f'latent dynamics to standard interval \n (long) {label_name}')
            
            self.plot_cluster(axs[1], 1, cate=cate)
            axs[1][ 0].set_title(f'sorted correlation matrix \n (long) {label_name}')
            axs[1][ 1].set_title(f'cross cluster correlation \n (long) {label_name}')
            axs[1][ 2].set_title(f'clustering evaluation metrics \n (long) {label_name}')
            axs[1][ 3].set_title(f'cluster fraction \n (long) {label_name}')
            axs[1][ 4].set_title(f'hierarchical clustering dendrogram \n (long) {label_name}')
            axs[1][ 5].set_title(f'response to standard interval \n (long) {label_name}')
            axs[1][ 6].set_title(f'response to oddball interval \n (long) {label_name}')
            axs[1][ 7].set_title(f'response to block transition \n (long to short) {label_name}')
            axs[1][ 8].set_title(f'response to standard interval \n (short) {label_name}')
            axs[1][ 9].set_title(f'latent dynamics to standard interval \n (long) {label_name}')
            axs[1][10].set_title(f'latent dynamics to oddball interval \n (long) {label_name}')
            axs[1][11].set_title(f'latent dynamics to block transition \n (long to short) {label_name}')
            axs[1][12].set_title(f'latent dynamics to standard interval \n (short) {label_name}')
        
        except: pass
    
    def cluster_inh(self, axs):
        cate = 1
        label_name = self.label_names[str(cate)]
        try:
            self.plot_cluster(axs[0], 0, cate=cate)
            axs[0][ 0].set_title(f'sorted correlation matrix \n (short) {label_name}')
            axs[0][ 1].set_title(f'cross cluster correlation \n (short) {label_name}')
            axs[0][ 2].set_title(f'clustering evaluation metrics \n (short) {label_name}')
            axs[0][ 3].set_title(f'cluster fraction \n (short) {label_name}')
            axs[0][ 4].set_title(f'hierarchical clustering dendrogram \n (short) {label_name}')
            axs[0][ 5].set_title(f'response to standard interval \n (short) {label_name}')
            axs[0][ 6].set_title(f'response to oddball interval \n (short) {label_name}')
            axs[0][ 7].set_title(f'response to block transition \n (short to long) {label_name}')
            axs[0][ 8].set_title(f'response to standard interval \n (long) {label_name}')
            axs[0][ 9].set_title(f'latent dynamics to standard interval \n (short) {label_name}')
            axs[0][10].set_title(f'latent dynamics to oddball interval \n (short) {label_name}')
            axs[0][11].set_title(f'latent dynamics to block transition \n (short to long) {label_name}')
            axs[0][12].set_title(f'latent dynamics to standard interval \n (long) {label_name}')
            
            self.plot_cluster(axs[1], 1, cate=cate)
            axs[1][ 0].set_title(f'sorted correlation matrix \n (long) {label_name}')
            axs[1][ 1].set_title(f'cross cluster correlation \n (long) {label_name}')
            axs[1][ 2].set_title(f'clustering evaluation metrics \n (long) {label_name}')
            axs[1][ 3].set_title(f'cluster fraction \n (long) {label_name}')
            axs[1][ 4].set_title(f'hierarchical clustering dendrogram \n (long) {label_name}')
            axs[1][ 5].set_title(f'response to standard interval \n (long) {label_name}')
            axs[1][ 6].set_title(f'response to oddball interval \n (long) {label_name}')
            axs[1][ 7].set_title(f'response to block transition \n (long to short) {label_name}')
            axs[1][ 8].set_title(f'response to standard interval \n (short) {label_name}')
            axs[1][ 9].set_title(f'latent dynamics to standard interval \n (long) {label_name}')
            axs[1][10].set_title(f'latent dynamics to oddball interval \n (long) {label_name}')
            axs[1][11].set_title(f'latent dynamics to block transition \n (long to short) {label_name}')
            axs[1][12].set_title(f'latent dynamics to standard interval \n (short) {label_name}')
        
        except: pass