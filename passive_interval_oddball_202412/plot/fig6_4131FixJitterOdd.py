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
from utils import get_neu_trial
from utils import get_frame_idx_from_time
from utils import get_isi_bin_neu
from utils import get_expect_interval
from utils import get_block_1st_idx
from utils import get_neu_sync
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
        self.alignment = run_get_stim_response(
                list_neural_trials, self.l_frames, self.r_frames, expected='none')
        self.list_stim_labels = self.alignment['list_stim_labels']
        self.list_odd_idx = [
            get_odd_stim_prepost_idx(sl) for sl in self.list_stim_labels]
        self.expect = np.array(np.mean([get_expect_interval(sl)[0] for sl in self.list_stim_labels]))
        self.list_block_start = [get_block_1st_idx(sl, 3) for sl in self.list_stim_labels]
        self.list_significance = list_significance
        self.bin_win = [500,2500]
        self.bin_num = 4
        self.n_clusters = 4
        self.max_clusters = 20
        self.d_latent = 3

    def plot_oddball_fix(self, ax, oddball, cate=None, roi_id=None):
        xlim = [-2000,5000]
        # collect data.
        neu_mean = []
        neu_sem = []
        n_trials = 0
        if 0 in oddball:
            [color0, color1, color2, _], [neu_short, _, stim_seq_short, stim_value_short], [nt, n_neurons] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_idx=[l[0] for l in self.list_odd_idx],
                trial_param=[None, None, [0], None, [0], [0]],
                cate=cate, roi_id=None)
            mean_short, sem_short = get_mean_sem(neu_short)
            neu_mean.append(mean_short)
            neu_sem.append(sem_short)
            n_trials += nt
        if 1 in oddball:
            [color0, color1, color2, _], [neu_long, _, stim_seq_long, stim_value_long], [nt, n_neurons] = get_neu_trial(
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
            [color0, color1, color2, _], [neu_short, _, stim_seq_short, stim_value_short], [nt, n_neurons] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_idx=[l[0] for l in self.list_odd_idx],
                trial_param=[None, None, [0], None, [0], [0]],
                cate=cate, roi_id=None)
            sync_short = get_neu_sync(neu_short, win_width)
            neu.append(sync_short)
            n_trials += nt
        if 1 in oddball:
            [color0, color1, color2, _], [neu_long, _, stim_seq_long, stim_value_long], [nt, n_neurons] = get_neu_trial(
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
         [neu_x, _, stim_seq, stim_value],
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
        [color0, color1, color2, _], [neu_seq, stim_seq, stim_value, pre_isi], [n_trials, n_neurons] = get_neu_trial(
            self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
            trial_idx=[l[oddball] for l in self.list_odd_idx],
            trial_param=[None, None, [1], None, [0], [0]],
            mean_sem=False,
            cate=cate, roi_id=roi_id)
        colors = get_cmap_color(self.bin_num, base_color=[color1, color2][oddball])
        # bin data based on isi.
        [bins, bin_center, _, bin_neu_mean, bin_neu_sem, bin_stim_seq, bin_stim_value] = get_isi_bin_neu(
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
        [color0, color1, color2, _], [neu_seq, stim_seq, stim_value, pre_isi], [n_trials, n_neurons] = get_neu_trial(
            self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
            trial_idx=[l[oddball] for l in self.list_odd_idx],
            trial_param=[None, None, [1], None, [0], [0]],
            mean_sem=False,
            cate=cate, roi_id=roi_id)
        colors = get_cmap_color(self.bin_num, base_color=[color1, color2][oddball])
        # bin data based on isi.
        [bins, bin_center, bin_neu_seq, _, _, bin_stim_seq, _] = get_isi_bin_neu(
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
    
    def plot_oddball_share(self, ax, oddball, cate=None, roi_id=None):
        isi_win = 100
        xlim = [-3000,4000]
        # collect data.
        n_trials = 0
        # fix.
        [color0, color1, color2, _], [neu_fix, _, stim_seq_fix, stim_value_fix], [nt, n_neurons] = get_neu_trial(
            self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
            trial_idx=[l[oddball] for l in self.list_odd_idx],
            trial_param=[None, None, [0], None, [0], [0]],
            cate=cate, roi_id=None)
        n_trials += nt
        mean_fix, sem_fix = get_mean_sem(neu_fix)
        # jitter.
        _, [neu_jitter, _, stim_value_jitter, pre_isi], _ = get_neu_trial(
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
        add_legend(ax, [c_fix,color0], ['fix', r'jitter within $\pm{}$ ms'.format(isi_win)],
                   n_trials, n_neurons, self.n_sess, 'upper left')
        
    def plot_cluster(self, axs, cate=None):
        n_latents = 25
        # collect data.
        [color0, color1, color2, cmap], [neu_seq, _, _, _], _ = get_neu_trial(
            self.alignment, self.list_labels, self.list_significance,
            [exclude_odd_stim(sl) for sl in self.list_stim_labels],
            trial_param=[[2,3,4,5], [0], [0], None, [0], [0]],
            cate=cate, roi_id=None)
        colors = get_cmap_color(self.n_clusters, cmap=plt.cm.nipy_spectral_r)
        lbl = ['cluster #'+str(i) for i in range(self.n_clusters)]
        cmap_3d, _ = get_cmap_color(2, base_color='#2C2C2C', return_cmap=True)
        # construct features for clustering.
        def prepare_data():
            win_eval = [-3000,2000]
            l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, win_eval[0], win_eval[1])
            # fix standard.
            _, [neu_1, _, _, _], _ = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance,
                [exclude_odd_stim(sl) for sl in self.list_stim_labels],
                trial_param=[[2,3,4,5], [0], [0], None, [0], [0]],
                cate=cate, roi_id=None)
            # jitter standard.
            _, [neu_2, _, _, _], _ = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance,
                [exclude_odd_stim(sl) for sl in self.list_stim_labels],
                trial_param=[[2,3,4,5], [0], [1], None, [0], [0]],
                cate=cate, roi_id=None)
            # fix short oddball.
            _, [neu_3, _, _, _], _ = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_idx=[l[2] for l in self.list_odd_idx],
                trial_param=[None, None, [0], None, [0], [0]],
                cate=cate, roi_id=None)
            # jitter short oddball.
            _, [neu_4, _, _, _], _ = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_idx=[l[2] for l in self.list_odd_idx],
                trial_param=[None, None, [1], None, [0], [0]],
                cate=cate, roi_id=None)
            # fix long oddball.
            _, [neu_5, _, _, _], _ = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_idx=[l[3] for l in self.list_odd_idx],
                trial_param=[None, None, [0], None, [0], [0]],
                cate=cate, roi_id=None)
            # jitter long oddball.
            _, [neu_6, _, _, _], _ = get_neu_trial(
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
        plot_info(axs[:3])
        # plot fix standard response.
        def plot_standard_fix(ax):
            xlim = [-4000, 3000]
            # collect data.
            _, [neu_seq, _, stim_seq, stim_value], [n_trials, n_neurons] = get_neu_trial(
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
                ax, neu_mean, neu_sem, norm_params, stim_seq, c_stim, colors)
            # adjust layout.
            ax.set_xlim(xlim)
            ax.set_xlabel('time since stim (ms)')
            add_legend(ax, [c_stim[0]]+colors, ['stim']+lbl, n_trials, n_neurons, self.n_sess, 'upper left')
        plot_standard_fix(axs[3])
        # plot jitter standard response.
        def plot_standard_jitter(ax):
            xlim = [-4000, 3000]
            # collect data.
            _, [neu_seq, stim_seq, stim_value, pre_isi], [n_trials, n_neurons] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance,
                [exclude_odd_stim(sl) for sl in self.list_stim_labels],
                trial_param=[[2,3,4,5], [0], [1], None, [0], [0]],
                mean_sem=False,
                cate=cate, roi_id=None)
            # bin data based on isi.
            [bins, bin_center, bin_neu_seq, _, _, bin_stim_seq, _] = get_isi_bin_neu(
                neu_seq, stim_seq, stim_value, pre_isi, self.bin_win, self.bin_num)
            # get response within cluster at each bin.
            cluster_bin_neu_mean = [get_mean_sem_cluster(neu, cluster_id)[0] for neu in bin_neu_seq]
            cluster_bin_neu_sem  = [get_mean_sem_cluster(neu, cluster_id)[1] for neu in bin_neu_seq]
            # organize into bin_num*n_clusters*time.
            cluster_bin_neu_mean = [np.expand_dims(neu, axis=0) for neu in cluster_bin_neu_mean]
            cluster_bin_neu_sem  = [np.expand_dims(neu, axis=0) for neu in cluster_bin_neu_sem]
            cluster_bin_neu_mean = np.concatenate(cluster_bin_neu_mean, axis=0)
            cluster_bin_neu_sem  = np.concatenate(cluster_bin_neu_sem, axis=0)
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
                    ax, cluster_bin_neu_mean[i,:,:], cluster_bin_neu_sem[i,:,:],
                    norm_params, stim_seq[i,:,:], [cs_all[i]]*stim_seq.shape[-2], cs[i])
            # adjust layout.
            ax.set_xlim(xlim)
            ax.set_xlabel('time since stim (ms)')
            add_legend(ax, cs_all, lbl, None, None, None, 'upper right')
        plot_standard_jitter(axs[4])
        # plot oddball response.
        def plot_oddball_fix(ax, oddball):
            xlim = [-2500,4500]
            # collect data.
            _, [neu_seq, _, stim_seq, stim_value], [n_trials, n_neurons] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_idx=[l[oddball] for l in self.list_odd_idx],
                trial_param=[None, None, [0], None, [0], [0]],
                cate=cate, roi_id=None)
            c_idx = int(stim_seq.shape[0]/2)
            c_stim = [color0] * stim_seq.shape[-2]
            c_stim[c_idx] = [color1, color2][oddball]
            # get response within cluster.
            neu_mean, neu_sem = get_mean_sem_cluster(neu_seq, cluster_id)
            # plot results.
            norm_params = [get_norm01_params(neu_mean[i,:]) for i in range(neu_mean.shape[0])]
            self.plot_cluster_mean_sem(
                ax, neu_mean, neu_sem, norm_params, stim_seq, c_stim, colors)
            # adjust layout.
            ax.set_xlim(xlim)
            ax.set_xlabel('time since stim before oddball interval (ms)')
            add_legend(
                ax, [color1, color2]+colors,
                ['pre oddball interval stim (short)', 'pre oddball interval stim (long)']+lbl,
                n_trials, n_neurons, self.n_sess, 'upper left')
        plot_oddball_fix(axs[5], 0)
        plot_oddball_fix(axs[7], 1)
        def plot_oddball_jitter(ax, oddball):
            xlim = [-2500,4500]
            # collect data.
            [_, color1, color2, _], [neu_seq, stim_seq, stim_value, pre_isi], [n_trials, n_neurons] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_idx=[l[oddball] for l in self.list_odd_idx],
                trial_param=[None, None, [1], None, [0], [0]],
                mean_sem=False,
                cate=cate, roi_id=None)
            # bin data based on isi.
            [bins, bin_center, bin_neu_seq, _, _, bin_stim_seq, _] = get_isi_bin_neu(
                neu_seq, stim_seq, stim_value, pre_isi, self.bin_win, self.bin_num)
            # get response within cluster at each bin.
            cluster_bin_neu_mean = [get_mean_sem_cluster(neu, cluster_id)[0] for neu in bin_neu_seq]
            cluster_bin_neu_sem  = [get_mean_sem_cluster(neu, cluster_id)[1] for neu in bin_neu_seq]
            # organize into bin_num*n_clusters*time.
            cluster_bin_neu_mean = [np.expand_dims(neu, axis=0) for neu in cluster_bin_neu_mean]
            cluster_bin_neu_sem  = [np.expand_dims(neu, axis=0) for neu in cluster_bin_neu_sem]
            cluster_bin_neu_mean = np.concatenate(cluster_bin_neu_mean, axis=0)
            cluster_bin_neu_sem  = np.concatenate(cluster_bin_neu_sem, axis=0)
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
                    ax, cluster_bin_neu_mean[i,:,:], cluster_bin_neu_sem[i,:,:],
                    norm_params, stim_seq[i,:,:], c_stim, cs[i])
            # adjust layout.
            ax.set_xlim(xlim)
            ax.set_xlabel('time since stim (ms)')
            add_legend(
                ax, [color1, color2]+cs_all,
                ['pre oddball interval stim (short)', 'pre oddball interval stim (long)']+lbl,
                n_trials, n_neurons, self.n_sess, 'upper left')
        plot_oddball_jitter(axs[6],0)
        plot_oddball_jitter(axs[8],1)
        
# colors = ['#989A9C', '#A4CB9E', '#9DB4CE', '#EDA1A4', '#F9C08A']
class plotter_main(plotter_utils):
    def __init__(self, neural_trials, labels, significance, label_names):
        super().__init__(neural_trials, labels, significance)
        self.label_names = label_names
    
    def oddball_fix_exc(self, axs):
        cate = [-1]
        label_name = self.label_names[str(cate[0])]
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
            
            self.plot_oddball_fix_latent(axs[7], 0, cate=cate)
            axs[7].set_title(f'latent dynamics to oddball interval \n (fix, long oddball) {label_name}')
            
        except: pass

    def oddball_fix_inh(self, axs):
        cate = [1]
        label_name = self.label_names[str(cate[0])]
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
            
            self.plot_oddball_fix_latent(axs[7], 0, cate=cate)
            axs[7].set_title(f'latent dynamics to oddball interval \n (fix, long oddball) {label_name}')

        except: pass
    
    def oddball_jitter_exc(self, axs):
        cate = [-1]
        label_name = self.label_names[str(cate[0])]
        try:
            
            self.plot_oddball_jitter(axs[0], 0, cate=cate)
            axs[0].set_title(f'response to oddball interval \n (jitter, short oddball) {label_name}')
            
            self.plot_oddball_jitter_box(axs[1], 0, cate=cate)
            axs[1].set_title(f'response to post oddball interval stim \n (jitter, short oddball) {label_name}')
            
            self.plot_oddball_jitter(axs[2], 1, cate=cate)
            axs[2].set_title(f'response to oddball interval \n (jitter, long oddball) {label_name}')
            
            self.plot_oddball_jitter_box(axs[3], 0, cate=cate)
            axs[3].set_title(f'response to post oddball interval stim \n (jitter, long oddball) {label_name}')
            
            self.plot_oddball_share(axs[4], 0, cate=cate)
            axs[4].set_title(f'response to oddball interval across blocks \n (short oddball) {label_name}')
            
            self.plot_oddball_share(axs[5], 1, cate=cate)
            axs[5].set_title(f'response to oddball interval across blocks \n (long oddball) {label_name}')
            
        except: pass
    
    def oddball_jitter_inh(self, axs):
        cate = [1]
        label_name = self.label_names[str(cate[0])]
        try:
            
            self.plot_oddball_jitter(axs[0], 0, cate=cate)
            axs[0].set_title(f'response to oddball interval \n (jitter, short oddball) {label_name}')
            
            self.plot_oddball_jitter_box(axs[1], 0, cate=cate)
            axs[1].set_title(f'response to post oddball interval stim \n (jitter, short oddball) {label_name}')
            
            self.plot_oddball_jitter(axs[2], 1, cate=cate)
            axs[2].set_title(f'response to oddball interval \n (jitter, long oddball) {label_name}')
            
            self.plot_oddball_jitter_box(axs[3], 0, cate=cate)
            axs[3].set_title(f'response to post oddball interval stim \n (jitter, long oddball) {label_name}')
            
            self.plot_oddball_share(axs[4], 0, cate=cate)
            axs[4].set_title(f'response to oddball interval across blocks \n (short oddball) {label_name}')
            
            self.plot_oddball_share(axs[5], 1, cate=cate)
            axs[5].set_title(f'response to oddball interval across blocks \n (long oddball) {label_name}')
            
        except: pass

    def cluster_exc(self, axs):
        cate = [-1]
        label_name = self.label_names[str(cate[0])]
        try:
            
            self.plot_cluster(axs, cate=cate)
            axs[ 0].set_title(f'sorted correlation matrix \n (short) {label_name}')
            axs[ 1].set_title(f'clustering evaluation metrics \n (short) {label_name}')
            axs[ 2].set_title(f'cluster fraction \n (short) {label_name}')
            axs[ 3].set_title(f'response to standard interval \n (fix) {label_name}')
            axs[ 4].set_title(f'response to standard interval \n (jitter) {label_name}')
            axs[ 5].set_title(f'response to oddball interval \n (fix, short oddball) {label_name}')
            axs[ 6].set_title(f'response to oddball interval \n (jitter, short oddball) {label_name}')
            axs[ 7].set_title(f'response to oddball interval \n (fix, long oddball) {label_name}')
            axs[ 8].set_title(f'response to oddball interval \n (jitter, long oddball) {label_name}')

        except: pass
    
    def cluster_inh(self, axs):
        cate = [1]
        label_name = self.label_names[str(cate[0])]
        try:
            
            self.plot_cluster(axs, cate=cate)
            axs[ 0].set_title(f'sorted correlation matrix \n (short) {label_name}')
            axs[ 1].set_title(f'clustering evaluation metrics \n (short) {label_name}')
            axs[ 2].set_title(f'cluster fraction \n (short) {label_name}')
            axs[ 3].set_title(f'response to standard interval \n (fix) {label_name}')
            axs[ 4].set_title(f'response to standard interval \n (jitter) {label_name}')
            axs[ 5].set_title(f'response to oddball interval \n (fix, short oddball) {label_name}')
            axs[ 6].set_title(f'response to oddball interval \n (jitter, short oddball) {label_name}')
            axs[ 7].set_title(f'response to oddball interval \n (fix, long oddball) {label_name}')
            axs[ 8].set_title(f'response to oddball interval \n (jitter, long oddball) {label_name}')

        except: pass