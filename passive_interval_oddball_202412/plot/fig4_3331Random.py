#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from modules.Alignment import run_get_stim_response
from modeling.clustering import clustering_neu_response_mode
from modeling.clustering import get_mean_sem_cluster
from modeling.generative import interp_input_seq
from modeling.generative import fit_glm
from modeling.generative import get_coding_score
from modeling.generative import get_explain_variance
from utils import get_norm01_params
from utils import get_bin_idx
from utils import get_mean_sem
from utils import get_mean_sem_win
from utils import get_neu_trial
from utils import get_frame_idx_from_time
from utils import get_epoch_idx
from utils import get_isi_bin_neu
from utils import get_neu_sync
from utils import get_cmap_color
from utils import apply_colormap
from utils import adjust_layout_neu
from utils import adjust_layout_heatmap
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
        self.l_frames = int(100*timescale)
        self.r_frames = int(80*timescale)
        self.list_labels = list_labels
        self.list_neural_trials = list_neural_trials
        self.alignment = run_get_stim_response(
                list_neural_trials, self.l_frames, self.r_frames, expected='none')
        self.list_stim_labels = self.alignment['list_stim_labels']
        self.list_significance = list_significance
        self.bin_win = [450,2550]
        self.bin_num = 3
        self.n_clusters = 4
        self.max_clusters = 20

    def run_clustering(self, cate):
        n_latents = 25
        # collect data.
        [[color0, color1, color2, cmap],
         [neu_seq, stim_seq, stim_value, pre_isi], neu_labels,
         [n_trials, n_neurons]] = get_neu_trial(
            self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
            trial_param=[[2,3,4,5], None, None, None, [1], [0]],
            mean_sem=False,
            cate=cate, roi_id=None)
        # bin data based on isi.
        [bins, bin_center, bin_neu_seq, _, _, bin_stim_seq, bin_stim_value] = get_isi_bin_neu(
            neu_seq, stim_seq, stim_value, pre_isi, self.bin_win, self.bin_num)
        # get correlation matrix within each bin.
        bin_corr = []
        for i in range(self.bin_num):
            # take timestamps with interval+image+interval.
            l_idx, r_idx = get_frame_idx_from_time(
                self.alignment['neu_time'], 0, -bins[i], self.bin_win[0])
            # compute correlation matrix.
            bin_corr.append(np.corrcoef(bin_neu_seq[i][:,l_idx:r_idx]))
        # combine all correlation matrix.
        bin_corr = np.concatenate(bin_corr, axis=1)
        # extract features.
        model = PCA(n_components=n_latents)
        neu_x = model.fit_transform(bin_corr)
        # run clustering on extracted correlation.
        metrics, cluster_id = clustering_neu_response_mode(
            neu_x, self.n_clusters, self.max_clusters)
        return metrics, cluster_id
    
    def plot_random_stim(self, ax, cate=None, roi_id=None):
        # collect data.
        [color0, _, color2, _], [neu_seq, _, stim_seq, stim_value], _, [n_trials, n_neurons] = get_neu_trial(
            self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
            trial_param=[[2,3,4,5], None, None, None, [1], [0]],
            cate=cate, roi_id=roi_id)
        mean_neu, sem_neu = get_mean_sem(neu_seq)
        # compute bounds.
        upper = np.nanmax(mean_neu) + np.nanmax(sem_neu)
        lower = np.nanmin(mean_neu) - np.nanmax(sem_neu)
        # plot stimulus.
        ax.fill_between(
            stim_seq[int(stim_seq.shape[0]/2),:],
            lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
            color=color0, edgecolor='none', alpha=0.25, step='mid')
        self.plot_vol(ax, self.alignment['stim_time'], stim_value.reshape(1,-1), color0, upper, lower)
        # plot neural traces.
        self.plot_mean_sem(ax, self.alignment['neu_time'], mean_neu, sem_neu, color2, None)
        # adjust layout.
        adjust_layout_neu(ax)
        ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
        ax.set_xlabel('time since stim (ms)')
        add_legend(ax, [color0, color2], ['stim', 'dff'], n_trials, n_neurons, self.n_sess, 'upper right')
    
    def plot_random_sync(self, ax, cate):
        win_width = 200
        l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, 0, win_width)
        win_width = r_idx - l_idx
        # collect data.
        [color0, _, color2, _], [neu_seq, _, stim_seq, stim_value], _, [n_trials, n_neurons] = get_neu_trial(
            self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
            trial_param=[[2,3,4,5], None, None, None, [1], [0]],
            cate=cate)
        # compute synchronization.
        neu_sync = get_neu_sync(neu_seq, win_width)
        # find bounds.
        upper = np.nanmax(neu_sync)
        lower = np.nanmin(neu_sync)
        # plot stimulus.
        ax.fill_between(
            stim_seq[int(stim_seq.shape[0]/2),:],
            lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
            color=color0, edgecolor='none', alpha=0.25, step='mid')
        self.plot_vol(ax, self.alignment['stim_time'], stim_value.reshape(1,-1), color0, upper, lower)
        # plot synchronization.
        ax.plot(self.alignment['neu_time'][win_width:], neu_sync, color=color2)
        ax.tick_params(axis='y', tick1On=False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_ylabel('sync level')
        ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
        ax.set_xlabel('time since stim (ms)')
        add_legend(ax, [color0, color2], ['stim', 'sync'], n_trials, n_neurons, self.n_sess, 'upper right')

    def plot_random_interval(self, ax, cate=None, roi_id=None):
        # collect data.
        [_, _, color2, _], [neu_seq, stim_seq, stim_value, pre_isi], _, [n_trials, n_neurons] = get_neu_trial(
            self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
            trial_param=[[2,3,4,5], None, None, None, [1], [0]],
            mean_sem=False,
            cate=cate, roi_id=roi_id)
        colors = get_cmap_color(self.bin_num, base_color=color2)
        # bin data based on isi.
        [bins, bin_center, _, bin_neu_mean, bin_neu_sem, bin_stim_seq, bin_stim_value] = get_isi_bin_neu(
            neu_seq, stim_seq, stim_value, pre_isi, self.bin_win, self.bin_num)
        # compute bounds.
        upper = np.nanmax(bin_neu_mean) + np.nanmax(bin_neu_sem)
        lower = np.nanmin(bin_neu_mean) - np.nanmax(bin_neu_sem)
        # plot stimulus.
        for i in range(self.bin_num):
            ax.fill_between(
                bin_stim_seq[i,int(bin_stim_seq.shape[1]/2),:],
                lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
                color=colors[i], edgecolor='none', alpha=0.25, step='mid')
            ax.fill_between(
                bin_stim_seq[i,int(bin_stim_seq.shape[1]/2)-1,:],
                lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
                color=colors[i], edgecolor='none', alpha=0.25, step='mid')
            self.plot_vol(ax, self.alignment['stim_time'], bin_stim_value[i,:].reshape(1,-1), colors[i], upper, lower)
        # plot neural traces.
        for i in range(self.bin_num):
            self.plot_mean_sem(ax, self.alignment['neu_time'], bin_neu_mean[i], bin_neu_sem[i], colors[i], None)
        # adjust layout.
        adjust_layout_neu(ax)
        ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
        ax.set_xlabel('time since stim (ms)')
        lbl = ['[{},{}] ms'.format(int(bins[i]),int(bins[i+1])) for i in range(self.bin_num)]
        add_legend(ax, colors, lbl, n_trials, n_neurons, self.n_sess, 'upper right')

    def plot_random_interval_box(self, ax, cate=None, roi_id=None):
        win_base = [-self.bin_win[1],0]
        offsets = np.arange(self.bin_num)/20
        # collect data.
        [_, _, color2, _], [neu_seq, stim_seq, stim_value, pre_isi], _, [n_trials, n_neurons] = get_neu_trial(
            self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
            trial_param=[[2,3,4,5], None, None, None, [1], [0]],
            mean_sem=False,
            cate=cate, roi_id=roi_id)
        colors = get_cmap_color(self.bin_num, base_color=color2)
        # bin data based on isi.
        [bins, bin_center, bin_neu_seq, _, _, _, _] = get_isi_bin_neu(
            neu_seq, stim_seq, stim_value, pre_isi, self.bin_win, self.bin_num)
        # plot errorbar.
        for i in range(self.bin_num):
            self.plot_win_mag_box(
                ax, bin_neu_seq[i], self.alignment['neu_time'], win_base,
                colors[i], 0, offsets[i])
        # adjust layout.
        lbl = ['[{},{}] ms'.format(int(bins[i]),int(bins[i+1])) for i in range(self.bin_num)]
        add_legend(ax, colors, lbl, n_trials, n_neurons, self.n_sess, 'upper right')

    def plot_random_interval_heatmap(self, ax, cate=None, roi_id=None):
        # collect data.
        [_, _, _, cmap], [neu_seq, stim_seq, stim_value, pre_isi], _, [n_trials, n_neurons] = get_neu_trial(
            self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
            trial_param=[[2,3,4,5], None, None, None, [1], [0]],
            mean_sem=False,
            cate=cate, roi_id=roi_id)
        # average across neurons.
        neu_seq = np.concatenate([np.nanmean(n,axis=1) for n in neu_seq],axis=0)
        # sort based on isi.
        pre_isi = np.concatenate(pre_isi)
        neu_seq = neu_seq[np.argsort(pre_isi),:]
        # plot heatmap.
        heatmap = apply_colormap(neu_seq, cmap)
        ax.imshow(
            heatmap,
            extent=[
                self.alignment['neu_time'][0], self.alignment['neu_time'][-1],
                1, heatmap.shape[0]],
            interpolation='nearest', aspect='auto')
        # adjust layout.
        adjust_layout_heatmap(ax)
        cbar = ax.figure.colorbar(
            plt.cm.ScalarMappable(cmap=cmap), ax=ax, ticks=[0.2,0.8], shrink=1, aspect=50)
        cbar.outline.set_visible(False)
        cbar.ax.set_ylabel('normalized dff', rotation=-90, va="bottom")
        cbar.ax.set_yticklabels(['0.2', '0.8'], rotation=-90)
        ax.set_xlabel('time since stim (ms)')
        ax.set_ylabel('preceeding interval (ms)')
        idx = np.arange(1, heatmap.shape[0], heatmap.shape[0]/5, dtype='int32') + int(heatmap.shape[0]/5/2)
        ax.set_yticks(idx)
        ax.set_yticklabels(pre_isi[np.argsort(pre_isi)][idx][::-1].astype('int32'))

    def plot_random_interval_curve(self, ax, cate=None, roi_id=None):
        win_base = [-self.bin_win[1],0]
        win_evoke = [200,400]
        l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, win_evoke[0], win_evoke[1])
        # collect data.
        [_, _, color2, _], [neu_seq, stim_seq, stim_value, pre_isi], _, [n_trials, n_neurons] = get_neu_trial(
            self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
            trial_param=[[2,3,4,5], None, None, None, [1], [0]],
            mean_sem=False,
            cate=cate, roi_id=roi_id)
        # average across neurons.
        neu_seq = np.concatenate([np.nanmean(n,axis=1) for n in neu_seq],axis=0)
        pre_isi = np.concatenate(pre_isi)
        # get evoked response.
        neu_baseline, _ = get_mean_sem_win(
            neu_seq.reshape(-1, neu_seq.shape[-1]),
            self.alignment['neu_time'], 0, win_base[0], win_base[1], mode='lower')
        neu_evoke = np.nanmean(neu_seq[:,l_idx:r_idx],axis=1) - neu_baseline
        # bin response data.
        bins, bin_center, bin_idx = get_bin_idx(pre_isi, self.bin_win, self.bin_num)
        neu_mean = []
        neu_sem = []
        for i in range(self.bin_num):
            m,s = get_mean_sem(neu_evoke[bin_idx==i].reshape(-1,1))
            neu_mean.append(m)
            neu_sem.append(s)
        neu_mean = np.concatenate(neu_mean)
        neu_sem = np.concatenate(neu_sem)
        # plot errorbar.
        ax.errorbar(
            bin_center,
            neu_mean,
            neu_sem,
            color=color2, capsize=2, marker='o', linestyle='none',
            markeredgecolor='white', markeredgewidth=1)
        # adjust layout.
        ax.tick_params(tick1On=False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xlabel('preceeding interval (ms)')
        ax.set_ylabel('response magnitude during \n [{},{}] ms'.format(
            win_evoke[0], win_evoke[1]))
        ax.set_xlim(self.bin_win)
        add_legend(ax, None, None, n_trials, n_neurons, self.n_sess, 'upper right')

    def plot_random_interval_corr_epoch(self, ax, cate=None, roi_id=None):
        win_base = [-self.bin_win[1],0]
        win_evoke = [200,400]
        epoch_len = 500
        l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, win_evoke[0], win_evoke[1])
        # collect data.
        [_, _, color2, _], [neu_seq, stim_seq, stim_value, pre_isi], _, [n_trials, n_neurons] = get_neu_trial(
            self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
            trial_param=[[2,3,4,5], None, None, None, [1], [0]],
            mean_sem=False,
            cate=cate, roi_id=roi_id)
        # divide into epoch.
        epoch_idx = [get_epoch_idx(sl, 'random', epoch_len) for sl in self.list_stim_labels]
        epoch_idx = np.concatenate(epoch_idx)
        # average across neurons.
        neu_seq = np.concatenate([np.nanmean(n,axis=1) for n in neu_seq],axis=0)
        pre_isi = np.concatenate(pre_isi)
        # get evoked response.
        neu_baseline = np.array([get_mean_sem_win(
            n.reshape(-1, n.shape[-1]),
            self.alignment['neu_time'], 0, win_base[0], win_base[1], mode='lower')[0]
            for n in neu_seq])
        neu_evoke = np.array([get_mean_sem_win(
            n.reshape(-1, n.shape[-1]),
            self.alignment['neu_time'], 0, win_evoke[0], win_evoke[1], mode='higher')[0]
            for n in neu_seq])
        neu_evoke -= neu_baseline
        # compute correlations.
        corr = []
        for i in np.unique(epoch_idx)[:-1]:
            c = np.corrcoef(neu_evoke[epoch_idx==i], pre_isi[epoch_idx==i])[0,1]
            corr.append(c)
        corr = np.array(corr)
        # plot results.
        ax.plot(np.unique(epoch_idx)[:-1], corr, color=color2)
        # adjust layout.
        ax.tick_params(tick1On=False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xlabel('epoch')
        ax.set_ylabel('correlation score during \n [{},{}] ms'.format(
            win_evoke[0], win_evoke[1]))
        ax.set_xlim([-1, np.nanmax(np.unique(epoch_idx))+1])
        add_legend(ax, None, None, n_trials, n_neurons, self.n_sess, 'upper right')
    
    def plot_random_cluster(self, axs, cate=None):
        metrics, cluster_id = self.run_clustering(cate)
        colors = get_cmap_color(self.n_clusters, cmap=plt.cm.nipy_spectral)
        # plot basic clustering results.
        def plot_info(axs):
            # collect data.
            [[color0, color1, color2, cmap],
             [neu_seq, stim_seq, stim_value, pre_isi], neu_labels,
             [n_trials, n_neurons]] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_param=[[2,3,4,5], None, None, None, [1], [0]],
                mean_sem=False,
                cate=cate, roi_id=None)
            neu = [np.nanmean(n, axis=0) for n in neu_seq]
            neu = np.concatenate(neu, axis=0)
            # plot info.
            self.plot_cluster_info(
                axs, colors, cmap, neu,
                self.n_clusters, self.max_clusters,
                metrics, cluster_id, neu_labels, self.label_names, cate)
        plot_info(axs[:5])
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
            c_neu = [get_cmap_color(self.bin_num, base_color=c) for c in colors]
            # convert to colors for each bin.
            c_neu = [[c_neu[i][j] for i in range(self.n_clusters)] for j in range(self.bin_num)]
            lbl = ['[{},{}] ms'.format(int(bins[i]),int(bins[i+1])) for i in range(self.bin_num)]
            lbl+= ['cluster #'+str(i) for i in range(self.n_clusters)]
            c_all = get_cmap_color(self.bin_num, base_color='#2C2C2C') + colors
            # only keep 2 stimulus.
            c_idx = int(bin_stim_seq.shape[1]/2)
            stim_seq = bin_stim_seq[:,c_idx-1:c_idx+1,:]
            # plot results.
            self.plot_cluster_interval_norm(
                ax, cluster_bin_neu_mean, cluster_bin_neu_sem,
                norm_params, stim_seq, c_neu)
            # adjust layout.
            add_legend(ax, c_all, lbl, n_trials, n_neurons, self.n_sess, 'upper right')
        plot_interval_norm(axs[5])
        # plot response on random for all clusters.
        def plot_random(ax):
            xlim = [-3500, 2500]
            # collect data.
            [[color0, color1, color2, cmap],
             [neu_seq, stim_seq, stim_value, pre_isi], neu_labels,
             [n_trials, n_neurons]] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_param=[[2,3,4,5], None, None, None, [1], [0]],
                mean_sem=False,
                cate=cate, roi_id=None)
            # bin data based on isi.
            [bins, bin_center, bin_neu_seq, _, _, bin_stim_seq, bin_stim_value] = get_isi_bin_neu(
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
            cs_all = get_cmap_color(self.bin_num, base_color='#2C2C2C')
            lbl = ['[{},{}] ms'.format(int(bins[i]),int(bins[i+1])) for i in range(self.bin_num)]
            # only keep 2 stimulus.
            c_idx = int(bin_stim_seq.shape[1]/2)
            stim_seq = bin_stim_seq[:,c_idx-1:c_idx+1,:]
            # plot results.
            for i in range(self.bin_num):
                self.plot_cluster_mean_sem(
                    ax, cluster_bin_neu_mean[i,:,:], cluster_bin_neu_sem[i,:,:],
                    norm_params, stim_seq[i,:,:], [cs_all[i]]*stim_seq.shape[-2], cs[i], xlim)
            # adjust layout.
            ax.set_xlabel('time since stim (ms)')
            ax.set_xlim(xlim)
            add_legend(ax, cs_all, lbl, n_trials, n_neurons, self.n_sess, 'upper right')
        plot_random(axs[6])
        # plot response heatmap on random for all clusters.
        def plot_heatmap_random(ax):
            xlim = [-3500, 2500]
            l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, xlim[0], xlim[1])
            # collect data.
            _, [neu_seq, _, _, _], _, _ = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_param=[[2,3,4,5], None, None, None, [1], [0]],
                cate=cate, roi_id=None)
            neu_seq = neu_seq[:,l_idx:r_idx]
            neu_time = self.alignment['neu_time'][l_idx:r_idx]
            # plot heatmap.
            self.plot_cluster_heatmap(ax, neu_seq, neu_time, neu_seq, cluster_id, colors)
            ax.set_xlabel('time since stim (ms)')
        plot_heatmap_random(axs[7])

    def plot_random_cluster_heatmaps(self, axs, sort_target, cate=None):
        metrics, cluster_id = self.run_clustering(cate)
        colors = get_cmap_color(self.n_clusters, cmap=plt.cm.nipy_spectral)
        xlim = [-3500, 2500]
        l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, xlim[0], xlim[1])
        # collect data.
        [_, [neu_seq, stim_seq, stim_value, pre_isi], _, [n_trials, n_neurons]] = get_neu_trial(
            self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
            trial_param=[None, None, None, None, None, [0]],
            mean_sem=False,
            cate=cate, roi_id=None)
        neu_time = self.alignment['neu_time'][l_idx:r_idx]
        # bin data based on isi.
        [_, _, neu_x, _, _, _, _] = get_isi_bin_neu(
            neu_seq, stim_seq, stim_value, pre_isi, self.bin_win, self.bin_num)
        if sort_target == 'shortest':
            neu_seq_sort = neu_x[0].copy()
        if sort_target == 'longest':
            neu_seq_sort = neu_x[-1].copy()
        # plot heatmaps.
        for i in range(len(neu_x)):
            self.plot_cluster_heatmap(
                axs[i], neu_x[i], neu_time, neu_seq_sort, cluster_id, colors)
            axs[i].set_xlabel('time since stim (ms)')
            
    def plot_random_glm(self, axs, cate=None):
        win_kernal = 2500
        l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, 0, win_kernal)
        len_kernel = r_idx - l_idx
        # collect data.
        [[color0, color1, color2, cmap],
         [neu_seq, stim_seq, stim_value, pre_isi], _,
         [n_trials, n_neurons]] = get_neu_trial(
            self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
            trial_param=[[2,3,4,5], None, None, None, [1], [0]],
            mean_sem=False,
            cate=cate, roi_id=None)
        input_seq = [interp_input_seq(sv, self.alignment['stim_time'], self.alignment['neu_time'])
                     for sv in stim_value]
        # fit glm model for all sessions.
        def run_glm():
            w_all = [fit_glm(x, y, len_kernel) for (x,y) in zip(neu_seq, input_seq)]
            return w_all
        w_all = run_glm()
        # plot coding score across kernal window.
        def plot_coding_score(ax):
            win_kernal_range = np.arange(
                self.bin_win[0], self.bin_win[1] + self.bin_win[0], self.bin_win[0])
            # run all glm for different kernal window.
            coding_score = []
            for win in win_kernal_range:
                # find kernal length.
                l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, 0, win)
                len_kernel = r_idx - l_idx
                # run glm.
                w_all = [fit_glm(x, y, len_kernel) for (x,y) in zip(neu_seq, input_seq)]
                r2 = [get_coding_score(x, y, w) for (x,y,w) in zip(neu_seq, input_seq, w_all)]
                r2 = np.concatenate(r2)
                coding_score.append(r2)
            # plot errorbar.
            score_mean = np.concatenate([get_mean_sem(s.reshape(-1,1))[0] for s in coding_score])
            score_sem = np.concatenate([get_mean_sem(s.reshape(-1,1))[1] for s in coding_score])
            ax.errorbar(
                win_kernal_range,
                score_mean,
                score_sem,
                color=color2, capsize=2, marker='o', linestyle='none',
                markeredgecolor='white', markeredgewidth=1)
            # adjust layout.
            ax.tick_params(tick1On=False)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.set_xlabel('kernal window (ms)')
            ax.set_ylabel(r'coding score ($R^2$)')
            ax.set_xticks(win_kernal_range)
            add_legend(ax, None, None, n_trials, n_neurons, self.n_sess, 'upper left')
        plot_coding_score(axs[0])
        # plot explain variance across kernal window.
        def plot_explain_variance(ax):
            win_kernal_range = np.arange(
                self.bin_win[0], self.bin_win[1] + self.bin_win[0], self.bin_win[0])
            # run all glm for different kernal window.
            explain_var = []
            for win in win_kernal_range:
                # find kernal length.
                l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, 0, win)
                len_kernel = r_idx - l_idx
                # run glm.
                w_all = [fit_glm(x, y, len_kernel) for (x,y) in zip(neu_seq, input_seq)]
                ev = [get_explain_variance(x, y, w) for (x,y,w) in zip(neu_seq, input_seq, w_all)]
                ev = np.concatenate(ev)
                explain_var.append(ev)
            # plot errorbar.
            score_mean = np.concatenate([get_mean_sem(s.reshape(-1,1))[0] for s in explain_var])
            score_sem = np.concatenate([get_mean_sem(s.reshape(-1,1))[1] for s in explain_var])
            ax.errorbar(
                win_kernal_range,
                score_mean,
                score_sem,
                color=color2, capsize=2, marker='o', linestyle='none',
                markeredgecolor='white', markeredgewidth=1)
            # adjust layout.
            ax.tick_params(tick1On=False)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.set_xlabel('kernal window (ms)')
            ax.set_ylabel('explain variance')
            ax.set_xticks(win_kernal_range)
            add_legend(ax, None, None, n_trials, n_neurons, self.n_sess, 'upper left')
        plot_explain_variance(axs[1])
        # clustering on glm weights.
        def plot_glm_cluster_w(ax):
            gap = 5
            w = np.concatenate(w_all)
            # run clustering.
            _, cluster_id = clustering_neu_response_mode(
                w, self.n_clusters, self.max_clusters)
            # group glm weights.
            w_group = []
            for i in range(self.n_clusters):
                w_group.append(w[cluster_id==i,:])
                w_group.append(np.zeros((gap,w.shape[1])))
            w_group = np.concatenate(w_group[:-1],axis=0)
            # plot heatmap.
            heatmap = apply_colormap(w_group, cmap)
            ax.imshow(
                heatmap,
                extent=[-win_kernal, 0, 1, heatmap.shape[0]],
                interpolation='nearest', aspect='auto',
                origin='lower')
            # adjust layout.
            adjust_layout_heatmap(ax)
            ax.set_xlabel('time from kernal window (ms)')
            pos_tick = [int(np.sum(cluster_id==i)/2 + i*gap + np.sum(cluster_id<i))
                        for i in range(self.n_clusters)]
            ax.set_yticks(pos_tick)
            ax.set_yticklabels(['{}'.format(i) for i in range(self.n_clusters)])
            ax.set_ylabel('cluster id')
        plot_glm_cluster_w(axs[2])

# colors = ['#989A9C', '#A4CB9E', '#9DB4CE', '#EDA1A4', '#F9C08A']
class plotter_main(plotter_utils):
    def __init__(self, neural_trials, labels, significance, label_names):
        super().__init__(neural_trials, labels, significance)
        self.label_names = label_names
    
    def random(self, axs_all):
        for cate, axs in zip([[-1],[1]], axs_all):
            label_name = self.label_names[str(cate[0])] if len(cate)==1 else 'all'
            try:
                
                self.plot_random_stim(axs[0], cate=cate)
                axs[0].set_title(f'response to random stim \n {label_name}')
                
                self.plot_random_sync(axs[1], cate=cate)
                axs[1].set_title(f'response to random stim synchronization level \n {label_name}')
                
                self.plot_random_interval(axs[2], cate=cate)
                axs[2].set_title(f'response to random preceeding interval with bins \n {label_name}')
                
                self.plot_random_interval_box(axs[3], cate=cate)
                axs[3].set_title(f'response to random preceeding interval with bins \n {label_name}')
                
                self.plot_random_interval_heatmap(axs[4], cate=cate)
                axs[4].set_title(f'response to random preceeding interval heatmap \n {label_name}')
                
                self.plot_random_interval_curve(axs[5], cate=cate)
                axs[5].set_title(f'evoked response to random VS interval \n {label_name}')
                
                self.plot_random_interval_corr_epoch(axs[6], cate=cate)
                axs[6].set_title(f'response and interval correlation across epochs \n {label_name}')

            except: pass

    def cluster(self, axs_all):
        for cate, axs in zip([[-1],[1],[-1,1]], axs_all):
            label_name = self.label_names[str(cate[0])] if len(cate)==1 else 'all'
            try:
                
                self.plot_random_cluster(axs, cate=cate)
                axs[0].set_title(f'sorted correlation matrix \n {label_name}')
                axs[1].set_title(f'clustering evaluation metrics \n {label_name}')
                axs[2].set_title(f'cluster calcium transient \n (short) {label_name}')
                axs[3].set_title(f'cluster fraction \n {label_name}')
                axs[4].set_title(f'cluster fraction for subtypes \n {label_name}')
                axs[5].set_title(f'time normalized reseponse to preceeding interval with bins \n {label_name}')
                axs[6].set_title(f'response to random preceeding interval with bins \n {label_name}')
                axs[7].set_title(f'response to random interval \n {label_name}')
            
            except: pass
    
    def cluster_heatmaps(self, axs_all):
        for cate, axs in zip([[-1,1]], axs_all):
            label_name = self.label_names[str(cate[0])] if len(cate)==1 else 'all'
            try:
                
                self.plot_random_cluster_heatmaps(axs[0], 'shortest', cate=cate)
                for i in range(3):
                    axs[0][i].set_title(f'response to random preceeding interval bin#{i} \n (sorted by shortest) {label_name}')
                
                self.plot_random_cluster_heatmaps(axs[1], 'longest', cate=cate)
                for i in range(3):
                    axs[1][i].set_title(f'response to random preceeding interval bin#{i} \n (sorted by longest) {label_name}')
                    
            except: pass

    def glm(self, axs_all):
        for cate, axs in zip([[-1],[1]], axs_all):
            label_name = self.label_names[str(cate[0])] if len(cate)==1 else 'all'
            try:
            
                self.plot_random_glm(axs, cate=cate)
                axs[0].set_title(f'coding score V.S. kernal window size \n {label_name}')
                axs[1].set_title(f'explain variance V.S. kernal window size \n {label_name}')
                axs[2].set_title(f'clustered kernal weights \n {label_name}')
            
            except: pass