#!/usr/bin/env python3

import numpy as np
import matplotlib.ticker as mtick
from sklearn.decomposition import PCA

from modules.Alignment import run_get_stim_response
from modeling.clustering import get_mean_sem_cluster
from modeling.clustering import get_bin_mean_sem_cluster
from modeling.clustering import get_cluster_cate
from modeling.generative import get_glm_cate
from utils import show_resource_usage
from utils import get_norm01_params
from utils import get_odd_stim_prepost_idx
from utils import get_mean_sem_win
from utils import get_mean_sem
from utils import get_neu_trial
from utils import get_frame_idx_from_time
from utils import get_split_idx
from utils import get_block_1st_idx
from utils import get_block_transition_idx
from utils import exclude_odd_stim
from utils import get_isi_bin_neu
from utils import get_expect_interval
from utils import get_roi_label_color
from utils import get_cmap_color
from utils import hide_all_axis
from utils import adjust_layout_neu
from utils import adjust_layout_2d_latent
from utils import adjust_layout_3d_latent
from utils import add_legend
from utils import utils_basic

# fig, ax = plt.subplots(1, 1, figsize=(6, 6))
# fig, ax = plt.subplots(1, 1, figsize=(2, 20))
# fig, ax = plt.subplots(1, 1, figsize=(3, 9))
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
        self.n_clusters = 9
        self.max_clusters = 10
        self.d_latent = 3
        self.glm = self.run_glm()
        self.cluster_id = self.run_clustering()

    def plot_cluster_all(self, axs, cate):
        kernel_all = get_glm_cate(self.glm, self.list_labels, self.list_significance, cate)
        cluster_id, neu_labels = get_cluster_cate(self.cluster_id, self.list_labels, self.list_significance, cate)
        @show_resource_usage
        def plot_glm_kernel(ax):
            ax.axis('off')
            ax = ax.inset_axes([0, 0, 0.5, 1], transform=ax.transAxes)
            # collect data.
            [[color0, _, _, _],
             [_, _, stim_seq, _], _, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_param=[[2,3,4,5], None, None, None, [1], [0]],
                cate=cate, roi_id=None)
            c_idx = stim_seq.shape[0]//2
            # get cluster average.
            glm_mean, glm_sem = get_mean_sem_cluster(kernel_all, self.n_clusters, cluster_id)
            # plot results.
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
                xlim = [-3500, 1500]
                isi = pre_isi
                isi_idx_offset = -1
            if mode == 'post':
                xlim = [-1000, 4000]
                isi = post_isi
                isi_idx_offset = 1
            [bins, bin_center, _, bin_neu_seq, _, _, bin_stim_seq, _] = get_isi_bin_neu(
                neu_seq, stim_seq, camera_pupil, isi, self.bin_win, self.bin_num)
            c_idx = bin_stim_seq.shape[1]//2
            # get response within cluster at each bin.
            cluster_bin_neu_mean, cluster_bin_neu_sem = get_bin_mean_sem_cluster(bin_neu_seq, self.n_clusters, cluster_id)
            norm_params = [get_norm01_params(cluster_bin_neu_mean[:,i,:]) for i in range(self.n_clusters)]
            # plot results.
            ax.fill_between(
                np.nanmean(bin_stim_seq, axis=0)[c_idx,:],
                0, self.n_clusters,
                color=color0, edgecolor='none', alpha=0.25, step='mid')
            for bi in range(self.bin_num):
                self.plot_cluster_mean_sem(
                    ax, cluster_bin_neu_mean[bi,:,:], cluster_bin_neu_sem[bi,:,:],
                    self.alignment['neu_time'], norm_params,
                    bin_stim_seq[bi, c_idx+isi_idx_offset, :].reshape(1,-1),
                    [colors[bi]], [colors[bi]]*self.n_clusters, xlim)
            # adjust layouts.
            ax.set_xlabel('time since stim (ms)')
        @show_resource_usage
        def plot_interval_bin_latent(ax):
            # collect data.
            [[color0, _, color2, _],
             [neu_seq, stim_seq, camera_pupil, pre_isi, post_isi], _,
             [n_trials, n_neurons]] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_param=[[2,3,4,5], None, None, None, [1], [0]],
                mean_sem=False,
                cate=cate, roi_id=None)
            colors = get_cmap_color(self.bin_num, base_color=color2)
            c_idx = np.nanmean(np.concatenate(stim_seq, axis=0), axis=0).shape[0]//2
            # bin data based on isi.
            [_, _, _, bin_neu_seq, _, _, bin_stim_seq, _] = get_isi_bin_neu(
                neu_seq, stim_seq, camera_pupil, post_isi, self.bin_win, self.bin_num)
            # get latent dynamics.
            neu_z = np.zeros([self.n_clusters, self.bin_num, 2, len(self.alignment['neu_time'])])
            for ci in range(self.n_clusters):
                if np.sum(cluster_id==ci) > 0:
                    neu_x = np.concatenate([bns[cluster_id==ci] for bns in bin_neu_seq], axis=1)
                    # fit model.
                    model = PCA(n_components=2)
                    model.fit(neu_x.reshape(neu_x.shape[0],-1).T)
                    z = model.transform(neu_x.reshape(neu_x.shape[0],-1).T)
                    z = z.reshape(self.bin_num, -1, 2).transpose([0,2,1])
                    neu_z[ci,:,:,:] = z
            # define layouts.
            ax.axis('off')
            axs = [ax.inset_axes([0, ci/self.n_clusters, 0.5, 0.8/self.n_clusters], transform=ax.transAxes)
                   for ci in range(self.n_clusters)]
            axs.reverse()
            # plot results for each class.
            for ci in range(self.n_clusters):
                if np.sum(cluster_id==ci) > 0:
                    for bi in range(self.bin_num):
                        # plot latent trajectories.
                        l_idx, r_idx = get_frame_idx_from_time(
                            self.alignment['neu_time'], 0, bin_stim_seq[bi, c_idx,0], bin_stim_seq[bi, c_idx+1,0])
                        z = neu_z[ci, bi, :, l_idx:r_idx]
                        axs[ci].plot(z[0,:]-z[0,0], z[1,:]-z[1,0], color=colors[bi])
                        # plot end point.
                        axs[ci].scatter(z[0,0]-z[0,0], z[1,0]-z[1,0], color='black', marker='x')
                        axs[ci].scatter(z[0,-1]-z[0,0], z[1,-1]-z[1,0], color='black', marker='o')
                    # adjust layouts.
                    adjust_layout_2d_latent(axs[ci])
        @show_resource_usage
        def plot_standard(ax, standard):
            xlim = [-2000,3000]
            # collect data.
            [[color0, color1, color2, _],
             [neu_seq, _, stim_seq, _], _, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_param=[[2,3,4,5], [standard], None, None, [0], [0]],
                cate=cate, roi_id=None)
            # get response within cluster.
            neu_mean, neu_sem = get_mean_sem_cluster(neu_seq, self.n_clusters, cluster_id)
            norm_params = [get_norm01_params(neu_mean[i,:]) for i in range(self.n_clusters)]
            # plot results.
            self.plot_cluster_mean_sem(
                ax, neu_mean, neu_sem,
                self.alignment['neu_time'], norm_params,
                stim_seq,
                [color0]*stim_seq.shape[0], [[color1, color2][standard]]*self.n_clusters, xlim)
            # adjust layouts.
            ax.set_xlabel('time since stim (ms)')
        @show_resource_usage
        def plot_standard_latent(ax):
            # collect data.
            [[color0, color1, color2, _],
             [neu_seq_0, _, stim_seq_0, _], _, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance,
                [exclude_odd_stim(sl) for sl in self.list_stim_labels],
                trial_param=[[2,3,4,5], [0], None, None, [0], [0]],
                cate=cate, roi_id=None)
            [[color0, color1, color2, _],
             [neu_seq_1, _, stim_seq_1, _], _, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance,
                [exclude_odd_stim(sl) for sl in self.list_stim_labels],
                trial_param=[[2,3,4,5], [1], None, None, [0], [0]],
                cate=cate, roi_id=None)
            c_idx = stim_seq_0.shape[0]//2
            # get latent dynamics.
            neu_z0 = []
            neu_z1 = []
            for ci in range(self.n_clusters):
                if np.sum(cluster_id==ci) > 0:
                    neu_x0 = neu_seq_0[cluster_id==ci,:].copy()
                    neu_x1 = neu_seq_1[cluster_id==ci,:].copy()
                    neu_x = np.concatenate([neu_x0, neu_x1], axis=1)
                    # fit model.
                    model = PCA(n_components=2)
                    model.fit(neu_x.reshape(neu_x.shape[0],-1).T)
                    z0 = model.transform(neu_x0.reshape(neu_x0.shape[0],-1).T).T.reshape(2, -1)
                    z1 = model.transform(neu_x1.reshape(neu_x1.shape[0],-1).T).T.reshape(2, -1)
                    neu_z0.append(z0)
                    neu_z1.append(z1)
            # define layouts.
            ax.axis('off')
            axs = [ax.inset_axes([0, ci/self.n_clusters, 0.5, 0.9/self.n_clusters], transform=ax.transAxes)
                   for ci in range(self.n_clusters)]
            axs.reverse()
            # plot results for each class.
            for ci in range(self.n_clusters):
                if np.sum(cluster_id==ci) > 0:
                    # plot latent trajectories.
                    l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, stim_seq_0[c_idx,0], stim_seq_0[c_idx+1,0])
                    z0 = neu_z0[ci][:, l_idx:r_idx]
                    axs[ci].plot(z0[0,:]-z0[0,0], z0[1,:]-z0[1,0], color=color1)
                    l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, stim_seq_1[c_idx,0], stim_seq_1[c_idx+1,0])
                    z1 = neu_z1[ci][:, l_idx:r_idx]
                    axs[ci].plot(z1[0,:]-z1[0,0], z1[1,:]-z1[1,0], color=color2)
                    # end point.
                    axs[ci].scatter(z0[0,0]-z0[0,0],  z0[1,0]-z0[1,0],  color='black', marker='x')
                    axs[ci].scatter(z0[0,-1]-z0[0,0], z0[1,-1]-z0[1,0], color='black', marker='o')
                    axs[ci].scatter(z1[0,0]-z1[0,0],  z1[1,0]-z1[1,0],  color='black', marker='x')
                    axs[ci].scatter(z1[0,-1]-z1[0,0], z1[1,-1]-z1[1,0], color='black', marker='o')
                    # adjust layouts.
                    adjust_layout_2d_latent(axs[ci])
        @show_resource_usage
        def plot_oddball(ax, oddball):
            xlim = [-3000,4000]
            # collect data.
            [[color0, color1, color2, _],
             [neu_seq, _, stim_seq, _],
             [neu_labels, neu_sig], _] = get_neu_trial(
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
                ax.axvline(stim_seq[c_idx,1]+self.expect[0], color='gold', lw=1, linestyle='--')
            self.plot_cluster_mean_sem(
                ax, neu_mean, neu_sem,
                self.alignment['neu_time'], norm_params,
                stim_seq,
                [color0]*stim_seq.shape[0], [[color1,color2][oddball]]*self.n_clusters, xlim)
            # adjust layouts.
            ax.set_xlabel('time since pre oddball stim (ms)')
        @show_resource_usage
        def plot_neu_fraction(ax):
            # collect data.
            [[_, _, color2, _], _, _, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_param=[None, None, [0], None, [0], [0]],
                cate=cate, roi_id=None)
            # plot results.
            self.plot_cluster_neu_fraction_in_cluster(ax, cluster_id, color2)
        @show_resource_usage
        def plot_fraction(ax):
            self.plot_cluster_cate_fraction_in_cluster(ax, cluster_id, neu_labels, self.label_names)
        @show_resource_usage
        def plot_legend(ax):
            [[color0, _, color2, _],
             [neu_seq, stim_seq, camera_pupil, pre_isi, post_isi],
             [neu_labels, _],
             [n_trials, n_neurons]] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_param=[[2,3,4,5], None, None, None, [0], [0]],
                mean_sem=False,
                cate=cate, roi_id=None)
            [bins, _, _, _, _, _, _, _] = get_isi_bin_neu(
                neu_seq, stim_seq, camera_pupil, post_isi, self.bin_win, self.bin_num)
            colors = get_cmap_color(self.bin_num, base_color=color2)
            lbl = ['[{},{}] ms'.format(int(bins[i]),int(bins[i+1])) for i in range(self.bin_num)]
            lbl+= [v for v in self.label_names.values()] + ['all cell-types']
            cs = colors + [get_roi_label_color(cate=[int(k)])[2] for k in self.label_names.keys()]
            cs+= [get_roi_label_color(cate=[-1,1,2])[2]]
            add_legend(ax, cs, lbl, n_trials, n_neurons, self.n_sess, 'upper right')
            ax.axis('off')
        # plot all.
        try: plot_glm_kernel(axs[0])
        except Exception as e: print(e)
        try: plot_random_bin(axs[1], 'pre')
        except Exception as e: print(e)
        try: plot_random_bin(axs[2], 'post')
        except Exception as e: print(e)
        try: plot_interval_bin_latent(axs[3])
        except Exception as e: print(e)
        try: plot_standard(axs[4], 0)
        except Exception as e: print(e)
        try: plot_standard(axs[5], 1)
        except Exception as e: print(e)
        try: plot_standard_latent(axs[6])
        except Exception as e: print(e)
        try: plot_oddball(axs[7], 0)
        except Exception as e: print(e)
        try: plot_oddball(axs[8], 1)
        except Exception as e: print(e)
        try: plot_neu_fraction(axs[9])
        except Exception as e: print(e)
        try: plot_fraction(axs[10])
        except Exception as e: print(e)
        try: plot_legend(axs[11])
        except Exception as e: print(e)

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
        def plot_standard(ax, standard):
            xlim = [-2000,3000]
            l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, xlim[0], xlim[1])
            # collect data.
            [[_, _, _, cmap],
             [neu_seq, _, stim_seq, _], _, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_param=[[2,3,4,5], [standard], None, None, [0], [0]],
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
        # plot all.
        try: plot_cluster_features(axs[0])
        except Exception as e: print(e)
        try: plot_hierarchical_dendrogram(axs[1])
        except Exception as e: print(e)
        try: plot_glm_kernel(axs[2])
        except Exception as e: print(e)
        try: plot_standard(axs[3], 0)
        except Exception as e: print(e)
        try: plot_standard(axs[4], 1)
        except Exception as e: print(e)

    def plot_cluster_adapt_all(self, axs, cate=None):
        trials_around = 10
        cluster_id, neu_labels = get_cluster_cate(self.cluster_id, self.list_labels, self.list_significance, cate)
        split_idx = get_split_idx(self.list_labels, self.list_significance, cate)
        day_cluster_id = np.split(cluster_id, split_idx)
        # collect data.
        [color0, color1, color2, _], [_, _, stim_seq, _], _, [n_trials, n_neurons] = get_neu_trial(
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
            gap_margin = 5
            # collect data.
            [[_, color1, color2, _],
             [neu_trans_0to1, _, _, _, _], _, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_param=[None, None, None, None, [0], [0]],
                trial_idx=list_trans_0to1,
                mean_sem=False,
                cate=cate, roi_id=None)
            [[_, color1, color2, _],
             [neu_trans_1to0, _, _, _, _], _, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_param=[None, None, None, None, [0], [0]],
                trial_idx=list_trans_1to0,
                mean_sem=False,
                cate=cate, roi_id=None)
            # reshape into [n_transition*n_trials*n_neurons*n_times]
            neu_trans_0to1 = [neu.reshape([-1, 2*trials_around, neu.shape[1], neu.shape[2]]) for neu in neu_trans_0to1]
            neu_trans_1to0 = [neu.reshape([-1, 2*trials_around, neu.shape[1], neu.shape[2]]) for neu in neu_trans_1to0]
            neu_trans_0to1 = np.concatenate(neu_trans_0to1, axis=2)
            neu_trans_1to0 = np.concatenate(neu_trans_1to0, axis=2)
            # get average across transition [n_trials*n_neurons*n_times]
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
                neu = np.full((4*trials_around+gap_margin, e2-z), np.nan)
                neu[:trials_around, :e1-z] = np.nanmean(neu_0to1[0:trials_around, cluster_id==ci, z:e1], axis=1)
                neu[trials_around:2*trials_around, :e2-z] = np.nanmean(neu_0to1[trials_around:2*trials_around, cluster_id==ci, z:e2], axis=1)
                neu[2*trials_around+gap_margin:3*trials_around+gap_margin, :e2-z] = np.nanmean(neu_1to0[:trials_around, cluster_id==ci, z:e2], axis=1)
                neu[3*trials_around+gap_margin:, :e1-z] = np.nanmean(neu_1to0[trials_around:2*trials_around, cluster_id==ci, z:e1], axis=1)
                neu_x.append(neu)
            # define layouts.
            ax.axis('off')
            ax = ax.inset_axes([0, 0.1, 1, 0.85], transform=ax.transAxes)
            ax0 = ax.inset_axes([0.1, 0, 0.5, 1], transform=ax.transAxes)
            ax1 = ax.inset_axes([0.7, 0, 0.3, 1], transform=ax.transAxes)
            axs_hm = [ax0.inset_axes([0, ci/self.n_clusters, 1, 0.8/self.n_clusters], transform=ax0.transAxes)
                      for ci in range(self.n_clusters)]
            axs_cb = [ax1.inset_axes([0, ci/self.n_clusters, 1, 0.8/self.n_clusters], transform=ax1.transAxes)
                      for ci in range(self.n_clusters)]
            axs_hm.reverse()
            axs_cb.reverse()
            # plot results for each class.
            cmap, _ = get_cmap_color(1, base_color=color2, return_cmap=True)
            self.plot_cluster_heatmap_trial(axs_hm, axs_cb, neu_x, neu_time, cmap, norm_mode)
            # adjust layouts.
            for ci in range(self.n_clusters):
                axs_hm[ci].set_yticks([trials_around, 3*trials_around+gap_margin])
                axs_hm[ci].set_yticklabels(['L\u2192S', 'S\u2192L'])
                axs_hm[ci].set_ylim([0, 4*trials_around+gap_margin])
                if ci != self.n_clusters-1:
                    axs_hm[ci].set_xticklabels([])
            axs_hm[self.n_clusters-1].set_xlabel('time since stim (ms)')
            ax.set_title(f'sorted with {norm_mode}')
            hide_all_axis(ax)
            hide_all_axis(ax0)
            hide_all_axis(ax1)
        @show_resource_usage
        def plot_tansition(ax):
            xlim = [-7500,8000]
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
            ax.axis('off')
            ax = ax.inset_axes([0, 0, 1, 0.95], transform=ax.transAxes)
            ax0 = ax.inset_axes([0.1, 0, 0.4, 1], transform=ax.transAxes)
            ax1 = ax.inset_axes([0.6, 0, 0.4, 1], transform=ax.transAxes)
            axs0 = [ax0.inset_axes([0, 0.1+ci/self.n_clusters, 1, 0.8/self.n_clusters], transform=ax0.transAxes)
                    for ci in range(self.n_clusters)]
            axs1 = [ax1.inset_axes([0, 0.1+ci/self.n_clusters, 1, 0.8/self.n_clusters], transform=ax1.transAxes)
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
                    axs0[ci].axvline(0, color='red', lw=1, linestyle='--')
                    axs1[ci].axvline(0, color='red', lw=1, linestyle='--')
                    # plot neural traces.
                    z_idx, _ = get_frame_idx_from_time(neu_time, 0, 0, 0)
                    self.plot_mean_sem(
                        axs0[ci], neu_time[:z_idx],
                        neu_mean_0to1[:z_idx], neu_sem_0to1[:z_idx], color1, None)
                    self.plot_mean_sem(
                        axs0[ci], neu_time[z_idx:],
                        neu_mean_0to1[z_idx:], neu_sem_0to1[z_idx:], color2, None)
                    self.plot_mean_sem(
                        axs1[ci], neu_time[:z_idx],
                        neu_mean_1to0[:z_idx], neu_sem_1to0[:z_idx], color2, None)
                    self.plot_mean_sem(
                        axs1[ci], neu_time[z_idx:],
                        neu_mean_1to0[z_idx:], neu_sem_1to0[z_idx:], color1, None)
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
            ax.set_ylabel('df/f (z-scored)')
            ax0.set_title('S\u2192L adaptation')
            ax1.set_title('L\u2192S adaptation')
            hide_all_axis(ax)
            hide_all_axis(ax0)
            hide_all_axis(ax1)
        @show_resource_usage
        def plot_trial_quant(ax):
            win = 250
            trials_eval = 6
            # collect data.
            [[color0, color1, color2, _],
             [neu_trans_0to1, _, stim_seq_0to1, _], _,
             [n_trials, n_neurons]] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_idx=[l[1] for l in self.list_block_start],
                cate=cate, roi_id=None)
            [[color0, color1, color2, _],
             [neu_trans_1to0, _, stim_seq_1to0, _], _,
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
            m_01 = [np.array([quant_0to1[ci][si][1] for si in range(2*trials_eval)]) for ci in range(self.n_clusters)]
            m_11 = [np.array([quant_1to0[ci][si][1] for si in range(trials_eval)])   for ci in range(self.n_clusters)]
            m_10 = [np.array([quant_1to0[ci][si][1] for si in range(2*trials_eval)]) for ci in range(self.n_clusters)]
            m_00 = [np.array([quant_0to1[ci][si][1] for si in range(trials_eval)])   for ci in range(self.n_clusters)]
            s_01 = [np.array([quant_0to1[ci][si][2] for si in range(2*trials_eval)]) for ci in range(self.n_clusters)]
            s_11 = [np.array([quant_1to0[ci][si][2] for si in range(trials_eval)])   for ci in range(self.n_clusters)]
            s_10 = [np.array([quant_1to0[ci][si][2] for si in range(2*trials_eval)]) for ci in range(self.n_clusters)]
            s_00 = [np.array([quant_0to1[ci][si][2] for si in range(trials_eval)])   for ci in range(self.n_clusters)]
            # define layouts.
            ax.axis('off')
            ax = ax.inset_axes([0, 0, 1, 0.95], transform=ax.transAxes)
            ax0 = ax.inset_axes([0.1, 0, 0.4, 1], transform=ax.transAxes)
            ax1 = ax.inset_axes([0.6, 0, 0.4, 1], transform=ax.transAxes)
            axs_01 = [ax0.inset_axes([0, 0.1+ci/self.n_clusters, 0.6, 0.7/self.n_clusters], transform=ax0.transAxes)
                      for ci in range(self.n_clusters)]
            axs_11 = [ax0.inset_axes([0.7, 0.1+ci/self.n_clusters, 0.3, 0.7/self.n_clusters], transform=ax0.transAxes)
                      for ci in range(self.n_clusters)]
            axs_10 = [ax1.inset_axes([0, 0.1+ci/self.n_clusters, 0.6, 0.7/self.n_clusters], transform=ax1.transAxes)
                      for ci in range(self.n_clusters)]
            axs_00 = [ax1.inset_axes([0.7, 0.1+ci/self.n_clusters, 0.3, 0.7/self.n_clusters], transform=ax1.transAxes)
                      for ci in range(self.n_clusters)]
            axs_01.reverse()
            axs_11.reverse()
            axs_10.reverse()
            axs_00.reverse()
            # plot results for each class.
            for ci in range(self.n_clusters):
                if np.sum(cluster_id==ci) > 0:
                    # find bounds.
                    upper = np.nanmax(np.concatenate([m_01[ci], m_11[ci], m_10[ci], m_00[ci]]))
                    lower = np.nanmin(np.concatenate([m_01[ci], m_11[ci], m_10[ci], m_00[ci]]))
                    # short to long tansition.
                    for ti in np.arange(2*trials_eval):
                        axs_01[ci].errorbar(
                            ti-trials_eval, m_01[ci][ti], s_01[ci][ti], None,
                            color=color1 if ti < trials_eval else color2,
                            capsize=2, marker='o', linestyle='none',
                            markeredgecolor='white', markeredgewidth=0.1)
                    axs_01[ci].axvline(0, color='red', lw=1, linestyle='--')
                    # late long.
                    for ti in np.arange(trials_eval):
                        axs_11[ci].errorbar(
                            ti-trials_eval, m_11[ci][ti], s_11[ci][ti], None,
                            color=color2,
                            capsize=2, marker='o', linestyle='none',
                            markeredgecolor='white', markeredgewidth=0.1)
                    # long to short tansition.
                    for ti in np.arange(2*trials_eval):
                        axs_10[ci].errorbar(
                            ti-trials_eval, m_10[ci][ti], s_10[ci][ti], None,
                            color=color2 if ti < trials_eval else color1,
                            capsize=2, marker='o', linestyle='none',
                            markeredgecolor='white', markeredgewidth=0.1)
                    axs_10[ci].axvline(0, color='red', lw=1, linestyle='--')
                    # late short.
                    for ti in np.arange(trials_eval):
                        axs_00[ci].errorbar(
                            ti-trials_eval, m_00[ci][ti], s_00[ci][ti], None,
                            color=color1,
                            capsize=2, marker='o', linestyle='none',
                            markeredgecolor='white', markeredgewidth=0.1)
                    # adjust layouts.
                    axs_01[ci].yaxis.set_major_locator(mtick.MaxNLocator(nbins=2))
                    for axs in [axs_01, axs_10]:
                        axs[ci].set_xlim([-trials_eval-1, trials_eval])
                        axs[ci].set_xticks([])
                    for axs in [axs_11, axs_00]:
                        axs[ci].set_xlim([-trials_eval-1, 1])
                        axs[ci].spines['left'].set_visible(False)
                        axs[ci].set_xticks([])
                    for axs in [axs_11, axs_10, axs_00]:
                        axs[ci].set_yticks([])
                    for axs in [axs_01, axs_11, axs_10, axs_00]:
                        axs[ci].tick_params(direction='in')
                        axs[ci].spines['right'].set_visible(False)
                        axs[ci].spines['top'].set_visible(False) 
                        axs[ci].set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])    
            for axs in [axs_01, axs_10]:
                axs[self.n_clusters-1].tick_params(axis='x', labelrotation=90)
                axs[self.n_clusters-1].set_xticks(np.arange(-trials_eval, trials_eval, 2))
                axs[self.n_clusters-1].set_xticklabels(np.arange(-trials_eval, trials_eval, 2))
                axs[self.n_clusters-1].set_xlabel('early epoch')
            for axs in [axs_11, axs_00]:
                axs[self.n_clusters-1].tick_params(axis='x', labelrotation=90)
                axs[self.n_clusters-1].set_xticks(np.arange(-trials_eval, 0, 2))
                axs[self.n_clusters-1].set_xticklabels(np.arange(-trials_eval, 0, 2))
                axs[self.n_clusters-1].set_xlabel('late epoch')
            ax.set_xlabel('trial since block transition')
            ax.set_ylabel('df/f (z-scored)')
            ax0.set_title('S\u2192L adaptation')
            ax1.set_title('L\u2192S adaptation')
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
            ax = ax.inset_axes([0, 0, 1, 0.95], transform=ax.transAxes)
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
        def plot_epoch_decode(ax):
            win = 250
            trials_eval = 6
            # collect data.
            [[color0, color1, color2, _],
             [neu_trans_0to1, _, stim_seq_0to1, _], _,
             [n_trials, n_neurons]] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_idx=[l[1] for l in self.list_block_start],
                cate=cate, roi_id=None)
            [[color0, color1, color2, _],
             [neu_trans_1to0, _, stim_seq_1to0, _], _,
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
            short_e0 = [np.array([quant_1to0[ci][trials_eval+si][1] for si in range(trials_eval)]) for ci in range(self.n_clusters)]
            short_e1 = [np.array([quant_0to1[ci][            si][1] for si in range(trials_eval)]) for ci in range(self.n_clusters)]
            long_e0  = [np.array([quant_0to1[ci][trials_eval+si][1] for si in range(trials_eval)]) for ci in range(self.n_clusters)]
            long_e1  = [np.array([quant_1to0[ci][            si][1] for si in range(trials_eval)]) for ci in range(self.n_clusters)]
        @show_resource_usage
        def plot_standard_heatmap(ax, standard, norm_mode):
            win_sort = [-500, 500]
            # short standard.
            [[_, _, _, cmap],
             [neu_seq_short, _, stim_seq_short, _], _, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance,
                [exclude_odd_stim(sl) for sl in self.list_stim_labels],
                trial_param=[[2,3,4,5], [0], None, None, [0], [0]],
                cate=cate, roi_id=None)
            # long standard.
            [[_, _, _, cmap],
             [neu_seq_long, _, stim_seq_long, _], _, _] = get_neu_trial(
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
            # define layouts.
            ax.axis('off')
            ax = ax.inset_axes([0, 0, 1, 0.95], transform=ax.transAxes)
            ax0 = ax.inset_axes([0.1, 0, 0.4, 1], transform=ax.transAxes)
            ax1 = ax.inset_axes([0.6, 0, 0.4, 1], transform=ax.transAxes)
            axs0 = [ax0.inset_axes([0, 0.1+ci/self.n_clusters, 1, 0.8/self.n_clusters], transform=ax0.transAxes)
                    for ci in range(self.n_clusters)]
            axs1 = [ax1.inset_axes([0, 0.1+ci/self.n_clusters, 1, 0.8/self.n_clusters], transform=ax1.transAxes)
                    for ci in range(self.n_clusters)]
            axs0.reverse()
            axs1.reverse()
            # plot results for each class.
            for ci in range(self.n_clusters):
                if np.sum(cluster_id==ci) > 0:
                    if standard == 0:
                        neu_x = [neu_seq_short[cluster_id==ci,:], neu_seq_long[cluster_id==ci,:]]
                    if standard == 1:
                        neu_x = [neu_seq_long[cluster_id==ci,:], neu_seq_short[cluster_id==ci,:]]
                    self.plot_heatmap_neuron(
                        axs0[ci], neu_x[0], neu_time, neu_x[0], cmap, win_sort,
                        norm_mode=norm_mode)
                    self.plot_heatmap_neuron(
                        axs1[ci], neu_x[1], neu_time, neu_x[0], cmap, win_sort,
                        norm_mode=norm_mode)
                    # adjust layouts.
                    axs0[ci].tick_params(axis='y', labelrotation=0)
                    axs0[ci].set_xlabel(None)
                    axs0[ci].set_ylabel(None)
                    axs0[ci].yaxis.set_major_locator(mtick.MaxNLocator(nbins=2))
                    axs1[ci].set_xlabel(None)
                    axs1[ci].set_ylabel(None)
                    axs1[ci].set_yticklabels([])
                    if ci != self.n_clusters-1:
                        axs0[ci].set_xticks([])
                        axs1[ci].set_xticks([])
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
        except Exception as e: print(e)
        try: plot_tansition_trial_heatmap(axs[1], 'minmax')
        except Exception as e: print(e)
        try: plot_tansition_trial_heatmap(axs[2], 'share')
        except Exception as e: print(e)
        try: plot_tansition(axs[3])
        except Exception as e: print(e)
        try: plot_trial_quant(axs[4])
        except Exception as e: print(e)
        try: plot_win_mag_scatter_epoch(axs[5])
        except Exception as e: print(e)
        try: plot_standard_heatmap(axs[6], 0, 'none')
        except Exception as e: print(e)
        try: plot_standard_heatmap(axs[7], 0, 'minmax')
        except Exception as e: print(e)
        try: plot_standard_heatmap(axs[8], 0, 'share')
        except Exception as e: print(e)
        try: plot_standard_heatmap(axs[9], 1, 'none')
        except Exception as e: print(e)
        try: plot_standard_heatmap(axs[10], 1, 'minmax')
        except Exception as e: print(e)
        try: plot_standard_heatmap(axs[11], 1, 'share')
        except Exception as e: print(e)
        try: plot_legend(axs[12])
        except Exception as e: print(e)

    def plot_oddball_latent_all(self, axs, oddball, cate):
        d_latent = 3
        # collect data.
        xlim = [-5000, 6000]
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
        model = PCA(n_components=d_latent)
        neu_z = model.fit_transform(
            neu_x.reshape(neu_x.shape[0],-1).T
            ).T.reshape(d_latent, -1)
        def plot_3d_dynamics(ax):
            # plot dynamics.
            self.plot_3d_latent_dynamics(ax, neu_z, stim_seq, neu_time, c_stim)
            # mark unexpected event.
            idx_unexpect = get_frame_idx_from_time(
                neu_time, 0, stim_seq[c_idx+1,0], stim_seq[c_idx,1]+self.expect[1-oddball])[oddball]
            ax.scatter(neu_z[0,idx_unexpect], neu_z[1,idx_unexpect], neu_z[2,idx_unexpect], color='gold', marker='o', lw=5)
            # adjust layouts.
            adjust_layout_3d_latent(ax, neu_z, self.latent_cmap, neu_time, 'time since pre oddball stim (ms)')
        def plot_mean_dynamics(ax):
            neu_mean, neu_sem = get_mean_sem(neu_x)
            c_neu = get_cmap_color(neu_mean.shape[0], cmap=self.latent_cmap)
            # find bounds.
            upper = np.nanmax(neu_mean) + np.nanmax(neu_sem)
            lower = np.nanmin(neu_mean) - np.nanmax(neu_sem)
            # plot stimulus.
            for si in range(stim_seq.shape[0]):
                ax.fill_between(
                    stim_seq[si,:],
                    lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
                    color=color0, edgecolor='none', alpha=0.25, step='mid')
            if oddball == 0:
                ax.axvline(stim_seq[c_idx+1,0], color='gold', lw=1, linestyle='--')
            if oddball == 1:
                ax.axvline(stim_seq[c_idx,1]+self.expect[1-oddball], color='gold', lw=1, linestyle='--')
            # plot neural traces.
            self.plot_mean_sem(ax, neu_time, neu_mean, neu_sem, color0, None)
            for t in range(neu_mean.shape[0]-1):
                ax.plot(neu_time[t:t+2], neu_mean[t:t+2], color=c_neu[t])
            # adjust layouts.
            adjust_layout_neu(ax)
            ax.set_xlim(xlim)
            ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
            ax.set_xlabel('time since pre oddball stim (ms)')
        # plot all.
        try: plot_3d_dynamics(axs[0])
        except Exception as e: print(e)
        try: plot_mean_dynamics(axs[1])
        except Exception as e: print(e)

    def plot_transition_latent_all(self, axs, block, cate):
        d_latent = 3
        # collect data.
        xlim = [-5000, 6000]
        l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, xlim[0], xlim[1])
        [[color0, color1, color2, _],
         [neu_x, _, stim_seq, _], _,
         [n_trials, n_neurons]] = get_neu_trial(
            self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
            trial_idx=[l[block] for l in self.list_block_start],
            trial_param=[None, None, [0], None, [0], [0]],
            cate=cate, roi_id=None)
        neu_x = neu_x[:,l_idx:r_idx]
        neu_time = self.alignment['neu_time'][l_idx:r_idx]
        cmap, c_neu = get_cmap_color(
            neu_x.shape[1], base_color=[color1, color2][block],
            return_cmap=True)
        c_stim = [color0] * stim_seq.shape[-2]
        # fit model.
        model = PCA(n_components=d_latent)
        neu_z = model.fit_transform(
            neu_x.reshape(neu_x.shape[0],-1).T
            ).T.reshape(d_latent, -1)
        def plot_3d_dynamics(ax):
            # plot dynamics.
            self.plot_3d_latent_dynamics(ax, neu_z, stim_seq, neu_time, c_stim)
            # mark unexpected event.
            idx_unexpect = get_frame_idx_from_time(neu_time, 0, 0, 0)[0]
            ax.scatter(neu_z[0,idx_unexpect], neu_z[1,idx_unexpect], neu_z[2,idx_unexpect], color='red', marker='o', lw=5)
            # adjust layouts.
            adjust_layout_3d_latent(ax, neu_z, self.latent_cmap, neu_time, 'time since pre oddball stim (ms)')
        def plot_mean_dynamics(ax):
            neu_mean, neu_sem = get_mean_sem(neu_x)
            c_neu = get_cmap_color(neu_mean.shape[0], cmap=self.latent_cmap)
            # find bounds.
            upper = np.nanmax(neu_mean) + np.nanmax(neu_sem)
            lower = np.nanmin(neu_mean) - np.nanmax(neu_sem)
            # plot stimulus.
            for si in range(stim_seq.shape[0]):
                ax.fill_between(
                    stim_seq[si,:],
                    lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
                    color=color0, edgecolor='none', alpha=0.25, step='mid')
            ax.axvline(0, color='red', lw=1, linestyle='--')
            # plot neural traces.
            self.plot_mean_sem(ax, neu_time, neu_mean, neu_sem, color0, None)
            for t in range(neu_mean.shape[0]-1):
                ax.plot(neu_time[t:t+2], neu_mean[t:t+2], color=c_neu[t])
            # adjust layouts.
            adjust_layout_neu(ax)
            ax.set_xlim(xlim)
            ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
            ax.set_xlabel('time since pre oddball stim (ms)')
        # plot all.
        try: plot_3d_dynamics(axs[0])
        except Exception as e: print(e)
        try: plot_mean_dynamics(axs[1])
        except Exception as e: print(e)

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
                axs[0].set_title(f'GLM kernels \n {label_name}')
                axs[1].set_title(f'response to binned interval in random \n {label_name}')
                axs[2].set_title(f'response to binned interval in random \n {label_name}')
                axs[3].set_title(f'response to short standard interval \n {label_name}')
                axs[4].set_title(f'response to long standard interval \n {label_name}')
                axs[5].set_title(f'latent dynamics on standard interval \n {label_name}')
                axs[6].set_title(f'response to short oddball interval \n {label_name}')
                axs[7].set_title(f'response to long oddball interval \n {label_name}')
                axs[8].set_title(f'fraction of neurons in cluster \n {label_name}')
                axs[9].set_title(f'fraction of cell-types in cluster \n {label_name}')
                axs[10].set_title(f'legend \n {label_name}')

            except Exception as e: print(e)

    def cluster_heatmap_all(self, axs_all):
        for cate, axs in zip(self.cate_list, axs_all):
            try:
                label_name = self.label_names[str(cate[0])] if len(cate)==1 else 'all'
                print(f'plotting results for {label_name}')

                self.plot_cluster_heatmap_all(axs, cate=cate)
                axs[0].set_title(f'clustered latent features \n {label_name}')
                axs[1].set_title(f'cluster dendrogram \n {label_name}')
                axs[2].set_title(f'GLM kernel course \n {label_name}')
                axs[3].set_title(f'response to short standard interval \n {label_name}')
                axs[4].set_title(f'response to long standard interval \n {label_name}')

            except Exception as e: print(e)

    def cluster_adapt_all(self, axs_all):
        for cate, axs in zip(self.cate_list, axs_all):
            try:
                label_name = self.label_names[str(cate[0])] if len(cate)==1 else 'all'
                print(f'plotting results for {label_name}')

                self.plot_cluster_adapt_all(axs, cate=cate)
                axs[ 0].set_title(f'single trial heatmap \n {label_name}')
                axs[ 1].set_title(f'single trial heatmap \n {label_name}')
                axs[ 2].set_title(f'single trial heatmap \n {label_name}')
                axs[ 3].set_title(f'response to block transition \n {label_name}')
                axs[ 4].set_title(f'response to block transition \n {label_name}')
                axs[ 5].set_title(f'comparison for first&last trial \n {label_name}')
                axs[ 6].set_title(f'response to standard (sorted by short) \n {label_name}')
                axs[ 7].set_title(f'response to standard (sorted by short) \n {label_name}')
                axs[ 8].set_title(f'response to standard (sorted by short) \n {label_name}')
                axs[ 9].set_title(f'response to standard (sorted by long) \n {label_name}')
                axs[10].set_title(f'response to standard (sorted by long) \n {label_name}')
                axs[11].set_title(f'response to standard (sorted by long) \n {label_name}')
                axs[12].set_title(f'legend \n {label_name}')
            except Exception as e: print(e)
    
    def latent_all(self, axs_all):
        for cate, axs in zip(self.cate_list, axs_all):
            try:
                label_name = self.label_names[str(cate[0])] if len(cate)==1 else 'all'
                print(f'plotting results for {label_name}')

                self.plot_oddball_latent_all(axs[0], 0, cate=cate)
                axs[0][0].set_title(f'latent dynamics around short oddball interval \n {label_name}')
                axs[0][1].set_title(f'response to short oddball interval \n {label_name}')
                
                self.plot_oddball_latent_all(axs[1], 1, cate=cate)
                axs[1][0].set_title(f'latent dynamics around long oddball interval \n {label_name}')
                axs[1][1].set_title(f'response to long oddball interval \n {label_name}')
                
                self.plot_transition_latent_all(axs[2], 0, cate=cate)
                axs[2][0].set_title(f'latent dynamics around short to long block transition \n {label_name}')
                axs[2][1].set_title(f'response to short to long block transition \n {label_name}')
                
                self.plot_transition_latent_all(axs[3], 1, cate=cate)
                axs[3][0].set_title(f'latent dynamics around long to short block transition \n {label_name}')
                axs[3][1].set_title(f'response to long to short block transition \n {label_name}')

            except Exception as e: print(e)
