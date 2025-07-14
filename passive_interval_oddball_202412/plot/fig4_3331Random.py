#!/usr/bin/env python3

import numpy as np
import matplotlib.colors as mcolors
import matplotlib.ticker as mtick
from sklearn.decomposition import PCA

from modules.Alignment import run_get_stim_response
from modeling.clustering import get_mean_sem_cluster
from modeling.clustering import get_bin_mean_sem_cluster
from modeling.clustering import get_cluster_cate
from modeling.decoding import fit_poly_line
from modeling.decoding import neu_pop_sample_decoding_slide_win
from modeling.generative import get_glm_cate
from modeling.quantifications import run_quantification
from utils import show_resource_usage
from utils import get_norm01_params
from utils import get_mean_sem_win
from utils import get_mean_sem
from utils import get_neu_trial
from utils import get_frame_idx_from_time
from utils import get_isi_bin_neu
from utils import get_expect_interval
from utils import get_split_idx
from utils import get_roi_label_color
from utils import get_cmap_color
from utils import apply_colormap
from utils import hide_all_axis
from utils import adjust_layout_neu
from utils import adjust_layout_heatmap
from utils import adjust_layout_2d_latent
from utils import add_legend
from utils import add_heatmap_colorbar
from utils import utils_basic

# fig, ax = plt.subplots(1, 1, figsize=(5, 5))
# fig, ax = plt.subplots(1, 1, figsize=(3, 9))
# fig, axs = plt.subplots(1, 8, figsize=(24, 3))
# axs = [plt.subplots(1, 8, figsize=(24, 3))[1], plt.subplots(1, 8, figsize=(24, 3))[1]]
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
        self.expect = np.array(np.mean([get_expect_interval(sl)[0] for sl in self.list_stim_labels]))
        self.list_significance = list_significance
        self.bin_win = [450,2550]
        self.bin_num = 4
        self.n_clusters = 7
        self.max_clusters = 10
        self.d_latent = 3
        self.glm = self.run_glm()
        self.cluster_id = self.run_clustering()

    def plot_cluster_interval_bin_all(self, axs, cate=None):
        color0 = 'dimgrey'
        color1 = 'chocolate'
        color2 = 'crimson'
        kernel_all = get_glm_cate(self.glm, self.list_labels, self.list_significance, cate)
        cluster_id, neu_labels = get_cluster_cate(self.cluster_id, self.list_labels, self.list_significance, cate)
        # collect data.
        [_, [neu_seq, stim_seq, camera_pupil, pre_isi, post_isi], _,
         [n_trials, n_neurons]] = get_neu_trial(
            self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
            trial_param=[[2,3,4,5], None, None, None, [1], [0]],
            mean_sem=False,
            cate=cate, roi_id=None)
        colors = get_cmap_color(self.bin_num, base_color=color2)
        c_idx = np.nanmean(np.concatenate(stim_seq, axis=0), axis=0).shape[0]//2
        @show_resource_usage
        def plot_glm_kernel(ax):
            ax.axis('off')
            ax = ax.inset_axes([0, 0, 0.5, 1], transform=ax.transAxes)
            # get cluster average.
            glm_mean, glm_sem = get_mean_sem_cluster(kernel_all, self.n_clusters, cluster_id)
            # plot results.
            norm_params = [get_norm01_params(glm_mean[i,:]) for i in range(self.n_clusters)]
            self.plot_cluster_mean_sem(
                ax, glm_mean, glm_sem, self.glm['kernel_time'],
                norm_params, np.nanmean(np.concatenate(stim_seq, axis=0), axis=0)[c_idx,:].reshape(1,-1),
                [color0], [color0]*self.n_clusters,
                [np.nanmin(self.glm['kernel_time']), np.nanmax(self.glm['kernel_time'])])
            # adjust layouts.
            ax.set_xlabel('time since stim (ms)')
        @show_resource_usage
        def plot_interval_bin(ax, mode):
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
        def plot_interval_scaling(ax):
            gap = 2
            bin_num = 50
            win_eval = [[-2500,0], [0, 250]]
            order = 2
            # bin data based on isi.
            [_, bin_center, _, bin_neu_seq, _, _, _, _] = get_isi_bin_neu(
                neu_seq, stim_seq, camera_pupil, pre_isi, self.bin_win, bin_num)
            bin_center = bin_center[gap:-gap]
            # get response within cluster.
            bin_win_baseline = [np.array([get_mean_sem_win(
                bns[cluster_id==ci,:], self.alignment['neu_time'], 0, win_eval[0][0], win_eval[0][1], 'lower')[1]
                for bns in bin_neu_seq])[gap:-gap] for ci in range(self.n_clusters)]
            bin_win_evoked = [np.array([get_mean_sem_win(
                bns[cluster_id==ci,:], self.alignment['neu_time'], 0, win_eval[1][0], win_eval[1][1], 'mean')[1]
                for bns in bin_neu_seq])[gap:-gap] for ci in range(self.n_clusters)]
            bin_win_neu_seq = [bwe - bwb for bwe, bwb in zip (bin_win_evoked, bin_win_baseline)]
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
                    upper = np.nanmax([bin_win_neu_seq[ci], bin_win_neu_seq[ci]])
                    lower = np.nanmin([bin_win_neu_seq[ci], bin_win_neu_seq[ci]])
                    # plot scatter.
                    axs[ci].scatter(bin_center, bin_win_neu_seq[ci], color=color0, s=2, alpha=0.5)
                    # fit line.
                    y_pred, mape = fit_poly_line(bin_center, bin_win_neu_seq[ci], order)
                    # plot line.
                    axs[ci].plot(bin_center, y_pred, color=color0, lw=2)
                    # plot goodness.
                    axs[ci].text(
                        bin_center[-1]*1.2, upper-0.4*(upper-lower), f'{mape:.3f}', color=color0,
                        ha='center', va='center')
                    # adjust layouts.
                    axs[ci].tick_params(direction='in')
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
            ax.set_ylabel(f'evoked magnitude in the {win_eval[1]} window with MAPE')
            hide_all_axis(ax)
        @show_resource_usage
        def plot_interval_bin_latent(ax):
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
        def plot_neu_fraction(ax):
            self.plot_cluster_neu_fraction_in_cluster(ax, cluster_id, color0)
        @show_resource_usage
        def plot_fraction(ax):
            self.plot_cluster_cate_fraction_in_cluster(ax, cluster_id, neu_labels, self.label_names, color0)
        @show_resource_usage
        def plot_legend(ax):
            [bins, _, _, _, _, _, _, _] = get_isi_bin_neu(
                neu_seq, stim_seq, camera_pupil, pre_isi, self.bin_win, self.bin_num)
            lbl = ['[{},{}] ms'.format(int(bins[i]),int(bins[i+1])) for i in range(self.bin_num)]
            lbl+= [v for v in self.label_names.values()] + ['all cell-types']
            cs = colors + [get_roi_label_color(cate=[int(k)])[2] for k in self.label_names.keys()]
            cs+= [get_roi_label_color(cate=[-1,1,2])[2]]
            add_legend(ax, cs, lbl, n_trials, n_neurons, self.n_sess, 'upper right')
            ax.axis('off')
        # plot all.
        try: plot_glm_kernel(axs[0])
        except Exception as e: print(e)
        try: plot_interval_bin(axs[1], 'pre')
        except Exception as e: print(e)
        try: plot_interval_bin(axs[2], 'post')
        except Exception as e: print(e)
        try: plot_interval_scaling(axs[3])
        except Exception as e: print(e)
        try: plot_interval_bin_latent(axs[4])
        except Exception as e: print(e)
        try: plot_neu_fraction(axs[5])
        except Exception as e: print(e)
        try: plot_fraction(axs[6])
        except Exception as e: print(e)
        try: plot_legend(axs[7])
        except Exception as e: print(e)
    
    def plot_cluster_heatmap_all(self, axs, cate):
        kernel_all = get_glm_cate(self.glm, self.list_labels, self.list_significance, cate)
        cluster_id, neu_labels = get_cluster_cate(self.cluster_id, self.list_labels, self.list_significance, cate)
        # collect data.
        [[_, _, _, cmap], [_, _, stim_seq, _], _, _] = get_neu_trial(
            self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
            trial_param=[[2,3,4,5], None, None, None, [1], [0]],
            cate=cate, roi_id=None)
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
            # plot results.
            self.plot_dendrogram(ax, kernel_all, cmap)
        @show_resource_usage
        def plot_glm_kernel(ax):
            ax.axis('off')
            ax = ax.inset_axes([0, 0, 0.5, 1], transform=ax.transAxes)
            c_idx = stim_seq.shape[0]//2
            # plot results.
            self.plot_cluster_heatmap(ax, kernel_all, self.glm['kernel_time'], cluster_id, 'minmax', cmap)
            # adjust layouts.
            ax.set_xlabel('time since stim (ms)')
            ax.axvline(stim_seq[c_idx,0], color='black', lw=1, linestyle='--')
        # plot all.
        try: plot_cluster_features(axs[0])
        except Exception as e: print(e)
        try: plot_hierarchical_dendrogram(axs[1])
        except Exception as e: print(e)
        try: plot_glm_kernel(axs[2])
        except Exception as e: print(e)
    
    def plot_cross_sess_adapt(self, axs, cate):
        epoch_len = 100
        cluster_id, neu_labels = get_cluster_cate(self.cluster_id, self.list_labels, self.list_significance, cate)
        split_idx = get_split_idx(self.list_labels, self.list_significance, cate)
        day_cluster_id = np.split(cluster_id, split_idx)
        [[color0, color1, color2, _],
         [_, _, stim_seq, _], _, _] = get_neu_trial(
            self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
            trial_param=[[2,3,4,5], None, None, None, [1], [0]],
            cate=cate, roi_id=None)
        c_idx = stim_seq.shape[0]//2
        @show_resource_usage
        def plot_dist_cluster_fraction(ax):
            bar_width = 0.5
            # get fraction in each category.
            fraction = np.zeros((len(split_idx)+1, self.n_clusters))
            for di in range(len(split_idx)+1):
                for ci in range(self.n_clusters):
                    nc = np.nansum(day_cluster_id[di]==ci)
                    nt = len(day_cluster_id[di]) + 1e-5
                    fraction[di,ci] = nc / nt
            # define layouts.
            ax.axis('off')
            ax1 = ax.inset_axes([0, 0, 1, 0.95], transform=ax.transAxes)
            axs = [ax1.inset_axes([0.2, ci/self.n_clusters, 0.5, 0.75/self.n_clusters], transform=ax1.transAxes)
                      for ci in range(self.n_clusters)]
            axs.reverse()
            # plot results for each class.
            for ci in range(self.n_clusters):
                if np.sum(cluster_id==ci) > 0:
                    for di in range(len(split_idx)+1):
                        axs[ci].bar(
                            di, fraction[di,ci],
                            bottom=0, edgecolor='white', width=bar_width, color=color0)
            # adjust layouts.
            for ci in range(self.n_clusters):
                axs[ci].tick_params(tick1On=False)
                axs[ci].spines['top'].set_visible(False)
                axs[ci].spines['left'].set_visible(False)
                axs[ci].spines['right'].set_visible(False)
                axs[ci].grid(True, axis='y', linestyle='--')
                axs[ci].set_ylim([0,np.nanmax(fraction[:,ci])+0.05])
                axs[ci].set_xticks([])
                axs[ci].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
                axs[ci].yaxis.set_major_locator(mtick.MaxNLocator(nbins=2))
            axs[self.n_clusters-1].set_xticks(np.arange(len(split_idx)+1))
            axs[self.n_clusters-1].set_xticklabels(
                [f'{di+1}' for di in range(len(split_idx)+1)])
            axs[self.n_clusters-1].set_xlabel('day')
            hide_all_axis(ax1)
        @show_resource_usage
        def plot_cross_epoch(ax):
            xlim = [-1000, 4000]
            # collect data.
            [_, [neu_seq_0, stim_seq_0, camera_pupil_0, _, isi_0], _, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_param=[[2,3,4,5], None, None, None, [1], [0]],
                trial_idx=[np.concatenate([np.ones(epoch_len), np.zeros(sl.shape[0]-epoch_len)]).astype('bool')
                           for sl in self.list_stim_labels],
                mean_sem=False, cate=cate, roi_id=None)
            [_, [neu_seq_1, stim_seq_1, camera_pupil_1, _, isi_1], _, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_param=[[2,3,4,5], None, None, None, [1], [0]],
                trial_idx=[np.concatenate([np.zeros(sl.shape[0]-epoch_len), np.ones(epoch_len)]).astype('bool')
                           for sl in self.list_stim_labels],
                mean_sem=False, cate=cate, roi_id=None)
            colors = get_cmap_color(self.bin_num, base_color=color2)
            # bin data based on isi.
            bin_results_0 = [get_isi_bin_neu(
                [neu_seq_0[di]], [stim_seq_0[di]], [camera_pupil_0[di]], [isi_0[di]], self.bin_win, 2)
                for di in range(len(day_cluster_id))]
            bin_results_1 = [get_isi_bin_neu(
                [neu_seq_1[di]], [stim_seq_1[di]], [camera_pupil_1[di]], [isi_1[di]], self.bin_win, 2)
                for di in range(len(day_cluster_id))]
            bin_neu_seq_0 = [bin_results_0[di][3] for di in range(len(day_cluster_id))]
            bin_neu_seq_1 = [bin_results_1[di][3] for di in range(len(day_cluster_id))]
            bin_stim_seq_0 = [bin_results_0[di][6] for di in range(len(day_cluster_id))]
            bin_stim_seq_1 = [bin_results_1[di][6] for di in range(len(day_cluster_id))]
            # get response within cluster at each bin.
            cluster_bin_neu_0 = [
                get_bin_mean_sem_cluster(bin_neu_seq_0[di], self.n_clusters, day_cluster_id[di])
                for di in range(len(day_cluster_id))]
            cluster_bin_neu_1 = [
                get_bin_mean_sem_cluster(bin_neu_seq_1[di], self.n_clusters, day_cluster_id[di])
                for di in range(len(day_cluster_id))]
            cluster_bin_neu_mean_0 = [cluster_bin_neu_0[di][0] for di in range(len(day_cluster_id))]
            cluster_bin_neu_mean_1 = [cluster_bin_neu_1[di][0] for di in range(len(day_cluster_id))]
            cluster_bin_neu_sem_0 = [cluster_bin_neu_0[di][1] for di in range(len(day_cluster_id))]
            cluster_bin_neu_sem_1 = [cluster_bin_neu_1[di][1] for di in range(len(day_cluster_id))]
            norm_params = [
                get_norm01_params(np.concatenate(cluster_bin_neu_mean_0 + cluster_bin_neu_mean_1, axis=0)[:,ci,:])
                for ci in range(self.n_clusters)]
            # define layout.
            ax.axis('off')
            ax0 = ax.inset_axes([0, 0, 0.45, 1], transform=ax.transAxes)
            ax1 = ax.inset_axes([0.5, 0, 0.45, 1], transform=ax.transAxes)
            axs0 = [ax0.inset_axes([di/len(day_cluster_id), 0, 0.8/len(day_cluster_id), 0.95],
                                   transform=ax0.transAxes) for di in range(len(day_cluster_id))]
            axs1 = [ax1.inset_axes([di/len(day_cluster_id), 0, 0.8/len(day_cluster_id), 0.95],
                                   transform=ax1.transAxes) for di in range(len(day_cluster_id))]
            # plot results for each day.
            for di in range(len(day_cluster_id)):
                for bi, axs in zip([0,1], [axs0, axs1]):
                    # plot stimulus.
                    axs[di].fill_between(
                        np.nanmean(bin_stim_seq_0[di], axis=0)[c_idx,:],
                        0, self.n_clusters,
                        color=color0, edgecolor='none', alpha=0.25, step='mid')
                    # plot response.
                    scale_bar=True if di==0 and bi==0 else False
                    self.plot_cluster_mean_sem(
                        axs[di], cluster_bin_neu_mean_0[di][bi,:,:], cluster_bin_neu_sem_0[di][bi,:,:],
                        self.alignment['neu_time'], norm_params,
                        None, None, [color0]*self.n_clusters, xlim, scale_bar)
                    self.plot_cluster_mean_sem(
                        axs[di], cluster_bin_neu_mean_1[di][bi,:,:], cluster_bin_neu_sem_1[di][bi,:,:],
                        self.alignment['neu_time'], norm_params,
                        bin_stim_seq_1[di][bi, c_idx+1, :].reshape(1,-1),
                        [colors[bi]], [colors[bi]]*self.n_clusters, xlim, False)
                    # adjust layouts.
                    axs[di].set_title(f'day {di+1}')
            ax.set_xlabel('time since stim (ms)')
            ax0.set_title('short bin')
            ax1.set_title('long bin')
            hide_all_axis(ax0)
            hide_all_axis(ax1)
        @show_resource_usage
        def plot_interval_scaling(ax):
            gap = 2
            bin_num = 25
            win_eval = [[-2500,0], [0, 250]]
            order = 2
            # collect data.
            [[color0, _, color2, _],
             [neu_seq, stim_seq, camera_pupil, pre_isi, _], _,
             [n_trials, n_neurons]] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_param=[[2,3,4,5], None, None, None, [1], [0]],
                mean_sem=False,
                cate=cate, roi_id=None)
            # bin data based on isi.
            [_, bin_center, bin_neu_seq_trial, _, _, _, _, _] = get_isi_bin_neu(
                neu_seq, stim_seq, camera_pupil, pre_isi, self.bin_win, bin_num)
            bin_center = bin_center[gap:-gap]
            bin_neu_seq_0 = [np.nanmean(bin_neu_seq_trial[bi][0], axis=0) for bi in range(bin_num)]
            bin_neu_seq_1 = [np.nanmean(bin_neu_seq_trial[bi][-1], axis=0) for bi in range(bin_num)]
            # get response within cluster.
            bin_win_0_baseline = [np.array([get_mean_sem_win(
                bns0[day_cluster_id[0]==ci,:], self.alignment['neu_time'], 0, win_eval[0][0], win_eval[0][1], 'lower')[1]
                for bns0 in bin_neu_seq_0])[gap:-gap] for ci in range(self.n_clusters)]
            bin_win_1_baseline = [np.array([get_mean_sem_win(
                bns1[day_cluster_id[-1]==ci,:], self.alignment['neu_time'], 0, win_eval[0][0], win_eval[0][1], 'lower')[1]
                for bns1 in bin_neu_seq_1])[gap:-gap] for ci in range(self.n_clusters)]
            bin_win_0_evoked = [np.array([get_mean_sem_win(
                bns0[day_cluster_id[0]==ci,:], self.alignment['neu_time'], 0, win_eval[1][0], win_eval[1][1], 'mean')[1]
                for bns0 in bin_neu_seq_0])[gap:-gap] for ci in range(self.n_clusters)]
            bin_win_1_evoked = [np.array([get_mean_sem_win(
                bns1[day_cluster_id[-1]==ci,:], self.alignment['neu_time'], 0, win_eval[1][0], win_eval[1][1], 'mean')[1]
                for bns1 in bin_neu_seq_1])[gap:-gap] for ci in range(self.n_clusters)]
            bin_win_0 = [bw0e - bw0b for bw0e, bw0b in zip (bin_win_0_evoked, bin_win_0_baseline)]
            bin_win_1 = [bw1e - bw1b for bw1e, bw1b in zip (bin_win_1_evoked, bin_win_1_baseline)]
            # define layouts.
            ax.axis('off')
            ax = ax.inset_axes([0, 0, 1, 0.95], transform=ax.transAxes)
            axs = [ax.inset_axes([0.1, 0.1+ci/self.n_clusters, 0.4, 0.7/self.n_clusters], transform=ax.transAxes)
                      for ci in range(self.n_clusters)]
            axs.reverse()
            # plot results for each class.
            for ci in range(self.n_clusters):
                if np.sum(day_cluster_id[0]==ci) > 0 and np.sum(day_cluster_id[-1]==ci) > 0:
                    # find bounds.
                    upper = np.nanmax([bin_win_0[ci], bin_win_1[ci]])
                    lower = np.nanmin([bin_win_0[ci], bin_win_1[ci]])
                    # plot scatter.
                    axs[ci].scatter(bin_center, bin_win_0[ci], color=color0, s=2, alpha=0.5)
                    axs[ci].scatter(bin_center, bin_win_1[ci], color=color2, s=2, alpha=0.5)
                    # fit line.
                    y_pred_0, mape_0 = fit_poly_line(bin_center, bin_win_0[ci], order)
                    y_pred_1, mape_1 = fit_poly_line(bin_center, bin_win_1[ci], order)
                    # plot line.
                    axs[ci].plot(bin_center, y_pred_0, color=color0, lw=2)
                    axs[ci].plot(bin_center, y_pred_1, color=color2, lw=2)
                    # plot goodness.
                    axs[ci].text(
                        bin_center[-1]*1.2, upper-0.4*(upper-lower), f'{mape_0:.3f}', color=color0,
                        ha='center', va='center')
                    axs[ci].text(
                        bin_center[-1]*1.2, upper-0.6*(upper-lower), f'{mape_1:.3f}', color=color2,
                        ha='center', va='center')
                    # plot change indicator.
                    arrow = '\u2191' if mape_1>mape_0 else '\u2193'
                    axs[ci].text(
                        bin_center[-1]*1.2, upper-0.9*(upper-lower), arrow, color='black',
                        ha='center', va='center')
                    # adjust layouts.
                    axs[ci].tick_params(direction='in')
                    axs[ci].spines['right'].set_visible(False)
                    axs[ci].spines['top'].set_visible(False)
                    axs[ci].set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
                    axs[ci].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
                    axs[ci].xaxis.set_major_locator(mtick.MaxNLocator(nbins=2))
                    axs[ci].yaxis.set_major_locator(mtick.MaxNLocator(nbins=2))
                    if ci != self.n_clusters-1:
                        axs[ci].set_xticklabels([])
            axs[self.n_clusters-1].set_xlabel('preceding interval (ms)')
            ax.set_ylabel(f'evoked magnitude in the {win_eval[1]} window with MAPE')
            hide_all_axis(ax)
        @show_resource_usage
        def plot_legend(ax):
            [[color0, color1, color2, _],
             [_, _, stim_seq, _], _,
             [n_trials, n_neurons]] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_param=[[2,3,4,5], None, None, None, [1], [0]],
                cate=cate, roi_id=None)
            cs = [color0, color2]
            lbl = ['first epoch', 'last epoch']
            add_legend(ax, cs, lbl, n_trials, n_neurons, self.n_sess, 'upper right')
            ax.axis('off')
        # plot all.
        try: plot_dist_cluster_fraction(axs[0])
        except Exception as e: print(e)
        try: plot_cross_epoch(axs[1])
        except Exception as e: print(e)
        try: plot_interval_scaling(axs[2])
        except Exception as e: print(e)
        try: plot_legend(axs[3])
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
                
                self.plot_cluster_interval_bin_all(axs, cate=cate)
                axs[0].set_title(f'GLM kernels \n {label_name}')
                axs[1].set_title(f'reponse to binned (pre) interval \n {label_name}')
                axs[2].set_title(f'reponse to binned (post) interval \n {label_name}')
                axs[3].set_title(f'response to interval regression \n {label_name}')
                axs[4].set_title(f'latent dynamics with binned interval \n {label_name}')
                axs[5].set_title(f'fraction of neurons in cluster \n {label_name}')
                axs[6].set_title(f'fraction of cell-types in cluster \n {label_name}')
                axs[7].set_title(f'legend \n {label_name}')
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

            except Exception as e: print(e)
    
    def cluster_individual_pre(self, axs_all):
        for cate, axs in zip(self.cate_list, axs_all):
            try:
                label_name = self.label_names[str(cate[0])] if len(cate)==1 else 'all'
                print(f'plotting results for {label_name}')
                
                self.plot_cluster_interval_bin_individual(axs, 'pre', cate=cate)

            except Exception as e: print(e)
    
    def cluster_individual_post(self, axs_all):
        for cate, axs in zip(self.cate_list, axs_all):
            try:
                label_name = self.label_names[str(cate[0])] if len(cate)==1 else 'all'
                print(f'plotting results for {label_name}')
                
                self.plot_cluster_interval_bin_individual(axs, 'post', cate=cate)

            except Exception as e: print(e)
    
    def separability_local(self, axs_all):
        for cate, axs in zip(self.cate_list, axs_all):
            try:
                label_name = self.label_names[str(cate[0])] if len(cate)==1 else 'all'
                print(f'plotting results for {label_name}')
                
                self.plot_separability_local(axs, cate=cate)

            except Exception as e: print(e)

    def cross_sess_adapt(self, axs_all):
        for cate, axs in zip(self.cate_list, axs_all):
            try:
                label_name = self.label_names[str(cate[0])] if len(cate)==1 else 'all'
                print(f'plotting results for {label_name}')
                
                self.plot_cross_sess_adapt(axs, cate=cate)

            except Exception as e: print(e)
        