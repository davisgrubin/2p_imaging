#!/usr/bin/env python3

import traceback
import numpy as np
import matplotlib.ticker as mtick
from sklearn.decomposition import PCA

from modules.Alignment import run_get_stim_response
from modeling.clustering import get_mean_sem_cluster
from modeling.clustering import get_bin_mean_sem_cluster
from modeling.clustering import get_cluster_cate
from modeling.decoding import fit_poly_line
from modeling.generative import get_glm_cate
from utils import show_resource_usage
from utils import get_norm01_params
from utils import get_mean_sem_win
from utils import get_mean_sem
from utils import get_peak_time
from utils import get_neu_trial
from utils import get_frame_idx_from_time
from utils import get_isi_bin_neu
from utils import get_expect_interval
from utils import get_split_idx
from utils import get_cmap_color
from utils import hide_all_axis
from utils import get_random_rotate_mat_3d
from utils import adjust_layout_isi_example_epoch
from utils import adjust_layout_2d_latent
from utils import adjust_layout_3d_latent
from utils import add_legend
from utils import add_heatmap_colorbar
from utils import utils_basic

# fig, ax = plt.subplots(1, 1, figsize=(3, 6))
# fig, axs = plt.subplots(1, 8, figsize=(24, 3))
# axs = [plt.subplots(1, 8, figsize=(24, 3))[1], plt.subplots(1, 8, figsize=(24, 3))[1]]
# fig, ax = plt.subplots(1, 1, figsize=(6, 6), subplot_kw={"projection": "3d"})
# cluster_id = run_clustering()

class plotter_utils(utils_basic):

    def __init__(
            self,
            list_neural_trials, list_labels,
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
        self.bin_win = [450,2550]
        self.bin_num = 4
        self.d_latent = 3
        self.glm = self.run_glm()
        self.n_pre = 2
        self.n_post = 2
        self.cluster_id = self.run_clustering(self.n_pre, self.n_post)
    
    def plot_neuron_fraction(self, ax):
        try:
            colors = ['cornflowerblue', 'violet', 'mediumseagreen']
            cluster_id, neu_labels = get_cluster_cate(self.cluster_id, self.list_labels, [-1,1,2])
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
        ax.hlines(0.5, 500+gap, 2500-gap, color='black')
        ax.vlines([500+gap, 2500-gap], 0, 0.5, color='black')
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
        # plot trials.
        ax.scatter(np.arange(trial_win[0], trial_win[1]-1), isi, c='black', s=5)
        # adjust layouts.
        adjust_layout_isi_example_epoch(ax, trial_win, self.bin_win)

    def plot_cluster_stim_all(self, axs, cate=None):
        color0 = 'dimgrey'
        cluster_id, neu_labels = get_cluster_cate(self.cluster_id, self.list_labels, cate)
        # collect data.
        [_, [neu_seq, _, stim_seq, _], _, _] = get_neu_trial(
            self.alignment, self.list_labels, self.list_stim_labels,
            trial_param=[[2,3,4,5], None, None, None, [1], [0]],
            cate=cate, roi_id=None)
        c_idx = stim_seq.shape[0]//2
        @show_resource_usage
        def plot_glm_kernel(ax):
            kernel_all = get_glm_cate(self.glm, self.list_labels, cate)
            self.plot_glm_kernel(ax, kernel_all, cluster_id, color0, 0.75)
        @show_resource_usage
        def plot_stim(ax, scaled):
            xlim = [-1000, 1500]
            # collect data.
            l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, xlim[0], xlim[1])
            # get response within cluster.
            neu_mean, neu_sem = get_mean_sem_cluster(neu_seq, self.n_clusters, cluster_id)
            if scaled:
                norm_params = [get_norm01_params(neu_mean[ci,:]) for ci in range(self.n_clusters)]
            else:
                norm_params = [get_norm01_params(neu_mean) for ci in range(self.n_clusters)]
            # define layouts.
            ax.axis('off')
            ax = ax.inset_axes([0, 0, 0.5, 0.75], transform=ax.transAxes)
            # plot results.
            self.plot_cluster_mean_sem(
                ax, neu_mean, neu_sem,
                self.alignment['neu_time'], norm_params,
                stim_seq[c_idx,:].reshape(1,2), [color0], [color0]*self.n_clusters, xlim)
            # adjust layouts.
            ax.set_xlabel('time since stim (ms)')
        @show_resource_usage
        def plot_stim_heatmap(ax, norm_mode):
            # collect data.
            xlim = [stim_seq[c_idx,0]-500, stim_seq[c_idx,1]+500]
            l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, xlim[0], xlim[1])
            neu_time = self.alignment['neu_time'][l_idx:r_idx]
            neu_ci = [neu_seq[cluster_id==ci,l_idx:r_idx] for ci in range(self.n_clusters)]
            # define layouts.
            ax.axis('off')
            ax = ax.inset_axes([0, 0, 1, 0.75], transform=ax.transAxes)
            axs_hm = [ax.inset_axes([0.2, ci/self.n_clusters, 0.3, 0.75/self.n_clusters], transform=ax.transAxes)
                      for ci in range(self.n_clusters)]
            axs_cb = [ax.inset_axes([0.6, ci/self.n_clusters, 0.1, 0.75/self.n_clusters], transform=ax.transAxes)
                      for ci in range(self.n_clusters)]
            axs_hm.reverse()
            axs_cb.reverse()
            # plot results for each class.
            for ci in range(self.n_clusters):
                if np.sum(cluster_id==ci) > 0:
                    self.plot_heatmap_neuron(axs_hm[ci], axs_cb[ci], neu_ci[ci], neu_time, neu_ci[ci],
                                             sort_method='shuffle', norm_mode=norm_mode, neu_seq_share=neu_ci)
                    # add stimulus line.
                    axs_hm[ci].axvline(0, color='black', lw=1, linestyle='--')
                # adjust layouts.
                axs_hm[ci].tick_params(axis='y', labelrotation=0)
                axs_hm[ci].set_ylabel(None)
                if ci != self.n_clusters-1:
                    axs_hm[ci].set_xticks([])
                hide_all_axis(axs_cb[ci])
            axs_hm[self.n_clusters-1].set_xlabel('time since stim (ms)')
            ax.set_ylabel('neuron id')
            hide_all_axis(ax)
        @show_resource_usage
        def plot_neu_fraction(ax):
            # define layouts.
            ax.axis('off')
            ax = ax.inset_axes([0, 0, 1, 0.75], transform=ax.transAxes)
            self.plot_cluster_neu_fraction_in_cluster(ax, cluster_id, color0)
        @show_resource_usage
        def plot_fraction(ax):
            # define layouts.
            ax.axis('off')
            ax = ax.inset_axes([0, 0, 1, 0.75], transform=ax.transAxes)
            self.plot_cluster_cate_fraction_in_cluster(ax, cluster_id, neu_labels, self.label_names, color0)
        # plot all.
        try: plot_glm_kernel(axs[0])
        except: traceback.print_exc()
        try: plot_stim(axs[1], False)
        except: traceback.print_exc()
        try: plot_stim(axs[2], True)
        except: traceback.print_exc()
        try: plot_stim_heatmap(axs[3], 'none')
        except: traceback.print_exc()
        try: plot_stim_heatmap(axs[4], 'minmax')
        except: traceback.print_exc()
        try: plot_stim_heatmap(axs[5], 'share')
        except: traceback.print_exc()
        try: plot_neu_fraction(axs[6])
        except: traceback.print_exc()
        try: plot_fraction(axs[7])
        except: traceback.print_exc()

    def plot_cluster_interval_bin_all(self, axs, cate=None):
        color0 = 'dimgrey'
        colors = get_cmap_color(self.bin_num, cmap=self.random_bin_cmap)
        cluster_id, neu_labels = get_cluster_cate(self.cluster_id, self.list_labels, cate)
        split_idx = get_split_idx(self.list_labels, cate)
        day_cluster_id = np.split(cluster_id, split_idx)
        # collect data.
        [_, [neu_seq, stim_seq, camera_pupil, pre_isi, post_isi], _,
         [n_trials, n_neurons]] = get_neu_trial(
            self.alignment, self.list_labels, self.list_stim_labels,
            trial_param=[[2,3,4,5], None, None, None, [1], [0]],
            mean_sem=False,
            cate=cate, roi_id=None)
        c_idx = np.nanmean(np.concatenate(stim_seq, axis=0), axis=0).shape[0]//2
        @show_resource_usage
        def plot_interval_heatmap(ax, norm_mode):
            xlim = [-1000, 3500]
            l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, xlim[0], xlim[1])
            neu_time = self.alignment['neu_time'][l_idx:r_idx]
            # get response within cluster.
            neu_ci = [np.concatenate(
                [np.nanmean(neu[:,dci==ci,:],axis=1)
                 for neu,dci in zip(neu_seq,day_cluster_id)], axis=0)
                for ci in range(self.n_clusters)]
            neu_ci = [nc[np.argsort(np.concatenate(post_isi)), l_idx:r_idx] for nc in neu_ci]
            # define layouts.
            ax0 = ax.inset_axes([0.2, 0, 0.5, 0.75], transform=ax.transAxes)
            ax1 = ax.inset_axes([0.7, 0, 0.1, 0.75], transform=ax.transAxes)
            axs_hm = [ax0.inset_axes([0, 0.05+ci/self.n_clusters, 1, 0.8/self.n_clusters], transform=ax0.transAxes)
                      for ci in range(self.n_clusters)]
            axs_cb = [ax1.inset_axes([0, 0.05+ci/self.n_clusters, 1, 0.8/self.n_clusters], transform=ax1.transAxes)
                      for ci in range(self.n_clusters)]
            axs_hm.reverse()
            axs_cb.reverse()
            # plot results for each class.
            for ci in range(self.n_clusters):
                if np.sum(cluster_id==ci) > 0:
                    self.plot_heatmap_trial(
                            axs_hm[ci], axs_cb[ci], neu_ci[ci], neu_time,
                            norm_mode=norm_mode, neu_seq_share=neu_ci)
                    hide_all_axis(axs_cb[ci])
            # adjust layouts.
            for ci in range(self.n_clusters):
                axs_hm[ci].set_xlim(xlim)
                axs_hm[ci].set_yticklabels((((np.arange(2)+0.5)/2)*2000+500)[::-1].astype('int32'))
                if ci != self.n_clusters-1:
                    axs_hm[ci].set_xticklabels([])
            axs_hm[self.n_clusters-1].set_xlabel('time since stim (ms)')
            ax.set_ylabel('interval (ms)')
            ax.set_title(f'sorted with {norm_mode}')
            hide_all_axis(ax)
            hide_all_axis(ax0)
            hide_all_axis(ax1)
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
            # define layouts.
            ax.axis('off')
            ax = ax.inset_axes([0, 0, 1, 0.75], transform=ax.transAxes)
            # plot results.
            ax.fill_between(
                np.nanmean(bin_stim_seq, axis=0)[c_idx,:],
                0, self.n_clusters,
                color=color0, edgecolor='none', alpha=0.25, step='mid')
            for bi in range(self.bin_num):
                ax.axvline(bin_stim_seq[bi, c_idx+isi_idx_offset, 0], color=colors[bi], lw=1, linestyle='--')
            for bi in range(self.bin_num):
                self.plot_cluster_mean_sem(
                    ax, cluster_bin_neu_mean[bi,:,:], cluster_bin_neu_sem[bi,:,:],
                    self.alignment['neu_time'], norm_params,
                    None, None, [colors[bi]]*self.n_clusters, xlim)
            # adjust layouts.
            ax.set_xlabel('time since stim (ms)')
        @show_resource_usage
        def plot_interval_scaling(ax):
            gap = 2
            bin_num = 50
            win_eval = [[-2500,0], [0, 400]]
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
            ax = ax.inset_axes([0, 0, 1, 0.75], transform=ax.transAxes)
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
            ax = ax.inset_axes([0, 0, 1, 0.75], transform=ax.transAxes)
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
                        axs[ci].scatter(z[0,0]-z[0,0], z[1,0]-z[1,0], color='red', marker='x')
                        axs[ci].scatter(z[0,-1]-z[0,0], z[1,-1]-z[1,0], color='red', marker='o')
                    # adjust layouts.
                    adjust_layout_2d_latent(axs[ci])
        @show_resource_usage
        def plot_legend(ax):
            [bins, _, _, _, _, _, _, _] = get_isi_bin_neu(
                neu_seq, stim_seq, camera_pupil, pre_isi, self.bin_win, self.bin_num)
            lbl = ['[{},{}] ms'.format(int(bins[i]),int(bins[i+1])) for i in range(self.bin_num)]
            cs = colors
            add_legend(ax, cs, lbl, n_trials, n_neurons, self.n_sess, 'upper right')
            ax.axis('off')
        # plot all.
        try: plot_interval_heatmap(axs[0], 'none')
        except: traceback.print_exc()
        try: plot_interval_heatmap(axs[1], 'minmax')
        except: traceback.print_exc()
        try: plot_interval_heatmap(axs[2], 'share')
        except: traceback.print_exc()
        try: plot_interval_bin(axs[3], 'pre')
        except: traceback.print_exc()
        try: plot_interval_bin(axs[4], 'post')
        except: traceback.print_exc()
        try: plot_interval_scaling(axs[5])
        except: traceback.print_exc()
        try: plot_interval_bin_latent(axs[6])
        except: traceback.print_exc()
        try: plot_legend(axs[7])
        except: traceback.print_exc()

    def plot_cluster_heatmap_all(self, axs, cate):
        kernel_all = get_glm_cate(self.glm, self.list_labels, cate)
        cluster_id, neu_labels = get_cluster_cate(self.cluster_id, self.list_labels, cate)
        # collect data.
        [[_, _, _, cmap], [neu_seq, _, stim_seq, _], _, _] = get_neu_trial(
            self.alignment, self.list_labels, self.list_stim_labels,
            trial_param=[[2,3,4,5], None, None, None, [1], [0]],
            cate=cate, roi_id=None)
        c_idx = stim_seq.shape[0]//2
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
            # plot results.
            self.plot_cluster_heatmap(ax, kernel_all, self.glm['kernel_time'], cluster_id, 'minmax')
            # adjust layouts.
            ax.set_xlabel('time since stim (ms)')
            ax.axvline(stim_seq[c_idx,0], color='black', lw=1, linestyle='--')
        @show_resource_usage
        def plot_stim(ax):
            # collect data.
            [[_, _, _, cmap], [neu_seq, _, stim_seq, _], _, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_stim_labels,
                trial_param=[[2,3,4,5], None, None, None, [1], [0]],
                cate=cate, roi_id=None)
            xlim = [stim_seq[c_idx,0]-500, stim_seq[c_idx,1]+500]
            l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, xlim[0], xlim[1])
            neu_seq = neu_seq[:,l_idx:r_idx]
            neu_time = self.alignment['neu_time'][l_idx:r_idx]
            # define layouts.
            ax_hm = ax.inset_axes([0, 0, 0.5, 1], transform=ax.transAxes)
            ax_cb = ax.inset_axes([0.6, 0, 0.1, 1], transform=ax.transAxes)
            # plot results.
            self.plot_heatmap_neuron(ax_hm, ax_cb, neu_seq, neu_time, neu_seq, norm_mode='minmax')
            # adjust layouts.
            ax_hm.set_xlabel('time since stim (ms)')
            ax_hm.axvline(stim_seq[c_idx,0], color='black', lw=1, linestyle='--')
            hide_all_axis(ax)
        # plot all.
        try: plot_cluster_features(axs[0])
        except: traceback.print_exc()
        try: plot_hierarchical_dendrogram(axs[1])
        except: traceback.print_exc()
        try: plot_glm_kernel(axs[2])
        except: traceback.print_exc()
        try: plot_stim(axs[3])
        except: traceback.print_exc()
    
    def plot_cross_sess_adapt(self, axs, cate):
        n_day = 5
        color0 = 'dimgrey'
        color1 = 'cornflowerblue'
        color_day = get_cmap_color(n_day, cmap=self.cross_day_cmap)
        epoch_len = 100
        cluster_id, neu_labels = get_cluster_cate(self.cluster_id, self.list_labels, cate)
        split_idx = get_split_idx(self.list_labels, cate)
        day_cluster_id = np.split(cluster_id, split_idx)
        [_, [_, _, stim_seq, _], _,
         [n_trials, n_neurons]] = get_neu_trial(
            self.alignment, self.list_labels, self.list_stim_labels,
            trial_param=[[2,3,4,5], None, None, None, [1], [0]],
            cate=cate, roi_id=None)
        c_idx = stim_seq.shape[0]//2
        @show_resource_usage
        def plot_dist_cluster_fraction(ax):
            bar_width = 0.5
            # collect data.
            con_day_cluster_id = [np.concatenate(day_cluster_id[di::n_day]) for di in range(n_day)]
            # get fraction in each category.
            fraction = np.zeros((n_day, self.n_clusters))
            for di in range(n_day):
                for ci in range(self.n_clusters):
                    nc = np.nansum(con_day_cluster_id[di]==ci)
                    nt = len(con_day_cluster_id[di]) + 1e-5
                    fraction[di,ci] = nc / nt
            # define layouts.
            ax.axis('off')
            ax = ax.inset_axes([0, 0, 1, 0.75], transform=ax.transAxes)
            axs = [ax.inset_axes([0.2, ci/self.n_clusters, 0.5, 0.75/self.n_clusters], transform=ax.transAxes)
                      for ci in range(self.n_clusters)]
            axs.reverse()
            # plot results for each class.
            for ci in range(self.n_clusters):
                if np.sum(cluster_id==ci) > 0:
                    for di in range(n_day):
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
            axs[self.n_clusters-1].set_xticks(np.arange(n_day))
            axs[self.n_clusters-1].set_xticklabels(
                [f'{di+1}' for di in range(n_day)])
            axs[self.n_clusters-1].set_xlabel('day')
            ax.set_ylabel('fraction of neurons')
            hide_all_axis(ax)
        @show_resource_usage
        def plot_cross_epoch(ax, scaled):
            xlim = [-1000, 1500]
            # collect data.
            [_, [neu_seq_0, _, _, _], _, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_stim_labels,
                trial_param=[[2,3,4,5], None, None, None, [1], [0]],
                trial_idx=[np.concatenate([np.ones(epoch_len), np.zeros(sl.shape[0]-epoch_len)]).astype('bool')
                           for sl in self.list_stim_labels],
                cate=cate, roi_id=None)
            [_, [neu_seq_1, _, _, _,], _, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_stim_labels,
                trial_param=[[2,3,4,5], None, None, None, [1], [0]],
                trial_idx=[np.concatenate([np.zeros(sl.shape[0]-epoch_len), np.ones(epoch_len)]).astype('bool')
                           for sl in self.list_stim_labels],
                cate=cate, roi_id=None)
            day_neu_seq_0 = np.split(neu_seq_0, split_idx)
            day_neu_seq_1 = np.split(neu_seq_1, split_idx)
            con_day_neu_seq_0 = [np.concatenate(day_neu_seq_0[di::n_day],axis=0) for di in range(n_day)]
            con_day_neu_seq_1 = [np.concatenate(day_neu_seq_1[di::n_day],axis=0) for di in range(n_day)]
            con_day_cluster_id = [np.concatenate(day_cluster_id[di::n_day]) for di in range(n_day)]
            # define layouts.
            ax.axis('off')
            ax = ax.inset_axes([0, 0, 1, 0.75], transform=ax.transAxes)
            axs = [ax.inset_axes([di/n_day, 0, 0.8/n_day, 1], transform=ax.transAxes) for di in range(n_day)]
            # plot results for each day.
            for di in range(n_day):
                # get response within cluster.
                neu_mean_0, neu_sem_0 = get_mean_sem_cluster(con_day_neu_seq_0[di], self.n_clusters, con_day_cluster_id[di])
                neu_mean_1, neu_sem_1 = get_mean_sem_cluster(con_day_neu_seq_1[di], self.n_clusters, con_day_cluster_id[di])
                if scaled:
                    norm_params = [get_norm01_params(
                        np.concatenate([neu_mean_0[ci,:], neu_mean_1[ci,:]]))
                        for ci in range(self.n_clusters)]
                else:
                    norm_params = [get_norm01_params(
                        np.concatenate([neu_mean_0, neu_mean_1]))
                        for ci in range(self.n_clusters)]
                # plot stimulus.
                axs[di].fill_between(
                    stim_seq[c_idx,:],
                    0, self.n_clusters,
                    color=color0, edgecolor='none', alpha=0.25, step='mid')
                # plot response.
                scale_bar=True if di==0 else False
                self.plot_cluster_mean_sem(
                    axs[di], neu_mean_0, neu_sem_0,
                    self.alignment['neu_time'], norm_params,
                    None, None, [color0]*self.n_clusters, xlim, scale_bar)
                self.plot_cluster_mean_sem(
                    axs[di], neu_mean_1, neu_sem_1,
                    self.alignment['neu_time'], norm_params,
                    None, None, [color1]*self.n_clusters, xlim, False)
                # adjust layouts.
                axs[di].set_title(f'day {di+1}')
            ax.set_xlabel('time since stim (ms)')
            hide_all_axis(ax)
        @show_resource_usage
        def plot_cross_day(ax, scaled):
            xlim = [-1000, 1500]
            # collect data.
            [_, [neu_seq, _, _, _], _, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_stim_labels,
                trial_param=[[2,3,4,5], None, None, None, [1], [0]],
                cate=cate, roi_id=None)
            day_neu_seq = np.split(neu_seq, split_idx)
            con_day_neu_seq = [np.concatenate(day_neu_seq[di::n_day],axis=0) for di in range(n_day)]
            con_day_cluster_id = [np.concatenate(day_cluster_id[di::n_day]) for di in range(n_day)]
            # get response within cluster.
            neu_ci = [get_mean_sem_cluster(
                con_day_neu_seq[di], self.n_clusters, con_day_cluster_id[di])
                for di in range(n_day)]
            if scaled:
                norm_params = [get_norm01_params(np.concatenate([nc[0][ci] for nc in neu_ci])) for ci in range(self.n_clusters)]
            else:
                norm_params = [get_norm01_params(np.concatenate([nc[0] for nc in neu_ci])) for ci in range(self.n_clusters)]
            # define layouts.
            ax.axis('off')
            ax = ax.inset_axes([0, 0, 1, 0.75], transform=ax.transAxes)
            ax.axis('off')
            ax = ax.inset_axes([0, 0, 0.5, 1], transform=ax.transAxes)
            # plot stimulus.
            ax.fill_between(
                stim_seq[c_idx,:],
                0, self.n_clusters,
                color=color0, edgecolor='none', alpha=0.25, step='mid')
            # plot response.
            for di in range(n_day):
                self.plot_cluster_mean_sem(
                    ax, neu_ci[di][0], neu_ci[di][1],
                    self.alignment['neu_time'], norm_params,
                    None, None, [color_day[di]]*self.n_clusters, xlim)
            # adjust layouts.
            ax.set_xlabel('time since stim (ms)')
            hide_all_axis(ax)
        @show_resource_usage
        def plot_legend(ax):
            cs = [color0, color1] + color_day
            lbl = ['first epoch', 'last epoch'] + [f'day {di+1}' for di in range(n_day)]
            add_legend(ax, cs, lbl, n_trials, n_neurons, self.n_sess, 'upper right')
            ax.axis('off')
        # plot all.
        try: plot_dist_cluster_fraction(axs[0])
        except: traceback.print_exc()
        try: plot_cross_epoch(axs[1], False)
        except: traceback.print_exc()
        try: plot_cross_epoch(axs[2], True)
        except: traceback.print_exc()
        try: plot_cross_day(axs[3], False)
        except: traceback.print_exc()
        try: plot_cross_day(axs[4], True)
        except: traceback.print_exc()
        try: plot_legend(axs[5])
        except: traceback.print_exc()
    
    def plot_cluster_local_all(self, axs, cate):
        isi_win = 200
        color0 = 'dimgrey'
        colors = get_cmap_color(self.bin_num, cmap=self.random_bin_cmap)
        cluster_id, neu_labels = get_cluster_cate(self.cluster_id, self.list_labels, cate)
        @show_resource_usage
        def plot_heatmap(ax):
            # collect data.
            [_, [neu_seq, stim_seq, camera_pupil, pre_isi, post_isi], _,
             [_, n_neurons]] = get_neu_trial(
                self.alignment, self.list_labels, self.list_stim_labels,
                trial_param=[[2,3,4,5], None, None, None, [1], [0]],
                mean_sem=False,
                cate=cate, roi_id=None)
            c_idx = np.nanmean(np.concatenate(stim_seq, axis=0), axis=0).shape[0]//2
            idx = [(isi>np.nanmean(isi)-isi_win)*(isi<np.nanmean(isi)+isi_win) for isi in post_isi]
            neu_seq  = [neu_seq[si][idx[si],:,:]  for si in range(self.n_sess)]
            stim_seq = [stim_seq[si][idx[si],:,:] for si in range(self.n_sess)]
            stim_seq = np.nanmean(np.concatenate(stim_seq, axis=0), axis=0)
            xlim = [stim_seq[c_idx,0]-500, stim_seq[c_idx+1,1]+500]
            l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, xlim[0], xlim[1])
            neu_seq = np.concatenate([np.nanmean(n, axis=0) for n in neu_seq], axis=0)
            neu_seq = neu_seq[:,l_idx:r_idx]
            neu_time = self.alignment['neu_time'][l_idx:r_idx]
            neu_ci = [neu_seq[cluster_id==ci,:] for ci in range(self.n_clusters)]
            # define layouts.
            axs_hm = [ax.inset_axes([0.1, ci/self.n_clusters, 0.5, 0.8/self.n_clusters], transform=ax.transAxes)
                      for ci in range(self.n_clusters)]
            axs_cb = [ax.inset_axes([0.7, ci/self.n_clusters, 0.1, 0.8/self.n_clusters], transform=ax.transAxes)
                      for ci in range(self.n_clusters)]
            axs_hm.reverse()
            axs_cb.reverse()
            # plot results for each class.
            for ci in range(self.n_clusters):
                if np.sum(cluster_id==ci) > 0:
                    self.plot_heatmap_neuron(axs_hm[ci], axs_cb[ci], neu_ci[ci], neu_time, neu_ci[ci],
                                             norm_mode='minmax', sort_method='shuffle')
                # adjust layouts.
                if ci != self.n_clusters-1:
                    axs_hm[ci].set_xticks([])
                axs_hm[ci].set_ylabel(None)
            axs_hm[self.n_clusters-1].set_xlabel('time since stim (ms)')
            ax.set_ylabel('neuron id')
            hide_all_axis(ax)
        @show_resource_usage
        def plot_interval_bin(ax):
            xlim = [-3500, 4000]
            # collect data.
            [_, [neu_seq, stim_seq, camera_pupil, pre_isi, post_isi], _,
             [_, n_neurons]] = get_neu_trial(
                self.alignment, self.list_labels, self.list_stim_labels,
                trial_param=[[2,3,4,5], None, None, None, [1], [0]],
                mean_sem=False,
                cate=cate, roi_id=None)
            c_idx = np.nanmean(np.concatenate(stim_seq, axis=0), axis=0).shape[0]//2
            idx = [(isi>np.nanmean(isi)-isi_win)*(isi<np.nanmean(isi)+isi_win) for isi in post_isi]
            neu_seq      = [neu_seq[si][idx[si],:,:]    for si in range(self.n_sess)]
            stim_seq     = [stim_seq[si][idx[si],:,:]   for si in range(self.n_sess)]
            camera_pupil = [camera_pupil[si][idx[si],:] for si in range(self.n_sess)]
            pre_isi      = [pre_isi[si][idx[si]]        for si in range(self.n_sess)]
            # bin data based on isi.
            [bins, bin_center, _, bin_neu_seq, _, _, bin_stim_seq, _] = get_isi_bin_neu(
                neu_seq, stim_seq, camera_pupil, pre_isi, self.bin_win, self.bin_num)
            stim_seq = np.nanmean(bin_stim_seq, axis=0)[c_idx:c_idx+2,:].reshape(-1,2)
            # get response within cluster at each bin.
            cluster_bin_neu_mean, cluster_bin_neu_sem = get_bin_mean_sem_cluster(bin_neu_seq, self.n_clusters, cluster_id)
            norm_params = [get_norm01_params(cluster_bin_neu_mean[:,i,:]) for i in range(self.n_clusters)]
            # define layouts.
            ax.axis('off')
            ax = ax.inset_axes([0.1, 0, 1, 1], transform=ax.transAxes)
            # plot results.
            for si in range(stim_seq.shape[0]):
                ax.fill_between(
                    stim_seq[si,:],
                    0, self.n_clusters,
                    color=color0, edgecolor='none', alpha=0.25, step='mid')
            for bi in range(self.bin_num):
                ax.axvline(bin_stim_seq[bi, c_idx-1, 0], color=colors[bi], lw=1, linestyle='--')
            for bi in range(self.bin_num):
                self.plot_cluster_mean_sem(
                    ax, cluster_bin_neu_mean[bi,:,:], cluster_bin_neu_sem[bi,:,:],
                    self.alignment['neu_time'], norm_params,
                    None, None, [colors[bi]]*self.n_clusters, xlim)
            # adjust layouts.
            ax.set_xlabel('time since stim (ms)')
            hide_all_axis(ax)
        def plot_interval_scaling(ax):
            post_color = 'lightcoral'
            gap = 2
            bin_num = 50
            win_peak = [-100,500]
            win_eval = [[-2500,0], [-100, 100]]
            order = 2
            # collect data.
            [_, [neu_seq, stim_seq, camera_pupil, pre_isi, post_isi], _,
             [_, n_neurons]] = get_neu_trial(
                self.alignment, self.list_labels, self.list_stim_labels,
                trial_param=[[2,3,4,5], None, None, None, [1], [0]],
                mean_sem=False,
                cate=cate, roi_id=None)
            c_idx = np.nanmean(np.concatenate(stim_seq, axis=0), axis=0).shape[0]//2
            idx = [(isi>np.nanmean(isi)-isi_win)*(isi<np.nanmean(isi)+isi_win) for isi in post_isi]
            neu_seq      = [neu_seq[si][idx[si],:,:]    for si in range(self.n_sess)]
            stim_seq     = [stim_seq[si][idx[si],:,:]   for si in range(self.n_sess)]
            camera_pupil = [camera_pupil[si][idx[si],:] for si in range(self.n_sess)]
            pre_isi      = [pre_isi[si][idx[si]]        for si in range(self.n_sess)]
            # bin data based on isi.
            [_, bin_center, _, bin_neu_seq, _, _, _, _] = get_isi_bin_neu(
                neu_seq, stim_seq, camera_pupil, pre_isi, self.bin_win, bin_num)
            bin_center = bin_center[gap:-gap]
            stim_seq = np.nanmean(np.concatenate(stim_seq, axis=0), axis=0)
            # get evluation windows.
            neu_ci = np.concatenate([np.nanmean(ns, axis=0) for ns in neu_seq], axis=0)
            neu_ci = [get_mean_sem(neu_ci[cluster_id==ci,:])[0] for ci in range(self.n_clusters)]
            peak_time = [get_peak_time(neu_ci[ci], self.alignment['neu_time'], win_peak) for ci in range(self.n_clusters)]
            # get response within cluster.
            bin_win_baseline = [np.array([get_mean_sem_win(
                bns[cluster_id==ci,:], self.alignment['neu_time'], 0,
                win_eval[0][0], win_eval[0][1], 'lower')[1]
                for bns in bin_neu_seq])[gap:-gap] for ci in range(self.n_clusters)]
            bin_win_evoked_pre = [np.array([get_mean_sem_win(
                bns[cluster_id==ci,:], self.alignment['neu_time'], 0,
                peak_time[ci]+stim_seq[c_idx,0]+win_eval[1][0], peak_time[ci]+stim_seq[c_idx,0]+win_eval[1][1], 'mean')[1]
                for bns in bin_neu_seq])[gap:-gap] for ci in range(self.n_clusters)]
            bin_win_evoked_post = [np.array([get_mean_sem_win(
                bns[cluster_id==ci,:], self.alignment['neu_time'], 0,
                peak_time[ci]+stim_seq[c_idx+1,0]+win_eval[1][0], peak_time[ci]+stim_seq[c_idx+1,0]+win_eval[1][1], 'mean')[1]
                for bns in bin_neu_seq])[gap:-gap] for ci in range(self.n_clusters)]
            bin_win_pre  = [bwe - bwb for bwe, bwb in zip (bin_win_evoked_pre, bin_win_baseline)]
            bin_win_post = [bwe - bwb for bwe, bwb in zip (bin_win_evoked_post, bin_win_baseline)]
            # define layouts.
            axs = [ax.inset_axes([0.1, ci/self.n_clusters, 0.4, 0.8/self.n_clusters], transform=ax.transAxes)
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
                    axs[ci].tick_params(direction='in')
                    axs[ci].spines['right'].set_visible(False)
                    axs[ci].spines['top'].set_visible(False)
                    axs[ci].set_xlim(self.bin_win)
                    axs[ci].set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
                    axs[ci].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
                    axs[ci].xaxis.set_major_locator(mtick.MaxNLocator(nbins=1))
                    axs[ci].yaxis.set_major_locator(mtick.MaxNLocator(nbins=1))
                    if ci != self.n_clusters-1:
                        axs[ci].set_xticklabels([])
            axs[self.n_clusters-1].set_xlabel('preceding interval (ms)')
            ax.set_ylabel(f'evoked peak magnitude in {win_eval[1]} ms')
            hide_all_axis(ax)
            add_legend(ax, [color0, post_color], ['pre','post'], None, None, None, 'upper right')
        @show_resource_usage
        def plot_legend(ax):
            # collect data.
            [_, [neu_seq, stim_seq, camera_pupil, pre_isi, post_isi], _,
             [_, n_neurons]] = get_neu_trial(
                self.alignment, self.list_labels, self.list_stim_labels,
                trial_param=[[2,3,4,5], None, None, None, [1], [0]],
                mean_sem=False,
                cate=cate, roi_id=None)
            idx = [(isi>np.nanmean(isi)-isi_win)*(isi<np.nanmean(isi)+isi_win) for isi in post_isi]
            n_trials = np.nansum(idx)
            [bins, _, _, _, _, _, _, _] = get_isi_bin_neu(
                neu_seq, stim_seq, camera_pupil, pre_isi, self.bin_win, self.bin_num)
            lbl = ['[{},{}] ms'.format(int(bins[i]),int(bins[i+1])) for i in range(self.bin_num)]
            cs = colors
            add_legend(ax, cs, lbl, n_trials, n_neurons, self.n_sess, 'upper right')
            ax.axis('off')
        # plot all.
        try: plot_heatmap(axs[0])
        except: traceback.print_exc()
        try: plot_interval_bin(axs[1])
        except: traceback.print_exc()
        try: plot_interval_scaling(axs[2])
        except: traceback.print_exc()
        try: plot_legend(axs[3])
        except: traceback.print_exc()
        
    def plot_latent_all(self, axs, cate=None):
        cluster_id, neu_labels = get_cluster_cate(self.cluster_id, self.list_labels, cate)
        # collect data.
        [_, [neu_seq, stim_seq, camera_pupil, pre_isi, post_isi], _,
         [n_trials, n_neurons]] = get_neu_trial(
            self.alignment, self.list_labels, self.list_stim_labels,
            trial_param=[[2,3,4,5], None, None, None, [1], [0]],
            mean_sem=False,
            cate=cate, roi_id=None)
        c_idx = np.nanmean(np.concatenate(stim_seq, axis=0), axis=0).shape[0]//2
        @show_resource_usage
        def plot_interval_bin_latent_all(axs):
            bin_num = 10
            colors = get_cmap_color(bin_num, cmap=self.random_bin_cmap)
            # bin data based on isi.
            [bins, _, _, bin_neu_seq, _, _, bin_stim_seq, _] = get_isi_bin_neu(
                neu_seq, stim_seq, camera_pupil, post_isi, self.bin_win, bin_num)
            # get latent dynamics.
            neu_x = np.concatenate([bns for bns in bin_neu_seq], axis=1)
            # fit model.
            model = PCA(n_components=3)
            model.fit(neu_x.reshape(neu_x.shape[0],-1).T)
            z = model.transform(neu_x.reshape(neu_x.shape[0],-1).T)
            neu_z = z.reshape(bin_num, -1, 3).transpose([0,2,1])
            # random rotate dynamics.
            for ai in range(len(axs)-1):
                # get random matrix.
                rm = get_random_rotate_mat_3d()
                # define layouts.
                axs[ai].axis('off')
                ax1 = axs[ai].inset_axes([0, 0, 0.6, 0.6], transform=axs[ai].transAxes, projection='3d')
                # plot 3d dynamics.
                for bi in range(bin_num):
                    l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, bin_stim_seq[bi,c_idx,0], bin_stim_seq[bi,c_idx+1,0])
                    neu_time = self.alignment['neu_time'][l_idx:r_idx]
                    cmap, _ = get_cmap_color(len(neu_time), base_color=['lemonchiffon', colors[bi], 'black'], return_cmap=True)
                    self.plot_3d_latent_dynamics(
                        ax1, np.matmul(rm, neu_z[bi,:,l_idx:r_idx]), None, neu_time,
                        cmap=cmap, end_color=colors[bi], add_stim=False)
                # adjust layouts.
                adjust_layout_3d_latent(ax1)
            ax2 = axs[-1].inset_axes([0, 0, 0.5, 1], transform=axs[-1].transAxes)
            ax3 = axs[-1].inset_axes([0.5, 0, 0.1, 0.6], transform=axs[-1].transAxes)
            # add colorbar.
            add_legend(ax2, colors, ['[{},{}] ms'.format(int(bins[i]),int(bins[i+1])) for i in range(bin_num)], None, None, None, 'upper left')
            t_cmap, _ = get_cmap_color(len(neu_time), base_color=['lemonchiffon', 'black'], return_cmap=True)
            add_heatmap_colorbar(ax3, t_cmap, None, 'interval progress since stim onset')
            hide_all_axis(axs[-1])
            hide_all_axis(ax2)
            hide_all_axis(ax3)
        # plot all.
        try: plot_interval_bin_latent_all(axs[0])
        except: traceback.print_exc()

# colors = ['#989A9C', '#A4CB9E', '#9DB4CE', '#EDA1A4', '#F9C08A']
class plotter_main(plotter_utils):
    def __init__(self, neural_trials, labels, label_names, temp_folder, cate_list):
        super().__init__(neural_trials, labels, temp_folder, cate_list)
        self.label_names = label_names
    
    def cluster_stim_all(self, axs_all):
        for cate, axs in zip(self.cate_list, axs_all):
            try:
                label_name = self.label_names[str(cate[0])] if len(cate)==1 else 'all'
                print(f'plotting results for {label_name}')
                
                self.plot_cluster_stim_all(axs, cate=cate)

            except: traceback.print_exc()

    def cluster_interval_bin_all(self, axs_all):
        for cate, axs in zip(self.cate_list, axs_all):
            try:
                label_name = self.label_names[str(cate[0])] if len(cate)==1 else 'all'
                print(f'plotting results for {label_name}')
                
                self.plot_cluster_interval_bin_all(axs, cate=cate)

            except: traceback.print_exc()
    
    def cluster_heatmap_all(self, axs_all):
        for cate, axs in zip(self.cate_list, axs_all):
            try:
                label_name = self.label_names[str(cate[0])] if len(cate)==1 else 'all'
                print(f'plotting results for {label_name}')

                self.plot_cluster_heatmap_all(axs, cate=cate)

            except: traceback.print_exc()

    def cross_sess_adapt(self, axs_all):
        for cate, axs in zip(self.cate_list, axs_all):
            try:
                label_name = self.label_names[str(cate[0])] if len(cate)==1 else 'all'
                print(f'plotting results for {label_name}')
                
                self.plot_cross_sess_adapt(axs, cate=cate)

            except: traceback.print_exc()
    
    def cluster_local_all(self, axs_all):
        for cate, axs in zip(self.cate_list, axs_all):
            try:
                label_name = self.label_names[str(cate[0])] if len(cate)==1 else 'all'
                print(f'plotting results for {label_name}')
                
                self.plot_cluster_local_all(axs, cate=cate)

            except: traceback.print_exc()
    
    def latent_all(self, axs_all):
        for cate, axs in zip(self.cate_list, axs_all):
            try:
                label_name = self.label_names[str(cate[0])] if len(cate)==1 else 'all'
                print(f'plotting results for {label_name}')
                
                self.plot_latent_all(axs, cate=cate)

            except: traceback.print_exc()