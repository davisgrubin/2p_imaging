#!/usr/bin/env python3

import numpy as np
import matplotlib.colors as mcolors
from sklearn.decomposition import PCA

from modules.Alignment import run_get_stim_response
from modeling.clustering import clustering_neu_response_mode
from modeling.clustering import remap_cluster_id
from modeling.clustering import get_mean_sem_cluster
from modeling.clustering import get_bin_mean_sem_cluster
from modeling.generative import get_glm_cate
from modeling.quantifications import run_quantification
from utils import get_norm01_params
from utils import get_mean_sem
from utils import get_neu_trial
from utils import get_frame_idx_from_time
from utils import get_isi_bin_neu
from utils import get_temporal_scaling_trial_multi_sess
from utils import get_roi_label_color
from utils import get_cmap_color
from utils import apply_colormap
from utils import adjust_layout_neu
from utils import adjust_layout_heatmap
from utils import add_legend
from utils import add_heatmap_colorbar
from utils import utils_basic

# fig, ax = plt.subplots(1, 1, figsize=(5, 5))
# fig, ax = plt.subplots(1, 1, figsize=(3, 12))
# fig, axs = plt.subplots(1, 8, figsize=(24, 3))
# axs = [plt.subplots(1, 8, figsize=(24, 3))[1], plt.subplots(1, 8, figsize=(24, 3))[1]]
# fig, ax = plt.subplots(1, 1, figsize=(6, 6), subplot_kw={"projection": "3d"})

class plotter_utils(utils_basic):

    def __init__(
            self,
            list_neural_trials, list_labels, list_significance,
            temp_folder,
            ):
        super().__init__()
        timescale = 1.0
        self.n_sess = len(list_neural_trials)
        self.l_frames = int(200*timescale)
        self.r_frames = int(200*timescale)
        self.list_labels = list_labels
        self.list_neural_trials = list_neural_trials
        self.alignment = run_get_stim_response(
            temp_folder,
            list_neural_trials, self.l_frames, self.r_frames, expected='none')
        self.list_stim_labels = self.alignment['list_stim_labels']
        self.list_significance = list_significance
        self.bin_win = [450,2550]
        self.bin_num = 5
        self.n_clusters = 9
        self.max_clusters = 10
        self.d_latent = 3
        self.glm = self.run_glm()

    def run_clustering(self, cate):
        n_latents = 15
        # get glm kernels.
        kernel_all, exp_var_all = get_glm_cate(self.glm, self.list_labels, self.list_significance, cate)
        # reduce kernel weights to pca features.
        model = PCA(n_components=n_latents if n_latents < np.min(kernel_all.shape) else np.min(kernel_all.shape))
        neu_x = model.fit_transform(kernel_all)
        # run clustering.
        _, cluster_id = clustering_neu_response_mode(neu_x, self.n_clusters, None)
        # relabel.
        _, [neu_seq, _, _, _], [neu_labels, _], _ = get_neu_trial(
            self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
            trial_param=[[2,3,4,5], None, None, None, [1], [0]],
            cate=cate, roi_id=None)
        cluster_id = remap_cluster_id(neu_seq, self.alignment['neu_time'], cluster_id)
        return kernel_all, cluster_id, neu_labels

    def plot_cluster_interval_bin_all(self, axs, mode, cate=None):
        kernel_all, cluster_id, neu_labels = self.run_clustering(cate)
        # collect data.
        [[color0, _, color2, _],
         [neu_seq, stim_seq, stim_value, pre_isi, post_isi],
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
            neu_seq, stim_seq, stim_value, isi, self.bin_win, self.bin_num)
        c_idx = bin_stim_seq.shape[1]//2
        def plot_glm_kernel(ax):
            ax.axis('off')
            ax = ax.inset_axes([0, 0, 0.5, 1], transform=ax.transAxes)
            # get cluster average.
            glm_mean, glm_sem = get_mean_sem_cluster(kernel_all, cluster_id)
            # plot results.
            norm_params = [get_norm01_params(glm_mean[i,:]) for i in range(self.n_clusters)]
            self.plot_cluster_mean_sem(
                ax, glm_mean, glm_sem, self.glm['kernel_time'],
                norm_params, np.nanmean(np.concatenate(stim_seq, axis=0), axis=0)[c_idx,:].reshape(1,-1),
                [color0], [color0]*self.n_clusters,
                [np.nanmin(self.glm['kernel_time']), np.nanmax(self.glm['kernel_time'])])
            # adjust layout.
            ax.set_xlabel('time since stim (ms)')
        def plot_interval_bin(ax):
            # get response within cluster at each bin.
            cluster_bin_neu_mean, cluster_bin_neu_sem = get_bin_mean_sem_cluster(bin_neu_seq, cluster_id)
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
            # adjust layout.
            ax.set_xlabel('time since stim (ms)')
        def plot_fraction(ax):
            ax.axis('off')
            ax = ax.inset_axes([0, 0, 0.4, 1], transform=ax.transAxes)
            # plot results.
            self.plot_cluster_cate_fraction_in_cluster(ax, cluster_id, neu_labels, self.label_names)
        def plot_legend(ax):
            lbl = ['[{},{}] ms'.format(int(bins[i]),int(bins[i+1])) for i in range(self.bin_num)]
            lbl+= [v for v in self.label_names.values()] + ['all cell-types']
            cs = colors + [get_roi_label_color(cate=[int(k)])[2] for k in self.label_names.keys()]
            cs+= [get_roi_label_color(cate=[-1,1,2])[2]]
            add_legend(ax, cs, lbl, n_trials, n_neurons, self.n_sess, 'upper right')
            ax.axis('off')
        # plot all.
        try: plot_glm_kernel(axs[0])
        except: pass
        try: plot_interval_bin(axs[1])
        except: pass
        try: plot_fraction(axs[2])
        except: pass
        try: plot_legend(axs[3])
        except: pass

    def plot_cluster_interval_bin_individual(self, axs, mode, cate=None):
        kernel_all, cluster_id, neu_labels = self.run_clustering(cate)
        lbl = ['cluster #'+str(ci) for ci in range(self.n_clusters)]
        # collect data.
        [[color0, _, color2, cmap],
         [neu_seq, stim_seq, stim_value, pre_isi, post_isi],
         [neu_labels, _],
         [n_trials, n_neurons]] = get_neu_trial(
            self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
            trial_param=[[2,3,4,5], None, None, None, [1], [0]],
            mean_sem=False,
            cate=cate, roi_id=None)
        # bin data based on isi.
        if mode == 'pre':
            xlim = [-3500, 1500]
            isi = pre_isi
            isi_idx_offset = -1
            isi_quant_idx = 0
        if mode == 'post':
            xlim = [-1000, 4000]
            isi = post_isi
            isi_idx_offset = 1
            isi_quant_idx = 1
        # bin data based on isi.
        [bins, bin_center, _, bin_neu_seq, _, _, bin_stim_seq, _] = get_isi_bin_neu(
            neu_seq, stim_seq, stim_value, isi, self.bin_win, self.bin_num)
        c_idx = bin_stim_seq.shape[1]//2
        cs = get_cmap_color(self.bin_num, base_color=color2)
        def plot_interval_bin(axs):
            # get response within cluster at each bin.
            cluster_bin_neu_mean, cluster_bin_neu_sem = get_bin_mean_sem_cluster(bin_neu_seq, cluster_id)
            # plot results for each class.
            for ci in range(self.n_clusters):
                # find bounds.
                upper = np.nanmax(cluster_bin_neu_mean[:,ci,:]) + np.nanmax(cluster_bin_neu_sem[:,ci,:])
                lower = np.nanmin(cluster_bin_neu_mean[:,ci,:]) - np.nanmax(cluster_bin_neu_sem[:,ci,:])
                # plot stimulus.
                axs[ci].fill_between(
                    np.nanmean(bin_stim_seq, axis=0)[c_idx,:],
                    lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
                    color=color0, edgecolor='none', alpha=0.25, step='mid')
                for bi in range(self.bin_num):
                    axs[ci].fill_between(
                        bin_stim_seq[bi,c_idx+isi_idx_offset,:],
                        lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
                        color=cs[bi], edgecolor='none', alpha=0.25, step='mid')
                # plot neural traces.
                l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, xlim[0], xlim[1])
                for bi in range(self.bin_num):
                    self.plot_mean_sem(
                        axs[ci], self.alignment['neu_time'][l_idx:r_idx],
                        cluster_bin_neu_mean[bi,ci,l_idx:r_idx], cluster_bin_neu_sem[bi,ci,l_idx:r_idx],
                        cs[bi], None)
                # adjust layouts.
                adjust_layout_neu(axs[ci])
                axs[ci].set_xlim(xlim)
                axs[ci].set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
                axs[ci].set_xlabel('time since stim (ms)')
                axs[ci].set_title(lbl[ci])
        def plot_interval_bin_quant(axs):
            target_metrics = ['evoke_mag', 'onset_drop']
            # plot results for each class.
            for ci in range(self.n_clusters):
                quant_all = [
                    run_quantification(ns[cluster_id==ci,:], self.alignment['neu_time'], ss[c_idx+isi_quant_idx,0])
                    for ns, ss in zip(bin_neu_seq, bin_stim_seq)]
                # plot each metric.
                for mi in range(len(target_metrics)):
                    axs[mi][ci].axis('off')
                    axs[mi][ci] = axs[mi][ci].inset_axes([0, 0, 0.4, 1], transform=axs[mi][ci].transAxes)
                    # plot each condition.
                    for di in range(self.bin_num-1):
                        m, s = get_mean_sem(quant_all[di+1][target_metrics[mi]].reshape(-1,1))
                        axs[mi][ci].errorbar(
                            di+1, m, s, None,
                            color=cs[di+1],
                            capsize=2, marker='o', linestyle='none',
                            markeredgecolor='white', markeredgewidth=0.1)
                    # adjust layout.
                    axs[mi][ci].tick_params(axis='y', tick1On=False)
                    axs[mi][ci].tick_params(axis='x', labelrotation=90)
                    axs[mi][ci].spines['right'].set_visible(False)
                    axs[mi][ci].spines['top'].set_visible(False)
                    axs[mi][ci].set_xlim([0.5, self.bin_num+0.5])
                    axs[mi][ci].set_xticks(np.arange(self.bin_num-1)+1)
                    axs[mi][ci].set_xticklabels(bin_center[1:], rotation='vertical')
                    axs[mi][ci].set_xlabel('interval')
                    axs[mi][ci].set_ylabel(target_metrics[mi])
                    axs[mi][ci].set_title(lbl[ci])
        def plot_interval_heatmap(axs):
            # bin data based on isi.
            bin_num_isi = 100
            [_, b_center, _, b_neu_seq, _, _, _, _] = get_isi_bin_neu(
                neu_seq, stim_seq, stim_value, isi, self.bin_win, bin_num_isi)
            # get response within cluster at each bin.
            cluster_bin_neu_mean, _ = get_bin_mean_sem_cluster(b_neu_seq, cluster_id)
            # plot results for each class.
            for ci in range(self.n_clusters):
                heatmap, _, _ = apply_colormap(
                    cluster_bin_neu_mean[:,ci,:],
                    cmap, norm_mode='none', data_share=cluster_bin_neu_mean[:,ci,:])
                axs[ci].imshow(
                    heatmap,
                    extent=[
                        self.alignment['neu_time'][0], self.alignment['neu_time'][-1],
                        1, heatmap.shape[0]],
                    interpolation='nearest', aspect='auto')
                # adjust layout.
                adjust_layout_heatmap(axs[ci])
                axs[ci].set_xlabel('time since stim (ms)')
                axs[ci].set_ylabel('interval (ms)')
                idx = np.arange(1, heatmap.shape[0], heatmap.shape[0]/5, dtype='int32') + int(heatmap.shape[0]/5/2)
                axs[ci].set_yticks(idx)
                axs[ci].set_yticklabels(b_center[idx][::-1].astype('int32'))
                axs[ci].set_title(lbl[ci])
                add_heatmap_colorbar(
                    axs[ci], cmap, mcolors.Normalize(
                        vmin=np.nanmin(cluster_bin_neu_mean[:,ci,:]),
                        vmax=np.nanmax(cluster_bin_neu_mean[:,ci,:])), None)
        def plot_legend(ax):
            lbl = ['[{},{}] ms'.format(int(bins[i]),int(bins[i+1])) for i in range(self.bin_num)]
            cs = get_cmap_color(self.bin_num, base_color=color2)
            add_legend(ax, cs, lbl, n_trials, n_neurons, self.n_sess, 'upper right')
            ax.axis('off')
        # plot all.
        try: plot_interval_bin(axs[0])
        except: pass
        try: plot_interval_bin_quant(axs[1])
        except: pass
        try: plot_interval_heatmap(axs[2])
        except: pass
        try: plot_legend(axs[3])
        except: pass

    def plot_categorization_features(self, axs, cate):
        bin_num = 3
        cate = [-1,1,2]
        n_latents = 15
        kernel_time, kernel_all, exp_var_all = self.run_glm(cate)
        model = PCA(n_components=n_latents if n_latents < np.min(kernel_all.shape) else np.min(kernel_all.shape))
        neu_x = model.fit_transform(kernel_all)
        _, cluster_id = clustering_neu_response_mode(neu_x, self.n_clusters, None)
        #results_all = self.run_features_categorization(cate)
        #cluster_id = results_all['cluster_id'].to_numpy()
        lbl = ['cluster #'+str(ci) for ci in range(self.n_clusters)]
        colors = get_cmap_color(len(lbl), cmap=self.cluster_cmap)
        [color0, _, _, _], _, [neu_labels, neu_sig], _ = get_neu_trial(
            self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
            trial_param=[[2,3,4,5], None, None, None, [1], [0]],
            mean_sem=False,
            cate=cate, roi_id=None)
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
        # comparison between short and long preceeding interval.
        def plot_local_isi(axs):
            xlim = [-3000, 2500]
            l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, xlim[0], xlim[1])
            # collect data.
            [[color0, _, _, _], [neu_seq, stim_seq, stim_value, pre_isi],
             [neu_labels, neu_sig], [n_trials, n_neurons]] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_param=[[2,3,4,5], None, None, None, [1], [0]],
                mean_sem=False,
                cate=cate, roi_id=None)
            # bin data based on isi.
            [bins, bin_center, _, bin_neu_seq, _, _, bin_stim_seq, bin_stim_value] = get_isi_bin_neu(
                neu_seq, stim_seq, stim_value, pre_isi, self.bin_win, bin_num)
            cluster_bin_neu_mean, cluster_bin_neu_sem = get_bin_mean_sem_cluster(bin_neu_seq, cluster_id)
            cluster_bin_neu_mean = cluster_bin_neu_mean[:,:,l_idx:r_idx]
            cluster_bin_neu_sem  = cluster_bin_neu_sem[:,:,l_idx:r_idx]
            neu_time = self.alignment['neu_time'][l_idx:r_idx]
            # plot results for each class.
            for ci in range(len(lbl)):
                # find bounds.
                upper = np.nanmax(cluster_bin_neu_mean[:,ci,:]) + np.nanmax(cluster_bin_neu_sem[:,ci,:])
                lower = np.nanmin(cluster_bin_neu_mean[:,ci,:]) - np.nanmax(cluster_bin_neu_sem[:,ci,:])
                # find class colors.
                cs = get_cmap_color(bin_num+1, base_color=colors[ci])[:-1]
                # only keep stimulus after bins.
                c_idx = bin_stim_seq.shape[1]//2
                stim_seq = bin_stim_seq[:,c_idx-1:c_idx+2,:]
                stim_seq[:,1:,:] = np.nanmean(stim_seq[:,1:,:], axis=0)
                # plot stimulus.
                for bi in range(bin_num):
                    axs[ci].fill_between(
                        stim_seq[bi,0,:],
                        lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
                        color=cs[bi], edgecolor='none', alpha=0.25, step='mid')
                    axs[ci].fill_between(
                        stim_seq[bi,1,:],
                        lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
                        color=cs[bi], edgecolor='none', alpha=0.25, step='mid')
                    self.plot_vol(axs[ci], self.alignment['stim_time'], bin_stim_value[bi,:].reshape(1,-1),
                                  cs[bi], upper, lower)
                # plot neural traces.
                for bi in range(bin_num):
                    self.plot_mean_sem(
                        axs[ci], neu_time,
                        cluster_bin_neu_mean[bi,ci,:], cluster_bin_neu_sem[bi,ci,:], cs[bi], None)
                # adjust layouts.
                adjust_layout_neu(axs[ci])
                axs[ci].set_xlim(xlim)
                axs[ci].set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
                axs[ci].set_xlabel('time since pre oddball stim (ms)')
                axs[ci].set_title(f'response to random interval comparison for cluster #{ci}')
                add_legend(axs[ci], cs,
                           ['[{},{}] ms'.format(int(bins[i]),int(bins[i+1])) for i in range(bin_num)],
                           n_trials, n_neurons, self.n_sess, 'upper right')
        # compare temporal scaling.
        def plot_temporal_scaling(axs):
            target_isi = 1500
            # collect data.
            [[color0, _, _, _], [neu_seq, stim_seq, stim_value, pre_isi],
             [neu_labels, neu_sig], [n_trials, n_neurons]] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_param=[[2,3,4,5], None, None, None, [1], [0]],
                mean_sem=False,
                cate=cate, roi_id=None)
            c_idx = stim_seq[0][0,:,:].shape[0]//2
            # compute mean time stamps.
            stim_seq_target = np.nanmean(np.concatenate(stim_seq, axis=0),axis=0)
            scale_neu_seq = get_temporal_scaling_trial_multi_sess(
                neu_seq, stim_seq, self.alignment['neu_time'], target_isi)
            # bin data based on isi.
            [bins, bin_center, _, bin_neu_seq, _, _, bin_stim_seq, bin_stim_value] = get_isi_bin_neu(
                scale_neu_seq, stim_seq, stim_value, pre_isi, self.bin_win, bin_num)
            cluster_bin_neu_mean, cluster_bin_neu_sem = get_bin_mean_sem_cluster(bin_neu_seq, cluster_id)
            # plot results for each class.
            l_idx, r_idx = get_frame_idx_from_time(
                self.alignment['neu_time'], 0, -target_isi, stim_seq_target[c_idx,1]+target_isi)
            neu_time = self.alignment['neu_time'][l_idx:r_idx]
            for ci in range(len(lbl)):
                # find bounds.
                upper = np.nanmax(cluster_bin_neu_mean[:,ci,:]) + np.nanmax(cluster_bin_neu_sem[:,ci,:])
                lower = np.nanmin(cluster_bin_neu_mean[:,ci,:]) - np.nanmax(cluster_bin_neu_sem[:,ci,:])
                # find class colors.
                cs = get_cmap_color(bin_num+1, base_color=colors[ci])[:-1]
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
                add_legend(axs[ci], cs,
                           ['[{},{}] ms'.format(int(bins[i]),int(bins[i+1])) for i in range(bin_num)],
                           None, None, None, 'upper right')
        # plot all.
        try: plot_info(axs[0])
        except: pass
        try: plot_local_isi(axs[1])
        except: pass
        try: plot_temporal_scaling(axs[2])
        except: pass
    
    def plot_sorted_heatmaps(self, axs, sort_target, norm_mode):
        bin_num = 3
        cate = [-1,1]
        win_sort = [-500, 500]
        xlim = [-3500, 2500]
        l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, xlim[0], xlim[1])
        # collect data.
        [_, [neu_seq, stim_seq, stim_value, pre_isi],
         [neu_labels, neu_sig], [n_trials, n_neurons]] = get_neu_trial(
            self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
            trial_param=[None, None, None, None, None, [0]],
            mean_sem=False,
            cate=cate, roi_id=None)
        neu_seq = [ns[:,:,l_idx:r_idx] for ns in neu_seq]
        neu_time = self.alignment['neu_time'][l_idx:r_idx]
        # bin data based on isi.
        [_, _, _, neu_x, _, _, bin_stim_seq, _] = get_isi_bin_neu(
            neu_seq, stim_seq, stim_value, pre_isi, self.bin_win, bin_num)
        # find sorting target.
        if sort_target == 'shortest':
            neu_seq_sort = neu_x[0].copy()
        if sort_target == 'longest':
            neu_seq_sort = neu_x[-1].copy()
        # plot heatmaps.
        for bi in range(len(neu_x)):
            self.plot_heatmap_neuron_cate(
                axs[bi], neu_x[bi], neu_time, neu_seq_sort, win_sort, neu_labels, neu_sig,
                norm_mode=norm_mode,
                neu_seq_share=neu_x,
                neu_labels_share=[neu_labels]*len(neu_x),
                neu_sig_share=[neu_sig]*len(neu_x))
            axs[bi].set_xlabel('time since stim (ms)')
            # add stimulus line.
            c_idx = bin_stim_seq.shape[1]//2
            xlines = [neu_time[np.searchsorted(neu_time, t)]
                      for t in [bin_stim_seq[bi,c_idx+i,0]
                                for i in [-1,0]]]
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
    
    def plot_cross_sess_adapt(self, axs):
        cate = [-1,1]
        xlim = [-2500, 2500]
        n_latents = 15
        kernel_time, kernel_all, exp_var_all = self.run_glm(cate)
        model = PCA(n_components=n_latents if n_latents < kernel_all.shape[1] else kernel_all.shape[1])
        neu_x = model.fit_transform(kernel_all)
        _, cluster_id = clustering_neu_response_mode(neu_x, self.n_clusters, None)
        lbl = ['cluster #'+str(ci) for ci in range(self.n_clusters)]
        colors = get_cmap_color(len(lbl), cmap=self.cluster_cmap)
        def plot_sorted_heatmaps(axs, norm_mode):
            xlim = [-2500, 2500]
            l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, xlim[0], xlim[1])
            win_sort = [-500, 500]
            # collect data.
            neu_x = []
            neu_time = self.alignment['neu_time'][l_idx:r_idx]
            _, [neu_seq, _, stim_seq, _], [neu_labels, neu_sig], _ = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_param=[[2,3,4,5], None, None, None, [1], [0]],
                cate=cate, roi_id=None)
            day_split_idx = np.cumsum([nt['dff'].shape[0] for nt in self.list_neural_trials])[:-1]
            neu_x = np.split(neu_seq, day_split_idx, axis=0)
            neu_labels = np.split(neu_labels, day_split_idx)
            neu_sig = np.split(neu_sig, day_split_idx)
            # plot heatmaps.
            for bi in range(len(neu_x)):
                self.plot_heatmap_neuron_cate(
                    axs[bi], neu_x[bi], neu_time, neu_x[bi], win_sort, neu_labels[bi], neu_sig[bi],
                    norm_mode=norm_mode,
                    neu_seq_share=neu_x,
                    neu_labels_share=neu_labels,
                    neu_sig_share=neu_sig)
                axs[bi].set_xlabel('time since pre oddball stim (ms) \n sorting window [{},{}] ms'.format(
                    win_sort[0], win_sort[1]))
            # add stimulus line.
            for bi in range(len(neu_x)):
                c_idx = stim_seq.shape[0]//2
                xlines = [neu_time[np.searchsorted(neu_time, t)]
                          for t in [stim_seq[c_idx+i,0]
                                    for i in [0]]]
                for xl in xlines:
                    if xl>neu_time[0] and xl<neu_time[-1]:
                        axs[bi].axvline(xl, color='black', lw=1, linestyle='--')
        # compare categorized response across sessions.
        def plot_categorization_adapt(axs):
            # collect data.
            [[color0, _, _, _], [neu_seq, _, stim_seq, stim_value],
             _, [n_trials, n_neurons]] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_param=[[2,3,4,5], None, None, None, [1], [0]],
                cate=cate, roi_id=None)
            day_split_idx = np.cumsum([nt['dff'].shape[0] for nt in self.list_neural_trials])[:-1]
            day_neu_seq = np.split(neu_seq, day_split_idx, axis=0)
            day_cluster_id = np.split(cluster_id, day_split_idx)
            # plot results for each class.
            for ci in range(len(lbl)):
                neu_mean = []
                neu_sem = []
                # find class colors.
                cn = get_cmap_color(len(day_split_idx)+2, base_color=colors[ci])
                # get class data for each day.
                for di in range(len(day_split_idx)+1):
                    m, s = get_mean_sem(day_neu_seq[di][day_cluster_id[di]==ci,:])
                    neu_mean.append(m)
                    neu_sem.append(s)
                # find bounds.
                upper = np.nanmax(neu_mean) + np.nanmax(neu_sem)
                lower = np.nanmin(neu_mean) - np.nanmax(neu_sem)
                # plot stimulus.
                axs[ci].fill_between(
                    stim_seq[int(stim_seq.shape[0]/2),:],
                    lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
                    color=color0, edgecolor='none', alpha=0.25, step='mid')
                self.plot_vol(
                    axs[ci], self.alignment['stim_time'],
                    stim_value.reshape(1,-1), color0, upper, lower)
                # plot neural traces.
                for di in range(len(day_split_idx)+1):
                    self.plot_mean_sem(
                        axs[ci], self.alignment['neu_time'],
                        neu_mean[di], neu_sem[di], cn[di], None)
                # adjust layouts.
                adjust_layout_neu(axs[ci])
                axs[ci].set_xlim(xlim)
                axs[ci].set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
                axs[ci].set_xlabel('time since stim (ms)')
                axs[ci].set_title(f'response adaptation across sessions for cluster #{ci}')
                add_legend(axs[ci], cn,
                           [f'day #{di}' for di in range(len(day_split_idx)+1)],
                           None, None, None, 'upper right')
        # compare categorized distribution across sessions.
        def plot_dist_cluster_fraction_in_cate(axs):
            cate_eval = [int(k) for k in self.label_names.keys()]
            # collect data.
            _, _, [neu_labels, _], _ = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_param=[[2,3,4,5], None, None, None, [1], [0]],
                cate=cate, roi_id=None)
            day_split_idx = np.cumsum([nt['dff'].shape[0] for nt in self.list_neural_trials])[:-1]
            day_neu_labels = np.split(neu_labels, day_split_idx)
            day_cluster_id = np.split(cluster_id, day_split_idx)
            # get fraction in each category.
            fraction = np.zeros((len(day_split_idx)+1, self.n_clusters, len(cate_eval)))
            for di in range(len(day_split_idx)+1):
                for i in range(self.n_clusters):
                    for j in range(len(cate_eval)):
                        nc = np.nansum((day_cluster_id[di]==i)*(day_neu_labels[di]==cate_eval[j]))
                        nt = np.nansum(day_neu_labels[di]==cate_eval[j]) + 1e-5
                        fraction[di,i,j] = nc / nt
            # plot lines.
            for ni in range(len(cate_eval)):
                for ci in range(self.n_clusters):
                    axs[ni].plot(fraction[:,ci,ni], color=colors[ci])
            # adjust layout.
            for ni in range(len(cate_eval)):
                axs[ni].tick_params(tick1On=False)
                axs[ni].spines['left'].set_visible(False)
                axs[ni].spines['right'].set_visible(False)
                axs[ni].grid(True, axis='y', linestyle='--')
                axs[ni].set_ylim([np.nanmin(fraction),np.nanmax(fraction)+0.1])
                axs[ni].set_xticks(np.arange(len(day_split_idx)+1))
                axs[ni].set_xticklabels(
                    ['day #{}'.format(di+1) for di in range(len(day_split_idx)+1)],
                    rotation='vertical')
        # plot all.
        try: plot_sorted_heatmaps(axs[0], 'none')
        except: pass
        try: plot_sorted_heatmaps(axs[1], 'minmax')
        except: pass
        try: plot_sorted_heatmaps(axs[2], 'share')
        except: pass
        try: plot_categorization_adapt(axs[3])
        except: pass
        try: plot_dist_cluster_fraction_in_cate(axs[4])
        except: pass

# colors = ['#989A9C', '#A4CB9E', '#9DB4CE', '#EDA1A4', '#F9C08A']
class plotter_main(plotter_utils):
    def __init__(self, neural_trials, labels, significance, label_names, temp_folder):
        super().__init__(neural_trials, labels, significance, temp_folder)
        self.label_names = label_names

    def cluster_all_pre(self, axs_all):
        for cate, axs in zip([[-1,1,2],[-1],[1],[2]], axs_all):
            label_name = self.label_names[str(cate[0])] if len(cate)==1 else 'all'
            print(f'plotting results for {label_name}')
            try:
                
                self.plot_cluster_interval_bin_all(axs, 'pre', cate=cate)
                axs[0].set_title(f'GLM kernels \n {label_name}')
                axs[1].set_title(f'reponse to binned interval \n {label_name}')
                axs[2].set_title(f'fraction of cell-types in cluster \n {label_name}')
                axs[3].set_title(f'legend \n {label_name}')
            except: pass
    
    def cluster_all_post(self, axs_all):
        for cate, axs in zip([[-1,1,2],[-1],[1],[2]], axs_all):
            label_name = self.label_names[str(cate[0])] if len(cate)==1 else 'all'
            print(f'plotting results for {label_name}')
            try:
                
                self.plot_cluster_interval_bin_all(axs, 'post', cate=cate)
                axs[0].set_title(f'GLM kernels \n {label_name}')
                axs[1].set_title(f'reponse to binned interval \n {label_name}')
                axs[2].set_title(f'fraction of cell-types in cluster \n {label_name}')
                axs[3].set_title(f'legend \n {label_name}')
            except: pass
    
    def cluster_individual_pre(self, axs_all):
        for cate, axs in zip([[-1,1,2],[-1],[1],[2]], axs_all):
            label_name = self.label_names[str(cate[0])] if len(cate)==1 else 'all'
            print(f'plotting results for {label_name}')
            try:
                
                self.plot_cluster_interval_bin_individual(axs, 'pre', cate=cate)

            except: pass
    
    def cluster_individual_post(self, axs_all):
        for cate, axs in zip([[-1,1,2],[-1],[1],[2]], axs_all):
            label_name = self.label_names[str(cate[0])] if len(cate)==1 else 'all'
            print(f'plotting results for {label_name}')
            try:
                
                self.plot_cluster_interval_bin_individual(axs, 'post', cate=cate)

            except: pass
        
        
    def sorted_heatmaps(self, axs_all):
        for norm_mode, axs in zip(['none', 'minmax', 'share'], axs_all):
            try:
                
                self.plot_sorted_heatmaps(axs[0], 'shortest', norm_mode)
                for i in range(3):
                    axs[0][i].set_title(
                        f'response to random preceeding interval bin#{i} \n (normalized with {norm_mode}, sorted by shortest)')
                
                self.plot_sorted_heatmaps(axs[1], 'longest', norm_mode)
                for i in range(3):
                    axs[1][i].set_title(
                        f'response to random preceeding interval bin#{i} \n (normalized with {norm_mode}, sorted by longest)')
                    
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
    
    def cross_sess_adapt(self, axs_all):
        for norm_mode, axs in zip(['none', 'minmax', 'share'], axs_all):
            try:

                self.plot_cross_sess_adapt(axs)
                for di in range(5):
                    axs[0][di].set_title(f'response to random stim day {di} \n (normalized with none)')
                    axs[1][di].set_title(f'response to random stim day {di} \n (normalized with minmax)')
                    axs[2][di].set_title(f'response to random stim day {di} \n (normalized with share)')

            except: pass
