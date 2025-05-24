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
        self.n_sess = len(list_neural_trials)
        self.list_labels = list_labels
        self.list_neural_trials = list_neural_trials
        self.alignment = run_get_stim_response(temp_folder, list_neural_trials, expected='none')
        self.list_stim_labels = self.alignment['list_stim_labels']
        self.list_significance = list_significance
        self.bin_win = [450,2550]
        self.bin_num = 5
        self.n_clusters = 8
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

    def plot_cluster_interval_bin_all(self, axs, cate=None):
        kernel_all, cluster_id, neu_labels = self.run_clustering(cate)
        # collect data.
        [[color0, _, color2, _],
         [neu_seq, stim_seq, _, pre_isi, post_isi],
         [neu_labels, _],
         [n_trials, n_neurons]] = get_neu_trial(
            self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
            trial_param=[[2,3,4,5], None, None, None, [1], [0]],
            mean_sem=False,
            cate=cate, roi_id=None)
        colors = get_cmap_color(self.bin_num, base_color=color2)
        def plot_glm_kernel(ax):
            c_idx = np.nanmean(np.concatenate(stim_seq, axis=0), axis=0).shape[0]//2
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
            [bins, bin_center, _, bin_neu_seq, _, _, bin_stim_seq] = get_isi_bin_neu(
                neu_seq, stim_seq, isi, self.bin_win, self.bin_num)
            c_idx = bin_stim_seq.shape[1]//2
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
        def plot_neu_fraction(ax):
            self.plot_cluster_neu_fraction_in_cluster(ax, cluster_id, color2)
        def plot_fraction(ax):
            self.plot_cluster_cate_fraction_in_cluster(ax, cluster_id, neu_labels, self.label_names)
        def plot_legend(ax):
            [bins, _, _, _, _, _, _] = get_isi_bin_neu(
                neu_seq, stim_seq, pre_isi, self.bin_win, self.bin_num)
            lbl = ['[{},{}] ms'.format(int(bins[i]),int(bins[i+1])) for i in range(self.bin_num)]
            lbl+= [v for v in self.label_names.values()] + ['all cell-types']
            cs = colors + [get_roi_label_color(cate=[int(k)])[2] for k in self.label_names.keys()]
            cs+= [get_roi_label_color(cate=[-1,1,2])[2]]
            add_legend(ax, cs, lbl, n_trials, n_neurons, self.n_sess, 'upper right')
            ax.axis('off')
        # plot all.
        try: plot_glm_kernel(axs[0])
        except: pass
        try: plot_interval_bin(axs[1], 'pre')
        except: pass
        try: plot_interval_bin(axs[2], 'post')
        except: pass
        try: plot_neu_fraction(axs[3])
        except: pass
        try: plot_fraction(axs[4])
        except: pass
        try: plot_legend(axs[5])
        except: pass

    def plot_cluster_interval_bin_individual(self, axs, mode, cate=None):
        kernel_all, cluster_id, neu_labels = self.run_clustering(cate)
        lbl = ['cluster #'+str(ci) for ci in range(self.n_clusters)]
        # collect data.
        [[color0, _, color2, cmap],
         [neu_seq, stim_seq, _, pre_isi, post_isi],
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
        [bins, bin_center, _, bin_neu_seq, _, _, bin_stim_seq] = get_isi_bin_neu(
            neu_seq, stim_seq, isi, self.bin_win, self.bin_num)
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
            [_, b_center, _, b_neu_seq, _, _, _] = get_isi_bin_neu(
                neu_seq, stim_seq, isi, self.bin_win, bin_num_isi)
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


# colors = ['#989A9C', '#A4CB9E', '#9DB4CE', '#EDA1A4', '#F9C08A']
class plotter_main(plotter_utils):
    def __init__(self, neural_trials, labels, significance, label_names, temp_folder):
        super().__init__(neural_trials, labels, significance, temp_folder)
        self.label_names = label_names

    def cluster_all(self, axs_all):
        for cate, axs in zip([[-1,1,2],[-1],[1],[2]], axs_all):
            try:
                label_name = self.label_names[str(cate[0])] if len(cate)==1 else 'all'
                print(f'plotting results for {label_name}')
                
                self.plot_cluster_interval_bin_all(axs, cate=cate)
                axs[0].set_title(f'GLM kernels \n {label_name}')
                axs[1].set_title(f'reponse to binned (pre) interval \n {label_name}')
                axs[2].set_title(f'reponse to binned (post) interval \n {label_name}')
                axs[3].set_title(f'fraction of neurons in cluster \n {label_name}')
                axs[4].set_title(f'fraction of cell-types in cluster \n {label_name}')
                axs[5].set_title(f'legend \n {label_name}')
            except: pass
    
    def cluster_individual_pre(self, axs_all):
        for cate, axs in zip([[-1,1,2],[-1],[1],[2]], axs_all):
            try:
                label_name = self.label_names[str(cate[0])] if len(cate)==1 else 'all'
                print(f'plotting results for {label_name}')
                
                self.plot_cluster_interval_bin_individual(axs, 'pre', cate=cate)

            except: pass
    
    def cluster_individual_post(self, axs_all):
        for cate, axs in zip([[-1,1,2],[-1],[1],[2]], axs_all):
            try:
                label_name = self.label_names[str(cate[0])] if len(cate)==1 else 'all'
                print(f'plotting results for {label_name}')
                
                self.plot_cluster_interval_bin_individual(axs, 'post', cate=cate)

            except: pass
        