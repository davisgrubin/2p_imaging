#!/usr/bin/env python3

import numpy as np
import matplotlib.ticker as mtick
from sklearn.decomposition import PCA

from modules.Alignment import run_get_stim_response
from modeling.clustering import get_mean_sem_cluster
from modeling.clustering import get_bin_mean_sem_cluster
from modeling.clustering import get_cluster_cate
from modeling.decoding import fit_poly_line
from modeling.generative import get_glm_cate
from utils import get_norm01_params
from utils import get_mean_sem_win
from utils import get_mean_sem
from utils import get_neu_trial
from utils import get_frame_idx_from_time
from utils import get_isi_bin_neu
from utils import get_expect_interval
from utils import get_roi_label_color
from utils import get_cmap_color
from utils import hide_all_axis
from utils import add_legend
from utils import utils_basic

# fig, ax = plt.subplots(1, 1, figsize=(5, 5))
# fig, ax = plt.subplots(1, 1, figsize=(3, 9))
# fig, axs = plt.subplots(1, 8, figsize=(24, 3))
# axs = [plt.subplots(1, 8, figsize=(24, 3))[1], plt.subplots(1, 8, figsize=(24, 3))[1]]
# fig, ax = plt.subplots(1, 1, figsize=(6, 6), subplot_kw={"projection": "3d"})

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
        self.bin_win = [2450,7550]
        self.bin_num = 11
        self.n_clusters = 7
        self.max_clusters = 10
        self.d_latent = 3
        self.glm = self.run_glm()
        self.cluster_id = self.run_clustering()

    def plot_cluster_interval_bin_all(self, axs, cate=None):
        kernel_all = get_glm_cate(self.glm, self.list_labels, self.list_significance, cate)
        cluster_id, neu_labels = get_cluster_cate(self.cluster_id, self.list_labels, self.list_significance, cate)
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
        def plot_interval_bin(ax, mode):
            # bin data based on isi.
            if mode == 'pre':
                xlim = [-9000, 1500]
                isi = pre_isi
                isi_idx_offset = -1
            if mode == 'post':
                xlim = [-1000, 9500]
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
        def plot_neu_fraction(ax):
            self.plot_cluster_neu_fraction_in_cluster(ax, cluster_id, color2)
        def plot_fraction(ax):
            self.plot_cluster_cate_fraction_in_cluster(ax, cluster_id, neu_labels, self.label_names)
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
        try: plot_neu_fraction(axs[4])
        except Exception as e: print(e)
        try: plot_fraction(axs[5])
        except Exception as e: print(e)
        try: plot_legend(axs[6])
        except Exception as e: print(e)
    
    def plot_cluster_heatmap_all(self, axs, cate):
        kernel_all = get_glm_cate(self.glm, self.list_labels, self.list_significance, cate)
        cluster_id, neu_labels = get_cluster_cate(self.cluster_id, self.list_labels, self.list_significance, cate)
        # collect data.
        [[_, _, _, cmap], [_, _, stim_seq, _], _, _] = get_neu_trial(
            self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
            trial_param=[[2,3,4,5], None, None, None, [1], [0]],
            cate=cate, roi_id=None)
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
        def plot_hierarchical_dendrogram(ax):
            # plot results.
            self.plot_dendrogram(ax, kernel_all, cmap)
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

# colors = ['#989A9C', '#A4CB9E', '#9DB4CE', '#EDA1A4', '#F9C08A']
class plotter_main(plotter_utils):
    def __init__(self, neural_trials, labels, significance, label_names, temp_folder, cate_list):
        super().__init__(neural_trials, labels, significance, temp_folder, cate_list)
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
            except Exception as e: print(e)
    
    def cluster_heatmap_all(self, axs_all):
        for cate, axs in zip([[-1,1,2],[-1],[1],[2]], axs_all):
            try:
                label_name = self.label_names[str(cate[0])] if len(cate)==1 else 'all'
                print(f'plotting results for {label_name}')

                self.plot_cluster_heatmap_all(axs, cate=cate)
                axs[0].set_title(f'clustered latent features \n {label_name}')
                axs[1].set_title(f'cluster dendrogram \n {label_name}')
                axs[2].set_title(f'GLM kernel course \n {label_name}')

            except Exception as e: print(e)