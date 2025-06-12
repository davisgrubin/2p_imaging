#!/usr/bin/env python3

import numpy as np
import matplotlib.colors as mcolors
from sklearn.decomposition import PCA

from modules.Alignment import run_get_stim_response
from modeling.clustering import get_mean_sem_cluster
from modeling.clustering import get_bin_mean_sem_cluster
from modeling.clustering import get_cluster_cate
from modeling.decoding import neu_pop_sample_decoding_slide_win
from modeling.generative import get_glm_cate
from modeling.quantifications import run_quantification
from utils import get_norm01_params
from utils import get_mean_sem
from utils import get_neu_trial
from utils import get_frame_idx_from_time
from utils import get_isi_bin_neu
from utils import get_expect_interval
from utils import get_split_idx
from utils import get_roi_label_color
from utils import get_cmap_color
from utils import apply_colormap
from utils import adjust_layout_neu
from utils import adjust_layout_heatmap
from utils import add_legend
from utils import add_heatmap_colorbar
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
            temp_folder,
            ):
        super().__init__()
        self.n_sess = len(list_neural_trials)
        self.list_labels = list_labels
        self.list_neural_trials = list_neural_trials
        self.alignment = run_get_stim_response(temp_folder, list_neural_trials, expected='none')
        self.list_stim_labels = self.alignment['list_stim_labels']
        self.expect = np.array(np.mean([get_expect_interval(sl)[0] for sl in self.list_stim_labels]))
        self.list_significance = list_significance
        self.bin_win = [450,2550]
        self.bin_num = 4
        self.n_clusters = 6
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
        def plot_glm_kernel(ax):
            c_idx = np.nanmean(np.concatenate(stim_seq, axis=0), axis=0).shape[0]//2
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
            # adjust layout.
            ax.set_xlabel('time since stim (ms)')
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
        try: plot_neu_fraction(axs[3])
        except Exception as e: print(e)
        try: plot_fraction(axs[4])
        except Exception as e: print(e)
        try: plot_legend(axs[5])
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
            # adjust layout.
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
            # adjust layout.
            ax.set_xlabel('time since stim (ms)')
            ax.axvline(stim_seq[c_idx,0], color='black', lw=1, linestyle='--')
        # plot all.
        try: plot_cluster_features(axs[0])
        except Exception as e: print(e)
        try: plot_hierarchical_dendrogram(axs[1])
        except Exception as e: print(e)
        try: plot_glm_kernel(axs[2])
        except Exception as e: print(e)

    def plot_cluster_interval_bin_individual(self, axs, mode, cate=None):
        cluster_id, neu_labels = get_cluster_cate(self.cluster_id, self.list_labels, self.list_significance, cate)
        lbl = ['cluster #'+str(ci) for ci in range(self.n_clusters)]
        split_idx = get_split_idx(self.list_labels, self.list_significance, cate)
        day_cluster_id = np.split(cluster_id, split_idx)
        # collect data.
        [[color0, _, color2, cmap],
         [neu_seq, stim_seq, camera_pupil, pre_isi, post_isi],
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
        [bins, bin_center, bin_neu_seq_trial, bin_neu_seq, _, _, bin_stim_seq, bin_camera_pupil] = get_isi_bin_neu(
            neu_seq, stim_seq, camera_pupil, isi, self.bin_win, self.bin_num)
        c_idx = bin_stim_seq.shape[1]//2
        cs = get_cmap_color(self.bin_num, base_color=color2)
        l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, xlim[0], xlim[1])
        def plot_interval_bin(axs):
            # get response within cluster at each bin.
            cluster_bin_neu_mean, cluster_bin_neu_sem = get_bin_mean_sem_cluster(bin_neu_seq, self.n_clusters, cluster_id)
            # plot results for each class.
            for ci in range(self.n_clusters):
                if np.sum(cluster_id==ci) > 0:
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
                    '''
                    # plot pupil.
                    for bi in range(self.bin_num):
                        self.plot_pupil(
                            axs[ci], self.alignment['neu_time'][l_idx:r_idx], bin_camera_pupil[bi, l_idx:r_idx],
                            cs[bi], upper, lower)
                    '''
                    # plot neural traces.
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
        def plot_trial_quant(axs):
            # plot results for each class.
            for ci in range(self.n_clusters):
                if np.sum(cluster_id==ci) > 0:
                    axs[ci].axis('off')
                    ax0 = axs[ci].inset_axes([0, 0.9, 1, 0.1], transform=axs[ci].transAxes)
                    ax1 = axs[ci].inset_axes([0, 0, 1, 0.7], transform=axs[ci].transAxes)
                    # plot evaluation window.
                    win_eval = 0
                    for i in [c_idx, c_idx+1]:
                        ax0.fill_between(
                            stim_seq[i,:], 0, 1,
                            color=color0, edgecolor='none', alpha=0.25, step='mid')
                    self.plot_win_mag_quant_win_eval(ax0, win_eval, color0, xlim)
                    for bi in range(self.bin_num):
                        # define evaluation windows.
                        win_eval = [[-2500, 0],
                                    [stim_seq[c_idx+1,0],stim_seq[c_idx+1,0]+250],
                                    [stim_seq[c_idx+1,0]+300,stim_seq[c_idx+1,0]+550],
                                    [stim_seq[c_idx+1,0]+600,stim_seq[c_idx+1,0]+850]]
                        # average across neurons within cluster.
                        bin_ci = np.concatenate(
                            [np.nanmean(bin_neu_seq_trial[:,sci==ci,:],axis=1)
                             for neu,sci in zip(neu_seq,day_cluster_id)], axis=0)
                    
        def plot_interval_bin_quant(axs):
            target_metrics = ['evoke_mag', 'onset_drop']
            # plot results for each class.
            for ci in range(self.n_clusters):
                if np.sum(cluster_id==ci) > 0:
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
                neu_seq, stim_seq, camera_pupil, isi, self.bin_win, bin_num_isi)
            # get response within cluster at each bin.
            cluster_bin_neu_mean, _ = get_bin_mean_sem_cluster(b_neu_seq, self.n_clusters, cluster_id)
            # plot results for each class.
            for ci in range(self.n_clusters):
                if np.sum(cluster_id==ci) > 0:
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
        except Exception as e: print(e)
        try: plot_interval_bin_quant(axs[1])
        except Exception as e: print(e)
        try: plot_interval_heatmap(axs[2])
        except Exception as e: print(e)
        try: plot_legend(axs[3])
        except Exception as e: print(e)
    
    def plot_separability_local(self, axs, cate):
        min_isi = 2000
        xlim = [-4000, 5000]
        win_sample = 500
        win_step = 1
        lbl = ['cluster #'+str(ci) for ci in range(self.n_clusters)]
        cluster_id, neu_labels = get_cluster_cate(self.cluster_id, self.list_labels, self.list_significance, cate)
        def plot_separability(axs):
            # collect data.
            [[color0, _, color2, _],
             [neu_seq, stim_seq, camera_pupil, pre_isi, post_isi], _, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_param=[[2,3,4,5], None, None, None, [1], [0]],
                mean_sem=False,
                cate=cate, roi_id=None)
            cs = get_cmap_color(self.bin_num, base_color=color2)
            neu_seq = [neu_seq[i][post_isi[i]>min_isi,:,:] for i in range(self.n_sess)]
            stim_seq = [stim_seq[i][post_isi[i]>min_isi,:,:] for i in range(self.n_sess)]
            pre_isi = [pre_isi[i][post_isi[i]>min_isi] for i in range(self.n_sess)]
            # bin data based on isi.
            [_, bin_center, bin_neu_seq_trial, bin_neu_seq, _, _, bin_stim_seq, _] = get_isi_bin_neu(
                neu_seq, stim_seq, camera_pupil, pre_isi, self.bin_win, self.bin_num)
            # prepare data for decoding.
            bss = np.nanmean(bin_stim_seq, axis=0)
            c_idx = bss.shape[0]//2
            win_decode = xlim
            neu_y = np.arange(self.bin_num)
            neu_x = np.concatenate([np.expand_dims(bns,axis=0) for bns in bin_neu_seq], axis=0)
            # run decoding for all neurons.
            [llh_time, llh_mean, llh_sem, llh_chance] = neu_pop_sample_decoding_slide_win(
                 neu_x, neu_y, self.alignment['neu_time'],
                 win_decode, win_sample, win_step)
            # plot results for each class.
            for ci in range(self.n_clusters):
                if np.sum(cluster_id==ci) > 0:
                    axs[ci].axis('off')
                    ax0 = axs[ci].inset_axes([0, 0.7,  0.7, 0.3], transform=axs[ci].transAxes)
                    ax1 = axs[ci].inset_axes([0, 0.25, 0.7, 0.4], transform=axs[ci].transAxes, sharex=ax0)
                    ax2 = axs[ci].inset_axes([0, 0,    0.7, 0.2], transform=axs[ci].transAxes, sharex=ax0)
                    axss = [ax0, ax1, ax2]
                    llh_results = neu_pop_sample_decoding_slide_win(
                        neu_x[:,cluster_id==ci,:], neu_y, self.alignment['neu_time'],
                        win_decode, win_sample, win_step)
                    # find bounds.
                    upper = [np.nanmax([llh_mean]), np.nanmax([llh_results[1]]), llh_chance+0.05]
                    lower = [np.nanmin([llh_mean]), np.nanmin([llh_results[1]]), llh_chance-0.05]
                    # plot stimulus.
                    stim_seq = np.nanmean(bin_stim_seq, axis=0)
                    c_idx = stim_seq.shape[0]//2
                    for ai in range(len(axss)):
                        axss[ai].axvline(stim_seq[c_idx,1]+self.expect, color='gold', lw=1, linestyle='--')
                        for bi in range(self.bin_num):
                            axss[ai].axvline(bin_stim_seq[bi,c_idx-1,0], color=cs[bi], lw=1, linestyle='--')
                        for si in [c_idx, c_idx+1]:
                            axss[ai].fill_between(
                                stim_seq[si,:],
                                lower[ai] - 0.1*(upper[ai]-lower[ai]), upper[ai] + 0.1*(upper[ai]-lower[ai]),
                                color=color0, edgecolor='none', alpha=0.25, step='mid')
                    # plot results.
                    self.plot_mean_sem(
                        axss[0], self.alignment['neu_time'][llh_time], llh_mean, llh_sem, color0, None)
                    self.plot_mean_sem(
                        axss[1], self.alignment['neu_time'][llh_time], llh_results[1], llh_results[2], color2, None)
                    axss[2].axhline(llh_chance, color=color0, lw=1, linestyle='--')
                    # adjust layouts.
                    axs[ci].set_title(lbl[ci])
                    axss[0].spines['bottom'].set_visible(False)
                    axss[0].xaxis.set_visible(False)
                    axss[1].set_ylabel('log likelihood')
                    axss[1].spines['bottom'].set_visible(False)
                    axss[1].xaxis.set_visible(False)
                    axss[2].set_xlabel('time since pre oddball stim (ms)')
                    axss[2].set_yticks(np.round(llh_chance, 2))
                    for ai in range(len(axss)):
                        axss[ai].tick_params(axis='y', tick1On=False)
                        axss[ai].spines['right'].set_visible(False)
                        axss[ai].spines['top'].set_visible(False)
                        axss[ai].set_xlim(xlim)
                        axss[ai].set_ylim([lower[ai], upper[ai]])
        def plot_legend(ax):
            [[color0, color1, color2, _],
             [neu_seq, stim_seq, camera_pupil, pre_isi, _], _,
             [n_trials, n_neurons]] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_param=[[2,3,4,5], None, None, None, [1], [0]],
                mean_sem=False,
                cate=cate, roi_id=None)
            [bins, _, _, _, _, _, _, _] = get_isi_bin_neu(
                neu_seq, stim_seq, camera_pupil, pre_isi, self.bin_win, self.bin_num)
            cs = ['gold'] + get_cmap_color(self.bin_num, base_color=color2)
            lbl = ['expected'] + ['[{},{}] ms'.format(int(bins[i]),int(bins[i+1])) for i in range(self.bin_num)]
            add_legend(ax, cs, lbl, n_trials, n_neurons, self.n_sess, 'upper right')
            ax.axis('off')
        # plot all.
        try: plot_separability(axs[0])
        except Exception as e: print(e)
        try: plot_legend(axs[1])
        except Exception as e: print(e)
    
    def plot_cross_sess_adapt(self, axs, cate):
        cluster_id, neu_labels = get_cluster_cate(self.cluster_id, self.list_labels, self.list_significance, cate)
        lbl = ['cluster #'+str(ci) for ci in range(self.n_clusters)]
        split_idx = get_split_idx(self.list_labels, self.list_significance, cate)
        day_cluster_id = np.split(cluster_id, split_idx)
        [[color0, color1, color2, _],
         [neu_seq, stim_seq, camera_pupil, _, _], _,
         [n_trials, n_neurons]] = get_neu_trial(
            self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
            trial_param=[[2,3,4,5], None, None, None, [1], [0]],
            mean_sem=False,
            cate=cate, roi_id=None)
        def plot_dist_cluster_fraction(axs):
            bar_width = 0.5
            # get fraction in each category.
            fraction = np.zeros((len(split_idx)+1, self.n_clusters))
            for di in range(len(split_idx)+1):
                for ci in range(self.n_clusters):
                    nc = np.nansum(day_cluster_id[di]==ci)
                    nt = len(day_cluster_id[di]) + 1e-5
                    fraction[di,ci] = nc / nt
            # plot results for each class.
            for ci in range(self.n_clusters):
                if np.sum(cluster_id==ci) > 0:
                    axs[ci].axis('off')
                    axs[ci] = axs[ci].inset_axes([0, 0, 1, 0.5], transform=axs[ci].transAxes)
                    for di in range(len(split_idx)+1):
                        axs[ci].bar(
                            di, fraction[di,ci],
                            bottom=0, edgecolor='white', width=bar_width, color=color2)
            # adjust layout.
            for ci in range(self.n_clusters):
                axs[ci].tick_params(tick1On=False)
                axs[ci].spines['top'].set_visible(False)
                axs[ci].spines['left'].set_visible(False)
                axs[ci].spines['right'].set_visible(False)
                axs[ci].grid(True, axis='y', linestyle='--')
                axs[ci].set_ylim([0,np.nanmax(fraction[:,ci])+0.05])
                axs[ci].set_xticks(np.arange(len(split_idx)+1))
                axs[ci].set_xticklabels(
                    ['day #{}'.format(di+1) for di in range(len(split_idx)+1)],
                    rotation='vertical')
                axs[ci].set_title(lbl[ci])
        def plot_raw_traces(axs, di):
            xlim = [-7500,7500]
            # collect data.
            day_cluster_id = np.split(cluster_id, split_idx)
            # plot results for each class.
            for ci in range(self.n_clusters):
                if np.sum(day_cluster_id[di-1]==ci) > 0:
                    axs[ci].axis('off')
                    ax0 = axs[ci].inset_axes([0, 0.5, 1, 0.4], transform=axs[ci].transAxes)
                    ax1 = axs[ci].inset_axes([0, 0, 1, 0.4], transform=axs[ci].transAxes)
                    neu_mean0, neu_sem0 = get_mean_sem(neu_seq[di-1][0,day_cluster_id[di-1]==ci,:])
                    neu_mean1, neu_sem1 = get_mean_sem(neu_seq[di-1][-1,day_cluster_id[di-1]==ci,:])
                    # find bounds.
                    upper = np.nanmax([neu_mean0, neu_mean1]) + np.nanmax([neu_sem0, neu_sem1])
                    lower = np.nanmin([neu_mean0, neu_mean1]) - np.nanmax([neu_sem0, neu_sem1])
                    # plot stimulus.
                    for i in range(stim_seq[di-1][0].shape[0]):
                        ax0.fill_between(
                            stim_seq[di-1][0,i,:],
                            lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
                            color=color0, edgecolor='none', alpha=0.25, step='mid')
                    for i in range(stim_seq[di-1][-1].shape[0]):
                        ax1.fill_between(
                            stim_seq[di-1][-1,i,:],
                            lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
                            color=color0, edgecolor='none', alpha=0.25, step='mid')
                    # plot traces.
                    self.plot_mean_sem(ax0, self.alignment['neu_time'], neu_mean0, neu_sem0, color0, None)
                    self.plot_mean_sem(ax1, self.alignment['neu_time'], neu_mean1, neu_sem1, color2, None)
                    # adjust layout.
                    axs[ci].set_title(lbl[ci] + '\n' + 'day {}'.format(di))
                    ax0.spines['bottom'].set_visible(False)
                    ax0.xaxis.set_visible(False)
                    ax1.set_ylabel('df/f (z-scored)')
                    ax1.set_xlabel('time since stim (ms)')
                    for ax in [ax0, ax1]:
                        ax.tick_params(axis='y', tick1On=False)
                        ax.spines['right'].set_visible(False)
                        ax.spines['top'].set_visible(False)
                        ax.set_xlim(xlim)
                        ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
        def plot_legend(ax):
            cs = [color0, color2]
            lbl = ['first epoch', 'last epoch']
            add_legend(ax, cs, lbl, n_trials, n_neurons, self.n_sess, 'upper right')
            ax.axis('off')
        # plot all.
        try: plot_dist_cluster_fraction(axs[0])
        except Exception as e: print(e)
        try: plot_raw_traces(axs[1], 1)
        except Exception as e: print(e)
        try: plot_raw_traces(axs[2], 5)
        except Exception as e: print(e)
        try: plot_legend(axs[3])
        except Exception as e: print(e)

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
    
    def cluster_individual_pre(self, axs_all):
        for cate, axs in zip([[-1,1,2],[-1],[1],[2]], axs_all):
            try:
                label_name = self.label_names[str(cate[0])] if len(cate)==1 else 'all'
                print(f'plotting results for {label_name}')
                
                self.plot_cluster_interval_bin_individual(axs, 'pre', cate=cate)

            except Exception as e: print(e)
    
    def cluster_individual_post(self, axs_all):
        for cate, axs in zip([[-1,1,2],[-1],[1],[2]], axs_all):
            try:
                label_name = self.label_names[str(cate[0])] if len(cate)==1 else 'all'
                print(f'plotting results for {label_name}')
                
                self.plot_cluster_interval_bin_individual(axs, 'post', cate=cate)

            except Exception as e: print(e)
    
    def separability_local(self, axs_all):
        for cate, axs in zip([[-1,1,2],[-1],[1],[2]], axs_all):
            try:
                label_name = self.label_names[str(cate[0])] if len(cate)==1 else 'all'
                print(f'plotting results for {label_name}')
                
                self.plot_separability_local(axs, cate=cate)

            except Exception as e: print(e)

    def cross_sess_adapt(self, axs_all):
        for cate, axs in zip([[-1,1,2],[-1],[1],[2]], axs_all):
            try:
                label_name = self.label_names[str(cate[0])] if len(cate)==1 else 'all'
                print(f'plotting results for {label_name}')
                
                self.plot_cross_sess_adapt(axs, cate=cate)

            except Exception as e: print(e)
        