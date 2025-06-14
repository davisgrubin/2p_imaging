#!/usr/bin/env python3

import numpy as np
from sklearn.decomposition import PCA

from modules.Alignment import run_get_stim_response
from modeling.clustering import get_mean_sem_cluster
from modeling.clustering import get_bin_mean_sem_cluster
from modeling.clustering import get_cluster_cate
from modeling.generative import get_glm_cate
from modeling.quantifications import run_quantification
from utils import get_norm01_params
from utils import get_odd_stim_prepost_idx
from utils import get_mean_sem
from utils import get_neu_trial
from utils import get_frame_idx_from_time
from utils import get_block_1st_idx
from utils import get_block_transition_idx
from utils import get_isi_bin_neu
from utils import get_expect_interval
from utils import get_roi_label_color
from utils import get_cmap_color
from utils import adjust_layout_neu
from utils import adjust_layout_3d_latent
from utils import add_legend
from utils import utils_basic

# fig, ax = plt.subplots(1, 1, figsize=(6, 6))
# fig, ax = plt.subplots(1, 1, figsize=(2, 20))
# fig, ax = plt.subplots(1, 1, figsize=(3, 9))
# fig, axs = plt.subplots(1, 8, figsize=(24, 3))
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
        self.list_odd_idx = [
            get_odd_stim_prepost_idx(sl) for sl in self.list_stim_labels]
        self.expect = np.nanmin(np.array([get_expect_interval(sl) for sl in self.list_stim_labels]),axis=0)
        self.list_block_start = [get_block_1st_idx(sl, 3) for sl in self.list_stim_labels]
        self.list_significance = list_significance
        self.bin_win = [450,2550]
        self.bin_num = 3
        self.n_clusters = 6
        self.max_clusters = 10
        self.d_latent = 3
        self.glm = self.run_glm()
        self.cluster_id = self.run_clustering()

    def plot_cluster_all(self, axs, cate):
        kernel_all = get_glm_cate(self.glm, self.list_labels, self.list_significance, cate)
        cluster_id, neu_labels = get_cluster_cate(self.cluster_id, self.list_labels, self.list_significance, cate)
        def plot_glm_kernel(ax):
            ax.axis('off')
            ax = ax.inset_axes([0, 0, 0.5, 1], transform=ax.transAxes)
            # collect data.
            [[color0, _, _, _], [_, _, stim_seq, _], _, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_param=[[2,3,4,5], None, [0], None, [0], [0]],
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
        def plot_standard(ax, standard):
            xlim = [-2000,3000]
            # collect data.
            [[color0, color1, color2, _],
             [neu_seq, _, stim_seq, _], _, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_param=[[2,3,4,5], [standard], None, None, [0], [0]],
                cate=cate, roi_id=None)
            # get response within cluster at each bin.
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
        def plot_neu_fraction(ax):
            # collect data.
            [[_, _, color2, _], _, _, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_param=[None, None, [0], None, [0], [0]],
                cate=cate, roi_id=None)
            # plot results.
            self.plot_cluster_neu_fraction_in_cluster(ax, cluster_id, color2)
        def plot_fraction(ax):
            self.plot_cluster_cate_fraction_in_cluster(ax, cluster_id, neu_labels, self.label_names)
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
        try: plot_standard(axs[3], 0)
        except Exception as e: print(e)
        try: plot_standard(axs[4], 1)
        except Exception as e: print(e)
        try: plot_oddball(axs[5], 0)
        except Exception as e: print(e)
        try: plot_oddball(axs[6], 1)
        except Exception as e: print(e)
        try: plot_neu_fraction(axs[7])
        except Exception as e: print(e)
        try: plot_fraction(axs[8])
        except Exception as e: print(e)
        try: plot_legend(axs[9])
        except Exception as e: print(e)

    def plot_cluster_heatmap_all(self, axs, cate):
        kernel_all = get_glm_cate(self.glm, self.list_labels, self.list_significance, cate)
        cluster_id, neu_labels = get_cluster_cate(self.cluster_id, self.list_labels, self.list_significance, cate)
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
            # collect data.
            [_, _, _, cmap], _, _, _ = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_param=[[2,3,4,5], None, [0], None, [0], [0]],
                cate=cate, roi_id=None)
            # plot results.
            self.plot_dendrogram(ax, kernel_all, cmap)
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

    def plot_cluster_adapt_individual(self, axs, cate=None):
        trials_around = 15
        trials_eval = 5
        cluster_id, neu_labels = get_cluster_cate(self.cluster_id, self.list_labels, self.list_significance, cate)
        lbl = ['cluster #'+str(ci) for ci in range(self.n_clusters)]
        # collect data.
        [_, color1, color2, _], [_, _, stim_seq, _], _, [n_trials, n_neurons] = get_neu_trial(
            self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
            trial_param=[None, None, None, None, [0], [0]],
            cate=cate, roi_id=None)
        c_idx = stim_seq.shape[0]//2
        # get transition trials indice.
        list_trans_0to1 = [get_block_transition_idx(sl[:,3], trials_around)[0] for sl in self.list_stim_labels]
        list_trans_1to0 = [get_block_transition_idx(sl[:,3], trials_around)[1] for sl in self.list_stim_labels]
        list_trans_0to1 = [np.nansum(ti, axis=0).astype('bool') for ti in list_trans_0to1]
        list_trans_1to0 = [np.nansum(ti, axis=0).astype('bool') for ti in list_trans_1to0]
        def plot_tansition_trial_heatmap(axs, norm_mode):
            gap_margin = 5
            xlim = [-500,2500]
            l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, xlim[0], xlim[1])
            neu_time = self.alignment['neu_time'][l_idx:r_idx]
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
            # extract transition for each class.
            neu_x = []
            zero_gap = np.full((gap_margin, r_idx - l_idx), np.nan)
            for ci in range(self.n_clusters):
                left = np.nanmean(neu_0to1[:, cluster_id==ci, l_idx:r_idx], axis=1)
                right = np.nanmean(neu_1to0[:, cluster_id==ci, l_idx:r_idx], axis=1)
                neu = np.concatenate([left, zero_gap, right], axis=0)
                neu_x.append(neu)
            # plot results for each class.
            for ci in range(self.n_clusters):
                cmap, _ = get_cmap_color(1, base_color=color2, return_cmap=True)
                self.plot_heatmap_trial(
                    axs[ci], neu_x[ci], neu_time, cmap, norm_mode, neu_x)
                axs[ci].set_xlabel('time since stim (ms)')
                axs[ci].set_ylabel('trial since transition')
            # add stimulus line.
            n_trials = neu_0to1.shape[0]
            total = 2 * n_trials + gap_margin
            for ci in range(self.n_clusters):
                z, _ = get_frame_idx_from_time(neu_time, 0, 0, 0)
                e1, e2 = get_frame_idx_from_time(
                    neu_time, 0,
                    stim_seq[c_idx,1]+self.expect[0],
                    stim_seq[c_idx,1]+self.expect[1])
                # stimulus before interval.
                axs[ci].axvline(neu_time[z], 0/total, n_trials/total, color='black', lw=1, linestyle='--')
                axs[ci].axvline(neu_time[z], (n_trials+gap_margin)/total, 1, color='black', lw=1, linestyle='--')
                # stimulus after short interval.
                axs[ci].axvline(neu_time[e1], 0, (n_trials/2)/total, color='black', lw=1, linestyle='--')
                axs[ci].axvline(neu_time[e1], (n_trials+gap_margin+n_trials/2)/total, 1, color='black', lw=1, linestyle='--')
                # stimulus after long interval.
                axs[ci].axvline(neu_time[e2], (n_trials/2)/total, n_trials/total, color='black', lw=1, linestyle='--')
                axs[ci].axvline(neu_time[e2], (n_trials+gap_margin)/total, (n_trials+gap_margin+n_trials/2)/total, color='black', lw=1, linestyle='--')
                # stimulus before and after transition.
                axs[ci].scatter(neu_time[e1], n_trials/2, color='red')
                axs[ci].scatter(neu_time[z], n_trials/2-0.5, color='red')
                axs[ci].scatter(neu_time[e2], n_trials+gap_margin+n_trials/2, color='red')
                axs[ci].scatter(neu_time[z], n_trials+gap_margin+n_trials/2-1, color='red')
            # adjust layouts.
            for ci in range(self.n_clusters):
                axs[ci].set_title(lbl[ci])
                axs[ci].set_yticks([
                    0, int(n_trials/2), n_trials,
                    n_trials + gap_margin, int(n_trials/2+n_trials + gap_margin), total])
                axs[ci].set_yticklabels([
                    int(n_trials/2), 0, -int(n_trials/2), int(n_trials/2), 0, -int(n_trials/2)])
                axs[ci].set_ylim([0, total])
        def plot_tansition(axs, standard):
            xlim = [-7500,9000]
            # collect data.
            [[color0, color1, color2, _],
             [neu_trans, _, stim_seq, _], _,
             [n_trials, n_neurons]] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_idx=[l[1-standard] for l in self.list_block_start],
                cate=cate, roi_id=None)
            # plot results for each class.
            for ci in range(self.n_clusters):
                axs[ci].axis('off')
                axs[ci] = axs[ci].inset_axes([0, 0, 1, 0.5], transform=axs[ci].transAxes)
                if np.sum(cluster_id==ci) > 0:
                    neu_mean, neu_sem = get_mean_sem(neu_trans[cluster_id==ci,:])
                    c_idx = stim_seq.shape[0]//2
                    c_stim = [[color1, color2][standard]] * (c_idx)
                    c_stim+= [[color1, color2][1-standard]] * (stim_seq.shape[0]-c_idx)
                    # find bounds.
                    upper = np.nanmax(neu_mean) + np.nanmax(neu_sem)
                    lower = np.nanmin(neu_mean) - np.nanmax(neu_sem)
                    # plot stimulus.
                    for i in range(stim_seq.shape[0]):
                        axs[ci].fill_between(
                            stim_seq[i,:],
                            lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
                            color=color0, edgecolor='none', alpha=0.25, step='mid')
                    axs[ci].axvline(0, color='red', lw=1, linestyle='--')
                    # plot neural traces.
                    z_idx, _ = get_frame_idx_from_time(self.alignment['neu_time'], 0, 0, 0)
                    self.plot_mean_sem(
                        axs[ci], self.alignment['neu_time'][:z_idx],
                        neu_mean[:z_idx], neu_sem[:z_idx], [color1, color2][standard], None)
                    self.plot_mean_sem(
                        axs[ci], self.alignment['neu_time'][z_idx:],
                        neu_mean[z_idx:], neu_sem[z_idx:], [color1, color2][1-standard], None)
                    # adjust layouts.
                    adjust_layout_neu(axs[ci])
                    axs[ci].set_xlim(xlim)
                    axs[ci].set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
                    axs[ci].set_xlabel('time since block transition (ms)')
                    axs[ci].set_title(lbl[ci])
        def plot_trial_quant(axs, standard):
            target_metrics = ['evoke_mag', 'onset_ramp', 'onset_drop']
            # collect data.
            [[color0, color1, color2, _],
             [neu_trans_0to1, _, stim_seq_0to1, _], _,
             [n_trials, n_neurons]] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_idx=[l[standard] for l in self.list_block_start],
                cate=cate, roi_id=None)
            [[color0, color1, color2, _],
             [neu_trans_1to0, _, stim_seq_1to0, _], _,
             [n_trials, n_neurons]] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_idx=[l[1-standard] for l in self.list_block_start],
                cate=cate, roi_id=None)
            neu_trans = [neu_trans_0to1, neu_trans_1to0][1-standard]
            neu_late = [neu_trans_0to1, neu_trans_1to0][standard]
            stim_seq_trans = [stim_seq_0to1, stim_seq_1to0][1-standard]
            stim_seq_late = [stim_seq_0to1, stim_seq_1to0][standard]
            idx_trans = np.arange(-trials_eval, trials_eval)
            idx_late = np.arange(-trials_eval, 0)
            color_trans = [[color1,color2][standard]]*trials_eval + [[color1,color2][1-standard]]*trials_eval
            color_late = [[color1,color2][1-standard]]*trials_eval
            c_idx = stim_seq.shape[0]//2
            # plot results for each class.
            for ci in range(self.n_clusters):
                if np.sum(cluster_id==ci) > 0:
                    quant_all_trans = [
                        run_quantification(neu_trans[cluster_id==ci,:], self.alignment['neu_time'], stim_seq_trans[c_idx+si,0])
                        for si in idx_trans]
                    quant_all_late = [
                        run_quantification(neu_late[cluster_id==ci,:], self.alignment['neu_time'], stim_seq_late[c_idx+si,0])
                        for si in idx_late]
                    # plot each metric.
                    for mi in range(len(target_metrics)):
                        axs[mi][ci].axis('off')
                        ax0 = axs[mi][ci].inset_axes([0, 0.5, 0.6, 0.5], transform=axs[mi][ci].transAxes)
                        ax1 = axs[mi][ci].inset_axes([0.7, 0.5, 0.3, 0.5], transform=axs[mi][ci].transAxes, sharey=ax0)
                        # plot each condition for transition.
                        ax0.axvline(0, color='red', lw=1, linestyle='--')
                        for di in range(2*trials_eval):
                            m, s = get_mean_sem(quant_all_trans[di][target_metrics[mi]].reshape(-1,1))
                            ax0.errorbar(
                                idx_trans[di], m, s, None,
                                color=color_trans[di],
                                capsize=2, marker='o', linestyle='none',
                                markeredgecolor='white', markeredgewidth=0.1)
                        # plot each condition for late epoch.
                        for di in range(trials_eval):
                            m, s = get_mean_sem(quant_all_late[di][target_metrics[mi]].reshape(-1,1))
                            ax1.errorbar(
                                idx_late[di], m, s, None,
                                color=color_late[di],
                                capsize=2, marker='o', linestyle='none',
                                markeredgecolor='white', markeredgewidth=0.1)
                        # adjust layouts.
                        axs[mi][ci].set_title(lbl[ci])
                        ax0.set_xlabel('first epoch')
                        ax0.set_ylabel(target_metrics[mi])
                        ax0.set_xlim([-trials_eval-0.5, trials_eval-1+0.5])
                        ax0.set_xticks(idx_trans)
                        ax0.set_xticklabels(idx_trans, rotation='vertical')
                        ax1.tick_params(axis='x', labelrotation=90)
                        ax1.spines['left'].set_visible(False)
                        ax1.set_xlim([-trials_eval-1+0.5, 0.5])
                        ax1.set_xticks(idx_late)
                        ax1.yaxis.set_visible(False)
                        ax1.set_xticklabels(idx_late, rotation='vertical')
                        ax1.set_xlabel('last epoch')
                        for ax in [ax0,ax1]:
                            ax.tick_params(axis='y', tick1On=False)
                            ax.spines['right'].set_visible(False)
                            ax.spines['top'].set_visible(False) 
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
        try: plot_tansition(axs[3], 0)
        except Exception as e: print(e)
        try: plot_trial_quant(axs[4], 0)
        except Exception as e: print(e)
        try: plot_tansition(axs[5], 1)
        except Exception as e: print(e)
        try: plot_trial_quant(axs[6], 1)
        except Exception as e: print(e)
        try: plot_legend(axs[7])
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
    def __init__(self, neural_trials, labels, significance, label_names, temp_folder):
        super().__init__(neural_trials, labels, significance, temp_folder)
        self.label_names = label_names
    
    def cluster_all(self, axs_all):
        for cate, axs in zip([[-1,1,2],[-1],[1],[2]], axs_all):
            try:
                label_name = self.label_names[str(cate[0])] if len(cate)==1 else 'all'
                print(f'plotting results for {label_name}')

                self.plot_cluster_all(axs, cate=cate)
                axs[0].set_title(f'GLM kernels \n {label_name}')
                axs[1].set_title(f'response to binned interval in random \n {label_name}')
                axs[2].set_title(f'response to binned interval in random \n {label_name}')
                axs[3].set_title(f'response to short standard interval \n {label_name}')
                axs[4].set_title(f'response to long standard interval \n {label_name}')
                axs[5].set_title(f'response to short oddball interval \n {label_name}')
                axs[6].set_title(f'response to long oddball interval \n {label_name}')
                axs[7].set_title(f'fraction of neurons in cluster \n {label_name}')
                axs[8].set_title(f'fraction of cell-types in cluster \n {label_name}')
                axs[9].set_title(f'legend \n {label_name}')

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
                axs[3].set_title(f'response to short standard interval \n {label_name}')
                axs[4].set_title(f'response to long standard interval \n {label_name}')

            except Exception as e: print(e)

    def cluster_adapt_individual(self, axs_all):
        for cate, axs in zip([[-1,1,2],[-1],[1],[2]], axs_all):
            try:
                label_name = self.label_names[str(cate[0])] if len(cate)==1 else 'all'
                print(f'plotting results for {label_name}')

                self.plot_cluster_adapt_individual(axs, cate=cate)

            except Exception as e: print(e)
    
    def latent_all(self, axs_all):
        for cate, axs in zip([[-1,1,2],[-1],[1],[2]], axs_all):
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
