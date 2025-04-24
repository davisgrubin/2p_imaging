#!/usr/bin/env python3

import numpy as np
from sklearn.decomposition import PCA

from modules.Alignment import run_get_stim_response
from modeling.clustering import clustering_neu_response_mode
from modeling.clustering import remap_cluster_id
from modeling.generative import get_glm_cate
from modeling.quantifications import run_quantification
from utils import get_odd_stim_prepost_idx
from utils import exclude_odd_stim
from utils import get_mean_sem
from utils import get_neu_trial
from utils import get_frame_idx_from_time
from utils import get_block_1st_idx
from utils import get_block_transition_idx
from utils import get_isi_bin_neu
from utils import get_expect_interval
from utils import get_temporal_scaling_data
from utils import run_wilcoxon_trial
from utils import get_cmap_color
from utils import adjust_layout_neu
from utils import add_legend
from utils import utils_basic

# fig, ax = plt.subplots(1, 1, figsize=(6, 6))
# fig, ax = plt.subplots(1, 1, figsize=(2, 20))
# fig, axs = plt.subplots(1, 8, figsize=(24, 3))
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
        self.r_frames = int(500*timescale)
        self.list_labels = list_labels
        self.list_neural_trials = list_neural_trials
        self.alignment = run_get_stim_response(
                list_neural_trials, self.l_frames, self.r_frames, expected='none')
        self.list_stim_labels = self.alignment['list_stim_labels']
        self.list_odd_idx = [
            get_odd_stim_prepost_idx(sl) for sl in self.list_stim_labels]
        self.expect = np.nanmin(np.array([get_expect_interval(sl) for sl in self.list_stim_labels]),axis=0)
        self.list_block_start = [get_block_1st_idx(sl, 3) for sl in self.list_stim_labels]
        self.list_significance = list_significance
        self.bin_win = [450,2550]
        self.bin_num = 2
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
            trial_param=[[2,3,4,5], None, None, None, [0], [0]],
            cate=cate, roi_id=None)
        cluster_id = remap_cluster_id(neu_seq, self.alignment['neu_time'], cluster_id)
        return kernel_all, cluster_id, neu_labels

    def plot_cluster_block_adapt_individual(self, axs, cate=None):
        lbl = ['cluster #'+str(ci) for ci in range(self.n_clusters)]
        trials_around = 15
        kernel_all, cluster_id, neu_labels = self.run_clustering(cate)
        # get transition trials indice.
        list_trans_0to1 = [get_block_transition_idx(sl[:,3], trials_around)[0] for sl in self.list_stim_labels]
        list_trans_1to0 = [get_block_transition_idx(sl[:,3], trials_around)[1] for sl in self.list_stim_labels]
        # collect data.
        [_, color1, color2, _], [_, _, stim_seq, _], _, [n_trials, n_neurons] = get_neu_trial(
            self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
            trial_param=[None, None, None, None, [0], [0]],
            cate=cate, roi_id=None)
        # preallocate arrays for 0to1 and 1to0 transitions
        neu_trans_0to1 = []
        for ti in range(list_trans_0to1[0].shape[0]):
            trial_idx = [t01[ti, :] for t01 in list_trans_0to1]
            trial_data = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_param=[None, None, None, None, [0], [0]],
                trial_idx=trial_idx,
                mean_sem=False,
                cate=cate, roi_id=None)[1][0]
            neu_trans_0to1.append(np.concatenate(trial_data, axis=1))
        neu_trans_0to1 = np.stack(neu_trans_0to1, axis=0)
        neu_trans_0to1 = np.nanmean(neu_trans_0to1, axis=0)
        neu_trans_1to0 = []
        for ti in range(list_trans_1to0[0].shape[0]):
            trial_idx = [t10[ti, :] for t10 in list_trans_1to0]
            trial_data = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_param=[None, None, None, None, [0], [0]],
                trial_idx=trial_idx,
                mean_sem=False,
                cate=cate, roi_id=None)[1][0]
            neu_trans_1to0.append(np.concatenate(trial_data, axis=1))
        neu_trans_1to0 = np.stack(neu_trans_1to0, axis=0)
        neu_trans_1to0 = np.nanmean(neu_trans_1to0, axis=0)
        def plot_trial_heatmap(axs, norm_mode):
            gap_margin = 5
            xlim = [-500,2500]
            l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, xlim[0], xlim[1])
            neu_time = self.alignment['neu_time'][l_idx:r_idx]
            # extract transition for each class.
            neu_x = []
            zero_gap = np.full((gap_margin, r_idx - l_idx), np.nan)
            for ci in range(self.n_clusters):
                left = np.nanmean(neu_trans_0to1[:, cluster_id==ci, l_idx:r_idx], axis=1)
                right = np.nanmean(neu_trans_1to0[:, cluster_id==ci, l_idx:r_idx], axis=1)
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
            n_trials = neu_trans_1to0.shape[0]
            total = 2 * n_trials + gap_margin
            for ci in range(self.n_clusters):
                c_idx = stim_seq.shape[0]//2
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
        def plot_trial(axs, standard):
            xlim = [-5000,6000]
            # collect data.
            [[color0, color1, color2, _],
             [neu_trans, _, stim_seq, stim_value], _,
             [n_trials, n_neurons]] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_idx=[l[1-standard] for l in self.list_block_start],
                cate=cate, roi_id=None)
            # plot results for each class.
            for ci in range(self.n_clusters):
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
                # adjust layout.
                adjust_layout_neu(axs[ci])
                axs[ci].set_xlim(xlim)
                axs[ci].set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
                axs[ci].set_xlabel('time since stim after interval block change (ms)')
                axs[ci].set_title(lbl[ci])
        def plot_trial_quant(axs, standard):
            trials_eval = 5
            target_metrics = ['evoke_mag', 'onset_ramp']
            # collect data.
            [[color0, color1, color2, _],
             [neu_trans, _, stim_seq, stim_value], _,
             [n_trials, n_neurons]] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_idx=[l[1-standard] for l in self.list_block_start],
                cate=cate, roi_id=None)
            c_idx = stim_seq.shape[0]//2
            cs = [[color1,color2][standard]]*trials_eval + [[color1,color2][1-standard]]*trials_eval
            # plot results for each class.
            idx_trial = np.arange(-trials_eval, trials_eval)
            for ci in range(self.n_clusters):
                quant_all = [
                    run_quantification(neu_trans[cluster_id==ci,:], self.alignment['neu_time'], stim_seq[c_idx+si,0])
                    for si in idx_trial]
                # plot each metric.
                for mi in range(len(target_metrics)):
                    axs[mi][ci].axis('off')
                    axs[mi][ci] = axs[mi][ci].inset_axes([0, 0, 1, 0.4], transform=axs[mi][ci].transAxes)
                    # plot each condition.
                    axs[mi][ci].axvline(0, color='red', lw=1, linestyle='--')
                    for di in range(2*trials_eval):
                        m, s = get_mean_sem(quant_all[di][target_metrics[mi]].reshape(-1,1))
                        axs[mi][ci].errorbar(
                            idx_trial[di], m, s, None,
                            color=cs[di],
                            capsize=2, marker='o', linestyle='none',
                            markeredgecolor='white', markeredgewidth=0.1)
                    # adjust layout.
                    axs[mi][ci].tick_params(axis='y', tick1On=False)
                    axs[mi][ci].tick_params(axis='x', labelrotation=90)
                    axs[mi][ci].spines['right'].set_visible(False)
                    axs[mi][ci].spines['top'].set_visible(False)
                    axs[mi][ci].set_xlim([-trials_eval-0.5, trials_eval-1+0.5])
                    axs[mi][ci].set_xticks(idx_trial)
                    axs[mi][ci].set_xticklabels(idx_trial, rotation='vertical')
                    axs[mi][ci].set_xlabel('trial since interval block change')
                    axs[mi][ci].set_ylabel(target_metrics[mi])
                    axs[mi][ci].set_title(lbl[ci])
        def plot_legend(ax):
            lbl = ['short', 'long']
            cs = [color1, color2]
            add_legend(ax, cs, lbl, n_trials, n_neurons, self.n_sess, 'upper right')
            ax.axis('off')
        # plot all.
        try: plot_trial_heatmap(axs[0], 'none')
        except: pass
        try: plot_trial_heatmap(axs[1], 'minmax')
        except: pass
        try: plot_trial_heatmap(axs[2], 'share')
        except: pass
        try: plot_trial(axs[3], 0)
        except: pass
        try: plot_trial_quant(axs[4], 0)
        except: pass
        try: plot_trial(axs[5], 1)
        except: pass
        try: plot_trial_quant(axs[6], 1)
        except: pass
        try: plot_legend(axs[7])
        except: pass

    def plot_sorted_heatmaps_standard(self, axs, norm_mode, standard):
        thres = 0.05
        cate = [-1,1]
        win_sort = [-500, 500]
        # collect data.
        neu_x = []
        stim_x = []
        # short standard.
        [_, [neu_seq_short, _, stim_seq_short, _], [neu_labels, neu_sig],
         [n_trials_short, n_neurons]] = get_neu_trial(
            self.alignment, self.list_labels, self.list_significance,
            [exclude_odd_stim(sl) for sl in self.list_stim_labels],
            trial_param=[[2,3,4,5], [0], None, None, [0], [0]],
            cate=cate, roi_id=None)
        # long standard.
        [_, [neu_seq_long, _, stim_seq_long, _], [neu_labels, neu_sig],
         [n_trials_long, n_neurons]] = get_neu_trial(
            self.alignment, self.list_labels, self.list_significance,
            [exclude_odd_stim(sl) for sl in self.list_stim_labels],
            trial_param=[[2,3,4,5], [1], None, None, [0], [0]],
            cate=cate, roi_id=None)
        # get indice for only 2 intervals.
        c_idx = stim_seq_short.shape[0]//2
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
        # organize data.
        if standard == 0:
            neu_x += [neu_seq_short, neu_seq_long]
            stim_x += [stim_seq_short, stim_seq_long]
        if standard == 1:
            neu_x += [neu_seq_long, neu_seq_short]
            stim_x += [stim_seq_long, stim_seq_short]
        # difference between short and long.
        neu_x.append(neu_x[0]-neu_x[1])
        stim_x.append(stim_seq_short)
        # wilcoxon test results.
        [_, [neu_seq_short, stim_seq_short, _, _], _, _] = get_neu_trial(
            self.alignment, self.list_labels, self.list_significance,
            [exclude_odd_stim(sl) for sl in self.list_stim_labels],
            trial_param=[[2,3,4,5], [0], None, None, [0], [0]],
            mean_sem=False,
            cate=cate, roi_id=None)
        [_, [neu_seq_long, stim_seq_long, _, _], _, _] = get_neu_trial(
            self.alignment, self.list_labels, self.list_significance,
            [exclude_odd_stim(sl) for sl in self.list_stim_labels],
            trial_param=[[2,3,4,5], [1], None, None, [0], [0]],
            mean_sem=False,
            cate=cate, roi_id=None)
        neu_seq_short = [ns[:,:,l_short:r_short] for ns in neu_seq_short]
        neu_seq_long = [ns[:,:,l_short:r_short] for ns in neu_seq_long]
        if standard == 0:
            _, p_bin = run_wilcoxon_trial(neu_seq_short, neu_seq_long, thres)
        if standard == 1:
            _, p_bin = run_wilcoxon_trial(neu_seq_long, neu_seq_short, thres)
        p_bin = np.concatenate([
            np.nan*np.zeros((p_bin.shape[0],l_short-l_long)),
            p_bin,
            np.nan*np.zeros((p_bin.shape[0],r_long-r_short))],
            axis=1)
        neu_x.append(p_bin)
        stim_x.append(stim_x[-1])
        # plot heatmaps.
        for bi in range(len(neu_x)):
            self.plot_heatmap_neuron_cate(
                axs[bi], neu_x[bi], neu_time, neu_x[0], win_sort, neu_labels, neu_sig,
                norm_mode=norm_mode if bi!=3 else 'binary',
                neu_seq_share=neu_x,
                neu_labels_share=[neu_labels]*len(neu_x),
                neu_sig_share=[neu_sig]*len(neu_x))
            axs[bi].set_xlabel('time since stim (ms) \n sorting window [{},{}] ms'.format(
                win_sort[0], win_sort[1]))
        # add stimulus line.
        for bi in [0,1,2,3]:
            c_idx = stim_x[0].shape[0]//2
            xlines = [neu_time[np.searchsorted(neu_time, t)]
                      for t in [stim_x[bi][c_idx+i,0]
                                for i in [0]]]
            for xl in xlines:
                if xl>neu_time[0] and xl<neu_time[-1]:
                    axs[bi].axvline(xl, color='black', lw=1, linestyle='--')

    def plot_temporal_scaling(self, axs, norm_mode):
        cate = [-1,1]
        win_sort = [-500, 500]
        # collect data.
        neu_x = []
        stim_x = []
        # short standard.
        [_, [neu_seq_short, _, stim_seq_short, _], [neu_labels, neu_sig],
         [n_trials_short, n_neurons]] = get_neu_trial(
            self.alignment, self.list_labels, self.list_significance,
            [exclude_odd_stim(sl) for sl in self.list_stim_labels],
            trial_param=[[2,3,4,5], [0], None, None, [0], [0]],
            cate=cate, roi_id=None)
        # long standard.
        [_, [neu_seq_long, _, stim_seq_long, _], [neu_labels, neu_sig],
         [n_trials_long, n_neurons]] = get_neu_trial(
            self.alignment, self.list_labels, self.list_significance,
            [exclude_odd_stim(sl) for sl in self.list_stim_labels],
            trial_param=[[2,3,4,5], [1], None, None, [0], [0]],
            cate=cate, roi_id=None)
        # get indice for only 2 intervals.
        c_idx = stim_seq_short.shape[0]//2
        t_onset, t_offset = get_frame_idx_from_time(
            self.alignment['neu_time'], 0, stim_seq_short[c_idx,0], stim_seq_short[c_idx,1])
        l_short, r_short = get_frame_idx_from_time(
            self.alignment['neu_time'], 0, stim_seq_short[c_idx-1,1], stim_seq_short[c_idx+1,0])
        l_long, r_long = get_frame_idx_from_time(
            self.alignment['neu_time'], 0, stim_seq_long[c_idx-1,1], stim_seq_long[c_idx+1,0])
        # scale data.
        neu_seq_short_scale = np.concatenate([
            get_temporal_scaling_data(
                neu_seq_short[:,l_short:t_onset],
                self.alignment['neu_time'][l_short:t_onset],
                self.alignment['neu_time'][l_long:t_onset]),
            neu_seq_short[:,t_onset:t_offset],
            get_temporal_scaling_data(
                neu_seq_short[:,t_offset:r_short],
                self.alignment['neu_time'][t_offset:r_short],
                self.alignment['neu_time'][t_offset:r_long])],
            axis=1)
        neu_seq_short = np.concatenate([
            np.nan*np.zeros((neu_seq_short.shape[0],l_short-l_long)),
            neu_seq_short[:,l_short:r_short],
            np.nan*np.zeros((neu_seq_short.shape[0],r_long-r_short))],
            axis=1)
        neu_seq_long = neu_seq_long[:,l_long:r_long]
        neu_time = self.alignment['neu_time'][l_long:r_long]
        # organize data.
        neu_x += [neu_seq_short, neu_seq_short_scale, neu_seq_long]
        stim_x += [stim_seq_short, stim_seq_short, stim_seq_long]
        # plot heatmaps.
        for bi in range(len(neu_x)):
            self.plot_heatmap_neuron_cate(
                axs[bi], neu_x[bi], neu_time, neu_x[0], win_sort, neu_labels, neu_sig,
                norm_mode=norm_mode if bi!=3 else 'binary',
                neu_seq_share=neu_x,
                neu_labels_share=[neu_labels]*len(neu_x),
                neu_sig_share=[neu_sig]*len(neu_x))
            axs[bi].set_xlabel('time since stim (ms) \n sorting window [{},{}] ms'.format(
                win_sort[0], win_sort[1]))
        # add stimulus line.
        for bi in [0,1,2]:
            c_idx = stim_x[0].shape[0]//2
            xlines = [neu_time[np.searchsorted(neu_time, t)]
                      for t in [stim_x[bi][c_idx+i,0]
                                for i in [0]]]
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

# colors = ['#989A9C', '#A4CB9E', '#9DB4CE', '#EDA1A4', '#F9C08A']
class plotter_main(plotter_utils):
    def __init__(self, neural_trials, labels, significance, label_names):
        super().__init__(neural_trials, labels, significance)
        self.label_names = label_names

    def cluster_block_adapt_individual(self, axs_all):
        for cate, axs in zip([[-1,1,2],[-1],[1],[2]], axs_all):
            label_name = self.label_names[str(cate[0])] if len(cate)==1 else 'all'
            print(f'plotting results for {label_name}')
            try:

                self.plot_cluster_block_adapt_individual(axs, cate=cate)

            except: pass


    def sorted_heatmaps_standard(self, axs_all):
        for norm_mode, axs in zip(['none', 'minmax', 'share'], axs_all):
            try:

                self.plot_sorted_heatmaps_standard(axs[0], norm_mode, 0)
                axs[0][0].set_title(f'response to short standard interval \n (normalized with {norm_mode})')
                axs[0][1].set_title(f'response to long standard interval \n (normalized with {norm_mode})')
                axs[0][2].set_title(f'response to standard interval \n (short minors long, normalized with {norm_mode})')
                axs[0][3].set_title('significance test across trials')

                self.plot_sorted_heatmaps_standard(axs[1], norm_mode, 1)
                axs[1][0].set_title(f'response to long standard interval \n (normalized with {norm_mode})')
                axs[1][1].set_title(f'response to short standard interval \n (normalized with {norm_mode})')
                axs[1][2].set_title(f'response to standard interval \n (long minors short, normalized with {norm_mode})')
                axs[1][3].set_title('significance test across trials')

            except: pass

    def temporal_scaling(self, axs_all):
        for norm_mode, axs in zip(['none', 'minmax', 'share'], axs_all):
            try:

                self.plot_temporal_scaling(axs, norm_mode)
                axs[0].set_title(f'response to short standard interval \n (normalized with {norm_mode})')
                axs[1].set_title(f'response to short standard interval \n (scaled, normalized with {norm_mode})')
                axs[2].set_title(f'response to long standard interval \n (normalized with {norm_mode})')

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
