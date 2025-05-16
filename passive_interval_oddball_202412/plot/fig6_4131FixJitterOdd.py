#!/usr/bin/env python3

import numpy as np
from sklearn.decomposition import PCA

from modules.Alignment import run_get_stim_response
from modeling.clustering import clustering_neu_response_mode
from modeling.clustering import remap_cluster_id
from modeling.clustering import get_mean_sem_cluster
from modeling.clustering import get_bin_mean_sem_cluster
from modeling.decoding import multi_sess_decoding_slide_win
from modeling.generative import get_glm_cate
from modeling.quantifications import run_quantification
from utils import get_norm01_params
from utils import get_odd_stim_prepost_idx
from utils import get_mean_sem
from utils import get_neu_trial
from utils import get_frame_idx_from_time
from utils import get_isi_bin_neu
from utils import get_expect_interval
from utils import get_block_1st_idx
from utils import run_wilcoxon_trial
from utils import get_cmap_color
from utils import adjust_layout_neu
from utils import adjust_layout_3d_latent
from utils import add_legend
from utils import utils_basic

# fig, ax = plt.subplots(1, 1, figsize=(5, 5))
# fig, ax = plt.subplots(1, 1, figsize=(3, 12))
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
        self.expect = np.array(np.mean([get_expect_interval(sl)[0] for sl in self.list_stim_labels]))
        self.list_block_start = [get_block_1st_idx(sl, 3) for sl in self.list_stim_labels]
        self.list_significance = list_significance
        self.bin_win = [450,2550]
        self.bin_num = 2
        self.n_clusters = 12
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
    
    def plot_cluster_oddball_fix_all(self, axs, cate):
        xlim = [-2500, 4000]
        kernel_all, cluster_id, neu_labels = self.run_clustering(cate)
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
            glm_mean, glm_sem = get_mean_sem_cluster(kernel_all, cluster_id)
            # plot results.
            norm_params = [get_norm01_params(glm_mean[i,:]) for i in range(self.n_clusters)]
            self.plot_cluster_mean_sem(
                ax, glm_mean, glm_sem, self.glm['kernel_time'],
                norm_params, stim_seq[c_idx,:].reshape(1,-1),
                [color0], [color0]*self.n_clusters,
                [np.nanmin(self.glm['kernel_time']), np.nanmax(self.glm['kernel_time'])])
            # adjust layout.
            ax.set_xlabel('time since stim (ms)')
        def plot_standard_fix(ax):
            # collect data.
            [[color0, color1, color2, _],
             [neu_seq, _, stim_seq, _],
             [neu_labels, neu_sig], _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_param=[[2,3,4,5], None, [0], None, [0], [0]],
                cate=cate, roi_id=None)
            # get response within cluster at each bin.
            neu_mean, neu_sem = get_mean_sem_cluster(neu_seq, cluster_id)
            norm_params = [get_norm01_params(neu_mean[i,:]) for i in range(self.n_clusters)]
            # plot results.
            self.plot_cluster_mean_sem(
                ax, neu_mean, neu_sem,
                self.alignment['neu_time'], norm_params,
                stim_seq,
                [color0]*stim_seq.shape[0], [color0]*self.n_clusters, xlim)
            # adjust layout.
            ax.set_xlabel('time since stim (ms)')
        def plot_oddball_fix(ax, oddball):
            # collect data.
            [[color0, color1, color2, _],
             [neu_seq, _, stim_seq, _],
             [neu_labels, neu_sig], _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_idx=[l[oddball] for l in self.list_odd_idx],
                trial_param=[None, None, [0], None, [0], [0]],
                cate=cate, roi_id=None)
            # get response within cluster at each bin.
            neu_mean, neu_sem = get_mean_sem_cluster(neu_seq, cluster_id)
            norm_params = [get_norm01_params(neu_mean[i,:]) for i in range(self.n_clusters)]
            # plot results.
            c_idx = stim_seq.shape[0]//2
            if oddball == 0:
                ax.axvline(stim_seq[c_idx+1,0], color='gold', lw=1, linestyle='--')
            if oddball == 1:
                ax.axvline(stim_seq[c_idx,1]+self.expect, color='gold', lw=1, linestyle='--')
            self.plot_cluster_mean_sem(
                ax, neu_mean, neu_sem,
                self.alignment['neu_time'], norm_params,
                stim_seq,
                [color0]*stim_seq.shape[0], [[color1,color2][oddball]]*self.n_clusters, xlim)
            # adjust layout.
            ax.set_xlabel('time since pre oddball stim (ms)')
        def plot_neu_fraction(ax):
            # collect data.
            [[_, _, color2, _], _, _, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_param=[None, None, [0], None, [0], [0]],
                cate=cate, roi_id=None)
            # plot results.
            self.plot_cluster_neu_fraction_in_cluster(ax, cluster_id, color2)
        def plot_cate_fraction(ax):
            self.plot_cluster_cate_fraction_in_cluster(ax, cluster_id, neu_labels, self.label_names)
        def plot_legend(ax):
            [[color0, color1, color2, _], _, _,
             [n_trials, n_neurons]] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_param=[[-1], None, [0], None, [0], [0]],
                cate=cate, roi_id=None)
            cs = [color0, color1, color2, 'gold']
            lbl = ['standard', 'short oddball', 'long oddball', 'unexpected event']
            add_legend(ax, cs, lbl, n_trials, n_neurons, self.n_sess, 'upper right')
            ax.axis('off')
        # plot all.
        try: plot_glm_kernel(axs[0])
        except: pass
        try: plot_standard_fix(axs[1])
        except: pass
        try: plot_oddball_fix(axs[2], 0)
        except: pass
        try: plot_oddball_fix(axs[3], 1)
        except: pass
        try: plot_neu_fraction(axs[4])
        except: pass
        try: plot_cate_fraction(axs[5])
        except: pass
        try: plot_legend(axs[6])
        except: pass
    
    def plot_sorted_heatmaps_fix_all(self, axs, norm_mode, cate):
        win_sort = [-500, 500]
        xlim = [-2500, 5000]
        l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, xlim[0], xlim[1])
        # collect data.
        neu_x = []
        stim_x = []
        neu_time = self.alignment['neu_time'][l_idx:r_idx]
        # standard.
        [_, _, _, cmap], [neu_seq, _, stim_seq, _], _, _ = get_neu_trial(
            self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
            trial_param=[[2,3,4,5], None, [0], None, [0], [0]],
            cate=cate, roi_id=None)
        neu_seq = neu_seq[:,l_idx:r_idx]
        neu_x.append(neu_seq)
        stim_x.append(stim_seq)
        # short oddball.
        _, [neu_seq, _, stim_seq, _], _, _ = get_neu_trial(
            self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
            trial_idx=[l[0] for l in self.list_odd_idx],
            trial_param=[None, None, [0], None, [0], [0]],
            cate=cate, roi_id=None)
        neu_seq = neu_seq[:,l_idx:r_idx]
        neu_x.append(neu_seq)
        stim_x.append(stim_seq)
        # long oddball.
        _, [neu_seq, _, stim_seq, _], _, _ = get_neu_trial(
            self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
            trial_idx=[l[1] for l in self.list_odd_idx],
            trial_param=[None, None, [0], None, [0], [0]],
            cate=cate, roi_id=None)
        neu_seq = neu_seq[:,l_idx:r_idx]
        neu_x.append(neu_seq)
        stim_x.append(stim_seq)
        # plot heatmaps.
        for bi in range(3):
            self.plot_heatmap_neuron(
                axs[bi], neu_x[bi], neu_time, neu_x[0], cmap, win_sort,
                norm_mode=norm_mode,
                neu_seq_share=neu_x,
                cbar_label='dF/F')
            axs[bi].set_xlabel('time since pre oddball stim (ms) \n sorting window [{},{}] ms'.format(
                win_sort[0], win_sort[1]))
        # add stimulus line.
        c_idx = stim_x[0].shape[0]//2
        for bi in range(3):
            xlines = [neu_time[np.searchsorted(neu_time, t)]
                      for t in [stim_x[bi][c_idx+i,0]
                                for i in [-2,-1,0,1,2]]]
            for xl in xlines:
                if xl>neu_time[0] and xl<neu_time[-1]:
                    axs[bi].axvline(xl, color='black', lw=1, linestyle='--')
        # adjust layout.
        axs[0].set_xlabel('time since stim (ms) \n sorting window [{},{}] ms'.format(
            win_sort[0], win_sort[1]))
        axs[1].set_xlabel('time since pre oddball stim (ms) \n sorting window [{},{}] ms'.format(
            win_sort[0], win_sort[1]))
        axs[2].set_xlabel('time since pre oddball stim (ms) \n sorting window [{},{}] ms'.format(
            win_sort[0], win_sort[1]))
        axs[1].axvline(stim_x[1][c_idx+1,0], color='gold', lw=1, linestyle='--')
        axs[2].axvline(stim_x[2][c_idx,1]+self.expect, color='gold', lw=1, linestyle='--')

    def plot_cluster_oddball_fix_individual(self, axs, cate):
        lbl = ['cluster #'+str(ci) for ci in range(self.n_clusters)]
        xlim = [-2500, 4000]
        kernel_all, cluster_id, neu_labels = self.run_clustering(cate)
        def plot_standard_fix(axs):
            # collect data.
            [[color0, color1, color2, _],
             [neu_seq, _, stim_seq, _],
             [neu_labels, neu_sig], _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_param=[[2,3,4,5], None, [0], None, [0], [0]],
                cate=cate, roi_id=None)
            # get response within cluster at each bin.
            neu_mean, neu_sem = get_mean_sem_cluster(neu_seq, cluster_id)
            # plot results for each class.
            for ci in range(self.n_clusters):
                # find bounds.
                upper = np.nanmax(neu_mean[ci,:]) + np.nanmax(neu_sem[ci,:])
                lower = np.nanmin(neu_mean[ci,:]) - np.nanmax(neu_sem[ci,:])
                # plot stimulus.
                for si in range(stim_seq.shape[0]):
                    axs[ci].fill_between(
                        stim_seq[si,:],
                        lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
                        color=color0, edgecolor='none', alpha=0.25, step='mid')
                # plot neural traces.
                self.plot_mean_sem(
                    axs[ci], self.alignment['neu_time'],
                    neu_mean[ci,:], neu_sem[ci,:], color0, None)
                # adjust layout.
                adjust_layout_neu(axs[ci])
                axs[ci].set_xlim(xlim)
                axs[ci].set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
                axs[ci].set_xlabel('time since stim (ms)')
                axs[ci].set_title(lbl[ci])
        def plot_oddball_fix(axs, oddball):
            # collect data.
            [[color0, color1, color2, _],
             [neu_seq, _, stim_seq, _],
             [neu_labels, neu_sig], _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_idx=[l[oddball] for l in self.list_odd_idx],
                trial_param=[None, None, [0], None, [0], [0]],
                cate=cate, roi_id=None)
            # get response within cluster.
            neu_mean, neu_sem = get_mean_sem_cluster(neu_seq, cluster_id)
            # plot results for each class.
            for ci in range(self.n_clusters):
                # find bounds.
                upper = np.nanmax(neu_mean[ci,:]) + np.nanmax(neu_sem[ci,:])
                lower = np.nanmin(neu_mean[ci,:]) - np.nanmax(neu_sem[ci,:])
                # plot stimulus.
                c_idx = stim_seq.shape[0]//2
                for si in range(stim_seq.shape[0]):
                    axs[ci].fill_between(
                        stim_seq[si,:],
                        lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
                        color=color0, edgecolor='none', alpha=0.25, step='mid')
                if oddball == 0:
                    axs[ci].axvline(stim_seq[c_idx+1,0], color='gold', lw=1, linestyle='--')
                if oddball == 1:
                    axs[ci].axvline(stim_seq[c_idx,1]+self.expect, color='gold', lw=1, linestyle='--')
                # plot neural traces.
                self.plot_mean_sem(
                    axs[ci], self.alignment['neu_time'],
                    neu_mean[ci,:], neu_sem[ci,:], [color1,color2][oddball], None)
                # adjust layout.
                adjust_layout_neu(axs[ci])
                axs[ci].set_xlim(xlim)
                axs[ci].set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
                axs[ci].set_xlabel('time since pre oddball stim (ms)')
                axs[ci].set_title(lbl[ci])
        def plot_oddball_fix_quant(axs):
            target_metrics = ['onset_ramp', 'onset_drop', 'evoke_mag']
            # standard.
            [[color0, color1, color2, _],
             [neu_seq_standard, _, stim_seq_standard, _], _, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_param=[[2,3,4,5], None, [0], None, [0], [0]],
                cate=cate, roi_id=None)
            # short oddball.
            [[color0, color1, color2, _],
             [neu_seq_short, _, stim_seq_short, _], _, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_idx=[l[0] for l in self.list_odd_idx],
                trial_param=[None, None, [0], None, [0], [0]],
                cate=cate, roi_id=None)
            # long oddball.
            [[color0, color1, color2, _],
             [neu_seq_long, _, stim_seq_long, _], _, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_idx=[l[1] for l in self.list_odd_idx],
                trial_param=[None, None, [0], None, [0], [0]],
                cate=cate, roi_id=None)
            # plot results for each class.
            c_idx = stim_seq_standard.shape[0]//2
            for ci in range(self.n_clusters):
                quant_all = [
                    run_quantification(ns[cluster_id==ci,:], self.alignment['neu_time'], ss[c_idx+1,0])
                    for ns, ss in zip(
                        [neu_seq_standard, neu_seq_short, neu_seq_long],
                        [stim_seq_standard, stim_seq_short, stim_seq_long])]
                # plot each metric.
                for mi in range(len(target_metrics)):
                    axs[mi][ci].axis('off')
                    axs[mi][ci] = axs[mi][ci].inset_axes([0, 0, 0.4, 1], transform=axs[mi][ci].transAxes)
                    # plot each condition.
                    for di in range(3):
                        m, s = get_mean_sem(quant_all[di][target_metrics[mi]].reshape(-1,1))
                        axs[mi][ci].errorbar(
                            di, m, s, None,
                            color=[color0, color1, color2][di],
                            capsize=2, marker='o', linestyle='none',
                            markeredgecolor='white', markeredgewidth=0.1)
                    # adjust layout.
                    axs[mi][ci].tick_params(axis='y', tick1On=False)
                    axs[mi][ci].tick_params(axis='x', labelrotation=90)
                    axs[mi][ci].spines['right'].set_visible(False)
                    axs[mi][ci].spines['top'].set_visible(False)
                    axs[mi][ci].set_xlim([-0.5, 2.5])
                    axs[mi][ci].set_xticks([0,1,2])
                    axs[mi][ci].set_xticklabels(['standard', 'short oddball', 'long oddball'], rotation='vertical')
                    axs[mi][ci].set_xlabel('interval condition')
                    axs[mi][ci].set_ylabel(target_metrics[mi])
                    axs[mi][ci].set_title(lbl[ci])
        def plot_legend(ax):
            [[color0, color1, color2, _], _, _,
             [n_trials, n_neurons]] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_param=[[-1], None, [0], None, [0], [0]],
                cate=cate, roi_id=None)
            cs = [color0, color1, color2]
            lbl = ['standard', 'short oddball', 'long oddball']
            add_legend(ax, cs, lbl, n_trials, n_neurons, self.n_sess, 'upper right')
            ax.axis('off')
        # plot all.
        try: plot_standard_fix(axs[0])
        except: pass
        try: plot_oddball_fix(axs[1], 0)
        except: pass
        try: plot_oddball_fix(axs[2], 1)
        except: pass
        try: plot_oddball_fix_quant(axs[3])
        except: pass
        try: plot_legend(axs[4])
        except: pass
    
    def plot_cluster_oddball_jitter_global_individual(self, axs, cate):
        lbl = ['cluster #'+str(ci) for ci in range(self.n_clusters)]
        xlim = [-2500, 4000]
        kernel_all, cluster_id, neu_labels = self.run_clustering(cate)
        def plot_oddball_jitter(axs, oddball):
            # fix oddball.
            [[color0, color1, color2, _],
             [neu_seq_fix, _, stim_seq_fix, _], _, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_idx=[l[oddball] for l in self.list_odd_idx],
                trial_param=[None, None, [0], None, [0], [0]],
                cate=cate, roi_id=None)
            # jitter oddball.
            [[color0, color1, color2, _],
             [neu_seq_jitter, _, stim_seq_jitter, _], _, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_idx=[l[oddball] for l in self.list_odd_idx],
                trial_param=[None, None, [1], None, [0], [0]],
                cate=cate, roi_id=None)
            # plot results for each class.
            for ci in range(len(lbl)):
                neu_mean = []
                neu_sem = []
                # get class data.
                mean_fix, sem_fix = get_mean_sem(neu_seq_fix[cluster_id==ci,:])
                neu_mean.append(mean_fix)
                neu_sem.append(sem_fix)
                mean_jitter, sem_jitter = get_mean_sem(neu_seq_jitter[cluster_id==ci,:])
                neu_mean.append(mean_jitter)
                neu_sem.append(sem_jitter)
                # find bounds.
                upper = np.nanmax(neu_mean) + np.nanmax(neu_sem)
                lower = np.nanmin(neu_mean) - np.nanmax(neu_sem)
                # plot stimulus.
                c_idx = stim_seq_fix.shape[0]//2
                if oddball == 0:
                    axs[ci].axvline(stim_seq_fix[c_idx+1,0], color='gold', lw=1, linestyle='--')
                if oddball == 1:
                    axs[ci].axvline(stim_seq_fix[c_idx,1]+self.expect, color='gold', lw=1, linestyle='--')
                for i in range(stim_seq_fix.shape[0]):
                    axs[ci].fill_between(
                        stim_seq_fix[i,:],
                        lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
                        color=color0, edgecolor='none', alpha=0.25, step='mid')
                # plot neural traces.
                l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, xlim[0], xlim[1])
                self.plot_mean_sem(
                    axs[ci], self.alignment['neu_time'][l_idx:r_idx],
                    mean_fix[l_idx:r_idx], sem_fix[l_idx:r_idx], color0, None)
                self.plot_mean_sem(
                    axs[ci], self.alignment['neu_time'][l_idx:r_idx],
                    mean_jitter[l_idx:r_idx], sem_jitter[l_idx:r_idx], [color1, color2][oddball], None)
                # adjust layouts.
                adjust_layout_neu(axs[ci])
                axs[ci].set_xlim(xlim)
                axs[ci].set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
                axs[ci].set_xlabel('time since pre oddball stim (ms)')
                axs[ci].set_title(lbl[ci])
        def plot_legend(ax):
            [[color0, color1, color2, _], _, _,
             [n_trials, n_neurons]] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_param=[[-1], None, [1], None, [0], [0]],
                cate=cate, roi_id=None)
            cs = [color0, color1, color2]
            lbl = ['fix', 'jitter short oddball', 'jitter long oddball']
            add_legend(ax, cs, lbl, n_trials, n_neurons, self.n_sess, 'upper right')
            ax.axis('off')
        # plot all.
        try: plot_oddball_jitter(axs[0], 0)
        except: pass
        try: plot_oddball_jitter(axs[1], 1)
        except: pass
        try: plot_legend(axs[2])
        except: pass
    
    def plot_cluster_oddball_jitter_local_individual(self, axs, cate):
        lbl = ['cluster #'+str(ci) for ci in range(self.n_clusters)]
        xlim = [-2500, 4000]
        kernel_all, cluster_id, neu_labels = self.run_clustering(cate)
        def plot_oddball_jitter(axs, oddball):
            # collect data.
            [[color0, color1, color2, _],
             [neu_seq, stim_seq, _, pre_isi, _], _, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_idx=[l[oddball] for l in self.list_odd_idx],
                trial_param=[None, None, [1], None, [0], [0]],
                mean_sem=False,
                cate=cate, roi_id=None)
            # bin data based on isi.
            [bins, bin_center, _, bin_neu_seq, _, _, bin_stim_seq] = get_isi_bin_neu(
                neu_seq, stim_seq, pre_isi, self.bin_win, self.bin_num)
            cluster_bin_neu_mean, cluster_bin_neu_sem = get_bin_mean_sem_cluster(bin_neu_seq, cluster_id)
            # plot results for each class.
            for ci in range(len(lbl)):
                cs = get_cmap_color(self.bin_num, base_color=[color1, color2][oddball])
                # find bounds.
                upper = np.nanmax(cluster_bin_neu_mean[:,ci,:]) + np.nanmax(cluster_bin_neu_sem[:,ci,:])
                lower = np.nanmin(cluster_bin_neu_mean[:,ci,:]) - np.nanmax(cluster_bin_neu_sem[:,ci,:])
                # plot stimulus.
                stim_seq = np.nanmean(bin_stim_seq, axis=0)
                c_idx = stim_seq.shape[0]//2
                if oddball == 0:
                    axs[ci].axvline(stim_seq[c_idx+1,0], color='gold', lw=1, linestyle='--')
                if oddball == 1:
                    axs[ci].axvline(stim_seq[c_idx,1]+self.expect, color='gold', lw=1, linestyle='--')
                for bi in range(self.bin_num):
                    axs[ci].axvline(bin_stim_seq[bi,c_idx-1,0], color=cs[bi], lw=1, linestyle='--')
                for si in [c_idx, c_idx+1]:
                    axs[ci].fill_between(
                        stim_seq[si,:],
                        lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
                        color=color0, edgecolor='none', alpha=0.25, step='mid')
                # plot neural traces.
                for bi in range(self.bin_num):
                    l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, xlim[0], xlim[1])
                    self.plot_mean_sem(
                        axs[ci], self.alignment['neu_time'][l_idx:r_idx],
                        cluster_bin_neu_mean[bi,ci,l_idx:r_idx], cluster_bin_neu_sem[bi,ci,l_idx:r_idx],
                        cs[bi], None)
                # adjust layouts.
                adjust_layout_neu(axs[ci])
                axs[ci].set_xlim(xlim)
                axs[ci].set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
                axs[ci].set_xlabel('time since pre oddball stim (ms)')
                axs[ci].set_title(lbl[ci])
        def plot_legend(ax):
            [[color0, _, color2, _],
             [neu_seq, stim_seq, _, pre_isi, _], _,
             [n_trials, n_neurons]] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_idx=[l[0] for l in self.list_odd_idx],
                trial_param=[None, None, [1], None, [0], [0]],
                mean_sem=False,
                cate=cate, roi_id=None)
            [bins, _, _, _, _, _, _] = get_isi_bin_neu(
                neu_seq, stim_seq, pre_isi, self.bin_win, self.bin_num)
            lbl = ['[{},{}] ms'.format(int(bins[i]),int(bins[i+1])) for i in range(self.bin_num)]
            cs = get_cmap_color(self.bin_num, base_color=color2)
            add_legend(ax, cs, lbl, n_trials, n_neurons, self.n_sess, 'upper right')
            ax.axis('off')
        # plot all.
        try: plot_oddball_jitter(axs[0], 0)
        except: pass
        try: plot_oddball_jitter(axs[1], 1)
        except: pass
        try: plot_legend(axs[2])
        except: pass
    
    def plot_oddball_win_decode_local(self, axs, cate):
        xlim = [-2500, 4000]
        win_sample = 200
        win_step = 1
        bin_num = 2
        def plot_win_decode(ax, oddball):
            # collect data.
            [[color0, color1, color2, _],
             [neu_seq, stim_seq, _, pre_isi, _], _, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_idx=[l[oddball] for l in self.list_odd_idx],
                trial_param=[None, None, [1], None, [0], [0]],
                mean_sem=False,
                cate=cate, roi_id=None)
            # bin data based on isi.
            [_, bin_center, bin_neu_seq_trial, _, _, _, bin_stim_seq] = get_isi_bin_neu(
                neu_seq, stim_seq, pre_isi, self.bin_win, bin_num)
            # prepare data for decoding.
            bss = np.nanmean(bin_stim_seq, axis=0)
            c_idx = bss.shape[0]//2
            win_decode = [bss[c_idx,0], bss[c_idx+1,1]+500]
            l_idx, r_idx = get_frame_idx_from_time(
                self.alignment['neu_time'], 0, win_decode[0], win_decode[1])
            neu_y = [np.concatenate([
                np.ones(bin_neu_seq_trial[bi][si].shape[0])*bi
                for bi in range(bin_num)], axis=0) for si in range(self.n_sess)]
            neu_x = [np.concatenate([
                bin_neu_seq_trial[bi][si]
                for bi in range(bin_num)], axis=0) for si in range(self.n_sess)]
            # run decoding.
            [acc_time,
             acc_model_mean, acc_model_sem,
             acc_chance_mean, acc_chance_sem] = multi_sess_decoding_slide_win(
                 neu_x, neu_y, self.alignment['neu_time'],
                 win_decode, win_sample, win_step)
            # find bounds.
            upper = np.nanmax([acc_model_mean, acc_chance_mean]) + np.nanmax([acc_model_sem, acc_chance_sem])
            lower = np.nanmin([acc_model_mean, acc_chance_mean]) - np.nanmax([acc_model_sem, acc_chance_sem])
            # plot stimulus.
            for si in [0,1]:
                ax.fill_between(
                    bss[c_idx+si,:],
                    lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
                    color=color0, edgecolor='none', alpha=0.25, step='mid')
            # plot results.
            self.plot_mean_sem(
                ax, acc_time, acc_chance_mean, acc_chance_sem, color0, None)
            self.plot_mean_sem(
                ax, acc_time, acc_model_mean, acc_model_sem, [color1, color2][oddball], None)
            # adjust layouts.
            ax.set_xlim(xlim)
            ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
            ax.set_xlabel('time since pre oddball stim (ms)')
            ax.tick_params(axis='y', tick1On=False)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.set_ylabel('test accuracy')
        def plot_legend(ax):
            [[color0, color1, color2, _], _, _,
             [n_trials, n_neurons]] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_param=[[-1], None, [0], None, [0], [0]],
                cate=cate, roi_id=None)
            cs = [color0, color1, color2]
            lbl = ['standard', 'short oddball', 'long oddball']
            add_legend(ax, cs, lbl, n_trials, n_neurons, self.n_sess, 'upper right')
            ax.axis('off')
        # plot all.
        plot_win_decode(axs[0], 0)
        plot_win_decode(axs[1], 1)
        plot_legend(axs[2])
        try: plot_win_decode(axs[0], 0)
        except: pass
        try: plot_win_decode(axs[1], 1)
        except: pass
        try: plot_legend(axs[2])
        except: pass
    
    def plot_oddball_latent_fix_all(self, axs, oddball, cate):
        # collect data.
        xlim = [-2500, 6000]
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
        model = PCA(n_components=self.d_latent)
        neu_z = model.fit_transform(
            neu_x.reshape(neu_x.shape[0],-1).T
            ).T.reshape(self.d_latent, -1)
        def plot_3d_dynamics(ax):
            # plot dynamics.
            self.plot_3d_latent_dynamics(ax, neu_z, stim_seq, neu_time, c_stim)
            # mark unexpected event.
            idx_unexpect = get_frame_idx_from_time(
                neu_time, 0, stim_seq[c_idx+1,0], stim_seq[c_idx,1]+self.expect)[oddball]
            ax.scatter(neu_z[0,idx_unexpect], neu_z[1,idx_unexpect], neu_z[2,idx_unexpect], color='gold', marker='o', lw=5)
            # adjust layout.
            adjust_layout_3d_latent(ax, neu_z, self.latent_cmap, neu_time, 'time since pre oddball stim (ms)')
        def plot_mean_dynamics(ax):
            ax.axis('off')
            ax = ax.inset_axes([0, 0, 1, 0.4], transform=ax.transAxes)
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
                ax.axvline(stim_seq[c_idx,1]+self.expect, color='gold', lw=1, linestyle='--')
            # plot neural traces.
            self.plot_mean_sem(ax, neu_time, neu_mean, neu_sem, color0, None)
            for t in range(neu_mean.shape[0]-1):
                ax.plot(neu_time[t:t+2], neu_mean[t:t+2], color=c_neu[t])
            # adjust layouts.
            adjust_layout_neu(ax)
            ax.set_xlim(xlim)
            ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
            ax.set_xlabel('time since pre oddball stim (ms)')
        def plot_legend(ax):
            [[color0, color1, color2, _], _, _,
             [n_trials, n_neurons]] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_param=[[-1], None, [0], None, [0], [0]],
                cate=cate, roi_id=None)
            cs = [color0, 'gold']
            lbl = ['stimulus', 'unexpected event']
            add_legend(ax, cs, lbl, n_trials, n_neurons, self.n_sess, 'upper right')
            ax.axis('off')
        # plot all.
        try: plot_3d_dynamics(axs[0])
        except: pass
        try: plot_mean_dynamics(axs[1])
        except: pass
        try: plot_legend(axs[2])
        except: pass
        
# colors = ['#989A9C', '#A4CB9E', '#9DB4CE', '#EDA1A4', '#F9C08A']
class plotter_main(plotter_utils):
    def __init__(self, neural_trials, labels, significance, label_names, temp_folder):
        super().__init__(neural_trials, labels, significance, temp_folder)
        self.label_names = label_names

    def cluster_oddball_fix_all(self, axs_all):
        for cate, axs in zip([[-1,1,2],[-1],[1],[2]], axs_all):
            try:
                label_name = self.label_names[str(cate[0])] if len(cate)==1 else 'all'
                print(f'plotting results for {label_name}')
                
                self.plot_cluster_oddball_fix_all(axs, cate=cate)
                axs[0].set_title(f'reponse to standard interval \n {label_name}')
                axs[1].set_title(f'reponse to short oddball interval \n {label_name}')
                axs[2].set_title(f'reponse to long oddball interval \n {label_name}')

            except: pass
        
    def sorted_heatmaps_fix_all(self, axs_all):
        for cate, axs in zip([[-1,1,2],[-1],[1],[2]], axs_all):
            try:
                label_name = self.label_names[str(cate[0])] if len(cate)==1 else 'all'
                print(f'plotting results for {label_name}')
                
                for axss, norm_mode in zip([axs[0:3], axs[3:6], axs[6:9]], ['none', 'minmax', 'share']):
                    self.plot_sorted_heatmaps_fix_all(axss, norm_mode, cate=cate)
                    axss[0].set_title(f'response to standard interval \n {label_name} (normalized with {norm_mode})')
                    axss[1].set_title(f'response to short oddball interval \n {label_name} (normalized with {norm_mode})')
                    axss[2].set_title(f'response to long oddball interval \n {label_name} (normalized with {norm_mode})')

            except: pass
        
    def cluster_oddball_fix_individual(self, axs_all):
        for cate, axs in zip([[-1,1,2],[-1],[1],[2]], axs_all):
            try:
                label_name = self.label_names[str(cate[0])] if len(cate)==1 else 'all'
                print(f'plotting results for {label_name}')
                
                self.plot_cluster_oddball_fix_individual(axs, cate=cate)

            except: pass
    
    def cluster_oddball_jitter_global_individual(self, axs_all):
        for cate, axs in zip([[-1,1,2],[-1],[1],[2]], axs_all):
            try:
                label_name = self.label_names[str(cate[0])] if len(cate)==1 else 'all'
                print(f'plotting results for {label_name}')
                
                self.plot_cluster_oddball_jitter_global_individual(axs, cate=cate)

            except: pass
    
    def cluster_oddball_jitter_local_individual(self, axs_all):
        for cate, axs in zip([[-1,1,2],[-1],[1],[2]], axs_all):
            try:
                label_name = self.label_names[str(cate[0])] if len(cate)==1 else 'all'
                print(f'plotting results for {label_name}')
                
                self.plot_cluster_oddball_jitter_local_individual(axs, cate=cate)

            except: pass
    
    def oddball_win_decode_local(self, axs_all):
        for cate, axs in zip([[-1,1,2],[-1],[1],[2]], axs_all):
            try:
                label_name = self.label_names[str(cate[0])] if len(cate)==1 else 'all'
                print(f'plotting results for {label_name}')
                
                self.plot_oddball_win_decode_local(axs, cate=cate)

            except: pass
    
    def oddball_latent_fix_all(self, axs_all):
        for cate, axs in zip([[-1,1,2],[-1],[1],[2]], axs_all):
            try:
                label_name = self.label_names[str(cate[0])] if len(cate)==1 else 'all'
                print(f'plotting results for {label_name}')
                
                self.plot_oddball_latent_fix_all(axs[0], 0, cate=cate)
                self.plot_oddball_latent_fix_all(axs[1], 1, cate=cate)

            except: pass

