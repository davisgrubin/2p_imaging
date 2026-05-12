#!/usr/bin/env python3

import traceback
import numpy as np
import matplotlib.ticker as mtick
from sklearn.decomposition import PCA

from modules.Alignment import run_get_stim_response
from modeling.clustering import get_mean_sem_cluster
from modeling.clustering import get_bin_mean_sem_cluster
from modeling.clustering import get_cluster_cate
from modeling.decoding import multi_sess_decoding_slide_win
from modeling.decoding import fit_poly_line
from modeling.decoding import cate_2_multi_sess_regression_pop
from modeling.generative import get_glm_cate
from utils import show_resource_usage
from utils import get_norm01_params
from utils import get_odd_stim_prepost_idx
from utils import get_mean_sem
from utils import get_mean_sem_win
from utils import get_neu_trial
from utils import get_frame_idx_from_time
from utils import get_modulation_index_neu_seq
from utils import get_isi_bin_neu
from utils import get_expect_interval
from utils import get_split_idx
from utils import pair_corr_subsample
from utils import get_cmap_color
from utils import hide_all_axis
from utils import get_random_rotate_mat_3d
from utils import add_ax_ticks
from utils import adjust_layout_isi_example_epoch
from utils import adjust_layout_3d_latent
from utils import adjust_layout_pupil
from utils import add_legend
from utils import add_heatmap_colorbar
from utils import utils_basic

# fig, ax = plt.subplots(1, 1, figsize=(3, 6))
# fig, ax = plt.subplots(1, 1, figsize=(3, 3))
# fig, axs = plt.subplots(1, 3, figsize=(9, 6))
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
        self.list_odd_idx = [
            get_odd_stim_prepost_idx(sl) for sl in self.list_stim_labels]
        self.expect = np.array(np.mean([get_expect_interval(sl)[0] for sl in self.list_stim_labels]))
        self.bin_win = [450,2550]
        self.bin_num = 2
        self.d_latent = 3
        self.glm = self.run_glm()
        self.n_pre = 2
        self.n_post = 3
        self.trf_model = self.run_trf_model()
        self.cluster_id, self.cluster_id_pre_layers, self.cluster_id_post_layers = self.run_clustering(self.n_pre, self.n_post)

    def get_neu_seq_trial_fix_jitter(self, jitter_trial_mode, oddball, cate, isi_win):
        # fix oddball.
        [_, [neu_seq_fix, _, _, _, _], _, _] = get_neu_trial(
            self.alignment, self.list_labels, self.list_stim_labels,
            trial_idx=[l[oddball] for l in self.list_odd_idx],
            trial_param=[None, None, [0], None, [0], [0]],
            mean_sem=False,
            cate=cate, roi_id=None)
        # jitter oddball.
        [_, [neu_seq_jitter, _, _, pre_isi, _], _, _] = get_neu_trial(
            self.alignment, self.list_labels, self.list_stim_labels,
            trial_idx=[l[oddball] for l in self.list_odd_idx],
            trial_param=[None, None, [1], None, [0], [0]],
            mean_sem=False,
            cate=cate, roi_id=None)
        if jitter_trial_mode == 'global':
            idx_jitter = [isi>0 for isi in pre_isi]
        if jitter_trial_mode == 'similar':
            idx_jitter = [(isi>np.nanmean(isi)-isi_win)*(isi<np.nanmean(isi)+isi_win) for isi in pre_isi]
        if jitter_trial_mode == 'lower':
            idx_jitter = [(isi<np.nanmean(isi)-isi_win) for isi in pre_isi]
        if jitter_trial_mode == 'higher':
            idx_jitter = [(isi>np.nanmean(isi)+isi_win) for isi in pre_isi]
        neu_seq_jitter = [neu_seq_jitter[si][idx_jitter[si],:,:] for si in range(self.n_sess)]
        return neu_seq_fix, neu_seq_jitter

    def plot_neuron_fraction(self, ax):
        try:
            colors = ['cornflowerblue', 'violet', 'mediumseagreen']
            _, cluster_id, neu_labels = get_cluster_cate(self.cluster_id, self.list_labels, [-1,1,2])
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
    
    def plot_ramp_type_cell_fraction_table(self, ax):
        try:
            # collect data.
            cate = [-1,1,2]
            neu_labels = np.concatenate(self.list_labels)
            # compute counts.
            rows = [('Ramp-up', (self.cluster_id<self.n_pre)&(self.cluster_id>=0)),
                    ('Ramp-down', self.cluster_id>=self.n_pre),
                    ('Excluded', self.cluster_id==-1)]
            counts = np.array([[np.sum(mask & (neu_labels == c)) for c in cate] for _, mask in rows])
            col_sums = counts.sum(axis=0) + 1e-8
            cell_text = [[f'{counts[i, j]} ({counts[i, j]/col_sums[j]:.2f})'
                          for j in range(len(cate))] for i in range(len(rows))]
            # plot table.
            tab = ax.table(
                cellText=cell_text,
                rowLabels=[r[0] for r in rows],
                colLabels=[f'{self.label_names[str(c)]}' for c in cate],
                loc='center',
                cellLoc='center',
                rowLoc='center')
            # adjust layouts.
            tab.scale(0.8, 2)
            for (i, j), cell in tab.get_celld().items():
                cell.set_linewidth(0)
                if i == 0:
                    cell.visible_edges = 'B'
                    cell.set_linewidth(1)
            ax.axis('off')
        except: traceback.print_exc()
        
    def plot_isi_seting(self, ax):
        # define layouts.
        ax.axis('off')
        ax = ax.inset_axes([0, 0, 0.6, 0.6], transform=ax.transAxes)
        # plot settings.
        gap = 25
        ax.hlines(0.5, 500+gap, 2500-gap, color='royalblue')
        ax.vlines([500+gap, 2500-gap], 0, 0.5, color='royalblue')
        ax.vlines(1500, 0, 1, color='deeppink')
        ax.vlines(500, 0, 0.6, color='black')
        ax.vlines(2500, 0, 0.6, color='black')
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xlim([450,2550])
        ax.set_ylim([0, 1.05])
        ax.set_xticks([500,1500,2500])
        ax.set_yticks([])
        ax.set_xticklabels([500,1500,2500])
        ax.set_xlabel('ISI (s)')
        ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, pos: f'{x/1000:.1f}'))
    
    def plot_isi_example_epoch(self, ax):
        trial_win = [1000,1500]
        # define layouts.
        ax.axis('off')
        ax = ax.inset_axes([0, 0, 1, 0.6], transform=ax.transAxes)
        # get isi and trial labels.
        stim_labels = self.list_neural_trials[0]['stim_labels'][trial_win[0]:trial_win[1],:]
        isi = stim_labels[1:,0] - stim_labels[:-1,1]
        img_seq_label    = stim_labels[:-1,2]
        fix_jitter_types = stim_labels[:-1,4]
        # plot trials.
        color_map = np.array(['deeppink', 'royalblue', 'black'], dtype=object)
        idx = np.where(img_seq_label == -1, 2, fix_jitter_types)
        colors = color_map[idx]
        ax.scatter(np.arange(trial_win[0], trial_win[1]-1), isi, c=colors, s=5)
        # adjust layouts.
        adjust_layout_isi_example_epoch(ax, trial_win, self.bin_win)
        ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, pos: f'{x/1000:.1f}'))
        
    def plot_cluster_oddball_fix_all(self, axs, cate):
        color0 = 'dimgrey'
        color1 = 'hotpink'
        color2 = 'darkviolet'
        xlim = [-2500, 4000]
        _, cluster_id, neu_labels = get_cluster_cate(self.cluster_id, self.list_labels, cate)
        @show_resource_usage
        def plot_glm_kernel(ax):
            kernel_all = get_glm_cate(self.glm, self.list_labels, cate)
            self.plot_glm_kernel(ax, kernel_all, cluster_id, color0, 1)
        @show_resource_usage
        def plot_standard_fix(ax):
            # collect data.
            [_, [neu_seq, _, stim_seq, _],
             neu_labels, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_stim_labels,
                trial_param=[[2,3,4,5], None, [0], None, [0], [0]],
                cate=cate, roi_id=None)
            # get response within cluster.
            neu_mean, neu_sem = get_mean_sem_cluster(neu_seq, self.n_clusters, cluster_id)
            norm_params = [get_norm01_params(neu_mean[i,:]) for i in range(self.n_clusters)]
            # plot results.
            self.plot_cluster_mean_sem(
                ax, neu_mean, neu_sem,
                self.alignment['neu_time'], norm_params,
                stim_seq,
                [color0]*stim_seq.shape[0], [color0]*self.n_clusters, xlim)
            # adjust layouts.
            ax.set_xlabel('Time from stim onset (s)')
        @show_resource_usage
        def plot_oddball_fix(ax, oddball):
            # collect data.
            [_, [neu_seq, _, stim_seq, _], _, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_stim_labels,
                trial_idx=[l[oddball] for l in self.list_odd_idx],
                trial_param=[None, None, [0], None, [0], [0]],
                cate=cate, roi_id=None)
            # get response within cluster at each bin.
            neu_mean, neu_sem = get_mean_sem_cluster(neu_seq, self.n_clusters, cluster_id)
            norm_params = [get_norm01_params(neu_mean[i,:]) for i in range(self.n_clusters)]
            # plot results.
            c_idx = stim_seq.shape[0]//2
            if oddball == 0:
                ax.axvline(stim_seq[c_idx+1,0], color='red', lw=1, linestyle='--')
            if oddball == 1:
                ax.axvline(stim_seq[c_idx,1]+self.expect, color='gold', lw=1, linestyle='--')
                ax.axvline(stim_seq[c_idx+1,0], color='red', lw=1, linestyle='--')
            self.plot_cluster_mean_sem(
                ax, neu_mean, neu_sem,
                self.alignment['neu_time'], norm_params,
                stim_seq,
                [color0]*stim_seq.shape[0], [[color1,color2][oddball]]*self.n_clusters, xlim)
            # adjust layouts.
            ax.set_xlabel('Time from deviant (s)')
        @show_resource_usage
        def plot_oddball_fix_quant(ax):
            win = 300
            # collect data.
            [_, [neu_seq_standard, _, stim_seq_standard, _], _, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_stim_labels,
                trial_param=[[2,3,4,5], None, [0], None, [0], [0]],
                cate=cate, roi_id=None)
            [_, [neu_seq_short, _, stim_seq_short, _], _, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_stim_labels,
                trial_idx=[l[0] for l in self.list_odd_idx],
                trial_param=[None, None, [0], None, [0], [0]],
                cate=cate, roi_id=None)
            [_, [neu_seq_long, _, stim_seq_long, _], _, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_stim_labels,
                trial_idx=[l[1] for l in self.list_odd_idx],
                trial_param=[None, None, [0], None, [0], [0]],
                cate=cate, roi_id=None)
            c_idx = stim_seq_standard.shape[0]//2
            # define layouts.
            ax.axis('off')
            ax = ax.inset_axes([0, 0, 1, 0.95], transform=ax.transAxes)
            axs = [ax.inset_axes([0.3, ci/self.n_clusters, 0.4, 0.7/self.n_clusters], transform=ax.transAxes)
                      for ci in range(self.n_clusters)]
            axs.reverse()
            # plot results for each class.
            for ci in range(self.n_clusters):
                if np.sum(cluster_id==ci) > 0:
                    quant_standard = get_mean_sem_win(
                        neu_seq_standard[cluster_id==ci,:],
                        self.alignment['neu_time'], 0, 0, win, mode='mean')
                    quant_short = get_mean_sem_win(
                        neu_seq_short[cluster_id==ci,:],
                        self.alignment['neu_time'], 0, stim_seq_short[c_idx+1,0], stim_seq_short[c_idx+1,0]+win, mode='mean')
                    quant_long = get_mean_sem_win(
                        neu_seq_long[cluster_id==ci,:],
                        self.alignment['neu_time'], 0, stim_seq_long[c_idx+1,0], stim_seq_long[c_idx+1,0]+win, mode='mean')
                    axs[ci].errorbar(
                        1, quant_standard[1], quant_standard[2], None,
                        color=color0,
                        capsize=2, marker='o', linestyle='none',
                        markeredgecolor='white', markeredgewidth=0.1)
                    axs[ci].errorbar(
                        2, quant_short[1], quant_short[2], None,
                        color=color1,
                        capsize=2, marker='o', linestyle='none',
                        markeredgecolor='white', markeredgewidth=0.1)
                    axs[ci].errorbar(
                        3, quant_long[1], quant_long[2], None,
                        color=color2,
                        capsize=2, marker='o', linestyle='none',
                        markeredgecolor='white', markeredgewidth=0.1)
                    # adjust layouts.
                    axs[ci].spines['right'].set_visible(False)
                    axs[ci].spines['top'].set_visible(False)
                    axs[ci].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
                    axs[ci].yaxis.set_major_locator(mtick.MaxNLocator(nbins=2))
                    axs[ci].set_xlim([0.5, 3.5])
                    axs[ci].set_xticks([])
            hide_all_axis(ax)
            ax.set_ylabel(f'response magnitude in {win} ms window since stimulus onset after interval \n')
            axs[self.n_clusters-1].tick_params(axis='x', labelrotation=90)
            axs[self.n_clusters-1].set_xticks([1,2,3])
            axs[self.n_clusters-1].set_xticklabels(['standard', 'short oddball', 'long oddball'])
        @show_resource_usage
        def plot_neu_fraction(ax):
            # define layouts.
            ax.axis('off')
            ax = ax.inset_axes([0, 0, 0.70, 0.95], transform=ax.transAxes)
            # plot results.
            self.plot_cluster_neu_fraction_in_cluster(ax, cluster_id, color0)
        @show_resource_usage
        def plot_cate_fraction(ax):
            # define layouts.
            ax.axis('off')
            ax = ax.inset_axes([0, 0, 0.70, 0.95], transform=ax.transAxes)
            # plot results.
            self.plot_cluster_cate_fraction_in_cluster(ax, cluster_id, neu_labels, self.label_names, color0)
        @show_resource_usage
        def plot_legend(ax):
            [[color0, color1, color2, _], _, _,
             [n_trials, n_neurons]] = get_neu_trial(
                self.alignment, self.list_labels, self.list_stim_labels,
                trial_param=[[-1], None, [0], None, [0], [0]],
                cate=cate, roi_id=None)
            cs = [color0, color1, color2, 'gold', 'red', 'red']
            lbl = ['Standard', 'Short deviant', 'Long deviant', 'Omission', 'Early stim', 'Late stim']
            add_legend(ax, cs, lbl, n_trials, n_neurons, self.n_sess, 'upper right')
            ax.axis('off')
        # plot all.
        try: plot_glm_kernel(axs[0])
        except: traceback.print_exc()
        try: plot_standard_fix(axs[1])
        except: traceback.print_exc()
        try: plot_oddball_fix(axs[2], 0)
        except: traceback.print_exc()
        try: plot_oddball_fix(axs[3], 1)
        except: traceback.print_exc()
        try: plot_oddball_fix_quant(axs[4])
        except: traceback.print_exc()
        try: plot_neu_fraction(axs[5])
        except: traceback.print_exc()
        try: plot_cate_fraction(axs[6])
        except: traceback.print_exc()
        try: plot_legend(axs[7])
        except: traceback.print_exc()
    
    def plot_cluster_oddball_jitter_global_all(self, axs, jitter_trial_mode, cate):
        color0 = 'dimgrey'
        color1 = 'deeppink'
        color2 = 'royalblue'
        color_model = 'plum'
        win_lbl = ['early', 'late', 'post']
        isi_win = 250
        xlim = [-2500, 4000]
        l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, xlim[0], xlim[1])
        _, cluster_id, neu_labels = get_cluster_cate(self.cluster_id, self.list_labels, cate)
        cluster_id_pre  = np.isin(cluster_id, np.arange(0, self.n_pre))
        cluster_id_post = np.isin(cluster_id, np.arange(self.n_pre, self.n_post+self.n_pre))
        split_idx = get_split_idx(self.list_labels, cate)
        day_cluster_id = np.split(cluster_id, split_idx)
        @show_resource_usage
        def plot_oddball_jitter(ax, oddball):
            # collect data.
            [_, [_, _, stim_seq, _], _, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_stim_labels,
                trial_idx=[l[oddball] for l in self.list_odd_idx],
                trial_param=[None, None, [0], None, [0], [0]],
                cate=cate, roi_id=None)
            c_idx = stim_seq.shape[0]//2
            neu_seq_fix, neu_seq_jitter = self.get_neu_seq_trial_fix_jitter(jitter_trial_mode, oddball, cate, isi_win)
            # get response within cluster.
            neu_seq_jitter = np.concatenate([np.nanmean(neu, axis=0) for neu in neu_seq_jitter], axis=0)
            neu_seq_fix = np.concatenate([np.nanmean(neu, axis=0) for neu in neu_seq_fix], axis=0)
            neu_fix_mean, neu_fix_sem = get_mean_sem_cluster(neu_seq_fix, self.n_clusters, cluster_id)
            neu_jitter_mean, neu_jitter_sem = get_mean_sem_cluster(neu_seq_jitter, self.n_clusters, cluster_id)
            norm_params = [get_norm01_params(
                np.concatenate([neu_fix_mean[ci,:], neu_jitter_mean[ci,:]]))
                for ci in range(self.n_clusters)]
            # define layouts.
            ax.axis('off')
            ax = ax.inset_axes([0, 0, 1, 0.95], transform=ax.transAxes)
            # plot results.
            if oddball == 0:
                ax.axvline(stim_seq[c_idx+1,0], color='red', lw=1, linestyle='--')
            if oddball == 1:
                ax.axvline(stim_seq[c_idx,1]+self.expect, color='gold', lw=1, linestyle='--')
                ax.axvline(stim_seq[c_idx+1,0], color='red', lw=1, linestyle='--')
            c_stim = [color0 if i in (c_idx, c_idx + 1) else f'empty{color0}' for i in range(stim_seq.shape[0])]
            self.plot_cluster_mean_sem(
                ax, neu_fix_mean, neu_fix_sem,
                self.alignment['neu_time'], norm_params,
                stim_seq,
                c_stim, [color1]*self.n_clusters, xlim)
            self.plot_cluster_mean_sem(
                ax, neu_jitter_mean, neu_jitter_sem,
                self.alignment['neu_time'], norm_params,
                None, None, [color2]*self.n_clusters, xlim,
                scale_bar=False)
            # adjust layouts.
            ax.set_xlabel('Time from deviant (s)')
        @show_resource_usage
        def plot_win_mag_scatter(ax, oddball, wi):
            average_axis = 1
            # collect data.
            neu_seq_fix, neu_seq_jitter = self.get_neu_seq_trial_fix_jitter(jitter_trial_mode, oddball, cate, isi_win)
            [_, [_, _, stim_seq, _], _, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_stim_labels,
                trial_idx=[l[oddball] for l in self.list_odd_idx],
                trial_param=[None, None, [0], None, [0], [0]],
                cate=cate, roi_id=None)
            c_idx = stim_seq.shape[0]//2
            # define evaluation windows.
            if oddball == 0:
                win_eval = [[-2500, 0],
                            [stim_seq[c_idx+1,0],stim_seq[c_idx+1,0]+250],
                            [stim_seq[c_idx+1,0]+300,stim_seq[c_idx+1,0]+550],
                            [stim_seq[c_idx+1,0]+600,stim_seq[c_idx+1,0]+850]]
            if oddball == 1:
                win_eval = [[-2500, 0],
                            [stim_seq[c_idx,1]+self.expect, stim_seq[c_idx,1]+self.expect+250],
                            [stim_seq[c_idx+1,0]-250,stim_seq[c_idx+1,0]],
                            [stim_seq[c_idx+1,1],stim_seq[c_idx+1,1]+250]]
            win_eval = [win_eval[0], win_eval[wi]]
            # define layouts.
            ax.axis('off')
            ax = ax.inset_axes([0, 0, 1, 0.95], transform=ax.transAxes)
            axs = [ax.inset_axes([0.3, ci/self.n_clusters, 0.5, 0.7/self.n_clusters], transform=ax.transAxes)
                      for ci in range(self.n_clusters)]
            axs.reverse()
            # plot results.
            self.plot_cluster_win_mag_scatter(
                axs, day_cluster_id, neu_seq_fix, neu_seq_jitter, self.alignment['neu_time'],
                win_eval, win_eval, color0, 0, average_axis)
            # adjust layouts.
            for ci in range(self.n_clusters):
                axs[ci].set_ylabel('jitter')
            axs[self.n_clusters-1].set_xlabel('fix')
            hide_all_axis(ax)
            ax.set_ylabel(f'Response comparison in the {win_lbl[wi-1]} window')
        @show_resource_usage
        def plot_win_mag_dist_var(ax, oddball, wi, cumulative):
            # collect data.
            neu_seq_1, neu_seq_2 = self.get_neu_seq_trial_fix_jitter(jitter_trial_mode, oddball, cate, isi_win)
            [_, [_, _, stim_seq, _], _, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_stim_labels,
                trial_idx=[l[oddball] for l in self.list_odd_idx],
                trial_param=[None, None, [0], None, [0], [0]],
                cate=cate, roi_id=None)
            c_idx = stim_seq.shape[0]//2
            # define evaluation windows.
            if oddball == 0:
                win_eval = [[-2500, 0], [stim_seq[c_idx+1,0],stim_seq[c_idx+1,0]+300]]
            if oddball == 1:
                win_eval = [[-2500, 0], [stim_seq[c_idx,1]+self.expect, stim_seq[c_idx,1]+self.expect+300]]
            # define layouts.
            ax.axis('off')
            ax = ax.inset_axes([0, 0, 1, 0.95], transform=ax.transAxes)
            axs = [ax.inset_axes([0.1, ci/self.n_clusters, 0.5, 1/self.n_clusters*0.7],
                                 transform=ax.transAxes) for ci in range(self.n_clusters)]
            axs.reverse()
            # plot results.
            self.plot_cluster_win_mag_dist_compare(
                axs, day_cluster_id, neu_seq_1, neu_seq_2, self.alignment['neu_time'],
                win_eval, [color1, color2], 0, cumulative)
            # adjust layouts.
            hide_all_axis(ax)
            ax.set_ylabel('Response comparison')
        @show_resource_usage
        def plot_win_mag_quant_stat(ax, oddball):
            average_axis = 0
            offset = [0, 0.1]
            # collect data.
            neu_seq_fix, neu_seq_jitter = self.get_neu_seq_trial_fix_jitter(jitter_trial_mode, oddball, cate, isi_win)
            [_, [_, _, stim_seq, _], _, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_stim_labels,
                trial_idx=[l[oddball] for l in self.list_odd_idx],
                trial_param=[None, None, [0], None, [0], [0]],
                cate=cate, roi_id=None)
            c_idx = stim_seq.shape[0]//2
            # define evaluation windows.
            if oddball == 0:
                win_eval = [[-2500, 0],
                            [stim_seq[c_idx+1,0],stim_seq[c_idx+1,0]+250],
                            [stim_seq[c_idx+1,0]+300,stim_seq[c_idx+1,0]+550],
                            [stim_seq[c_idx+1,0]+600,stim_seq[c_idx+1,0]+850]]
            if oddball == 1:
                win_eval = [[-2500, 0],
                            [stim_seq[c_idx,1]+self.expect, stim_seq[c_idx,1]+self.expect+300],
                            [stim_seq[c_idx+1,0]-300,stim_seq[c_idx+1,0]],
                            [stim_seq[c_idx+1,1]-100,stim_seq[c_idx+1,1]+300]]
            # define layout.
            ax.axis('off')
            ax0 = ax.inset_axes([0, 0.97, 1, 0.03], transform=ax.transAxes)
            ax1 = ax.inset_axes([0, 0, 1, 0.95], transform=ax.transAxes)
            ax_box = [ax1.inset_axes([0.3, ci/self.n_clusters, 0.5, 0.65/self.n_clusters], transform=ax1.transAxes)
                      for ci in range(self.n_clusters)]
            axs_stat_m = [ax1.inset_axes([0.3, (0.8+ci)/self.n_clusters, 0.5, 0.1/self.n_clusters], transform=ax1.transAxes)
                   for ci in range(self.n_clusters)]
            ax_box.reverse()
            axs_stat_m.reverse()
            # plot evaluation window.
            if oddball == 0:
                ax0.axvline(stim_seq[c_idx+1,0], color='red', lw=1, linestyle='--')
            if oddball == 1:
                ax0.axvline(stim_seq[c_idx,1]+self.expect, color='gold', lw=1, linestyle='--')
                ax0.axvline(stim_seq[c_idx+1,0], color='red', lw=1, linestyle='--')
            for i in range(stim_seq.shape[0]):
                ax0.fill_between(
                    stim_seq[i,:], 0, 1,
                    color=color0, edgecolor='none', alpha=0.25, step='mid')
            self.plot_win_mag_quant_win_eval(ax0, win_eval, color0, xlim)
            # plot errorbar.
            self.plot_cluster_win_mag_quant(
                ax_box, day_cluster_id, neu_seq_fix, self.alignment['neu_time'],
                win_eval, color1, 0, offset[0], average_axis)
            self.plot_cluster_win_mag_quant(
                ax_box, day_cluster_id, neu_seq_jitter, self.alignment['neu_time'],
                win_eval, color2, 0, offset[1], average_axis)
            # plot statistics test.
            self.plot_cluster_win_mag_quant_stat(
                axs_stat_m, day_cluster_id, neu_seq_fix, neu_seq_jitter, self.alignment['neu_time'],
                win_eval, 0, average_axis, 'ttest_ind')
            # adjust layouts.
            hide_all_axis(ax1)
            ax0.set_xticks([])
            ax1.set_ylabel('Evoked magnitude')
        @show_resource_usage
        def plot_block_win_decode(ax, oddball):
            win_sample = 100
            # collect data.
            neu_seq_fix, neu_seq_jitter = self.get_neu_seq_trial_fix_jitter(jitter_trial_mode, oddball, cate, isi_win)
            [_, [_, _, stim_seq, _], _, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_stim_labels,
                trial_idx=[l[oddball] for l in self.list_odd_idx],
                trial_param=[None, None, [0], None, [0], [0]],
                cate=cate, roi_id=None)
            c_idx = stim_seq.shape[0]//2
            # run decoding for each class.
            results_all = []
            win_decode = [-500, stim_seq[c_idx+1,1]+500]
            for ci in range(self.n_clusters):
                if np.sum(cluster_id==ci) > 0:
                    # construct input data.
                    neu_x = [
                        [nsf[:,dci==ci,:] for nsf,dci in zip(neu_seq_fix, day_cluster_id)],
                        [nsj[:,dci==ci,:] for nsj,dci in zip(neu_seq_jitter, day_cluster_id)]]
                    # run decoding.
                    r = multi_sess_decoding_slide_win(
                         neu_x, self.alignment['neu_time'],
                         win_decode, win_sample)
                    results_all.append(r)
                else:
                    results_all.append([np.nan]*5)
            # split results.
            decode_model_mean  = [results_all[ci][1] for ci in range(self.n_clusters)]
            decode_model_sem   = [results_all[ci][2] for ci in range(self.n_clusters)]
            decode_chance_mean = [results_all[ci][3] for ci in range(self.n_clusters)]
            decode_chance_sem  = [results_all[ci][4] for ci in range(self.n_clusters)]
            # define layouts.
            ax.axis('off')
            ax = ax.inset_axes([0, 0, 1, 0.95], transform=ax.transAxes)
            axs = [ax.inset_axes([0, ci/self.n_clusters, 0.8, 0.8/self.n_clusters], transform=ax.transAxes)
                      for ci in range(self.n_clusters)]
            axs.reverse()
            # plot results for each class.
            for ci in range(self.n_clusters):
                if np.sum(cluster_id==ci) > 0:
                    # find bounds.
                    upper = np.nanmax([decode_model_mean[ci], decode_chance_mean[ci]])
                    lower = np.nanmin([decode_model_mean[ci], decode_chance_mean[ci]])
                    # plot stimulus.
                    c_idx = stim_seq.shape[0]//2
                    if oddball == 0:
                        axs[ci].axvline(stim_seq[c_idx+1,0], color='red', lw=1, linestyle='--')
                    if oddball == 1:
                        axs[ci].axvline(stim_seq[c_idx,1]+self.expect, color='gold', lw=1, linestyle='--')
                        axs[ci].axvline(stim_seq[c_idx+1,0], color='red', lw=1, linestyle='--')
                    for si in [0,1]:
                        axs[ci].fill_between(
                            stim_seq[c_idx+si,:],
                            0, 1,
                            color=color0, edgecolor='none', alpha=0.25, step='mid')
                    # plot decoding results.
                    self.plot_mean_sem(
                        axs[ci],
                        results_all[ci][0], decode_chance_mean[ci], decode_chance_sem[ci],
                        color0, None)
                    self.plot_mean_sem(
                        axs[ci],
                        results_all[ci][0], decode_model_mean[ci], decode_model_sem[ci],
                        color_model, None)
                    # adjust layouts.
                    axs[ci].set_xlim(xlim)
                    axs[ci].set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
            for ci in range(self.n_clusters):     
                add_ax_ticks(axs[ci], 'x', 8)
                axs[ci].spines['right'].set_visible(False)
                axs[ci].spines['top'].set_visible(False)
                axs[ci].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
                axs[ci].yaxis.set_major_locator(mtick.MaxNLocator(nbins=2))
                axs[ci].set_ylim([0.4, 0.9])
                if ci != self.n_clusters-1:
                    axs[ci].set_xticks([])
            axs[self.n_clusters-1].set_xlabel('Time from deviant (s)')
            axs[self.n_clusters-1].set_ylabel('Decoding accuracy (cross validated) \n (fixed vs jittered)')
            hide_all_axis(ax)
        @show_resource_usage
        def plot_neu_fraction(ax):
            # define layouts.
            ax.axis('off')
            ax = ax.inset_axes([0, 0, 0.70, 0.95], transform=ax.transAxes)
            # plot results.
            self.plot_cluster_neu_fraction_in_cluster(ax, cluster_id, color0)
        @show_resource_usage
        def plot_cate_fraction(ax):
            # define layouts.
            ax.axis('off')
            ax = ax.inset_axes([0, 0, 0.70, 0.95], transform=ax.transAxes)
            # plot results.
            self.plot_cluster_cate_fraction_in_cluster(ax, cluster_id, neu_labels, self.label_names, color0)
        @show_resource_usage
        def plot_legend(ax):
            [_, _, _, [n_trials, n_neurons]] = get_neu_trial(
                self.alignment, self.list_labels, self.list_stim_labels,
                trial_param=[[-1], None, [0], None, [0], [0]],
                cate=cate, roi_id=None)
            cs = [color1, color2, 'gold', 'red', 'red', color0, color_model]
            lbl = ['Fix', 'Jitter', 'Omission', 'Early stim', 'Late stim', 'Shuffle','Data']
            add_legend(ax, cs, lbl, n_trials, n_neurons, self.n_sess, 'upper right')
            ax.axis('off')
        @show_resource_usage
        def plot_block_win_decode_all(ax, oddball):
            color_model = 'plum'
            win_sample = 200
            # collect data.
            neu_seq_fix, neu_seq_jitter = self.get_neu_seq_trial_fix_jitter(jitter_trial_mode, oddball, cate, isi_win)
            [_, [_, _, stim_seq, _], _, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_stim_labels,
                trial_idx=[l[oddball] for l in self.list_odd_idx],
                trial_param=[None, None, [0], None, [0], [0]],
                cate=cate, roi_id=None)
            c_idx = stim_seq.shape[0]//2
            # run decoding for each class.
            win_decode = [-500, stim_seq[c_idx+1,1]+500]
            # construct input data.
            neu_x = [neu_seq_fix, neu_seq_jitter]
            # run decoding.
            acc_time, acc_model_mean, acc_model_sem, acc_chance_mean, acc_chance_sem = multi_sess_decoding_slide_win(
                neu_x, self.alignment['neu_time'],
                win_decode, win_sample)
            # define layouts.
            ax.axis('off')
            ax = ax.inset_axes([0, 0, 0.8, 0.6], transform=ax.transAxes)
            # plot stimulus.
            c_idx = stim_seq.shape[0]//2
            if oddball == 0:
                ax.axvline(stim_seq[c_idx+1,0], color='red', lw=1, linestyle='--')
            if oddball == 1:
                ax.axvline(stim_seq[c_idx,1]+self.expect, color='gold', lw=1, linestyle='--')
                ax.axvline(stim_seq[c_idx+1,0], color='red', lw=1, linestyle='--')
            for si in [0,1]:
                ax.fill_between(
                    stim_seq[c_idx+si,:],
                    0, 1,
                    color=color0, edgecolor='none', alpha=0.25, step='mid')
            # plot decoding results.
            self.plot_mean_sem(ax, acc_time, acc_chance_mean, acc_chance_sem, color0, None)
            self.plot_mean_sem(ax, acc_time, acc_model_mean, acc_model_sem, color_model, None)
            # adjust layouts.
            ax.set_xlim(xlim)
            ax.set_ylim([0.4, 0.9])
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, pos: f'{x/1000:.1f}'))
            ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
            ax.xaxis.set_major_locator(mtick.MaxNLocator(nbins=4))
            ax.yaxis.set_major_locator(mtick.MaxNLocator(nbins=3))
            ax.set_xlabel('Time from deviant (s)')
            ax.set_ylabel('Decoding accuracy (cross validated) \n (fixed vs jittered)')
        @show_resource_usage
        def plot_oddball_jitter_layers(ax):
            oddball = 1
            nc_pre  = [len(np.unique(col))-1 for col in self.cluster_id_pre_layers.T]
            nc_post = [len(np.unique(col))-1 for col in self.cluster_id_post_layers.T]
            cid_pre  = [get_cluster_cate(col, self.list_labels, cate)[1] for col in self.cluster_id_pre_layers.T]
            cid_post = [get_cluster_cate(col, self.list_labels, cate)[1] for col in self.cluster_id_post_layers.T]
            # collect data.
            [_, [_, _, stim_seq, _], _, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_stim_labels,
                trial_idx=[l[oddball] for l in self.list_odd_idx],
                trial_param=[None, None, [0], None, [0], [0]],
                cate=cate, roi_id=None)
            c_idx = stim_seq.shape[0]//2
            neu_seq_fix, neu_seq_jitter = self.get_neu_seq_trial_fix_jitter(jitter_trial_mode, oddball, cate, isi_win)
            neu_seq_jitter = np.concatenate([np.nanmean(neu, axis=0) for neu in neu_seq_jitter], axis=0)
            neu_seq_fix = np.concatenate([np.nanmean(neu, axis=0) for neu in neu_seq_fix], axis=0)
            neu_seq_fix_pre  = neu_seq_fix[cluster_id_pre, l_idx:r_idx]
            neu_seq_fix_post = neu_seq_fix[cluster_id_post, l_idx:r_idx]
            neu_time = self.alignment['neu_time'][l_idx:r_idx]
            # define layouts.
            axs0 = [ax.inset_axes([0.30, 0, 0.1, 0.2], transform=ax.transAxes),
                    ax.inset_axes([0.15, 0, 0.1, 0.4], transform=ax.transAxes),
                    ax.inset_axes([0.00, 0, 0.1, 0.8], transform=ax.transAxes)]
            axs1 = [ax.inset_axes([0.55, 0, 0.1, 0.2], transform=ax.transAxes),
                    ax.inset_axes([0.70, 0, 0.1, 0.4], transform=ax.transAxes),
                    ax.inset_axes([0.85, 0, 0.1, 0.8], transform=ax.transAxes)]
            axs2 = [ax.inset_axes([0.30, 0.6, 0.1, 0.2], transform=ax.transAxes),
                    ax.inset_axes([0.55, 0.6, 0.1, 0.2], transform=ax.transAxes)]
            axs_hm = [axs2[0].inset_axes([0, 0, 0.7, 1], transform=axs2[0].transAxes),
                      axs2[1].inset_axes([0, 0, 0.7, 1], transform=axs2[1].transAxes)]
            axs_cb = [axs2[0].inset_axes([0.8, 0, 0.1, 1], transform=axs2[0].transAxes),
                      axs2[1].inset_axes([0.8, 0, 0.1, 1], transform=axs2[1].transAxes)]
            # get response within cluster.
            for axi, n_clusters, cluster_id in zip(axs0+axs1, nc_pre+nc_post, cid_pre+cid_post):
                neu_fix_mean, neu_fix_sem = get_mean_sem_cluster(neu_seq_fix, n_clusters, cluster_id)
                neu_jitter_mean, neu_jitter_sem = get_mean_sem_cluster(neu_seq_jitter, n_clusters, cluster_id)
                norm_params = [get_norm01_params(
                    np.concatenate([neu_fix_mean[ci,:], neu_jitter_mean[ci,:]]))
                    for ci in range(n_clusters)]
                # plot results.
                if oddball == 0:
                    axi.axvline(stim_seq[c_idx+1,0], color='red', lw=1, linestyle='--')
                if oddball == 1:
                    axi.axvline(stim_seq[c_idx,1]+self.expect, color='gold', lw=1, linestyle='--')
                    axi.axvline(stim_seq[c_idx+1,0], color='red', lw=1, linestyle='--')
                c_stim = [color0 if i in (c_idx, c_idx + 1) else f'empty{color0}' for i in range(stim_seq.shape[0])]
                self.plot_cluster_mean_sem(
                    axi, neu_fix_mean, neu_fix_sem,
                    self.alignment['neu_time'], norm_params,
                    stim_seq,
                    c_stim, [color1]*n_clusters, xlim)
                self.plot_cluster_mean_sem(
                    axi, neu_jitter_mean, neu_jitter_sem,
                    self.alignment['neu_time'], norm_params,
                    None, None, [color2]*n_clusters, xlim,
                    scale_bar=False)
            # plot heatmaps.
            self.plot_heatmap_neuron(
                axs_hm[0], axs_cb[0], neu_seq_fix_pre, neu_time, neu_seq_fix_pre,
                norm_mode='minmax',
                neu_seq_share=[neu_seq_fix_pre, neu_seq_fix_post])
            self.plot_heatmap_neuron(
                axs_hm[1], axs_cb[1], neu_seq_fix_post, neu_time, neu_seq_fix_post,
                norm_mode='minmax',
                neu_seq_share=[neu_seq_fix_pre, neu_seq_fix_post])
            # add stimulus line.
            for axi in axs_hm:
                xlines = [self.alignment['neu_time'][np.searchsorted(self.alignment['neu_time'], t)]
                          for t in [stim_seq[c_idx+i,0]
                                    for i in [-2,-1,0,1,2]]]
                for xl in xlines:
                    if xl>neu_time[0] and xl<neu_time[-1]:
                        axi.axvline(xl, color='black', lw=1, linestyle='--')
            # adjust layouts.
            axs0[1].set_xlabel('Time from deviant (s)')
            axs1[1].set_xlabel('Time from deviant (s)')
            hide_all_axis(ax)
            hide_all_axis(axs2[0])
            hide_all_axis(axs2[1])
        # plot all.
        try: plot_oddball_jitter(axs[0], 0)
        except: traceback.print_exc()
        try: plot_win_mag_quant_stat(axs[1], 0)
        except: traceback.print_exc()
        try: plot_win_mag_scatter(axs[2], 0, 2)
        except: traceback.print_exc()
        try: plot_win_mag_dist_var(axs[3], 0, 2, False)
        except: traceback.print_exc()
        try: plot_block_win_decode(axs[4], 0)
        except: traceback.print_exc()
        try: plot_oddball_jitter(axs[5], 1)
        except: traceback.print_exc()
        try: plot_win_mag_quant_stat(axs[6], 1)
        except: traceback.print_exc()
        try: plot_win_mag_scatter(axs[7], 1, 2)
        except: traceback.print_exc()
        try: plot_win_mag_dist_var(axs[8], 1, 2, False)
        except: traceback.print_exc()
        try: plot_block_win_decode(axs[9], 1)
        except: traceback.print_exc()
        try: plot_neu_fraction(axs[10])
        except: traceback.print_exc()
        try: plot_cate_fraction(axs[11])
        except: traceback.print_exc()
        try: plot_legend(axs[12])
        except: traceback.print_exc()
        try: plot_block_win_decode_all(axs[13], 0)
        except: traceback.print_exc()
        try: plot_block_win_decode_all(axs[14], 1)
        except: traceback.print_exc()
        try: plot_oddball_jitter_layers(axs[15])
        except: traceback.print_exc()

    def plot_cluster_oddball_jitter_local_all(self, axs, jitter_trial_mode, cate):
        color0 = 'dimgrey'
        color1 = 'mediumseagreen'
        color2 = 'coral'
        cs = [color1, color2]
        win_lbl = ['early', 'late', 'post']
        xlim = [-3000, 4000]
        l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, xlim[0], xlim[1])
        _, cluster_id, neu_labels = get_cluster_cate(self.cluster_id, self.list_labels, cate)
        split_idx = get_split_idx(self.list_labels, cate)
        day_cluster_id = np.split(cluster_id, split_idx)
        @show_resource_usage
        def plot_oddball_jitter(ax, oddball):
            # collect data.
            [_, [neu_seq, stim_seq, camera_pupil, pre_isi, _], _, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_stim_labels,
                trial_idx=[l[oddball] for l in self.list_odd_idx],
                trial_param=[None, None, [1], None, [0], [0]],
                mean_sem=False,
                cate=cate, roi_id=None)
            # bin data based on isi.
            [bins, bin_center, _, bin_neu_seq, _, _, bin_stim_seq, bin_camera_pupil] = get_isi_bin_neu(
                neu_seq, stim_seq, camera_pupil, pre_isi, self.bin_win, self.bin_num)
            stim_seq = np.nanmean(np.concatenate(stim_seq, axis=0), axis=0)
            c_idx = stim_seq.shape[0]//2
            cluster_bin_neu_mean, cluster_bin_neu_sem = get_bin_mean_sem_cluster(bin_neu_seq, self.n_clusters, cluster_id)
            norm_params = [get_norm01_params(cluster_bin_neu_mean[:,ci,:]) for ci in range(self.n_clusters)]
            # define layouts.
            ax.axis('off')
            ax = ax.inset_axes([0, 0, 1, 0.95], transform=ax.transAxes)
            # plot results.
            for si in [0,1]:
                ax.fill_between(
                    stim_seq[c_idx+si,:],
                    0, self.n_clusters,
                    color=color0, edgecolor='none', alpha=0.25, step='mid')
            if oddball == 0:
                ax.axvline(stim_seq[c_idx+1,0], color='red', lw=1, linestyle='--')
            if oddball == 1:
                ax.axvline(stim_seq[c_idx,1]+self.expect, color='gold', lw=1, linestyle='--')
                ax.axvline(stim_seq[c_idx+1,0], color='red', lw=1, linestyle='--')
            for bi in range(self.bin_num):
                ax.fill_between(
                    bin_stim_seq[bi, c_idx-1, :],
                    0, self.n_clusters,
                    color='none', edgecolor=cs[bi], alpha=0.25, step='mid')
            for bi in range(self.bin_num):
                self.plot_cluster_mean_sem(
                    ax, cluster_bin_neu_mean[bi,:,:], cluster_bin_neu_sem[bi,:,:],
                    self.alignment['neu_time'], norm_params,
                    None, None, [cs[bi]]*self.n_clusters, xlim,
                    scale_bar=True if bi==0 else False)
            # adjust layouts.
            ax.set_xlabel('Time from stim onset (s)')
        @show_resource_usage
        def plot_win_mag_quant_stat(ax, oddball):
            average_axis = 1
            offset = [0, 0.1]
            # collect data.
            [_, [neu_seq, stim_seq, camera_pupil, pre_isi, _], _, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_stim_labels,
                trial_idx=[l[oddball] for l in self.list_odd_idx],
                trial_param=[None, None, [1], None, [0], [0]],
                mean_sem=False,
                cate=cate, roi_id=None)
            # bin data based on isi.
            [_, _, bin_neu_seq_trial, _, _, _, _, _] = get_isi_bin_neu(
                neu_seq, stim_seq, camera_pupil, pre_isi, self.bin_win, self.bin_num)
            stim_seq = np.nanmean(np.concatenate(stim_seq, axis=0), axis=0)
            c_idx = stim_seq.shape[0]//2
            # define evaluation windows.
            if oddball == 0:
                win_eval = [[-2500, 0],
                            [stim_seq[c_idx+1,0],stim_seq[c_idx+1,0]+250],
                            [stim_seq[c_idx+1,0]+300,stim_seq[c_idx+1,0]+550],
                            [stim_seq[c_idx+1,0]+600,stim_seq[c_idx+1,0]+850]]
            if oddball == 1:
                win_eval = [[-2500, 0],
                            [stim_seq[c_idx,1]+self.expect, stim_seq[c_idx,1]+self.expect+250],
                            [stim_seq[c_idx+1,0]-250,stim_seq[c_idx+1,0]],
                            [stim_seq[c_idx+1,1],stim_seq[c_idx+1,1]+250]]
            # define layout.
            ax.axis('off')
            ax0 = ax.inset_axes([0, 0.97, 1, 0.03], transform=ax.transAxes)
            ax1 = ax.inset_axes([0, 0, 1, 0.95], transform=ax.transAxes)
            ax_box = [ax1.inset_axes([0.3, ci/self.n_clusters, 0.5, 0.65/self.n_clusters], transform=ax1.transAxes)
                      for ci in range(self.n_clusters)]
            axs_stat_m = [ax1.inset_axes([0.3, (0.8+ci)/self.n_clusters, 0.5, 0.1/self.n_clusters], transform=ax1.transAxes)
                   for ci in range(self.n_clusters)]
            axs_stat_v = [ax1.inset_axes([0.3, (0.7+ci)/self.n_clusters, 0.5, 0.1/self.n_clusters], transform=ax1.transAxes)
                   for ci in range(self.n_clusters)]
            ax_box.reverse()
            axs_stat_m.reverse()
            axs_stat_v.reverse()
            # plot evaluation window.
            if oddball == 0:
                ax0.axvline(stim_seq[c_idx+1,0], color='gold', lw=1, linestyle='--')
            if oddball == 1:
                ax0.axvline(stim_seq[c_idx,1]+self.expect, color='gold', lw=1, linestyle='--')
            for i in range(stim_seq.shape[0]):
                ax0.fill_between(
                    stim_seq[i,:], 0, 1,
                    color=color0, edgecolor='none', alpha=0.25, step='mid')
            self.plot_win_mag_quant_win_eval(ax0, win_eval, color0, xlim)
            # plot errorbar.
            for bi in range(self.bin_num):
                self.plot_cluster_win_mag_quant(
                    ax_box, day_cluster_id, bin_neu_seq_trial[bi], self.alignment['neu_time'],
                    win_eval, cs[bi], 0, offset[1], average_axis)
            # plot statistics test.
            self.plot_cluster_win_mag_quant_stat(
                axs_stat_m, day_cluster_id, bin_neu_seq_trial[0], bin_neu_seq_trial[-1], self.alignment['neu_time'],
                win_eval, 0, average_axis, 'ttest_ind')
            self.plot_cluster_win_mag_quant_stat(
                axs_stat_v, day_cluster_id, bin_neu_seq_trial[0], bin_neu_seq_trial[-1], self.alignment['neu_time'],
                win_eval, 0, average_axis, 'levene')
            # adjust layouts.
            hide_all_axis(ax1)
            ax0.set_xticks([])
            ax1.set_ylabel('Evoked magnitude')
        @show_resource_usage
        def plot_win_mag_scatter(ax, oddball, wi):
            average_axis = 1
            # collect data.
            [_, [neu_seq, stim_seq, camera_pupil, pre_isi, _], _, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_stim_labels,
                trial_idx=[l[oddball] for l in self.list_odd_idx],
                trial_param=[None, None, [1], None, [0], [0]],
                mean_sem=False,
                cate=cate, roi_id=None)
            # bin data based on isi.
            [_, _, bin_neu_seq_trial, _, _, _, _, _] = get_isi_bin_neu(
                neu_seq, stim_seq, camera_pupil, pre_isi, self.bin_win, self.bin_num)
            stim_seq = np.nanmean(np.concatenate(stim_seq, axis=0), axis=0)
            c_idx = stim_seq.shape[0]//2
            # define evaluation windows.
            if oddball == 0:
                win_eval = [[-2500, 0],
                            [stim_seq[c_idx+1,0],stim_seq[c_idx+1,0]+250],
                            [stim_seq[c_idx+1,0]+300,stim_seq[c_idx+1,0]+550],
                            [stim_seq[c_idx+1,0]+600,stim_seq[c_idx+1,0]+850]]
            if oddball == 1:
                win_eval = [[-2500, 0],
                            [stim_seq[c_idx,1]+self.expect, stim_seq[c_idx,1]+self.expect+250],
                            [stim_seq[c_idx+1,0]-250,stim_seq[c_idx+1,0]],
                            [stim_seq[c_idx+1,1],stim_seq[c_idx+1,1]+250]]
            win_eval = [win_eval[0], win_eval[wi]]
            # define layouts.
            ax.axis('off')
            ax = ax.inset_axes([0, 0, 1, 0.95], transform=ax.transAxes)
            axs = [ax.inset_axes([0.3, ci/self.n_clusters, 0.5, 0.7/self.n_clusters], transform=ax.transAxes)
                      for ci in range(self.n_clusters)]
            axs.reverse()
            # plot results.
            self.plot_cluster_win_mag_scatter(
                axs, day_cluster_id, bin_neu_seq_trial[0], bin_neu_seq_trial[-1], self.alignment['neu_time'],
                win_eval, win_eval, color0, 0, average_axis)
            # adjust layouts.
            for ci in range(self.n_clusters):
                axs[ci].set_ylabel('jitter')
            axs[self.n_clusters-1].set_xlabel('fix')
            hide_all_axis(ax)
            ax.set_ylabel(f'Response comparison in the {win_lbl[wi-1]} window \n')
        @show_resource_usage
        def plot_win_mag_dist(ax, oddball, wi, cumulative):
            # collect data.
            [_, [neu_seq, stim_seq, camera_pupil, pre_isi, _], _, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_stim_labels,
                trial_idx=[l[oddball] for l in self.list_odd_idx],
                trial_param=[None, None, [1], None, [0], [0]],
                mean_sem=False,
                cate=cate, roi_id=None)
            # bin data based on isi.
            [_, _, bin_neu_seq_trial, _, _, _, _, _] = get_isi_bin_neu(
                neu_seq, stim_seq, camera_pupil, pre_isi, self.bin_win, self.bin_num)
            stim_seq = np.nanmean(np.concatenate(stim_seq, axis=0), axis=0)
            c_idx = stim_seq.shape[0]//2
            # define evaluation windows.
            if oddball == 0:
                win_eval = [[-2500, 0],
                            [stim_seq[c_idx+1,0],stim_seq[c_idx+1,0]+250],
                            [stim_seq[c_idx+1,0]+300,stim_seq[c_idx+1,0]+550],
                            [stim_seq[c_idx+1,0]+600,stim_seq[c_idx+1,0]+850]]
            if oddball == 1:
                win_eval = [[-2500, 0],
                            [stim_seq[c_idx,1]+self.expect, stim_seq[c_idx,1]+self.expect+250],
                            [stim_seq[c_idx+1,0]-250,stim_seq[c_idx+1,0]],
                            [stim_seq[c_idx+1,1],stim_seq[c_idx+1,1]+250]]
            win_eval = [win_eval[0], win_eval[wi]]
            # define layouts.
            ax.axis('off')
            ax = ax.inset_axes([0, 0, 1, 0.95], transform=ax.transAxes)
            axs = [ax.inset_axes([0.1, ci/self.n_clusters, 0.5, 1/self.n_clusters*0.7],
                                 transform=ax.transAxes) for ci in range(self.n_clusters)]
            axs.reverse()
            # plot results.
            self.plot_cluster_win_mag_dist_compare(
                axs, day_cluster_id, bin_neu_seq_trial[0], bin_neu_seq_trial[-1], self.alignment['neu_time'],
                win_eval, [cs[0], cs[-1]], 0, cumulative)
            # adjust layouts.
            hide_all_axis(ax)
            ax.set_ylabel(f'Response comparison in the {win_lbl[wi-1]} window')
        @show_resource_usage
        def plot_interval_scaling(ax, oddball):
            post_color = 'lightcoral'
            gap = 2
            bin_num = 50
            win_eval = [[-2500,0], [0, 400]]
            order = 2
            # collect data.
            [_, [neu_seq, stim_seq, camera_pupil, pre_isi, _], _, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_stim_labels,
                trial_idx=[l[oddball] for l in self.list_odd_idx],
                trial_param=[None, None, [1], None, [0], [0]],
                mean_sem=False,
                cate=cate, roi_id=None)
            # bin data based on isi.
            [_, bin_center, _, bin_neu_seq, _, _, _, _] = get_isi_bin_neu(
                neu_seq, stim_seq, camera_pupil, pre_isi, self.bin_win, bin_num)
            bin_center = bin_center[gap:-gap]
            stim_seq = np.nanmean(np.concatenate(stim_seq, axis=0), axis=0)
            c_idx = stim_seq.shape[0]//2
            # get response within cluster.
            bin_win_baseline = [np.array([get_mean_sem_win(
                bns[cluster_id==ci,:], self.alignment['neu_time'], 0,
                win_eval[0][0], win_eval[0][1], 'lower')[1]
                for bns in bin_neu_seq])[gap:-gap] for ci in range(self.n_clusters)]
            bin_win_evoked_pre = [np.array([get_mean_sem_win(
                bns[cluster_id==ci,:], self.alignment['neu_time'], 0,
                stim_seq[c_idx,0]+win_eval[1][0], stim_seq[c_idx,0]+win_eval[1][1], 'mean')[1]
                for bns in bin_neu_seq])[gap:-gap] for ci in range(self.n_clusters)]
            bin_win_evoked_post = [np.array([get_mean_sem_win(
                bns[cluster_id==ci,:], self.alignment['neu_time'], 0,
                stim_seq[c_idx+1,0]+win_eval[1][0], stim_seq[c_idx+1,0]+win_eval[1][1], 'mean')[1]
                for bns in bin_neu_seq])[gap:-gap] for ci in range(self.n_clusters)]
            bin_win_pre  = [bwe - bwb for bwe, bwb in zip (bin_win_evoked_pre, bin_win_baseline)]
            bin_win_post = [bwe - bwb for bwe, bwb in zip (bin_win_evoked_post, bin_win_baseline)]
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
                    axs[ci].spines['right'].set_visible(False)
                    axs[ci].spines['top'].set_visible(False)
                    axs[ci].set_xlim(self.bin_win)
                    axs[ci].set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
                    axs[ci].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
                    axs[ci].xaxis.set_major_locator(mtick.MaxNLocator(nbins=2))
                    axs[ci].yaxis.set_major_locator(mtick.MaxNLocator(nbins=2))
                    if ci != self.n_clusters-1:
                        axs[ci].set_xticklabels([])
            axs[self.n_clusters-1].set_xlabel('Preceding interval (ms)')
            ax.set_ylabel(f'Rvoked magnitude in the {win_eval[1]} window')
            hide_all_axis(ax)
            add_legend(ax, [color0, post_color], ['pre','post'], None, None, None, 'upper right')
        @show_resource_usage
        def plot_block_win_decode(ax, oddball):
            color_model = 'plum'
            win_decode = [-2500, 4000]
            win_sample = 400
            # collect data.
            [_, [neu_seq, stim_seq, camera_pupil, pre_isi, _], _, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_stim_labels,
                trial_idx=[l[oddball] for l in self.list_odd_idx],
                trial_param=[None, None, [1], None, [0], [0]],
                mean_sem=False,
                cate=cate, roi_id=None)
            # bin data based on isi.
            [_, _, bin_neu_seq_trial, _, _, _, bin_stim_seq, _] = get_isi_bin_neu(
                neu_seq, stim_seq, camera_pupil, pre_isi, self.bin_win, self.bin_num)
            stim_seq = np.nanmean(np.concatenate(stim_seq, axis=0), axis=0)
            c_idx = stim_seq.shape[0]//2
            # run decoding for each class.
            results_all = []
            for ci in range(self.n_clusters):
                if np.sum(cluster_id==ci) > 0:
                    # construct input data.
                    neu_x = [
                        [ns[:,dci==ci,:] for ns,dci in zip(bin_neu_seq_trial[0], day_cluster_id)],
                        [ns[:,dci==ci,:] for ns,dci in zip(bin_neu_seq_trial[-1], day_cluster_id)]]
                    # run decoding.
                    r = multi_sess_decoding_slide_win(
                         neu_x, self.alignment['neu_time'],
                         win_decode, win_sample)
                    results_all.append(r)
                else:
                    results_all.append([np.nan]*5)
            # split results.
            decode_model_mean  = [results_all[ci][1] for ci in range(self.n_clusters)]
            decode_model_sem   = [results_all[ci][2] for ci in range(self.n_clusters)]
            decode_chance_mean = [results_all[ci][3] for ci in range(self.n_clusters)]
            decode_chance_sem  = [results_all[ci][4] for ci in range(self.n_clusters)]
            # define layouts.
            ax.axis('off')
            ax = ax.inset_axes([0, 0, 1, 0.95], transform=ax.transAxes)
            axs = [ax.inset_axes([0, ci/self.n_clusters, 0.8, 0.9/self.n_clusters], transform=ax.transAxes)
                      for ci in range(self.n_clusters)]
            axs.reverse()
            # plot results for each class.
            for ci in range(self.n_clusters):
                if np.sum(cluster_id==ci) > 0:
                    # find bounds.
                    upper = np.nanmax([decode_model_mean[ci], decode_chance_mean[ci]])
                    lower = np.nanmin([decode_model_mean[ci], decode_chance_mean[ci]])
                    # plot stimulus.
                    for si in [0,1]:
                        axs[ci].fill_between(
                            stim_seq[c_idx+si,:],
                            0, self.n_clusters,
                            color=color0, edgecolor='none', alpha=0.25, step='mid')
                    if oddball == 0:
                        axs[ci].axvline(stim_seq[c_idx+1,0], color='red', lw=1, linestyle='--')
                    if oddball == 1:
                        axs[ci].axvline(stim_seq[c_idx,1]+self.expect, color='gold', lw=1, linestyle='--')
                        axs[ci].axvline(stim_seq[c_idx+1,0], color='red', lw=1, linestyle='--')
                    for bi in range(self.bin_num):
                        axs[ci].fill_between(
                            bin_stim_seq[bi, c_idx-1, :],
                            0, self.n_clusters,
                            color='none', edgecolor=cs[bi], alpha=0.25, step='mid')
                    # plot decoding results.
                    self.plot_mean_sem(
                        axs[ci],
                        results_all[ci][0], decode_chance_mean[ci], decode_chance_sem[ci],
                        color0, None)
                    self.plot_mean_sem(
                        axs[ci],
                        results_all[ci][0], decode_model_mean[ci], decode_model_sem[ci],
                        color_model, None)
                    # adjust layouts.
                    axs[ci].set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
            for ci in range(self.n_clusters):           
                axs[ci].spines['right'].set_visible(False)
                axs[ci].spines['top'].set_visible(False)
                axs[ci].xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, pos: f'{x/1000:.1f}'))
                axs[ci].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
                axs[ci].xaxis.set_major_locator(mtick.MaxNLocator(nbins=4))
                axs[ci].yaxis.set_major_locator(mtick.MaxNLocator(nbins=2))
                axs[ci].set_ylim([0.4, 0.9])
                if ci != self.n_clusters-1:
                    axs[ci].set_xticks([])
            axs[self.n_clusters-1].set_xlabel('Time from deviant (s)')
            axs[self.n_clusters-1].set_ylabel('Decoding accuracy \n (fixed vs jittered)')
            hide_all_axis(ax)
        @show_resource_usage
        def plot_neu_fraction(ax):
            # define layouts.
            ax.axis('off')
            ax = ax.inset_axes([0, 0, 0.70, 0.95], transform=ax.transAxes)
            # plot results.
            self.plot_cluster_neu_fraction_in_cluster(ax, cluster_id, color0)
        @show_resource_usage
        def plot_cate_fraction(ax):
            # define layouts.
            ax.axis('off')
            ax = ax.inset_axes([0, 0, 0.70, 0.95], transform=ax.transAxes)
            # plot results.
            self.plot_cluster_cate_fraction_in_cluster(ax, cluster_id, neu_labels, self.label_names, color0)
        @show_resource_usage
        def plot_legend(ax):
            [_, _, _, [n_trials, n_neurons]] = get_neu_trial(
                self.alignment, self.list_labels, self.list_stim_labels,
                trial_param=[[-1], None, [0], None, [0], [0]],
                cate=cate, roi_id=None)
            cs = [color1, color2, 'gold']
            lbl = ['Short preceding ISI', 'Long preceding ISI', 'unexpected event']
            add_legend(ax, cs, lbl, n_trials, n_neurons, self.n_sess, 'upper right')
            ax.axis('off')
        # plot all.
        try: plot_oddball_jitter(axs[0], 0)
        except: traceback.print_exc()
        try: plot_win_mag_quant_stat(axs[1], 0)
        except: traceback.print_exc()
        try: plot_win_mag_scatter(axs[2], 0, 2)
        except: traceback.print_exc()
        try: plot_win_mag_dist(axs[3], 0, 2, False)
        except: traceback.print_exc()
        try: plot_interval_scaling(axs[4], 0)
        except: traceback.print_exc()
        try: plot_block_win_decode(axs[5], 0)
        except: traceback.print_exc()
        try: plot_oddball_jitter(axs[6], 1)
        except: traceback.print_exc()
        try: plot_win_mag_quant_stat(axs[7], 1)
        except: traceback.print_exc()
        try: plot_win_mag_scatter(axs[8], 1, 2)
        except: traceback.print_exc()
        try: plot_win_mag_dist(axs[9], 1, 2, False)
        except: traceback.print_exc()
        try: plot_interval_scaling(axs[10], 1)
        except: traceback.print_exc()
        try: plot_block_win_decode(axs[11], 1)
        except: traceback.print_exc()
        try: plot_neu_fraction(axs[12])
        except: traceback.print_exc()
        try: plot_cate_fraction(axs[13])
        except: traceback.print_exc()
        try: plot_legend(axs[14])
        except: traceback.print_exc()
    
    def plot_cluster_oddball_jitter_global_modulation(self, axs, cate):
        jitter_trial_mode = 'global'
        color0 = 'dimgrey'
        color1 = 'deeppink'
        color2 = 'royalblue'
        isi_win = 250
        xlim = [-2500, 4000]
        l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, xlim[0], xlim[1])
        _, cluster_id, neu_labels = get_cluster_cate(self.cluster_id, self.list_labels, cate)
        split_idx = get_split_idx(self.list_labels, cate)
        day_cluster_id = np.split(cluster_id, split_idx)
        @show_resource_usage
        def plot_pred_mod_index_box(ax, oddball, pe):
            mi_tol = 0.1
            n_trials = 10
            # collect data.
            [_, [_, _, stim_seq, _], _, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_stim_labels,
                trial_idx=[l[oddball] for l in self.list_odd_idx],
                trial_param=[None, None, [0], None, [0], [0]],
                cate=cate, roi_id=None)
            c_idx = stim_seq.shape[0]//2
            neu_seq_fix, neu_seq_jitter = self.get_neu_seq_trial_fix_jitter(jitter_trial_mode, oddball, cate, isi_win)
            neu_seq_0 = np.concatenate([np.nanmean(neu, axis=0) for neu in neu_seq_fix], axis=0)
            neu_seq_1 = np.concatenate([np.nanmean(neu, axis=0) for neu in neu_seq_jitter], axis=0)
            # define evaluation windows.
            if pe == 'pos':
                win_eval = [[-2500, 4000],
                            [stim_seq[c_idx,0]-100, stim_seq[c_idx,0]+500],
                            [stim_seq[c_idx+1,0]-100, stim_seq[c_idx+1,0]+500]]
            if pe == 'neg':
                win_eval = [[-2500, 4000],
                            [stim_seq[c_idx,1]+self.expect-500, stim_seq[c_idx,1]+self.expect],
                            [stim_seq[c_idx,1]+self.expect, stim_seq[c_idx,1]+self.expect+500]]
            # compute index.
            mi1 = np.full(neu_seq_0.shape[0], np.nan)
            mi2 = np.full(neu_seq_1.shape[0], np.nan)
            for ni, _ in enumerate(mi1):
                mi1[ni] = get_modulation_index_neu_seq(neu_seq_0[ni,:].reshape(1,-1), self.alignment['neu_time'], 0, win_eval)
                mi2[ni] = get_modulation_index_neu_seq(neu_seq_1[ni,:].reshape(1,-1), self.alignment['neu_time'], 0, win_eval)
            # get response within cluster.
            neu_seq_0 = np.concatenate([np.nanmean(neu[:n_trials,:,:], axis=0) for neu in neu_seq_fix], axis=0)
            neu_seq_1 = np.concatenate([np.nanmean(neu[:n_trials,:,:], axis=0) for neu in neu_seq_jitter], axis=0)
            cluster_id_high = cluster_id.copy()
            cluster_id_low  = cluster_id.copy()
            cluster_id_high[(mi1 < mi_tol) | (mi2 < mi_tol)] = -1
            cluster_id_low[(mi1 > -mi_tol) | (mi2 > -mi_tol)] = -1
            neu_0_mean_high, neu_0_sem_high = get_mean_sem_cluster(neu_seq_0, self.n_clusters, cluster_id_high)
            neu_0_mean_low,  neu_0_sem_low  = get_mean_sem_cluster(neu_seq_0, self.n_clusters, cluster_id_low)
            neu_1_mean_high, neu_1_sem_high = get_mean_sem_cluster(neu_seq_1, self.n_clusters, cluster_id_high)
            neu_1_mean_low,  neu_1_sem_low  = get_mean_sem_cluster(neu_seq_1, self.n_clusters, cluster_id_low)
            norm_params = [get_norm01_params(
                np.concatenate([neu_0_mean_high[ci,:], neu_0_mean_low[ci,:], neu_1_mean_high[ci,:], neu_1_mean_low[ci,:]]))
                for ci in range(self.n_clusters)]
            # define layouts.
            ax0 = ax.inset_axes([0, 0.97, 0.25, 0.03], transform=ax.transAxes)
            ax1 = ax.inset_axes([0, 0, 0.25, 0.95], transform=ax.transAxes)
            ax2 = ax.inset_axes([0.4, 0, 0.25, 0.95], transform=ax.transAxes)
            ax3 = ax.inset_axes([0.75, 0, 0.25, 0.95], transform=ax.transAxes)
            axs = [ax1.inset_axes([0, ci/self.n_clusters, 1, 0.7/self.n_clusters], transform=ax1.transAxes)
                      for ci in range(self.n_clusters)]
            axs.reverse()
            # plot evaluation window.
            if oddball == 0:
                ax0.axvline(stim_seq[c_idx+1,0], color='red', lw=1, linestyle='--')
            if oddball == 1:
                ax0.axvline(stim_seq[c_idx,1]+self.expect, color='gold', lw=1, linestyle='--')
                ax0.axvline(stim_seq[c_idx+1,0], color='red', lw=1, linestyle='--')
            for i in range(stim_seq.shape[0]):
                ax0.fill_between(
                    stim_seq[i,:], 0, 1,
                    color=color0, edgecolor='none', alpha=0.25, step='mid')
            self.plot_win_mag_quant_win_eval(ax0, win_eval, color0, xlim, False)
            # plot results for each class.
            for ci in range(self.n_clusters):
                if np.sum(cluster_id==ci) > 0:
                    self.plot_pred_mod_index_dist(axs[ci], mi1[cluster_id==ci], mi2[cluster_id==ci], color0, color1, color2)
            # plot high index.
            if oddball == 0:
                ax2.axvline(stim_seq[c_idx+1,0], color='red', lw=1, linestyle='--')
            if oddball == 1:
                ax2.axvline(stim_seq[c_idx,1]+self.expect, color='gold', lw=1, linestyle='--')
                ax2.axvline(stim_seq[c_idx+1,0], color='red', lw=1, linestyle='--')
            c_stim = [color0 if i in (c_idx, c_idx + 1) else f'empty{color0}' for i in range(stim_seq.shape[0])]
            self.plot_cluster_mean_sem(
                ax2, neu_0_mean_high, neu_0_sem_high,
                self.alignment['neu_time'], norm_params,
                stim_seq,
                c_stim, [color1]*self.n_clusters, xlim)
            self.plot_cluster_mean_sem(
                ax2, neu_1_mean_high, neu_1_sem_high,
                self.alignment['neu_time'], norm_params,
                None, None, [color2]*self.n_clusters, xlim,
                scale_bar=False)
            # plot low index.
            if oddball == 0:
                ax3.axvline(stim_seq[c_idx+1,0], color='red', lw=1, linestyle='--')
            if oddball == 1:
                ax3.axvline(stim_seq[c_idx,1]+self.expect, color='gold', lw=1, linestyle='--')
                ax3.axvline(stim_seq[c_idx+1,0], color='red', lw=1, linestyle='--')
            c_stim = [color0 if i in (c_idx, c_idx + 1) else f'empty{color0}' for i in range(stim_seq.shape[0])]
            self.plot_cluster_mean_sem(
                ax3, neu_0_mean_low, neu_0_sem_low,
                self.alignment['neu_time'], norm_params,
                stim_seq,
                c_stim, [color1]*self.n_clusters, xlim)
            self.plot_cluster_mean_sem(
                ax3, neu_1_mean_low, neu_1_sem_low,
                self.alignment['neu_time'], norm_params,
                None, None, [color2]*self.n_clusters, xlim,
                scale_bar=False)
            # adjust layouts.
            for ci in range(self.n_clusters):
                axs[ci].xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
                axs[ci].xaxis.set_major_locator(mtick.MaxNLocator(nbins=3))
                axs[ci].xaxis.set_minor_locator(mtick.AutoMinorLocator(2))
                axs[ci].tick_params(axis='x', which='minor', labelbottom=False)
                axs[ci].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
                axs[ci].yaxis.set_major_locator(mtick.MaxNLocator(nbins=3))
                if ci != self.n_clusters-1:
                    axs[ci].set_xticklabels([])
            axs[self.n_clusters-1].set_xlabel('Modulation Index \n (single neuron)')
            axs[self.n_clusters-1].set_ylabel('Density')
            ax.set_title(pe+' PE')
            ax2.set_title(f'MI>{mi_tol}')
            ax3.set_title(f'MI<-{mi_tol}')
            ax2.set_xlabel('Time from deviant (s)')
            ax3.set_xlabel('Time from deviant (s)')
            ax3.set_ylabel(r'$\Delta F/F$ (z-scored)')
            ax3.yaxis.set_label_position('right')
            hide_all_axis(ax)
            hide_all_axis(ax0)
            hide_all_axis(ax1)
        @show_resource_usage
        def plot_pred_mod_index_box_trial(ax, oddball, pe):
            # collect data.
            [_, [_, _, stim_seq, _], _, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_stim_labels,
                trial_idx=[l[oddball] for l in self.list_odd_idx],
                trial_param=[None, None, [0], None, [0], [0]],
                cate=cate, roi_id=None)
            c_idx = stim_seq.shape[0]//2
            neu_seq_fix, neu_seq_jitter = self.get_neu_seq_trial_fix_jitter(jitter_trial_mode, oddball, cate, isi_win)
            # define evaluation windows.
            if pe == 'pos':
                win_eval = [[-2500, 4000],
                            [stim_seq[c_idx,0]-100, stim_seq[c_idx,0]+500],
                            [stim_seq[c_idx+1,0]-100, stim_seq[c_idx+1,0]+500]]
            if pe == 'neg':
                win_eval = [[-2500, 4000],
                            [stim_seq[c_idx,1]+self.expect-500, stim_seq[c_idx,1]+self.expect],
                            [stim_seq[c_idx,1]+self.expect, stim_seq[c_idx,1]+self.expect+500]]
            # define layouts.
            ax.axis('off')
            ax0 = ax.inset_axes([0, 0.97, 0.6, 0.03], transform=ax.transAxes)
            ax = ax.inset_axes([0, 0, 1, 0.95], transform=ax.transAxes)
            axs = [ax.inset_axes([0.2, ci/self.n_clusters, 0.6, 0.7/self.n_clusters], transform=ax.transAxes)
                      for ci in range(self.n_clusters)]
            axs.reverse()
            # plot evaluation window.
            if oddball == 0:
                ax0.axvline(stim_seq[c_idx+1,0], color='red', lw=1, linestyle='--')
            if oddball == 1:
                ax0.axvline(stim_seq[c_idx,1]+self.expect, color='gold', lw=1, linestyle='--')
                ax0.axvline(stim_seq[c_idx+1,0], color='red', lw=1, linestyle='--')
            for i in range(stim_seq.shape[0]):
                ax0.fill_between(
                    stim_seq[i,:], 0, 1,
                    color=color0, edgecolor='none', alpha=0.25, step='mid')
            self.plot_win_mag_quant_win_eval(ax0, win_eval, color0, xlim, False)
            # plot results for each class.
            for ci in range(self.n_clusters):
                if np.sum(cluster_id==ci) > 0:
                    # get neurons within class.
                    neu_seq_ci_0 = [ns[:,dcid==ci,:] for ns, dcid in zip(neu_seq_fix, day_cluster_id)]
                    neu_seq_ci_1 = [ns[:,dcid==ci,:] for ns, dcid in zip(neu_seq_jitter, day_cluster_id)]
                    # average across neurons.
                    neu_seq_ci_0 = np.concatenate([np.nanmean(ns, axis=1) for ns in neu_seq_ci_0], axis=0)
                    neu_seq_ci_1 = np.concatenate([np.nanmean(ns, axis=1) for ns in neu_seq_ci_1], axis=0)
                    # compute index.
                    mi1 = get_modulation_index_neu_seq(neu_seq_ci_0, self.alignment['neu_time'], 0, win_eval)
                    mi2 = get_modulation_index_neu_seq(neu_seq_ci_1, self.alignment['neu_time'], 0, win_eval)
                    # plot results.
                    self.plot_pred_mod_index_dist(axs[ci], mi1, mi2, color0, color1, color2)
                    # adjust layouts.
                    axs[ci].xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
                    axs[ci].xaxis.set_major_locator(mtick.MaxNLocator(nbins=3))
                    axs[ci].xaxis.set_minor_locator(mtick.AutoMinorLocator(2))
                    axs[ci].tick_params(axis='x', which='minor', labelbottom=False)
                    axs[ci].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
                    axs[ci].yaxis.set_major_locator(mtick.MaxNLocator(nbins=3))
                    if ci != self.n_clusters-1:
                        axs[ci].set_xticklabels([])
            axs[self.n_clusters-1].set_xlabel('Modulation Index \n (single trial)')
            axs[self.n_clusters-1].set_ylabel('Density')
            hide_all_axis(ax)
            hide_all_axis(ax0)
        try: plot_pred_mod_index_box(axs[0], 1, 'neg')
        except: traceback.print_exc()
        try: plot_pred_mod_index_box(axs[1], 1, 'pos')
        except: traceback.print_exc()
        try: plot_pred_mod_index_box(axs[2], 0, 'pos')
        except: traceback.print_exc()
        try: plot_pred_mod_index_box_trial(axs[3], 1, 'neg')
        except: traceback.print_exc()
        try: plot_pred_mod_index_box_trial(axs[4], 1, 'pos')
        except: traceback.print_exc()
        try: plot_pred_mod_index_box_trial(axs[5], 0, 'pos')
        except: traceback.print_exc()
        
    
    def plot_cell_communication(self, axs, cate):
        color0 = 'dimgrey'
        color1 = 'deeppink'
        color2 = 'royalblue'
        _, cluster_id, neu_labels = get_cluster_cate(self.cluster_id, self.list_labels, cate)
        split_idx = get_split_idx(self.list_labels, cate)
        day_cluster_id = np.split(cluster_id, split_idx)
        day_neu_labels = np.split(neu_labels, split_idx)
        def plot_corr_time(ax, oddball):
            terget_cate = [-1,1,2]
            xlim = [-2500, 4000]
            # collect data.
            [_, [_, _, stim_seq, _], _, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_stim_labels,
                trial_idx=[l[oddball] for l in self.list_odd_idx],
                trial_param=[None, None, [0], None, [0], [0]],
                cate=cate, roi_id=None)
            c_idx = stim_seq.shape[0]//2
            neu_seq_fix, neu_seq_jitter = self.get_neu_seq_trial_fix_jitter('global', 1, cate, None)
            neu_seq_jitter = np.concatenate([np.nanmean(neu, axis=0) for neu in neu_seq_jitter], axis=0)
            neu_seq_fix = np.concatenate([np.nanmean(neu, axis=0) for neu in neu_seq_fix], axis=0)
            # get data within range.
            win = [-500, stim_seq[c_idx+1,1]+500]
            l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, win[0], win[1])
            neu_seq_fix = neu_seq_fix[:,l_idx:r_idx]
            neu_seq_jitter = neu_seq_jitter[:,l_idx:r_idx]
            neu_time = self.alignment['neu_time'][l_idx:r_idx]
            # split data into populations.
            list_neu_seq_fix = [neu_seq_fix[np.isin(neu_labels, [tc])] for tc in terget_cate]
            list_neu_seq_jitter = [neu_seq_jitter[np.isin(neu_labels, [tc])] for tc in terget_cate]
            # compute similarity across time.
            corr_fix = pair_corr_subsample(list_neu_seq_fix)
            corr_jitter = pair_corr_subsample(list_neu_seq_jitter)
            # define layouts.
            pairs = [(ci, cj) for ci, _ in enumerate(terget_cate)
                 for cj, _ in enumerate(terget_cate)
                 if ci > cj]
            w = (1-0.05*(len(pairs)+1))/len(pairs)
            axs = {}
            for k, (ci, cj) in enumerate(pairs):
                axs[(ci, cj)] = ax.inset_axes([0.05+k*(w+0.05), 0.15, (1-0.05*(len(pairs)+1))/len(pairs), 0.6])
            # plot results.
            for ci, _ in enumerate(terget_cate):
                for cj, _ in enumerate(terget_cate):
                    if ci <= cj: continue
                    axs[(ci, cj)] = axs[(ci, cj)]
                    m_0, s_0 = get_mean_sem(corr_fix[ci, cj, :, :].reshape(-1, corr_fix.shape[3]))
                    m_1, s_1 = get_mean_sem(corr_jitter[ci, cj, :, :].reshape(-1, corr_jitter.shape[3]))
                    # plot stimulus.
                    if oddball == 0:
                        axs[(ci, cj)].axvline(stim_seq[c_idx + 1, 0], color='red', lw=1, linestyle='--')
                    if oddball == 1:
                        axs[(ci, cj)].axvline(stim_seq[c_idx, 1] + self.expect, color='gold', lw=1, linestyle='--')
                        axs[(ci, cj)].axvline(stim_seq[c_idx + 1, 0], color='red', lw=1, linestyle='--')
                    for si in [0, 1]:
                        axs[(ci, cj)].fill_between(
                            stim_seq[c_idx + si, :], -0.03, 0.03,
                            color=color0, edgecolor='none', alpha=0.25, step='mid')
                    for si in [-1, 2]:
                        axs[(ci, cj)].fill_between(
                            stim_seq[c_idx + si, :], -0.03, 0.03,
                            color='none', edgecolor=color0, alpha=0.25, step='mid')
                    # plot traces.
                    self.plot_mean_sem(axs[(ci, cj)], neu_time, m_0, s_0, color1)
                    self.plot_mean_sem(axs[(ci, cj)], neu_time, m_1, s_1, color2)
                    # adjust layouts.
                    axs[(ci, cj)].spines['right'].set_visible(False)
                    axs[(ci, cj)].spines['top'].set_visible(False)
                    add_ax_ticks(axs[(ci, cj)], 'x', 8)
                    axs[(ci, cj)].set_xlim(xlim)
                    axs[(ci, cj)].set_ylim([-0.03, 0.03])
                    axs[(ci, cj)].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
                    axs[(ci, cj)].yaxis.set_major_locator(mtick.MaxNLocator(nbins=5))
                    if (ci, cj) != pairs[0]:
                        axs[(ci, cj)].set_yticks([])
                    # clear title for each subplot
                    axs[(ci, cj)].set_title(f'{self.label_names[str(terget_cate[ci])]} - {self.label_names[str(terget_cate[cj])]}')
            # shared labels on first subplot
            axs[pairs[0]].set_xlabel('Time from deviant (s)')
            axs[pairs[0]].set_ylabel('Correlation coefficient')
            hide_all_axis(ax)
        def plot_interaction_strength(ax):
            target_cate = [-1,1]
            xlim = [-500, 4000]
            # collect data.
            neu_seq_fix_odd_1, neu_seq_jitter_odd_1 = self.get_neu_seq_trial_fix_jitter('global', 1, cate, None)
            neu_seq_fix_odd_0, neu_seq_jitter_odd_0 = self.get_neu_seq_trial_fix_jitter('global', 0, cate, None)
            [_, [neu_seq_fix_standard, _, _, _, _],
             neu_labels, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_stim_labels,
                trial_param=[[2,3,4,5], None, [0], None, [0], [0]],
                mean_sem=False,
                cate=cate, roi_id=None)
            [_, [neu_seq_jitter_standard, _, _, _, _],
             neu_labels, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_stim_labels,
                trial_param=[[2,3,4,5], None, [1], None, [0], [0]],
                mean_sem=False,
                cate=cate, roi_id=None)
            [_, [_, _, stim_seq_standard, _],
             neu_labels, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_stim_labels,
                trial_param=[[2,3,4,5], None, [0], None, [0], [0]],
                cate=cate, roi_id=None)
            [_, [_, _, stim_seq_odd_1, _], _, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_stim_labels,
                trial_idx=[l[1] for l in self.list_odd_idx],
                trial_param=[None, None, [0], None, [0], [0]],
                cate=cate, roi_id=None)
            [_, [_, _, stim_seq_odd_0, _], _, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_stim_labels,
                trial_idx=[l[0] for l in self.list_odd_idx],
                trial_param=[None, None, [0], None, [0], [0]],
                cate=cate, roi_id=None)
            c_idx = stim_seq_standard.shape[0]//2
            # get data within range.
            r_standard = [stim_seq_standard[c_idx,0], stim_seq_standard[c_idx,1]+500]
            r_odd_1 = [stim_seq_odd_1[c_idx,1]+self.expect, stim_seq_odd_1[c_idx+1,1]+500]
            r_odd_0 = [stim_seq_odd_0[c_idx+1,0],stim_seq_odd_0[c_idx+1,1]+500]
            # filter cell types.
            neu_seq_fix_standard_x = []
            neu_seq_jitter_standard_x = []
            neu_seq_fix_odd_1_x = []
            neu_seq_jitter_odd_1_x = []
            neu_seq_fix_odd_0_x = []
            neu_seq_jitter_odd_0_x = []
            day_cluster_id_x = []
            day_neu_labels_x = []
            for nsfs, nsjs, nsfo_1, nsjo_1, nsfo_0, nsjo_0, cid, lbl in zip(
                    neu_seq_fix_standard, neu_seq_jitter_standard,
                    neu_seq_fix_odd_1,    neu_seq_jitter_odd_1,
                    neu_seq_fix_odd_0,    neu_seq_jitter_odd_0,
                    day_cluster_id, day_neu_labels):
                keep = (cid >= 0) & np.isin(lbl, [-1,1])
                if np.sum(keep) == 0: continue
                # standard.
                l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, r_standard[0], r_standard[1])
                neu_seq_fix_standard_x.append(nsfs[:,keep,l_idx:r_idx])
                neu_seq_jitter_standard_x.append(nsjs[:,keep,l_idx:r_idx])
                # long oddball.
                l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, r_odd_1[0], r_odd_1[1])
                neu_seq_fix_odd_1_x.append(nsfo_1[:,keep,l_idx:r_idx])
                neu_seq_jitter_odd_1_x.append(nsjo_1[:,keep,l_idx:r_idx])
                # short oddball.
                l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, r_odd_0[0], r_odd_0[1])
                neu_seq_fix_odd_0_x.append(nsfo_0[:,keep,l_idx:r_idx])
                neu_seq_jitter_odd_0_x.append(nsjo_0[:,keep,l_idx:r_idx])
                # labels.
                day_cluster_id_x.append(cid[keep])
                day_neu_labels_x.append(lbl[keep])
            # decoding two cell types.
            r2 = dict()
            r2['standard_fix']    = cate_2_multi_sess_regression_pop(neu_seq_fix_standard_x,    day_neu_labels_x, target_cate)
            r2['standard_jitter'] = cate_2_multi_sess_regression_pop(neu_seq_jitter_standard_x, day_neu_labels_x, target_cate)
            r2['fix_odd_1']       = cate_2_multi_sess_regression_pop(neu_seq_fix_odd_1_x,       day_neu_labels_x, target_cate)
            r2['jitter_odd_1']    = cate_2_multi_sess_regression_pop(neu_seq_jitter_odd_1_x,    day_neu_labels_x, target_cate)
            r2['fix_odd_0']       = cate_2_multi_sess_regression_pop(neu_seq_fix_odd_0_x,       day_neu_labels_x, target_cate)
            r2['jitter_odd_0']    = cate_2_multi_sess_regression_pop(neu_seq_jitter_odd_0_x,    day_neu_labels_x, target_cate)
            # find bounds.
            m = np.stack([np.nanmean(v[i].ravel()) for i in [0, 1] for v in r2.values()])
            upper = 0.5
            lower = -0.1
            # define layouts.
            ax.axis('off')
            ax0 = ax.inset_axes([0, 0, 0.1, 0.6], transform=ax.transAxes)
            axs_w = [ax.inset_axes([0.1+i*0.3, 0.9, 0.2, 0.1], transform=ax.transAxes) for i in range(3)]
            ax_xy = ax.inset_axes([0.1, 0, 0.4, 0.6], transform=ax.transAxes)
            ax_yx = ax.inset_axes([0.6, 0, 0.4, 0.6], transform=ax.transAxes)
            axs_xy = [ax_xy.inset_axes([0.1+i*0.3, 0, 0.2, 1], transform=ax_xy.transAxes) for i in range(3)]
            axs_yx = [ax_yx.inset_axes([0.1+i*0.3, 0, 0.2, 1], transform=ax_yx.transAxes) for i in range(3)]
            # plot evaluation window.
            for si in [0, 1, 2]:
                axs_w[0].fill_between(
                    stim_seq_standard[c_idx+si,:], 0, 1,
                    color=color0, edgecolor='none', alpha=0.25, step='mid')
            self.plot_win_mag_quant_win_eval(axs_w[0], [[], r_standard], color0, xlim, False)
            # plot evaluation window.
            for si in [0, 1, 2]:
                axs_w[0].fill_between(
                    stim_seq_standard[c_idx+si,:], 0, 1,
                    color=color0, edgecolor='none', alpha=0.25, step='mid')
            self.plot_win_mag_quant_win_eval(axs_w[0], [[], r_standard], color0, xlim, False)
            for si in [0, 1]:
                axs_w[1].fill_between(
                    stim_seq_odd_1[c_idx+si,:], 0, 1,
                    color=color0, edgecolor='none', alpha=0.25, step='mid')
                axs_w[1].axvline(stim_seq_odd_1[c_idx,1]+self.expect, color='gold', lw=1, linestyle='--')
                axs_w[1].axvline(stim_seq_odd_1[c_idx+1,0], color='red', lw=1, linestyle='--')
            self.plot_win_mag_quant_win_eval(axs_w[1], [[], r_odd_1], color0, xlim, False)
            for si in [0, 1]:
                axs_w[2].fill_between(
                    stim_seq_odd_0[c_idx+si,:], 0, 1,
                    color=color0, edgecolor='none', alpha=0.25, step='mid')
                axs_w[2].fill_between(
                    stim_seq_odd_0[c_idx+2,:], 0, 1,
                    color='none', edgecolor=color0, alpha=0.25, step='mid')
                axs_w[2].axvline(stim_seq_odd_0[c_idx+1,0], color='red', lw=1, linestyle='--')
            self.plot_win_mag_quant_win_eval(axs_w[2], [[], r_odd_0], color0, xlim, False)
            # plot results.
            for axs, direction_idx in [(axs_xy, 0), (axs_yx, 1)]:
                for ax, fix_key, jitter_key in zip(
                    axs, ['standard_fix', 'fix_odd_1', 'fix_odd_0'], ['standard_jitter', 'jitter_odd_1', 'jitter_odd_0']
                    ):
                    for pos, key, color in [(0, fix_key, color1), (1, jitter_key, color2)]:
                        r = r2[key][direction_idx].copy()
                        m, s = get_mean_sem(r.reshape(-1, 1), method_s='confidence interval')
                        # plot scatters.
                        ax.scatter(
                            np.ones_like(r)*pos+np.random.uniform(0,0.4, len(r))-0.2, r,
                            color=color, alpha=0.2, s=3)
                        # plot errorbar.
                        ax.errorbar(
                            pos, m, s, color=color0,
                            capsize=2, marker='o', linestyle='none', markeredgecolor='white', markeredgewidth=0.1)
                    # plot stat test.
                    #r = get_stat_test(r2[fix_key][direction_idx], r2[jitter_key][direction_idx], 'wilcoxon')[1]
                    #ax.text(0.5, upper + 0.3 * (upper - lower), self.stat_sym[r], ha='center', va='center')
            # adjust layouts.
            for axi in axs_w:
                add_ax_ticks(axi, 'x', 6)
            ax0.spines['bottom'].set_visible(False)
            ax0.spines['top'].set_visible(False)
            ax0.spines['right'].set_visible(False)
            ax0.set_xticks([])
            ax0.set_ylim([lower - 0.4*(upper-lower), upper + 0.4*(upper-lower)])
            ax0.set_ylabel('Predictive performance')
            ax_xy.set_title(f'{self.label_names[str(target_cate[0])]}\u2192{self.label_names[str(target_cate[1])]}')
            ax_yx.set_title(f'{self.label_names[str(target_cate[1])]}\u2192{self.label_names[str(target_cate[0])]}')
            for axss in [axs_xy, axs_yx]: 
                for i, axi in enumerate(axss):
                    axi.spines['left'].set_visible(False)
                    axi.spines['right'].set_visible(False)
                    axi.spines['top'].set_visible(False)
                    axi.set_xlim(-1,2)
                    axi.set_xlabel(['Standard', 'Long \n deviant', 'Short \n deviant'][i]),
                    axi.set_xticks([])
                    axi.set_ylim([lower - 0.4*(upper-lower), upper + 0.4*(upper-lower)])
                    axi.set_yticks([])
            hide_all_axis(ax_xy)
            hide_all_axis(ax_yx)
        # plot all.
        try: plot_corr_time(axs[0], 0)
        except: traceback.print_exc()
        try: plot_corr_time(axs[1], 1)
        except: traceback.print_exc()
        try: plot_interaction_strength(axs[2])
        except: traceback.print_exc()
        
    def plot_sorted_heatmaps_fix_all(self, axs, norm_mode, cate):
        xlim = [-2500, 5000]
        l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, xlim[0], xlim[1])
        # collect data.
        neu_x = []
        stim_x = []
        neu_time = self.alignment['neu_time'][l_idx:r_idx]
        # standard.
        _, [neu_seq, _, stim_seq, _], _, _ = get_neu_trial(
            self.alignment, self.list_labels, self.list_stim_labels,
            trial_param=[[2,3,4,5], None, [0], None, [0], [0]],
            cate=cate, roi_id=None)
        neu_seq = neu_seq[:,l_idx:r_idx]
        neu_x.append(neu_seq)
        stim_x.append(stim_seq)
        # short oddball.
        _, [neu_seq, _, stim_seq, _], _, _ = get_neu_trial(
            self.alignment, self.list_labels, self.list_stim_labels,
            trial_idx=[l[0] for l in self.list_odd_idx],
            trial_param=[None, None, [0], None, [0], [0]],
            cate=cate, roi_id=None)
        neu_seq = neu_seq[:,l_idx:r_idx]
        neu_x.append(neu_seq)
        stim_x.append(stim_seq)
        # long oddball.
        _, [neu_seq, _, stim_seq, _], _, _ = get_neu_trial(
            self.alignment, self.list_labels, self.list_stim_labels,
            trial_idx=[l[1] for l in self.list_odd_idx],
            trial_param=[None, None, [0], None, [0], [0]],
            cate=cate, roi_id=None)
        neu_seq = neu_seq[:,l_idx:r_idx]
        neu_x.append(neu_seq)
        stim_x.append(stim_seq)
        # define layouts.
        axs_hm = [axs[ai].inset_axes([0, 0, 0.7, 1], transform=axs[ai].transAxes) for ai in range(3)]
        axs_cb = [axs[ai].inset_axes([0.8, 0, 0.1, 1], transform=axs[ai].transAxes) for ai in range(3)]
        # plot heatmaps.
        for ai in range(3):
            self.plot_heatmap_neuron(
                axs_hm[ai], axs_cb[ai], neu_x[ai], neu_time, neu_x[0],
                norm_mode=norm_mode,
                neu_seq_share=neu_x)
            hide_all_axis(axs[ai])
        # add stimulus line.
        c_idx = stim_x[0].shape[0]//2
        for ai in range(3):
            xlines = [neu_time[np.searchsorted(neu_time, t)]
                      for t in [stim_x[ai][c_idx+i,0]
                                for i in [-2,-1,0,1,2]]]
            for xl in xlines:
                if xl>neu_time[0] and xl<neu_time[-1]:
                    axs_hm[ai].axvline(xl, color='black', lw=1, linestyle='--')
        axs_hm[1].axvline(stim_x[1][c_idx+1,0], color='gold', lw=1, linestyle='--')
        axs_hm[2].axvline(stim_x[2][c_idx,1]+self.expect, color='gold', lw=1, linestyle='--')
        # adjust layouts.
        axs_hm[0].set_xlabel('Time from stim onset (ms)')
        axs_hm[1].set_xlabel('Time from deviant (ms)')
        axs_hm[2].set_xlabel('Time from deviant (ms)')
        
    def plot_latent_all(self, axs, cate):
        xlim = [-1000, 3500]
        color0 = 'dimgrey'
        def plot_jitter_global_oddball(axs, oddball, fix_jitter):
            base_color1 = ['chocolate', 'crimson']
            base_color2 = ['deepskyblue', 'royalblue']
            # collect data.
            [_, [_, _, stim_seq, _], _, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_stim_labels,
                trial_idx=[l[oddball] for l in self.list_odd_idx],
                trial_param=[None, None, [0], None, [0], [0]],
                cate=cate, roi_id=None)
            c_idx = stim_seq.shape[0]//2
            neu_seq_0, neu_seq_1 = self.get_neu_seq_trial_fix_jitter('global', oddball, cate, None)
            neu_seq_0 = np.concatenate([np.nanmean(neu, axis=0) for neu in neu_seq_0], axis=0)
            neu_seq_1 = np.concatenate([np.nanmean(neu, axis=0) for neu in neu_seq_1], axis=0)
            c_stim = [color0] * stim_seq.shape[-2]
            # fit model.
            neu_x = np.concatenate([neu_seq_0, neu_seq_1], axis=1)
            model = PCA(n_components=self.d_latent)
            model.fit(neu_x.reshape(neu_x.shape[0],-1).T)
            neu_z_0 = model.transform(neu_seq_0.reshape(neu_seq_0.shape[0],-1).T).T.reshape(self.d_latent, -1)
            neu_z_1 = model.transform(neu_seq_1.reshape(neu_seq_1.shape[0],-1).T).T.reshape(self.d_latent, -1)
            # get data within range.
            l_idx, r_idx   = get_frame_idx_from_time(self.alignment['neu_time'], 0, stim_seq[c_idx,0]-500, stim_seq[c_idx+1,1]+500)
            neu_time = self.alignment['neu_time'][l_idx:r_idx]
            neu_seq_0 = neu_seq_0[:,l_idx:r_idx]
            neu_seq_1 = neu_seq_1[:,l_idx:r_idx]
            neu_z_0 = neu_z_0[:,l_idx:r_idx]
            neu_z_1 = neu_z_1[:,l_idx:r_idx]
            cmap_0, c_neu_0 = get_cmap_color(neu_seq_0.shape[1], base_color=['lemonchiffon']+base_color1+['black'], return_cmap=True)
            cmap_1, c_neu_1 = get_cmap_color(neu_seq_1.shape[1], base_color=['lemonchiffon']+base_color2+['black'], return_cmap=True)
            # random rotate dynamics.
            for ai in range(len(axs)):
                rm = get_random_rotate_mat_3d()
                axs[ai].axis('off')
                ax0 = axs[ai].inset_axes([0, 0, 0.6, 0.6], transform=axs[ai].transAxes, projection='3d')
                ax10 = axs[ai].inset_axes([0, 0.6, 0.5, 0.15], transform=axs[ai].transAxes)
                ax11 = axs[ai].inset_axes([0, 0.8, 0.5, 0.15], transform=axs[ai].transAxes)
                ax20 = axs[ai].inset_axes([0.6, 0.2, 0.1, 0.3], transform=axs[ai].transAxes)
                ax21 = axs[ai].inset_axes([0.8, 0.2, 0.1, 0.3], transform=axs[ai].transAxes)
                hide_all_axis(ax20)
                hide_all_axis(ax21)
                # fix.
                if 0 in fix_jitter:
                    # plot mean traces.
                    neu_mean_0, neu_sem_0 = get_mean_sem(neu_seq_0)
                    # find bounds.
                    upper = np.nanmax(neu_mean_0) + np.nanmax(neu_sem_0)
                    lower = np.nanmin(neu_mean_0) - np.nanmax(neu_sem_0)
                    # plot stimulus.
                    if oddball == 0:
                        ax10.axvline(stim_seq[c_idx + 1, 0], color='red', lw=1, linestyle='--')
                    if oddball == 1:
                        ax10.axvline(stim_seq[c_idx, 1] + self.expect, color='gold', lw=1, linestyle='--')
                        ax10.axvline(stim_seq[c_idx + 1, 0], color='red', lw=1, linestyle='--')
                    for si in [0, 1]:
                        ax10.fill_between(
                            stim_seq[c_idx + si, :], -1, 1,
                            color=color0, edgecolor='none', alpha=0.25, step='mid')
                    for si in [-1, 2]:
                        ax10.fill_between(
                            stim_seq[c_idx + si, :], -1, 1,
                            color='none', edgecolor=color0, alpha=0.25, step='mid')
                    # plot neural traces.
                    for t in range(neu_mean_0.shape[0] - 1):
                        ax10.plot(neu_time[t:t + 2], neu_mean_0[t:t + 2], color=c_neu_0[t])
                        ax10.fill_between(neu_time[t:t + 2], lower-(upper-lower)*0.1, lower, color=c_neu_0[t], linewidth=0)
                    ax10.scatter(neu_time[0], neu_mean_0[0], color='black', marker='x', lw=2)
                    ax10.scatter(neu_time[-1], neu_mean_0[-1], color='black', marker='o', lw=2)
                    # plot dynamics.
                    t_unexpect = [stim_seq[c_idx + 1, 0], stim_seq[c_idx, 1] + self.expect][oddball]
                    self.plot_3d_latent_dynamics(
                        ax0, np.matmul(rm, neu_z_0), stim_seq, neu_time,
                        end_color='black', c_stim=c_stim, cmap=cmap_0, add_mark=[(t_unexpect, 'gold')])
                    # adjust layouts.
                    add_ax_ticks(ax10, 'x', 6)
                    ax10.spines['left'].set_visible(False)
                    ax10.spines['right'].set_visible(False)
                    ax10.spines['top'].set_visible(False)
                    ax10.set_yticks([])
                    ax10.set_xlim(xlim)
                    ax10.set_ylim([lower - 0.1 * (upper - lower), upper + 0.1 * (upper - lower)])
                    add_heatmap_colorbar(
                        ax20, cmap_0, None, 'Time from deviant (s)',
                        [f'{(neu_time[0] + 0.2 * (neu_time[-1] - neu_time[0]))/1000:.1f}',
                         f'{(neu_time[-1] - 0.2 * (neu_time[-1] - neu_time[0]))/1000:.1f}'])
                # jitter.
                if 1 in fix_jitter:
                    # plot mean traces.
                    neu_mean_1, neu_sem_1 = get_mean_sem(neu_seq_1)
                    # find bounds.
                    upper = np.nanmax(neu_mean_1) + np.nanmax(neu_sem_1)
                    lower = np.nanmin(neu_mean_1) - np.nanmax(neu_sem_1)
                    # plot stimulus.
                    if oddball == 0:
                        ax11.axvline(stim_seq[c_idx + 1, 0], color='red', lw=1, linestyle='--')
                    if oddball == 1:
                        ax11.axvline(stim_seq[c_idx, 1] + self.expect, color='gold', lw=1, linestyle='--')
                        ax11.axvline(stim_seq[c_idx + 1, 0], color='red', lw=1, linestyle='--')
                    for si in [0, 1]:
                        ax11.fill_between(
                            stim_seq[c_idx + si, :], -1, 1,
                            color=color0, edgecolor='none', alpha=0.25, step='mid')
                    for si in [-1, 2]:
                        ax11.fill_between(
                            stim_seq[c_idx + si, :], -1, 1,
                            color='none', edgecolor=color0, alpha=0.25, step='mid')
                    # plot neural traces.
                    for t in range(neu_mean_1.shape[0] - 1):
                        ax11.plot(neu_time[t:t + 2], neu_mean_1[t:t + 2], color=c_neu_1[t])
                        ax11.fill_between(neu_time[t:t + 2], lower-(upper-lower)*0.1, lower, color=c_neu_1[t], linewidth=0)
                    ax11.scatter(neu_time[0], neu_mean_1[0], color='black', marker='x', lw=2)
                    ax11.scatter(neu_time[-1], neu_mean_1[-1], color='black', marker='o', lw=2)
                    # plot dynamics.
                    t_unexpect = [stim_seq[c_idx + 1, 0], stim_seq[c_idx, 1] + self.expect][oddball]
                    self.plot_3d_latent_dynamics(
                        ax0, np.matmul(rm, neu_z_1), stim_seq, neu_time,
                        end_color='black', c_stim=c_stim, cmap=cmap_1, add_mark=[(t_unexpect, 'gold')])
                    # adjust layouts.
                    add_ax_ticks(ax11, 'x', 6)
                    ax11.spines['left'].set_visible(False)
                    ax11.spines['right'].set_visible(False)
                    ax11.spines['top'].set_visible(False)
                    ax11.set_yticks([])
                    ax11.set_xlim(xlim)
                    ax11.set_ylim([lower - 0.1 * (upper - lower), upper + 0.1 * (upper - lower)])
                    add_heatmap_colorbar(
                        ax21, cmap_1, None, 'Time from deviant (s)',
                        [f'{(neu_time[0] + 0.2 * (neu_time[-1] - neu_time[0]))/1000:.1f}',
                         f'{(neu_time[-1] - 0.2 * (neu_time[-1] - neu_time[0]))/1000:.1f}'])
                adjust_layout_3d_latent(ax0)
        def plot_jitter_local_oddball(axs, oddball):
            base_color1 = ['coral', 'crimson']
            base_color2 = ['mediumseagreen', 'forestgreen']
            # collect data.
            [_, [neu_seq, stim_seq, camera_pupil, pre_isi, _], _, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_stim_labels,
                trial_idx=[l[oddball] for l in self.list_odd_idx],
                trial_param=[None, None, [1], None, [0], [0]],
                mean_sem=False,
                cate=cate, roi_id=None)
            # bin data based on isi.
            [bins, bin_center, _, bin_neu_seq, _, _, bin_stim_seq, bin_camera_pupil] = get_isi_bin_neu(
                neu_seq, stim_seq, camera_pupil, pre_isi, self.bin_win, self.bin_num)
            stim_seq = np.nanmean(np.concatenate(stim_seq, axis=0), axis=0)
            c_idx = stim_seq.shape[0]//2
            c_stim = [color0] * stim_seq.shape[-2]
        # plot all.
        try: plot_jitter_global_oddball(axs[0], 0, [0])
        except: traceback.print_exc()
        try: plot_jitter_global_oddball(axs[1], 0, [1])
        except: traceback.print_exc()
        try: plot_jitter_global_oddball(axs[2], 0, [0,1])
        except: traceback.print_exc()
        try: plot_jitter_global_oddball(axs[3], 1, [0])
        except: traceback.print_exc()
        try: plot_jitter_global_oddball(axs[4], 1, [1])
        except: traceback.print_exc()
        try: plot_jitter_global_oddball(axs[5], 1, [0,1])
        except: traceback.print_exc()
    
    def plot_excluded(self, axs, cate):
        color0 = 'dimgrey'
        color1 = 'deeppink'
        color2 = 'royalblue'
        color_model = 'plum'
        isi_win = 250
        xlim = [-2500, 4000]
        l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, xlim[0], xlim[1])
        _, cluster_id, neu_labels = get_cluster_cate(self.cluster_id, self.list_labels, cate)
        split_idx = get_split_idx(self.list_labels, cate)
        day_cluster_id = np.split(cluster_id, split_idx)
        @show_resource_usage
        def plot_ramp_cate_fraction(ax):
            colors = ['skyblue', 'peachpuff', 'palegreen']
            nl_excluded = neu_labels[np.isin(cluster_id, [-1])]
            frac_excluded = np.array([np.nansum(nl_excluded==ci)/np.nansum(neu_labels==ci) for ci in cate])
            ax.pie(
                frac_excluded/np.nansum(frac_excluded),
                labels=['Exc', 'VIP', 'SST'],
                colors=colors,
                autopct='%1.1f%%',
                wedgeprops={'linewidth': 1, 'edgecolor':'white', 'width':0.2})
            ax.set_title('Mixed')
        @show_resource_usage
        def plot_oddball_jitter(ax, oddball):
            # collect data.
            [_, [_, _, stim_seq, _], _, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_stim_labels,
                trial_idx=[l[oddball] for l in self.list_odd_idx],
                trial_param=[None, None, [0], None, [0], [0]],
                cate=cate, roi_id=None)
            c_idx = stim_seq.shape[0]//2
            neu_seq_fix, neu_seq_jitter = self.get_neu_seq_trial_fix_jitter('global', oddball, cate, isi_win)
            # get response within cluster.
            neu_seq_jitter = np.concatenate([np.nanmean(neu, axis=0) for neu in neu_seq_jitter], axis=0)
            neu_seq_fix = np.concatenate([np.nanmean(neu, axis=0) for neu in neu_seq_fix], axis=0)
            # get exclude cluster response.
            neu_fix_mean, neu_fix_sem = get_mean_sem_cluster(neu_seq_fix, 1, -cluster_id-1)
            neu_jitter_mean, neu_jitter_sem = get_mean_sem_cluster(neu_seq_jitter, 1, -cluster_id-1)
            norm_params = [get_norm01_params(
                np.concatenate([neu_fix_mean[ci,:], neu_jitter_mean[ci,:]]))
                for ci in range(1)]
            # define layouts.
            ax.axis('off')
            ax = ax.inset_axes([0, 0, 1, 0.95], transform=ax.transAxes)
            ax.axis('off')
            ax = ax.inset_axes([0, 0, 1, 1/self.n_clusters], transform=ax.transAxes)
            # plot results.
            if oddball == 0:
                ax.axvline(stim_seq[c_idx+1,0], color='red', lw=1, linestyle='--')
            if oddball == 1:
                ax.axvline(stim_seq[c_idx,1]+self.expect, color='gold', lw=1, linestyle='--')
                ax.axvline(stim_seq[c_idx+1,0], color='red', lw=1, linestyle='--')
            c_stim = [color0 if i in (c_idx, c_idx + 1) else f'empty{color0}' for i in range(stim_seq.shape[0])]
            self.plot_cluster_mean_sem(
                ax, neu_fix_mean, neu_fix_sem,
                self.alignment['neu_time'], norm_params,
                stim_seq,
                c_stim, [color1], xlim)
            self.plot_cluster_mean_sem(
                ax, neu_jitter_mean, neu_jitter_sem,
                self.alignment['neu_time'], norm_params,
                None, None, [color2], xlim,
                scale_bar=False)
            # adjust layouts.
            ax.set_xlabel('Time from deviant (s)')
        @show_resource_usage
        def plot_block_win_decode(ax, oddball):
            win_sample = 100
            # collect data.
            neu_seq_fix, neu_seq_jitter = self.get_neu_seq_trial_fix_jitter('global', oddball, cate, isi_win)
            [_, [_, _, stim_seq, _], _, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_stim_labels,
                trial_idx=[l[oddball] for l in self.list_odd_idx],
                trial_param=[None, None, [0], None, [0], [0]],
                cate=cate, roi_id=None)
            c_idx = stim_seq.shape[0]//2
            # run decoding for each class.
            results_all = []
            win_decode = [-500, stim_seq[c_idx+1,1]+500]
            # construct input data.
            neu_x = [
                [nsf[:,dci==-1,:] for nsf,dci in zip(neu_seq_fix, day_cluster_id)],
                [nsj[:,dci==-1,:] for nsj,dci in zip(neu_seq_jitter, day_cluster_id)]]
            # run decoding.
            r = multi_sess_decoding_slide_win(
                 neu_x, self.alignment['neu_time'],
                 win_decode, win_sample)
            results_all.append(r)
            # split results.
            decode_model_mean  = [results_all[0][1]]
            decode_model_sem   = [results_all[0][2]]
            decode_chance_mean = [results_all[0][3]]
            decode_chance_sem  = [results_all[0][4]]
            # define layouts.
            ax.axis('off')
            ax = ax.inset_axes([0, 0, 1, 0.95], transform=ax.transAxes)
            axs = [ax.inset_axes([0, 1, 0.8, 0.8/self.n_clusters], transform=ax.transAxes)]
            axs.reverse()
            # plot results for each class.
            for ci in range(1):
                # find bounds.
                upper = np.nanmax([decode_model_mean[ci], decode_chance_mean[ci]])
                lower = np.nanmin([decode_model_mean[ci], decode_chance_mean[ci]])
                # plot stimulus.
                c_idx = stim_seq.shape[0]//2
                if oddball == 0:
                    axs[ci].axvline(stim_seq[c_idx+1,0], color='red', lw=1, linestyle='--')
                if oddball == 1:
                    axs[ci].axvline(stim_seq[c_idx,1]+self.expect, color='gold', lw=1, linestyle='--')
                    axs[ci].axvline(stim_seq[c_idx+1,0], color='red', lw=1, linestyle='--')
                for si in [0,1]:
                    axs[ci].fill_between(
                        stim_seq[c_idx+si,:],
                        0, 1,
                        color=color0, edgecolor='none', alpha=0.25, step='mid')
                # plot decoding results.
                self.plot_mean_sem(
                    axs[ci],
                    results_all[ci][0], decode_chance_mean[ci], decode_chance_sem[ci],
                    color0, None)
                self.plot_mean_sem(
                    axs[ci],
                    results_all[ci][0], decode_model_mean[ci], decode_model_sem[ci],
                    color_model, None)
                # adjust layouts.
                axs[ci].set_xlim(xlim)
                axs[ci].set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
            for ci in range(1):     
                add_ax_ticks(axs[ci], 'x', 8)
                axs[ci].spines['right'].set_visible(False)
                axs[ci].spines['top'].set_visible(False)
                axs[ci].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
                axs[ci].yaxis.set_major_locator(mtick.MaxNLocator(nbins=2))
                axs[ci].set_ylim([0.4, 0.9])
            axs[0].set_xlabel('Time from deviant (s)')
            axs[0].set_ylabel('Decoding accuracy (cross validated) \n (fixed vs jittered)')
            hide_all_axis(ax)
        # plot all.
        try: plot_ramp_cate_fraction(axs[0])
        except: traceback.print_exc()
        try: plot_oddball_jitter(axs[1], 0)
        except: traceback.print_exc()
        try: plot_block_win_decode(axs[2], 0)
        except: traceback.print_exc()
        try: plot_oddball_jitter(axs[3], 1)
        except: traceback.print_exc()
        try: plot_block_win_decode(axs[4], 1)
        except: traceback.print_exc()
    
    def plot_pupil(self, axs):
        color0 = 'dimgrey'
        color1 = 'deeppink'
        color2 = 'royalblue'
        def plot_standard_global(ax):
            xlim = [-2000, 2500]
            l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, xlim[0], xlim[1])
            neu_time = self.alignment['neu_time'][l_idx:r_idx]
            # collect data.
            [_, [_, stim_seq_0, camera_pupil_0, _, _], _, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_stim_labels,
                trial_param=[[2,3,4,5], None, [0], None, [0], [0]],
                mean_sem=False,
                cate=[-1,1,2], roi_id=None) 
            [_, [_, stim_seq_1, camera_pupil_1, _, _], _, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_stim_labels,
                trial_param=[[2,3,4,5], None, [1], None, [0], [0]],
                mean_sem=False,
                cate=[-1,1,2], roi_id=None)
            stim_seq = np.nanmean(np.concatenate(stim_seq_0, axis=0), axis=0)
            c_idx = stim_seq.shape[0]//2
            # get data within range.
            camera_pupil_0 = np.concatenate(camera_pupil_0, axis=0)[:,l_idx:r_idx]
            camera_pupil_1 = np.concatenate(camera_pupil_1, axis=0)[:,l_idx:r_idx]
            m_0, s_0 = get_mean_sem(camera_pupil_0)
            m_1, s_1 = get_mean_sem(camera_pupil_1)
            # find bounds.
            upper = np.nanmax([m_0, m_1]) + np.nanmax([s_0, s_1])
            lower = np.nanmin([m_0, m_1]) - np.nanmax([s_0, s_1])
            # plot stimulus.
            ax.fill_between(
                stim_seq[c_idx,:],
                lower, upper,
                color=color0, edgecolor='none', alpha=0.25, step='mid')
            # plot traces.
            self.plot_mean_sem(ax, neu_time, m_0, s_0, color1)
            self.plot_mean_sem(ax, neu_time, m_1, s_1, color2)
            # adjust layouts.
            adjust_layout_pupil(ax)
            ax.set_xlim(xlim)
            ax.set_xlabel('Time from stim onset (ms)')
            ax.set_ylim([lower, upper])
        def plot_oddball_global(ax, oddball):
            xlim = [-2500, 4000]
            l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, xlim[0], xlim[1])
            neu_time = self.alignment['neu_time'][l_idx:r_idx]
            # collect data.
            [_, [_, stim_seq_0, camera_pupil_0, _, _], _, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_stim_labels,
                trial_idx=[l[oddball] for l in self.list_odd_idx],
                trial_param=[None, None, [0], None, [0], [0]],
                mean_sem=False,
                cate=[-1,1,2], roi_id=None)
            [_, [_, stim_seq_1, camera_pupil_1, _, _], _, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_stim_labels,
                trial_idx=[l[oddball] for l in self.list_odd_idx],
                trial_param=[None, None, [1], None, [0], [0]],
                mean_sem=False,
                cate=[-1,1,2], roi_id=None)
            stim_seq = np.nanmean(np.concatenate(stim_seq_0, axis=0), axis=0)
            c_idx = stim_seq.shape[0]//2
            # get data within range.
            camera_pupil_0 = np.concatenate(camera_pupil_0, axis=0)[:,l_idx:r_idx]
            camera_pupil_1 = np.concatenate(camera_pupil_1, axis=0)[:,l_idx:r_idx]
            m_0, s_0 = get_mean_sem(camera_pupil_0)
            m_1, s_1 = get_mean_sem(camera_pupil_1)
            # find bounds.
            upper = np.nanmax([m_0, m_1]) + np.nanmax([s_0, s_1])
            lower = np.nanmin([m_0, m_1]) - np.nanmax([s_0, s_1])
            # plot stimulus.
            if oddball == 0:
                ax.axvline(stim_seq[c_idx+1,0], color='red', lw=1, linestyle='--')
            if oddball == 1:
                ax.axvline(stim_seq[c_idx,1]+self.expect, color='gold', lw=1, linestyle='--')
                ax.axvline(stim_seq[c_idx+1,0], color='red', lw=1, linestyle='--')
            for si in range(stim_seq.shape[0]):
                if stim_seq[si,0] >= xlim[0] and stim_seq[si,1] <= xlim[1]:
                    ax.fill_between(
                        stim_seq[si,:],
                        lower, upper,
                        color=color0, edgecolor='none', alpha=0.25, step='mid')
            # plot traces.
            self.plot_mean_sem(ax, neu_time, m_0, s_0, color1)
            self.plot_mean_sem(ax, neu_time, m_1, s_1, color2)
            # adjust layouts.
            adjust_layout_pupil(ax)
            ax.set_xlim(xlim)
            ax.set_xlabel('Time from deviant (ms)')
            ax.set_ylim([lower, upper])
        # plot all.
        try: plot_standard_global(axs[0])
        except: traceback.print_exc()
        try: plot_oddball_global(axs[1], 0)
        except: traceback.print_exc()
        try: plot_oddball_global(axs[2], 1)
        except: traceback.print_exc()
            
# colors = ['#989A9C', '#A4CB9E', '#9DB4CE', '#EDA1A4', '#F9C08A']
class plotter_main(plotter_utils):
    def __init__(self, neural_trials, labels, label_names, temp_folder, cate_list):
        super().__init__(neural_trials, labels, temp_folder, cate_list)
        self.label_names = label_names

    def cluster_oddball_fix_all(self, axs_all):
        for cate, axs in zip(self.cate_list, axs_all):
            try:
                label_name = self.label_names[str(cate[0])] if len(cate)==1 else 'all'
                print(f'plotting results for {label_name}')
                
                self.plot_cluster_oddball_fix_all(axs, cate=cate)

            except: traceback.print_exc()
            
    def cluster_oddball_jitter_global_all(self, axs_all):
        for cate, axs in zip(self.cate_list, axs_all):
            try:
                label_name = self.label_names[str(cate[0])] if len(cate)==1 else 'all'
                print(f'plotting results for {label_name}')
                
                self.plot_cluster_oddball_jitter_global_all(axs, 'global', cate=cate)

            except: traceback.print_exc()
    
    def cluster_oddball_jitter_local_all(self, axs_all):
        for cate, axs in zip(self.cate_list, axs_all):
            try:
                label_name = self.label_names[str(cate[0])] if len(cate)==1 else 'all'
                print(f'plotting results for {label_name}')
                
                self.plot_cluster_oddball_jitter_local_all(axs, 'global', cate=cate)
                
            except: traceback.print_exc()
    
    def cluster_oddball_jitter_global_modulation(self, axs_all):
        for cate, axs in zip(self.cate_list, axs_all):
            try:
                label_name = self.label_names[str(cate[0])] if len(cate)==1 else 'all'
                print(f'plotting results for {label_name}')
                
                self.plot_cluster_oddball_jitter_global_modulation(axs, cate=cate)

            except: traceback.print_exc()
    
    def cell_communication(self, axs_all):
        for cate, axs in zip(self.cate_list, axs_all):
            try:
                label_name = self.label_names[str(cate[0])] if len(cate)==1 else 'all'
                print(f'plotting results for {label_name}')
                
                self.plot_cell_communication(axs, cate=cate)

            except: traceback.print_exc()
        
    def sorted_heatmaps_fix_all(self, axs_all):
        for cate, axs in zip(self.cate_list, axs_all):
            try:
                label_name = self.label_names[str(cate[0])] if len(cate)==1 else 'all'
                print(f'plotting results for {label_name}')
                
                for axss, norm_mode in zip([axs[0:3], axs[3:6], axs[6:9]], ['none', 'minmax', 'share']):
                    self.plot_sorted_heatmaps_fix_all(axss, norm_mode, cate=cate)

            except: traceback.print_exc()
    
    def latent_all(self, axs_all):
        for cate, axs in zip(self.cate_list, axs_all):
            try:
                label_name = self.label_names[str(cate[0])] if len(cate)==1 else 'all'
                print(f'plotting results for {label_name}')
                
                self.plot_latent_all(axs, cate=cate)

            except: traceback.print_exc()
    
    def excluded(self, axs_all):
        for cate, axs in zip(self.cate_list, axs_all):
            try:
                label_name = self.label_names[str(cate[0])] if len(cate)==1 else 'all'
                print(f'plotting results for {label_name}')

                self.plot_excluded(axs, cate=cate)

            except: traceback.print_exc()

    def pupil(self, axs_all):
        try:

            self.plot_pupil(axs_all)

        except: traceback.print_exc()
'''
from cebra.integrations.sklearn.metrics import consistency_score
# compute trial pairwise consistency score between latent dynamics.
def get_latent_consistency(z0, z1):
    s = np.zeros([z0.shape[0],z1.shape[0]])
    for t0 in range(z0.shape[0]):
        for t1 in range(z1.shape[0]):
            s[t0,t1] = np.nanmean(consistency_score(embeddings=[z0[t0].T, z1[t0].T], between='runs')[0])
    return s
'''