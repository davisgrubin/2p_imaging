#!/usr/bin/env python3

import numpy as np
from sklearn.decomposition import PCA

from modules.Alignment import run_get_stim_response
from modeling.clustering import get_mean_sem_cluster
from modeling.clustering import get_bin_mean_sem_cluster
from modeling.clustering import get_cluster_cate
from modeling.decoding import neu_pop_sample_decoding_slide_win
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
from utils import get_split_idx
from utils import get_block_1st_idx
from utils import get_cmap_color
from utils import adjust_layout_neu
from utils import adjust_layout_3d_latent
from utils import add_legend
from utils import utils_basic

# fig, ax = plt.subplots(1, 1, figsize=(5, 5))
# fig, ax = plt.subplots(1, 1, figsize=(3, 9))
# fig, axs = plt.subplots(1, 6, figsize=(18, 3))
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
        self.n_clusters = 6
        self.max_clusters = 10
        self.d_latent = 3
        self.glm = self.run_glm()
        self.cluster_id = self.run_clustering()

    def get_neu_seq_trial_fix_jitter(self, mode, oddball, cate, isi_win):
        # jitter oddball.  
        [_, [neu_seq_jitter, _, _, pre_isi, _], _, _] = get_neu_trial(
            self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
            trial_idx=[l[oddball] for l in self.list_odd_idx],
            trial_param=[None, None, [1], None, [0], [0]],
            mean_sem=False,
            cate=cate, roi_id=None)
        # fix oddball.
        [_, [neu_seq_fix, _, _, _, _], _, _] = get_neu_trial(
            self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
            trial_idx=[l[oddball] for l in self.list_odd_idx],
            trial_param=[None, None, [0], None, [0], [0]],
            mean_sem=False,
            cate=cate, roi_id=None)
        if mode == 'global':
            pass
        if mode == 'common':
            # subsampling.
            n_samples = 50
            idx_jitter = [(isi>np.nanmean(isi)-isi_win)*(isi<np.nanmean(isi)+isi_win) for isi in pre_isi]
            neu_seq_jitter = [neu_seq_jitter[i][idx_jitter[i],:,:] for i in range(self.n_sess)]
            neu_seq_fix = [
                np.concatenate([np.nanmean(nsf[np.random.choice(nsf.shape[0], n_samples, replace=True)], axis=0)[None, ...]
                                for _ in range(nsj.shape[0])], axis=0)
                for nsj, nsf in zip(neu_seq_jitter, neu_seq_fix)]
        return neu_seq_fix, neu_seq_jitter

    def plot_cluster_oddball_fix_all(self, axs, cate):
        xlim = [-2500, 4000]
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
            neu_mean, neu_sem = get_mean_sem_cluster(neu_seq, self.n_clusters, cluster_id)
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
            neu_mean, neu_sem = get_mean_sem_cluster(neu_seq, self.n_clusters, cluster_id)
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
        except Exception as e: print(e)
        try: plot_standard_fix(axs[1])
        except Exception as e: print(e)
        try: plot_oddball_fix(axs[2], 0)
        except Exception as e: print(e)
        try: plot_oddball_fix(axs[3], 1)
        except Exception as e: print(e)
        try: plot_neu_fraction(axs[4])
        except Exception as e: print(e)
        try: plot_cate_fraction(axs[5])
        except Exception as e: print(e)
        try: plot_legend(axs[6])
        except Exception as e: print(e)
    
    def plot_cluster_oddball_jitter_common_all(self, axs, cate):
        isi_win = 250
        xlim = [-2500, 4000]
        kernel_all = get_glm_cate(self.glm, self.list_labels, self.list_significance, cate)
        cluster_id, neu_labels = get_cluster_cate(self.cluster_id, self.list_labels, self.list_significance, cate)
        split_idx = get_split_idx(self.list_labels, self.list_significance, cate)
        day_cluster_id = np.split(cluster_id, split_idx)
        def plot_glm_kernel(ax):
            ax.axis('off')
            ax = ax.inset_axes([0, 0, 0.5, 0.95], transform=ax.transAxes)
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
            # adjust layout.
            ax.set_xlabel('time since stim (ms)')
        def plot_random_bin(ax, mode):
            ax.axis('off')
            ax = ax.inset_axes([0, 0, 1, 0.95], transform=ax.transAxes)
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
                xl = [-3500, 1500]
                isi = pre_isi
                isi_idx_offset = -1
            if mode == 'post':
                xl = [-1000, 4000]
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
                    [colors[bi]], [colors[bi]]*self.n_clusters, xl)
            # adjust layout.
            ax.set_xlabel('time since stim (ms)')
        def plot_oddball_jitter(ax, oddball):
            ax.axis('off')
            ax = ax.inset_axes([0, 0, 1, 0.95], transform=ax.transAxes)
            # collect data.
            [[color0, color1, color2, _],
             [_, _, stim_seq, _], _, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_idx=[l[oddball] for l in self.list_odd_idx],
                trial_param=[None, None, [0], None, [0], [0]],
                cate=cate, roi_id=None)
            neu_seq_fix, neu_seq_jitter = self.get_neu_seq_trial_fix_jitter('common', oddball, cate, isi_win)
            neu_seq_jitter = np.concatenate([np.nanmean(neu, axis=0) for neu in neu_seq_jitter], axis=0)
            neu_seq_fix = np.concatenate([np.nanmean(neu, axis=0) for neu in neu_seq_fix], axis=0)
            # get response within cluster at each bin.
            neu_fix_mean, neu_fix_sem = get_mean_sem_cluster(neu_seq_fix, self.n_clusters, cluster_id)
            neu_jitter_mean, neu_jitter_sem = get_mean_sem_cluster(neu_seq_jitter, self.n_clusters, cluster_id)
            norm_params = [get_norm01_params(
                np.concatenate([neu_fix_mean[i,:], neu_jitter_mean[i,:]]))
                for i in range(self.n_clusters)]
            # plot results.
            c_idx = stim_seq.shape[0]//2
            if oddball == 0:
                ax.axvline(stim_seq[c_idx+1,0], color='gold', lw=1, linestyle='--')
            if oddball == 1:
                ax.axvline(stim_seq[c_idx,1]+self.expect, color='gold', lw=1, linestyle='--')
            self.plot_cluster_mean_sem(
                ax, neu_fix_mean, neu_fix_sem,
                self.alignment['neu_time'], norm_params,
                stim_seq,
                [color0]*stim_seq.shape[0], [color0]*self.n_clusters, xlim)
            self.plot_cluster_mean_sem(
                ax, neu_jitter_mean, neu_jitter_sem,
                self.alignment['neu_time'], norm_params,
                None, None, [[color1,color2][oddball]]*self.n_clusters, xlim)
            # adjust layout.
            ax.set_xlabel('time since pre oddball stim (ms)')
        def plot_win_mag_quant_stat(ax, oddball):
            offset = [0, 0.1]
            ax.axis('off')
            ax0 = ax.inset_axes([0, 0.97, 0.5, 0.03], transform=ax.transAxes)
            ax1 = ax.inset_axes([0, 0, 1, 0.95], transform=ax.transAxes)
            ax1.axis('off')
            ax_box = [ax1.inset_axes([0, ci/self.n_clusters, 0.5, 0.75/self.n_clusters], transform=ax1.transAxes)
                      for ci in range(self.n_clusters)]
            axs_stat = [ax1.inset_axes([0, (0.8+ci)/self.n_clusters, 0.5, 0.1/self.n_clusters], transform=ax1.transAxes)
                   for ci in range(self.n_clusters)]
            # collect data.
            neu_seq_fix, neu_seq_jitter = self.get_neu_seq_trial_fix_jitter('common', oddball, cate, isi_win)
            [[color0, color1, color2, _],
             [_, _, stim_seq, _], _, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
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
            self.plot_cluster_win_mag_quant(
                ax_box, day_cluster_id, neu_seq_fix, self.alignment['neu_time'],
                win_eval, color0, 0, offset[0])
            self.plot_cluster_win_mag_quant(
                ax_box, day_cluster_id, neu_seq_jitter, self.alignment['neu_time'],
                win_eval, [color1,color2][oddball], 0, offset[1])
            # plot statistics test.
            self.plot_cluster_win_mag_quant_stat(
                axs_stat, day_cluster_id, neu_seq_fix, neu_seq_jitter, self.alignment['neu_time'],
                win_eval, 0)
            # adjust layout.
            ax0.set_xticks([])
        def plot_win_decode(ax, oddball):
            win_decode = [-2500, 4000]
            win_sample = 200
            win_step = 1
            ax.axis('off')
            ax = ax.inset_axes([0, 0, 0.5, 0.95], transform=ax.transAxes)
            ax.axis('off')
            axs = [ax.inset_axes([0, ci/self.n_clusters, 1, 0.8/self.n_clusters], transform=ax.transAxes)
                      for ci in range(self.n_clusters)]
            # collect data.
            neu_seq_fix, neu_seq_jitter = self.get_neu_seq_trial_fix_jitter('common', oddball, cate, isi_win)
            [[color0, color1, color2, _],
             [_, _, stim_seq, _], _, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_idx=[l[oddball] for l in self.list_odd_idx],
                trial_param=[None, None, [0], None, [0], [0]],
                cate=cate, roi_id=None)
            results_all = []
            # run decoding for each class.
            for ci in range(self.n_clusters):
                if np.sum(cluster_id==ci) > 0:
                    neu_x = [
                        [nsf[:,dci==ci,:] for nsf,dci in zip(neu_seq_fix, day_cluster_id)],
                        [nsj[:,dci==ci,:] for nsj,dci in zip(neu_seq_jitter, day_cluster_id)]]
                    r = multi_sess_decoding_slide_win(
                         neu_x, self.alignment['neu_time'],
                         win_decode, win_sample, win_step)
                    results_all.append(r)
            # split results.
            decode_time = results_all[0][0]
            decode_model_mean  = [results_all[ci][1] for ci in range(self.n_clusters)]
            decode_model_sem   = [results_all[ci][2] for ci in range(self.n_clusters)]
            decode_chance_mean = [results_all[ci][3] for ci in range(self.n_clusters)]
            decode_chance_sem  = [results_all[ci][4] for ci in range(self.n_clusters)]
            # plot results for each class.
            for ci in range(self.n_clusters):
                # find bounds.
                upper = np.nanmax([decode_model_mean[ci], decode_chance_mean[ci]])
                lower = np.nanmin([decode_model_mean[ci], decode_chance_mean[ci]])
                # plot stimulus.
                c_idx = stim_seq.shape[0]//2
                if oddball == 0:
                    axs[ci].axvline(stim_seq[c_idx+1,0], color='gold', lw=1, linestyle='--')
                if oddball == 1:
                    axs[ci].axvline(stim_seq[c_idx,1]+self.expect, color='gold', lw=1, linestyle='--')
                for i in range(stim_seq.shape[0]):
                    axs[ci].fill_between(
                        stim_seq[i,:],
                        lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
                        color=color0, edgecolor='none', alpha=0.25, step='mid')
                # plot decoding results.
                self.plot_mean_sem(axs[ci], decode_time, decode_model_mean[ci], decode_model_sem[ci], color2, None)
                self.plot_mean_sem(axs[ci], decode_time, decode_chance_mean[ci], decode_chance_sem[ci], color0, None)
                # adjust layout.
                axs[ci].tick_params(tick1On=False)
                axs[ci].spines['right'].set_visible(False)
                axs[ci].spines['top'].set_visible(False)
                axs[ci].set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
                if ci>0:
                    axs[ci].set_xticks([])
            axs[0].set_xlabel('time since pre oddball stim (ms)')
        def plot_neu_fraction(ax):
            ax.axis('off')
            ax = ax.inset_axes([0, 0, 1, 0.95], transform=ax.transAxes)
            # collect data.
            [[_, _, color2, _], _, _, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_param=[None, None, [0], None, [0], [0]],
                cate=cate, roi_id=None)
            # plot results.
            self.plot_cluster_neu_fraction_in_cluster(ax, cluster_id, color2)
        def plot_cate_fraction(ax):
            ax.axis('off')
            ax = ax.inset_axes([0, 0, 1, 0.95], transform=ax.transAxes)
            self.plot_cluster_cate_fraction_in_cluster(ax, cluster_id, neu_labels, self.label_names)
        def plot_legend(ax):
            [[color0, color1, color2, _], _, _,
             [n_trials, n_neurons]] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_param=[[-1], None, [0], None, [0], [0]],
                cate=cate, roi_id=None)
            cs = [color0, color1, color2, 'gold']
            lbl = ['fix', 'short oddball', 'long oddball', 'unexpected event']
            add_legend(ax, cs, lbl, n_trials, n_neurons, self.n_sess, 'upper right')
            ax.axis('off')
        # plot all.
        try: plot_glm_kernel(axs[0])
        except Exception as e: print(e)
        try: plot_random_bin(axs[1], 'pre')
        except Exception as e: print(e)
        try: plot_random_bin(axs[2], 'post')
        except Exception as e: print(e)
        try: plot_oddball_jitter(axs[3], 0)
        except Exception as e: print(e)
        try: plot_win_mag_quant_stat(axs[4], 0)
        except Exception as e: print(e)
        try: plot_win_decode(axs[5], 0)
        except Exception as e: print(e)
        try: plot_oddball_jitter(axs[6], 1)
        except Exception as e: print(e)
        try: plot_win_mag_quant_stat(axs[7], 1)
        except Exception as e: print(e)
        try: plot_win_decode(axs[8], 1)
        except Exception as e: print(e)
        try: plot_neu_fraction(axs[9])
        except Exception as e: print(e)
        try: plot_cate_fraction(axs[10])
        except Exception as e: print(e)
        try: plot_legend(axs[11])
        except Exception as e: print(e)
    
    def plot_cluster_oddball_fix_heatmap_all(self, axs, cate):
        xlim = [-2500, 4000]
        l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, xlim[0], xlim[1])
        kernel_all = get_glm_cate(self.glm, self.list_labels, self.list_significance, cate)
        cluster_id, neu_labels = get_cluster_cate(self.cluster_id, self.list_labels, self.list_significance, cate)
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
            # adjust layout.
            ax.set_xlabel('time since stim (ms)')
            ax.axvline(stim_seq[c_idx,0], color='black', lw=1, linestyle='--')
        def plot_standard_fix(ax):
            # collect data.
            [[_, _, _, cmap],
             [neu_seq, _, stim_seq, _], _, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_param=[[2,3,4,5], None, [0], None, [0], [0]],
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
            # adjust layout.
            ax.set_xlabel('time since stim (ms)')
        def plot_oddball_fix(ax, oddball):
            # collect data.
            [[_, _, _, cmap],
             [neu_seq, _, stim_seq, _], _, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_idx=[l[oddball] for l in self.list_odd_idx],
                trial_param=[None, None, [0], None, [0], [0]],
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
            if oddball == 0:
                ax.axvline(stim_seq[c_idx+1,0], color='gold', lw=1, linestyle='--')
            if oddball == 1:
                ax.axvline(stim_seq[c_idx,1]+self.expect, color='gold', lw=1, linestyle='--')
            # adjust layout.
            ax.set_xlabel('time since pre oddball stim (ms)')
        # plot all.
        try: plot_cluster_features(axs[0])
        except Exception as e: print(e)
        try: plot_hierarchical_dendrogram(axs[1])
        except Exception as e: print(e)
        try: plot_glm_kernel(axs[2])
        except Exception as e: print(e)
        try: plot_standard_fix(axs[3])
        except Exception as e: print(e)
        try: plot_oddball_fix(axs[4], 0)
        except Exception as e: print(e)
        try: plot_oddball_fix(axs[5], 1)
        except Exception as e: print(e)
        
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
        xlim = [-2500, 4000]
        lbl = ['cluster #'+str(ci) for ci in range(self.n_clusters)]
        cluster_id, neu_labels = get_cluster_cate(self.cluster_id, self.list_labels, self.list_significance, cate)
        def plot_standard_fix(axs):
            # collect data.
            [[color0, color1, color2, _],
             [neu_seq, _, stim_seq, _],
             [neu_labels, neu_sig], _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_param=[[2,3,4,5], None, [0], None, [0], [0]],
                cate=cate, roi_id=None)
            # get response within cluster at each bin.
            neu_mean, neu_sem = get_mean_sem_cluster(neu_seq, self.n_clusters, cluster_id)
            # plot results for each class.
            for ci in range(self.n_clusters):
                if np.sum(cluster_id==ci) > 0:
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
            neu_mean, neu_sem = get_mean_sem_cluster(neu_seq, self.n_clusters, cluster_id)
            # plot results for each class.
            for ci in range(self.n_clusters):
                if np.sum(cluster_id==ci) > 0:
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
                if np.sum(cluster_id==ci) > 0:
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
        def plot_oddball_fix_index(axs):
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
                if np.sum(cluster_id==ci) > 0:
                    axs[ci].axis('off')
                    axs[ci] = axs[ci].inset_axes([0, 0, 0.6, 1], transform=axs[ci].transAxes)
                    quant_all = [
                        run_quantification(ns[cluster_id==ci,:], self.alignment['neu_time'], ss[c_idx+1,0])
                        for ns, ss in zip(
                            [neu_seq_standard, neu_seq_short, neu_seq_long],
                            [stim_seq_standard, stim_seq_short, stim_seq_long])]
                    evoke_standard, evoke_short, evoke_long = [np.abs(quant_all[di]['evoke_mag']) for di in range(3)]
                    ori_short = (evoke_short - evoke_standard) / (evoke_short + evoke_standard + 1e-5)
                    ori_long = (evoke_long - evoke_standard) / (evoke_long + evoke_standard + 1e-5)
                    # plot errorbar.
                    axs[ci].scatter(
                        0*np.ones_like(ori_short), ori_short, None,
                        color=color1, alpha=0.5)
                    axs[ci].scatter(
                        1*np.ones_like(ori_long), ori_long, None,
                        color=color2, alpha=0.5)
                    # adjust layout.
                    axs[ci].tick_params(axis='y', tick1On=False)
                    axs[ci].tick_params(axis='x', labelrotation=90)
                    axs[ci].spines['right'].set_visible(False)
                    axs[ci].spines['top'].set_visible(False)
                    axs[ci].set_xlim([-0.5, 1.5])
                    axs[ci].set_ylim([-0.5, 0.5])
                    axs[ci].set_xticks([0,1])
                    axs[ci].set_xticklabels(['short', 'long'], rotation='vertical')
                    axs[ci].set_xlabel('interval condition')
                    axs[ci].set_ylabel('oddball response index')
                    axs[ci].set_title(lbl[ci])
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
        except Exception as e: print(e)
        try: plot_oddball_fix(axs[1], 0)
        except Exception as e: print(e)
        try: plot_oddball_fix(axs[2], 1)
        except Exception as e: print(e)
        try: plot_oddball_fix_quant(axs[3])
        except Exception as e: print(e)
        try: plot_oddball_fix_index(axs[4])
        except Exception as e: print(e)
        try: plot_legend(axs[5])
        except Exception as e: print(e)
    
    def plot_cluster_oddball_jitter_individual(self, axs, mode, cate):
        isi_win = 150
        xlim = [-2500, 4000]
        l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, xlim[0], xlim[1])
        lbl = ['cluster #'+str(ci) for ci in range(self.n_clusters)]
        cluster_id, neu_labels = get_cluster_cate(self.cluster_id, self.list_labels, self.list_significance, cate)
        split_idx = get_split_idx(self.list_labels, self.list_significance, cate)
        day_cluster_id = np.split(cluster_id, split_idx)
        def plot_oddball_jitter(axs, oddball):
            # collect data.
            neu_seq_fix, neu_seq_jitter = self.get_neu_seq_trial_fix_jitter(mode, oddball, cate, isi_win)
            neu_seq_jitter = np.concatenate([np.nanmean(neu, axis=0) for neu in neu_seq_jitter], axis=0)
            neu_seq_fix    = np.concatenate([np.nanmean(neu, axis=0) for neu in neu_seq_fix], axis=0)
            [[color0, color1, color2, _],
             [_, _, stim_seq, _], _, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_idx=[l[oddball] for l in self.list_odd_idx],
                trial_param=[None, None, [0], None, [0], [0]],
                cate=cate, roi_id=None)
            c_idx = stim_seq.shape[0]//2
            # plot results for each class.
            for ci in range(self.n_clusters):
                if np.sum(cluster_id==ci) > 0:
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
                    if oddball == 0:
                        axs[ci].axvline(stim_seq[c_idx+1,0], color='gold', lw=1, linestyle='--')
                    if oddball == 1:
                        axs[ci].axvline(stim_seq[c_idx,1]+self.expect, color='gold', lw=1, linestyle='--')
                    for i in range(stim_seq.shape[0]):
                        axs[ci].fill_between(
                            stim_seq[i,:],
                            lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
                            color=color0, edgecolor='none', alpha=0.25, step='mid')
                    # plot neural traces.
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
        def plot_trial_quant(axs, oddball):
            offset = [0, 0.1]
            # collect data.
            neu_seq_fix, neu_seq_jitter = self.get_neu_seq_trial_fix_jitter(mode, oddball, cate, isi_win)
            [[color0, color1, color2, _],
             [_, _, stim_seq, _], _, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
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
            # plot results for each class.
            for ci in range(self.n_clusters):
                if np.sum(cluster_id==ci) > 0:
                    axs[ci].axis('off')
                    ax0 = axs[ci].inset_axes([0, 0.9, 1, 0.1], transform=axs[ci].transAxes)
                    ax1 = axs[ci].inset_axes([0, 0, 1, 0.7], transform=axs[ci].transAxes)
                    # average across neurons within cluster.
                    neu_ci_fix = np.concatenate(
                        [np.nanmean(neu[:,sci==ci,:],axis=1)
                         for neu,sci in zip(neu_seq_fix,day_cluster_id)], axis=0)
                    neu_ci_jitter = np.concatenate(
                        [np.nanmean(neu[:,sci==ci,:],axis=1)
                         for neu,sci in zip(neu_seq_jitter,day_cluster_id)], axis=0)
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
                    self.plot_win_mag_quant(
                        ax1, neu_ci_fix, self.alignment['neu_time'],
                        win_eval, color0, 0, offset[0])
                    self.plot_win_mag_quant(
                        ax1, neu_ci_jitter, self.alignment['neu_time'],
                        win_eval, [color1, color2][oddball], 0, offset[1])
        def plot_legend(ax):
            [[color0, color1, color2, _], _, _,
             [n_trials, n_neurons]] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_param=[[-1], None, [1], None, [0], [0]],
                cate=cate, roi_id=None)
            cs = [color0, color1, color2, color1, color2]
            lbl = ['fix', 'jitter short oddball', 'jitter long oddball',
                   r'jitter within $\pm{}$ ms'.format(isi_win), r'jitter within $\pm{}$ ms'.format(isi_win)]
            add_legend(ax, cs, lbl, n_trials, n_neurons, self.n_sess, 'upper right')
            ax.axis('off')
        # plot all.
        try: plot_oddball_jitter(axs[0], 0)
        except Exception as e: print(e)
        try: plot_trial_quant(axs[1], 0)
        except Exception as e: print(e)
        try: plot_oddball_jitter(axs[2], 1)
        except Exception as e: print(e)
        try: plot_trial_quant(axs[3], 1)
        except Exception as e: print(e)
        try: plot_legend(axs[4])
        except Exception as e: print(e)
    
    def plot_cluster_oddball_jitter_local_individual(self, axs, cate):
        xlim = [-2500, 4000]
        lbl = ['cluster #'+str(ci) for ci in range(self.n_clusters)]
        cluster_id, neu_labels = get_cluster_cate(self.cluster_id, self.list_labels, self.list_significance, cate)
        def plot_oddball_jitter(axs, oddball):
            # collect data.
            [[color0, color1, color2, _],
             [neu_seq, stim_seq, camera_pupil, pre_isi, _], _, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_idx=[l[oddball] for l in self.list_odd_idx],
                trial_param=[None, None, [1], None, [0], [0]],
                mean_sem=False,
                cate=cate, roi_id=None)
            cs = get_cmap_color(self.bin_num, base_color=[color1, color2][oddball])
            # bin data based on isi.
            [bins, bin_center, _, bin_neu_seq, _, _, bin_stim_seq, bin_camera_pupil] = get_isi_bin_neu(
                neu_seq, stim_seq, camera_pupil, pre_isi, self.bin_win, self.bin_num)
            cluster_bin_neu_mean, cluster_bin_neu_sem = get_bin_mean_sem_cluster(bin_neu_seq, self.n_clusters, cluster_id)
            # plot results for each class.
            for ci in range(self.n_clusters):
                if np.sum(cluster_id==ci) > 0:
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
        def plot_trial_quant(axs, oddball):
            target_metrics = ['evoke_mag', 'onset_ramp', 'onset_drop']
            # collect data.
            [[color0, color1, color2, _],
             [neu_seq, stim_seq, camera_pupil, pre_isi, _], _, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_idx=[l[oddball] for l in self.list_odd_idx],
                trial_param=[None, None, [1], None, [0], [0]],
                mean_sem=False,
                cate=cate, roi_id=None)
            cs = get_cmap_color(self.bin_num, base_color=[color1, color2][oddball])
            # bin data based on isi.
            [bins, bin_center, _, bin_neu_seq, _, _, bin_stim_seq, _] = get_isi_bin_neu(
                neu_seq, stim_seq, camera_pupil, pre_isi, self.bin_win, self.bin_num)
            c_idx = bin_stim_seq.shape[1]//2
            cluster_bin_neu_mean, cluster_bin_neu_sem = get_bin_mean_sem_cluster(bin_neu_seq, self.n_clusters, cluster_id)
            # plot results for each class.
            for ci in range(self.n_clusters):
                if np.sum(cluster_id==ci) > 0:
                    quant_all = [
                        run_quantification(ns[cluster_id==ci,:], self.alignment['neu_time'], ss[c_idx+1,0])
                        for ns, ss in zip(bin_neu_seq, bin_stim_seq)]
                    # plot each metric.
                    for mi in range(len(target_metrics)):
                        axs[mi][ci].axis('off')
                        axs[mi][ci] = axs[mi][ci].inset_axes([0, 0, 0.4, 1], transform=axs[mi][ci].transAxes)
                        # plot each condition.
                        for di in range(self.bin_num):
                            m, s = get_mean_sem(quant_all[di][target_metrics[mi]].reshape(-1,1))
                            axs[mi][ci].errorbar(
                                di, m, s, None,
                                color=cs[di],
                                capsize=2, marker='o', linestyle='none',
                                markeredgecolor='white', markeredgewidth=0.1)
                        # adjust layout.
                        axs[mi][ci].tick_params(axis='y', tick1On=False)
                        axs[mi][ci].tick_params(axis='x', labelrotation=90)
                        axs[mi][ci].spines['right'].set_visible(False)
                        axs[mi][ci].spines['top'].set_visible(False)
                        axs[mi][ci].set_xlim([-0.5, self.bin_num-0.5])
                        axs[mi][ci].set_xticks(np.arange(self.bin_num))
                        axs[mi][ci].set_xticklabels(bin_center, rotation='vertical')
                        axs[mi][ci].set_xlabel('interval')
                        axs[mi][ci].set_ylabel(target_metrics[mi])
                        axs[mi][ci].set_title(lbl[ci])
        def plot_legend(ax):
            [[color0, _, color2, _],
             [neu_seq, stim_seq, camera_pupil, pre_isi, _], _,
             [n_trials, n_neurons]] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_idx=[l[0] for l in self.list_odd_idx],
                trial_param=[None, None, [1], None, [0], [0]],
                mean_sem=False,
                cate=cate, roi_id=None)
            [bins, _, _, _, _, _, _, _] = get_isi_bin_neu(
                neu_seq, stim_seq, camera_pupil, pre_isi, self.bin_win, self.bin_num)
            lbl = ['[{},{}] ms'.format(int(bins[i]),int(bins[i+1])) for i in range(self.bin_num)]
            cs = get_cmap_color(self.bin_num, base_color=color2)
            add_legend(ax, cs, lbl, n_trials, n_neurons, self.n_sess, 'upper right')
            ax.axis('off')
        # plot all.
        try: plot_oddball_jitter(axs[0], 0)
        except Exception as e: print(e)
        try: plot_trial_quant(axs[1], 0)
        except Exception as e: print(e)
        try: plot_oddball_jitter(axs[2], 1)
        except Exception as e: print(e)
        try: plot_trial_quant(axs[3], 1)
        except Exception as e: print(e)
        try: plot_legend(axs[4])
        except Exception as e: print(e)

    def plot_oddball_separability_global(self, axs, cate):
        xlim = [-4000, 5000]
        win_sample = 500
        win_step = 1
        lbl = ['cluster #'+str(ci) for ci in range(self.n_clusters)]
        cluster_id, neu_labels = get_cluster_cate(self.cluster_id, self.list_labels, self.list_significance, cate)
        def plot_separability(axs, oddball):
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
            # prepare data for decoding.
            c_idx = stim_seq_fix.shape[0]//2
            win_decode = xlim
            neu_y = np.arange(2)
            neu_x = np.concatenate([np.expand_dims(neu_seq_fix,axis=0), np.expand_dims(neu_seq_jitter,axis=0)], axis=0)              
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
                    c_idx = stim_seq_fix.shape[0]//2
                    for ai in range(len(axss)):
                        if oddball == 0:
                            axss[ai].axvline(stim_seq_fix[c_idx+1,0], color='gold', lw=1, linestyle='--')
                        if oddball == 1:
                            axss[ai].axvline(stim_seq_fix[c_idx,1]+self.expect, color='gold', lw=1, linestyle='--')
                        for si in [c_idx, c_idx+1]:
                            axss[ai].fill_between(
                                stim_seq_fix[si,:],
                                lower[ai] - 0.1*(upper[ai]-lower[ai]), upper[ai] + 0.1*(upper[ai]-lower[ai]),
                                color=color0, edgecolor='none', alpha=0.25, step='mid')
                    # plot results.
                    self.plot_mean_sem(
                        axss[0], self.alignment['neu_time'][llh_time], llh_mean, llh_sem, color0, None)
                    self.plot_mean_sem(
                        axss[1], self.alignment['neu_time'][llh_time], llh_results[1], llh_results[2], [color1, color2][oddball], None)
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
            [[color0, color1, color2, _], _, _,
             [n_trials, n_neurons]] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_param=[[-1], None, [1], None, [0], [0]],
                cate=cate, roi_id=None)
            cs = [color1, color2]
            lbl = ['short oddball', 'long oddball']
            add_legend(ax, cs, lbl, n_trials, n_neurons, self.n_sess, 'upper right')
            ax.axis('off')
        # plot all.
        try: plot_separability(axs[0], 0)
        except Exception as e: print(e)
        try: plot_separability(axs[1], 1)
        except Exception as e: print(e)
        try: plot_legend(axs[2])
        except Exception as e: print(e)

    def plot_oddball_separability_local(self, axs, cate):
        xlim = [-4000, 5000]
        win_sample = 500
        win_step = 1
        lbl = ['cluster #'+str(ci) for ci in range(self.n_clusters)]
        cluster_id, neu_labels = get_cluster_cate(self.cluster_id, self.list_labels, self.list_significance, cate)
        def plot_separability(axs, oddball):
            # collect data.
            [[color0, color1, color2, _],
             [neu_seq, stim_seq, camera_pupil, pre_isi, _], _, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_idx=[l[oddball] for l in self.list_odd_idx],
                trial_param=[None, None, [1], None, [0], [0]],
                mean_sem=False,
                cate=cate, roi_id=None)
            cs = get_cmap_color(self.bin_num, base_color=[color1, color2][oddball])
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
                        if oddball == 0:
                            axss[ai].axvline(stim_seq[c_idx+1,0], color='gold', lw=1, linestyle='--')
                        if oddball == 1:
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
                        axss[1], self.alignment['neu_time'][llh_time], llh_results[1], llh_results[2], [color1, color2][oddball], None)
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
                trial_idx=[l[0] for l in self.list_odd_idx],
                trial_param=[None, None, [1], None, [0], [0]],
                mean_sem=False,
                cate=cate, roi_id=None)
            [bins, _, _, _, _, _, _, _, _] = get_isi_bin_neu(
                neu_seq, stim_seq, camera_pupil, pre_isi, self.bin_win, self.bin_num)
            cs = [color0, color1, color2, 'gold']
            cs+= get_cmap_color(self.bin_num, base_color=color1)
            cs+= get_cmap_color(self.bin_num, base_color=color2)
            lbl = ['standard', 'short oddball', 'long oddball', 'unexpected event']
            lbl+= ['[{},{}] ms'.format(int(bins[i]),int(bins[i+1])) for i in range(self.bin_num)]
            lbl+= ['[{},{}] ms'.format(int(bins[i]),int(bins[i+1])) for i in range(self.bin_num)]
            add_legend(ax, cs, lbl, n_trials, n_neurons, self.n_sess, 'upper right')
            ax.axis('off')
        # plot all.
        try: plot_separability(axs[0], 0)
        except Exception as e: print(e)
        try: plot_separability(axs[1], 1)
        except Exception as e: print(e)
        try: plot_legend(axs[2])
        except Exception as e: print(e)
    
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
        # plot all.
        try: plot_3d_dynamics(axs[0])
        except Exception as e: print(e)
        try: plot_mean_dynamics(axs[1])
        except Exception as e: print(e)
    
    def plot_cross_sess_adapt(self, axs, fix_jitter, cate):
        xlim = [-2500, 4000]
        cluster_id, neu_labels = get_cluster_cate(self.cluster_id, self.list_labels, self.list_significance, cate)
        lbl = ['cluster #'+str(ci) for ci in range(self.n_clusters)]
        split_idx = get_split_idx(self.list_labels, self.list_significance, cate)
        day_cluster_id = np.split(cluster_id, split_idx)
        def plot_oddball_day_trace(axs, oddball):
            # collect data.
            [[color0, color1, color2, _],
             [_, _, stim_seq, _], _, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_idx=[l[0] for l in self.list_odd_idx],
                trial_param=[None, None, [0], None, [0], [0]],
                cate=cate, roi_id=None)
            _, [neu_seq, _, _, _, _], _, _ = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_idx=[l[oddball] for l in self.list_odd_idx],
                trial_param=[None, None, [fix_jitter], None, [0], [0]],
                mean_sem=False,
                cate=cate, roi_id=None)
            c_idx = stim_seq.shape[0]//2
            cs = get_cmap_color(len(day_cluster_id), base_color=[color1, color2][oddball])
            # plot results for each class.
            for ci in range(self.n_clusters):
                neu_mean = []
                neu_sem = []
                neu_color = []
                for di in range(len(day_cluster_id)):
                    if np.sum(day_cluster_id[di]==ci) > 0:
                        m, s = get_mean_sem(neu_seq[di][0,day_cluster_id[di]==ci,:])
                        neu_mean.append(m)
                        neu_sem.append(s)
                        neu_color.append(cs[di])
                # find bounds.
                upper = np.nanmax(neu_mean) + np.nanmax(neu_sem)
                lower = np.nanmin(neu_mean) - np.nanmax(neu_sem)
                # plot stimulus.
                if oddball == 0:
                    axs[ci].axvline(stim_seq[c_idx+1,0], color='gold', lw=1, linestyle='--')
                if oddball == 1:
                    axs[ci].axvline(stim_seq[c_idx,1]+self.expect, color='gold', lw=1, linestyle='--')
                if fix_jitter == 0:
                    for si in range(stim_seq.shape[0]):
                        axs[ci].fill_between(
                            stim_seq[si,:],
                            lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
                            color=color0, edgecolor='none', alpha=0.25, step='mid')
                if fix_jitter == 1:
                    for si in [c_idx, c_idx+1]:
                        axs[ci].fill_between(
                            stim_seq[si,:],
                            lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
                            color=color0, edgecolor='none', alpha=0.25, step='mid')
                # plot neural traces.
                for ni in range(len(neu_mean)):
                    self.plot_mean_sem(
                        axs[ci], self.alignment['neu_time'],
                        neu_mean[ni], neu_sem[ni], neu_color[ni], None)
                # adjust layout.
                adjust_layout_neu(axs[ci])
                axs[ci].set_xlim(xlim)
                axs[ci].set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
                axs[ci].set_xlabel('time since pre oddball stim (ms)')
                axs[ci].set_title(lbl[ci])
        def plot_trial_quant(axs, oddball):
            offset = np.arange(len(day_cluster_id))*0.05
            # collect data.
            [[color0, color1, color2, _],
             [_, _, stim_seq, _], _, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_idx=[l[0] for l in self.list_odd_idx],
                trial_param=[None, None, [0], None, [0], [0]],
                cate=cate, roi_id=None)
            _, [neu_seq, _, _, _, _], _, _ = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_idx=[l[oddball] for l in self.list_odd_idx],
                trial_param=[None, None, [fix_jitter], None, [0], [0]],
                mean_sem=False,
                cate=cate, roi_id=None)
            c_idx = stim_seq.shape[0]//2
            cs = get_cmap_color(len(day_cluster_id), base_color=[color1, color2][oddball])
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
            for ci in range(self.n_clusters):
                if np.sum(cluster_id==ci) > 0:
                    axs[ci].axis('off')
                    ax0 = axs[ci].inset_axes([0, 0.9, 0.5, 0.1], transform=axs[ci].transAxes)
                    ax1 = axs[ci].inset_axes([0, 0, 1, 0.7], transform=axs[ci].transAxes)
                    # average across neurons within cluster.
                    neu = [np.nanmean(neu[:,sci==ci,:],axis=1)
                         for neu,sci in zip(neu_seq,day_cluster_id)]
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
                    for ni in range(len(neu)):
                        self.plot_win_mag_quant(
                            ax1, neu[ni], self.alignment['neu_time'],
                            win_eval, cs[ni], 0, offset[ni])
        def plot_legend(ax):
            [[color0, color1, color2, _], _, _,
             [n_trials, n_neurons]] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_param=[[-1], None, [1], None, [0], [0]],
                cate=cate, roi_id=None)
            cs = get_cmap_color(len(day_cluster_id), color1)
            cs+= get_cmap_color(len(day_cluster_id), color2)
            lbl = [f'day #{di}' for di in range(len(day_cluster_id))]*2
            add_legend(ax, cs, lbl, n_trials, n_neurons, self.n_sess, 'upper right')
            ax.axis('off')
        # plot all.
        try: plot_oddball_day_trace(axs[0], 0)
        except Exception as e: print(e)
        try: plot_trial_quant(axs[1], 0)
        except Exception as e: print(e)
        try: plot_oddball_day_trace(axs[2], 1)
        except Exception as e: print(e)
        try: plot_trial_quant(axs[3], 1)
        except Exception as e: print(e)
        try: plot_legend(axs[4])
        except Exception as e: print(e)
        
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
                axs[0].set_title(f'GLM kernel course \n {label_name}')
                axs[1].set_title(f'reponse to standard interval \n {label_name}')
                axs[2].set_title(f'reponse to short oddball interval \n {label_name}')
                axs[3].set_title(f'reponse to long oddball interval \n {label_name}')

            except Exception as e: print(e)
            
    def cluster_oddball_jitter_common_all(self, axs_all):
        for cate, axs in zip([[-1,1,2],[-1],[1],[2]], axs_all):
            try:
                label_name = self.label_names[str(cate[0])] if len(cate)==1 else 'all'
                print(f'plotting results for {label_name}')
                
                self.plot_cluster_oddball_jitter_common_all(axs, cate=cate)

            except Exception as e: print(e)
    
    def cluster_oddball_fix_heatmap_all(self, axs_all):
        for cate, axs in zip([[-1,1,2],[-1],[1],[2]], axs_all):
            try:
                label_name = self.label_names[str(cate[0])] if len(cate)==1 else 'all'
                print(f'plotting results for {label_name}')
                
                self.plot_cluster_oddball_fix_heatmap_all(axs, cate=cate)
                axs[0].set_title(f'clustered latent features \n {label_name}')
                axs[1].set_title(f'cluster dendrogram \n {label_name}')
                axs[2].set_title(f'GLM kernel course \n {label_name}')
                axs[3].set_title(f'reponse to standard interval \n {label_name}')
                axs[4].set_title(f'reponse to short oddball interval \n {label_name}')
                axs[5].set_title(f'reponse to long oddball interval \n {label_name}')

            except Exception as e: print(e)
        
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

            except Exception as e: print(e)
        
    def cluster_oddball_fix_individual(self, axs_all):
        for cate, axs in zip([[-1,1,2],[-1],[1],[2]], axs_all):
            try:
                label_name = self.label_names[str(cate[0])] if len(cate)==1 else 'all'
                print(f'plotting results for {label_name}')
                
                self.plot_cluster_oddball_fix_individual(axs, cate=cate)

            except Exception as e: print(e)
    
    def cluster_oddball_jitter_global_individual(self, axs_all):
        for cate, axs in zip([[-1,1,2],[-1],[1],[2]], axs_all):
            try:
                label_name = self.label_names[str(cate[0])] if len(cate)==1 else 'all'
                print(f'plotting results for {label_name}')
                
                self.plot_cluster_oddball_jitter_individual(axs, 'global', cate=cate)

            except Exception as e: print(e)
    
    def cluster_oddball_jitter_common_individual(self, axs_all):
        for cate, axs in zip([[-1,1,2],[-1],[1],[2]], axs_all):
            try:
                label_name = self.label_names[str(cate[0])] if len(cate)==1 else 'all'
                print(f'plotting results for {label_name}')
                
                self.plot_cluster_oddball_jitter_individual(axs, 'common', cate=cate)

            except Exception as e: print(e)
    
    def cluster_oddball_jitter_local_individual(self, axs_all):
        for cate, axs in zip([[-1,1,2],[-1],[1],[2]], axs_all):
            try:
                label_name = self.label_names[str(cate[0])] if len(cate)==1 else 'all'
                print(f'plotting results for {label_name}')
                
                self.plot_cluster_oddball_jitter_local_individual(axs, cate=cate)

            except Exception as e: print(e)

    def oddball_separability_global(self, axs_all):
        for cate, axs in zip([[-1,1,2],[-1],[1],[2]], axs_all):
            try:
                label_name = self.label_names[str(cate[0])] if len(cate)==1 else 'all'
                print(f'plotting results for {label_name}')
                
                self.plot_oddball_separability_global(axs, cate=cate)

            except Exception as e: print(e)
        
    def oddball_separability_local(self, axs_all):
        for cate, axs in zip([[-1,1,2],[-1],[1],[2]], axs_all):
            try:
                label_name = self.label_names[str(cate[0])] if len(cate)==1 else 'all'
                print(f'plotting results for {label_name}')
                
                self.plot_oddball_separability_local(axs, cate=cate)

            except Exception as e: print(e)
    
    def oddball_latent_fix_all(self, axs_all):
        for cate, axs in zip([[-1,1,2],[-1],[1],[2]], axs_all):
            try:
                label_name = self.label_names[str(cate[0])] if len(cate)==1 else 'all'
                print(f'plotting results for {label_name}')
                
                self.plot_oddball_latent_fix_all(axs[0], 0, cate=cate)
                self.plot_oddball_latent_fix_all(axs[1], 1, cate=cate)

            except Exception as e: print(e)
    
    def cross_sess_adapt_fix(self, axs_all):
        for cate, axs in zip([[-1,1,2],[-1],[1],[2]], axs_all):
            try:
                label_name = self.label_names[str(cate[0])] if len(cate)==1 else 'all'
                print(f'plotting results for {label_name}')
                
                self.plot_cross_sess_adapt(axs, 0, cate=cate)

            except Exception as e: print(e)
    
    def cross_sess_adapt_jitter(self, axs_all):
        for cate, axs in zip([[-1,1,2],[-1],[1],[2]], axs_all):
            try:
                label_name = self.label_names[str(cate[0])] if len(cate)==1 else 'all'
                print(f'plotting results for {label_name}')
                
                self.plot_cross_sess_adapt(axs, 1, cate=cate)

            except Exception as e: print(e)

