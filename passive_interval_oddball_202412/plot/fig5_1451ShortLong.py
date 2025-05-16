#!/usr/bin/env python3

import numpy as np
from sklearn.decomposition import PCA

from modules.Alignment import run_get_stim_response
from modeling.clustering import clustering_neu_response_mode
from modeling.clustering import remap_cluster_id
from modeling.clustering import get_mean_sem_cluster
from modeling.clustering import get_bin_mean_sem_cluster
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
from utils import add_legend
from utils import utils_basic

# fig, ax = plt.subplots(1, 1, figsize=(6, 6))
# fig, ax = plt.subplots(1, 1, figsize=(2, 20))
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

    def plot_cluster_all(self, axs, cate):
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
        def plot_random_bin(ax, mode):
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
        def plot_standard(ax, standard):
            xlim = [-2000,3000]
            # collect data.
            [[color0, color1, color2, _],
             [neu_seq, _, stim_seq, _], _, _] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_param=[[2,3,4,5], [standard], None, None, [0], [0]],
                cate=cate, roi_id=None)
            # get response within cluster at each bin.
            neu_mean, neu_sem = get_mean_sem_cluster(neu_seq, cluster_id)
            norm_params = [get_norm01_params(neu_mean[i,:]) for i in range(self.n_clusters)]
            # plot results.
            self.plot_cluster_mean_sem(
                ax, neu_mean, neu_sem,
                self.alignment['neu_time'], norm_params,
                stim_seq,
                [color0]*stim_seq.shape[0], [[color1, color2][standard]]*self.n_clusters, xlim)
            # adjust layout.
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
            neu_mean, neu_sem = get_mean_sem_cluster(neu_seq, cluster_id)
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
        def plot_fraction(ax):
            self.plot_cluster_cate_fraction_in_cluster(ax, cluster_id, neu_labels, self.label_names)
        def plot_legend(ax):
            [[color0, _, color2, _],
             [neu_seq, stim_seq, _, pre_isi, post_isi],
             [neu_labels, _],
             [n_trials, n_neurons]] = get_neu_trial(
                self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
                trial_param=[[2,3,4,5], None, None, None, [0], [0]],
                mean_sem=False,
                cate=cate, roi_id=None)
            [bins, bin_center, _, bin_neu_seq, _, _, bin_stim_seq] = get_isi_bin_neu(
                neu_seq, stim_seq, post_isi, self.bin_win, self.bin_num)
            colors = get_cmap_color(self.bin_num, base_color=color2)
            lbl = ['[{},{}] ms'.format(int(bins[i]),int(bins[i+1])) for i in range(self.bin_num)]
            lbl+= [v for v in self.label_names.values()] + ['all cell-types']
            cs = colors + [get_roi_label_color(cate=[int(k)])[2] for k in self.label_names.keys()]
            cs+= [get_roi_label_color(cate=[-1,1,2])[2]]
            add_legend(ax, cs, lbl, n_trials, n_neurons, self.n_sess, 'upper right')
            ax.axis('off')
        # plot all.
        try: plot_glm_kernel(axs[0])
        except: pass
        try: plot_random_bin(axs[1], 'pre')
        except: pass
        try: plot_random_bin(axs[2], 'post')
        except: pass
        try: plot_standard(axs[3], 0)
        except: pass
        try: plot_standard(axs[4], 1)
        except: pass
        try: plot_oddball(axs[5], 0)
        except: pass
        try: plot_oddball(axs[6], 1)
        except: pass
        try: plot_neu_fraction(axs[7])
        except: pass
        try: plot_fraction(axs[8])
        except: pass
        try: plot_legend(axs[9])
        except: pass

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
             [neu_trans, _, stim_seq, _], _,
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
                axs[ci].set_xlabel('time since block transition (ms)')
                axs[ci].set_title(lbl[ci])
        def plot_trial_quant(axs, standard):
            trials_eval = 5
            target_metrics = ['evoke_mag', 'onset_ramp']
            # collect data.
            [[color0, color1, color2, _],
             [neu_trans, _, stim_seq, _], _,
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
                axs[1].set_title(f'reponse to binned interval in random \n {label_name}')
                axs[2].set_title(f'reponse to binned interval in random \n {label_name}')
                axs[3].set_title(f'reponse to short standard interval \n {label_name}')
                axs[4].set_title(f'reponse to long standard interval \n {label_name}')
                axs[5].set_title(f'reponse to short oddball interval \n {label_name}')
                axs[6].set_title(f'reponse to long oddball interval \n {label_name}')
                axs[7].set_title(f'fraction of neurons in cluster \n {label_name}')
                axs[8].set_title(f'fraction of cell-types in cluster \n {label_name}')
                axs[9].set_title(f'legend \n {label_name}')

            except: pass

    def cluster_block_adapt_individual(self, axs_all):
        for cate, axs in zip([[-1,1,2],[-1],[1],[2]], axs_all):
            try:
                label_name = self.label_names[str(cate[0])] if len(cate)==1 else 'all'
                print(f'plotting results for {label_name}')

                self.plot_cluster_block_adapt_individual(axs, cate=cate)

            except: pass
