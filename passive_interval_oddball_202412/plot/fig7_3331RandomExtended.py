#!/usr/bin/env python3

import traceback
import numpy as np
from sklearn.decomposition import PCA

from modules.Alignment import run_get_stim_response
from modeling.clustering import get_mean_sem_cluster
from modeling.clustering import get_bin_mean_sem_cluster
from modeling.clustering import get_cluster_cate
from modeling.generative import get_glm_cate
from utils import show_resource_usage
from utils import get_norm01_params
from utils import get_neu_trial
from utils import get_frame_idx_from_time
from utils import get_isi_bin_neu
from utils import get_expect_interval
from utils import get_split_idx
from utils import get_cmap_color
from utils import hide_all_axis
from utils import get_random_rotate_mat_3d
from utils import adjust_layout_isi_example_epoch
from utils import adjust_layout_3d_latent
from utils import add_legend
from utils import add_heatmap_colorbar
from utils import utils_basic

# fig, ax = plt.subplots(1, 1, figsize=(3, 6))
# fig, axs = plt.subplots(1, 8, figsize=(24, 3))
# axs = [plt.subplots(1, 8, figsize=(24, 3))[1], plt.subplots(1, 8, figsize=(24, 3))[1]]
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
        self.expect = np.array(np.mean([get_expect_interval(sl)[0] for sl in self.list_stim_labels]))
        self.bin_win = [2950,7550]
        self.bin_num = 10
        self.d_latent = 3
        self.glm = self.run_glm([-5000,5000])
        self.n_pre = 2
        self.n_post = 1
        self.cluster_id = self.run_clustering(self.n_pre, self.n_post)
    
    def plot_neuron_fraction(self, ax):
        try:
            colors = ['cornflowerblue', 'violet', 'mediumseagreen']
            cluster_id, neu_labels = get_cluster_cate(self.cluster_id, self.list_labels, [-1,1,2])
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

    def plot_isi_seting(self, ax):
        for i in range(2500,7500,500):
            ax.vlines(500+i, 0, 0.5, color='black')
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xlim([2950,7550])
        ax.set_ylim([0, 1.05])
        ax.set_xticks([3000,5000,7500])
        ax.set_yticks([])
        ax.set_xticklabels([3000,5000,7500])
        
    def plot_isi_example_epoch(self, ax):
        trial_win = [0,500]
        # get isi and trial labels.
        stim_labels = self.list_neural_trials[2]['stim_labels'][trial_win[0]:trial_win[1],:]
        isi = stim_labels[1:,0] - stim_labels[:-1,1]
        # plot trials.
        ax.scatter(np.arange(trial_win[0], trial_win[1]-1), isi, c='black', s=5)
        # adjust layouts.
        adjust_layout_isi_example_epoch(ax, trial_win, self.bin_win)

    def plot_cluster_stim_all(self, axs, cate=None):
        color0 = 'dimgrey'
        cluster_id, neu_labels = get_cluster_cate(self.cluster_id, self.list_labels, cate)
        # collect data.
        [_, [neu_seq, _, stim_seq, _], _, _] = get_neu_trial(
            self.alignment, self.list_labels, self.list_stim_labels,
            trial_param=[[2,3,4,5], None, None, None, [1], [0]],
            cate=cate, roi_id=None)
        c_idx = stim_seq.shape[0]//2
        @show_resource_usage
        def plot_glm_kernel(ax):
            kernel_all = get_glm_cate(self.glm, self.list_labels, cate)
            self.plot_glm_kernel(ax, kernel_all, cluster_id, color0, 0.6)
        @show_resource_usage
        def plot_stim(ax, scaled):
            xlim = [-1000, 6000]
            # collect data.
            l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, xlim[0], xlim[1])
            # get response within cluster.
            neu_mean, neu_sem = get_mean_sem_cluster(neu_seq, self.n_clusters, cluster_id)
            if scaled:
                norm_params = [get_norm01_params(neu_mean[ci,:]) for ci in range(self.n_clusters)]
            else:
                norm_params = [get_norm01_params(neu_mean) for ci in range(self.n_clusters)]
            # define layouts.
            ax.axis('off')
            ax = ax.inset_axes([0, 0, 0.5, 0.6], transform=ax.transAxes)
            # plot results.
            self.plot_cluster_mean_sem(
                ax, neu_mean, neu_sem,
                self.alignment['neu_time'], norm_params,
                stim_seq[c_idx,:].reshape(1,2), [color0], [color0]*self.n_clusters, xlim)
            # adjust layouts.
            ax.set_xlabel('time since stim (ms)')
        @show_resource_usage
        def plot_stim_heatmap(ax, norm_mode):
            # collect data.
            xlim = [stim_seq[c_idx,0]-500, stim_seq[c_idx,1]+500]
            l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, xlim[0], xlim[1])
            neu_time = self.alignment['neu_time'][l_idx:r_idx]
            neu_ci = [neu_seq[cluster_id==ci,l_idx:r_idx] for ci in range(self.n_clusters)]
            # define layouts.
            ax.axis('off')
            ax = ax.inset_axes([0, 0, 1, 0.6], transform=ax.transAxes)
            axs_hm = [ax.inset_axes([0.2, ci/self.n_clusters, 0.3, 0.75/self.n_clusters], transform=ax.transAxes)
                      for ci in range(self.n_clusters)]
            axs_cb = [ax.inset_axes([0.6, ci/self.n_clusters, 0.1, 0.75/self.n_clusters], transform=ax.transAxes)
                      for ci in range(self.n_clusters)]
            axs_hm.reverse()
            axs_cb.reverse()
            # plot results for each class.
            for ci in range(self.n_clusters):
                if np.sum(cluster_id==ci) > 0:
                    self.plot_heatmap_neuron(axs_hm[ci], axs_cb[ci], neu_ci[ci], neu_time, neu_ci[ci],
                                             sort_method='shuffle', norm_mode=norm_mode, neu_seq_share=neu_ci)
                    # add stimulus line.
                    axs_hm[ci].axvline(0, color='black', lw=1, linestyle='--')
                # adjust layouts.
                axs_hm[ci].tick_params(axis='y', labelrotation=0)
                axs_hm[ci].set_ylabel(None)
                if ci != self.n_clusters-1:
                    axs_hm[ci].set_xticks([])
                hide_all_axis(axs_cb[ci])
            axs_hm[self.n_clusters-1].set_xlabel('time since stim (ms)')
            ax.set_ylabel('neuron id')
            hide_all_axis(ax)
        @show_resource_usage
        def plot_neu_fraction(ax):
            # define layouts.
            ax.axis('off')
            ax = ax.inset_axes([0, 0, 1, 0.6], transform=ax.transAxes)
            self.plot_cluster_neu_fraction_in_cluster(ax, cluster_id, color0)
        @show_resource_usage
        def plot_fraction(ax):
            # define layouts.
            ax.axis('off')
            ax = ax.inset_axes([0, 0, 1, 0.6], transform=ax.transAxes)
            self.plot_cluster_cate_fraction_in_cluster(ax, cluster_id, neu_labels, self.label_names, color0)
        # plot all.
        try: plot_glm_kernel(axs[0])
        except: traceback.print_exc()
        try: plot_stim(axs[1], False)
        except: traceback.print_exc()
        try: plot_stim(axs[2], True)
        except: traceback.print_exc()
        try: plot_stim_heatmap(axs[3], 'none')
        except: traceback.print_exc()
        try: plot_stim_heatmap(axs[4], 'minmax')
        except: traceback.print_exc()
        try: plot_stim_heatmap(axs[5], 'share')
        except: traceback.print_exc()
        try: plot_neu_fraction(axs[6])
        except: traceback.print_exc()
        try: plot_fraction(axs[7])
        except: traceback.print_exc()

    def plot_cluster_interval_bin_all(self, axs, cate=None):
        color0 = 'dimgrey'
        colors = get_cmap_color(self.bin_num, cmap=self.random_bin_cmap)
        cluster_id, neu_labels = get_cluster_cate(self.cluster_id, self.list_labels, cate)
        split_idx = get_split_idx(self.list_labels, cate)
        day_cluster_id = np.split(cluster_id, split_idx)
        # collect data.
        [_, [neu_seq, stim_seq, camera_pupil, pre_isi, post_isi], _,
         [n_trials, n_neurons]] = get_neu_trial(
            self.alignment, self.list_labels, self.list_stim_labels,
            trial_param=[[2,3,4,5], None, None, None, [1], [0]],
            mean_sem=False,
            cate=cate, roi_id=None)
        c_idx = np.nanmean(np.concatenate(stim_seq, axis=0), axis=0).shape[0]//2
        @show_resource_usage
        def plot_interval_heatmap(ax, norm_mode):
            xlim = [-1000, 10000]
            l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, xlim[0], xlim[1])
            neu_time = self.alignment['neu_time'][l_idx:r_idx]
            # get response within cluster.
            neu_ci = [np.concatenate(
                [np.nanmean(neu[:,dci==ci,:],axis=1)
                 for neu,dci in zip(neu_seq,day_cluster_id)], axis=0)
                for ci in range(self.n_clusters)]
            neu_ci = [nc[np.argsort(np.concatenate(post_isi)), l_idx:r_idx] for nc in neu_ci]
            # define layouts.
            ax0 = ax.inset_axes([0.2, 0, 0.5, 0.6], transform=ax.transAxes)
            ax1 = ax.inset_axes([0.7, 0, 0.1, 0.6], transform=ax.transAxes)
            axs_hm = [ax0.inset_axes([0, 0.05+ci/self.n_clusters, 1, 0.8/self.n_clusters], transform=ax0.transAxes)
                      for ci in range(self.n_clusters)]
            axs_cb = [ax1.inset_axes([0, 0.05+ci/self.n_clusters, 1, 0.8/self.n_clusters], transform=ax1.transAxes)
                      for ci in range(self.n_clusters)]
            axs_hm.reverse()
            axs_cb.reverse()
            # plot results for each class.
            for ci in range(self.n_clusters):
                if np.sum(cluster_id==ci) > 0:
                    self.plot_heatmap_trial(
                            axs_hm[ci], axs_cb[ci], neu_ci[ci], neu_time,
                            norm_mode=norm_mode, neu_seq_share=neu_ci)
                    hide_all_axis(axs_cb[ci])
            # adjust layouts.
            for ci in range(self.n_clusters):
                axs_hm[ci].set_xlim(xlim)
                axs_hm[ci].set_yticklabels((((np.arange(2)+0.5)/2)*2000+500)[::-1].astype('int32'))
                if ci != self.n_clusters-1:
                    axs_hm[ci].set_xticklabels([])
            axs_hm[self.n_clusters-1].set_xlabel('time since stim (ms)')
            ax.set_ylabel('interval (ms)')
            ax.set_title(f'sorted with {norm_mode}')
            hide_all_axis(ax)
            hide_all_axis(ax0)
            hide_all_axis(ax1)
        @show_resource_usage
        def plot_interval_bin(ax, mode):
            # bin data based on isi.
            if mode == 'pre':
                xlim = [-8500, 1500]
                isi = pre_isi
                isi_idx_offset = -1
            if mode == 'post':
                xlim = [-1000, 10000]
                isi = post_isi
                isi_idx_offset = 1
            [bins, bin_center, _, bin_neu_seq, _, _, bin_stim_seq, _] = get_isi_bin_neu(
                neu_seq, stim_seq, camera_pupil, isi, self.bin_win, self.bin_num)
            # get response within cluster at each bin.
            cluster_bin_neu_mean, cluster_bin_neu_sem = get_bin_mean_sem_cluster(bin_neu_seq, self.n_clusters, cluster_id)
            norm_params = [get_norm01_params(cluster_bin_neu_mean[:,i,:]) for i in range(self.n_clusters)]
            # define layouts.
            ax.axis('off')
            ax = ax.inset_axes([0, 0, 1, 0.6], transform=ax.transAxes)
            # plot results.
            ax.fill_between(
                np.nanmean(bin_stim_seq, axis=0)[c_idx,:],
                0, self.n_clusters,
                color=color0, edgecolor='none', alpha=0.25, step='mid')
            for bi in range(self.bin_num):
                ax.axvline(bin_stim_seq[bi, c_idx+isi_idx_offset, 0], color=colors[bi], lw=1, linestyle='--')
            for bi in range(self.bin_num):
                self.plot_cluster_mean_sem(
                    ax, cluster_bin_neu_mean[bi,:,:], cluster_bin_neu_sem[bi,:,:],
                    self.alignment['neu_time'], norm_params,
                    None, None, [colors[bi]]*self.n_clusters, xlim)
            # adjust layouts.
            ax.set_xlabel('time since stim (ms)')
        @show_resource_usage
        def plot_legend(ax):
            [bins, _, _, _, _, _, _, _] = get_isi_bin_neu(
                neu_seq, stim_seq, camera_pupil, pre_isi, self.bin_win, self.bin_num)
            lbl = ['[{},{}] ms'.format(int(bins[i]),int(bins[i+1])) for i in range(self.bin_num)]
            cs = colors
            add_legend(ax, cs, lbl, n_trials, n_neurons, self.n_sess, 'upper right')
            ax.axis('off')
        # plot all.
        try: plot_interval_heatmap(axs[0], 'none')
        except: traceback.print_exc()
        try: plot_interval_heatmap(axs[1], 'minmax')
        except: traceback.print_exc()
        try: plot_interval_heatmap(axs[2], 'share')
        except: traceback.print_exc()
        try: plot_interval_bin(axs[3], 'pre')
        except: traceback.print_exc()
        try: plot_interval_bin(axs[4], 'post')
        except: traceback.print_exc()
        try: plot_legend(axs[5])
        except: traceback.print_exc()
        
    def plot_latent_all(self, axs, cate=None):
        # collect data.
        [_, [neu_seq, stim_seq, camera_pupil, pre_isi, post_isi], _,
         [n_trials, n_neurons]] = get_neu_trial(
            self.alignment, self.list_labels, self.list_stim_labels,
            trial_param=[[2,3,4,5], None, None, None, [1], [0]],
            mean_sem=False,
            cate=cate, roi_id=None)
        c_idx = np.nanmean(np.concatenate(stim_seq, axis=0), axis=0).shape[0]//2
        @show_resource_usage
        def plot_interval_bin_latent_all(axs):
            colors = get_cmap_color(self.bin_num, cmap=self.random_bin_cmap)
            # bin data based on isi.
            [bins, _, _, bin_neu_seq, _, _, bin_stim_seq, _] = get_isi_bin_neu(
                neu_seq, stim_seq, camera_pupil, post_isi, self.bin_win, self.bin_num)
            # get latent dynamics.
            neu_x = np.concatenate([bns for bns in bin_neu_seq], axis=1)
            # fit model.
            model = PCA(n_components=3)
            model.fit(neu_x.reshape(neu_x.shape[0],-1).T)
            z = model.transform(neu_x.reshape(neu_x.shape[0],-1).T)
            neu_z = z.reshape(self.bin_num, -1, 3).transpose([0,2,1])
            # random rotate dynamics.
            for ai in range(len(axs)-1):
                # get random matrix.
                rm = get_random_rotate_mat_3d()
                # define layouts.
                axs[ai].axis('off')
                ax1 = axs[ai].inset_axes([0, 0, 0.6, 0.6], transform=axs[ai].transAxes, projection='3d')
                # plot 3d dynamics.
                for bi in range(self.bin_num):
                    l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, bin_stim_seq[bi,c_idx,0], bin_stim_seq[bi,c_idx+1,0])
                    neu_time = self.alignment['neu_time'][l_idx:r_idx]
                    cmap, _ = get_cmap_color(len(neu_time), base_color=['lemonchiffon', colors[bi], 'black'], return_cmap=True)
                    self.plot_3d_latent_dynamics(
                        ax1, np.matmul(rm, neu_z[bi,:,l_idx:r_idx]), None, neu_time,
                        cmap=cmap, end_color=colors[bi], add_stim=False)
                # adjust layouts.
                adjust_layout_3d_latent(ax1)
            ax2 = axs[-1].inset_axes([0, 0, 0.5, 1], transform=axs[-1].transAxes)
            ax3 = axs[-1].inset_axes([0.5, 0, 0.1, 0.6], transform=axs[-1].transAxes)
            # add colorbar.
            add_legend(ax2, colors, ['[{},{}] ms'.format(int(bins[i]),int(bins[i+1])) for i in range(self.bin_num)], None, None, None, 'upper left')
            t_cmap, _ = get_cmap_color(len(neu_time), base_color=['lemonchiffon', 'black'], return_cmap=True)
            add_heatmap_colorbar(ax3, t_cmap, None, 'interval progress since stim onset')
            hide_all_axis(axs[-1])
            hide_all_axis(ax2)
            hide_all_axis(ax3)
        # plot all.
        try: plot_interval_bin_latent_all(axs[0])
        except: traceback.print_exc()

# colors = ['#989A9C', '#A4CB9E', '#9DB4CE', '#EDA1A4', '#F9C08A']
class plotter_main(plotter_utils):
    def __init__(self, neural_trials, labels, label_names, temp_folder, cate_list):
        super().__init__(neural_trials, labels, temp_folder, cate_list)
        self.label_names = label_names
    
    def cluster_stim_all(self, axs_all):
        for cate, axs in zip(self.cate_list, axs_all):
            try:
                label_name = self.label_names[str(cate[0])] if len(cate)==1 else 'all'
                print(f'plotting results for {label_name}')
                
                self.plot_cluster_stim_all(axs, cate=cate)

            except: traceback.print_exc()

    def cluster_interval_bin_all(self, axs_all):
        for cate, axs in zip(self.cate_list, axs_all):
            try:
                label_name = self.label_names[str(cate[0])] if len(cate)==1 else 'all'
                print(f'plotting results for {label_name}')
                
                self.plot_cluster_interval_bin_all(axs, cate=cate)

            except: traceback.print_exc()
    
    def latent_all(self, axs_all):
        for cate, axs in zip(self.cate_list, axs_all):
            try:
                label_name = self.label_names[str(cate[0])] if len(cate)==1 else 'all'
                print(f'plotting results for {label_name}')
                
                self.plot_latent_all(axs, cate=cate)

            except: traceback.print_exc()