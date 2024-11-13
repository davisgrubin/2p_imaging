#!/usr/bin/env python3

import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from modules.Alignment import run_get_stim_response
from plot.utils import norm01
from plot.utils import exclude_post_odd_stim
from plot.utils import get_frame_idx_from_time
from plot.utils import get_mean_sem
from plot.utils import get_multi_sess_neu_trial_average
from plot.utils import get_roi_label_color
from plot.utils import get_epoch_idx
from plot.utils import get_expect_time
from plot.utils import adjust_layout_neu
from plot.utils import utils

# fig, ax = plt.subplots(1, 1, figsize=(6, 6))

class plotter_utils(utils):

    def __init__(
            self,
            list_neural_trials, list_labels, list_significance
            ):
        super().__init__()
        timescale = 1.0
        self.n_sess = len(list_neural_trials)
        self.l_frames = int(80*timescale)
        self.r_frames = int(100*timescale)
        self.list_stim_labels = [
            nt['stim_labels'][1:-1,:] for nt in list_neural_trials]
        self.list_stim_labels = [
            exclude_post_odd_stim(sl) for sl in self.list_stim_labels]
        self.list_labels = list_labels
        self.expect = np.array([
            np.mean([get_expect_time(sl)[0] for sl in self.list_stim_labels]),
            np.mean([get_expect_time(sl)[1] for sl in self.list_stim_labels])])
        self.epoch_early = [get_epoch_idx(sl)[0] for sl in self.list_stim_labels]
        self.epoch_late  = [get_epoch_idx(sl)[1] for sl in self.list_stim_labels]
        self.alignment = run_get_stim_response(
                list_neural_trials, self.l_frames, self.r_frames, expected='none')
        self.list_significance = list_significance

    def plot_normal(self, ax, normal, fix_jitter, cate=None, roi_id=None):
        if cate != None:
            _, color1, color2, _ = get_roi_label_color([cate], 0)
            neu_cate = [
                self.alignment['list_neu_seq'][i][:,(self.list_labels[i]==cate)*self.list_significance[i]['r_normal'],:]
                for i in range(self.n_sess)]
        if roi_id != None:
            _, color1, color2, _ = get_roi_label_color(self.labels, roi_id)
            neu_cate = [np.expand_dims(self.alignment['list_neu_seq'][0][:,roi_id,:], axis=1)]
        # collect data.
        neu_short, _, stim_seq_short, stim_value_short, _ = get_multi_sess_neu_trial_average(
            self.list_stim_labels, neu_cate, self.alignment['list_stim_seq'],
            self.alignment['list_stim_value'], self.alignment['list_pre_isi'],
            trial_param=[[2,3,4,5], [0], [fix_jitter], None, [0]])
        mean_short, sem_short = get_mean_sem(neu_short)
        # compute bounds.
        upper = np.nanmax(mean_short) + np.nanmax(sem_short)
        lower = np.nanmin(mean_short) - np.nanmax(sem_short)
        # plot stimulus.
        for i in range(3):
            ax.fill_between(
                stim_seq_short[i,:],
                lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
                color=color1, alpha=0.15, step='mid')
        self.plot_vol(ax, self.alignment['stim_time'], stim_value_short.reshape(1,-1), color1, upper, lower)
        # plot neural traces.
        self.plot_mean_sem(ax, self.alignment['neu_time'], mean_short, sem_short, color2, None)
        adjust_layout_neu(ax)
        ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
        ax.set_xlabel('time since stim (ms)')

    def plot_normal_pca(self, ax, normal, fix_jitter):
        _, _, color, _ = get_roi_label_color([0], 0)
        d_latent = 6
        scale = 0.9
        neu_cate = [
            self.alignment['list_neu_seq'][i][:,self.list_significance[i]['r_normal'],:]
            for i in range(self.n_sess)]
        # collect data.
        neu, _, stim_seq, _, _ = get_multi_sess_neu_trial_average(
            self.list_stim_labels, neu_cate, self.alignment['list_stim_seq'],
            self.alignment['list_stim_value'], self.alignment['list_pre_isi'],
            trial_param=[[2,3,4,5], [normal], [fix_jitter], None, [0]])
        neu = neu[~np.isnan(np.sum(neu,axis=1)),:]
        # fit model.
        model = PCA(n_components=d_latent)
        z = model.fit_transform(neu.T).T
        for i in range(d_latent):
            z[i,:] = norm01(z[i,:])
        # plot stimulus.
        upper = d_latent
        lower = 0
        for i in range(3):
            ax.fill_between(
                stim_seq[i,:],
                lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
                color=color, alpha=0.15, step='mid')
        # plot latents.
        for i in range(d_latent):
            ax.plot(self.alignment['neu_time'], norm01(z[i,:])*scale+d_latent-i-1, color=color)
        ax.tick_params(axis='y', tick1On=False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_yticks([])
        ax.set_xlabel('time (ms)')
        ax.set_ylabel('df/f (z-scored)')
        ax.set_ylim([-0.1, d_latent+0.1])

    def plot_normal_cluster(self, ax, normal, fix_jitter, cate):
        n_clusters = 8
        scale = 0.9
        win_sort = [-500,500]
        l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, win_sort[0], win_sort[1])
        _, color1, color2, cmap = get_roi_label_color([cate], 0)
        neu_cate = [
            self.alignment['list_neu_seq'][i][:,(self.list_labels[i]==cate)*self.list_significance[i]['r_normal'],:]
            for i in range(self.n_sess)]
        # collect data.
        neu, _, stim_seq, _, _ = get_multi_sess_neu_trial_average(
            self.list_stim_labels, neu_cate, self.alignment['list_stim_seq'],
            self.alignment['list_stim_value'], self.alignment['list_pre_isi'],
            trial_param=[[2,3,4,5], [normal], [fix_jitter], None, [0]])
        neu = neu[~np.isnan(np.sum(neu,axis=1)),:]
        # fit model.
        corr_matrix = np.corrcoef(neu)
        model = KMeans(n_clusters)
        cluster_id = model.fit_predict(corr_matrix)
        # collect neural traces based on clusters and sorted by peak timing.
        neu_mean = np.zeros((n_clusters, len(self.alignment['neu_time'])))
        neu_sem  = np.zeros((n_clusters, len(self.alignment['neu_time'])))
        for i in range(n_clusters):
            neu_mean[i,:], neu_sem[i,:] = get_mean_sem(neu[np.where(cluster_id==i)[0], :])
        sort_idx_neu = np.argmax(neu_mean[:,l_idx:r_idx], axis=1).reshape(-1).argsort()
        neu_mean = neu_mean[sort_idx_neu,:]
        neu_sem  = neu_sem[sort_idx_neu,:]
        # plot sorted correlation matrix.
        sorted_indices = np.argsort(cluster_id)
        sorted_corr_matrix = corr_matrix[sorted_indices, :][:, sorted_indices]
        ax[0].imshow(sorted_corr_matrix, cmap=cmap)
        ax[0].set_xlabel('neuron id')
        ax[0].set_ylabel('neuron id')
        ax[0].spines['left'].set_visible(False)
        ax[0].spines['right'].set_visible(False)
        ax[0].spines['top'].set_visible(False)
        ax[0].spines['bottom'].set_visible(False)
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        # plot stimulus.
        upper = n_clusters
        lower = 0
        for i in range(3):
            ax[1].fill_between(
                stim_seq[i,:],
                lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
                color=color1, alpha=0.15, step='mid')
        # plot neural traces averaged within clusters.
        for i in range(n_clusters):
            ax[1].plot(self.alignment['neu_time'], norm01(neu_mean[i,:])*scale+n_clusters-i-1, color=color2)
        ax[1].tick_params(axis='y', tick1On=False)
        ax[1].spines['left'].set_visible(False)
        ax[1].spines['right'].set_visible(False)
        ax[1].spines['top'].set_visible(False)
        ax[1].set_yticks([])
        ax[1].set_xlabel('time (ms)')
        ax[1].set_ylabel('df/f (z-scored)')
        ax[1].set_ylim([-0.1, n_clusters+0.1])

    def plot_normal_select(self, ax, normal, fix_jitter, cate=None, roi_id=None):
        if cate != None:
            _, color1, color2, _ = get_roi_label_color([cate], 0)
            neu_cate = [
                self.alignment['list_neu_seq'][i][:,(self.list_labels[i]==cate)*self.list_significance[i]['r_normal'],:]
                for i in range(self.n_sess)]
        if roi_id != None:
            _, color1, color2, _ = get_roi_label_color(self.labels, roi_id)
            neu_cate = [np.expand_dims(self.alignment['list_neu_seq'][0][:,roi_id,:], axis=1)]
        # collect data.
        neu_x = []
        for img_id in [2,3,4,5]:
            neu, _, _, _, _ = get_multi_sess_neu_trial_average(
                self.list_stim_labels, neu_cate, self.alignment['list_stim_seq'],
                self.alignment['list_stim_value'], self.alignment['list_pre_isi'],
                trial_param=[[img_id], [normal], [fix_jitter], None, [0]])
            neu = neu[~np.isnan(np.sum(neu,axis=1)),:]
            neu_x.append(np.expand_dims(neu,axis=2))
        neu_x = np.concatenate(neu_x, axis=2)


# colors = ['#989A9C', '#A4CB9E', '#9DB4CE', '#EDA1A4', '#F9C08A']
class plotter_VIPTD_G8_align_stim(plotter_utils):
    def __init__(self, neural_trials, labels, significance):
        super().__init__(neural_trials, labels, significance)

    def normal_exc(self, axs):
        self.plot_normal(axs[0], [0], 0, cate=-1)
        axs[0].set_title('response to normal \n excitatory')

    def normal_inh(self, axs):
        self.plot_normal(axs[0], [0], 0, cate=1)
        axs[0].set_title('response to normal \n inhibitory')    

    def normal_heatmap(self, axs):
        win_sort = [-200, 1000]
        labels = np.concatenate(self.list_labels)
        sig = np.concatenate([self.list_significance[n]['r_normal'] for n in range(self.n_sess)])
        neu_short_fix, _, _, _, _ = get_multi_sess_neu_trial_average(
            self.list_stim_labels, self.alignment['list_neu_seq'], self.alignment['list_stim_seq'],
            self.alignment['list_stim_value'], self.alignment['list_pre_isi'],
            trial_param=[[2,3,4,5], [0], [0], None, [0]])
        axs[0].set_xlabel('time since stim (ms)')
        self.plot_heatmap_neuron(
            axs[0], neu_short_fix, self.alignment['neu_time'], neu_short_fix,
            win_sort, labels, sig)
        axs[0].set_title('response to normal')
    
    def normal_mode(self, axs):
        self.plot_normal_pca(axs[0], 0, 0)
        axs[0].set_title('top principle components')
        self.plot_normal_cluster([axs[1], axs[2]], 0, 0, cate=-1)
        axs[1].set_title('sorted correlation matrix \n excitatory')
        axs[2].set_title('clustering response \n excitatory')
        self.plot_normal_cluster([axs[3], axs[4]], 0, 0, cate=1)
        axs[3].set_title('sorted correlation matrix \n inhibitory')
        axs[4].set_title('clustering response \n inhibitory')
        
        
        
        
        
        
        
        
        
        