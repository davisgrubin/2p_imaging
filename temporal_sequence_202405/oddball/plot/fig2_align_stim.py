#!/usr/bin/env python3

import numpy as np
from sklearn.manifold import TSNE
from scipy.signal import cwt, ricker
from matplotlib.colors import LinearSegmentedColormap

from modules.Alignment import run_get_stim_response
from modeling.decoding import multi_sess_decoding_num_neu
from modeling.decoding import multi_sess_decoding_slide_win
from modeling.clustering import clustering_neu_response_mode
from plot.utils import norm01
from plot.utils import exclude_post_odd_stim
from plot.utils import get_frame_idx_from_time
from plot.utils import get_change_prepost_idx
from plot.utils import get_mean_sem
from plot.utils import get_neu_sync
from plot.utils import get_multi_sess_neu_trial_average
from plot.utils import get_roi_label_color
from plot.utils import get_epoch_idx
from plot.utils import get_expect_time
from plot.utils import adjust_layout_neu
from plot.utils import adjust_layout_spectral
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
            color0, color1, color2, _ = get_roi_label_color([cate], 0)
            neu_cate = [
                self.alignment['list_neu_seq'][i][:,(self.list_labels[i]==cate)*self.list_significance[i]['r_normal'],:]
                for i in range(self.n_sess)]
        if roi_id != None:
            _, color1, color2, _ = get_roi_label_color(self.labels, roi_id)
            neu_cate = [np.expand_dims(self.alignment['list_neu_seq'][0][:,roi_id,:], axis=1)]
        # collect data.
        neu_short, _, stim_seq_short, stim_value_short, _ = get_multi_sess_neu_trial_average(
            self.list_stim_labels, neu_cate, self.alignment,
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
                color=color0, alpha=0.15, step='mid')
        self.plot_vol(ax, self.alignment['stim_time'], stim_value_short.reshape(1,-1), color0, upper, lower)
        # plot neural traces.
        self.plot_mean_sem(ax, self.alignment['neu_time'], mean_short, sem_short, color2, None)
        adjust_layout_neu(ax)
        ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
        ax.set_xlabel('time since stim (ms)')
    
    def plot_normal_spectral(self, ax, normal, fix_jitter, cate):
        max_freq = 64
        win_eval = [-2500,3500]
        l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, win_eval[0], win_eval[1])
        _, _, _, cmap = get_roi_label_color([cate], 0)
        neu_cate = [
            self.alignment['list_neu_seq'][i][:,(self.list_labels[i]==cate)*self.list_significance[i]['r_normal'],:]
            for i in range(self.n_sess)]
        neu_x, _, _, _, _ = get_multi_sess_neu_trial_average(
            self.list_stim_labels, neu_cate, self.alignment,
            trial_param=[[2,3,4,5], [0], [fix_jitter], None, [0]])
        # wavelet decomposition.
        neu_cwt = [
            norm01(cwt(neu_x[i, :], ricker, np.arange(1, max_freq-1)))
            for i in range(neu_x.shape[0])]
        neu_cwt = np.nanmean(neu_cwt, axis=0)
        # plot spectrogram.
        ax.imshow(
            neu_cwt,
            extent=[self.alignment['neu_time'][0], self.alignment['neu_time'][-1], 1, max_freq-1],
            aspect='auto', origin='lower', cmap=cmap)
        adjust_layout_spectral(ax)
        ax.set_xlabel('time since stim (ms)')
        ax.axvline(0, color='black', lw=1, label='stim', linestyle='--')
        ax.set_xlim([win_eval[0], win_eval[1]])
    
    def plot_normal_sync(self, ax, normal, fix_jitter, cate):
        win_width = 200
        l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, 0, win_width)
        win_width = r_idx - l_idx
        neu_cate = [
            self.alignment['list_neu_seq'][i][:,(self.list_labels[i]==cate)*self.list_significance[i]['r_oddball'],:]
            for i in range(self.n_sess)]
        color0, color1, color2, _ = get_roi_label_color([cate], 0)
        # collect data.
        neu_short, _, stim_seq_short, stim_value_short, _ = get_multi_sess_neu_trial_average(
            self.list_stim_labels, neu_cate, self.alignment,
            trial_param=[[2,3,4,5], [0], [fix_jitter], None, [0]])
        # compute synchronization.
        sync_short = get_neu_sync(neu_short, win_width)
        # find bounds.
        upper = np.nanmax(sync_short)
        lower = np.nanmin(sync_short)
        # plot stimulus.
        for i in range(3):
            ax.fill_between(
                stim_seq_short[i,:],
                lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
                color=color0, alpha=0.15, step='mid')
        self.plot_vol(ax, self.alignment['stim_time'], stim_value_short.reshape(1,-1), color0, upper, lower)
        # plot synchronization.
        ax.plot(self.alignment['neu_time'][win_width:], sync_short, color=color2)
        ax.tick_params(axis='y', tick1On=False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_ylabel('sync level')
        ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
        ax.set_xlabel('time since stim (ms)')
        ax.legend(loc='upper left')

    def plot_normal_cluster(self, axs, normal, fix_jitter, cate):
        n_clusters = 4
        max_clusters = 32
        win_sort = [-500,500]
        colors=['#A4CB9E', '#9DB4CE', '#EDA1A4', '#F9C08A']
        metric_lbl = ['silhouette', 'calinski_harabasz', 'davies_bouldin', 'inertia']
        l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, win_sort[0], win_sort[1])
        color0, color1, color2, cmap = get_roi_label_color([cate], 0)
        neu_cate = [
            self.alignment['list_neu_seq'][i][:,(self.list_labels[i]==cate)*self.list_significance[i]['r_normal'],:]
            for i in range(self.n_sess)]
        # collect data.
        neu, _, stim_seq, _, _ = get_multi_sess_neu_trial_average(
            self.list_stim_labels, neu_cate, self.alignment,
            trial_param=[[2,3,4,5], [normal], [fix_jitter], None, [0]])
        neu = neu[~np.isnan(np.sum(neu,axis=1)),:]
        # fit model.
        [metrics, sorted_neu_corr, neu_mean, neu_sem, cluster_corr] = clustering_neu_response_mode(
            neu, n_clusters, max_clusters, l_idx, r_idx)
        # plot sorted correlation matrix.
        axs[0].imshow(sorted_neu_corr, cmap=cmap)
        axs[0].set_xlabel('neuron id')
        axs[0].set_ylabel('neuron id')
        axs[0].spines['left'].set_visible(False)
        axs[0].spines['right'].set_visible(False)
        axs[0].spines['top'].set_visible(False)
        axs[0].spines['bottom'].set_visible(False)
        axs[0].set_xticks([])
        axs[0].set_yticks([])
        # plot stimulus.
        upper = n_clusters
        lower = 0
        for i in range(3):
            axs[1].fill_between(
                stim_seq[i,:],
                lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
                color=[color1, color2][normal], alpha=0.15, step='mid')
        # plot neural traces averaged within clusters.
        for i in range(n_clusters):
            a = 1 / (np.nanmax(neu_mean[i,:]) - np.nanmin(neu_mean[i,:]))
            b = - np.nanmin(neu_mean[i,:]) / (np.nanmax(neu_mean[i,:]) - np.nanmin(neu_mean[i,:]))
            self.plot_mean_sem(
                axs[1], self.alignment['neu_time'],
                (a*neu_mean[i,:]+b)+n_clusters-i-1, np.abs(a)*neu_sem[i,:],
                [color1, color2][normal], None)
        axs[1].tick_params(axis='y', tick1On=False)
        axs[1].spines['left'].set_visible(False)
        axs[1].spines['right'].set_visible(False)
        axs[1].spines['top'].set_visible(False)
        axs[1].set_yticks([])
        axs[1].set_xlabel('time since stim (ms)')
        axs[1].set_ylabel('df/f (z-scored)')
        axs[1].set_ylim([-0.1, n_clusters+0.1])
        # plot clustering metrics.
        for i in range(4):
            axs[2].plot(metrics['n_clusters'], metrics[metric_lbl[i]], color=colors[i], label=metric_lbl[i])
        axs[2].tick_params(axis='y', tick1On=False)
        axs[2].spines['right'].set_visible(False)
        axs[2].spines['top'].set_visible(False)
        axs[2].set_xlabel('n_clusters')
        axs[2].set_ylabel('normalized value')
        axs[2].set_xlim(2, max_clusters+2)
        axs[2].legend(loc='upper right')
        # plot cross cluster correlations.
        mask = np.tril(np.ones_like(cluster_corr, dtype=bool), k=0)
        masked_corr = np.where(mask, cluster_corr, np.nan)
        axs[3].matshow(masked_corr, interpolation='nearest', cmap=cmap)
        axs[3].tick_params(bottom=False, top=False, labelbottom=True, labeltop=False)
        axs[3].tick_params(tick1On=False)
        axs[3].spines['left'].set_visible(False)
        axs[3].spines['right'].set_visible(False)
        axs[3].spines['top'].set_visible(False)
        axs[3].spines['bottom'].set_visible(False)
        axs[3].set_xticks(np.arange(n_clusters))
        axs[3].set_yticks(np.arange(n_clusters))
        axs[2].set_xlabel('cluster id')
        axs[2].set_ylabel('cluster id')
        for i in range(cluster_corr.shape[0]):
            for j in range(cluster_corr.shape[1]):
                if not np.isnan(masked_corr[i, j]):
                    axs[3].text(j, i, f'{masked_corr[i, j]:.2f}',
                                ha='center', va='center', color='grey')
        
    def plot_select(self, axs, normal, fix_jitter, cate):
        colors = ['cornflowerblue', 'mediumseagreen', 'hotpink', 'coral']
        lbl = ['img#1', 'img#2', 'img#3', 'img#4']
        win_eval = [-500,500]
        l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, win_eval[0], win_eval[1])
        color0, color1, color2, _ = get_roi_label_color([cate], 0)
        neu_cate = [
            self.alignment['list_neu_seq'][i][:,(self.list_labels[i]==cate)*self.list_significance[i]['r_normal'],:]
            for i in range(self.n_sess)]
        # collect data.
        neu_x = []
        for img_id in [2,3,4,5]:
            neu, _, _, _, _ = get_multi_sess_neu_trial_average(
                self.list_stim_labels, neu_cate, self.alignment,
                trial_param=[[img_id], [normal], [fix_jitter], None, [0]])
            neu = neu[~np.isnan(np.sum(neu,axis=1)),:]
            neu_x.append(np.expand_dims(neu,axis=2))
        neu_x = np.concatenate(neu_x, axis=2)
        _, _, stim_seq, _, _ = get_multi_sess_neu_trial_average(
            self.list_stim_labels, neu_cate, self.alignment,
            trial_param=[[2,3,4,5], [normal], [fix_jitter], None, [0]])
        # find preferred stimulus from maximum response.
        neu_x_evoke = np.nanmean(neu_x[:,l_idx:r_idx,:], axis=1)
        prefer_idx = np.argmax(neu_x_evoke, axis=1)
        neu_mean = []
        neu_sem = []
        for ai in range(4):
            for img_id in range(3):
                m, s = get_mean_sem(neu_x[np.where(prefer_idx==img_id)[0],:,ai])
                neu_mean.append(m)
                neu_sem.append(s)
        # compute bounds.
        upper = np.nanmax(neu_mean) + np.nanmax(neu_sem)
        lower = np.nanmin(neu_mean) - np.nanmax(neu_sem)
        # plot stimulus.
        for ai in range(4):
            for i in range(3):
                axs[ai].fill_between(
                    stim_seq[i,:],
                    lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
                    color=color0, alpha=0.15, step='mid')
        # plot neural traces.
        for ai in range(4):
            for img_id in range(4):
                m, s = get_mean_sem(neu_x[np.where(prefer_idx==img_id)[0],:,ai])
                self.plot_mean_sem(axs[ai], self.alignment['neu_time'], m, s, colors[img_id], lbl[img_id])
            adjust_layout_neu(axs[ai])
            axs[ai].set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
            axs[ai].set_xlabel('time since stim (ms)')
    
    def plot_select_pie(self, ax, normal, fix_jitter, cate):
        colors = ['cornflowerblue', 'mediumseagreen', 'hotpink', 'coral']
        win_eval = [-500,500]
        l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, win_eval[0], win_eval[1])
        _, color1, color2, _ = get_roi_label_color([cate], 0)
        neu_cate = [
            self.alignment['list_neu_seq'][i][:,(self.list_labels[i]==cate)*self.list_significance[i]['r_normal'],:]
            for i in range(self.n_sess)]
        # collect data.
        neu_x = []
        for img_id in [2,3,4,5]:
            neu, _, _, _, _ = get_multi_sess_neu_trial_average(
                self.list_stim_labels, neu_cate, self.alignment,
                trial_param=[[img_id], [normal], [fix_jitter], None, [0]])
            neu = neu[~np.isnan(np.sum(neu,axis=1)),:]
            neu_x.append(np.expand_dims(neu,axis=2))
        neu_x = np.concatenate(neu_x, axis=2)
        _, _, stim_seq, _, _ = get_multi_sess_neu_trial_average(
            self.list_stim_labels, neu_cate, self.alignment,
            trial_param=[[2,3,4,5], [normal], [fix_jitter], None, [0]])
        # find preferred stimulus from maximum response.
        neu_x_evoke = np.nanmean(neu_x[:,l_idx:r_idx,:], axis=1)
        prefer_idx = np.argmax(neu_x_evoke, axis=1)
        # plot percentage.
        prefer1 = np.sum(prefer_idx==0)
        prefer2 = np.sum(prefer_idx==1)
        prefer3 = np.sum(prefer_idx==2)
        prefer4 = np.sum(prefer_idx==3)
        ax.pie(
            [prefer1, prefer2, prefer3, prefer4],
            labels=['{} img#1'.format(prefer1),
                    '{} img#2'.format(prefer2),
                    '{} img#3'.format(prefer3),
                    '{} img#4'.format(prefer4)],
            colors=colors,
            autopct='%1.1f%%',
            wedgeprops={'linewidth': 1, 'edgecolor':'white'})
    
    def plot_select_box(self, ax, normal, fix_jitter, cate):
        win_base = [-1500,0]
        win_eval = [-500,500]
        offsets = [0.0, 0.1]
        l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, win_eval[0], win_eval[1])
        color0, color1, color2, _ = get_roi_label_color([cate], 0)
        neu_cate = [
            self.alignment['list_neu_seq'][i][:,(self.list_labels[i]==cate)*self.list_significance[i]['r_normal'],:]
            for i in range(self.n_sess)]
        # collect data.
        neu_x = []
        for img_id in [2,3,4,5]:
            neu, _, _, _, _ = get_multi_sess_neu_trial_average(
                self.list_stim_labels, neu_cate, self.alignment,
                trial_param=[[img_id], [normal], [fix_jitter], None, [0]])
            neu = neu[~np.isnan(np.sum(neu,axis=1)),:]
            neu_x.append(np.expand_dims(neu,axis=2))
        neu_x = np.concatenate(neu_x, axis=2)
        # find preferred stimulus from maximum response.
        neu_x_evoke = np.nanmean(neu_x[:,l_idx:r_idx,:], axis=1)
        prefer_idx = np.argmax(neu_x_evoke, axis=1)
        neu_hate = []
        neu_like = []
        for img_id in range(4):
            for i in range(4):
                if img_id==i:
                    neu_like.append(neu_x[np.where(prefer_idx==img_id)[0],:,i])
                else:
                    neu_hate.append(neu_x[np.where(prefer_idx==img_id)[0],:,i])
        neu_hate = np.concatenate(neu_hate, axis=0)
        neu_like = np.concatenate(neu_like, axis=0)
        # plot errorbar.
        self.plot_win_mag_box(ax, neu_hate, self.alignment['neu_time'], win_base, color0, 0, offsets[0])
        self.plot_win_mag_box(ax, neu_like, self.alignment['neu_time'], win_base, color2, 0, offsets[1])
        ax.plot([], color=color0, label='hate')
        ax.plot([], color=color2, label='like')
        ax.legend(loc='upper right')

    def plot_select_decode_num_neu(self, ax, normal, fix_jitter, cate):
        num_step = 5
        n_decode = 1
        win_eval = [-300,500]
        l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, win_eval[0], win_eval[1])
        color0, color1, color2, _ = get_roi_label_color([cate], 0)
        neu_cate = [
            self.alignment['list_neu_seq'][i][:,(self.list_labels[i]==cate)*self.list_significance[i]['r_normal'],:]
            for i in range(self.n_sess)]
        # collect data.
        neu_x = []
        for img_id in [2,3,4,5]:
            neu, _, _, _ = get_multi_sess_neu_trial_average(
                self.list_stim_labels, neu_cate, self.alignment,
                trial_param=[[img_id], [normal], [fix_jitter], None, [0]],
                mean_sem=False)
            neu_x.append(neu)
        # organize data.
        neu_y = [np.concatenate([
            np.ones(neu_x[i][s].shape[0])*i
            for i in range(4)], axis=0) for s in range(self.n_sess)]
        neu_x = [np.concatenate([
            neu_x[i][s][:,:,l_idx:r_idx]
            for i in range(4)], axis=0) for s in range(self.n_sess)]
        # run decoding.
        sampling_nums, acc_model, acc_chance = multi_sess_decoding_num_neu(
            neu_x, neu_y, num_step, n_decode)
        # plot results.
        self.plot_multi_sess_decoding_num_neu(
            ax, sampling_nums, acc_model, acc_chance, color0, color2)
        ax.set_xlim([sampling_nums[0]-num_step, sampling_nums[-1]+num_step])
        ax.set_ylabel('ACC \n window [{},{}] ms'.format(win_eval[0], win_eval[1]))
    
    def plot_select_features(self, ax, normal, fix_jitter, cate):
        d_latent = 2
        colors = ['cornflowerblue', 'mediumseagreen', 'hotpink', 'coral']
        lbl = ['img#1', 'img#2', 'img#3', 'img#4']
        win_eval = [-200,400]
        l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, win_eval[0], win_eval[1])
        _, color1, color2, _ = get_roi_label_color([cate], 0)
        neu_cate = [
            self.alignment['list_neu_seq'][i][:,(self.list_labels[i]==cate)*self.list_significance[i]['r_normal'],:]
            for i in range(self.n_sess)]
        # collect data.
        neu_x = []
        for img_id in [2,3,4,5]:
            neu, _, _, _, _ = get_multi_sess_neu_trial_average(
                self.list_stim_labels, neu_cate, self.alignment,
                trial_param=[[img_id], [normal], [fix_jitter], None, [0]])
            neu = neu[~np.isnan(np.sum(neu,axis=1)),:]
            neu_x.append(np.expand_dims(neu,axis=2))
        neu_x = np.concatenate(neu_x, axis=2)
        # find preferred stimulus from maximum response.
        neu_x_evoke = np.nanmean(neu_x[:,l_idx:r_idx,:], axis=1)
        prefer_idx = np.argmax(neu_x_evoke, axis=1)
        # fit model.
        model = TSNE(n_components=d_latent)
        neu_z = model.fit_transform(neu_x_evoke)
        # plot results.
        for i in range(4):
            ax.scatter(neu_z[prefer_idx==i,0], neu_z[prefer_idx==i,1], color=colors[i], label=lbl[i])
        ax.tick_params(tick1On=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel('dim 1')
        ax.set_ylabel('dim 2')
        ax.legend(loc='upper left')

    def plot_select_decode_slide_win(self, ax, normal, fix_jitter, cate):
        win_step = 3
        n_decode = 1
        win_width = 500
        l_idx, r_idx = get_frame_idx_from_time(
            self.alignment['neu_time'], 0, 0, win_width)
        num_frames = r_idx - l_idx
        win_range = [-1500, 2500]
        start_idx, end_idx = get_frame_idx_from_time(
            self.alignment['neu_time'], 0, win_range[0], win_range[1])
        color0, color1, color2, _ = get_roi_label_color([cate], 0)
        neu_cate = [
            self.alignment['list_neu_seq'][i][:,(self.list_labels[i]==cate)*self.list_significance[i]['r_normal'],:]
            for i in range(self.n_sess)]
        # collect data.
        neu_x = []
        for img_id in [2,3,4,5]:
            neu, _, _, _ = get_multi_sess_neu_trial_average(
                self.list_stim_labels, neu_cate, self.alignment,
                trial_param=[[img_id], [normal], [fix_jitter], None, [0]],
                mean_sem=False)
            neu_x.append(neu)
        _, _, stim_seq, _, _ = get_multi_sess_neu_trial_average(
            self.list_stim_labels, neu_cate, self.alignment,
            trial_param=[[2,3,4,5], [normal], [fix_jitter], None, [0]])
        # organize data.
        neu_y = [np.concatenate([
            np.ones(neu_x[i][s].shape[0])*i
            for i in range(4)], axis=0) for s in range(self.n_sess)]
        neu_x = [np.concatenate([
            neu_x[i][s]
            for i in range(4)], axis=0) for s in range(self.n_sess)]
        # run decoding.
        acc_model, acc_chance = multi_sess_decoding_slide_win(
            neu_x, neu_y, start_idx, end_idx, win_step, n_decode, num_frames)
        # compute bounds.
        upper = np.nanmax([acc_model, acc_chance])
        lower = np.nanmin([acc_model, acc_chance])
        # plot stimulus.
        for i in range(3):
            ax.fill_between(
                stim_seq[i,:],
                lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
                color=color0, alpha=0.15, step='mid')
        # plot results.
        eval_time = self.alignment['neu_time'][
            np.arange(start_idx, end_idx, win_step)]
        self.plot_multi_sess_decoding_slide_win(
            ax, eval_time, acc_model, acc_chance, color0, color2)
        ax.set_xlabel('time since stim (ms)')
        ax.set_ylabel('ACC')
        ax.set_xlim([self.alignment['neu_time'][0], self.alignment['neu_time'][-1]])
            
    def plot_change_prepost(self, ax, normal, fix_jitter, cate=None, roi_id=None):
        if cate != None:
            color0, color1, color2, _ = get_roi_label_color([cate], 0)
            neu_cate = [
                self.alignment['list_neu_seq'][i][:,(self.list_labels[i]==cate)*self.list_significance[i]['r_normal'],:]
                for i in range(self.n_sess)]
        if roi_id != None:
            _, color1, color2, _ = get_roi_label_color(self.labels, roi_id)
            neu_cate = [np.expand_dims(self.alignment['list_neu_seq'][0][:,roi_id,:], axis=1)]
        # get pre and post indice.
        list_idx_pre  = [get_change_prepost_idx(sl)[0] for sl in self.list_stim_labels]
        list_idx_post = [get_change_prepost_idx(sl)[1] for sl in self.list_stim_labels]
        # collect data.
        neu_pre, _, _, _, _ = get_multi_sess_neu_trial_average(
            self.list_stim_labels, neu_cate, self.alignment,
            trial_idx=list_idx_pre,
            trial_param=[[2,3,4,5], [normal], [fix_jitter], None, [0]])
        neu_post, _, stim_seq, stim_value, _ = get_multi_sess_neu_trial_average(
            self.list_stim_labels, neu_cate, self.alignment,
            trial_idx=list_idx_post,
            trial_param=[[-2,-3,-4,-5], [normal], [fix_jitter], None, [0]])
        m_pre,  s_pre  = get_mean_sem(neu_pre)
        m_post, s_post = get_mean_sem(neu_post)
        # compute bounds.
        upper = np.nanmax([m_pre, m_post]) + np.nanmax([s_pre, s_post])
        lower = np.nanmin([m_pre, m_post]) - np.nanmax([s_pre, s_post])
        # plot stimulus.
        ax.fill_between(
            stim_seq[1,:],
            lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
            color=color0, alpha=0.15, step='mid')
        # plot neural traces.
        self.plot_mean_sem(ax, self.alignment['neu_time'], m_pre,  s_pre,  color1, 'pre')
        self.plot_mean_sem(ax, self.alignment['neu_time'], m_post, s_post, color2, 'post')
        adjust_layout_neu(ax)
        ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
        ax.set_xlabel('time since image change (ms)')

    def plot_change_decode_num_neu(self, ax, normal, fix_jitter, cate=None):
        num_step = 5
        n_decode = 1
        win_eval = [-300,500]
        l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, win_eval[0], win_eval[1])
        color0, color1, color2, _ = get_roi_label_color([cate], 0)
        neu_cate = [
            self.alignment['list_neu_seq'][i][:,(self.list_labels[i]==cate)*self.list_significance[i]['r_normal'],:]
            for i in range(self.n_sess)]
        # get pre and post indice.
        list_idx_pre  = [get_change_prepost_idx(sl)[0] for sl in self.list_stim_labels]
        list_idx_post = [get_change_prepost_idx(sl)[1] for sl in self.list_stim_labels]
        # collect data.
        neu_pre, _, _, _ = get_multi_sess_neu_trial_average(
            self.list_stim_labels, neu_cate, self.alignment,
            trial_idx=list_idx_pre,
            trial_param=[[2,3,4,5], [normal], [fix_jitter], None, [0]],
            mean_sem=False)
        neu_post, _, _, _ = get_multi_sess_neu_trial_average(
            self.list_stim_labels, neu_cate, self.alignment,
            trial_idx=list_idx_post,
            trial_param=[[-2,-3,-4,-5], [normal], [fix_jitter], None, [0]],
            mean_sem=False)
        # combine data matrix.
        neu_x = [np.concatenate([neu_pre[i], neu_post[i]], axis=0)
                 for i in range(self.n_sess)]
        neu_y = [np.concatenate([np.ones(neu_pre[i].shape[0])*0, np.ones(neu_post[i].shape[0])*1], axis=0)
                 for i in range(self.n_sess)]
        # run decoding.
        sampling_nums, acc_model, acc_chance = multi_sess_decoding_num_neu(
            neu_x, neu_y, num_step, n_decode)
        self.plot_multi_sess_decoding_num_neu(
                ax, sampling_nums, acc_model, acc_chance, color0, color2)
        ax.set_xlim([sampling_nums[0]-num_step, sampling_nums[-1]+num_step])

    def plot_change_latent(self, ax, normal, fix_jitter, cate=None, roi_id=None):
        d_latent = 2
        colors = ['cornflowerblue', 'mediumseagreen', 'hotpink', 'coral']
        lbl = ['img#1', 'img#2', 'img#3', 'img#4']
        if cate != None:
            _, color1, color2, _ = get_roi_label_color([cate], 0)
            neu_cate = [
                self.alignment['list_neu_seq'][i][:,(self.list_labels[i]==cate)*self.list_significance[i]['r_normal'],:]
                for i in range(self.n_sess)]
        if roi_id != None:
            _, color1, color2, _ = get_roi_label_color(self.labels, roi_id)
            neu_cate = [np.expand_dims(self.alignment['list_neu_seq'][0][:,roi_id,:], axis=1)]
        # collect data.
        list_idx_post = [get_change_prepost_idx(sl)[1] for sl in self.list_stim_labels]
        neu_x = []
        for img_id in [2,3,4,5]:
            neu, _, stim_seq, _, _ = get_multi_sess_neu_trial_average(
                self.list_stim_labels, neu_cate, self.alignment,
                trial_idx=list_idx_post,
                trial_param=[[-img_id], [normal], [fix_jitter], None, [0]])
            neu = neu[~np.isnan(np.sum(neu,axis=1)),:]
            neu_x.append(np.expand_dims(neu,axis=2))
        neu_x = np.concatenate(neu_x, axis=2)
        # fit model.
        model = TSNE(n_components=d_latent)
        neu_z = model.fit_transform(
            neu_x.reshape(neu_x.shape[0],-1).T
            ).T.reshape(d_latent, -1, 4)
        # plot dynamics.
        for i in range(4):
            # trajaectory.
            cmap = LinearSegmentedColormap.from_list(lbl[i], ['white', colors[i]])
            for t in range(neu_z.shape[1]-1):
                c = cmap(np.linspace(0, 1, neu_z.shape[1]))
                ax.plot(neu_z[0,t:t+2,i], neu_z[1,t:t+2,i], color=c[t,:])
            # end point.
            ax.scatter(neu_z[0,-1,i], neu_z[1,-1,i], color=colors[i], label='to '+lbl[i])
        # iamge change.
        l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, stim_seq[1,0], stim_seq[1,1])
        for i in range(4):
            ax.plot(neu_z[0,l_idx:r_idx,i], neu_z[1,l_idx:r_idx,i], linestyle=':', color='black')
        ax.plot([], linestyle=':', color='black', label='change')
        ax.tick_params(tick1On=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel('latent 1')
        ax.set_ylabel('latent 2')
        ax.legend(loc='upper left')

# colors = ['#989A9C', '#A4CB9E', '#9DB4CE', '#EDA1A4', '#F9C08A']
class plotter_VIPTD_G8_align_stim(plotter_utils):
    def __init__(self, neural_trials, labels, significance):
        super().__init__(neural_trials, labels, significance)

    def normal_exc(self, axs):
        
        self.plot_normal(axs[0], [0], 0, cate=-1)
        axs[0].set_title('response to normal \n excitatory')
        
        self.plot_normal_spectral(axs[1], 0, 0, cate=-1)
        axs[1].set_title('response spectrogram to normal \n excitatory')
        
        self.plot_normal_sync(axs[2], 0, 0, cate=-1)
        axs[2].set_title('synchronization level to normal \n excitatory')
        
    def normal_inh(self, axs):
        
        self.plot_normal(axs[0], [0], 0, cate=1)
        axs[0].set_title('response to normal \n inhibitory')
        
        self.plot_normal_spectral(axs[1], 0, 0, cate=1)
        axs[1].set_title('response spectrogram to normal \n inhibitory')
        
        self.plot_normal_sync(axs[2], 0, 0, cate=1)
        axs[2].set_title('synchronization level to normal \n inhibitory')

    def normal_heatmap(self, axs):
        win_sort = [-200, 1000]
        labels = np.concatenate(self.list_labels)
        sig = np.concatenate([self.list_significance[n]['r_normal'] for n in range(self.n_sess)])
        neu_short_fix, _, _, _, _ = get_multi_sess_neu_trial_average(
            self.list_stim_labels, self.alignment['list_neu_seq'], self.alignment,
            trial_param=[[2,3,4,5], [0], [0], None, [0]])
        axs[0].set_xlabel('time since stim (ms)')
        
        self.plot_heatmap_neuron(
            axs[0], neu_short_fix, self.alignment['neu_time'], neu_short_fix,
            win_sort, labels, sig)
        axs[0].set_title('response to normal')
    
    def normal_mode(self, axs):
        
        self.plot_normal_cluster([axs[0], axs[1], axs[2], axs[3]], 0, 0, cate=-1)
        axs[0].set_title('sorted correlation matrix \n excitatory')
        axs[1].set_title('clustering response \n excitatory')
        axs[2].set_title('clustering evaluation \n excitatory')
        axs[3].set_title('cross cluster correlation \n excitatory')
        
        self.plot_normal_cluster([axs[4], axs[5], axs[6], axs[7]], 0, 0, cate=1)
        axs[4].set_title('sorted correlation matrix \n inhibitory')
        axs[5].set_title('clustering response \n inhibitory')
        axs[6].set_title('clustering evaluation \n inhibitory')
        axs[7].set_title('cross cluster correlation \n inhibitory')

    def select_exc(self, axs):
        lbl = ['img#1', 'img#2', 'img#3', 'img#4']
        
        self.plot_select([axs[i] for i in range(4)], 0, 0, cate=-1)
        for i in range(4):
            axs[i].set_title('response to normal \n excitatory prefer {}'.format(lbl[i]))
        
        self.plot_select_pie(axs[4], 0, 0, cate=-1)
        axs[4].set_title('percentage of preferred stimlus \n excitatory')
        
        self.plot_select_box(axs[5], 0, 0, cate=-1)
        axs[5].set_title('presponse to preferred stimlus \n excitatory')
        
        self.plot_select_decode_num_neu(axs[6], 0, 0, cate=-1)
        axs[6].set_title('single trial decoding accuracy for images \n with number of neurons \n excitatory')

        self.plot_select_features(axs[7], 0, 0, cate=-1)
        axs[7].set_title('response features projection with image preference \n excitatory')
        
        self.plot_select_decode_slide_win(axs[8], 0, 0, cate=-1)
        axs[8].set_title('single trial decoding accuracy for images \n with sliding window \n excitatory')

        self.plot_change_prepost(axs[9], 0, 0, cate=-1)
        axs[9].set_title('response to change \n excitatory')
        
        self.plot_change_decode_num_neu(axs[10], 0, 0, cate=-1)
        axs[10].set_title('single trial decoding accuracy for pre&post change \n with number of neurons \n excitatory')
        
        self.plot_change_latent(axs[11], 0, 0, cate=-1)
        axs[11].set_title('latent dynamics response to change \n excitatory')

    def select_inh(self, axs):
        lbl = ['img#1', 'img#2', 'img#3', 'img#4']

        self.plot_select([axs[i] for i in range(4)], [0], 0, cate=1)
        for i in range(4):
            axs[i].set_title('response to normal \n inhibitory prefer {}'.format(lbl[i]))

        self.plot_select_pie(axs[4], 0, 0, cate=1)
        axs[4].set_title('percentage of preferred stimlus \n inhibitory')
        
        self.plot_select_box(axs[5], 0, 0, cate=-1)
        axs[5].set_title('presponse to preferred stimlus \n inhibitory')
        
        self.plot_select_decode_num_neu(axs[6], 0, 0, cate=1)
        axs[6].set_title('decoding accuracy for images VS number of neurons \n inhibitory')
        
        self.plot_select_features(axs[7], 0, 0, cate=1)
        axs[7].set_title('response features projection with image preference \n inhibitory')
        
        self.plot_select_decode_slide_win(axs[8], 0, 0, cate=1)
        axs[8].set_title('decoding accuracy for images with sliding window \n inhibitory')
        
        self.plot_change_prepost(axs[9], 0, 0, cate=1)
        axs[9].set_title('response to change \n inhibitory')
        
        self.plot_change_decode_num_neu(axs[10], 0, 0, cate=1)
        axs[10].set_title('decoding accuracy for pre&post change VS number of neurons \n excitatory')
        
        self.plot_change_latent(axs[11], 0, 0, cate=1)
        axs[11].set_title('latent dynamics response to change \n inhibitory')
