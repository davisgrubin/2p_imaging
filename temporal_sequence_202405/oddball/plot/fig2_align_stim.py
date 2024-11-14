#!/usr/bin/env python3

import numpy as np
from scipy.signal import find_peaks
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

from modules.Alignment import run_get_stim_response
from plot.utils import norm01
from plot.utils import exclude_post_odd_stim
from plot.utils import get_bin_stat
from plot.utils import get_frame_idx_from_time
from plot.utils import get_change_prepost_idx
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

    def plot_normal_component(self, ax, normal, fix_jitter):
        _, color1, color2, _ = get_roi_label_color([0], 0)
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
                color=color1, alpha=0.15, step='mid')
        # plot latents.
        for i in range(d_latent):
            ax.plot(self.alignment['neu_time'], norm01(z[i,:])*scale+d_latent-i-1, color=color2)
        ax.tick_params(axis='y', tick1On=False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_yticks([])
        ax.set_xlabel('time since stim (ms)')
        ax.set_ylabel('df/f (z-scored)')
        ax.set_ylim([-0.1, d_latent+0.1])

    def plot_normal_cluster(self, axs, normal, fix_jitter, cate):
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
        axs[0].imshow(sorted_corr_matrix, cmap=cmap)
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
                color=color1, alpha=0.15, step='mid')
        # plot neural traces averaged within clusters.
        for i in range(n_clusters):
            axs[1].plot(self.alignment['neu_time'], norm01(neu_mean[i,:])*scale+n_clusters-i-1, color=color2)
        axs[1].tick_params(axis='y', tick1On=False)
        axs[1].spines['left'].set_visible(False)
        axs[1].spines['right'].set_visible(False)
        axs[1].spines['top'].set_visible(False)
        axs[1].set_yticks([])
        axs[1].set_xlabel('time since stim (ms)')
        axs[1].set_ylabel('df/f (z-scored)')
        axs[1].set_ylim([-0.1, n_clusters+0.1])

    def plot_normal_peak(self, ax, normal, fix_jitter, cate):
        win_conv = 15
        win_sort = [-1000,1500]
        bin_size = 100
        l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, win_sort[0], win_sort[1])
        _, color1, color2, _ = get_roi_label_color([cate], 0)
        neu_cate = [
            self.alignment['list_neu_seq'][i][:,(self.list_labels[i]==cate)*self.list_significance[i]['r_normal'],:]
            for i in range(self.n_sess)]
        # collect data.
        neu, _, stim_seq, _, _ = get_multi_sess_neu_trial_average(
            self.list_stim_labels, neu_cate, self.alignment['list_stim_seq'],
            self.alignment['list_stim_value'], self.alignment['list_pre_isi'],
            trial_param=[[2,3,4,5], [normal], [fix_jitter], None, [0]])
        neu = neu[~np.isnan(np.sum(neu,axis=1)),:]
        # compute peak timing and bin.
        smoothed_mean = np.array(
            [np.convolve(row, np.ones(win_conv)/win_conv, mode='same')
             for row in neu[:,l_idx:r_idx]])
        peak_time_max = self.alignment['neu_time'][l_idx:r_idx][np.argmax(smoothed_mean, axis=1)]
        peak_time_min = self.alignment['neu_time'][l_idx:r_idx][np.argmin(smoothed_mean, axis=1)]
        bin_max_c, bin_max_n, _, _ = get_bin_stat(peak_time_max, win_sort, bin_size)
        bin_min_c, bin_min_n, _, _ = get_bin_stat(peak_time_min, win_sort, bin_size)
        # compute bounds.
        upper = np.nanmax(np.concatenate([bin_max_n, bin_min_n]))
        lower = np.nanmin(np.concatenate([bin_max_n, bin_min_n]))
        # plot stimulus.
        ax.fill_between(
            stim_seq[1,:],
            lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
            color=color1, alpha=0.15, step='mid')
        # plot bin results.
        ax.plot(bin_max_c, bin_max_n, color=color2, label='max', alpha=1.0)
        ax.plot(bin_min_c, bin_min_n, color=color2, label='min', alpha=0.5)
        ax.tick_params(axis='y', tick1On=False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_ylabel('fraction of neurons')
        ax.legend(loc='upper right')
        ax.set_xlabel('time since stim (ms)')
        ax.set_xlim([self.alignment['neu_time'][0], self.alignment['neu_time'][-1]])
        ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
        
    def plot_change_select(self, axs, normal, fix_jitter, cate=None):
        colors = ['royalblue', 'mediumseagreen', 'violet', 'coral']
        lbl = ['img#1', 'img#2', 'img#3', 'img#4']
        win_test = [-500,500]
        l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, win_test[0], win_test[1])
        _, color1, color2, _ = get_roi_label_color([cate], 0)
        neu_cate = [
            self.alignment['list_neu_seq'][i][:,(self.list_labels[i]==cate)*self.list_significance[i]['r_normal'],:]
            for i in range(self.n_sess)]
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
        _, _, stim_seq, _, _ = get_multi_sess_neu_trial_average(
            self.list_stim_labels, neu_cate, self.alignment['list_stim_seq'],
            self.alignment['list_stim_value'], self.alignment['list_pre_isi'],
            trial_param=[[2,3,4,5], [normal], [fix_jitter], None, [0]])
        # find preferred stimulus from maximum response.
        neu_x_evoke = np.nanmean(neu_x[:,l_idx:r_idx,:], axis=1)
        prefer_idx = np.argmax(neu_x_evoke, axis=1)
        # compute bounds.
        neu_mean = []
        neu_sem = []
        for ai in range(4):
            for img_id in range(3):
                m, s = get_mean_sem(neu_x[np.where(prefer_idx==img_id)[0],:,ai])
                neu_mean.append(m)
                neu_sem.append(s)
        upper = np.nanmax(neu_mean) + np.nanmax(neu_sem)
        lower = np.nanmin(neu_mean) - np.nanmax(neu_sem)
        # plot stimulus.
        for ai in range(4):
            for i in range(3):
                axs[ai].fill_between(
                    stim_seq[i,:],
                    lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
                    color=color1, alpha=0.15, step='mid')
        # plot neural traces.
        for ai in range(4):
            for img_id in range(4):
                m, s = get_mean_sem(neu_x[np.where(prefer_idx==img_id)[0],:,ai])
                self.plot_mean_sem(axs[ai], self.alignment['neu_time'], m, s, colors[img_id], lbl[img_id])
            adjust_layout_neu(axs[ai])
            axs[ai].set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
            axs[ai].set_xlabel('time since stim (ms)')
        # plot percentage.
        prefer1 = np.sum(prefer_idx==0)
        prefer2 = np.sum(prefer_idx==1)
        prefer3 = np.sum(prefer_idx==2)
        prefer4 = np.sum(prefer_idx==3)
        axs[4].pie(
            [prefer1, prefer2, prefer3, prefer4],
            labels=['{} img#1'.format(prefer1),
                    '{} img#2'.format(prefer2),
                    '{} img#3'.format(prefer3),
                    '{} img#4'.format(prefer4)],
            colors=colors,
            autopct='%1.1f%%',
            wedgeprops={'linewidth': 1, 'edgecolor':'white'})
    
    def plot_change_prepost(self, ax, normal, fix_jitter, cate=None, roi_id=None):
        if cate != None:
            _, color1, color2, _ = get_roi_label_color([cate], 0)
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
            self.list_stim_labels, neu_cate, self.alignment['list_stim_seq'],
            self.alignment['list_stim_value'], self.alignment['list_pre_isi'],
            trial_idx=list_idx_pre,
            trial_param=[[2,3,4,5], [normal], [fix_jitter], None, [0]])
        neu_post, _, stim_seq, stim_value, _ = get_multi_sess_neu_trial_average(
            self.list_stim_labels, neu_cate, self.alignment['list_stim_seq'],
            self.alignment['list_stim_value'], self.alignment['list_pre_isi'],
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
            color='gold', alpha=0.15, step='mid')
        # plot neural traces.
        self.plot_mean_sem(ax, self.alignment['neu_time'], m_pre,  s_pre,  color1, 'pre')
        self.plot_mean_sem(ax, self.alignment['neu_time'], m_post, s_post, color2, 'post')
        adjust_layout_neu(ax)
        ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
        ax.set_xlabel('time since image change (ms)')
    
    def plot_change_decode(self, ax, normal, fix_jitter, cate=None):
        num_step = 5
        n_decode = 50
        test_size = 0.5
        win_eval = [-200,400]
        l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, win_eval[0], win_eval[1])
        _, color1, color2, _ = get_roi_label_color([cate], 0)
        neu_cate = [
            self.alignment['list_neu_seq'][i][:,(self.list_labels[i]==cate)*self.list_significance[i]['r_normal'],:]
            for i in range(self.n_sess)]
        # get pre and post indice.
        list_idx_pre  = [get_change_prepost_idx(sl)[0] for sl in self.list_stim_labels]
        list_idx_post = [get_change_prepost_idx(sl)[1] for sl in self.list_stim_labels]
        # collect data.
        neu_pre, _, _, _ = get_multi_sess_neu_trial_average(
            self.list_stim_labels, neu_cate, self.alignment['list_stim_seq'],
            self.alignment['list_stim_value'], self.alignment['list_pre_isi'],
            trial_idx=list_idx_pre,
            trial_param=[[2,3,4,5], [normal], [fix_jitter], None, [0]],
            mean_sem=False)
        neu_post, _, _, _ = get_multi_sess_neu_trial_average(
            self.list_stim_labels, neu_cate, self.alignment['list_stim_seq'],
            self.alignment['list_stim_value'], self.alignment['list_pre_isi'],
            trial_idx=list_idx_post,
            trial_param=[[-2,-3,-4,-5], [normal], [fix_jitter], None, [0]],
            mean_sem=False)
        # extract response projection for each neuron.
        neu_pre  = [np.mean(neu_pre[i][:,:,l_idx:r_idx],  axis=2) for i in range(self.n_sess)]
        neu_post = [np.mean(neu_post[i][:,:,l_idx:r_idx], axis=2) for i in range(self.n_sess)]
        # combine data matrix.
        neu_x = [np.concatenate([neu_pre[i], neu_post[i]], axis=0)
                 for i in range(self.n_sess)]
        neu_y = [np.concatenate([np.zeros(neu_pre[i].shape[0]), np.ones(neu_post[i].shape[0])], axis=0)
                 for i in range(self.n_sess)]
        # define sampling numbers.
        max_num = np.nanmax([neu_x[i].shape[1] for i in range(self.n_sess)])
        sampling_nums = np.arange(num_step, ((max_num//num_step)+1)*num_step, num_step)
        # run decoding.
        acc_model   = []
        acc_chance = []
        for n_neu in sampling_nums:
            results_model = []
            results_chance = []
            for s in range(self.n_sess):
                # not enough neurons.
                if n_neu > neu_x[s].shape[1]:
                    results_model.append(np.nan)
                    results_chance.append(np.nan)
                # random sampling n_decode times.
                else:
                    for _ in range(n_decode):
                        sub_idx = np.random.choice(neu_x[s].shape[1], n_neu, replace=False)
                        x = neu_x[s][:,sub_idx].copy()
                        y = neu_y[s].copy()
                        # reparate training and testing sets.
                        x_train, x_test, y_train, y_test = train_test_split(
                            x, y, test_size=test_size, stratify=y)
                        x_train, y_train = shuffle(x_train, y_train)
                        # model.
                        model = SVC(kernel='linear')
                        model.fit(x_train, y_train)
                        results_model.append(model.score(x_test, y_test))
                        # chance.
                        chance = SVC(kernel='linear')
                        x_shuffle, y_shuffle = shuffle(x_train, y_train)
                        chance.fit(np.random.permutation(x_train), np.random.permutation(y_train))
                        results_chance.append(chance.score(x_test, y_test))
            acc_model.append(np.array(results_model).reshape(-1,1))
            acc_chance.append(np.array(results_chance).reshape(-1,1))
        # compute mean and sem.
        acc_mean_model = np.array([get_mean_sem(a)[0] for a in acc_model]).reshape(-1)
        acc_sem_model  = np.array([get_mean_sem(a)[1] for a in acc_model]).reshape(-1)
        acc_mean_chance = np.array([get_mean_sem(a)[0] for a in acc_chance]).reshape(-1)
        acc_sem_chance  = np.array([get_mean_sem(a)[1] for a in acc_chance]).reshape(-1)
        # plot decoding results.
        self.plot_mean_sem(ax, sampling_nums, acc_mean_model,  acc_sem_model,  color2, 'model')
        self.plot_mean_sem(ax, sampling_nums, acc_mean_chance, acc_sem_chance, color1, 'chance')
        ax.tick_params(tick1On=False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.legend(loc='lower left')
        ax.set_xlabel('number of sampled neurons')
        ax.set_ylabel('ACC')
    
    def plot_change_latent(self, ax, normal, fix_jitter, cate=None, roi_id=None):
        d_latent = 2
        colors = ['royalblue', 'mediumseagreen', 'violet', 'coral']
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
                self.list_stim_labels, neu_cate, self.alignment['list_stim_seq'],
                self.alignment['list_stim_value'], self.alignment['list_pre_isi'],
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
            ax.plot(neu_z[0,:,i], neu_z[1,:,i], color=colors[i], label='to '+lbl[i])
            ax.scatter(neu_z[0,-1,i], neu_z[1,-1,i], color=colors[i])
        l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, stim_seq[1,0], stim_seq[1,1])
        for i in range(4):
            ax.plot(neu_z[0,l_idx:r_idx,i], neu_z[1,l_idx:r_idx,i], linestyle=':', color='gold')
        ax.plot([], linestyle=':', color='gold', label='change')
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
        
        self.plot_normal_peak(axs[1], [0], 0, cate=-1)
        axs[1].set_title('peak timing distribution for normal \n excitatory')
        
    def normal_inh(self, axs):
        
        self.plot_normal(axs[0], [0], 0, cate=1)
        axs[0].set_title('response to normal \n inhibitory')
        
        self.plot_normal_peak(axs[1], [0], 0, cate=1)
        axs[1].set_title('peak timing distribution for normal \n inhibitory')

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
        
        self.plot_normal_component(axs[0], 0, 0)
        axs[0].set_title('top principle components')
        
        self.plot_normal_cluster([axs[1], axs[2]], 0, 0, cate=-1)
        axs[1].set_title('sorted correlation matrix \n excitatory')
        axs[2].set_title('clustering response \n excitatory')
        
        self.plot_normal_cluster([axs[3], axs[4]], 0, 0, cate=1)
        axs[3].set_title('sorted correlation matrix \n inhibitory')
        axs[4].set_title('clustering response \n inhibitory')
        
    def change_exc(self, axs):
        lbl = ['img#1', 'img#2', 'img#3', 'img#4']
        
        self.plot_change_select([axs[i] for i in range(5)], [0], 0, cate=-1)
        for i in range(4):
            axs[i].set_title('response to normal \n excitatory prefer {}'.format(lbl[i]))
        axs[4].set_title('percentage of preferred stimlus \n excitatory')
        
        self.plot_change_prepost(axs[5], 0, 0, cate=-1)
        axs[5].set_title('response to change \n excitatory')
        
        self.plot_change_decode(axs[6], 0, 0, cate=-1)
        axs[6].set_title('decoding accuracy VS number of neurons \n excitatory')
        
        self.plot_change_latent(axs[7], 0, 0, cate=-1)
        axs[7].set_title('latent dynamics response to change \n excitatory')

    def change_inh(self, axs):
        lbl = ['img#1', 'img#2', 'img#3', 'img#4']

        self.plot_change_select([axs[i] for i in range(5)], [0], 0, cate=1)
        for i in range(4):
            axs[i].set_title('response to normal \n inhibitory prefer {}'.format(lbl[i]))
        axs[4].set_title('percentage of preferred stimlus \n inhibitory')
        
        self.plot_change_prepost(axs[5], 0, 0, cate=1)
        axs[5].set_title('response to change \n inhibitory')
        
        self.plot_change_decode(axs[6], 0, 0, cate=1)
        axs[6].set_title('decoding accuracy for pre&post change VS number of neurons \n excitatory')
        
        self.plot_change_latent(axs[7], 0, 0, cate=1)
        axs[7].set_title('latent dynamics response to change \n inhibitory')
        
        
        
        
        
        
        