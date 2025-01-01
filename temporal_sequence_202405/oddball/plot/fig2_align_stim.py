#!/usr/bin/env python3

import numpy as np
from sklearn.manifold import TSNE
from scipy.signal import cwt, ricker
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from modules.Alignment import run_get_stim_response
from modeling.decoding import multi_sess_decoding_num_neu
from modeling.decoding import multi_sess_decoding_slide_win
from plot.utils import norm01
from plot.utils import exclude_odd_stim
from plot.utils import get_frame_idx_from_time
from plot.utils import get_change_prepost_idx
from plot.utils import get_mean_sem
from plot.utils import get_neu_sync
from plot.utils import get_multi_sess_neu_trial_average
from plot.utils import get_roi_label_color
from plot.utils import get_epoch_idx
from plot.utils import adjust_layout_neu
from plot.utils import adjust_layout_spectral
from plot.utils import adjust_layout_3d_latent
from plot.utils import add_legend
from plot.utils import utils

# fig, ax = plt.subplots(1, 1, figsize=(6, 6))
# fig, ax = plt.subplots(1, 1, figsize=(6, 6), subplot_kw={"projection": "3d"})

class plotter_utils(utils):

    def __init__(
            self,
            list_neural_trials, list_labels, list_significance
            ):
        super().__init__()
        timescale = 1.0
        self.n_sess = len(list_neural_trials)
        self.l_frames = int(100*timescale)
        self.r_frames = int(120*timescale)
        self.list_stim_labels = [
            nt['stim_labels'][2:-2,:] for nt in list_neural_trials]
        self.list_stim_labels = [
            exclude_odd_stim(sl) for sl in self.list_stim_labels]
        self.list_labels = list_labels
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
        for i in range(stim_seq_short.shape[0]):
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

    def plot_normal_latent(self, ax, normal, fix_jitter, cate):
        d_latent = 3
        _, _, _, cmap = get_roi_label_color([cate], 0)
        neu_cate = [
            self.alignment['list_neu_seq'][i][:,(self.list_labels[i]==cate)*self.list_significance[i]['r_normal'],:]
            for i in range(self.n_sess)]
        # collect data.
        neu_x, _, stim_seq, _, _ = get_multi_sess_neu_trial_average(
            self.list_stim_labels, neu_cate, self.alignment,
            trial_param=[[2,3,4,5], [0], [fix_jitter], None, [0]])
        # fit model.
        model = TSNE(n_components=d_latent)
        neu_z = model.fit_transform(
            neu_x.reshape(neu_x.shape[0],-1).T
            ).T.reshape(d_latent, -1)
        # plot dynamics.
        for t in range(neu_z.shape[1]-1):
            c = cmap(np.linspace(0, 1, neu_z.shape[1]))
            # trajaectory.
            ax.plot(neu_z[0,t:t+2], neu_z[1,t:t+2], neu_z[2,t:t+2], color=c[t,:])
        # end point.
        ax.scatter(neu_z[0,0], neu_z[1,0], neu_z[2,0], color='black', marker='x')
        ax.scatter(neu_z[0,-1], neu_z[1,-1], neu_z[2,-1], color='black', marker='o')
        # stimulus.
        for i in range(stim_seq.shape[0]):
            l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, stim_seq[i,0], stim_seq[i,1])
            ax.plot(neu_z[0,l_idx:r_idx], neu_z[1,l_idx:r_idx], neu_z[2,l_idx:r_idx], lw=4, color='grey')
        adjust_layout_3d_latent(ax, neu_z, cmap, self.alignment['neu_time'], 'time since stim (ms)')
        add_legend(ax, ['grey'], ['stim'], 'upper right', dim=3)

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
        for i in range(stim_seq_short.shape[0]):
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
            for i in range(stim_seq.shape[0]):
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
        for ai in range(4):
            add_legend(axs[ai], colors, lbl, 'upper right')

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
            wedgeprops={'linewidth': 1, 'edgecolor':'white', 'width':0.2})

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
        add_legend(ax, [color0,color2], ['non-prefer','prefer'], 'upper right')

    def plot_select_decode_num_neu(self, ax, normal, fix_jitter, cate):
        mode = 'spatial'
        num_step = 5
        n_decode = 10
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
            neu_x, neu_y, num_step, n_decode, mode)
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
        add_legend(ax, colors, lbl, 'upper left')

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
        for i in range(stim_seq.shape[0]):
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
            stim_seq[int(stim_seq.shape[0]/2),:],
            lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
            color=color0, alpha=0.15, step='mid')
        # plot neural traces.
        self.plot_mean_sem(ax, self.alignment['neu_time'], m_pre,  s_pre,  color1, 'pre')
        self.plot_mean_sem(ax, self.alignment['neu_time'], m_post, s_post, color2, 'post')
        adjust_layout_neu(ax)
        ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
        ax.set_xlabel('time since image change (ms)')
        add_legend(ax, [color1,color2], ['pre','post'], 'upper left')

    def plot_change_decode_num_neu(self, ax, normal, fix_jitter, cate=None):
        mode = 'temporal'
        num_step = 5
        n_decode = 10
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
            neu_x, neu_y, num_step, n_decode, mode)
        self.plot_multi_sess_decoding_num_neu(
                ax, sampling_nums, acc_model, acc_chance, color0, color2)
        ax.set_xlim([sampling_nums[0]-num_step, sampling_nums[-1]+num_step])

    def plot_change_latent(self, ax, normal, fix_jitter, cate=None):
        d_latent = 3
        colors = ['cornflowerblue', 'mediumseagreen', 'hotpink', 'coral', 'grey']
        lbl = ['to img#1', 'to img#2', 'to img#3', 'to img#4', 'change']
        _, color1, color2, _ = get_roi_label_color([cate], 0)
        neu_cate = [
            self.alignment['list_neu_seq'][i][:,(self.list_labels[i]==cate)*self.list_significance[i]['r_normal'],:]
            for i in range(self.n_sess)]
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
            cmap = LinearSegmentedColormap.from_list(lbl[i], ['white', colors[i], 'black'])
            for t in range(neu_z.shape[1]-1):
                c = cmap(np.linspace(0, 1, neu_z.shape[1]))
                ax.plot(neu_z[0,t:t+2,i], neu_z[1,t:t+2,i], neu_z[2,t:t+2,i], color=c[t,:])
            # end point.
            ax.scatter(neu_z[0,0,i], neu_z[1,0,i], neu_z[2,0,i], color='black', marker='x')
            ax.scatter(neu_z[0,-1,i], neu_z[1,-1,i], neu_z[2,-1,i], color='black', marker='o')
        # image change.
        l_idx, r_idx = get_frame_idx_from_time(
            self.alignment['neu_time'], 0,
            stim_seq[int(stim_seq.shape[0]/2),0], stim_seq[int(stim_seq.shape[0]/2),1])
        for i in range(4):
            ax.plot(neu_z[0,l_idx:r_idx,i], neu_z[1,l_idx:r_idx,i], neu_z[2,l_idx:r_idx,i], lw=4, color=colors[-1])
        cmap = LinearSegmentedColormap.from_list('', ['white', 'black'])
        adjust_layout_3d_latent(ax, neu_z, cmap, self.alignment['neu_time'], 'time since stim change (ms)')
        add_legend(ax, colors, lbl, 'upper right', dim=3)

# colors = ['#989A9C', '#A4CB9E', '#9DB4CE', '#EDA1A4', '#F9C08A']
class plotter_main_align_stim(plotter_utils):
    def __init__(self, neural_trials, labels, significance, label_names):
        super().__init__(neural_trials, labels, significance)
        self.label_names = label_names

    def normal_exc(self, axs):
        try:
            cate = -1
            label_name = self.label_names[str(cate)]

            self.plot_normal(axs[0], [0], 0, cate=cate)
            axs[0].set_title(f'response to normal \n {label_name}')

            self.plot_normal_spectral(axs[1], 0, 0, cate=cate)
            axs[1].set_title(f'response spectrogram to normal \n {label_name}')

            self.plot_normal_sync(axs[2], 0, 0, cate=cate)
            axs[2].set_title(f'synchronization level to normal \n {label_name}')

            self.plot_normal_latent(axs[3], 0, 0, cate=cate)
            axs[3].set_title(f'response to normal \n {label_name}')

        except:
            pass

    def normal_inh(self, axs):
        try:
            cate = 1
            label_name = self.label_names[str(cate)]

            self.plot_normal(axs[0], [0], 0, cate=cate)
            axs[0].set_title(f'response to normal \n {label_name}')

            self.plot_normal_spectral(axs[1], 0, 0, cate=cate)
            axs[1].set_title(f'response spectrogram to normal \n {label_name}')

            self.plot_normal_sync(axs[2], 0, 0, cate=cate)
            axs[2].set_title(f'synchronization level to normal \n {label_name}')

            self.plot_normal_latent(axs[3], 0, 0, cate=cate)
            axs[3].set_title(f'response spectrogram to normal \n {label_name}')

        except:
            pass

    def normal_heatmap(self, axs):
        try:
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

        except:
            pass

    def select_exc(self, axs):
        try:
            cate = -1
            label_name = self.label_names[str(cate)]
            lbl = ['img#1', 'img#2', 'img#3', 'img#4']

            self.plot_select([axs[i] for i in range(4)], 0, 0, cate=cate)
            for i in range(4):
                axs[i].set_title(f'response to normal \n {label_name} prefer {lbl[i]}')

            self.plot_select_pie(axs[4], 0, 0, cate=cate)
            axs[4].set_title(f'percentage of preferred stimlus \n {label_name}')

            self.plot_select_box(axs[5], 0, 0, cate=cate)
            axs[5].set_title(f'presponse to preferred stimlus \n {label_name}')

            #self.plot_select_decode_num_neu(axs[6], 0, 0, cate=cate)
            axs[6].set_title(f'single trial decoding accuracy for images \n with number of neurons \n {label_name}')

            self.plot_select_features(axs[7], 0, 0, cate=cate)
            axs[7].set_title(f'response features projection with image preference \n {label_name}')

            #self.plot_select_decode_slide_win(axs[8], 0, 0, cate=cate)
            axs[8].set_title(f'single trial decoding accuracy for images \n with sliding window \n {label_name}')

            self.plot_change_prepost(axs[9], 0, 0, cate=cate)
            axs[9].set_title(f'response to change \n {label_name}')

            #self.plot_change_decode_num_neu(axs[10], 0, 0, cate=cate)
            axs[10].set_title(f'single trial decoding accuracy for pre&post change \n with number of neurons \n {label_name}')

            self.plot_change_latent(axs[11], 0, 0, cate=cate)
            axs[11].set_title(f'latent dynamics response to change \n {label_name}')

        except:
            pass

    def select_inh(self, axs):
        try:
            cate = 1
            label_name = self.label_names[str(cate)]
            lbl = ['img#1', 'img#2', 'img#3', 'img#4']

            self.plot_select([axs[i] for i in range(4)], [0], 0, cate=cate)
            for i in range(4):
                axs[i].set_title(f'response to normal \n {label_name} prefer {lbl[i]}')

            self.plot_select_pie(axs[4], 0, 0, cate=cate)
            axs[4].set_title(f'percentage of preferred stimlus \n {label_name}')

            self.plot_select_box(axs[5], 0, 0, cate=cate)
            axs[5].set_title(f'presponse to preferred stimlus \n {label_name}')

            #self.plot_select_decode_num_neu(axs[6], 0, 0, cate=cate)
            axs[6].set_title(f'decoding accuracy for images VS number of neurons \n {label_name}')

            self.plot_select_features(axs[7], 0, 0, cate=cate)
            axs[7].set_title(f'response features projection with image preference \n {label_name}')

            #self.plot_select_decode_slide_win(axs[8], 0, 0, cate=cate)
            axs[8].set_title(f'decoding accuracy for images with sliding window \n {label_name}')

            self.plot_change_prepost(axs[9], 0, 0, cate=cate)
            axs[9].set_title(f'response to change \n {label_name}')

            #self.plot_change_decode_num_neu(axs[10], 0, 0, cate=cate)
            axs[10].set_title(f'decoding accuracy for pre&post change VS number of neurons \n {label_name}')

            self.plot_change_latent(axs[11], 0, 0, cate=cate)
            axs[11].set_title(f'latent dynamics response to change \n {label_name}')

        except:
            pass
