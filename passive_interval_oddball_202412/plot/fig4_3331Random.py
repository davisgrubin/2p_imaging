#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from modules.Alignment import run_get_stim_response
from utils import get_bin_idx
from utils import get_mean_sem
from utils import get_base_mean_win
from utils import get_neu_trial
from utils import get_frame_idx_from_time
from utils import get_isi_bin_neu
from utils import get_neu_sync
from utils import get_cmap_color
from utils import apply_colormap
from utils import adjust_layout_neu
from utils import adjust_layout_heatmap
from utils import add_legend
from utils import utils_basic

# fig, ax = plt.subplots(1, 1, figsize=(6, 6))
# fig, ax = plt.subplots(1, 1, figsize=(6, 6), subplot_kw={"projection": "3d"})

class plotter_utils(utils_basic):

    def __init__(
            self,
            list_neural_trials, list_labels, list_significance
            ):
        super().__init__()
        timescale = 1.0
        self.bin_win = [500,2500]
        self.bin_num = 8
        self.n_sess = len(list_neural_trials)
        self.l_frames = int(100*timescale)
        self.r_frames = int(80*timescale)
        self.list_stim_labels = [
            nt['stim_labels'][2:-2,:] for nt in list_neural_trials]
        self.list_labels = list_labels
        self.alignment = run_get_stim_response(
                list_neural_trials, self.l_frames, self.r_frames, expected='none')
        self.list_significance = list_significance
    
    def plot_random_stim(self, ax, cate=None, roi_id=None):
        # collect data.
        [color0, _, color2, _], [neu_seq, _, stim_seq, stim_value, _] = get_neu_trial(
            self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
            trial_param=[[2,3,4,5], None, None, None, [1], [0]],
            cate=cate, roi_id=roi_id)
        mean_neu, sem_neu = get_mean_sem(neu_seq)
        # compute bounds.
        upper = np.nanmax(mean_neu) + np.nanmax(sem_neu)
        lower = np.nanmin(mean_neu) - np.nanmax(sem_neu)
        # plot stimulus.
        ax.fill_between(
            stim_seq[int(stim_seq.shape[0]/2),:],
            lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
            color=color0, alpha=0.15, step='mid')
        self.plot_vol(ax, self.alignment['stim_time'], stim_value.reshape(1,-1), color0, upper, lower)
        # plot neural traces.
        self.plot_mean_sem(ax, self.alignment['neu_time'], mean_neu, sem_neu, color2, None)
        # adjust layout.
        adjust_layout_neu(ax)
        ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
        ax.set_xlabel('time since stim (ms)')
        add_legend(ax, [color0, color2], ['stim', 'dff'], 'upper right')
    
    def plot_random_sync(self, ax, cate):
        win_width = 200
        l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, 0, win_width)
        win_width = r_idx - l_idx
        # collect data.
        [color0, _, color2, _], [neu_seq, _, stim_seq, stim_value, _] = get_neu_trial(
            self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
            trial_param=[[2,3,4,5], None, None, None, [1], [0]],
            cate=cate)
        # compute synchronization.
        neu_sync = get_neu_sync(neu_seq, win_width)
        # find bounds.
        upper = np.nanmax(neu_sync)
        lower = np.nanmin(neu_sync)
        # plot stimulus.
        ax.fill_between(
            stim_seq[int(stim_seq.shape[0]/2),:],
            lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
            color=color0, alpha=0.15, step='mid')
        self.plot_vol(ax, self.alignment['stim_time'], stim_value.reshape(1,-1), color0, upper, lower)
        # plot synchronization.
        ax.plot(self.alignment['neu_time'][win_width:], neu_sync, color=color2)
        ax.tick_params(axis='y', tick1On=False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_ylabel('sync level')
        ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
        ax.set_xlabel('time since stim (ms)')

    def plot_random_interval(self, ax, cate=None, roi_id=None):
        # collect data.
        [_, _, color2, _], [neu_seq, stim_seq, stim_value, pre_isi] = get_neu_trial(
            self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
            trial_param=[[2,3,4,5], None, None, None, [1], [0]],
            mean_sem=False,
            cate=cate, roi_id=roi_id)
        colors = get_cmap_color(self.bin_num, base_color=color2)
        # bin data based on isi.
        [bins, bin_center, _, bin_neu_mean, bin_neu_sem, bin_stim_seq, bin_stim_value] = get_isi_bin_neu(
            neu_seq, stim_seq, stim_value, pre_isi, self.bin_win, self.bin_num)
        # compute bounds.
        upper = np.nanmax(bin_neu_mean) + np.nanmax(bin_neu_sem)
        lower = np.nanmin(bin_neu_mean) - np.nanmax(bin_neu_sem)
        # plot stimulus.
        for i in range(self.bin_num):
            ax.fill_between(
                bin_stim_seq[i,int(bin_stim_seq.shape[1]/2),:],
                lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
                color=colors[i], alpha=0.15, step='mid')
            ax.fill_between(
                bin_stim_seq[i,int(bin_stim_seq.shape[1]/2)-1,:],
                lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
                color=colors[i], alpha=0.15, step='mid')
            self.plot_vol(ax, self.alignment['stim_time'], bin_stim_value[i,:].reshape(1,-1), colors[i], upper, lower)
        # plot neural traces.
        for i in range(self.bin_num):
            self.plot_mean_sem(ax, self.alignment['neu_time'], bin_neu_mean[i], bin_neu_sem[i], colors[i], None)
        # adjust layout.
        adjust_layout_neu(ax)
        ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
        ax.set_xlabel('time since stim (ms)')
        lbl = ['[{},{}] ms'.format(int(bins[i]),int(bins[i+1])) for i in range(self.bin_num)]
        add_legend(ax, colors, lbl, 'upper right')

    def plot_random_interval_box(self, ax, cate=None, roi_id=None):
        win_base = [-self.bin_win[1],0]
        offsets = np.arange(self.bin_num)/20
        # collect data.
        [_, _, color2, _], [neu_seq, stim_seq, stim_value, pre_isi] = get_neu_trial(
            self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
            trial_param=[[2,3,4,5], None, None, None, [1], [0]],
            mean_sem=False,
            cate=cate, roi_id=roi_id)
        colors = get_cmap_color(self.bin_num, base_color=color2)
        # bin data based on isi.
        [bins, bin_center, bin_neu_seq, _, _, _, _] = get_isi_bin_neu(
            neu_seq, stim_seq, stim_value, pre_isi, self.bin_win, self.bin_num)
        # plot errorbar.
        for i in range(self.bin_num):
            self.plot_win_mag_box(
                ax, bin_neu_seq[i], self.alignment['neu_time'], win_base,
                colors[i], 0, offsets[i])
        # adjust layout.
        lbl = ['[{},{}] ms'.format(int(bins[i]),int(bins[i+1])) for i in range(self.bin_num)]
        add_legend(ax, colors, lbl, 'upper right')

    def plot_random_interval_heatmap(self, ax, cate=None, roi_id=None):
        # collect data.
        [_, _, _, cmap], [neu_seq, stim_seq, stim_value, pre_isi] = get_neu_trial(
            self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
            trial_param=[[2,3,4,5], None, None, None, [1], [0]],
            mean_sem=False,
            cate=cate, roi_id=roi_id)
        # average across neurons.
        neu_seq = np.concatenate([np.nanmean(n,axis=1) for n in neu_seq],axis=0)
        # sort based on isi.
        pre_isi = np.concatenate(pre_isi)
        neu_seq = neu_seq[np.argsort(pre_isi),:]
        # plot heatmap.
        heatmap = apply_colormap(neu_seq, cmap)
        ax.imshow(
            heatmap,
            extent=[
                self.alignment['neu_time'][0], self.alignment['neu_time'][-1],
                1, heatmap.shape[0]],
            interpolation='nearest', aspect='auto')
        # adjust layout.
        adjust_layout_heatmap(ax)
        cbar = ax.figure.colorbar(
            plt.cm.ScalarMappable(cmap=cmap), ax=ax, ticks=[0.2,0.8], shrink=1, aspect=50)
        cbar.outline.set_visible(False)
        cbar.ax.set_ylabel('normalized dff', rotation=-90, va="bottom")
        cbar.ax.set_yticklabels(['0.2', '0.8'], rotation=-90)
        ax.set_xlabel('time since stim (ms)')
        ax.set_ylabel('preceeding interval (ms)')
        idx = np.arange(1, heatmap.shape[0], heatmap.shape[0]/5, dtype='int32') + int(heatmap.shape[0]/5/2)
        ax.set_yticks(idx)
        ax.set_yticklabels(pre_isi[np.argsort(pre_isi)][idx][::-1].astype('int32'))
    
    def plot_random_interval_curve(self, ax, cate=None, roi_id=None):
        win_base = [-self.bin_win[1],0]
        win_eval = [0,200]
        l_idx, r_idx = get_frame_idx_from_time(self.alignment['neu_time'], 0, win_eval[0], win_eval[1])
        # collect data.
        [_, _, color2, _], [neu_seq, stim_seq, stim_value, pre_isi] = get_neu_trial(
            self.alignment, self.list_labels, self.list_significance, self.list_stim_labels,
            trial_param=[[2,3,4,5], None, None, None, [1], [0]],
            mean_sem=False,
            cate=cate, roi_id=roi_id)
        # average across neurons.
        neu_seq = np.concatenate([np.nanmean(n,axis=1) for n in neu_seq],axis=0)
        pre_isi = np.concatenate(pre_isi)
        # get evoked response.
        neu_baseline = [get_base_mean_win(n.reshape(1, -1), self.alignment['neu_time'], 0, win_base)
                    for n in neu_seq]
        neu_evoke = np.nanmean(neu_seq[:,l_idx:r_idx],axis=1) - neu_baseline
        bins, bin_center, bin_idx = get_bin_idx(pre_isi, self.bin_win, self.bin_num)
        # bin response data.
        neu_mean = []
        neu_sem = []
        for i in range(self.bin_num):
            m,s = get_mean_sem(neu_evoke[bin_idx==i].reshape(-1,1))
            neu_mean.append(m)
            neu_sem.append(s)
        neu_mean = np.concatenate(neu_mean)
        neu_sem = np.concatenate(neu_sem)
        # plot errorbar.
        ax.errorbar(
            bin_center,
            neu_mean,
            neu_sem,
            color=color2, capsize=2, marker='o', linestyle='none',
            markeredgecolor='white', markeredgewidth=1, label='excitory')
        # adjust layout.
        ax.tick_params(tick1On=False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xlabel('preceeding interval (ms)')
        ax.set_ylabel('response magnitude during [{},{}] ms'.format(
            win_eval[0], win_eval[1]))
        ax.set_xlim(self.bin_win)


# colors = ['#989A9C', '#A4CB9E', '#9DB4CE', '#EDA1A4', '#F9C08A']
class plotter_main(plotter_utils):
    def __init__(self, neural_trials, labels, significance, label_names):
        super().__init__(neural_trials, labels, significance)
        self.label_names = label_names

    def random_exc(self, axs):
        try:
            cate = -1
            label_name = self.label_names[str(cate)]

            self.plot_random_stim(axs[0], cate=cate)
            axs[0].set_title(f'response to random stim \n {label_name}')
            
            self.plot_random_sync(axs[1], cate=cate)
            axs[1].set_title(f'response to random stim synchronization level \n {label_name}')
            
            self.plot_random_interval(axs[2], cate=cate)
            axs[2].set_title(f'response to random preceeding interval with bins \n {label_name}')
            
            self.plot_random_interval_box(axs[3], cate=cate)
            axs[3].set_title(f'response to random preceeding interval with bins \n {label_name}')
            
            self.plot_random_interval_heatmap(axs[4], cate=cate)
            axs[4].set_title(f'response to random preceeding interval heatmap \n {label_name}')
            
            self.plot_random_interval_curve(axs[5], cate=cate)
            axs[5].set_title(f'response to random preceeding interval scatter \n {label_name}')

        except: pass

    def random_inh(self, axs):
        try:
            cate = 1
            label_name = self.label_names[str(cate)]
            
            self.plot_random_stim(axs[0], cate=cate)
            axs[0].set_title(f'response to random stim \n {label_name}')
            
            self.plot_random_sync(axs[1], cate=cate)
            axs[1].set_title(f'response to random stim synchronization level \n {label_name}')
            
            self.plot_random_interval(axs[2], cate=cate)
            axs[2].set_title(f'response to random preceeding interval with bins \n {label_name}')
            
            self.plot_random_interval_box(axs[3], cate=cate)
            axs[3].set_title(f'response to random preceeding interval with bins \n {label_name}')
            
            self.plot_random_interval_heatmap(axs[4], cate=cate)
            axs[4].set_title(f'response to random preceeding interval heatmap \n {label_name}')
            
            self.plot_random_interval_curve(axs[5], cate=cate)
            axs[5].set_title(f'response to random preceeding interval scatter \n {label_name}')

        except: pass
