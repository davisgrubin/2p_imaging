#!/usr/bin/env python3

import numpy as np

from modules.Alignment import run_get_stim_response
from plot.utils import get_mean_sem
from plot.utils import get_multi_sess_neu_trial_average
from plot.utils import get_roi_label_color
from plot.utils import get_epoch_idx
from plot.utils import get_odd_stim_prepost_idx
from plot.utils import get_expect_time
from plot.utils import adjust_layout_neu
from plot.utils import utils


class plotter_utils(utils):

    def __init__(
            self,
            list_neural_trials, list_labels, list_significance
            ):
        super().__init__()
        timescale = 1.0
        self.n_sess = len(list_neural_trials)
        self.l_frames = int(100*timescale)
        self.r_frames = int(200*timescale)
        self.cut_frames = int(80*timescale)
        self.list_stim_labels = [
            nt['stim_labels'][1:-1,:] for nt in list_neural_trials]
        self.list_labels = list_labels
        self.alignment = run_get_stim_response(
                list_neural_trials, self.l_frames, self.r_frames, expected='none')
        self.expect = np.array([
            np.mean([get_expect_time(sl)[0] for sl in self.list_stim_labels]),
            np.mean([get_expect_time(sl)[1] for sl in self.list_stim_labels])])
        self.epoch_early = [get_epoch_idx(sl)[0] for sl in self.list_stim_labels]
        self.epoch_late  = [get_epoch_idx(sl)[1] for sl in self.list_stim_labels]
        self.list_odd_idx = [
            get_odd_stim_prepost_idx(sl) for sl in self.list_stim_labels]
        self.list_significance = list_significance

    def plot_pre_odd_isi_distribution(self, ax):
        isi_short = [self.alignment['list_pre_isi'][n][self.list_odd_idx[n][0]] for n in range(self.n_sess)]
        isi_long  = [self.alignment['list_pre_isi'][n][self.list_odd_idx[n][1]] for n in range(self.n_sess)]
        isi_short = np.concatenate(isi_short)
        isi_long  = np.concatenate(isi_long)
        isi_max = np.nanmax(np.concatenate([isi_short, isi_long]))
        ax.hist(isi_short,
            bins=100, range=[0, isi_max], align='left',
            color='#9DB4CE', density=True)
        ax.hist(isi_long,
            bins=100, range=[0, isi_max], align='right',
            color='dodgerblue', density=True)
        ax.set_title('pre oddball isi distribution')
        ax.tick_params(tick1On=False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xlabel('time (ms)')
        ax.set_ylabel('percentage')
        ax.set_xlim([0, isi_max])
        ax.set_xticks(500*np.arange(0, isi_max/500+1).astype('int32'))
        ax.set_xticklabels(
            500*np.arange(0, isi_max/500+1).astype('int32'),
            rotation='vertical')
        ax.plot([], color='#9DB4CE', label='short')
        ax.plot([], color='dodgerblue', label='long')
        ax.legend(loc='upper left')

    def plot_odd_normal(self, ax, normal, cate=None, roi_id=None):
        if cate != None:
            neu_cate = [
                self.alignment['list_neu_seq'][i][:,(self.list_labels[i]==cate)*self.list_significance[i]['r_oddball'],:]
                for i in range(self.n_sess)]
            _, color1, color2, _ = get_roi_label_color([cate], 0)
        if roi_id != None:
            neu_cate = [np.expand_dims(self.alignment['list_neu_seq'][0][:,roi_id,:], axis=1)]
            _, color1, color2, _ = get_roi_label_color(self.labels, roi_id)
        neu_mean = []
        neu_sem = []
        # collect data.
        if 0 in normal:
            neu_short, _, stim_seq_short, stim_value_short, _ = get_multi_sess_neu_trial_average(
                self.list_stim_labels, neu_cate, self.alignment['list_stim_seq'],
                self.alignment['list_stim_value'], self.alignment['list_pre_isi'],
                trial_idx=[l[0] for l in self.list_odd_idx],
                trial_param=[[-1], None, None, [0], [0]])
            mean_short, sem_short = get_mean_sem(neu_short)
            neu_mean.append(mean_short)
            neu_sem.append(sem_short)
        if 1 in normal:
            neu_long, _, stim_seq_long, stim_value_long, _ = get_multi_sess_neu_trial_average(
                self.list_stim_labels, neu_cate, self.alignment['list_stim_seq'],
                self.alignment['list_stim_value'], self.alignment['list_pre_isi'],
                trial_idx=[l[0] for l in self.list_odd_idx],
                trial_param=[[-1], None, None, [1], [0]])
            mean_long, sem_long = get_mean_sem(neu_long)
            neu_mean.append(mean_long)
            neu_sem.append(sem_long)
        # find bounds.
        upper = np.nanmax(neu_mean) + np.nanmax(neu_sem)
        lower = np.nanmin(neu_mean) - np.nanmax(neu_sem)
        # plot stimulus.
        if 0 in normal:
            for i in range(3):
                ax.fill_between(
                    stim_seq_short[i,:],
                    lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
                    color=color1, alpha=0.15, step='mid')
            self.plot_vol(
                ax, self.alignment['stim_time'],
                stim_value_short.reshape(1,-1), color1, upper, lower)
        if 1 in normal:
            for i in range(3):
                ax.fill_between(
                    stim_seq_long[i,:],
                    lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
                    color=color2, alpha=0.15, step='mid')
            self.plot_vol(
                ax, self.alignment['stim_time'],
                stim_value_long.reshape(1,-1), color2, upper, lower)
        # plot neural traces.
        if 0 in normal:
            self.plot_mean_sem(
                ax, self.alignment['neu_time'],
                mean_short, sem_short, color1, 'short')
        if 1 in normal:
            self.plot_mean_sem(
                ax, self.alignment['neu_time'],
                mean_long, sem_long, color2, 'long')
        adjust_layout_neu(ax)
        ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
        ax.set_xlabel('time since pre oddball stim (ms)')
        ax.legend(loc='upper left')

class plotter_VIPTD_G8_align_odd(plotter_utils):
    def __init__(self, neural_trials, labels, significance):
        super().__init__(neural_trials, labels, significance)

    def odd_normal_exc(self, axs):
        
        self.plot_odd_normal(axs[0], [0], cate=-1)
        axs[0].set_title('response to oddball \n excitatory (short)')
        
        self.plot_odd_normal(axs[1], [1], cate=-1)
        axs[1].set_title('response to oddball \n excitatory (long)')
        
        self.plot_odd_normal(axs[2], [0,1], cate=-1)
        axs[2].set_title('response to oddball \n excitatory')

    def odd_normal_inh(self, axs):
        
        self.plot_odd_normal(axs[0], [0], cate=1)
        axs[0].set_title('response to oddball \n inhibitory (short)')
        
        self.plot_odd_normal(axs[1], [1], cate=1)
        axs[1].set_title('response to oddball \n inhibitory (long)')
        
        self.plot_odd_normal(axs[2], [0,1], cate=1)
        axs[2].set_title('response to oddball \n inhibitory')

    def odd_normal_heatmap(self, axs):
        win_sort = [-500, 1500]
        labels = np.concatenate(self.list_labels)
        sig = np.concatenate([self.list_significance[n]['r_oddball'] for n in range(self.n_sess)])
        neu_short, _, stim_seq_short, _, _ = get_multi_sess_neu_trial_average(
            self.list_stim_labels, self.alignment['list_neu_seq'], self.alignment['list_stim_seq'],
            self.alignment['list_stim_value'], self.alignment['list_pre_isi'],
            trial_idx=[l[0] for l in self.list_odd_idx],
            trial_param=[[-1], None, None, [0], [0]])
        neu_long, _, stim_seq_long, _, _ = get_multi_sess_neu_trial_average(
            self.list_stim_labels, self.alignment['list_neu_seq'], self.alignment['list_stim_seq'],
            self.alignment['list_stim_value'], self.alignment['list_pre_isi'],
            trial_idx=[l[0] for l in self.list_odd_idx],
            trial_param=[[-1], None, None, [1], [0]])
        for i in range(4):
            axs[i].set_xlabel('time since pre oddball stim (ms)')
            
        self.plot_heatmap_neuron(axs[0], neu_short, self.alignment['neu_time'], neu_short, win_sort, labels, sig)
        axs[0].set_title('response to oddball \n (short sorted by short)')
        
        self.plot_heatmap_neuron(axs[1], neu_long, self.alignment['neu_time'], neu_short, win_sort, labels, sig)
        axs[1].set_title('response to oddball \n (long sorted by short)')
        
        self.plot_heatmap_neuron(axs[2], neu_short, self.alignment['neu_time'], neu_long, win_sort, labels, sig)
        axs[2].set_title('response to oddball \n (short sorted by long)')
        
        self.plot_heatmap_neuron(axs[3], neu_long, self.alignment['neu_time'], neu_long, win_sort, labels, sig)
        axs[3].set_title('response to oddball \n (long sorted by long)')
        