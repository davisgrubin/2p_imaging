#!/usr/bin/env python3

import numpy as np

from modules.Alignment import run_get_stim_response
from plot.utils import exclude_post_odd_stim
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
        self.alignment_local = run_get_stim_response(
                list_neural_trials, self.l_frames, self.r_frames, expected='local')
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
        neu_mean = []
        neu_sem = []
        if 0 in normal:
            neu_short, _, stim_seq_short, stim_value_short, _ = get_multi_sess_neu_trial_average(
                self.list_stim_labels, neu_cate, self.alignment['list_stim_seq'],
                self.alignment['list_stim_value'], self.alignment['list_pre_isi'],
                trial_param=[[2,3,4,5], [0], [fix_jitter], None, [0]])
            mean_short, sem_short = get_mean_sem(neu_short)
            neu_mean.append(mean_short)
            neu_sem.append(sem_short)
        if 1 in normal:
            neu_long, _, stim_seq_long, stim_value_long, _ = get_multi_sess_neu_trial_average(
                self.list_stim_labels, neu_cate, self.alignment['list_stim_seq'],
                self.alignment['list_stim_value'], self.alignment['list_pre_isi'],
                trial_param=[[2,3,4,5], [1], [fix_jitter], None, [0]])
            mean_long, sem_long = get_mean_sem(neu_long)
            neu_mean.append(mean_long)
            neu_sem.append(sem_long)
        # compute bounds.
        upper = np.nanmax(neu_mean) + np.nanmax(neu_sem)
        lower = np.nanmin(neu_mean) - np.nanmax(neu_sem)
        # plot voltages.
        if 0 in normal:
            for i in range(3):
                ax.fill_between(
                    stim_seq_short[i,:],
                    lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
                    color=color1, alpha=0.15, step='mid')
            self.plot_vol(ax, self.alignment['stim_time'], stim_value_short.reshape(1,-1), color1, upper, lower)
        if 1 in normal:
            for i in range(3):
                ax.fill_between(
                    stim_seq_long[i,:],
                    lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
                    color=color2, alpha=0.15, step='mid')
            self.plot_vol(ax, self.alignment['stim_time'], stim_value_long.reshape(1,-1), color2, upper, lower)
        # plot neural traces.
        if 0 in normal:
            self.plot_mean_sem(ax, self.alignment['neu_time'], mean_short, sem_short, color1, 'short')
        if 1 in normal:
            self.plot_mean_sem(ax, self.alignment['neu_time'], mean_long,  sem_long,  color2, 'long')
        adjust_layout_neu(ax)
        ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
        ax.set_xlabel('time since stim (ms)')

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
        axs[0].set_title('response to normal \n excitatory (short, fix)')
        self.plot_normal(axs[1], [1], 0, cate=-1)
        axs[1].set_title('response to normal \n excitatory (long, fix)')
        self.plot_normal(axs[2], [0,1], 0, cate=-1)
        axs[2].set_title('response to normal \n excitatory (fix)')
        self.plot_normal_box(axs[3], 0, cate=-1)
        axs[3].set_title('response to normal \n excitatory (fix)')
        self.plot_normal(axs[4], [0], 1, cate=-1)
        axs[4].set_title('response to normal \n excitatory (short, jitter)')
        self.plot_normal(axs[5], [1], 1, cate=-1)
        axs[5].set_title('response to normal \n excitatory (long, jitter)')
        self.plot_normal(axs[6], [0,1], 1, cate=-1)
        axs[6].set_title('response to normal \n excitatory (jitter)')
        self.plot_normal_box(axs[7], 1, cate=-1)
        axs[7].set_title('response to normal \n excitatory (jitter)')

    def normal_inh(self, axs):
        self.plot_normal(axs[0], [0], 0, cate=1)
        axs[0].set_title('response to normal \n inhibitory (short, fix)')
        self.plot_normal(axs[1], [1], 0, cate=1)
        axs[1].set_title('response to normal \n inhibitory (long, fix)')
        self.plot_normal(axs[2], [0,1], 0, cate=1)
        axs[2].set_title('response to normal \n inhibitory (fix)')
        self.plot_normal_box(axs[3], 0, cate=1)
        axs[3].set_title('response to normal \n inhibitory (fix)')
        self.plot_normal(axs[4], [0], 1, cate=1)
        axs[4].set_title('response to normal \n inhibitory (short, jitter)')
        self.plot_normal(axs[5], [1], 1, cate=1)
        axs[5].set_title('response to normal \n inhibitory (long, jitter)')
        self.plot_normal(axs[6], [0,1], 1, cate=1)
        axs[6].set_title('response to normal \n inhibitory (jitter)')
        self.plot_normal_box(axs[7], 1, cate=1)
        axs[7].set_title('response to normal \n inhibitory (jitter)')

    def normal_heatmap(self, axs):
        win_sort = [-200, 1000]
        labels = np.concatenate(self.list_labels)
        sig = np.concatenate([self.list_significance[n]['r_normal'] for n in range(self.n_sess)])
        neu_short_fix, _, _, _, _ = get_multi_sess_neu_trial_average(
            self.list_stim_labels, self.alignment['list_neu_seq'], self.alignment['list_stim_seq'],
            self.alignment['list_stim_value'], self.alignment['list_pre_isi'],
            trial_param=[[2,3,4,5], [0], [0], None, [0]])
        neu_long_fix, _, _, _, _ = get_multi_sess_neu_trial_average(
            self.list_stim_labels, self.alignment['list_neu_seq'], self.alignment['list_stim_seq'],
            self.alignment['list_stim_value'], self.alignment['list_pre_isi'],
            trial_param=[[2,3,4,5], [1], [0], None, [0]])
        neu_short_jitter, _, _, _, _ = get_multi_sess_neu_trial_average(
            self.list_stim_labels, self.alignment['list_neu_seq'], self.alignment['list_stim_seq'],
            self.alignment['list_stim_value'], self.alignment['list_pre_isi'],
            trial_param=[[2,3,4,5], [0], [1], None, [0]])
        neu_long_jitter, _, _, _, _ = get_multi_sess_neu_trial_average(
            self.list_stim_labels, self.alignment['list_neu_seq'], self.alignment['list_stim_seq'],
            self.alignment['list_stim_value'], self.alignment['list_pre_isi'],
            trial_param=[[2,3,4,5], [1], [1], None, [0]])
        for i in range(8):
            axs[i].set_xlabel('time since stim (ms)')
        self.plot_heatmap_neuron(
            axs[0], neu_short_fix, self.alignment['neu_time'], neu_short_fix,
            win_sort, labels, sig)
        axs[0].set_title('response to normal \n (short sorted by short, fix)')
        self.plot_heatmap_neuron(
            axs[1], neu_long_fix, self.alignment['neu_time'], neu_short_fix,
            win_sort, labels, sig)
        axs[1].set_title('response to normal \n (long sorted by short, fix)')
        self.plot_heatmap_neuron(
            axs[2], neu_short_fix, self.alignment['neu_time'], neu_long_fix,
            win_sort, labels, sig)
        axs[2].set_title('response to normal \n (short sorted by long, fix)')
        self.plot_heatmap_neuron(
            axs[3], neu_long_fix, self.alignment['neu_time'], neu_long_fix,
            win_sort, labels, sig)
        axs[3].set_title('response to normal \n (long sorted by long, fix)')
        self.plot_heatmap_neuron(
            axs[4], neu_short_jitter, self.alignment['neu_time'], neu_short_jitter,
            win_sort, labels, sig)
        axs[4].set_title('response to normal \n (short sorted by short, jitter)')
        self.plot_heatmap_neuron(
            axs[5], neu_long_jitter, self.alignment['neu_time'], neu_short_jitter,
            win_sort, labels, sig)
        axs[5].set_title('response to normal \n (long sorted by short, jitter)')
        self.plot_heatmap_neuron(
            axs[6], neu_short_jitter, self.alignment['neu_time'], neu_long_jitter,
            win_sort, labels, sig)
        axs[6].set_title('response to normal \n (short sorted by long, jitter)')
        self.plot_heatmap_neuron(
            axs[7], neu_long_jitter, self.alignment['neu_time'], neu_long_jitter,
            win_sort, labels, sig)
        axs[7].set_title('response to normal \n (long sorted by long, jitter)')

    def normal_isi_exc(self, axs):
        self.plot_normal_global(axs[0], 0, cate=-1)
        axs[0].set_title('response to normal with proceeding ISI \n excitatory (global, short)')
        self.plot_normal_global(axs[1], 1, cate=-1)
        axs[1].set_title('response to normal with proceeding ISI \n excitatory (global, long)')
        self.plot_normal_overlap(axs[2], cate=-1)
        axs[2].set_title('response to normal with proceeding ISI \n excitatory (global, {}-{} ms overlap)'.format(
            int(self.expect[0]), int(self.expect[1])))
        self.plot_normal_local(axs[3], 0, cate=-1)
        axs[3].set_title('response to normal with proceeding ISI \n excitatory (local, short)')
        self.plot_normal_local(axs[4], 1, cate=-1)
        axs[4].set_title('response to normal with proceeding ISI \n excitatory (local, long)')

    def normal_isi_inh(self, axs):
        self.plot_normal_global(axs[0], 0, cate=1)
        axs[0].set_title('response to normal with proceeding ISI \n inhibitory (global, short)')
        self.plot_normal_global(axs[1], 1, cate=1)
        axs[1].set_title('response to normal with proceeding ISI \n inhibitory (global, long)')
        self.plot_normal_overlap(axs[2], cate=1)
        axs[2].set_title('response to normal with proceeding ISI \n inhibitory (global, {}-{} ms overlap)'.format(
            int(self.expect[0]), int(self.expect[1])))
        self.plot_normal_local(axs[3], 0, cate=1)
        axs[3].set_title('response to normal with proceeding ISI \n inhibitory (local, short)')
        self.plot_normal_local(axs[4], 1, cate=1)
        axs[4].set_title('response to normal with proceeding ISI \n inhibitory (local, long)')

    def normal_epoch_exc(self, axs):
        self.plot_normal_epoch(axs[0], 0, 0, cate=-1)
        axs[0].set_title('response to normal \n excitatory (short, fix)')
        self.plot_normal_epoch(axs[1], 1, 0, cate=-1)
        axs[1].set_title('response to normal \n excitatory (long, fix)')
        self.plot_normal_epoch(axs[2], 0, 1, cate=-1)
        axs[2].set_title('response to normal \n excitatory (short, jitter)')
        self.plot_normal_epoch(axs[3], 1, 1, cate=-1)
        axs[3].set_title('response to normal \n excitatory (long, jitter)')

    def normal_epoch_inh(self, axs):
        self.plot_normal_epoch(axs[0], 0, 0, cate=1)
        axs[0].set_title('response to normal \n inhibitory (short, fix)')
        self.plot_normal_epoch(axs[1], 1, 0, cate=1)
        axs[1].set_title('response to normal \n inhibitory (long, fix)')
        self.plot_normal_epoch(axs[2], 0, 1, cate=1)
        axs[2].set_title('response to normal \n inhibitory (short, jitter)')
        self.plot_normal_epoch(axs[3], 1, 1, cate=1)
        axs[3].set_title('response to normal \n inhibitory (long, jitter)')




