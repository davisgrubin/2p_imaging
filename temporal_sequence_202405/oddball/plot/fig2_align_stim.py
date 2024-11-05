#!/usr/bin/env python3

import numpy as np

from modules.Alignment import run_get_stim_response
from plot.utils import exclude_post_odd_stim
from plot.utils import get_mean_sem
from plot.utils import get_multi_sess_neu_trial_average
from plot.utils import get_roi_label_color
from plot.utils import get_epoch_idx
from plot.utils import get_change_prepost_idx
from plot.utils import pick_trial
from plot.utils import adjust_layout_neu
from plot.utils import utils

# fig, ax = plt.subplots(1, 1, figsize=(12, 6))

class plotter_utils(utils):

    def __init__(
            self,
            list_neural_trials, list_labels, list_significance
            ):
        timescale = 1.0
        self.n_sess = len(list_neural_trials)
        self.l_frames = int(50*timescale)
        self.r_frames = int(100*timescale)
        self.list_stim_labels = [
            nt['stim_labels'][1:-1,:] for nt in list_neural_trials]
        self.list_stim_labels = [
            exclude_post_odd_stim(sl) for sl in self.list_stim_labels]
        self.list_labels = list_labels
        self.list_epoch = [
            get_epoch_idx(sl) for sl in self.list_stim_labels]
        self.epoch = [[self.list_epoch[n][0][e] for n in range(self.n_sess)] for e in range(4)]
        [self.list_neu_seq, self.list_neu_time,
         self.list_stim_seq, self.list_stim_value, self.list_stim_time,
         _, _] = run_get_stim_response(
                list_neural_trials, self.l_frames, self.r_frames)
        self.list_significance = list_significance

    def plot_normal(self, ax, normal, cate=None, roi_id=None):
        colors = ['royalblue', 'mediumseagreen', 'violet', 'coral']
        lbl = ['img#1', 'img#2', 'img#3', 'img#4']
        if cate != None:
            neu_cate = [
                self.list_neu_seq[i][:,(self.list_labels[i]==cate)*self.list_significance[i]['r_normal'],:]
                for i in range(self.n_sess)]
        if roi_id != None:
            neu_cate = [np.expand_dims(self.list_neu_seq[0][:,roi_id,:], axis=1)]
        neu_mean = []
        neu_sem = []
        for i in [2,3,4,5]:
            neu, _ = get_multi_sess_neu_trial_average(
                self.list_stim_labels, neu_cate,
                trial_param=[[i], [normal], None, None, [0]])
            m, s = get_mean_sem(neu)
            neu_mean.append(m)
            neu_sem.append(s)
        upper = np.nanmax(neu_mean) + np.nanmax(neu_sem)
        lower = np.nanmin(neu_mean) - np.nanmax(neu_sem)
        idx = pick_trial(self.list_stim_labels[0], [2,3,4,5], [normal], None, None, [0])
        stim = np.mean(self.list_stim_seq[0][idx,1,:],axis=0)
        ax.fill_between(
            stim,
            lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
            color='gold', alpha=0.15, step='mid', label='stim')
        self.plot_vol(
            ax, self.list_stim_time[0], self.list_stim_value[0][idx,:],
            'gold', upper, lower)
        for i in range(4):
            self.plot_mean_sem(
                ax, self.list_neu_time[0], neu_mean[i], neu_sem[i],
                colors[i], lbl[i])
        adjust_layout_neu(ax)
        ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
        ax.set_xlabel('time since stim (ms)')

    def plot_normal_box(self, ax, normal, cate=None, roi_id=None):
        win_base = [-1000,0]
        colors = ['royalblue', 'mediumseagreen', 'violet', 'coral', 'grey']
        lbl = ['img#1', 'img#2', 'img#3', 'img#4', 'all']
        offsets = [0.0, 0.1, 0.2, 0.3, 0.4]
        if cate != None:
            neu_cate = [
                self.list_neu_seq[i][:,(self.list_labels[i]==cate)*self.list_significance[i]['r_normal'],:]
                for i in range(self.n_sess)]
        if roi_id != None:
            neu_cate = [np.expand_dims(self.list_neu_seq[0][:,roi_id,:], axis=1)]
        neu = []
        for i in [2,3,4,5]:
            neu.append(get_multi_sess_neu_trial_average(
                self.list_stim_labels, neu_cate,
                trial_param=[[i], [normal], None, None, [0]])[0])
        neu.append(get_multi_sess_neu_trial_average(
            self.list_stim_labels, neu_cate,
            trial_param=[[2,3,4,5], [normal], None, None, [0]])[0])
        for i in range(5):
            self.plot_win_mag_box(
                ax, neu[i], self.list_neu_time[0], win_base, colors[i], 0, offsets[i])
            ax.plot([], color=colors[i], label=lbl[i])
        ax.legend(loc='upper right')

    def plot_epoch(self, ax, epoch, normal, colors, cate=None, roi_id=None):
        lbl = ['ep11', 'ep12', 'ep21', 'ep22']
        if cate != None:
            neu_cate = [
                self.list_neu_seq[i][:,(self.list_labels[i]==cate)*self.list_significance[i]['r_normal'],:]
                for i in range(self.n_sess)]
        if roi_id != None:
            neu_cate = [np.expand_dims(self.list_neu_seq[0][:,roi_id,:], axis=1)]
        neu_mean = []
        neu_sem = []
        for i in range(4):
            neu, _ = get_multi_sess_neu_trial_average(
                self.list_stim_labels, neu_cate,
                trial_idx=epoch[i],
                trial_param=[[2,3,4,5], [normal], None, None, [0]])
            m, s = get_mean_sem(neu)
            neu_mean.append(m)
            neu_sem.append(s)
        upper = np.nanmax(neu_mean) + np.nanmax(neu_sem)
        lower = np.nanmin(neu_mean) - np.nanmax(neu_sem)
        idx = pick_trial(self.list_stim_labels[0], [2,3,4,5], [normal], None, None, [0])
        stim = np.mean(self.list_stim_seq[0][idx,1,:],axis=0)
        ax.fill_between(
            stim,
            lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
            color='gold', alpha=0.15, step='mid', label='stim')
        self.plot_vol(
            ax, self.list_stim_time[0], self.list_stim_value[0][idx,:],
            'gold', upper, lower)
        for i in range(4):
            self.plot_mean_sem(
                ax, self.list_neu_time[0], neu_mean[i], neu_sem[i],
                colors[i], lbl[i])
        adjust_layout_neu(ax)
        ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
        ax.set_xlabel('time since stim (ms)')

    def plot_epoch_box(self, ax, epoch, normal, colors, cate=None, roi_id=None):
        win_base = [-1000,0]
        lbl = ['ep11', 'ep12', 'ep21', 'ep22']
        offsets = [0.0, 0.1, 0.2, 0.3]
        if cate != None:
            neu_cate = [
                self.list_neu_seq[i][:,(self.list_labels[i]==cate)*self.list_significance[i]['r_normal'],:]
                for i in range(self.n_sess)]
        if roi_id != None:
            neu_cate = [np.expand_dims(self.list_neu_seq[0][:,roi_id,:], axis=1)]
        neu = []
        for i in range(4):
            neu.append(get_multi_sess_neu_trial_average(
                self.list_stim_labels, neu_cate,
                trial_idx=epoch[i],
                trial_param=[[2,3,4,5], [normal], None, None, [0]])[0])
        for i in range(4):
            self.plot_win_mag_box(
                ax, neu[i], self.list_neu_time[0], win_base, colors[i], 0, offsets[i])
            ax.plot([], color=colors[i], label=lbl[i])
        ax.legend(loc='upper right')
    
    def plot_context(self, ax, img_id, cate=None, roi_id=None):
        if cate != None:
            _, _, color, _ = get_roi_label_color([cate], 0)
            neu_cate = [
                self.list_neu_seq[i][:,(self.list_labels[i]==cate)*self.list_significance[i]['r_normal'],:]
                for i in range(self.n_sess)]
        if roi_id != None:
            _, _, color, _ = get_roi_label_color(self.labels, roi_id)
            neu_cate = [np.expand_dims(self.list_neu_seq[0][:,roi_id,:], axis=1)]
        neu, _ = get_multi_sess_neu_trial_average(
            self.list_stim_labels, neu_cate,
            trial_param=[img_id, [0], None, None, [0]])
        m, s = get_mean_sem(neu)
        upper = np.nanmax(m) + np.nanmax(s)
        lower = np.nanmin(m) - np.nanmax(s)
        idx = pick_trial(self.list_stim_labels[0], img_id, [0], None, None, [0])
        ax.fill_between(
            np.mean(self.list_stim_seq[0][idx,1,:],axis=0),
            lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
            color=color, alpha=0.15, step='mid')
        ax.fill_between(
            np.mean(self.list_stim_seq[0][idx,2,:],axis=0),
            lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
            color=color, alpha=0.15, step='mid')
        self.plot_vol(
            ax, self.list_stim_time[0], self.list_stim_value[0][idx,:],
            color, upper, lower)
        self.plot_mean_sem(ax, self.list_neu_time[0], m, s, color, None)
        adjust_layout_neu(ax)
        ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
        ax.set_xlabel('time since stim (ms)')

    def plot_context_box(self, ax, cate=None, roi_id=None):
        win_base = [-1500,-300]
        lbl = ['dff']
        offsets = [0.0]
        if cate != None:
            _, _, color, _ = get_roi_label_color([cate], 0)
            neu_cate = [
                self.list_neu_seq[i][:,(self.list_labels[i]==cate)*self.list_significance[i]['r_normal'],:]
                for i in range(self.n_sess)]
        if roi_id != None:
            _, _, color, _ = get_roi_label_color(self.labels, roi_id)
            neu_cate = [np.expand_dims(self.list_neu_seq[0][:,roi_id,:], axis=1)]
        neu = get_multi_sess_neu_trial_average(
            self.list_stim_labels, neu_cate,
            trial_param=[[2,3,4,5], [0], None, None, [0]])[0]
        self.plot_win_mag_box(
                ax, neu, self.list_neu_time[0], win_base, color, 0, offsets[0])
        ax.plot([], color=color, label=lbl[0])
        ax.legend(loc='upper right')

    def plot_change_prepost(self, ax, normal, cate=None, roi_id=None):
        if cate != None:
            _, color1, color2, _ = get_roi_label_color([cate], 0)
            neu_cate = [
                self.list_neu_seq[i][:,(self.list_labels[i]==cate)*self.list_significance[i]['r_normal'],:]
                for i in range(self.n_sess)]
        if roi_id != None:
            _, color1, color2, _ = get_roi_label_color(self.labels, roi_id)
            neu_cate = [np.expand_dims(self.list_neu_seq[0][:,roi_id,:], axis=1)]
        list_idx_pre  = [get_change_prepost_idx(sl)[0] for sl in self.list_stim_labels]
        list_idx_post = [get_change_prepost_idx(sl)[1] for sl in self.list_stim_labels]
        neu_pre, _ = get_multi_sess_neu_trial_average(
            self.list_stim_labels, neu_cate,
            trial_idx=list_idx_pre,
            trial_param=[[2,3,4,5], [normal], None, None, [0]])
        neu_post, _ = get_multi_sess_neu_trial_average(
            self.list_stim_labels, neu_cate,
            trial_idx=list_idx_post,
            trial_param=[[-2,-3,-4,-5], [normal], None, None, [0]])
        neu_mean_pre,  neu_sem_pre  = get_mean_sem(neu_pre)
        neu_mean_post, neu_sem_post = get_mean_sem(neu_post)
        idx = pick_trial(self.list_stim_labels[0], [-2,-3,-4,-5], [normal], None, None, [0])
        stim_seq = np.mean(self.list_stim_seq[0][idx,1,:],axis=0)
        upper = np.nanmax([neu_mean_pre, neu_mean_post]) + np.nanmax([neu_sem_pre, neu_sem_post])
        lower = np.nanmin([neu_mean_pre, neu_mean_post]) - np.nanmax([neu_sem_pre, neu_sem_post])
        ax.fill_between(
            stim_seq,
            lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
            color='gold', alpha=0.25, step='mid', label='stim')
        self.plot_vol(
            ax, self.list_stim_time[0], self.list_stim_value[0][idx,:],
            'gold', upper, lower)
        self.plot_mean_sem(ax, self.list_neu_time[0], neu_mean_pre,  neu_sem_pre,  color1, 'pre')
        self.plot_mean_sem(ax, self.list_neu_time[0], neu_mean_post, neu_sem_post, color2, 'post')
        adjust_layout_neu(ax)
        ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
        ax.set_xlabel('time since stim (ms)')

    # roi response to normal stimulus.
    def roi_normal(self, ax, roi_id):
        self.plot_normal(ax, roi_id=roi_id)
        ax.set_title('response to normal stimulus')

    # roi response to normal stimulus quantification.
    def roi_normal_box(self, ax, roi_id):
        self.plot_normal_box(ax, roi_id=roi_id)
        ax.set_title('response to normal stimulus')

    # roi response to normal stimulus single trial heatmap.
    def roi_normal_heatmap_trials(self, ax, roi_id):
        neu_cate = self.neu_seq[:,roi_id,:]
        neu_cate = neu_cate[self.stim_labels[:,2]>0,:]
        _, _, _, cmap = get_roi_label_color(self.labels, roi_id)
        self.plot_heatmap_trials(ax, neu_cate, cmap)
        ax.set_title('single trial response to normal stimulus')

    # roi response to image change in short block.
    def roi_change_short(self, ax, roi_id):
        self.plot_change(ax, 0, roi_id=roi_id)
        ax.set_title('response to image change in short block')

    # roi response to image change in short block.
    def roi_change_long(self, ax, roi_id):
        self.plot_change(ax, 1, roi_id=roi_id)
        ax.set_title('response to image change in long block')

    # roi response to image change quantification.
    def roi_change_box(self, ax, roi_id):
        self.plot_change_box(ax, roi_id=roi_id)
        ax.set_title('response to image change')

    # roi response to image change single trial heatmap.
    def roi_change_heatmap_trials(self, ax, roi_id):
        neu_cate = self.neu_seq[:,roi_id,:]
        neu_cate = neu_cate[self.stim_labels[:,2]<-1,:]
        _, _, _, cmap = get_roi_label_color(self.labels, roi_id)
        self.plot_heatmap_trials(ax, neu_cate, cmap)
        ax.set_title('response to image change')

    # roi response to short long context for all normal image.
    def roi_context_all(self, ax, roi_id):
        self.plot_context(ax, 5, roi_id=roi_id)
        ax.set_title('response to all normal image')

    # roi response to all stimulus average across trial for short.
    def roi_context_all_short_heatmap_trial(self, ax, roi_id):
        idx = (self.stim_labels[:,2]>0) * (self.stim_labels[:,3]==0)
        neu_cate = self.neu_seq[idx,:,:]
        neu_cate = neu_cate[:,roi_id,:]
        _, _, _, cmap = get_roi_label_color(self.labels, roi_id)
        self.plot_heatmap_trials(ax, neu_cate, cmap)
        ax.set_title('single trial response to normal stim (short)')

    # roi response to all stimulus average across trial for long.
    def roi_context_all_long_heatmap_trial(self, ax, roi_id):
        idx = (self.stim_labels[:,2]>0) * (self.stim_labels[:,3]==1)
        neu_cate = self.neu_seq[idx,:,:]
        neu_cate = neu_cate[:,roi_id,:]
        _, _, _, cmap = get_roi_label_color(self.labels, roi_id)
        self.plot_heatmap_trials(ax, neu_cate, cmap)
        ax.set_title('single trial response to normal stim (long)')

    # roi response to short long context for individual image.
    def roi_context_individual(self, axs, roi_id):
        titles = [
            'response to normal img#1',
            'response to normal img#2',
            'response to normal img#3',
            'response to normal img#4']
        for i in range(4):
            self.plot_context(axs[i], i+1, roi_id=roi_id)
            axs[i].set_title(titles[i])

    # roi response to short long context quantification
    def roi_context_box(self, ax, roi_id):
        self.plot_context_box(ax, roi_id=roi_id)
        ax.set_title('response to different image')


# fig, ax = plt.subplots(1, 1, figsize=(6, 6))
# colors = ['#989A9C', '#A4CB9E', '#9DB4CE', '#EDA1A4', '#F9C08A']
class plotter_VIPTD_G8_align_stim(plotter_utils):
    def __init__(self, neural_trials, labels, significance):
        super().__init__(neural_trials, labels, significance)

    def normal_exc(self, axs):
        # excitatory response to normal stimulus.
        self.plot_normal(axs[0], 0, cate=-1)
        axs[0].set_title('excitatory response to normal')
        # excitatory response to normal stimulus quantification.
        self.plot_normal_box(axs[1], 0, cate=-1)
        axs[1].set_title('excitatory response to normal')

    def normal_inh(self, axs):
        # inhibitory response to normal stimulus.
        self.plot_normal(axs[0], 0, cate=1)
        axs[0].set_title('inhibitory response to normal')
        # inhibitory response to normal stimulus quantification.
        self.plot_normal_box(axs[1], 0, cate=1)
        axs[1].set_title('inhibitory response to normal')

    def normal_heatmap(self, axs):
        sig = np.concatenate([self.list_significance[n]['r_normal'] for n in range(self.n_sess)])
        labels = np.concatenate(self.list_labels)
        neu, _ = get_multi_sess_neu_trial_average(
            self.list_stim_labels, self.list_neu_seq,
            trial_param=[[2,3,4,5], [0], None, None, [0]])
        axs[0].set_xlabel('time since stim (ms)')
        # response to normal stimulus heatmap average across trials.
        self.plot_heatmap_neuron(
            axs[0], neu, self.list_neu_time[0], neu,
            labels, sig)
        axs[0].set_title('response to normal')

    def context_exc(self, axs):
        # excitatory response to context.
        titles = [
            'excitatory response to all image',
            'excitatory response to img#1',
            'excitatory response to img#2',
            'excitatory response to img#3',
            'excitatory response to img#4']
        img_id = [[2,3,4,5], [2], [3], [4], [5]]
        for i in range(5):
            self.plot_context(axs[i], img_id[i], cate=-1)
            axs[i].set_title(titles[i])
        # excitatory response to context quantification.
        self.plot_context_box(axs[5], cate=-1)
        axs[5].set_title('excitatory response to all image')

    def context_inh(self, axs):
        # inhibitory response to short long context.
        titles = [
            'inhibitory response to all image',
            'inhibitory response to img#1',
            'inhibitory response to img#2',
            'inhibitory response to img#3',
            'inhibitory response to img#4']
        img_id = [[2,3,4,5], [2], [3], [4], [5]]
        for i in range(5):
            self.plot_context(axs[i], img_id[i], cate=1)
            axs[i].set_title(titles[i])
        # inhibitory response to short long context quantification.
        self.plot_context_box(axs[5], cate=1)
        axs[5].set_title('inhibitory response to all image')
    
    def epoch_exc(self, axs):
        colors = ['#969696', '#6BAED6', '#2171B5', '#504099']
        # excitatory response to normal stimulus with epoch.
        self.plot_epoch(axs[0], self.epoch, 0, colors, cate=-1)
        axs[0].set_title('excitatory response to normal with epoch')
        # excitatory response to normal stimulus with epoch quantification.
        self.plot_epoch_box(axs[1], self.epoch, 0, colors, cate=-1)
        axs[1].set_title('excitatory response to normal with epoch')

    def epoch_inh(self, axs):
        colors = ['#969696', '#FE7BBF', '#974EC3', '#504099']
        # inhibitory response to normal stimulus with epoch.
        self.plot_epoch(axs[0], self.epoch, 0, colors, cate=1)
        axs[0].set_title('inhibitory response to normal with epoch')
        # inhibitory response to normal stimulus with epochquantification.
        self.plot_epoch_box(axs[1], self.epoch, 0, colors, cate=1)
        axs[1].set_title('inhibitory response to normal with epoch')

    def change_exc(self, axs):
        # excitatory response to image change for all images.
        self.plot_change_prepost(axs[0], 0, cate=-1)
        axs[0].set_title('excitatory response to pre & post change')

    def change_inh(self, axs):
        # inhibitory response to image change for all images.
        self.plot_change_prepost(axs[0], 0, cate=1)
        axs[0].set_title('inhibitory response to pre & post change')
