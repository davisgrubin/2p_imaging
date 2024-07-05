#!/usr/bin/env python3

import numpy as np
from scipy.stats import sem

from modules.Alignment import get_lick_response
from plot.utils import get_mean_sem_win
from plot.utils import get_roi_label_color
from plot.utils import adjust_layout_neu
from plot.utils import utils


class plotter_utils(utils):
    def __init__(self, neural_trials, labels, significance):
        super().__init__(labels)
        
        self.l_frames = 40
        self.r_frames = 50
        [self.neu_seq_lick_all,
         self.neu_time_lick_all,
         self.lick_direc_all] = get_lick_response(
                 neural_trials, 'trial_lick',
                 self.l_frames, self.r_frames)
        [self.neu_seq_lick_reaction,
         self.neu_time_lick_reaction,
         self.lick_direc_reaction] = get_lick_response(
                 neural_trials, 'trial_reaction',
                 self.l_frames, self.r_frames)
        [self.neu_seq_lick_decision,
         self.neu_time_lick_decision,
         self.lick_direc_decision] = get_lick_response(
                 neural_trials, 'trial_decision',
                 self.l_frames, self.r_frames)
        self.significance = significance
        
    def plot_lick(
            self, ax,
            neu_seq_lick, neu_time_lick, lick_direc,
            s, cate=None, roi_id=None
            ):
        if cate != None:
            neu_cate = neu_seq_lick[:,(self.labels==cate)*s,:]
            _, color1, color2, _ = get_roi_label_color([cate], 0)
        if roi_id != None:
            neu_cate = np.expand_dims(neu_seq_lick[:,roi_id,:], axis=1)
            _, color1, color2, _ = get_roi_label_color(self.labels, roi_id)
        neu_seq_left = neu_cate[lick_direc==0, :, :].copy()
        neu_seq_left = neu_seq_left.reshape(-1, self.l_frames+self.r_frames)
        neu_seq_right = neu_cate[lick_direc==1, :, :].copy()
        neu_seq_right = neu_seq_right.reshape(-1, self.l_frames+self.r_frames)
        mean_left  = np.mean(neu_seq_left, axis=0)
        mean_right = np.mean(neu_seq_right, axis=0)
        sem_left   = sem(neu_seq_left, axis=0)
        sem_right  = sem(neu_seq_right, axis=0)
        upper = np.max([np.max(mean_left), np.max(mean_right)]) + \
                np.max([np.max(sem_left), np.max(sem_right)])
        lower = np.min([np.min(mean_left), np.min(mean_right)]) - \
                np.max([np.max(sem_left), np.max(sem_right)])
        ax.axvline(0, color='gold', lw=1, label='licking', linestyle='--')
        self.plot_mean_sem(ax, neu_time_lick, mean_left, sem_left, 'mediumseagreen', 'left')
        self.plot_mean_sem(ax, neu_time_lick, mean_right, sem_right, 'coral', 'right')
        adjust_layout_neu(ax)
        ax.set_xlim([np.min(neu_time_lick), np.max(neu_time_lick)])
        ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
        ax.set_xlabel('time since licking event (ms)')
    
    # roi response to all licking events.
    def roi_all(self, ax, roi_id):
        self.plot_lick(
            ax,
            self.neu_seq_lick_all, self.neu_time_lick_all, self.lick_direc_all,
            None, roi_id=roi_id)
        ax.set_title('response to all licking events')
    
    # roi response to all licking events quantification.
    def roi_all_box(self, ax, roi_id):
        _, _, color, _ = get_roi_label_color(self.labels, roi_id) 
        neu_cate = np.expand_dims(self.neu_seq_lick_all[:,roi_id,:], axis=1)
        neu_left  = neu_cate[self.lick_direc_all==0,:,:].copy()
        neu_right = neu_cate[self.lick_direc_all==1,:,:].copy()
        self.plot_win_mag_box(
            ax, neu_left, self.neu_time_lick_all, 'mediumseagreen', 0, -0.1)
        self.plot_win_mag_box(
            ax, neu_right, self.neu_time_lick_all, 'coral', 0, 0.1)
        ax.plot([], color='mediumseagreen', label='left')
        ax.plot([], color='coral', label='right')
        ax.legend(loc='upper right')
        ax.set_title('response to all licking events')
    
    # roi mean response to all licking events single trial heatmap.
    def roi_all_heatmap_trials(self, ax, roi_id):
        _, _, _, cmap = get_roi_label_color(self.labels, roi_id)
        neu_cate = np.expand_dims(self.neu_seq_lick_all[:,roi_id,:], axis=1)
        self.plot_heatmap_trials(
            ax, neu_cate, self.neu_time_lick_all, cmap, norm=True)
        ax.set_title('response to all licking events')
    
    # roi response to reaction licking.
    def roi_reaction(self, ax, roi_id):
        self.plot_lick(
            ax,
            self.neu_seq_lick_reaction, self.neu_time_lick_reaction, self.lick_direc_reaction,
            None, roi_id=roi_id)
        ax.set_title('response to reaction licking')
    
    # roi response to reaction licking quantification.
    def roi_reaction_box(self, ax, roi_id):
        _, _, color, _ = get_roi_label_color(self.labels, roi_id) 
        neu_cate = np.expand_dims(self.neu_seq_lick_reaction[:,roi_id,:], axis=1)
        neu_left  = neu_cate[self.lick_direc_reaction==0,:,:].copy()
        neu_right = neu_cate[self.lick_direc_reaction==1,:,:].copy()
        self.plot_win_mag_box(
            ax, neu_left, self.neu_time_lick_reaction, 'mediumseagreen', 0, -0.1)
        self.plot_win_mag_box(
            ax, neu_right, self.neu_time_lick_reaction, 'coral', 0, 0.1)
        ax.plot([], color='mediumseagreen', label='left')
        ax.plot([], color='coral', label='right')
        ax.legend(loc='upper right')
        ax.set_title('response to reaction licking')
    
    # roi mean response to reaction licking single trial heatmap.
    def roi_reaction_heatmap_trials(self, ax, roi_id):
        _, _, _, cmap = get_roi_label_color(self.labels, roi_id)
        neu_cate = np.expand_dims(self.neu_seq_lick_reaction[:,roi_id,:], axis=1)
        self.plot_heatmap_trials(
            ax, neu_cate, self.neu_time_lick_reaction, cmap, norm=True)
        ax.set_title('response to reaction licking')

    # roi response to decision licking.
    def roi_decision(self, ax, roi_id):
        self.plot_lick(
            ax,
            self.neu_seq_lick_decision, self.neu_time_lick_decision, self.lick_direc_decision,
            None, roi_id=roi_id)
        ax.set_title('response to decision licking')
    
    # roi response to decision licking quantification.
    def roi_decision_box(self, ax, roi_id):
        _, _, color, _ = get_roi_label_color(self.labels, roi_id) 
        neu_cate = np.expand_dims(self.neu_seq_lick_decision[:,roi_id,:], axis=1)
        neu_left  = neu_cate[self.lick_direc_decision==0,:,:].copy()
        neu_right = neu_cate[self.lick_direc_decision==1,:,:].copy()
        self.plot_win_mag_box(
            ax, neu_left, self.neu_time_lick_decision, 'mediumseagreen', 0, -0.1)
        self.plot_win_mag_box(
            ax, neu_right, self.neu_time_lick_decision, 'coral', 0, 0.1)
        ax.plot([], color='mediumseagreen', label='left')
        ax.plot([], color='coral', label='right')
        ax.legend(loc='upper right')
        ax.set_title('response to decision licking')
    
    # roi mean response to decision licking single trial heatmap.
    def roi_decision_heatmap_trials(self, ax, roi_id):
        _, _, _, cmap = get_roi_label_color(self.labels, roi_id)
        neu_cate = np.expand_dims(self.neu_seq_lick_decision[:,roi_id,:], axis=1)
        self.plot_heatmap_trials(
            ax, neu_cate, self.neu_time_lick_decision, cmap, norm=True)
        ax.set_title('response to decision licking')
        
class plotter_VIPTD_G8_align_beh(plotter_utils):
    def __init__(self, neural_trials, labels, significance):
        super().__init__(neural_trials, labels, significance)

    # excitory mean response to all licking events.
    def all_exc(self, ax):
        self.plot_lick(
            ax,
            self.neu_seq_lick_all,
            self.neu_time_lick_all,
            self.lick_direc_all,
            self.significance['r_lick_all'],
            cate=-1)
        ax.set_title('excitory response to all licking events')

    # inhibitory mean response to all licking events.
    def all_inh(self, ax):
        self.plot_lick(
            ax,
            self.neu_seq_lick_all,
            self.neu_time_lick_all,
            self.lick_direc_all,
            self.significance['r_lick_all'],
            cate=1)
        ax.set_title('inhibitory response to all licking events')

    # response heatmap to all licking events average across trials.
    def all_heatmap_neuron(self, ax):
        self.plot_heatmap_neuron(
            ax, self.neu_seq_lick_all, self.neu_time_lick_all)
        ax.set_title('response heatmap to all licking events')

    # excitory mean response to reaction licking.
    def reaction_exc(self, ax):
        self.plot_lick(
            ax,
            self.neu_seq_lick_reaction,
            self.neu_time_lick_reaction,
            self.lick_direc_reaction,
            self.significance['r_lick_reaction'],
            cate=-1)
        ax.set_title('excitory response to reaction licking')

    # inhibitory mean response to reaction licking.
    def reaction_inh(self, ax):
        self.plot_lick(
            ax,
            self.neu_seq_lick_reaction,
            self.neu_time_lick_reaction,
            self.lick_direc_reaction,
            self.significance['r_lick_reaction'],
            cate=1)
        ax.set_title('inhibitory response to reaction licking')

    # response heatmap to reaction licking average across trials.
    def reaction_heatmap_neuron(self, ax):
        self.plot_heatmap_neuron(
            ax, self.neu_seq_lick_reaction, self.neu_time_lick_reaction)
        ax.set_title('response heatmap to reaction licking')

    # excitory mean response to decision licking.
    def decision_exc(self, ax):
        self.plot_lick(
            ax,
            self.neu_seq_lick_decision,
            self.neu_time_lick_decision,
            self.lick_direc_decision,
            self.significance['r_lick_decision'],
            cate=-1)
        ax.set_title('excitory response to decision licking')

    # inhibitory mean response to decision licking.
    def decision_inh(self, ax):
        self.plot_lick(
            ax,
            self.neu_seq_lick_decision,
            self.neu_time_lick_decision,
            self.lick_direc_decision,
            self.significance['r_lick_decision'],
            cate=1)
        ax.set_title('inhibitory response to decision licking')

    # response heatmap to decision licking average across trials.
    def decision_heatmap_neuron(self, ax):
        self.plot_heatmap_neuron(
            ax, self.neu_seq_lick_decision, self.neu_time_lick_decision)
        ax.set_title('response heatmap to decision licking')
