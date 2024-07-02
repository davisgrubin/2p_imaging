import numpy as np
from scipy.stats import sem

from modules.Alignment import get_motor_response
from plot.utils import get_roi_label_color
from plot.utils import adjust_layout_neu
from plot.utils import utils


class plotter_utils(utils):
    
    def __init__(self, neural_trials, labels, significance):
        super().__init__(labels)
        
        self.l_frames = 30
        self.r_frames = 50
        
        [self.neu_seq_press1, self.neu_time_press1,
         self.outcome_press1] = get_motor_response(
            neural_trials, 'trial_press1', self.l_frames, self.r_frames)
        [self.neu_seq_retract1, self.neu_time_retract1,
         self.outcome_retract1] = get_motor_response(
            neural_trials, 'trial_retract1', self.l_frames, self.r_frames)
        [self.neu_seq_press2, self.neu_time_press2,
         self.outcome_press2] = get_motor_response(
            neural_trials, 'trial_press2', self.l_frames, self.r_frames)
        [self.neu_seq_lick, self.neu_time_lick, _] = get_motor_response(
            neural_trials, 'trial_lick', self.l_frames, self.r_frames)
        self.significance = significance
        
    def plot_lick(self, ax, s, cate=None, roi_id=None):
        if cate != None:
            neu_cate = self.neu_seq_lick[:,(self.labels==cate)*s,:]
            _, _, color, _ = get_roi_label_color([cate], 0)
        if roi_id != None:
            neu_cate = np.expand_dims(self.neu_seq_lick[:,roi_id,:], axis=1)
            _, _, color, _ = get_roi_label_color(self.labels, roi_id) 
        mean_neu = np.mean(neu_cate.reshape(-1, neu_cate.shape[2]), axis=0)
        sem_neu = sem(neu_cate.reshape(-1, neu_cate.shape[2]), axis=0)
        self.plot_mean_sem(ax, self.neu_time_lick, mean_neu, sem_neu, color, 'dff')
        upper = np.max(mean_neu) + np.max(sem_neu)
        lower = np.min(mean_neu) - np.max(sem_neu)
        ax.axvline(0, color='gold', lw=1, label='licking', linestyle='--')
        adjust_layout_neu(ax)
        ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
        ax.set_xlabel('time since licking (ms)')
    
    def plot_moto(self, ax, neu_seq, neu_time, outcome, s, cate=None, roi_id=None):
        if cate != None:
            neu_cate = neu_seq[:,(self.labels==cate)*s,:]
        if roi_id != None:
            neu_cate = np.expand_dims(neu_seq[:,roi_id,:], axis=1)
        mean_all = np.mean(neu_cate.reshape(-1, neu_cate.shape[2]), axis=0)
        mean_reward = np.mean(neu_cate[outcome==1,:,:].reshape(-1, neu_cate.shape[2]), axis=0)
        mean_punish = np.mean(neu_cate[outcome==-1,:,:].reshape(-1, neu_cate.shape[2]), axis=0)
        sem_all = sem(neu_cate.reshape(-1, neu_cate.shape[2]), axis=0)
        sem_reward = sem(neu_cate[outcome==1,:,:].reshape(-1, neu_cate.shape[2]), axis=0)
        sem_punish = sem(neu_cate[outcome==-1,:,:].reshape(-1, neu_cate.shape[2]), axis=0)
        self.plot_mean_sem(ax, neu_time, mean_all,    sem_all,    'grey',           'all')
        self.plot_mean_sem(ax, neu_time, mean_reward, sem_reward, 'mediumseagreen', 'reward')
        self.plot_mean_sem(ax, neu_time, mean_punish, sem_punish, 'coral',          'punish')
        upper = np.max([mean_all, mean_reward, mean_punish]) +\
                np.max([sem_all, sem_reward, sem_punish])
        lower = np.min([mean_all, mean_reward, mean_punish]) -\
                np.max([sem_all, sem_reward, sem_punish])
        ax.axvline(0, color='gold', lw=1, linestyle='--')
        adjust_layout_neu(ax)
        ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
    
    # roi mean response to 1st press.
    def roi_press1(self, ax, roi_id):
        self.plot_moto(
            ax, self.neu_seq_press1, self.neu_time_press1,
            self.outcome_press1, None, roi_id=roi_id)
        ax.set_xlabel('time since 1st press window end (ms)')
        ax.set_title('response to 1st press')
        
    # roi mean response to 1st press quantification.
    def roi_press1_box(self, ax, roi_id):
        neu_cate = np.expand_dims(self.neu_seq_press1[:,roi_id,:], axis=1)
        neu_reward = neu_cate[self.outcome_press1==1,:,:].copy()
        neu_punish = neu_cate[self.outcome_press1==-1,:,:].copy()
        self.plot_win_mag_box(ax, neu_reward, self.neu_time_press1, 'mediumseagreen', 0, -0.1)
        self.plot_win_mag_box(ax, neu_punish, self.neu_time_press1, 'coral', 0, 0.1)
        ax.set_title('response to 1st press')
        ax.plot([], color='mediumseagreen', label='reward')
        ax.plot([], color='coral', label='punish')
        ax.legend(loc='upper right')
        
    # roi mean response to 1st press single trial heatmap.
    def roi_press1_heatmap_trials(self, ax, roi_id):
        _, _, _, cmap = get_roi_label_color(self.labels, roi_id)
        self.plot_heatmap_trials(
            ax, self.neu_seq_press1[:,roi_id,:], self.neu_time_press1, cmap, norm=True)
        ax.set_title('response to 1st press')
        
    # roi mean response to 1st retract.
    def roi_retract1(self, ax, roi_id):
        self.plot_moto(
            ax, self.neu_seq_retract1, self.neu_time_retract1,
            self.outcome_retract1, None, roi_id=roi_id)
        ax.set_xlabel('time since 1st retract (ms)')
        ax.set_title('response to 1st retract')
    
    # roi mean response to 1st retract quantification.
    def roi_retract1_box(self, ax, roi_id):
        neu_cate = np.expand_dims(self.neu_seq_retract1[:,roi_id,:], axis=1)
        neu_reward = neu_cate[self.outcome_retract1==1,:,:].copy()
        neu_punish = neu_cate[self.outcome_retract1==-1,:,:].copy()
        self.plot_win_mag_box(ax, neu_reward, self.neu_time_retract1, 'mediumseagreen', 0, -0.1)
        self.plot_win_mag_box(ax, neu_punish, self.neu_time_retract1, 'coral', 0, 0.1)
        ax.set_title('response to 1st retract')
        ax.plot([], color='mediumseagreen', label='reward')
        ax.plot([], color='coral', label='punish')
        ax.legend(loc='upper right')
    
    # roi mean response to 2nd press single trial heatmap.
    def roi_retract1_heatmap_trials(self, ax, roi_id):
        _, _, _, cmap = get_roi_label_color(self.labels, roi_id)
        self.plot_heatmap_trials(
            ax, self.neu_seq_retract1[:,roi_id,:], self.neu_time_retract1, cmap, norm=True)
        ax.set_title('response to 1st retract')
        
    # roi mean response to 2nd press.
    def roi_press2(self, ax, roi_id):
        self.plot_moto(
            ax, self.neu_seq_press2, self.neu_time_press2,
            self.outcome_press2, None, roi_id=roi_id)
        ax.set_xlabel('time since 2nd press window end (ms)')
        ax.set_title('response to 2nd press')
    
    # roi mean response to 2nd press quantification.
    def roi_press2_box(self, ax, roi_id):
        neu_cate = np.expand_dims(self.neu_seq_press2[:,roi_id,:], axis=1)
        neu_reward = neu_cate[self.outcome_press2==1,:,:].copy()
        neu_punish = neu_cate[self.outcome_press2==-1,:,:].copy()
        self.plot_win_mag_box(ax, neu_reward, self.neu_time_press2, 'mediumseagreen', 0, -0.1)
        self.plot_win_mag_box(ax, neu_punish, self.neu_time_press2, 'coral', 0, 0.1)
        ax.set_title('response to 2nd press')
        ax.plot([], color='mediumseagreen', label='reward')
        ax.plot([], color='coral', label='punish')
        ax.legend(loc='upper right')
    
    # roi mean response to 2nd press single trial heatmap.
    def roi_press2_heatmap_trials(self, ax, roi_id):
        _, _, _, cmap = get_roi_label_color(self.labels, roi_id)
        self.plot_heatmap_trials(
            ax, self.neu_seq_press2[:,roi_id,:], self.neu_time_press2, cmap, norm=True)
        ax.set_title('response to 2nd press')
    
    # roi mean response to lick.
    def roi_lick(self, ax, roi_id):
        self.plot_lick(ax, None, roi_id=roi_id)
        ax.set_xlabel('time since lick (ms)')
        ax.set_title('response to lick')
    
    # roi mean response to lick quantification.
    def roi_lick_box(self, ax, roi_id):
        _, _, color, _ = get_roi_label_color(self.labels, roi_id) 
        neu_cate = np.expand_dims(self.neu_seq_lick[:,roi_id,:], axis=1)
        self.plot_win_mag_box(ax, neu_cate, self.neu_time_lick, color, 0, 0)
        ax.set_title('response to lick')
    
    # roi mean response to lick single trial heatmap.
    def roi_lick_heatmap_trials(self, ax, roi_id):
        _, _, _, cmap = get_roi_label_color(self.labels, roi_id)
        self.plot_heatmap_trials(
            ax, self.neu_seq_lick[:,roi_id,:], self.neu_time_lick, cmap, norm=True)
        ax.set_title('response to lick')
        
class plotter_VIPTD_G8_motor(plotter_utils):
    def __init__(self, neural_trials, labels, significance):
        super().__init__(neural_trials, labels, significance)
    
    # excitory mean response to 1st press.
    def press1_exc(self, ax):
        self.plot_moto(
            ax, self.neu_seq_press1, self.neu_time_press1,
            self.outcome_press1, self.significance['r_press1'], cate=-1)
        ax.set_xlabel('time since 1st press window end (ms)')
        ax.set_title('excitory response to 1st press')
    
    # inhibitory mean response to 1st press.
    def press1_inh(self, ax):
        self.plot_moto(
            ax, self.neu_seq_press1, self.neu_time_press1,
            self.outcome_press1, self.significance['r_press1'], cate=1)
        ax.set_xlabel('time since 1st press window end (ms)')
        ax.set_title('inhibitory response to 1st press')
    
    # response to 1st pressing heatmap average across trials.
    def press1_heatmap_neuron(self, ax):
        self.plot_heatmap_neuron(ax, self.neu_seq_press1, self.neu_time_press1)
        ax.set_xlabel('time since 1st press window end (ms)')
        ax.set_title('response to 1st press')
    
    # excitory mean response to 1st retract.
    def retract1_exc(self, ax):
        self.plot_moto(
            ax, self.neu_seq_retract1, self.neu_time_retract1,
            self.outcome_retract1, self.significance['r_retract1'], cate=-1)
        ax.set_xlabel('time since 1st retract (ms)')
        ax.set_title('excitory response to 1st retract')
    
    # inhibitory mean response to 1st press.
    def retract1_inh(self, ax):
        self.plot_moto(
            ax, self.neu_seq_retract1, self.neu_time_retract1,
            self.outcome_retract1, self.significance['r_retract1'], cate=1)
        ax.set_xlabel('time since 1st retract (ms)')
        ax.set_title('inhibitory response to 1st retract')
    
    # response to 1st retract heatmap average across trials.
    def retract1_heatmap_neuron(self, ax):
        self.plot_heatmap_neuron(ax, self.neu_seq_retract1, self.neu_time_retract1)
        ax.set_xlabel('time since 1st retract (ms)')
        ax.set_title('response to 1st retract')
    
    # excitory mean response to 2nd press.
    def press2_exc(self, ax):
        self.plot_moto(
            ax, self.neu_seq_press2, self.neu_time_press2,
            self.outcome_press2, self.significance['r_press2'], cate=-1)
        ax.set_xlabel('time since 2nd press window end (ms)')
        ax.set_title('excitory response to 2nd pressing')
    
    # inhibitory mean response to 2nd press.
    def press2_inh(self, ax):
        self.plot_moto(
            ax, self.neu_seq_press2, self.neu_time_press2,
            self.outcome_press2, self.significance['r_press2'], cate=1)
        ax.set_xlabel('time since 2nd press window end (ms)')
        ax.set_title('inhibitory response to 2nd pressing')
    
    # response to 2nd pressing heatmap average across trials.
    def press2_heatmap_neuron(self, ax):
        self.plot_heatmap_neuron(ax, self.neu_seq_press2, self.neu_time_press2)
        ax.set_xlabel('time since 2nd press window end (ms)')
        ax.set_title('response to 2nd pressing')
    
    # excitory mean response to licking.
    def lick_exc(self, ax):
        self.plot_lick(ax, self.significance['r_lick'], cate=-1)
        ax.set_xlabel('time since lick (ms)')
        ax.set_title('excitory response to lick')
    
    # inhibitory mean response to licking.
    def lick_inh(self, ax):
        self.plot_lick(ax, self.significance['r_lick'], cate=1)
        ax.set_xlabel('time since lick (ms)')
        ax.set_title('inhibitory response to lick')
    
    # response to licking heatmap average across trials.
    def lick_heatmap_neuron(self, ax):
        self.plot_heatmap_neuron(ax, self.neu_seq_lick, self.neu_time_lick)
        ax.set_xlabel('time since lick (ms)')
        ax.set_title('response to lick')


class plotter_VIPG8_motor(plotter_utils):
    def __init__(self, neural_trials, labels, significance):
        super().__init__(neural_trials, labels, significance)
    
    # mean response to licking.
    def lick(self, ax):
        self.plot_lick(ax, cate=1)
        ax.set_title('response to licking')
    
    # response to licking heatmap average across trials.
    def lick_heatmap_neuron(self, ax):
        self.plot_heatmap_neuron(ax, self.neu_seq_lick, self.neu_time_lick)
        ax.set_title('response to licking')

    # mean response to 1st press.
    def press1(self, ax):
        self.plot_moto(
            ax, self.neu_seq_press1, self.neu_time_press1,
            self.outcome_press1, cate=1)
        ax.set_title('response to 1st pressing')
    
    # response to 1st pressing heatmap average across trials.
    def press1_heatmap_neuron(self, ax):
        self.plot_heatmap_neuron(ax, self.neu_seq_press1, self.neu_time_press1)
        ax.set_title('response to 1st pressing')

    # mean response to 2nd press.
    def press2(self, ax):
        self.plot_moto(
            ax, self.neu_seq_press2, self.neu_time_press2,
            self.outcome_press2, cate=1)
        ax.set_title('response to 2nd pressing')
    
    # response to 2nd pressing heatmap average across trials.
    def press2_heatmap_neuron(self, ax):
        self.plot_heatmap_neuron(ax, self.neu_seq_press2, self.neu_time_press2)
        ax.set_title('response to 2nd pressing')
    