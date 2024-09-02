import numpy as np

from modules.Alignment import get_motor_response
from modules.Alignment import get_lick_response
from plot.utils import get_block_epoch
from plot.utils import get_trial_type
from plot.utils import get_mean_sem
from plot.utils import get_roi_label_color
from plot.utils import adjust_layout_neu
from plot.utils import utils


class plotter_utils(utils):
    
    def __init__(self, neural_trials, labels, significance, cate_delay):
        super().__init__(labels)
        self.l_frames = 30
        self.r_frames = 50
        [self.neu_seq_push1, self.neu_time_push1,
         self.outcome_push1, self.delay_push1] = get_motor_response(
            neural_trials, 'trial_push1', self.l_frames, self.r_frames)
        [self.neu_seq_retract1, self.neu_time_retract1,
         self.outcome_retract1, self.delay_retract1] = get_motor_response(
            neural_trials, 'trial_retract1', self.l_frames, self.r_frames)
        [self.neu_seq_push2, self.neu_time_push2,
         self.outcome_push2, self.delay_push2] = get_motor_response(
            neural_trials, 'trial_push2', self.l_frames, self.r_frames)
        [self.neu_seq_wait2, self.neu_time_wait2,
         self.outcome_wait2, self.delay_wait2] = get_motor_response(
            neural_trials, 'trial_wait2', self.l_frames, self.r_frames)
        [self.neu_seq_retract2, self.neu_time_retract2,
         self.outcome_retract2, self.delay_retract2] = get_motor_response(
            neural_trials, 'trial_retract2', self.l_frames, self.r_frames)
        [self.neu_seq_lick, self.neu_time_lick, self.lick_label] = get_lick_response(
            neural_trials, self.l_frames, self.r_frames)
        self.significance = significance
        self.cate_delay = cate_delay
        
    def plot_lick(self, ax, s, cate=None, roi_id=None):
        if not np.isnan(np.sum(self.neu_seq_lick)):
            if cate != None:
                neu_cate = self.neu_seq_lick[:,(self.labels==cate)*s,:]
                _, color1, color2, _ = get_roi_label_color([cate], 0)
            if roi_id != None:
                neu_cate = np.expand_dims(self.neu_seq_lick[:,roi_id,:], axis=1)
                _, color1, color2, _ = get_roi_label_color(self.labels, roi_id)
            mean_spont, sem_spont = get_mean_sem(neu_cate[self.lick_label==0,:,:])
            mean_consume, sem_consume = get_mean_sem(neu_cate[self.lick_label==1,:,:])
            self.plot_mean_sem(ax, self.neu_time_lick, mean_spont, sem_spont, color1, 'spont')
            self.plot_mean_sem(ax, self.neu_time_lick, mean_consume, sem_consume, color2, 'consume')
            upper = np.max([mean_spont, mean_consume]) + np.max([sem_spont, sem_consume])
            lower = np.min([mean_spont, mean_consume]) - np.max([sem_spont, sem_consume])
            ax.axvline(0, color='grey', lw=1, label='licking', linestyle='--')
            adjust_layout_neu(ax)
            ax.set_ylim([lower, upper])
            ax.set_xlabel('time since licking (ms)')
    
    def plot_moto_outcome(
            self, ax,
            neu_seq, neu_time, outcome,
            delay, block, s,
            cate=None, roi_id=None):
        if not np.isnan(np.sum(neu_seq)):
            if cate != None:
                neu_cate = neu_seq[:,(self.labels==cate)*s,:]
            if roi_id != None:
                neu_cate = np.expand_dims(neu_seq[:,roi_id,:], axis=1)
            idx = get_trial_type(self.cate_delay, delay, block)
            mean = []
            sem = []
            for i in range(4):
                trial_idx = idx*(outcome==i)
                if len(trial_idx) >= self.min_num_trial:
                    m, s = get_mean_sem(neu_cate[trial_idx,:,:])
                    self.plot_mean_sem(ax, neu_time, m, s, self.colors[i], self.states[i])
                    mean.append(m)
                    sem.append(s)
            upper = np.nanmax(mean) + np.nanmax(sem)
            lower = np.nanmin(mean) - np.nanmax(sem)
            ax.axvline(0, color='grey', lw=1, linestyle='--')
            adjust_layout_neu(ax)
            ax.set_ylim([lower, upper])
    
    def plot_moto(
            self, ax,
            neu_seq, neu_time,
            delay, block, s,
            cate=None, roi_id=None):
        if not np.isnan(np.sum(neu_seq)):
            if cate != None:
                neu_cate = neu_seq[:,(self.labels==cate)*s,:]
                _, _, color, _ = get_roi_label_color([cate], 0)
            if roi_id != None:
                neu_cate = np.expand_dims(neu_seq[:,roi_id,:], axis=1)
                _, _, color, _ = get_roi_label_color(self.labels, roi_id)
            idx = get_trial_type(self.cate_delay, delay, block)
            neu_mean, neu_sem = get_mean_sem(neu_cate[idx,:,:])
            self.plot_mean_sem(ax, neu_time, neu_mean, neu_sem, color, 'all')
            upper = np.max(neu_mean) + np.max(neu_sem)
            lower = np.min(neu_mean) - np.max(neu_sem)
            ax.axvline(0, color='grey', lw=1, linestyle='--')
            adjust_layout_neu(ax)
            ax.set_ylim([lower, upper])
    
    def plot_motor_epoch(
            self, ax,
            neu_seq, neu_time, outcome,
            delay, block, s,
            cate=None, roi_id=None):
        if cate != None:
            neu_cate = neu_seq[:,(self.labels==cate)*s,:]
            _, color1, color2, _ = get_roi_label_color([cate], 0)
        if roi_id != None:
            neu_cate = np.expand_dims(neu_seq[:,roi_id,:], axis=1)
            _, color1, color2, _ = get_roi_label_color(self.labels, roi_id)
        idx = get_trial_type(self.cate_delay, delay, block)
        trial_idx, block_tran = get_block_epoch(idx)
        i_ep1 = (block_tran==1) * trial_idx * idx * (outcome==0)
        i_ep2 = (block_tran==0) * trial_idx * idx * (outcome==0)
        m_ep1, s_ep1 = get_mean_sem(neu_cate[i_ep1,:,:])
        m_ep2, s_ep2 = get_mean_sem(neu_cate[i_ep2,:,:])
        if not np.isnan(np.sum(m_ep1)) and not np.isnan(np.sum(m_ep2)):
            self.plot_mean_sem(ax, neu_time, m_ep1, s_ep1, color1, 'ep1')
            self.plot_mean_sem(ax, neu_time, m_ep2, s_ep2, color2, 'ep2')
            upper = np.nanmax([m_ep1, m_ep2]) + np.nanmax([s_ep1, s_ep2])
            lower = np.nanmin([m_ep1, m_ep2]) - np.nanmax([s_ep1, s_ep2])
            ax.axvline(0, color='grey', lw=1, linestyle='--')
            adjust_layout_neu(ax)
            ax.set_ylim([lower, upper])
    
    def plot_push_onset(
            self, ax,
            s, cate=None, roi_id=None):
        if cate != None:
            neu_push1 = self.neu_seq_push1[:,(self.labels==cate)*s,:]
            neu_push2 = self.neu_seq_push2[:,(self.labels==cate)*s,:]
            _, color1, _, _ = get_roi_label_color([cate], 0)
        if roi_id != None:
            neu_push1 = np.expand_dims(self.neu_seq_push1[:,roi_id,:], axis=1)
            neu_push2 = np.expand_dims(self.neu_seq_push2[:,roi_id,:], axis=1)
            _, color1, _, _ = get_roi_label_color(self.labels, roi_id)
        if not np.isnan(np.sum(neu_push1)) and not np.isnan(np.sum(neu_push2)):
            m_push1_all,    s_push1_all    = get_mean_sem(neu_push1[self.outcome_push1!=1,:,:])
            m_push2_reward, s_push2_reward = get_mean_sem(neu_push2[self.outcome_push2==0,:,:])
            m_push2_punish, s_push2_punish = get_mean_sem(neu_push2[self.outcome_push2==3,:,:])
            ax.axvline(0, color='grey', lw=1, label='PushOnset', linestyle='--')
            self.plot_mean_sem(
                ax, self.neu_time_push1, m_push1_all, s_push1_all,
                color1, 'push1 all')
            self.plot_mean_sem(
                ax, self.neu_time_push2, m_push2_reward, s_push2_reward,
                self.colors[0], 'push2 reward')
            self.plot_mean_sem(
                ax, self.neu_time_push2, m_push2_punish, s_push2_punish,
                self.colors[3], 'push2 early')
            upper = np.nanmax([m_push1_all, m_push2_reward, m_push2_punish]) +\
                    np.nanmax([s_push1_all, s_push2_reward, s_push2_punish])
            lower = np.nanmin([m_push1_all, m_push2_reward, m_push2_punish]) -\
                    np.nanmax([s_push1_all, s_push2_reward, s_push2_punish])
            adjust_layout_neu(ax)
            ax.set_ylim([lower, upper])
        
    # roi response to PushOnset1 (short).
    def roi_short_push1(self, ax, roi_id):
        self.plot_moto_outcome(
            ax, self.neu_seq_push1, self.neu_time_push1, self.outcome_push1,
            self.delay_push1, 0, None, roi_id=roi_id)
        ax.set_xlabel('time since PushOnset1 (ms)')
        ax.set_title('response to PushOnset1 (short)')
    
    # roi response to Retract1 end (short).
    def roi_short_retract1(self, ax, roi_id):
        self.plot_moto_outcome(
            ax, self.neu_seq_retract1, self.neu_time_retract1, self.outcome_retract1,
            self.delay_retract1, 0, None, roi_id=roi_id)
        ax.set_xlabel('time since Retract1 end (ms)')
        ax.set_title('response to Retract1 end (short)')
    
    # roi response to WaitForPush2 start (short).
    def roi_short_wait2(self, ax, roi_id):
        self.plot_moto_outcome(
            ax, self.neu_seq_wait2, self.neu_time_wait2, self.outcome_wait2,
            self.delay_wait2, 0, None, roi_id=roi_id)
        ax.set_xlabel('time since WaitForPush2 start (ms)')
        ax.set_title('response to WaitForPush2 start (short)')
      
    # roi response to PushOnset2 (short).
    def roi_short_push2(self, ax, roi_id):
        self.plot_moto_outcome(
            ax, self.neu_seq_push2, self.neu_time_push2, self.outcome_push2,
            self.delay_push2, 0, None, roi_id=roi_id)
        ax.set_xlabel('time since PushOnset2 (ms)')
        ax.set_title('response to PushOnset2 (short)')
    
    # roi response to Retract2 (short).
    def roi_short_retract2(self, ax, roi_id):
        self.plot_moto(
            ax, self.neu_seq_retract2, self.neu_time_retract2,
            self.delay_retract2, 0, None, roi_id=roi_id)
        ax.set_xlabel('time since Retract2 (ms)')
        ax.set_title('response to Retract2 (short)')
    
    # roi response to PushOnset1 (long).
    def roi_long_push1(self, ax, roi_id):
        self.plot_moto_outcome(
            ax, self.neu_seq_push1, self.neu_time_push1, self.outcome_push1,
            self.delay_push1, 0, None, roi_id=roi_id)
        ax.set_xlabel('time since PushOnset1 (ms)')
        ax.set_title('response to PushOnset1 (long)')
    
    # roi response to Retract1 end (long).
    def roi_long_retract1(self, ax, roi_id):
        self.plot_moto_outcome(
            ax, self.neu_seq_retract1, self.neu_time_retract1, self.outcome_retract1,
            self.delay_retract1, 0, None, roi_id=roi_id)
        ax.set_xlabel('time since Retract1 end (ms)')
        ax.set_title('response to Retract1 end (long)')
    
    # roi response to WaitForPush2 start (long).
    def roi_long_wait2(self, ax, roi_id):
        self.plot_moto_outcome(
            ax, self.neu_seq_wait2, self.neu_time_wait2, self.outcome_wait2,
            self.delay_wait2, 0, None, roi_id=roi_id)
        ax.set_xlabel('time since WaitForPush2 start (ms)')
        ax.set_title('response to WaitForPush2 start (long)')
      
    # roi response to PushOnset2 (long).
    def roi_long_push2(self, ax, roi_id):
        self.plot_moto_outcome(
            ax, self.neu_seq_push2, self.neu_time_push2, self.outcome_push2,
            self.delay_push2, 0, None, roi_id=roi_id)
        ax.set_xlabel('time since PushOnset2 (ms)')
        ax.set_title('response to PushOnset2 (long)')
    
    # roi response to Retract2 (long).
    def roi_long_retract2(self, ax, roi_id):
        self.plot_moto(
            ax, self.neu_seq_retract2, self.neu_time_retract2,
            self.delay_retract2, 0, None, roi_id=roi_id)
        ax.set_xlabel('time since Retract2 (ms)')
        ax.set_title('response to Retract2 (long)')
    
    # roi response to push osnet.
    def roi_onset(self, ax, roi_id):
        self.plot_push_onset(ax, None, roi_id=roi_id)
        ax.set_xlabel('time since push onset (ms)')
        ax.set_title('response to all push onset')
    
    # roi response to licking.
    def roi_lick(self, ax, roi_id):
        self.plot_lick(ax, None, roi_id=roi_id)
        ax.set_xlabel('time since lick (ms)')
        ax.set_title('response to lick')
    
    # roi response to PushOnset1 with epoch (short).
    def roi_short_epoch_push1(self, ax, roi_id):
        self.plot_motor_epoch(
            ax,
            self.neu_seq_push1, self.neu_time_push1, self.outcome_push1,
            self.delay_push1, 0,
            None, roi_id=roi_id)
        ax.set_xlabel('time since PushOnset1 (ms)')
        ax.set_title('response to PushOnset1 (reward)')
    
    # roi response to Retract1 end with epoch (short).
    def roi_short_epoch_retract1(self, ax, roi_id):
        self.plot_motor_epoch(
            ax,
            self.neu_seq_retract1, self.neu_time_retract1, self.outcome_retract1,
            self.delay_retract1, 0,
            None, roi_id=roi_id)
        ax.set_xlabel('time since Retract1 end (ms)')
        ax.set_title('response to Retract1 end (reward)')
    
    # roi response to WaitForPush2 start with epoch (short).
    def roi_short_epoch_wait2(self, ax, roi_id):
        self.plot_motor_epoch(
            ax,
            self.neu_seq_wait2, self.neu_time_wait2, self.outcome_wait2,
            self.delay_wait2, 0,
            None, roi_id=roi_id)
        ax.set_xlabel('time since WaitForPush2 start (ms)')
        ax.set_title('response to WaitForPush2 start (reward)')
    
    # roi response to PushOnset2 with epoch (short).
    def roi_short_epoch_push2(self, ax, roi_id):
        self.plot_motor_epoch(
            ax,
            self.neu_seq_push2, self.neu_time_push2, self.outcome_push2,
            self.delay_push2, 0,
            None, roi_id=roi_id)
        ax.set_xlabel('time since PushOnset2 (ms)')
        ax.set_title('response to PushOnset2 (reward)')
    
    # roi response to Retract2 with epoch (short).
    def roi_short_epoch_retract2(self, ax, roi_id):
        self.plot_motor_epoch(
            ax,
            self.neu_seq_retract2, self.neu_time_retract2, self.outcome_retract2,
            self.delay_retract2, 0,
            None, roi_id=roi_id)
        ax.set_xlabel('time since Retract2 (ms)')
        ax.set_title('response to Retract2 (reward)')
    
    # roi response to PushOnset1 with epoch (long).
    def roi_long_epoch_push1(self, ax, roi_id):
        self.plot_motor_epoch(
            ax,
            self.neu_seq_push1, self.neu_time_push1, self.outcome_push1,
            self.delay_push1, 1,
            None, roi_id=roi_id)
        ax.set_xlabel('time since PushOnset1 (ms)')
        ax.set_title('response to PushOnset1 (reward)')
    
    # roi response to Retract1 end with epoch (long).
    def roi_long_epoch_retract1(self, ax, roi_id):
        self.plot_motor_epoch(
            ax,
            self.neu_seq_retract1, self.neu_time_retract1, self.outcome_retract1,
            self.delay_retract1, 1,
            None, roi_id=roi_id)
        ax.set_xlabel('time since Retract1 end (ms)')
        ax.set_title('response to Retract1 end (reward)')
    
    # roi response to WaitForPush2 start with epoch (long).
    def roi_long_epoch_wait2(self, ax, roi_id):
        self.plot_motor_epoch(
            ax,
            self.neu_seq_wait2, self.neu_time_wait2, self.outcome_wait2,
            self.delay_wait2, 1,
            None, roi_id=roi_id)
        ax.set_xlabel('time since WaitForPush2 start (ms)')
        ax.set_title('response to WaitForPush2 start (reward)')
    
    # roi response to PushOnset2 with epoch (long).
    def roi_long_epoch_push2(self, ax, roi_id):
        self.plot_motor_epoch(
            ax,
            self.neu_seq_push2, self.neu_time_push2, self.outcome_push2,
            self.delay_push2, 1,
            None, roi_id=roi_id)
        ax.set_xlabel('time since PushOnset2 (ms)')
        ax.set_title('response to PushOnset2 (reward)')
    
    # roi response to Retract2 with epoch (long).
    def roi_long_epoch_retract2(self, ax, roi_id):
        self.plot_motor_epoch(
            ax,
            self.neu_seq_retract2, self.neu_time_retract2, self.outcome_retract2,
            self.delay_retract2, 1,
            None, roi_id=roi_id)
        ax.set_xlabel('time since Retract2 (ms)')
        ax.set_title('response to Retract2 (reward)')
    
    def all_roi_motor_align(self, axs, roi_id):
        self.roi_short_push1(axs[0][0], roi_id)
        self.roi_short_retract1(axs[0][1], roi_id)
        self.roi_short_wait2(axs[0][2], roi_id)
        self.roi_short_push2(axs[0][3], roi_id)
        self.roi_short_retract2(axs[0][4], roi_id)
        self.roi_long_push1(axs[1][0], roi_id)
        self.roi_long_retract1(axs[1][1], roi_id)
        self.roi_long_wait2(axs[1][2], roi_id)
        self.roi_long_push2(axs[1][3], roi_id)
        self.roi_long_retract2(axs[1][4], roi_id)
        self.roi_onset(axs[2], roi_id)
        self.roi_lick(axs[3], roi_id)
    
    def all_roi_epoch_motor_align(self, axs, roi_id):
        self.roi_short_epoch_push1(axs[0][0], roi_id)
        self.roi_short_epoch_retract1(axs[0][1], roi_id)
        self.roi_short_epoch_wait2(axs[0][2], roi_id)
        self.roi_short_epoch_push2(axs[0][3], roi_id)
        self.roi_short_epoch_retract2(axs[0][4], roi_id)
        self.roi_long_epoch_push1(axs[1][0], roi_id)
        self.roi_long_epoch_retract1(axs[1][1], roi_id)
        self.roi_long_epoch_wait2(axs[1][2], roi_id)
        self.roi_long_epoch_push2(axs[1][3], roi_id)
        self.roi_long_epoch_retract2(axs[1][4], roi_id)
        
        
class plotter_VIPTD_G8_motor(plotter_utils):
    def __init__(self, neural_trials, labels, significance, cate_delay):
        super().__init__(neural_trials, labels, significance, cate_delay)
    
    def plot_exc_inh(self, ax, neu_seq, neu_time, delay, block, s):
        if not np.isnan(np.sum(neu_seq)):
            _, _, color_exc, _ = get_roi_label_color([-1], 0)
            _, _, color_inh, _ = get_roi_label_color([1], 0)
            idx = get_trial_type(self.cate_delay, delay, block)
            neu_cate = neu_seq[idx,:,:].copy()
            mean_exc, sem_exc = get_mean_sem(neu_cate[:,(self.labels==-1)*s,:])
            mean_inh, sem_inh = get_mean_sem(neu_cate[:,(self.labels==1)*s,:])
            self.plot_mean_sem(ax, neu_time, mean_exc, sem_exc, color_exc, 'exc')
            self.plot_mean_sem(ax, neu_time, mean_inh, sem_inh, color_inh, 'inh')
            upper = np.nanmax([mean_exc, mean_inh]) + np.nanmax([sem_exc, sem_inh])
            lower = np.nanmin([mean_exc, mean_inh]) - np.nanmax([sem_exc, sem_inh])
            adjust_layout_neu(ax)
            ax.axvline(0, color='grey', lw=1, linestyle='--')
            ax.set_ylim([lower, upper])
    
    def all_short_motor_align_exc(self, axs):
        self.short_push1_exc(axs[0])
        self.short_retract1_exc(axs[1])
        self.short_wait2_exc(axs[2])
        self.short_push2_exc(axs[3])
        self.short_retract2_exc(axs[4])
    
    def all_short_motor_align_inh(self, axs):
        self.short_push1_inh(axs[0])
        self.short_retract1_inh(axs[1])
        self.short_wait2_inh(axs[2])
        self.short_push2_inh(axs[3])
        self.short_retract2_inh(axs[4])
        
    def all_short_motor_align_heatmap_neuron(self, axs):
        self.short_push1_heatmap_neuron(axs[0])
        self.short_retract1_heatmap_neuron(axs[1])
        self.short_wait2_heatmap_neuron(axs[2])
        self.short_push2_heatmap_neuron(axs[3])
        self.short_retract2_heatmap_neuron(axs[4])
    
    def all_long_motor_align_exc(self, axs):
        self.long_push1_exc(axs[0])
        self.long_retract1_exc(axs[1])
        self.long_wait2_exc(axs[2])
        self.long_push2_exc(axs[3])
        self.long_retract2_exc(axs[4])
    
    def all_long_motor_align_inh(self, axs):
        self.long_push1_inh(axs[0])
        self.long_retract1_inh(axs[1])
        self.long_wait2_inh(axs[2])
        self.long_push2_inh(axs[3])
        self.long_retract2_inh(axs[4])
        
    def all_long_motor_align_heatmap_neuron(self, axs):
        self.long_push1_heatmap_neuron(axs[0])
        self.long_retract1_heatmap_neuron(axs[1])
        self.long_wait2_heatmap_neuron(axs[2])
        self.long_push2_heatmap_neuron(axs[3])
        self.long_retract2_heatmap_neuron(axs[4])
    
    def all_short_epoch_motor_align_exc(self, axs):
        self.short_epoch_push1_exc(axs[0])
        self.short_epoch_retract1_exc(axs[1])
        self.short_epoch_wait2_exc(axs[2])
        self.short_epoch_push2_exc(axs[3])
        self.short_epoch_retract2_exc(axs[4])
    
    def all_short_epoch_motor_align_inh(self, axs):
        self.short_epoch_push1_inh(axs[0])
        self.short_epoch_retract1_inh(axs[1])
        self.short_epoch_wait2_inh(axs[2])
        self.short_epoch_push2_inh(axs[3])
        self.short_epoch_retract2_inh(axs[4])
    
    def all_long_epoch_motor_align_exc(self, axs):
        self.long_epoch_push1_exc(axs[0])
        self.long_epoch_retract1_exc(axs[1])
        self.long_epoch_wait2_exc(axs[2])
        self.long_epoch_push2_exc(axs[3])
        self.long_epoch_retract2_exc(axs[4])
    
    def all_long_epoch_motor_align_inh(self, axs):
        self.long_epoch_push1_inh(axs[0])
        self.long_epoch_retract1_inh(axs[1])
        self.long_epoch_wait2_inh(axs[2])
        self.long_epoch_push2_inh(axs[3])
        self.long_epoch_retract2_inh(axs[4])
    
    # excitory response to PushOnset1 (short).
    def short_push1_exc(self, ax):
        self.plot_moto_outcome(
            ax, self.neu_seq_push1, self.neu_time_push1, self.outcome_push1,
            self.delay_push1, 0, self.significance['r_push'], cate=-1)
        ax.set_xlabel('time since PushOnset1 (ms)')
        ax.set_title('excitory response to PushOnset1')
    
    # inhibitory response to PushOnset1 (short).
    def short_push1_inh(self, ax):
        self.plot_moto_outcome(
            ax, self.neu_seq_push1, self.neu_time_push1, self.outcome_push1,
            self.delay_push1, 0, self.significance['r_push'], cate=1)
        ax.set_xlabel('time since PushOnset1 (ms)')
        ax.set_title('inhibitory response to PushOnset1')
    
    # response to PushOnset1ing heatmap average across trials (short).
    def short_push1_heatmap_neuron(self, ax):
        idx = get_trial_type(self.cate_delay, self.delay_push1, 0)
        self.plot_heatmap_neuron(
            ax, self.neu_seq_push1[idx,:,:], self.neu_time_push1, self.significance['r_push'])
        ax.set_xlabel('time since PushOnset1 (ms)')
        ax.set_title('response to PushOnset1')
    
    # excitory response to Retract1 end (short).
    def short_retract1_exc(self, ax):
        self.plot_moto_outcome(
            ax, self.neu_seq_retract1, self.neu_time_retract1, self.outcome_retract1,
            self.delay_retract1, 0, self.significance['r_retract'], cate=-1)
        ax.set_xlabel('time since Retract1 end (ms)')
        ax.set_title('excitory response to Retract1 end')
    
    # inhibitory response to PushOnset1 (short).
    def short_retract1_inh(self, ax):
        self.plot_moto_outcome(
            ax, self.neu_seq_retract1, self.neu_time_retract1, self.outcome_retract1,
            self.delay_retract1, 0, self.significance['r_retract'], cate=1)
        ax.set_xlabel('time since Retract1 end (ms)')
        ax.set_title('inhibitory response to Retract1 end')
    
    # response to Retract1 end heatmap average across trials (short).
    def short_retract1_heatmap_neuron(self, ax):
        idx = get_trial_type(self.cate_delay, self.delay_retract1, 0)
        self.plot_heatmap_neuron(
            ax, self.neu_seq_retract1[idx,:,:], self.neu_time_retract1, self.significance['r_retract'])
        ax.set_xlabel('time since Retract1 end (ms)')
        ax.set_title('response to Retract1 end')
    
    # excitory response to WaitForPush2 start (short).
    def short_wait2_exc(self, ax):
        self.plot_moto_outcome(
            ax, self.neu_seq_wait2, self.neu_time_wait2, self.outcome_wait2,
            self.delay_wait2, 0, self.significance['r_wait'], cate=-1)
        ax.set_xlabel('time since WaitForPush2 start (ms)')
        ax.set_title('excitory response to WaitForPush2 start')
    
    # inhibitory response to WaitForPush2 start (short).
    def short_wait2_inh(self, ax):
        self.plot_moto_outcome(
            ax, self.neu_seq_wait2, self.neu_time_wait2, self.outcome_wait2,
            self.delay_wait2, 0, self.significance['r_wait'], cate=1)
        ax.set_xlabel('time since WaitForPush2 start (ms)')
        ax.set_title('inhibitory response to WaitForPush2 start')
    
    # response to WaitForPush2 start heatmap average across trials (short).
    def short_wait2_heatmap_neuron(self, ax):
        idx = get_trial_type(self.cate_delay, self.delay_wait2, 0)
        self.plot_heatmap_neuron(
            ax, self.neu_seq_wait2[idx,:,:], self.neu_time_wait2, self.significance['r_wait'])
        ax.set_xlabel('time since WaitForPush2 start (ms)')
        ax.set_title('response to WaitForPush2 start')
    
    # excitory response to PushOnset2 (short).
    def short_push2_exc(self, ax):
        self.plot_moto_outcome(
            ax, self.neu_seq_push2, self.neu_time_push2, self.outcome_push2,
            self.delay_push2, 0, self.significance['r_push'], cate=-1)
        ax.set_xlabel('time since PushOnset2 (ms)')
        ax.set_title('excitory response to PushOnset2')
    
    # inhibitory response to PushOnset2 (short).
    def short_push2_inh(self, ax):
        self.plot_moto_outcome(
            ax, self.neu_seq_push2, self.neu_time_push2, self.outcome_push2,
            self.delay_push2, 0, self.significance['r_push'], cate=1)
        ax.set_xlabel('time since PushOnset2 (ms)')
        ax.set_title('inhibitory response to PushOnset2')
    
    # response to PushOnset2  heatmap average across trials (short).
    def short_push2_heatmap_neuron(self, ax):
        idx = get_trial_type(self.cate_delay, self.delay_push2, 0)
        self.plot_heatmap_neuron(
            ax, self.neu_seq_push2[idx,:,:], self.neu_time_push2, self.significance['r_push'])
        ax.set_xlabel('time since PushOnset2 (ms)')
        ax.set_title('response to PushOnset2 ')
    
    # excitory response to Retract2 (short).
    def short_retract2_exc(self, ax):
        self.plot_moto(
            ax, self.neu_seq_retract2, self.neu_time_retract2,
            self.delay_retract2, 0, self.significance['r_retract'], cate=-1)
        ax.set_xlabel('time since Retract2 (ms)')
        ax.set_title('excitory response to Retract2')
    
    # inhibitory response to PushOnset2 (short).
    def short_retract2_inh(self, ax):
        self.plot_moto(
            ax, self.neu_seq_retract2, self.neu_time_retract2,
            self.delay_retract2, 0, self.significance['r_retract'], cate=1)
        ax.set_xlabel('time since Retract2 (ms)')
        ax.set_title('inhibitory response to Retract2')
    
    # response to Retract2 heatmap average across trials (short).
    def short_retract2_heatmap_neuron(self, ax):
        idx = get_trial_type(self.cate_delay, self.delay_retract2, 0)
        self.plot_heatmap_neuron(
            ax, self.neu_seq_retract2[idx,:,:], self.neu_time_retract2, self.significance['r_retract'])
        ax.set_xlabel('time since Retract2 (ms)')
        ax.set_title('response to Retract2')
    
    # excitory response to PushOnset1 with epoch (short).
    def short_epoch_push1_exc(self, ax):
        self.plot_motor_epoch(
            ax,
            self.neu_seq_push1, self.neu_time_push1, self.outcome_push1,
            self.delay_push1, 0,
            self.significance['r_push'], cate=-1)
        ax.set_xlabel('time since PushOnset1 (ms)')
        ax.set_title('excitory response to PushOnset1 (reward)')
    
    # inhibitory response to PushOnset1 with epoch (short).
    def short_epoch_push1_inh(self, ax):
        self.plot_motor_epoch(
            ax,
            self.neu_seq_push1, self.neu_time_push1, self.outcome_push1,
            self.delay_push1, 0,
            self.significance['r_push'], cate=1)
        ax.set_xlabel('time since PushOnset1 (ms)')
        ax.set_title('inhibitory response to PushOnset1 (reward)')
    
    # excitory response to Retract1 end with epoch (short).
    def short_epoch_retract1_exc(self, ax):
        self.plot_motor_epoch(
            ax,
            self.neu_seq_retract1, self.neu_time_retract1, self.outcome_retract1,
            self.delay_retract1, 0,
            self.significance['r_retract'], cate=-1)
        ax.set_xlabel('time since Retract1 end (ms)')
        ax.set_title('excitory response to Retract1 end (reward)')
    
    # inhibitory response to Retract1 end with epoch (short).
    def short_epoch_retract1_inh(self, ax):
        self.plot_motor_epoch(
            ax,
            self.neu_seq_retract1, self.neu_time_retract1, self.outcome_retract1,
            self.delay_retract1, 0,
            self.significance['r_retract'], cate=1)
        ax.set_xlabel('time since Retract1 end (ms)')
        ax.set_title('inhibitory response to Retract1 end (reward)')
    
    # excitory response to WaitForPush2 start with epoch (short).
    def short_epoch_wait2_exc(self, ax):
        self.plot_motor_epoch(
            ax,
            self.neu_seq_wait2, self.neu_time_wait2, self.outcome_wait2,
            self.delay_wait2, 0,
            self.significance['r_wait'], cate=-1)
        ax.set_xlabel('time since WaitForPush2 start (ms)')
        ax.set_title('excitory response to WaitForPush2 start (reward)')
    
    # inhibitory response to WaitForPush2 start with epoch (short).
    def short_epoch_wait2_inh(self, ax):
        self.plot_motor_epoch(
            ax,
            self.neu_seq_wait2, self.neu_time_wait2, self.outcome_wait2,
            self.delay_wait2, 0,
            self.significance['r_wait'], cate=1)
        ax.set_xlabel('time since WaitForPush2 start (ms)')
        ax.set_title('inhibitory response to WaitForPush2 start (reward)')
    
    # excitory response to PushOnset2 with epoch (short).
    def short_epoch_push2_exc(self, ax):
        self.plot_motor_epoch(
            ax,
            self.neu_seq_push2, self.neu_time_push2, self.outcome_push2,
            self.delay_push2, 0,
            self.significance['r_push'], cate=-1)
        ax.set_xlabel('time since PushOnset2 (ms)')
        ax.set_title('excitory response to PushOnset2 (reward)')
    
    # inhibitory response to PushOnset2 with epoch (short).
    def short_epoch_push2_inh(self, ax):
        self.plot_motor_epoch(
            ax,
            self.neu_seq_push2, self.neu_time_push2, self.outcome_push2,
            self.delay_push2, 0,
            self.significance['r_push'], cate=1)
        ax.set_xlabel('time since PushOnset2 (ms)')
        ax.set_title('inhibitory response to PushOnset2 (reward)')
    
    # excitory response to Retract2 with epoch (short).
    def short_epoch_retract2_exc(self, ax):
        self.plot_motor_epoch(
            ax,
            self.neu_seq_retract2, self.neu_time_retract2, self.outcome_retract2,
            self.delay_retract2, 0,
            self.significance['r_retract'], cate=-1)
        ax.set_xlabel('time since Retract2 (ms)')
        ax.set_title('excitory response to Retract2 (reward)')
    
    # inhibitory response to Retract2 with epoch (short).
    def short_epoch_retract2_inh(self, ax):
        self.plot_motor_epoch(
            ax,
            self.neu_seq_retract2, self.neu_time_retract2, self.outcome_retract2,
            self.delay_retract2, 0,
            self.significance['r_retract'], cate=1)
        ax.set_xlabel('time since Retract2 (ms)')
        ax.set_title('inhibitory response to Retract2 (reward)')       
    
    # excitory response to PushOnset1 (long).
    def long_push1_exc(self, ax):
        self.plot_moto_outcome(
            ax, self.neu_seq_push1, self.neu_time_push1, self.outcome_push1,
            self.delay_push1, 1, self.significance['r_push'], cate=-1)
        ax.set_xlabel('time since PushOnset1 (ms)')
        ax.set_title('excitory response to PushOnset1')
    
    # inhibitory response to PushOnset1 (long).
    def long_push1_inh(self, ax):
        self.plot_moto_outcome(
            ax, self.neu_seq_push1, self.neu_time_push1, self.outcome_push1,
            self.delay_push1, 1, self.significance['r_push'], cate=1)
        ax.set_xlabel('time since PushOnset1 (ms)')
        ax.set_title('inhibitory response to PushOnset1')
    
    # response to PushOnset1ing heatmap average across trials (long).
    def long_push1_heatmap_neuron(self, ax):
        idx = get_trial_type(self.cate_delay, self.delay_push1, 1)
        self.plot_heatmap_neuron(
            ax, self.neu_seq_push1[idx,:,:], self.neu_time_push1, self.significance['r_push'])
        ax.set_xlabel('time since PushOnset1 (ms)')
        ax.set_title('response to PushOnset1')
    
    # excitory response to Retract1 end (long).
    def long_retract1_exc(self, ax):
        self.plot_moto_outcome(
            ax, self.neu_seq_retract1, self.neu_time_retract1, self.outcome_retract1,
            self.delay_retract1, 1, self.significance['r_retract'], cate=-1)
        ax.set_xlabel('time since Retract1 end (ms)')
        ax.set_title('excitory response to Retract1 end')
    
    # inhibitory response to PushOnset1 (long).
    def long_retract1_inh(self, ax):
        self.plot_moto_outcome(
            ax, self.neu_seq_retract1, self.neu_time_retract1, self.outcome_retract1,
            self.delay_retract1, 1, self.significance['r_retract'], cate=1)
        ax.set_xlabel('time since Retract1 end (ms)')
        ax.set_title('inhibitory response to Retract1 end')
    
    # response to Retract1 end heatmap average across trials (long).
    def long_retract1_heatmap_neuron(self, ax):
        idx = get_trial_type(self.cate_delay, self.delay_retract1, 1)
        self.plot_heatmap_neuron(
            ax, self.neu_seq_retract1[idx,:,:], self.neu_time_retract1, self.significance['r_retract'])
        ax.set_xlabel('time since Retract1 end (ms)')
        ax.set_title('response to Retract1 end')
    
    # excitory response to WaitForPush2 start (long).
    def long_wait2_exc(self, ax):
        self.plot_moto_outcome(
            ax, self.neu_seq_wait2, self.neu_time_wait2, self.outcome_wait2,
            self.delay_wait2, 1, self.significance['r_wait'], cate=-1)
        ax.set_xlabel('time since WaitForPush2 start (ms)')
        ax.set_title('excitory response to WaitForPush2 start')
    
    # inhibitory response to WaitForPush2 start (long).
    def long_wait2_inh(self, ax):
        self.plot_moto_outcome(
            ax, self.neu_seq_wait2, self.neu_time_wait2, self.outcome_wait2,
            self.delay_wait2, 1, self.significance['r_wait'], cate=1)
        ax.set_xlabel('time since WaitForPush2 start (ms)')
        ax.set_title('inhibitory response to WaitForPush2 start')
    
    # response to WaitForPush2 start heatmap average across trials (long).
    def long_wait2_heatmap_neuron(self, ax):
        idx = get_trial_type(self.cate_delay, self.delay_wait2, 1)
        self.plot_heatmap_neuron(
            ax, self.neu_seq_wait2[idx,:,:], self.neu_time_wait2, self.significance['r_wait'])
        ax.set_xlabel('time since WaitForPush2 start (ms)')
        ax.set_title('response to WaitForPush2 start')
    
    # excitory response to PushOnset2 (long).
    def long_push2_exc(self, ax):
        self.plot_moto_outcome(
            ax, self.neu_seq_push2, self.neu_time_push2, self.outcome_push2,
            self.delay_push2, 1, self.significance['r_push'], cate=-1)
        ax.set_xlabel('time since PushOnset2 (ms)')
        ax.set_title('excitory response to PushOnset2')
    
    # inhibitory response to PushOnset2 (long).
    def long_push2_inh(self, ax):
        self.plot_moto_outcome(
            ax, self.neu_seq_push2, self.neu_time_push2, self.outcome_push2,
            self.delay_push2, 1, self.significance['r_push'], cate=1)
        ax.set_xlabel('time since PushOnset2 (ms)')
        ax.set_title('inhibitory response to PushOnset2')
    
    # response to PushOnset2  heatmap average across trials (long).
    def long_push2_heatmap_neuron(self, ax):
        idx = get_trial_type(self.cate_delay, self.delay_push2, 1)
        self.plot_heatmap_neuron(
            ax, self.neu_seq_push2[idx,:,:], self.neu_time_push2, self.significance['r_push'])
        ax.set_xlabel('time since PushOnset2 (ms)')
        ax.set_title('response to PushOnset2 ')
    
    # excitory response to Retract2 (long).
    def long_retract2_exc(self, ax):
        self.plot_moto(
            ax, self.neu_seq_retract2, self.neu_time_retract2,
            self.delay_retract2, 1, self.significance['r_retract'], cate=-1)
        ax.set_xlabel('time since Retract2 (ms)')
        ax.set_title('excitory response to Retract2')
    
    # inhibitory response to PushOnset2 (long).
    def long_retract2_inh(self, ax):
        self.plot_moto(
            ax, self.neu_seq_retract2, self.neu_time_retract2,
            self.delay_retract2, 1, self.significance['r_retract'], cate=1)
        ax.set_xlabel('time since Retract2 (ms)')
        ax.set_title('inhibitory response to Retract2')
    
    # response to Retract2 heatmap average across trials (long).
    def long_retract2_heatmap_neuron(self, ax):
        idx = get_trial_type(self.cate_delay, self.delay_retract2, 1)
        self.plot_heatmap_neuron(
            ax, self.neu_seq_retract2[idx,:,:], self.neu_time_retract2, self.significance['r_retract'])
        ax.set_xlabel('time since Retract2 (ms)')
        ax.set_title('response to Retract2')
    
    # excitory response to PushOnset1 with epoch (long).
    def long_epoch_push1_exc(self, ax):
        self.plot_motor_epoch(
            ax,
            self.neu_seq_push1, self.neu_time_push1, self.outcome_push1,
            self.delay_push1, 1,
            self.significance['r_push'], cate=-1)
        ax.set_xlabel('time since PushOnset1 (ms)')
        ax.set_title('excitory response to PushOnset1 (reward)')
    
    # inhibitory response to PushOnset1 with epoch (long).
    def long_epoch_push1_inh(self, ax):
        self.plot_motor_epoch(
            ax,
            self.neu_seq_push1, self.neu_time_push1, self.outcome_push1,
            self.delay_push1, 1,
            self.significance['r_push'], cate=1)
        ax.set_xlabel('time since PushOnset1 (ms)')
        ax.set_title('inhibitory response to PushOnset1 (reward)')
    
    # excitory response to Retract1 end with epoch (long).
    def long_epoch_retract1_exc(self, ax):
        self.plot_motor_epoch(
            ax,
            self.neu_seq_retract1, self.neu_time_retract1, self.outcome_retract1,
            self.delay_retract1, 1,
            self.significance['r_retract'], cate=-1)
        ax.set_xlabel('time since Retract1 end (ms)')
        ax.set_title('excitory response to Retract1 end (reward)')
    
    # inhibitory response to Retract1 end with epoch (long).
    def long_epoch_retract1_inh(self, ax):
        self.plot_motor_epoch(
            ax,
            self.neu_seq_retract1, self.neu_time_retract1, self.outcome_retract1,
            self.delay_retract1, 1,
            self.significance['r_retract'], cate=1)
        ax.set_xlabel('time since Retract1 end (ms)')
        ax.set_title('inhibitory response to Retract1 end (reward)')
    
    # excitory response to WaitForPush2 start with epoch (long).
    def long_epoch_wait2_exc(self, ax):
        self.plot_motor_epoch(
            ax,
            self.neu_seq_wait2, self.neu_time_wait2, self.outcome_wait2,
            self.delay_wait2, 1,
            self.significance['r_wait'], cate=-1)
        ax.set_xlabel('time since WaitForPush2 start (ms)')
        ax.set_title('excitory response to WaitForPush2 start (reward)')
    
    # inhibitory response to WaitForPush2 start with epoch (long).
    def long_epoch_wait2_inh(self, ax):
        self.plot_motor_epoch(
            ax,
            self.neu_seq_wait2, self.neu_time_wait2, self.outcome_wait2,
            self.delay_wait2, 1,
            self.significance['r_wait'], cate=1)
        ax.set_xlabel('time since WaitForPush2 start (ms)')
        ax.set_title('inhibitory response to WaitForPush2 start (reward)')
    
    # excitory response to PushOnset2 with epoch (long).
    def long_epoch_push2_exc(self, ax):
        self.plot_motor_epoch(
            ax,
            self.neu_seq_push2, self.neu_time_push2, self.outcome_push2,
            self.delay_push2, 1,
            self.significance['r_push'], cate=-1)
        ax.set_xlabel('time since PushOnset2 (ms)')
        ax.set_title('excitory response to PushOnset2 (reward)')
    
    # inhibitory response to PushOnset2 with epoch (long).
    def long_epoch_push2_inh(self, ax):
        self.plot_motor_epoch(
            ax,
            self.neu_seq_push2, self.neu_time_push2, self.outcome_push2,
            self.delay_push2, 1,
            self.significance['r_push'], cate=1)
        ax.set_xlabel('time since PushOnset2 (ms)')
        ax.set_title('inhibitory response to PushOnset2 (reward)')
    
    # excitory response to Retract2 with epoch (long).
    def long_epoch_retract2_exc(self, ax):
        self.plot_motor_epoch(
            ax,
            self.neu_seq_retract2, self.neu_time_retract2, self.outcome_retract2,
            self.delay_retract2, 1,
            self.significance['r_retract'], cate=-1)
        ax.set_xlabel('time since Retract2 (ms)')
        ax.set_title('excitory response to Retract2 (reward)')
    
    # inhibitory response to Retract2 with epoch (long).
    def long_epoch_retract2_inh(self, ax):
        self.plot_motor_epoch(
            ax,
            self.neu_seq_retract2, self.neu_time_retract2, self.outcome_retract2,
            self.delay_retract2, 1,
            self.significance['r_retract'], cate=1)
        ax.set_xlabel('time since Retract2 (ms)')
        ax.set_title('inhibitory response to Retract2 (reward)')
    
    # excitory response to push osnet.
    def onset_exc(self, ax):
        self.plot_push_onset(ax, self.significance['r_push'], cate=-1)
        ax.set_xlabel('time since push onset (ms)')
        ax.set_title('excitory response to all push onset')
    
    # inhibitory response to push osnet.
    def onset_inh(self, ax):
        self.plot_push_onset(ax, self.significance['r_push'], cate=1)
        ax.set_xlabel('time since push onset (ms)')
        ax.set_title('inhibitory response to all push onset')
    
    # response to push osnet.
    def onset_heatmap_neuron(self, axs):
        neu_seq = [self.neu_seq_push1[self.outcome_push1!=1,:,:],
                   self.neu_seq_push2[self.outcome_push2==0,:,:],
                   self.neu_seq_push2[self.outcome_push2==3,:,:]]
        for i in range(3):
            self.plot_heatmap_neuron(
                axs[i], neu_seq[i], self.neu_time_push1, self.significance['r_push'])
            axs[i].set_xlabel('time since push onset (ms)')
        axs[0].set_title('response to PO1 all')
        axs[1].set_title('response to PO2 reward')
        axs[2].set_title('response to PO2 early')
        
    # excitory response to licking.
    def lick_exc(self, ax):
        self.plot_lick(ax, self.significance['r_lick'], cate=-1)
        ax.set_xlabel('time since lick (ms)')
        ax.set_title('excitory response to lick')
    
    # inhibitory response to licking.
    def lick_inh(self, ax):
        self.plot_lick(ax, self.significance['r_lick'], cate=1)
        ax.set_xlabel('time since lick (ms)')
        ax.set_title('inhibitory response to lick')
    
    # response to licking heatmap average across trials.
    def lick_heatmap_neuron(self, ax):
        self.plot_heatmap_neuron(
            ax, self.neu_seq_lick, self.neu_time_lick, self.significance['r_lick'])
        ax.set_xlabel('time since lick (ms)')
        ax.set_title('response to all lick')
    
    # response to PushOnset1 (short).
    def short_exc_inh_push1(self, ax):
        self.plot_exc_inh(
            ax,
            self.neu_seq_push1, self.neu_time_push1,
            self.delay_push1, 0,
            self.significance['r_push'])
        ax.set_xlabel('time since PushOnset1 (ms)')
        ax.set_title('response to PushOnset1 (short)')
    
    # response to Retract1 end (short).
    def short_exc_inh_retract1(self, ax):
        self.plot_exc_inh(
            ax,
            self.neu_seq_retract1, self.neu_time_retract1,
            self.delay_retract1, 0,
            self.significance['r_retract'])
        ax.set_xlabel('time since Retract1 end (ms)')
        ax.set_title('response to Retract1 end (short)')
    
    # response to WaitForPush2 start (short).
    def short_exc_inh_wait2(self, ax):
        self.plot_exc_inh(
            ax,
            self.neu_seq_push2, self.neu_time_push2,
            self.delay_push2, 0,
            self.significance['r_wait'])
        ax.set_xlabel('time since WaitForPush2 start (ms)')
        ax.set_title('response to WaitForPush2 start onset (short)')
    
    # response to PushOnset2 (short).
    def short_exc_inh_push2(self, ax):
        self.plot_exc_inh(
            ax,
            self.neu_seq_push2, self.neu_time_push2,
            self.delay_push2, 0,
            self.significance['r_push'])
        ax.set_xlabel('time since PushOnset2 (ms)')
        ax.set_title('response to 2nd push onset (short)')
    
    # response to Retract2 (short).
    def short_exc_inh_retract2(self, ax):
        self.plot_exc_inh(
            ax,
            self.neu_seq_retract2, self.neu_time_retract2,
            self.delay_retract2, 0,
            self.significance['r_retract'])
        ax.set_xlabel('time since Retract2 (ms)')
        ax.set_title('response to Retract2 (short)')

    # response to PushOnset1 (long).
    def long_exc_inh_push1(self, ax):
        self.plot_exc_inh(
            ax,
            self.neu_seq_push1, self.neu_time_push1,
            self.delay_push1, 1,
            self.significance['r_push'])
        ax.set_xlabel('time since PushOnset1 (ms)')
        ax.set_title('response to PushOnset1 (long)')
    
    # response to Retract1 end (long).
    def long_exc_inh_retract1(self, ax):
        self.plot_exc_inh(
            ax,
            self.neu_seq_retract1, self.neu_time_retract1,
            self.delay_retract1, 1,
            self.significance['r_retract'])
        ax.set_xlabel('time since Retract1 end (ms)')
        ax.set_title('response to Retract1 end (long)')
    
    # response to WaitForPush2 start (long).
    def long_exc_inh_wait2(self, ax):
        self.plot_exc_inh(
            ax,
            self.neu_seq_push2, self.neu_time_push2,
            self.delay_push2, 1,
            self.significance['r_wait'])
        ax.set_xlabel('time since WaitForPush2 start (ms)')
        ax.set_title('response to WaitForPush2 start onset (long)')
    
    # response to PushOnset2 (long).
    def long_exc_inh_push2(self, ax):
        self.plot_exc_inh(
            ax,
            self.neu_seq_push2, self.neu_time_push2,
            self.delay_push2, 1,
            self.significance['r_push'])
        ax.set_xlabel('time since PushOnset2 (ms)')
        ax.set_title('response to PushOnset2 (long)')
    
    # response to Retract2 (long).
    def long_exc_inh_retract2(self, ax):
        self.plot_exc_inh(
            ax,
            self.neu_seq_retract2, self.neu_time_retract2,
            self.delay_retract2, 1,
            self.significance['r_retract'])
        ax.set_xlabel('time since Retract2 (ms)')
        ax.set_title('response to Retract2 (long)')


class plotter_L7G8_motor(plotter_utils):
    def __init__(self, neural_trials, labels, significance, cate_delay):
        super().__init__(neural_trials, labels, significance, cate_delay)
    
    def all_short_motor_align(self, axs):
        self.short_push1(axs[0])
        self.short_retract1(axs[1])
        self.short_wait2(axs[2])
        self.short_push2(axs[3])
        self.short_retract2(axs[4])
        
    def all_short_motor_align_heatmap_neuron(self, axs):
        self.short_push1_heatmap_neuron(axs[0])
        self.short_retract1_heatmap_neuron(axs[1])
        self.short_wait2_heatmap_neuron(axs[2])
        self.short_push2_heatmap_neuron(axs[3])
        self.short_retract2_heatmap_neuron(axs[4])
    
    def all_long_motor_align(self, axs):
        self.long_push1(axs[0])
        self.long_retract1(axs[1])
        self.long_wait2(axs[2])
        self.long_push2(axs[3])
        self.long_retract2(axs[4])

    def all_long_motor_align_heatmap_neuron(self, axs):
        self.long_push1_heatmap_neuron(axs[0])
        self.long_retract1_heatmap_neuron(axs[1])
        self.long_wait2_heatmap_neuron(axs[2])
        self.long_push2_heatmap_neuron(axs[3])
        self.long_retract2_heatmap_neuron(axs[4])
    
    def all_short_epoch_motor_align(self, axs):
        self.short_epoch_push1(axs[0])
        self.short_epoch_retract1(axs[1])
        self.short_epoch_wait2(axs[2])
        self.short_epoch_push2(axs[3])
        self.short_epoch_retract2(axs[4])

    def all_long_epoch_motor_align(self, axs):
        self.long_epoch_push1(axs[0])
        self.long_epoch_retract1(axs[1])
        self.long_epoch_wait2(axs[2])
        self.long_epoch_push2(axs[3])
        self.long_epoch_retract2(axs[4])
    
    # response to PushOnset1 (short).
    def short_push1(self, ax):
        self.plot_moto_outcome(
            ax, self.neu_seq_push1, self.neu_time_push1, self.outcome_push1,
            self.delay_push1, 0, self.significance['r_push'], cate=-1)
        ax.set_xlabel('time since PushOnset1 (ms)')
        ax.set_title('response to PushOnset1')
    
    # response to PushOnset1ing heatmap average across trials (short).
    def short_push1_heatmap_neuron(self, ax):
        idx = get_trial_type(self.cate_delay, self.delay_push1, 0)
        self.plot_heatmap_neuron(
            ax, self.neu_seq_push1[idx,:,:], self.neu_time_push1, self.significance['r_push'])
        ax.set_xlabel('time since PushOnset1 (ms)')
        ax.set_title('response to PushOnset1')
    
    # response to PushOnset1 (short).
    def short_retract1(self, ax):
        self.plot_moto_outcome(
            ax, self.neu_seq_retract1, self.neu_time_retract1, self.outcome_retract1,
            self.delay_retract1, 0, self.significance['r_retract'], cate=-1)
        ax.set_xlabel('time since Retract1 end (ms)')
        ax.set_title('response to Retract1 end')
    
    # response to Retract1 end heatmap average across trials (short).
    def short_retract1_heatmap_neuron(self, ax):
        idx = get_trial_type(self.cate_delay, self.delay_retract1, 0)
        self.plot_heatmap_neuron(
            ax, self.neu_seq_retract1[idx,:,:], self.neu_time_retract1, self.significance['r_retract'])
        ax.set_xlabel('time since Retract1 end (ms)')
        ax.set_title('response to Retract1 end')
    
    # response to WaitForPush2 start (short).
    def short_wait2(self, ax):
        self.plot_moto_outcome(
            ax, self.neu_seq_wait2, self.neu_time_wait2, self.outcome_wait2,
            self.delay_wait2, 0, self.significance['r_wait'], cate=-1)
        ax.set_xlabel('time since WaitForPush2 start (ms)')
        ax.set_title('response to WaitForPush2 start')
    
    # response to WaitForPush2 start heatmap average across trials (short).
    def short_wait2_heatmap_neuron(self, ax):
        idx = get_trial_type(self.cate_delay, self.delay_wait2, 0)
        self.plot_heatmap_neuron(
            ax, self.neu_seq_wait2[idx,:,:], self.neu_time_wait2, self.significance['r_wait'])
        ax.set_xlabel('time since WaitForPush2 start (ms)')
        ax.set_title('response to WaitForPush2 start')
    
    # response to PushOnset2 (short).
    def short_push2(self, ax):
        self.plot_moto_outcome(
            ax, self.neu_seq_push2, self.neu_time_push2, self.outcome_push2,
            self.delay_push2, 0, self.significance['r_push'], cate=-1)
        ax.set_xlabel('time since PushOnset2 (ms)')
        ax.set_title('response to PushOnset2')
    
    # response to PushOnset2  heatmap average across trials (short).
    def short_push2_heatmap_neuron(self, ax):
        idx = get_trial_type(self.cate_delay, self.delay_push2, 0)
        self.plot_heatmap_neuron(
            ax, self.neu_seq_push2[idx,:,:], self.neu_time_push2, self.significance['r_push'])
        ax.set_xlabel('time since PushOnset2 (ms)')
        ax.set_title('response to PushOnset2 ')
    
    # response to PushOnset2 (short).
    def short_retract2(self, ax):
        self.plot_moto(
            ax, self.neu_seq_retract2, self.neu_time_retract2,
            self.delay_retract2, 0, self.significance['r_retract'], cate=-1)
        ax.set_xlabel('time since Retract2 (ms)')
        ax.set_title('response to Retract2')
    
    # response to Retract2 heatmap average across trials (short).
    def short_retract2_heatmap_neuron(self, ax):
        idx = get_trial_type(self.cate_delay, self.delay_retract2, 0)
        self.plot_heatmap_neuron(
            ax, self.neu_seq_retract2[idx,:,:], self.neu_time_retract2, self.significance['r_retract'])
        ax.set_xlabel('time since Retract2 (ms)')
        ax.set_title('response to Retract2')
    
    # response to PushOnset1 with epoch (short).
    def short_epoch_push1(self, ax):
        self.plot_motor_epoch(
            ax,
            self.neu_seq_push1, self.neu_time_push1, self.outcome_push1,
            self.delay_push1, 0,
            self.significance['r_push'], cate=-1)
        ax.set_xlabel('time since PushOnset1 (ms)')
        ax.set_title('response to PushOnset1 (reward)')
    
    # response to Retract1 end with epoch (short).
    def short_epoch_retract1(self, ax):
        self.plot_motor_epoch(
            ax,
            self.neu_seq_retract1, self.neu_time_retract1, self.outcome_retract1,
            self.delay_retract1, 0,
            self.significance['r_retract'], cate=-1)
        ax.set_xlabel('time since Retract1 end (ms)')
        ax.set_title('response to Retract1 end (reward)')
    
    # response to WaitForPush2 start with epoch (short).
    def short_epoch_wait2(self, ax):
        self.plot_motor_epoch(
            ax,
            self.neu_seq_wait2, self.neu_time_wait2, self.outcome_wait2,
            self.delay_wait2, 0,
            self.significance['r_wait'], cate=-1)
        ax.set_xlabel('time since WaitForPush2 start (ms)')
        ax.set_title('response to WaitForPush2 start (reward)')
    
    # response to PushOnset2 with epoch (short).
    def short_epoch_push2(self, ax):
        self.plot_motor_epoch(
            ax,
            self.neu_seq_push2, self.neu_time_push2, self.outcome_push2,
            self.delay_push2, 0,
            self.significance['r_push'], cate=-1)
        ax.set_xlabel('time since PushOnset2 (ms)')
        ax.set_title('response to PushOnset2 (reward)')
    
    # response to Retract2 with epoch (short).
    def short_epoch_retract2(self, ax):
        self.plot_motor_epoch(
            ax,
            self.neu_seq_retract2, self.neu_time_retract2, self.outcome_retract2,
            self.delay_retract2, 0,
            self.significance['r_retract'], cate=-1)
        ax.set_xlabel('time since Retract2 (ms)')
        ax.set_title('response to Retract2 (reward)')       
    
    # response to PushOnset1 (long).
    def long_push1(self, ax):
        self.plot_moto_outcome(
            ax, self.neu_seq_push1, self.neu_time_push1, self.outcome_push1,
            self.delay_push1, 1, self.significance['r_push'], cate=-1)
        ax.set_xlabel('time since PushOnset1 (ms)')
        ax.set_title('response to PushOnset1')
    
    # response to PushOnset1ing heatmap average across trials (long).
    def long_push1_heatmap_neuron(self, ax):
        idx = get_trial_type(self.cate_delay, self.delay_push1, 1)
        self.plot_heatmap_neuron(
            ax, self.neu_seq_push1[idx,:,:], self.neu_time_push1, self.significance['r_push'])
        ax.set_xlabel('time since PushOnset1 (ms)')
        ax.set_title('response to PushOnset1')
    
    # response to PushOnset1 (long).
    def long_retract1(self, ax):
        self.plot_moto_outcome(
            ax, self.neu_seq_retract1, self.neu_time_retract1, self.outcome_retract1,
            self.delay_retract1, 1, self.significance['r_retract'], cate=-1)
        ax.set_xlabel('time since Retract1 end (ms)')
        ax.set_title('response to Retract1 end')
    
    # response to Retract1 end heatmap average across trials (long).
    def long_retract1_heatmap_neuron(self, ax):
        idx = get_trial_type(self.cate_delay, self.delay_retract1, 1)
        self.plot_heatmap_neuron(
            ax, self.neu_seq_retract1[idx,:,:], self.neu_time_retract1, self.significance['r_retract'])
        ax.set_xlabel('time since Retract1 end (ms)')
        ax.set_title('response to Retract1 end')
    
    # response to WaitForPush2 start (long).
    def long_wait2(self, ax):
        self.plot_moto_outcome(
            ax, self.neu_seq_wait2, self.neu_time_wait2, self.outcome_wait2,
            self.delay_wait2, 1, self.significance['r_wait'], cate=-1)
        ax.set_xlabel('time since WaitForPush2 start (ms)')
        ax.set_title('response to WaitForPush2 start')
    
    # response to WaitForPush2 start heatmap average across trials (long).
    def long_wait2_heatmap_neuron(self, ax):
        idx = get_trial_type(self.cate_delay, self.delay_wait2, 1)
        self.plot_heatmap_neuron(
            ax, self.neu_seq_wait2[idx,:,:], self.neu_time_wait2, self.significance['r_wait'])
        ax.set_xlabel('time since WaitForPush2 start (ms)')
        ax.set_title('response to WaitForPush2 start')
    
    # response to PushOnset2 (long).
    def long_push2(self, ax):
        self.plot_moto_outcome(
            ax, self.neu_seq_push2, self.neu_time_push2, self.outcome_push2,
            self.delay_push2, 1, self.significance['r_push'], cate=-1)
        ax.set_xlabel('time since PushOnset2 (ms)')
        ax.set_title('response to PushOnset2')
    
    # response to PushOnset2  heatmap average across trials (long).
    def long_push2_heatmap_neuron(self, ax):
        idx = get_trial_type(self.cate_delay, self.delay_push2, 1)
        self.plot_heatmap_neuron(
            ax, self.neu_seq_push2[idx,:,:], self.neu_time_push2, self.significance['r_push'])
        ax.set_xlabel('time since PushOnset2 (ms)')
        ax.set_title('response to PushOnset2 ')
    
    # response to PushOnset2 (long).
    def long_retract2(self, ax):
        self.plot_moto(
            ax, self.neu_seq_retract2, self.neu_time_retract2,
            self.delay_retract2, 1, self.significance['r_retract'], cate=-1)
        ax.set_xlabel('time since Retract2 (ms)')
        ax.set_title('response to Retract2')
    
    # response to Retract2 heatmap average across trials (long).
    def long_retract2_heatmap_neuron(self, ax):
        idx = get_trial_type(self.cate_delay, self.delay_retract2, 1)
        self.plot_heatmap_neuron(
            ax, self.neu_seq_retract2[idx,:,:], self.neu_time_retract2, self.significance['r_retract'])
        ax.set_xlabel('time since Retract2 (ms)')
        ax.set_title('response to Retract2')
    
    # response to PushOnset1 with epoch (long).
    def long_epoch_push1(self, ax):
        self.plot_motor_epoch(
            ax,
            self.neu_seq_push1, self.neu_time_push1, self.outcome_push1,
            self.delay_push1, 1,
            self.significance['r_push'], cate=-1)
        ax.set_xlabel('time since PushOnset1 (ms)')
        ax.set_title('response to PushOnset1 (reward)')
    
    # response to Retract1 end with epoch (long).
    def long_epoch_retract1(self, ax):
        self.plot_motor_epoch(
            ax,
            self.neu_seq_retract1, self.neu_time_retract1, self.outcome_retract1,
            self.delay_retract1, 1,
            self.significance['r_retract'], cate=-1)
        ax.set_xlabel('time since Retract1 end (ms)')
        ax.set_title('response to Retract1 end (reward)')
    
    # response to WaitForPush2 start with epoch (long).
    def long_epoch_wait2(self, ax):
        self.plot_motor_epoch(
            ax,
            self.neu_seq_wait2, self.neu_time_wait2, self.outcome_wait2,
            self.delay_wait2, 1,
            self.significance['r_wait'], cate=-1)
        ax.set_xlabel('time since WaitForPush2 start (ms)')
        ax.set_title('response to WaitForPush2 start (reward)')
    
    # response to PushOnset2 with epoch (long).
    def long_epoch_push2(self, ax):
        self.plot_motor_epoch(
            ax,
            self.neu_seq_push2, self.neu_time_push2, self.outcome_push2,
            self.delay_push2, 1,
            self.significance['r_push'], cate=-1)
        ax.set_xlabel('time since PushOnset2 (ms)')
        ax.set_title('response to PushOnset2 (reward)')
    
    # response to Retract2 with epoch (long).
    def long_epoch_retract2(self, ax):
        self.plot_motor_epoch(
            ax,
            self.neu_seq_retract2, self.neu_time_retract2, self.outcome_retract2,
            self.delay_retract2, 1,
            self.significance['r_retract'], cate=-1)
        ax.set_xlabel('time since Retract2 (ms)')
        ax.set_title('response to Retract2 (reward)')       
        
    # response to licking.
    def lick(self, ax):
        self.plot_lick(ax, self.significance['r_lick'], cate=-1)
        ax.set_xlabel('time since lick (ms)')
        ax.set_title('response to lick')
    
    # response to licking heatmap average across trials.
    def lick_heatmap_neuron(self, ax):
        self.plot_heatmap_neuron(
            ax, self.neu_seq_lick, self.neu_time_lick, self.significance['r_lick'])
        ax.set_xlabel('time since lick (ms)')
        ax.set_title('response to all lick')
    
    # response to push osnet.
    def onset(self, ax):
        self.plot_push_onset(ax, self.significance['r_push'], cate=-1)
        ax.set_xlabel('time since push onset (ms)')
        ax.set_title('response to all push onset')
    
    # response to push osnet.
    def onset_heatmap_neuron(self, axs):
        neu_seq = [self.neu_seq_push1[self.outcome_push1!=1,:,:],
                   self.neu_seq_push2[self.outcome_push2==0,:,:],
                   self.neu_seq_push2[self.outcome_push2==3,:,:]]
        for i in range(3):
            self.plot_heatmap_neuron(
                axs[i], neu_seq[i], self.neu_time_push1, self.significance['r_push'])
            axs[i].set_xlabel('time since push onset (ms)')
        axs[0].set_title('response to PO1 all')
        axs[1].set_title('response to PO2 reward')
        axs[2].set_title('response to PO2 early')


class plotter_VIPG8_motor(plotter_utils):
    def __init__(self, neural_trials, labels, significance, cate_delay):
        super().__init__(neural_trials, labels, significance, cate_delay)
    
    def all_short_motor_align(self, axs):
        self.short_push1(axs[0])
        self.short_retract1(axs[1])
        self.short_wait2(axs[2])
        self.short_push2(axs[3])
        self.short_retract2(axs[4])
        
    def all_short_motor_align_heatmap_neuron(self, axs):
        self.short_push1_heatmap_neuron(axs[0])
        self.short_retract1_heatmap_neuron(axs[1])
        self.short_wait2_heatmap_neuron(axs[2])
        self.short_push2_heatmap_neuron(axs[3])
        self.short_retract2_heatmap_neuron(axs[4])
    
    def all_long_motor_align(self, axs):
        self.long_push1(axs[0])
        self.long_retract1(axs[1])
        self.long_wait2(axs[2])
        self.long_push2(axs[3])
        self.long_retract2(axs[4])

    def all_long_motor_align_heatmap_neuron(self, axs):
        self.long_push1_heatmap_neuron(axs[0])
        self.long_retract1_heatmap_neuron(axs[1])
        self.long_wait2_heatmap_neuron(axs[2])
        self.long_push2_heatmap_neuron(axs[3])
        self.long_retract2_heatmap_neuron(axs[4])
    
    def all_short_epoch_motor_align(self, axs):
        self.short_epoch_push1(axs[0])
        self.short_epoch_retract1(axs[1])
        self.short_epoch_wait2(axs[2])
        self.short_epoch_push2(axs[3])
        self.short_epoch_retract2(axs[4])

    def all_long_epoch_motor_align(self, axs):
        self.long_epoch_push1(axs[0])
        self.long_epoch_retract1(axs[1])
        self.long_epoch_wait2(axs[2])
        self.long_epoch_push2(axs[3])
        self.long_epoch_retract2(axs[4])
    
    # response to PushOnset1 (short).
    def short_push1(self, ax):
        self.plot_moto_outcome(
            ax, self.neu_seq_push1, self.neu_time_push1, self.outcome_push1,
            self.delay_push1, 0, self.significance['r_push'], cate=1)
        ax.set_xlabel('time since PushOnset1 (ms)')
        ax.set_title('response to PushOnset1')
    
    # response to PushOnset1ing heatmap average across trials (short).
    def short_push1_heatmap_neuron(self, ax):
        idx = get_trial_type(self.cate_delay, self.delay_push1, 0)
        self.plot_heatmap_neuron(
            ax, self.neu_seq_push1[idx,:,:], self.neu_time_push1, self.significance['r_push'])
        ax.set_xlabel('time since PushOnset1 (ms)')
        ax.set_title('response to PushOnset1')
    
    # response to PushOnset1 (short).
    def short_retract1(self, ax):
        self.plot_moto_outcome(
            ax, self.neu_seq_retract1, self.neu_time_retract1, self.outcome_retract1,
            self.delay_retract1, 0, self.significance['r_retract'], cate=1)
        ax.set_xlabel('time since Retract1 end (ms)')
        ax.set_title('response to Retract1 end')
    
    # response to Retract1 end heatmap average across trials (short).
    def short_retract1_heatmap_neuron(self, ax):
        idx = get_trial_type(self.cate_delay, self.delay_retract1, 0)
        self.plot_heatmap_neuron(
            ax, self.neu_seq_retract1[idx,:,:], self.neu_time_retract1, self.significance['r_retract'])
        ax.set_xlabel('time since Retract1 end (ms)')
        ax.set_title('response to Retract1 end')
    
    # response to WaitForPush2 start (short).
    def short_wait2(self, ax):
        self.plot_moto_outcome(
            ax, self.neu_seq_wait2, self.neu_time_wait2, self.outcome_wait2,
            self.delay_wait2, 0, self.significance['r_wait'], cate=1)
        ax.set_xlabel('time since WaitForPush2 start (ms)')
        ax.set_title('response to WaitForPush2 start')
    
    # response to WaitForPush2 start heatmap average across trials (short).
    def short_wait2_heatmap_neuron(self, ax):
        idx = get_trial_type(self.cate_delay, self.delay_wait2, 0)
        self.plot_heatmap_neuron(
            ax, self.neu_seq_wait2[idx,:,:], self.neu_time_wait2, self.significance['r_wait'])
        ax.set_xlabel('time since WaitForPush2 start (ms)')
        ax.set_title('response to WaitForPush2 start')
    
    # response to PushOnset2 (short).
    def short_push2(self, ax):
        self.plot_moto_outcome(
            ax, self.neu_seq_push2, self.neu_time_push2, self.outcome_push2,
            self.delay_push2, 0, self.significance['r_push'], cate=1)
        ax.set_xlabel('time since PushOnset2 (ms)')
        ax.set_title('response to PushOnset2')
    
    # response to PushOnset2  heatmap average across trials (short).
    def short_push2_heatmap_neuron(self, ax):
        idx = get_trial_type(self.cate_delay, self.delay_push2, 0)
        self.plot_heatmap_neuron(
            ax, self.neu_seq_push2[idx,:,:], self.neu_time_push2, self.significance['r_push'])
        ax.set_xlabel('time since PushOnset2 (ms)')
        ax.set_title('response to PushOnset2 ')
    
    # response to PushOnset2 (short).
    def short_retract2(self, ax):
        self.plot_moto(
            ax, self.neu_seq_retract2, self.neu_time_retract2,
            self.delay_retract2, 0, self.significance['r_retract'], cate=1)
        ax.set_xlabel('time since Retract2 (ms)')
        ax.set_title('response to Retract2')
    
    # response to Retract2 heatmap average across trials (short).
    def short_retract2_heatmap_neuron(self, ax):
        idx = get_trial_type(self.cate_delay, self.delay_retract2, 0)
        self.plot_heatmap_neuron(
            ax, self.neu_seq_retract2[idx,:,:], self.neu_time_retract2, self.significance['r_retract'])
        ax.set_xlabel('time since Retract2 (ms)')
        ax.set_title('response to Retract2')
    
    # response to PushOnset1 with epoch (short).
    def short_epoch_push1(self, ax):
        self.plot_motor_epoch(
            ax,
            self.neu_seq_push1, self.neu_time_push1, self.outcome_push1,
            self.delay_push1, 0,
            self.significance['r_push'], cate=1)
        ax.set_xlabel('time since PushOnset1 (ms)')
        ax.set_title('response to PushOnset1 (reward)')
    
    # response to Retract1 end with epoch (short).
    def short_epoch_retract1(self, ax):
        self.plot_motor_epoch(
            ax,
            self.neu_seq_retract1, self.neu_time_retract1, self.outcome_retract1,
            self.delay_retract1, 0,
            self.significance['r_retract'], cate=1)
        ax.set_xlabel('time since Retract1 end (ms)')
        ax.set_title('response to Retract1 end (reward)')
    
    # response to WaitForPush2 start with epoch (short).
    def short_epoch_wait2(self, ax):
        self.plot_motor_epoch(
            ax,
            self.neu_seq_wait2, self.neu_time_wait2, self.outcome_wait2,
            self.delay_wait2, 0,
            self.significance['r_wait'], cate=1)
        ax.set_xlabel('time since WaitForPush2 start (ms)')
        ax.set_title('response to WaitForPush2 start (reward)')
    
    # response to PushOnset2 with epoch (short).
    def short_epoch_push2(self, ax):
        self.plot_motor_epoch(
            ax,
            self.neu_seq_push2, self.neu_time_push2, self.outcome_push2,
            self.delay_push2, 0,
            self.significance['r_push'], cate=1)
        ax.set_xlabel('time since PushOnset2 (ms)')
        ax.set_title('response to PushOnset2 (reward)')
    
    # response to Retract2 with epoch (short).
    def short_epoch_retract2(self, ax):
        self.plot_motor_epoch(
            ax,
            self.neu_seq_retract2, self.neu_time_retract2, self.outcome_retract2,
            self.delay_retract2, 0,
            self.significance['r_retract'], cate=1)
        ax.set_xlabel('time since Retract2 (ms)')
        ax.set_title('response to Retract2 (reward)')       
    
    # response to PushOnset1 (long).
    def long_push1(self, ax):
        self.plot_moto_outcome(
            ax, self.neu_seq_push1, self.neu_time_push1, self.outcome_push1,
            self.delay_push1, 1, self.significance['r_push'], cate=1)
        ax.set_xlabel('time since PushOnset1 (ms)')
        ax.set_title('response to PushOnset1')
    
    # response to PushOnset1ing heatmap average across trials (long).
    def long_push1_heatmap_neuron(self, ax):
        idx = get_trial_type(self.cate_delay, self.delay_push1, 1)
        self.plot_heatmap_neuron(
            ax, self.neu_seq_push1[idx,:,:], self.neu_time_push1, self.significance['r_push'])
        ax.set_xlabel('time since PushOnset1 (ms)')
        ax.set_title('response to PushOnset1')
    
    # response to PushOnset1 (long).
    def long_retract1(self, ax):
        self.plot_moto_outcome(
            ax, self.neu_seq_retract1, self.neu_time_retract1, self.outcome_retract1,
            self.delay_retract1, 1, self.significance['r_retract'], cate=1)
        ax.set_xlabel('time since Retract1 end (ms)')
        ax.set_title('response to Retract1 end')
    
    # response to Retract1 end heatmap average across trials (long).
    def long_retract1_heatmap_neuron(self, ax):
        idx = get_trial_type(self.cate_delay, self.delay_retract1, 1)
        self.plot_heatmap_neuron(
            ax, self.neu_seq_retract1[idx,:,:], self.neu_time_retract1, self.significance['r_retract'])
        ax.set_xlabel('time since Retract1 end (ms)')
        ax.set_title('response to Retract1 end')
    
    # response to WaitForPush2 start (long).
    def long_wait2(self, ax):
        self.plot_moto_outcome(
            ax, self.neu_seq_wait2, self.neu_time_wait2, self.outcome_wait2,
            self.delay_wait2, 1, self.significance['r_wait'], cate=1)
        ax.set_xlabel('time since WaitForPush2 start (ms)')
        ax.set_title('response to WaitForPush2 start')
    
    # response to WaitForPush2 start heatmap average across trials (long).
    def long_wait2_heatmap_neuron(self, ax):
        idx = get_trial_type(self.cate_delay, self.delay_wait2, 1)
        self.plot_heatmap_neuron(
            ax, self.neu_seq_wait2[idx,:,:], self.neu_time_wait2, self.significance['r_wait'])
        ax.set_xlabel('time since WaitForPush2 start (ms)')
        ax.set_title('response to WaitForPush2 start')
    
    # response to PushOnset2 (long).
    def long_push2(self, ax):
        self.plot_moto_outcome(
            ax, self.neu_seq_push2, self.neu_time_push2, self.outcome_push2,
            self.delay_push2, 1, self.significance['r_push'], cate=1)
        ax.set_xlabel('time since PushOnset2 (ms)')
        ax.set_title('response to PushOnset2')
    
    # response to PushOnset2  heatmap average across trials (long).
    def long_push2_heatmap_neuron(self, ax):
        idx = get_trial_type(self.cate_delay, self.delay_push2, 1)
        self.plot_heatmap_neuron(
            ax, self.neu_seq_push2[idx,:,:], self.neu_time_push2, self.significance['r_push'])
        ax.set_xlabel('time since PushOnset2 (ms)')
        ax.set_title('response to PushOnset2 ')
    
    # response to PushOnset2 (long).
    def long_retract2(self, ax):
        self.plot_moto(
            ax, self.neu_seq_retract2, self.neu_time_retract2,
            self.delay_retract2, 1, self.significance['r_retract'], cate=1)
        ax.set_xlabel('time since Retract2 (ms)')
        ax.set_title('response to Retract2')
    
    # response to Retract2 heatmap average across trials (long).
    def long_retract2_heatmap_neuron(self, ax):
        idx = get_trial_type(self.cate_delay, self.delay_retract2, 1)
        self.plot_heatmap_neuron(
            ax, self.neu_seq_retract2[idx,:,:], self.neu_time_retract2, self.significance['r_retract'])
        ax.set_xlabel('time since Retract2 (ms)')
        ax.set_title('response to Retract2')
    
    # response to PushOnset1 with epoch (long).
    def long_epoch_push1(self, ax):
        self.plot_motor_epoch(
            ax,
            self.neu_seq_push1, self.neu_time_push1, self.outcome_push1,
            self.delay_push1, 1,
            self.significance['r_push'], cate=1)
        ax.set_xlabel('time since PushOnset1 (ms)')
        ax.set_title('response to PushOnset1 (reward)')
    
    # response to Retract1 end with epoch (long).
    def long_epoch_retract1(self, ax):
        self.plot_motor_epoch(
            ax,
            self.neu_seq_retract1, self.neu_time_retract1, self.outcome_retract1,
            self.delay_retract1, 1,
            self.significance['r_retract'], cate=1)
        ax.set_xlabel('time since Retract1 end (ms)')
        ax.set_title('response to Retract1 end (reward)')
    
    # response to WaitForPush2 start with epoch (long).
    def long_epoch_wait2(self, ax):
        self.plot_motor_epoch(
            ax,
            self.neu_seq_wait2, self.neu_time_wait2, self.outcome_wait2,
            self.delay_wait2, 1,
            self.significance['r_wait'], cate=1)
        ax.set_xlabel('time since WaitForPush2 start (ms)')
        ax.set_title('response to WaitForPush2 start (reward)')
    
    # response to PushOnset2 with epoch (long).
    def long_epoch_push2(self, ax):
        self.plot_motor_epoch(
            ax,
            self.neu_seq_push2, self.neu_time_push2, self.outcome_push2,
            self.delay_push2, 1,
            self.significance['r_push'], cate=1)
        ax.set_xlabel('time since PushOnset2 (ms)')
        ax.set_title('response to PushOnset2 (reward)')
    
    # response to Retract2 with epoch (long).
    def long_epoch_retract2(self, ax):
        self.plot_motor_epoch(
            ax,
            self.neu_seq_retract2, self.neu_time_retract2, self.outcome_retract2,
            self.delay_retract2, 1,
            self.significance['r_retract'], cate=1)
        ax.set_xlabel('time since Retract2 (ms)')
        ax.set_title('response to Retract2 (reward)')       
        
    # response to licking.
    def lick(self, ax):
        self.plot_lick(ax, self.significance['r_lick'], cate=1)
        ax.set_xlabel('time since lick (ms)')
        ax.set_title('response to lick')
    
    # response to licking heatmap average across trials.
    def lick_heatmap_neuron(self, ax):
        self.plot_heatmap_neuron(
            ax, self.neu_seq_lick, self.neu_time_lick, self.significance['r_lick'])
        ax.set_xlabel('time since lick (ms)')
        ax.set_title('response to all lick')
    
    # response to push osnet.
    def onset(self, ax):
        self.plot_push_onset(ax, self.significance['r_push'], cate=1)
        ax.set_xlabel('time since push onset (ms)')
        ax.set_title('response to all push onset')
    
    # response to push osnet.
    def onset_heatmap_neuron(self, axs):
        neu_seq = [self.neu_seq_push1[self.outcome_push1!=1,:,:],
                   self.neu_seq_push2[self.outcome_push2==0,:,:],
                   self.neu_seq_push2[self.outcome_push2==3,:,:]]
        for i in range(3):
            self.plot_heatmap_neuron(
                axs[i], neu_seq[i], self.neu_time_push1, self.significance['r_push'])
            axs[i].set_xlabel('time since push onset (ms)')
        axs[0].set_title('response to PO1 all')
        axs[1].set_title('response to PO2 reward')
        axs[2].set_title('response to PO2 early')