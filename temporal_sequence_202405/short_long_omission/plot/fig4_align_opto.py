#!/usr/bin/env python3

import numpy as np

from modules.Alignment import get_stim_response
from plot.utils import exclude_opto_neighbor
from plot.utils import get_expect_stim_time
from plot.utils import get_omi_stim_idx
from plot.utils import get_mean_sem
from plot.utils import get_roi_label_color
from plot.utils import get_epoch_idx
from plot.utils import get_opto_stim
from plot.utils import pick_trial
from plot.utils import adjust_layout_neu
from plot.utils import utils
    

class plotter_utils(utils):

    def __init__(
            self,
            neural_trials, labels
            ):
        self.l_frames = 150
        self.r_frames = 150
        self.cut_frames = 50
        self.stim_labels = neural_trials['stim_labels'][1:-1,:]
        self.stim_labels = exclude_opto_neighbor(self.stim_labels)
        self.labels = labels
        [self.epoch_short, self.epoch_long] = get_epoch_idx(self.stim_labels)
        [self.neu_seq, self.neu_time,
         self.stim_seq, self.stim_value, self.stim_time, self.led_value, _] = get_stim_response(
            neural_trials, self.l_frames, self.r_frames)
        self.expect_short, self.expect_long = get_expect_stim_time(
            self.stim_labels)
        [self.idx_pre_short, self.idx_pre_long,
         self.idx_post_short, self.idx_post_long] = get_omi_stim_idx(
            self.stim_labels)
        self.stim_labels_org = neural_trials['stim_labels'][1:-1,:]
        self.idx_opto_pre, self.idx_opto_post = get_opto_stim(self.stim_labels_org)
        
    def plot_normal_opto(self, ax, s, normal, cate=None, roi_id=None):
        if cate != None:
            neu_cate = self.neu_seq[:,self.labels==cate,:]
            _, color1, color2, _ = get_roi_label_color([cate], 0)
        if roi_id != None:
            neu_cate = np.expand_dims(self.neu_seq[:,roi_id,:], axis=1)
            _, color1, color2, _ = get_roi_label_color(self.labels, roi_id) 
        idx_none = pick_trial(self.stim_labels, None, [normal], None, None, [0])
        idx_opto = pick_trial(self.stim_labels, None, [normal], None, None, [3])
        mean_none, sem_none = get_mean_sem(neu_cate[idx_none,:,:])
        mean_opto, sem_opto = get_mean_sem(neu_cate[idx_opto,:,:])
        stim_seq = np.mean(self.stim_seq[idx_none,1,:],axis=0)
        upper = np.max([mean_none, mean_opto]) + np.max([sem_none, sem_opto])
        lower = np.min([mean_none, mean_opto]) - np.max([sem_none, sem_opto])
        ax.fill_between(
            stim_seq,
            lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
            color='gold', alpha=0.15, step='mid', label='stim')
        self.plot_vol(
            ax, self.stim_time, self.stim_value[idx_none,:],
            'gold', upper, lower)
        self.plot_mean_sem(ax, self.neu_time, mean_none, sem_none, color1, 'control')
        self.plot_mean_sem(ax, self.neu_time, mean_opto, sem_opto, color2, 'opto')
        adjust_layout_neu(ax)
        ax.set_xlim([self.neu_time[np.argmin(np.abs(self.neu_time))-2*self.cut_frames],
                     self.neu_time[np.argmin(np.abs(self.neu_time))+1*self.cut_frames]])
        ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
        ax.set_xlabel('time since stim (ms)')
    
    def plot_omi_normal_post_opto(self, ax, s, cate=None, roi_id=None):
        if cate != None:
            neu_cate = self.neu_seq[:,self.labels==cate,:]
            _, color1, color2, _ = get_roi_label_color([cate], 0)
        if roi_id != None:
            neu_cate = np.expand_dims(self.neu_seq[:,roi_id,:], axis=1)
            _, color1, color2, _ = get_roi_label_color(self.labels, roi_id)
        idx_short = self.idx_post_short * pick_trial(self.stim_labels, None, [0], None, None, [2])
        idx_long  = self.idx_post_long  * pick_trial(self.stim_labels, None, [1], None, None, [2])
        stim_seq_short = np.mean(self.stim_seq[idx_short,:,:], axis=0)
        stim_seq_long  = np.mean(self.stim_seq[idx_long,:,:], axis=0)
        mean_short, sem_short = get_mean_sem(neu_cate[idx_short,:,:])
        mean_long,  sem_long  = get_mean_sem(neu_cate[idx_long,:,:])
        upper = np.max([mean_short, mean_long]) + np.max([sem_short, sem_long])
        lower = np.min([mean_short, mean_long]) - np.max([sem_short, sem_long])
        ax.fill_between(
            stim_seq_short[1,:],
            lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
            color='gold', alpha=0.15, step='mid', label='post')
        ax.axvline(
            self.expect_short+stim_seq_short[0,0],
            color=color1, label='omi (short)',
            lw=1, linestyle='--')
        ax.axvline(
            self.expect_long+stim_seq_long[0,0],
            color=color2, label='omi (long)',
            lw=1, linestyle='--')
        ax.fill_between(
            stim_seq_short[0,:],
            lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
            color=color1, alpha=0.15, step='mid', label='pre (short)')
        ax.fill_between(
            stim_seq_long[0,:],
            lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
            color=color2, alpha=0.15, step='mid', label='pre (long)')
        self.plot_mean_sem(ax, self.neu_time, mean_short, sem_short, color1, 'short')
        self.plot_mean_sem(ax, self.neu_time, mean_long, sem_long, color2, 'long')
        adjust_layout_neu(ax)
        ax.set_xlim([self.neu_time[0],
                     self.neu_time[np.argmin(np.abs(self.neu_time))+self.cut_frames]])
        ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
        ax.set_xlabel('time since post omi stim (ms)')
        ax.legend(loc='upper left')
        
        

        
# fig, ax = plt.subplots(1, 1, figsize=(6, 6))

class plotter_VIPTD_G8_align_opto(plotter_utils):
    def __init__(self, neural_trials, labels):
        super().__init__(neural_trials, labels)
    
    # excitory response to normal with opto for short.
    def normal_opto_exc_short(self, ax):
        self.plot_normal_opto(ax, None, 0, cate=-1)
        ax.set_title('excitory response to normal with opto (short)')
    
    # inhibitory response to normal with opto for short.
    def normal_opto_inh_short(self, ax):
        self.plot_normal_opto(ax, None, 0, cate=1)
        ax.set_title('inhibitory response to normal with opto (short)')
    
    # excitory response to normal with opto for long.
    def normal_opto_exc_long(self, ax):
        self.plot_normal_opto(ax, None, 1, cate=-1)
        ax.set_title('excitory response to normal with opto (long)')
    
    # inhibitory response to normal with opto for short.
    def normal_opto_inh_long(self, ax):
        self.plot_normal_opto(ax, None, 1, cate=1)
        ax.set_title('inhibitory response to normal with opto (long)')
    
    # excitory response to post omission stimulus with opto.
    def omi_normal_post_opto_exc(self, ax):
        self.plot_omi_normal_post_opto(ax, None, cate=-1)
        ax.set_title('excitory response to post omi stim')
    
    # inhibitory response to post omission stimulus with opto.
    def omi_normal_post_opto_inh(self, ax):
        self.plot_omi_normal_post_opto(ax, None, cate=1)
        ax.set_title('inhibitory response to post omi stim')

    
    def plot_opto_normal(self, ax, s, normal, cate=None, roi_id=None):
        for roi_id in tqdm(np.argsort(labels, kind='stable')):
            cate = None
            #roi_id=None
            normal = 1
            if cate != None:
                neu_cate = neu_seq[:,labels==cate,:]
                _, color1, color2, _ = get_roi_label_color([cate], 0)
            if roi_id != None:
                neu_cate = np.expand_dims(neu_seq[:,roi_id,:], axis=1)
                _, color1, color2, _ = get_roi_label_color(labels, roi_id)
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            idx_control = pick_trial(stim_labels, [2,3,4,5], [normal], None, None, [0])
            idx_opto    = pick_trial(stim_labels, [2,3,4,5], [normal], None, None, [3])
            stim = np.mean(stim_seq[idx_control,1,:],axis=0)
            m_control, s_control = get_mean_sem(neu_cate[idx_control,:,:])
            m_opto,    s_opto    = get_mean_sem(neu_cate[idx_opto,:,:])
            upper = np.nanmax([m_control, m_opto]) + np.nanmax([s_control, s_opto])
            lower = np.nanmin([m_control, m_opto]) - np.nanmax([s_control, s_opto])
            ax.fill_between(
                stim,
                lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
                color='gold', alpha=0.15, step='mid', label='stim')
            plot_vol(
                ax, stim_time, led_value[idx_opto,:],
                'indigo', upper, lower)
            ax.plot([], color='indigo', label='LED on')
            plot_mean_sem(ax, neu_time, m_control, s_control, color1, 'control')
            plot_mean_sem(ax, neu_time, m_opto,    s_opto,    color2, 'opto')
            adjust_layout_neu(ax)
            ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
            ax.set_xlabel('time since stim (ms)')
            ax.set_title('response to normal stimulus w/o opto')
            fig.savefig(os.path.join(
                ops['save_path0'], 'figures',
                str(roi_id).zfill(4)+'.pdf'),
                dpi=300)
            plt.close()