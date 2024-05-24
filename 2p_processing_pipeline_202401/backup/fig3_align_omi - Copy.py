#!/usr/bin/env python3

import numpy as np
from scipy.stats import mode
from scipy.stats import sem
from matplotlib.colors import LinearSegmentedColormap


# normalization into [0,1].
def norm01(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-5)


# rescale voltage recordings.
def rescale(data, upper, lower):
    data = data.copy()
    data = ( data - np.min(data) ) / (np.max(data) - np.min(data))
    data = data * (upper - lower) + lower
    return data


# cut sequence into the same length as the shortest one given pivots.
def trim_seq(
        data,
        pivots,
        ):
    if len(data[0].shape) == 1:
        len_l_min = np.min(pivots)
        len_r_min = np.min([len(data[i])-pivots[i] for i in range(len(data))])
        data = [data[i][pivots[i]-len_l_min:pivots[i]+len_r_min]
                for i in range(len(data))]
    if len(data[0].shape) == 3:
        len_l_min = np.min(pivots)
        len_r_min = np.min([len(data[i][0,0,:])-pivots[i] for i in range(len(data))])
        data = [data[i][:, :, pivots[i]-len_l_min:pivots[i]+len_r_min]
                for i in range(len(data))]
    return data


# compute omission start point.
def get_omi_time(vol_stim, vol_time):
    stim_neg = vol_stim.copy() / np.max(vol_stim)
    stim_neg[stim_neg>0] = 0
    diff_stim_neg = np.diff(stim_neg, prepend=0)
    omi_time = vol_time[diff_stim_neg == -1]
    return omi_time


# compute expected grating time.
def get_expect_stim_time(stim_vol_fix, fix_stim_time):
    diff_stim = np.diff(stim_vol_fix, prepend=0)
    up = np.where(diff_stim==1)[0][1:]
    down = np.where(diff_stim==-1)[0][:-1]
    isi = np.median(fix_stim_time[up] - fix_stim_time[down])
    expect_stim_time = fix_stim_time[diff_stim==-1]
    expect_stim_time = expect_stim_time[expect_stim_time<0][-1] + isi
    return expect_stim_time


# compute isi before omission
def get_isi_pre_omi(vol_stim, vol_time, omi_time):
    diff_stim = np.diff(vol_stim.copy(), prepend=0)
    edge_time = vol_time[diff_stim!=0]
    isi_pre_omi = []
    for t in omi_time:
        omi_edge_idx = np.where(edge_time==t)[0]
        isi = edge_time[omi_edge_idx-2] - edge_time[omi_edge_idx-3]
        isi_pre_omi.append(isi)
    return isi_pre_omi


# bin the data with timestamps.
def get_bin_stat(data, time, time_range, bin_size):
    bins = np.arange(time_range[0], time_range[1] + bin_size, bin_size)
    bins = bins - bin_size / 2
    bin_indices = np.digitize(time, bins) - 1
    bin_mean = []
    bin_sem = []
    for i in range(len(bins)-1):
        bin_values = data[bin_indices == i]
        m = np.mean(bin_values) if len(bin_values) > 0 else np.nan
        s = sem(bin_values) if len(bin_values) > 0 else np.nan
        bin_mean.append(m)
        bin_sem.append(s)
    bin_mean = np.array(bin_mean)
    bin_sem = np.array(bin_sem)
    return bins, bin_mean, bin_sem


# extract response around omissions.
def get_omi_response(
        neural_trials,
        trial_idx,
        l_frames, r_frames
        ):
    # initialize list.
    neu_seq  = []
    neu_time = []
    stim_vol  = []
    stim_time = []
    omi_isi = []
    # loop over trials.
    for trials in trial_idx:
        # read trial data.
        fluo = neural_trials[str(trials)]['dff']
        time = neural_trials[str(trials)]['time']
        vol_stim = neural_trials[str(trials)]['vol_stim']
        vol_time = neural_trials[str(trials)]['vol_time']
        # compute stimulus start point in ms.
        omi_time = get_omi_time(vol_stim, vol_time)
        isi_pre_omi = get_isi_pre_omi(vol_stim, vol_time, omi_time)
        for i in range(len(omi_time)):
            idx = np.argmin(np.abs(time - omi_time[i]))
            if idx > l_frames and idx < len(time)-r_frames:
                # isi before omission.
                omi_isi.append(isi_pre_omi[i])
                # signal response.
                f = fluo[:, idx-l_frames : idx+r_frames]
                f = np.expand_dims(f, axis=0)
                neu_seq.append(f)
                # signal time stamps.
                t = time[idx-l_frames : idx+r_frames] - time[idx]
                neu_time.append(t)
                # voltage recordings.
                vidx = np.where(
                    (vol_time > time[idx-l_frames]) &
                    (vol_time < time[idx+r_frames]))[0]
                stim_vol.append(np.abs(vol_stim[vidx]) / np.max(vol_stim))
                # voltage time stamps.
                stim_time.append(vol_time[vidx] - time[idx])
    # correct voltage recordings centering at perturbation.
    stim_time_zero = [np.argmin(np.abs(st)) for st in stim_time]
    stim_time = trim_seq(stim_time, stim_time_zero)
    stim_vol = trim_seq(stim_vol, stim_time_zero)
    # correct neuron time stamps centering at perturbation.
    neu_time_zero = [np.argmin(np.abs(nt)) for nt in neu_time]
    neu_time = trim_seq(neu_time, neu_time_zero)
    neu_seq = trim_seq(neu_seq, neu_time_zero)
    # concatenate results.
    omi_isi = np.array(omi_isi).reshape(-1)
    neu_time  = [nt.reshape(1,-1) for nt in neu_time]
    stim_time = [st.reshape(1,-1) for st in stim_time]
    stim_vol  = [sv.reshape(1,-1) for sv in stim_vol]
    neu_seq   = np.concatenate(neu_seq, axis=0)
    neu_time  = np.concatenate(neu_time, axis=0)
    stim_vol  = np.concatenate(stim_vol, axis=0)
    stim_time = np.concatenate(stim_time, axis=0)
    # get mean time stamps.
    neu_time  = np.mean(neu_time, axis=0)
    stim_time = np.mean(stim_time, axis=0)
    # get mode and mean stimulus.
    stim_vol_fix, _ = mode(stim_vol, axis=0)
    stim_vol_jitter = np.mean(stim_vol, axis=0)
    return [neu_seq, neu_time,
            stim_vol_fix, stim_vol_jitter, stim_time,
            omi_isi]


# run computation to separate fix and jitter response.
def get_fix_jitter_response(
        vol_stim_bin, vol_time,
        neural_trials,
        l_frames, r_frames
        ):
    # find trial id for jitter and fix.
    fix_idx = [i for i in range(len(neural_trials))
               if neural_trials[str(i)]['trial_types'] == 2]
    jitter_idx = [i for i in range(len(neural_trials))
               if neural_trials[str(i)]['trial_types'] == 1]
    # fix data.
    [fix_neu_seq,  fix_neu_time,
     stim_vol_fix, _,
     fix_stim_time,
     omi_isi_fix] = get_omi_response(
        neural_trials, fix_idx,
        l_frames, r_frames)
    # jitter data.
    [jitter_neu_seq,  jitter_neu_time,
     _, stim_vol_jitter,
     jitter_stim_time,
     omi_isi_jitter] = get_omi_response(
        neural_trials, jitter_idx,
        l_frames, r_frames)
    return [fix_neu_seq, fix_neu_time,
            stim_vol_fix, fix_stim_time,
            jitter_neu_seq, jitter_neu_time,
            stim_vol_jitter, jitter_stim_time,
            omi_isi_fix, omi_isi_jitter]


# get ROI color from label.
def get_roi_label_color(labels, roi_id):
    if labels[roi_id] == -1:
        cate = 'excitory'
        color1 = 'royalblue'
        color2 = 'turquoise'
        cmap = LinearSegmentedColormap.from_list(
            'excitory', ['white', 'seagreen', 'black'])
    if labels[roi_id] == 0:
        cate = 'unsure'
        color1 = 'violet'
        color2 = 'dodgerblue'
        cmap = LinearSegmentedColormap.from_list(
            'unsure', ['white', 'dodgerblue', 'black'])
    if labels[roi_id] == 1:
        cate = 'inhibitory'
        color1 = 'brown'
        color2 = 'coral'
        cmap = LinearSegmentedColormap.from_list(
            'inhibitory', ['white', 'coral', 'black'])
    return cate, color1, color2, cmap


# adjust layout for omission average.
def adjust_layout_mean(ax):
    ax.axvline(0, color='red', lw=2, label='omission', linestyle='--')
    ax.tick_params(axis='y', tick1On=False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel('time since first grating after omission (ms)')
    ax.set_ylabel('df/f (z-scored)')
    ax.legend(loc='upper right')


# adjust layout for heatmap.
def adjust_layout_heatmap(ax):
    ax.tick_params(tick1On=False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_yticks([])
    ax.set_xlabel('time since first grating after omission (ms)')


class plotter_VIPTD_G8_align_omi:
    
    def __init__(
            self,
            vol_stim_bin, vol_time, neural_trials,
            labels
            ):
        self.l_frames = 100
        self.r_frames = 100
        self.vol_stim_bin = vol_stim_bin
        self.vol_time = vol_time
        self.neural_trials = neural_trials
        self.labels = labels
        [self.fix_neu_seq,
         self.fix_neu_time,
         self.stim_vol_fix,
         self.fix_stim_time,
         self.jitter_neu_seq,
         self.jitter_neu_time,
         self.stim_vol_jitter,
         self.jitter_stim_time,
         self.omi_isi_fix,
         self.omi_isi_jitter] = get_fix_jitter_response(
             vol_stim_bin, vol_time, neural_trials,
             self.l_frames, self.r_frames)
        
    # mean response of excitory.
    def exc_omi_mean(self, ax):
        fix_all  = self.fix_neu_seq[:, self.labels==-1, :].reshape(
            -1, self.l_frames+self.r_frames)
        fix_mean = np.mean(fix_all, axis=0)
        fix_sem  = sem(fix_all, axis=0)
        jitter_all  = self.jitter_neu_seq[:, self.labels==-1, :].reshape(
            -1, self.l_frames+self.r_frames)
        jitter_mean = np.mean(jitter_all, axis=0)
        jitter_sem  = sem(jitter_all, axis=0)
        upper = np.max(fix_mean) + np.max(fix_sem)
        lower = np.min(fix_mean) - np.max(fix_sem)
        ax.fill_between(
            self.fix_stim_time,
            lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
            where=(self.stim_vol_fix==1),
            color='silver', step='mid', label='fix stim')
        ax.plot(
            self.fix_stim_time,
            rescale(self.stim_vol_jitter, upper, lower),
            color='grey', linestyle=':', label='jitter stim')
        ax.plot(
            self.fix_neu_time,
            fix_mean,
            color='royalblue', label='excitory_fix')
        ax.plot(
            self.jitter_neu_time,
            jitter_mean,
            color='turquoise', label='excitory_jitter')
        ax.fill_between(
            self.fix_neu_time,
            fix_mean - fix_sem,
            fix_mean + fix_sem,
            color='royalblue', alpha=0.2)
        ax.fill_between(
            self.jitter_neu_time,
            jitter_mean - jitter_sem,
            jitter_mean + jitter_sem,
            color='turquoise', alpha=0.2)
        ax.axvline(
            get_expect_stim_time(self.stim_vol_fix, self.fix_stim_time),
            color='black', lw=2, label='expectation', linestyle='--')
        adjust_layout_mean(ax)
        ax.set_xlim([np.min(self.fix_stim_time), np.max(self.fix_stim_time)])
        ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
        ax.set_title(
            'omission average trace of {} excitory neurons'.format(
            np.sum(self.labels==-1)))
    
    # mean response of inhibitory.
    def inh_omi_mean(self, ax):
        fix_all  = self.fix_neu_seq[:, self.labels==1, :].reshape(
            -1, self.l_frames+self.r_frames)
        fix_mean = np.mean(fix_all, axis=0)
        fix_sem  = sem(fix_all, axis=0)
        jitter_all  = self.jitter_neu_seq[:, self.labels==1, :].reshape(
            -1, self.l_frames+self.r_frames)
        jitter_mean = np.mean(jitter_all, axis=0)
        jitter_sem  = sem(jitter_all, axis=0)
        upper = np.max(fix_mean) + np.max(fix_sem)
        lower = np.min(fix_mean) - np.max(fix_sem)
        ax.fill_between(
            self.fix_stim_time,
            lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
            where=(self.stim_vol_fix==1),
            color='silver', step='mid', label='fix stim')
        ax.plot(
            self.fix_stim_time,
            rescale(self.stim_vol_jitter, upper, lower),
            color='grey', linestyle=':', label='jitter stim')
        ax.plot(
            self.fix_neu_time,
            fix_mean,
            color='brown', label='inhitory_fix')
        ax.plot(
            self.jitter_neu_time,
            jitter_mean,
            color='coral', label='inhitory_jitter')
        ax.fill_between(
            self.fix_neu_time,
            fix_mean - fix_sem,
            fix_mean + fix_sem,
            color='brown', alpha=0.2)
        ax.fill_between(
            self.jitter_neu_time,
            jitter_mean - jitter_sem,
            jitter_mean + jitter_sem,
            color='coral', alpha=0.2)
        ax.axvline(
            get_expect_stim_time(self.stim_vol_fix, self.fix_stim_time),
            color='black', lw=2, label='expectation', linestyle='--')
        adjust_layout_mean(ax)
        ax.set_xlim([np.min(self.fix_stim_time), np.max(self.fix_stim_time)])
        ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
        ax.set_title(
            'omission average trace of {} inhitory neurons'.format(
            np.sum(self.labels==1)))

    # response heat map around omission for fix with average across trials.
    def omi_fix_heatmap_neuron(self, ax):
        mean = np.mean(self.fix_neu_seq, axis=0)
        for i in range(mean.shape[0]):
            mean[i,:] = norm01(mean[i,:])
        zero = np.where(self.fix_neu_time==0)[0][0]
        expect = get_expect_stim_time(self.stim_vol_fix, self.fix_stim_time)
        expect = np.argmin(np.abs(self.fix_neu_time-expect))
        sort_idx_fix = mean[:, zero].reshape(-1).argsort()
        sort_fix = mean[sort_idx_fix,:]
        cmap = LinearSegmentedColormap.from_list(
            'fix', ['white','dodgerblue', 'black'])
        im_fix = ax.imshow(
            sort_fix, interpolation='nearest', aspect='auto', cmap=cmap)
        adjust_layout_heatmap(ax)
        ax.set_ylabel('neuron id (sorted)')
        diff_stim_up = np.diff(self.stim_vol_fix, prepend=0)
        diff_stim_up = self.fix_stim_time[diff_stim_up==1]
        for stim_idx in diff_stim_up:
            ax.axvline(
                np.argmin(np.abs(self.fix_neu_time-stim_idx)), color='black', lw=1,
                label='grating', linestyle='--')
        ax.axvline(zero, color='red', lw=1, label='omission', linestyle='--')
        ax.axvline(
            expect, color='grey', lw=1,
            label='expect grating', linestyle='--')
        ax.axvline(zero, color='red', lw=1, label='omission', linestyle='--')
        ax.set_xticks([0, zero, len(self.fix_neu_time)])
        ax.set_xticklabels([int(self.fix_neu_time[0]), 0, int(self.fix_neu_time[-1])])
        cbar = ax.figure.colorbar(im_fix, ax=ax, ticks=[0.2,0.8])
        cbar.ax.set_ylabel('normalized response', rotation=-90, va="bottom")
        cbar.ax.set_yticklabels(['0.2', '0.8'])
        ax.set_title('omission response heatmap on fix')

    # response heat map around omission for jitter with average across trials.
    def omi_jitter_heatmap_neuron(self, ax):
        mean = np.mean(self.jitter_neu_seq, axis=0)
        for i in range(mean.shape[0]):
            mean[i,:] = norm01(mean[i,:])
        zero = np.where(self.fix_neu_time==0)[0][0]
        expect = get_expect_stim_time(self.stim_vol_fix, self.fix_stim_time)
        expect = np.argmin(np.abs(self.fix_neu_time-expect))
        sort_idx_fix = mean[:, zero].reshape(-1).argsort()
        sort_fix = mean[sort_idx_fix,:]
        cmap = LinearSegmentedColormap.from_list(
            'jitter', ['white','violet', 'black'])
        im_fix = ax.imshow(
            sort_fix, interpolation='nearest', aspect='auto', cmap=cmap)
        adjust_layout_heatmap(ax)
        ax.set_ylabel('neuron id (sorted)')
        ax.axvline(zero, color='red', lw=1, label='omission', linestyle='--')
        ax.axvline(
            expect, color='black', lw=1,
            label='expect grating', linestyle='--')
        ax.set_xticks([0, zero, len(self.fix_neu_time)])
        ax.set_xticklabels([int(self.fix_neu_time[0]), 0, int(self.fix_neu_time[-1])])
        cbar = ax.figure.colorbar(im_fix, ax=ax, ticks=[0.2,0.8])
        cbar.ax.set_ylabel('normalized response', rotation=-90, va="bottom")
        cbar.ax.set_yticklabels(['0.2', '0.8'])
        ax.set_title('omission response heatmap on jitter')

    # response heat map around omission for fix with average across excitory.
    def omi_fix_exc_heatmap_trials(self, ax):
        mean = np.mean(self.fix_neu_seq[:,self.labels==-1,:], axis=1)
        for i in range(mean.shape[0]):
            mean[i,:] = norm01(mean[i,:])
        zero = np.where(self.fix_neu_time==0)[0][0]
        expect = get_expect_stim_time(self.stim_vol_fix, self.fix_stim_time)
        expect = np.argmin(np.abs(self.fix_neu_time-expect))
        cmap = LinearSegmentedColormap.from_list(
            'exc', ['white','seagreen', 'black'])
        im_fix = ax.imshow(
            mean, interpolation='nearest', aspect='auto', cmap=cmap)
        adjust_layout_heatmap(ax)
        ax.set_ylabel('trial id')
        ax.axvline(zero, color='red', lw=1, label='omission', linestyle='--')
        diff_stim_up = np.diff(self.stim_vol_fix, prepend=0)
        diff_stim_up = self.fix_stim_time[diff_stim_up==1]
        for stim_idx in diff_stim_up:
            ax.axvline(
                np.argmin(np.abs(self.fix_neu_time-stim_idx)), color='black', lw=1,
                label='grating', linestyle='--')
        ax.axvline(zero, color='red', lw=1, label='omission', linestyle='--')
        ax.axvline(
            expect, color='grey', lw=1,
            label='expect grating', linestyle='--')
        ax.set_xticks([0, zero, len(self.fix_neu_time)])
        ax.set_xticklabels([int(self.fix_neu_time[0]), 0, int(self.fix_neu_time[-1])])
        cbar = ax.figure.colorbar(im_fix, ax=ax, ticks=[0.2,0.8])
        cbar.ax.set_ylabel('normalized response', rotation=-90, va="bottom")
        cbar.ax.set_yticklabels(['0.2', '0.8'])
        ax.set_title('excitory single trial omission response heatmap on fix')
    
    # response heat map around omission for fix with average across inhibitory.
    def omi_fix_inh_heatmap_trials(self, ax):
        mean = np.mean(self.fix_neu_seq[:,self.labels==1,:], axis=1)
        for i in range(mean.shape[0]):
            mean[i,:] = norm01(mean[i,:])
        zero = np.where(self.fix_neu_time==0)[0][0]
        expect = get_expect_stim_time(self.stim_vol_fix, self.fix_stim_time)
        expect = np.argmin(np.abs(self.fix_neu_time-expect))
        cmap = LinearSegmentedColormap.from_list(
            'exc', ['white', 'coral', 'black'])
        im_fix = ax.imshow(
            mean, interpolation='nearest', aspect='auto', cmap=cmap)
        adjust_layout_heatmap(ax)
        ax.set_ylabel('trial id')
        ax.axvline(zero, color='red', lw=1, label='omission', linestyle='--')
        diff_stim_up = np.diff(self.stim_vol_fix, prepend=0)
        diff_stim_up = self.fix_stim_time[diff_stim_up==1]
        for stim_idx in diff_stim_up:
            ax.axvline(
                np.argmin(np.abs(self.fix_neu_time-stim_idx)), color='black', lw=1,
                label='grating', linestyle='--')
        ax.axvline(zero, color='red', lw=1, label='omission', linestyle='--')
        ax.axvline(
            expect, color='grey', lw=1,
            label='expect grating', linestyle='--')
        ax.set_xticks([0, zero, len(self.fix_neu_time)])
        ax.set_xticklabels([int(self.fix_neu_time[0]), 0, int(self.fix_neu_time[-1])])
        cbar = ax.figure.colorbar(im_fix, ax=ax, ticks=[0.2,0.8])
        cbar.ax.set_ylabel('normalized response', rotation=-90, va="bottom")
        cbar.ax.set_yticklabels(['0.2', '0.8'])
        ax.set_title('inhibitory single trial omission response heatmap on fix')
    
    # omission response magnitude and isi before omission.
    def omi_isi(self, ax):
        frames = 1
        time_range = [200,1000]
        bin_size = 100
        offset = 2
        zero = np.where(self.jitter_neu_time==0)[0][0]
        mag_exc = self.jitter_neu_seq[:,self.labels==-1, zero:zero+frames]
        mag_exc = np.mean(mag_exc,axis=2).reshape(-1)
        mag_inh = self.jitter_neu_seq[:,self.labels==1, zero:zero+frames]
        mag_inh = np.mean(mag_inh,axis=2).reshape(-1)
        isi_exc = np.tile(self.omi_isi_jitter, np.sum(self.labels==-1))
        isi_inh = np.tile(self.omi_isi_jitter, np.sum(self.labels==1))
        bins, bin_mean_exc, bin_sem_exc = get_bin_stat(
            mag_exc, isi_exc, time_range, bin_size)
        bins, bin_mean_inh, bin_sem_inh = get_bin_stat(
            mag_inh, isi_inh, time_range, bin_size)
        ax.errorbar(
            bins[:-1] + (bins[1]-bins[0]) / 2 - offset,
            bin_mean_exc,
            bin_sem_exc,
            color='mediumaquamarine', capsize=2, marker='o',
            markeredgecolor='white', markeredgewidth=1, label='excitory')
        ax.errorbar(
            bins[:-1] + (bins[1]-bins[0]) / 2 + offset,
            bin_mean_inh,
            bin_sem_inh,
            color='lightcoral', capsize=2, marker='o',
            markeredgecolor='white', markeredgewidth=1, label='inhibitory')
        ax.tick_params(tick1On=False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_title('omission response and isi beforee omission')
        ax.set_xlabel('isi before omission (ms)')
        ax.set_ylabel('df/f (mean$\pm$sem)')
        ax.yaxis.grid(True)
        ax.legend(loc='upper left')
                 
    # ROI mean omission response.
    def roi_omi_mean(self, ax, roi_id):
        fix_all  = self.fix_neu_seq[:, roi_id, :].reshape(
            -1, self.l_frames+self.r_frames)
        fix_mean = np.mean(fix_all, axis=0)
        fix_sem  = sem(fix_all, axis=0)
        jitter_all  = self.jitter_neu_seq[:, roi_id, :].reshape(
            -1, self.l_frames+self.r_frames)
        jitter_mean = np.mean(jitter_all, axis=0)
        jitter_sem  = sem(jitter_all, axis=0)
        upper = np.max(fix_mean) + np.max(fix_sem)
        lower = np.min(fix_mean) - np.max(fix_sem)
        _, color1, color2, _ = get_roi_label_color(self.labels, roi_id)
        ax.fill_between(
            self.fix_stim_time,
            lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
            where=(self.stim_vol_fix==1),
            color='silver', step='mid', label='fix stim')
        ax.plot(
            self.fix_stim_time,
            rescale(self.stim_vol_jitter, upper, lower),
            color='grey', linestyle=':', label='jitter stim')
        ax.plot(
            self.fix_neu_time,
            fix_mean,
            color=color1, label='dff on fix')
        ax.plot(
            self.jitter_neu_time,
            jitter_mean,
            color=color2, label='dff on jitter')
        ax.fill_between(
            self.fix_neu_time,
            fix_mean - fix_sem,
            fix_mean + fix_sem,
            color=color1, alpha=0.2)
        ax.fill_between(
            self.jitter_neu_time,
            jitter_mean - jitter_sem,
            jitter_mean + jitter_sem,
            color=color2, alpha=0.2)
        ax.axvline(
            get_expect_stim_time(self.stim_vol_fix, self.fix_stim_time),
            color='black', lw=2, label='expectation', linestyle='--')
        adjust_layout_mean(ax)
        ax.set_xlim([np.min(self.fix_stim_time), np.max(self.fix_stim_time)])
        ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
        ax.set_title('ROI # {} mean response to omission'.format(str(roi_id).zfill(4)))
    
    # ROI response heat map around omission for fix.
    def roi_omi_fix_heatmap_trials(self, ax, roi_id):
        fix_neu_seq = self.fix_neu_seq[:,roi_id,:]
        for i in range(fix_neu_seq.shape[0]):
            fix_neu_seq[i,:] = norm01(fix_neu_seq[i,:])
        zero = np.where(self.fix_neu_time==0)[0][0]
        expect = get_expect_stim_time(self.stim_vol_fix, self.fix_stim_time)
        expect = np.argmin(np.abs(self.fix_neu_time-expect))
        _, _, _, cmap = get_roi_label_color(self.labels, roi_id)
        im_fix = ax.imshow(
            fix_neu_seq, interpolation='nearest', aspect='auto', cmap=cmap)
        adjust_layout_heatmap(ax)
        ax.set_ylabel('trial id')
        ax.axvline(zero, color='red', lw=1, label='omission', linestyle='--')
        diff_stim_up = np.diff(self.stim_vol_fix, prepend=0)
        diff_stim_up = self.fix_stim_time[diff_stim_up==1]
        for stim_idx in diff_stim_up:
            ax.axvline(
                np.argmin(np.abs(self.fix_neu_time-stim_idx)), color='black', lw=1,
                label='grating', linestyle='--')
        ax.axvline(zero, color='red', lw=1, label='omission', linestyle='--')
        ax.axvline(
            expect, color='grey', lw=1,
            label='expect grating', linestyle='--')
        ax.set_xticks([0, zero, len(self.fix_neu_time)])
        ax.set_xticklabels([int(self.fix_neu_time[0]), 0, int(self.fix_neu_time[-1])])
        cbar = ax.figure.colorbar(im_fix, ax=ax, ticks=[0.2,0.8])
        cbar.ax.set_ylabel('normalized response', rotation=-90, va="bottom")
        cbar.ax.set_yticklabels(['0.2', '0.8'])
        ax.set_title('ROI # {} single trial omission response heatmap on fix'.format(
            str(roi_id).zfill(4)))

    # ROI omission response magnitude and isi before omission.
    def roi_omi_isi(self, ax, roi_id):
        frames = 3
        time_range = [200,1000]
        bin_size = 100
        zero = np.where(self.jitter_neu_time==0)[0][0]
        mag = self.jitter_neu_seq[:,roi_id, zero:zero+frames]
        mag = np.mean(mag, axis=1)
        _, color1, _, _ = get_roi_label_color(self.labels, roi_id)
        bins, bin_mean, bin_sem = get_bin_stat(
            mag, self.omi_isi_jitter, time_range, bin_size)
        ax.errorbar(
            bins[:-1] + (bins[1]-bins[0]) / 2,
            bin_mean,
            bin_sem,
            color=color1, capsize=2, marker='o',
            markeredgecolor='white', markeredgewidth=1)
        ax.tick_params(tick1On=False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_title('ROI # {} omission response and isi beforee omission'.format(
            str(roi_id).zfill(4)))
        ax.set_xlabel('isi before omission (ms)')
        ax.set_ylabel('df/f (mean$\pm$sem)')
        ax.yaxis.grid(True)
    
    # ROI omission response magnitude for fix and jitter.
    def roi_omi_fix_jitter(self, ax, roi_id):
        frames = 5
        zero = np.where(self.jitter_neu_time==0)[0][0]
        mag_fix = self.fix_neu_seq[:,roi_id, zero-frames:zero+frames]
        mag_fix = np.mean(mag_fix, axis=1)
        mag_jitter = self.jitter_neu_seq[:,roi_id, zero-frames:zero+frames]
        mag_jitter = np.mean(mag_jitter, axis=1)
        _, color1, color2, _ = get_roi_label_color(self.labels, roi_id)
        bp = ax.boxplot(
            [mag_fix, mag_jitter],
            showfliers=False, labels=['fix', 'jitter'])
        colors = [color1, color2]
        for i in range(2):
            for item in ['boxes', 'medians']:
                bp[item][i].set_color(colors[i])
        for i in range(2):
            for item in ['whiskers', 'caps']:
                bp[item][2*i].set_color(colors[i])
                bp[item][2*i+1].set_color(colors[i])
        ax.tick_params(tick1On=False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_title('ROI # {} omission response for fix and jitter'.format(
            str(roi_id).zfill(4)))
        ax.set_ylabel('df/f')

    
class plotter_L7G8_align_omi:
    
    def __init__(
            self,
            vol_stim_bin, vol_time, neural_trials,
            labels
            ):
        self.l_frames = 100
        self.r_frames = 100
        self.vol_stim_bin = vol_stim_bin
        self.vol_time = vol_time
        self.neural_trials = neural_trials
        self.labels = labels
        [self.fix_neu_seq,
         self.fix_neu_time,
         self.stim_vol_fix,
         self.fix_stim_time,
         self.jitter_neu_seq,
         self.jitter_neu_time,
         self.stim_vol_jitter,
         self.jitter_stim_time,
         self.omi_isi_fix,
         self.omi_isi_jitter] = get_fix_jitter_response(
             vol_stim_bin, vol_time, neural_trials,
             self.l_frames, self.r_frames)
    
    # mean response.
    def omi_mean(self, ax):
        fix_all  = self.fix_neu_seq.reshape(-1, self.l_frames+self.r_frames)
        fix_mean = np.mean(fix_all, axis=0)
        fix_sem  = sem(fix_all, axis=0)
        jitter_all  = self.jitter_neu_seq.reshape(-1, self.l_frames+self.r_frames)
        jitter_mean = np.mean(jitter_all, axis=0)
        jitter_sem  = sem(jitter_all, axis=0)
        upper = np.max(fix_mean) + np.max(fix_sem)
        lower = np.min(fix_mean) - np.max(fix_sem)
        ax.fill_between(
            self.fix_stim_time,
            lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
            where=(self.stim_vol_fix!=0),
            color='silver', step='mid', label='fix stim')
        ax.plot(
            self.fix_stim_time,
            rescale(self.stim_vol_jitter, upper, lower),
            color='grey', linestyle=':', label='jitter stim')
        ax.plot(
            self.fix_neu_time,
            fix_mean,
            color='royalblue', label='fix')
        ax.plot(
            self.jitter_neu_time,
            jitter_mean,
            color='coral', label='jitter')
        ax.fill_between(
            self.fix_neu_time,
            fix_mean - fix_sem,
            fix_mean + fix_sem,
            color='royalblue', alpha=0.2)
        ax.fill_between(
            self.jitter_neu_time,
            jitter_mean - jitter_sem,
            jitter_mean + jitter_sem,
            color='coral', alpha=0.2)
        ax.axvline(
            get_expect_stim_time(self.stim_vol_fix, self.fix_stim_time),
            color='black', lw=2, label='expectation', linestyle='--')
        adjust_layout_mean(ax)
        ax.set_xlim([np.min(self.fix_stim_time), np.max(self.fix_stim_time)])
        ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
        ax.set_title(
            'omission average trace of {} ROIs'.format(
            self.fix_neu_seq.shape[1]))

    # response heat map around omission for fix with average across trials.
    def omi_fix_heatmap_neuron(self, ax):
        mean = np.mean(self.fix_neu_seq, axis=0)
        for i in range(mean.shape[0]):
            mean[i,:] = norm01(mean[i,:])
        zero = np.where(self.fix_neu_time==0)[0][0]
        expect = get_expect_stim_time(self.stim_vol_fix, self.fix_stim_time)
        sort_idx_fix = mean[:, zero].reshape(-1).argsort()
        sort_fix = mean[sort_idx_fix,:]
        cmap = LinearSegmentedColormap.from_list(
            'fix', ['white','dodgerblue', 'black'])
        im_fix = ax.imshow(
            sort_fix, interpolation='nearest', aspect='auto', cmap=cmap)
        adjust_layout_heatmap(ax)
        ax.set_ylabel('neuron id')
        diff_stim_up = np.diff(self.stim_vol_fix, prepend=0)
        diff_stim_up = self.fix_stim_time[diff_stim_up==1]
        for stim_idx in diff_stim_up:
            ax.axvline(
                np.argmin(np.abs(self.fix_neu_time-stim_idx)), color='black', lw=1,
                label='grating', linestyle='--')
        ax.axvline(zero, color='red', lw=1, label='omission', linestyle='--')
        ax.axvline(
            np.argmin(np.abs(self.fix_neu_time-expect)), color='grey', lw=1,
            label='expect grating', linestyle='--')
        ax.set_xticks([0, zero, len(self.fix_neu_time)])
        ax.set_xticklabels([int(self.fix_neu_time[0]), 0, int(self.fix_neu_time[-1])])
        cbar = ax.figure.colorbar(im_fix, ax=ax, ticks=[0.2,0.8])
        cbar.ax.set_ylabel('normalized response', rotation=-90, va="bottom")
        cbar.ax.set_yticklabels(['0.2', '0.8'])
        ax.set_title('omission response heatmap on fix')
    
    # response heat map around omission for jitter with average across trials.
    def omi_jitter_heatmap_neuron(self, ax):
        mean = np.mean(self.jitter_neu_seq, axis=0)
        for i in range(mean.shape[0]):
            mean[i,:] = norm01(mean[i,:])
        zero = np.where(self.fix_neu_time==0)[0][0]
        expect = get_expect_stim_time(self.stim_vol_fix, self.fix_stim_time)
        sort_idx_fix = mean[:, zero].reshape(-1).argsort()
        sort_fix = mean[sort_idx_fix,:]
        cmap = LinearSegmentedColormap.from_list(
            'jitter', ['white','violet', 'black'])
        im_fix = ax.imshow(
            sort_fix, interpolation='nearest', aspect='auto', cmap=cmap)
        adjust_layout_heatmap(ax)
        ax.set_ylabel('neuron id')
        diff_stim_up = np.diff(self.stim_vol_fix, prepend=0)
        diff_stim_up = self.fix_stim_time[diff_stim_up==1]
        for stim_idx in diff_stim_up:
            ax.axvline(
                np.argmin(np.abs(self.fix_neu_time-stim_idx)), color='black', lw=1,
                label='grating', linestyle='--')
        ax.axvline(zero, color='red', lw=1, label='omission', linestyle='--')
        ax.axvline(
            np.argmin(np.abs(self.fix_neu_time-expect)), color='grey', lw=1,
            label='expect grating', linestyle='--')
        ax.set_xticks([0, zero, len(self.fix_neu_time)])
        ax.set_xticklabels([int(self.fix_neu_time[0]), 0, int(self.fix_neu_time[-1])])
        cbar = ax.figure.colorbar(im_fix, ax=ax, ticks=[0.2,0.8])
        cbar.ax.set_ylabel('normalized response', rotation=-90, va="bottom")
        cbar.ax.set_yticklabels(['0.2', '0.8'])
        ax.set_title('omission response heatmap on jitter')

    # omission response magnitude and isi before omission.
    def omi_isi(self, ax):
        frames = 1
        time_range = [200,1000]
        bin_size = 100
        offset = 2
        zero = np.where(self.jitter_neu_time==0)[0][0]
        mag = self.jitter_neu_seq[:,:, zero:zero+frames]
        mag = np.mean(mag,axis=2).reshape(-1)
        isi = np.tile(self.omi_isi_jitter, self.jitter_neu_seq.shape[1])
        bins, bin_mean, bin_sem = get_bin_stat(
            mag, isi, time_range, bin_size)
        ax.errorbar(
            bins[:-1] + (bins[1]-bins[0]) / 2 - offset,
            bin_mean,
            bin_sem,
            color='mediumaquamarine',
            capsize=2,
            marker='o',
            markeredgecolor='white',
            markeredgewidth=1,
            label='excitory')
        ax.tick_params(tick1On=False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_title('omission response and isi beforee omission')
        ax.set_xlabel('isi before omission (ms)')
        ax.set_ylabel('df/f (mean$\pm$sem)')
        ax.yaxis.grid(True)


class plotter_L7G8_roi_align_omi:
    
    def __init__(
            self,
            vol_stim_bin, vol_time, neural_trials,
            ):
        self.l_frames = 100
        self.r_frames = 100
        self.vol_stim_bin = vol_stim_bin
        self.vol_time = vol_time
        self.neural_trials = neural_trials
        [self.fix_neu_seq,
         self.fix_neu_time,
         self.stim_vol_fix,
         self.fix_stim_time,
         self.jitter_neu_seq,
         self.jitter_neu_time,
         self.stim_vol_jitter,
         self.jitter_stim_time,
         self.omi_isi_fix,
         self.omi_isi_jitter] = get_fix_jitter_response(
             vol_stim_bin, vol_time, neural_trials,
             self.l_frames, self.r_frames)
                 
    # ROI mean omission response.
    def roi_omi_mean(self, ax, roi_id):
        fix_all  = self.fix_neu_seq[:, roi_id, :].reshape(
            -1, self.l_frames+self.r_frames)
        fix_mean = np.mean(fix_all, axis=0)
        fix_sem  = sem(fix_all, axis=0)
        jitter_all  = self.jitter_neu_seq[:, roi_id, :].reshape(
            -1, self.l_frames+self.r_frames)
        jitter_mean = np.mean(jitter_all, axis=0)
        jitter_sem  = sem(jitter_all, axis=0)
        upper = np.max(fix_mean) + np.max(fix_sem)
        lower = np.min(fix_mean) - np.max(fix_sem)
        ax.fill_between(
            self.fix_stim_time,
            lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
            where=(self.stim_vol_fix==1),
            color='silver', step='mid', label='fix stim')
        ax.plot(
            self.fix_stim_time,
            rescale(self.stim_vol_jitter, upper, lower),
            color='grey', linestyle=':', label='jitter stim')
        ax.plot(
            self.fix_neu_time,
            fix_mean,
            color='royalblue', label='dff on fix')
        ax.plot(
            self.jitter_neu_time,
            jitter_mean,
            color='turquoise', label='dff on jitter')
        ax.fill_between(
            self.fix_neu_time,
            fix_mean - fix_sem,
            fix_mean + fix_sem,
            color='royalblue', alpha=0.2)
        ax.fill_between(
            self.jitter_neu_time,
            jitter_mean - jitter_sem,
            jitter_mean + jitter_sem,
            color='turquoise', alpha=0.2)
        ax.axvline(
            get_expect_stim_time(self.stim_vol_fix, self.fix_stim_time),
            color='black', lw=2, label='expectation', linestyle='--')
        adjust_layout_mean(ax)
        ax.set_xlim([np.min(self.fix_stim_time), np.max(self.fix_stim_time)])
        ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
        ax.set_title('ROI # {} mean response to omission'.format(str(roi_id).zfill(4)))
    
    # ROI response heat map around omission for fix.
    def roi_omi_fix_heatmap_trials(self, ax, roi_id):
        fix_neu_seq = self.fix_neu_seq[:,roi_id,:]
        for i in range(fix_neu_seq.shape[0]):
            fix_neu_seq[i,:] = norm01(fix_neu_seq[i,:])
        zero = np.where(self.fix_neu_time==0)[0][0]
        expect = get_expect_stim_time(self.stim_vol_fix, self.fix_stim_time)
        expect = np.argmin(np.abs(self.fix_neu_time-expect))
        cmap = LinearSegmentedColormap.from_list(
            'excitory', ['white', 'seagreen', 'black'])
        im_fix = ax.imshow(
            fix_neu_seq, interpolation='nearest', aspect='auto', cmap=cmap)
        adjust_layout_heatmap(ax)
        ax.set_ylabel('trial id')
        diff_stim_up = np.diff(self.stim_vol_fix, prepend=0)
        diff_stim_up = self.fix_stim_time[diff_stim_up==1]
        for stim_idx in diff_stim_up:
            ax.axvline(
                np.argmin(np.abs(self.fix_neu_time-stim_idx)), color='black', lw=1,
                label='grating', linestyle='--')
        ax.axvline(zero, color='red', lw=1, label='omission', linestyle='--')
        ax.axvline(
            expect, color='grey', lw=1,
            label='expect grating', linestyle='--')
        ax.set_xticks([0, zero, len(self.fix_neu_time)])
        ax.set_xticklabels([int(self.fix_neu_time[0]), 0, int(self.fix_neu_time[-1])])
        cbar = ax.figure.colorbar(im_fix, ax=ax, ticks=[0.2,0.8])
        cbar.ax.set_ylabel('normalized response', rotation=-90, va="bottom")
        cbar.ax.set_yticklabels(['0.2', '0.8'])
        ax.set_title('ROI # {} single trial omission response heatmap on fix'.format(
            str(roi_id).zfill(4)))

    # ROI omission response magnitude and isi before omission.
    def roi_omi_isi(self, ax, roi_id):
        frames = 3
        time_range = [200,1000]
        bin_size = 100
        zero = np.where(self.jitter_neu_time==0)[0][0]
        mag = self.jitter_neu_seq[:,roi_id, zero:zero+frames]
        mag = np.mean(mag, axis=1)
        bins, bin_mean, bin_sem = get_bin_stat(
            mag, self.omi_isi_jitter, time_range, bin_size)
        ax.errorbar(
            bins[:-1] + (bins[1]-bins[0]) / 2,
            bin_mean,
            bin_sem,
            color='royalblue', capsize=2, marker='o',
            markeredgecolor='white', markeredgewidth=1)
        ax.tick_params(tick1On=False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_title('ROI # {} omission response and isi beforee omission'.format(
            str(roi_id).zfill(4)))
        ax.set_xlabel('isi before omission (ms)')
        ax.set_ylabel('df/f (mean$\pm$sem)')
        ax.yaxis.grid(True)
    
    # ROI omission response magnitude for fix and jitter.
    def roi_omi_fix_jitter(self, ax, roi_id):
        frames = 5
        zero = np.where(self.jitter_neu_time==0)[0][0]
        mag_fix = self.fix_neu_seq[:,roi_id, zero-frames:zero+frames]
        mag_fix = np.mean(mag_fix, axis=1)
        mag_jitter = self.jitter_neu_seq[:,roi_id, zero-frames:zero+frames]
        mag_jitter = np.mean(mag_jitter, axis=1)
        bp = ax.boxplot(
            [mag_fix, mag_jitter],
            showfliers=False, labels=['fix', 'jitter'])
        colors = ['royalblue', 'turquoise']
        for i in range(2):
            for item in ['boxes', 'medians']:
                bp[item][i].set_color(colors[i])
        for i in range(2):
            for item in ['whiskers', 'caps']:
                bp[item][2*i].set_color(colors[i])
                bp[item][2*i+1].set_color(colors[i])
        ax.tick_params(tick1On=False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_title('ROI # {} omission response for fix and jitter'.format(
            str(roi_id).zfill(4)))
        ax.set_ylabel('df/f')
        

class plotter_VIPG8_align_omi:
    
    def __init__(
            self,
            vol_stim_bin, vol_time, neural_trials,
            labels
            ):
        self.l_frames = 100
        self.r_frames = 100
        self.vol_stim_bin = vol_stim_bin
        self.vol_time = vol_time
        self.neural_trials = neural_trials
        self.labels = labels
        [self.fix_neu_seq,
         self.fix_neu_time,
         self.stim_vol_fix,
         self.fix_stim_time,
         self.jitter_neu_seq,
         self.jitter_neu_time,
         self.stim_vol_jitter,
         self.jitter_stim_time,
         self.omi_isi_fix,
         self.omi_isi_jitter] = get_fix_jitter_response(
             vol_stim_bin, vol_time, neural_trials,
             self.l_frames, self.r_frames)
    
    # mean response.
    def omi_mean(self, ax):
        fix_all  = self.fix_neu_seq.reshape(-1, self.l_frames+self.r_frames)
        fix_mean = np.mean(fix_all, axis=0)
        fix_sem  = sem(fix_all, axis=0)
        jitter_all  = self.jitter_neu_seq.reshape(-1, self.l_frames+self.r_frames)
        jitter_mean = np.mean(jitter_all, axis=0)
        jitter_sem  = sem(jitter_all, axis=0)
        upper = np.max(fix_mean) + np.max(fix_sem)
        lower = np.min(fix_mean) - np.max(fix_sem)
        ax.fill_between(
            self.fix_stim_time,
            lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
            where=(self.stim_vol_fix!=0),
            color='silver', step='mid', label='fix stim')
        ax.plot(
            self.fix_stim_time,
            rescale(self.stim_vol_jitter, upper, lower),
            color='grey', linestyle=':', label='jitter stim')
        ax.plot(
            self.fix_neu_time,
            fix_mean,
            color='brown', label='fix')
        ax.plot(
            self.jitter_neu_time,
            jitter_mean,
            color='coral', label='jitter')
        ax.fill_between(
            self.fix_neu_time,
            fix_mean - fix_sem,
            fix_mean + fix_sem,
            color='brown', alpha=0.2)
        ax.fill_between(
            self.jitter_neu_time,
            jitter_mean - jitter_sem,
            jitter_mean + jitter_sem,
            color='coral', alpha=0.2)
        ax.axvline(
            get_expect_stim_time(self.stim_vol_fix, self.fix_stim_time),
            color='black', lw=2, label='expectation', linestyle='--')
        adjust_layout_mean(ax)
        ax.set_xlim([np.min(self.fix_stim_time), np.max(self.fix_stim_time)])
        ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
        ax.set_title(
            'omission average trace of {} ROIs'.format(
            self.fix_neu_seq.shape[1]))

    # response heat map around omission for fix with average across trials.
    def omi_fix_heatmap_neuron(self, ax):
        mean = np.mean(self.fix_neu_seq, axis=0)
        for i in range(mean.shape[0]):
            mean[i,:] = norm01(mean[i,:])
        zero = np.where(self.fix_neu_time==0)[0][0]
        expect = get_expect_stim_time(self.stim_vol_fix, self.fix_stim_time)
        expect = np.argmin(np.abs(self.fix_neu_time-expect))
        sort_idx_fix = mean[:, zero].reshape(-1).argsort()
        sort_fix = mean[sort_idx_fix,:]
        cmap = LinearSegmentedColormap.from_list(
            'fix', ['white','dodgerblue', 'black'])
        im_fix = ax.imshow(
            sort_fix, interpolation='nearest', aspect='auto', cmap=cmap)
        adjust_layout_heatmap(ax)
        ax.set_ylabel('neuron id')
        diff_stim_up = np.diff(self.stim_vol_fix, prepend=0)
        diff_stim_up = self.fix_stim_time[diff_stim_up==1]
        for stim_idx in diff_stim_up:
            ax.axvline(
                np.argmin(np.abs(self.fix_neu_time-stim_idx)), color='black', lw=1,
                label='grating', linestyle='--')
        ax.axvline(zero, color='red', lw=1, label='omission', linestyle='--')
        ax.axvline(
            expect, color='black', lw=1,
            label='expect grating', linestyle='--')
        ax.set_xticks([0, zero, len(self.fix_neu_time)])
        ax.set_xticklabels([int(self.fix_neu_time[0]), 0, int(self.fix_neu_time[-1])])
        cbar = ax.figure.colorbar(im_fix, ax=ax, ticks=[0.2,0.8])
        cbar.ax.set_ylabel('normalized response', rotation=-90, va="bottom")
        cbar.ax.set_yticklabels(['0.2', '0.8'])
        ax.set_title('omission response heatmap on fix')
    
    # response heat map around omission for jitter with average across trials.
    def omi_jitter_heatmap_neuron(self, ax):
        mean = np.mean(self.jitter_neu_seq, axis=0)
        for i in range(mean.shape[0]):
            mean[i,:] = norm01(mean[i,:])
        zero = np.where(self.fix_neu_time==0)[0][0]
        expect = get_expect_stim_time(self.stim_vol_fix, self.fix_stim_time)
        expect = np.argmin(np.abs(self.fix_neu_time-expect))
        sort_idx_fix = mean[:, zero].reshape(-1).argsort()
        sort_fix = mean[sort_idx_fix,:]
        cmap = LinearSegmentedColormap.from_list(
            'jitter', ['white','violet', 'black'])
        im_fix = ax.imshow(
            sort_fix, interpolation='nearest', aspect='auto', cmap=cmap)
        adjust_layout_heatmap(ax)
        ax.set_ylabel('neuron id')
        ax.axvline(zero, color='red', lw=1, label='omission', linestyle='--')
        ax.axvline(
            expect, color='black', lw=1,
            label='expect grating', linestyle='--')
        ax.set_xticks([0, zero, len(self.fix_neu_time)])
        ax.set_xticklabels([int(self.fix_neu_time[0]), 0, int(self.fix_neu_time[-1])])
        cbar = ax.figure.colorbar(im_fix, ax=ax, ticks=[0.2,0.8])
        cbar.ax.set_ylabel('normalized response', rotation=-90, va="bottom")
        cbar.ax.set_yticklabels(['0.2', '0.8'])
        ax.set_title('omission response heatmap on jitter')

    # omission response magnitude and isi before omission.
    def omi_isi(self, ax):
        frames = 1
        time_range = [200,1000]
        bin_size = 100
        offset = 2
        zero = np.where(self.jitter_neu_time==0)[0][0]
        mag = self.jitter_neu_seq[:,:, zero:zero+frames]
        mag = np.mean(mag,axis=2).reshape(-1)
        isi = np.tile(self.omi_isi_jitter, self.jitter_neu_seq.shape[1])
        bins, bin_mean, bin_sem = get_bin_stat(
            mag, isi, time_range, bin_size)
        ax.errorbar(
            bins[:-1] + (bins[1]-bins[0]) / 2 - offset,
            bin_mean,
            bin_sem,
            color='coral',
            capsize=2,
            marker='o',
            markeredgecolor='white',
            markeredgewidth=1,
            label='excitory')
        ax.tick_params(tick1On=False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_title('omission response and isi beforee omission')
        ax.set_xlabel('isi before omission (ms)')
        ax.set_ylabel('df/f (mean$\pm$sem)')
        ax.yaxis.grid(True)


class plotter_VIPG8_roi_align_omi:
    
    def __init__(
            self,
            vol_stim_bin, vol_time, neural_trials,
            ):
        self.l_frames = 100
        self.r_frames = 100
        self.vol_stim_bin = vol_stim_bin
        self.vol_time = vol_time
        self.neural_trials = neural_trials
        [self.fix_neu_seq,
         self.fix_neu_time,
         self.stim_vol_fix,
         self.fix_stim_time,
         self.jitter_neu_seq,
         self.jitter_neu_time,
         self.stim_vol_jitter,
         self.jitter_stim_time,
         self.omi_isi_fix,
         self.omi_isi_jitter] = get_fix_jitter_response(
             vol_stim_bin, vol_time, neural_trials,
             self.l_frames, self.r_frames)
                 
    # ROI mean omission response.
    def roi_omi_mean(self, ax, roi_id):
        fix_all  = self.fix_neu_seq[:, roi_id, :].reshape(
            -1, self.l_frames+self.r_frames)
        fix_mean = np.mean(fix_all, axis=0)
        fix_sem  = sem(fix_all, axis=0)
        jitter_all  = self.jitter_neu_seq[:, roi_id, :].reshape(
            -1, self.l_frames+self.r_frames)
        jitter_mean = np.mean(jitter_all, axis=0)
        jitter_sem  = sem(jitter_all, axis=0)
        upper = np.max(fix_mean) + np.max(fix_sem)
        lower = np.min(fix_mean) - np.max(fix_sem)
        ax.fill_between(
            self.fix_stim_time,
            lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
            where=(self.stim_vol_fix==1),
            color='silver', step='mid', label='fix stim')
        ax.plot(
            self.fix_stim_time,
            rescale(self.stim_vol_jitter, upper, lower),
            color='grey', linestyle=':', label='jitter stim')
        ax.plot(
            self.fix_neu_time,
            fix_mean,
            color='brown', label='dff on fix')
        ax.plot(
            self.jitter_neu_time,
            jitter_mean,
            color='coral', label='dff on jitter')
        ax.fill_between(
            self.fix_neu_time,
            fix_mean - fix_sem,
            fix_mean + fix_sem,
            color='brown', alpha=0.2)
        ax.fill_between(
            self.jitter_neu_time,
            jitter_mean - jitter_sem,
            jitter_mean + jitter_sem,
            color='coral', alpha=0.2)
        ax.axvline(
            get_expect_stim_time(self.stim_vol_fix, self.fix_stim_time),
            color='black', lw=2, label='expectation', linestyle='--')
        adjust_layout_mean(ax)
        ax.set_xlim([np.min(self.fix_stim_time), np.max(self.fix_stim_time)])
        ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
        ax.set_title('ROI # {} mean response to omission'.format(str(roi_id).zfill(4)))
    
    # ROI response heat map around omission for fix.
    def roi_omi_fix_heatmap_trials(self, ax, roi_id):
        fix_neu_seq = self.fix_neu_seq[:,roi_id,:]
        for i in range(fix_neu_seq.shape[0]):
            fix_neu_seq[i,:] = norm01(fix_neu_seq[i,:])
        zero = np.where(self.fix_neu_time==0)[0][0]
        expect = get_expect_stim_time(self.stim_vol_fix, self.fix_stim_time)
        expect = np.argmin(np.abs(self.fix_neu_time-expect))
        cmap = LinearSegmentedColormap.from_list(
            'excitory', ['white', 'seagreen', 'black'])
        im_fix = ax.imshow(
            fix_neu_seq, interpolation='nearest', aspect='auto', cmap=cmap)
        adjust_layout_heatmap(ax)
        ax.set_ylabel('trial id')
        ax.axvline(zero, color='red', lw=1, label='omission', linestyle='--')
        ax.axvline(
            expect, color='black', lw=1,
            label='expect grating', linestyle='--')
        ax.set_xticks([0, zero, len(self.fix_neu_time)])
        ax.set_xticklabels([int(self.fix_neu_time[0]), 0, int(self.fix_neu_time[-1])])
        cbar = ax.figure.colorbar(im_fix, ax=ax, ticks=[0.2,0.8])
        cbar.ax.set_ylabel('normalized response', rotation=-90, va="bottom")
        cbar.ax.set_yticklabels(['0.2', '0.8'])
        ax.set_title('ROI # {} single trial omission response heatmap on fix'.format(
            str(roi_id).zfill(4)))

    # ROI omission response magnitude and isi before omission.
    def roi_omi_isi(self, ax, roi_id):
        frames = 3
        time_range = [200,1000]
        bin_size = 100
        zero = np.where(self.jitter_neu_time==0)[0][0]
        mag = self.jitter_neu_seq[:,roi_id, zero:zero+frames]
        mag = np.mean(mag, axis=1)
        bins, bin_mean, bin_sem = get_bin_stat(
            mag, self.omi_isi_jitter, time_range, bin_size)
        ax.errorbar(
            bins[:-1] + (bins[1]-bins[0]) / 2,
            bin_mean,
            bin_sem,
            color='coral', capsize=2, marker='o',
            markeredgecolor='white', markeredgewidth=1)
        ax.tick_params(tick1On=False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_title('ROI # {} omission response and isi beforee omission'.format(
            str(roi_id).zfill(4)))
        ax.set_xlabel('isi before omission (ms)')
        ax.set_ylabel('df/f (mean$\pm$sem)')
        ax.yaxis.grid(True)
    
    # ROI omission response magnitude for fix and jitter.
    def roi_omi_fix_jitter(self, ax, roi_id):
        frames = 5
        zero = np.where(self.jitter_neu_time==0)[0][0]
        mag_fix = self.fix_neu_seq[:,roi_id, zero-frames:zero+frames]
        mag_fix = np.mean(mag_fix, axis=1)
        mag_jitter = self.jitter_neu_seq[:,roi_id, zero-frames:zero+frames]
        mag_jitter = np.mean(mag_jitter, axis=1)
        bp = ax.boxplot(
            [mag_fix, mag_jitter],
            showfliers=False, labels=['fix', 'jitter'])
        colors = ['brown', 'coral']
        for i in range(2):
            for item in ['boxes', 'medians']:
                bp[item][i].set_color(colors[i])
        for i in range(2):
            for item in ['whiskers', 'caps']:
                bp[item][2*i].set_color(colors[i])
                bp[item][2*i+1].set_color(colors[i])
        ax.tick_params(tick1On=False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_title('ROI # {} omission response for fix and jitter'.format(
            str(roi_id).zfill(4)))
        ax.set_ylabel('df/f')