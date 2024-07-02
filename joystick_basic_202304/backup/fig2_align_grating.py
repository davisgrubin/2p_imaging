#!/usr/bin/env python3

import numpy as np
from scipy.stats import mode
from scipy.stats import sem
from matplotlib.colors import LinearSegmentedColormap


#%% utils


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


# extract response around 1st grating.

def get_1st_response(
        neural_trials,
        l_frames, r_frames
        ):
    # initialize list.
    neu_seq  = []
    neu_time = []
    stim_vol  = []
    stim_time = []
    outcome = []
    # loop over trials.
    for trials in neural_trials.keys():
        # read trial data.
        fluo = neural_trials[trials]['dff']
        time = neural_trials[trials]['time']
        vol_stim = neural_trials[str(trials)]['vol_stim']
        vol_time = neural_trials[str(trials)]['vol_time']
        time_vis1 = neural_trials[str(trials)]['time_vis1']
        time_punish = neural_trials[str(trials)]['time_punish']
        time_reward = neural_trials[str(trials)]['time_reward']
        if not np.isnan(time_punish[0]):
            trial_outcome = -1
        if not np.isnan(time_reward[0]):
            trial_outcome = 1
        # compute stimulus start point in ms.
        if not np.isnan(time_vis1[0]):
            idx = np.argmin(np.abs(time - time_vis1[0]))
            if idx > l_frames and idx < len(time)-r_frames:
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
                # outcome.
                outcome.append(trial_outcome)
    # correct voltage recordings centering at perturbation.
    stim_time_zero = [np.argmin(np.abs(st)) for st in stim_time]
    stim_time = trim_seq(stim_time, stim_time_zero)
    stim_vol = trim_seq(stim_vol, stim_time_zero)
    # correct neuron time stamps centering at perturbation.
    neu_time_zero = [np.argmin(np.abs(nt)) for nt in neu_time]
    neu_time = trim_seq(neu_time, neu_time_zero)
    neu_seq = trim_seq(neu_seq, neu_time_zero)
    # concatenate results.
    neu_time  = [nt.reshape(1,-1) for nt in neu_time]
    stim_time = [st.reshape(1,-1) for st in stim_time]
    stim_vol  = [sv.reshape(1,-1) for sv in stim_vol]
    neu_seq   = np.concatenate(neu_seq, axis=0)
    neu_time  = np.concatenate(neu_time, axis=0)
    stim_vol  = np.concatenate(stim_vol, axis=0)
    stim_time = np.concatenate(stim_time, axis=0)
    outcome   = np.array(outcome)
    # get mean time stamps.
    neu_time  = np.mean(neu_time, axis=0)
    stim_time = np.mean(stim_time, axis=0)
    # get mode and mean stimulus.
    stim_vol, _ = mode(stim_vol, axis=0)
    return [neu_seq, neu_time, stim_vol, stim_time, outcome]


# extract response around 2nd grating.

def get_2nd_response(
        neural_trials,
        l_frames, r_frames
        ):
    # initialize list.
    neu_seq  = []
    neu_time = []
    stim_vol  = []
    stim_time = []
    outcome = []
    # loop over trials.
    for trials in neural_trials.keys():
        # read trial data.
        fluo = neural_trials[trials]['dff']
        time = neural_trials[trials]['time']
        vol_stim = neural_trials[str(trials)]['vol_stim']
        vol_time = neural_trials[str(trials)]['vol_time']
        time_vis2 = neural_trials[str(trials)]['time_vis2']
        time_punish = neural_trials[str(trials)]['time_punish']
        time_reward = neural_trials[str(trials)]['time_reward']
        if not np.isnan(time_punish[0]):
            trial_outcome = -1
        if not np.isnan(time_reward[0]):
            trial_outcome = 1
        # compute stimulus start point in ms.
        if not np.isnan(time_vis2[0]):
            idx = np.argmin(np.abs(time - time_vis2[0]))
            if idx > l_frames and idx < len(time)-r_frames:
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
                # outcome.
                outcome.append(trial_outcome)
    # correct voltage recordings centering at perturbation.
    stim_time_zero = [np.argmin(np.abs(st)) for st in stim_time]
    stim_time = trim_seq(stim_time, stim_time_zero)
    stim_vol = trim_seq(stim_vol, stim_time_zero)
    # correct neuron time stamps centering at perturbation.
    neu_time_zero = [np.argmin(np.abs(nt)) for nt in neu_time]
    neu_time = trim_seq(neu_time, neu_time_zero)
    neu_seq = trim_seq(neu_seq, neu_time_zero)
    # concatenate results.
    neu_time  = [nt.reshape(1,-1) for nt in neu_time]
    stim_time = [st.reshape(1,-1) for st in stim_time]
    stim_vol  = [sv.reshape(1,-1) for sv in stim_vol]
    neu_seq   = np.concatenate(neu_seq, axis=0)
    neu_time  = np.concatenate(neu_time, axis=0)
    stim_vol  = np.concatenate(stim_vol, axis=0)
    stim_time = np.concatenate(stim_time, axis=0)
    outcome   = np.array(outcome)
    # get mean time stamps.
    neu_time  = np.mean(neu_time, axis=0)
    stim_time = np.mean(stim_time, axis=0)
    # get mode and mean stimulus.
    stim_vol, _ = mode(stim_vol, axis=0)
    return [neu_seq, neu_time, stim_vol, stim_time, outcome]


# adjust layout for grating average.

def adjust_layout(ax):
    ax.tick_params(axis='y', tick1On=False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
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
    
    
#%% ppc


# first grating alignment response for excitory with outcome.

def plot_ppc_exc_1st_grating_mean_outcome(
        ax,
        neural_trials, labels
        ):
    l_frames = 5
    r_frames = 50
    [neu_seq, neu_time,
     stim_vol, stim_time,
     outcome] = get_1st_response(neural_trials, l_frames, r_frames)
    exc_all = neu_seq[:,labels==-1,:]
    neu_reward = exc_all[outcome==1,:,:].reshape(-1, l_frames+r_frames)
    neu_punish = exc_all[outcome==-1,:,:].reshape(-1, l_frames+r_frames)
    mean_reward = np.mean(neu_reward, axis=0)
    mean_punish = np.mean(neu_punish, axis=0)
    sem_reward = sem(neu_reward, axis=0)
    sem_punish = sem(neu_punish, axis=0)
    upper = np.max([mean_reward, mean_punish]) + np.max([sem_reward, sem_punish])
    lower = np.min([mean_reward, mean_punish]) - np.max([sem_reward, sem_punish])
    ax.fill_between(
        stim_time,
        lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
        where=(stim_vol==1),
        color='silver', step='mid', label='stim')
    ax.plot(
        neu_time,
        mean_reward,
        color='turquoise', label='reward')
    ax.fill_between(
        neu_time,
        mean_reward - sem_reward,
        mean_reward + sem_reward,
        color='turquoise', alpha=0.2)
    ax.plot(
        neu_time,
        mean_punish,
        color='violet', label='punish')
    ax.fill_between(
        neu_time,
        mean_punish - sem_punish,
        mean_punish + sem_punish,
        color='violet', alpha=0.2)
    adjust_layout(ax)
    ax.set_xlim([np.min(neu_time), np.max(neu_time)])
    ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
    ax.set_xlabel('time since the 1st grating (ms)')
    ax.set_title('response to the 1st grating for excitory')


# first grating alignment response for inhibitory.

def plot_ppc_inh_1st_grating_mean_outcome(
        ax,
        neural_trials, labels
        ):
    l_frames = 5
    r_frames = 50
    [neu_seq, neu_time,
     stim_vol, stim_time,
     outcome] = get_1st_response(neural_trials, l_frames, r_frames)
    inh_all = neu_seq[:,labels==1,:]
    neu_reward = inh_all[outcome==1,:,:].reshape(-1, l_frames+r_frames)
    neu_punish = inh_all[outcome==-1,:,:].reshape(-1, l_frames+r_frames)
    mean_reward = np.mean(neu_reward, axis=0)
    mean_punish = np.mean(neu_punish, axis=0)
    sem_reward = sem(neu_reward, axis=0)
    sem_punish = sem(neu_punish, axis=0)
    upper = np.max([mean_reward, mean_punish]) + np.max([sem_reward, sem_punish])
    lower = np.min([mean_reward, mean_punish]) - np.max([sem_reward, sem_punish])
    ax.fill_between(
        stim_time,
        lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
        where=(stim_vol==1),
        color='silver', step='mid', label='stim')
    ax.plot(
        neu_time,
        mean_reward,
        color='orangered', label='reward')
    ax.fill_between(
        neu_time,
        mean_reward - sem_reward,
        mean_reward + sem_reward,
        color='orangered', alpha=0.2)
    ax.plot(
        neu_time,
        mean_punish,
        color='violet', label='punish')
    ax.fill_between(
        neu_time,
        mean_punish - sem_punish,
        mean_punish + sem_punish,
        color='violet', alpha=0.2)
    adjust_layout(ax)
    ax.set_xlim([np.min(neu_time), np.max(neu_time)])
    ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
    ax.set_xlabel('time since the 1st grating (ms)')
    ax.set_title('response to the 1st grating for inhibitory')
    
    
# second grating alignment response for excitory with outcome.

def plot_ppc_exc_2nd_grating_mean_outcome(
        ax,
        neural_trials, labels
        ):
    l_frames = 25
    r_frames = 40
    [neu_seq, neu_time,
     stim_vol, stim_time,
     outcome] = get_2nd_response(neural_trials, l_frames, r_frames)
    exc_all = neu_seq[:,labels==-1,:]
    neu_reward = exc_all[outcome==1,:,:].reshape(-1, l_frames+r_frames)
    neu_punish = exc_all[outcome==-1,:,:].reshape(-1, l_frames+r_frames)
    mean_reward = np.mean(neu_reward, axis=0)
    mean_punish = np.mean(neu_punish, axis=0)
    sem_reward = sem(neu_reward, axis=0)
    sem_punish = sem(neu_punish, axis=0)
    upper = np.max([mean_reward, mean_punish]) + np.max([sem_reward, sem_punish])
    lower = np.min([mean_reward, mean_punish]) - np.max([sem_reward, sem_punish])
    ax.fill_between(
        stim_time,
        lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
        where=(stim_vol==1),
        color='silver', step='mid', label='stim')
    ax.plot(
        neu_time,
        mean_reward,
        color='turquoise', label='reward')
    ax.fill_between(
        neu_time,
        mean_reward - sem_reward,
        mean_reward + sem_reward,
        color='turquoise', alpha=0.2)
    ax.plot(
        neu_time,
        mean_punish,
        color='violet', label='punish')
    ax.fill_between(
        neu_time,
        mean_punish - sem_punish,
        mean_punish + sem_punish,
        color='violet', alpha=0.2)
    adjust_layout(ax)
    ax.set_xlim([np.min(neu_time), np.max(neu_time)])
    ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
    ax.set_xlabel('time since the 2nd grating (ms)')
    ax.set_title('response to the 2nd grating for excitory')


# second grating alignment response for inhibitory.

def plot_ppc_inh_2nd_grating_mean_outcome(
        ax,
        neural_trials, labels
        ):
    l_frames = 25
    r_frames = 40
    [neu_seq, neu_time,
     stim_vol, stim_time,
     outcome] = get_2nd_response(neural_trials, l_frames, r_frames)
    inh_all = neu_seq[:,labels==1,:]
    neu_reward = inh_all[outcome==1,:,:].reshape(-1, l_frames+r_frames)
    neu_punish = inh_all[outcome==-1,:,:].reshape(-1, l_frames+r_frames)
    mean_reward = np.mean(neu_reward, axis=0)
    mean_punish = np.mean(neu_punish, axis=0)
    sem_reward = sem(neu_reward, axis=0)
    sem_punish = sem(neu_punish, axis=0)
    upper = np.max([mean_reward, mean_punish]) + np.max([sem_reward, sem_punish])
    lower = np.min([mean_reward, mean_punish]) - np.max([sem_reward, sem_punish])
    ax.fill_between(
        stim_time,
        lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
        where=(stim_vol==1),
        color='silver', step='mid', label='stim')
    ax.plot(
        neu_time,
        mean_reward,
        color='orangered', label='reward')
    ax.fill_between(
        neu_time,
        mean_reward - sem_reward,
        mean_reward + sem_reward,
        color='orangered', alpha=0.2)
    ax.plot(
        neu_time,
        mean_punish,
        color='violet', label='punish')
    ax.fill_between(
        neu_time,
        mean_punish - sem_punish,
        mean_punish + sem_punish,
        color='violet', alpha=0.2)
    adjust_layout(ax)
    ax.set_xlim([np.min(neu_time), np.max(neu_time)])
    ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
    ax.set_xlabel('time since the 2nd grating (ms)')
    ax.set_title('response to the 2nd grating for inhibitory')
    
    
# plot reward response heat map around second grating with average across trials.
    
def plot_ppc_2nd_grating_heatmap_reward_neuron(
        ax,
        neural_trials, labels
        ):
    l_frames = 25
    r_frames = 40
    [neu_seq, neu_time,
     stim_vol, stim_time,
     outcome] = get_2nd_response(neural_trials, l_frames, r_frames)
    mean = np.mean(neu_seq[outcome==1,:,:], axis=0)
    for i in range(mean.shape[0]):
        mean[i,:] = norm01(mean[i,:])
    zero = np.where(neu_time==0)[0][0]
    sort_idx = mean[:, zero].reshape(-1).argsort()
    sort_mean = mean[sort_idx,:]
    cmap = LinearSegmentedColormap.from_list(
        'reward', ['white','seagreen', 'black'])
    im_fix = ax.imshow(
        sort_mean, interpolation='nearest', aspect='auto', cmap=cmap)
    adjust_layout_heatmap(ax)
    ax.set_ylabel('sorted neuron id')
    ax.axvline(zero, color='red', lw=1, label='2nd grating start', linestyle='--')
    ax.set_xticks([0, zero, len(neu_time)])
    ax.set_xticklabels([int(neu_time[0]), 0, int(neu_time[-1])])
    cbar = ax.figure.colorbar(im_fix, ax=ax, ticks=[0.2,0.8])
    cbar.ax.set_ylabel('normalized response', rotation=-90, va="bottom")
    cbar.ax.set_yticklabels(['0.2', '0.8'])
    ax.set_title('2nd grating response heatmap with reward for neurons')
    
    
# plot punish response heat map around second grating with average across trials.
    
def plot_ppc_2nd_grating_heatmap_punish_neuron(
        ax,
        neural_trials, labels
        ):
    l_frames = 25
    r_frames = 40
    [neu_seq, neu_time,
     stim_vol, stim_time,
     outcome] = get_2nd_response(neural_trials, l_frames, r_frames)
    mean = np.mean(neu_seq[outcome==-1,:,:], axis=0)
    for i in range(mean.shape[0]):
        mean[i,:] = norm01(mean[i,:])
    zero = np.where(neu_time==0)[0][0]
    sort_idx = mean[:, zero].reshape(-1).argsort()
    sort_mean = mean[sort_idx,:]
    cmap = LinearSegmentedColormap.from_list(
        'punish', ['white','coral', 'black'])
    im_fix = ax.imshow(
        sort_mean, interpolation='nearest', aspect='auto', cmap=cmap)
    adjust_layout_heatmap(ax)
    ax.set_ylabel('sorted neuron id')
    ax.axvline(zero, color='red', lw=1, label='2nd grating start', linestyle='--')
    ax.set_xticks([0, zero, len(neu_time)])
    ax.set_xticklabels([int(neu_time[0]), 0, int(neu_time[-1])])
    cbar = ax.figure.colorbar(im_fix, ax=ax, ticks=[0.2,0.8])
    cbar.ax.set_ylabel('normalized response', rotation=-90, va="bottom")
    cbar.ax.set_yticklabels(['0.2', '0.8'])
    ax.set_title('2nd grating response heatmap with punish for neurons')


# plot reward response heat map around second grating with average across neurons.
    
def plot_ppc_2nd_grating_heatmap_reward_trial(
        ax,
        neural_trials, labels
        ):
    l_frames = 25
    r_frames = 40
    [neu_seq, neu_time,
     stim_vol, stim_time,
     outcome] = get_2nd_response(neural_trials, l_frames, r_frames)
    mean = np.mean(neu_seq[outcome==1,:,:], axis=1)
    for i in range(mean.shape[0]):
        mean[i,:] = norm01(mean[i,:])
    zero = np.where(neu_time==0)[0][0]
    cmap = LinearSegmentedColormap.from_list(
        'reward', ['white','seagreen', 'black'])
    im_fix = ax.imshow(
        mean, interpolation='nearest', aspect='auto', cmap=cmap)
    adjust_layout_heatmap(ax)
    ax.set_ylabel('trial id')
    ax.axvline(zero, color='red', lw=1, label='2nd grating start', linestyle='--')
    ax.set_xticks([0, zero, len(neu_time)])
    ax.set_xticklabels([int(neu_time[0]), 0, int(neu_time[-1])])
    cbar = ax.figure.colorbar(im_fix, ax=ax, ticks=[0.2,0.8])
    cbar.ax.set_ylabel('normalized response', rotation=-90, va="bottom")
    cbar.ax.set_yticklabels(['0.2', '0.8'])
    ax.set_title('2nd grating response heatmap with reward for trials')
    
    
# plot punish response heat map around second grating with average across neurons.
    
def plot_ppc_2nd_grating_heatmap_punish_trial(
        ax,
        neural_trials, labels
        ):
    l_frames = 25
    r_frames = 40
    [neu_seq, neu_time,
     stim_vol, stim_time,
     outcome] = get_2nd_response(neural_trials, l_frames, r_frames)
    mean = np.mean(neu_seq[outcome==-1,:,:], axis=0)
    for i in range(mean.shape[0]):
        mean[i,:] = norm01(mean[i,:])
    zero = np.where(neu_time==0)[0][0]
    cmap = LinearSegmentedColormap.from_list(
        'punish', ['white','coral', 'black'])
    im_fix = ax.imshow(
        mean, interpolation='nearest', aspect='auto', cmap=cmap)
    adjust_layout_heatmap(ax)
    ax.set_ylabel('trial id')
    ax.axvline(zero, color='red', lw=1, label='2nd grating start', linestyle='--')
    ax.set_xticks([0, zero, len(neu_time)])
    ax.set_xticklabels([int(neu_time[0]), 0, int(neu_time[-1])])
    cbar = ax.figure.colorbar(im_fix, ax=ax, ticks=[0.2,0.8])
    cbar.ax.set_ylabel('normalized response', rotation=-90, va="bottom")
    cbar.ax.set_yticklabels(['0.2', '0.8'])
    ax.set_title('2nd grating response heatmap with punish for trials')


# ROI first grating alignment response with outcome.

def plot_ppc_roi_1st_grating_mean_outcome(
        ax, roi_id,
        neural_trials
        ):
    l_frames = 5
    r_frames = 50
    [neu_seq, neu_time,
     stim_vol, stim_time,
     outcome] = get_1st_response(neural_trials, l_frames, r_frames)
    neu_reward = neu_seq[outcome==1,roi_id,:].reshape(-1, l_frames+r_frames)
    neu_punish = neu_seq[outcome==-1,roi_id,:].reshape(-1, l_frames+r_frames)
    mean_reward = np.mean(neu_reward, axis=0)
    mean_punish = np.mean(neu_punish, axis=0)
    sem_reward = sem(neu_reward, axis=0)
    sem_punish = sem(neu_punish, axis=0)
    upper = np.max([mean_reward, mean_punish]) + np.max([sem_reward, sem_punish])
    lower = np.min([mean_reward, mean_punish]) - np.max([sem_reward, sem_punish])
    ax.fill_between(
        stim_time,
        lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
        where=(stim_vol==1),
        color='silver', step='mid', label='stim')
    ax.plot(
        neu_time,
        mean_reward,
        color='turquoise', label='reward')
    ax.fill_between(
        neu_time,
        mean_reward - sem_reward,
        mean_reward + sem_reward,
        color='turquoise', alpha=0.2)
    ax.plot(
        neu_time,
        mean_punish,
        color='violet', label='punish')
    ax.fill_between(
        neu_time,
        mean_punish - sem_punish,
        mean_punish + sem_punish,
        color='violet', alpha=0.2)
    adjust_layout(ax)
    ax.set_xlim([np.min(neu_time), np.max(neu_time)])
    ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
    ax.set_xlabel('time since the 1st grating (ms)')
    ax.set_title('ROI # {} response to the 1st grating'.format(
        str(roi_id).zfill(4)))
    
    
# ROI second grating alignment response with outcome.

def plot_ppc_roi_2nd_grating_mean_outcome(
        ax, roi_id,
        neural_trials
        ):
    l_frames = 25
    r_frames = 40
    [neu_seq, neu_time,
     stim_vol, stim_time,
     outcome] = get_2nd_response(neural_trials, l_frames, r_frames)
    neu_reward = neu_seq[outcome==1,roi_id,:].reshape(-1, l_frames+r_frames)
    neu_punish = neu_seq[outcome==-1,roi_id,:].reshape(-1, l_frames+r_frames)
    mean_reward = np.mean(neu_reward, axis=0)
    mean_punish = np.mean(neu_punish, axis=0)
    sem_reward = sem(neu_reward, axis=0)
    sem_punish = sem(neu_punish, axis=0)
    upper = np.max([mean_reward, mean_punish]) + np.max([sem_reward, sem_punish])
    lower = np.min([mean_reward, mean_punish]) - np.max([sem_reward, sem_punish])
    ax.fill_between(
        stim_time,
        lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
        where=(stim_vol==1),
        color='silver', step='mid', label='stim')
    ax.plot(
        neu_time,
        mean_reward,
        color='turquoise', label='reward')
    ax.fill_between(
        neu_time,
        mean_reward - sem_reward,
        mean_reward + sem_reward,
        color='turquoise', alpha=0.2)
    ax.plot(
        neu_time,
        mean_punish,
        color='violet', label='punish')
    ax.fill_between(
        neu_time,
        mean_punish - sem_punish,
        mean_punish + sem_punish,
        color='violet', alpha=0.2)
    adjust_layout(ax)
    ax.set_xlim([np.min(neu_time), np.max(neu_time)])
    ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
    ax.set_xlabel('time since the 2nd grating (ms)')
    ax.set_title('ROI # {} response to the 2nd grating'.format(
        str(roi_id).zfill(4)))
    
    
# ROI single trial first grating alignment response heatmap with reward.

def plot_ppc_roi_1st_grating_reward_hearmap(
        ax, roi_id,
        neural_trials, labels
        ):
    l_frames = 5
    r_frames = 50
    [neu_seq, neu_time,
     stim_vol, stim_time,
     outcome] = get_1st_response(neural_trials, l_frames, r_frames)
    neu_seq = neu_seq[outcome==1,roi_id,:]
    for i in range(neu_seq.shape[0]):
        neu_seq[i,:] = norm01(neu_seq[i,:])
    zero = np.where(neu_time==0)[0][0]
    if labels[roi_id] == -1:
        cmap = LinearSegmentedColormap.from_list(
            'excitory', ['white', 'seagreen', 'black'])
    if labels[roi_id] == 0:
        cmap = LinearSegmentedColormap.from_list(
            'unsure', ['white', 'dodgerblue', 'black'])
    if labels[roi_id] == 1:
        cmap = LinearSegmentedColormap.from_list(
            'inhibitory', ['white', 'coral', 'black'])
    im_fix = ax.imshow(
        neu_seq, interpolation='nearest', aspect='auto', cmap=cmap)
    adjust_layout_heatmap(ax)
    ax.set_ylabel('trial id')
    ax.axvline(zero, color='red', lw=1, label='1st grating start', linestyle='--')
    ax.set_xticks([0, zero, len(neu_time)])
    ax.set_xticklabels([int(neu_time[0]), 0, int(neu_time[-1])])
    cbar = ax.figure.colorbar(im_fix, ax=ax, ticks=[0.2,0.8])
    cbar.ax.set_ylabel('normalized response', rotation=-90, va="bottom")
    cbar.ax.set_yticklabels(['0.2', '0.8'])
    ax.set_title('1st grating single trial response heatmap with reward')
    

# ROI single trial first grating alignment response heatmap with reward.

def plot_ppc_roi_2nd_grating_reward_hearmap(
        ax, roi_id,
        neural_trials, labels
        ):
    l_frames = 25
    r_frames = 40
    [neu_seq, neu_time,
     stim_vol, stim_time,
     outcome] = get_2nd_response(neural_trials, l_frames, r_frames)
    neu_seq = neu_seq[outcome==1,roi_id,:]
    for i in range(neu_seq.shape[0]):
        neu_seq[i,:] = norm01(neu_seq[i,:])
    zero = np.where(neu_time==0)[0][0]
    if labels[roi_id] == -1:
        cmap = LinearSegmentedColormap.from_list(
            'excitory', ['white', 'seagreen', 'black'])
    if labels[roi_id] == 0:
        cmap = LinearSegmentedColormap.from_list(
            'unsure', ['white', 'dodgerblue', 'black'])
    if labels[roi_id] == 1:
        cmap = LinearSegmentedColormap.from_list(
            'inhibitory', ['white', 'coral', 'black'])
    im_fix = ax.imshow(
        neu_seq, interpolation='nearest', aspect='auto', cmap=cmap)
    adjust_layout_heatmap(ax)
    ax.set_ylabel('trial id')
    ax.axvline(zero, color='red', lw=1, label='2nd grating start', linestyle='--')
    ax.set_xticks([0, zero, len(neu_time)])
    ax.set_xticklabels([int(neu_time[0]), 0, int(neu_time[-1])])
    cbar = ax.figure.colorbar(im_fix, ax=ax, ticks=[0.2,0.8])
    cbar.ax.set_ylabel('normalized response', rotation=-90, va="bottom")
    cbar.ax.set_yticklabels(['0.2', '0.8'])
    ax.set_title('2nd grating single trial response heatmap with reward')
    
    
    
    
    
#%% crbl
