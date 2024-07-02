#!/usr/bin/env python3

import numpy as np
from scipy.stats import sem
from scipy.interpolate import interp1d
from sklearn.linear_model import Ridge


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


# align joystick trajectory at given state.

def align_pos(neural_trials, state):
    interval = 1
    joystick_time = [neural_trials[str(i)]['joystick_time']
                    for i in range(len(neural_trials))]
    joystick_pos = [neural_trials[str(i)]['joystick_pos']
                    for i in range(len(neural_trials))]
    reward = np.array([not np.isnan(neural_trials[str(i)]['time_reward'][0])
              for i in range(len(neural_trials))])
    punish = np.array([not np.isnan(neural_trials[str(i)]['time_punish'][0])
              for i in range(len(neural_trials))])
    trial_types = np.array([neural_trials[str(i)]['trial_types']
              for i in range(len(neural_trials))])
    inter_time = []
    inter_pos = []
    for (pos, time) in zip(joystick_pos, joystick_time):
        interpolator = interp1d(time, pos, bounds_error=False)
        new_time = np.arange(np.min(time), np.max(time), interval)
        new_pos = interpolator(new_time)
        inter_time.append(new_time)
        inter_pos.append(new_pos)
    time_state = [
        neural_trials[str(i)][state][0] - neural_trials[str(i)]['vol_time'][0]
        for i in range(len(neural_trials))]
    trial_types = [trial_types[i]
                   for i in range(len(inter_time))
                   if not np.isnan(time_state)[i]]
    reward = np.array([reward[i] * 1
              for i in range(len(inter_time))
              if not np.isnan(time_state)[i]])
    punish = np.array([punish[i] * -1
              for i in range(len(inter_time))
              if not np.isnan(time_state)[i]])
    outcome = reward + punish
    zero_state = [np.argmin(np.abs(inter_time[i] - time_state[i]))
                  for i in range(len(inter_time))
                  if not np.isnan(time_state)[i]]
    align_data = [inter_pos[i]
                  for i in range(len(inter_pos))
                  if not np.isnan(time_state)[i]]
    align_time = [inter_time[i]
                  for i in range(len(inter_time))
                  if not np.isnan(time_state)[i]]
    align_time = [align_time[i] - align_time[i][zero_state[i]]
                  for i in range(len(align_time))]
    align_data = np.array(trim_seq(align_data, zero_state))
    align_time = np.array(trim_seq(align_time, zero_state))[0,:]
    return [align_data, align_time, trial_types, outcome]


# extract response around reward.

def get_reward_response(
        neural_trials,
        l_frames, r_frames
        ):
    # initialize list.
    neu_seq  = []
    neu_time = []
    reward = []
    # loop over trials.
    for trials in neural_trials.keys():
        # read trial data.
        fluo = neural_trials[trials]['dff']
        time = neural_trials[trials]['time']
        time_reward = neural_trials[str(trials)]['time_reward']
        # compute stimulus start point in ms.
        if not np.isnan(time_reward[0]):
            idx = np.argmin(np.abs(time - time_reward[0]))
            if idx > l_frames and idx < len(time)-r_frames:
                # signal response.
                f = fluo[:, idx-l_frames : idx+r_frames]
                f = np.expand_dims(f, axis=0)
                neu_seq.append(f)
                # signal time stamps.
                t = time[idx-l_frames : idx+r_frames] - time[idx]
                neu_time.append(t)
                # get outcome timestamps
                reward.append(time_reward - time_reward[0])
    # correct neuron time stamps centering at perturbation.
    neu_time_zero = [np.argmin(np.abs(nt)) for nt in neu_time]
    neu_time = trim_seq(neu_time, neu_time_zero)
    neu_seq = trim_seq(neu_seq, neu_time_zero)
    # concatenate results.
    neu_time  = [nt.reshape(1,-1) for nt in neu_time]
    neu_seq   = np.concatenate(neu_seq, axis=0)
    neu_time  = np.concatenate(neu_time, axis=0)
    # get mean time stamps.
    neu_time  = np.mean(neu_time, axis=0)
    # outcome period.
    reward = np.mean(reward, axis=0)
    return [neu_seq, neu_time, reward]


# extract response around punish.

def get_punish_response(
        neural_trials,
        l_frames, r_frames
        ):
    # initialize list.
    neu_seq  = []
    neu_time = []
    punish = []
    # loop over trials.
    for trials in neural_trials.keys():
        # read trial data.
        fluo = neural_trials[trials]['dff']
        time = neural_trials[trials]['time']
        time_punish = neural_trials[str(trials)]['time_punish']
        # compute stimulus start point in ms.
        if not np.isnan(time_punish[0]):
            idx = np.argmin(np.abs(time - time_punish[0]))
            if idx > l_frames and idx < len(time)-r_frames:
                # signal response.
                f = fluo[:, idx-l_frames : idx+r_frames]
                f = np.expand_dims(f, axis=0)
                neu_seq.append(f)
                # signal time stamps.
                t = time[idx-l_frames : idx+r_frames] - time[idx]
                neu_time.append(t)
                # get outcome timestamps
                punish.append(time_punish - time_punish[0])
    # correct neuron time stamps centering at perturbation.
    neu_time_zero = [np.argmin(np.abs(nt)) for nt in neu_time]
    neu_time = trim_seq(neu_time, neu_time_zero)
    neu_seq = trim_seq(neu_seq, neu_time_zero)
    # concatenate results.
    neu_time  = [nt.reshape(1,-1) for nt in neu_time]
    neu_seq   = np.concatenate(neu_seq, axis=0)
    neu_time  = np.concatenate(neu_time, axis=0)
    # get mean time stamps.
    neu_time  = np.mean(neu_time, axis=0)
    # outcome period.
    punish = np.mean(punish, axis=0)
    return [neu_seq, neu_time, punish]


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


# adjust layout for align trajectory.

def adjust_layout_trajectory(ax):
    ax.tick_params(tick1On=False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_ylabel('joystick deflection (deg)')
    ax.legend(loc='upper right')
    

class plotter_VIPTD_G8_beh:
    
    def __init__(
            self,
            neural_trials,
            ):
        self.neural_trials = neural_trials

    # outcome percentage.
    def outcome(self, ax):
        reward = np.array([not np.isnan(self.neural_trials[str(i)]['time_reward'][0])
                  for i in range(len(self.neural_trials))])
        punish = np.array([not np.isnan(self.neural_trials[str(i)]['time_punish'][0])
                  for i in range(len(self.neural_trials))])
        trial_types = np.array([self.neural_trials[str(i)]['trial_types']
                  for i in range(len(self.neural_trials))])
        reward_short = np.sum(reward[trial_types==1])
        punish_short = np.sum(punish[trial_types==1])
        reward_long  = np.sum(reward[trial_types==2])
        punish_long  = np.sum(punish[trial_types==2])
        ax.bar(
            0, reward_short/(reward_short+punish_short),
            bottom=0,
            edgecolor='white', width=0.25,
            color='#A4CB9E', label='reward')
        ax.bar(
            0, punish_short/(reward_short+punish_short),
            bottom=reward_short/(reward_short+punish_short),
            edgecolor='white', width=0.25,
            color='#EDA1A4', label='punish')
        ax.bar(
            1, reward_long/(reward_long+punish_long),
            bottom=0,
            edgecolor='white', width=0.25,
            color='#A4CB9E', label='reward')
        ax.bar(
            1, punish_long/(reward_long+punish_long),
            bottom=reward_long/(reward_long+punish_long),
            edgecolor='white', width=0.25,
            color='#EDA1A4', label='punish')
        ax.tick_params(tick1On=False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.grid(True)
        ax.set_xlabel('trial type')
        ax.set_ylabel('percentage')
        ax.set_xlim([-1,2])
        ax.set_xticks([0,1])
        ax.set_xticklabels(['short', 'long'])
        handles, labels = ax.get_legend_handles_labels()
        handles = [handles[0], handles[-1]]
        labels = [labels[0], labels[-1]]
        ax.legend(handles, labels, loc='upper right')
        ax.set_title('percentage of outcome')
    
    # trajectory aligned at reward.
    def align_pos_reward(self, ax):
        [align_data, align_time, _, _] = align_pos(self.neural_trials, 'time_reward')
        align_mean = np.mean(align_data, axis=0)
        align_sem  = sem(align_data, axis=0)
        upper = np.max(align_mean) + np.max(align_sem)
        lower = -0.1
        ax.plot(
            align_time,
            align_mean,
            color='dodgerblue',
            label='excitory_fix')
        ax.fill_between(
            align_time,
            align_mean - align_sem,
            align_mean + align_sem,
            color='dodgerblue',
            alpha=0.2)
        ax.axvline(0, color='grey', lw=2, label='reward', linestyle='--')
        adjust_layout_trajectory(ax)
        ax.set_xlabel('time since reward (ms)')
        ax.set_xlim([np.min(align_time), np.max(align_time)])
        ax.set_ylim([lower, upper + 0.1*(upper-lower)])
        ax.set_title(
            'reward aligned trajectories for {} trials'.format(
            align_data.shape[0]))
    
    # trajectory aligned at first stimuli.
    def align_pos_vis1(self, ax):
        [align_data, align_time, _, outcome] = align_pos(self.neural_trials, 'time_vis1')
        mean_reward = np.mean(align_data[outcome==1], axis=0)
        mean_punish = np.mean(align_data[outcome==-1], axis=0)
        sem_reward  = sem(align_data[outcome==1], axis=0)
        sem_punish  = sem(align_data[outcome==-1], axis=0)
        upper = np.max([mean_reward, mean_punish]) + np.max([sem_reward, sem_punish])
        lower = np.min([mean_reward, mean_punish]) - np.max([sem_reward, sem_punish])
        ax.plot(
            align_time,
            mean_reward,
            color='turquoise',
            label='1st press success')
        ax.fill_between(
            align_time,
            mean_reward - sem_reward,
            mean_reward + sem_reward,
            color='turquoise',
            alpha=0.2)
        ax.plot(
            align_time,
            mean_punish,
            color='coral',
            label='1st press failure')
        ax.fill_between(
            align_time,
            mean_punish - sem_punish,
            mean_punish + sem_punish,
            color='coral',
            alpha=0.2)
        ax.axvline(0, color='grey', lw=2, label='vis1', linestyle='--')
        adjust_layout_trajectory(ax)
        ax.set_xlabel('time since 1st stimuli (ms)')
        ax.set_xlim([np.min(align_time), np.max(align_time)])
        ax.set_ylim([lower, upper + 0.1*(upper-lower)])
        ax.set_title(
            '1st stimuli aligned trajectories for {} trials'.format(
            align_data.shape[0]))
    
    # trajectory aligned at second stimuli.
    def align_pos_vis2(self, ax):
        [align_data, align_time, _, outcome] = align_pos(self.neural_trials, 'time_vis2')
        mean_reward = np.mean(align_data[outcome==1], axis=0)
        mean_punish = np.mean(align_data[outcome==-1], axis=0)
        sem_reward  = sem(align_data[outcome==1], axis=0)
        sem_punish  = sem(align_data[outcome==-1], axis=0)
        upper = np.max([mean_reward, mean_punish]) + np.max([sem_reward, sem_punish])
        lower = np.min([mean_reward, mean_punish]) - np.max([sem_reward, sem_punish])
        ax.plot(
            align_time,
            mean_reward,
            color='turquoise',
            label='2nd press success')
        ax.fill_between(
            align_time,
            mean_reward - sem_reward,
            mean_reward + sem_reward,
            color='turquoise',
            alpha=0.2)
        ax.plot(
            align_time,
            mean_punish,
            color='coral',
            label='2nd press failure')
        ax.fill_between(
            align_time,
            mean_punish - sem_punish,
            mean_punish + sem_punish,
            color='coral',
            alpha=0.2)
        ax.axvline(0, color='grey', lw=2, label='vis2', linestyle='--')
        adjust_layout_trajectory(ax)
        ax.set_xlabel('time since 2nd stimuli (ms)')
        ax.set_xlim([np.min(align_time), np.max(align_time)])
        ax.set_ylim([lower, upper + 0.1*(upper-lower)])
        ax.set_title(
            '2nd stimuli aligned trajectories for {} trials'.format(
            align_data.shape[0]))


#%% ppc


# reward response.

def plot_ppc_reward(
        ax,
        neural_trials, labels
        ):
    l_frames = 15
    r_frames = 25
    [neu_seq, neu_time,
     reward] = get_reward_response(neural_trials, l_frames, r_frames)
    exc_all = neu_seq[:,labels==-1,:].reshape(-1, l_frames+r_frames)
    inh_all = neu_seq[:,labels==1,:].reshape(-1, l_frames+r_frames)
    mean_exc = np.mean(exc_all, axis=0)
    mean_inh = np.mean(inh_all, axis=0)
    sem_exc = sem(exc_all, axis=0)
    sem_inh = sem(inh_all, axis=0)
    upper = np.max([mean_exc, mean_inh]) + np.max([sem_exc, sem_inh])
    lower = np.min([mean_exc, mean_inh]) - np.max([sem_exc, sem_inh])
    ax.fill_between(
        reward,
        lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
        color='gold', alpha=0.25, step='mid', label='reward')
    ax.plot(
        neu_time,
        mean_exc,
        color='turquoise', label='exc')
    ax.fill_between(
        neu_time,
        mean_exc - sem_exc,
        mean_exc + sem_exc,
        color='turquoise', alpha=0.2)
    ax.plot(
        neu_time,
        mean_inh,
        color='coral', label='inh')
    ax.fill_between(
        neu_time,
        mean_inh - sem_inh,
        mean_inh + sem_inh,
        color='coral', alpha=0.2)
    adjust_layout(ax)
    ax.set_xlim([np.min(neu_time), np.max(neu_time)])
    ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
    ax.set_xlabel('time since reward (ms)')
    ax.set_title('response to reward')


# reward response.

def plot_ppc_punish(
        ax,
        neural_trials, labels
        ):
    l_frames = 5
    r_frames = 100
    [neu_seq, neu_time,
     punish] = get_punish_response(neural_trials, l_frames, r_frames)
    exc_all = neu_seq[:,labels==-1,:].reshape(-1, l_frames+r_frames)
    inh_all = neu_seq[:,labels==1,:].reshape(-1, l_frames+r_frames)
    mean_exc = np.mean(exc_all, axis=0)
    mean_inh = np.mean(inh_all, axis=0)
    sem_exc = sem(exc_all, axis=0)
    sem_inh = sem(inh_all, axis=0)
    upper = np.max([mean_exc, mean_inh]) + np.max([sem_exc, sem_inh])
    lower = np.min([mean_exc, mean_inh]) - np.max([sem_exc, sem_inh])
    ax.fill_between(
        punish,
        lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
        color='gold', alpha=0.25, step='mid', label='punish')
    ax.plot(
        neu_time,
        mean_exc,
        color='turquoise', label='exc')
    ax.fill_between(
        neu_time,
        mean_exc - sem_exc,
        mean_exc + sem_exc,
        color='turquoise', alpha=0.2)
    ax.plot(
        neu_time,
        mean_inh,
        color='coral', label='inh')
    ax.fill_between(
        neu_time,
        mean_inh - sem_inh,
        mean_inh + sem_inh,
        color='coral', alpha=0.2)
    adjust_layout(ax)
    ax.set_xlim([np.min(neu_time), np.max(neu_time)])
    ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
    ax.set_xlabel('time since punish (ms)')
    ax.set_title('response to punish')
    

# ROI reward response.

def plot_ppc_roi_reward(
        ax, roi_id,
        neural_trials, labels,
        ):
    l_frames = 15
    r_frames = 25
    [neu_seq, neu_time,
     reward] = get_reward_response(neural_trials, l_frames, r_frames)
    neu_all = neu_seq[:,roi_id,:].reshape(-1, l_frames+r_frames)
    neu_mean = np.mean(neu_all, axis=0)
    neu_sem = sem(neu_all, axis=0)
    upper = np.max(neu_mean) + np.max(neu_sem)
    lower = np.min(neu_mean) - np.max(neu_sem)
    if labels[roi_id]==-1:
        color = 'seagreen'
        label = 'excitory'
    if labels[roi_id]==0:
        color = 'dodgerblue'
        label = 'unsure'
    if labels[roi_id]==1:
        color = 'coral'
        label = 'inhibitory'
    ax.fill_between(
        reward,
        lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
        color='gold', alpha=0.25, step='mid', label='reward')
    ax.plot(
        neu_time,
        neu_mean,
        color=color, label=label)
    ax.fill_between(
        neu_time,
        neu_mean - neu_sem,
        neu_mean + neu_sem,
        color=color, alpha=0.2)
    adjust_layout(ax)
    ax.set_xlim([np.min(neu_time), np.max(neu_time)])
    ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
    ax.set_xlabel('time since reward (ms)')
    ax.set_title('ROI # {} response to reward'.format(str(roi_id).zfill(4)))


# ROI reward response.

def plot_ppc_roi_punish(
        ax, roi_id,
        neural_trials, labels
        ):
    l_frames = 5
    r_frames = 100
    [neu_seq, neu_time,
     punish] = get_punish_response(neural_trials, l_frames, r_frames)
    neu_all = neu_seq[:,roi_id,:].reshape(-1, l_frames+r_frames)
    neu_mean = np.mean(neu_all, axis=0)
    neu_sem = sem(neu_all, axis=0)
    upper = np.max(neu_mean) + np.max(neu_sem)
    lower = np.min(neu_mean) - np.max(neu_sem)
    if labels[roi_id]==-1:
        color = 'seagreen'
        label = 'excitory'
    if labels[roi_id]==0:
        color = 'dodgerblue'
        label = 'unsure'
    if labels[roi_id]==1:
        color = 'coral'
        label = 'inhibitory'
    ax.fill_between(
        punish,
        lower - 0.1*(upper-lower), upper + 0.1*(upper-lower),
        color='gold', alpha=0.25, step='mid', label='punish')
    ax.plot(
        neu_time,
        neu_mean,
        color=color, label=label)
    ax.fill_between(
        neu_time,
        neu_mean - neu_sem,
        neu_mean + neu_sem,
        color=color, alpha=0.2)
    adjust_layout(ax)
    ax.set_xlim([np.min(neu_time), np.max(neu_time)])
    ax.set_ylim([lower - 0.1*(upper-lower), upper + 0.1*(upper-lower)])
    ax.set_xlabel('time since punish (ms)')
    ax.set_title('ROI # {} response to punish'.format(str(roi_id).zfill(4)))
    

# single trial behavior decoding with outcome.

def plot_ppc_roi_beh_decode(
        ax, roi_id,
        neural_trials, labels
        ):
    dff = [neural_trials[str(i)]['dff'][roi_id,:]
           for i in range(len(neural_trials))]
    time = [neural_trials[str(i)]['time'] - neural_trials[
        str(i)]['time'][0] for i in range(len(neural_trials))]
    joystick_pos = [neural_trials[str(i)]['joystick_pos']
           for i in range(len(neural_trials))]
    joystick_time = [neural_trials[str(i)]['joystick_time']
           for i in range(len(neural_trials))]
    time_reward = [neural_trials[str(i)]['time_reward'][0] - neural_trials[
        str(i)]['time'][0] for i in range(len(neural_trials))]
    time_punish = [neural_trials[str(i)]['time_punish'][0] - neural_trials[
        str(i)]['time'][0] for i in range(len(neural_trials))]
    align_pos = []
    for js_pos, js_time, t in zip(joystick_pos, joystick_time, time):
        interpolator = interp1d(js_time, js_pos, bounds_error=False, fill_value=0)
        align_pos.append(interpolator(t))
    dff_reward = [dff[i] for i in range(len(dff))
                  if not np.isnan(time_reward[i])]
    pos_reward = [align_pos[i] for i in range(len(dff))
                  if not np.isnan(time_reward[i])]
    dff_punish = [dff[i] for i in range(len(dff))
                  if not np.isnan(time_punish[i])]
    pos_punish = [align_pos[i] for i in range(len(dff))
                  if not np.isnan(time_punish[i])]
    results_reward = []
    results_punish = []
    for i in range(len(dff_reward)):
        x = dff_reward[i].reshape(-1, 1)
        y = pos_reward[i].reshape(-1, 1)
        model = Ridge().fit(x, y)
        results_reward.append(model.score(x, y))
    for i in range(len(dff_punish)):
        x = dff_punish[i].reshape(-1, 1)
        y = pos_punish[i].reshape(-1, 1)
        model = Ridge().fit(x, y)
        results_punish.append(model.score(x, y))
    if labels[roi_id]==-1:
        color = 'seagreen'
    if labels[roi_id]==0:
        color = 'dodgerblue'
    if labels[roi_id]==1:
        color = 'coral'
    vp = ax.violinplot(
        [results_reward, results_punish],
        showmeans=False, showmedians=True,
        positions=[0,1])
    for i in range(2):
        vp['bodies'][i].set_facecolor(color)
        vp['bodies'][i].set_edgecolor(color)
        vp['bodies'][i].set_alpha(0.2)
    for p in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
        vp[p].set_edgecolor(color)
    ax.set_xlabel('outcome')
    ax.set_xticks([0,1])
    ax.set_xticklabels(['reward', 'punish'])
    ax.set_ylabel('decoding $R^2$')
    ax.set_title('single trial decoding performance')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(tick1On=False)

    
#%% crbl
