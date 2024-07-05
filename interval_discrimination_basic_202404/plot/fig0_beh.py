#!/usr/bin/env python3

import numpy as np
from scipy.stats import sem


# organize outcome of behavior data.
def get_beh_outcomes(neural_trials):
    out_reward = 1 - np.isnan([neural_trials[i]['trial_reward'][0]
                                 for i in neural_trials.keys()])
    out_punish = 1 - np.isnan([neural_trials[i]['trial_punish'][0]
                                 for i in neural_trials.keys()])
    out_wrinit = 1 - np.isnan([neural_trials[i]['trial_wrinit'][0]
                                 for i in neural_trials.keys()])
    out_notcho = 1 - np.isnan([neural_trials[i]['trial_notcho'][0]
                                 for i in neural_trials.keys()])
    outcomes = 1*out_reward + 2*out_punish + 3*out_wrinit + 4*out_notcho
    return outcomes


# compute perturbation start time.
def get_perturb_time(neural_trials):
    perturb_time = []
    for trial in neural_trials.keys():
        trial_stim_seq = neural_trials[trial]['trial_stim_seq']
        trial_stim_start = neural_trials[trial]['trial_stim_start']
        if not np.isnan(trial_stim_seq[0,0]) and trial_stim_seq.shape[1]>=3:
            perturb_time.append(trial_stim_seq[1,2] - trial_stim_start)
    perturb_time = np.median(perturb_time)
    return perturb_time


# organize reaction.
def get_reaction(neural_trials):
    reaction = [neural_trials[i]['trial_reaction'] for i in neural_trials.keys()]
    reaction = np.concatenate(reaction, axis=1)
    stim_start = [np.array(neural_trials[i]['trial_stim_start']).reshape(-1)
                  for i in neural_trials.keys()]
    stim_start = np.concatenate(stim_start)
    trial_types = [neural_trials[i]['trial_types'] for i in neural_trials.keys()]
    trial_types = np.array(trial_types).reshape(1,-1)
    reaction = np.concatenate([reaction, trial_types], axis=0)
    reaction[0,:] -= stim_start
    non_nan = (1-np.isnan(reaction[0,:])).astype('bool')
    reaction = reaction[:,non_nan]
    # row 0: time.
    # row 1: direction.
    # row 2: correctness.
    # row 3: targets.
    return reaction


# bin reaction data.
def get_reaction_bin_stat(reaction):
    max_time=6000
    bin_size=200
    min_samples=2
    bins = np.arange(0, max_time + bin_size, bin_size)
    bins = bins - bin_size / 2
    bin_indices = np.digitize(reaction[0,:], bins) - 1
    bin_mean = []
    bin_sem = []
    for i in range(len(bins)-1):
        bin_values = reaction[2,bin_indices == i]
        m = np.mean(bin_values) if len(bin_values) > min_samples else np.nan
        s = sem(bin_values) if len(bin_values) > min_samples else np.nan
        bin_mean.append(m)
        bin_sem.append(s)
    bin_mean = np.array(bin_mean)
    bin_sem = np.array(bin_sem)
    return bins, bin_mean, bin_sem


# organize decision.
def get_decision(neural_trials):
    decision = [neural_trials[i]['trial_decision'] for i in neural_trials.keys()]
    decision = np.concatenate(decision, axis=1)
    stim_start = [np.array(neural_trials[i]['trial_stim_start']).reshape(-1)
                  for i in neural_trials.keys()]
    stim_start = np.concatenate(stim_start)
    decision[0,:] -= stim_start
    isi = []
    for trial in neural_trials.keys():
        trial_stim_seq = neural_trials[trial]['trial_stim_seq']
        if trial_stim_seq.shape[1]<4:
            stim_isi = np.array([np.nan]).reshape(1)
        else:
            stim_isi = np.mean(trial_stim_seq[0,3:] - trial_stim_seq[1,2:-1]).reshape(1)
        isi.append(stim_isi)
    isi = np.concatenate(isi).reshape(1,-1)
    decision = np.concatenate([decision, isi], axis=0)
    non_nan = (1-np.isnan(np.sum(decision, axis=0))).astype('bool')
    decision = decision[:,non_nan]
    # row 0: time.
    # row 1: direction.
    # row 2: correctness.
    # row 3: isi.
    return decision


class plotter_VIPTD_G8_beh:
    
    def __init__(
            self,
            neural_trials
            ):
        self.outcomes = get_beh_outcomes(neural_trials)
        self.perturb_time = get_perturb_time(neural_trials)
        self.decision = get_decision(neural_trials)
        self.reaction = get_reaction(neural_trials)

    # outcome percentage.
    def outcomes_precentage(self, ax):
        reward = np.sum(self.outcomes==1)
        punish = np.sum(self.outcomes==2)
        wrinit = np.sum(self.outcomes==3)
        notcho = np.sum(self.outcomes==4)
        ax.pie(
            [reward, punish, wrinit, notcho],
            labels=['{} reward'.format(reward),
                    '{} punish'.format(punish),
                    '{} wrong init'.format(wrinit),
                    '{} did not choose'.format(notcho)],
            colors=['#A4CB9E', '#EDA1A4', '#F9C08A', '#9DB4CE'],
            autopct='%1.1f%%',
            wedgeprops={'linewidth': 1, 'edgecolor':'white'})
        ax.set_title('percentage of outcomes')
    
    # completed trial correctness.
    def correctness_precentage(self, ax):
        reward = np.sum(self.outcomes==1)
        punish = np.sum(self.outcomes==2)
        ax.pie(
            [reward, punish],
            labels=['{} reward'.format(reward),
                    '{} punish'.format(punish)],
            colors=['#A4CB9E', '#EDA1A4'],
            autopct='%1.1f%%',
            wedgeprops={'linewidth': 1, 'edgecolor':'white'})
        ax.set_title('percentage of correctness')
    
    # choice for all completed trials.
    def choice_percentage(self, ax):
        l = np.sum(self.decision[1,:]==0)
        r = np.sum(self.decision[1,:]==1)
        ax.pie(
            [l, r],
            labels=['{} left'.format(l),
                    '{} right'.format(r)],
            colors=['#A4CB9E', '#EDA1A4'],
            autopct='%1.1f%%',
            wedgeprops={'linewidth': 1, 'edgecolor':'white'})
        ax.set_title('percentage of choice')
    
    # psychometric function.
    def psych_func(self, ax):
        bin_size=100
        least_trials=3
        bins = np.arange(0, 1000 + bin_size, bin_size)
        bins = bins - bin_size / 2
        bin_indices = np.digitize(self.decision[3,:], bins) - 1
        bin_mean = []
        bin_sem = []
        for i in range(len(bins)-1):
            direction = self.decision[1, bin_indices == i].copy()
            m = np.mean(direction) if len(direction) > least_trials else np.nan
            s = sem(direction) if len(direction) > least_trials else np.nan
            bin_mean.append(m)
            bin_sem.append(s)
        bin_mean = np.array(bin_mean)
        bin_sem  = np.array(bin_sem)
        bin_isi  = bins[:-1] + (bins[1]-bins[0]) / 2
        non_nan  = (1-np.isnan(bin_mean)).astype('bool')
        bin_mean = bin_mean[non_nan]
        bin_sem  = bin_sem[non_nan]
        bin_isi  = bin_isi[non_nan]
        ax.plot(
            bin_isi,
            bin_mean,
            color='dodgerblue', marker='.', label='fix', markersize=4)
        ax.fill_between(
            bin_isi,
            bin_mean - bin_sem,
            bin_mean + bin_sem,
            color='dodgerblue', alpha=0.2)
        ax.hlines(0.5, 0.0, 1000, linestyle=':', color='grey')
        ax.vlines(500, 0.0, 1.0, linestyle=':', color='grey')
        ax.tick_params(tick1On=False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xlim([-50,1050])
        ax.set_ylim([-0.05,1.05])
        ax.set_xticks(np.arange(6)*200)
        ax.set_yticks(np.arange(5)*0.25)
        ax.set_xlabel('isi')
        ax.set_ylabel('prob. of choosing the right side (mean$\pm$sem)')
        ax.set_title('session pychometric function')
        
    # decision time distribution and correctness.
    def decision_correct(self, ax):
        correct_all = self.decision[:,self.decision[2]==1]
        correct_left = correct_all[:, correct_all[3]==1]
        correct_right = correct_all[:, correct_all[3]==2]
        wrong_all = self.decision[:,self.decision[2]==0]
        wrong_left = wrong_all[:, wrong_all[3]==1]
        wrong_right = wrong_all[:, wrong_all[3]==2]
        correct_mean = [
            np.mean(correct_all[0,:]),
            np.mean(correct_left[0,:]),
            np.mean(correct_right[0,:])]
        correct_sem = [
            sem(correct_all[0,:]),
            sem(correct_left[0,:]),
            sem(correct_right[0,:])]
        wrong_mean = [
            np.mean(wrong_all[0,:]),
            np.mean(wrong_left[0,:]),
            np.mean(wrong_right[0,:])]
        wrong_sem = [
            sem(wrong_all[0,:]),
            sem(wrong_left[0,:]),
            sem(wrong_right[0,:])]
        colors = ['#A4CB9E', '#EDA1A4', '#9DB4CE']
        offset = [-0.1, 0, 0.1]
        label = ['all', 'short', 'long']
        ax.hlines(
            self.perturb_time, -0.2, 1.2,
            linestyle=':', color='grey', label='perturbation')
        for i in range(3):
            ax.errorbar(
                0 + offset[i],
                correct_mean[i], correct_sem[i],
                linestyle='none', color=colors[i], capsize=2, marker='o',
                markeredgecolor='white', markeredgewidth=1)
            ax.errorbar(
                1 + offset[i],
                wrong_mean[i], wrong_sem[i],
                linestyle='none', color=colors[i], capsize=2, marker='o',
                markeredgecolor='white', markeredgewidth=1)
        ax.tick_params(tick1On=False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_ylabel('Decision time (since stim onset) / s')
        ax.set_xticks([0,1])
        ax.set_xticklabels(['correct', 'wrong'])
        ax.set_xlim([-0.2, 1.5])
        for i in range(3):
            ax.plot([], label=label[i], color=colors[i])
        ax.legend(loc='upper right')
        ax.set_title('Decision time V.S. correctness')
