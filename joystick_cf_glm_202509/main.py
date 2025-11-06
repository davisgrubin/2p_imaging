#!/usr/bin/env python3

import warnings
warnings.filterwarnings("ignore")

import os
import traceback
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from modules.ReadResults import read_neural_trials
from modules.ReadResults import get_labels
#from modules import Trialization 

from modeling.generative import get_target_dff_time
from modeling.generative import get_factor_all
from modeling.generative import run_glm_multi_sess
from modeling.generative import get_coding_score_fraction
from modeling.generative import get_coding_score_dropout

from utils import get_mean_sem
from utils import plot_mean_sem
from utils import plot_heatmap_neuron

# read data.
list_session_data_path = [
    os.path.join('./results', 'SA11', 'SA11_20250813'),
    os.path.join('./results', 'SA11', 'SA11_20250814'),
    os.path.join('./results', 'SA11', 'SA11_20250815')
    ]
list_neural_trials = []
for session_data_path in list_session_data_path:
    ops = np.load(
        os.path.join(session_data_path, 'suite2p', 'plane0','ops.npy'),
        allow_pickle=True).item()
    ops['save_path0'] = os.path.join(session_data_path)
    #Trialization.run(ops)
    neural_trials = read_neural_trials(ops)
    neural_trials = get_labels(neural_trials, first_epoch = 6, last_epoch = 10)
    list_neural_trials.append(neural_trials)
'''
trial_types = np.concatenate([[nt[0][k]['trial_types'] for k in nt[0].keys()] for nt in list_neural_trials])
trial_delay = np.concatenate([[nt[0][k]['trial_delay'] for k in nt[0].keys()] for nt in list_neural_trials])
block_epoch = np.concatenate([[nt[0][k]['block_epoch'] for k in nt[0].keys()] for nt in list_neural_trials])
trial_probe = np.concatenate([[nt[0][k]['trial_probe'] for k in nt[0].keys()] for nt in list_neural_trials])
fig, ax = plt.subplots(1, 1, figsize=(15, 3))
ax.plot(np.arange(len(trial_types)), trial_types)
fig, ax = plt.subplots(1, 1, figsize=(15, 3))
ax.plot(np.arange(len(trial_delay)), trial_delay)
fig, ax = plt.subplots(1, 1, figsize=(15, 3))
ax.plot(np.arange(len(block_epoch)), block_epoch)
fig, ax = plt.subplots(1, 1, figsize=(15, 3))
ax.plot(np.arange(len(trial_probe)), trial_probe)
'''

# fit model.
def main():
    kernel_win = [-3000, 3000]
    # prepare glm input.
    list_glm_time, list_glm_dff = get_target_dff_time(list_neural_trials)
    model = get_factor_all(list_neural_trials)
    list_factor_names, list_glm_factor = model.run()
    list_all_factor_in = [[f[fi] for f in list_glm_factor] for fi in range(len(list_glm_dff))]
    # fit full glm model.
    print('Fitting GLM full model')
    kernel_time, kernel_all, exp_var_all = run_glm_multi_sess(list_glm_dff, list_all_factor_in, list_glm_time, kernel_win)
    results_all = {
        'kernel_time': kernel_time,
        'kernel_all': kernel_all,
        'exp_var_all': exp_var_all}
    # fit single glm model.
    print('Fitting GLM single factor model')
    results_single = {
        'kernel_time': [],
        'kernel_all': [],
        'exp_var_all': []}
    for fi, lgf in enumerate(list_glm_factor):
        print(f'Fitting GLM with only {list_factor_names[fi]}, {fi+1}/{len(list_factor_names)}')
        list_factor_in = [[f] for f in lgf]
        kernel_time, kernel_all, exp_var_all = run_glm_multi_sess(list_glm_dff, list_factor_in, list_glm_time, kernel_win)
        results_single['kernel_time'].append(kernel_time)
        results_single['kernel_all'].append(kernel_all[:,0,:])
        results_single['exp_var_all'].append(exp_var_all)
    # fit dropout glm model.
    print('Fitting GLM factor dropout model')
    results_dropout = {
        'kernel_time': [],
        'kernel_all': [],
        'exp_var_all': []}
    for fi, lgf in enumerate(list_glm_factor):
        print(f'Fitting GLM by dropout {list_factor_names[fi]}, {fi+1}/{len(list_factor_names)}')
        list_factor_in = [f[:fi]+f[fi+1:] for f in list_all_factor_in]
        kernel_time, kernel_all, exp_var_all = run_glm_multi_sess(list_glm_dff, list_factor_in, list_glm_time, kernel_win)
        results_dropout['kernel_time'].append(kernel_time)
        results_dropout['kernel_all'].append(kernel_all)
        results_dropout['exp_var_all'].append(exp_var_all)
    # evaluation.
    score_fraction = get_coding_score_fraction(results_all['exp_var_all'], results_single['exp_var_all'])
    score_dropout = get_coding_score_dropout(results_all['exp_var_all'], results_dropout['exp_var_all'])
    return [list_factor_names,
            results_all, results_single, results_dropout,
            score_fraction, score_dropout]
[list_factor_names,
 results_all, results_single, results_dropout,
 score_fraction, score_dropout] = main()

# visualize factor from dropout model.
fig, axs = plt.subplots(3, len(list_factor_names), figsize=(4*len(list_factor_names), 8))
for fi in range(len(list_factor_names)):
    try:
        idx = np.argsort(score_dropout[:,fi])[::-1][:int(len(score_dropout[:,fi])*0.5)]
        m, s = get_mean_sem(results_all['kernel_all'][idx,fi,:])
        plot_mean_sem(axs[0,fi], results_all['kernel_time'], m, s, 'dimgrey')
        axs[0,fi].set_title(list_factor_names[fi])
        axs[0,fi].spines['right'].set_visible(False)
        axs[0,fi].spines['top'].set_visible(False)
        axs[0,fi].axvline(0, color='black', lw=1, linestyle='--')
        axs[0,fi].xaxis.set_major_locator(mtick.MaxNLocator(nbins=3))
        axs[0,fi].yaxis.set_major_locator(mtick.MaxNLocator(nbins=3))
        axs[1,fi].hist(score_dropout[:,fi], color='dimgrey', bins=200)
        axs[1,fi].spines['right'].set_visible(False)
        axs[1,fi].spines['top'].set_visible(False)
        axs[1,fi].set_xlim([-0.1,0.4])
        axs[1,fi].xaxis.set_major_locator(mtick.MaxNLocator(nbins=3))
        axs[1,fi].yaxis.set_major_locator(mtick.MaxNLocator(nbins=3))
        plot_heatmap_neuron(
            axs[2,fi], None, results_all['kernel_all'][idx,fi,:], results_all['kernel_time'],
            results_all['kernel_all'][idx,fi,:], norm_mode='minmax')
    except: traceback.print_exc()
# visualize factor from single model.
fig, axs = plt.subplots(3, len(list_factor_names), figsize=(4*len(list_factor_names), 8))
for fi in range(len(list_factor_names)):
    try:
        idx = np.argsort(score_fraction[:,fi])[::-1][:int(len(score_fraction[:,fi])*0.5)]
        m, s = get_mean_sem(results_single['kernel_all'][fi][idx,:])
        plot_mean_sem(axs[0,fi], results_all['kernel_time'], m, s, 'dimgrey')
        axs[0,fi].set_title(list_factor_names[fi])
        axs[0,fi].spines['right'].set_visible(False)
        axs[0,fi].spines['top'].set_visible(False)
        axs[0,fi].axvline(0, color='black', lw=1, linestyle='--')
        axs[0,fi].xaxis.set_major_locator(mtick.MaxNLocator(nbins=3))
        axs[0,fi].yaxis.set_major_locator(mtick.MaxNLocator(nbins=3))
        axs[1,fi].hist(score_fraction[:,fi], color='dimgrey', bins=200)
        axs[1,fi].spines['right'].set_visible(False)
        axs[1,fi].spines['top'].set_visible(False)
        axs[1,fi].set_xlim([-0.1,0.8])
        axs[1,fi].xaxis.set_major_locator(mtick.MaxNLocator(nbins=3))
        axs[1,fi].yaxis.set_major_locator(mtick.MaxNLocator(nbins=3))
        plot_heatmap_neuron(
            axs[2,fi], None, results_single['kernel_all'][fi][idx,:], results_all['kernel_time'],
            results_single['kernel_all'][fi][idx,:], norm_mode='minmax')
    except: traceback.print_exc()
