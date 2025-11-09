#!/usr/bin/env python3

import warnings
warnings.filterwarnings('ignore')

import gc
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from modules.ReadResults import filter_session_config_list
from modules.ReadResults import read_all
from plot.misc import plot_significance
from plot.fig3_intervals import plot_standard_type
from plot.fig3_intervals import plot_fix_jitter_type
from plot.fig3_intervals import plot_oddball_type
from plot.fig3_intervals import plot_random_type
from plot.fig3_intervals import plot_standard_isi_distribution
from plot.fig3_intervals import plot_jitter_isi_distribution
from plot.fig3_intervals import plot_oddball_isi_distribution
from plot.fig3_intervals import plot_random_isi_distribution
from plot.fig3_intervals import plot_stim_type
from plot.fig3_intervals import plot_stim_label
from plot.fig3_intervals import plot_trial_legend
from plot.fig7_3331RandomExtended import plotter_main

def run(session_config_list, smooth, cate_list):
    size_scale = 3
    target_sess = 'extended_random'
    idx_target_sess = np.array(list(session_config_list['list_session_name'].values())) == target_sess
    print('Found {} {} sessions'.format(np.sum(idx_target_sess), target_sess))
    if (np.sum(idx_target_sess)==0) or (np.sum(idx_target_sess)==1 and not idx_target_sess[0]):
        return []
    else:
        print('Reading saved results')
        sub_session_config_list = filter_session_config_list(session_config_list, target_sess)
        [list_labels, _, list_neural_trials, _
         ] = read_all(sub_session_config_list, smooth)
        print('Read {} session results'.format(np.sum(idx_target_sess)))
        print('Initiating alignment results')
        plotter = plotter_main(
            list_neural_trials, list_labels,
            session_config_list['label_names'], 'temp_'+session_config_list['subject_name'], cate_list)
        def plot_cell_fraction():
            title = 'cell type fraction'
            print('-----------------------------------------------')
            print(title)
            filename = '3331RandomExtended_fraction'
            n_row = 1
            n_col = 1
            fig = plt.figure(figsize=(n_col*size_scale, n_row*size_scale), layout='tight')
            gs = GridSpec(n_row, n_col, figure=fig)
            ax = plt.subplot(gs[0, 0])
            plotter.plot_neuron_fraction(ax)
            fig.set_size_inches(n_col*size_scale, n_row*size_scale)
            fig.savefig(os.path.join('results', 'temp_'+session_config_list['subject_name'], filename+'.svg'), dpi=300, format='svg')
            plt.close(fig)
            return [filename, n_row, n_col, title]
        def plot_intervals():
            title = 'interval distribution'
            print('-----------------------------------------------')
            print(title)
            filename = '3331RandomExtended_interval_distribution'
            n_row = 2
            n_col = 4
            fig = plt.figure(figsize=(n_col*size_scale, n_row*size_scale), layout='tight')
            gs = GridSpec(n_row, n_col, figure=fig)
            type_ax01 = plt.subplot(gs[0, 0])
            type_ax02 = plt.subplot(gs[0, 1])
            type_ax03 = plt.subplot(gs[0, 2])
            type_ax04 = plt.subplot(gs[0, 3])
            plot_standard_type(type_ax01, list_neural_trials)
            plot_fix_jitter_type(type_ax02, list_neural_trials)
            plot_oddball_type(type_ax03, list_neural_trials)
            plot_random_type(type_ax04, list_neural_trials)
            isi_ax01 = plt.subplot(gs[1, 0])
            isi_ax02 = plt.subplot(gs[1, 1])
            isi_ax03 = plt.subplot(gs[1, 2])
            isi_ax04 = plt.subplot(gs[1, 3])
            plot_standard_isi_distribution(isi_ax01, list_neural_trials)
            plot_jitter_isi_distribution(isi_ax02, list_neural_trials)
            plot_oddball_isi_distribution(isi_ax03, list_neural_trials)
            plot_random_isi_distribution(isi_ax04, list_neural_trials)
            fig.set_size_inches(n_col*size_scale, n_row*size_scale)
            fig.savefig(os.path.join('results', 'temp_'+session_config_list['subject_name'], filename+'.svg'), dpi=300, format='svg')
            plt.close(fig)
            return [filename, n_row, n_col, title]
        def plot_trial():
            title = 'interval trial structure'
            print('-----------------------------------------------')
            print(title)
            filename = '3331RandomExtended_trial_structure'
            n_row = 1
            n_col = 10
            fig = plt.figure(figsize=(n_col*size_scale, n_row*size_scale), layout='tight')
            gs = GridSpec(n_row, n_col, figure=fig)
            trial_ax01 = plt.subplot(gs[0, 0])
            trial_ax02 = plt.subplot(gs[0, 1:5])
            trial_ax03 = plt.subplot(gs[0, 5])
            trial_ax04 = plt.subplot(gs[0, 6])
            trial_ax05 = plt.subplot(gs[0, 7:9])
            plot_stim_type(trial_ax01, list_neural_trials)
            plot_stim_label(trial_ax02, list_neural_trials, plotter.bin_win)
            plot_trial_legend(trial_ax03)
            plotter.plot_isi_seting(trial_ax04)
            plotter.plot_isi_example_epoch(trial_ax05)
            fig.set_size_inches(n_col*size_scale, n_row*size_scale)
            fig.savefig(os.path.join('results', 'temp_'+session_config_list['subject_name'], filename+'.svg'), dpi=300, format='svg')
            plt.close(fig)
            return [filename, n_row, n_col, title]
        def plot_cluster_stim_all():
            title = 'all cluster neural traces on stimulus'
            print('-----------------------------------------------')
            print(title)
            filename = '3331RandomExtended_cluster_stim_all'
            cate_gap = 3
            n_row = cate_gap*len(plotter.cate_list)
            n_col = 10
            fig = plt.figure(figsize=(n_col*size_scale, n_row*size_scale), layout='tight')
            gs = GridSpec(n_row, n_col, figure=fig)
            axs_all = []
            for s in cate_gap*np.arange(len(plotter.cate_list)):
                a = [plt.subplot(gs[s:s+2, i]) for i in range(10)]
                axs_all.append(a)
            plotter.cluster_stim_all(axs_all)
            fig.set_size_inches(n_col*size_scale, n_row*size_scale)
            fig.savefig(os.path.join('results', 'temp_'+session_config_list['subject_name'], filename+'.svg'), dpi=300, format='svg')
            plt.close(fig)
            return [filename, n_row, n_col, title]
        def plot_cluster_interval_bin_all():
            title = 'all cluster neural traces on binned stimlus interval'
            print('-----------------------------------------------')
            print(title)
            filename = '3331RandomExtended_cluster_interval_bin_all'
            cate_gap = 4
            n_row = cate_gap*len(plotter.cate_list)
            n_col = 8
            fig = plt.figure(figsize=(n_col*size_scale, n_row*size_scale), layout='tight')
            gs = GridSpec(n_row, n_col, figure=fig)
            axs_all = []
            for s in cate_gap*np.arange(len(plotter.cate_list)):
                a = [plt.subplot(gs[s:s+2, i]) for i in range(8)]
                axs_all.append(a)
            plotter.cluster_interval_bin_all(axs_all)
            fig.set_size_inches(n_col*size_scale, n_row*size_scale)
            fig.savefig(os.path.join('results', 'temp_'+session_config_list['subject_name'], filename+'.svg'), dpi=300, format='svg')
            plt.close(fig)
            return [filename, n_row, n_col, title]
        def plot_latent_all():
            title = 'all cluster latent dynamics on binned stimlus interval'
            print('-----------------------------------------------')
            print(title)
            filename = '3331RandomExtended_latent_all'
            cate_gap = 3
            n_row = cate_gap*len(plotter.cate_list)
            n_col = 20
            fig = plt.figure(figsize=(n_col*size_scale, n_row*size_scale), layout='tight')
            gs = GridSpec(n_row, n_col, figure=fig)
            axs_all = []
            for s in cate_gap*np.arange(len(plotter.cate_list)):
                a = [[plt.subplot(gs[s+0:s+2, i:i+2]) for i in 2*np.arange(10)]]
                axs_all.append(a)
            plotter.latent_all(axs_all)
            fig.set_size_inches(n_col*size_scale, n_row*size_scale)
            fig.savefig(os.path.join('results', 'temp_'+session_config_list['subject_name'], filename+'.svg'), dpi=300, format='svg')
            plt.close(fig)
            return [filename, n_row, n_col, title]
        fig_all = [
            plot_intervals(),
            plot_trial(),
            plot_cluster_stim_all(),
            plot_cluster_interval_bin_all(),
            plot_latent_all(),
            ]
        print('Clearing memory usage')
        del list_labels
        del list_neural_trials
        del plotter
        gc.collect()
        return fig_all
