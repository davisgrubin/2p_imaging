#!/usr/bin/env python3

import warnings
warnings.filterwarnings('ignore')

import gc
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

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
from plot.fig4_3331Random import plotter_main

def run(
        session_config_list,
        list_labels, list_vol, list_dff, list_neural_trials, list_significance
        ):
    size_scale = 5
    # filter data.
    target_sess = 'random'
    idx = np.array(list(session_config_list['list_session_name'].values())) == target_sess
    sess_names = np.array(list(session_config_list['list_session_name'].keys()))[idx].copy().tolist()
    list_labels = np.array(list_labels,dtype='object')[idx].copy().tolist()
    list_vol = np.array(list_vol,dtype='object')[idx].copy().tolist()
    list_dff = np.array(list_dff,dtype='object')[idx].copy().tolist()
    list_neural_trials = np.array(list_neural_trials,dtype='object')[idx].copy().tolist()
    list_significance = np.array(list_significance,dtype='object')[idx].copy().tolist()
    print('Found {} {} sessions'.format(len(sess_names), target_sess))
    if len(sess_names) == 0:
        return []
    else:
        # create plotter.
        print('Initiating alignment results')
        plotter = plotter_main(list_neural_trials, list_labels, list_significance, session_config_list['label_names'])
        # significance test.
        print('Plotting significance test results')
        def plot_sess_significance():
            title = 'significance'
            filename = '3331Random01_significance'
            n_row = 1
            n_col = 1
            fig = plt.figure(figsize=(n_col*size_scale, n_row*size_scale), layout='constrained')
            gs = GridSpec(n_row, n_col, figure=fig)
            sign_ax = plt.subplot(gs[0, 0])
            plot_significance(sign_ax, list_significance, list_labels)
            fig.set_size_inches(n_col*size_scale, n_row*size_scale)
            fig.savefig(os.path.join('results', session_config_list['subject_name']+'_temp', filename+'.svg'), dpi=300, format='svg')
            fig.savefig(os.path.join('results', session_config_list['subject_name']+'_temp', filename+'.pdf'), dpi=300, format='pdf')
            plt.close(fig)
            return [filename, n_row, n_col, title]
        # stimulus types and interval distributions.
        print('Plotting stimulus types and interval distributions')
        def plot_intervals():
            title = 'interval distribution'
            filename = '3331Random02_interval_distribution'
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
            fig.savefig(os.path.join('results', session_config_list['subject_name']+'_temp', filename+'.svg'), dpi=300, format='svg')
            fig.savefig(os.path.join('results', session_config_list['subject_name']+'_temp', filename+'.pdf'), dpi=300, format='pdf')
            plt.close(fig)
            return [filename, n_row, n_col, title]
        # trial structure.
        print('Plotting trial structure')
        def plot_trial():
            title = 'interval trial structure'
            filename = '3331Random03_trial_structure'
            n_row = 1
            n_col = 6
            fig = plt.figure(figsize=(n_col*size_scale, n_row*size_scale), layout='tight')
            gs = GridSpec(n_row, n_col, figure=fig)
            trial_ax01 = plt.subplot(gs[0, 0])
            trial_ax02 = plt.subplot(gs[0, 1:5])
            plot_stim_type(trial_ax01, list_neural_trials)
            plot_stim_label(trial_ax02, list_neural_trials)
            fig.set_size_inches(n_col*size_scale, n_row*size_scale)
            fig.savefig(os.path.join('results', session_config_list['subject_name']+'_temp', filename+'.svg'), dpi=300, format='svg')
            fig.savefig(os.path.join('results', session_config_list['subject_name']+'_temp', filename+'.pdf'), dpi=300, format='pdf')
            plt.close(fig)
            return [filename, n_row, n_col, title]
        # neural alignments.
        print('Plotting neural traces alignments')
        def plot_random_alignment():
            title = 'neural traces alignment on random interval'
            filename = '3331Random04_random_alignment'
            n_row = 2
            n_col = 7
            fig = plt.figure(figsize=(n_col*size_scale, n_row*size_scale), layout='tight')
            gs = GridSpec(n_row, n_col, figure=fig)
            axs_all = []
            for s in [0,1]:
                a = [plt.subplot(gs[s+0, i]) for i in range(7)]
                axs_all.append(a)
            plotter.random(axs_all)
            fig.set_size_inches(n_col*size_scale, n_row*size_scale)
            fig.savefig(os.path.join('results', session_config_list['subject_name']+'_temp', filename+'.svg'), dpi=300, format='svg')
            fig.savefig(os.path.join('results', session_config_list['subject_name']+'_temp', filename+'.pdf'), dpi=300, format='pdf')
            plt.close(fig)
            return [filename, n_row, n_col, title]
        # clustering analysis.
        print('Plotting clustering analysis')
        def plot_clustering():
            title = 'clustering on random interval'
            filename = '3331Random05_random_clustering'
            n_row = 12
            n_col = 3
            fig = plt.figure(figsize=(n_col*size_scale, n_row*size_scale), layout='tight')
            gs = GridSpec(n_row, n_col, figure=fig)
            axs_all = []
            for s in [0,4,8]:
                a = [plt.subplot(gs[s+0, 0]), plt.subplot(gs[s+1, 0]),
                     plt.subplot(gs[s+2, 0]), plt.subplot(gs[s+3, 0]),
                     plt.subplot(gs[s+0:s+2, 1])]
                a+= [plt.subplot(gs[s+2:s+4, 1])]
                a+= [plt.subplot(gs[s+0:s+2, 2])]
                a+= [plt.subplot(gs[s+2:s+4, 2])]
                axs_all.append(a)
            plotter.cluster(axs_all)
            fig.set_size_inches(n_col*size_scale, n_row*size_scale)
            fig.savefig(os.path.join('results', session_config_list['subject_name']+'_temp', filename+'.svg'), dpi=300, format='svg')
            fig.savefig(os.path.join('results', session_config_list['subject_name']+'_temp', filename+'.pdf'), dpi=300, format='pdf')
            plt.close(fig)
            return [filename, n_row, n_col, title]
        # clustering latents.
        print('Plotting clustering latent dynamics')
        def plot_clustering_latents():
            title = 'clustering latent dynamics on random interval'
            filename = '3331Random06_random_clustering_latents'
            n_row = 1
            n_col = 4
            fig = plt.figure(figsize=(n_col*size_scale, n_row*size_scale), layout='tight')
            gs = GridSpec(n_row, n_col, figure=fig)
            axs_all = []
            for s in [0]:
                a = [plt.subplot(gs[s+0, i], projection='3d') for i in range(4)]
                axs_all.append(a)
            plotter.cluster_latents(axs_all)
            fig.set_size_inches(n_col*size_scale, n_row*size_scale)
            fig.savefig(os.path.join('results', session_config_list['subject_name']+'_temp', filename+'.svg'), dpi=300, format='svg')
            fig.savefig(os.path.join('results', session_config_list['subject_name']+'_temp', filename+'.pdf'), dpi=300, format='pdf')
            plt.close(fig)
            return [filename, n_row, n_col, title]
        # clustering heatmaps.
        print('Plotting sorted heatmaps')
        def plot_sorted_heatmaps():
            title = 'sorted heatmaps on random interval'
            filename = '3331Random07_random_sorted_heatmaps'
            n_row = 4
            n_col = 3
            fig = plt.figure(figsize=(n_col*size_scale, n_row*size_scale), layout='tight')
            gs = GridSpec(n_row, n_col, figure=fig)
            axs_all = []
            for s in [0]:
                a = [[plt.subplot(gs[s+0:s+2, i]) for i in range(3)],
                     [plt.subplot(gs[s+2:s+4, i]) for i in range(3)]]
                axs_all.append(a)
            plotter.sorted_heatmaps(axs_all)
            fig.set_size_inches(n_col*size_scale, n_row*size_scale)
            fig.savefig(os.path.join('results', session_config_list['subject_name']+'_temp', filename+'.svg'), dpi=300, format='svg')
            fig.savefig(os.path.join('results', session_config_list['subject_name']+'_temp', filename+'.pdf'), dpi=300, format='pdf')
            plt.close(fig)
            return [filename, n_row, n_col, title]
        # glm.
        print('Plotting GLM analysis')
        def plot_glm():
            title = 'GLM neural coding analysis'
            filename = '3331Random08_glm_coding'
            n_row = 3
            n_col = 3
            fig = plt.figure(figsize=(n_col*size_scale, n_row*size_scale), layout='tight')
            gs = GridSpec(n_row, n_col, figure=fig)
            axs_all = []
            for s in [0,1,2]:
                a = [plt.subplot(gs[s+0, i]) for i in range(3)]
                axs_all.append(a)
            plotter.glm(axs_all)
            fig.set_size_inches(n_col*size_scale, n_row*size_scale)
            fig.savefig(os.path.join('results', session_config_list['subject_name']+'_temp', filename+'.svg'), dpi=300, format='svg')
            fig.savefig(os.path.join('results', session_config_list['subject_name']+'_temp', filename+'.pdf'), dpi=300, format='pdf')
            plt.close(fig)
            return [filename, n_row, n_col, title]
        # plot all.
        fig_all = [
            plot_sess_significance(),
            plot_intervals(),
            plot_trial(),
            plot_random_alignment(),
            plot_clustering(),
            plot_clustering_latents(),
            plot_sorted_heatmaps(),
            plot_glm(),
            ]
        # clear memory.
        print('Clearing memory usage')
        del list_labels
        del list_vol
        del list_dff
        del list_neural_trials
        del list_significance
        del plotter
        gc.collect()
        # combine temp filenames
        return fig_all
