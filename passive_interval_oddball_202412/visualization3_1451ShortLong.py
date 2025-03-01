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
from plot.fig5_1451ShortLong import plotter_main

def run(
        session_config_list,
        list_labels, list_vol, list_dff, list_neural_trials, list_significance
        ):
    size_scale = 5
    target_sess = 'short_long'
    idx = np.array(list(session_config_list['list_session_name'].values())) == target_sess
    sess_names = np.array(list(session_config_list['list_session_name'].keys()))[idx].copy().tolist()
    list_labels = np.array(list_labels,dtype='object')[idx].copy().tolist()
    list_vol = np.array(list_vol,dtype='object')[idx].copy().tolist()
    list_dff = np.array(list_dff,dtype='object')[idx].copy().tolist()
    list_neural_trials = np.array(list_neural_trials,dtype='object')[idx].copy().tolist()
    list_significance = np.array(list_significance,dtype='object')[idx].copy().tolist()
    print('Found {} {} sessions'.format(len(sess_names), target_sess))
    if (len(sess_names) == 0) or (len(idx)==1 and not idx[0]):
        return []
    else:
        print('Initiating alignment results')
        plotter = plotter_main(list_neural_trials, list_labels, list_significance, session_config_list['label_names'])
        def plot_sess_significance():
            title = 'significance'
            print(title)
            filename = '1451ShortLong01_significance'
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
        def plot_intervals():
            title = 'interval distribution'
            print(title)
            filename = '1451ShortLong02_interval_distribution'
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
        def plot_trial():
            title = 'interval trial structure'
            print(title)
            filename = '1451ShortLong03_trial_structure'
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
        def plot_standard():
            title = 'neural traces alignment on standard intervals'
            print(title)
            filename = '1451ShortLong04_standard'
            n_row = 2
            n_col = 10
            fig = plt.figure(figsize=(n_col*size_scale, n_row*size_scale), layout='tight')
            gs = GridSpec(n_row, n_col, figure=fig)
            axs_all = []
            for s in [0,1]:
                a = [plt.subplot(gs[s+0, i]) for i in range(4)]
                a+= [plt.subplot(gs[s+0, i+4], projection='3d') for i in range(2)]
                axs_all.append(a)
            plotter.standard(axs_all)
            fig.set_size_inches(n_col*size_scale, n_row*size_scale)
            fig.savefig(os.path.join('results', session_config_list['subject_name']+'_temp', filename+'.svg'), dpi=300, format='svg')
            fig.savefig(os.path.join('results', session_config_list['subject_name']+'_temp', filename+'.pdf'), dpi=300, format='pdf')
            plt.close(fig)
            return [filename, n_row, n_col, title]
        def plot_oddball():
            title = 'neural traces alignment on oddball intervals'
            print(title)
            filename = '1451ShortLong05_oddball'
            n_row = 4
            n_col = 3
            fig = plt.figure(figsize=(n_col*size_scale, n_row*size_scale), layout='tight')
            gs = GridSpec(n_row, n_col, figure=fig)
            axs_all = []
            for s in [0,2]:
                a = [plt.subplot(gs[s+i, 0]) for i in range(2)]
                a+= [plt.subplot(gs[s+i, 1]) for i in range(2)]
                a+= [plt.subplot(gs[s+i, 2], projection='3d') for i in range(2)]
                axs_all.append(a)
            plotter.oddball(axs_all)
            fig.set_size_inches(n_col*size_scale, n_row*size_scale)
            fig.savefig(os.path.join('results', session_config_list['subject_name']+'_temp', filename+'.svg'), dpi=300, format='svg')
            fig.savefig(os.path.join('results', session_config_list['subject_name']+'_temp', filename+'.pdf'), dpi=300, format='pdf')
            plt.close(fig)
            return [filename, n_row, n_col, title]
        def plot_block():
            title = 'neural traces alignment on block transition'
            print(title)
            filename = '1451ShortLong06_block'
            n_row = 4
            n_col = 3
            fig = plt.figure(figsize=(n_col*size_scale, n_row*size_scale), layout='tight')
            gs = GridSpec(n_row, n_col, figure=fig)
            axs_all = []
            for s in [0,2]:
                a = [plt.subplot(gs[s+i, 0:2]) for i in [0,1]]
                a+= [plt.subplot(gs[s+i, 2], projection='3d') for i in [0,1]]
                axs_all.append(a)
            plotter.block(axs_all)
            fig.set_size_inches(n_col*size_scale, n_row*size_scale)
            fig.savefig(os.path.join('results', session_config_list['subject_name']+'_temp', filename+'.svg'), dpi=300, format='svg')
            fig.savefig(os.path.join('results', session_config_list['subject_name']+'_temp', filename+'.pdf'), dpi=300, format='pdf')
            plt.close(fig)
            return [filename, n_row, n_col, title]
        def plot_clustering():
            title = 'clustering on short long interval'
            print(title)
            filename = '1451ShortLong07_clustering'
            n_row = 12
            n_col = 12
            fig = plt.figure(figsize=(n_col*size_scale, n_row*size_scale), layout='tight')
            gs = GridSpec(n_row, n_col, figure=fig)
            axs_all = []
            for s in [0,4,8]:
                a = [plt.subplot(gs[s+0, 0]), plt.subplot(gs[s+1, 0]),
                     plt.subplot(gs[s+2, 0]), plt.subplot(gs[s+3, 0]),
                     plt.subplot(gs[s+0:s+2, 1])]
                a+= [plt.subplot(gs[s+2:s+4, 1])]
                a+= [plt.subplot(gs[s+0:s+2, i]) for i in [2,3,4,5]]
                a+= [plt.subplot(gs[s+0:s+2, 6:8]), plt.subplot(gs[s+0:s+2, 8:10])]
                a+= [plt.subplot(gs[s+2:s+4, i]) for i in [2,3,4,5]]
                a+= [plt.subplot(gs[s+2:s+4, 6:8]), plt.subplot(gs[s+2:s+4, 8:10])]
                axs_all.append(a)
            plotter.cluster(axs_all)
            fig.set_size_inches(n_col*size_scale, n_row*size_scale)
            fig.savefig(os.path.join('results', session_config_list['subject_name']+'_temp', filename+'.svg'), dpi=300, format='svg')
            fig.savefig(os.path.join('results', session_config_list['subject_name']+'_temp', filename+'.pdf'), dpi=300, format='pdf')
            plt.close(fig)
            return [filename, n_row, n_col, title]
        def plot_feature_categorization():
            title = 'feature categorization analysis'
            print(title)
            filename = '1451ShortLong08_feature_categorization'
            n_row = 12
            n_col = 7
            fig = plt.figure(figsize=(n_col*size_scale, n_row*size_scale), layout='tight')
            gs = GridSpec(n_row, n_col, figure=fig)
            axs_all = []
            for s in [0]:
                a = [[plt.subplot(gs[s+0, i]) for i in [0,1,2,3]]]
                a+= [[plt.subplot(gs[s+1, i]) for i in [0,1,2,3,4,5,6]]]
                a+= [[plt.subplot(gs[s+2, i]) for i in [0,1,2,3,4,5,6]]]
                a+= [[plt.subplot(gs[s+3, i]) for i in [0,1,2,3,4,5,6]]]
                a+= [[plt.subplot(gs[s+4, i]) for i in [0,1,2,3,4,5,6]]]
                a+= [[plt.subplot(gs[s+5:s+7,  i]) for i in [0,1,2,3,4,5,6]]]
                a+= [[plt.subplot(gs[s+7:s+9,  i]) for i in [0,1,2,3,4,5,6]]]
                a+= [[plt.subplot(gs[s+9:s+11, i]) for i in [0,1,2,3,4,5,6]]]
                a+= [plt.subplot(gs[s+11, 0])]
                axs_all.append(a)
            plotter.categorization_features(axs_all)
            fig.set_size_inches(n_col*size_scale, n_row*size_scale)
            fig.savefig(os.path.join('results', session_config_list['subject_name']+'_temp', filename+'.svg'), dpi=300, format='svg')
            fig.savefig(os.path.join('results', session_config_list['subject_name']+'_temp', filename+'.pdf'), dpi=300, format='pdf')
            plt.close(fig)
            return [filename, n_row, n_col, title]
        def plot_sorted_heatmaps_standard():
            title = 'sorted heatmaps on short long interval'
            print(title)
            filename = '1451ShortLong09_sorted_heatmaps_standard'
            n_row = 12
            n_col = 4
            fig = plt.figure(figsize=(n_col*size_scale, n_row*size_scale), layout='tight')
            gs = GridSpec(n_row, n_col, figure=fig)
            axs_all = []
            for s in [0,4,8]:
                a = [[plt.subplot(gs[s+0:s+2, i]) for i in range(4)],
                     [plt.subplot(gs[s+2:s+4, i]) for i in range(4)]]
                axs_all.append(a)
            plotter.sorted_heatmaps_standard(axs_all)
            fig.set_size_inches(n_col*size_scale, n_row*size_scale)
            fig.savefig(os.path.join('results', session_config_list['subject_name']+'_temp', filename+'.svg'), dpi=300, format='svg')
            fig.savefig(os.path.join('results', session_config_list['subject_name']+'_temp', filename+'.pdf'), dpi=300, format='pdf')
            plt.close(fig)
            return [filename, n_row, n_col, title]
        def plot_temporal_scaling():
            title = 'temporal scaling effect on short long interval'
            print(title)
            filename = '1451ShortLong10_temporal_scaling'
            n_row = 6
            n_col = 4
            fig = plt.figure(figsize=(n_col*size_scale, n_row*size_scale), layout='tight')
            gs = GridSpec(n_row, n_col, figure=fig)
            axs_all = []
            for s in [0,2,4]:
                a = [plt.subplot(gs[s+0:s+2, i]) for i in range(3)]
                axs_all.append(a)
            plotter.temporal_scaling(axs_all)
            fig.set_size_inches(n_col*size_scale, n_row*size_scale)
            fig.savefig(os.path.join('results', session_config_list['subject_name']+'_temp', filename+'.svg'), dpi=300, format='svg')
            fig.savefig(os.path.join('results', session_config_list['subject_name']+'_temp', filename+'.pdf'), dpi=300, format='pdf')
            plt.close(fig)
            return [filename, n_row, n_col, title]
        def plot_glm():
            title = 'GLM neural coding analysis'
            print(title)
            filename = '1451ShortLong11_glm_coding'
            n_row = 6
            n_col = 3
            fig = plt.figure(figsize=(n_col*size_scale, n_row*size_scale), layout='tight')
            gs = GridSpec(n_row, n_col, figure=fig)
            axs_all = []
            for s in [0,2,4]:
                a = [plt.subplot(gs[s+0:s+2, i]) for i in range(3)]
                axs_all.append(a)
            plotter.glm(axs_all)
            fig.set_size_inches(n_col*size_scale, n_row*size_scale)
            fig.savefig(os.path.join('results', session_config_list['subject_name']+'_temp', filename+'.svg'), dpi=300, format='svg')
            fig.savefig(os.path.join('results', session_config_list['subject_name']+'_temp', filename+'.pdf'), dpi=300, format='pdf')
            plt.close(fig)
            return [filename, n_row, n_col, title]
        fig_all = [
            #plot_sess_significance(),
            #plot_intervals(),
            plot_trial(),
            #plot_standard(),
            #plot_oddball(),
            #plot_block(),
            #plot_clustering(),
            plot_feature_categorization(),
            plot_sorted_heatmaps_standard(),
            plot_temporal_scaling(),
            #plot_glm(),
            ]
        print('Clearing memory usage')
        del list_labels
        del list_vol
        del list_dff
        del list_neural_trials
        del list_significance
        del plotter
        gc.collect()
        return fig_all