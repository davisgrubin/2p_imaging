#!/usr/bin/env python3

import warnings
warnings.filterwarnings('ignore')

import gc
import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from modules.ReadResults import read_all
from plot.fig1_mask import plotter_main_masks
from plot.fig2_raw_traces import plot_sess_example_traces
from plot.misc import plot_surgery_window

def run(session_config_list, smooth, cate_list):
    size_scale = 3
    [list_labels, list_masks, list_neural_trials, list_move_offset
     ] = read_all(session_config_list, smooth)
    def plot_window():
        title = 'surgery window'
        print('-----------------------------------------------')
        print(title)
        filename = 'fov_window'
        session_folder = [
            os.path.join('results', sc['session_folder'])
            for sc in session_config_list['list_config']]
        path_window = []
        path_note = []
        for sf in session_folder:
            path_window += [os.path.join(sf, f) for f in os.listdir(sf) if 'window' in f]
            path_note +=[os.path.join(sf, f) for f in os.listdir(sf) if 'note' in f]
        img_window = [plt.imread(wp) for wp in path_window]
        img_note = [plt.imread(wp) for wp in path_note]
        n_row = 2
        n_col = 5
        fig = plt.figure(figsize=(n_col*size_scale, n_row*size_scale), layout='tight')
        gs = GridSpec(n_row, n_col, figure=fig)
        win_ax01 = [plt.subplot(gs[0, i]) for i in range(len(path_window))]
        win_ax02 = [plt.subplot(gs[1, i]) for i in range(len(path_note))]
        plot_surgery_window(win_ax01, img_window)
        plot_surgery_window(win_ax02, img_note)
        fig.set_size_inches(n_col*size_scale, n_row*size_scale)
        fig.savefig(os.path.join('results', 'temp_'+session_config_list['subject_name'], filename+'.svg'), dpi=300, format='svg')
        plt.close(fig)
        return [filename, n_row, n_col, title]
    def plot_masks():
        title = 'imaging fov'
        print('-----------------------------------------------')
        print(title)
        filename = 'fov_imaging'
        n_row = 4
        n_col = 12
        fig = plt.figure(figsize=(n_col*size_scale, n_row*size_scale), layout='tight')
        gs = GridSpec(n_row, n_col, figure=fig)
        plotter = plotter_main_masks(
            list_labels[0],
            list_masks[0][0],
            list_masks[0][1],
            list_masks[0][2],
            list_masks[0][3],
            list_masks[0][4])
        try:
            mask_axs = [plt.subplot(gs[i:i+2, j:j+2]) for j in range(0,12,2) for i in [0,2]]
            plotter.all_2chan(mask_axs)
        except:
            mask_axs = [plt.subplot(gs[0:2, j:j+2]) for j in range(0,10,2)]
            plotter.all_1chan(mask_axs)
        fig.set_size_inches(n_col*size_scale, n_row*size_scale)
        fig.savefig(os.path.join('results', 'temp_'+session_config_list['subject_name'], filename+'.svg'), dpi=300, format='svg')
        plt.close(fig)
        del plotter
        gc.collect()
        return [filename, n_row, n_col, title]
    def plot_example_traces():
        title = 'example traces'
        print('-----------------------------------------------')
        print(title)
        filename = 'fov_example_traces'
        n_row = 12
        n_col = 5
        fig = plt.figure(figsize=(n_col*size_scale, n_row*size_scale), layout='tight')
        gs = GridSpec(n_row, n_col, figure=fig)
        example_ax = plt.subplot(gs[0:12, 0:4])
        plot_sess_example_traces(example_ax, list_labels, list_neural_trials, session_config_list['label_names'])
        fig.set_size_inches(n_col*size_scale, n_row*size_scale)
        fig.savefig(os.path.join('results', 'temp_'+session_config_list['subject_name'], filename+'.svg'), dpi=300, format='svg')
        plt.close(fig)
        return [filename, n_row, n_col, title]
    fig_all = [
        #plot_window(),
        plot_masks(),
        plot_example_traces(),
        ]
    return fig_all

