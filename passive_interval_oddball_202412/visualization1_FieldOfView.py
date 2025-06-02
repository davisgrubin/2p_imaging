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
from plot.fig2_raw_traces import plot_ca_transient
from plot.misc import plot_surgery_window
from plot.misc import plot_inh_exc_label_pc

def run(session_config_list, smooth):
    size_scale = 3
    n_examples = 3
    n_sess = len(session_config_list['list_session_name'])
    n_examples = n_examples if n_examples < n_sess else n_sess
    [list_labels, list_masks, list_neural_trials, list_move_offset, _
     ] = read_all(session_config_list, smooth)
    def plot_window():
        title = 'surgery window'
        print('-----------------------------------------------')
        print(title)
        filename = 'fov01_window'
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
        filename = 'fov02_imaging'
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
        filename = 'fov03_example_traces'
        n_row = 4
        n_col = 1
        fig = plt.figure(figsize=(n_col*size_scale, n_row*size_scale), layout='tight')
        gs = GridSpec(n_row, n_col, figure=fig)
        example_ax = plt.subplot(gs[0:4, 0])
        plot_sess_example_traces(example_ax, list_labels, list_neural_trials, session_config_list['label_names'])
        fig.set_size_inches(n_col*size_scale, n_row*size_scale)
        fig.savefig(os.path.join('results', 'temp_'+session_config_list['subject_name'], filename+'.svg'), dpi=300, format='svg')
        plt.close(fig)
        return [filename, n_row, n_col, title]
    def plot_ca():
        title = 'calcium transient analysis'
        print('-----------------------------------------------')
        print(title)
        filename = 'fov04_ca'
        n_row = 2
        n_col = 3
        fig = plt.figure(figsize=(n_col*size_scale, n_row*size_scale), layout='tight')
        gs = GridSpec(n_row, n_col, figure=fig)
        ca_axs01 = [plt.subplot(gs[0, i]) for i in range(3)]
        ca_axs02 = [plt.subplot(gs[1, i]) for i in range(3)]
        plot_ca_transient(ca_axs01, list_labels, list_neural_trials, session_config_list['label_names'], [-1])
        plot_ca_transient(ca_axs02, list_labels, list_neural_trials, session_config_list['label_names'], [1])
        fig.set_size_inches(n_col*size_scale, n_row*size_scale)
        fig.savefig(os.path.join('results', 'temp_'+session_config_list['subject_name'], filename+'.svg'), dpi=300, format='svg')
        plt.close(fig)
        return [filename, n_row, n_col, title]
    def plot_misc():
        title = 'misc imaging results'
        print('-----------------------------------------------')
        print(title)
        filename = 'fov05_misc'
        n_row = 1
        n_col = 1
        fig = plt.figure(figsize=(n_col*size_scale, n_row*size_scale), layout='tight')
        gs = GridSpec(n_row, n_col, figure=fig)
        label_ax = plt.subplot(gs[0, 0])
        plot_inh_exc_label_pc(label_ax, list_labels)
        fig.set_size_inches(n_col*size_scale, n_row*size_scale)
        fig.savefig(os.path.join('results', 'temp_'+session_config_list['subject_name'], filename+'.svg'), dpi=300, format='svg')
        plt.close(fig)
        return [filename, n_row, n_col, title]
    fig_all = [
        #plot_window(),
        plot_masks(),
        #plot_example_traces(),
        #plot_ca(),
        #plot_misc(),
        ]
    return fig_all

