#!/usr/bin/env python3

import warnings
warnings.filterwarnings('ignore')

import gc
import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from plot.fig1_mask import plotter_main_masks
from plot.fig2_raw_traces import plot_sess_example_traces
from plot.fig2_raw_traces import plot_ca_tran
from plot.fig2_raw_traces import plot_ca_tran_half_dist
from plot.misc import plot_motion_offset_hist
from plot.misc import plot_inh_exc_label_pc

def run(
        session_config_list,
        list_labels, list_masks, list_vol, list_dff, list_move_offset
        ):
    size_scale = 7
    # masks.
    print('Plotting fov images with masks')
    def plot_masks():
        title = 'imaging fov'
        filename = 'masks01_fov'
        n_row = 4
        n_col = 12
        fig = plt.figure(figsize=(n_col*size_scale, n_row*size_scale), layout='tight')
        gs = GridSpec(n_row, n_col, figure=fig)
        plotter = plotter_main_masks(
            list_labels[-1],
            list_masks[-1][0],
            list_masks[-1][1],
            list_masks[-1][2],
            list_masks[-1][3],
            list_masks[-1][4])
        if len(session_config_list['label_names'])==1:
            print('Plotting masks for 1 channel data')
            mask_axs = [plt.subplot(gs[0:2, j:j+2]) for j in range(0,12,2)]
            plotter.all_1chan(mask_axs)
        if len(session_config_list['label_names'])==2:
            print('Plotting masks for 2 channel data')
            mask_axs = [plt.subplot(gs[i:i+2, j:j+2]) for j in range(0,12,2) for i in [0,2]]
            plotter.all_2chan(mask_axs)
        fig.set_size_inches(n_col*size_scale, n_row*size_scale)
        fig.savefig(os.path.join('results', session_config_list['subject_name']+'_temp', filename+'.svg'), dpi=300, format='svg')
        fig.savefig(os.path.join('results', session_config_list['subject_name']+'_temp', filename+'.pdf'), dpi=300, format='pdf')
        plt.close(fig)
        del plotter
        gc.collect()
        return [filename, n_row, n_col, title]
    f1 = plot_masks()
    # example traces.
    print('Plotting example traces')
    def plot_example_traces():
        title = 'example traces'
        filename = 'masks02_example_traces'
        n_row = 4
        n_col = 1
        fig = plt.figure(figsize=(n_col*size_scale, n_row*size_scale), layout='tight')
        gs = GridSpec(n_row, n_col, figure=fig)
        example_ax = plt.subplot(gs[0:4, 0])
        plot_sess_example_traces(
            example_ax,
            list_dff[-1], list_labels[-1], list_vol[-1],
            session_config_list['label_names'])
        fig.set_size_inches(n_col*size_scale, n_row*size_scale)
        fig.savefig(os.path.join('results', session_config_list['subject_name']+'_temp', filename+'.svg'), dpi=300, format='svg')
        fig.savefig(os.path.join('results', session_config_list['subject_name']+'_temp', filename+'.pdf'), dpi=300, format='pdf')
        plt.close(fig)
        return [filename, n_row, n_col, title]
    f2 = plot_example_traces()
    # misc.
    def plot_misc():
        title = 'misc imaging analysis'
        filename = 'masks03_misc'
        n_row = 2
        n_col = 3
        fig = plt.figure(figsize=(n_col*size_scale, n_row*size_scale), layout='tight')
        gs = GridSpec(n_row, n_col, figure=fig)
        # labels.
        print('Plotting neural labels')
        label_ax = plt.subplot(gs[0, 0])
        plot_inh_exc_label_pc(label_ax, list_labels)
        # offset.
        print('Plotting motion offsets')
        offset_ax = plt.subplot(gs[1, 0])
        plot_motion_offset_hist(offset_ax, list_move_offset)
        # calcium transients.
        print('Plotting calcium transients')
        ca_axs01 = [plt.subplot(gs[0, 1]), plt.subplot(gs[1, 1])]
        ca_axs02 = [plt.subplot(gs[0, 2]), plt.subplot(gs[1, 2])]
        plot_ca_tran(
            ca_axs01,
            list_dff[-1], list_labels[-1], list_vol[-1],
            session_config_list['label_names'])
        plot_ca_tran_half_dist(
            ca_axs02,
            list_dff[-1], list_labels[-1], list_vol[-1],
            session_config_list['label_names'])
        fig.set_size_inches(n_col*size_scale, n_row*size_scale)
        fig.savefig(os.path.join('results', session_config_list['subject_name']+'_temp', filename+'.svg'), dpi=300, format='svg')
        fig.savefig(os.path.join('results', session_config_list['subject_name']+'_temp', filename+'.pdf'), dpi=300, format='pdf')
        plt.close(fig)
        return [filename, n_row, n_col, title]
    f3 = plot_misc()
    # combine temp filenames
    return [f1, f2, f3]

