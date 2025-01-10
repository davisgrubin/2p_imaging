#!/usr/bin/env python3

import warnings
warnings.filterwarnings('ignore')

import os
import fitz
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from plot.fig1_mask import plotter_main_masks
from plot.fig2_raw_traces import plot_sess_example_traces
from plot.fig2_raw_traces import plot_ca_tran
from plot.fig2_raw_traces import plot_ca_tran_half_dist
from plot.misc import plot_motion_offset_hist
from plot.misc import plot_inh_exc_label_pc

def run(
        session_config, session_report,
        list_labels, list_masks, list_vol, list_dff, list_move_offset
        ):
    n_row = 5
    n_col = 20
    size_scale = 7
    plotter_masks = plotter_main_masks(
        list_labels[-1],
        list_masks[-1][0],
        list_masks[-1][1],
        list_masks[-1][2],
        list_masks[-1][3],
        list_masks[-1][4])
    # create canvas.
    fig = plt.figure(figsize=(n_col*size_scale, n_row*size_scale), layout='tight')
    gs = GridSpec(n_row, n_col, figure=fig)
    # masks.
    if len(session_config['label_names'])==1:
        print('Plotting masks for 1 channel data')
        mask_axs = [plt.subplot(gs[0:2, j:j+2]) for j in range(0,12,2)]
        plotter_masks.all_1chan(mask_axs)
    if len(session_config['label_names'])==2:
        print('Plotting masks for 2 channel data')
        mask_axs = [plt.subplot(gs[i:i+2, j:j+2]) for j in range(0,12,2) for i in [0,2]]
        plotter_masks.all_2chan(mask_axs)
    # example traces.
    print('Plotting example traces')
    example_ax = plt.subplot(gs[0:4, 12])
    plot_sess_example_traces(
        example_ax,
        list_dff[-1], list_labels[-1], list_vol[-1],
        session_config['label_names'])
    # labels.
    print('Plotting neural labels')
    label_ax = plt.subplot(gs[0, 13])
    plot_inh_exc_label_pc(label_ax, list_labels)
    # offset.
    print('Plotting motion offsets')
    offset_ax = plt.subplot(gs[1, 13])
    plot_motion_offset_hist(offset_ax, list_move_offset)
    # calcium transients.
    print('Plotting calcium transients')
    ca_axs01 = [plt.subplot(gs[2, 13]), plt.subplot(gs[3, 13])]
    ca_axs02 = [plt.subplot(gs[2, 14]), plt.subplot(gs[3, 14])]
    plot_ca_tran(
        ca_axs01,
        list_dff[-1], list_labels[-1], list_vol[-1],
        session_config['label_names'])
    plot_ca_tran_half_dist(
        ca_axs02,
        list_dff[-1], list_labels[-1], list_vol[-1],
        session_config['label_names'])
    # save temp file.
    print('Saving results')
    fname = os.path.join('results', 'masks.pdf')
    fig.set_size_inches(n_col*size_scale, n_row*size_scale)
    fig.savefig(fname, dpi=300)
    plt.close()
    # insert pdf.
    canvas = fitz.open(fname)
    session_report.insert_pdf(canvas)
    canvas.close()
    os.remove(fname)
