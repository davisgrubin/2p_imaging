#!/usr/bin/env python3

import warnings
warnings.filterwarnings('ignore')

import os
import fitz
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from plot.misc import plot_sess_name
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

def run(
        session_config, session_report,
        list_labels, list_vol, list_dff, list_neural_trials, list_significance
        ):
    target_sess = 'short_long'
    n_row = 5
    n_col = 20
    size_scale = 7
    # filter data.
    idx = np.array(list(session_config['list_session_name'].values())) == target_sess
    sess_names = np.array(list(session_config['list_session_name'].keys()))[idx].copy().tolist()
    list_labels = np.array(list_labels,dtype='object')[idx].copy().tolist()
    list_vol = np.array(list_vol,dtype='object')[idx].copy().tolist()
    list_dff = np.array(list_dff,dtype='object')[idx].copy().tolist()
    list_neural_trials = np.array(list_neural_trials,dtype='object')[idx].copy().tolist()
    list_significance = np.array(list_significance,dtype='object')[idx].copy().tolist()
    print('Found {} {} sessions'.format(len(sess_names), target_sess))
    if len(sess_names) > 0:
        # create canvas.
        fig = plt.figure(figsize=(n_col*size_scale, n_row*size_scale), layout='tight')
        gs = GridSpec(n_row, n_col, figure=fig)
        # used sessions.
        sess_ax = plt.subplot(gs[0, 0])
        plot_sess_name(sess_ax, sess_names, target_sess)
        # significance.
        sign_ax = plt.subplot(gs[1, 0])
        plot_significance(sign_ax, list_significance, list_labels)
        # stimulus types.
        print('Plotting stimulus type fractions')
        type_ax01 = plt.subplot(gs[0, 1])
        type_ax02 = plt.subplot(gs[0, 2])
        type_ax03 = plt.subplot(gs[0, 3])
        type_ax04 = plt.subplot(gs[0, 4])
        plot_standard_type(type_ax01, list_neural_trials)
        plot_fix_jitter_type(type_ax02, list_neural_trials)
        plot_oddball_type(type_ax03, list_neural_trials)
        plot_random_type(type_ax04, list_neural_trials)
        # interval distributions.
        print('Plotting interval distributions')
        isi_ax01 = plt.subplot(gs[1, 1])
        isi_ax02 = plt.subplot(gs[1, 2])
        isi_ax03 = plt.subplot(gs[1, 3])
        isi_ax04 = plt.subplot(gs[1, 4])
        plot_standard_isi_distribution(isi_ax01, list_neural_trials)
        plot_jitter_isi_distribution(isi_ax02, list_neural_trials)
        plot_oddball_isi_distribution(isi_ax03, list_neural_trials)
        plot_random_isi_distribution(isi_ax04, list_neural_trials)
        # trial structure.
        print('Plotting trial structure')
        trial_ax01 = plt.subplot(gs[2, 0])
        trial_ax02 = plt.subplot(gs[2, 1:5])
        plot_stim_type(trial_ax01, list_neural_trials)
        plot_stim_label(trial_ax02, list_neural_trials)
        # save temp file.
        fname = os.path.join('results', target_sess+'.pdf')
        fig.set_size_inches(n_col*size_scale, n_row*size_scale)
        fig.savefig(fname, dpi=300)
        plt.close()
        # insert pdf.
        canvas = fitz.open(fname)
        session_report.insert_pdf(canvas)
        canvas.close()
        os.remove(fname)
