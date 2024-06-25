#!/usr/bin/env python3

import os
import fitz
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.gridspec import GridSpec
# fig, ax = plt.subplots(1, 1, figsize=(6, 6))
from modules import Trialization
from modules.ReadResults import read_masks
from modules.ReadResults import read_raw_voltages
from modules.ReadResults import read_dff
from modules.ReadResults import read_neural_trials
from modules.ReadResults import read_move_offset

def read_ops(session_data_path):
    ops = np.load(
        os.path.join(session_data_path, 'suite2p', 'plane0','ops.npy'),
        allow_pickle=True).item()
    ops['save_path0'] = os.path.join(session_data_path)
    return ops

from plot.fig0_beh import plotter_VIPTD_G8_beh
from plot.fig1_mask import plotter_VIPTD_G8_masks
from plot.fig2_align_percept import plotter_VIPTD_G8_align_perc
from plot.fig3_align_beh import plotter_VIPTD_G8_align_beh
from plot.fig5_raw_traces import plot_VIPTD_G8_example_traces
from plot.fig5_raw_traces import plot_roi_raw_trace
from plot.misc import plot_motion_offset_hist
from plot.misc import plot_inh_exc_label_pc
from plot.misc import plot_isi_distribution


def plot_2afc_VIPTD_G8(ops, session_data_name):

    # read data.
    print('===============================================')
    print('============ reading saved results ============')
    print('===============================================')
    [labels,
     masks,
     mean_func, max_func,
     mean_anat] = read_masks(ops)
    [vol_time,
     vol_start_bin,
     vol_stim_bin,
     vol_img_bin] = read_raw_voltages(ops)
    dff = read_dff(ops)
    neural_trials = read_neural_trials(ops)
    [xoff, yoff] = read_move_offset(ops)

    print('Processing masks')
    plotter_masks = plotter_VIPTD_G8_masks(
        labels, masks, mean_func, max_func, mean_anat)
    print('Processing behavior')
    plotter_beh = plotter_VIPTD_G8_beh(
        neural_trials)
    print('Processing behavior alignment')
    plotter_align_beh = plotter_VIPTD_G8_align_beh(
        neural_trials, labels)
    print('Processing perception alignment')
    plotter_align_perc = plotter_VIPTD_G8_align_perc(
        neural_trials, labels)

    def plot_session_report(session_data_name):
        print('Plotting session report')
        fig = plt.figure(figsize=(70, 63))
        gs = GridSpec(9, 10, figure=fig)
        # masks.
        mask_ax01 = plt.subplot(gs[0:2, 0:2])
        mask_ax02 = plt.subplot(gs[2:4, 0:2])
        mask_ax03 = plt.subplot(gs[0:2, 2:4])
        mask_ax04 = plt.subplot(gs[2:4, 2:4])
        mask_ax05 = plt.subplot(gs[0:2, 4:6])
        mask_ax06 = plt.subplot(gs[2:4, 4:6])
        plotter_masks.func_max(mask_ax01)
        plotter_masks.func_masks(mask_ax02)
        plotter_masks.anat_mean(mask_ax03)
        plotter_masks.anat_label_masks(mask_ax04)
        plotter_masks.superimpose(mask_ax05)
        plotter_masks.shared_masks(mask_ax06)
        # behavior.
        beh_ax01 = plt.subplot(gs[0, 8])
        beh_ax02 = plt.subplot(gs[1, 8])
        beh_ax03 = plt.subplot(gs[2, 8])
        beh_ax04 = plt.subplot(gs[3, 8])
        plotter_beh.outcomes_precentage(beh_ax01)
        plotter_beh.correctness_precentage(beh_ax02)
        plotter_beh.psych_func(beh_ax03)
        plotter_beh.decision_correct(beh_ax04)
        # example traces.
        example_ax = plt.subplot(gs[2:4, 6:8])
        plot_VIPTD_G8_example_traces(
            example_ax, dff, labels, vol_stim_bin, vol_img_bin, vol_time)
        # perception alignment.
        all_ax01 = plt.subplot(gs[4, 0])
        all_ax02 = plt.subplot(gs[5, 0])
        all_ax03 = plt.subplot(gs[6:8, 0])
        plotter_align_perc.all_exc(all_ax01)
        plotter_align_perc.all_inh(all_ax02)
        plotter_align_perc.all_heatmap_neuron(all_ax03)
        onset_ax01 = plt.subplot(gs[4, 1])
        onset_ax02 = plt.subplot(gs[5, 1])
        onset_ax03 = plt.subplot(gs[6:8, 1])
        plotter_align_perc.onset_exc(onset_ax01)
        plotter_align_perc.onset_inh(onset_ax02)
        plotter_align_perc.onset_heatmap_neuron(onset_ax03)
        pre_ax01 = plt.subplot(gs[4, 2])
        pre_ax02 = plt.subplot(gs[5, 2])
        pre_ax03 = plt.subplot(gs[6:8, 2])
        plotter_align_perc.pre_exc(pre_ax01)
        plotter_align_perc.pre_inh(pre_ax02)
        plotter_align_perc.pre_heatmap_neuron(pre_ax03)
        pert_ax01 = plt.subplot(gs[4, 3])
        pert_ax02 = plt.subplot(gs[5, 3])
        pert_ax03 = plt.subplot(gs[6:8, 3])
        plotter_align_perc.pert_exc(pert_ax01)
        plotter_align_perc.pert_inh(pert_ax02)
        plotter_align_perc.pert_heatmap_neuron(pert_ax03)
        post_ax01 = plt.subplot(gs[4, 4])
        post_ax02 = plt.subplot(gs[5, 4])
        plotter_align_perc.post_isi_exc(post_ax01)
        plotter_align_perc.post_isi_inh(post_ax02)
        outcome_ax01 = plt.subplot(gs[4, 5])
        outcome_ax02 = plt.subplot(gs[5, 5])
        outcome_ax03 = plt.subplot(gs[6:8, 5])
        outcome_ax04 = plt.subplot(gs[4, 6])
        outcome_ax05 = plt.subplot(gs[5, 6])
        outcome_ax06 = plt.subplot(gs[6:8, 6])
        plotter_align_perc.reward_exc(outcome_ax01)
        plotter_align_perc.reward_inh(outcome_ax02)
        plotter_align_perc.reward_heatmap_neuron(outcome_ax03)
        plotter_align_perc.punish_exc(outcome_ax04)
        plotter_align_perc.punish_inh(outcome_ax05)
        plotter_align_perc.punish_heatmap_neuron(outcome_ax06)
        # behavior alignment.
        lick_ax01 = plt.subplot(gs[4, 7])
        lick_ax02 = plt.subplot(gs[5, 7])
        lick_ax03 = plt.subplot(gs[6:8, 7])
        plotter_align_beh.all_exc(lick_ax01)
        plotter_align_beh.all_inh(lick_ax02)
        plotter_align_beh.all_heatmap_neuron(lick_ax03)
        reaction_ax01 = plt.subplot(gs[4, 8])
        reaction_ax02 = plt.subplot(gs[5, 8])
        reaction_ax03 = plt.subplot(gs[6:8, 8])
        plotter_align_beh.reaction_exc(reaction_ax01)
        plotter_align_beh.reaction_inh(reaction_ax02)
        plotter_align_beh.reaction_heatmap_neuron(reaction_ax03)
        decision_ax01 = plt.subplot(gs[4, 9])
        decision_ax02 = plt.subplot(gs[5, 9])
        decision_ax03 = plt.subplot(gs[6:8, 9])
        plotter_align_beh.decision_exc(decision_ax01)
        plotter_align_beh.decision_inh(decision_ax02)
        plotter_align_beh.decision_heatmap_neuron(decision_ax03)
        # offset.
        offset_ax = plt.subplot(gs[0, 6])
        plot_motion_offset_hist(offset_ax, xoff, yoff)
        # labels.
        label_ax = plt.subplot(gs[0, 7])
        plot_inh_exc_label_pc(label_ax, labels)
        # isi distribution.
        isi_ax = plt.subplot(gs[1, 7])
        plot_isi_distribution(isi_ax, neural_trials)
        # save figure.
        fig.set_size_inches(70, 63)
        fig.savefig(os.path.join(
            ops['save_path0'], 'figures', session_data_name),
            dpi=300)
        plt.close()
        print('Visualization for session report completed')

    def plot_individual_roi():
        print('Plotting roi report')
        roi_report = fitz.open()
        for roi_id in tqdm(np.argsort(labels, kind='stable')):
            fig = plt.figure(figsize=(56, 21))
            gs = GridSpec(3, 8, figure=fig)
            # masks.
            mask_ax01 = plt.subplot(gs[0:2, 0:2])
            mask_ax02 = plt.subplot(gs[0, 2])
            mask_ax03 = plt.subplot(gs[0, 3])
            mask_ax04 = plt.subplot(gs[1, 2])
            mask_ax05 = plt.subplot(gs[1, 3])
            plotter_masks.roi_loc(mask_ax01, roi_id)
            plotter_masks.roi_func(mask_ax02, roi_id)
            plotter_masks.roi_anat(mask_ax03, roi_id)
            plotter_masks.roi_superimpose(mask_ax04, roi_id)
            plotter_masks.roi_masks(mask_ax05, roi_id)
            # behavior alignment.
            beh_alg_ax01 = plt.subplot(gs[0, 6])
            beh_alg_ax02 = plt.subplot(gs[1, 6])
            beh_alg_ax03 = plt.subplot(gs[0, 7])
            beh_alg_ax04 = plt.subplot(gs[1, 7])
            plotter_align_beh.roi_lick_all(beh_alg_ax01, roi_id)
            plotter_align_beh.roi_lick_all_box(beh_alg_ax02, roi_id)
            plotter_align_beh.roi_lick_decision(beh_alg_ax03, roi_id)
            plotter_align_beh.roi_lick_decision_box(beh_alg_ax04, roi_id)
            # perception alignment.
            stim_ax01 = plt.subplot(gs[2, 0])
            stim_ax02 = plt.subplot(gs[2, 1])
            stim_ax03 = plt.subplot(gs[2, 2])
            stim_ax04 = plt.subplot(gs[2, 3])
            stim_ax05 = plt.subplot(gs[2, 4])
            stim_ax06 = plt.subplot(gs[2, 5])
            stim_ax07 = plt.subplot(gs[2, 6:8])
            stim_ax08 = plt.subplot(gs[0, 4])
            stim_ax09 = plt.subplot(gs[1, 4])
            stim_ax10 = plt.subplot(gs[0, 5])
            stim_ax11 = plt.subplot(gs[1, 5])
            plotter_align_perc.roi_stim_onset(stim_ax01, roi_id)
            plotter_align_perc.roi_stim_all(stim_ax02, roi_id)
            plotter_align_perc.roi_stim_pre(stim_ax03, roi_id)
            plotter_align_perc.roi_stim_perturb(stim_ax04, roi_id)
            plotter_align_perc.roi_stim_perturb_box(stim_ax05, roi_id)
            plotter_align_perc.roi_stim_post_isi(stim_ax06, roi_id)
            plotter_align_perc.roi_stim_post_isi_box(stim_ax07, roi_id)
            plotter_align_perc.roi_reward(stim_ax08, roi_id)
            plotter_align_perc.roi_reward_heatmap(stim_ax09, roi_id)
            plotter_align_perc.roi_punish(stim_ax10, roi_id)
            plotter_align_perc.roi_punish_heatmap(stim_ax11, roi_id)
            # save figure.
            fname = os.path.join(
                ops['save_path0'], 'figures',
                str(roi_id).zfill(4)+'.pdf')
            fig.set_size_inches(56, 21)
            fig.savefig(fname, dpi=300)
            plt.close()
            roi_fig = fitz.open(fname)
            roi_report.insert_pdf(roi_fig)
            roi_fig.close()
            os.remove(fname)
        roi_report.save(os.path.join(ops['save_path0'], 'figures', 'roi_report.pdf'))
        roi_report.close()


    def plot_raw_traces():
        max_ms = 300000
        if not os.path.exists(os.path.join(
                ops['save_path0'], 'figures', 'raw_traces')):
            os.makedirs(os.path.join(
                ops['save_path0'], 'figures', 'raw_traces'))
        if np.max(vol_time) < max_ms:
            trace_num_fig = 1
        else:
            trace_num_fig = int(np.max(vol_time)/max_ms)
        for roi_id in tqdm(range(len(labels))):
            fig, axs = plt.subplots(trace_num_fig, 1, figsize=(5, 5))
            plt.subplots_adjust(hspace=0.6)
            plot_roi_raw_trace(
                axs, roi_id, max_ms,
                labels, dff, vol_img_bin, vol_stim_bin, vol_time)
            fig.set_size_inches(max_ms/5000, trace_num_fig*2)
            fig.tight_layout()
            fig.savefig(os.path.join(
                ops['save_path0'], 'figures', 'raw_traces',
                str(roi_id).zfill(4)+'.pdf'),
                dpi=300)
            plt.close()

    print('===============================================')
    print('============= plot session report =============')
    print('===============================================')
    plot_session_report('session_report_{}.pdf'.format(session_data_name))
    print('===============================================')
    print('=============== plot roi report ===============')
    print('===============================================')
    #plot_individual_roi()
    print('===============================================')
    print('=============== plot raw traces ===============')
    print('===============================================')
    #plot_raw_traces()


def run(session_data_path):
    session_data_name = session_data_path.split('/')[-1]
    ops = read_ops(session_data_path)
    print('===============================================')
    print('Processing '+ session_data_name)
    print('===============================================')
    print('============= trials segmentation =============')
    print('===============================================')
    #Trialization.run(ops)
    plot_2afc_VIPTD_G8(ops, session_data_name)
    print('===============================================')
    print('Processing {} completed'.format(session_data_name))


if __name__ == "__main__":

    session_data_path = 'C:/Users/yhuang887/Projects/interval_discrimination_basic_202404/results/FN14_P_20240530_2afc_t'
    run(session_data_path)
