#!/usr/bin/env python3

import os
import fitz
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.gridspec import GridSpec
# fig, ax = plt.subplots(1, 1, figsize=(6, 6))
from modules import Trialization
from modules import StatTest
from modules.ReadResults import read_masks
from modules.ReadResults import read_raw_voltages
from modules.ReadResults import read_dff
from modules.ReadResults import read_neural_trials
from modules.ReadResults import read_move_offset
from modules.ReadResults import read_significance

def read_ops(session_data_path):
    ops = np.load(
        os.path.join(session_data_path, 'suite2p', 'plane0','ops.npy'),
        allow_pickle=True).item()
    ops['save_path0'] = os.path.join(session_data_path)
    return ops

def get_roi_sign(significance, roi_id):
    r = significance['r_stim_all'][roi_id] +\
        significance['r_stim_onset'][roi_id] +\
        significance['r_stim_pre'][roi_id] +\
        significance['r_stim_post_first'][roi_id] +\
        significance['r_stim_post_all'][roi_id] +\
        significance['r_reward'][roi_id] +\
        significance['r_punish'][roi_id] +\
        significance['r_lick_all'][roi_id] +\
        significance['r_lick_reaction'][roi_id] +\
        significance['r_lick_decision'][roi_id]
    return r

def reset_significance(significance):
    sign = {}
    sign['r_stim_all']   = np.ones_like(['r_stim_all'])
    sign['r_stim_onset'] = np.ones_like(['r_stim_onset'])
    sign['r_stim_pre']   = np.ones_like(['r_stim_pre'])
    sign['r_stim_post_first'] = np.ones_like(['r_stim_post_first'])
    sign['r_stim_post_all']   = np.ones_like(['r_stim_post_all'])
    sign['r_reward'] = np.ones_like(['r_reward'])
    sign['r_punish'] = np.ones_like(['r_punish'])
    sign['r_lick_all']      = np.ones_like(['r_lick_all'])
    sign['r_lick_reaction'] = np.ones_like(['r_lick_reaction'])
    sign['r_lick_decision'] = np.ones_like(['r_lick_decision'])
    return sign

from plot.fig0_beh import plotter_VIPTD_G8_beh
from plot.fig1_mask import plotter_all_masks
from plot.fig2_align_percept import plotter_VIPTD_G8_align_perc
from plot.fig3_align_beh import plotter_VIPTD_G8_align_beh
from plot.fig5_raw_traces import plot_VIPTD_G8_example_traces
from plot.fig5_raw_traces import plot_roi_example_traces
from plot.fig5_raw_traces import plot_roi_raw_trace
from plot.misc import plot_motion_offset_hist
from plot.misc import plot_inh_exc_label_pc
from plot.misc import plot_isi_distribution
from plot.misc import plot_significance
from plot.misc import plot_roi_significance


def plot_2afc_VIPTD_G8(ops, session_data_name):

    def plot_session_report():
        print('Plotting session report')
        fig = plt.figure(figsize=(77, 63))
        gs = GridSpec(9, 11, figure=fig)
        # masks.
        mask_ax01 = plt.subplot(gs[0:2, 0:2])
        mask_ax02 = plt.subplot(gs[2:4, 0:2])
        mask_ax03 = plt.subplot(gs[0:2, 2:4])
        mask_ax04 = plt.subplot(gs[2:4, 2:4])
        mask_ax05 = plt.subplot(gs[0:2, 4:6])
        mask_ax06 = plt.subplot(gs[2:4, 4:6])
        mask_ax07 = plt.subplot(gs[0:2, 6:8])
        mask_ax08 = plt.subplot(gs[2:4, 6:8])
        plotter_masks.func(mask_ax01, 'max')
        plotter_masks.func_masks(mask_ax02)
        plotter_masks.anat_cellpose(mask_ax03)
        plotter_masks.masks_superimpose(mask_ax04)
        plotter_masks.anat(mask_ax05)
        plotter_masks.anat_label_masks(mask_ax06)
        plotter_masks.superimpose(mask_ax07, 'max')
        plotter_masks.shared_masks(mask_ax08)
        # behavior.
        beh_ax01 = plt.subplot(gs[0, 10])
        beh_ax02 = plt.subplot(gs[1, 10])
        beh_ax03 = plt.subplot(gs[1, 9])
        beh_ax04 = plt.subplot(gs[2, 9])
        beh_ax05 = plt.subplot(gs[2, 10])
        plotter_beh.outcomes_precentage(beh_ax01)
        plotter_beh.correctness_precentage(beh_ax02)
        plotter_beh.choice_percentage(beh_ax03)
        plotter_beh.psych_func(beh_ax04)
        plotter_beh.decision_correct(beh_ax05)
        # example traces.
        example_ax = plt.subplot(gs[2:4, 8])
        plot_VIPTD_G8_example_traces(
            example_ax, dff, labels, vol_img_bin, vol_time)
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
        plotter_align_perc.pert_exc(pert_ax01)
        plotter_align_perc.pert_inh(pert_ax02)
        post_ax01 = plt.subplot(gs[4, 4])
        post_ax02 = plt.subplot(gs[5, 4])
        plotter_align_perc.post_isi_exc(post_ax01)
        plotter_align_perc.post_isi_inh(post_ax02)
        reward_ax01 = plt.subplot(gs[4, 5])
        reward_ax02 = plt.subplot(gs[5, 5])
        reward_ax03 = plt.subplot(gs[6:8, 5])
        plotter_align_perc.reward_exc(reward_ax01)
        plotter_align_perc.reward_inh(reward_ax02)
        plotter_align_perc.reward_heatmap_neuron(reward_ax03)
        punish_ax01 = plt.subplot(gs[4, 6])
        punish_ax02 = plt.subplot(gs[5, 6])
        punish_ax03 = plt.subplot(gs[6:8, 6])
        plotter_align_perc.punish_exc(punish_ax01)
        plotter_align_perc.punish_inh(punish_ax02)
        plotter_align_perc.punish_heatmap_neuron(punish_ax03)
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
        # significance.
        sign_ax = plt.subplot(gs[3, 9:11])
        plot_significance(sign_ax, significance)
        # offset.
        offset_ax = plt.subplot(gs[0, 8])
        plot_motion_offset_hist(offset_ax, xoff, yoff)
        # labels.
        label_ax = plt.subplot(gs[0, 9])
        plot_inh_exc_label_pc(label_ax, labels)
        # isi distribution.
        isi_ax = plt.subplot(gs[1, 8])
        plot_isi_distribution(isi_ax, neural_trials)
        # save figure.
        fig.set_size_inches(77, 63)
        fig.savefig(os.path.join(
            ops['save_path0'], 'figures',
            'session_report_{}.pdf'.format(session_data_name)),
            dpi=300)
        plt.close()
        print('Visualization for session report completed')

    def plot_individual_roi():
        print('Plotting roi report')
        roi_report = fitz.open()
        for roi_id in tqdm(np.argsort(labels, kind='stable')):
            if get_roi_sign(significance, roi_id):
                fig = plt.figure(figsize=(70, 35))
                gs = GridSpec(5, 10, figure=fig)
                # masks.
                mask_ax01 = plt.subplot(gs[0:2, 0:2])
                mask_ax02 = plt.subplot(gs[0, 2])
                mask_ax03 = plt.subplot(gs[0, 3])
                mask_ax04 = plt.subplot(gs[0, 4])
                mask_ax05 = plt.subplot(gs[0, 5])
                plotter_masks.roi_loc_2chan(mask_ax01, roi_id, 'max')
                plotter_masks.roi_func(mask_ax02, roi_id, 'max')
                plotter_masks.roi_anat(mask_ax03, roi_id)
                plotter_masks.roi_superimpose(mask_ax04, roi_id, 'max')
                plotter_masks.roi_masks(mask_ax05, roi_id)
                # perception alignments.
                stim_all_ax01 = plt.subplot(gs[2, 0])
                stim_all_ax02 = plt.subplot(gs[3, 0])
                stim_all_ax03 = plt.subplot(gs[4, 0])
                plotter_align_perc.roi_all(stim_all_ax01, roi_id)
                plotter_align_perc.roi_all_box(stim_all_ax02, roi_id)
                plotter_align_perc.roi_all_heatmap_trials(stim_all_ax03, roi_id)
                stim_onset_ax01 = plt.subplot(gs[2, 1])
                stim_onset_ax02 = plt.subplot(gs[3, 1])
                stim_onset_ax03 = plt.subplot(gs[4, 1])
                plotter_align_perc.roi_onset(stim_onset_ax01, roi_id)
                plotter_align_perc.roi_onset_box(stim_onset_ax02, roi_id)
                plotter_align_perc.roi_onset_heatmap_trials(stim_onset_ax03, roi_id)
                stim_pre_ax01 = plt.subplot(gs[2, 2])
                stim_pre_ax02 = plt.subplot(gs[3, 2])
                stim_pre_ax03 = plt.subplot(gs[4, 2])
                plotter_align_perc.roi_pre(stim_pre_ax01, roi_id)
                plotter_align_perc.roi_pre_box(stim_pre_ax02, roi_id)
                plotter_align_perc.roi_pre_heatmap_trials(stim_pre_ax03, roi_id)
                stim_pert_ax01 = plt.subplot(gs[2, 3])
                stim_pert_ax02 = plt.subplot(gs[3, 3])
                stim_pert_ax03 = plt.subplot(gs[4, 3])
                plotter_align_perc.roi_pert(stim_pert_ax01, roi_id)
                plotter_align_perc.roi_pert_box(stim_pert_ax02, roi_id)
                plotter_align_perc.roi_pert_heatmap_trials(stim_pert_ax03, roi_id)
                stim_post_ax01 = plt.subplot(gs[2, 4])
                plotter_align_perc.roi_post_isi(stim_post_ax01, roi_id)
                reward_ax01 = plt.subplot(gs[2, 5])
                reward_ax02 = plt.subplot(gs[3, 5])
                reward_ax03 = plt.subplot(gs[4, 5])
                plotter_align_perc.roi_reward(reward_ax01, roi_id)
                plotter_align_perc.roi_reward_box(reward_ax02, roi_id)
                plotter_align_perc.roi_reward_heatmap_trials(reward_ax03, roi_id)
                punish_ax01 = plt.subplot(gs[2, 6])
                punish_ax02 = plt.subplot(gs[3, 6])
                punish_ax03 = plt.subplot(gs[4, 6])
                plotter_align_perc.roi_punish(punish_ax01, roi_id)
                plotter_align_perc.roi_punish_box(punish_ax02, roi_id)
                plotter_align_perc.roi_punish_heatmap_trials(punish_ax03, roi_id)
                # behavior alignments.
                lick_all_ax01 = plt.subplot(gs[2, 7])
                lick_all_ax02 = plt.subplot(gs[3, 7])
                lick_all_ax03 = plt.subplot(gs[4, 7])
                plotter_align_beh.roi_all(lick_all_ax01, roi_id)
                plotter_align_beh.roi_all_box(lick_all_ax02, roi_id)
                plotter_align_beh.roi_all_heatmap_trials(lick_all_ax03, roi_id)
                lick_reaction_ax01 = plt.subplot(gs[2, 8])
                lick_reaction_ax02 = plt.subplot(gs[3, 8])
                lick_reaction_ax03 = plt.subplot(gs[4, 8])
                plotter_align_beh.roi_reaction(lick_reaction_ax01, roi_id)
                plotter_align_beh.roi_reaction_box(lick_reaction_ax02, roi_id)
                plotter_align_beh.roi_reaction_heatmap_trials(lick_reaction_ax03, roi_id)
                lick_decision_ax01 = plt.subplot(gs[2, 9])
                lick_decision_ax02 = plt.subplot(gs[3, 9])
                lick_decision_ax03 = plt.subplot(gs[4, 9])
                plotter_align_beh.roi_decision(lick_decision_ax01, roi_id)
                plotter_align_beh.roi_decision_box(lick_decision_ax02, roi_id)
                plotter_align_beh.roi_decision_heatmap_trials(lick_decision_ax03, roi_id)
                # significance.
                sign_ax = plt.subplot(gs[1, 2])
                plot_roi_significance(sign_ax, significance, roi_id)
                # example traces.
                example_ax = plt.subplot(gs[0, 6:8])
                plot_roi_example_traces(example_ax, dff, labels, vol_img_bin, vol_time, roi_id)
                # save figure.
                fname = os.path.join(
                    ops['save_path0'], 'figures',
                    str(roi_id).zfill(4)+'.pdf')
                fig.set_size_inches(70, 35)
                fig.savefig(fname, dpi=300)
                plt.close()
                roi_fig = fitz.open(fname)
                roi_report.insert_pdf(roi_fig)
                roi_fig.close()
                os.remove(fname)
        roi_report.save(
            os.path.join(
                ops['save_path0'], 'figures',
                'roi_report_{}.pdf'.format(session_data_name)))
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

    # read data.
    print('===============================================')
    print('============ reading saved results ============')
    print('===============================================')
    [labels,
     masks,
     mean_func, max_func,
     mean_anat, masks_anat] = read_masks(ops)
    [vol_time,
     vol_start_bin,
     vol_stim_bin,
     vol_img_bin] = read_raw_voltages(ops)
    dff = read_dff(ops)
    neural_trials = read_neural_trials(ops)
    [xoff, yoff] = read_move_offset(ops)
    significance = read_significance(ops)
    if RESET_SIGNIFICANCE:
        significance = reset_significance(significance)
    print('Processing masks')
    plotter_masks = plotter_all_masks(
        labels, masks, mean_func, max_func, mean_anat, masks_anat)
    print('Processing behavior')
    plotter_beh = plotter_VIPTD_G8_beh(
        neural_trials)
    print('Processing behavior alignment')
    plotter_align_beh = plotter_VIPTD_G8_align_beh(
        neural_trials, labels, significance)
    print('Processing perception alignment')
    plotter_align_perc = plotter_VIPTD_G8_align_perc(
        neural_trials, labels, significance)
    print('===============================================')
    print('============= plot session report =============')
    print('===============================================')
    plot_session_report()
    print('===============================================')
    print('=============== plot roi report ===============')
    print('===============================================')
    plot_individual_roi()
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
    Trialization.run(ops)
    StatTest.run(ops)
    plot_2afc_VIPTD_G8(ops, session_data_name)
    print('===============================================')
    print('Processing {} completed'.format(session_data_name))


if __name__ == "__main__":
    RESET_SIGNIFICANCE = False
    
    session_data_path = 'C:/Users/yhuang887/Projects/interval_discrimination_basic_202404/results/FN14_P_20240530_2afc_t'
    run(session_data_path)
