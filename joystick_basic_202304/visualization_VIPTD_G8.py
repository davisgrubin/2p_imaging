#!/usr/bin/env python3

import os
import fitz
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.gridspec import GridSpec

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
    r = significance['r_vis1'][roi_id] +\
        significance['r_press1'][roi_id] +\
        significance['r_retract1'][roi_id] +\
        significance['r_vis2'][roi_id] +\
        significance['r_press2'][roi_id] +\
        significance['r_reward'][roi_id] +\
        significance['r_punish'][roi_id] +\
        significance['r_lick'][roi_id]
    return r

def reset_significance(significance):
    sign = {}
    sign['r_vis1']     = np.ones_like(significance['r_vis1'])
    sign['r_press1']   = np.ones_like(significance['r_press1'])
    sign['r_retract1'] = np.ones_like(significance['r_retract1'])
    sign['r_vis2']     = np.ones_like(significance['r_vis2'])
    sign['r_press2']   = np.ones_like(significance['r_press2'])
    sign['r_reward']   = np.ones_like(significance['r_reward'])
    sign['r_punish']   = np.ones_like(significance['r_punish'])
    sign['r_lick']     = np.ones_like(significance['r_lick'])
    return sign

from plot.fig0_beh import plotter_all_beh
from plot.fig1_mask import plotter_all_masks
from plot.fig2_align_percept import plotter_VIPTD_G8_percept
from plot.fig3_align_motor import plotter_VIPTD_G8_motor
from plot.fig5_raw_traces import plot_VIPTD_G8_example_traces
from plot.fig5_raw_traces import plot_roi_example_traces
from plot.fig5_raw_traces import plot_roi_raw_trace
from plot.misc import plot_motion_offset_hist
from plot.misc import plot_inh_exc_label_pc
from plot.misc import plot_significance
from plot.misc import plot_roi_significance


def plot_js_VIPTD_G8(ops, session_data_name):
    
    def plot_session_report(session_data_name):
        fig = plt.figure(figsize=(77, 70))
        gs = GridSpec(10, 11, figure=fig)
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
        beh_ax01 = plt.subplot(gs[2, 9])
        beh_ax02 = plt.subplot(gs[4, 0])
        beh_ax03 = plt.subplot(gs[4, 1])
        beh_ax04 = plt.subplot(gs[4, 2])
        beh_ax05 = plt.subplot(gs[4, 3])
        beh_ax06 = plt.subplot(gs[4, 4])
        beh_ax07 = plt.subplot(gs[4, 5])
        plotter_beh.outcome(beh_ax01)
        plotter_beh.align_pos_vis1(beh_ax02)
        plotter_beh.align_pos_press1(beh_ax03)
        plotter_beh.align_pos_retract1(beh_ax04)
        plotter_beh.align_pos_vis2(beh_ax05)
        plotter_beh.align_pos_press2(beh_ax06)
        plotter_beh.align_pos_outcome(beh_ax07)
        # alignment.
        vis1_ax01 = plt.subplot(gs[5, 0])
        vis1_ax02 = plt.subplot(gs[6, 0])
        vis1_ax03 = plt.subplot(gs[7:9, 0])
        plotter_percept.vis1_outcome_exc(vis1_ax01)
        plotter_percept.vis1_outcome_inh(vis1_ax02)
        plotter_percept.vis1_heatmap_neuron(vis1_ax03)
        press1_ax01 = plt.subplot(gs[5, 1])
        press1_ax02 = plt.subplot(gs[6, 1])
        press1_ax03 = plt.subplot(gs[7:9, 1])
        plotter_motor.press1_exc(press1_ax01)
        plotter_motor.press1_inh(press1_ax02)
        plotter_motor.press1_heatmap_neuron(press1_ax03)
        retract1_ax01 = plt.subplot(gs[5, 2])
        retract1_ax02 = plt.subplot(gs[6, 2])
        retract1_ax03 = plt.subplot(gs[7:9, 2])
        plotter_motor.retract1_exc(retract1_ax01)
        plotter_motor.retract1_inh(retract1_ax02)
        plotter_motor.retract1_heatmap_neuron(retract1_ax03)
        vis2_ax01 = plt.subplot(gs[5, 3])
        vis2_ax02 = plt.subplot(gs[6, 3])
        vis2_ax03 = plt.subplot(gs[7:9, 3])
        plotter_percept.vis2_outcome_exc(vis2_ax01)
        plotter_percept.vis2_outcome_inh(vis2_ax02)
        plotter_percept.vis2_heatmap_neuron(vis2_ax03)
        press2_ax01 = plt.subplot(gs[5, 4])
        press2_ax02 = plt.subplot(gs[6, 4])
        press2_ax03 = plt.subplot(gs[7:9, 4])
        plotter_motor.press2_exc(press2_ax01)
        plotter_motor.press2_inh(press2_ax02)
        plotter_motor.press2_heatmap_neuron(press2_ax03)
        outcome_ax01 = plt.subplot(gs[5, 5])
        outcome_ax02 = plt.subplot(gs[6, 5])
        outcome_ax03 = plt.subplot(gs[7:9, 5])
        outcome_ax04 = plt.subplot(gs[5, 6])
        outcome_ax05 = plt.subplot(gs[6, 6])
        outcome_ax06 = plt.subplot(gs[7:9, 6])
        plotter_percept.reward_exc(outcome_ax01)
        plotter_percept.reward_inh(outcome_ax02)
        plotter_percept.reward_heatmap_neuron(outcome_ax03)
        plotter_percept.punish_exc(outcome_ax04)
        plotter_percept.punish_inh(outcome_ax05)
        plotter_percept.punish_heatmap_neuron(outcome_ax06)
        lick_ax01 = plt.subplot(gs[5, 7])
        lick_ax02 = plt.subplot(gs[6, 7])
        lick_ax03 = plt.subplot(gs[7:9, 7])
        plotter_motor.lick_exc(lick_ax01)
        plotter_motor.lick_inh(lick_ax02)
        plotter_motor.lick_heatmap_neuron(lick_ax03)
        # example traces.
        example_ax = plt.subplot(gs[0:3, 8])
        plot_VIPTD_G8_example_traces(
            example_ax, dff, labels, vol_img_bin, vol_time)
        # offset.
        offset_ax = plt.subplot(gs[0, 9])
        plot_motion_offset_hist(offset_ax, xoff, yoff)
        # labels.
        label_ax = plt.subplot(gs[1, 9])
        plot_inh_exc_label_pc(label_ax, labels)
        # significance.
        sign_ax = plt.subplot(gs[3, 8:10])
        plot_significance(sign_ax, significance)
        # save figure.
        fig.set_size_inches(77, 70)
        fig.savefig(os.path.join(
            ops['save_path0'], 'figures', session_data_name),
            dpi=300)
        plt.close()

    def plot_individual_roi(session_data_name):
        roi_report = fitz.open()
        for roi_id in tqdm(np.argsort(labels, kind='stable')):
            if get_roi_sign(significance, roi_id):
                fig = plt.figure(figsize=(56, 35))
                gs = GridSpec(5, 8, figure=fig)
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
                # alignments.
                vis1_ax01 = plt.subplot(gs[2, 0])
                vis1_ax02 = plt.subplot(gs[3, 0])
                vis1_ax03 = plt.subplot(gs[4, 0])
                plotter_percept.roi_vis1_outcome(vis1_ax01, roi_id)
                plotter_percept.roi_vis1_outcome_box(vis1_ax02, roi_id)
                plotter_percept.roi_vis1_heatmap_trials(vis1_ax03, roi_id)
                press1_ax01 = plt.subplot(gs[2, 1])
                press1_ax02 = plt.subplot(gs[3, 1])
                press1_ax03 = plt.subplot(gs[4, 1])
                plotter_motor.roi_press1(press1_ax01, roi_id)
                plotter_motor.roi_press1_box(press1_ax02, roi_id)
                plotter_motor.roi_press1_heatmap_trials(press1_ax03, roi_id)
                retract1_ax01 = plt.subplot(gs[2, 2])
                retract1_ax02 = plt.subplot(gs[3, 2])
                retract1_ax03 = plt.subplot(gs[4, 2])
                plotter_motor.roi_retract1(retract1_ax01, roi_id)
                plotter_motor.roi_retract1_box(retract1_ax02, roi_id)
                plotter_motor.roi_retract1_heatmap_trials(retract1_ax03, roi_id)
                vis2_ax01 = plt.subplot(gs[2, 3])
                vis2_ax02 = plt.subplot(gs[3, 3])
                vis2_ax03 = plt.subplot(gs[4, 3])
                plotter_percept.roi_vis2_outcome(vis2_ax01, roi_id)
                plotter_percept.roi_vis2_outcome_box(vis2_ax02, roi_id)
                plotter_percept.roi_vis2_heatmap_trials(vis2_ax03, roi_id)
                press2_ax01 = plt.subplot(gs[2, 4])
                press2_ax02 = plt.subplot(gs[3, 4])
                press2_ax03 = plt.subplot(gs[4, 4])
                plotter_motor.roi_press2(press2_ax01, roi_id)
                plotter_motor.roi_press2_box(press2_ax02, roi_id)
                plotter_motor.roi_press2_heatmap_trials(press2_ax03, roi_id)
                reward_ax01 = plt.subplot(gs[2, 5])
                reward_ax02 = plt.subplot(gs[3, 5])
                reward_ax03 = plt.subplot(gs[4, 5])
                plotter_percept.roi_reward(reward_ax01, roi_id)
                plotter_percept.roi_reward_box(reward_ax02, roi_id)
                plotter_percept.roi_reward_heatmap_trials(reward_ax03, roi_id)
                punish_ax01 = plt.subplot(gs[2, 6])
                punish_ax02 = plt.subplot(gs[3, 6])
                punish_ax03 = plt.subplot(gs[4, 6])
                plotter_percept.roi_punish(punish_ax01, roi_id)
                plotter_percept.roi_punish_box(punish_ax02, roi_id)
                plotter_percept.roi_punish_heatmap_trials(punish_ax03, roi_id)
                lick_ax01 = plt.subplot(gs[2, 7])
                lick_ax02 = plt.subplot(gs[3, 7])
                lick_ax03 = plt.subplot(gs[4, 7])
                plotter_motor.roi_lick(lick_ax01, roi_id)
                plotter_motor.roi_lick_box(lick_ax02, roi_id)
                plotter_motor.roi_lick_heatmap_trials(lick_ax03, roi_id)
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
                fig.set_size_inches(56, 35)
                fig.savefig(fname, dpi=300)
                plt.close()
                roi_fig = fitz.open(fname)
                roi_report.insert_pdf(roi_fig)
                roi_fig.close()
                os.remove(fname)
        roi_report.save(
            os.path.join(ops['save_path0'], 'figures', session_data_name))
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
    plotter_beh = plotter_all_beh(
        neural_trials)
    print('Processing perception')
    plotter_percept = plotter_VIPTD_G8_percept(
        neural_trials, labels, significance)
    print('Processing locomotion')
    plotter_motor = plotter_VIPTD_G8_motor(
        neural_trials, labels, significance)
    print('===============================================')
    print('====== plot session report with all ROIs ======')
    print('===============================================')
    plot_session_report('session_report_{}.pdf'.format(session_data_name))
    print('===============================================')
    print('=============== plot roi report ===============')
    print('===============================================')
    plot_individual_roi('roi_report_{}.pdf'.format(session_data_name))
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
    plot_js_VIPTD_G8(ops, session_data_name)
    print('===============================================')
    print('Processing {} completed'.format(session_data_name))
    
    
if __name__ == "__main__":
    RESET_SIGNIFICANCE = False
    
    session_data_path = 'C:/Users/yhuang887/Projects/joystick_basic_202304/results/FN16_P_20240603_js_t'
    run(session_data_path)

    