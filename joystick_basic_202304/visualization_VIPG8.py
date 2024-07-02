#!/usr/bin/env python3

import os
import fitz
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.gridspec import GridSpec

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

from plot.fig0_beh import plotter_all_beh
from plot.fig1_mask import plotter_all_masks
from plot.fig2_align_percept import plotter_VIPG8_percept
from plot.fig3_align_motor import plotter_VIPG8_motor
from plot.fig5_raw_traces import plot_VIPG8_example_traces
from plot.misc import plot_motion_offset_hist
from plot.misc import plot_inh_exc_label_pc


def plot_js_VIPG8(ops, session_data_name):
    
    def plot_session_report(session_data_name):
        fig = plt.figure(figsize=(70, 35))
        gs = GridSpec(5, 10, figure=fig)
        # masks.
        mask_ax01 = plt.subplot(gs[0:2, 0:2])
        mask_ax02 = plt.subplot(gs[0:2, 2:4])
        mask_ax03 = plt.subplot(gs[0:2, 4:6])
        plotter_masks.func(mask_ax01, 'mean')
        plotter_masks.func(mask_ax02, 'max')
        plotter_masks.func_masks(mask_ax03)
        # behavior.
        beh_ax01 = plt.subplot(gs[1, 7])
        beh_ax02 = plt.subplot(gs[1, 8])
        beh_ax03 = plt.subplot(gs[2, 7])
        beh_ax04 = plt.subplot(gs[3, 7])
        beh_ax05 = plt.subplot(gs[2, 8])
        beh_ax06 = plt.subplot(gs[3, 8])
        plotter_beh.outcome(beh_ax01)
        plotter_beh.align_pos_outcome(beh_ax02)
        plotter_beh.align_pos_vis1(beh_ax03)
        plotter_beh.align_pos_vis2(beh_ax04)
        plotter_beh.align_pos_press1(beh_ax05)
        plotter_beh.align_pos_press2(beh_ax06)
        # perception.
        stim_ax01 = plt.subplot(gs[2, 0])
        stim_ax02 = plt.subplot(gs[3:5, 0])
        stim_ax03 = plt.subplot(gs[2, 1])
        stim_ax04 = plt.subplot(gs[3:5, 1])
        plotter_percept.vis1_outcome(stim_ax01)
        plotter_percept.vis1_heatmap_neuron(stim_ax02)
        plotter_percept.vis2_outcome(stim_ax03)
        plotter_percept.vis2_heatmap_neuron(stim_ax04)
        outcome_ax02 = plt.subplot(gs[2, 2])
        outcome_ax03 = plt.subplot(gs[3:5, 2])
        outcome_ax05 = plt.subplot(gs[2, 3])
        outcome_ax06 = plt.subplot(gs[3:5, 3])
        plotter_percept.reward(outcome_ax02)
        plotter_percept.reward_heatmap_neuron(outcome_ax03)
        plotter_percept.punish(outcome_ax05)
        plotter_percept.punish_heatmap_neuron(outcome_ax06)
        # locomotion.
        lick_ax01 = plt.subplot(gs[2, 4])
        lick_ax02 = plt.subplot(gs[3:5, 4])
        plotter_motor.lick(lick_ax01)
        plotter_motor.lick_heatmap_neuron(lick_ax02)
        press_ax01 = plt.subplot(gs[2, 5])
        press_ax02 = plt.subplot(gs[3:5, 5])
        press_ax03 = plt.subplot(gs[2, 6])
        press_ax04 = plt.subplot(gs[3:5, 6])
        plotter_motor.press1(press_ax01)
        plotter_motor.press1_heatmap_neuron(press_ax02)
        plotter_motor.press2(press_ax03)
        plotter_motor.press2_heatmap_neuron(press_ax04)
        # example traces.
        example_ax = plt.subplot(gs[0:2, 6])
        plot_VIPG8_example_traces(
            example_ax, dff, labels, vol_img_bin, vol_time)
        # offset.
        offset_ax = plt.subplot(gs[0, 7])
        plot_motion_offset_hist(offset_ax, xoff, yoff)
        # labels.
        label_ax = plt.subplot(gs[0, 8])
        plot_inh_exc_label_pc(label_ax, labels)
        # save figure.
        fig.set_size_inches(70, 35)
        fig.savefig(os.path.join(
            ops['save_path0'], 'figures', session_data_name),
            dpi=300)
        plt.close()

    def plot_individual_roi():
        roi_report = fitz.open()
        for roi_id in tqdm(np.argsort(labels, kind='stable')[:2]):
            fig = plt.figure(figsize=(56, 35))
            gs = GridSpec(5, 8, figure=fig)
            # masks.
            mask_ax01 = plt.subplot(gs[0:2, 0:2])
            mask_ax02 = plt.subplot(gs[0, 2])
            mask_ax03 = plt.subplot(gs[0, 3])
            mask_ax04 = plt.subplot(gs[1, 2])
            mask_ax05 = plt.subplot(gs[1, 3])
            plotter_masks.roi_loc_2chan(mask_ax01, roi_id, 'max')
            plotter_masks.roi_func(mask_ax02, roi_id, 'max')
            plotter_masks.roi_anat(mask_ax03, roi_id)
            plotter_masks.roi_superimpose(mask_ax04, roi_id, 'max')
            plotter_masks.roi_masks(mask_ax05, roi_id)

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
            os.path.join(ops['save_path0'], 'figures', 'roi_report_{}.pdf'.format(
            session_data_name)))
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
    labels = np.ones_like(labels)
    [vol_time,
     vol_start_bin,
     vol_stim_bin,
     vol_img_bin] = read_raw_voltages(ops)
    dff = read_dff(ops)
    neural_trials = read_neural_trials(ops)
    [xoff, yoff] = read_move_offset(ops)
    print('Processing masks')
    plotter_masks = plotter_all_masks(
        labels, masks, mean_func, max_func, mean_anat, masks_anat)
    print('Processing perception')
    plotter_percept = plotter_VIPG8_percept(
        neural_trials, labels)
    print('Processing locomotion')
    plotter_motor = plotter_VIPG8_motor(
        neural_trials, labels)
    print('Processing behavior')
    plotter_beh = plotter_all_beh(
        neural_trials)
    print('===============================================')
    print('====== plot session report with all ROIs ======')
    print('===============================================')
    plot_session_report('session_report_all_{}.pdf'.format(session_data_name))
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
    Trialization.run(ops)
    plot_js_VIPG8(ops, session_data_name)
    print('===============================================')
    print('Processing {} completed'.format(session_data_name))
    
    
if __name__ == "__main__":
    
    session_data_path = 'C:/Users/yhuang887/Projects/joystick_basic_202304/results/VG01_P_20240530_js_DCNCNO_t'
    run(session_data_path)

    