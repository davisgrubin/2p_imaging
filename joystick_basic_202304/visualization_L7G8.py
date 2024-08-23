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
    r = significance['r_vis'][roi_id] +\
        significance['r_push'][roi_id] +\
        significance['r_retract'][roi_id] +\
        significance['r_wait'][roi_id] +\
        significance['r_reward'][roi_id] +\
        significance['r_punish'][roi_id] +\
        significance['r_lick'][roi_id]
    return r

def reset_significant_roi(significance):
    sign = {}
    sign['r_vis']     = np.ones_like(significance['r_vis'])
    sign['r_push']    = np.ones_like(significance['r_push'])
    sign['r_retract'] = np.ones_like(significance['r_retract'])
    sign['r_wait']    = np.ones_like(significance['r_wait'])
    sign['r_reward']  = np.ones_like(significance['r_reward'])
    sign['r_punish']  = np.ones_like(significance['r_punish'])
    sign['r_lick']    = np.ones_like(significance['r_lick'])
    return sign

from plot.fig0_beh import plotter_all_beh
from plot.fig1_mask import plotter_all_masks
from plot.fig2_align_percept import plotter_L7G8_percept
from plot.fig3_align_motor import plotter_L7G8_motor
from plot.fig4_model import plotter_L7G8_model
from plot.fig5_raw_traces import plot_L7G8_example_traces
from plot.fig5_raw_traces import plot_roi_example_traces
from plot.fig5_raw_traces import plot_roi_raw_trace
from plot.misc import plot_motion_offset_hist
from plot.misc import plot_inh_exc_label_pc
from plot.misc import plot_significance
from plot.misc import plot_roi_significance


def plot_js_L7G8(ops, session_data_name):
    
    def plot_session_report():
        fig = plt.figure(figsize=(77, 168))
        gs = GridSpec(24, 11, figure=fig)
        # masks.
        mask_ax01 = plt.subplot(gs[0:2, 0:2])
        mask_ax02 = plt.subplot(gs[0:2, 2:4])
        mask_ax03 = plt.subplot(gs[0:2, 4:6])
        mask_ax04 = plt.subplot(gs[0:2, 6:8])
        plotter_masks.func(mask_ax01, 'mean', with_mask=False)
        plotter_masks.func(mask_ax02, 'mean')
        plotter_masks.func(mask_ax03, 'max')
        plotter_masks.func_masks_color(mask_ax04)
        # behavior.
        beh_misc_ax01 = plt.subplot(gs[2, 9:11])
        beh_misc_ax02 = plt.subplot(gs[3, 10])
        plotter_beh.delay_dist(beh_misc_ax01)
        plotter_beh.session_outcome(beh_misc_ax02)
        beh_js_short_ax01 = plt.subplot(gs[2, 0])
        beh_js_short_ax02 = plt.subplot(gs[2, 1])
        beh_js_short_ax03 = plt.subplot(gs[2, 2])
        beh_js_short_ax04 = plt.subplot(gs[2, 3])
        beh_js_short_ax05 = plt.subplot(gs[2, 4])
        beh_js_short_ax06 = plt.subplot(gs[2, 5])
        beh_js_short_ax07 = plt.subplot(gs[2, 6])
        beh_js_short_ax08 = plt.subplot(gs[2, 7])
        plotter_beh.short_align_pos_vis1(beh_js_short_ax01)
        plotter_beh.short_align_pos_push1(beh_js_short_ax02)
        plotter_beh.short_align_pos_retract1(beh_js_short_ax03)
        plotter_beh.short_align_pos_vis2(beh_js_short_ax04)
        plotter_beh.short_align_pos_wait2(beh_js_short_ax05)
        plotter_beh.short_align_pos_push2(beh_js_short_ax06)
        plotter_beh.short_align_pos_retract2(beh_js_short_ax07)
        plotter_beh.short_align_pos_outcome(beh_js_short_ax08)
        beh_js_long_ax01 = plt.subplot(gs[6, 0])
        beh_js_long_ax02 = plt.subplot(gs[6, 1])
        beh_js_long_ax03 = plt.subplot(gs[6, 2])
        beh_js_long_ax04 = plt.subplot(gs[6, 3])
        beh_js_long_ax05 = plt.subplot(gs[6, 4])
        beh_js_long_ax06 = plt.subplot(gs[6, 5])
        beh_js_long_ax07 = plt.subplot(gs[6, 6])
        beh_js_long_ax08 = plt.subplot(gs[6, 7])
        plotter_beh.long_align_pos_vis1(beh_js_long_ax01)
        plotter_beh.long_align_pos_push1(beh_js_long_ax02)
        plotter_beh.long_align_pos_retract1(beh_js_long_ax03)
        plotter_beh.long_align_pos_vis2(beh_js_long_ax04)
        plotter_beh.long_align_pos_wait2(beh_js_long_ax05)
        plotter_beh.long_align_pos_push2(beh_js_long_ax06)
        plotter_beh.long_align_pos_retract2(beh_js_long_ax07)
        plotter_beh.long_align_pos_outcome(beh_js_long_ax08)
        # short.
        vis1_ax01 = plt.subplot(gs[3, 0])
        vis1_ax02 = plt.subplot(gs[4, 0])
        plotter_percept.short_vis1_outcome(vis1_ax01)
        plotter_percept.short_vis1_heatmap_neuron(vis1_ax02)
        push1_ax01 = plt.subplot(gs[3, 1])
        push1_ax02 = plt.subplot(gs[4, 1])
        plotter_motor.short_push1(push1_ax01)
        plotter_motor.short_push1_heatmap_neuron(push1_ax02)
        retract1_ax01 = plt.subplot(gs[3, 2])
        retract1_ax02 = plt.subplot(gs[4, 2])
        plotter_motor.short_retract1(retract1_ax01)
        plotter_motor.short_retract1_heatmap_neuron(retract1_ax02)
        vis2_ax01 = plt.subplot(gs[3, 3])
        vis2_ax02 = plt.subplot(gs[4, 3])
        plotter_percept.short_vis2_outcome(vis2_ax01)
        plotter_percept.short_vis2_heatmap_neuron(vis2_ax02)
        wait2_ax01 = plt.subplot(gs[3, 4])
        wait2_ax02 = plt.subplot(gs[4, 4])
        plotter_motor.short_wait2(wait2_ax01)
        plotter_motor.short_wait2_heatmap_neuron(wait2_ax02)
        push2_ax01 = plt.subplot(gs[3, 5])
        push2_ax02 = plt.subplot(gs[4, 5])
        plotter_motor.short_push2(push2_ax01)
        plotter_motor.short_push2_heatmap_neuron(push2_ax02)
        retract2_ax01 = plt.subplot(gs[3, 6])
        retract2_ax02 = plt.subplot(gs[4, 6])
        plotter_motor.short_retract2(retract2_ax01)
        plotter_motor.short_retract2_heatmap_neuron(retract2_ax02)
        outcome_ax01 = plt.subplot(gs[3, 7])
        outcome_ax02 = plt.subplot(gs[4, 7])
        outcome_ax03 = plt.subplot(gs[3, 8])
        outcome_ax04 = plt.subplot(gs[4, 8])
        plotter_percept.short_reward(outcome_ax01)
        plotter_percept.short_reward_heatmap_neuron(outcome_ax02)
        plotter_percept.short_punish(outcome_ax03)
        plotter_percept.short_punish_heatmap_neuron(outcome_ax04)
        # long.
        vis1_ax01 = plt.subplot(gs[7, 0])
        vis1_ax02 = plt.subplot(gs[8, 0])
        plotter_percept.long_vis1_outcome(vis1_ax01)
        plotter_percept.long_vis1_heatmap_neuron(vis1_ax02)
        push1_ax01 = plt.subplot(gs[7, 1])
        push1_ax02 = plt.subplot(gs[8, 1])
        plotter_motor.long_push1(push1_ax01)
        plotter_motor.long_push1_heatmap_neuron(push1_ax02)
        retract1_ax01 = plt.subplot(gs[7, 2])
        retract1_ax02 = plt.subplot(gs[8, 2])
        plotter_motor.long_retract1(retract1_ax01)
        plotter_motor.long_retract1_heatmap_neuron(retract1_ax02)
        vis2_ax01 = plt.subplot(gs[7, 3])
        vis2_ax02 = plt.subplot(gs[8, 3])
        plotter_percept.long_vis2_outcome(vis2_ax01)
        plotter_percept.long_vis2_heatmap_neuron(vis2_ax02)
        wait2_ax01 = plt.subplot(gs[7, 4])
        wait2_ax02 = plt.subplot(gs[8, 4])
        plotter_motor.long_wait2(wait2_ax01)
        plotter_motor.long_wait2_heatmap_neuron(wait2_ax02)
        push2_ax01 = plt.subplot(gs[7, 5])
        push2_ax02 = plt.subplot(gs[8, 5])
        plotter_motor.long_push2(push2_ax01)
        plotter_motor.long_push2_heatmap_neuron(push2_ax02)
        retract2_ax01 = plt.subplot(gs[7, 6])
        retract2_ax02 = plt.subplot(gs[8, 6])
        plotter_motor.long_retract2(retract2_ax01)
        plotter_motor.long_retract2_heatmap_neuron(retract2_ax02)
        outcome_ax01 = plt.subplot(gs[7, 7])
        outcome_ax02 = plt.subplot(gs[8, 7])
        outcome_ax03 = plt.subplot(gs[7, 8])
        outcome_ax04 = plt.subplot(gs[8, 8])
        plotter_percept.long_reward(outcome_ax01)
        plotter_percept.long_reward_heatmap_neuron(outcome_ax02)
        plotter_percept.long_punish(outcome_ax03)
        plotter_percept.long_punish_heatmap_neuron(outcome_ax04)
        # epoch.
        ep_short_ax01 = plt.subplot(gs[5, 0])
        ep_short_ax03 = plt.subplot(gs[5, 1])
        ep_short_ax05 = plt.subplot(gs[5, 2])
        ep_short_ax07 = plt.subplot(gs[5, 3])
        ep_short_ax09 = plt.subplot(gs[5, 4])
        ep_short_ax11 = plt.subplot(gs[5, 5])
        ep_short_ax13 = plt.subplot(gs[5, 6])
        plotter_percept.short_epoch_vis1(ep_short_ax01)
        plotter_motor.short_epoch_push1(ep_short_ax03)
        plotter_motor.short_epoch_retract1(ep_short_ax05)
        plotter_percept.short_epoch_vis2(ep_short_ax07)
        plotter_motor.short_epoch_wait2(ep_short_ax09)
        plotter_motor.short_epoch_push2(ep_short_ax11)
        plotter_motor.short_epoch_retract2(ep_short_ax13)
        ep_long_ax01 = plt.subplot(gs[9, 0])
        ep_long_ax03 = plt.subplot(gs[9, 1])
        ep_long_ax05 = plt.subplot(gs[9, 2])
        ep_long_ax07 = plt.subplot(gs[9, 3])
        ep_long_ax09 = plt.subplot(gs[9, 4])
        ep_long_ax11 = plt.subplot(gs[9, 5])
        ep_long_ax13 = plt.subplot(gs[9, 6])
        plotter_percept.long_epoch_vis1(ep_long_ax01)
        plotter_motor.long_epoch_push1(ep_long_ax03)
        plotter_motor.long_epoch_retract1(ep_long_ax05)
        plotter_percept.long_epoch_vis2(ep_long_ax07)
        plotter_motor.long_epoch_wait2(ep_long_ax09)
        plotter_motor.long_epoch_push2(ep_long_ax11)
        plotter_motor.long_epoch_retract2(ep_long_ax13)
        # lick
        lick_ax01 = plt.subplot(gs[3, 9])
        lick_ax02 = plt.subplot(gs[4, 9])
        plotter_motor.lick(lick_ax01)
        plotter_motor.lick_heatmap_neuron(lick_ax02)
        # model.
        model_ax01 = plt.subplot(gs[10, 0:2])
        model_ax02 = plt.subplot(gs[10, 2:4])
        model_ax03 = plt.subplot(gs[10, 4:6])
        model_ax04 = plt.subplot(gs[10, 6:8])
        plotter_model.plot_block_type_population_decoding(model_ax01)
        plotter_model.plot_block_tran_decode_all(model_ax02)
        plotter_model.plot_block_tran_decode_short(model_ax03)
        plotter_model.plot_block_tran_decode_long(model_ax04)
        # example traces.
        example_ax = plt.subplot(gs[0:3, 8])
        plot_L7G8_example_traces(
            example_ax, dff, labels, vol_img, vol_time)
        # offset.
        offset_ax = plt.subplot(gs[0, 9])
        plot_motion_offset_hist(offset_ax, xoff, yoff)
        # labels.
        label_ax = plt.subplot(gs[0, 10])
        plot_inh_exc_label_pc(label_ax, labels)
        # significance.
        sign_ax = plt.subplot(gs[1, 9:11])
        plot_significance(sign_ax, significance, labels)
        # save figure.
        fig.set_size_inches(77, 168)
        fig.savefig(os.path.join(
            ops['save_path0'], 'figures',
            'session_report_{}.pdf'.format(session_data_name)),
            dpi=300)
        plt.close()

    def plot_individual_roi():
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
                push1_ax01 = plt.subplot(gs[2, 1])
                push1_ax02 = plt.subplot(gs[3, 1])
                push1_ax03 = plt.subplot(gs[4, 1])
                plotter_motor.roi_push1(push1_ax01, roi_id)
                plotter_motor.roi_push1_box(push1_ax02, roi_id)
                plotter_motor.roi_push1_heatmap_trials(push1_ax03, roi_id)
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
                push2_ax01 = plt.subplot(gs[2, 4])
                push2_ax02 = plt.subplot(gs[3, 4])
                push2_ax03 = plt.subplot(gs[4, 4])
                plotter_motor.roi_push2(push2_ax01, roi_id)
                plotter_motor.roi_push2_box(push2_ax02, roi_id)
                plotter_motor.roi_push2_heatmap_trials(push2_ax03, roi_id)
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
                plot_roi_example_traces(example_ax, dff, labels, vol_img, vol_time, roi_id)
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
        roi_report.save(os.path.join(
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
                labels, dff, vol_img, vol_stim_vis, vol_time)
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
    labels = -1*np.ones_like(labels)
    [vol_time, vol_start, vol_stim_vis, vol_img, 
     vol_hifi, vol_stim_aud, vol_flir,
     vol_pmt, vol_led] = read_raw_voltages(ops)
    dff = read_dff(ops)
    neural_trials = read_neural_trials(ops)
    [xoff, yoff] = read_move_offset(ops)
    significance = read_significance(ops)
    if reset_significance:
        significance = reset_significant_roi(significance)
    print('Processing masks')
    plotter_masks = plotter_all_masks(
        labels, masks, mean_func, max_func, mean_anat, masks_anat)
    print('Processing behavior')
    plotter_beh = plotter_all_beh(
        neural_trials, cate_delay)
    print('Processing perception')
    plotter_percept = plotter_L7G8_percept(
        neural_trials, labels, significance, cate_delay)
    print('Processing locomotion')
    plotter_motor = plotter_L7G8_motor(
        neural_trials, labels, significance, cate_delay)
    print('Processing Modeling')
    plotter_model = plotter_L7G8_model(
        neural_trials, labels, significance, cate_delay)
    print('===============================================')
    print('============= plot session report =============')
    print('===============================================')
    plot_session_report()
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
    StatTest.run(ops)
    plot_js_L7G8(ops, session_data_name)
    print('===============================================')
    print('Processing {} completed'.format(session_data_name))
    
    
if __name__ == "__main__":
    reset_significance = False
    SELF_TIME = False
    cate_delay = 40
    #delay = [neural_trials[str(i)]['trial_delay'] for i in range(len(neural_trials))]
    
    session_data_path = 'C:/Users/yhuang887/Projects/joystick_basic_202304/results/LG07_CRBL_20240804_js_t'
    run(session_data_path)

    