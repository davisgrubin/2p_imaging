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
from plot.fig2_align_percept import plotter_VIPG8_percept
from plot.fig3_align_motor import plotter_VIPG8_motor
from plot.fig4_model import plotter_VIPG8_model
from plot.fig5_raw_traces import plot_VIPG8_example_traces
from plot.fig5_raw_traces import plot_roi_example_traces
from plot.fig5_raw_traces import plot_roi_raw_trace
from plot.misc import plot_motion_offset_hist
from plot.misc import plot_inh_exc_label_pc
from plot.misc import plot_significance
from plot.misc import plot_roi_significance


def plot_js_VIPG8(ops, session_data_name):
    
    def plot_session_report():
        fig = plt.figure(figsize=(105, 175))
        gs = GridSpec(25, 15, figure=fig)
        # masks.
        mask_ax01 = plt.subplot(gs[0:2, 0:2])
        mask_ax02 = plt.subplot(gs[0:2, 2:4])
        mask_ax03 = plt.subplot(gs[0:2, 4:6])
        mask_ax04 = plt.subplot(gs[0:2, 6:8])
        plotter_masks.func(mask_ax01, 'mean')
        plotter_masks.func(mask_ax02, 'max')
        plotter_masks.func_masks_color(mask_ax03)
        plotter_masks.func(mask_ax04, 'mean', with_mask=False)
        # behavior.
        beh_misc_ax01 = plt.subplot(gs[5, 10:12])
        beh_misc_ax02 = plt.subplot(gs[6, 10])
        beh_js_short_axs = [plt.subplot(gs[3, i]) for i in range(9)]
        beh_js_long_axs  = [plt.subplot(gs[9, i]) for i in range(9)]
        beh_js_epoch_short_axs = [plt.subplot(gs[6, i]) for i in range(8)]
        beh_js_epoch_long_axs  = [plt.subplot(gs[12, i]) for i in range(8)]
        beh_js_onset_ax = plt.subplot(gs[15, 2])
        plotter_beh.delay_dist(beh_misc_ax01)
        plotter_beh.session_outcome(beh_misc_ax02)
        plotter_beh.all_short_align(beh_js_short_axs)
        plotter_beh.all_long_align(beh_js_long_axs)
        plotter_beh.all_short_epoch(beh_js_epoch_short_axs)
        plotter_beh.all_long_epoch(beh_js_epoch_long_axs)
        plotter_beh.onset(beh_js_onset_ax)
        # short.
        percept_axs01 = [plt.subplot(gs[4, i]) for i in [0,3,6,7]]
        percept_axs02 = [plt.subplot(gs[5, i]) for i in [0,3,6,7]]
        plotter_percept.all_short_percept_align(percept_axs01)
        plotter_percept.all_short_percept_align_heatmap_neuron(percept_axs02)
        motor_axs01 = [plt.subplot(gs[4, i]) for i in [1,2,4,5,8]]
        motor_axs02 = [plt.subplot(gs[5, i]) for i in [1,2,4,5,8]]
        plotter_motor.all_short_motor_align(motor_axs01)
        plotter_motor.all_short_motor_align_heatmap_neuron(motor_axs02)
        # long.
        percept_axs01 = [plt.subplot(gs[10, i]) for i in [0,3,6,7]]
        percept_axs02 = [plt.subplot(gs[11, i]) for i in [0,3,6,7]]
        plotter_percept.all_long_percept_align(percept_axs01)
        plotter_percept.all_long_percept_align_heatmap_neuron(percept_axs02)
        motor_axs01 = [plt.subplot(gs[10, i]) for i in [1,2,4,5,8]]
        motor_axs02 = [plt.subplot(gs[11, i]) for i in [1,2,4,5,8]]
        plotter_motor.all_long_motor_align(motor_axs01)
        plotter_motor.all_long_motor_align_heatmap_neuron(motor_axs02)
        # epoch.
        ep_short_axs01 = [plt.subplot(gs[7, i]) for i in [0,3,6]]
        ep_short_axs02 = [plt.subplot(gs[7, i]) for i in [1,2,4,5,7]]
        plotter_percept.all_short_epoch_percept_align(ep_short_axs01)
        plotter_motor.all_short_epoch_motor_align(ep_short_axs02)
        ep_long_axs01 = [plt.subplot(gs[13, i]) for i in [0,3,6]]
        ep_long_axs02 = [plt.subplot(gs[13, i]) for i in [1,2,4,5,7]]
        plotter_percept.all_long_epoch_percept_align(ep_long_axs01)
        plotter_motor.all_long_epoch_motor_align(ep_long_axs02)
        # push onset.
        push_ax01 = plt.subplot(gs[15, 2])
        push_ax02 = [plt.subplot(gs[15, i]) for i in [2,3,4]]
        plotter_motor.onset(push_ax01)
        plotter_motor.onset_heatmap_neuron(push_ax02)
        # lick.
        lick_ax01 = plt.subplot(gs[15, 6])
        lick_ax02 = plt.subplot(gs[15, 7])
        plotter_motor.lick(lick_ax01)
        plotter_motor.lick_heatmap_neuron(lick_ax02)
        # model.
        decode_axs = [[plt.subplot(gs[10, 10:12]), plt.subplot(gs[11, 10:12])],
                      [plt.subplot(gs[12, 10:12]), plt.subplot(gs[13, 10:12])],
                      [plt.subplot(gs[14, 10:12]), plt.subplot(gs[15, 10:12])],
                      [plt.subplot(gs[16, 10:12]), plt.subplot(gs[17, 10:12])],
                      [plt.subplot(gs[18, 10:12]), plt.subplot(gs[19, 10:12])],
                      [plt.subplot(gs[20, 10:12]), plt.subplot(gs[21, 10:12])]]
        model_axs07 = [plt.subplot(gs[10, i]) for i in [12,13,14]]
        model_axs08 = [plt.subplot(gs[16, i]) for i in [12,13,14]]
        model_axs09 = [plt.subplot(gs[11, i]) for i in [12,13,14]]
        model_axs10 = [plt.subplot(gs[17, i]) for i in [12,13,14]]
        plotter_model.all_decode(decode_axs)
        plotter_model.block_type_population_pca(model_axs07)
        plotter_model.block_tran_population_pca(model_axs08)
        plotter_model.block_type_dynamics(model_axs09)
        plotter_model.block_tran_dynamics(model_axs10)
        # example traces.
        example_ax = plt.subplot(gs[0:2, 8])
        plot_VIPG8_example_traces(
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
        fig.set_size_inches(105, 140)
        fig.savefig(os.path.join(
            ops['save_path0'], 'figures',
            'session_report_{}.pdf'.format(session_data_name)),
            dpi=300)
        plt.close()

    def plot_individual_roi():
        roi_report = fitz.open()
        for roi_id in tqdm(np.argsort(labels, kind='stable')):
            if get_roi_sign(significance, roi_id):
                fig = plt.figure(figsize=(70, 56))
                gs = GridSpec(8, 10, figure=fig)
                # masks.
                mask_ax01 = plt.subplot(gs[0:2, 0:2])
                mask_ax02 = plt.subplot(gs[0, 2])
                mask_ax03 = plt.subplot(gs[0, 3])
                mask_ax04 = plt.subplot(gs[0, 4])
                plotter_masks.roi_loc_1chan(mask_ax01, roi_id, 'max')
                plotter_masks.roi_func(mask_ax02, roi_id, 'max')
                plotter_masks.roi_func(mask_ax03, roi_id, 'mean')
                plotter_masks.roi_masks(mask_ax04, roi_id)
                # alignment.
                percept_axs = [[plt.subplot(gs[2, i]) for i in [0,3,6,7]],
                               [plt.subplot(gs[4, i]) for i in [0,3,6,7]]]
                moto_axs = [[plt.subplot(gs[2, i]) for i in [1,2,4,5,8]],
                            [plt.subplot(gs[4, i]) for i in [1,2,4,5,8]],
                            plt.subplot(gs[1, 3]), plt.subplot(gs[1, 4])]
                plotter_percept.all_roi_percept_align(percept_axs, roi_id)
                plotter_motor.all_roi_motor_align(moto_axs, roi_id)
                # epoch.
                ep_percept_axs = [[plt.subplot(gs[3, i]) for i in [0,3,6]],
                               [plt.subplot(gs[5, i]) for i in [0,3,6]]]
                ep_moto_axs = [[plt.subplot(gs[3, i]) for i in [1,2,4,5,7]],
                               [plt.subplot(gs[5, i]) for i in [1,2,4,5,7]]]
                plotter_percept.all_roi_epoch_percept_align(ep_percept_axs, roi_id)
                plotter_motor.all_roi_epoch_motor_align(ep_moto_axs, roi_id)
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
                fig.set_size_inches(70, 56)
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
    labels = 1*np.ones_like(labels)
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
    plotter_percept = plotter_VIPG8_percept(
        neural_trials, labels, significance, cate_delay)
    print('Processing locomotion')
    plotter_motor = plotter_VIPG8_motor(
        neural_trials, labels, significance, cate_delay)
    print('Processing Modeling')
    plotter_model = plotter_VIPG8_model(
        neural_trials, labels, significance, cate_delay)
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
    #Trialization.run(ops)
    #StatTest.run(ops)
    plot_js_VIPG8(ops, session_data_name)
    print('===============================================')
    print('Processing {} completed'.format(session_data_name))
    
    
if __name__ == "__main__":
    reset_significance = False
    cate_delay = 15
    #delay = [neural_trials[str(i)]['trial_delay'] for i in range(len(neural_trials))]
    
    session_data_path = 'C:/Users/yhuang887/Projects/joystick_basic_202304/results/VG01_P_20240624_js_t'
    run(session_data_path)

    