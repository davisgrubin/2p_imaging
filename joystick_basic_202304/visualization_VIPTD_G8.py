#!/usr/bin/env python3

import os
import fitz
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.gridspec import GridSpec

import warnings
warnings.filterwarnings("ignore")

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
        significance['r_push1'][roi_id] +\
        significance['r_retract1'][roi_id] +\
        significance['r_vis2'][roi_id] +\
        significance['r_push2'][roi_id] +\
        significance['r_retract2'][roi_id] +\
        significance['r_wait'][roi_id] +\
        significance['r_reward'][roi_id] +\
        significance['r_punish'][roi_id] +\
        significance['r_lick'][roi_id]
    return r

def reset_significant_roi(significance):
    sign = {}
    sign['r_vis1']     = np.ones_like(significance['r_vis1'])
    sign['r_push1']    = np.ones_like(significance['r_push1'])
    sign['r_retract1'] = np.ones_like(significance['r_retract1'])
    sign['r_vis2']     = np.ones_like(significance['r_vis2'])
    sign['r_push2']    = np.ones_like(significance['r_push2'])
    sign['r_retract2'] = np.ones_like(significance['r_retract2'])
    sign['r_wait']     = np.ones_like(significance['r_wait'])
    sign['r_reward']   = np.ones_like(significance['r_reward'])
    sign['r_punish']   = np.ones_like(significance['r_punish'])
    sign['r_lick']     = np.ones_like(significance['r_lick'])
    return sign

from plot.fig0_beh import plotter_all_beh
from plot.fig1_mask import plotter_all_masks
from plot.fig2_align_percept import plotter_VIPTD_G8_percept
from plot.fig3_align_motor import plotter_VIPTD_G8_motor
from plot.fig4_model import plotter_VIPTD_G8_model
from plot.fig5_raw_traces import plot_VIPTD_G8_example_traces
from plot.fig5_raw_traces import plot_roi_example_traces
from plot.fig5_raw_traces import plot_roi_raw_trace
from plot.misc import plot_motion_offset_hist
from plot.misc import plot_inh_exc_label_pc
from plot.misc import plot_significance
from plot.misc import plot_roi_significance


def plot_js_VIPTD_G8(ops, session_data_name):
    
    def plot_session_report():
        fig = plt.figure(figsize=(105, 210))
        gs = GridSpec(30, 15, figure=fig)
        # masks.
        print('Plotting masks')
        mask_ax01 = plt.subplot(gs[0:2, 0:2])
        mask_ax02 = plt.subplot(gs[2:4, 0:2])
        mask_ax03 = plt.subplot(gs[0:2, 2:4])
        mask_ax04 = plt.subplot(gs[2:4, 2:4])
        mask_ax05 = plt.subplot(gs[0:2, 4:6])
        mask_ax06 = plt.subplot(gs[2:4, 4:6])
        mask_ax07 = plt.subplot(gs[0:2, 6:8])
        mask_ax08 = plt.subplot(gs[2:4, 6:8])
        mask_ax09 = plt.subplot(gs[0:2, 8:10])
        mask_ax10 = plt.subplot(gs[2:4, 8:10])
        mask_ax11 = plt.subplot(gs[0:2, 10:12])
        plotter_masks.func(mask_ax01, 'max')
        plotter_masks.func_masks(mask_ax02)
        plotter_masks.anat_cellpose(mask_ax03)
        plotter_masks.masks_superimpose(mask_ax04)
        plotter_masks.anat(mask_ax05)
        plotter_masks.anat_label_masks(mask_ax06)
        plotter_masks.superimpose(mask_ax07, 'max')
        plotter_masks.shared_masks(mask_ax08)
        plotter_masks.func(mask_ax09, 'max', with_mask=False)
        plotter_masks.anat(mask_ax10, with_mask=False)
        plotter_masks.superimpose(mask_ax11, 'max', with_mask=False)
        # behavior.
        beh_misc_ax01 = plt.subplot(gs[5, 10:12])
        beh_misc_ax02 = plt.subplot(gs[6, 10])
        beh_js_short_axs = [plt.subplot(gs[5, i]) for i in range(9)]
        beh_js_long_axs  = [plt.subplot(gs[14, i]) for i in range(9)]
        beh_js_both_axs  = [plt.subplot(gs[23, i]) for i in range(9)]
        beh_js_epoch_short_axs = [plt.subplot(gs[10, i]) for i in range(8)]
        beh_js_epoch_long_axs  = [plt.subplot(gs[19, i]) for i in range(8)]
        beh_js_onset_ax = plt.subplot(gs[23, 9])
        print('Plotting behavior overview')
        plotter_beh.delay_dist(beh_misc_ax01)
        plotter_beh.session_outcome(beh_misc_ax02)
        print('Plotting behavior alignment for short')
        plotter_beh.all_short_align(beh_js_short_axs)
        print('Plotting behavior alignment for long')
        plotter_beh.all_long_align(beh_js_long_axs)
        print('Plotting behavior alignment for both')
        plotter_beh.all_both_align(beh_js_both_axs)
        print('Plotting behavior alignment for short with epoch')
        plotter_beh.all_short_epoch(beh_js_epoch_short_axs)
        print('Plotting behavior alignment for long with epoch')
        plotter_beh.all_long_epoch(beh_js_epoch_long_axs)
        print('Plotting behavior alignment for all onset')
        plotter_beh.onset(beh_js_onset_ax)
        # short.
        print('Plotting neural trace alignment for short')
        percept_axs01 = [plt.subplot(gs[6, i]) for i in [0,3,6,7]]
        percept_axs02 = [plt.subplot(gs[7, i]) for i in [0,3,6,7]]
        percept_axs03 = [plt.subplot(gs[8, i]) for i in [0,3,6,7]]
        plotter_percept.all_short_percept_align_exc(percept_axs01)
        plotter_percept.all_short_percept_align_inh(percept_axs02)
        plotter_percept.all_short_percept_align_heatmap_neuron(percept_axs03)
        motor_axs01 = [plt.subplot(gs[6, i]) for i in [1,2,4,5,8]]
        motor_axs02 = [plt.subplot(gs[7, i]) for i in [1,2,4,5,8]]
        motor_axs03 = [plt.subplot(gs[8, i]) for i in [1,2,4,5,8]]
        plotter_motor.all_short_motor_align_exc(motor_axs01)
        plotter_motor.all_short_motor_align_inh(motor_axs02)
        plotter_motor.all_short_motor_align_heatmap_neuron(motor_axs03)
        # long.
        print('Plotting neural trace alignment for long')
        percept_axs01 = [plt.subplot(gs[15, i]) for i in [0,3,6,7]]
        percept_axs02 = [plt.subplot(gs[16, i]) for i in [0,3,6,7]]
        percept_axs03 = [plt.subplot(gs[17, i]) for i in [0,3,6,7]]
        plotter_percept.all_long_percept_align_exc(percept_axs01)
        plotter_percept.all_long_percept_align_inh(percept_axs02)
        plotter_percept.all_long_percept_align_heatmap_neuron(percept_axs03)
        motor_axs01 = [plt.subplot(gs[15, i]) for i in [1,2,4,5,8]]
        motor_axs02 = [plt.subplot(gs[16, i]) for i in [1,2,4,5,8]]
        motor_axs03 = [plt.subplot(gs[17, i]) for i in [1,2,4,5,8]]
        plotter_motor.all_long_motor_align_exc(motor_axs01)
        plotter_motor.all_long_motor_align_inh(motor_axs02)
        plotter_motor.all_long_motor_align_heatmap_neuron(motor_axs03)
        # epoch.
        print('Plotting neural trace alignment for short with epoch')
        ep_short_axs01 = [plt.subplot(gs[11, i]) for i in [0,3,6]]
        ep_short_axs02 = [plt.subplot(gs[12, i]) for i in [0,3,6]]
        ep_short_axs03 = [plt.subplot(gs[11, i]) for i in [1,2,4,5,7]]
        ep_short_axs04 = [plt.subplot(gs[12, i]) for i in [1,2,4,5,7]]
        plotter_percept.all_short_epoch_percept_align_exc(ep_short_axs01)
        plotter_percept.all_short_epoch_percept_align_inh(ep_short_axs02)
        plotter_motor.all_short_epoch_motor_align_exc(ep_short_axs03)
        plotter_motor.all_short_epoch_motor_align_inh(ep_short_axs04)
        print('Plotting neural trace alignment for long with epoch')
        ep_long_axs01 = [plt.subplot(gs[20, i]) for i in [0,3,6]]
        ep_long_axs02 = [plt.subplot(gs[21, i]) for i in [0,3,6]]
        ep_long_axs03 = [plt.subplot(gs[20, i]) for i in [1,2,4,5,7]]
        ep_long_axs04 = [plt.subplot(gs[21, i]) for i in [1,2,4,5,7]]
        plotter_percept.all_long_epoch_percept_align_exc(ep_long_axs01)
        plotter_percept.all_long_epoch_percept_align_inh(ep_long_axs02)
        plotter_motor.all_long_epoch_motor_align_exc(ep_long_axs03)
        plotter_motor.all_long_epoch_motor_align_inh(ep_long_axs04)
        # both.
        print('Plotting neural trace alignment for both')
        percept_axs01 = [plt.subplot(gs[24, i]) for i in [0,3,6,7]]
        percept_axs02 = [plt.subplot(gs[25, i]) for i in [0,3,6,7]]
        percept_axs03 = [plt.subplot(gs[26, i]) for i in [0,3,6,7]]
        plotter_percept.all_both_percept_align_exc(percept_axs01)
        plotter_percept.all_both_percept_align_inh(percept_axs02)
        plotter_percept.all_both_percept_align_heatmap_neuron(percept_axs03)
        motor_axs01 = [plt.subplot(gs[24, i]) for i in [1,2,4,5,8]]
        motor_axs02 = [plt.subplot(gs[25, i]) for i in [1,2,4,5,8]]
        motor_axs03 = [plt.subplot(gs[26, i]) for i in [1,2,4,5,8]]
        plotter_motor.all_both_motor_align_exc(motor_axs01)
        plotter_motor.all_both_motor_align_inh(motor_axs02)
        plotter_motor.all_both_motor_align_heatmap_neuron(motor_axs03)
        # push onset.
        print('Plotting neural trace alignment for all onset')
        push_ax01 = plt.subplot(gs[24, 9])
        push_ax02 = plt.subplot(gs[25, 9])
        push_ax03 = [plt.subplot(gs[i, 9]) for i in [26,27,28]]
        plotter_motor.onset_exc(push_ax01)
        plotter_motor.onset_inh(push_ax02)
        plotter_motor.onset_heatmap_neuron(push_ax03)
        # lick.
        print('Plotting neural trace alignment for lick')
        lick_ax01 = plt.subplot(gs[24, 10])
        lick_ax02 = plt.subplot(gs[25, 10])
        lick_ax03 = plt.subplot(gs[26, 10])
        plotter_motor.lick_exc(lick_ax01)
        plotter_motor.lick_inh(lick_ax02)
        plotter_motor.lick_heatmap_neuron(lick_ax03)
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
        print('Plotting decoding model')
        plotter_model.all_decode(decode_axs)
        print('Plotting dimensionality reduction')
        plotter_model.block_type_population_pca(model_axs07)
        plotter_model.epoch_population_pca(model_axs08)
        print('Plotting latent dynamics')
        plotter_model.block_type_dynamics(model_axs09)
        plotter_model.epoch_dynamics(model_axs10)
        # example traces.
        print('Plotting example traces')
        example_ax = plt.subplot(gs[0:4, 12:14])
        plot_VIPTD_G8_example_traces(
            example_ax, dff, labels, vol_img, vol_time)
        print('Plotting 2p misc results')
        # offset.
        offset_ax = plt.subplot(gs[2, 10])
        plot_motion_offset_hist(offset_ax, xoff, yoff)
        # labels.
        label_ax = plt.subplot(gs[2, 11])
        plot_inh_exc_label_pc(label_ax, labels)
        # significance.
        sign_ax = plt.subplot(gs[3, 10])
        plot_significance(sign_ax, significance, labels)
        # save figure.
        fig.set_size_inches(105, 210)
        fig.savefig(os.path.join(
            ops['save_path0'], 'figures',
            'session_report_{}_{}.pdf'.format(sig_tag, session_data_name)),
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
                mask_ax05 = plt.subplot(gs[0, 5])
                plotter_masks.roi_loc_2chan(mask_ax01, roi_id, 'max')
                plotter_masks.roi_func(mask_ax02, roi_id, 'max')
                plotter_masks.roi_anat(mask_ax03, roi_id)
                plotter_masks.roi_superimpose(mask_ax04, roi_id, 'max')
                plotter_masks.roi_masks(mask_ax05, roi_id)
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
            'roi_report_{}_{}.pdf'.format(sig_tag, session_data_name)))
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
    [vol_time, vol_start, vol_stim_vis, vol_img,
     vol_hifi, vol_stim_aud, vol_flir,
     vol_pmt, vol_led] = read_raw_voltages(ops)
    dff = read_dff(ops)
    neural_trials = read_neural_trials(ops, cate_delay)
    [xoff, yoff] = read_move_offset(ops)
    significance = read_significance(ops)
    sig_tag = 'sig'
    if reset_significance:
        significance = reset_significant_roi(significance)
        sig_tag = 'all'
    print('Processing masks')
    plotter_masks = plotter_all_masks(
        labels, masks, mean_func, max_func, mean_anat, masks_anat)
    print('Processing behavior')
    plotter_beh = plotter_all_beh(
        neural_trials, cate_delay)
    print('Processing perception')
    plotter_percept = plotter_VIPTD_G8_percept(
        neural_trials, labels, significance, cate_delay)
    print('Processing locomotion')
    plotter_motor = plotter_VIPTD_G8_motor(
        neural_trials, labels, significance, cate_delay)
    print('Processing Modeling')
    plotter_model = plotter_VIPTD_G8_model(
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
    #Trialization.run(ops)
    StatTest.run(ops, cate_delay)
    plot_js_VIPTD_G8(ops, session_data_name)
    print('===============================================')
    print('Processing {} completed'.format(session_data_name))
    
    
if __name__ == "__main__":
    cate_delay = 25
    session_data_path = 'C:/Users/yhuang887/Projects/joystick_basic_202304/results/FN13_P_20240618_js_DCNCHEMO_t'
    reset_significance = False
    run(session_data_path)
    reset_significance = True
    run(session_data_path)

