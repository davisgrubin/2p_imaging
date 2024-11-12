#!/usr/bin/env python3

import os
import fitz
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.gridspec import GridSpec

from modules import Trialization
from modules import StatTest
from modules.ReadResults import read_ops
from modules.ReadResults import read_all

def get_roi_sign(significance, roi_id):
    r = significance['r_normal'][roi_id] +\
        significance['r_change'][roi_id] +\
        significance['r_oddball'][roi_id]
    return r

from plot.fig1_mask import plotter_all_masks
from plot.fig2_align_stim import plotter_VIPTD_G8_align_stim
from plot.fig3_align_odd import plotter_VIPTD_G8_align_odd
from plot.fig5_raw_traces import plot_VIPTD_G8_example_traces
from plot.fig5_raw_traces import plot_roi_raw_trace
from plot.misc import plot_sess_name
from plot.misc import plot_motion_offset_hist
from plot.misc import plot_inh_exc_label_pc
from plot.misc import plot_isi_distribution
from plot.misc import plot_stim_type
from plot.misc import plot_normal_type
from plot.misc import plot_fix_jitter_type
from plot.misc import plot_oddball_type
from plot.misc import plot_significance
from plot.misc import plot_roi_significance

def run(list_session_data_path, sig_tag):

    def plot_session_report():
        fig = plt.figure(figsize=(140, 210))
        gs = GridSpec(30, 20, figure=fig)
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
        # normal alignment.
        print('Plotting normal alignment')
        normal_axs01 = [plt.subplot(gs[5, 0])]
        normal_axs02 = [plt.subplot(gs[6, 0])]
        normal_axs03 = [plt.subplot(gs[5:7, 1])]
        plotter_align_stim.normal_exc(normal_axs01)
        plotter_align_stim.normal_inh(normal_axs02)
        plotter_align_stim.normal_heatmap(normal_axs03)
        # oddball alignment.
        print('Plotting oddball alignment')
        odd_normal_axs01 = [
            plt.subplot(gs[10, 0:2]), plt.subplot(gs[10, 2:4]),  plt.subplot(gs[10, 4:6])]
        odd_normal_axs02 = [
            plt.subplot(gs[11, 0:2]), plt.subplot(gs[11, 2:4]),  plt.subplot(gs[11, 4:6])]
        odd_normal_axs03 = [plt.subplot(gs[10:12, i]) for i in [6,7,8,9]]
        plotter_align_odd.odd_normal_exc(odd_normal_axs01)
        plotter_align_odd.odd_normal_inh(odd_normal_axs02)
        plotter_align_odd.odd_normal_heatmap(odd_normal_axs03)
        # example traces.
        print('Plotting example traces')
        example_ax = plt.subplot(gs[0:4, 12])
        plot_VIPTD_G8_example_traces(
            example_ax, list_dff[0], list_labels[0], list_vol[0][3], list_vol[0][0])
        print('Plotting stimulus types')
        # isi distribution.
        isi_ax01 = plt.subplot(gs[0, 14])
        isi_ax02 = plt.subplot(gs[0, 15])
        plot_isi_distribution(isi_ax01, list_neural_trials)
        plotter_align_odd.plot_pre_odd_isi_distribution(isi_ax02)
        # stimulus types.
        type_ax01 = plt.subplot(gs[1, 14])
        type_ax02 = plt.subplot(gs[1, 15])
        type_ax03 = plt.subplot(gs[2, 14])
        type_ax04 = plt.subplot(gs[2, 15])
        plot_stim_type(type_ax01, list_neural_trials)
        plot_normal_type(type_ax02, list_neural_trials)
        plot_fix_jitter_type(type_ax03, list_neural_trials)
        plot_oddball_type(type_ax04, list_neural_trials)
        print('Plotting 2p misc results')
        # offset.
        offset_ax = plt.subplot(gs[2, 10:12])
        plot_motion_offset_hist(offset_ax, list_move_offset)
        # labels.
        label_ax = plt.subplot(gs[3, 10])
        plot_inh_exc_label_pc(label_ax, list_labels)
        # significance.
        sign_ax = plt.subplot(gs[3, 11])
        plot_significance(sign_ax, list_significance, list_labels)
        # used sessions.
        sess_ax = plt.subplot(gs[3, 14])
        plot_sess_name(sess_ax, list_session_data_path)
        # save figure.
        fig.set_size_inches(140, 210)
        fig.savefig(os.path.join(
            list_ops[0]['save_path0'], 'figures',
            'session_report_{}_{}.pdf'.format(
                sig_tag,
                list_session_data_path[0].split('/')[-1])),
            dpi=300)
        plt.close()
    '''
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
        for roi_id in [50,55,60,65,70]:
            fig, axs = plt.subplots(trace_num_fig, 1, figsize=(max_ms/5000, trace_num_fig*4))
            plt.subplots_adjust(hspace=0.6)
            plot_roi_raw_trace(
                axs, roi_id, max_ms,
                labels, dff,
                vol_img, vol_stim_vis, vol_led, vol_time)
            fig.set_size_inches(max_ms/2500, trace_num_fig*4)
            fig.tight_layout()
            fig.savefig(os.path.join(
                ops['save_path0'], 'figures', 'raw_traces',
                str(roi_id).zfill(4)+'.pdf'),
                dpi=300)
            plt.close()
    '''

    # main
    print('===============================================')
    print('============ Start Processing Data ============')
    print('===============================================')
    print('Processing {} sessions'.format(len(list_session_data_path)))
    for session_data_path in list_session_data_path:
        print(session_data_path.split('/')[-1])
    print('Reading ops.npy')
    list_ops = read_ops(list_session_data_path)
    '''
    print('===============================================')
    print('============= trials segmentation =============')
    print('===============================================')
    for ops in list_ops:
        Trialization.run(ops)
    print('===============================================')
    print('============== significance test ==============')
    print('===============================================')
    for ops in list_ops:
        StatTest.run(ops)
    '''
    print('===============================================')
    print('============ reading saved results ============')
    print('===============================================')
    [list_labels, list_masks, list_vol, list_dff,
     list_neural_trials, list_move_offset, list_significance
     ] = read_all(list_ops, sig_tag)
    print('Preparing masks')
    plotter_masks = plotter_all_masks(
        list_labels[0],
        list_masks[0][0],
        list_masks[0][1],
        list_masks[0][2],
        list_masks[0][3],
        list_masks[0][4])
    print('Preparing stimulus alignments')
    plotter_align_stim = plotter_VIPTD_G8_align_stim(
        list_neural_trials, list_labels, list_significance)
    print('Preparing omission alignments')
    plotter_align_odd = plotter_VIPTD_G8_align_odd(
        list_neural_trials, list_labels, list_significance)
    print('Preparing misc results')

    plot_session_report()
    print('===============================================')
    print('=============== plot roi report ===============')
    print('===============================================')
    #plot_individual_roi()
    print('===============================================')
    print('=============== plot raw traces ===============')
    print('===============================================')
    #plot_raw_traces()
    print('===============================================')
    print('Processing completed')

if __name__ == "__main__":
    '''
    list_session_data_path = [
        'C:/Users/yhuang887/Projects/temporal_sequence_202405/oddball/results/FN14_P_20240627_seq1130_t',
        'C:/Users/yhuang887/Projects/temporal_sequence_202405/oddball/results/FN14_P_20240701_seq1130_t',
        'C:/Users/yhuang887/Projects/temporal_sequence_202405/oddball/results/FN14_P_20240702_seq1130_t',
        'C:/Users/yhuang887/Projects/temporal_sequence_202405/oddball/results/FN14_P_20240705_seq1130_t',
        'C:/Users/yhuang887/Projects/temporal_sequence_202405/oddball/results/FN14_P_20240708_seq1131_t',
        'C:/Users/yhuang887/Projects/temporal_sequence_202405/oddball/results/FN14_P_20240710_seq1130_t',
        ]
    '''
    list_session_data_path = [
        'C:/Users/yhuang887/Projects/temporal_sequence_202405/oddball/results/FN14_P_20240627_seq1130_t',
        'C:/Users/yhuang887/Projects/temporal_sequence_202405/oddball/results/FN14_P_20240701_seq1130_t',
        'C:/Users/yhuang887/Projects/temporal_sequence_202405/oddball/results/FN14_P_20240702_seq1130_t',
        'C:/Users/yhuang887/Projects/temporal_sequence_202405/oddball/results/FN14_P_20240705_seq1130_t',
        'C:/Users/yhuang887/Projects/temporal_sequence_202405/oddball/results/FN14_P_20240708_seq1131_t',
        'C:/Users/yhuang887/Projects/temporal_sequence_202405/oddball/results/FN14_P_20240710_seq1130_t',
        ]
    #run(list_session_data_path, 'sig')
    run(list_session_data_path, 'all')
    '''
    list_ops = read_ops(list_session_data_path)
    [list_labels, list_masks, list_vol, list_dff,
     list_neural_trials, list_move_offset, list_significance
     ] = read_all(list_ops, 'all')
    '''
