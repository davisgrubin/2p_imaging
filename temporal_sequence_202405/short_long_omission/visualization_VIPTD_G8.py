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
#from modules.ReadResults import read_significance
RESET_SIGNIFICANCE = False

def read_ops(session_data_path):
    ops = np.load(
        os.path.join(session_data_path, 'suite2p', 'plane0','ops.npy'),
        allow_pickle=True).item()
    ops['save_path0'] = os.path.join(session_data_path)
    return ops

def get_roi_sign(significance, roi_id):
    r = significance['r_normal'][roi_id] +\
        significance['r_change'][roi_id] +\
        significance['r_oddball'][roi_id]
    return r

def reset_significance(significance):
    sign = {}
    sign['r_normal'] = np.ones_like(['r_normal'])
    sign['r_change'] = np.ones_like(['r_change'])
    sign['r_oddball'] = np.ones_like(['r_oddball'])
    return sign

from plot.fig1_mask import plotter_all_masks
from plot.fig2_align_stim import plotter_VIPTD_G8_align_stim
from plot.fig3_align_omi import plotter_VIPTD_G8_align_omi
from plot.fig5_raw_traces import plot_VIPTD_G8_example_traces
from plot.fig5_raw_traces import plot_roi_raw_trace
from plot.misc import plot_motion_offset_hist
from plot.misc import plot_inh_exc_label_pc
from plot.misc import plot_isi_distribution
from plot.misc import plot_stim_type
from plot.misc import plot_normal_type
from plot.misc import plot_fix_jitter_type
from plot.misc import plot_oddball_type
from plot.misc import plot_oddball_distribution

def plot_omi_VIPTD_G8(ops, session_data_name):
    
    def plot_session_report(session_report_name):
        fig = plt.figure(figsize=(91, 91))
        gs = GridSpec(13, 13, figure=fig)
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
        # stimulus alignment.
        normal_ax01 = plt.subplot(gs[4, 0])
        normal_ax02 = plt.subplot(gs[5, 0])
        normal_ax03 = plt.subplot(gs[4, 1:3])
        normal_ax04 = plt.subplot(gs[5, 1:3])
        normal_ax05 = plt.subplot(gs[4, 3])
        normal_ax06 = plt.subplot(gs[5, 3])
        normal_ax07 = plt.subplot(gs[4, 4])
        normal_ax08 = plt.subplot(gs[4, 5])
        normal_ax09 = plt.subplot(gs[5, 4])
        normal_ax10 = plt.subplot(gs[5, 5])
        plotter_align_stim.normal_exc(normal_ax01)
        plotter_align_stim.normal_inh(normal_ax02)
        plotter_align_stim.normal_exc_box(normal_ax03)
        plotter_align_stim.normal_inh_box(normal_ax04)
        plotter_align_stim.normal_exc_heatmap_trials(normal_ax05)
        plotter_align_stim.normal_inh_heatmap_trials(normal_ax06)
        plotter_align_stim.normal_epoch_exc_short(normal_ax07)
        plotter_align_stim.normal_epoch_exc_long(normal_ax08)
        plotter_align_stim.normal_epoch_inh_short(normal_ax09)
        plotter_align_stim.normal_epoch_inh_long(normal_ax10)
        change_ax01 = plt.subplot(gs[4, 6])
        change_ax02 = plt.subplot(gs[4, 7])
        change_ax03 = plt.subplot(gs[5, 6])
        change_ax04 = plt.subplot(gs[5, 7])
        change_ax05 = plt.subplot(gs[4, 8])
        change_ax06 = plt.subplot(gs[5, 8])
        change_ax07 = plt.subplot(gs[4, 9])
        change_ax08 = plt.subplot(gs[5, 9])
        change_ax09 = plt.subplot(gs[4:6, 10])
        plotter_align_stim.change_exc_short(change_ax01)
        plotter_align_stim.change_exc_long(change_ax02)
        plotter_align_stim.change_inh_short(change_ax03)
        plotter_align_stim.change_inh_long(change_ax04)
        plotter_align_stim.change_exc_box(change_ax05)
        plotter_align_stim.change_inh_box(change_ax06)
        plotter_align_stim.change_exc_heatmap_trials(change_ax07)
        plotter_align_stim.change_inh_heatmap_trials(change_ax08)
        plotter_align_stim.change_heatmap_neuron(change_ax09)
        context_ax01 = [plt.subplot(gs[6, i]) for i in range(5)]
        context_ax02 = [plt.subplot(gs[7, i]) for i in range(5)]
        context_ax03 = plt.subplot(gs[6, 5:7])
        context_ax04 = plt.subplot(gs[7, 5:7])
        context_ax05 = plt.subplot(gs[6:8, 7])
        context_ax06 = plt.subplot(gs[6:8, 8])
        context_ax07 = [plt.subplot(gs[8, i]) for i in range(5)]
        plotter_align_stim.context_exc(context_ax01)
        plotter_align_stim.context_inh(context_ax02)
        plotter_align_stim.context_exc_box(context_ax03)
        plotter_align_stim.context_inh_box(context_ax04)
        plotter_align_stim.context_all_short_heatmap_neuron(context_ax05)
        plotter_align_stim.context_all_long_heatmap_neuron(context_ax06)
        plotter_align_stim.context_evoke(context_ax07)
        # omission alignment.
        omi_normal_ax01 = plt.subplot(gs[9, 0])
        omi_normal_ax02 = plt.subplot(gs[10, 0])
        omi_normal_ax03 = plt.subplot(gs[9:11, 1])
        omi_normal_ax04 = plt.subplot(gs[9:11, 2])
        omi_normal_ax05 = plt.subplot(gs[9, 3])
        omi_normal_ax06 = plt.subplot(gs[10, 3])
        omi_normal_ax07 = plt.subplot(gs[9:11, 4])
        omi_normal_ax08 = plt.subplot(gs[9:11, 5])
        omi_normal_ax09 = plt.subplot(gs[11, 0])
        omi_normal_ax10 = plt.subplot(gs[11, 1])
        omi_normal_ax11 = plt.subplot(gs[12, 0])
        omi_normal_ax12 = plt.subplot(gs[12, 1])
        omi_normal_ax13 = plt.subplot(gs[11, 2])
        omi_normal_ax14 = plt.subplot(gs[12, 2])
        omi_normal_ax15 = plt.subplot(gs[8, 5])
        omi_normal_ax16 = plt.subplot(gs[8, 6])
        plotter_align_omi.omi_normal_pre_exc(omi_normal_ax01)
        plotter_align_omi.omi_normal_pre_inh(omi_normal_ax02)
        plotter_align_omi.omi_normal_pre_short_heatmap_neuron(omi_normal_ax03)
        plotter_align_omi.omi_normal_pre_long_heatmap_neuron(omi_normal_ax04)
        plotter_align_omi.omi_normal_post_exc(omi_normal_ax05)
        plotter_align_omi.omi_normal_post_inh(omi_normal_ax06)
        plotter_align_omi.omi_normal_post_short_heatmap_neuron(omi_normal_ax07)
        plotter_align_omi.omi_normal_post_long_heatmap_neuron(omi_normal_ax08)
        plotter_align_omi.omi_normal_exc_short(omi_normal_ax09)
        plotter_align_omi.omi_normal_exc_long(omi_normal_ax10)
        plotter_align_omi.omi_normal_inh_short(omi_normal_ax11)
        plotter_align_omi.omi_normal_inh_long(omi_normal_ax12)
        plotter_align_omi.omi_normal_exc_box(omi_normal_ax13)
        plotter_align_omi.omi_normal_inh_box(omi_normal_ax14)
        plotter_align_omi.omi_isi_short(omi_normal_ax15)
        plotter_align_omi.omi_isi_long(omi_normal_ax16)
        omi_epoch_ax01 = plt.subplot(gs[9, 6])
        omi_epoch_ax02 = plt.subplot(gs[9, 7])
        omi_epoch_ax03 = plt.subplot(gs[10, 6])
        omi_epoch_ax04 = plt.subplot(gs[10, 7])
        omi_epoch_ax05 = plt.subplot(gs[9, 8:10])
        omi_epoch_ax06 = plt.subplot(gs[10, 8:10])
        omi_epoch_ax07 = plt.subplot(gs[9, 10])
        omi_epoch_ax08 = plt.subplot(gs[9, 11])
        omi_epoch_ax09 = plt.subplot(gs[10, 10])
        omi_epoch_ax10 = plt.subplot(gs[10, 11])
        plotter_align_omi.omi_epoch_post_exc_short(omi_epoch_ax01)
        plotter_align_omi.omi_epoch_post_exc_long(omi_epoch_ax02)
        plotter_align_omi.omi_epoch_post_inh_short(omi_epoch_ax03)
        plotter_align_omi.omi_epoch_post_inh_long(omi_epoch_ax04)
        plotter_align_omi.omi_epoch_post_exc_box(omi_epoch_ax05)
        plotter_align_omi.omi_epoch_post_inh_box(omi_epoch_ax06)
        plotter_align_omi.omi_post_exc_short_heatmap_trials(omi_epoch_ax07)
        plotter_align_omi.omi_post_exc_long_heatmap_trials(omi_epoch_ax08)
        plotter_align_omi.omi_post_inh_short_heatmap_trials(omi_epoch_ax09)
        plotter_align_omi.omi_post_inh_long_heatmap_trials(omi_epoch_ax10)
        omi_context_ax01 = [plt.subplot(gs[11, i+3]) for i in range(4)]
        omi_context_ax02 = [plt.subplot(gs[12, i+3]) for i in range(4)]
        omi_context_ax03 = plt.subplot(gs[11, 7:9])
        omi_context_ax04 = plt.subplot(gs[12, 7:9])
        plotter_align_omi.omi_context_exc(omi_context_ax01)
        plotter_align_omi.omi_context_inh(omi_context_ax02)
        plotter_align_omi.omi_context_exc_box(omi_context_ax03)
        plotter_align_omi.omi_context_inh_box(omi_context_ax04)
        omi_jitter_ax01 = plt.subplot(gs[11, 9])
        omi_jitter_ax02 = plt.subplot(gs[11, 10])
        omi_jitter_ax03 = plt.subplot(gs[11, 11])
        omi_jitter_ax04 = plt.subplot(gs[11, 12])
        omi_jitter_ax05 = plt.subplot(gs[12, 9])
        omi_jitter_ax06 = plt.subplot(gs[12, 10])
        omi_jitter_ax07 = plt.subplot(gs[12, 11])
        omi_jitter_ax08 = plt.subplot(gs[12, 12])
        plotter_align_omi.omi_post_isi_exc_short(omi_jitter_ax01)
        plotter_align_omi.omi_post_isi_exc_short_box(omi_jitter_ax02)
        plotter_align_omi.omi_post_isi_exc_long(omi_jitter_ax03)
        plotter_align_omi.omi_post_isi_exc_long_box(omi_jitter_ax04)
        plotter_align_omi.omi_post_isi_inh_short(omi_jitter_ax05)
        plotter_align_omi.omi_post_isi_inh_short_box(omi_jitter_ax06)
        plotter_align_omi.omi_post_isi_inh_long(omi_jitter_ax07)
        plotter_align_omi.omi_post_isi_inh_long_box(omi_jitter_ax08)
        # example traces.
        example_ax = plt.subplot(gs[0:4, 10])
        plot_VIPTD_G8_example_traces(
            example_ax, dff, labels, vol_img_bin, vol_time)
        # offset.
        offset_ax = plt.subplot(gs[0, 8])
        plot_motion_offset_hist(offset_ax, xoff, yoff)
        # labels.
        label_ax = plt.subplot(gs[0, 9])
        plot_inh_exc_label_pc(label_ax, labels)
        # isi distribution.
        isi_ax = plt.subplot(gs[1, 8])
        plot_isi_distribution(isi_ax, neural_trials)
        # stimulus types.
        type_ax01 = plt.subplot(gs[1, 9])
        type_ax02 = plt.subplot(gs[2, 8])
        type_ax03 = plt.subplot(gs[2, 9])
        type_ax04 = plt.subplot(gs[3, 8])
        type_ax05 = plt.subplot(gs[3, 9])
        plot_stim_type(type_ax01, neural_trials)
        plot_normal_type(type_ax02, neural_trials)
        plot_fix_jitter_type(type_ax03, neural_trials)
        plot_oddball_type(type_ax04, neural_trials)
        plot_oddball_distribution(type_ax05, neural_trials)
        # save figure.
        fig.set_size_inches(84, 91)
        fig.savefig(os.path.join(
            ops['save_path0'], 'figures', session_report_name),
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
            # stimulus alignment.
            normal_ax01 = plt.subplot(gs[0, 4])
            normal_ax02 = plt.subplot(gs[0, 5:7])
            normal_ax03 = plt.subplot(gs[0, 7])
            plotter_align_stim.roi_normal(normal_ax01, roi_id)
            plotter_align_stim.roi_normal_box(normal_ax02, roi_id)
            plotter_align_stim.roi_normal_heatmap_trials(normal_ax03, roi_id)
            change_ax01 = plt.subplot(gs[1, 4])
            change_ax02 = plt.subplot(gs[1, 5])
            change_ax03 = plt.subplot(gs[1, 6])
            change_ax04 = plt.subplot(gs[1, 7])
            plotter_align_stim.roi_change_short(change_ax01, roi_id)
            plotter_align_stim.roi_change_long(change_ax02, roi_id)
            plotter_align_stim.roi_change_box(change_ax03, roi_id)
            plotter_align_stim.roi_change_heatmap_trials(change_ax04, roi_id)
            context_ax01 = plt.subplot(gs[2, 0])
            context_ax02 = plt.subplot(gs[2, 1])
            context_ax03 = plt.subplot(gs[2, 2])
            context_ax04 = [plt.subplot(gs[2, i+3]) for i in range(4)]
            context_ax05 = plt.subplot(gs[2, 7:9])
            plotter_align_stim.roi_context_all(context_ax01, roi_id)
            plotter_align_stim.roi_context_all_short_heatmap_trial(context_ax02, roi_id)
            plotter_align_stim.roi_context_all_long_heatmap_trial(context_ax03, roi_id)
            plotter_align_stim.roi_context_individual(context_ax04, roi_id)
            plotter_align_stim.roi_context_box(context_ax05, roi_id)
            # omission alignment.
            '''
            omi_normal_ax01 = plt.subplot(gs[3, 0])
            omi_normal_ax02 = plt.subplot(gs[3, 1])
            omi_normal_ax03 = plt.subplot(gs[3, 2])
            omi_normal_ax04 = plt.subplot(gs[3, 3])
            omi_normal_ax05 = plt.subplot(gs[4, 0])
            omi_normal_ax06 = plt.subplot(gs[4, 1])
            omi_normal_ax07 = plt.subplot(gs[4, 2])
            omi_normal_ax08 = plt.subplot(gs[3, 4])
            omi_normal_ax09 = plt.subplot(gs[3, 5])
            omi_normal_ax10 = plt.subplot(gs[3, 6])
            omi_normal_ax11 = plt.subplot(gs[3, 7])
            plotter_align_omi.roi_omi_normal_pre(omi_normal_ax01, roi_id)
            plotter_align_omi.roi_omi_normal_post(omi_normal_ax02, roi_id)
            plotter_align_omi.roi_omi_normal_post_heatmap_trials_short(omi_normal_ax03, roi_id)
            plotter_align_omi.roi_omi_normal_post_heatmap_trials_long(omi_normal_ax04, roi_id)
            plotter_align_omi.roi_omi_normal_short(omi_normal_ax05, roi_id)
            plotter_align_omi.roi_omi_normal_long(omi_normal_ax06, roi_id)
            plotter_align_omi.roi_omi_normal_box(omi_normal_ax07, roi_id)
            plotter_align_omi.roi_omi_isi_short(omi_normal_ax08, roi_id)
            plotter_align_omi.roi_omi_isi_long(omi_normal_ax09, roi_id)
            plotter_align_omi.roi_omi_isi_post_short(omi_normal_ax10, roi_id)
            plotter_align_omi.roi_omi_isi_post_long(omi_normal_ax11, roi_id)
            omi_context_ax01 = [plt.subplot(gs[4, i+3]) for i in range(4)]
            omi_context_ax02 = plt.subplot(gs[4, 7])
            plotter_align_omi.roi_omi_context(omi_context_ax01, roi_id)
            plotter_align_omi.roi_omi_context_box(omi_context_ax02, roi_id)
            '''
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
    [vol_time, vol_start, vol_stim_vis, vol_img, 
     vol_hifi, vol_stim_aud, vol_flir,
     vol_pmt, vol_led] = read_raw_voltages(ops)
    dff = read_dff(ops)
    neural_trials = read_neural_trials(ops)
    [xoff, yoff] = read_move_offset(ops)
    significance = None
#    significance = read_significance(ops)
    if RESET_SIGNIFICANCE:
        significance = reset_significance(significance)
    print('Processing masks')
    plotter_masks = plotter_all_masks(
        labels, masks, mean_func, max_func, mean_anat, masks_anat)
    print('===============================================')
    print('====== plot session report with all ROIs ======')
    print('===============================================')
    print('Processing stimulus alignments')
    plotter_align_stim = plotter_VIPTD_G8_align_stim(neural_trials, labels)
    print('Processing omission alignments')
    plotter_align_omi = plotter_VIPTD_G8_align_omi(neural_trials, labels)
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
    #Trialization.run(ops)
    #plot_omi_VIPTD_G8(ops, session_data_name)
    print('===============================================')
    print('Processing {} completed'.format(session_data_name))
    
    
if __name__ == "__main__":
    
    session_data_path = 'C:/Users/yhuang887/Projects/temporal_sequence_202405/short_long_omission/results/FN14_P_20240718_seq1423_DCNOPTO_t'
    run(session_data_path)
    '''
    for session_data_path in [
        'C:/Users/yhuang887/Projects/passive_omission_202304/results/FN15_P_20240412_omi_t',
        'C:/Users/yhuang887/Projects/passive_omission_202304/results/FN16_P_20240410_omi_t',
        'C:/Users/yhuang887/Projects/passive_omission_202304/results/FN16_P_20240413_omi_t'
        ]:
        run(session_data_path)
    '''


    