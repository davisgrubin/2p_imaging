#!/usr/bin/env python3


import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_pdf import PdfPages
    
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


from plot.fig1_mask import plot_ppc_func_max
from plot.fig1_mask import plot_ppc_func_masks
from plot.fig1_mask import plot_ppc_anat_mean
from plot.fig1_mask import plot_ppc_anat_label_masks
from plot.fig1_mask import plot_ppc_superimpose
from plot.fig1_mask import plot_ppc_shared_masks
from plot.fig2_align_grating import plot_ppc_exc_1st_grating_mean_outcome
from plot.fig2_align_grating import plot_ppc_inh_1st_grating_mean_outcome
from plot.fig2_align_grating import plot_ppc_exc_2nd_grating_mean_outcome
from plot.fig2_align_grating import plot_ppc_inh_2nd_grating_mean_outcome
from plot.fig2_align_grating import plot_ppc_2nd_grating_heatmap_reward_neuron
from plot.fig2_align_grating import plot_ppc_2nd_grating_heatmap_punish_neuron
from plot.fig2_align_grating import plot_ppc_2nd_grating_heatmap_reward_trial
from plot.fig2_align_grating import plot_ppc_2nd_grating_heatmap_punish_trial
from plot.fig3_align_beh import plot_beh_outcome
from plot.fig3_align_beh import plot_align_pos_reward
from plot.fig3_align_beh import plot_align_pos_vis1
from plot.fig3_align_beh import plot_align_pos_vis2
from plot.fig3_align_beh import plot_ppc_reward
from plot.fig3_align_beh import plot_ppc_punish
from plot.fig5_raw_traces import plot_ppc_examplt_traces
from plot.misc import plot_motion_offset_hist
from plot.misc import plot_inh_exc_label_pc
from plot.misc import plot_isi_distribution


def plot_js_ppc_seesion_report(ops):
    
    # read data.
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
    
    # define grid space for ppc.
    fig = plt.figure()
    gs = GridSpec(6, 9, figure=fig)
    
    # masks.
    mask_ax01 = plt.subplot(gs[0:2, 0:2])
    mask_ax02 = plt.subplot(gs[2:4, 0:2])
    mask_ax03 = plt.subplot(gs[0:2, 2:4])
    mask_ax04 = plt.subplot(gs[2:4, 2:4])
    mask_ax05 = plt.subplot(gs[0:2, 4:6])
    mask_ax06 = plt.subplot(gs[2:4, 4:6])
    plot_ppc_func_max(mask_ax01, max_func, masks)
    plot_ppc_func_masks(mask_ax02, masks)
    plot_ppc_anat_mean(mask_ax03, mean_anat, masks, labels)
    plot_ppc_anat_label_masks(mask_ax04, masks, labels)
    plot_ppc_superimpose(mask_ax05, max_func, mean_anat, masks, labels)
    plot_ppc_shared_masks(mask_ax06, masks, labels)
    
    # example traces.
    example_ax = plt.subplot(gs[2:4, 6:8])
    plot_ppc_examplt_traces(example_ax,
        dff, labels, vol_stim_bin, vol_img_bin, vol_time)
    
    # grating average.
    stim_ax01 = plt.subplot(gs[4, 0])
    stim_ax02 = plt.subplot(gs[5, 0])
    stim_ax03 = plt.subplot(gs[4, 1])
    stim_ax04 = plt.subplot(gs[5, 1])
    stim_ax05 = plt.subplot(gs[4, 2])
    stim_ax06 = plt.subplot(gs[5, 2])
    stim_ax07 = plt.subplot(gs[4, 3])
    stim_ax08 = plt.subplot(gs[5, 3])
    plot_ppc_exc_1st_grating_mean_outcome(stim_ax01, neural_trials, labels)
    plot_ppc_inh_1st_grating_mean_outcome(stim_ax02, neural_trials, labels)
    plot_ppc_exc_2nd_grating_mean_outcome(stim_ax03, neural_trials, labels)
    plot_ppc_inh_2nd_grating_mean_outcome(stim_ax04, neural_trials, labels)
    plot_ppc_2nd_grating_heatmap_reward_neuron(stim_ax05, neural_trials, labels)
    plot_ppc_2nd_grating_heatmap_punish_neuron(stim_ax06, neural_trials, labels)
    plot_ppc_2nd_grating_heatmap_reward_trial(stim_ax07, neural_trials, labels)
    plot_ppc_2nd_grating_heatmap_punish_trial(stim_ax08, neural_trials, labels)
    
    # behavior results.
    beh_ax01 = plt.subplot(gs[0, 8])
    beh_ax02 = plt.subplot(gs[1, 8])
    beh_ax03 = plt.subplot(gs[2, 8])
    beh_ax04 = plt.subplot(gs[3, 8])
    plot_beh_outcome(beh_ax01, neural_trials)
    plot_align_pos_reward(beh_ax02, neural_trials)
    plot_align_pos_vis1(beh_ax03, neural_trials)
    plot_align_pos_vis2(beh_ax04, neural_trials)
    
    # outcome average.
    outcome_ax01 = plt.subplot(gs[4, 4])
    outcome_ax02 = plt.subplot(gs[5, 4])
    plot_ppc_reward(outcome_ax01, neural_trials, labels)
    plot_ppc_punish(outcome_ax02, neural_trials, labels)
    
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
    fig.set_size_inches(63, 42)
    fig.savefig(os.path.join(
        ops['save_path0'], 'figures', 'session_report.pdf'),
        dpi=300)
    plt.close()
    print('Finished visualizing {}'.format(session_data_path))


from plot.fig1_mask import plot_ppc_roi_loc
from plot.fig1_mask import plot_ppc_roi_func
from plot.fig1_mask import plot_ppc_roi_anat
from plot.fig1_mask import plot_ppc_roi_superimpose
from plot.fig1_mask import plot_ppc_roi_masks
from plot.fig2_align_grating import plot_ppc_roi_1st_grating_mean_outcome
from plot.fig2_align_grating import plot_ppc_roi_2nd_grating_mean_outcome
from plot.fig2_align_grating import plot_ppc_roi_1st_grating_reward_hearmap
from plot.fig2_align_grating import plot_ppc_roi_2nd_grating_reward_hearmap
from plot.fig3_align_beh import plot_ppc_roi_reward
from plot.fig3_align_beh import plot_ppc_roi_punish
from plot.fig3_align_beh import plot_ppc_roi_beh_decode
from plot.fig5_raw_traces import plot_ppc_roi_raw_trace



def plot_js_ppc_individual_roi(ops):

    # read data.
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
    
    max_ms = 600000
    if np.max(vol_time) < max_ms:
        trace_num_fig = 1
    else:
        trace_num_fig = int(np.max(vol_time)/max_ms)
    
    # define grid space for each ROI.
    with PdfPages(os.path.join(
            ops['save_path0'], 'figures', 'roi_report.pdf')
            ) as pdf:
        #dff.shape[0]
        for roi_id in tqdm(range(dff.shape[0])):
            fig = plt.figure(layout='tight')
            gs = GridSpec(2+trace_num_fig, 10, figure=fig)
            
            # masks.
            mask_ax01 = plt.subplot(gs[0:2, 0:2])
            mask_ax02 = plt.subplot(gs[0, 2])
            mask_ax03 = plt.subplot(gs[0, 3])
            mask_ax04 = plt.subplot(gs[1, 2])
            mask_ax05 = plt.subplot(gs[1, 3])
            plot_ppc_roi_loc(mask_ax01, roi_id, max_func, mean_anat, masks, labels)
            plot_ppc_roi_func(mask_ax02, roi_id, max_func, masks)
            plot_ppc_roi_anat(mask_ax03, roi_id, mean_anat, masks)
            plot_ppc_roi_superimpose(mask_ax04, roi_id, max_func, mean_anat, masks)
            plot_ppc_roi_masks(mask_ax05, roi_id, masks)
    
            # grating average.
            stim_ax01 = plt.subplot(gs[0, 4])
            stim_ax02 = plt.subplot(gs[1, 4])
            stim_ax03 = plt.subplot(gs[0, 5])
            stim_ax04 = plt.subplot(gs[1, 5])
            plot_ppc_roi_1st_grating_mean_outcome(stim_ax01, roi_id, neural_trials)
            plot_ppc_roi_2nd_grating_mean_outcome(stim_ax02, roi_id, neural_trials)
            plot_ppc_roi_1st_grating_reward_hearmap(stim_ax03, roi_id, neural_trials, labels)
            plot_ppc_roi_2nd_grating_reward_hearmap(stim_ax04, roi_id,neural_trials, labels)
            
            # outcome average.
            outcome_ax01 = plt.subplot(gs[0, 6])
            outcome_ax02 = plt.subplot(gs[1, 6])
            outcome_ax03 = plt.subplot(gs[0:2, 7])
            plot_ppc_roi_reward(outcome_ax01, roi_id, neural_trials, labels)
            plot_ppc_roi_punish(outcome_ax02, roi_id, neural_trials, labels)
            plot_ppc_roi_beh_decode(outcome_ax03, roi_id, neural_trials, labels)
            
            # raw traces.
            trace_axs = [plt.subplot(gs[i+2, :]) for i in range(trace_num_fig)]
            plot_ppc_roi_raw_trace(trace_axs, roi_id, max_ms, labels, dff, vol_img_bin, vol_stim_bin, vol_time)
            
            # save figure.
            fig.set_size_inches(60, (2+trace_num_fig)*6)
            pdf.savefig(fig)
            plt.close(fig)
            

if __name__ == "__main__":
    
    session_data_path = 'C:/Users/yhuang887/Projects/joystick_basic_202304/results/FN16_P_20240411_js_t'
    ops = read_ops(session_data_path)
    # Trialization.run(ops)
    #plot_js_ppc_seesion_report(ops)
    plot_js_ppc_individual_roi(ops)