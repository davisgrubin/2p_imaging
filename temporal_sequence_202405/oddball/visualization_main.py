#!/usr/bin/env python3

import warnings
warnings.filterwarnings('ignore')

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

from plot.fig1_mask import plotter_main_masks
from plot.fig2_align_stim import plotter_main_align_stim
from plot.fig3_align_odd import plotter_main_align_odd
from plot.fig5_raw_traces import plot_sess_example_traces
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

def run(session_config):

    def plot_session_report():
        fig = plt.figure(figsize=(140, 210), layout='tight')
        gs = GridSpec(30, 20, figure=fig)
        # masks.
        print('Plotting masks')
        if len(session_config['label_names'])==1:
            mask_axs = [plt.subplot(gs[0:2, j:j+2]) for j in range(0,12,2)]
            plotter_masks.all_1chan(mask_axs)
        if len(session_config['label_names'])==2:
            mask_axs = [plt.subplot(gs[i:i+2, j:j+2]) for j in range(0,12,2) for i in [0,2]]
            plotter_masks.all_2chan(mask_axs)
        # normal alignment.
        print('Plotting normal alignment')
        normal_axs01 = [plt.subplot(gs[5, i]) for i in range(3)]
        normal_axs01+= [plt.subplot(gs[5, 3], projection='3d')]
        normal_axs02 = [plt.subplot(gs[6, i]) for i in range(3)]
        normal_axs02+= [plt.subplot(gs[6, 3], projection='3d')]
        normal_axs03 = [plt.subplot(gs[5:7, 4])]
        plotter_align_stim.normal_exc(normal_axs01)
        plotter_align_stim.normal_inh(normal_axs02)
        plotter_align_stim.normal_heatmap(normal_axs03)
        # selectivity alignment.
        print('Plotting selectivity alignment')
        select_axs01 = [plt.subplot(gs[5, i+7]) for i in range(6)]
        select_axs01+= [plt.subplot(gs[7, i+7]) for i in range(5)]
        select_axs01+= [plt.subplot(gs[7, 12], projection='3d')]
        select_axs02 = [plt.subplot(gs[6, i+7]) for i in range(6)]
        select_axs02+= [plt.subplot(gs[8, i+7]) for i in range(5)]
        select_axs02+= [plt.subplot(gs[8, 12], projection='3d')]
        plotter_align_stim.select_exc(select_axs01)
        plotter_align_stim.select_inh(select_axs02)
        # oddball alignment.
        print('Plotting oddball alignment')
        odd_normal_axs01 = [plt.subplot(gs[10, 0:2]), plt.subplot(gs[10, 2:4]), plt.subplot(gs[10, 4:6])]
        odd_normal_axs01+= [plt.subplot(gs[10, i+6]) for i in range(6)]
        odd_normal_axs02 = [plt.subplot(gs[11, 0:2]), plt.subplot(gs[11, 2:4]), plt.subplot(gs[11, 4:6])]
        odd_normal_axs02+= [plt.subplot(gs[11, i+6]) for i in range(6)]
        odd_normal_axs03 = [plt.subplot(gs[12:14, i]) for i in range(4)]
        odd_normal_axs04 = [plt.subplot(gs[12, 4:6]), plt.subplot(gs[12, 6])]
        odd_normal_axs04+= [plt.subplot(gs[12, 7], projection='3d')]
        odd_normal_axs05 = [plt.subplot(gs[13, 4:6]), plt.subplot(gs[13, 6])]
        odd_normal_axs05+= [plt.subplot(gs[13, 7], projection='3d')]
        plotter_align_odd.odd_normal_exc(odd_normal_axs01)
        plotter_align_odd.odd_normal_inh(odd_normal_axs02)
        plotter_align_odd.odd_normal_heatmap(odd_normal_axs03)
        plotter_align_odd.odd_normal_pop_exc(odd_normal_axs04)
        plotter_align_odd.odd_normal_pop_inh(odd_normal_axs05)
        # cluster analysis.
        cluster_axs01 = [
            plt.subplot(gs[15:18, 0:3]),
            plt.subplot(gs[15, 3]), plt.subplot(gs[16, 3]), plt.subplot(gs[17, 3]),
            plt.subplot(gs[15:18, 4]), plt.subplot(gs[15:18, 5]), plt.subplot(gs[15:18, 6])]
        cluster_axs02 = [       
            plt.subplot(gs[18:21, 0:3]),
            plt.subplot(gs[18, 3]), plt.subplot(gs[19, 3]), plt.subplot(gs[20, 3]),
            plt.subplot(gs[18:21, 4]), plt.subplot(gs[18:21, 5]), plt.subplot(gs[18:21, 6])]
        plotter_align_odd.odd_cluster_exc(cluster_axs01)
        plotter_align_odd.odd_cluster_inh(cluster_axs02)
        # example traces.
        print('Plotting example traces')
        example_ax = plt.subplot(gs[0:4, 13])
        plot_sess_example_traces(example_ax, list_dff, list_labels, list_vol, session_config['label_names'])
        print('Plotting stimulus types')
        # isi distribution.
        isi_ax01 = plt.subplot(gs[0, 15])
        isi_ax02 = plt.subplot(gs[0, 16])
        plot_isi_distribution(isi_ax01, list_neural_trials)
        plotter_align_odd.plot_pre_odd_isi_distribution(isi_ax02)
        # stimulus types.
        type_ax01 = plt.subplot(gs[1, 15])
        type_ax02 = plt.subplot(gs[1, 16])
        type_ax03 = plt.subplot(gs[2, 15])
        type_ax04 = plt.subplot(gs[2, 16])
        plot_stim_type(type_ax01, list_neural_trials)
        plot_normal_type(type_ax02, list_neural_trials)
        plot_fix_jitter_type(type_ax03, list_neural_trials)
        plot_oddball_type(type_ax04, list_neural_trials)
        print('Plotting 2p misc results')
        # used sessions.
        sess_ax = plt.subplot(gs[0, 12])
        plot_sess_name(sess_ax, list_session_data_path)
        # labels.
        label_ax = plt.subplot(gs[1, 12])
        plot_inh_exc_label_pc(label_ax, list_labels)
        # significance.
        sign_ax = plt.subplot(gs[2, 12])
        plot_significance(sign_ax, list_significance, list_labels)
        # offset.
        offset_ax = plt.subplot(gs[3, 12])
        plot_motion_offset_hist(offset_ax, list_move_offset)
        # save figure.
        fig.set_size_inches(140, 210)
        fig.savefig(os.path.join('results', session_config['output_filename']), dpi=300)
        plt.close()

    # main
    print('===============================================')
    print('============ Start Processing Data ============')
    print('===============================================')
    list_session_data_path = [
        os.path.join('results', n)
        for n in session_config['list_session_name'].keys()]
    print('Processing {} sessions'.format(len(list_session_data_path)))
    for session_data_path in list_session_data_path:
        print(session_data_path.split('/')[-1])
    print('Reading ops.npy')
    list_ops = read_ops(list_session_data_path)
    print('===============================================')
    print('============= trials segmentation =============')
    print('===============================================')
    #for ops in list_ops:
    #    Trialization.run(ops)
    print('===============================================')
    print('============== significance test ==============')
    print('===============================================')
    #for ops in list_ops:
    #    StatTest.run(ops)
    print('===============================================')
    print('============ reading saved results ============')
    print('===============================================')
    [list_labels, list_masks, list_vol, list_dff,
     list_neural_trials, list_move_offset, list_significance
     ] = read_all(list_ops, session_config['sig_tag'], session_config['force_label'])
    print('Preparing masks')
    plotter_masks = plotter_main_masks(
        list_labels[0],
        list_masks[0][0],
        list_masks[0][1],
        list_masks[0][2],
        list_masks[0][3],
        list_masks[0][4])
    print('Preparing stimulus alignments')
    plotter_align_stim = plotter_main_align_stim(
        list_neural_trials, list_labels, list_significance, session_config['label_names'])
    print('Preparing omission alignments')
    plotter_align_odd = plotter_main_align_odd(
        list_neural_trials, list_labels, list_significance, session_config['label_names'])
    print('===============================================')
    print('=========== plotting session report ===========')
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
    print('===============================================')
    print('Processing completed')
    print('File saved as '+os.path.join('results', session_config['output_filename']))

if __name__ == "__main__":
    
    session_config = {
        'list_session_name' : {
            'VT02_PPC_20241129_seq1131_t' : 'oddball',
            'VT02_PPC_20241130_seq1131_t' : 'oddball',
            'VT02_PPC_20241201_seq1131_t' : 'oddball',
            'VT02_PPC_20241205_seq1131_t' : 'oddball',
            'VT02_PPC_20241209_seq1131_t' : 'oddball',
            },
        'sig_tag' : 'all',
        'label_names' : {
            '-1':'excitatory',
            '1':'inhibitory'
            },
        'force_label' : None,
        'output_filename' : 'VT02_PPC_oddball.pdf'
        }
    run(session_config)
    
    '''

    session_config = {
        'list_session_name' : {
            'FN14_PPC_20241129_seq1131_t' : 'oddball',
            'FN14_PPC_20241130_seq1131_t' : 'oddball',
            'FN14_PPC_20241202_seq1131_t' : 'oddball',
            'FN14_PPC_20241209_seq1131_t' : 'oddball',
            },
        'sig_tag' : 'all',
        'label_names' : {
            '-1':'excitatory',
            '1':'inhibitory'
            },
        'force_label' : None,
        'output_filename' : 'FN14_PPC_oddball.pdf'
        }
    run(session_config)
    
    session_config = {
        'list_session_name' : {
            'VT02_PPC_20241129_seq1131_t' : 'oddball',
            'VT02_PPC_20241130_seq1131_t' : 'oddball',
            'VT02_PPC_20241201_seq1131_t' : 'oddball',
            'VT02_PPC_20241205_seq1131_t' : 'oddball',
            'VT02_PPC_20241209_seq1131_t' : 'oddball',
            },
        'sig_tag' : 'all',
        'label_names' : {
            '-1':'excitatory',
            '1':'inhibitory'
            },
        'force_label' : None,
        'output_filename' : 'VT02_PPC_oddball.pdf'
        }
    run(session_config)
    
    session_config = {
        'list_session_name' : {
            'E4LG_CRBL_crux2_20241202_seq1131_t' : 'crux2',
            'E4LG_CRBL_lobulevi_20241205_seq1131_t' : 'lobulevi',
            'E4LG_CRBL_crux1_20241203_seq1131_t' : 'crux1',
            'E4LG_CRBL_crux1_20241213_seq1131_t' : 'crux1',
            'E4LG_CRBL_crux1_20241216_seq1131_t' : 'crux1',
            },
        'sig_tag' : 'all',
        'label_names' : {
            '-1':'dendrites',
            },
        'force_label' : -1,
        'output_filename' : 'E4LG_CRBL_oddball.pdf'
        }
    run(session_config)
    
    list_session_data_path = [
        'C:/Users/yhuang887/Projects/temporal_sequence_202405/oddball/results/FN14_P_20240627_seq1130_t',
        'C:/Users/yhuang887/Projects/temporal_sequence_202405/oddball/results/FN14_P_20240701_seq1130_t',
        'C:/Users/yhuang887/Projects/temporal_sequence_202405/oddball/results/FN14_P_20240702_seq1130_t',
        'C:/Users/yhuang887/Projects/temporal_sequence_202405/oddball/results/FN14_P_20240705_seq1130_t',
        'C:/Users/yhuang887/Projects/temporal_sequence_202405/oddball/results/FN14_P_20240708_seq1131_t',
        'C:/Users/yhuang887/Projects/temporal_sequence_202405/oddball/results/FN14_P_20240710_seq1130_t',
        'C:/Users/yhuang887/Projects/temporal_sequence_202405/oddball/results/FN14_P_20241021_seq1131_t',
        ]
    
    list_session_data_path = [
        os.path.join('results', n)
        for n in session_config['list_session_name'].keys()]
    list_ops = read_ops(list_session_data_path)
    [list_labels, list_masks, list_vol, list_dff,
     list_neural_trials, list_move_offset, list_significance
     ] = read_all(list_ops, session_config['sig_tag'], session_config['force_label'])

    ops = list_ops[0]
    plt.plot(vol_time, vol_stim_vis)
    
    run(list_session_data_path, 'sig')
    '''
