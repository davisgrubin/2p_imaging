# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 10:22:53 2025

@author: saminnaji3
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import fitz
from matplotlib.gridspec import GridSpec
from plot import Basic_alignments_mega_session

def plot_beh(axs, align_data, align_time, trial_type, epoch, outcome, state, type_plot = np.nan):
    
    color_tag = 'k'
    color_all = ['green' , 'y' , 'orange' , 'pink', 'b']
    label_all = ['Reward' , 'DidNotPress1', 'DidNotPress2' , 'LatePress2', 'EarlyPress2']
    
    if np.isnan(type_plot):
        align_data_1 = align_data
        trajectory_mean = np.mean(align_data_1 , axis = 0)
        trajectory_sem = np.std(align_data_1 , axis = 0)/np.sqrt(len(align_data_1))
        axs.fill_between(align_time/1000 ,trajectory_mean-trajectory_sem , trajectory_mean+trajectory_sem, color = color_tag , alpha = 0.2)
        axs.plot(align_time/1000 , trajectory_mean , color = color_tag, linewidth = 0.5, label = label_all[type_plot])
    else:
        align_data_1 = align_data[outcome == type_plot, :]
        align_data_short = align_data[(outcome == type_plot) & (trial_type == 1), :]
        align_data_long = align_data[(outcome == type_plot) & (trial_type == 2) , :]
        color_tag = color_all[type_plot]
        
        trajectory_mean = np.mean(align_data_short , axis = 0)
        trajectory_sem = np.std(align_data_short , axis = 0)/np.sqrt(len(align_data_short))
        axs.fill_between(align_time/1000 ,trajectory_mean-trajectory_sem , trajectory_mean+trajectory_sem, color = color_tag , alpha = 0.2)
        axs.plot(align_time/1000 , trajectory_mean , color = color_tag, linewidth = 0.5, label = label_all[type_plot])
        
        trajectory_mean = np.mean(align_data_long , axis = 0)
        trajectory_sem = np.std(align_data_long , axis = 0)/np.sqrt(len(align_data_long))
        axs.fill_between(align_time/1000 ,trajectory_mean-trajectory_sem , trajectory_mean+trajectory_sem, color = color_tag , alpha = 0.2)
        axs.plot(align_time/1000 , trajectory_mean , color = color_tag, linewidth = 0.5, label = label_all[type_plot], linestyle = '--')
    
    
    axs.axvline(x = 0, color = 'gray', linestyle='--')
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    axs.set_xlabel('Time (s)')
    axs.set_ylabel('joystick deflection (deg)')
    axs.set_xlim([-0.5 , 2])
    axs.set_title('aligned on '+ state)
    
def plot_2p(axs, align_data, align_time, trial_type, epoch, outcome, state, type_plot = np.nan):
    color_tag = 'k'
    color_all = ['green' , 'y' , 'orange' , 'pink', 'b']
    if np.isnan(type_plot):
        align_data_1 = align_data
        trajectory_mean = np.mean(align_data_1 , axis = 0)
        trajectory_sem = np.std(align_data_1 , axis = 0)/np.sqrt(len(align_data_1))
        axs.fill_between(align_time/1000 ,trajectory_mean-trajectory_sem , trajectory_mean+trajectory_sem, color = color_tag , alpha = 0.2)
        axs.plot(align_time/1000 , trajectory_mean , color = color_tag, linewidth = 0.5)
    else:
        align_data_1 = align_data[outcome == type_plot, :]
        align_data_short = align_data[(outcome == type_plot) & (trial_type == 1), :]
        align_data_long = align_data[(outcome == type_plot) & (trial_type == 2) , :]
        color_tag = color_all[type_plot]
        
        trajectory_mean = np.mean(align_data_short , axis = 0)
        trajectory_sem = np.std(align_data_short , axis = 0)/np.sqrt(len(align_data_short))
        axs.fill_between(align_time/1000 ,trajectory_mean-trajectory_sem , trajectory_mean+trajectory_sem, color = color_tag , alpha = 0.2)
        axs.plot(align_time/1000 , trajectory_mean , color = color_tag, linewidth = 0.5)
        
        trajectory_mean = np.mean(align_data_long , axis = 0)
        trajectory_sem = np.std(align_data_long , axis = 0)/np.sqrt(len(align_data_long))
        axs.fill_between(align_time/1000 ,trajectory_mean-trajectory_sem , trajectory_mean+trajectory_sem, color = color_tag , alpha = 0.2)
        axs.plot(align_time/1000 , trajectory_mean , color = color_tag, linewidth = 0.5, linestyle = '--')
        
    axs.axvline(x = 0, color = 'gray', linestyle='--')
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    axs.set_xlabel('Time  (s)')
    axs.set_ylabel('df/f')
    axs.set_xlim([-0.5 , 2])
    
def plot_heatmap_neuron(ax, neu_seq_raw, neu_time, ROI_id, trial_type, outcome, state, type_plot = np.nan, sort = 1, short_long = np.nan):
    neu_seq = neu_seq_raw
    if not np.isnan(np.nansum(neu_seq)) and len(neu_seq)>0:
        label_all = ['Reward' , 'DidNotPress1', 'DidNotPress2' , 'LatePress2', 'EarlyPress2']
        mean = []
        for i in np.unique(ROI_id):
            #print(len(trial_type), len(outcome) , len(ROI_id))
            #print(len(neu_seq[(ROI_id == i) & (outcome == type_plot) & (trial_type == short_long), :]))
            mean.append( np.nanmean(neu_seq[(ROI_id == i) & (outcome == type_plot) & (trial_type == short_long), :], axis=0))
        smoothed_mean = np.array([np.convolve(row, np.ones(5)/5, mode='same') for row in mean])
        
        if not sort == 0:
            sort_idx_neu = np.argmax(smoothed_mean, axis=1).reshape(-1).argsort()
            mean = np.array(mean)
            mean = mean[sort_idx_neu,:].copy()
        sl_label = ''
        if short_long == 1:
            sl_label = 'Short '
        elif short_long == 2:
            sl_label = 'Long '
        num_ROI = len(np.unique(ROI_id))
        cmap = plt.cm.inferno
        im = ax.imshow(mean, interpolation='nearest', aspect='auto', vmin = -0.5, vmax = 0.7, cmap=cmap, extent = [np.min(neu_time)/1000 , np.max(neu_time)/1000, 0 , int(num_ROI)])
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.tick_params(tick1On=False)
        if not np.isnan(mean).all():
            if not sort == 0:
                ax.set_ylabel('neuron id (sorted)')
            else:
                ax.set_ylabel('neuron id')
            ax.tick_params(tick1On=True)
            ax.set_yticks([])
            ax.set_xlabel('Time (s)')
            ax.axvline(0, color='black', lw=1, label='stim', linestyle='--')
            num_ROI = len(np.unique(ROI_id))
            ax.set_yticks([0, int(num_ROI/3), int(num_ROI*2/3), int(num_ROI)])
            ax.set_yticklabels([0, int(num_ROI/3), int(num_ROI*2/3), int(num_ROI)])
            ax.set_xlim([-0.5 , 2])
            ax.set_title(sl_label + label_all[type_plot])
            
    return im

def run(mega_session_data, output_dir_onedrive, subject, st):
    subject_report = fitz.open()
    fig = plt.figure(layout='constrained', figsize=(30, 15))
    gs = GridSpec(12 , 11, figure=fig)
    if st == 0:
        all_states = ['trial_vis1' , 'trial_push1' , 'trial_retract1_init' , 'trial_retract1'  , 'trial_vis2', 'trial_wait2', 'trial_push2', 'trial_reward', 'trial_punish' ,'trial_retract2',  'trial_iti']
        title_str = 'All'
        legend_loc = 9
        iti_loc = 10
    elif st == 1:
        all_states = ['trial_vis1' , 'trial_push1' , 'trial_retract1_init' , 'trial_retract1', 'trial_wait2', 'trial_push2', 'trial_reward', 'trial_punish' ,'trial_retract2',  'trial_iti']
        title_str = 'Self_Timed'
        legend_loc = 8
        iti_loc = 9
    else:
        all_states = ['trial_vis1' , 'trial_push1' , 'trial_retract1_init' , 'trial_retract1'  , 'trial_vis2', 'trial_wait2', 'trial_push2', 'trial_reward', 'trial_punish' ,'trial_retract2',  'trial_iti']
        title_str = 'Vissually_Guided'
        legend_loc = 9
        iti_loc = 10
    
    num = 0
    for state in all_states:
        print(state)
        [align_data, align_time, trial_type, epoch, outcome] = Basic_alignments_mega_session.get_js_pos(mega_session_data, state)
        if state == 'trial_iti':
            [neu_seq, neu_time, ROI_id, outcome_label, post_outcome_label, outcome, post_outcome, stim_seq, delay, delay_label, epoch] = Basic_alignments_mega_session.get_stim_response(mega_session_data, state,104, 1, end_align = 1)
            plot_2p(plt.subplot(gs[1, num]), neu_seq, neu_time, delay_label, epoch, post_outcome_label, state, 0)
            plot_2p(plt.subplot(gs[1, num]), neu_seq, neu_time, delay_label, epoch, post_outcome_label, state, 4)
            plot_2p(plt.subplot(gs[1, num]), neu_seq, neu_time, delay_label, epoch, post_outcome_label, state, 3)
            plot_2p(plt.subplot(gs[1, num]), neu_seq, neu_time, delay_label, epoch, post_outcome_label, state, 2)
            plot_2p(plt.subplot(gs[1, num]), neu_seq, neu_time, delay_label, epoch, post_outcome_label, state, 1)
            plot_heatmap_neuron(plt.subplot(gs[2, num]), neu_seq, neu_time, ROI_id, delay_label, post_outcome_label, state, type_plot = 0, sort = 1, short_long = 1)
            plot_heatmap_neuron(plt.subplot(gs[4, num]), neu_seq, neu_time, ROI_id, delay_label, post_outcome_label, state, type_plot = 4, sort = 1, short_long = 1)
            plot_heatmap_neuron(plt.subplot(gs[6, num]), neu_seq, neu_time, ROI_id, delay_label, post_outcome_label, state, type_plot = 3, sort = 1, short_long = 1)
            plot_heatmap_neuron(plt.subplot(gs[8, num]), neu_seq, neu_time, ROI_id, delay_label, post_outcome_label, state, type_plot = 2, sort = 1, short_long = 1)
            plot_heatmap_neuron(plt.subplot(gs[10, num]), neu_seq, neu_time, ROI_id, delay_label, post_outcome_label, state, type_plot = 1, sort = 1, short_long = 1)
            plot_heatmap_neuron(plt.subplot(gs[3, num]), neu_seq, neu_time, ROI_id, delay_label, post_outcome_label, state, type_plot = 0, sort = 1, short_long = 2)
            plot_heatmap_neuron(plt.subplot(gs[5, num]), neu_seq, neu_time, ROI_id, delay_label, post_outcome_label, state, type_plot = 4, sort = 1, short_long = 2)
            plot_heatmap_neuron(plt.subplot(gs[7, num]), neu_seq, neu_time, ROI_id, delay_label, post_outcome_label, state, type_plot = 3, sort = 1, short_long = 2)
            plot_heatmap_neuron(plt.subplot(gs[9, num]), neu_seq, neu_time, ROI_id, delay_label, post_outcome_label, state, type_plot = 2, sort = 1, short_long = 2)
            plot_heatmap_neuron(plt.subplot(gs[11, num]), neu_seq, neu_time, ROI_id, delay_label, post_outcome_label, state, type_plot = 1, sort = 1, short_long = 2)
            for i in range(1,12):
                plt.subplot(gs[i, num]).set_xlim([-2,0.01])
        
        else:
            plot_beh(plt.subplot(gs[0, num]), align_data, align_time, trial_type, epoch, outcome, state, 0)
            plot_beh(plt.subplot(gs[0, num]), align_data, align_time, trial_type, epoch, outcome, state, 4)
            plot_beh(plt.subplot(gs[0, num]), align_data, align_time, trial_type, epoch, outcome, state, 3)
            plot_beh(plt.subplot(gs[0, num]), align_data, align_time, trial_type, epoch, outcome, state, 2)
            plot_beh(plt.subplot(gs[0, num]), align_data, align_time, trial_type, epoch, outcome, state, 1)
            [neu_seq, neu_time, ROI_id, outcome_label, outcome, post_outcome_label, post_outcome, stim_seq, delay, delay_label, epoch] = Basic_alignments_mega_session.get_stim_response(mega_session_data, state,15, 90)
            plot_2p(plt.subplot(gs[1, num]), neu_seq, neu_time, delay_label, epoch, outcome_label, state, 0)
            plot_2p(plt.subplot(gs[1, num]), neu_seq, neu_time, delay_label, epoch, outcome_label, state, 1)
            plot_2p(plt.subplot(gs[1, num]), neu_seq, neu_time, delay_label, epoch, outcome_label, state, 2)
            plot_2p(plt.subplot(gs[1, num]), neu_seq, neu_time, delay_label, epoch, outcome_label, state, 3)
            plot_2p(plt.subplot(gs[1, num]), neu_seq, neu_time, delay_label, epoch, outcome_label, state, 4)
            plot_heatmap_neuron(plt.subplot(gs[2, num]), neu_seq, neu_time, ROI_id, delay_label, outcome_label, state, type_plot = 0, sort = 1, short_long = 1)
            plot_heatmap_neuron(plt.subplot(gs[4, num]), neu_seq, neu_time, ROI_id, delay_label, outcome_label, state, type_plot = 4, sort = 1, short_long = 1)
            plot_heatmap_neuron(plt.subplot(gs[6, num]), neu_seq, neu_time, ROI_id, delay_label, outcome_label, state, type_plot = 3, sort = 1, short_long = 1)
            plot_heatmap_neuron(plt.subplot(gs[8, num]), neu_seq, neu_time, ROI_id, delay_label, outcome_label, state, type_plot = 2, sort = 1, short_long = 1)
            plot_heatmap_neuron(plt.subplot(gs[10, num]), neu_seq, neu_time, ROI_id, delay_label, outcome_label, state, type_plot = 1, sort = 1, short_long = 1)
            plot_heatmap_neuron(plt.subplot(gs[3, num]), neu_seq, neu_time, ROI_id, delay_label, outcome_label, state, type_plot = 0, sort = 1, short_long = 2)
            plot_heatmap_neuron(plt.subplot(gs[5, num]), neu_seq, neu_time, ROI_id, delay_label, outcome_label, state, type_plot = 4, sort = 1, short_long = 2)
            plot_heatmap_neuron(plt.subplot(gs[7, num]), neu_seq, neu_time, ROI_id, delay_label, outcome_label, state, type_plot = 3, sort = 1, short_long = 2)
            plot_heatmap_neuron(plt.subplot(gs[9, num]), neu_seq, neu_time, ROI_id, delay_label, outcome_label, state, type_plot = 2, sort = 1, short_long = 2)
            plot_heatmap_neuron(plt.subplot(gs[11, num]), neu_seq, neu_time, ROI_id, delay_label, outcome_label, state, type_plot = 1, sort = 1, short_long = 2)
            #plt.colorbar(im , ax = plt.subplot(gs[10, 0]))
        
        num = num + 1
    #plt.subplot(gs[0, legend_loc]).legend()
    plt.subplot(gs[1, iti_loc]).set_title('previous trial ITI')
    plt.suptitle(subject + ' ' + title_str)
    fname = os.path.join(str(0).zfill(4)+'.pdf')
    fig.set_size_inches(30, 15)
    fig.savefig(fname, dpi=300)
    plt.close()
    roi_fig = fitz.open(fname)
    subject_report.insert_pdf(roi_fig)
    roi_fig.close()
    os.remove(fname)
    subject_report.save(output_dir_onedrive + 'Summary_Short_Long_' + title_str + '.pdf')
    subject_report.close()
    
def run_sep(mega_session_data, output_dir_onedrive, subject, st, s_l):
    subject_report = fitz.open()
    fig = plt.figure(layout='constrained', figsize=(30, 15))
    gs = GridSpec(7 , 11, figure=fig)
    if st == 0:
        all_states = ['trial_vis1' , 'trial_push1' , 'trial_retract1_init' , 'trial_retract1'  , 'trial_vis2', 'trial_wait2', 'trial_push2', 'trial_reward', 'trial_punish' ,'trial_retract2',  'trial_iti']
        title_str = 'All'
        legend_loc = 9
        iti_loc = 10
    elif st == 1:
        all_states = ['trial_vis1' , 'trial_push1' , 'trial_retract1_init' , 'trial_retract1', 'trial_wait2', 'trial_push2', 'trial_reward', 'trial_punish' ,'trial_retract2',  'trial_iti']
        title_str = 'Self_Timed'
        legend_loc = 8
        iti_loc = 9
    else:
        all_states = ['trial_vis1' , 'trial_push1' , 'trial_retract1_init' , 'trial_retract1'  , 'trial_vis2', 'trial_wait2', 'trial_push2', 'trial_reward', 'trial_punish' ,'trial_retract2',  'trial_iti']
        title_str = 'Vissually_Guided'
        legend_loc = 9
        iti_loc = 10
    
    num = 0
    for state in all_states:
        print(state)
        [align_data, align_time, trial_type, epoch, outcome] = Basic_alignments_mega_session.get_js_pos(mega_session_data, state)
        if state == 'trial_iti':
            [neu_seq, neu_time, ROI_id, outcome_label, post_outcome_label, outcome, post_outcome, stim_seq, delay, delay_label, epoch] = Basic_alignments_mega_session.get_stim_response(mega_session_data, state,104, 1, end_align = 1)
            plot_2p(plt.subplot(gs[1, num]), neu_seq, neu_time, delay_label, epoch, post_outcome_label, state, 0)
            plot_2p(plt.subplot(gs[1, num]), neu_seq, neu_time, delay_label, epoch, post_outcome_label, state, 4)
            plot_2p(plt.subplot(gs[1, num]), neu_seq, neu_time, delay_label, epoch, post_outcome_label, state, 3)
            plot_2p(plt.subplot(gs[1, num]), neu_seq, neu_time, delay_label, epoch, post_outcome_label, state, 2)
            plot_2p(plt.subplot(gs[1, num]), neu_seq, neu_time, delay_label, epoch, post_outcome_label, state, 1)
            plot_heatmap_neuron(plt.subplot(gs[2, num]), neu_seq, neu_time, ROI_id, delay_label, post_outcome_label, state, type_plot = 0, sort = 1, short_long = s_l)
            plot_heatmap_neuron(plt.subplot(gs[3, num]), neu_seq, neu_time, ROI_id, delay_label, post_outcome_label, state, type_plot = 4, sort = 1, short_long = s_l)
            plot_heatmap_neuron(plt.subplot(gs[4, num]), neu_seq, neu_time, ROI_id, delay_label, post_outcome_label, state, type_plot = 3, sort = 1, short_long = s_l)
            plot_heatmap_neuron(plt.subplot(gs[5, num]), neu_seq, neu_time, ROI_id, delay_label, post_outcome_label, state, type_plot = 2, sort = 1, short_long = s_l)
            plot_heatmap_neuron(plt.subplot(gs[6, num]), neu_seq, neu_time, ROI_id, delay_label, post_outcome_label, state, type_plot = 1, sort = 1, short_long = s_l)
            for i in range(1,7):
                plt.subplot(gs[i, num]).set_xlim([-2,0.01])
        
        else:
            plot_beh(plt.subplot(gs[0, num]), align_data, align_time, trial_type, epoch, outcome, state, 0)
            plot_beh(plt.subplot(gs[0, num]), align_data, align_time, trial_type, epoch, outcome, state, 4)
            plot_beh(plt.subplot(gs[0, num]), align_data, align_time, trial_type, epoch, outcome, state, 3)
            plot_beh(plt.subplot(gs[0, num]), align_data, align_time, trial_type, epoch, outcome, state, 2)
            plot_beh(plt.subplot(gs[0, num]), align_data, align_time, trial_type, epoch, outcome, state, 1)
            [neu_seq, neu_time, ROI_id, outcome_label, outcome, post_outcome_label, post_outcome, stim_seq, delay, delay_label, epoch] = Basic_alignments_mega_session.get_stim_response(mega_session_data, state,15, 90)
            plot_2p(plt.subplot(gs[1, num]), neu_seq, neu_time, delay_label, epoch, outcome_label, state, 0)
            plot_2p(plt.subplot(gs[1, num]), neu_seq, neu_time, delay_label, epoch, outcome_label, state, 1)
            plot_2p(plt.subplot(gs[1, num]), neu_seq, neu_time, delay_label, epoch, outcome_label, state, 2)
            plot_2p(plt.subplot(gs[1, num]), neu_seq, neu_time, delay_label, epoch, outcome_label, state, 3)
            plot_2p(plt.subplot(gs[1, num]), neu_seq, neu_time, delay_label, epoch, outcome_label, state, 4)
            plot_heatmap_neuron(plt.subplot(gs[2, num]), neu_seq, neu_time, ROI_id, delay_label, outcome_label, state, type_plot = 0, sort = 1, short_long = s_l)
            plot_heatmap_neuron(plt.subplot(gs[3, num]), neu_seq, neu_time, ROI_id, delay_label, outcome_label, state, type_plot = 4, sort = 1, short_long = s_l)
            plot_heatmap_neuron(plt.subplot(gs[4, num]), neu_seq, neu_time, ROI_id, delay_label, outcome_label, state, type_plot = 3, sort = 1, short_long = s_l)
            plot_heatmap_neuron(plt.subplot(gs[5, num]), neu_seq, neu_time, ROI_id, delay_label, outcome_label, state, type_plot = 2, sort = 1, short_long = s_l)
            plot_heatmap_neuron(plt.subplot(gs[6, num]), neu_seq, neu_time, ROI_id, delay_label, outcome_label, state, type_plot = 1, sort = 1, short_long = s_l)
            #plt.colorbar(im , ax = plt.subplot(gs[10, 0]))
        
        num = num + 1
    #plt.subplot(gs[0, legend_loc]).legend()
    plt.subplot(gs[1, iti_loc]).set_title('previous trial ITI')
    plt.suptitle(subject + ' ' + title_str)
    fname = os.path.join(str(0).zfill(4)+'.pdf')
    fig.set_size_inches(30, 15)
    fig.savefig(fname, dpi=300)
    plt.close()
    roi_fig = fitz.open(fname)
    subject_report.insert_pdf(roi_fig)
    roi_fig.close()
    os.remove(fname)
    subject_report.save(output_dir_onedrive + 'Summary_Short_Long_sep_' + title_str + '.pdf')
    subject_report.close()