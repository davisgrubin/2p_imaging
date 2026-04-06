# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 08:23:08 2026

@author: saminnaji3
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# Use Type 42 fonts (TrueType) so text is editable in Illustrator
matplotlib.rcParams['pdf.fonttype'] = 42 
# Ensure vector paths aren't simplified into too many small segments
matplotlib.rcParams['path.simplify'] = False
import os
import fitz
from matplotlib.gridspec import GridSpec
from skimage.segmentation import find_boundaries
from scipy.interpolate import interp1d

from Modules import ReadResults 

colors = np.array([
            [255, 105, 180],  # hot pink
            [  0, 114, 178],  # blue
            [127, 255, 212],  # aquamarine
            [230,  25,  75],  # red
            [255, 225,  25],  # yellow
            [  0, 225,   0],  # green
            [255,   0,  25],  # purple
            [ 77, 175, 124],  # teal
            [152,   0, 119],  # fuchsia
            [255, 105,   0],  # orange
            [  0,  76,  153],  # dark blue
            [204,  121, 167],  # rose
            [  0, 255, 255],  # cyan
            [127, 255, 212],  # aquamarine
            [255, 105, 180],  # hot pink
            [  0, 0, 255],    # blue (pure)
            [213,  94,   0],  # vermilion (orange-red)
            [0, 0, 0], #black
            [100, 100, 100], #gray
            [  0,  76,  153],  # dark blue
        ], dtype=np.uint8)

    
def save_temp_fig(fig, report):
    fname = os.path.join(str(0).zfill(4)+'.pdf')
    fig.set_size_inches(30, 15)
    fig.savefig(fname, dpi=300)
    plt.close()
    temp_fig = fitz.open(fname)
    report.insert_pdf(temp_fig)
    temp_fig.close()
    os.remove(fname)
    
def adjust_contrast(org_img, lower_percentile=50, upper_percentile=99):
    lower = np.percentile(org_img, lower_percentile)
    upper = np.percentile(org_img, upper_percentile)
    img = np.clip((org_img - lower) * 255 / (upper - lower), 0, 255)
    img = img.astype('int32')
    return img

def get_labeled_masks_img(masks, labels, cate):
    neuron_idx = np.where(labels == cate)[0] + 1
    labeled_masks_img = np.zeros(
        (masks.shape[0], masks.shape[1]), dtype='int64')
    for i in range(masks.shape[0]):
        for j in range(masks.shape[1]):
            if masks[i,j] in neuron_idx:
                labeled_masks_img[i,j] = masks[i,j]
    return labeled_masks_img

def plot_fov_blank(file_paths_1, output_dir_onedrive, img = 'mean', with_mask=False):
    report = fitz.open()
    n_row = 3
    n_col = 5
    for page in range(len(file_paths_1)//(n_row*n_col)+1):
        fig = plt.figure(layout='constrained', figsize=(30, 15))
        gs = GridSpec(n_row , n_col, figure=fig)
        start = page*(n_row*n_col)
        end = min((1+page)*(n_row*n_col), len(file_paths_1))
        for i in range(start, end):
            file_temp = file_paths_1[i]
            page_ind = i - start
            print(file_temp[-8:])
            ops = np.load(
                os.path.join(file_temp, 'suite2p', 'plane0', 'ops.npy'),
                allow_pickle=True).item()
            ops['save_path0'] = os.path.join(file_temp)
            [labels, masks, mean_func, max_func, mean_anat, masks_anat, masks_anat_corrected] = ReadResults.read_masks(ops)
            ax = plt.subplot(gs[page_ind//n_col, np.mod(page_ind,n_col)])
            f = mean_func if img == 'mean' else max_func
            func_img = np.zeros(
                (f.shape[0], f.shape[1], 3), dtype='int32')
            func_img[:, :, 1] = adjust_contrast(f)
            func_img = adjust_contrast(func_img)
            if with_mask:
                x_all, y_all = np.where(find_boundaries(masks))
                for x,y in zip(x_all, y_all):
                    func_img[x,y,:] = np.array([255,255,255])
            ax.matshow(func_img)
            ax.tick_params(tick1On=False)
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])
            
            ax.set_title(img + 'projection: '+file_temp[-8:])
        save_temp_fig(fig, report)
    report.save(output_dir_onedrive+ 'blank_fov.pdf')
    report.close()

def plot_fov(file_paths_1, output_dir_onedrive, img = 'mean'):
    report = fitz.open()
    n_row = 3
    n_col = 5
    for page in range(len(file_paths_1)//(n_row*n_col)+1):
        fig = plt.figure(layout='constrained', figsize=(30, 15))
        gs = GridSpec(n_row , n_col, figure=fig)
        start = page*(n_row*n_col)
        end = min((1+page)*(n_row*n_col), len(file_paths_1))
        for i in range(start, end):
            file_temp = file_paths_1[i]
            page_ind = i - start
            print(file_temp[-8:])
            ops = np.load(
                os.path.join(file_temp, 'suite2p', 'plane0', 'ops.npy'),
                allow_pickle=True).item()
            ops['save_path0'] = os.path.join(file_temp, 'session_data')
            cluster_labels = ReadResults.read_cluster_labels(ops, 'cluster_labels')
            ops['save_path0'] = os.path.join(file_temp)
            [labels, masks, mean_func, max_func, mean_anat, masks_anat, masks_anat_corrected] = ReadResults.read_masks(ops)
            ax = plt.subplot(gs[page_ind//n_col, np.mod(page_ind,n_col)])
            
            f = mean_func if img == 'mean' else max_func
            H, W = f.shape[:2]
        
            # make a 3-channel background from f (you had contrast on G; keep that idea)
            func_img = np.zeros((H, W, 3), dtype=np.uint8)
            bg = adjust_contrast(f)                       # expected to return 0–255
            bg = np.clip(bg, 0, 255).astype(np.uint8)
            func_img[:, :, 1] = bg                        # show background in green channel
            func_img = adjust_contrast(func_img)          # your existing call; keep as is
        
            for c in np.unique(cluster_labels):
                # expect a boolean/0-1 mask of shape (H, W) for category c
                roi_mask = get_labeled_masks_img(masks, cluster_labels, c).astype(bool)
                if roi_mask.any():
                    func_img[roi_mask] = colors[c % len(colors)]
        
            ax.matshow(func_img)
            ax.tick_params(tick1On=False)
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])
           
        save_temp_fig(fig, report)
    report.save(output_dir_onedrive + 'cluster_fov.pdf')
    report.close()
 
def align_arrays(output_array, output_time):
    """
    Aligns a list of arrays along a common time axis and fills missing values with NaN.
    
    Parameters
    ----------
    output_array : list of 1D np.arrays
        Each row contains the values to align.
    output_time : list of 1D np.arrays
        Each row contains the corresponding time points.
        
    Returns
    -------
    aligned_array : 2D np.array
        Each row corresponds to an input array, with NaNs for missing points.
    common_time : 1D np.array
        The common time axis from min to max time.
    """
    # find the overall min and max time
    min_time = min([t[0] for t in output_time])
    max_time = max([t[-1] for t in output_time])
    
    # determine the finest time resolution across all time arrays
    dt = np.min([np.min(np.diff(t)) for t in output_time if len(t) > 1])
    
    # create a common time axis
    n_steps = int(np.round((max_time - min_time) / dt)) + 1
    common_time = min_time + np.arange(n_steps) * dt
    
    # create aligned array with NaNs
    aligned_array = np.full((len(output_array), len(common_time)), np.nan)
    
    for i, (vals, t) in enumerate(zip(output_array, output_time)):
        # find indices in common_time corresponding to each t
        idx = np.searchsorted(common_time, t)
        idx = np.clip(idx, 0, len(common_time)-1)
        aligned_array[i, idx] = vals
        
    return aligned_array, common_time

def align_2d_trials(output_array, output_time):
    # find global time limits
    min_time = min(t[0] for t in output_time)
    max_time = max(t[-1] for t in output_time)

    # smallest timestep
    dt = np.min([np.min(np.diff(t)) for t in output_time if len(t) > 1])

    # build common time axis
    n_steps = int(np.round((max_time - min_time) / dt)) + 1
    common_time = min_time + np.arange(n_steps) * dt

    n_trials = len(output_array)
    n_neurons = output_array[0].shape[0]

    aligned_array = np.full((n_trials, n_neurons, len(common_time)), np.nan)

    for trial_idx, (vals, t) in enumerate(zip(output_array, output_time)):
        idx = np.searchsorted(common_time, t)
        idx = np.clip(idx, 0, len(common_time)-1)
    
        for j, t_idx in enumerate(idx):
            aligned_array[trial_idx, :, t_idx] = vals[:, j]

    return aligned_array, common_time

def create_matrix(idx, input_array, time_array, reference_time, pre, post):
    output_array = []
    output_time = []
    if input_array.ndim == 2:
        #print(1)
        for i in idx:
            output_time.append(time_array[max(0, i-pre):min(len(time_array), i+post)]-time_array[i])
            output_array.append(input_array[:, max(0, i-pre):min(len(time_array), i+post)])
        aligned_array, common_time = align_2d_trials(output_array, output_time)
            
    if input_array.ndim == 1:
        if len(time_array) == len(reference_time):
            for i in idx:
                output_time.append(time_array[max(0, i-pre):min(len(time_array), i+post)]-time_array[i])
                output_array.append(input_array[max(0, i-pre):min(len(time_array), i+post)])      
        else:
            interval = 1
            for i in idx:
                start_time = reference_time[max(0, i-pre)]
                end_time = reference_time[min(len(reference_time)-1, i+post)]
                
                start_idx = np.argmin(np.abs(time_array - start_time))
                end_idx = np.argmin(np.abs(time_array - end_time))
                zero_idx = np.argmin(np.abs(time_array - reference_time[i]))

                # --- ADD THIS CHECK TO PREVENT THE ERROR ---
                # Check if the slice is empty or has fewer than 2 points (interp1d needs at least 2 for linear)
                if start_idx >= end_idx or (end_idx - start_idx) < 2:
                    # Append an array of NaNs based on the expected length
                    new_time = np.arange(start_time, end_time, interval)
                    output_time.append(new_time - time_array[zero_idx])
                    output_array.append(np.full(len(new_time), np.nan))
                    continue 
                # --------------------------------------------

                interpolator = interp1d(time_array[start_idx:end_idx], input_array[start_idx:end_idx], bounds_error=False)
                new_time = np.arange(start_time, end_time, interval)
                new_pos = interpolator(new_time)
                output_time.append(new_time - time_array[zero_idx])
                output_array.append(new_pos)
                
        aligned_array, common_time = align_arrays(output_array, output_time)
    
    return aligned_array, common_time
                
def trace_plot(ax, data, time, color_tag, dim = 1):
    if dim == 1:
        y_values = np.nanmean(data, axis=0)
        y_sem = np.nanstd(data, axis=0) / np.sqrt(np.sum(~np.isnan(data), axis=0))
    else:
        heatmap_values = np.nanmean(data, axis=0)  # shape: (n_neurons, n_timepoints)
        y_values = np.nanmean(data, axis=(0, 1))  
        n_valid = np.sum(~np.isnan(data), axis=(0, 1))
        y_sem = np.nanstd(data, axis=(0, 1)) / np.sqrt(n_valid)  # shape: (n_timepoints,)
        
    ax.plot(time/1000, y_values, color = color_tag, linewidth = 0.3)
    #ax.fill_between(time/1000, y_values-y_sem, y_values+y_sem, color = color_tag, alpha = 0.2)
    ax.fill_between(time/1000, y_values-y_sem, y_values+y_sem, 
                    color=color_tag, 
                    alpha=0.2, 
                    edgecolor='none',  # This removes the white/aliased border
                    linewidth=0,       # Ensures no stroke is drawn around the shaded box
                    antialiased=True)  # Helps Illustrator interpret the boundary better
    ax.axvline(0, color = 'gray', linestyle = '--', linewidth = 0.5)
        
def lay_out_plot(axs, x_label = '', y_label = '', title = '', x_lim = [], y_lim = [], legend = 0, right = 0):
    if not len(x_label) == 0:
        axs.set_xlabel(x_label)
    if not len(y_label) == 0:
        axs.set_ylabel(y_label)
    if not len(title) == 0:
        axs.set_title(title)
    if not len(x_lim) == 0:
        axs.set_xlim(x_lim)
    if not len(y_lim) == 0:
        axs.set_ylim(y_lim)
    if not legend == 0:
        axs.legend()
    if not right:
        axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)

def session_colormap(rgb_color, num_sessions, dim_factor=0.2):
    base = np.array(rgb_color) / 255.0
    dim_color = base * dim_factor

    colors = np.linspace(dim_color, base, num_sessions)

    return colors
    
def select_trials_from_data(neural_data, conditions, combine='and'):
    trial_start_indices = np.where(neural_data['trial_start'] == 1)[0]
    n_trials = len(trial_start_indices)
    mask = np.ones(n_trials, dtype=bool) if combine == 'and' else np.zeros(n_trials, dtype=bool)

    for key, cond in conditions.items():
        # Trialization.py stores info like 'outcome' or 'types' at trial_start indices
        trial_values = neural_data[key][trial_start_indices]
        
        # Build the mask (supports scalar, tuple ('in', [...]), or callable)
        if isinstance(cond, tuple):
            op, val = cond[0], cond[1]
            if op == '==': m = (trial_values == val)
            elif op == 'in': m = np.isin(trial_values, np.array(list(val)))
            # ... add other operators as needed ...
        else:
            m = (trial_values == cond)
        
        mask = (mask & m) if combine == 'and' else (mask | m)

    selected_trial_num = np.where(mask)[0]
    start_time_indices = trial_start_indices[selected_trial_num]
    return selected_trial_num, start_time_indices
        
def make_data_final(output_dir_onedrive, list_neural_data, event_list, pre_list, post_list, list_labels, 
                    grant = 0, conditions = None, plot_mode = 'separate', target_cluster = None,
                    existing_pdf_path = None):
    
    if existing_pdf_path and os.path.exists(existing_pdf_path):
        report = fitz.open(existing_pdf_path)
    else:
        report = fitz.open()

    fig = plt.figure(layout='constrained', figsize=(30, 18))
    gs = GridSpec(6 , len(event_list)+2, figure=fig)
    fs = 30.0 
    
    for event_num, event in enumerate(event_list):
        ax_js = plt.subplot(gs[0, event_num]); ax_vel = plt.subplot(gs[1, event_num]) 
        ax_lick = plt.subplot(gs[2, event_num]); ax_dff = plt.subplot(gs[3, event_num])
        ax_spk = plt.subplot(gs[4, event_num]); ax_coact = plt.subplot(gs[5, event_num]) 
        
        t_grand = (np.arange(pre_list[event_num] + post_list[event_num]) - pre_list[event_num]) * (1000/fs)

        sess_js, sess_vel, sess_lick = [], [], []
        sess_dff = {cat: [] for cat in range(-1, len(colors))}
        sess_spk = {cat: [] for cat in range(-1, len(colors))}
        sess_coact = {cat: [] for cat in range(-1, len(colors))}
        
        # --- NEW: Trial Counter for this event column ---
        total_trials_for_event = 0

        print('Analyzing: ', event)

        for session, neural_data in enumerate(list_neural_data):
            all_starts = np.where(neural_data['trial_start'] == 1)[0]
            all_ends = np.where(neural_data['trial_end'] == 1)[0]
            all_ends = all_ends[all_ends > 0]
            if len(all_ends) < len(all_starts):
                all_ends = np.append(all_ends, len(neural_data['time']) - 1)

            if conditions is not None:
                selected_trial_indices, _ = select_trials_from_data(neural_data, conditions)
                sel_starts = all_starts[selected_trial_indices]
                sel_ends = all_ends[selected_trial_indices]
            else:
                sel_starts, sel_ends = all_starts, all_ends

            all_event_idx = np.where(neural_data[event] == 1)[0]
            idx = []
            for s, e in zip(sel_starts, sel_ends):
                trial_events = all_event_idx[(all_event_idx >= s) & (all_event_idx <= e)]
                idx.extend(trial_events)
            idx = np.array(idx)
            
            if len(idx) == 0: continue
            
            # Update the total trial count
            total_trials_for_event += len(idx)
            
            # --- Position & Velocity Processing ---
            adj_js, t_js = create_matrix(idx, neural_data['js_pos'], neural_data['js_time'], 
                                         neural_data['time'], pre_list[event_num], post_list[event_num])
            vel_matrix = np.array([np.gradient(trial, t_js) for trial in adj_js])
            adj_lick, t_lick = create_matrix(idx, neural_data['lick'], neural_data['time'], 
                                             neural_data['time'], pre_list[event_num], post_list[event_num])

            if grant:
                sess_js.append(np.interp(t_grand, t_js, np.nanmean(adj_js, axis=0), left=np.nan, right=np.nan))
                sess_vel.append(np.interp(t_grand, t_js, np.nanmean(vel_matrix, axis=0), left=np.nan, right=np.nan))
                sess_lick.append(np.interp(t_grand, t_lick, np.nanmean(adj_lick, axis=0), left=np.nan, right=np.nan))
            else:
                color = session_colormap(colors[-1], len(list_neural_data)+3)[session]
                trace_plot(ax_js, adj_js, t_js, color, dim=1)
                trace_plot(ax_vel, vel_matrix, t_js, color, dim=1)
                trace_plot(ax_lick, adj_lick, t_lick, color, dim=1)

            # (Neural Logic code remains the same...)
            current_labels = list_labels[session]
            if plot_mode == 'pool': categories = [-1]
            elif plot_mode == 'specific': categories = [target_cluster] if target_cluster in current_labels else []
            else: categories = np.unique(current_labels)

            for cat in categories:
                if cat == -1:
                    neurons = np.arange(neural_data['dff'].shape[0])
                    c_idx = -1 
                else:
                    neurons = np.squeeze(np.where(current_labels == cat))
                    c_idx = cat
                if neurons.size == 0: continue
                adj_d, t_d = create_matrix(idx, neural_data['dff'][neurons, :], neural_data['time'], 
                                           neural_data['time'], pre_list[event_num], post_list[event_num])
                adj_s, t_s = create_matrix(idx, neural_data['spikes'][neurons, :], neural_data['time'], 
                                           neural_data['time'], pre_list[event_num], post_list[event_num])
                coact_matrix = np.mean(adj_s > 0, axis=1) 
                if grant:
                    n_avg_d = np.nanmean(adj_d, axis=0); n_avg_s = np.nanmean(adj_s, axis=0)
                    sess_dff[c_idx].append(np.array([np.interp(t_grand, t_d, n, left=np.nan, right=np.nan) for n in n_avg_d]))
                    sess_spk[c_idx].append(np.array([np.interp(t_grand, t_s, n, left=np.nan, right=np.nan) for n in n_avg_s]))
                    sess_coact[c_idx].append(np.interp(t_grand, t_s, np.nanmean(coact_matrix, axis=0), left=np.nan, right=np.nan))
                else:
                    cat_color = session_colormap(colors[c_idx], len(list_neural_data)+3)[session]
                    trace_plot(ax_dff, adj_d, t_d, cat_color, dim=2)
                    trace_plot(ax_spk, adj_s, t_s, cat_color, dim=2)
                    trace_plot(ax_coact, coact_matrix, t_s, cat_color, dim=1)

        # --- Plot Grand Average ---
        if grant:
            c_main = colors[-1]/255.0 
            if sess_js: trace_plot(ax_js, np.array(sess_js), t_grand, c_main, dim=1)
            # (Remaining Grand Average traces...)
            if sess_vel: trace_plot(ax_vel, np.array(sess_vel), t_grand, c_main, dim=1)
            if sess_lick: trace_plot(ax_lick, np.array(sess_lick), t_grand, c_main, dim=1)
            for cat in sess_dff:
                if sess_dff[cat]:
                    c_cat = colors[cat]/255.0
                    trace_plot(ax_dff, np.vstack(sess_dff[cat]), t_grand, c_cat, dim=1)
                    trace_plot(ax_spk, np.vstack(sess_spk[cat]), t_grand, c_cat, dim=1)
                    trace_plot(ax_coact, np.array(sess_coact[cat]), t_grand, c_cat, dim=1)

        # --- MODIFIED Formatting: Add Trial Count to Legend or Title ---
        y_labels = ['joystick pos', 'velocity', 'lick rate', 'df/f', 'spikes', 'fraction active']
        for ax, ylab in zip([ax_js, ax_vel, ax_lick, ax_dff, ax_spk, ax_coact], y_labels):
            # If it's the joystick plot, append (n=X) to the title
            current_title = f"{event}\n(n={total_trials_for_event} trials)" if ylab == 'joystick pos' else ''
            
            lay_out_plot(ax, x_label='Time (s)', y_label=ylab, title=current_title, 
                         x_lim=[-pre_list[event_num]/fs, post_list[event_num]/fs])
                
    save_temp_fig(fig, report)
    # Determine save path
    if existing_pdf_path:
        save_path = existing_pdf_path
    else:
        fname = 'grand_averages.pdf' if grant else 'initial_alignments.pdf'
        save_path = os.path.join(output_dir_onedrive, fname)
    
    # Save incrementally (use incremental=True if appending to an existing file)
    if existing_pdf_path and os.path.exists(existing_pdf_path):
        report.save(report.name, incremental=True, encryption=0) #
    else:
        report.save(save_path) #
        
    report.close()
    
def make_data_final_with_pupil(output_dir_onedrive, list_neural_data, event_list, pre_list, post_list, list_labels, 
                    grant = 0, conditions = None, plot_mode = 'separate', target_cluster = None,
                    existing_pdf_path = None):
    
    if existing_pdf_path and os.path.exists(existing_pdf_path):
        report = fitz.open(existing_pdf_path)
    else:
        report = fitz.open()

    fig = plt.figure(layout='constrained', figsize=(30, 21)) # Increased height slightly for 7th row
    gs = GridSpec(7 , len(event_list)+2, figure=fig) # Changed rows from 6 to 7
    fs = 30.0 
    
    for event_num, event in enumerate(event_list):
        ax_js = plt.subplot(gs[0, event_num]); ax_vel = plt.subplot(gs[1, event_num]) 
        ax_lick = plt.subplot(gs[2, event_num]); ax_pupil = plt.subplot(gs[3, event_num]) # New Pupil Row
        ax_dff = plt.subplot(gs[4, event_num]); ax_spk = plt.subplot(gs[5, event_num])
        ax_coact = plt.subplot(gs[6, event_num]) 
        
        t_grand = (np.arange(pre_list[event_num] + post_list[event_num]) - pre_list[event_num]) * (1000/fs)

        sess_js, sess_vel, sess_lick, sess_pupil = [], [], [], [] # Added sess_pupil
        sess_dff = {cat: [] for cat in range(-1, len(colors))}
        sess_spk = {cat: [] for cat in range(-1, len(colors))}
        sess_coact = {cat: [] for cat in range(-1, len(colors))}
        
        total_trials_for_event = 0
        print('Analyzing: ', event)

        for session, neural_data in enumerate(list_neural_data):
            # ... (Trial start/end logic remains the same) ...
            all_starts = np.where(neural_data['trial_start'] == 1)[0]
            all_ends = np.where(neural_data['trial_end'] == 1)[0]
            all_ends = all_ends[all_ends > 0]
            if len(all_ends) < len(all_starts):
                all_ends = np.append(all_ends, len(neural_data['time']) - 1)

            if conditions is not None:
                selected_trial_indices, _ = select_trials_from_data(neural_data, conditions)
                sel_starts = all_starts[selected_trial_indices]
                sel_ends = all_ends[selected_trial_indices]
            else:
                sel_starts, sel_ends = all_starts, all_ends

            all_event_idx = np.where(neural_data[event] == 1)[0]
            idx = []
            for s, e in zip(sel_starts, sel_ends):
                trial_events = all_event_idx[(all_event_idx >= s) & (all_event_idx <= e)]
                idx.extend(trial_events)
            idx = np.array(idx)
            
            if len(idx) == 0: continue
            total_trials_for_event += len(idx)
            
            # --- Position, Velocity, & Lick Processing ---
            adj_js, t_js = create_matrix(idx, neural_data['js_pos'], neural_data['js_time'], 
                                         neural_data['time'], pre_list[event_num], post_list[event_num])
            vel_matrix = np.array([np.gradient(trial, t_js) for trial in adj_js])
            adj_lick, t_lick = create_matrix(idx, neural_data['lick'], neural_data['time'], 
                                             neural_data['time'], pre_list[event_num], post_list[event_num])

            # --- NEW: Pupil Processing ---
            # Slice flir_time to match camera_pupil length to handle potential extra value at end
            p_data = neural_data['camera_pupil']
            p_time = neural_data['flir_time'][:len(p_data)]
            adj_pupil, t_pupil = create_matrix(idx, p_data, p_time, 
                                               neural_data['time'], pre_list[event_num], post_list[event_num])

            if grant:
                sess_js.append(np.interp(t_grand, t_js, np.nanmean(adj_js, axis=0), left=np.nan, right=np.nan))
                sess_vel.append(np.interp(t_grand, t_js, np.nanmean(vel_matrix, axis=0), left=np.nan, right=np.nan))
                sess_lick.append(np.interp(t_grand, t_lick, np.nanmean(adj_lick, axis=0), left=np.nan, right=np.nan))
                sess_pupil.append(np.interp(t_grand, t_pupil, np.nanmean(adj_pupil, axis=0), left=np.nan, right=np.nan))
            else:
                color = session_colormap(colors[-1], len(list_neural_data)+3)[session]
                trace_plot(ax_js, adj_js, t_js, color, dim=1)
                trace_plot(ax_vel, vel_matrix, t_js, color, dim=1)
                trace_plot(ax_lick, adj_lick, t_lick, color, dim=1)
                trace_plot(ax_pupil, adj_pupil, t_pupil, color, dim=1) # Plot pupil individual traces

            # ... (Neural Logic remains the same) ...
            current_labels = list_labels[session]
            if plot_mode == 'pool': categories = [-1]
            elif plot_mode == 'specific': categories = [target_cluster] if target_cluster in current_labels else []
            else: categories = np.unique(current_labels)

            for cat in categories:
                if cat == -1:
                    neurons = np.arange(neural_data['dff'].shape[0])
                    c_idx = -1 
                else:
                    neurons = np.squeeze(np.where(current_labels == cat))
                    c_idx = cat
                if neurons.size == 0: continue
                adj_d, t_d = create_matrix(idx, neural_data['dff'][neurons, :], neural_data['time'], 
                                           neural_data['time'], pre_list[event_num], post_list[event_num])
                adj_s, t_s = create_matrix(idx, neural_data['spikes'][neurons, :], neural_data['time'], 
                                           neural_data['time'], pre_list[event_num], post_list[event_num])
                coact_matrix = np.mean(adj_s > 0, axis=1) 
                if grant:
                    n_avg_d = np.nanmean(adj_d, axis=0); n_avg_s = np.nanmean(adj_s, axis=0)
                    sess_dff[c_idx].append(np.array([np.interp(t_grand, t_d, n, left=np.nan, right=np.nan) for n in n_avg_d]))
                    sess_spk[c_idx].append(np.array([np.interp(t_grand, t_s, n, left=np.nan, right=np.nan) for n in n_avg_s]))
                    sess_coact[c_idx].append(np.interp(t_grand, t_s, np.nanmean(coact_matrix, axis=0), left=np.nan, right=np.nan))
                else:
                    cat_color = session_colormap(colors[c_idx], len(list_neural_data)+3)[session]
                    trace_plot(ax_dff, adj_d, t_d, cat_color, dim=2)
                    trace_plot(ax_spk, adj_s, t_s, cat_color, dim=2)
                    trace_plot(ax_coact, coact_matrix, t_s, cat_color, dim=1)

        # --- Plot Grand Averages ---
        if grant:
            c_main = colors[-1]/255.0 
            if sess_js: trace_plot(ax_js, np.array(sess_js), t_grand, c_main, dim=1)
            if sess_vel: trace_plot(ax_vel, np.array(sess_vel), t_grand, c_main, dim=1)
            if sess_lick: trace_plot(ax_lick, np.array(sess_lick), t_grand, c_main, dim=1)
            if sess_pupil: trace_plot(ax_pupil, np.array(sess_pupil), t_grand, c_main, dim=1) # Grand avg pupil
            for cat in sess_dff:
                if sess_dff[cat]:
                    c_cat = colors[cat]/255.0
                    trace_plot(ax_dff, np.vstack(sess_dff[cat]), t_grand, c_cat, dim=1)
                    trace_plot(ax_spk, np.vstack(sess_spk[cat]), t_grand, c_cat, dim=1)
                    trace_plot(ax_coact, np.array(sess_coact[cat]), t_grand, c_cat, dim=1)

        # --- Formatting and Layout ---
        y_labels = ['joystick pos', 'velocity', 'lick rate', 'pupil area', 'df/f', 'spikes', 'fraction active']
        plot_axes = [ax_js, ax_vel, ax_lick, ax_pupil, ax_dff, ax_spk, ax_coact]
        
        for ax, ylab in zip(plot_axes, y_labels):
            current_title = f"{event}\n(n={total_trials_for_event} trials)" if ylab == 'joystick pos' else ''
            lay_out_plot(ax, x_label='Time (s)', y_label=ylab, title=current_title, 
                         x_lim=[-pre_list[event_num]/fs, post_list[event_num]/fs])
                
    save_temp_fig(fig, report)
    # Determine save path and close report (omitted here for brevity, same as original code)
    # Determine save path
    if existing_pdf_path:
        save_path = existing_pdf_path
    else:
        fname = 'grand_averages.pdf' if grant else 'initial_alignments.pdf'
        save_path = os.path.join(output_dir_onedrive, fname)
    
    # Save incrementally (use incremental=True if appending to an existing file)
    if existing_pdf_path and os.path.exists(existing_pdf_path):
        report.save(report.name, incremental=True, encryption=0) #
    else:
        report.save(save_path) #
        
    report.close()

def create_initial_pdf(output_path, filename="all_sessions_plots.pdf"):
    """
    Creates a PDF file with an initial title page to act as a container.
    """
    import fitz
    import os
    
    # Create a new document
    doc = fitz.open()
    
    # Add a blank page (PyMuPDF requirement for saving)
    page = doc.new_page()
    
    # Optional: Add a title to the first page so it's not just empty
    text = f"Analysis Report: {filename}"
    page.insert_text((50, 50), text, fontsize=20)
    
    # Ensure the output directory exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    full_path = os.path.join(output_path, filename)
    
    # Save the document
    doc.save(full_path)
    doc.close()
    
    print(f"Initial PDF created with title page at: {full_path}")
    return full_path

def make_data_final1(output_dir_onedrive, list_neural_data, event_list, pre_list, post_list, list_labels, 
                    grant = 0, conditions = None, plot_mode = 'separate', target_cluster = None,
                    existing_pdf_path = None):
    
    # Fix: Use dark gray for pooled traces to allow dimming
    plot_colors = colors.copy()
    plot_colors[-1] = [100, 100, 100] 

    if existing_pdf_path and os.path.exists(existing_pdf_path):
        report = fitz.open(existing_pdf_path)
    else:
        report = fitz.open()

    # Updated to 7 rows to accommodate the Heatmap
    fig = plt.figure(layout='constrained', figsize=(30, 21))
    gs = GridSpec(7 , len(event_list)+2, figure=fig)
    fs = 30.0 
    
    for event_num, event in enumerate(event_list):
        ax_js = plt.subplot(gs[0, event_num]); ax_vel = plt.subplot(gs[1, event_num]) 
        ax_lick = plt.subplot(gs[2, event_num]); ax_dff = plt.subplot(gs[3, event_num])
        ax_spk = plt.subplot(gs[4, event_num]); ax_coact = plt.subplot(gs[5, event_num])
        ax_hm = plt.subplot(gs[6, event_num]) # New Heatmap Row

        t_grand = (np.arange(pre_list[event_num] + post_list[event_num]) - pre_list[event_num]) * (1000/fs)

        sess_js, sess_vel, sess_lick = [], [], []
        sess_dff = {cat: [] for cat in range(-1, len(plot_colors))}
        sess_spk = {cat: [] for cat in range(-1, len(plot_colors))}
        sess_coact = {cat: [] for cat in range(-1, len(plot_colors))}
        
        # Container for heatmap aggregation
        sess_hm_data = [] 

        for session, neural_data in enumerate(list_neural_data):
            # 1. Trial Selection
            all_starts = np.where(neural_data['trial_start'] == 1)[0]
            all_ends = np.where(neural_data['trial_end'] == 1)[0]
            all_ends = all_ends[all_ends > 0]
            if len(all_ends) < len(all_starts):
                all_ends = np.append(all_ends, len(neural_data['time']) - 1)

            if conditions is not None:
                selected_trial_indices, _ = select_trials_from_data(neural_data, conditions)
                sel_starts = all_starts[selected_trial_indices]
                sel_ends = all_ends[selected_trial_indices]
            else:
                sel_starts, sel_ends = all_starts, all_ends

            all_event_idx = np.where(neural_data[event] == 1)[0]
            idx = [ev for s, e in zip(sel_starts, sel_ends) for ev in all_event_idx if s <= ev <= e]
            if not idx: continue
            idx = np.array(idx)
            
            # --- 1. Behavior ---
            adj_js, t_js = create_matrix(idx, neural_data['js_pos'], neural_data['js_time'], 
                                         neural_data['time'], pre_list[event_num], post_list[event_num])
            vel_matrix = np.array([np.gradient(trial, t_js) for trial in adj_js])
            adj_lick, t_lick = create_matrix(idx, neural_data['lick'], neural_data['time'], 
                                             neural_data['time'], pre_list[event_num], post_list[event_num])

            if grant:
                sess_js.append(np.interp(t_grand, t_js, np.nanmean(adj_js, axis=0), left=np.nan, right=np.nan))
                sess_vel.append(np.interp(t_grand, t_js, np.nanmean(vel_matrix, axis=0), left=np.nan, right=np.nan))
                sess_lick.append(np.interp(t_grand, t_lick, np.nanmean(adj_lick, axis=0), left=np.nan, right=np.nan))
            else:
                # Point 1: Dimming logic for session traces
                color = session_colormap(plot_colors[-1], len(list_neural_data)+3)[session]
                trace_plot(ax_js, adj_js, t_js, color, dim=1)
                trace_plot(ax_vel, vel_matrix, t_js, color, dim=1)
                trace_plot(ax_lick, adj_lick, t_lick, color, dim=1)

            # --- 2. Neural Logic ---
            current_labels = list_labels[session]
            if plot_mode == 'pool':
                categories = [-1]
            elif plot_mode == 'specific':
                categories = [target_cluster] if target_cluster in current_labels else []
            else:
                categories = np.sort(np.unique(current_labels))

            for cat in categories:
                if cat == -1:
                    neurons = np.arange(neural_data['dff'].shape[0])
                    c_idx = -1 
                else:
                    neurons = np.squeeze(np.where(current_labels == cat))
                    c_idx = cat

                if neurons.size == 0: continue
                
                adj_d, t_d = create_matrix(idx, neural_data['dff'][neurons, :], neural_data['time'], 
                                           neural_data['time'], pre_list[event_num], post_list[event_num])
                adj_s, t_s = create_matrix(idx, neural_data['spikes'][neurons, :], neural_data['time'], 
                                           neural_data['time'], pre_list[event_num], post_list[event_num])
                
                coact_matrix = np.mean(adj_s > 0, axis=1) #
                
                if grant:
                    n_avg_d = np.nanmean(adj_d, axis=0) # Shape: (neurons, time)
                    res_d = np.array([np.interp(t_grand, t_d, n, left=np.nan, right=np.nan) for n in n_avg_d])
                    sess_dff[c_idx].append(res_d)
                    
                    res_s = np.array([np.interp(t_grand, t_s, n, left=np.nan, right=np.nan) for n in np.nanmean(adj_s, axis=0)])
                    sess_spk[c_idx].append(res_s)
                    
                    res_c = np.interp(t_grand, t_s, np.nanmean(coact_matrix, axis=0), left=np.nan, right=np.nan)
                    sess_coact[c_idx].append(res_c)
                else:
                    cat_color = session_colormap(plot_colors[c_idx], len(list_neural_data)+3)[session]
                    trace_plot(ax_dff, adj_d, t_d, cat_color, dim=2)
                    trace_plot(ax_spk, adj_s, t_s, cat_color, dim=2)
                    trace_plot(ax_coact, coact_matrix, t_s, cat_color, dim=1)

        # --- 3. Heatmap Row (Cluster-grouped, not internally sorted) ---
        if grant:
            all_hm_neurons = []
            divider_positions = []
            for cat in sorted(sess_dff.keys()):
                if sess_dff[cat]:
                    # Stack all neurons for this cluster
                    cluster_neurons = np.vstack(sess_dff[cat])
                    all_hm_neurons.append(cluster_neurons)
                    # Track where to draw lines
                    if not divider_positions: divider_positions.append(len(cluster_neurons))
                    else: divider_positions.append(divider_positions[-1] + len(cluster_neurons))
            
            if all_hm_neurons:
                hm_matrix = np.vstack(all_hm_neurons)
                # Plot heatmap with 0 at the vertical line
                im = ax_hm.imshow(hm_matrix, aspect='auto', cmap='viridis', 
                                  extent=[t_grand[0]/1000, t_grand[-1]/1000, 0, len(hm_matrix)])
                # Add white dividers between clusters
                for pos in divider_positions[:-1]:
                    ax_hm.axhline(len(hm_matrix) - pos, color='white', lw=1)

        # --- 4. Grand Average Traces ---
        if grant:
            c_main = plot_colors[-1]/255.0 
            if sess_js: trace_plot(ax_js, np.array(sess_js), t_grand, c_main, dim=1)
            if sess_vel: trace_plot(ax_vel, np.array(sess_vel), t_grand, c_main, dim=1)
            if sess_lick: trace_plot(ax_lick, np.array(sess_lick), t_grand, c_main, dim=1)
            
            for cat in sess_dff:
                if sess_dff[cat]:
                    c_cat = plot_colors[cat]/255.0
                    trace_plot(ax_dff, np.vstack(sess_dff[cat]), t_grand, c_cat, dim=1)
                    trace_plot(ax_spk, np.vstack(sess_spk[cat]), t_grand, c_cat, dim=1)
                    trace_plot(ax_coact, np.array(sess_coact[cat]), t_grand, c_cat, dim=1)

        # Formatting
        y_labels = ['joystick pos', 'velocity', 'lick rate', 'df/f', 'spikes', 'fraction active', 'heatmap']
        for ax, ylab in zip([ax_js, ax_vel, ax_lick, ax_dff, ax_spk, ax_coact, ax_hm], y_labels):
            lay_out_plot(ax, x_label='Time (s)', y_label=ylab, title=event if ylab == 'joystick pos' else '', 
                         x_lim=[-pre_list[event_num]/fs, post_list[event_num]/fs])
                
    save_temp_fig(fig, report)
    
    # Point 2: Incremental Saving
    if existing_pdf_path and os.path.exists(existing_pdf_path):
        report.save(report.name, incremental=True, encryption=0)
    else:
        fname = 'grand_averages.pdf' if grant else 'initial_alignments.pdf'
        report.save(os.path.join(output_dir_onedrive, fname))
        
    report.close()