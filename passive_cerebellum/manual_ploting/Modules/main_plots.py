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
            [255, 105,   0],  # orange
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
            ops['save_path0'] = os.path.join(file_temp)
            cluster_labels = ReadResults.read_cluster_labels(ops)
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
                    #print(c)
                    func_img[roi_mask] = colors[int(c)]
        
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
                #print(reference_time.shape, max(0, i-pre))
                start_time = reference_time[max(0, i-pre)]
                #print(min(len(reference_time), i+post), len(reference_time))
                end_time = reference_time[min(len(reference_time)-1, i+post)]
                start_idx = np.argmin(np.abs(time_array-start_time))
                end_idx = np.argmin(np.abs(time_array-end_time))
                zero_idx = np.argmin(np.abs(time_array-reference_time[i]))
                # output_time.append(time_array[start_idx:end_idx]-time_array[zero_idx])
                # output_array.append(input_array[start_idx:end_idx])
                #print(start_time, end_time, start_idx, end_idx, zero_idx)
        
                interpolator = interp1d(time_array[start_idx:end_idx], input_array[start_idx:end_idx], bounds_error=False)
                new_time = np.arange(start_time, end_time, interval)
                new_pos = interpolator(new_time)
                output_time.append(new_time-time_array[zero_idx])
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
    """
    Selects stimulus events based on columns in stim_labels.
    
    Mapping based on your get_stim_labels:
    2: img_seq_label, 3: standard_types, 4: fix_jitter_types, 
    5: oddball_types, 6: random_types, 7: opto_types, 8: sequence_number
    """
    stim_labels = neural_data['stim_labels']
    n_stims = stim_labels.shape[0]
    
    # Mapping dictionary to translate human-readable keys to column indices
    col_map = {
        'img_seq': 2, 'standard': 3, 'jitter': 4,
        'oddball': 5, 'random': 6, 'opto': 7, 'seq_num': 8, 'pre_isi': 9,   # The preceding ISI we just added
    'next_isi': 10  # The next ISI we just added
    }
    
    mask = np.ones(n_stims, dtype=bool) if combine == 'and' else np.zeros(n_stims, dtype=bool)

    for key, cond in conditions.items():
        if key not in col_map:
            continue
            
        # Extract the column values for this condition
        col_idx = col_map[key]
        stim_values = stim_labels[:, col_idx]
        
        # Build the mask
        if isinstance(cond, tuple):
            op, val = cond[0], cond[1]
            if op == '==': m = (stim_values == val)
            elif op == 'in': m = np.isin(stim_values, np.array(list(val)))
            elif op == '!=': m = (stim_values != val)
        else:
            m = (stim_values == cond)
        
        mask = (mask & m) if combine == 'and' else (mask | m)

    selected_stim_indices = np.where(mask)[0]
    # Return the timestamps from column 0 associated with filtered stims
    start_times = stim_labels[selected_stim_indices, 0]
    
    return selected_stim_indices, start_times
        
def make_data_final(output_dir_onedrive, list_neural_data, event_list, pre_list, post_list, list_labels, 
                    grant=0, conditions=None, plot_mode='separate', target_cluster=None,
                    existing_pdf_path=None):
    """
    Final version: Supports alignment to behavioral events OR stimulus-specific 
    events from stim_labels using a unified conditions dictionary.
    """
    if existing_pdf_path and os.path.exists(existing_pdf_path):
        report = fitz.open(existing_pdf_path)
    else:
        report = fitz.open()

    fig = plt.figure(layout='constrained', figsize=(30, 18))
    gs = GridSpec(6, len(event_list) + 2, figure=fig)
    fs = 30.0 

    # Mapping for stim_labels columns
    col_map = {
        'img_seq': 2, 'standard': 3, 'jitter': 4,
        'oddball': 5, 'random': 6, 'opto': 7, 'seq_num': 8,
        'pre_isi': 9,   # The preceding ISI we just added
        'next_isi': 10  # The next ISI we just added
    }

    for event_num, event in enumerate(event_list):
        ax_js = plt.subplot(gs[0, event_num]); ax_vel = plt.subplot(gs[1, event_num]) 
        ax_lick = plt.subplot(gs[2, event_num]); ax_dff = plt.subplot(gs[3, event_num])
        ax_spk = plt.subplot(gs[4, event_num]); ax_coact = plt.subplot(gs[5, event_num]) 
        
        t_grand = (np.arange(pre_list[event_num] + post_list[event_num]) - pre_list[event_num]) * (1000/fs)

        sess_js, sess_vel, sess_lick = [], [], []
        sess_dff = {cat: [] for cat in range(-1, 10)} # Adjust range if you have >10 clusters
        sess_spk = {cat: [] for cat in range(-1, 10)}
        sess_coact = {cat: [] for cat in range(-1, 10)}
        
        print(f'Analyzing: {event}')

        for session, neural_data in enumerate(list_neural_data):
            # --- 1. Event Timing & Condition Filtering ---
            if event == 'stim':
                # STIMULUS ALIGNMENT LOGIC
                stim_labels = neural_data['stim_labels']
                n_stims = stim_labels.shape[0]
                mask = np.ones(n_stims, dtype=bool)

                if conditions is not None:
                    for key, cond in conditions.items():
                        if key in col_map:
                            val_arr = stim_labels[:, col_map[key]]
                            if isinstance(cond, tuple):
                                op, val = cond[0], cond[1]
                                if op == '==': m = (val_arr == val)
                                elif op == 'in': m = np.isin(val_arr, np.array(list(val)))
                                elif op == '!=': m = (val_arr != val)
                            else:
                                m = (val_arr == cond)
                            mask &= m
                
                # Get timestamps from column 0 and convert to indices
                stim_times = stim_labels[mask, 0]
                idx = np.searchsorted(neural_data['time'], stim_times)
            
            else:
                # BEHAVIORAL TRIAL ALIGNMENT LOGIC
                all_starts = np.where(neural_data['trial_start'] == 1)[0]
                all_ends = np.where(neural_data['trial_end'] == 1)[0]
                all_ends = all_ends[all_ends > 0]
                if len(all_ends) < len(all_starts):
                    all_ends = np.append(all_ends, len(neural_data['time']) - 1)

                if conditions is not None:
                    # Uses your original select_trials_from_data function
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
            
            # --- 2. Create Matrices ---
            # adj_js, t_js = create_matrix(idx, neural_data['js_pos'], neural_data['js_time'], 
            #                              neural_data['time'], pre_list[event_num], post_list[event_num])
            # vel_matrix = np.array([np.gradient(trial, t_js) for trial in adj_js])
            # adj_lick, t_lick = create_matrix(idx, neural_data['lick'], neural_data['time'], 
            #                                  neural_data['time'], pre_list[event_num], post_list[event_num])

            # if grant:
            #     sess_js.append(np.interp(t_grand, t_js, np.nanmean(adj_js, axis=0), left=np.nan, right=np.nan))
            #     sess_vel.append(np.interp(t_grand, t_js, np.nanmean(vel_matrix, axis=0), left=np.nan, right=np.nan))
            #     sess_lick.append(np.interp(t_grand, t_lick, np.nanmean(adj_lick, axis=0), left=np.nan, right=np.nan))
            # else:
            #     color = session_colormap(colors[-1], len(list_neural_data)+3)[session]
            #     trace_plot(ax_js, adj_js, t_js, color, dim=1)
            #     trace_plot(ax_vel, vel_matrix, t_js, color, dim=1)
            #     trace_plot(ax_lick, adj_lick, t_lick, color, dim=1)

            # --- 3. Neural Logic ---
            current_labels = list_labels[session]
            if plot_mode == 'pool':
                categories = [-1]
            elif plot_mode == 'specific':
                categories = [target_cluster] if target_cluster in current_labels else []
            else:
                categories = np.unique(current_labels)

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
                    cat_color = session_colormap(colors[int(c_idx)], len(list_neural_data)+3)[session]
                    trace_plot(ax_dff, adj_d, t_d, cat_color, dim=2)
                    trace_plot(ax_spk, adj_s, t_s, cat_color, dim=2)
                    trace_plot(ax_coact, coact_matrix, t_s, cat_color, dim=1)

        # --- 4. Plot Grand Average & Formatting ---
        if grant:
            c_main = colors[-1]/255.0 
            if sess_js: trace_plot(ax_js, np.array(sess_js), t_grand, c_main, dim=1)
            if sess_vel: trace_plot(ax_vel, np.array(sess_vel), t_grand, c_main, dim=1)
            if sess_lick: trace_plot(ax_lick, np.array(sess_lick), t_grand, c_main, dim=1)
            
            for cat in sess_dff:
                if sess_dff[cat]:
                    c_cat = colors[cat]/255.0
                    trace_plot(ax_dff, np.vstack(sess_dff[cat]), t_grand, c_cat, dim=1)
                    trace_plot(ax_spk, np.vstack(sess_spk[cat]), t_grand, c_cat, dim=1)
                    trace_plot(ax_coact, np.array(sess_coact[cat]), t_grand, c_cat, dim=1)

        y_labels = ['joystick pos', 'velocity', 'lick rate', 'df/f', 'spikes', 'fraction active']
        for ax, ylab in zip([ax_js, ax_vel, ax_lick, ax_dff, ax_spk, ax_coact], y_labels):
            lay_out_plot(ax, x_label='Time (s)', y_label=ylab, title=event if ylab == 'joystick pos' else '', 
                         x_lim=[-pre_list[event_num]/fs, post_list[event_num]/fs])
                
    save_temp_fig(fig, report)
    
    save_path = existing_pdf_path if existing_pdf_path else os.path.join(output_dir_onedrive, 'initial_alignments.pdf')
    report.save(save_path, incremental=bool(existing_pdf_path))
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

def make_data_final_new(output_dir_onedrive, list_neural_data, pre_list, post_list, list_labels, 

                    event_list=None, column_conditions=None,

                    grant=0, plot_mode='separate', target_cluster=None,

                    existing_pdf_path=None):

    """

    Modified version:

    - Removes behavioral plots (js, vel, lick).

    - If column_conditions is provided, each plot column represents a condition 

      aligned to 'stim' rather than a different event type.

    """

    if existing_pdf_path and os.path.exists(existing_pdf_path):

        report = fitz.open(existing_pdf_path)

    else:

        report = fitz.open()



    # Determine number of columns

    num_cols = len(column_conditions) if column_conditions else len(event_list)

    

    # We only need 3 rows now: df/f, spikes, and coactivity

    fig = plt.figure(layout='constrained', figsize=(6 * num_cols, 15))

    gs = GridSpec(3, num_cols, figure=fig)

    fs = 30.0 



    col_map = {

        'img_seq': 2, 'standard': 3, 'jitter': 4,

        'oddball': 5, 'random': 6, 'opto': 7, 'seq_num': 8,
    'pre_isi': 9,  # New Column 1
    'next_isi': 10 # New Column 2

    }



    for col_idx in range(num_cols):

        # UI Setup: Only Neural Plots

        ax_dff = plt.subplot(gs[0, col_idx])

        ax_spk = plt.subplot(gs[1, col_idx])

        ax_coact = plt.subplot(gs[2, col_idx])

        

        # Determine the event and condition for this column

        current_event = event_list[col_idx] if event_list else 'stim'

        current_cond = column_conditions[col_idx] if column_conditions else None

        

        # Timing setup (uses first item in pre/post lists if they aren't long enough)

        pre = pre_list[col_idx] if len(pre_list) > col_idx else pre_list[0]

        post = post_list[col_idx] if len(post_list) > col_idx else post_list[0]

        t_grand = (np.arange(pre + post) - pre) * (1000/fs)



        sess_dff = {cat: [] for cat in range(-1, 20)}

        sess_spk = {cat: [] for cat in range(-1, 20)}

        sess_coact = {cat: [] for cat in range(-1, 20)}

        

        print(f'Column {col_idx}: Aligning to {current_event} with cond {current_cond}')



        for session, neural_data in enumerate(list_neural_data):

            # --- 1. Timing Logic ---

            if current_event == 'stim':

                stim_labels = neural_data['stim_labels']

                mask = np.ones(stim_labels.shape[0], dtype=bool)

                if current_cond:
                    for key, cond in current_cond.items():
                        if key in col_map:
                            val_arr = stim_labels[:, col_map[key]]
                            
                            # 1. Handle string comparisons like '>500' or '<1000'
                            if isinstance(cond, str) and any(op in cond for op in ['>', '<', '=']):
                                # We use np.vectorize or a simple comparison if it's a single op
                                # Example: for '>500', we eval 'val_arr > 500'
                                try:
                                    mask &= eval(f"val_arr {cond}")
                                except Exception as e:
                                    print(f"Error evaluating condition {cond} on column {key}: {e}")
                            
                            # 2. Handle lists or specific values
                            else:
                                m = np.isin(val_arr, cond) if isinstance(cond, (list, tuple)) else (val_arr == cond)
                                mask &= m

                idx = np.searchsorted(neural_data['time'], stim_labels[mask, 0])

            else:

                # Fallback to standard trial-based event logic

                all_event_idx = np.where(neural_data[current_event] == 1)[0]

                idx = all_event_idx # Simplified for brevity; add trial filtering if needed



            if len(idx) == 0: continue

            

            # --- 2. Neural Processing ---

            current_labels = list_labels[session]

            categories = [-1] if plot_mode == 'pool' else ([target_cluster] if plot_mode == 'specific' else np.unique(current_labels))



            for cat in categories:

                neurons = np.arange(neural_data['dff'].shape[0]) if cat == -1 else np.squeeze(np.where(current_labels == cat))

                if neurons.size == 0: continue

                

                adj_d, t_d = create_matrix(idx, neural_data['dff'][neurons, :], neural_data['time'], neural_data['time'], pre, post)

                adj_s, t_s = create_matrix(idx, neural_data['spikes'][neurons, :], neural_data['time'], neural_data['time'], pre, post)

                coact_matrix = np.mean(adj_s > 0, axis=1) 

                

                if grant:

                    sess_dff[cat].append(np.array([np.interp(t_grand, t_d, n, left=np.nan, right=np.nan) for n in np.nanmean(adj_d, axis=0)]))

                    sess_spk[cat].append(np.array([np.interp(t_grand, t_s, n, left=np.nan, right=np.nan) for n in np.nanmean(adj_s, axis=0)]))

                    sess_coact[cat].append(np.interp(t_grand, t_s, np.nanmean(coact_matrix, axis=0), left=np.nan, right=np.nan))

                else:

                    cat_color = session_colormap(colors[int(cat)], len(list_neural_data)+3)[session]

                    trace_plot(ax_dff, adj_d, t_d, cat_color, dim=2)

                    trace_plot(ax_spk, adj_s, t_s, cat_color, dim=2)

                    trace_plot(ax_coact, coact_matrix, t_s, cat_color, dim=1)



        # --- 3. Grand Average Plotting ---

        if grant:

            for cat in sess_dff:

                if sess_dff[cat]:

                    c_cat = colors[int(cat)]/255.0

                    trace_plot(ax_dff, np.vstack(sess_dff[cat]), t_grand, c_cat, dim=1)

                    trace_plot(ax_spk, np.vstack(sess_spk[cat]), t_grand, c_cat, dim=1)

                    trace_plot(ax_coact, np.array(sess_coact[cat]), t_grand, c_cat, dim=1)



        # Formatting

        title_str = f"Cond: {current_cond}" if current_cond else current_event

        lay_out_plot(ax_dff, y_label='df/f', title=title_str, x_lim=[-pre/fs, post/fs])

        lay_out_plot(ax_spk, y_label='spikes', x_lim=[-pre/fs, post/fs])

        lay_out_plot(ax_coact, x_label='Time (s)', y_label='active', x_lim=[-pre/fs, post/fs])

                

    save_temp_fig(fig, report)



    # Determine the final save path

    save_path = existing_pdf_path if existing_pdf_path else os.path.join(output_dir_onedrive, 'condition_analysis.pdf')

    

    try:

        # If the file exists, we try to save incrementally. 

        # If it fails due to encryption/structure changes, we fall back to a full save.

        if existing_pdf_path and os.path.exists(existing_pdf_path):

            report.save(report.name, incremental=True, encryption=0)

        else:

            report.save(save_path)

    except Exception as e:

        # Fallback: Save as a regular file if incremental fails

        print(f"Incremental save failed, performing full save: {e}")

        report.save(save_path)

    

    report.close()

    

def make_data_comparison(output_dir_onedrive, list_neural_data, pre_list, post_list, list_labels, 

                         event_list=None, column_conditions1=None, column_conditions2=None,

                         grant=0, plot_mode='pool', target_cluster=None,

                         existing_pdf_path=None):

    """

    Plots two sets of conditions superimposed with SEM shaded boxes.

    - column_conditions1: Solid lines

    - column_conditions2: Dashed lines

    """

    if existing_pdf_path and os.path.exists(existing_pdf_path):

        report = fitz.open(existing_pdf_path)

    else:

        report = fitz.open()



    num_cols = len(column_conditions1) if column_conditions1 else len(event_list)

    fig = plt.figure(layout='constrained', figsize=(6 * num_cols, 15))

    gs = GridSpec(3, num_cols, figure=fig)

    fs = 30.0 



    col_map = {

        'img_seq': 2, 'standard': 3, 'jitter': 4, 'oddball': 5, 

        'random': 6, 'opto': 7, 'seq_num': 8, 
    'pre_isi': 9,  # New Column 1
    'next_isi': 10 # New Column 2

    }



    for col_idx in range(num_cols):

        ax_dff = plt.subplot(gs[0, col_idx])

        ax_spk = plt.subplot(gs[1, col_idx])

        ax_coact = plt.subplot(gs[2, col_idx])

        

        pre = pre_list[col_idx] if len(pre_list) > col_idx else pre_list[0]

        post = post_list[col_idx] if len(post_list) > col_idx else post_list[0]

        t_grand = (np.arange(pre + post) - pre) * (1000/fs)



        for run_idx, current_cond in enumerate([column_conditions1[col_idx], column_conditions2[col_idx]]):

            linestyle = '-' if run_idx == 0 else '--'

            alpha_line = 1.0 if run_idx == 0 else 0.8

            alpha_shade = 0.2 if run_idx == 0 else 0.1 # Lighter shade for dashed comparison

            

            sess_dff = {cat: [] for cat in range(-1, 20)}

            sess_spk = {cat: [] for cat in range(-1, 20)}

            sess_coact = {cat: [] for cat in range(-1, 20)}



            for session, neural_data in enumerate(list_neural_data):

                stim_labels = neural_data['stim_labels']

                mask = np.ones(stim_labels.shape[0], dtype=bool)

                if current_cond:
                    for key, cond in current_cond.items():
                        if key in col_map:
                            val_arr = stim_labels[:, col_map[key]]
                            
                            # 1. Handle string comparisons like '>500' or '<1000'
                            if isinstance(cond, str) and any(op in cond for op in ['>', '<', '=']):
                                # We use np.vectorize or a simple comparison if it's a single op
                                # Example: for '>500', we eval 'val_arr > 500'
                                try:
                                    mask &= eval(f"val_arr {cond}")
                                except Exception as e:
                                    print(f"Error evaluating condition {cond} on column {key}: {e}")
                            
                            # 2. Handle lists or specific values
                            else:
                                m = np.isin(val_arr, cond) if isinstance(cond, (list, tuple)) else (val_arr == cond)
                                mask &= m

                

                idx = np.searchsorted(neural_data['time'], stim_labels[mask, 0])

                if len(idx) == 0: continue

                

                current_labels = list_labels[session]

                categories = [-1] if plot_mode == 'pool' else ([target_cluster] if plot_mode == 'specific' else np.unique(current_labels))



                for cat in categories:

                    neurons = np.arange(neural_data['dff'].shape[0]) if cat == -1 else np.squeeze(np.where(current_labels == cat))

                    if neurons.size == 0: continue

                    

                    adj_d, t_d = create_matrix(idx, neural_data['dff'][neurons, :], neural_data['time'], neural_data['time'], pre, post)

                    adj_s, t_s = create_matrix(idx, neural_data['spikes'][neurons, :], neural_data['time'], neural_data['time'], pre, post)

                    coact_matrix = np.mean(adj_s > 0, axis=1) 

                    

                    if grant:

                        # Interpolate to common grand time axis for averaging across sessions

                        sess_dff[cat].append(np.array([np.interp(t_grand, t_d, n, left=np.nan, right=np.nan) for n in np.nanmean(adj_d, axis=0)]))

                        sess_spk[cat].append(np.array([np.interp(t_grand, t_s, n, left=np.nan, right=np.nan) for n in np.nanmean(adj_s, axis=0)]))

                        sess_coact[cat].append(np.interp(t_grand, t_s, np.nanmean(coact_matrix, axis=0), left=np.nan, right=np.nan))

                    else:

                        # Single session plotting with SEM shading

                        cat_color = session_colormap(colors[int(cat)], len(list_neural_data)+3)[session]

                        # Flatten across neurons and trials for simple session mean

                        y_vals = np.nanmean(adj_d, axis=(0,1))

                        y_sem = np.nanstd(adj_d, axis=(0,1)) / np.sqrt(np.sum(~np.isnan(adj_d), axis=(0,1)))

                        

                        ax_dff.plot(t_d/1000, y_vals, color=cat_color, linestyle=linestyle, alpha=alpha_line, linewidth=0.5)

                        ax_dff.fill_between(t_d/1000, y_vals-y_sem, y_vals+y_sem, color=cat_color, alpha=alpha_shade, edgecolor='none')



            if grant:

                for cat in sess_dff:

                    if sess_dff[cat]:

                        c_cat = colors[int(cat)]/255.0

                        

                        # Helper to plot mean + SEM for grand averages

                        for ax, data_list in zip([ax_dff, ax_spk, ax_coact], [sess_dff[cat], sess_spk[cat], sess_coact[cat]]):

                            data_stack = np.vstack(data_list) if ax != ax_coact else np.array(data_list)

                            y_mean = np.nanmean(data_stack, axis=0)

                            y_sem = np.nanstd(data_stack, axis=0) / np.sqrt(np.sum(~np.isnan(data_stack), axis=0))

                            

                            ax.plot(t_grand/1000, y_mean, color=c_cat, linestyle=linestyle, linewidth=1.5)

                            ax.fill_between(t_grand/1000, y_mean-y_sem, y_mean+y_sem, color=c_cat, alpha=alpha_shade, edgecolor='none')

                            ax.axvline(0, color='gray', linestyle=':', linewidth=0.5)



        lay_out_plot(ax_dff, y_label='df/f', title=f"Comparison Col {col_idx}", x_lim=[-pre/fs, post/fs])

        lay_out_plot(ax_spk, y_label='spikes', x_lim=[-pre/fs, post/fs])

        lay_out_plot(ax_coact, x_label='Time (s)', y_label='active', x_lim=[-pre/fs, post/fs])

                

    save_temp_fig(fig, report)

    

    save_path = existing_pdf_path if existing_pdf_path else os.path.join(output_dir_onedrive, 'comparison_sem.pdf')

    try:

        if existing_pdf_path and os.path.exists(existing_pdf_path):

            report.save(report.name, incremental=True, encryption=0)

        else:

            report.save(save_path)

    except:

        report.save(save_path.replace('.pdf', '_new.pdf'))

    report.close()