# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 09:52:45 2025

@author: saminnaji3
"""

import numpy as np
import os

import h5py
from scipy.signal import savgol_filter
from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
#%matplotlib qt

# %%

def smooth_dff(dff, window_length=7, polyorder=3):
    """
    Apply Savitzky-Golay filter to smooth DF/F data.

    Parameters:
    dff (np.ndarray): The DF/F data to be smoothed.
    window_length (int): The length of the filter window (must be odd).
    polyorder (int): The order of the polynomial used to fit the samples.

    Returns:
    np.ndarray: The smoothed DF/F data.
    """

    # Copy the original data to avoid modifying it
    dff_smoothed = dff.copy()

    # Apply Savitzky-Golay filter to each neuron's data
    for i in range(dff.shape[0]):
        dff_smoothed[i] = savgol_filter(dff[i], window_length=window_length, polyorder=polyorder)

    return dff_smoothed

def read_dff(ops):
    """
    Reads DF/F data from an HDF5 file.

    This function opens the 'dff.h5' file located in the directory specified by ops['save_path0'],
    and extracts the DF/F data.

    Parameters:
    ops (dict): A dictionary containing operation parameters, including the 'save_path0' key
                which specifies the directory where the HDF5 file is located.

    Returns:
    np.array: An array containing the DF/F data.
    """

    f = h5py.File(os.path.join(ops['save_path0'], 'dff.h5'), 'r')
    dff = np.array(f['dff'])
    f.close()
    return dff

def read_ops(session_data_path):
    print('Processing {}'.format(session_data_path))
    ops = np.load(
        os.path.join(session_data_path, 'ops.npy'),
        allow_pickle=True).item()
    ops['save_path0'] = os.path.join(session_data_path)
    return ops

def read_masks(ops):
    f = h5py.File(os.path.join(ops['save_path0'], 'masks.h5'), 'r')
    labels = np.array(f['labels'])
    masks = np.array(f['masks_func'])
    mean_func = np.array(f['mean_func'])
    max_func = np.array(f['max_func'])
    mean_anat = np.array(f['mean_anat']) if ops['nchannels'] == 2 else None
    masks_anat = np.array(f['masks_anat']) if ops['nchannels'] == 2 else None
    f.close()
    return [labels, masks, mean_func, max_func, mean_anat, masks_anat]

def detect_spikes_from_dff(dff_trace, fs=30):
    """
    Detect spikes in dF/F trace using a moving window approach.
    
    Parameters:
    - dff_trace: 1D numpy array of dF/F values
    - fs: Sampling frequency (Hz)
    
    Returns:
    - spike_times: Array of spike times
    - spike_indices: Array of spike indices
    """
    # I tuned the parameters to detect spikes in the data (85th percentile, and 15 points around peak)
    
    # Calculate 85th percentile
    percentile_85 = np.percentile(dff_trace, 75)
    
    # Number of points around peak
    n_points = 5
    
    # Calculate the area under this window as a threshold  
    gaussian_area = percentile_85*n_points
    
    # Window for spike detection
    window_size = n_points
    
    # Detect spikes (I am using nan between as I will use nan values as a dectector in future steps)
    spike_indices = np.full(len(dff_trace), np.nan)
    
    for i in range(len(dff_trace) - window_size + 1):
        # Extract window
        window = dff_trace[i:i+window_size]
        
        # Calculate area of this window
        window_area = np.sum(window)
        
        # Compare areas
        if window_area > gaussian_area:
            spike_indices[i + window_size//5] = i + window_size//5
    
    # Convert indices to times
    spike_times = np.where(~np.isnan(spike_indices))[0] / fs
    
    return spike_times, spike_indices, gaussian_area

def group_spikes(spike_indices):
    """
    Group continuous spike indices, ignoring NaN values
    
    Parameters:
    - spike_indices: Array with spike indices and NaN values
    
    Returns:
    - List of lists containing grouped spike indices
    """
    # Find indices of non-NaN values
    valid_spikes = ~np.isnan(spike_indices)
    
    # Find the indices where the boolean array changes
    changes = np.diff(valid_spikes.astype(int))
    
    # Find start and end of each group
    group_starts = np.where(changes == 1)[0] + 1
    group_ends = np.where(changes == -1)[0] + 1
    
    # Handle edge cases
    if valid_spikes[0]:
        group_starts = np.insert(group_starts, 0, 0)
    if valid_spikes[-1]:
        group_ends = np.append(group_ends, len(spike_indices))
    
    # Extract groups
    spike_groups = []
    for start, end in zip(group_starts, group_ends):
        # Get indices of spikes in this group
        group = np.where(~np.isnan(spike_indices[start:end]))[0] + start
        spike_groups.append(group.tolist())
    
    return spike_groups

def area(dff_trace, spike_groups):
    """
    Finidng area of each Idendified calcium transient
    
    Parameters:
    - dff_trace: Original dF/F trace
    - spike_groups: List of spike groups (indices) // this can be filtered 
    
    Returns:
    - aligned_segments: List of aligned and padded segments
    - first_indices: First indices of each group
    """
    # Area
    area_segments = []
    
    for group in spike_groups:
        
        # Fill segment with original dF/F values
        area = np.sum(np.abs(dff_trace[group]))
        
        area_segments.append(area)
    
    return area_segments

def align_spike_segments(dff_trace, spike_groups, left_frames=0, right_frames=0):
    """
    Align spike segments based on the first index of each group and pad with dF/F values,
    then add extra frames to the left and right of each segment.

    Parameters:
    - dff_trace: Original dF/F trace
    - spike_groups: List of spike groups (indices)
    - left_frames: Number of frames to add on the left of each segment
    - right_frames: Number of frames to add on the right of each segment

    Returns:
    - aligned_segments: List of aligned and padded segments
    - first_indices: First indices of each group
    """
    # Determine the size of the largest segment and calculate padding
    segment_lengths = [len(group) for group in spike_groups]
    max_segment_length = max(segment_lengths)
    total_length = max_segment_length + left_frames + right_frames  # Total length after adding frames

    # Align and pad segments
    aligned_segments = []
    first_indices = []

    for group in spike_groups:
        # Start index for the padded segment
        start_index = max(group[0] - left_frames, 0)
        # End index for the padded segment
        end_index = min(group[-1] + right_frames + 1, len(dff_trace))  # +1 to include the last value

        # Extract the padded segment from dff_trace
        segment = dff_trace[start_index:end_index]

        # If the segment is shorter than the total length, pad with edge values
        if len(segment) < total_length:
            segment = np.pad(
                segment, 
                (0, total_length - len(segment)), 
                mode='edge'
            )

        aligned_segments.append(segment[:total_length])  # Ensure it's exactly the total length
        first_indices.append(group[0])  # Keep track of the original first index

    return aligned_segments, first_indices

def calculate_half_decay_time(segments, peak_values, peak_indices, left_frames, sampling_rate=30):
    """
    Calculate the half decay time of a given segment.

    Parameters:
    segments (np.ndarray): The aligned segments data.
    peak_values (np.ndarray): The peak values of each segment calculated before alingment.
    sampling_rate (int): The sampling rate in Hz. Default is 30.

    Returns:
    half_decay_time (np. ndarray): The half decay time in seconds.
    """
    
    # Find the index of the maximum value
    max_index = peak_indices + left_frames
    
    # Calculate the half maximum value
    half_max_value = peak_values / 2
    
    # Find the index where the segment falls below the half maximum value after the peak
    half_max_index = []
    for segment, max_idx, half_max_val in zip(segments, max_index, half_max_value):
        indices = np.where(segment[max_idx:] <= half_max_val)[0]
        if len(indices) > 0:
            half_max_index.append(indices[0])
        else:
            half_max_index.append(np.nan)  # or handle this case as needed

    half_max_index = np.array(half_max_index)
    
    # Calculate the half decay time in seconds
    half_decay_time = half_max_index / sampling_rate
    
    return half_decay_time

def create_violin_plot(ax, exc_data, inh_data, ylabel, filter_type):
    # Remove NaN values
    exc_data = np.array(exc_data)
    inh_data = np.array(inh_data)
    exc_data = exc_data[~np.isnan(exc_data)]
    inh_data = inh_data[~np.isnan(inh_data)]
    
    # Check if we have any valid data
    if len(exc_data) == 0 or len(inh_data) == 0:
        ax.text(0.5, 0.5, 'No valid data available', 
                horizontalalignment='center',
                verticalalignment='center',
                transform=ax.transAxes)
        ax.set_title(f'{ylabel} ({filter_type})')
        return
    
    # Calculate mean values
    exc_mean = np.mean(exc_data)
    inh_mean = np.mean(inh_data)
    
    # Create violin plot
    parts = ax.violinplot([exc_data, inh_data], showmeans=True, showmedians=True, showextrema=False)
    
    # Customize violin plot colors
    for pc in parts['bodies']:
        pc.set_facecolor('blue' if pc.get_paths()[0].vertices[0][0] < 1.5 else 'red')
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)
    
    parts['cmeans'].set_color('orange')
    parts['cmedians'].set_color('purple')
    
    # Adding box for quartiles
    for j, data in enumerate([exc_data, inh_data]):
        q1, q3 = np.percentile(data, [25, 75])
        ax.plot([j + 1, j + 1], [q1, q3], color='green', lw=5)
    
    # Create x-tick labels with counts
    x_labels = [f'Exc\n(n={len(exc_data)})', f'Inh\n(n={len(inh_data)})']
    ax.set_xticks([1, 2])
    ax.set_xticklabels(x_labels)
    
    ax.set_xlabel('Neuron Type')
    ax.set_ylabel(ylabel)
    ax.set_title(f'{ylabel} ({filter_type})')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Adding legend with mean values
    handles = [
        plt.Line2D([0], [0], color='orange', lw=2, label='Mean'),
        plt.Line2D([0], [0], color='purple', lw=2, label='Median'),
        plt.Line2D([0], [0], color='green', lw=5, label='Quartiles'),
        plt.Line2D([0], [0], color='blue', lw=0, label=f'Exc mean: {exc_mean:.2f}'),
        plt.Line2D([0], [0], color='red', lw=0, label=f'Inh mean: {inh_mean:.2f}')
    ]
    ax.legend(handles=handles)

    
# Helper function to pad arrays with NaN to make them homogeneous
def pad_with_nan(arrays):
    max_length = max(map(len, arrays))
    return np.array([np.pad(arr, (0, max_length - len(arr)), constant_values=np.nan) for arr in arrays])

# Helper function to calculate SEM
def calculate_sem(data):
    return np.nanstd(data, axis=0) / np.sqrt(np.sum(~np.isnan(data), axis=0))

def run_ca_transient(dff , ops):
    [labels_main, masks, mean_func, max_func, mean_anat, masks_anat] = read_masks(ops)
    
    print(f'{dff.shape[0]} neurons, recorded in {dff.shape[1]/30:f} second')
    unique_labels, counts = np.unique(labels_main, return_counts=True)
    label_mapping = {-1: 'excitatory', 1: 'inhibitory', 0: 'unsure'}
    print(f'neuron types report within {labels_main.shape[0]}')
    for label, count in zip(unique_labels, counts):
        print(f'{label_mapping[label]}: {count}')
        
    # Parameters
    window_length = 7  # Window size (must be odd) (Larger value smoother)
    polyorder = 3      # Polynomial order (lower values for smoother results)
    
    # Apply the smoothing function
    dff_filtered = smooth_dff(dff, window_length, polyorder)

    time = np.linspace(0,dff_filtered.shape[1]/30,dff_filtered.shape[1])
    
    all_spike_times = []
    all_spike_indices = []
    
    for i in tqdm(range(dff_filtered.shape[0])):
        spike_times, spike_indices, _ = detect_spikes_from_dff(dff_filtered[i])
        all_spike_times.append(spike_times)
        all_spike_indices.append(spike_indices)
    
    all_calcium_tran = []
    
    for i in tqdm(range(len(all_spike_indices))):
        all_calcium_tran.append(group_spikes(all_spike_indices[i]))
    

    # Define the different k values
    k_values = {'strict': 0.01, 'medium': 0.1, 'relaxed': 0.5}
    
    # Initialize the dictionary to store results
    results = {}
    
    # Iterate over all neurons
    for neuron_index in range(len(all_calcium_tran)):
        groups_1 = all_calcium_tran[neuron_index][1:-1]
        lengths = [len(lst) for lst in groups_1]
        mean_length = np.mean(lengths)
        std_length = np.std(lengths)
        
        # Initialize the dictionary for this neuron
        results[neuron_index] = {'initial_length': len(groups_1)}
        
        # Apply the statistical check for each k value
        for k_name, k in k_values.items():
            filtered_lists = [lst for lst in groups_1 if len(lst) >= mean_length - k * std_length]
            results[neuron_index][f'{k_name}_filtered_length'] = len(filtered_lists)
            results[neuron_index][f'{k_name}_filtered_lists'] = filtered_lists
            results[neuron_index][f'{k_name}_filtered_length'] = len(filtered_lists)
        
    for neuron_index in results.keys():
        for filter_type in ['strict', 'medium', 'relaxed']:
            spike_groups = results[neuron_index][f'{filter_type}_filtered_lists']
            dff_trace = dff_filtered[neuron_index]
            area_segments = area(dff_trace, spike_groups)
            results[neuron_index][f'{filter_type}_area_segments'] = area_segments
            # Peak vlaue
            results[neuron_index][f'{filter_type}_peak_segments'] = [np.max(dff_trace[group]) for group in spike_groups]
            # Peak time
            results[neuron_index][f'{filter_type}_peak_time'] = [np.argmax(dff_trace[group]) for group in spike_groups]


    left_frames = 30
    right_frames = 30
    
    for neuron_index in results.keys():
        for filter_type in ['strict', 'medium', 'relaxed']:
            spike_groups = results[neuron_index][f'{filter_type}_filtered_lists']
            dff_trace = dff_filtered[neuron_index]
            aligned_segments, first_indices = align_spike_segments(dff_trace, spike_groups, left_frames, right_frames)
            results[neuron_index][f'{filter_type}_aligned_segments'] = aligned_segments
            results[neuron_index][f'{filter_type}_average_segment'] = np.nanmean(aligned_segments, axis=0)
            results[neuron_index][f'{filter_type}_half_decay'] = calculate_half_decay_time(np.array(aligned_segments), 
                                                                                           np.array(results[neuron_index][f'{filter_type}_peak_segments']), 
                                                                                           np.array(results[neuron_index][f'{filter_type}_peak_time']), 
                                                                                           left_frames)
            results[neuron_index][f'{filter_type}_time_peak_segment'] = [time / 30 for time in results[neuron_index][f'{filter_type}_peak_time']]
            results[neuron_index][f'{filter_type}_sem_segment'] = np.nanstd(aligned_segments, axis=0) / np.sqrt(len(aligned_segments))
            results[neuron_index][f'{filter_type}_first_indices'] = first_indices
    return labels_main , time, dff_filtered, results

    # Define the PDF file name
def plot_individual_ca_transient(labels_main , results, pdf_file_name):
    
    label_mapping = {-1: 'excitatory', 1: 'inhibitory', 0: 'unsure'}
    left_frames = 30
    right_frames = 30
    # Create a PdfPages object
    with PdfPages(pdf_file_name) as pdf:
        # Iterate over all neurons
        for neuron_index in tqdm(range(len(labels_main))):
            neuron_type = label_mapping[labels_main[neuron_index]]
            
            # Create a new figure for each neuron
            fig, axs = plt.subplots(3, 3, figsize=(24, 18))
            fig.suptitle(f'Neuron {neuron_index} - Type: {neuron_type}', fontsize=16)
            
            filter_types = ['relaxed', 'medium', 'strict']
            
            for i, filter_type in enumerate(filter_types):
                # Extract area segments for the specified neuron and filter types
                area_segments = results[neuron_index][f'{filter_type}_area_segments']
                
                # Plot violin plot for area segments
                parts = axs[0, i].violinplot(area_segments, showmeans=True, showmedians=True)
                
                # Customize violin plot colors
                parts['cmeans'].set_color('orange')    # Set mean color
                parts['cmedians'].set_color('purple')  # Set median color
    
                # Adding box for quartiles
                q1, q3 = np.percentile(area_segments, [25, 75])
                axs[0, i].plot([1, 1], [q1, q3], color='green', lw=5)
    
                axs[0, i].set_title(f'Area Segments ({filter_type})')
                axs[0, i].set_xlabel('Filter Type')
                axs[0, i].set_ylabel('Calcium Transient Area')
                axs[0, i].set_xticks([1])
                axs[0, i].set_xticklabels([filter_type])
                
                # Report the number of initial_length and after applying filter
                initial_length = results[neuron_index]['initial_length']
                filtered_length = results[neuron_index][f'{filter_type}_filtered_length']
                axs[0, i].text(0.5, 0.95, f'Initial: {initial_length}\nFiltered: {filtered_length}', 
                               transform=axs[0, i].transAxes, fontsize=12, verticalalignment='top', horizontalalignment='right')
                
                axs[0, i].spines['top'].set_visible(False)
                axs[0, i].spines['right'].set_visible(False)
                
                # Plot average segment
                average_segment = results[neuron_index][f'{filter_type}_average_segment']
                std_segment = results[neuron_index][f'{filter_type}_sem_segment']
                tim_sta = np.linspace(0, average_segment.shape[0]/30, average_segment.shape[0])
                
                peak_time = np.argmax(average_segment)
                peak_time_s = (peak_time - left_frames) / 30
                
                axs[1, i].plot(tim_sta, average_segment, color='k', label='STA')
                axs[1, i].fill_between(tim_sta, average_segment - std_segment, average_segment + std_segment, color='gray', alpha=0.5, label='STD')
                axs[1, i].plot(tim_sta[peak_time], average_segment[peak_time], marker='o', color='r', label=f'onset to peak time: {peak_time_s:.2f} sec')
                axs[1, i].axvline(x=left_frames/30, color='r', linestyle='--', label='Calcium_transient onset')
                axs[1, i].set_xlabel('time (s)')
                axs[1, i].set_ylabel('dff')
                axs[1, i].legend()
                axs[1, i].set_title(f'Average of identified calcium transient ({filter_type})')
                axs[1, i].spines['top'].set_visible(False)
                axs[1, i].spines['right'].set_visible(False)
                
                # Plot all segments
                aligned_segments = results[neuron_index][f'{filter_type}_aligned_segments']
                for segment in aligned_segments:
                    axs[2, i].plot(tim_sta, segment, color='gray', alpha=0.5)
                axs[2, i].plot(tim_sta, average_segment, color='k', label='STA')
                axs[2, i].axvline(x=left_frames/30, color='k', linestyle='--', label='Calcium_transient onset')
                axs[2, i].set_xlabel('time (s)')
                axs[2, i].set_ylabel('dff')
                axs[2, i].legend()
                axs[2, i].set_title(f'All identified calcium transients ({filter_type})')
                axs[2, i].spines['top'].set_visible(False)
                axs[2, i].spines['right'].set_visible(False)
            
            # Adjust layout and save the figure to the PDF
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            pdf.savefig(fig)
            plt.close(fig)
            
def plot_average_ca_transient(labels_main , results, pdf_file_name, inhibit = 0):
    # Initialize lists to store pooled data for excitatory neurons
    Exc_all_areas_strict = []
    Exc_all_peaks_strict = []
    Exc_all_decay_times_strict = []
    Exc_all_segments_strict = []
    Exc_all_time_peak_segments_strict = []

    Exc_all_areas_medium = []
    Exc_all_peaks_medium = []
    Exc_all_decay_times_medium = []
    Exc_all_segments_medium = []
    Exc_all_time_peak_segments_medium = []

    Exc_all_areas_relaxed = []
    Exc_all_peaks_relaxed = []
    Exc_all_decay_times_relaxed = []
    Exc_all_segments_relaxed = []
    Exc_all_time_peak_segments_relaxed = []

    # Initialize lists to store pooled data for inhibitory neurons
    if inhibit:
        Inh_all_areas_strict = []
        Inh_all_peaks_strict = []
        Inh_all_decay_times_strict = []
        Inh_all_segments_strict = []
        Inh_all_time_peak_segments_strict = []
    
        Inh_all_areas_medium = []
        Inh_all_peaks_medium = []
        Inh_all_decay_times_medium = []
        Inh_all_segments_medium = []
        Inh_all_time_peak_segments_medium = []
    
        Inh_all_areas_relaxed = []
        Inh_all_peaks_relaxed = []
        Inh_all_decay_times_relaxed = []
        Inh_all_segments_relaxed = []
        Inh_all_time_peak_segments_relaxed = []
        
    for neuron_index in results.keys():
        if labels_main[neuron_index] == -1:  # Check if the neuron is excitatory
            Exc_all_areas_strict.append(results[neuron_index]['strict_area_segments'])
            Exc_all_peaks_strict.append(results[neuron_index]['strict_peak_segments'])
            Exc_all_decay_times_strict.append(results[neuron_index]['strict_half_decay'])
            Exc_all_segments_strict.append(results[neuron_index]['strict_aligned_segments'])
            Exc_all_time_peak_segments_strict.append(results[neuron_index]['strict_time_peak_segment'])

            Exc_all_areas_medium.append(results[neuron_index]['medium_area_segments'])
            Exc_all_peaks_medium.append(results[neuron_index]['medium_peak_segments'])
            Exc_all_decay_times_medium.append(results[neuron_index]['medium_half_decay'])
            Exc_all_segments_medium.append(results[neuron_index]['medium_aligned_segments'])
            Exc_all_time_peak_segments_medium.append(results[neuron_index]['medium_time_peak_segment'])

            Exc_all_areas_relaxed.append(results[neuron_index]['relaxed_area_segments'])
            Exc_all_peaks_relaxed.append(results[neuron_index]['relaxed_peak_segments'])
            Exc_all_decay_times_relaxed.append(results[neuron_index]['relaxed_half_decay'])
            Exc_all_segments_relaxed.append(results[neuron_index]['relaxed_aligned_segments'])
            Exc_all_time_peak_segments_relaxed.append(results[neuron_index]['relaxed_time_peak_segment'])
        elif labels_main[neuron_index] == 1:  # Check if the neuron is inhibitory
            if inhibit:
                Inh_all_areas_strict.append(results[neuron_index]['strict_area_segments'])
                Inh_all_peaks_strict.append(results[neuron_index]['strict_peak_segments'])
                Inh_all_decay_times_strict.append(results[neuron_index]['strict_half_decay'])
                Inh_all_segments_strict.append(results[neuron_index]['strict_aligned_segments'])
                Inh_all_time_peak_segments_strict.append(results[neuron_index]['strict_time_peak_segment'])
    
                Inh_all_areas_medium.append(results[neuron_index]['medium_area_segments'])
                Inh_all_peaks_medium.append(results[neuron_index]['medium_peak_segments'])
                Inh_all_decay_times_medium.append(results[neuron_index]['medium_half_decay'])
                Inh_all_segments_medium.append(results[neuron_index]['medium_aligned_segments'])
                Inh_all_time_peak_segments_medium.append(results[neuron_index]['medium_time_peak_segment'])
    
                Inh_all_areas_relaxed.append(results[neuron_index]['relaxed_area_segments'])
                Inh_all_peaks_relaxed.append(results[neuron_index]['relaxed_peak_segments'])
                Inh_all_decay_times_relaxed.append(results[neuron_index]['relaxed_half_decay'])
                Inh_all_segments_relaxed.append(results[neuron_index]['relaxed_aligned_segments'])
                Inh_all_time_peak_segments_relaxed.append(results[neuron_index]['relaxed_time_peak_segment'])
    
    
    Exc_all_segments = {'strict': [], 'medium': [], 'relaxed': []}
    if inhibit:
        Inh_all_segments = {'strict': [], 'medium': [], 'relaxed': []}

    # Iterate through the results and pool the data for each neuron type
    for neuron_index in results.keys():
        if labels_main[neuron_index] == -1:  # Excitatory neurons
            for filter_type in ['strict', 'medium', 'relaxed']:
                Exc_all_segments[filter_type].extend(results[neuron_index][f'{filter_type}_aligned_segments'])
        elif labels_main[neuron_index] == 1:  # Inhibitory neurons
            if inhibit:
                for filter_type in ['strict', 'medium', 'relaxed']:
                    Inh_all_segments[filter_type].extend(results[neuron_index][f'{filter_type}_aligned_segments'])
    fig, axs = plt.subplots(5, 3, figsize=(18, 20))

    filter_types = ['relaxed', 'medium', 'strict']

    for i, filter_type in enumerate(filter_types):
        left_frames = 30
        right_frames = 30
        # Pad segments with NaN to make them homogeneous
        Exc_segments_padded = pad_with_nan(Exc_all_segments[filter_type])
        
        # Calculate average and SEM
        Exc_average = np.nanmean(Exc_segments_padded, axis=0)
        Exc_sem = calculate_sem(Exc_segments_padded)
        
        if inhibit:
            Inh_segments_padded = pad_with_nan(Inh_all_segments[filter_type])
            Inh_average = np.nanmean(Inh_segments_padded, axis=0)
            Inh_sem = calculate_sem(Inh_segments_padded)
            time_array_inh = np.linspace(0, Inh_average.shape[0] / 30, Inh_average.shape[0])
            axs[0, i].plot(time_array_inh, Inh_average, color='red', label=f'Inh Average (n={len(Inh_all_segments[filter_type])})')
            axs[0, i].fill_between(time_array_inh, Inh_average - Inh_sem, Inh_average + Inh_sem, color='red', alpha=0.3)
        
        # Time array
        time_array_exc = np.linspace(0, Exc_average.shape[0] / 30, Exc_average.shape[0])
        
        
        # Plot average and SEM for excitatory neurons
        axs[0, i].plot(time_array_exc, Exc_average, color='blue', label=f'Exc Average (n={len(Exc_all_segments[filter_type])})')
        axs[0, i].fill_between(time_array_exc, Exc_average - Exc_sem, Exc_average + Exc_sem, color='blue', alpha=0.3)
        
       
        
        # Add black dashed line at left_frames/30
        axs[0, i].axvline(x=left_frames/30, color='k', linestyle='--', label='Identified Calcium Transient onset')
        
        # Limit x-axis to 6 seconds
        axs[0, i].set_xlim([0, 6])
        
        # Add labels and title
        axs[0, i].set_title(f'Average of Identified Calcium Transients ({filter_type})')
        axs[0, i].set_xlabel('Time (s)')
        axs[0, i].set_ylabel('dF/F +/- SEM')
        axs[0, i].legend()
        axs[0, i].spines['top'].set_visible(False)
        axs[0, i].spines['right'].set_visible(False)

    # Plot violin plots for all rows
    for i, filter_type in enumerate(filter_types):
        # Area plots (second row)
        if filter_type == 'strict':
            Exc_areas = [area for neuron_areas in Exc_all_areas_strict for area in neuron_areas]
            Inh_areas = [area for neuron_areas in Inh_all_areas_strict for area in neuron_areas]
            
            Exc_time_peaks = [tp for neuron_tp in Exc_all_time_peak_segments_strict for tp in neuron_tp]
            Inh_time_peaks = [tp for neuron_tp in Inh_all_time_peak_segments_strict for tp in neuron_tp]
            
            Exc_decay = [d for neuron_d in Exc_all_decay_times_strict for d in neuron_d]
            Inh_decay = [d for neuron_d in Inh_all_decay_times_strict for d in neuron_d]
            
            Exc_peaks = [p for neuron_p in Exc_all_peaks_strict for p in neuron_p]
            Inh_peaks = [p for neuron_p in Inh_all_peaks_strict for p in neuron_p]
            
        elif filter_type == 'medium':
            Exc_areas = [area for neuron_areas in Exc_all_areas_medium for area in neuron_areas]
            Inh_areas = [area for neuron_areas in Inh_all_areas_medium for area in neuron_areas]
            
            Exc_time_peaks = [tp for neuron_tp in Exc_all_time_peak_segments_medium for tp in neuron_tp]
            Inh_time_peaks = [tp for neuron_tp in Inh_all_time_peak_segments_medium for tp in neuron_tp]
            
            Exc_decay = [d for neuron_d in Exc_all_decay_times_medium for d in neuron_d]
            Inh_decay = [d for neuron_d in Inh_all_decay_times_medium for d in neuron_d]
            
            Exc_peaks = [p for neuron_p in Exc_all_peaks_medium for p in neuron_p]
            Inh_peaks = [p for neuron_p in Inh_all_peaks_medium for p in neuron_p]
            
        else:  # relaxed
            Exc_areas = [area for neuron_areas in Exc_all_areas_relaxed for area in neuron_areas]
            Inh_areas = [area for neuron_areas in Inh_all_areas_relaxed for area in neuron_areas]
            
            Exc_time_peaks = [tp for neuron_tp in Exc_all_time_peak_segments_relaxed for tp in neuron_tp]
            Inh_time_peaks = [tp for neuron_tp in Inh_all_time_peak_segments_relaxed for tp in neuron_tp]
            
            Exc_decay = [d for neuron_d in Exc_all_decay_times_relaxed for d in neuron_d]
            Inh_decay = [d for neuron_d in Inh_all_decay_times_relaxed for d in neuron_d]
            
            Exc_peaks = [p for neuron_p in Exc_all_peaks_relaxed for p in neuron_p]
            Inh_peaks = [p for neuron_p in Inh_all_peaks_relaxed for p in neuron_p]

        # Create violin plots for each metric
        create_violin_plot(axs[1, i], Exc_areas, Inh_areas, 'Calcium Transient Area', filter_type)
        create_violin_plot(axs[2, i], Exc_time_peaks, Inh_time_peaks, 'Time to Peak', filter_type)
        create_violin_plot(axs[3, i], Exc_decay, Inh_decay, 'Decay Time', filter_type)
        create_violin_plot(axs[4, i], Exc_peaks, Inh_peaks, 'Peak Value', filter_type)
        
    plt.tight_layout()
    plt.show()