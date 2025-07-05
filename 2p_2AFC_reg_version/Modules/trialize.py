import numpy as np
import os
import h5py

from Modules.reader import read_dff
from Modules.reader import read_raw_voltages
from Modules.reader import read_bpod_mat_data

def remove_start_impulse(vol_time, vol_stim_vis):
    """Remove short-duration voltage impulses from a stimulation signal.

    This function identifies and removes voltage impulses in the stimulation signal
    (vol_stim_vis) that have a duration less than a specified minimum threshold.
    Impulses are detected by finding transitions in the signal, and those shorter
    than min_duration are set to zero.

    Args:
        vol_time (numpy.ndarray): Array of time points corresponding to the stimulation signal.
        vol_stim_vis (numpy.ndarray): Array representing the stimulation signal (binary, 0 or 1).

    Returns:
        numpy.ndarray: Modified stimulation signal with short impulses removed.

    Notes:
        - The minimum duration threshold is set to 100 units (assumed to be in the same units as vol_time).
        - The function assumes vol_stim_vis contains binary values (0 or 1).
        - If the signal starts or ends with an impulse, the function handles these edge cases appropriately.
    """
    min_duration = 100
    changes = np.diff(vol_stim_vis.astype(int))
    start_indices = np.where(changes == 1)[0] + 1
    end_indices = np.where(changes == -1)[0] + 1
    if vol_stim_vis[0] == 1:
        start_indices = np.insert(start_indices, 0, 0)
    if vol_stim_vis[-1] == 1:
        end_indices = np.append(end_indices, len(vol_stim_vis))
    for start, end in zip(start_indices, end_indices):
        duration = vol_time[end-1] - vol_time[start]
        if duration < min_duration:
            vol_stim_vis[start:end] = 0
    return vol_stim_vis

def correct_vol_start(vol_stim_vis):
    """Correct the stimulation signal to ensure it starts with a zero value.

    This function checks if the stimulation signal starts with a value of 1. If it does,
    it sets all initial values up to the first zero to 0, ensuring the signal begins
    with a low state.

    Args:
        vol_stim_vis (numpy.ndarray): Array representing the stimulation signal (binary, 0 or 1).

    Returns:
        numpy.ndarray: Modified stimulation signal with the beginning corrected to start at 0.

    Notes:
        - The function assumes vol_stim_vis contains binary values (0 or 1).
        - If the signal already starts with 0, no changes are made.
        - The correction is applied up to the first occurrence of 0 in the signal.
    """
    if vol_stim_vis[0] == 1:
        vol_stim_vis[:np.where(vol_stim_vis==0)[0][0]] = 0
    return vol_stim_vis

def get_trigger_time(vol_time, vol_bin):
    """Detect rising and falling edges in a binary signal and return their corresponding times.

    This function identifies the rising (0 to 1) and falling (1 to 0) edges in a binary signal
    by computing the difference of the signal and mapping the edge indices to their corresponding
    time points.

    Args:
        vol_time (numpy.ndarray): Array of time points corresponding to the binary signal.
        vol_bin (numpy.ndarray): Array representing the binary signal (0 or 1).

    Returns:
        tuple: A tuple containing two numpy arrays:
            - time_up (numpy.ndarray): Time points of rising edges (0 to 1 transitions).
            - time_down (numpy.ndarray): Time points of falling edges (1 to 0 transitions).

    Notes:
        - The function assumes vol_bin contains binary values (0 or 1).
        - Time points are returned in the same units as vol_time.
        - The np.diff with prepend=0 is used to detect edges, ensuring the first element is considered.
    """
    # find the edge with np.diff and correct it by preappend one 0.
    diff_vol = np.diff(vol_bin, prepend=0)
    idx_up = np.where(diff_vol == 1)[0]
    idx_down = np.where(diff_vol == -1)[0]
    # select the indice for rising and falling.
    # give the edges in ms.
    time_up = vol_time[idx_up]
    time_down = vol_time[idx_down]
    return time_up, time_down

def get_session_start_time(vol_time, vol_start):
    """Find the start time of a session based on the first rising edge in a binary signal.

    This function uses the get_trigger_time function to detect the rising edges in the
    binary signal (vol_start) and returns the time of the first rising edge, indicating
    the session start time.

    Args:
        vol_time (numpy.ndarray): Array of time points corresponding to the binary signal.
        vol_start (numpy.ndarray): Array representing the binary session start signal (0 or 1).

    Returns:
        float: The time of the first rising edge in vol_start, representing the session start time.

    Notes:
        - The function assumes vol_start contains binary values (0 or 1).
        - The time is returned in the same units as vol_time.
        - The function expects at least one rising edge in vol_start; otherwise, an error may occur.
    """
    time_up, _ = get_trigger_time(vol_time, vol_start)
    session_start_time = time_up[0]
    return session_start_time

def correct_time_img_center(time_img):
    """Correct fluorescence signal timing to the center of photon integration intervals.

    This function adjusts the timing of fluorescence imaging data by shifting each time point
    to the center of its corresponding photon integration interval. The intervals are calculated
    using the differences between consecutive time points, with the last interval estimated as
    the mean of the previous intervals.

    Args:
        time_img (numpy.ndarray): Array of time points for the fluorescence imaging data.

    Returns:
        numpy.ndarray: Corrected time points, shifted to the center of each photon integration interval.

    Notes:
        - The function assumes time_img contains monotonically increasing time points.
        - The last time interval is estimated as the mean of all prior intervals to handle the edge case.
        - The correction shifts each time point by half the duration of its interval.
    """
    # find the frame internal.
    diff_time_img = np.diff(time_img, append=0)
    # correct the last element.
    diff_time_img[-1] = np.mean(diff_time_img[:-1])
    # move the image timing to the center of photon integration interval.
    diff_time_img = diff_time_img / 2
    # correct each individual timing.
    time_neuro = time_img + diff_time_img
    return time_neuro

def save_trials(
        ops, time_neuro, dff, trial_labels,
        vol_time, vol_stim_vis,
        vol_stim_aud, vol_flir,
        vol_pmt, vol_led
        ):
    """Save neural trial data and associated labels to files.

    This function saves neural trial data, including fluorescence signals, stimulation signals,
    and corresponding time points, to an HDF5 file. Trial labels are saved separately as a CSV file.
    The data is stored in a specified directory, overwriting any existing files with the same names.

    Args:
        ops (dict): Dictionary containing configuration options, including 'save_path0' for the output directory.
        time_neuro (numpy.ndarray): Array of corrected time points for neural data.
        dff (numpy.ndarray): Array of fluorescence signal data (delta F/F).
        trial_labels (pandas.DataFrame): DataFrame containing trial labels.
        vol_time (numpy.ndarray): Array of time points for voltage signals.
        vol_stim_vis (numpy.ndarray): Array of visual stimulation signals (binary, 0 or 1).
        vol_stim_aud (numpy.ndarray): Array of auditory stimulation signals (binary, 0 or 1).
        vol_flir (numpy.ndarray): Array of FLIR camera signals.
        vol_pmt (numpy.ndarray): Array of photomultiplier tube signals.
        vol_led (numpy.ndarray): Array of LED signals.

    Returns:
        None

    Notes:
        - The HDF5 file is saved as 'neural_trials.h5' in the directory specified by ops['save_path0'].
        - The trial labels are saved as 'trial_labels.csv' in the same directory.
        - If an HDF5 file already exists at the specified path, it is overwritten.
        - The HDF5 file contains a group 'neural_trials' with datasets for time, dff, vol_time,
          vol_stim_vis, vol_stim_aud, vol_flir, vol_pmt, and vol_led.
    """
    # file structure:
    # ops['save_path0'] / neural_trials.h5
    # ---- time
    # ---- stim
    # ---- dff
    # ---- vol_stim
    # ---- vol_time
    # trial_labels.csv
    h5_path = os.path.join(ops['save_path0'], 'neural_trials.h5')
    if os.path.exists(h5_path):
        os.remove(h5_path)
    f = h5py.File(h5_path, 'w')
    grp = f.create_group('neural_trials')
    grp['time']         = time_neuro
    grp['dff']          = dff
    grp['vol_time']     = vol_time
    grp['vol_stim_vis'] = vol_stim_vis
    grp['vol_stim_aud'] = vol_stim_aud
    grp['vol_flir']     = vol_flir
    grp['vol_pmt']      = vol_pmt
    grp['vol_led']      = vol_led
    f.close()
    trial_labels.to_csv(os.path.join(ops['save_path0'], 'trial_labels.csv'))


def trialize(ops_dir):
    """Trialize neural data and save it to HDF5 and CSV files.

    This function orchestrates the processing of neural data by reading fluorescence traces
    and voltage recordings, correcting stimulation signals and imaging timings, and saving
    the processed data. It integrates several helper functions to remove short impulses,
    correct signal starts, detect session start times, adjust imaging timings, and save the
    results in an HDF5 file and a CSV file for trial labels.

    Args:
        ops_dir (dict): Dictionary containing configuration options, including paths for
            reading data files and saving output (e.g., 'save_path0' for output directory).

    Returns:
        None

    Notes:
        - The function assumes the existence of helper functions: `read_dff`,
          `read_raw_voltages`, `remove_start_impulse`, `correct_vol_start`,
          `get_session_start_time`, `get_trigger_time`, `correct_time_img_center`,
          and `save_trials`.
        - Outputs are saved as 'neural_trials.h5' (containing neural and voltage data)
          and 'trial_labels.csv' (containing trial labels) in the directory specified by
          ops_dir['save_path0'].
        - The function prints progress messages during execution.
    """
    print('Reading dff traces and voltage recordings')
    dff = read_dff(ops_dir)
    [vol_time, vol_start, vol_stim_vis, vol_img,
        vol_hifi, vol_stim_aud, vol_flir,
        vol_pmt, vol_led] = read_raw_voltages(ops_dir)
    vol_stim_vis = remove_start_impulse(vol_time, vol_stim_vis)
    vol_stim_vis = correct_vol_start(vol_stim_vis)
    session_start_time = get_session_start_time(vol_time, vol_start)
    trial_labels = read_bpod_mat_data(ops_dir, session_start_time)
    print('Correcting 2p camera trigger time')
    # signal trigger time stamps.
    time_img, _ = get_trigger_time(vol_time, vol_img)
    # correct imaging timing.
    time_neuro = correct_time_img_center(time_img)
    # save the final data.
    print('Saving trial data')
    save_trials(
        ops_dir, time_neuro, dff, trial_labels,
        vol_time, vol_stim_vis,
        vol_stim_aud, vol_flir,
        vol_pmt, vol_led)
