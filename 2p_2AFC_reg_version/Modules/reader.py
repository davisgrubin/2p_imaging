# Liberary for reading session data from multiple directories
import os
import numpy as np
import h5py
import shutil
import pandas as pd
import scipy.io as sio
from scipy.signal import savgol_filter

# First fuction to read for reading sessions data from multiple directories
def read_ops(list_session_data_path):
    """
    Reads and processes operation data from multiple session data paths.

    This function loads operation data stored in 'ops.npy' files from a list of session
    directories, updates each operation dictionary with its corresponding session path,
    and returns a list of these operation dictionaries.

    Args:
        list_session_data_path (list): A list of strings, where each string is a path to
            a directory containing an 'ops.npy' file with operation data.

    Returns:
        list: A list of dictionaries, where each dictionary contains the operation data
            loaded from an 'ops.npy' file, with an additional 'save_path0' key pointing
            to the corresponding session directory.

    Notes:
        - The function uses `numpy.load` with `allow_pickle=True` to load the 'ops.npy'
          files, which are expected to contain pickled Python objects (dictionaries).
        - The 'ops.npy' file must exist in each session directory specified in
          `list_session_data_path`.
        - The function assumes that the loaded `ops` data is a dictionary and modifies it
          by adding or updating the 'save_path0' key with the session directory path.

    Example:
        >>> session_paths = ['/path/to/session1', '/path/to/session2']
        >>> ops_list = read_ops(session_paths)
        >>> print(ops_list[0]['save_path0'])
        '/path/to/session1'

    Raises:
        FileNotFoundError: If an 'ops.npy' file is not found in any of the specified
            session directories.
        ValueError: If the loaded 'ops.npy' file does not contain a dictionary.
    """
    list_ops = []
    for session_data_path in list_session_data_path:
        ops = np.load(
            os.path.join(session_data_path, 'ops.npy'),
            allow_pickle=True).item()
        ops['save_path0'] = os.path.join(session_data_path)
        list_ops.append(ops)
    return list_ops

def create_memmap(data, dtype, mmap_path):
    """
    Creates a memory-mapped array from input data and saves it to a specified file path.

    This function creates a memory-mapped array with the same shape and data type as the
    input data, writes the input data to the memory-mapped file, and returns the
    memory-mapped array for further use.

    Args:
        data (numpy.ndarray): The input NumPy array whose data will be written to the
            memory-mapped file.
        dtype (numpy.dtype): The data type of the memory-mapped array.
        mmap_path (str): The file path where the memory-mapped array will be stored.

    Returns:
        numpy.memmap: A memory-mapped array with the specified shape and data type,
            containing the data from the input array.

    Notes:
        - The memory-mapped array is created in 'w+' mode, which creates a new file or
          overwrites an existing file for reading and writing.
        - The function uses `numpy.memmap` to create the memory-mapped array, which allows
          efficient handling of large arrays by mapping them directly to disk.
        - The input `data` is copied into the memory-mapped array using slice assignment.

    Example:
        >>> import numpy as np
        >>> data = np.array([[1, 2], [3, 4]], dtype=np.int32)
        >>> mmap_arr = create_memmap(data, np.int32, 'data.mmap')
        >>> print(mmap_arr)
        [[1 2]
         [3 4]]

    Raises:
        ValueError: If the `dtype` does not match the data type of the input `data` or if
            the `mmap_path` is invalid.
        OSError: If there are issues with file creation or access at `mmap_path`.
    """
    memmap_arr = np.memmap(mmap_path, dtype=dtype, mode='w+', shape=data.shape)
    memmap_arr[:] = data[...]
    return memmap_arr

def get_memmap_path(ops, h5_file_name):
    """
    Generates paths for memory-mapped data and the corresponding HDF5 file.

    This function creates a directory for memory-mapped data based on the provided HDF5
    file name and constructs paths for both the memory-mapped directory and the HDF5 file.
    The memory-mapped directory is created under a 'memmap' subdirectory within the
    session's save path if it does not already exist.

    Args:
        ops (dict): A dictionary containing session metadata, including the key
            'save_path0' which specifies the base directory path for the session.
        h5_file_name (str): The name of the HDF5 file (e.g., 'data.h5').

    Returns:
        tuple: A tuple containing two strings:
            - mm_path (str): The path to the memory-mapped data directory.
            - file_path (str): The path to the HDF5 file.

    Notes:
        - The memory-mapped directory is named after the HDF5 file name (without its
          extension) and is created under '<save_path0>/memmap/'.
        - If the memory-mapped directory does not exist, it is created automatically.
        - The function assumes that 'save_path0' exists in the `ops` dictionary and is a
          valid directory path.

    Example:
        >>> ops = {'save_path0': '/path/to/session'}
        >>> h5_file_name = 'data.h5'
        >>> mm_path, file_path = get_memmap_path(ops, h5_file_name)
        >>> print(mm_path)
        '/path/to/session/memmap/data'
        >>> print(file_path)
        '/path/to/session/data.h5'

    Raises:
        KeyError: If 'save_path0' is not present in the `ops` dictionary.
        OSError: If there are issues creating the memory-mapped directory or accessing
            the file system.
    """
    mm_folder_name, _ = os.path.splitext(h5_file_name)
    if not os.path.exists(os.path.join(ops['save_path0'], 'memmap', mm_folder_name)):
        os.makedirs(os.path.join(ops['save_path0'], 'memmap', mm_folder_name))
    mm_path = os.path.join(ops['save_path0'], 'memmap', mm_folder_name)
    file_path = os.path.join(ops['save_path0'], h5_file_name)
    return mm_path, file_path

def read_masks(ops):
    """
    Reads mask-related data from an HDF5 file and creates memory-mapped arrays.

    This function retrieves paths for memory-mapped data and an HDF5 file using
    `get_memmap_path`, then loads specific datasets from the HDF5 file into memory-mapped
    arrays. It handles both functional and anatomical data, with anatomical data loaded
    only if the session has two channels.

    Args:
        ops (dict): A dictionary containing session metadata, including:
            - 'save_path0': The base directory path for the session.
            - 'nchannels': The number of channels in the session (e.g., 1 or 2).

    Returns:
        list: A list containing the following memory-mapped arrays (or None for anatomical
            data if `ops['nchannels'] != 2`):
            - labels (numpy.memmap): Array of integer labels (dtype: int8).
            - masks (numpy.memmap): Functional mask data (dtype: float32).
            - mean_func (numpy.memmap): Mean functional data (dtype: float32).
            - max_func (numpy.memmap): Maximum functional data (dtype: float32).
            - mean_anat (numpy.memmap or None): Mean anatomical data (dtype: float32) if
                `ops['nchannels'] == 2`, otherwise None.
            - masks_anat (numpy.memmap or None): Anatomical mask data (dtype: float32) if
                `ops['nchannels'] == 2`, otherwise None.

    Notes:
        - The function assumes the HDF5 file 'masks.h5' exists in the session directory
          specified by `ops['save_path0']` and contains the datasets 'labels',
          'masks_func', 'mean_func', 'max_func', and optionally 'mean_anat' and
          'masks_anat' for two-channel data.
        - Memory-mapped arrays are created using the `create_memmap` function and stored
          in a 'memmap' subdirectory under the session path.
        - The HDF5 file is accessed in read-only mode ('r') using `h5py.File`.

    Example:
        >>> ops = {'save_path0': '/path/to/session', 'nchannels': 2}
        >>> result = read_masks(ops)
        >>> print(result[0])  # Access the labels memory-mapped array
        <numpy.memmap object with dtype=int8>

    Raises:
        KeyError: If required keys ('save_path0', 'nchannels') are missing from `ops` or
            required datasets are missing from the HDF5 file.
        FileNotFoundError: If the 'masks.h5' file does not exist at the specified path.
        OSError: If there are issues creating memory-mapped files or accessing the file
            system.
    """
    mm_path, file_path = get_memmap_path(ops, 'masks.h5')
    with h5py.File(file_path, 'r') as f:
        labels     = create_memmap(f['labels'],     'int8',    os.path.join(mm_path, 'labels.mmap'))
        masks      = create_memmap(f['masks_func'], 'float32', os.path.join(mm_path, 'masks_func.mmap'))
        mean_func  = create_memmap(f['mean_func'],  'float32', os.path.join(mm_path, 'mean_func.mmap'))
        max_func   = create_memmap(f['max_func'],   'float32', os.path.join(mm_path, 'max_func.mmap'))
        mean_anat  = create_memmap(f['mean_anat'],  'float32', os.path.join(mm_path, 'mean_anat.mmap')) if ops['nchannels'] == 2 else None
        masks_anat = create_memmap(f['masks_anat'], 'float32', os.path.join(mm_path, 'masks_anat.mmap')) if ops['nchannels'] == 2 else None
    return [labels, masks, mean_func, max_func, mean_anat, masks_anat]


def read_raw_voltages(ops):
    """
    Reads raw voltage data from an HDF5 file and creates memory-mapped arrays.

    This function retrieves paths for memory-mapped data and an HDF5 file using
    `get_memmap_path`, then loads specific voltage-related datasets from the HDF5 file
    into memory-mapped arrays. The datasets include timestamps and various voltage signals
    related to experimental events (e.g., visual stimuli, audio stimuli, imaging triggers).

    Args:
        ops (dict): A dictionary containing session metadata, including:
            - 'save_path0': The base directory path for the session.

    Returns:
        list: A list containing the following memory-mapped arrays:
            - vol_time (numpy.memmap): Timestamps for voltage signals (dtype: float32).
            - vol_start (numpy.memmap): Trial start trigger signals (dtype: int8).
            - vol_stim_vis (numpy.memmap): Visual stimulus signals (dtype: int8).
            - vol_img (numpy.memmap): Imaging trigger signals (dtype: int8).
            - vol_hifi (numpy.memmap): HiFi (audio) trigger signals (dtype: int8).
            - vol_stim_aud (numpy.memmap): Audio stimulus signals (dtype: float32).
            - vol_flir (numpy.memmap): FLIR camera trigger signals (dtype: int8).
            - vol_pmt (numpy.memmap): PMT (photomultiplier tube) signals (dtype: int8).
            - vol_led (numpy.memmap): LED trigger signals (dtype: int8).

    Notes:
        - The function assumes the HDF5 file 'raw_voltages.h5' exists in the session
          directory specified by `ops['save_path0']` and contains a 'raw' group with
          datasets: 'vol_time', 'vol_start', 'vol_stim_vis', 'vol_img', 'vol_hifi',
          'vol_stim_aud', 'vol_flir', 'vol_pmt', and 'vol_led'.
        - Memory-mapped arrays are created using the `create_memmap` function and stored
          in a 'memmap' subdirectory under the session path.
        - The HDF5 file is accessed in read-only mode ('r') using `h5py.File`.

    Example:
        >>> ops = {'save_path0': '/path/to/session'}
        >>> voltages = read_raw_voltages(ops)
        >>> print(voltages[0])  # Access the vol_time memory-mapped array
        <numpy.memmap object with dtype=float32>

    Raises:
        KeyError: If required keys (e.g., 'save_path0') are missing from `ops` or if
            expected datasets are missing in the HDF5 file.
        FileNotFoundError: If the 'raw_voltages.h5' file does not exist at the specified
            path.
        OSError: If there are issues creating memory-mapped files or accessing the file
            system.
    """
    mm_path, file_path = get_memmap_path(ops, 'raw_voltages.h5')
    with h5py.File(file_path, 'r') as f:
        vol_time     = create_memmap(f['raw']['vol_time'],     'float32', os.path.join(mm_path, 'vol_time.mmap'))
        vol_start    = create_memmap(f['raw']['vol_start'],    'int8',    os.path.join(mm_path, 'vol_start.mmap'))
        vol_stim_vis = create_memmap(f['raw']['vol_stim_vis'], 'int8',    os.path.join(mm_path, 'vol_stim_vis.mmap'))
        vol_hifi     = create_memmap(f['raw']['vol_hifi'],     'int8',    os.path.join(mm_path, 'vol_hifi.mmap'))
        vol_img      = create_memmap(f['raw']['vol_img'],      'int8',    os.path.join(mm_path, 'vol_img.mmap'))
        vol_stim_aud = create_memmap(f['raw']['vol_stim_aud'], 'float32', os.path.join(mm_path, 'vol_stim_aud.mmap'))
        vol_flir     = create_memmap(f['raw']['vol_flir'],     'int8',    os.path.join(mm_path, 'vol_flir.mmap'))
        vol_pmt      = create_memmap(f['raw']['vol_pmt'],      'int8',    os.path.join(mm_path, 'vol_pmt.mmap'))
        vol_led      = create_memmap(f['raw']['vol_led'],      'int8',    os.path.join(mm_path, 'vol_led.mmap'))
    return [vol_time, vol_start, vol_stim_vis, vol_img,
            vol_hifi, vol_stim_aud, vol_flir,
            vol_pmt, vol_led]

def read_dff(ops):
    """
    Reads dF/F (delta F over F) traces from an HDF5 file and creates a memory-mapped array.

    This function retrieves paths for memory-mapped data and an HDF5 file using
    `get_memmap_path`, then loads the dF/F dataset from the HDF5 file into a
    memory-mapped array.

    Args:
        ops (dict): A dictionary containing session metadata, including:
            - 'save_path0': The base directory path for the session.

    Returns:
        numpy.memmap: A memory-mapped array containing dF/F traces (dtype: float32).

    Notes:
        - The function assumes the HDF5 file 'dff.h5' exists in the session directory
          specified by `ops['save_path0']` and contains a 'dff' dataset.
        - The memory-mapped array is created using the `create_memmap` function and
          stored in a 'memmap' subdirectory under the session path.
        - The HDF5 file is accessed in read-only mode ('r') using `h5py.File`.

    Example:
        >>> ops = {'save_path0': '/path/to/session'}
        >>> dff_data = read_dff(ops)
        >>> print(dff_data)  # Access the dF/F memory-mapped array
        <numpy.memmap object with dtype=float32>

    Raises:
        KeyError: If required keys (e.g., 'save_path0') are missing from `ops` or if the
            'dff' dataset is missing in the HDF5 file.
        FileNotFoundError: If the 'dff.h5' file does not exist at the specified path.
        OSError: If there are issues creating the memory-mapped file or accessing the file
            system.
    """
    mm_path, file_path = get_memmap_path(ops, 'dff.h5')
    with h5py.File(file_path, 'r') as f:
        dff = create_memmap(f['dff'], 'float32', os.path.join(mm_path, 'dff.mmap'))
    return dff
            
# Reading bpod from matlab file
def read_bpod_mat_data(ops, session_start_time):
    """
    Reads and processes Bpod session data from a MATLAB file into a structured DataFrame.

    This function loads behavioral data from a 'bpod_session_data.mat' file, processes
    trial-related information (e.g., trial timings, outcomes, stimulus sequences, and
    licking events), and organizes it into a pandas DataFrame. It handles nested MATLAB
    structures and converts them into Python dictionaries, adjusts timestamps relative to
    a session start time, and labels trial outcomes and states.

    Args:
        ops (dict): A dictionary containing session metadata, including:
            - 'save_path0': The base directory path where the 'bpod_session_data.mat' file
              is located.
        session_start_time (float): The reference start time (in milliseconds) for
            correcting trial timestamps.

    Returns:
        pandas.DataFrame: A DataFrame containing trial-related data with the following
            columns:
            - time_trial_start (float): Trial start timestamps (ms, adjusted to session
              start).
            - time_trial_end (float): Trial end timestamps (ms, adjusted to session start).
            - trial_type (int): Trial type indicator (0-based, derived from raw data).
            - outcome (str): Trial outcome ('punish', 'reward', 'naive_punish',
              'naive_reward', 'no_choose', or 'other').
            - state_window_choice (array): Timestamps for the 'WindowChoice' state [start, end].
            - state_reward (array): Timestamps for the 'Reward' state [start, end].
            - state_punish (array): Timestamps for the 'Punish' state [start, end].
            - stim_seq (array): Stimulus sequence timestamps [[BNC1High], [BNC1Low]].
            - isi (float): Inter-stimulus interval (ms) between BNC1High and BNC1Low.
            - lick (array): Licking events with [timestamps, direction, correctness, lick_type].

    Notes:
        - The function assumes the 'bpod_session_data.mat' file exists in the directory
          specified by `ops['save_path0']` and contains a 'SessionData' structure with
          fields like 'nTrials', 'RawEvents', 'TrialStartTimestamp', 'TrialEndTimestamp',
          and 'TrialTypes'.
        - Timestamps are converted to milliseconds and adjusted relative to the session
          start time.
        - Nested MATLAB structures are recursively converted to Python dictionaries using
          helper functions `_check_keys`, `_todict`, and `_tolist`.
        - Trial outcomes are determined by the `states_labeling` helper function based on
          the presence of specific states in the trial data.
        - Stimulus sequences and licking events are processed to handle missing or invalid
          data (e.g., NaN values for absent events).
        - The 'yicong_forever' string is used as a temporary placeholder to handle array
          length alignment and is removed before returning the DataFrame.

    Example:
        >>> ops = {'save_path0': '/path/to/session'}
        >>> session_start_time = 1000.0
        >>> df = read_bpod_mat_data(ops, session_start_time)
        >>> print(df[['time_trial_start', 'outcome']].head())
           time_trial_start    outcome
        0           1000.0    reward
        1           1500.0    punish
        ...

    Raises:
        KeyError: If required keys (e.g., 'save_path0') are missing from `ops` or if
            expected fields are missing in the MATLAB file.
        FileNotFoundError: If the 'bpod_session_data.mat' file does not exist.
        ValueError: If the data structure in the MATLAB file is malformed or incompatible.
    """
    def _check_keys(d):
        """
        Recursively converts MATLAB structs to dictionaries.

        Args:
            d (dict): Input dictionary containing MATLAB struct objects.

        Returns:
            dict: Dictionary with MATLAB structs converted to nested dictionaries.
        """
        for key in d:
            if isinstance(d[key], sio.matlab.mat_struct):
                d[key] = _todict(d[key])
        return d

    def _todict(matobj):
        """
        Converts a MATLAB struct to a Python dictionary.

        Args:
            matobj (sio.matlab.mat_struct): MATLAB struct object.

        Returns:
            dict: Dictionary with field names as keys and converted values.
        """
        d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, sio.matlab.mat_struct):
                d[strg] = _todict(elem)
            elif isinstance(elem, np.ndarray):
                d[strg] = _tolist(elem)
            else:
                d[strg] = elem
        return d

    def _tolist(ndarray):
        """
        Recursively converts NumPy arrays to lists, handling nested MATLAB structs.

        Args:
            ndarray (numpy.ndarray): Input NumPy array.

        Returns:
            list: List of converted elements.
        """
        elem_list = []
        for sub_elem in ndarray:
            if isinstance(sub_elem, sio.matlab.mat_struct):
                elem_list.append(_todict(sub_elem))
            elif isinstance(sub_elem, np.ndarray):
                elem_list.append(_tolist(sub_elem))
            else:
                elem_list.append(sub_elem)
        return elem_list

    def states_labeling(trial_states):
        """
        Labels trial outcomes based on trial states.

        Args:
            trial_states (dict): Dictionary of trial states and their timestamps.

        Returns:
            str: Outcome label ('punish', 'reward', 'naive_punish', 'naive_reward',
                'no_choose', or 'other').
        """
        if 'Punish' in trial_states.keys() and not np.isnan(trial_states['Punish'][0]):
            outcome = 'punish'
        elif 'Reward' in trial_states.keys() and not np.isnan(trial_states['Reward'][0]):
            outcome = 'reward'
        elif 'PunishNaive' in trial_states.keys() and not np.isnan(trial_states['PunishNaive'][0]):
            outcome = 'naive_punish'
        elif 'RewardNaive' in trial_states.keys() and not np.isnan(trial_states['RewardNaive'][0]):
            outcome = 'naive_reward'
        elif 'DidNotChoose' in trial_states.keys() and not np.isnan(trial_states['DidNotChoose'][0]):
            outcome = 'no_choose'
        else:
            outcome = 'other'
        return outcome

    def get_state(trial_state_dict, target_state, trial_start):
        """
        Retrieves timestamps for a specific trial state.

        Args:
            trial_state_dict (dict): Dictionary of trial states.
            target_state (str): Name of the target state.
            trial_start (float): Trial start timestamp (ms).

        Returns:
            numpy.ndarray: Array of [start, end] timestamps (ms) or [NaN, NaN] if the
                state is not found.
        """
        if target_state in trial_state_dict:
            time_state = 1000 * np.array(trial_state_dict[target_state]) + trial_start
        else:
            time_state = np.array([np.nan, np.nan])
        return time_state

    # Read raw data from MATLAB file
    raw = sio.loadmat(
        os.path.join(ops['save_path0'], 'bpod_session_data.mat'),
        struct_as_record=False, squeeze_me=True)
    raw = _check_keys(raw)['SessionData']

    trial_labels = dict()
    n_trials = raw['nTrials']
    trial_states = [raw['RawEvents']['Trial'][ti]['States'] for ti in range(n_trials)]
    trial_events = [raw['RawEvents']['Trial'][ti]['Events'] for ti in range(n_trials)]

    # Trial start and end timestamps
    trial_labels['time_trial_start'] = 1000 * np.array(raw['TrialStartTimestamp']).reshape(-1)
    trial_labels['time_trial_end'] = 1000 * np.array(raw['TrialEndTimestamp']).reshape(-1)
    trial_labels['time_trial_end'] = trial_labels['time_trial_end'] - trial_labels['time_trial_start'][0] + session_start_time
    trial_labels['time_trial_start'] = trial_labels['time_trial_start'] - trial_labels['time_trial_start'][0] + session_start_time

    # Trial type and outcome
    trial_labels['trial_type'] = np.array(raw['TrialTypes']).reshape(-1) - 1
    trial_labels['outcome'] = np.array([states_labeling(ts) for ts in trial_states], dtype='object')

    # Trial state timings
    trial_labels['state_window_choice'] = np.array([
        get_state(trial_states[ti], 'WindowChoice', trial_labels['time_trial_start'][ti])
        for ti in range(n_trials)] + ['yicong_forever'], dtype='object')[:-1]
    trial_labels['state_reward'] = np.array([
        get_state(trial_states[ti], 'Reward', trial_labels['time_trial_start'][ti])
        for ti in range(n_trials)] + ['yicong_forever'], dtype='object')[:-1]
    trial_labels['state_punish'] = np.array([
        get_state(trial_states[ti], 'Punish', trial_labels['time_trial_start'][ti])
        for ti in range(n_trials)] + ['yicong_forever'], dtype='object')[:-1]

    # Stimulus timing
    trial_isi = []
    trial_stim_seq = []
    for ti in range(n_trials):
        if ('BNC1High' in trial_events[ti].keys() and
            'BNC1Low' in trial_events[ti].keys() and
            len(np.array(trial_events[ti]['BNC1High']).reshape(-1)) == 2 and
            len(np.array(trial_events[ti]['BNC1Low']).reshape(-1)) == 2):
            stim_seq = 1000 * np.array([trial_events[ti]['BNC1High'], trial_events[ti]['BNC1Low']]) + trial_labels['time_trial_start'][ti]
            stim_seq = np.transpose(stim_seq, [1, 0])
            isi = 1000 * np.array(trial_events[ti]['BNC1High'][1] - trial_events[ti]['BNC1Low'][0])
        else:
            stim_seq = np.array([[np.nan, np.nan], [np.nan, np.nan]])
            isi = np.nan
        trial_stim_seq.append(stim_seq)
        trial_isi.append(isi)
    trial_labels['stim_seq'] = np.array(trial_stim_seq + ['yicong_forever'], dtype='object')[:-1]
    trial_labels['isi'] = np.array(trial_isi + ['yicong_forever'], dtype='object')[:-1]

    # Licking events
    trial_lick = []
    for ti in range(n_trials):
        licking_events = []
        direction = []
        correctness = []
        if 'Port1In' in trial_events[ti].keys():
            lick_left = np.array(trial_events[ti]['Port1In']).reshape(-1)
            licking_events.append(lick_left)
            direction.append(np.zeros_like(lick_left))
            if trial_labels['trial_type'][ti] == 0:
                correctness.append(np.ones_like(lick_left))
            else:
                correctness.append(np.zeros_like(lick_left))
        if 'Port3In' in trial_events[ti].keys():
            lick_right = np.array(trial_events[ti]['Port3In']).reshape(-1)
            licking_events.append(lick_right)
            direction.append(np.ones_like(lick_right))
            if trial_labels['trial_type'][ti] == 1:
                correctness.append(np.ones_like(lick_right))
            else:
                correctness.append(np.zeros_like(lick_right))
        if len(licking_events) > 0:
            licking_events = 1000 * np.concatenate(licking_events).reshape(1, -1) + trial_labels['time_trial_start'][ti]
            direction = np.concatenate(direction).reshape(1, -1)
            correctness = np.concatenate(correctness).reshape(1, -1)
            lick = np.concatenate([licking_events, direction, correctness], axis=0)
            lick = lick[:, np.argsort(lick[0, :])]
            lick = lick[:, lick[0, :] >= trial_labels['state_window_choice'][ti][0]]
            if np.size(lick) != 0:
                lick_type = np.full(lick.shape[1], np.nan)
                lick_type[0] = 1
                if (not np.isnan(trial_labels['state_reward'][ti][1]) and
                    len(lick_type) > 1):
                    lick_type[1:][lick[0, 1:] > trial_labels['state_reward'][ti][0]] = 0
                lick_type = lick_type.reshape(1, -1)
                lick = np.concatenate([lick, lick_type], axis=0)
            else:
                lick = np.array([[np.nan], [np.nan], [np.nan], [np.nan]])
        else:
            lick = np.array([[np.nan], [np.nan], [np.nan], [np.nan]])
        trial_lick.append(lick)
    trial_labels['lick'] = np.array(trial_lick + ['yicong_forever'], dtype='object')[:-1]

    # Convert to DataFrame
    trial_labels = pd.DataFrame(trial_labels)
    return trial_labels

# Readiing Trialized data
def read_trial_label(ops):
    """Read trial label data from a CSV file into a pandas DataFrame.

    This function reads a CSV file containing trial label data and converts specific columns
    into numpy arrays with appropriate data types and shapes. Some columns containing array-like
    data are parsed from string representations back into numpy arrays. The processed data is
    returned as a structured pandas DataFrame.

    Args:
        ops (dict): Dictionary containing configuration options, including 'save_path0' for the
            directory where the 'trial_labels.csv' file is located.

    Returns:
        pandas.DataFrame: A DataFrame containing the trial label data with columns:
            - time_trial_start (float32): Start times of trials.
            - time_trial_end (float32): End times of trials.
            - trial_type (int8): Type of each trial.
            - outcome (object): Outcome of each trial.
            - state_window_choice (object): Array of choice window states per trial.
            - state_reward (object): Array of reward states per trial.
            - state_punish (object): Array of punishment states per trial.
            - stim_seq (object): 2D array of stimulus sequences per trial.
            - isi (float32): Inter-stimulus intervals.
            - lick (object): 2D array of lick data per trial.

    Notes:
        - The CSV file is expected to be located at ops['save_path0']/trial_labels.csv.
        - Columns with array-like data (e.g., state_window_choice, stim_seq, lick) are stored as
          strings in the CSV and are parsed back into numpy arrays with specified shapes.
        - The function uses a helper function, object_parse, to handle array parsing.
        - The 'yicong_forever' string is appended during parsing and then removed to maintain
          data integrity.
    """
    raw_csv = pd.read_csv(os.path.join(ops['save_path0'], 'trial_labels.csv'), index_col=0)
    # recover object numpy array from csv str.
    def object_parse(k, shape):
        arr = np.array(
            [np.fromstring(s.replace('[', '').replace(']', ''), sep=' ').reshape(shape)
             for s in raw_csv[k].to_list()] + ['yicong_forever'],
            dtype='object')[:-1]
        return arr
    # parse all array.
    time_trial_start = raw_csv['time_trial_start'].to_numpy(dtype='float32')
    time_trial_end = raw_csv['time_trial_end'].to_numpy(dtype='float32')
    trial_type = raw_csv['trial_type'].to_numpy(dtype='int8')
    outcome = raw_csv['outcome'].to_numpy(dtype='object')
    state_window_choice = object_parse('state_window_choice', [-1])
    state_reward = object_parse('state_reward', [-1])
    state_punish = object_parse('state_punish', [-1])
    stim_seq = object_parse('stim_seq', [-1, 2])
    isi = raw_csv['isi'].to_numpy(dtype='float32')
    lick = object_parse('lick', [4, -1])
    # convert to dataframe.
    trial_labels = pd.DataFrame({
        'time_trial_start': time_trial_start,
        'time_trial_end': time_trial_end,
        'trial_type': trial_type,
        'outcome': outcome,
        'state_window_choice': state_window_choice,
        'state_reward': state_reward,
        'state_punish': state_punish,
        'stim_seq': stim_seq,
        'isi': isi,
        'lick': lick,
        })
    return trial_labels

def read_neural_trials(ops, smooth):
    """Read neural trial data from an HDF5 file and create memory-mapped arrays.

    This function reads neural trial data from an HDF5 file, including fluorescence signals,
    time points, and voltage signals, and optionally applies a Savitzky-Golay filter to smooth
    the fluorescence data. The data is stored as memory-mapped arrays for efficient access and
    returned as a dictionary. Trial labels are read using the read_trial_label function.

    Args:
        ops (dict): Dictionary containing configuration options, including 'save_path0' for the
            directory where the 'neural_trials.h5' file is located.
        smooth (bool): If True, apply Savitzky-Golay smoothing to the fluorescence data (dff).

    Returns:
        dict: A dictionary containing memory-mapped arrays and trial labels:
            - dff (numpy.memmap): Fluorescence signal data (float32).
            - time (numpy.memmap): Corrected neural time points (float32).
            - trial_labels (pandas.DataFrame): Trial label data from read_trial_label.
            - vol_time (numpy.memmap): Voltage signal time points (float32).
            - vol_stim_vis (numpy.memmap): Visual stimulation signals (int8).
            - vol_stim_aud (numpy.memmap): Auditory stimulation signals (float32).
            - vol_flir (numpy.memmap): FLIR camera signals (int8).
            - vol_pmt (numpy.memmap): Photomultiplier tube signals (int8).
            - vol_led (numpy.memmap): LED signals (int8).

    Notes:
        - The HDF5 file is expected to be located at ops['save_path0']/neural_trials.h5.
        - If smooth is True, a Savitzky-Golay filter is applied to the dff data with a window
          length of 9 and polynomial order of 3.
        - Memory-mapped files are created in a directory specified by get_memmap_path.
        - The function assumes the existence of helper functions: get_memmap_path,
          read_trial_label, and create_memmap.
    """
    mm_path, file_path = get_memmap_path(ops, 'neural_trials.h5')
    trial_labels = read_trial_label(ops)
    with h5py.File(file_path, 'r') as f:
        neural_trials = dict()
        dff = np.array(f['neural_trials']['dff'])
        if smooth:
            window_length = 9
            polyorder = 3
            dff = np.apply_along_axis(
                savgol_filter, 1, dff,
                window_length=window_length,
                polyorder=polyorder)
        else:
            pass
        neural_trials['dff']          = create_memmap(dff,                                'float32', os.path.join(mm_path, 'dff.mmap'))
        neural_trials['time']         = create_memmap(f['neural_trials']['time'],         'float32', os.path.join(mm_path, 'time.mmap'))
        neural_trials['trial_labels'] = trial_labels
        neural_trials['vol_time']     = create_memmap(f['neural_trials']['vol_time'],     'float32', os.path.join(mm_path, 'vol_time.mmap'))
        neural_trials['vol_stim_vis'] = create_memmap(f['neural_trials']['vol_stim_vis'], 'int8',    os.path.join(mm_path, 'vol_stim_vis.mmap'))
        neural_trials['vol_stim_aud'] = create_memmap(f['neural_trials']['vol_stim_aud'], 'float32', os.path.join(mm_path, 'vol_stim_aud.mmap'))
        neural_trials['vol_flir']     = create_memmap(f['neural_trials']['vol_flir'],     'int8',    os.path.join(mm_path, 'vol_flir.mmap'))
        neural_trials['vol_pmt']      = create_memmap(f['neural_trials']['vol_pmt'],      'int8',    os.path.join(mm_path, 'vol_pmt.mmap'))
        neural_trials['vol_led']      = create_memmap(f['neural_trials']['vol_led'],      'int8',    os.path.join(mm_path, 'vol_led.mmap'))
    return neural_trials

# clean memory mapping files.
def clean_memap_path(ops):
    try:
        if os.path.exists(os.path.join(ops['save_path0'], 'memmap')):
            shutil.rmtree(os.path.join(ops['save_path0'], 'memmap'))
    except: pass