import numpy as np
from tqdm import tqdm

def trim_seq(data, pivots):
    """Trim sequences to the same length based on pivot points.

    This function trims a list of sequences (1D or 3D arrays) to the same length, aligning them
    around specified pivot points. The trimmed length is determined by the shortest available
    segments before and after the pivots across all sequences. For 1D arrays, the function trims
    the sequences directly. For 3D arrays, it trims along the last dimension (time axis).

    Args:
        data (list): List of numpy arrays, either 1D (e.g., time series) or 3D (e.g., multi-channel
            time series with shape [channels, features, time]).
        pivots (numpy.ndarray or list): Array of pivot indices for each sequence, indicating the
            alignment points for trimming.

    Returns:
        list: List of trimmed numpy arrays, all having the same length, aligned around the pivot points.

    Notes:
        - For 1D arrays, each sequence is trimmed to [pivot - len_l_min : pivot + len_r_min].
        - For 3D arrays, trimming is applied along the last dimension (time axis).
        - len_l_min is the minimum distance from the start of any sequence to its pivot.
        - len_r_min is the minimum distance from any pivot to the end of its sequence.
        - The function assumes all sequences in data have compatible shapes (all 1D or all 3D).
    """
    if len(data[0].shape) == 1:
        len_l_min = np.min(pivots)
        len_r_min = np.min([len(data[i]) - pivots[i] for i in range(len(data))])
        data = [data[i][pivots[i] - len_l_min:pivots[i] + len_r_min]
                for i in range(len(data))]
    if len(data[0].shape) == 3:
        len_l_min = np.min(pivots)
        len_r_min = np.min([len(data[i][0, 0, :]) - pivots[i] for i in range(len(data))])
        data = [data[i][:, :, pivots[i] - len_l_min:pivots[i] + len_r_min]
                for i in range(len(data))]
    return data

def get_lick_response(neural_trials, l_frames, r_frames):
    """Extract neural responses aligned to licking events with associated properties.

    This function processes neural trial data to extract fluorescence signal segments
    (dff) centered around licking events, along with their corresponding time stamps
    and lick properties (direction, correction, and type). The segments are trimmed
    to a uniform length using the trim_seq function, aligning them at the licking event
    time (zero-centered). The results are concatenated and returned as a list of arrays.

    Args:
        neural_trials (dict): Dictionary containing neural trial data with keys:
            - 'time' (numpy.memmap): Time points for neural data.
            - 'dff' (numpy.memmap): Fluorescence signal data.
            - 'trial_labels' (pandas.DataFrame): Trial labels including lick data.
        l_frames (int): Number of frames to include before each licking event.
        r_frames (int): Number of frames to include after each licking event.

    Returns:
        list: A list containing:
            - neu_seq (numpy.ndarray): Concatenated fluorescence signal segments for each lick event.
            - neu_time (numpy.ndarray): Mean time stamps relative to each lick event (zero-centered).
            - direction (numpy.ndarray): Array of lick directions.
            - correction (numpy.ndarray): Array of lick correction indicators.
            - lick_type (numpy.ndarray): Array of lick types.

    Notes:
        - Lick data is extracted from neural_trials['trial_labels']['lick'], assumed to be a
          2D array with shape [4, num_licks], where rows represent time, direction, correction,
          and type.
        - Only non-NaN lick events within the valid time range (allowing l_frames before and
          r_frames after) are processed.
        - The trim_seq function is used to ensure all neural sequences and time stamps are of
          equal length, aligned at the lick event time.
        - The progress of the loop over lick events is displayed using tqdm.
    """
    # initialization.
    time = neural_trials['time']
    neu_seq = []
    neu_time = []
    direction = []
    correction = []
    lick_type = []
    # get all licking events.
    lick = np.concatenate(neural_trials['trial_labels']['lick'].to_numpy(), axis=1)
    # loop over licks.
    for li in (range(lick.shape[1])):
        t = lick[0, li]
        if not np.isnan(t):
            # get state start timing.
            idx = np.searchsorted(neural_trials['time'], t)
            if idx > l_frames and idx < len(neural_trials['time']) - r_frames:
                # signal response.
                f = neural_trials['dff'][:, idx - l_frames : idx + r_frames]
                f = np.expand_dims(f, axis=0)
                neu_seq.append(f)
                # signal time stamps.
                neu_time.append(neural_trials['time'][idx - l_frames : idx + r_frames] - time[idx])
                # licking properties.
                direction.append(lick[1, li])
                correction.append(lick[2, li])
                lick_type.append(lick[3, li])
    # correct neural data centering at zero.
    neu_time_zero = [np.argmin(np.abs(nt)) for nt in neu_time]
    neu_seq = trim_seq(neu_seq, neu_time_zero)
    neu_time = trim_seq(neu_time, neu_time_zero)
    # concatenate results.
    neu_seq = np.concatenate(neu_seq, axis=0)
    neu_time = [nt.reshape(1, -1) for nt in neu_time]
    neu_time = np.concatenate(neu_time, axis=0)
    direction = np.array(direction)
    correction = np.array(correction)
    lick_type = np.array(lick_type)
    # get mean time stamps.
    neu_time = np.mean(neu_time, axis=0)
    # combine results.
    return [neu_seq, neu_time, direction, correction, lick_type]

def get_perception_response(neural_trials, target_state, l_frames, r_frames, indices=0):
    """Extract neural and stimulus responses around specific trial states.

    This function processes neural trial data to extract fluorescence signal segments (dff),
    voltage signals, and task-related variables aligned to a specified trial state (e.g., choice
    or reward). The data is centered around the state timing, trimmed to uniform lengths using
    the trim_seq function, and returned as a list of arrays. The first and last few trials are
    excluded to avoid edge effects.

    Args:
        neural_trials (dict): Dictionary containing neural trial data with keys:
            - 'time' (numpy.memmap): Time points for neural data.
            - 'dff' (numpy.memmap): Fluorescence signal data.
            - 'vol_time' (numpy.memmap): Time points for voltage signals.
            - 'vol_stim_vis' (numpy.memmap): Visual stimulation signals.
            - 'vol_led' (numpy.memmap): LED signals.
            - 'trial_labels' (pandas.DataFrame): Trial labels including state, stim_seq, trial_type, isi, lick, and outcome.
        target_state (str): The trial state (e.g., 'state_window_choice', 'state_reward') to align responses to.
        l_frames (int): Number of frames to include before the state event.
        r_frames (int): Number of frames to include after the state event.
        indices (int, optional): Index to select from the flattened target_state array (default: 0).

    Returns:
        list: A list containing:
            - neu_seq (numpy.ndarray): Concatenated fluorescence signal segments for each trial.
            - neu_time (numpy.ndarray): Mean neural time stamps relative to the state event (zero-centered).
            - stim_seq (numpy.ndarray): Concatenated stimulus sequence arrays, time-adjusted.
            - stim_value (numpy.ndarray): Concatenated visual stimulation signals.
            - stim_time (numpy.ndarray): Mean voltage time stamps relative to the state event (zero-centered).
            - led_value (numpy.ndarray): Concatenated LED signals.
            - trial_type (numpy.ndarray): Array of trial types.
            - isi (numpy.ndarray): Array of inter-stimulus intervals.
            - decision (numpy.ndarray): Array of lick decisions (first lick direction).
            - outcome (numpy.ndarray): Array of trial outcomes.

    Notes:
        - Excludes the first 2 and last 2 trials to avoid edge effects.
        - Only processes non-NaN state events within the valid time range (allowing l_frames before and r_frames after).
        - The trim_seq function ensures all sequences (neural, time, and voltage) are of equal length, aligned at the state event.
        - The progress of the trial loop is displayed using tqdm.
        - The stim_seq is reshaped to [1, 2, 2] per trial and adjusted relative to the state event time.
    """
    exclude_start_trials = 2
    exclude_end_trials = 2
    # initialization.
    time = neural_trials['time']
    neu_seq = []
    neu_time = []
    stim_seq = []
    stim_value = []
    stim_time = []
    led_value = []
    trial_type = []
    block_type = []
    isi = []
    decision = []
    outcome = []
    # loop over trials.
    for ti in (range(len(neural_trials['trial_labels']))):
        t = neural_trials['trial_labels'][target_state][ti].flatten()[indices]
        if (not np.isnan(t) and
            ti >= exclude_start_trials and
            ti < len(neural_trials['trial_labels']) - exclude_end_trials):
            # get state start timing.
            idx = np.searchsorted(neural_trials['time'], t)
            if idx > l_frames and idx < len(neural_trials['time']) - r_frames:
                # signal response.
                f = neural_trials['dff'][:, idx - l_frames : idx + r_frames]
                f = np.expand_dims(f, axis=0)
                neu_seq.append(f)
                # signal time stamps.
                neu_time.append(neural_trials['time'][idx - l_frames : idx + r_frames] - time[idx])
                # voltage.
                vol_t_c = np.searchsorted(neural_trials['vol_time'], neural_trials['time'][idx])
                vol_t_l = np.searchsorted(neural_trials['vol_time'], neural_trials['time'][idx - l_frames])
                vol_t_r = np.searchsorted(neural_trials['vol_time'], neural_trials['time'][idx + r_frames])
                stim_time.append(neural_trials['vol_time'][vol_t_l:vol_t_r] - neural_trials['vol_time'][vol_t_c])
                stim_value.append(neural_trials['vol_stim_vis'][vol_t_l:vol_t_r])
                led_value.append(neural_trials['vol_led'][vol_t_l:vol_t_r])
                # task variables.
                stim_seq.append(neural_trials['trial_labels']['stim_seq'][ti].reshape(1, 2, 2) - t)
                trial_type.append(neural_trials['trial_labels']['trial_type'][ti])
                block_type.append(neural_trials['trial_labels']['block_type'][ti])
                isi.append(neural_trials['trial_labels']['isi'][ti])
                decision.append(neural_trials['trial_labels']['lick'][ti][1, 0])
                outcome.append(neural_trials['trial_labels']['outcome'][ti])
        else:
            pass
    # correct neural data centering at zero.
    neu_time_zero = [np.argmin(np.abs(nt)) for nt in neu_time]
    neu_time = trim_seq(neu_time, neu_time_zero)
    neu_seq = trim_seq(neu_seq, neu_time_zero)
    # correct voltage data centering at zero.
    stim_time_zero = [np.argmin(np.abs(sv)) for sv in stim_value]
    stim_time = trim_seq(stim_time, stim_time_zero)
    stim_value = trim_seq(stim_value, stim_time_zero)
    led_value = trim_seq(led_value, stim_time_zero)
    # concatenate results.
    neu_seq = np.concatenate(neu_seq, axis=0)
    neu_time = [nt.reshape(1, -1) for nt in neu_time]
    neu_time = np.concatenate(neu_time, axis=0)
    stim_seq = np.concatenate(stim_seq, axis=0)
    stim_value = [sv.reshape(1, -1) for sv in stim_value]
    stim_value = np.concatenate(stim_value, axis=0)
    stim_time = [st.reshape(1, -1) for st in stim_time]
    stim_time = np.concatenate(stim_time, axis=0)
    led_value = [lv.reshape(1, -1) for lv in led_value]
    led_value = np.concatenate(led_value, axis=0)
    trial_type = np.array(trial_type)
    block_type = np.array(block_type)
    isi = np.array(isi)
    decision = np.array(decision)
    outcome = np.array(outcome)
    # get mean time stamps.
    neu_time = np.mean(neu_time, axis=0)
    stim_time = np.mean(stim_time, axis=0)
    # combine results.
    return [neu_seq, neu_time, stim_seq, stim_value, stim_time, led_value, trial_type, block_type, isi, decision, outcome]