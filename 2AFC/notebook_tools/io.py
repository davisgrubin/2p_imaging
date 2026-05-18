"""Common I/O and session-path helper implementations migrated from `Test_pilot/test_nb_io.py` and `Test_pilot/test_nb_session_paths.py`.

This module is now self-contained so notebooks do not depend on `Test_pilot/test_nb_*`.
"""

import os
import re
import glob
import numpy as np
import h5py
import matplotlib.pyplot as plt

USE_VIS_STIM_PAIR_TRIAL_START_FALLBACK = True
FORCE_VIS_STIM_PAIR_TRIAL_START = False

def configure_trial_start_flags(use_vis_stim_pair_trial_start_fallback=True,
                                force_vis_stim_pair_trial_start=False):
    global USE_VIS_STIM_PAIR_TRIAL_START_FALLBACK
    global FORCE_VIS_STIM_PAIR_TRIAL_START
    USE_VIS_STIM_PAIR_TRIAL_START_FALLBACK = use_vis_stim_pair_trial_start_fallback
    FORCE_VIS_STIM_PAIR_TRIAL_START = force_vis_stim_pair_trial_start

# create a numpy memmap from an h5py dataset.
def create_memmap(data, dtype, mmap_path, chunk_size=1000000):
    memmap_arr = np.memmap(mmap_path, dtype=dtype, mode='w+', shape=data.shape)
    if data.ndim == 0:
        memmap_arr[...] = data[()]
    elif data.ndim == 1:
        for start in range(0, data.shape[0], chunk_size):
            end = min(start + chunk_size, data.shape[0])
            memmap_arr[start:end] = data[start:end]
    else:
        for start in range(0, data.shape[0], chunk_size):
            end = min(start + chunk_size, data.shape[0])
            memmap_arr[start:end, ...] = data[start:end, ...]
    memmap_arr.flush()
    return memmap_arr

# create folder for h5 data.
def get_memmap_path(ops, h5_file_name, subdir=None):
    mm_folder_name, _ = os.path.splitext(h5_file_name)
    mm_root = os.path.join(ops['save_path0'], 'memmap')
    file_root = ops['save_path0']
    if subdir is not None:
        mm_root = os.path.join(mm_root, subdir)
        file_root = os.path.join(file_root, subdir)
    if not os.path.exists(os.path.join(mm_root, mm_folder_name)):
        os.makedirs(os.path.join(mm_root, mm_folder_name))
    mm_path = os.path.join(mm_root, mm_folder_name)
    file_path = os.path.join(file_root, h5_file_name)
    return mm_path, file_path

####################################################################
# read masks.
def read_masks(ops):
    mm_path, file_path = get_memmap_path(ops, 'masks.h5')
    with h5py.File(file_path, 'r') as f:
        labels     = create_memmap(f['labels'],     'int8',    os.path.join(mm_path, 'labels.mmap'))
        masks      = create_memmap(f['masks_func'], 'float32', os.path.join(mm_path, 'masks_func.mmap'))
        mean_func  = create_memmap(f['mean_func'],  'float32', os.path.join(mm_path, 'mean_func.mmap'))
        max_func   = create_memmap(f['max_func'],   'float32', os.path.join(mm_path, 'max_func.mmap'))
        mean_anat  = create_memmap(f['mean_anat'],  'float32', os.path.join(mm_path, 'mean_anat.mmap')) if ops['nchannels'] == 2 else None
        masks_anat = create_memmap(f['masks_anat'], 'float32', os.path.join(mm_path, 'masks_anat.mmap')) if ops['nchannels'] == 2 else None
    return [labels, masks, mean_func, max_func, mean_anat, masks_anat]

def visualize_calcium_signal(image, reference_image=None, method='percentile', channel='red', **kwargs):
    """
    Visualizes single-channel calcium imaging data in red shades using various methods,
    with optional reference image for consistent scaling.
    
    Parameters:
    - image (numpy.ndarray): Single-channel input image
    - reference_image (numpy.ndarray, optional): Reference image for consistent scaling
    - method (str): Visualization method ('percentile', 'zscore', 'minmax', or 'adaptive')
    - kwargs: Additional parameters for specific methods
        - percentile_low, percentile_high: For 'percentile' method (default: 1, 99)
    
    Returns:
    - numpy.ndarray: RGB image with red channel visualization
    """
    # Ensure input is float
    img = image.astype(float)
    
    if method == 'percentile':
        p_low = kwargs.get('percentile_low', 1)
        p_high = kwargs.get('percentile_high', 99)
        
        # If reference image is provided, use its statistics for normalization
        if reference_image is not None:
            ref_img = reference_image.astype(float)
            low = np.percentile(ref_img, p_low)
            high = np.percentile(ref_img, p_high)
        else:
            low = np.percentile(img, p_low)
            high = np.percentile(img, p_high)
            
        # Apply normalization using reference statistics
        img = np.clip((img - low) / (high - low + 1e-9), 0, 1)
    
    # Create RGB image (rest of your existing code)
    rgb_img = np.zeros((*img.shape, 3))
    if channel == 'red':
        rgb_img[..., 0] = img
    elif channel == 'green':
        rgb_img[..., 1] = img
    
    return rgb_img

from skimage.segmentation import find_boundaries

import numpy as np

def get_labeled_masks_img(masks, roi_labels, cate):
    """
    Generates an RGB image highlighting the masks that belong to a specific category.

    Parameters:
    -----------
    masks : np.ndarray
        2D labeled mask of ROIs. 0 = background, and each ROI has a unique integer ID.
    roi_labels : np.ndarray
        1D array of labels corresponding to roi_ids, where label ∈ {-1, 0, 1}.
        -1 indicates excitatory, 0 unsure, 1 inhibitory.
    cate : int
        Category of interest. Must be one of {1, -1, 0}.
         -  1 → inhibitory
         - -1 → excitatory
         -  0 → unsure

    Returns:
    --------
    labeled_masks_img : np.ndarray
        3D image (height × width × 3), where pixels belonging to the specified `cate`
        are highlighted (by default, in the red channel).
    """
    # Create an empty RGB image (int32 to store 0..255 values if desired)
    labeled_masks_img = np.zeros(
        (masks.shape[0], masks.shape[1], 3), dtype='int32'
    )

    # Iterate over ROI IDs and their labels
    for roi_id, roi_label in enumerate(roi_labels):
        if roi_id == 0:
            # Skip the background ROI (id 0)
            continue
        if roi_label == cate:
            # Create a binary mask for this ROI
            roi_binary_mask = (masks == roi_id).astype('int32')

            # Convert to intensity 255 where ROI is present
            roi_binary_mask *= 255

            # Add it to the RED channel of our output image
            labeled_masks_img[:, :, 0] += roi_binary_mask

    return labeled_masks_img

def create_superimposed_mask_images(mean_func, max_func, masks, labels, mean_anat):
    """
    Generate superimposed mask images for visualization of functional and anatomical channels.

    Parameters
    ----------
    mean_func : np.ndarray
        Mean functional channel image (2D array).
    max_func : np.ndarray
        Max functional channel image (2D array).
    masks : np.ndarray
        2D labeled mask of ROIs. 0 = background, and each ROI has a unique integer ID.
    labels : np.ndarray
        1D array of labels corresponding to roi_ids, where label ∈ {-1, 0, 1}.
        -1 indicates excitatory, 0 unsure, 1 inhibitory.
    mean_anat : np.ndarray or None, optional
        Mean anatomical channel image (2D array), or None if not available.

    Returns
    -------
    mean_fun_channel : np.ndarray
        Visualized mean functional channel (RGB image).
    max_fun_channel : np.ndarray
        Visualized max functional channel (RGB image).
    superimpose_mask_func : np.ndarray
        RGB image of mean functional channel with superimposed ROI boundaries.
    superimpose_mask_anat : np.ndarray or None
        RGB image of mean anatomical channel with superimposed ROI boundaries, or None if mean_anat is None.
    """
    # Visualize the mean and max functional channels
    mean_fun_channel = visualize_calcium_signal(mean_func, method='percentile', channel='green')
    max_fun_channel = visualize_calcium_signal(max_func, method='percentile', channel='green')

    # Get ROI masks for each neuron type
    inhibitory_masks_img = get_labeled_masks_img(masks, labels, 1)
    excitatory_masks_img = get_labeled_masks_img(masks, labels, -1)
    unsure_masks_img = get_labeled_masks_img(masks, labels, 0)

    # Find boundaries for each neuron type
    boundaries_inhibitory = find_boundaries(inhibitory_masks_img[..., 0], mode='outer')
    boundaries_excitatory = find_boundaries(excitatory_masks_img[..., 0], mode='thick')
    boundaries_unsure = find_boundaries(unsure_masks_img[..., 0], mode='outer')

    # Create RGB images for boundaries
    boundary_image_inh = np.zeros_like(inhibitory_masks_img)
    boundary_image_inh[boundaries_inhibitory] = [1, 0, 1]  # Magenta for inhibitory

    boundary_image_exc = np.zeros_like(excitatory_masks_img)
    boundary_image_exc[boundaries_excitatory] = [0, 0, 1]  # Blue for excitatory

    boundary_image_uns = np.zeros_like(unsure_masks_img)
    boundary_image_uns[boundaries_unsure] = [1, 1, 1]      # White for unsure

    # Combine boundary images with the mean functional channel
    superimpose_mask_func = boundary_image_inh + boundary_image_exc + boundary_image_uns + mean_fun_channel

    # If anatomical channel is provided, visualize and superimpose boundaries
    if mean_anat is not None:
        mean_anat_channel = visualize_calcium_signal(mean_anat, method='percentile', channel='red')
        superimpose_mask_anat = mean_anat_channel + boundary_image_inh + boundary_image_exc + boundary_image_uns
    else:
        superimpose_mask_anat = None

    return mean_fun_channel, max_fun_channel, superimpose_mask_func, superimpose_mask_anat

def _plot_fov_summary_single(
    mean_fun_channel,
    max_fun_channel,
    masks,
    superimpose_mask_func,
    mean_anat,
    superimpose_mask_anat,
    save_path=None,
    session_title=None,
    roi_count=None,
    show=True
):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gsp
    import numpy as np

    fig = plt.figure(figsize=(45, 30))
    gs = gsp.GridSpec(2, 4)

    def _imshow_safe(ax, img):
        arr = np.asarray(img)
        # Avoid matplotlib RGB clipping warnings by normalizing RGB float arrays.
        if arr.ndim == 3 and arr.shape[-1] in (3, 4):
            if np.issubdtype(arr.dtype, np.floating):
                arr = np.clip(arr, 0.0, 1.0)
            else:
                arr = np.clip(arr, 0, 255)
        ax.imshow(arr)

    title_parts = []
    if session_title is not None:
        title_parts.append(str(session_title))
    if roi_count is not None:
        title_parts.append(f'{int(roi_count)} identified ROIs')
    if title_parts:
        fig.suptitle(' | '.join(title_parts), fontsize=28, y=0.98)

    # first row for functional channel
    ax = plt.subplot(gs[0, 0])
    _imshow_safe(ax, mean_fun_channel)
    plt.title("Mean Functional Channel (green colorized)")

    ax = plt.subplot(gs[0, 1])
    _imshow_safe(ax, max_fun_channel)
    plt.title("Max Functional Channel (green colorized)")

    ax = plt.subplot(gs[0, 2])
    _imshow_safe(ax, masks)
    plt.title("Masks")

    ax = plt.subplot(gs[0, 3])
    _imshow_safe(ax, superimpose_mask_func)
    plt.title("Superimposed Mask + Mean Func\nInhibitory (magenta), Excitatory (blue), Unsure (white)")

    # second row for anatomical channel
    if (mean_anat is not None) and (superimpose_mask_anat is not None):
        ax = plt.subplot(gs[1, 0])
        _imshow_safe(ax, mean_anat)
        plt.title("Mean Anatomical Channel (red colorized)")

        ax = plt.subplot(gs[1, 3])
        _imshow_safe(ax, superimpose_mask_anat)
        plt.title("Superimposed Mask + Mean Anat\nInhibitory (magenta), Excitatory (blue), Unsure (white)")

    if save_path is not None:
        figures_dir = os.path.join(save_path, 'Figures', 'fov')
        if not os.path.exists(figures_dir):
            os.makedirs(figures_dir)
        save_path = os.path.join(figures_dir, 'FOV_summary.pdf')
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
        return None
    return fig


def plot_fov_summary(
    mean_fun_channel,
    max_fun_channel,
    masks,
    superimpose_mask_func,
    mean_anat,
    superimpose_mask_anat,
    save_path=None,
    session_names=None,
    use_slider=False,
    session_title=None,
    navigator="slider",
    roi_counts=None
):
    """
    Plot FOV summary for either one session (existing behavior) or multiple sessions
    with an interactive slider.

    Multi-session mode is enabled when list/tuple inputs are passed (or `use_slider=True`).
    In multi-session mode, each argument should be a list with matching length.
    """
    is_sequence = isinstance(mean_fun_channel, (list, tuple))
    if not is_sequence and not use_slider:
        inferred_title = session_title
        if inferred_title is None and isinstance(save_path, str):
            inferred_title = os.path.basename(os.path.normpath(save_path))
        _plot_fov_summary_single(
            mean_fun_channel, max_fun_channel, masks, superimpose_mask_func,
            mean_anat, superimpose_mask_anat, save_path=save_path,
            session_title=inferred_title,
            roi_count=roi_counts
        )
        return

    # Normalize to lists for multi-session mode.
    mean_fun_list = list(mean_fun_channel) if isinstance(mean_fun_channel, (list, tuple)) else [mean_fun_channel]
    max_fun_list = list(max_fun_channel) if isinstance(max_fun_channel, (list, tuple)) else [max_fun_channel]
    masks_list = list(masks) if isinstance(masks, (list, tuple)) else [masks]
    sup_func_list = list(superimpose_mask_func) if isinstance(superimpose_mask_func, (list, tuple)) else [superimpose_mask_func]
    mean_anat_list = list(mean_anat) if isinstance(mean_anat, (list, tuple)) else [mean_anat]
    sup_anat_list = list(superimpose_mask_anat) if isinstance(superimpose_mask_anat, (list, tuple)) else [superimpose_mask_anat]

    n_sessions = len(mean_fun_list)
    lengths = [len(max_fun_list), len(masks_list), len(sup_func_list), len(mean_anat_list), len(sup_anat_list)]
    if any(length != n_sessions for length in lengths):
        raise ValueError("All multi-session inputs must have the same length.")

    if session_names is None:
        session_names = [f"Session {i}" for i in range(n_sessions)]
    elif len(session_names) != n_sessions:
        raise ValueError("session_names length must match number of sessions.")

    if roi_counts is None:
        roi_count_list = [None] * n_sessions
    elif isinstance(roi_counts, (list, tuple)):
        if len(roi_counts) != n_sessions:
            raise ValueError("roi_counts length must match number of sessions.")
        roi_count_list = list(roi_counts)
    else:
        roi_count_list = [roi_counts] * n_sessions

    save_paths = None
    if isinstance(save_path, (list, tuple)):
        if len(save_path) != n_sessions:
            raise ValueError("save_path list length must match number of sessions.")
        save_paths = list(save_path)

    try:
        import ipywidgets as widgets
        from IPython.display import display
    except Exception as exc:
        raise ImportError(
            "ipywidgets is required for slider mode. Install it or call with single-session inputs."
        ) from exc

    slider = widgets.IntSlider(
        value=0, min=0, max=n_sessions - 1, step=1, description="Session"
    )
    prev_button = widgets.Button(description="◀ Prev")
    next_button = widgets.Button(description="Next ▶")
    title_html = widgets.HTML()
    image_widget = widgets.Image(format="png")

    def _render(idx):
        idx = int(idx)
        title_html.value = f"<b>{session_names[idx]}</b> (index {idx})"
        import io
        import matplotlib.pyplot as plt

        with plt.ioff():
            fig = _plot_fov_summary_single(
                mean_fun_list[idx], max_fun_list[idx], masks_list[idx], sup_func_list[idx],
                mean_anat_list[idx], sup_anat_list[idx],
                save_path=(save_paths[idx] if save_paths is not None else None),
                session_title=session_names[idx],
                roi_count=roi_count_list[idx],
                show=False
            )
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        image_widget.value = buf.getvalue()

    slider.observe(lambda change: _render(change["new"]), names="value")

    def _prev(_):
        slider.value = max(0, slider.value - 1)

    def _next(_):
        slider.value = min(n_sessions - 1, slider.value + 1)

    prev_button.on_click(_prev)
    next_button.on_click(_next)

    display(title_html)
    if navigator == "arrows":
        controls = widgets.HBox([prev_button, next_button])
    else:
        controls = widgets.VBox([slider, widgets.HBox([prev_button, next_button])])
    display(controls)
    _render(0)
    display(image_widget)

# read raw_voltages.h5.
def read_raw_voltages(ops, load_hifi=False):
    mm_path, file_path = get_memmap_path(ops, 'raw_voltages.h5')
    with h5py.File(file_path, 'r') as f:
        vol_time     = create_memmap(f['raw']['vol_time'],     'float32', os.path.join(mm_path, 'vol_time.mmap'))
        vol_start    = create_memmap(f['raw']['vol_start'],    'int8',    os.path.join(mm_path, 'vol_start.mmap'))
        vol_stim_vis = create_memmap(f['raw']['vol_stim_vis'], 'int8',    os.path.join(mm_path, 'vol_stim_vis.mmap'))
        if load_hifi:
            vol_hifi = create_memmap(f['raw']['vol_hifi'], 'int8', os.path.join(mm_path, 'vol_hifi.mmap'))
        else:
            vol_hifi = np.zeros(f['raw']['vol_start'].shape, dtype='int8')
        vol_img      = create_memmap(f['raw']['vol_img'],      'int8',    os.path.join(mm_path, 'vol_img.mmap'))
        vol_stim_aud = create_memmap(f['raw']['vol_stim_aud'], 'float32', os.path.join(mm_path, 'vol_stim_aud.mmap'))
        vol_flir     = create_memmap(f['raw']['vol_flir'],     'int8',    os.path.join(mm_path, 'vol_flir.mmap'))
        vol_pmt      = create_memmap(f['raw']['vol_pmt'],      'int8',    os.path.join(mm_path, 'vol_pmt.mmap'))
        vol_led      = create_memmap(f['raw']['vol_led'],      'int8',    os.path.join(mm_path, 'vol_led.mmap'))
    return [vol_time, vol_start, vol_stim_vis, vol_img,
            vol_hifi, vol_stim_aud, vol_flir,
            vol_pmt, vol_led]

# read dff traces.
def read_dff(ops):
    qc_file_path = os.path.join(ops['save_path0'], 'qc_results', 'dff.h5')
    root_file_path = os.path.join(ops['save_path0'], 'dff.h5')
    if os.path.exists(qc_file_path):
        mm_path, file_path = get_memmap_path(ops, 'dff.h5', subdir='qc_results')
    elif os.path.exists(root_file_path):
        mm_path, file_path = get_memmap_path(ops, 'dff.h5')
    else:
        raise FileNotFoundError(
            f'Could not find dff.h5 in either {qc_file_path} or {root_file_path}')
    with h5py.File(file_path, 'r') as f:
        dff = create_memmap(f['dff'], 'float32', os.path.join(mm_path, 'dff.mmap'))
    return dff

import scipy.io as sio
import pandas as pd

def get_bpod_mat_path(ops):
    local_path = os.path.join(ops['save_path0'], 'bpod_session_data.mat')
    if os.path.exists(local_path):
        return local_path
    session_name = os.path.basename(ops['save_path0'])
    mouse_name = session_name.split('_')[0]
    date_match = re.search(r'_(\d{8})_', session_name)
    if date_match is None:
        raise FileNotFoundError(f'Could not parse session date from {session_name}.')
    session_date = date_match.group(1)
    shared_dir = os.path.join(
        '/storage/project/r-fnajafi3-0/shared/single_interval_discrimination/session_data',
        mouse_name)
    matches = sorted(glob.glob(os.path.join(shared_dir, f'*{session_date}*.mat')))
    if len(matches) == 0:
        raise FileNotFoundError(
            f'Could not find bpod_session_data.mat for {session_name} in {shared_dir}.')
    return matches[0]

# read bpod session data.
def read_bpod_mat_data(ops, session_start_time):
    def _check_keys(d):
        for key in d:
            if isinstance(d[key], sio.matlab.mat_struct):
                d[key] = _todict(d[key])
        return d
    def _todict(matobj):
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
        elem_list = []
        for sub_elem in ndarray:
            if isinstance(sub_elem, sio.matlab.mat_struct):
                elem_list.append(_todict(sub_elem))
            elif isinstance(sub_elem, np.ndarray):
                elem_list.append(_tolist(sub_elem))
            else:
                elem_list.append(sub_elem)
        return elem_list
    # labeling every trials for a subject
    def states_labeling(trial_states):
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
    # find state timing.
    def get_state(trial_state_dict, target_state, trial_start):
        if target_state in trial_state_dict:
            time_state = 1000*np.array(trial_state_dict[target_state]) + trial_start
        else:
            time_state = np.array([np.nan, np.nan])
        return time_state
    # read raw data.
    raw = sio.loadmat(
        get_bpod_mat_path(ops),
        struct_as_record=False, squeeze_me=True)
    raw = _check_keys(raw)['SessionData']
    trial_labels = dict()
    n_trials = raw['nTrials']
    trial_states = [raw['RawEvents']['Trial'][ti]['States'] for ti in range(n_trials)]
    trial_events = [raw['RawEvents']['Trial'][ti]['Events'] for ti in range(n_trials)]
    # trial start time stamps.
    trial_labels['time_trial_start'] = 1000*np.array(raw['TrialStartTimestamp']).reshape(-1)
    # trial end time stamps.
    trial_labels['time_trial_end'] = 1000*np.array(raw['TrialEndTimestamp']).reshape(-1)
    # correct timestamps starting from session start.
    trial_labels['time_trial_end'] = trial_labels['time_trial_end'] - trial_labels['time_trial_start'][0] + session_start_time
    trial_labels['time_trial_start'] = trial_labels['time_trial_start'] - trial_labels['time_trial_start'][0] + session_start_time
    # trial target.
    trial_labels['trial_type'] = np.array(raw['TrialTypes']).reshape(-1)-1
    if 'BlockTypes' in raw:
        trial_labels['block_type'] = np.array(raw['BlockTypes']).reshape(-1)
    else:
        trial_labels['block_type'] = np.zeros(n_trials, dtype='int32')
    # trial outcomes.
    trial_labels['outcome'] = np.array([states_labeling(ts) for ts in trial_states], dtype='object')
    # trial state timings.
    trial_labels['state_window_choice'] = np.array([
        get_state(trial_states[ti], 'WindowChoice', trial_labels['time_trial_start'][ti])
        for ti in range(n_trials)] + ['yicong_forever'], dtype='object')[:-1]
    trial_labels['state_reward'] = np.array([
        get_state(trial_states[ti], 'Reward', trial_labels['time_trial_start'][ti])
        for ti in range(n_trials)] + ['yicong_forever'], dtype='object')[:-1]
    trial_labels['state_punish'] = np.array([
        get_state(trial_states[ti], 'Punish', trial_labels['time_trial_start'][ti])
        for ti in range(n_trials)] + ['yicong_forever'], dtype='object')[:-1]
    # stimulus timing.
    trial_isi = []
    trial_stim_seq = []
    for ti in range(n_trials):
        # stimulus sequence.
        if ('BNC1High' in trial_events[ti].keys() and
            'BNC1Low' in trial_events[ti].keys() and
            len(np.array(trial_events[ti]['BNC1High']).reshape(-1))==2 and
            len(np.array(trial_events[ti]['BNC1Low']).reshape(-1))==2
            ):
            stim_seq = 1000*np.array([trial_events[ti]['BNC1High'], trial_events[ti]['BNC1Low']]) + trial_labels['time_trial_start'][ti]
            stim_seq = np.transpose(stim_seq, [1,0])
            isi = 1000*np.array(trial_events[ti]['BNC1High'][1] - trial_events[ti]['BNC1Low'][0])
        else:
            stim_seq = np.array([[np.nan, np.nan], [np.nan, np.nan]])
            isi = np.nan
        trial_stim_seq.append(stim_seq)
        trial_isi.append(isi)
    trial_labels['stim_seq'] = np.array(trial_stim_seq + ['yicong_forever'], dtype='object')[:-1]
    trial_labels['isi'] = np.array(trial_isi + ['yicong_forever'], dtype='object')[:-1]
    # licking.
    trial_lick = []
    for ti in range(n_trials):
        licking_events = []
        # 0 left 1 right.
        direction = []
        # 0 wrong 1 correct.
        correctness = []
        # left.
        if 'Port1In' in trial_events[ti].keys():
            lick_left = np.array(trial_events[ti]['Port1In']).reshape(-1)
            licking_events.append(lick_left)
            direction.append(np.zeros_like(lick_left))
            if trial_labels['trial_type'][ti] == 0:
                correctness.append(np.ones_like(lick_left))
            else:
                correctness.append(np.zeros_like(lick_left))
        # right.
        if 'Port3In' in trial_events[ti].keys():
            lick_right = np.array(trial_events[ti]['Port3In']).reshape(-1)
            licking_events.append(lick_right)
            direction.append(np.ones_like(lick_right))
            if trial_labels['trial_type'][ti] == 1:
                correctness.append(np.ones_like(lick_right))
            else:
                correctness.append(np.zeros_like(lick_right))
        if len(licking_events) > 0:
            # combine all licking.
            licking_events = 1000*np.concatenate(licking_events).reshape(1,-1) + trial_labels['time_trial_start'][ti]
            direction = np.concatenate(direction).reshape(1,-1)
            correctness = np.concatenate(correctness).reshape(1,-1)
            lick = np.concatenate([licking_events, direction, correctness], axis=0)
            # sort based on timing.
            lick = lick[:,np.argsort(lick[0,:])]
            # filter false detection before licking window.
            lick = lick[:,lick[0,:] >= trial_labels['state_window_choice'][ti][0]]
            # classify licking.
            if np.size(lick) != 0:
                lick_type = np.full(lick.shape[1], np.nan)
                lick_type[0] = 1
                if (not np.isnan(trial_labels['state_reward'][ti][1]) and
                    len(lick_type) > 1
                    ):
                    lick_type[1:][lick[0,1:] > trial_labels['state_reward'][ti][0]] = 0
                lick_type = lick_type.reshape(1,-1)
                lick = np.concatenate([lick, lick_type], axis=0)
            else:
                lick = np.array([[np.nan], [np.nan], [np.nan], [np.nan]])
        else:
            lick = np.array([[np.nan], [np.nan], [np.nan], [np.nan]])
        # all licking events.
        trial_lick.append(lick)
    trial_labels['lick'] = np.array(trial_lick + ['yicong_forever'], dtype='object')[:-1]
    # convert to dataframe.
    trial_labels = pd.DataFrame(trial_labels)
    return trial_labels

# remove trial start trigger voltage impulse.
def remove_start_impulse(vol_time, vol_stim_vis):
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

# correct beginning vol_stim_vis if not start from 0.
def correct_vol_start(vol_stim_vis):
    if vol_stim_vis[0] == 1:
        vol_stim_vis[:np.where(vol_stim_vis==0)[0][0]] = 0
    return vol_stim_vis

# detect the rising edge and falling edge of binary series.
def get_trigger_time(
        vol_time,
        vol_bin
        ):
    # find the edge with np.diff and correct it by preappend one 0.
    diff_vol = np.diff(vol_bin, prepend=0)
    idx_up = np.where(diff_vol == 1)[0]
    idx_down = np.where(diff_vol == -1)[0]
    # select the indice for risging and falling.
    # give the edges in ms.
    time_up   = vol_time[idx_up]
    time_down = vol_time[idx_down]
    return time_up, time_down

def infer_trial_starts_from_stim_pairs(ops, vol_time, vol_stim_vis):
    stim_up, _ = get_trigger_time(vol_time, vol_stim_vis)
    if len(stim_up) < 2:
        raise ValueError('Not enough vol_stim_vis pulses to infer trial starts.')
    raw = sio.loadmat(
        get_bpod_mat_path(ops),
        struct_as_record=False, squeeze_me=True)['SessionData']
    bpod_first_offsets = []
    bpod_isis = []
    for trial in np.array(raw.RawEvents.Trial).reshape(-1):
        if not hasattr(trial.Events, 'BNC1High'):
            continue
        pulse_times = 1000*np.array(trial.Events.BNC1High).reshape(-1)
        if len(pulse_times) < 2:
            continue
        bpod_first_offsets.append(pulse_times[0])
        bpod_isis.append(pulse_times[1] - pulse_times[0])
    if len(bpod_isis) == 0:
        raise ValueError('No usable BNC1High pulse pairs found in bpod_session_data.mat.')
    bpod_first_offsets = np.array(bpod_first_offsets)
    bpod_isis = np.array(bpod_isis)
    candidates = []
    for start_idx in [0, 1]:
        if len(stim_up[start_idx:]) < 2:
            continue
        stim_first = stim_up[start_idx::2]
        stim_second = stim_up[start_idx+1::2]
        n_pairs = min(len(stim_first), len(stim_second), len(bpod_isis))
        if n_pairs == 0:
            continue
        stim_first = stim_first[:n_pairs]
        stim_second = stim_second[:n_pairs]
        stim_isi = stim_second - stim_first
        score = np.median(np.abs(stim_isi - bpod_isis[:n_pairs]))
        inferred_starts = stim_first - bpod_first_offsets[:n_pairs]
        candidates.append((score, start_idx, inferred_starts))
    if len(candidates) == 0:
        raise ValueError('Unable to pair vol_stim_vis pulses into trial starts.')
    _, _, inferred_starts = min(candidates, key=lambda x: x[0])
    return inferred_starts

def get_trial_start_times(vol_time, vol_start, ops=None, vol_stim_vis=None):
    if FORCE_VIS_STIM_PAIR_TRIAL_START:
        if ops is None or vol_stim_vis is None:
            raise ValueError('ops and vol_stim_vis are required when forcing vis_stim trial starts.')
        return infer_trial_starts_from_stim_pairs(ops, vol_time, vol_stim_vis)
    time_up, _ = get_trigger_time(vol_time, vol_start)
    if len(time_up) > 0:
        return time_up
    if USE_VIS_STIM_PAIR_TRIAL_START_FALLBACK:
        if ops is None or vol_stim_vis is None:
            raise ValueError('ops and vol_stim_vis are required for vis_stim trial-start fallback.')
        return infer_trial_starts_from_stim_pairs(ops, vol_time, vol_stim_vis)
    raise ValueError('No valid vol_start rising edge found for this session.')

# find when bpod session timer start.
def get_session_start_time(vol_time, vol_start, ops=None, vol_stim_vis=None):
    return get_trial_start_times(vol_time, vol_start, ops, vol_stim_vis)[0]

# correct the fluorescence signal timing.
def correct_time_img_center(time_img):
    # find the frame internal.
    diff_time_img = np.diff(time_img, append=0)
    # correct the last element.
    diff_time_img[-1] = np.mean(diff_time_img[:-1])
    # move the image timing to the center of photon integration interval.
    diff_time_img = diff_time_img / 2
    # correct each individual timing.
    time_neuro = time_img + diff_time_img
    return time_neuro

# save trial neural data.
def save_trials(
        ops, time_neuro, dff, trial_labels,
        vol_time, vol_stim_vis,
        vol_stim_aud, vol_flir,
        vol_pmt, vol_led
        ):
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

from scipy.signal import savgol_filter

# read trial label csv file into dataframe.
def read_trial_label(ops):
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
    block_type = raw_csv['block_type'].to_numpy(dtype='int8')
    outcome = raw_csv['outcome'].to_numpy(dtype='object')
    state_window_choice = object_parse('state_window_choice', [-1])
    state_reward = object_parse('state_reward', [-1])
    state_punish = object_parse('state_punish', [-1])
    stim_seq = object_parse('stim_seq', [-1,2])
    isi = raw_csv['isi'].to_numpy(dtype='float32')
    lick = object_parse('lick', [4,-1])
    # convert to dataframe.
    trial_labels = pd.DataFrame({
        'time_trial_start': time_trial_start,
        'time_trial_end': time_trial_end,
        'trial_type': trial_type,
        'block_type': block_type,
        'outcome': outcome,
        'state_window_choice': state_window_choice,
        'state_reward': state_reward,
        'state_punish': state_punish,
        'stim_seq': stim_seq,
        'isi': isi,
        'lick': lick,
        })
    return trial_labels

# read trailized neural traces with stimulus alignment.
def read_neural_trials(ops, smooth):
    mm_path, file_path = get_memmap_path(ops, 'neural_trials.h5')
    trial_labels = read_trial_label(ops)
    with h5py.File(file_path, 'r') as f:
        neural_trials = dict()
        dff = np.array(f['neural_trials']['dff'])
        if smooth:
            window_length=9
            polyorder=3
            dff = np.apply_along_axis(
                savgol_filter, 1, dff,
                window_length=window_length,
                polyorder=polyorder)
        else: pass
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

# Session path helpers migrated from Test_pilot/test_nb_session_paths.py
import os
import numpy as np

def get_mouse_name(session_name):
    return session_name.split('_')[0]

def get_session_output_dir(session_name, output_root):
    mouse_name = get_mouse_name(session_name)
    output_dir = os.path.join(output_root, mouse_name, session_name)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'Figures'), exist_ok=True)
    return output_dir

def get_session_output_dirs(list_session_names, output_root):
    return [get_session_output_dir(session_name, output_root) for session_name in list_session_names]

def read_ops(list_session_data_path):
    list_ops = []
    for session_data_path in list_session_data_path:
        ops = np.load(
            os.path.join(session_data_path, 'ops.npy'),
            allow_pickle=True).item()
        ops['save_path0'] = os.path.join(session_data_path)
        list_ops.append(ops)
    return list_ops

def resolve_session_path(
    session_name,
    environment,
    local_data_root,
    pace_cedar_data_root,
    pace_cedar_yh24_processed_root,
    pace_scratch_mc11_root,
):
    if environment == 'PACE':
        if session_name.startswith('MC11_'):
            return os.path.join(pace_scratch_mc11_root, session_name)
        mouse_name = session_name.split('_')[0]
        if session_name.startswith('YH24LG_'):
            return os.path.join(pace_cedar_yh24_processed_root, mouse_name, session_name)
        return os.path.join(pace_cedar_data_root, mouse_name, session_name)
    return os.path.join(local_data_root, session_name)

def zscore_normalize(data, axis=1, min_std=1e-8):
    """Perform z-score normalization while avoiding divide-by-zero."""
    data = np.asarray(data)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    mean = np.mean(data, axis=axis, keepdims=True)
    std = np.std(data, axis=axis, keepdims=True)
    std = np.maximum(std, min_std)
    return (data - mean) / std
