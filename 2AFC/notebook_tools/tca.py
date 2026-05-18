"""TCA/block-transition helpers extracted from `test_TCA_block_transition.ipynb`."""

import numpy as np


def process_transitions(
    transition_indices,
    num_before,
    num_after,
    total_trials,
    time_trial_start,
    time_trial_end,
    time_all,
    dff,
    num_neurons,
):
    """Extract fixed-width, transition-centered dF/F segments across block transitions."""
    segments = []
    trans_indices = []
    lengths = []
    trial_bounds = []

    for ti in transition_indices:
        intended_start = ti - (num_before - 1)
        intended_end = ti + num_after
        if intended_start < 0 or intended_end >= total_trials:
            continue

        start_trial = intended_start
        end_trial = intended_end

        t_start = int(np.searchsorted(time_all, time_trial_start[start_trial].flatten()[0]))
        t_end = int(np.searchsorted(time_all, time_trial_end[end_trial].flatten()[0]))

        segment_time = time_all[t_start:t_end]
        dff_seg = dff[:, t_start:t_end]

        t_trans = time_trial_start[ti + 1].flatten()[0]
        trans_idx = int(np.searchsorted(segment_time, t_trans))

        bounds = []
        for tr in range(start_trial, end_trial + 1):
            ts = time_trial_start[tr].flatten()[0]
            te = time_trial_end[tr].flatten()[0]
            idx_s = int(np.searchsorted(segment_time, ts))
            idx_e = int(np.searchsorted(segment_time, te))
            bounds.append((idx_s, idx_e))

        segments.append(dff_seg)
        trans_indices.append(trans_idx)
        lengths.append(dff_seg.shape[1])
        trial_bounds.append(bounds)

    if not segments:
        return None, None, None, None

    min_pre = min(trans_indices)
    min_post = min(lengths[m] - trans_indices[m] for m in range(len(segments)))
    cropped_len = min_pre + min_post
    cropped_dff = np.empty((len(segments), num_neurons, cropped_len))
    cropped_bounds = []
    cropped_trans_indices = []

    for m, dff_seg in enumerate(segments):
        trans_idx = trans_indices[m]
        if trans_idx < min_pre or (lengths[m] - trans_idx) < min_post:
            continue

        crop_start = trans_idx - min_pre
        crop_end = trans_idx + min_post
        cropped_dff[m] = dff_seg[:, crop_start:crop_end]
        crop_bounds = [(s - crop_start, e - crop_start) for s, e in trial_bounds[m]]
        cropped_bounds.append(crop_bounds)
        cropped_trans_indices.append(trans_idx - crop_start)

    return cropped_dff, cropped_bounds, cropped_len, cropped_trans_indices

