#!/usr/bin/env python3

import gc
import numpy as np
from tqdm import tqdm

# cut sequence into the same length as the shortest one given pivots.
def trim_seq(
        data,
        pivots,
        ):
    if len(data[0].shape) == 1:
        len_l_min = np.min(pivots)
        len_r_min = np.min([len(data[i])-pivots[i] for i in range(len(data))])
        data = [data[i][pivots[i]-len_l_min:pivots[i]+len_r_min]
                for i in range(len(data))]
    if len(data[0].shape) == 3:
        len_l_min = np.min(pivots)
        len_r_min = np.min([len(data[i][0,0,:])-pivots[i] for i in range(len(data))])
        data = [data[i][:, :, pivots[i]-len_l_min:pivots[i]+len_r_min]
                for i in range(len(data))]
    return data

def get_lick_response(
    neural_trials,
    l_frames, r_frames
    ):
    # initialization.
    time = neural_trials['time']
    neu_seq    = []
    neu_time   = []
    direction  = []
    correction = []
    lick_type  = []
    # get all licking events.
    lick = np.concatenate(neural_trials['trial_labels']['lick'].to_numpy(), axis=1)
    # loop over licks.
    for li in tqdm(range(lick.shape[1])):
        t = lick[0,li]
        if not np.isnan(t):
            # get state start timing.
            idx = np.searchsorted(neural_trials['time'], t)
            if idx > l_frames and idx < len(neural_trials['time'])-r_frames:
                # signal response.
                f = neural_trials['dff'][:, idx-l_frames : idx+r_frames]
                f = np.expand_dims(f, axis=0)
                neu_seq.append(f)
                # signal time stamps.
                neu_time.append(neural_trials['time'][idx-l_frames : idx+r_frames] - time[idx])
                # licking properties.
                direction.append(lick[1,li])
                correction.append(lick[2,li])
                lick_type.append(lick[3,li])
    # correct neural data centering at zero.
    neu_time_zero = [np.argmin(np.abs(nt)) for nt in neu_time]
    neu_time = trim_seq(neu_time, neu_time_zero)
    neu_seq = trim_seq(neu_seq, neu_time_zero)
    # concatenate results.
    neu_seq    = np.concatenate(neu_seq, axis=0)
    neu_time   = [nt.reshape(1,-1) for nt in neu_time]
    neu_time   = np.concatenate(neu_time, axis=0)
    direction  = np.array(direction)
    correction = np.array(correction)
    lick_type  = np.array(lick_type)
    # get mean time stamps.
    neu_time = np.mean(neu_time, axis=0)
    # combine results.
    return [neu_seq, neu_time, direction, correction, lick_type]

# extract response around stimulus.
def get_perception_response(
        neural_trials, target_state,
        l_frames, r_frames,
        ):
    exclude_start_trials = 25
    exclude_end_trials = 20
    # initialization.
    time = neural_trials['time']
    neu_seq    = []
    neu_time   = []
    stim_seq   = []
    stim_value = []
    stim_time  = []
    led_value  = []
    trial_type = []
    isi        = []
    decision   = []
    outcome    = []
    # loop over trials.
    for ti in tqdm(range(len(neural_trials['trial_labels']))):
        t = neural_trials['trial_labels'][target_state][ti].flatten()[0]
        if (not np.isnan(t) and
            ti >= exclude_start_trials and
            ti < len(neural_trials['trial_labels'])-exclude_end_trials
            ):
            # get state start timing.
            idx = np.searchsorted(neural_trials['time'], t)
            if idx > l_frames and idx < len(neural_trials['time'])-r_frames:
                # signal response.
                f = neural_trials['dff'][:, idx-l_frames : idx+r_frames]
                f = np.expand_dims(f, axis=0)
                neu_seq.append(f)
                # signal time stamps.
                neu_time.append(neural_trials['time'][idx-l_frames : idx+r_frames] - time[idx])
                # voltage.
                vol_t_c = np.searchsorted(neural_trials['vol_time'], neural_trials['time'][idx])
                vol_t_l = np.searchsorted(neural_trials['vol_time'], neural_trials['time'][idx-l_frames])
                vol_t_r = np.searchsorted(neural_trials['vol_time'], neural_trials['time'][idx+r_frames])
                stim_time.append(neural_trials['vol_time'][vol_t_l:vol_t_r] - neural_trials['vol_time'][vol_t_c])
                stim_value.append(neural_trials['vol_stim_vis'][vol_t_l:vol_t_r])
                led_value.append(neural_trials['vol_led'][vol_t_l:vol_t_r])
                # task variables.
                stim_seq.append(neural_trials['trial_labels']['stim_seq'][ti].reshape(1,2,2) - t)
                trial_type.append(neural_trials['trial_labels']['trial_type'][ti])
                isi.append(neural_trials['trial_labels']['isi'][ti])
                decision.append(neural_trials['trial_labels']['lick'][ti][1,0])
                outcome.append(neural_trials['trial_labels']['outcome'][ti])
        else: pass
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
    neu_seq    = np.concatenate(neu_seq, axis=0)
    neu_time   = [nt.reshape(1,-1) for nt in neu_time]
    neu_time   = np.concatenate(neu_time, axis=0)
    stim_seq   = np.concatenate(stim_seq, axis=0)
    stim_value = [sv.reshape(1,-1) for sv in stim_value]
    stim_value = np.concatenate(stim_value, axis=0)
    stim_time  = [st.reshape(1,-1) for st in stim_time]
    stim_time  = np.concatenate(stim_time, axis=0)
    led_value  = [lv.reshape(1,-1) for lv in led_value]
    led_value  = np.concatenate(led_value, axis=0)
    trial_type = np.array(trial_type)
    isi        = np.array(isi)
    decision   = np.array(decision)
    outcome    = np.array(outcome)
    # get mean time stamps.
    neu_time  = np.mean(neu_time, axis=0)
    stim_time = np.mean(stim_time, axis=0)
    # combine results.
    return [neu_seq, neu_time, stim_seq, stim_value, stim_time, led_value, trial_type, isi, decision, outcome]

# run alignment for all sessions.
def run_get_perception_response(
        temp_folder,
        list_neural_trials, target_state,
        l_frames, r_frames,
        ):
    # run alignment for each session.
    list_neu_seq    = []
    list_neu_time   = []
    list_stim_seq   = []
    list_stim_value = []
    list_stim_time  = []
    list_led_value  = []
    list_trial_type = []
    list_isi        = []
    list_decision   = []
    list_outcome    = []
    for si in range(len(list_neural_trials)):
        [neu_seq, neu_time, stim_seq,
         stim_value, stim_time, led_value,
         trial_type, isi, decision, outcome] = get_perception_response(
             list_neural_trials[si], target_state,
             l_frames, r_frames)
        list_neu_seq.append(neu_seq)
        list_neu_time.append(neu_time)
        list_stim_seq.append(stim_seq)
        list_stim_value.append(stim_value)
        list_stim_time.append(stim_time)
        list_led_value.append(led_value)
        list_trial_type.append(trial_type)
        list_isi.append(isi)
        list_decision.append(decision)
        list_outcome.append(outcome)
        del neu_seq
        del neu_time
        del stim_seq
        del stim_value
        del stim_time
        del led_value
        del trial_type
        del isi
        del decision
        del outcome
        gc.collect()
    # combine neu_time.
    neu_time = np.nanmean(np.concatenate([nt.reshape(1,-1) for nt in list_neu_time]),axis=0)
    # combine stim_time.
    st_min  = int(np.nanmin(np.concatenate(list_stim_time)))
    st_max  = int(np.nanmax(np.concatenate(list_stim_time)))
    st_rate = np.nanmean(np.concatenate([np.diff(st) for st in list_stim_time]))
    stim_time = np.arange(st_min, st_max, st_rate)
    # interpolate stim_value on the common stim_time.
    list_stim_value = [
        np.apply_along_axis(
            lambda row: np.interp(stim_time, list_stim_time[i], row), 
            axis=1, 
            arr=list_stim_value[i])
        for i in range(len(list_stim_value))]
    # interpolate led_value on the common led_time.
    list_led_value = [
        np.apply_along_axis(
            lambda row: np.interp(stim_time, list_stim_time[i], row), 
            axis=1, 
            arr=list_led_value[i])
        for i in range(len(list_led_value))]
    # combine results.
    alignment = {
        'list_neu_seq': list_neu_seq,
        'neu_time': neu_time,
        'list_stim_seq': list_stim_seq,
        'list_stim_value': list_stim_value,
        'stim_time': stim_time,
        'list_led_value': list_led_value,
        'list_trial_type': list_trial_type,
        'list_isi': list_isi,
        'list_decision': list_decision,
        'list_outcome': list_outcome,
        }
    return alignment

# run alignment for all sessions.
def run_get_lick_response(
        temp_folder,
        list_neural_trials,
        l_frames, r_frames,
        ):
    # run alignment for each session.
    list_neu_seq    = []
    list_neu_time   = []
    list_direction  = []
    list_correction = []
    list_lick_type  = []
    for si in range(len(list_neural_trials)):
        [neu_seq, neu_time,
         direction, correction, lick_type] = get_lick_response(
             list_neural_trials[si],
             l_frames, r_frames)
        list_neu_seq.append(neu_seq)
        list_neu_time.append(neu_time)
        list_direction.append(direction)
        list_correction.append(correction)
        list_lick_type.append(lick_type)
        del neu_seq
        del neu_time
        gc.collect()
    # combine neu_time.
    neu_time = np.nanmean(np.concatenate([nt.reshape(1,-1) for nt in list_neu_time]),axis=0)
    # combine results.
    alignment = {
        'list_neu_seq': list_neu_seq,
        'neu_time': neu_time,
        'list_direction': list_direction,
        'list_correction': list_correction,
        'list_lick_type': list_lick_type,
        }
    return alignment

