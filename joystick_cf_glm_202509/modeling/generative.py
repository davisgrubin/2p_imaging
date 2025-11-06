#!/usr/bin/env python3

import numpy as np
from tqdm import tqdm
from scipy.interpolate import interp1d
from sklearn.metrics import explained_variance_score

from utils import get_frame_idx_from_time


#%% utils

# interpolate input trace to the same length as neural response.
def interp_factor_in(input_value, input_time, neu_time):
    model = interp1d(input_time, input_value, bounds_error=False, fill_value=0)
    return model(neu_time)

# linear convolution to obtain neural response from one factor and kernel.
def factor_dff_neu(factor_target, kernel, l_idx, r_idx):
    padded = np.pad(factor_target, (r_idx, l_idx-1), mode='constant')
    return np.convolve(padded, kernel, mode='valid')

# retrieve glm kernels for category.
def get_glm_cate(glm, list_labels, cate):
    idx = np.concatenate([np.in1d(list_labels[i],cate)
           for i in range(len(list_labels))])
    kernel_all = glm['kernel_all'][idx,:]
    return kernel_all

# set factor to 1 when state is presented.
def set_factor(time, trial_factor, state_time):
    if len(state_time) == 1:
        l_idx, _ = get_frame_idx_from_time(time, 0, state_time[0], 0)
        trial_factor[l_idx] = 1
    if len(state_time) == 2:
        l_idx, r_idx = get_frame_idx_from_time(time, 0, state_time[0], state_time[1])
        trial_factor[l_idx:r_idx] = 1
    return trial_factor


#%% construct factors

# process dff data.
def get_target_dff_time(list_neural_trials):
    list_glm_time = []
    list_glm_dff = []
    for neural_trials in list_neural_trials:
        # collect factors based on trial.
        glm_time = []
        glm_dff = []
        # iterate through trials.
        for ti in range(len(neural_trials[0])):
            nt = neural_trials[0][str(ti + np.array(sorted(neural_trials[0].keys()), dtype='int32').min())]
            # collect data.
            dff = nt['dff']
            time = nt['time']
            # collect results.
            glm_time.append(time)
            glm_dff.append(dff)
        # concatenate session result.
        glm_time = np.concatenate(glm_time)
        glm_dff = np.concatenate(glm_dff,axis=1)
        # collect into list.
        list_glm_time.append(glm_time)
        list_glm_dff.append(glm_dff)
    return list_glm_time, list_glm_dff

# get all factors.
class get_factor_all:

    def __init__(self, list_neural_trials):
        self.list_neural_trials = list_neural_trials
        self.d1, self.d2 = self.get_type_trial_delay(list_neural_trials)

    def get_type_trial_delay(self, list_neural_trials):
        trial_delay = np.concatenate([[nt[0][k]['trial_delay'] for k in nt[0].keys()] for nt in list_neural_trials])
        d = np.unique(trial_delay).astype('int32')
        d1 = np.nanmin(d)
        d2 = np.nanmax(d)
        print(f'Found 2 delay: {d1}ms, {d2}ms')
        if len(d) > 2:
            print(f'Found multiple delay values: {d}')
        return d1, d2
    
    # both visual cues for all trials.
    def factor_vis_all(self, nt):
        time = nt['time']
        trial_factor = np.zeros_like(time)
        if 'trial_vis1' in nt.keys() and not np.isnan(np.sum(nt['trial_vis1'])):
            trial_factor = set_factor(time, trial_factor, [nt['trial_vis1'][0], nt['trial_vis1'][1]])
        if 'trial_vis2' in nt.keys() and not np.isnan(np.sum(nt['trial_vis2'])):
            trial_factor = set_factor(time, trial_factor, [nt['trial_vis2'][0], nt['trial_vis2'][1]])
        return trial_factor
    
    # prediction of 2nd visual cues in late epoch.
    def factor_vis_2nd_prediction(self, nt):
        win = 300
        time = nt['time']
        trial_factor = np.zeros_like(time)
        if (nt['block_epoch'] == 2 and
            'trial_vis2' in nt.keys() and
            not np.isnan(np.sum(nt['trial_vis2']))
            ):
            trial_factor = set_factor(time, trial_factor, [nt['trial_vis2'][0]-win, nt['trial_vis2'][0]])
        return trial_factor
    
    # prediction error of 2nd visual cue omission in probe trials.
    def factor_vis_2nd_prediction_error(self, nt):
        win = 300
        time = nt['time']
        trial_factor = np.zeros_like(time)
        if (nt['trial_probe'] == 1 and
            not np.isnan(np.sum(nt['trial_push1']) and
            'trial_vis2' in nt.keys() and
            not np.isnan(np.sum(nt['trial_vis2'])))
            ):
            trial_factor = set_factor(time, trial_factor, [nt['trial_vis2'][0]-win, nt['trial_vis2'][1]])
        return trial_factor
    
    # from long to short transition as early epoch visual cue presents earlier.
    def factor_vis_2nd_present_prediction_error(self, nt):
        time = nt['time']
        trial_factor = np.zeros_like(time)
        if (nt['block_epoch'] == 0 and
            nt['trial_types'] == 1 and
            'trial_vis2' in nt.keys() and
            not np.isnan(np.sum(nt['trial_vis2']))
            ):
            trial_factor = set_factor(time, trial_factor, [nt['trial_vis2'][0], nt['trial_vis2'][1]])
        return trial_factor

    # from short to long transition as early epoch visual cue is missing.
    def factor_vis_2nd_omisison_prediction_error(self, nt):
        win = 100
        time = nt['time']
        trial_factor = np.zeros_like(time)
        if (nt['block_epoch'] == 0 and
            nt['trial_types'] == 2 and
            'trial_early2ndpush' in nt.keys() and
            not np.isnan(np.sum(nt['trial_early2ndpush']))
            ):
            trial_factor = set_factor(time, trial_factor, [nt['trial_retract2'][0]+self.d1, nt['trial_retract2'][0]+self.d1+win])
        return trial_factor

    # retract auditory noise.
    def factor_aud_all(self, nt):
        win = 200
        time = nt['time']
        trial_factor = np.zeros_like(time)
        if 'trial_retract1' in nt.keys() and not np.isnan(np.sum(nt['trial_retract1'])):
            trial_factor = set_factor(time, trial_factor, [nt['trial_retract1'][0], nt['trial_retract1'][0]+win])
        if 'trial_retract2' in nt.keys() and not np.isnan(np.sum(nt['trial_retract2'])):
            trial_factor = set_factor(time, trial_factor, [nt['trial_retract2'][0], nt['trial_retract2'][0]+win])
        return trial_factor

    # reward stimulus.
    def factor_reward_prediction(self, nt):
        win = 300
        time = nt['time']
        trial_factor = np.zeros_like(time)
        if 'trial_reward' in nt.keys() and not np.isnan(np.sum(nt['trial_reward'])):
            trial_factor = set_factor(time, trial_factor, [nt['trial_reward'][0]-win, nt['trial_reward'][0]])
        if (nt['trial_types'] == 2 and
            'trial_early2ndpush' in nt.keys() and
            not np.isnan(np.sum(nt['trial_early2ndpush']))
            ):
            trial_factor = set_factor(time, trial_factor, [nt['trial_early2ndpush'][1], nt['trial_early2ndpush'][1]+win])
        return trial_factor
    
    # reward prediction error when early push.
    def factor_reward_prediction_error(self, nt):
        delay = 200
        win = 300
        time = nt['time']
        trial_factor = np.zeros_like(time)
        if (nt['block_epoch'] == 0 and
            nt['trial_types'] == 2 and
            'trial_early2ndpush' in nt.keys() and
            not np.isnan(np.sum(nt['trial_early2ndpush']))
            ):
            trial_factor = set_factor(time, trial_factor, [nt['trial_early2ndpush'][1]+delay, nt['trial_early2ndpush'][1]+delay+win])
        return trial_factor

    # joystick trajectory.
    def factor_js_all(self, nt):
        time = nt['time']
        vol_time = nt['vol_time']
        trial_js_pos = nt['trial_js_pos']
        trial_js_time = nt['trial_js_time'] + vol_time[0]
        trial_factor = interp_factor_in(trial_js_pos, trial_js_time, time)
        return trial_factor

    # push onset.
    def factor_pre_push(self, nt):
        win = 200
        time = nt['time']
        trial_factor = np.zeros_like(time)
        if 'trial_push1' in nt.keys() and not np.isnan(np.sum(nt['trial_push1'])):
            trial_factor = set_factor(time, trial_factor, [nt['trial_push1'][0]-win, nt['trial_push1'][0]])
        if 'trial_push2' in nt.keys() and not np.isnan(np.sum(nt['trial_push2'])):
            trial_factor = set_factor(time, trial_factor, [nt['trial_push2'][0]-win, nt['trial_push2'][0]])
        return trial_factor
    
    # self initiated push from prediction on probe trial.
    def factor_push_self_pred(self, nt):
        win = 200
        time = nt['time']
        trial_factor = np.zeros_like(time)
        if (nt['trial_probe'] == 1 and
            not np.isnan(np.sum(nt['trial_push2']))
            ):
            trial_factor = set_factor(time, trial_factor, [nt['trial_push2'][0]-win, nt['trial_push2'][0]])
        return trial_factor

    # all lick.
    def factor_lick_all(self, nt):
        time = nt['time']
        trial_factor = np.zeros_like(time)
        if 'trial_lick' in nt.keys() and not np.isnan(np.sum(nt['trial_lick'])):
            for lt in nt['trial_lick'][0,:]:
                trial_factor = set_factor(time, trial_factor, [lt])
        return trial_factor

    def run(self, ):
        factor_funcs = [
            self.factor_vis_all,
            self.factor_vis_2nd_prediction,
            self.factor_vis_2nd_prediction_error,
            self.factor_vis_2nd_present_prediction_error,
            self.factor_vis_2nd_omisison_prediction_error,
            self.factor_aud_all,
            self.factor_reward_prediction,
            self.factor_reward_prediction_error,
            self.factor_js_all,
            self.factor_pre_push,
            self.factor_push_self_pred,
            self.factor_lick_all,
        ]
        list_factor_names = [
            'vis_all',
            'vis_2nd_pred',
            'vis_2nd_probe_pe',
            'vis_2nd_transition_present_pe',
            'vis_2nd_transition_omisison_pe',
            'aud_all',
            'reward_pred',
            'reward_pe',
            'js_all',
            'pre_push',
            'push_self_pred',
            'lick_all',
            ]
        list_glm_factor = [[] for _ in factor_funcs]
        # loop through sessions.
        for neural_trials in self.list_neural_trials:
            sess_factors = [[] for _ in factor_funcs]
            # loop through trials.
            for ti in range(len(neural_trials[0])):
                nt = neural_trials[0][str(ti + np.array(sorted(neural_trials[0].keys()), dtype='int32').min())]
                for fi, func in enumerate(factor_funcs):
                    # construct factors.
                    sess_factors[fi].append(func(nt))
            # collect results.
            for si in range(len(factor_funcs)):
                list_glm_factor[si].append(np.concatenate(sess_factors[si]))
        return list_factor_names, list_glm_factor


#%% fit model

# construct the input design matrix for glm.
def construct_design_matrix(factor_target, l_idx, r_idx):
    total_window = l_idx + r_idx + 2
    padded = np.pad(factor_target, (r_idx+1, l_idx+1), mode='constant')
    windows = np.lib.stride_tricks.sliding_window_view(padded, window_shape=total_window)
    return windows[:len(factor_target), :]

# fit regression model with multiple factors (no cross interactions).
def fit_glm_factor(dff, all_factor_in, time, l_idx, r_idx):
    alpha = 52025
    n_neu, n_T = dff.shape
    n_factors = len(all_factor_in)
    k_len = l_idx + r_idx
    # pre-allocate result arrays.
    kernel = np.nan * np.zeros([n_neu, n_factors, k_len])
    exp_var = np.nan * np.zeros([n_neu])
    # build design matrix for each factor and concatenate (no interaction terms).
    X_list = [construct_design_matrix(f, l_idx, r_idx) for f in all_factor_in]
    X = np.concatenate(X_list, axis=1)
    # precompute ridge pseudo-inverse.
    TW = (l_idx + r_idx + 2) * n_factors
    A = X.T @ X + alpha * np.eye(TW)
    M = np.linalg.solve(A, X.T)
    # indices to slice betas per factor
    seg = l_idx + r_idx + 2
    slices = [slice(i*seg, (i+1)*seg) for i in range(n_factors)]
    # for each neuron solve and write results.
    for ni in tqdm(range(n_neu), desc='neurons'):
        y = dff[ni, :]
        beta = M @ y
        # collect kernels per factor and reconstruct prediction
        y_hat = np.zeros_like(y)
        for fi in range(n_factors):
            b_seg = beta[slices[fi]]
            k = np.flip(b_seg)[1:-1]
            kernel[ni, fi, :] = k
            y_hat += factor_dff_neu(all_factor_in[fi], k, l_idx, r_idx)
        exp_var[ni] = explained_variance_score(y, y_hat)
    return kernel, exp_var

# fit glm for multiple sessions.
def run_glm_multi_sess(
        list_dff, list_factor_in, list_time, kernel_win):
    # get kernel time.
    l_idx = np.nanmin([get_frame_idx_from_time(time, time[1106], kernel_win[0], kernel_win[1])[0] for time in list_time])
    r_idx = np.nanmax([get_frame_idx_from_time(time, time[1106], kernel_win[0], kernel_win[1])[1] for time in list_time])
    kernel_time = []
    for time in list_time:
        kernel_time.append(time[l_idx:r_idx] - time[1106])
    kernel_time = np.nanmean(np.stack(kernel_time,axis=0),axis=0)
    l_idx = np.searchsorted(kernel_time, 0)
    r_idx = len(kernel_time) - np.searchsorted(kernel_time, 0)
    # run glm on all sessions.
    kernel_all = []
    exp_var_all = []
    for si, (dff, factor_in, time) in enumerate(zip(list_dff, list_factor_in, list_time)):
        print(f'Fitting GLM for session {si+1}/{len(list_dff)}')
        kernel, exp_var = fit_glm_factor(dff, factor_in, time, l_idx, r_idx)
        kernel_all.append(kernel)
        exp_var_all.append(exp_var)
    # concatenate results.
    kernel_all = np.concatenate(kernel_all, axis=0)
    exp_var_all = np.concatenate(exp_var_all)
    return kernel_time, kernel_all, exp_var_all


#%% evaluation

# compute coding score by fraction of explained variance.
def get_coding_score_fraction(exp_var_all, list_exp_var_single):
    bound = [0, 1]
    s = np.stack([ev/exp_var_all for ev in list_exp_var_single], axis=1)
    s[s<bound[0]] = 0
    s[s>bound[1]] = 1
    return s

# compute coding score by dropout single factors
def get_coding_score_dropout(exp_var_all, list_exp_var_dropout):
    bound = [0, 1]
    s = np.stack([1-ev/exp_var_all for ev in list_exp_var_dropout], axis=1)
    s[s<bound[0]] = 0
    s[s>bound[1]] = 1
    return s
