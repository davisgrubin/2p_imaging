#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

from modeling.utils import norm01
from modeling.utils import get_frame_idx_from_time

# average across subsampling neurons.
def sub_sampling(neu_seq):
    samping_size = 0.2
    sampling_times = 50
    # compute number of samples.
    n_samples = int(samping_size*neu_seq.shape[0])+1
    # average across ramdom subset of neurons.
    neu_seq_sub = np.zeros([sampling_times, neu_seq.shape[1]])
    for qi in range(sampling_times):
        sub_idx = np.random.choice(neu_seq.shape[0], n_samples, replace=False)
        neu_seq_sub[qi,:] = np.nanmean(neu_seq[sub_idx,:], axis=0)
    return neu_seq_sub

# define temporal receptive field model.
def trf_model(t, b, a, m, r, s, ramp_sign=1):
    t_eff = ramp_sign * t
    term_factor = ramp_sign*np.abs(a)/2
    term_exp = np.exp((2*m + r**2) / (2*s) - t_eff/s)
    term_erfc = erfc((m + (r**2)/s - t_eff) / (np.sqrt(2)*r))
    trf = b + term_factor * term_exp * term_erfc
    return trf
def trf_model_up(t, b, a, m, r, s):
    return trf_model(t, b, a, m, r, s, 1)
def trf_model_dn(t, b, a, m, r, s):
    return trf_model(t, b, a, m, r, s, -1)

# fit response model for single trace.
def fit_trf_model(nsl, nsr, l_time, r_time):
    # initial conditions.
    init_b = 0.0
    init_a = 0.5
    init_m = 0.3
    init_r = 0.15
    init_s = 0.5
    p0 = [init_b, init_a, init_m, init_r, init_s]
    # bounds.
    bounds_b = [-2.0, 2.0]
    bounds_a = [0.0, 25.0]
    bounds_m = [-2, 2]
    bounds_r = [1e-3, 0.7]
    bounds_s = [1e-3, 0.9]
    bounds = [np.array([bounds_b[0], bounds_a[0], bounds_m[0], bounds_r[0], bounds_s[0]]),
              np.array([bounds_b[1], bounds_a[1], bounds_m[1], bounds_r[1], bounds_s[1]])]
    # fit models.
    try:
        popt_up, _ = curve_fit(trf_model_up, l_time, nsl, p0=p0, bounds=bounds, maxfev=20000)
        popt_dn, _ = curve_fit(trf_model_dn, r_time, nsr, p0=p0, bounds=bounds, maxfev=20000)
        y_pred_up = trf_model_up(l_time, *popt_up)
        y_pred_dn = trf_model_dn(r_time, *popt_dn)
        r2_up = r2_score(nsl, y_pred_up)
        r2_dn = r2_score(nsr, y_pred_dn)
    except:
        popt_up = np.full(5, np.nan)
        popt_dn = np.full(5, np.nan)
        y_pred_up = np.full_like(nsl, np.nan)
        y_pred_dn = np.full_like(nsr, np.nan)
        r2_up = np.nan
        r2_dn = np.nan
    return popt_up, popt_dn, y_pred_up, y_pred_dn, r2_up, r2_dn

# fit response model for all input neurons.
def fit_trf_model_all(neu_seq, neu_time):
    margin_time = 300
    # subsampling neurons.
    neu_seq_sub = sub_sampling(neu_seq)
    # get time zero.
    l_bound, r_bound = get_frame_idx_from_time(neu_time, 0, margin_time, -margin_time)
    # z score data.
    neu_seq_sub = (neu_seq_sub - np.nanmean(neu_seq)) / (np.nanstd(neu_seq_sub) + 1e-8)
    neu_seq_l = neu_seq_sub[:,:l_bound]
    neu_seq_r = neu_seq_sub[:,r_bound:]
    # normlize time.
    l_time = norm01(neu_time[:l_bound])
    r_time = norm01(neu_time[r_bound:])
    # initialize results.
    popt_up = np.zeros([5,neu_seq_sub.shape[0]])
    popt_dn = np.zeros([5,neu_seq_sub.shape[0]])
    y_pred_up = np.zeros_like(neu_seq_l)
    y_pred_dn = np.zeros_like(neu_seq_r)
    r2_up = np.zeros([neu_seq_sub.shape[1]])
    r2_dn = np.zeros([neu_seq_sub.shape[1]])
    # fit model for each neuron.
    for ni in range(neu_seq_sub.shape[0]):
        results = fit_trf_model(neu_seq_l[ni,:], neu_seq_r[ni,:], l_time, r_time)
        popt_up[:,ni] = results[0]
        popt_dn[:,ni] = results[1]
        y_pred_up[ni,:] = results[2]
        y_pred_dn[ni,:] = results[3]
        r2_up[ni] = results[4]
        r2_dn[ni] = results[5]

'''

neu_time = neu_time_1
neu_seq = neu_seq_1[0,cluster_id==2,:]
nsl, nsr = neu_seq_l[0,:], neu_seq_r[0,:]

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
for i in range(sampling_times):
    ax.plot(l_time, neu_seq_l[i,:], color='black', alpha=0.1)
    ax.plot(l_time, y_pred_up[i,:], color='dodgerblue', alpha=0.1)
    
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
for i in range(sampling_times):
    ax.plot(r_time, neu_seq_r[i,:], color='black', alpha=0.2)
    ax.plot(r_time, y_pred_dn[i,:], color='dodgerblue', alpha=0.1)
'''


# ---------- Fitting ----------

def fit_half(t_half, y_half, ramp='up', a_sign='pos'):
    # initial guesses
    b0 = np.median(y_half)
    a0 = max(0.2, np.percentile(y_half, 95) - np.percentile(y_half, 5))
    if a_sign == 'neg':
        a0 = -a0
    mu0 = -0.8  # works for both due to form
    r0, s0 = 0.4, 1.0

    # bounds
    b_bounds = (y_half.min() - 2.0, y_half.max() + 2.0)
    if a_sign == 'pos':
        a_bounds = (0.0, 50.0)
    else:
        a_bounds = (-50.0, 0.0)
    mu_bounds = (-T_pre, 1.0)  # broad
    r_bounds = (0.05, 2.0)
    s_bounds = (0.05, 3.0)

    p0 = (b0, a0, mu0, r0, s0)
    bounds = (np.array([b_bounds[0], a_bounds[0], mu_bounds[0], r_bounds[0], s_bounds[0]]),
              np.array([b_bounds[1], a_bounds[1], mu_bounds[1], r_bounds[1], s_bounds[1]]))

    model = trf_model_up if ramp == 'up' else trf_model_down

    try:
        popt, _ = curve_fit(model, t_half, y_half, p0=p0, bounds=bounds, maxfev=20000)
        y_pred = model(t_half, *popt)
        r2 = r2_score(y_half, y_pred)
        return popt, y_pred, r2
    except Exception:
        return (np.full(5, np.nan), np.full_like(y_half, np.nan), np.nan)


# Containers
trf_param_up = np.zeros((n_neurons, 5))   # b, a, mu, r, s
trf_param_down = np.zeros((n_neurons, 5))
r2_all_up = np.zeros(n_neurons)
r2_all_down = np.zeros(n_neurons)

# Fit each neuron
for i in range(n_neurons):
    # Pre-stim fit (ramp-up model), constrain a >= 0
    p_up, yhat_up, r2_up = fit_half(t_neg, neu_seq[i, :split], ramp='up', a_sign='pos')
    trf_param_up[i] = p_up
    r2_all_up[i] = r2_up

    # Post-stim fit (ramp-down model), constrain sign by neuron type
    a_sign = 'neg' if mask_typeA[i] else 'pos'
    p_down, yhat_down, r2_down = fit_half(t_pos, neu_seq[i, split:], ramp='down', a_sign=a_sign)
    trf_param_down[i] = p_down
    r2_all_down[i] = r2_down

# ---------- Plotting (20 neurons per type) ----------

def plot_group(indices, title_prefix):
    fig, axes = plt.subplots(4, 5, figsize=(20, 12))
    axes = axes.ravel()
    for ax, i in zip(axes, indices):
        y = neu_seq[i]
        # model preds
        y_up = mexg_fixed_up(t_neg, *trf_param_up[i])
        y_down = mexg_fixed_down(t_pos, *trf_param_down[i])
        ax.plot(t_neg, y[:split], linewidth=1.0)
        ax.plot(t_neg, y_up, linewidth=2.0)
        ax.plot(t_pos, y[split:], linewidth=1.0)
        ax.plot(t_pos, y_down, linewidth=2.0)
        ax.axvline(0, linestyle='--', linewidth=1.0)
        ax.set_title(
            f"{title_prefix} {i} | R2_up={r2_all_up[i]:.2f}, R2_down={r2_all_down[i]:.2f}",
            fontsize=9)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Rate (a.u.)")
    plt.tight_layout()
    return fig

# Select 20 from each type
idx_up = rng.choice(np.where(mask_typeA)[0], size=20, replace=False)
idx_down = rng.choice(np.where(mask_typeB)[0], size=20, replace=False)

fig1 = plot_group(idx_up, "RampUp→Inhibit")
fig2 = plot_group(idx_down, "Excited→RampDown")

plt.show()

print("trf_param_up shape:", trf_param_up.shape)
print("trf_param_down shape:", trf_param_down.shape)
print("r2_all_up (mean±sd):", np.nanmean(r2_all_up), np.nanstd(r2_all_up))
print("r2_all_down (mean±sd):", np.nanmean(r2_all_down), np.nanstd(r2_all_down))
