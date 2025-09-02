#!/usr/bin/env python3

import numpy as np
from tqdm import tqdm
from scipy.special import erfc
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

from modeling.utils import norm01

# define temporal receptive field model.
def trf_model(t, b, a, m, r, s, ramp_sign=1, sign=1):
    t_eff = ramp_sign * t
    term_factor = sign*np.abs(a)/2
    term_exp = np.exp((2*m + r**2) / (2*s) - t_eff/s)
    term_erfc = erfc((m + (r**2)/s - t_eff) / (np.sqrt(2)*r))
    trf = b + term_factor * term_exp * term_erfc
    return trf
def trf_model_up(t, b, a, m, r, s):
    return trf_model(t, b, a, m, r, s, ramp_sign=1, sign=1)
def trf_model_dn(t, b, a, m, r, s):
    return trf_model(t, b, a, m, r, s, ramp_sign=-1, sign=1)
def trf_model_causal_up(t, b, a, m, r, s):
    return trf_model(t, b, a, m, r, s, ramp_sign=-1, sign=-1)
def trf_model_causal_dn(t, b, a, m, r, s):
    return trf_model(t, b, a, m, r, s, ramp_sign=-1, sign=1)
    
# fit response model for single trace.
def fit_trf_model(neu_seq_l, neu_time_l, neu_seq_r, neu_time_r):
    # normalize data.
    nsl = np.apply_along_axis(norm01, 1, neu_seq_l)
    nsr = np.apply_along_axis(norm01, 1, neu_seq_r)
    ntl = norm01(neu_time_l)-1
    ntr = norm01(neu_time_r)
    # initialize results.
    pred_up = np.zeros_like(nsl) * np.nan
    pred_dn = np.zeros_like(nsr) * np.nan
    trf_param_up = np.zeros((neu_seq_l.shape[0], 5)) * np.nan
    trf_param_dn = np.zeros((neu_seq_r.shape[0], 5)) * np.nan
    r2_all_up = np.zeros(neu_seq_l.shape[0]) * np.nan
    r2_all_dn = np.zeros(neu_seq_r.shape[0]) * np.nan
    # fit ramp up model.
    print('Fitting ramp up model')
    for ni in tqdm(range(neu_seq_l.shape[0]), desc='neuron'):
        try:
            popt, _ = curve_fit(
                trf_model_up, ntl, nsl[ni,:],
                p0=[0.5, 1.5, -0.25, 0.2, 0.5],
                bounds=([0, 0.0, -0.5, 0.001, 0], [1, 5.0, 0, 1.0, 5.0]))
            trf_param_up[ni] = popt
            pred_up[ni,:] = trf_model_up(ntl, *popt)
            r2_all_up[ni] = r2_score(nsl[ni,:], pred_up[ni,:])
        except: pass
    # fit ramp down model.
    print('Fitting ramp down model')
    for ni in tqdm(range(neu_seq_r.shape[0]), desc='neuron'):
        try:
            popt, _ = curve_fit(
                trf_model_dn, ntr, nsr[ni,:],
                p0=[0.5, 1.5, -0.25, 0.2, 0.5],
                bounds=([0, 0.0, -0.5, 0.001, 0], [1, 5.0, 0, 1.0, 5.0]))
            trf_param_dn[ni] = popt
            pred_dn[ni,:] = trf_model_dn(ntr, *popt)
            r2_all_dn[ni] = r2_score(nsr[ni,:], pred_dn[ni,:])
        except: pass
    return [trf_param_up, pred_up, r2_all_up,
            trf_param_dn, pred_dn, r2_all_dn]

# fit response model for single trace.
def fit_trf_model_causal(neu_seq, neu_time):
    # normalize data.
    ns = np.apply_along_axis(norm01, 1, neu_seq)
    nt = norm01(neu_time)
    # initialize results.
    pred_up = np.zeros_like(ns) * np.nan
    pred_dn = np.zeros_like(ns) * np.nan
    trf_param_up = np.zeros((neu_seq.shape[0], 5)) * np.nan
    trf_param_dn = np.zeros((neu_seq.shape[0], 5)) * np.nan
    r2_all_up = np.zeros(neu_seq.shape[0]) * np.nan
    r2_all_dn = np.zeros(neu_seq.shape[0]) * np.nan
    # fit ramp up model.
    print('Fitting ramp up model')
    for ni in tqdm(range(neu_seq.shape[0]), desc='neuron'):
        try:
            popt, _ = curve_fit(
                trf_model_causal_up, nt, ns[ni,:],
                p0=[0.5, 1.5, -0.25, 0.2, 0.5],
                bounds=([-1, 0.0, -1, 0.001, 0], [1, 5.0, 0, 1.0, 5.0]))
            trf_param_up[ni] = popt
            pred_up[ni,:] = trf_model_causal_up(nt, *popt)
            r2_all_up[ni] = r2_score(ns[ni,:], pred_up[ni,:])
        except: pass
    # fit ramp down model.
    print('Fitting ramp down model')
    for ni in tqdm(range(neu_seq.shape[0]), desc='neuron'):
        try:
            popt, _ = curve_fit(
                trf_model_causal_dn, nt, ns[ni,:],
                p0=[0.5, 1.5, -0.25, 0.2, 0.5],
                bounds=([-1, 0.0, -1, 0.001, 0], [1, 5.0, 0, 1.0, 5.0]))
            trf_param_dn[ni] = popt
            pred_dn[ni,:] = trf_model_causal_dn(nt, *popt)
            r2_all_dn[ni] = r2_score(ns[ni,:], pred_dn[ni,:])
        except: pass
    return [trf_param_up, pred_up, r2_all_up,
            trf_param_dn, pred_dn, r2_all_dn]

'''

axs[0].plot(ntl, trf_model_up(ntl, 0, 1.5, -0.4, 0.15, 0.5))
axs[8].plot(ntl, trf_model_up(ntl, 0.2, 1.5, -0.1, 0.1, 0.5))
axs[10].plot(ntl, trf_model_up(ntl, 0.2, 1.5, -0.1, 0.1, 0.5))

fig, axs = plt.subplots(6, 8, figsize=(24, 18))
axs = [x for xs in axs for x in xs]
for ni in range(48):
    axs[ni].plot(ntl,nsl[ni])
    axs[ni].plot(ntl,pred_up[ni])
    
fig, axs = plt.subplots(6, 8, figsize=(24, 18))
axs = [x for xs in axs for x in xs]
for ni in range(48):
    axs[ni].plot(ntr,nsr[ni])
    axs[ni].plot(ntr,pred_dn[ni])

fig, ax = plt.subplots(1, 1, figsize=(3, 3))
ax.plot(ntr,nsr[12])
ax.plot(ntr, trf_model_dn(ntr, 0, 1.5, -0.5, 0.4, 1))

fig, ax = plt.subplots(1, 1, figsize=(3, 3))
ax.hist(r2_all_up,bins=100)

thres = 0.251314
fig, ax = plt.subplots(1, 1, figsize=(3, 3))
ax.axis('off')
ax_hm = ax.inset_axes([0, 0, 0.5, 1], transform=ax.transAxes)
ax_cb = ax.inset_axes([0.6, 0, 0.1, 1], transform=ax.transAxes)
plot_heatmap_neuron(ax_hm, ax_cb, nsl[r2_all_up>thres], ntl, nsl[r2_all_up>thres], norm_mode='minmax')

fig, ax = plt.subplots(1, 1, figsize=(3, 3))
ax.axis('off')
ax_hm = ax.inset_axes([0, 0, 0.5, 1], transform=ax.transAxes)
ax_cb = ax.inset_axes([0.6, 0, 0.1, 1], transform=ax.transAxes)
plot_heatmap_neuron(ax_hm, ax_cb, nsr[r2_all_dn>thres], ntr, nsr[r2_all_dn>thres], norm_mode='minmax')
'''
            