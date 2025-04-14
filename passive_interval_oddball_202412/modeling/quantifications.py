#!/usr/bin/env python3

import pywt
import numpy as np
from skimage.feature import peak_local_max
from sklearn.cluster import KMeans
from scipy.stats import linregress

from modeling.utils import get_frame_idx_from_time

# compute the curvature of line.
def get_curvature(data):
    t = np.arange(len(data))
    dx = np.gradient(data)
    ddx = np.gradient(dx)
    curvature = np.polyfit(ddx, t, 1)[0]
    return curvature

# test linearity of a line.
def stat_test_linear(data):
    t = np.arange(len(data))
    _, _, _, p_value, _ = linregress(t, data)
    return p_value

# onset detection with wavelet.
def get_change_onset(nsm, neu_time, win_eval):
    num_peaks = 2
    pct = 90
    win_val = [-300,300]
    # evaluation window.
    l_idx, r_idx = get_frame_idx_from_time(neu_time, 0, win_eval[0], win_eval[1])
    nsm = nsm[l_idx:r_idx].copy()
    nt = neu_time[l_idx:r_idx]
    try:
        # wavelet transform.
        wavelets = ['cgau1','cmor2.5-0.5','fbsp']
        scales = np.arange(1, 32)
        sampling_period = np.mean(np.diff(neu_time))
        coef = [
            pywt.cwt(nsm, scales, wavelets[wi], sampling_period=sampling_period)[0]
            for wi in range(len(wavelets))]
        # collect all real and imaginery results.
        coef = [coef[wi].real for wi in range(len(wavelets))] + [coef[wi].imag for wi in range(len(wavelets))]
        # max in time frequency map is dropping onset.
        candi_drop = [peak_local_max(c, threshold_abs=np.percentile(c, pct), num_peaks=num_peaks) for c in coef]
        # min in time frequency map is ramping onset.
        candi_ramp = [peak_local_max(-c, threshold_abs=np.percentile(c, pct), num_peaks=num_peaks) for c in coef]
        # validation window index.
        val_c_idx, _ = get_frame_idx_from_time(neu_time, 0, 0, 0)
        val_l_idx, val_r_idx = get_frame_idx_from_time(neu_time, 0, win_val[0], win_val[1])
        val_l_idx = val_c_idx - val_l_idx
        val_r_idx = val_r_idx - val_c_idx
        # collect target trace within validation window.
        val_drop, val_ramp = [], []
        idx_drop, idx_ramp = [], []
        for ci in range(len(coef)):
            for pi in range(candi_drop[ci].shape[0]):
                i = candi_drop[ci][pi,1]
                if (i > val_l_idx and i < len(nt)-val_r_idx):
                    val_drop.append(nsm[i-val_l_idx:i+val_r_idx].reshape(1,-1))
                    idx_drop.append(i.reshape(-1))
            for pi in range(candi_ramp[ci].shape[0]):
                i = candi_ramp[ci][pi,1]
                if (i > val_l_idx and i < len(nt)-val_r_idx):
                    val_ramp.append(nsm[i-val_l_idx:i+val_r_idx].reshape(1,-1))
                    idx_ramp.append(i.reshape(-1))
        val_drop = np.concatenate(val_drop, axis=0)
        val_ramp = np.concatenate(val_ramp, axis=0)
        idx_drop = np.concatenate(idx_drop)
        idx_ramp = np.concatenate(idx_ramp)
        # run clustering to divide into two groups.
        cluster_id_drop = KMeans(n_clusters=2).fit_predict(val_drop)
        cluster_id_ramp = KMeans(n_clusters=2).fit_predict(val_ramp)
        # compute clustering mean.
        cluster_mean_drop = [np.nanmean(val_drop[cluster_id_drop==i], axis=0) for i in range(2)]
        cluster_mean_ramp = [np.nanmean(val_ramp[cluster_id_ramp==i], axis=0) for i in range(2)]
        # find the group with less linearity.
        p_drop = [stat_test_linear(m) for m in cluster_mean_drop]
        p_ramp = [stat_test_linear(m) for m in cluster_mean_ramp]
        val_drop = val_drop[cluster_id_drop==np.argmax(p_drop),:]
        val_ramp = val_ramp[cluster_id_ramp==np.argmax(p_ramp),:]
        idx_drop = idx_drop[cluster_id_drop==np.argmax(p_drop)]
        idx_ramp = idx_ramp[cluster_id_ramp==np.argmax(p_ramp)]
        # keep the one closet to center stimulus onset.
        stim_onset_idx, _ = get_frame_idx_from_time(neu_time, 0, 0, 0)
        val_drop = val_drop[np.argmin(np.abs(idx_drop - stim_onset_idx)),:]
        val_ramp = val_ramp[np.argmin(np.abs(idx_ramp - stim_onset_idx)),:]
        idx_drop = np.array(idx_drop[np.argmin(np.abs(idx_drop - stim_onset_idx))])
        idx_ramp = np.array(idx_ramp[np.argmin(np.abs(idx_ramp - stim_onset_idx))])
        # refine with local optimum.
        idx_drop = idx_drop - val_l_idx + np.argmax(val_drop)
        idx_ramp = idx_ramp - val_l_idx + np.argmin(val_ramp)
        # convert to time.
        onset_drop = np.array(nt[idx_drop])
        onset_ramp = np.array(nt[idx_ramp])
    except:
        onset_drop = np.nan
        onset_ramp = np.nan
    return onset_drop, onset_ramp
    

# peak and valley value.
def get_peak_valley(nsm, neu_time, onset_drop, onset_ramp):
    if not np.isnan(onset_drop) and not np.isnan(onset_ramp):
        idx_drop, idx_ramp = get_frame_idx_from_time(neu_time, 0, onset_drop, onset_ramp)
        neu_peak   = nsm[idx_drop]
        neu_valley = nsm[idx_ramp]
    else:
        neu_peak = np.nan
        neu_valley = np.nan
    return neu_peak, neu_valley

# ramping or dropping slope.



# compute all metrics.
def get_all_metrics(nsm, neu_time, win_eval):
    onset_drop, onset_ramp = get_change_onset(nsm, neu_time, win_eval)
    neu_peak, neu_valley = get_peak_valley(nsm, neu_time, onset_drop, onset_ramp)
    quant = {
        'onset_drop': onset_drop,
        'onset_ramp': onset_ramp,
        'neu_peak': neu_peak,
        'neu_valley': neu_valley,
        }
    return quant

'''
neu_time = alignment['neu_time']
neu_peak = []
neu_valley = []
for i in range(bin_num):
    win_eval = [bin_stim_seq[i,4,1], 1000]
    quant = get_all_metrics(bin_neu_mean[i,:], neu_time, win_eval)
    neu_peak.append(quant['neu_peak'])
    neu_valley.append(quant['neu_valley'])

fig, ax = plt.subplots(1, 1, figsize=(6, 6))
ax.plot(bin_center, neu_peak)
ax.tick_params(axis='y', tick1On=False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_ylabel('peak value')
ax.set_xlabel('binned isi')

fig, ax = plt.subplots(1, 1, figsize=(6, 6))
ax.plot(bin_center, neu_valley)
ax.tick_params(axis='y', tick1On=False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_ylabel('valley value')
ax.set_xlabel('binned isi')

'''