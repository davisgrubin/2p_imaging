#!/usr/bin/env python3

import pywt
import numpy as np
from tqdm import tqdm
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
def get_change_onset(nsm, neu_time, win_eval_c):
    num_peaks = 2
    pct = 80
    win_eval = [-1000, 1500]
    win_val = [-350,350]
    # evaluation window.
    l_idx, r_idx = get_frame_idx_from_time(neu_time, win_eval_c, win_eval[0], win_eval[1])
    nsm = nsm[l_idx:r_idx].copy()
    nt = neu_time[l_idx:r_idx]
    try:
        # wavelet transform.
        wavelets = ['cgau1','cmor2.5-0.5','fbsp']
        scales = np.arange(1, 64)
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
        c_idx, _ = get_frame_idx_from_time(nt, 0, win_eval_c, 0)
        val_drop = val_drop[np.argmin(np.abs(idx_drop - c_idx)),:]
        val_ramp = val_ramp[np.argmin(np.abs(idx_ramp - c_idx)),:]
        idx_drop = np.array(idx_drop[np.argmin(np.abs(idx_drop - c_idx))])
        idx_ramp = np.array(idx_ramp[np.argmin(np.abs(idx_ramp - c_idx))])
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

# stimulus evoked latency.
def get_stim_evoke_latency(onset_drop, onset_ramp):
    try:
        evoke_latency = onset_ramp - onset_drop
    except:
        evoke_latency = np.nan
    return evoke_latency

# stimulus evoked response.
def get_stim_evoke_mag(onset_drop, onset_ramp, neu_peak, neu_valley):
    try:
        sign = np.sign(np.nanmean(onset_ramp) - np.nanmean(onset_drop))
        evoke_mag = sign * (neu_valley - neu_peak)
    except:
        evoke_mag = np.nan
    return evoke_mag

# stimulus evoked response slope.
def get_stim_evoke_slope(onset_drop, onset_ramp, neu_peak, neu_valley):
    try:
        evoke_slope = (onset_drop - onset_ramp) / (neu_peak - neu_valley)
    except:
        evoke_slope = np.nan
    return evoke_slope

# compute all metrics.
def get_all_metrics(nsm, neu_time, win_eval_c):
    onset_drop, onset_ramp = get_change_onset(nsm, neu_time, win_eval_c)
    neu_peak, neu_valley = get_peak_valley(nsm, neu_time, onset_drop, onset_ramp)
    evoke_latency = get_stim_evoke_latency(onset_drop, onset_ramp)
    evoke_mag = get_stim_evoke_mag(onset_drop, onset_ramp, neu_peak, neu_valley)
    evoke_slope = get_stim_evoke_slope(onset_drop, onset_ramp, neu_peak, neu_valley)
    quant = {
        'onset_drop': onset_drop - win_eval_c,
        'onset_ramp': onset_ramp - win_eval_c,
        'neu_peak': neu_peak,
        'neu_valley': neu_valley,
        'evoke_latency': evoke_latency,
        'evoke_mag': evoke_mag,
        'evoke_slope': evoke_slope,
        }
    return quant

# compute all metrics for each traces.
def run_quantification(neu_seq, neu_time, win_eval_c, samping_size=0.2):
    quant = []
    # average subset of traces.
    if samping_size > 0:
        sampling_time = 10
        n_samples = int(samping_size*neu_seq.shape[0])+1
        for qi in tqdm(range(sampling_time)):
            sub_idx = np.random.choice(neu_seq.shape[0], n_samples, replace=False)
            nsm = np.nanmean(neu_seq[sub_idx,:], axis=0)
            q = get_all_metrics(nsm, neu_time, win_eval_c)
            quant.append(q)
    # compute for all individual traces.
    else:
        for ni in tqdm(range(neu_seq.shape[0])):
            nsm = neu_seq[ni,:].copy()
            q = get_all_metrics(nsm, neu_time, win_eval_c)
            quant.append(q)
    quant = {k: np.array([q[k] for q in quant]) for k in q.keys()}
    return quant
    

    


'''
n_clusters = 6

lbl = ['cluster #'+str(ci) for ci in range(n_clusters)]
xlim = [-2500, 4000]
kernel_time, kernel_all, cluster_id, neu_labels = run_clustering(cate)
oddball = 1

# collect data.
[[color0, color1, color2, _],
 [neu_seq, _, stim_seq, stim_value],
 [neu_labels, neu_sig], _] = get_neu_trial(
    alignment, list_labels, list_significance, list_stim_labels,
    trial_idx=[l[oddball] for l in list_odd_idx],
    trial_param=[None, None, [0], None, [0], [0]],
    cate=cate, roi_id=None)

# get response within cluster.
neu_mean, neu_sem = get_mean_sem_cluster(neu_seq, cluster_id)
neu_time = alignment['neu_time']

c_idx = int(stim_seq.shape[0]/2)
win_eval_c = stim_seq[c_idx+1,0]

fig, axs = plt.subplots(2, n_clusters, figsize=(2*n_clusters,4))
for cli in range(n_clusters):
    nsm = neu_mean[cli,:]
    onset_drop, onset_ramp = get_change_onset(nsm, neu_time, win_eval_c)
    axs[0,cli].plot(neu_time, nsm, color='black')
    axs[0,cli].axvline(onset_drop, color='crimson', linestyle=':')
    axs[0,cli].axvline(onset_ramp, color='royalblue', linestyle=':')
    axs[0,cli].set_xlim(xlim)

fig, axs = plt.subplots(9, n_clusters, figsize=(2*n_clusters,20))
for cli in range(n_clusters):
    nsm = neu_mean[cli,:]
    
    num_peaks = 2
    pct = 80
    win_eval = [-1000, 1500]
    win_val = [-350,350]
    # evaluation window.
    l_idx, r_idx = get_frame_idx_from_time(neu_time, win_eval_c, win_eval[0], win_eval[1])
    nsm = nsm[l_idx:r_idx].copy()
    nt = neu_time[l_idx:r_idx]
    # wavelet transform.
    wavelets = ['cgau1','cmor2.5-0.5','fbsp']
    scales = np.arange(1, 64)
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
    c_idx, _ = get_frame_idx_from_time(nt, 0, win_eval_c, 0)
    val_drop = val_drop[np.argmin(np.abs(idx_drop - c_idx)),:]
    val_ramp = val_ramp[np.argmin(np.abs(idx_ramp - c_idx)),:]
    idx_drop = np.array(idx_drop[np.argmin(np.abs(idx_drop - c_idx))])
    idx_ramp = np.array(idx_ramp[np.argmin(np.abs(idx_ramp - c_idx))])
    # refine with local optimum.
    idx_drop = idx_drop - val_l_idx + np.argmax(val_drop)
    idx_ramp = idx_ramp - val_l_idx + np.argmin(val_ramp)
    # convert to time.
    onset_drop = np.array(nt[idx_drop])
    onset_ramp = np.array(nt[idx_ramp])

    axs[0,cli].plot(nt, nsm, color='black')
    axs[0,cli].axvline(onset_drop, color='crimson', linestyle=':')
    axs[0,cli].axvline(onset_ramp, color='royalblue', linestyle=':')
    dnsm = np.diff(nsm, prepend=nsm[0])
    axs[1,cli].plot(nt, dnsm, color='black')
    ddnsm = np.diff(dnsm, prepend=dnsm[0])
    axs[2,cli].plot(nt, ddnsm, color='black')
    
    extent=[nt[0], nt[-1], scales[-1], scales[0]]
    for wi in range(6):
        axs[wi+3,cli].imshow(coef[wi], extent=extent, aspect='auto', cmap='coolwarm')  
    for wi in range(6):
        for i in candi_drop[wi][:,1]:
            axs[wi+3,cli].axvline(nt[i], color='crimson', linestyle=':')
        for i in candi_ramp[wi][:,1]:
            axs[wi+3,cli].axvline(nt[i], color='royalblue', linestyle=':')

'''