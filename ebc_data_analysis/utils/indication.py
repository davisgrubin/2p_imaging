import numpy as np
import math
from scipy.stats import ttest_rel

def find_index(home_array, event):
    return np.searchsorted(home_array, event, side='right')

def block_and_CR_fec(CR_stat,fec_0, shorts, longs):
    short_CRp = []
    short_CRn = []
    long_CRp = []
    long_CRn = []

    for id in shorts:
        if CR_stat[id] == 1:
            short_CRp.append(fec_0[id] * 0.01)
        if CR_stat[id] == 0:
            short_CRn.append(fec_0[id]* 0.01)

    for id in longs:
        if CR_stat[id] == 1:
            long_CRp.append(fec_0[id]* 0.01)
        if CR_stat[id] == 0:
            long_CRn.append(fec_0[id]* 0.01)

    return short_CRp, short_CRn, long_CRp, long_CRn

def CR_stat_indication(trials , static_threshold, AP_delay):
    fec_index_0 = {}
    fec_index_led = {}
    fec_index_ap = {}
    fec_index_cr = {}
    fec_index_bl = {}
    fec_time_0 = {}
    fec_0 = {}
    base_line_avg = {}
    CR_interval_avg = {}
    CR_stat = {}

    for i , id in enumerate(trials):
        if trials[id]["trial_type"][()] == 1:
            interval_type = 0
        if trials[id]["trial_type"][()] == 2:
            interval_type = 50
        fec_time_0[id] = trials[id]["FECTimes"][:] - trials[id]["LED"][0]
        fec_0[id] = trials[id]["FEC"][:]

        fec_index_0[id] = np.abs(fec_time_0[id]).argmin()
        fec_index_led[id] = find_index(fec_time_0[id], 0.0)
        fec_index_ap[id] = find_index(fec_time_0[id] , trials[id]["AirPuff"][0]- trials[id]["LED"][0] + 12)
        fec_index_cr[id] = find_index(fec_time_0[id], trials[id]["AirPuff"][0]- trials[id]["LED"][0] - 50)
        fec_index_bl[id] = find_index(fec_time_0[id] , trials[id]["LED"][0]- trials[id]["LED"][0] - 200)

        # print("fec indexes", fec_index_led - fec_index_0)

        CR_interval = fec_0[id][fec_index_cr[id] :fec_index_ap[id]]
        CR_interval_avg[id] = np.average(CR_interval)

        base_line = np.sort(fec_0[id][fec_index_bl[id]: fec_index_led[id]])
        base_line_indexes = int(0.3 * len(base_line))
        base_line_avg[id] = np.average(base_line[:base_line_indexes])

        if CR_interval_avg[id] - base_line_avg[id] > static_threshold:
            CR_stat[id] = 1
        else:
            CR_stat[id] = 0
    return CR_stat, CR_interval_avg, base_line_avg 

def block_type(trials):
    shorts = []
    longs = []
    for i , id in enumerate(trials):
        if trials[id]["trial_type"][()] == 1:
            shorts.append(id)
        if trials[id]["trial_type"][()] == 2:
            longs.append(id)
    return shorts, longs

def intervals(dff_signal, led_index, ap_index, interval_window_led, interval_window_cr, interval_window_ap, interval_window_bl, isi_time, sample_id):
    cr_interval = {}
    led_interval = {}
    ap_interval = {}
    base_line_interval = {}

    for roi in range(len(dff_signal)):
        cr_interval[roi] = {}
        led_interval[roi] = {}
        ap_interval[roi] = {}
        base_line_interval[roi] = {}

        for id in dff_signal[roi]:
            # Ensure indices are within bounds
            if led_index[id] < math.floor(interval_window_bl / 33.6) or \
               led_index[id] + math.floor(isi_time / 33.6) + math.floor(interval_window_ap / 33.6) > len(dff_signal[roi][id]):
                continue  # Skip trials with invalid indices

            # Compute intervals
            cr_interval[roi][id] = dff_signal[roi][id][
                ap_index[id] - math.floor(interval_window_cr / 33.6): 
                ap_index[id]
            ]
            led_interval[roi][id] = dff_signal[roi][id][
                led_index[id]: 
                led_index[id] + math.floor(interval_window_led / 33.6)
            ]
            ap_interval[roi][id] = dff_signal[roi][id][
                ap_index[id]: 
                ap_index[id] + math.floor(interval_window_ap / 33.6)
            ]
            base_line_interval[roi][id] = dff_signal[roi][id][
                led_index[id] - math.floor(interval_window_bl / 33.6): 
                led_index[id]
            ]

    return cr_interval, led_interval, ap_interval, base_line_interval


def ttest_intervals(interval_under_test, base_interval, roi_list):
    t_stat = {}
    p_value = {}
    nan_ids = []
    for roi in roi_list:
        t_stat[roi] = {}
        p_value[roi] = {}
    for roi in roi_list:
        for id in interval_under_test[roi]:
            test_data = interval_under_test[roi][id]
            base_data = base_interval[roi][id]
            
            if len(test_data) == 0 or len(base_data) == 0 or (
                np.all(test_data == test_data[0]) and np.all(base_data == base_data[0])
            ):
                t_stat[roi][id] = np.nan
                p_value[roi][id] = np.nan
                nan_ids.append(id)
                continue
            
            t_stat[roi][id], p_value[roi][id] = ttest_rel(base_data, test_data)
    if nan_ids:
        print(f"IDs resulting in NaN: {set(nan_ids)}")
    return t_stat, p_value

def calculate_average_ttest(t_stat):
    roi_avg_t_stat = {}
    # roi_avg_p_value = {}
    
    # Calculate average for each ROI
    for roi in t_stat:
        t_values = [t_stat[roi][id] for id in t_stat[roi] if not np.isnan(t_stat[roi][id])]
        # p_values = [p_value[roi][id] for id in p_value[roi] if not np.isnan(p_value[roi][id])]
        
        roi_avg_t_stat[roi] = np.mean(t_values) if t_values else np.nan
        # roi_avg_p_value[roi] = np.mean(p_values) if p_values else np.nan

    # # Calculate overall average across all ROIs
    # all_t_stats = [roi_avg_t_stat[roi] for roi in roi_avg_t_stat if not np.isnan(roi_avg_t_stat[roi])]
    # all_p_values = [roi_avg_p_value[roi] for roi in roi_avg_p_value if not np.isnan(roi_avg_p_value[roi])]

    # overall_avg_t_stat = np.mean(all_t_stats) if all_t_stats else np.nan
    # overall_avg_p_value = np.mean(all_p_values) if all_p_values else np.nan

    return roi_avg_t_stat

# def sort_rois_by_ttest(roi_avg_t_stat, top_n):
#     # Filter out ROIs with NaN values in their t_stat
#     valid_rois = {roi: abs(roi_avg_t_stat[roi]) for roi in roi_avg_t_stat if not np.isnan(roi_avg_t_stat[roi])}

#     # Sort ROIs by absolute t_stat values in descending order
#     sorted_rois = sorted(valid_rois.items(), key=lambda x: x[1], reverse=True)

#     # Select the top N ROIs
#     top_rois = sorted_rois[:top_n]

#     return top_rois

def sort_rois_by_ttest(t_stats, top_n=7):
    """Sorts ROIs by absolute t-statistic values and returns the top N ROIs."""
    return sorted(
        [(roi, abs(stat)) for roi, stat in t_stats.items() if not np.isnan(stat)],
        key=lambda x: x[1],
        reverse=True
    )[:top_n]

def extract_top_rois(t_stats_list):
    """Extracts ROI names from multiple top ROI lists."""
    return np.concatenate([np.array([roi for roi, _ in sort_rois_by_ttest(t_stats)]) for t_stats in t_stats_list])
    

def sort_dff_avg(dff, led_index, ap_index):
    avg_cr = {}
    cr_window = {}
    for roi in dff:
        cr_window[roi] = dff[roi][led_index:ap_index]
        avg_cr[roi] = np.mean(cr_window[roi])
    sorted_roi = {roi: avg_cr[roi] for roi in sorted(avg_cr, key=avg_cr.get, reverse=True)}
    return sorted_roi

def sort_dff_max(dff, led_index, ap_index):
    max_cr = {}
    cr_window = {}
    for roi in dff:
        cr_window[roi] = dff[roi][led_index:ap_index]
        max_cr[roi] = np.max(cr_window[roi])
    sorted_roi = {roi: max_cr[roi] for roi in sorted(max_cr, key=max_cr.get, reverse=True)}
    return sorted_roi

def sort_dff_max_index(dff, led_index, ap_index):
    max_index_cr = {}
    cr_window = {}
    for roi in dff:
        cr_window[roi] = dff[roi][led_index:ap_index]
        max_index_cr[roi] = np.argmax(cr_window[roi])
    sorted_roi = {roi: max_index_cr[roi] for roi in sorted(max_index_cr, key=max_index_cr.get)}
    return sorted_roi