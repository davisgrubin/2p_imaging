import numpy as np
from scipy.interpolate import interp1d

def aligning_times(trials , aligning_event = "LED"):
    initial_times = []
    ending_times = []
    for id in trials:
        reference_time = trials[id][aligning_event][0]
        aligned_time = trials[id]["time"][:] - reference_time
        initial_times.append(aligned_time[0])
        ending_times.append(np.max(aligned_time))
    init_time =   max(initial_times)
    ending_time = min(ending_times) 
    init_index = {}
    ending_index = {}
    event_index = {}
    ap_index = {}
    event_diff = []
    ending_diff = []
    init_index_0 = {}
    ending_index_0 = {}
    for id in trials:
        init_index[id] = np.searchsorted(trials[id]["time"][:] - trials[id][aligning_event][0], init_time, side='right')
        ending_index[id]=np.searchsorted(trials[id]["time"][:] - trials[id][aligning_event][0], ending_time, side='left')
        event_index[id] =np.searchsorted(trials[id]["time"][:],  trials[id][aligning_event][0], side='right')
        ap_index[id] =np.searchsorted(trials[id]["time"][:],  trials[id]["AirPuff"][0], side='right')
        event_diff.append(event_index[id] - init_index[id])
        ending_diff.append(ending_index[id] - event_index[id])
    ending_incr = min(ending_diff)
    init_incr = min(event_diff)
    for id in trials:
        init_index_0[id] = event_index[id] - init_incr
        ending_index_0[id] = event_index[id] + ending_incr

    return(init_time, init_index_0, ending_time, ending_index_0, event_index, ap_index)

def index_differences(init_index , event_index, ending_index, ap_index):
    for id in init_index:
        event_diff = event_index[id] - init_index[id]
        ending_diff = ending_index[id] - init_index[id]
        ap_diff = ap_index[id] - init_index[id]
        event_diff_0 = event_diff  
        ending_diff_0 = ending_diff
        ap_diff_0 = ap_diff
        if event_diff !=event_diff_0:
            print(id)
        if ending_diff != ending_diff_0:
            print(id)
        if ap_diff != ap_diff_0:
            print(id)
    return event_diff_0, ap_diff_0, ending_diff_0


def FEC_alignment(trials):
    fec_aligned = {}
    fec_time_aligned = {}  # Use a clear name to avoid confusion with fec_time as a variable
    for id in trials:
        # Extract data
        time = trials[id]["time"][:]
        fec = trials[id]["FEC"][:]
        fec_time = trials[id]["FECTimes"][:]

        # Filter time to be within fec_time range
        time = time[(time >= fec_time.min()) & (time <= fec_time.max())]

        # Interpolate FEC data
        interp_function = interp1d(fec_time, fec, kind='linear', bounds_error=True)
        aligned_FEC = interp_function(time)

        # Store in dictionaries
        fec_aligned[id] = aligned_FEC
        fec_time_aligned[id] = time

    return fec_aligned, fec_time_aligned

def aligned_dff(trials,trials_id, cr, cr_stat, init_index, ending_index, sample_id):
    id = sample_id
    aligned_time = {}
    aligned_dff = {}
    for roi in range(len(trials[id]["dff"])):
        aligned_dff[roi] = {}
    for roi in range(len(trials[id]["dff"])):
        for id in trials_id:
            if cr[id] == cr_stat:
                aligned_time[id] = trials[id]["time"][init_index[id]:ending_index[id]] - trials[id]["LED"][0]
                aligned_dff[roi][id] = trials[id]["dff"][roi][init_index[id]: ending_index[id]]

    return aligned_dff , aligned_time

def calculate_average_dff_roi(aligned_dff):
    average_dff = {}
    sem_dff = {}

    for roi in range(len(aligned_dff)):
        # Gather all dff arrays for the current ROI into a list
        all_trials_dff = []
        for trial_id in aligned_dff[roi]:
            all_trials_dff.append(aligned_dff[roi][trial_id])

        # print(len(aligned_dff[roi]))

        # Stack dff arrays along a new axis and compute mean and SEM along the trial axis
        if all_trials_dff:
            stacked_dff = np.vstack(all_trials_dff)  # Shape: (num_trials, time_points)
            average_dff[roi] = np.mean(stacked_dff, axis=0)  # Shape: (time_points,)
            sem_dff[roi] = np.std(stacked_dff, axis=0) / np.sqrt(stacked_dff.shape[0])  # Shape: (time_points,)
            # print("number of trials included in the average (Grand average):" , stacked_dff.shape[0])
        else:
            average_dff[roi] = []
            sem_dff[roi] = []

    return average_dff, sem_dff

def calculate_average_dff_pool(aligned_dff):
    all_trials_dff = []

    for roi in range(len(aligned_dff)):
        for trial_id in aligned_dff[roi]:
            all_trials_dff.append(aligned_dff[roi][trial_id])

    if all_trials_dff:
        # Stack all trial arrays along a new axis and calculate statistics
        stacked_dff = np.vstack(all_trials_dff)  # Shape: (num_trials, time_points)
        average_dff = np.mean(stacked_dff, axis=0)  # Mean across trials (time_points,)
        sem_dff = np.std(stacked_dff, axis=0) / np.sqrt(stacked_dff.shape[0])  # SEM (time_points,)
        # print("number of trials included in the average (Pooled average)" , stacked_dff.shape[0])
    else:
        # Handle the empty case: return empty arrays
        average_dff = np.array([])
        sem_dff = np.array([])

    return average_dff, sem_dff


def calculate_average_sig(aligned_dff, roi_indices):
    all_trials_dff = []

    for roi in roi_indices:
        for trial_id in aligned_dff[roi]:
            all_trials_dff.append(aligned_dff[roi][trial_id])

    if all_trials_dff:
        # Stack all trial arrays along a new axis and calculate statistics
        stacked_dff = np.vstack(all_trials_dff)  # Shape: (num_trials, time_points)
        average_dff = np.mean(stacked_dff, axis=0)  # Mean across trials (time_points,)
        sem_dff = np.std(stacked_dff, axis=0) / np.sqrt(stacked_dff.shape[0])  # SEM (time_points,)
        # print("number of trials included in the average (Pooled average)" , stacked_dff.shape[0])
    else:
        # Handle the empty case: return empty arrays
        average_dff = np.array([])
        sem_dff = np.array([])

    return average_dff, sem_dff

def average_over_roi(dff_dict):
    dff_list = list(dff_dict.values())
    
    # Compute mean and SEM
    avg = np.mean(dff_list, axis=0)
    sem = np.std(dff_list, axis=0) / np.sqrt(len(dff_list))
    print("0" , len(dff_list))
    
    return avg, sem


def sort_numbers_as_strings(numbers):
    # Convert strings to integers for sorting
    sorted_numbers = sorted(numbers, key=int)
    # Convert back to strings
    return list(map(str, sorted_numbers))

def fec_zero(trials):
    fec_time_0 = {}
    fec_0 = {}

    for i , id in enumerate(trials):
        fec_time_0[id] = trials[id]["FECTimes"][:] - trials[id]["LED"][0]
        fec_0[id] = trials[id]["FEC"][:]

    return fec_0,fec_time_0

# def fec_zero(trials):
#     fec_time_0 = {}
#     fec_0 = {}

#     for i , id in enumerate(trials):
#         for val in trials[id]["FEC"][:]:
#             if val<0:
#                 print(f"Negative FEC value in {id}")
#                 break
#             else:
#                 fec_time_0[id] = trials[id]["FECTimes"][:] - trials[id]["LED"][0]
#                 fec_0[id] = trials[id]["FEC"][:]
#     return fec_0,fec_time_0

def moving_average(fec_0, window_size):
    smoothed_fec = {}
    for id in fec_0:
        if window_size < 1:
            raise ValueError("Window size must be a positive integer.")    
        kernel = np.ones(window_size) / window_size
        smoothed_fec[id] = np.convolve(fec_0[id], kernel, mode='same')
    return smoothed_fec
