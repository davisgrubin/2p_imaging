from numbers import Number
import numpy as np
from scipy.interpolate import interp1d

def find_index(array, value):
    return np.searchsorted(array, value, side='right')

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
    led_index = {}
    ap_index = {}
    event_diff = []
    ending_diff = []
    init_index_0 = {}
    ending_index_0 = {}
    for id in trials:
        init_index[id] = np.searchsorted(trials[id]["time"][:] - trials[id][aligning_event][0], init_time, side='right')
        ending_index[id]=np.searchsorted(trials[id]["time"][:] - trials[id][aligning_event][0], ending_time, side='left')
        led_index[id] =np.searchsorted(trials[id]["time"][:],  trials[id][aligning_event][0], side='right')
        ap_index[id] =np.searchsorted(trials[id]["time"][:],  trials[id]["AirPuff"][0], side='right')
        event_diff.append(led_index[id] - init_index[id])
        ending_diff.append(ending_index[id] - led_index[id])
    ending_incr = min(ending_diff)
    init_incr = min(event_diff)
    for id in trials:
        init_index_0[id] = led_index[id] - init_incr
        ending_index_0[id] = led_index[id] + ending_incr

    return(init_time, init_index_0, ending_time, ending_index_0, led_index, ap_index)

def index_differences(init_index , led_index, ending_index, ap_index):
    for id in init_index:
        event_diff = led_index[id] - init_index[id]
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


# interpolating the fec points to get a complete line.
#adding a code to align the times to the favored intervals.
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
    # spike_threshold = 3
    # constant_threshold = 0.0000000001
    id = sample_id
    aligned_time = {}
    aligned_dff = {}
    for roi in range(len(trials[id]["dff"])):
        aligned_dff[roi] = {}
    for id in trials_id:
        for roi in range(len(trials[id]["dff"])):
    # for roi in range(len(trials[id]["dff"])):
    #     for id in trials_id:
            if cr[id] == cr_stat:
                dff_value = trials[id]["dff"][roi][init_index[id]: ending_index[id]]
                std = np.nanstd(dff_value)
                # mean = np.nanmean(dff_value)
                # if std< constant_threshold:
                #     print(f"roi {roi} is constant")
                #     continue
                aligned_time[id] = trials[id]["time"][init_index[id]:ending_index[id]] - trials[id]["LED"][0]
                aligned_dff[roi][id] = dff_value
                # print(f'this is 0 {dff_value}')
            # else:
                # print('yes')

    return aligned_dff , aligned_time

def filter_dff(aligned_dff, aligned_time, led_index):

    filtered_dff = {}
    filtered_time = {}
    for roi in range(len(aligned_dff)):
        filtered_dff[roi] = {}
        filtered_time[roi] = {}
        for id in aligned_dff[roi]:
            led = led_index[id]

            dff_value = aligned_dff[roi][id][led - 4:led + 18]

            if len(dff_value) == 22:
                filtered_dff[roi][id] = dff_value
                filtered_time[roi][id] = aligned_time[id][led - 4:led + 18]

            else:
                print(id, led)
                # breakpoint()
                continue

    return filtered_dff, filtered_time

def aligned_dff_br(trials,trials_id, init_index, ending_index, sample_id):
    # spike_threshold = 3
    # constant_threshold = 0.0000000001
    id = sample_id
    aligned_time = {}
    aligned_dff = {}
    for roi in range(len(trials[id]["dff"])):
        aligned_dff[roi] = {}
    for id in trials_id:
        for roi in range(len(trials[id]["dff"])):
            dff_value = trials[id]["dff"][roi][init_index[id]: ending_index[id]]
            aligned_time[id] = trials[id]["time"][init_index[id]:ending_index[id]] - trials[id]["LED"][0]
            aligned_dff[roi][id] = dff_value
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
            stacked_dff = np.array([])
            sem_dff[roi] = []

    return average_dff, sem_dff , stacked_dff.shape[0]

def pooling_info(pooling_stack, dff_array, led_index):
    number_of_roi = 0
    number_of_trials = []
    for roi in dff_array:
        number_of_roi += 1
        number_of_trials0 = 0
        for trial_id in dff_array[roi]:
            number_of_trials0 += 1
            filtered_array = dff_array[roi][trial_id][led_index[trial_id] -4 :led_index[trial_id] + 18] 
            if len(filtered_array) == 22:
                pooling_stack.append(filtered_array)
        number_of_trials.append(number_of_trials0)
    number_of_trials = np.max(number_of_trials)
    number_of_roi = np.max(number_of_roi)
    return pooling_stack, number_of_roi, number_of_trials

def pooling(pooling_stack, dff_array, led_index):
    for roi in dff_array:
        for trial_id in dff_array[roi]:
            filtered_array = dff_array[roi][trial_id][led_index[trial_id] -4 :led_index[trial_id] + 18] 
            if len(filtered_array) == 22:
                pooling_stack.append(filtered_array)

            # else:
                # print(f'problem is with {trial_id},{dff_array[roi][trial_id][led_index[trial_id] -4] ,led_index[trial_id] + 18}, {filtered_array} ')
                # continue
    return pooling_stack

def zscore(arr):
    return (arr - np.mean(arr)) / np.std(arr)

def pooling_z(pooling_stack, dff_array, led_index):
    for roi in dff_array:
        for trial_id in dff_array[roi]:
            filtered_array = dff_array[roi][trial_id][led_index[trial_id] -4 :led_index[trial_id] + 18] 
            if len(filtered_array) == 22:
                filtered_array = (filtered_array - np.mean(filtered_array)) / np.std(filtered_array)
                pooling_stack.append(filtered_array)
    return pooling_stack

def pooling_sig(pooling_stack, dff_array, led_index, roi_indices):
    for roi in roi_indices:
        for trial_id in dff_array[roi]:
            filtered_array = dff_array[roi][trial_id][led_index[trial_id] -4 :led_index[trial_id] + 18] 
            if len(filtered_array) == 22:
                pooling_stack.append(filtered_array)
            else:
                # print(f'problem is with {trial_id}, {len(filtered_array)}')
                continue
    return pooling_stack

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
        stacked_dff = np.array([])
        sem_dff = np.array([])

    return average_dff, sem_dff , stacked_dff.shape[0]


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
        stacked_dff = np.array([])
        average_dff = np.array([])
        sem_dff = np.array([])

    return average_dff, sem_dff, stacked_dff.shape[0]

def calculate_average_over_roi_sig(aligned_dff, roi_indices):
    sig_roi_trials = {}
    for id in aligned_dff:
        averaging = []
        for roi in roi_indices:
            #calculate an average over the rois with respect to their trials.
            averaging.append(aligned_dff[roi][id])
            average = np.nanmean(averaging)


def average_over_roi(dff_dict):
    dff_list = list(dff_dict.values())
    
    # Compute mean and SEM
    avg = np.mean(dff_list, axis=0)
    sem = np.std(dff_list, axis=0) / np.sqrt(len(dff_list))
    print("0" , len(dff_list))
    
    return avg, sem


def sort_numbers_as_strings(numbers):
    source = []
    sorted_numbers = []
    sorted_str = []
    
    for item in numbers:
        source.append(int(item))
                
    sorted_numbers = quicksort(source)

    for item in sorted_numbers:
        sorted_str.append(str(item))

    return sorted_str

def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[0]
    left = [x for x in arr[1:] if x < pivot]
    right = [x for x in arr[1:] if x >= pivot]
    return quicksort(left) + [pivot] + quicksort(right)

# def fec_zero(trials):
#     valid_trials = {}
#     fec_time_0 = {}
#     fec_time = {}
#     fec = {}

#     until_led = []
#     from_led = []
    
#     # Compute segment lengths
#     for id in trials:
#         fec_time_0[id] = trials[id]["FECTimes"][:] - trials[id]["LED"][0]
#         init_index = find_index(fec_time_0[id], -100)
#         ending_index = find_index(fec_time_0[id], 600)
#         led_index = find_index(fec_time_0[id], 0)
#         until_led.append(led_index - init_index)
#         from_led.append(ending_index - led_index)

#     # Determine the most common length
#     target_until_led = max(set(until_led), key=until_led.count)  # Most frequent length
#     target_from_led = max(set(from_led), key=from_led.count)  # Most frequent length

#     for id in trials:
#         led_index = find_index(fec_time_0[id], 0)
#         fec_time[id] = fec_time_0[id][led_index - target_until_led: target_from_led + led_index]
#         fec[id] = trials[id]["FEC"][led_index - target_until_led: target_from_led + led_index]

#     return fec, fec_time, valid_trials

# def fec_zero(trials):
#     valid_trials = {}
#     fec_time_0 = {}
#     fec_time = {}
#     fec = {}

#     for id in trials:
#         fec_time_0[id] = trials[id]["FECTimes"][:] - trials[id]["LED"][0]
#         init_index = find_index(fec_time_0[id], -100)
#         ending_index = find_index(fec_time_0[id], 600)

#         segment_length = ending_index - init_index
#         if segment_length == 175:
#             fec_time[id] = fec_time_0[id][init_index:ending_index]
#             fec[id] = trials[id]["FEC"][init_index:ending_index]
#         elif segment_length == 176:
#             fec_time[id] = fec_time_0[id][max(0, init_index - 1):ending_index]
#             fec[id] = trials[id]["FEC"][max(0, init_index - 1):ending_index]
#         elif segment_length == 174:
#             fec_time[id] = fec_time_0[id][max(0, init_index + 1):ending_index]
#             fec[id] = trials[id]["FEC"][max(0, init_index + 1):ending_index]
#         else:
#             print(f"Skipping trial {id} due to unexpected segment length: {segment_length}")
#             breakpoint()
#             continue  # Skip this trial instead of breaking

#         valid_trials[id] = trials[id]

#     return fec, fec_time, valid_trials
def fec_cr_aligned(trials, cr_time):
    valid_trials = {}
    fec_time_0 = {}
    fec_time = {}
    fec_0 = {}
    fec = {}

    for i , id in enumerate(trials):
        # try:
        # if cr_time[id]:
        fec_time_0[id] = trials[id]["FECTimes"][:]
        init_index = find_index(fec_time_0[id], cr_time[id] -100)
        # ending_index = find_index(fec_time_0[id], 600)
        ending_index = init_index + 175
        fec_time[id] = fec_time_0[id][init_index:ending_index] - cr_time[id]
        fec_0[id] = trials[id]["FEC"][:]
        fec[id] = fec_0[id][init_index:ending_index]
        valid_trials[id] = trials[id]
        # else:
            # continue
        # except Exception as e:
        #     # print(f"trial {id} had fec data problems {e}")
        #     continue

    return fec,fec_time, valid_trials

def fec_zero(trials):
    valid_trials = {}
    fec_time_0 = {}
    fec_0 = {}

    for i , id in enumerate(trials):
        try:
            fec_time_0[id] = trials[id]["FECTimes"][:] - trials[id]["LED"][0]
            fec_0[id] = trials[id]["FEC"][:]
            valid_trials[id] = trials[id]
        except Exception as e:
            # print(f"trial {id} had fec data problems {e}")
            continue

    return fec_0,fec_time_0, valid_trials

def moving_average(fec_0, window_size):
    smoothed_fec = {}
    for id in fec_0:
        if window_size < 1:
            raise ValueError("Window size must be a positive integer.")    
        kernel = np.ones(window_size) / window_size
        smoothed_fec[id] = np.convolve(fec_0[id], kernel, mode='same')
    return smoothed_fec

def fec_crop(fec, time):
    cropped_fec = {}
    cropped_time = {}
    for i in fec:
        init_index = find_index(time[i], -100)
        ending_index = init_index + 175
        # ending_index = find_index(time[i], 600)
        cropped_fec[i] = fec[i][init_index:ending_index]
        cropped_time[i] = time[i][init_index:ending_index]

    return cropped_fec, cropped_time
        

def min_max_normalize(data):
    normalized_fec = {}
    for id in data:
        normalized_fec[id] = data[id] - np.min(data[id]) / np.max(data[id]) - np.min(data[id])
    return normalized_fec
