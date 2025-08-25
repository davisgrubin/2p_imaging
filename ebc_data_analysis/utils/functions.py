import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.gridspec import GridSpec
# import h5py
import scipy.io as sio
from scipy.signal import savgol_filter
import h5py
import traceback
from utils.alignment import find_index

def color_list(slice_i, stack_i, number_of_session_slices):
    # Calculate brightness from 1 (brightest) to 0.2 (darkest)
    brightness = np.linspace(1.0, 0.02, number_of_session_slices)[slice_i]


    if stack_i % 2 == 1:
        # Green shades: (R, G, B)
        color = (0, brightness, 0)
    else:
        # Blue shades: (R, G, B)
        color = (0, 0, brightness)

    return color

def save_trials(trials, exclude_start, exclude_end, save_path):
    trial_ids = sorted(map(int, trials.keys()))  # Ensure trial IDs are sorted

    # if len(trial_ids) == 0:
    #     raise ValueError('Empty trials')

    if exclude_start + exclude_end >= len(trial_ids) and exclude_start != 0 and exclude_end != 0 :
        raise ValueError("Exclusion range removes all trials.")

    with h5py.File(save_path, 'w') as f:
        grp = f.create_group('trial_id')

        for trial in range(exclude_start, len(trial_ids) - exclude_end):
            trial_id = str(trial_ids[trial])
            trial_group = grp.create_group(trial_id)
            for k, v in trials[trial_id].items():
                trial_group[k] = v


#for the fec only files
def processing_beh(bpod_file, save_path, exclude_start, exclude_end):
    bpod_mat_data_0 = read_bpod_mat_data(bpod_file)
    trials = trial_label_fec(bpod_mat_data_0)
    save_trials(trials, exclude_start, exclude_end, save_path)

def read_bpod_mat_data(bpod_file):
    # Load .mat file
    raw = sio.loadmat(bpod_file, struct_as_record=False, squeeze_me=True)
    try:
        raw = check_keys(raw)
    except Exception as e:
        print(e)
        breakpoint()
        # raise ValueError(f'check keys {e}')


    try:
        sleep_dep = raw['SessionData']['SleepDeprived']
        print(sleep_dep)
    except Exception as e:
        print('Sleep Deprivation data is missing', e)
        sleep_dep = 0
        # breakpoint()
    #raw = raw['SessionData']

    # Initialize variables
    trial_type = []
    trial_FEC_TIME = []
    LED_on = []
    LED_off = []
    AirPuff_on = []
    AirPuff_off = []
    Airpuff_on_post = []
    Airpuff_off_post = []
    eye_area = []
    test_type = []


    # try:
        # Loop through trials
    for i in range(raw['SessionData']['nTrials']):
        # trial_states = raw['RawEvents']['Trial'][i]['States']
        trial_data = raw['SessionData']['RawEvents']['Trial'][i]['Data']
        trial_event = raw['SessionData']['RawEvents']['Trial'][i]['Events']

        # Handle LED_on and LED_off
        if 'GlobalTimer1_Start' in trial_event:
            LED_on.append(1000 * np.array(trial_event['GlobalTimer1_Start']).reshape(-1))
        else:
            print(f'led on missing from bpod trial {i}')
            # breakpoint()
            # continue
            LED_on.append([])

        if 'GlobalTimer1_End' in trial_event:
            LED_off.append(1000 * np.array(trial_event['GlobalTimer1_End']).reshape(-1))
        else:
            print(f'led off missing from bpod trial {i}')
            # breakpoint()
            # continue
            LED_off.append([])

        # Handle AirPuff_on and AirPuff_off
        if 'GlobalTimer2_Start' in trial_event:
            AirPuff_on.append(1000 * np.array(trial_event['GlobalTimer2_Start']).reshape(-1))
        else:
            print(f'airpuff on missing from bpod trial {i}')
            # breakpoint()
            # continue
            AirPuff_on.append([])

        if 'GlobalTimer2_End' in trial_event:
            AirPuff_off.append(1000 * np.array(trial_event['GlobalTimer2_End']).reshape(-1))
        else:
            print(f'airpuff off missing from bpod trial {i}')
            # breakpoint()
            # continue
            AirPuff_off.append([])
       
        if 'FECTimes' in trial_data:
            trial_FEC_TIME.append(1000 * np.array(trial_data['FECTimes']).reshape(-1))
        else:
            print(f'FEC times missing from bpod trial {i}')
            # breakpoint()
            # continue
            trial_FEC_TIME.append([])

        # Determine trial type
        if trial_data.get('BlockType') == 'short':
            trial_type.append(1)
        else:
            trial_type.append(2)
        
        if 'eyeAreaPixels' in trial_data:
            eye_area.append(trial_data['eyeAreaPixels'])
        else:
            eye_area.append([])
            continue

        test_type.append(sleep_dep)
    # except Exception as e:
    #     print('error with n trials', e)



    # Prepare output dictionary
    bpod_sess_data = {
        'trial_types': trial_type,
        'trial_AirPuff_ON': AirPuff_on,
        'trial_AirPuff_OFF': AirPuff_off,
        'trial_LED_ON': LED_on,
        'trial_LED_OFF': LED_off,
        'eye_area': eye_area,
        'trial_FEC_TIME': trial_FEC_TIME,
        'test_type' : test_type,
    }

    return bpod_sess_data

# def trial_label_fec(bpod_sess_data):
#     valid_trials = {}  # Create a new dictionary to store only valid trials
#     #print(len(bpod_sess_data['trial_types']))
#     for i in range(len(bpod_sess_data['trial_types'])):
#         valid_trials[str(i)] = {}
#     for i in range(len(bpod_sess_data['trial_types'])):
#         # Initialize a flag to check if the trial contains invalid data
#         is_valid = True

#         # Check each field for NaN or unexpected values or are empty
#         if np.isnan(bpod_sess_data['trial_FEC_TIME'][i]).any() or bpod_sess_data['trial_FEC_TIME'][i] == []:
#             print(f"trial_FEC_TIME trial {i}")
#             is_valid = False

#         if np.isnan(bpod_sess_data['trial_LED_ON'][i]).any() or np.isnan(bpod_sess_data['trial_LED_OFF'][i]).any() or bpod_sess_data['trial_LED_OFF'][i] == []:
#             print(f"invalid trial_LED trial {i}")
#             breakpoint()
#             is_valid = False
#     
#         if not bpod_sess_data['trial_LED_ON'][i] or not bpod_sess_data['trial_LED_OFF'][i]:
#             print(f"Warning: Empty list found in trial_LED for trial {i}")
#             is_valid = False

#         if np.isnan(bpod_sess_data['trial_AirPuff_ON'][i]).any():
#             print(f"Warning: NaN found in trial_AirPuff for trial {i}")
#             is_valid = False

#         if not bpod_sess_data['trial_AirPuff_ON'][i] or not bpod_sess_data['trial_AirPuff_OFF'][i]:
#             print(f"Warning: Empty list found in air puff for trial {i}")
#             is_valid = False

#         if np.isnan(bpod_sess_data['eye_area'][i]).any():
#             print('no eye area')
#             is_valid = False

#         if np.isnan(bpod_sess_data['test_type'][i]).any():
#             print('no sd data 0')
#             is_valid = False
#     
#         if is_valid:

#             led_on_time = bpod_sess_data['trial_LED_ON'][i] 
#             led_off_time = bpod_sess_data['trial_LED_OFF'][i] 
#             airpuff_on = bpod_sess_data['trial_AirPuff_ON'][i]
#             airpuff_off = bpod_sess_data['trial_AirPuff_OFF'][i]

#             valid_trials[str(i)]['trial_type'] = bpod_sess_data['trial_types'][i]
#             valid_trials[str(i)]['LED'] = [led_on_time[0], led_off_time[0]]
#             valid_trials[str(i)]['AirPuff'] = [airpuff_on[0], airpuff_off[0]]
#             valid_trials[str(i)]['test_type'] = bpod_sess_data['test_type']
#             valid_trials[str(i)]['FEC'] = 1- ((bpod_sess_data['eye_area'][i]- np.min(bpod_sess_data['eye_area'][i])) / (np.max(bpod_sess_data['eye_area'][i]) - np.min(bpod_sess_data['eye_area'][i])))
#             valid_trials[str(i)]['FECTimes'] = bpod_sess_data['trial_FEC_TIME'][i]

#     return valid_trials



def trial_label_fec(bpod_sess_data):
    indicator = 0
    valid_trials = {} 

    for i in range(len(bpod_sess_data['trial_types'])):
        is_valid = True  
        print(i)
        print(len(bpod_sess_data['trial_types']))

        # Check for empty lists or NaN values
        if len(bpod_sess_data['trial_FEC_TIME'][i]) == 0 or np.isnan(bpod_sess_data['trial_FEC_TIME'][i]).any():
            is_valid = False
            indicator = 1

        if len(bpod_sess_data['trial_LED_ON'][i]) == 0 or len(bpod_sess_data['trial_LED_OFF'][i]) == 0 or \
           np.isnan(bpod_sess_data['trial_LED_ON'][i]).any() or np.isnan(bpod_sess_data['trial_LED_OFF'][i]).any():
            is_valid = False
            indicator = 2

        if len(bpod_sess_data['trial_AirPuff_ON'][i]) == 0:
            is_valid = False
            indicator = 30

        if len(bpod_sess_data['trial_AirPuff_OFF'][i]) == 0:
            is_valid = False
            indicator = 31
            
        if np.isnan(bpod_sess_data['trial_AirPuff_ON'][i]).any():
            is_valid = False
            indicator = 32

        if len(bpod_sess_data['eye_area'][i]) == 0 or np.isnan(bpod_sess_data['eye_area'][i]).any():
            is_valid = False
            indicator = 4

        if np.isnan(bpod_sess_data['test_type'][i]).any():
            is_valid = False
            indicator = 5

        if is_valid:
            valid_trials[str(i)] = {
                'trial_type': bpod_sess_data['trial_types'][i],
                'LED': [bpod_sess_data['trial_LED_ON'][i][0], bpod_sess_data['trial_LED_OFF'][i][0]],
                'AirPuff': [bpod_sess_data['trial_AirPuff_ON'][i][0], bpod_sess_data['trial_AirPuff_OFF'][i][0]],
                'test_type': bpod_sess_data['test_type'][i],
                'FEC': 1 - ((bpod_sess_data['eye_area'][i] - np.min(bpod_sess_data['eye_area'][i])) /
                            (np.max(bpod_sess_data['eye_area'][i]) - np.min(bpod_sess_data['eye_area'][i]))),
                'FECTimes': bpod_sess_data['trial_FEC_TIME'][i]
            }
        else:
            print(indicator)
            # breakpoint()

    return valid_trials


def processing_files(bpod_file = "bpod_session_data.mat",
                     raw_voltage_file = "raw_voltages.h5",
                     dff_file = "dff.h5",
                     save_path = 'saved_trials.h5',
                     exclude_start=20, exclude_end=20):
    bpod_mat_data_0 = read_bpod_mat_data(bpod_file)
    #processing dff from here
    dff = read_dff(dff_file)
    [vol_time,
    vol_start,
    vol_stim_vis,
    vol_img,
    vol_hifi,
    vol_stim_aud,
    vol_flir,
    vol_pmt,
    vol_led] = read_raw_voltages(raw_voltage_file)
    print('Correcting 2p camera trigger time')
    # signal trigger time stamps. = {}
    time_img, _   = get_trigger_time(vol_time, vol_img)
    # correct imaging timing.
    time_neuro = correct_time_img_center(time_img)
    # stimulus alignment.
    print('Aligning stimulus to 2p frame')
    stim = align_stim(vol_time, time_neuro, vol_stim_vis, vol_stim_vis)
    # trial segmentation.
    print('Segmenting trials')
    start, end = get_trial_start_end(vol_time, vol_start)
    neural_trials = trial_split(
        start, end,
        dff, stim, time_neuro,
        vol_stim_vis, vol_time)

    neural_trials = trial_label(neural_trials , bpod_mat_data_0)
    save_trials(neural_trials, exclude_start, exclude_end, save_path)

def trial_label(neural_trials, bpod_sess_data):
    valid_trials = {}  # Create a new dictionary to store only valid trials

    for i in range(np.min([len(neural_trials), len(bpod_sess_data['trial_types'])])):
        # Initialize a flag to check if the trial contains invalid data
        is_valid = True

        # Check each field for NaN or unexpected values
        if np.isnan(bpod_sess_data['trial_LED_ON'][i]).any() or np.isnan(bpod_sess_data['trial_LED_OFF'][i]).any():
            print(f"Warning: NaN found in trial_LED for trial {i}")
            is_valid = False
        
        if not bpod_sess_data['trial_LED_ON'][i] or not bpod_sess_data['trial_LED_OFF'][i]:
            print(f"Warning: Empty list found in trial_LED for trial {i}")
            is_valid = False
        if np.isnan(bpod_sess_data['trial_AirPuff_ON'][i]).any():
            print(f"Warning: NaN found in trial_AirPuff for trial {i}")
            is_valid = False
        if not isinstance(bpod_sess_data['trial_types'][i], int):  # Assuming trial types should be integers
            print(f"Warning: Invalid trial_type for trial {i}")
            is_valid = False
        if np.isnan(bpod_sess_data['trial_ITI'][i]).any():
            print(f"Warning: NaN found in trial_ITI for trial {i}")
            is_valid = False
        # if np.isnan(bpod_sess_data['trial_FEC'][i]).any():
        #     print(f"Warning: NaN found in trial_FEC for trial {i}")
        #     is_valid = False
        if np.isnan(bpod_sess_data['trial_FEC_TIME'][i]).any():
            print(f"Warning: NaN found in trial_FEC_TIME for trial {i}")
            is_valid = False
        if np.isnan(bpod_sess_data['eye_area'][i]).any():
            print('*')
            is_valid = False

        # Only add the trial to valid_trials if it is valid
        if is_valid:
            valid_trials[str(i)] = neural_trials[str(i)]  # Add the trial to the valid dictionary

            # Modify the 'LED' field with new logic
            led_on_time = bpod_sess_data['trial_LED_ON'][i] + neural_trials[str(i)]['vol_time'][0]
            led_off_time = bpod_sess_data['trial_LED_OFF'][i] + neural_trials[str(i)]['vol_time'][0]
            airpuff_on = bpod_sess_data['trial_AirPuff_ON'][i]+ neural_trials[str(i)]['vol_time'][0]
            airpuff_off = bpod_sess_data['trial_AirPuff_OFF'][i]+ neural_trials[str(i)]['vol_time'][0]
            valid_trials[str(i)]['LED'] = [led_on_time[0], led_off_time[0]]
            valid_trials[str(i)]['AirPuff'] = [airpuff_on[0], airpuff_off[0]]

            valid_trials[str(i)]['trial_type'] = bpod_sess_data['trial_types'][i]
            valid_trials[str(i)]['ITI'] = bpod_sess_data['trial_ITI'][i] + neural_trials[str(i)]['vol_time'][0]
            valid_trials[str(i)]['FEC'] = 1- ((bpod_sess_data['eye_area'][i]- np.min(bpod_sess_data['eye_area'][i])) / (np.max(bpod_sess_data['eye_area'][i]) - np.min(bpod_sess_data['eye_area'][i])))
            valid_trials[str(i)]['FECTimes'] = bpod_sess_data['trial_FEC_TIME'][i] + neural_trials[str(i)]['vol_time'][0]
            valid_trials[str(i)]['LED_on'] = led_on_time
            valid_trials[str(i)]['LED_off'] = led_off_time

    return valid_trials



def read_raw_voltages(voltage_file):
    f = h5py.File(voltage_file,'r')
    try:
        vol_time = np.array(f['raw']['vol_time'])
        vol_start = np.array(f['raw']['vol_start'])
        vol_stim_vis = np.array(f['raw']['vol_stim_vis'])
        vol_hifi = np.array(f['raw']['vol_hifi'])
        vol_img = np.array(f['raw']['vol_img'])
        vol_stim_aud = np.array(f['raw']['vol_stim_aud'])
        vol_flir = np.array(f['raw']['vol_flir'])
        vol_pmt = np.array(f['raw']['vol_pmt'])
        vol_led = np.array(f['raw']['vol_led'])
    except:
        vol_time = np.array(f['raw']['vol_time'])
        vol_start = np.array(f['raw']['vol_start_bin'])
        vol_stim_vis = np.array(f['raw']['vol_stim_bin'])
        vol_img = np.array(f['raw']['vol_img_bin'])
        vol_hifi = np.zeros_like(vol_time)
        vol_stim_aud = np.zeros_like(vol_time)
        vol_flir = np.zeros_like(vol_time)
        vol_pmt = np.zeros_like(vol_time)
        vol_led = np.zeros_like(vol_time)
    f.close()

    return [vol_time, vol_start, vol_stim_vis, vol_img,
            vol_hifi, vol_stim_aud, vol_flir,
            vol_pmt, vol_led]

def get_trigger_time(
        vol_time,
        vol_bin
        ):
    # find the edge with np.diff and correct it by preappend one 0.
    diff_vol = np.diff(vol_bin, prepend=0)
    idx_up = np.where(diff_vol == 1)[0]
    idx_down = np.where(diff_vol == -1)[0]
    # select the indice for risging and falling.
    # give the edges in ms.
    time_up   = vol_time[idx_up]
    time_down = vol_time[idx_down]
    return time_up, time_down

def correct_time_img_center(time_img):
    # find the frame internal.
    diff_time_img = np.diff(time_img, append=0)
    # correct the last element.
    diff_time_img[-1] = np.mean(diff_time_img[:-1])
    # move the image timing to the center of photon integration interval.
    diff_time_img = diff_time_img / 2
    # correct each individual timing.
    time_neuro = time_img + diff_time_img
    return time_neuro

def align_stim(
        vol_time,
        time_neuro,
        vol_stim_vis,
        label_stim,
        ):
    # find the rising and falling time of stimulus.
    stim_time_up, stim_time_down = get_trigger_time(
        vol_time, vol_stim_vis)
    # avoid going up but not down again at the end.
    stim_time_up = stim_time_up[:len(stim_time_down)]
    # assign the start and end time to fluorescence frames.
    stim_start = []
    stim_end = []
    for i in range(len(stim_time_up)):
        # find the nearest frame that stimulus start or end.
        stim_start.append(
            np.argmin(np.abs(time_neuro - stim_time_up[i])))
        stim_end.append(
            np.argmin(np.abs(time_neuro - stim_time_down[i])))
    # reconstruct stimulus sequence.
    stim = np.zeros(len(time_neuro))
    for i in range(len(stim_start)):
        label = label_stim[vol_time==stim_time_up[i]][0]
        stim[stim_start[i]:stim_end[i]] = label
    return stim



def get_trial_start_end(
        vol_time,
        vol_start,
        ):
    time_up, time_down = get_trigger_time(vol_time, vol_start)
    # find the impulse start signal.
    time_start = [time_up[0]]
    for i in range(len(time_up)-1):
        if time_up[i+1] - time_up[i] > 5:
            time_start.append(time_up[i])
    start = []
    end = []
    # assume the current trial end at the next start point.
    for i in range(len(time_start)):
        s = time_start[i]
        e = time_start[i+1] if i != len(time_start)-1 else -1
        start.append(s)
        end.append(e)
    return start, end



def trial_split(
        start, end,
        dff, stim, time_neuro,
        label_stim, vol_time,
        ):
    neural_trials = dict()
    for i in range(len(start)):
        neural_trials[str(i)] = dict()
        start_idx_dff = np.where(time_neuro > start[i])[0][0]
        end_idx_dff   = np.where(time_neuro < end[i])[0][-1] if end[i] != -1 else -1
        neural_trials[str(i)]['time'] = time_neuro[start_idx_dff:end_idx_dff]
        neural_trials[str(i)]['stim'] = stim[start_idx_dff:end_idx_dff]
        neural_trials[str(i)]['dff'] = dff[:,start_idx_dff:end_idx_dff]
        start_idx_vol = np.where(vol_time > start[i])[0][0]
        end_idx_vol   = np.where(vol_time < end[i])[0][-1] if end[i] != -1 else -1
        neural_trials[str(i)]['vol_stim'] = label_stim[start_idx_vol:end_idx_vol]
        neural_trials[str(i)]['vol_time'] = vol_time[start_idx_vol:end_idx_vol]
    return neural_trials

def check_keys(d):
    """Recursively converts MATLAB structs to Python dictionaries."""
    for key in d:
        if isinstance(d[key], sio.matlab.mat_struct):
            d[key] = todict(d[key])
        elif isinstance(d[key], np.ndarray):
            d[key] = tolist(d[key])
    return d

def todict(matobj):
    """Converts a MATLAB struct object to a Python dictionary."""
    d = {}
    for field in getattr(matobj, '_fieldnames', []):  # Ensure _fieldnames exists
        elem = getattr(matobj, field, None)
        if isinstance(elem, sio.matlab.mat_struct):
            d[field] = todict(elem)
        elif isinstance(elem, np.ndarray):
            d[field] = tolist(elem)
        else:
            d[field] = elem
    return d

def tolist(ndarray):
    elem_list = []
    if ndarray.ndim == 0:  # Handle 0-d arrays
        return ndarray.item()  # Convert to Python scalar
    for sub_elem in ndarray:
        if isinstance(sub_elem, sio.matlab.mat_struct):
            elem_list.append(todict(sub_elem))
        elif isinstance(sub_elem, np.ndarray):
            elem_list.append(tolist(sub_elem))
        else:
            elem_list.append(sub_elem)
    return elem_list
    # return [tolist(elem) if isinstance(elem, np.ndarray) else elem for elem in ndarray]

def roi_group_analysis(trials, trial_id, roi_group):

    group = []

    for roi in roi_group:
        group.append(trials[trial_id]["dff"][roi])
    avg = np.nanmean(group , axis=0)
    std = np.nanstd(group , axis=0)

    return avg , std

def read_dff(dff_file_path):
    window_length = 9
    polyorder = 3

    with h5py.File(dff_file_path, 'r') as f:
        dff = np.array(f['name'])

    dff = np.apply_along_axis(
        savgol_filter, 1, dff.copy(),
        window_length=window_length,
        polyorder=polyorder
    )
    
    return dff

def interval_averaging(interval):
    interval_avg = {}
    for roi in interval:
        dummy = []
        for id in interval[roi]:
            # print(interval[roi][id])
            dummy.append(interval[roi][id])
        interval_avg[roi] = np.nanmean(dummy, axis=0)
    return interval_avg

def zscore(trace):
    return (trace - np.mean(trace)) / np.std(trace)

def sig_trial_func(all_id, trials, transition_0, transition_1):

    sig_trial_ids = []
    slot_ids = []

    if isi_type(trials[all_id[0]]) == 2:
        print('first block is long')
        sig_trial_ids.append([])
        slot_ids.append([])

    for trial_id in all_id:
        try:
            next_id = int(trial_id) + 1

            while str(next_id) not in trials:
                next_id += 1

                if next_id - int(trial_id) >= 20:
                    break

            if trials[trial_id]["trial_type"][()] != trials[str(next_id)]["trial_type"][()] :
                sig_trial_ids.append(trial_id)
                slot_ids.append([str(i) for i in range(int(trial_id) - transition_0 , int(trial_id) + transition_1)])

        except Exception as e:
            print(f"Exception: {e}")
            continue

    return sig_trial_ids, slot_ids

def isi_type(trial):
    airpuff = trial["AirPuff"][0] - trial["LED"][0]
    if airpuff > (Long_airpuff_off - 10) and airpuff < (Long_airpuff_on + 10):
        trial_type = 2
    elif airpuff > (Short_airpuff_on - 10) and airpuff < (Short_airpuff_on + 10):
        trial_type = 1
    else:
        print("FATAL ERROR. The isi duration is not as expected. It is {airpuff}")
        ValueError()
        trial_type = None

    return trial_type


def sd_stat(test_type, session_date):
    if 'V_3_18' in session_date or 'V_3_19' in session_date:
        return test_type
    elif 'V_3_17' in session_date:
        if test_type == 2:
            return 1
        elif test_type == 3:
            return 2
    elif 'V_3_16' in session_date:
        if test_type == 1:
            return 3
        elif test_type == 0:
            return 1
