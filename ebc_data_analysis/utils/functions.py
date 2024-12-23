import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.gridspec import GridSpec
# import h5py
import scipy.io as sio
import h5py

def processing_files(bpod_file = "bpod_session_data.mat", 
                     raw_voltage_file = "raw_voltages.h5", 
                     dff_file = "raw_voltages.h5", 
                     save_path = 'saved_trials.h5', 
                     exclude_start=20, exclude_end=20):
    bpod_mat_data_0 = read_bpod_mat_data(bpod_file)
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
    # signal trigger time stamps.
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
        if np.isnan(bpod_sess_data['trial_AirPuff'][i]).any():
            print(f"Warning: NaN found in trial_AirPuff for trial {i}")
            is_valid = False
        if not isinstance(bpod_sess_data['trial_types'][i], int):  # Assuming trial types should be integers
            print(f"Warning: Invalid trial_type for trial {i}")
            is_valid = False
        if np.isnan(bpod_sess_data['trial_ITI'][i]).any():
            print(f"Warning: NaN found in trial_ITI for trial {i}")
            is_valid = False
        if np.isnan(bpod_sess_data['trial_FEC'][i]).any():
            print(f"Warning: NaN found in trial_FEC for trial {i}")
            is_valid = False
        if np.isnan(bpod_sess_data['trial_FEC_TIME'][i]).any():
            print(f"Warning: NaN found in trial_FEC_TIME for trial {i}")
            is_valid = False

        # Only add the trial to valid_trials if it is valid
        if is_valid:
            valid_trials[str(i)] = neural_trials[str(i)]  # Add the trial to the valid dictionary

            # Modify the 'LED' field with new logic
            led_on_time = bpod_sess_data['trial_LED_ON'][i] + neural_trials[str(i)]['vol_time'][0]
            led_off_time = bpod_sess_data['trial_LED_OFF'][i] + neural_trials[str(i)]['vol_time'][0]
            valid_trials[str(i)]['LED'] = [led_on_time[0], led_off_time[0]]

            valid_trials[str(i)]['AirPuff'] = bpod_sess_data['trial_AirPuff'][i] + neural_trials[str(i)]['vol_time'][0]
            valid_trials[str(i)]['trial_type'] = bpod_sess_data['trial_types'][i]
            valid_trials[str(i)]['ITI'] = bpod_sess_data['trial_ITI'][i] + neural_trials[str(i)]['vol_time'][0]
            valid_trials[str(i)]['FEC'] = bpod_sess_data['trial_FEC'][i]
            valid_trials[str(i)]['FECTimes'] = bpod_sess_data['trial_FEC_TIME'][i] + neural_trials[str(i)]['vol_time'][0]
            valid_trials[str(i)]['LED_on'] = led_on_time
            valid_trials[str(i)]['LED_off'] = led_off_time
            
            print(f"Data for trial {i} has been added.")
        else:
            print(f"Skipping trial {i} due to invalid data.")

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

def save_trials(neural_trials, exclude_start, exclude_end, save_path):
    f = h5py.File(save_path, 'w')
    grp = f.create_group('trial_id')
    trial_ids = list(map(int, neural_trials.keys()))  # Get all valid trial IDs as integers

    for trial in range(len(trial_ids)):  # Iterate through valid trial IDs
        if trial > exclude_start and trial < len(trial_ids) - exclude_end:
            trial_id = str(trial_ids[trial])  # Convert back to string for dictionary access
            if trial_id in neural_trials:  # Check if the trial ID exists in the dictionary
                trial_group = grp.create_group(trial_id)
                for k in neural_trials[trial_id].keys():
                    trial_group[k] = neural_trials[trial_id][k]
    f.close()

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
    for key in d:
        if isinstance(d[key], sio.matlab.mat_struct):
            d[key] = todict(d[key])
    return d

def todict(matobj):
    d = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, sio.matlab.mat_struct):
            d[strg] = todict(elem)
        elif isinstance(elem, np.ndarray):
            d[strg] = tolist(elem)
        else:
            d[strg] = elem
    return d

def tolist(ndarray):
    elem_list = []
    for sub_elem in ndarray:
        if isinstance(sub_elem, sio.matlab.mat_struct):
            elem_list.append(todict(sub_elem))
        elif isinstance(sub_elem, np.ndarray):
            elem_list.append(tolist(sub_elem))
        else:
            elem_list.append(sub_elem)
    return elem_list

import scipy.io as sio
import numpy as np

def read_bpod_mat_data(bpod_file):
    # Load .mat file
    raw = sio.loadmat(bpod_file, struct_as_record=False, squeeze_me=True)
    raw = check_keys(raw)['SessionData']
    
    # Initialize variables
    trial_type = []
    trial_delay = []
    trial_AirPuff = []
    trial_LED = []
    trial_ITI = []
    trial_FEC = []
    trial_FEC_TIME = []
    LED_on = []
    LED_off = []
    
    # Loop through trials
    for i in range(raw['nTrials']):
        trial_states = raw['RawEvents']['Trial'][i]['States']
        trial_data = raw['RawEvents']['Trial'][i]['Data']
        trial_event = raw['RawEvents']['Trial'][i]['Events']
        
        # Handle trial_LED
        if 'LED_Onset' in trial_states:
            trial_LED.append(1000 * np.array(trial_states['LED_Onset']).reshape(-1))
        else:
            trial_LED.append([])
        
        # Handle LED_on and LED_off
        if 'GlobalTimer1_Start' in trial_event:
            LED_on.append(1000 * np.array(trial_event['GlobalTimer1_Start']).reshape(-1))
        else:
            LED_on.append([])
        if 'GlobalTimer1_End' in trial_event:
            LED_off.append(1000 * np.array(trial_event['GlobalTimer1_End']).reshape(-1))
        else:
            LED_off.append([])
        
        # Handle trial_AirPuff
        if 'AirPuff' in trial_states:
            trial_AirPuff.append(1000 * np.array(trial_states['AirPuff']).reshape(-1))
        else:
            trial_AirPuff.append([])
        
        # Handle trial_ITI
        if 'ITI' in trial_states:
            trial_ITI.append(1000 * np.array(trial_states['ITI']).reshape(-1))
        else:
            trial_ITI.append([])
        
        # Handle trial_FEC and trial_FEC_TIME
        if 'FEC' in trial_data:
            trial_FEC.append(np.array(trial_data['FEC']).reshape(-1))
        else:
            trial_FEC.append([])
        if 'FECTimes' in trial_data:
            trial_FEC_TIME.append(1000 * np.array(trial_data['FECTimes']).reshape(-1))
        else:
            trial_FEC_TIME.append([])
        
        # Determine trial type
        if trial_data.get('BlockType') == 'short':
            trial_type.append(1)
        else:
            trial_type.append(2)
        
        # Handle trial_delay
        if 'AirPuff_OnsetDelay' in trial_data:
            trial_delay.append(1000 * np.array(trial_data['AirPuff_OnsetDelay']))
        else:
            trial_delay.append(None)
    
    # Prepare output dictionary
    bpod_sess_data = {
        'trial_types': trial_type,
        'trial_delay': trial_delay,
        'trial_AirPuff': trial_AirPuff,
        'trial_LED': trial_LED,
        'trial_LED_ON': LED_on,
        'trial_LED_OFF': LED_off,
        'trial_ITI': trial_ITI,
        'trial_FEC': trial_FEC,
        'trial_FEC_TIME': trial_FEC_TIME
    }

    return bpod_sess_data



########
def indexing_time(value1 , value2 , time):
    for i in range (len(time)):
        if float(time[i]) < value1:
            if float(time[i]) > value1 - 100:
                index_1 = i

    for j in range (len(time)):
        if float(time[j]) < value2:
            if float(time[j]) > value2 - 100:
                index_2 = j
    return index_1, index_2

########
def roi_group_analysis(trials, trial_id, roi_group):

    group = []

    for roi in roi_group:
        group.append(trials[trial_id]["dff"][roi])
    avg = np.nanmean(group , axis=0)
    std = np.nanstd(group , axis=0)

    return avg , std

def read_dff(dff_file_path):
    f = h5py.File(dff_file_path)
    dff = np.array(f['name'])
    f.close()
    return dff