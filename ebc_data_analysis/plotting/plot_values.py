import numpy as np

def compute_fec_CR_data(base_line_avg, CR_interval_avg, CR_stat):
    # Initialize arrays for absolute values, relative changes, and baselines
    cr_amplitudes = []
    cr_relative_changes = []
    baselines = []

    # Calculate baseline, absolute CR amplitudes, and relative changes
    for id in base_line_avg:
        baseline = base_line_avg[id]
        cr_amplitude = abs(CR_interval_avg[id])
        cr_relative_change = CR_interval_avg[id] - baseline

        baselines.append(baseline)
        cr_amplitudes.append(cr_amplitude)
        cr_relative_changes.append(cr_relative_change)

    # Convert lists to numpy arrays
    baselines = np.array(baselines)
    cr_amplitudes = np.array(cr_amplitudes)
    cr_relative_changes = np.array(cr_relative_changes)

    # Separate data based on CR+ and CR-
    baselines_crp = []
    baselines_crn = []
    cr_amplitudes_crp = []
    cr_amplitudes_crn = []
    cr_relative_changes_crp = []
    cr_relative_changes_crn = []

    for id in base_line_avg:
        baseline = base_line_avg[id]
        cr_amplitude = abs(CR_interval_avg[id])
        cr_relative_change = CR_interval_avg[id] - baseline

        if CR_stat[id] == 1:  # CR+
            baselines_crp.append(baseline)
            cr_amplitudes_crp.append(cr_amplitude)
            cr_relative_changes_crp.append(cr_relative_change)
        elif CR_stat[id] == 0:  # CR-
            baselines_crn.append(baseline)
            cr_amplitudes_crn.append(cr_amplitude)
            cr_relative_changes_crn.append(cr_relative_change)

    # Convert lists to numpy arrays
    baselines_crp = np.array(baselines_crp)
    baselines_crn = np.array(baselines_crn)
    cr_amplitudes_crp = np.array(cr_amplitudes_crp)
    cr_amplitudes_crn = np.array(cr_amplitudes_crn)
    cr_relative_changes_crp = np.array(cr_relative_changes_crp)
    cr_relative_changes_crn = np.array(cr_relative_changes_crn)

    # Combine CR+ and CR- data
    all_baselines = np.concatenate([baselines_crp, baselines_crn])
    all_relative_changes = np.concatenate([cr_relative_changes_crp, cr_relative_changes_crn])

    # Return all data as a dictionary
    return {
        'baselines': baselines,
        'cr_amplitudes': cr_amplitudes,
        'cr_relative_changes': cr_relative_changes,
        'baselines_crp': baselines_crp,
        'baselines_crn': baselines_crn,
        'cr_amplitudes_crp': cr_amplitudes_crp,
        'cr_amplitudes_crn': cr_amplitudes_crn,
        'cr_relative_changes_crp': cr_relative_changes_crp,
        'cr_relative_changes_crn': cr_relative_changes_crn,
        'all_baselines': all_baselines,
        'all_relative_changes': all_relative_changes
    }

def compute_fec_averages(short_CRp_fec, short_CRn_fec, long_CRp_fec, long_CRn_fec, fec_time_0, shorts, longs, trials):
    # Compute mean and standard error for Short Trials
    mean1_short = np.mean(short_CRp_fec, axis=0)
    std1_short = np.std(short_CRp_fec, axis=0) / np.sqrt(len(short_CRp_fec))
    mean0_short = np.mean(short_CRn_fec, axis=0)
    std0_short = np.std(short_CRn_fec, axis=0) / np.sqrt(len(short_CRn_fec))
    id_short = shorts[0]

    # Compute mean and standard error for Long Trials
    mean1_long = np.mean(long_CRp_fec, axis=0)
    std1_long = np.std(long_CRp_fec, axis=0) / np.sqrt(len(long_CRp_fec))
    mean0_long = np.mean(long_CRn_fec, axis=0)
    std0_long = np.std(long_CRn_fec, axis=0) / np.sqrt(len(long_CRn_fec))
    id_long = longs[0]

    # Return the computed values as a dictionary
    return {
        "short_trials": {
            "mean1": mean1_short,
            "std1": std1_short,
            "mean0": mean0_short,
            "std0": std0_short,
            "id": id_short,
            "time": fec_time_0[id_short],
            "led": trials[id_short]["LED"],
            "airpuff": trials[id_short]["AirPuff"]
        },
        "long_trials": {
            "mean1": mean1_long,
            "std1": std1_long,
            "mean0": mean0_long,
            "std0": std0_long,
            "id": id_long,
            "time": fec_time_0[id_long],
            "led": trials[id_long]["LED"],
            "airpuff": trials[id_long]["AirPuff"]
        }
    }


def compute_trial_averages(
    n_value_p, n_value_n, aligned_time, crp_avg, crp_sem, crn_avg, crn_sem, x_min, x_max):
    time = aligned_time[list(aligned_time.keys())[0]]
    mask = (time >= x_min) & (time <= x_max)

    crp_filtered = crp_avg[mask]
    crp_sem_filtered = crp_sem[mask]
    crn_filtered = crn_avg[mask]
    crn_sem_filtered = crn_sem[mask]

    global_y_min = min(
        np.min(crp_filtered - crp_sem_filtered), np.min(crn_filtered - crn_sem_filtered)
    )
    global_y_max = max(
        np.max(crp_filtered + crp_sem_filtered), np.max(crn_filtered + crn_sem_filtered)
    )

    return time, crp_filtered, crn_filtered, crp_sem_filtered, crn_sem_filtered, global_y_min, global_y_max
