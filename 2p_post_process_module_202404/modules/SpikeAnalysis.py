import numpy as np
import pandas as pd
# Ensure this is the correct import for your SpikeDeconv module
from modules import SpikeDeconv


def corrected_sample_var(std, n):
    """
    Apply Bessel's correction to the sample variance.
    """
    return np.square(std) * (n / (n - 1))


def pooled_variance(neuron_data, n):
    """
    Compute the pooled variance for a single tau across neurons.

    Args:
        neuron_data (pd.Series): Series containing standard deviations for neurons of a specific tau.
        n (int): Sample size per neuron.

    Returns:
        float: The pooled variance across neurons for this tau.
    """
    numerator = ((neuron_data ** 2) * (n - 1)).sum()
    denominator = len(neuron_data) * (n - 1)
    return numerator / denominator if denominator != 0 else 0


def aggregate_statistics(df, n):
    """
    Averages the statistical properties across all neurons for each tau.

    Args:
        df (pd.DataFrame): DataFrame containing neuron statistics for each tau.
        n (int): Sample size per neuron.

    Returns:
        pd.DataFrame: A DataFrame containing aggregated statistics per tau.
    """
    agg_results = []

    for tau, group in df.groupby('tau'):
        agg_stats = {
            'tau': tau,
            'agg_mean': group['mean'].mean(),
            'agg_max': group['max'].mean(),
            'agg_min': group['min'].mean(),
            'agg_med': group['median'].mean(),
            'agg_total_spike': group['total_spikes'].mean(),
            # Compute pooled standard deviation across neurons for this tau
            'pooled_std': np.sqrt(pooled_variance(group['std'], n))
        }
        agg_results.append(agg_stats)

    return pd.DataFrame(agg_results)


def compute_statistics(spikes):
    """
    Compute statistical properties of the spike traces.

    Args:
        spikes (np.array): Array of spike traces.

    Returns:
        dict: A dictionary containing statistical properties.
    """
    stats = {
        'mean': np.mean(spikes, axis=1),
        'std': np.std(spikes, axis=1),
        'max': np.max(spikes, axis=1),
        'min': np.min(spikes, axis=1),
        'median': np.median(spikes, axis=1),
        'total_spikes': np.sum(spikes, axis=1)
    }
    return stats


def analyze_spike_traces(ops, dff, tau_spike_dict, neurons=np.arange(100), n=333):
    """
    Analyze spike traces for different tau values and save results to a CSV file.

    Args:
        ops (dict): Operation parameters.
        dff (np.array): DF/F data array.
        taus (list): List of tau values for OASIS deconvolution.
        neurons (list): List of neuron indices to analyze.
        n (int): Sample size for all neurons.
    """
    results = []

    for tau in tau_spike_dict.keys():
        print(f'Analyzing spikes for tau = {tau}')
        # smoothed, spikes = SpikeDeconv.run(
        #     ops, dff, oasis_tau=tau, neurons=neurons)

        # Compute statistics for the spikes
        spikes = tau_spike_dict[tau]
        stats = compute_statistics(spikes)

        # Store results with corresponding tau
        for i, neuron in enumerate(neurons):
            result = {
                'tau': tau,
                'neuron': neuron,
                'mean': stats['mean'][i],
                'std': stats['std'][i],
                'max': stats['max'][i],
                'min': stats['min'][i],
                'median': stats['median'][i],
                'total_spikes': stats['total_spikes'][i]
            }
            results.append(result)

    # Convert results to DataFrame
    df_n = pd.DataFrame(results)
    df_agg = aggregate_statistics(df_n, n)

    # Save per-neuron results to CSV
    df_n.to_csv('spike_analysis_results.csv', index=False)

    # Save aggregated results across neurons for each tau to a CSV
    df_agg.to_csv('aggregated_analysis_results.csv', index=False)
    print('Spike analysis results saved to spike_analysis_results.csv')
