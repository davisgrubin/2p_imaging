import numpy as np
import pandas as pd
# Ensure this is the correct import for your SpikeDeconv module
from modules import SpikeDeconv


def corrected_sample_var(std, n):
    return np.square(std) * (n / (n-1))


def aggregate_statistics(results):
    """
    Averages the statistical properties across all neurons.

    Args:
        results (list): A list of dictionaries containing spike statistics.

    Returns:
        dict: A dictionary containing the averaged statistics.
    """
    # Convert results to a DataFrame for easier manipulation
    df = pd.DataFrame(results)

    # Group by tau and compute the mean for each statistic
    agg_df = df.groupby('tau').reset_index()

    # simply average additive stats
    agg_stats = {
        'agg_mean': agg_df['mean'].mean(),
        'agg_max': agg_df['max'].mean(),
        'agg_min': agg_df['min'].mean(),
        'agg_med': agg_df['median'].mean(),
        'agg_total_spike': agg_df['total_spikes'].mean()
    }
    # for std, we aggregate the statistics across neurons by computing
    # pooled variance.
    agg_stats['std'] = agg_df['std'].apply(
        lambda row: corrected_sample_var(row, 333)).pow(2).mean()

    return pd.DataFrame(agg_stats)


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


def analyze_spike_traces(ops, dff, tau_spike_dict, neurons=np.arange(100)):
    """
    Analyze spike traces for different tau values and save results to a CSV file.

    Args:
        ops (dict): Operation parameters.
        dff (np.array): DF/F data array.
        taus (list): List of tau values for OASIS deconvolution.
        neurons (list): List of neuron indices to analyze.
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
    df_agg = aggregate_statistics(results)

    # Save per-neuron results to CSV
    df_n.to_csv('spike_analysis_results.csv', index=False)

    # save results averaged across each neuron for a particular tau value
    df_agg.to_csv('aggregated_analysis_results.csv', index=False)
    print('Spike analysis results saved to spike_analysis_results.csv')
