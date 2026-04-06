# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 17:15:00 2026
@author: saminnaji3
"""

import os
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from suite2p.extraction.dcnv import oasis
from modules.ReadResults import read_dff

# --- Configuration ---
subject = 'SA16_LG'
subject_id = 'SA16_'
initial_path = r'C:\Users\saminnaji3\Downloads'
output_base = r'C:\Users\saminnaji3\OneDrive - Georgia Institute of Technology\2p imaging\Figures'

# Active session dates
data_dates = ['20260105', '20260107', '20260106']
test_taus = [0.25, 0.32, 0.5, 0.75, 1.0, 1.5]

def run_multi_tau_validation(session_path, output_dir, taus, data_date, subset=None, fs=30):
    """Generates a single interactive HTML comparing multiple Taus per neuron."""
    ops_path = os.path.join(session_path, 'suite2p', 'plane0', 'ops.npy')
    ops = np.load(ops_path, allow_pickle=True).item()
    ops['save_path0'] = session_path
    
    try:
        # Use manual_qc_results as established in your pipeline
        dff = read_dff(ops, 'manual_qc_results') 
    except Exception as e:
        print(f"   [Error] Could not find manual_qc_results for {data_date}: {e}")
        return

    # 1. Pre-calculate spikes for all Taus using OASIS
    print(f"   Deconvolving {len(taus)} Taus...")
    spike_results = {}
    for tau in taus:
        spike_results[tau] = oasis(F=dff, batch_size=ops['batch_size'], tau=tau, fs=fs)

    # 2. Setup Plotting
    time_min = np.arange(0, dff.shape[1]) / (fs * 60)
    fig = go.Figure()
    
    # Generate a color palette for the Taus
    colors = px.colors.qualitative.Plotly
    tau_colors = {tau: colors[i % len(colors)] for i, tau in enumerate(taus)}

    shift, curr_max = 0, 0
    neuron_indices = sorted(subset) if subset is not None else np.arange(min(20, dff.shape[0]))

    for i in neuron_indices:
        curr_dff = dff[i, :] - np.nanmedian(dff[i, :])
        
        # Stacking logic based on your main script
        curr_min = np.nanmin(curr_dff)
        shift = shift + np.abs(curr_min) + curr_max
        curr_max = np.nanmax(curr_dff)
        
        # Plot DFF Trace
        fig.add_trace(go.Scatter(
            x=time_min, y=list(curr_dff + shift), 
            mode='lines', name=f'Neu {i} Trace',
            line=dict(width=1, color='rgba(150,150,150,0.4)'), 
            hoverinfo='skip', showlegend=False
        ))
        
        # 3. Layer spikes for each Tau
        for tau in taus:
            curr_spikes = spike_results[tau][i, :]
            spike_idx = np.where(curr_spikes > 1e-4)[0]
            
            if len(spike_idx) > 0:
                fig.add_trace(go.Scatter(
                    x=time_min[spike_idx], 
                    y=list(curr_spikes[spike_idx] + shift),
                    mode='markers', 
                    name=f'Tau {tau}',
                    marker=dict(color=tau_colors[tau], size=4),
                    legendgroup=f'Tau {tau}',
                    # The FIX: Cast numpy.bool_ to standard Python bool
                    showlegend=bool(i == neuron_indices[0]), 
                    hoverinfo='text',
                    text=[f'Neu: {i} | Tau: {tau} | Amp: {a:.3f}' for a in curr_spikes[spike_idx]]
                ))

    # Layout & UI
    fig.update_layout(
        updatemenus=[dict(
            type="buttons", direction="right", x=0.1, y=1.15,
            buttons=[
                dict(label="Reset View", method="relayout", args=[{"xaxis.range": [0, time_min[-1]]}]),
                dict(label="Zoom 1-Min Window", method="relayout", args=[{"xaxis.range": [time_min[-1]/2, time_min[-1]/2 + 1]}])
            ]
        )],
        title=f'Multi-Tau Comparison | Subject: {subject} | Date: {data_date}',
        xaxis_title='Time (min)', yaxis_title='Stacked Activity',
        template='plotly_white',
        legend=dict(title="Tau Values", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
          
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    save_path = os.path.join(output_dir, f"{data_date}_multi_tau_comparison.html")
    fig.write_html(save_path)
    print(f"   [Saved] {os.path.basename(save_path)}")

# --- Main Execution ---
if __name__ == "__main__":
    final_output_dir = os.path.join(output_base, subject, 'RawTraces', 'Tau_Comparison')

    for data_date in data_dates:
        print(f"\nProcessing Session: {data_date}")
        path = os.path.join(initial_path, subject, subject_id + data_date)
        
        if not os.path.exists(path):
            print(f"   [Skipping] Path not found: {path}")
            continue

        try:
            # Re-read ops to get current neuron count for random sampling
            ops = np.load(os.path.join(path, 'suite2p', 'plane0', 'ops.npy'), allow_pickle=True).item()
            ops['save_path0'] = path
            dff_temp = read_dff(ops, 'manual_qc_results')
            subset_neurons = np.random.choice(np.arange(dff_temp.shape[0]), size=min(20, dff_temp.shape[0]), replace=False)
            
            run_multi_tau_validation(path, final_output_dir, test_taus, data_date, subset=subset_neurons)
        except Exception as e:
            print(f"   [Error] Failed to process session {data_date}: {e}")