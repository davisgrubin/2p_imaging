# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 16:54:49 2026

@author: saminnaji3
"""


import os
import numpy as np
import plotly.graph_objects as go
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

# --- Helper Functions ---

def run_interactive_spike_validation(session_path, output_dir, tau, data_date, subset=None, fs=30):
    """Generates an interactive Plotly HTML for spike validation."""
    ops = np.load(os.path.join(session_path, 'suite2p', 'plane0', 'ops.npy'), allow_pickle=True).item()
    ops['save_path0'] = session_path
    
    try:
        dff = read_dff(ops, 'manual_qc_results')
    except Exception:
        print(f"   [Error] Could not find manual_qc_results for {data_date}.")
        return

    # Run deconvolution for this specific tau
    spikes_matrix = oasis(F=dff, batch_size=ops['batch_size'], tau=tau, fs=fs)

    # Handle neuron indexing (1-based for naming, 0-based for slicing)
    total_rois = dff.shape[0]
    if subset is not None:
        neuron_indices = subset
    else:
        num_to_sample = min(20, total_rois)
        neuron_indices = np.random.choice(np.arange(total_rois), size=num_to_sample, replace=False)

    time_min = np.arange(0, dff.shape[1]) / (fs * 60)
    fig = go.Figure()
    shift, curr_max = 0, 0 

    for i in sorted(neuron_indices):
        curr_dff = dff[i, :] - np.nanmedian(dff[i, :])
        curr_spikes = spikes_matrix[i, :]
        
        # Stacking logic
        curr_min = np.nanmin(curr_dff)
        shift = shift + np.abs(curr_min) + curr_max
        curr_max = np.nanmax(curr_dff)
        
        # Plot DFF Trace
        fig.add_trace(go.Scatter(
            x=time_min, y=list(curr_dff + shift), 
            mode='lines', name=f'Neu {i}',
            line=dict(width=1, color='gray'), opacity=0.6
        ))
        
        # Superimpose Spikes
        spike_idx = np.where(curr_spikes > 1e-4)[0] # Filter very tiny noise
        if len(spike_idx) > 0:
            fig.add_trace(go.Scatter(
                x=time_min[spike_idx], y=list(curr_spikes[spike_idx] + shift),
                mode='markers', marker=dict(color='red', size=3),
                hoverinfo='text', text=[f'Neu: {i} | Amp: {a:.3f}' for a in curr_spikes[spike_idx]]
            ))

    # Layout & Buttons
    fig.update_layout(
        updatemenus=[dict(
            type="buttons", direction="right", x=0.1, y=1.15,
            buttons=[
                dict(label="Reset View", method="relayout", args=[{"xaxis.range": [0, time_min[-1]]}]),
                dict(label="Zoom 1-Min Window", method="relayout", args=[{"xaxis.range": [time_min[-1]/2, time_min[-1]/2 + 1]}])
            ]
        )],
        title=f'Spike Validation | Subject: {subject} | Date: {data_date} | Tau: {tau}',
        xaxis_title='Time (min)', yaxis_title='Stacked Activity (DFF + Spikes)',
        template='plotly_white', showlegend=False
    )
          
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    save_path = os.path.join(output_dir, f"{data_date}_tau_{tau}_validation.html")
    fig.write_html(save_path)
    print(f"   [Saved] {os.path.basename(save_path)}")

# --- Main Execution ---

if __name__ == "__main__":
    # Generate session paths
    session_paths = [os.path.join(initial_path, subject, subject_id + d) for d in data_dates]
    
    # Target output folder
    final_output_dir = os.path.join(output_base, subject, 'RawTraces', 'Tau_Validation')

    for data_date, path in zip(data_dates, session_paths):

        print(f"\nProcessing Session: {data_date}")
        
        if not os.path.exists(path):
            print(f"   [Skipping] Path not found: {path}")
            continue

        # Pick 20 random neurons once per session to compare across all Taus
        # Using a dummy check to see total neurons available
        try:
            #ops = np.load(os.path.join(path, 'suite2p', 'plane0', 'ops.npy'), allow_pickle=True).item()
            ops_path = os.path.join(path, 'suite2p', 'plane0', 'ops.npy')
            ops = np.load(ops_path, allow_pickle=True).item()
            ops['save_path0'] = path
            dff_temp = read_dff(ops, 'manual_qc_results')
            subset_neurons = np.random.choice(np.arange(dff_temp.shape[0]), size=min(20, dff_temp.shape[0]), replace=False)
        except:
            print(f"   [Error] Could not load data for session {data_date}")
            continue

        for tau in test_taus:
            run_interactive_spike_validation(
                path, 
                final_output_dir, 
                tau, 
                data_date, 
                subset=subset_neurons
            )