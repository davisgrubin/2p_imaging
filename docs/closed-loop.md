# Closed Loop

The closest closed-loop or optogenetic pilot stack in this repository is `opto_pilot/`.

This directory is more script-like than the passive and single-interval projects, but it clearly contains the building blocks for opto versus control comparisons aligned to imaging and behavior.

## Main entry points

- `opto_pilot/main_opto.py`
- `opto_pilot/modules/Trialization_Opto.py`
- `opto_pilot/modules/FlirFrames.py`
- `opto_pilot/modules/Alignment.py`
- `opto_pilot/modules/ReadResults.py`
- `opto_pilot/plot/`

## What `main_opto.py` does

`main_opto.py` is a monolithic analysis script that:

1. reads a processed session directory and restores `ops.npy`
2. loads masks, `dff`, significance labels, raw voltages, and Bpod data
3. reads or constructs `neural_trials`
4. separates control and opto trials
5. plots mean trajectories and heatmaps
6. identifies opto-responsive neurons with threshold-based heuristics
7. writes report-style figures

The code currently uses hard-coded local paths, which suggests it is intended for targeted analysis runs rather than batch execution on arbitrary datasets.

## Module roles

### `Trialization_Opto.py`

Specialized trial segmentation for optogenetic sessions, likely extending the shared trialization logic with opto-specific state handling.

### `FlirFrames.py`

Helpers for working with FLIR timing or frame alignment. This is notable because the preprocessing voltage files also preserve FLIR-related channels.

### `Alignment.py`

Reusable event-alignment helpers for constructing trial-locked neural activity sequences.

### `ReadResults.py`

Readers for masks, ΔF/F, raw voltages, ROI labels, motion offsets, and Bpod session data.

### `plot/`

Contains analysis and figure helpers for:

- basic alignments
- short/long comparisons
- raw traces
- calcium transients
- coactivation fractions
- model-style summaries

## Relationship to the shared pipeline

Like the other experiment directories, `opto_pilot/` assumes the shared preprocessing and postprocessing layers have already produced:

- Suite2p outputs
- voltage sidecars
- ROI masks and labels
- ΔF/F traces

The opto-specific work then starts from those files and focuses on trial alignment and response comparison.

## Current maturity

Compared with `passive_interval_oddball_202412/` and `single_interval_discrimination_202505/`, the closed-loop stack is less standardized:

- fewer reusable top-level config objects
- more hard-coded local paths
- more notebook or script-like execution style

That said, the core module split is already visible and would support future cleanup into a more reusable pipeline.
