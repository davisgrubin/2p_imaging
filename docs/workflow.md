# 2p Data Preparation Workflow

This section groups the shared upstream imaging pipeline that most downstream projects in this repository depend on.

## Scope

The data preparation workflow is split into two stages:

- `Preprocessing`: Suite2p-oriented ingestion, voltage extraction, and session metadata handoff
- `Postprocessing`: ROI quality control, channel-aware labeling, and `dff.h5` generation

## Expected handoff files

Most downstream analysis projects assume these files already exist in each processed session directory:

- `suite2p/plane0/ops.npy`
- `raw_voltages.h5`
- `bpod_session_data.mat`
- `masks.h5`
- `dff.h5`
- often `neural_trials.h5` after trialization

Use the next two pages for the actual preprocessing and postprocessing details.
