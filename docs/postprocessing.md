# Postprocessing

The postprocessing layer lives primarily in `2p_post_process_module_202404/` and is designed to operate on outputs produced by the preprocessing or Suite2p stage.

## Main entry points

- `2p_post_process_module_202404/run_postprocess.py`
- `2p_post_process_module_202404/run_postprocess.sh`
- `2p_post_process_module_202404/modules/QualControlDataIO.py`
- `2p_post_process_module_202404/modules/LabelExcInh.py`
- `2p_post_process_module_202404/modules/DffTraces.py`

## Pipeline responsibilities

The current postprocessing layer is organized into three major steps:

1. Quality control and data organization via `QualControlDataIO`
2. Cross-channel ROI labeling via `LabelExcInh`
3. `dff.h5` computation via `DffTraces`

The top-level orchestration lives in `run_postprocess.py`, where `process_session()` runs those modules in sequence.

## `run_postprocess.py`

Important responsibilities:

- parse QC thresholds from the command line
- read `suite2p/plane0/ops.npy`
- reattach `save_path0` to the current session directory
- execute the full postprocessing workflow for one or more sessions

The README update notes indicate that the postprocessing layer has evolved over time to support:

- batch processing of session lists
- improved PMT or shutter artifact handling
- default smoothing during ΔF/F preparation
- separation of raw `dff` saving from downstream filtering

## `QualControlDataIO`

This module is responsible for reading Suite2p outputs, computing ROI-level QC metrics, filtering ROIs, and saving cleaned results for downstream modules.

Saved artifacts include:

- `qc_results/fluo.npy`
- `qc_results/neuropil.npy`
- `qc_results/stat.npy`
- `qc_results/masks.npy`
- `qc_results/ops.npy`
- `move_offset.h5`

The QC metrics described in the code and README include skew, connectivity, aspect ratio, footprint, and compactness thresholds.

## `LabelExcInh`

This module handles channel-aware ROI labeling, especially for dual-channel recordings.

Major tasks:

- load QC masks and cropped reference images
- optionally run Cellpose on the anatomical channel
- estimate bleedthrough from functional to anatomical channels
- compare overlap between functional and anatomical masks
- save labeled masks to `masks.h5`

For single-channel recordings, the module falls back to a simpler labeling path.

## `DffTraces`

This module converts QC-filtered fluorescence and neuropil signals into ΔF/F traces.

Major tasks:

- apply PMT or LED artifact handling where needed
- compute a baseline-normalized trace
- optionally normalize traces
- save the resulting data to `dff.h5`

The later experiment directories generally treat `dff.h5` as the main starting point for trialization and plotting.

## CLI example

```bash
python run_postprocess.py \
  --session_data_path /path/to/session \
  --range_skew -5,5 \
  --max_connect 1 \
  --range_aspect 1,1.35 \
  --range_footprint 1,2 \
  --range_compact 0,1.05 \
  --diameter 6
```
