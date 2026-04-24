# Postprocessing

The postprocessing layer lives primarily in `2p_post_process_module_202404/` and is designed to operate on outputs produced by the preprocessing/Suite2p stage.

## Main entry points

- `2p_post_process_module_202404/run_postprocess.py`
- `2p_post_process_module_202404/run_postprocess.sh`
- `2p_post_process_module_202404/modules/QualControlDataIO.py`
- `2p_post_process_module_202404/modules/LabelExcInh.py`
- `2p_post_process_module_202404/modules/DffTraces.py`

## Pipeline responsibilities

The current postprocessing pipeline is organized into three major module-level operations:

1. **Quality control and data organization** via `QualControlDataIO`
2. **Cross-channel ROI labeling** via `LabelExcInh`
3. **ΔF/F computation** via `DffTraces`

The top-level orchestration happens in `2p_post_process_module_202404/run_postprocess.py:33`, where `process_session()` runs the modules in this order:

- `QualControlDataIO.run(...)`
- `LabelExcInh.run(...)`
- `DffTraces.run(...)`

## Top-level script

### `run_postprocess.py`

Key functions:

#### `get_qc_args(args)`

Defined in `2p_post_process_module_202404/run_postprocess.py:12`.

Purpose:

- parses CLI threshold ranges for QC into numeric NumPy arrays

#### `read_ops(session_data_path)`

Defined in `2p_post_process_module_202404/run_postprocess.py:22`.

Purpose:

- loads `suite2p/plane0/ops.npy`
- reattaches `save_path0` to the current session directory

#### `process_session(session_data_path, args)`

Defined in `2p_post_process_module_202404/run_postprocess.py:33`.

Purpose:

- executes the full postprocessing workflow for one session

#### `main()`

Defined in `2p_post_process_module_202404/run_postprocess.py:41`.

Purpose:

- provides the command-line interface for postprocessing a single session

## Module reference

## `QualControlDataIO`

Defined in `2p_post_process_module_202404/modules/QualControlDataIO.py`.

This module is responsible for reading raw Suite2p outputs, computing ROI-level quality metrics, filtering ROIs, and saving cleaned results for downstream modules.

### Core functions

#### `read_raw(ops)`

Defined in `2p_post_process_module_202404/modules/QualControlDataIO.py:9`.

Purpose:

- loads `F.npy`, `Fneu.npy`, and `stat.npy` from Suite2p output

#### `get_metrics(ops, stat)`

Defined in `2p_post_process_module_202404/modules/QualControlDataIO.py:22`.

Purpose:

- computes QC metrics used to reject ROIs, including skew, connectivity, aspect, compactness, and footprint

#### `thres_stat(...)`

Defined in `2p_post_process_module_202404/modules/QualControlDataIO.py:41`.

Purpose:

- thresholds ROI metrics against the user-provided QC ranges
- returns the indices of ROIs to reject

#### `reset_roi(...)`

Defined in `2p_post_process_module_202404/modules/QualControlDataIO.py:72`.

Purpose:

- zeroes out rejected ROIs and keeps only the surviving fluorescence, neuropil, and `stat` entries

#### `save_qc_results(...)`

Defined in `2p_post_process_module_202404/modules/QualControlDataIO.py:90`.

Purpose:

- writes filtered outputs to `qc_results/`

Saved artifacts include:

- `fluo.npy`
- `neuropil.npy`
- `stat.npy`
- `masks.npy`
- `ops.npy`

#### `stat_to_masks(ops, stat)`

Defined in `2p_post_process_module_202404/modules/QualControlDataIO.py:103`.

Purpose:

- converts surviving Suite2p ROI pixel coordinates into a dense mask image

#### `save_move_offset(ops)`

Defined in `2p_post_process_module_202404/modules/QualControlDataIO.py:112`.

Purpose:

- writes motion correction offsets to `move_offset.h5`

#### `run(...)`

Defined in `2p_post_process_module_202404/modules/QualControlDataIO.py:121`.

Purpose:

- performs the end-to-end QC step and saves filtered outputs for downstream processing

## `LabelExcInh`

Defined in `2p_post_process_module_202404/modules/LabelExcInh.py`.

This module uses functional and anatomical channels to label ROIs, especially for distinguishing excitatory/inhibitory candidates in dual-channel recordings.

### Core functions

#### `normz(data)`

Defined in `2p_post_process_module_202404/modules/LabelExcInh.py:14`.

Purpose:

- z-score normalization helper used internally

#### `get_cellpose_model(model_type="cyto3")`

Defined in `2p_post_process_module_202404/modules/LabelExcInh.py:18`.

Purpose:

- initializes a Cellpose model with some compatibility handling for different Cellpose APIs and optional GPU usage

#### `run_cellpose(...)`

Defined in `2p_post_process_module_202404/modules/LabelExcInh.py:48`.

Purpose:

- writes the anatomical mean image to disk
- runs Cellpose segmentation
- saves Cellpose segmentation side products

#### `get_mask(ops)`

Defined in `2p_post_process_module_202404/modules/LabelExcInh.py:96`.

Purpose:

- reads QC masks and crops the corresponding functional/anatomical mean images using Suite2p ranges

#### `get_ch_traces(ops)`

Defined in `2p_post_process_module_202404/modules/LabelExcInh.py:114`.

Purpose:

- loads `F.npy` and `F_chan2.npy` for cross-channel comparisons

#### `anat_bleedthrough_correction(...)`

Defined in `2p_post_process_module_202404/modules/LabelExcInh.py:120`.

Purpose:

- estimates and applies a linear bleedthrough correction to the anatomical channel based on fluorescence traces

#### `get_label(...)`

Defined in `2p_post_process_module_202404/modules/LabelExcInh.py:155`.

Purpose:

- computes overlap between functional and anatomical masks
- assigns coarse ROI labels based on overlap strength thresholds

#### `save_masks(...)`

Defined in `2p_post_process_module_202404/modules/LabelExcInh.py:192`.

Purpose:

- writes channel-aware masks and labels to `masks.h5`

#### `run(ops, diameter)`

Defined in `2p_post_process_module_202404/modules/LabelExcInh.py:206`.

Purpose:

- handles the main ROI labeling workflow
- falls back to a trivial labeling scheme for single-channel recordings

## `DffTraces`

Defined in `2p_post_process_module_202404/modules/DffTraces.py`.

This module computes ΔF/F from QC-filtered fluorescence and neuropil traces.

### Core functions

#### `pmt_led_handler(fluo, dff)`

Defined in `2p_post_process_module_202404/modules/DffTraces.py:11`.

Purpose:

- applies PMT/LED artifact correction to opto/shutter sessions by interpolating over detected artifact periods

#### `get_dff(ops, dff, norm)`

Defined in `2p_post_process_module_202404/modules/DffTraces.py:33`.

Purpose:

- computes ΔF/F using a Gaussian-filtered baseline
- optionally z-normalizes each trace

#### `save_dff(ops, dff, fluo)`

Defined in `2p_post_process_module_202404/modules/DffTraces.py:50`.

Purpose:

- writes `dff.h5` containing both processed `dff` and raw `fluo`

#### `run(ops, norm=True, correct_pmt=False)`

Defined in `2p_post_process_module_202404/modules/DffTraces.py:57`.

Purpose:

- loads QC-filtered fluorescence and neuropil
- applies neuropil subtraction using `ops['neucoeff']`
- computes ΔF/F
- optionally runs PMT/LED correction

## Typical outputs

After postprocessing, a session directory commonly contains:

- `qc_results/`
- `masks.h5`
- `move_offset.h5`
- `dff.h5`
- Cellpose intermediate files under `cellpose/` for dual-channel data

## CLI example

Example command structure:

```bash
python run_postprocess.py \
  --session_data_path /path/to/session \
  --range_skew -5,5 \
  --max_connect 1 \
  --range_footprint 1,2 \
  --range_aspect 0,5 \
  --range_compact 0,1.06 \
  --diameter 6
```
