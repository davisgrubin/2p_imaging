# Preprocessing

The preprocessing layer lives primarily in `2p_processing_pipeline_202401/` and is responsible for configuring and launching Suite2p, organizing voltage recordings, and preserving session metadata needed downstream.

## Main entry points

- `2p_processing_pipeline_202401/run_suite2p_pipeline.py`
- `2p_processing_pipeline_202401/run_suite2p_pipeline.sh`
- `2p_processing_pipeline_202401/config_neuron.json`
- `2p_processing_pipeline_202401/config_neuron_1chan.json`
- `2p_processing_pipeline_202401/config_dendrite.json`

## Pipeline responsibilities

The preprocessing script performs four main steps:

1. Load a Suite2p parameter template based on the requested target structure
2. Override key runtime parameters from command-line arguments
3. Read and binarize raw voltage recordings, then save them to `raw_voltages.h5`
4. Copy the Bpod session `.mat` file into the processed output directory and run Suite2p

The core workflow is launched at the bottom of `2p_processing_pipeline_202401/run_suite2p_pipeline.py:200`:

- parse CLI arguments
- call `set_params(args)`
- call `process_vol(args)`
- call `move_bpod_mat(args)`
- call `run_s2p(ops=ops, db=db)`

## Key functions

### `set_params(args)`

Defined in `2p_processing_pipeline_202401/run_suite2p_pipeline.py:24`.

Purpose:

- selects a JSON config template based on `target_structure`
- flattens the config into a Suite2p `ops` dictionary
- injects runtime parameters such as input path, output path, channel count, and functional channel
- constructs the `db` dictionary required by `suite2p.run_s2p`

Important inputs:

- `--target_structure`
- `--data_path`
- `--save_path`
- `--nchannels`
- `--functional_chan`
- `--denoise`
- `--spatial_scale`

Important outputs:

- `ops`: runtime Suite2p parameter dictionary
- `db`: Suite2p database/input descriptor

### `process_vol(args)`

Defined in `2p_processing_pipeline_202401/run_suite2p_pipeline.py:70`.

Purpose:

- discovers the voltage CSV file in the session directory
- reads analog input channels into NumPy arrays
- thresholds relevant channels into binary signals
- saves the resulting signals into `raw_voltages.h5`

Recorded channels include:

- `vol_start`
- `vol_stim_vis`
- `vol_hifi`
- `vol_img`
- `vol_stim_aud`
- `vol_flir`
- `vol_pmt`
- `vol_led`
- `vol_2p_stim`

The saved HDF5 layout is documented directly in the `save_vol()` helper inside `process_vol()` around `2p_processing_pipeline_202401/run_suite2p_pipeline.py:161`.

### `move_bpod_mat(args)`

Defined in `2p_processing_pipeline_202401/run_suite2p_pipeline.py:192`.

Purpose:

- copies the Bpod session `.mat` file from the raw acquisition directory into the output directory as `bpod_session_data.mat`

This is important because downstream trialization and behavior alignment code frequently assumes this file exists next to the processed imaging outputs.

## Inputs and outputs

### Inputs

Typical preprocessing inputs include:

- raw imaging data directory
- voltage CSV exported by the acquisition system
- a single Bpod session `.mat` file
- a config JSON defining Suite2p defaults

### Outputs

Typical outputs include:

- Suite2p outputs under the configured results directory
- `raw_voltages.h5`
- `bpod_session_data.mat`
- `ops.npy` and other Suite2p-generated files

## Configuration files

The preprocessing layer relies on JSON configuration templates:

- `2p_processing_pipeline_202401/config_neuron.json`
- `2p_processing_pipeline_202401/config_neuron_1chan.json`
- `2p_processing_pipeline_202401/config_dendrite.json`

These define the baseline Suite2p settings used for different acquisition/ROI detection modes. Runtime arguments then override selected fields.

## CLI example

Example from the project README:

```bash
python run_suite2p_pipeline.py \
  --denoise 0 \
  --spatial_scale 1 \
  --data_path /path/to/raw/session \
  --save_path ./results/session_name \
  --nchannels 2 \
  --functional_chan 2 \
  --target_structure neuron
```
