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

1. Load a Suite2p parameter template based on the requested target structure.
2. Override key runtime parameters from command-line arguments.
3. Read and binarize raw voltage recordings, then save them to `raw_voltages.h5`.
4. Copy the Bpod session `.mat` file into the processed output directory and run Suite2p.

The core workflow is launched at the bottom of `2p_processing_pipeline_202401/run_suite2p_pipeline.py`:

- parse CLI arguments
- call `set_params(args)`
- call `process_vol(args)`
- call `move_bpod_mat(args)`
- call `run_s2p(ops=ops, db=db)`

## Key functions

### `set_params(args)`

Purpose:

- selects a JSON config template based on `target_structure`
- flattens the config into a Suite2p `ops` dictionary
- injects runtime parameters such as input path, output path, channel count, and functional channel
- constructs the `db` dictionary required by `suite2p.run_s2p`

Important arguments:

- `--target_structure`
- `--data_path`
- `--save_path`
- `--nchannels`
- `--functional_chan`
- `--denoise`
- `--spatial_scale`

### `process_vol(args)`

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

### `move_bpod_mat(args)`

Purpose:

- copies the Bpod session `.mat` file from the raw acquisition directory into the output directory as `bpod_session_data.mat`

This handoff matters because many downstream `Trialization` and `ReadResults` modules assume the MATLAB session file exists beside the processed imaging outputs.

## Inputs and outputs

Typical inputs:

- raw imaging data directory
- voltage CSV exported by the acquisition system
- one Bpod session `.mat` file
- one config JSON defining Suite2p defaults

Typical outputs:

- Suite2p outputs under the configured result directory
- `raw_voltages.h5`
- `bpod_session_data.mat`
- `ops.npy` and related Suite2p artifacts

## Configuration templates

The JSON templates define baseline Suite2p settings for different acquisition modes:

- `config_neuron.json`
- `config_neuron_1chan.json`
- `config_dendrite.json`

Runtime arguments override selected fields without requiring manual editing of the templates.

## CLI example

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
