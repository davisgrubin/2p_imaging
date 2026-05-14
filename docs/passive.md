# Passive

The main passive-analysis stack in this repository is `passive_interval_oddball_202412/`.

This project is organized around passive visual interval paradigms such as:

- `3331Random`
- `1451ShortLong`
- `4131FixJitterOdd`
- `3331RandomExtended`

## Main entry points

- `passive_interval_oddball_202412/main.py`
- `passive_interval_oddball_202412/run_main.sh`
- `passive_interval_oddball_202412/session_configs.py`
- `passive_interval_oddball_202412/modules/`
- `passive_interval_oddball_202412/modeling/`
- `passive_interval_oddball_202412/webpage/`

## High-level workflow

`main.py` is the orchestration entry point. Its `run(session_config_list, cate_list)` function does the following:

1. Expand the selected subject-level configuration into a list of session result directories.
2. Read `ops.npy` for each session via `modules.ReadResults.read_ops`.
3. Optionally trialize each session via `modules.Trialization`.
4. Run visualization modules for the supported paradigms.
5. Package outputs into an HTML report with `webpage.pack_webpage_main`.

The script currently mixes production-style code with commented toggles for enabling or disabling individual analyses. That is typical of this repository: the project is a working lab analysis stack, not a strict library API.

## Directory roles

### `modules/`

This folder contains the low-level readers and alignment or trialization helpers that turn processed session outputs into per-trial analysis inputs.

Common responsibilities include:

- reading `dff.h5`, masks, labels, and motion offsets
- reading and unpacking Bpod session data
- constructing `neural_trials.h5`
- cleaning temporary memmap files

### `modeling/`

This folder holds the heavier computational analyses:

- clustering
- decoding
- GLM and generative modeling
- quantification helpers
- shared project utilities

The update history in `README.md` shows this layer has grown to include cross-session clustering, temporal scaling, latent dynamics, decoding, quantification, onset detection, and pupil-trace support.

### `visualization*.py`

These files are paradigm-specific report builders:

- `visualization1_FieldOfView.py`
- `visualization2_3331Random.py`
- `visualization3_1451ShortLong.py`
- `visualization4_4131FixJitterOdd.py`
- `visualization5_3331RandomExtended.py`

Each file typically assembles figures for one experiment family using `plot/`, `modeling/`, and `modules/`.

### `webpage/`

This folder contains the HTML assembly layer used to publish session reports:

- CSS and JavaScript generators
- HTML fragments for logos, menus, and session lists
- `pack_webpage_main.py` for final report assembly

This is why many passive outputs in this repo are webpage-based rather than static PDF figures.

## Session configuration pattern

`session_configs.py` stores subject-level configuration dictionaries that group sessions and map raw session names to paradigm labels such as `random`, `short_long`, or `fix_jitter_odd`.

At runtime, `main.py` flattens these grouped configs into:

- `list_session_name`
- `list_session_data_path`
- one output HTML target per subject or pooled report

## Typical inputs and outputs

Inputs:

- postprocessed session directories under `results/`
- `ops.npy`
- `dff.h5`
- mask and label files
- Bpod and voltage-derived trial metadata

Outputs:

- `neural_trials.h5` and related cached intermediates
- per-paradigm figure panels
- one assembled HTML report in `results/`

## Related files

The passive project also contains notebooks and helper scripts for focused analyses:

- `quick_start.ipynb`
- `image_alignment.ipynb`
- `run_spike_inference.py`
- `summarize_image_change_decoders.py`
