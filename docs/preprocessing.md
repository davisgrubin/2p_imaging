# Preprocessing

## Step 1: Prep

The Prep step organizes the non-imaging session inputs so that Suite2p can run on a standardized session directory.

Core responsibilities:

- load the requested Suite2p config template
- set runtime parameters such as `data_path`, `save_path`, `nchannels`, and `functional_chan`
- read the voltage recording CSV
- threshold and save voltage channels into `raw_voltages.h5`
- copy the matching Bpod session `.mat` file into the output directory as `bpod_session_data.mat`

The relevant functions in `run_suite2p_pipeline.py` are:

- `set_params(args)`
- `process_vol(args)`
- `move_bpod_mat(args)`

### Sample raw session directory

A typical dual-channel raw session directory on shared storage looks like:

```text
/storage/cedar/cedar0/cedarp-fnajafi3-0/2p_imaging/MC11_VT/MC11_20260330_2afc_PFC-1377/
├── MC11_20260330_2afc_PFC-1377_Cycle00001_Ch1_000001.ome.tif
├── MC11_20260330_2afc_PFC-1377_Cycle00001_Ch1_000002.ome.tif
├── ...
├── MC11_20260330_2afc_PFC-1377_Cycle00001_Ch1_000024.ome.tif
├── MC11_20260330_2afc_PFC-1377_Cycle00001_Ch2_000001.ome.tif
├── MC11_20260330_2afc_PFC-1377_Cycle00001_Ch2_000002.ome.tif
├── ...
├── MC11_20260330_2afc_PFC-1377_Cycle00001_Ch2_000024.ome.tif
├── MC11_20260330_2afc_PFC-1377_Cycle00001_VoltageRecording_001.csv
└── MC11_VT_single_interval_discrimination_... .mat
```

Here:

- `Ch1` and `Ch2` are two measured imaging channels from the same recording session
- `functional_chan=2` in the standard two-channel config means channel 2 is treated as the functional channel for ROI detection and trace extraction
- the voltage CSV records stimulus, timing, and synchronization signals
- the Bpod `.mat` file records trial structure and behavioral metadata

### Example session scale

Using the dual-channel `MC11_20260330_2afc_PFC-1377` session as a concrete reference:

- imaging field size: `512 x 512`
- frame rate: `30.0 Hz`
- total frames processed: `96087`
- total recording duration: about `3202.9 s` or `53.4 min`
- raw file count: `24` channel-1 OME-TIFFs and `24` channel-2 OME-TIFFs

At `512 x 512`, `16-bit`, and two channels, a session of this length is on the order of:

- about `46.9 GiB` per channel uncompressed
- about `93.8 GiB` total across both channels uncompressed

Actual on-disk size depends on acquisition/export settings and TIFF compression, but this is the right order of magnitude for a roughly 53 minute two-channel session.

### Prep outputs

Prep writes or prepares:

- `raw_voltages.h5`
- `bpod_session_data.mat`
- the session output directory and Suite2p config state used in the next step

### `raw_voltages.h5`

This file is created by `process_vol(args)` and contains one HDF5 group named `raw`.

Typical structure:

```text
raw_voltages.h5
└── /raw
    ├── vol_time
    ├── vol_start
    ├── vol_stim_vis
    ├── vol_hifi
    ├── vol_img
    ├── vol_stim_aud
    ├── vol_flir
    ├── vol_pmt
    ├── vol_led
    └── vol_2p_stim
```

Fields:

| Field | Meaning | Typical datatype |
|---|---|---|
| `vol_time` | Voltage sample timestamps in milliseconds | numeric 1D array |
| `vol_start` | Bpod trial start signal | numeric 1D array, binarized to `0/1` |
| `vol_stim_vis` | Visual stimulus or photodiode sync signal | numeric 1D array, binarized to `0/1` |
| `vol_hifi` | HIFI BNC output | numeric 1D array, binarized to `0/1` |
| `vol_img` | 2p microscope imaging trigger | numeric 1D array, binarized to `0/1` |
| `vol_stim_aud` | Audio waveform signal | numeric 1D array |
| `vol_flir` | FLIR camera output | numeric 1D array, binarized to `0/1` |
| `vol_pmt` | PMT shutter signal | numeric 1D array, binarized to `0/1` |
| `vol_led` | LED signal | numeric 1D array, binarized to `0/1` |
| `vol_2p_stim` | 2p stimulation signal | numeric 1D array |

The values are stored as numeric HDF5 datasets. Signals that are thresholded in `process_vol(args)` are converted to binary `0/1` series before saving.

### `bpod_session_data.mat`

This file is copied by `move_bpod_mat(args)` from the raw session directory into the processed session directory under a standardized name.

Expected top-level structure:

- top-level MATLAB variable: `SessionData`

Common fields expected by downstream readers:

| Field | Meaning | Typical datatype |
|---|---|---|
| `SessionData.nTrials` | Number of trials in the session | scalar integer |
| `SessionData.TrialStartTimestamp` | Trial start times | numeric vector |
| `SessionData.TrialEndTimestamp` | Trial end times | numeric vector |
| `SessionData.TrialTypes` | Trial condition labels | numeric vector |
| `SessionData.BlockTypes` | Block labels when present | numeric vector or field absent |
| `SessionData.RawEvents` | Per-trial state and event structure | MATLAB struct |
| `SessionData.RawEvents.Trial[i].States` | State timing information for trial `i` | MATLAB struct |
| `SessionData.RawEvents.Trial[i].Events` | Event timing information for trial `i` | MATLAB struct |

The exact field set depends on the experiment protocol, but the standard filename and top-level `SessionData` struct are what downstream code assumes.

## Step 2: Suite2p

After Prep has standardized the session inputs, Suite2p performs the actual image registration, ROI detection, and trace extraction.

Core responsibilities:

- read the dual-channel OME-TIFF acquisition
- perform registration and motion correction
- detect ROIs on the functional channel
- extract raw ROI fluorescence and neuropil traces
- save run metadata and per-ROI statistics for downstream modules

The core call is:

```python
run_s2p(ops=ops, db=db)
```

### Key runtime parameters

The parameters most relevant for understanding the preprocessing stage are:

| Parameter | Meaning | Typical datatype |
|---|---|---|
| `target_structure` | Which JSON template to load, e.g. `neuron` or `dendrite` | string |
| `data_path` | Path to the raw session directory | string |
| `save_path` | Path to the processed output directory | string |
| `nchannels` | Number of recorded imaging channels | integer |
| `functional_chan` | Channel used for ROI detection and functional traces | integer |
| `align_by_chan` | Channel used for image alignment | integer |
| `denoise` | Suite2p denoising flag | integer or boolean-like flag |
| `spatial_scale` | Characteristic ROI scale | integer |

In the standard two-channel configuration used here:

- `nchannels = 2`
- `functional_chan = 2`
- `align_by_chan = 1`

### Sample processed directory after Suite2p

```text
SESSION_OUTPUT/
├── raw_voltages.h5
├── bpod_session_data.mat
└── suite2p/
    └── plane0/
        ├── F.npy
        ├── Fneu.npy
        ├── F_chan2.npy
        ├── stat.npy
        ├── iscell.npy
        └── ops.npy
```

### Main Suite2p output files

| File | Meaning | Typical datatype |
|---|---|---|
| `suite2p/plane0/F.npy` | Raw fluorescence traces for all detected ROIs on the functional channel | NumPy `float32` array, shape `(n_rois, n_frames)` |
| `suite2p/plane0/Fneu.npy` | Neuropil traces for the same ROIs | NumPy `float32` array, shape `(n_rois, n_frames)` |
| `suite2p/plane0/F_chan2.npy` | Second-channel fluorescence values associated with the same ROI set | NumPy `float32` array, shape `(n_rois, n_frames)` |
| `suite2p/plane0/stat.npy` | Per-ROI metadata and geometry | NumPy object array of Python dicts |
| `suite2p/plane0/iscell.npy` | Suite2p cell classifier output | NumPy `float64` array, shape `(n_rois, 2)` |
| `suite2p/plane0/ops.npy` | Full Suite2p run configuration and summary metadata | NumPy object array containing one Python dict |

### `F.npy` and `Fneu.npy`

For the reference session `MC11_20260330_2afc_PFC-1377`:

- `F.npy`: shape `(647, 96087)`, dtype `float32`
- `Fneu.npy`: shape `(647, 96087)`, dtype `float32`
- `F_chan2.npy`: shape `(647, 96087)`, dtype `float32`

Interpretation:

- rows correspond to detected ROIs
- columns correspond to imaging frames across the session

### `stat.npy`

`stat.npy` is a NumPy object array where each entry is a dictionary describing one ROI.

Common fields seen in this repo include:

| Field | Meaning | Typical datatype |
|---|---|---|
| `xpix` | X pixel indices belonging to the ROI | integer array |
| `ypix` | Y pixel indices belonging to the ROI | integer array |
| `lam` | Pixel weights within the ROI | float array |
| `med` | ROI center or median location | numeric vector |
| `npix` | ROI pixel count | integer |
| `footprint` | ROI footprint metric used in QC | float |
| `aspect_ratio` | ROI elongation metric used in QC | float |
| `compact` | ROI compactness metric used in QC | float |
| `skew` | ROI skewness metric used in QC | float |
| `radius` | Effective ROI radius | float |
| `std` | ROI trace variability summary | float |
| `neuropil_mask` | Pixels used for neuropil estimation | integer or boolean array |
| `overlap` | Whether ROI overlaps neighboring ROIs | boolean-like or integer flag |

These ROI-level statistics are what the QC step later filters on.

### `iscell.npy`

`iscell.npy` is a two-column numeric array:

| Column | Meaning | Typical datatype |
|---|---|---|
| column `0` | Cell or non-cell decision | float64, usually `0` or `1` |
| column `1` | Classifier confidence or score | float64 |

This file is saved by Suite2p but is not the only QC decision used in this repo; the downstream QC step applies additional morphology-based filtering.

### `ops.npy`

`ops.npy` is a single Python dictionary saved in NumPy format.

For the reference session, some representative fields are:

| Field | Meaning | Typical datatype |
|---|---|---|
| `nchannels` | Number of imaging channels | integer |
| `functional_chan` | Functional imaging channel index | integer |
| `align_by_chan` | Alignment channel index | integer |
| `fs` | Imaging frame rate in Hz | float |
| `tau` | Calcium decay timescale parameter | float |
| `Ly`, `Lx` | Image height and width | integers |
| `neucoeff` | Neuropil subtraction coefficient | float |
| `xoff`, `yoff` | Frame-wise rigid motion offsets | NumPy `int32` arrays |
| `meanImg` | Mean image for the functional channel | NumPy `float32` array |
| `meanImg_chan2` | Mean image for the second channel | NumPy `float32` array |
| `max_proj` | Maximum projection image | NumPy `float32` array |
| `filelist` | Raw TIFF files that were processed | Python list of strings |
| `frames_per_file` | Frame counts per input file block | NumPy `int64` array |

Downstream modules rely heavily on `ops.npy` to recover session geometry, channel identity, frame counts, motion offsets, and file provenance.

## Summary

Prep standardizes the raw session inputs by organizing the voltage, behavioral, and configuration information needed to run Suite2p. Suite2p then performs registration, ROI detection, and fluorescence extraction, producing the raw ROI-level imaging outputs that are cleaned and refined in the postprocessing stage.
