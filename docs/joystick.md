# Joystick

The repository contains both an older joystick analysis stack and a newer reorganization:

- `joystick_basic_202304/`
- `JoystickProcessing2026/`

## `joystick_basic_202304/`

This is the older, report-oriented joystick project.

Main components:

- `modules/`: `Trialization.py`, `Alignment.py`, `ReadResults.py`, `StatTest.py`
- `plot/`: field-of-view, perception, motor-alignment, model, raw-trace, and behavior figures
- `visualization_VIPG8.py`, `visualization_VIPTD_G8.py`, `visualization_L7G8.py`: subject or cohort-specific report builders

The update notes in `README.md` show that this stack supports:

- significance testing
- ROI plots
- short versus long block comparisons
- outcome discrimination
- block and transient decoding
- joystick trajectory analyses
- pooled session summaries

This project follows the same general pattern seen elsewhere in the repo:

1. read processed imaging outputs
2. trialize and align the session
3. compute significance labels
4. generate experiment-specific figure panels

## `JoystickProcessing2026/`

This newer directory is split by pipeline stage rather than by one monolithic analysis script.

Subdirectories:

- `SessionsTrialization/`: session-level trialization
- `InitialPlotting/`: early analysis and plotting scripts
- `PostProcessing/`: later postprocessing, spike-analysis summaries, and tau-selection utilities

Representative entry points include:

- `JoystickProcessing2026/SessionsTrialization/main.py`
- `JoystickProcessing2026/InitialPlotting/main.py`
- `JoystickProcessing2026/InitialPlotting/main_laptop.py`
- `JoystickProcessing2026/PostProcessing/main.py`
- `JoystickProcessing2026/PostProcessing/run_postprocess.py`
- `JoystickProcessing2026/PostProcessing/run_manual_postprocess.py`

This separation suggests a gradual migration away from the older one-directory project style toward more explicit stages.

## Shared assumptions

Both joystick stacks assume upstream data from the common preprocessing and postprocessing layers, especially:

- `ops.npy`
- motion-correction outputs
- ROI masks and labels
- `dff.h5`
- voltage recordings and Bpod metadata

## Practical use

Use `joystick_basic_202304/` if you need the established figure-generation code for existing joystick experiments.

Use `JoystickProcessing2026/` if you are working with the newer reorganized workflow and want the staged scripts for trialization, initial plotting, and postprocessing.
