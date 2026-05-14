# EBC

The repository contains two EBC-related codebases:

- `ebc_basic_202410/`
- `ebc_data_analysis/`

## `ebc_basic_202410/`

This directory follows the same broad structure as several other experiment stacks in the repo:

- `modules/`: `Trialization.py`, `Alignment.py`, `ReadResults.py`, `StatTest.py`
- `plot/`: figure helpers for masks, perception-aligned responses, motor-aligned responses, model summaries, raw traces, and behavior
- `visualization_VIPG8.py`, `visualization_VIPTD_G8.py`, `visualization_L7G8.py`: report builders for different cohorts

Unlike the passive and single-interval stacks, `main.py` is currently a script-style analysis notebook in Python form rather than a reusable orchestrator. It performs:

1. direct loading of one hard-coded session path
2. voltage and `dff` reading
3. trial alignment for conditions such as `LED`, `AirPuff`, and `ITI`
4. ROI-mask plotting
5. large per-ROI report generation

That means `ebc_basic_202410/` contains reusable modules, but its top-level execution path still looks exploratory.

## `ebc_data_analysis/`

This is a broader analysis workspace around EBC data rather than a single report pipeline.

It includes:

- preprocessing helpers such as `2p_preprocess.py` and `beh_preprocess.py`
- session and summary scripts such as `session_analysis.py`, `summary.py`, `br_summary.py`, and `adaptation_summary.py`
- transition-focused analyses such as `transition.py`, `transition_meta.py`, and `check_transition.py`
- shared utilities in `utils/`
- plotting support in `plotting/`

This directory appears to be the more flexible, script-driven environment for exploratory or aggregated EBC analysis across sessions and subjects.

## Shared inputs

Both EBC codebases depend on the usual upstream outputs from the shared imaging pipeline:

- processed Suite2p session directories
- ROI masks and labels
- `dff.h5`
- raw voltage timing
- MATLAB session metadata where applicable

## When to use which

- Use `ebc_basic_202410/` when you need the figure-oriented modules and trial-alignment logic for the established 2024 EBC workflow.
- Use `ebc_data_analysis/` when you need broader exploratory analysis, summary scripts, or custom transition and adaptation analyses.
