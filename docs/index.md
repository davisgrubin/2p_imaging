# `2p_imaging`

This site documents the major analysis stacks in the `2p_imaging` repository.

The repo is not a single polished package. It is a collection of shared preprocessing layers plus experiment-specific downstream projects. The docs therefore focus on:

- where each pipeline lives
- which scripts act as entry points
- what intermediate files are expected between stages
- how the experiment-specific directories are organized

## 2p Data Preparation Workflow

- `2p_processing_pipeline_202401/`: Suite2p-oriented preprocessing, voltage extraction, and session metadata handoff
- `2p_post_process_module_202404/`: ROI QC, cross-channel labeling, and `dff.h5` generation

Most downstream projects assume these outputs already exist in each session directory:

- `suite2p/plane0/ops.npy`
- `raw_voltages.h5`
- `bpod_session_data.mat`
- `masks.h5`
- `dff.h5`
- often `neural_trials.h5` after trialization

## Experiment families

- `passive_interval_oddball_202412/`: passive visual oddball and interval paradigms with HTML report generation
- `2p_2AFC_double_block_version/`, `2p_2AFC_reg_version/`, `single_interval_discrimination_202505/`: 2AFC and related decision-task analyses
- `opto_pilot/`: optogenetic or closed-loop pilot analyses with opto/control comparisons
- `joystick_basic_202304/` and `JoystickProcessing2026/`: joystick behavior and imaging analyses
- `ebc_basic_202410/` and `ebc_data_analysis/`: eyeblink conditioning pipelines and exploratory analyses

## Local preview

Install the docs dependencies and start a local preview from the repository root:

```bash
python -m pip install -r requirements-docs.txt
mkdocs serve
```

To build the site without serving it:

```bash
mkdocs build
```

## GitHub Pages deployment

The repository includes a GitHub Actions workflow at `.github/workflows/docs.yml` that builds the site with MkDocs and deploys the generated `site/` directory to GitHub Pages.
