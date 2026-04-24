# `2p_imaging`

Welcome to the documentation site for the `2p_imaging` repository.

This documentation is intended to give a project-level overview of the codebase and provide an entry point for understanding how the major pipelines are organized. The initial focus is on the shared pipeline layers:

- **Preprocessing**: Suite2p-oriented ingestion, parameter setup, and raw voltage/session sidecar handling
- **Postprocessing**: ROI quality control, channel-based ROI labeling, and ΔF/F extraction

Experiment-specific sections are included in the navigation so the documentation structure is stable, but those pages are intentionally placeholders for now.

## Repository structure

At a high level, the repository mixes shared pipeline components with experiment-specific downstream analysis projects.

- `2p_processing_pipeline_202401/`: preprocessing entry points and Suite2p configuration
- `2p_post_process_module_202404/`: postprocessing modules applied after Suite2p
- `passive_interval_oddball_202412/`: passive visual experiment analysis
- `2p_2AFC_double_block_version/` and `2p_2AFC_reg_version/`: 2AFC-related analysis
- `joystick_basic_202304/` and `JoystickProcessing2026/`: joystick-related analysis stacks
- `ebc_basic_202410/` and `ebc_data_analysis/`: eyeblink conditioning analysis

## Documentation scope

The current version of this documentation site populates:

- `Preprocessing`
- `Postprocessing`

The following sections are present as placeholders only:

- `Passive`
- `2AFC`
- `Closed Loop`
- `Joystick`
- `EBC`

## Local preview

This docs site is configured for MkDocs using the Read the Docs theme.

To preview locally from the repository root:

```bash
python -m pip install mkdocs
mkdocs serve
```

To build the static site:

```bash
mkdocs build
```

To deploy to GitHub Pages:

```bash
mkdocs gh-deploy
```

## GitHub Pages hosting

This repository also includes a GitHub Actions workflow for static hosting via GitHub Pages:

- workflow file: `.github/workflows/docs.yml`
- build system: `mkdocs build`
- published artifact: the generated `site/` directory

To enable hosting on GitHub without your own server:

1. Push the workflow and docs files to GitHub
2. Open `Settings` → `Pages`
3. Set the source to `GitHub Actions`

After that, pushes to `main` will automatically rebuild and deploy the docs site.
