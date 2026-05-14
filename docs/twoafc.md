# 2AFC

The repository contains multiple 2AFC-related analysis stacks rather than one unified package.

The main directories are:

- `2p_2AFC_reg_version/`
- `2p_2AFC_double_block_version/`
- `single_interval_discrimination_202505/`

## Common structure

The older `2p_2AFC_*` directories share a similar layout:

- `Main.py` as the top-level orchestration script
- `Modules/` for readers, trialization, clustering, and decoding
- `Plot/` for field-of-view and event-aligned figures

Common module names include:

- `reader.py`
- `trialize.py`
- `clustering_neurons.py`
- `decoding.py`

The workflow in both older variants is broadly:

1. Read one or more processed session directories.
2. Load masks and mean images.
3. Trialize the session to produce event-aligned neural data.
4. Read `neural_trials` and optional trial labels.
5. Generate per-session and pooled figures.
6. Optionally run clustering or decoding analyses.

## `2p_2AFC_reg_version/`

This looks like the earlier, simpler analysis variant.

`Main.py`:

- defines a hard-coded list of session paths
- loads masks and field-of-view images
- trializes each session
- plots licking and event-aligned neural responses
- performs pooled clustering and decoding when multiple sessions are provided

The main pooled analysis hooks are:

- `Modules.clustering_neurons.clustering_GMM`
- `Modules.decoding.decoder_decision`
- `Plot.plot_licking_neural_response`
- `Plot.plot_events_neural_response`

## `2p_2AFC_double_block_version/`

This is the more feature-rich 2AFC branch currently present in the repo.

Compared with the earlier version, it adds:

- additional plotting modules such as `plot_epoch_response.py` and `plot_licking.py`
- alternate decoding implementations including `decoding_four.py` and `decoding_multiple_settings.py`
- explicit handling of block structure, epoch summaries, and licking-pattern outputs

`Main.py` follows the same broad pattern as the earlier version, but the analysis surface is larger and more configurable.

## `single_interval_discrimination_202505/`

This directory is a newer project with a different organization:

- `main.py` orchestrates the analysis
- `session_configs.py` defines subject-level grouping
- `modules/` contains trialization, significance testing, alignment, and readers
- `visualization*.py` splits reporting into field-of-view, behavior, perception, and decision sections
- `plot/` contains figure-specific helpers

Its `run(session_config_list)` function performs:

1. session config expansion
2. optional trialization
3. optional significance testing
4. report generation through the visualization modules
5. final packaging through `webpage.pack_webpage_main`

This makes it structurally closer to the passive project than to the older `2p_2AFC_*` scripts.

## Inputs and outputs

Across all 2AFC-related projects, the expected upstream inputs are generally:

- postprocessed session folders
- mask and ROI label files
- `dff.h5`
- voltage-derived event timing
- Bpod behavioral metadata

Typical outputs include:

- `neural_trials.h5`
- pooled trial labels
- field-of-view summaries
- event-aligned neural and licking figures
- clustering or decoding summaries
- in the newer stack, HTML reports

## Practical note

Most 2AFC entry scripts currently rely on hard-coded session paths or imported session config blocks. They are best treated as lab analysis drivers that you edit for a specific dataset, not as stable general-purpose CLIs.
