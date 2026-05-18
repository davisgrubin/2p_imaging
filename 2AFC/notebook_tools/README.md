# `notebook_tools`

This folder consolidates helper functions that were previously defined inline across the notebooks in `2AFC/Test_pilot`.

## Categories

- `io.py`
  - Shared I/O, voltage, mask, trial-save, and session-path helpers
  - Implementations migrated from `Test_pilot/test_nb_io.py` and `Test_pilot/test_nb_session_paths.py`
- `alignment.py`
  - Shared alignment functions like `trim_seq`, `get_perception_response`, and `get_lick_response`
  - Implementation migrated from `Test_pilot/test_nb_alignment.py`
- `stim.py`
  - Stimulus-response plotting and pooling helpers
  - Implementation migrated from `Test_pilot/test_nb_stim_analysis.py`
- `clustering.py`
  - Clustering helpers and `NeuralActivityClustering`
  - Implementation migrated from `Test_pilot/test_nb_clustering.py`
- `decoding.py`
  - Decoder helpers and decision-decoding entry points
  - Implementation migrated from `Test_pilot/test_nb_decoding.py`
- `alignments.py`
  - Extra alignment-notebook utilities extracted from `test_alignemnts*.ipynb`
- `glm.py`
  - GLM design-matrix, fitting, and kernel-clustering helpers from `test_GLM.ipynb`
- `lfads.py`
  - LFADS export helper from `test_LFAD.ipynb`
- `tca.py`
  - Block-transition/TCA helper from `test_TCA_block_transition.ipynb`

## Source notebooks covered

- `Test_pilot/test.ipynb`
- `Test_pilot/test_GLM.ipynb`
- `Test_pilot/test_LFAD.ipynb`
- `Test_pilot/test_TCA_block_transition.ipynb`
- `Test_pilot/test_alignemnts - Copy.ipynb`
- `Test_pilot/test_alignemnts.ipynb`
- `Test_pilot/test_alignemnts_V3.ipynb`
- `Test_pilot/test_licking.ipynb`

## Notes

- Some interactive notebook utilities were cleaned up slightly to use explicit parameters instead of notebook globals.
- `notebook_tools/` is now the source of truth for migrated notebook helper code; `Test_pilot/test_nb_*.py` are legacy copies.
