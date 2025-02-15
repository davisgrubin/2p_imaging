
# Update note

## 2025.01.13
- First release.

## 2025.01.19
- Added basic analysis for 1451ShortLong.

## 2025.01.24
- Now all figures are webpage based.

## 2025.01.25
- Now pooling subject is allowed.
- Added .sh file for PACE support.

## 2025.01.26
- Added basic analysis for 4131FixJitterOdd.

## 2025.01.31
- Now all clustering are done on all conditions.

## 2025.02.01
- Now get_neu_trial receives cate as a list.

## 2025.02.05
- Added surgery window picture and note picture into webpage.
- Combined all calcium transient computation into get_ca_transient in utils.
- Merged plot_ca_transient for general calcium transient analysis.
- Rewritten plot_sess_example_traces accordingly.
- Added calcium transient for all clusters.

## 2025.02.06
- Eliminated axis for cluster average and added scalebars.
- Eliminated axis for masks.

## 2025.02.08
- Added labels in get_neu_trial.
- Added label fraction for neuron categories in plot_cluster_info.

## 2025.02.10
- Added time normalization wrapping for clsuters.
- Added cluster heatmap.
- Combined all functions for subtypes in plot_main.
- Changed canvas creation with loop.
- Now read_dff can use both raw and smooth dff.

## 2025.02.14
- Now run_clustering is an independent function.
- Added cluster heatmap sorting.
- Added sorted cluster heatmaps.

## 2025.02.16
- Moved dff filtering into ReadResults.
- Now time normalizing analysis plots before/within/after interval responses.