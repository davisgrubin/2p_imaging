
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

## 2025.02.18
- Fixed sorted clustering heatmap scale.
- Added binned latent dynamics for clusters.

## 2025.02.19
- Now get_neu_trial also returns significance test.
- Replaced clustering heatmaps with population heatmaps.
- Now temp folder will be removed with its content.
- Now size_scale is set to 5.
- Now all functions in ReadResults are using memory mapping.
- Added memory mapping files cleaning.

## 2025.02.20
- Completed GLM full model fitting.
- Added basic GLM kernel analysis.

## 2025.02.22
- Now sort_heatmap_neuron is an independent function.
- Now add_heatmap_colorbar is an independent function.
- Beautified heatmap and added colorbar.

## 2025.02.25
- Now heatmap can specify scale.

## 2025.02.26
- Added run_wilcoxon_trial.
- Rewritten plot_sorted_heatmaps_fix_jitter.
- Now apply_colormap can handle binary heatmaps.
- Now plot_heatmap_neuron can plot binary heatmaps.
- Now plot_heatmap_neuron only excludes rows with all nan.
- Rewritten plot_sorted_heatmaps_standard.
- Added get_temporal_scaling_data.

## 2025.02.28
- Added plot_cross_sess_adapt.
- Completed run_features_categorization.
- Completed feature_categorization.
- Added get_bin_mean_sem_cluster.
- Completed plot_categorization_features for all paradigms.
- Deleted plot_interval_corr_epoch.
- Added plot_heatmap_trial.
- Added plot_block_adapatation_trial_heatmap.
- Added get_slope_speeds.
- Addded plot_slope_dist.

## 2025.03.05
- Added get_block_transition_idx.
- Now plot_block_adapatation_trial_heatmap shows trials around transition.
- Added plot_dist_cluster_fraction_in_cate.
- Now for random interval trials temporal scaling is computed before binning.

## 2025.03.15
- Now fraction of clusters in plot_cross_sess_adapt is a line plot.
- Now by default dff smoothing is set to false.
- Now trials_around is specified in plot_block_adapatation_trial_heatmap.

## 2025.03.19
- Added filter_session_config_list for session filtering.
- Now session filtering is within each visualization file.
- Now subject configs are in a separate file.
















