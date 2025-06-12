
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

## 2025.03.28
- Improved memory management in fitting GLM.
- Completed basic quantification funcitons.
- Added plot_cluster_metric_box to plot metrics for clusters under conditions.
- Added plot_oddball_fix_jitter_box.
- Now get_all_metrics receives list_win_eval for different evaluation windows.
- Added plot_standard_box.
- Added plot_temporal_scaling_box.
- Deleted get_slope_speeds.

## 2025.04.02
- Now clustering_neu_response_mode has option not to collect clustering metrics.
- Improved GLM fitting efficiency with analytical solution.
- Removed all pdf file.
- Now all functions in ReadResults can specify dtype to reduce memory usage.
- Assigned all memory efficient dtype when running memory mapping.

## 2025.04.05
- Added back axis to plot_cluster_mean_sem.
- Rewritten plot_categorization_features in fig5_1451ShortLong.

## 2025.04.07
- Now one common temp folder is used.
- Now Alignment is using memory mapping with files in the temp folder.

## 2025.04.09
- Complete wavelet denoising for ramping and dropping onset detection.

## 2025.04.13
- Finalized onset detection.

## 2025.04.15
- Now get_roi_label_color accepts cate as input.
- Now get_neu_trial and get_stim_response returns post_isi as wel.
- Deleted relabeling in clustering_neu_response_mode.
- Added remap_cluster_id.
- Rewritten plot_interval_bin in 3331Random.
- Rewritten plot_interval_bin_box in 3331Random
- In quantifications added get_evoke_value.
- In quantifications added get_evoke_time.
- In quantifications added run_quantification.
- Rewritten plot_cluster_metric_box.

## 2025.04.16
- Now filter_candidate_linearity is a separate function.
- Set the default wavelet scale in get_change_onset to 64.
- Moved run_glm into utils_basic.
- Now plot_cluster_mean_sem can set stim_seq to None to avoid plotting stimulus.
- Rewritten plot_cluster_oddball_fix_all in 4131FixJitterOdd.

## 2025.04.18
- Added plot_cluster_oddball_fix_individual.
- Added plot_cluster_oddball_fix_all.
- All layout set to constrained.
- Removed plot_sorted_heatmaps_fix_jitter.d
- Added plot_sorted_heatmaps_fix_all.
- Separate plot_heatmap_neuron and plot_heatmap_neuron_cate.
- Added plot_sorted_heatmaps_fix_all.

## 2025.04.20
- Now get_cmap_color gives uniformly distributed colors.
- Added plot_oddball_latent_fix_all.
- Rewritten multi_sess_decoding_slide_win.

## 2025.04.22
- Now get_change_onset receives win_eval_c as the evaluation center and has the window predefined.
- Added run_quantification into quantifications.
- Added plot_trial_legend into intervals.
- Added get_stim_evoke_mag into quantifications.
- Added get_stim_evoke_slope into quantifications.

## 2025.04.23
- Added get_stim_evoke_latency into quantifications.
- Rewritten remap_cluster_id in clustering.
- Now run_glm is initialized and run at the beginning.
- Added get_glm_cate into generative.

## 2025.04.25
- Removed reading vol and dff independently.
- Fixed dff smoothing.
- Now temporal files are separeted into subjects.
- Now all individual plot xlim is applied before plotting.
- Added a factor in apply_colormap to remap data distribution.
- Added plot_cluster_neu_fraction_in_cluster.
- Now pack_webpage_main can be assigned pages more flexibly with lists.

## 2025.05.08
- Added pupil traces processing in ReadResults and Trialization.
- Added list_target_sess into list_target_sess and changed html_session_list.
- Improved independence of html_session_list.
- Now l_frames and r_frames are within get_stim_response.

## 2025.05.10
- Now pupil data is ready for analysis.
- Added into ShortLong.
- Improved plot_cluster_neu_fraction_in_cluster.
- Combined cluster_all_pre and cluster_all_post in 3331Random
- Removed stimulus and led traces alignment in Alignment.
- Now clean_memap_path also applies at the beginning.

## 2025.05.15
- Fixed filter_stimulus when the last trial happens to be oddball. 
- Now apply_colormap can clip data into percentile range.
- Added plot_cluster_heatmap.
- Added plot_cluster_oddball_fix_heatmap_all.
- Added plot_dendrogram.

## 2025.05.23
- Improved plot_cluster_block_adapt_individual efficiency.
- get_multi_sess_neu_trial returns std instead of sem.
- Added plot_cluster_epoch_adapt_individual.
- Added plot_oddball_win_likelihood_local.
- Added plot_oddball_win_likelihood_global.
- Added plot_trial_quant to plot_cluster_oddball_jitter_local_individual.
- Added plot_trial_quant to plot_cluster_oddball_jitter_global_individual.
- Added command line window control.

## 2025.05.28
- Now plot_oddball_win_likelihood_local and plot_oddball_win_likelihood_global use broken axis.
- Changed the decoding range of plot_oddball_win_likelihood_local and plot_oddball_win_likelihood_global.
- Now plot_oddball_win_likelihood_local and plot_oddball_win_likelihood_global is for individual clusters.
- Now plot_win_mag_quant can specify 3 windows.
- Now plot_cluster_oddball_jitter_global_individual uses mean within window for quantification.

## 2025.05.29
- Deleted metrics in clustering_neu_response_mode.
- Now clustering is run when init the plotter.
- Added log info for alignment.
- Merge plot_cluster_block_adapt_individual and plot_cluster_epoch_adapt_individual
- Added error in get_neu_trial.

## 2025.06.05
- Fixed transition heatmap labels.
- Changed transition trial quantification order.
- Changed plot_tansition range and layout.
- Now plot_cluster_oddball_jitter_individual supports both global and common modes.
- Added get_split_idx.
- Added plot_dist_cluster_fraction.
- Added plot_raw_traces.

## 2025.06.07
- Added subsampling to plot_cluster_oddball_jitter_individual.
- Added plot_win_mag_quant_win_eval.
- Added baseline and post into plot_win_mag_quant.
- Changed isi_range in plot_standard_isi_distribution.
- Removed all numbers in figure filename.

## 2025.06.10
- Added command line control.
- Added subsampling iteration for get_neu_seq_trial_fix_jitter.
- Fine tuned regularization in run_glm_multi_sess.
- Added gap control for plot_cluster_mean_sem.
- Rewritten plot_cluster_cate_fraction_in_cluster.

## 2025.06.11
- Fixed get_mean_sem_win on taking percentile values.
- Rewritten get_wilcoxon_test.
- Now get_mean_sem_win also returns data points before average.
- Now mode is defined inside plot_cluster_win_mag_quant.