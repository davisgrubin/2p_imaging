
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

## 2025.04.30
- Added Dash interactive decoding viewer for decoder timecourse exploration.

---

## Dash interactive decoding viewer

Run an interactive dashboard to explore decoder accuracy summaries from the exported CSVs produced by `generate_session_plots.py`.

1. Install the required packages (Dash, Plotly, Pandas are needed in addition to NumPy):
   ```bash
   python -m pip install dash plotly pandas numpy
   ```
2. Launch the viewer (the default export directory is `plot_exports`):
   ```bash
   python dash_interactive_decoding_viewer.py --data-root plot_exports
   ```
3. Open the provided URL in your browser to:
   - Switch between region-, subject-, session-, or neuron-type summaries.
   - Filter by region, subject, session, neuron subset, and block type (short vs. long).
   - Toggle automatic versus manual y-axis ranges for close inspection of accuracy traces.
   - Inspect supporting counts for each aggregation level directly below the plot.
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
- Fixed coloring of decoding.
- Improved labeling for plot_win_decode.
- Improved labeling for plot_win_mag_quant_stat.
- Added hide_all_axis.
- Added baseline correction to plot_win_decode.
- Fixed plot_win_decode clustering bug.
- Fixed plot_cluster_win_mag_quant_stat clustering bug.
- Fixed plot_cluster_win_mag_quant clustering bug.
- Added fit_linear_r2.
- Added plot_cluster_win_mag_scatter.

## 2025.06.15
- Now plot_cluster_cate_fraction_in_cluster uses common maximum in set_ylim.
- Added plot_oddball_jitter_variance.
- Now plot_cluster_win_mag_scatter is none average.
- Now plot_cluster_win_mag_scatter also added distribution.

## 2025.06.16
- Added bsaeline correction as optional for quantifications.
- Removed subsammpling for fix trials.
- Added y axis to plot_cluster_win_mag_quant.
- Added scale bar for plot_cluster_win_mag_scatter.
- Added stat_sym.
- Rewritten plot_cluster_win_mag_dist with np.histogram.
- Added 0 for clustering tracesto adjust_layout_cluster_neu.
- Now quantifications have option to change average axis.
- Now plot_oddball_jitter_variability is single neuron single trial.
- Added adjust_layout_scatter.
- Adjusted position of statistics test of plot_cluster_win_mag_dist_compare.
- Added plot_dist.
- Fixed cumulative distribution.
- Removed plot_raw_traces in random.

## 2025.06.19
- Added plot_cross_epochin random.
- Adjusted plot_dist_cluster_fraction layout.
- Created fig7_3331RandomExtended.
- Now plot_stim_label accept isi_range manually.
- Now plot_win_mag_scatter can specify evaluation window.
- Changed layout of plot_dist_cluster_fraction.

## 2025.06.21
- Now plot_cross_epoch compares epoch around stimulus.
- Now get_stat_test can test variability as well.

## 2025.06.25
- Now plot_cross_epoch compares epoch for all days.
- Rewritten plot_cluster_adapt_individual with plot_cluster_adapt_all.
- Rewritten plot_tansition_trial_heatmap.
- Added axs.reverse() to all cluster plots with.
- Improved add_heatmap_colorbar layout.
- Rewritten plot_tansition.
- Added more options to get_neu_seq_trial_fix_jitter.
- Now plot_cluster_win_mag_scatter allows different win_eval.
- Added plot_win_mag_scatter_trans_oddball.

## 2025.06.26
- Set alignment window longer.
- Rewritten plot_trial_quant in plot_cluster_adapt_all.
- Adjusted plot_tansition layout.
- Separated plot_scatter.
- Removed quantification scripts.

## 2025.07.01
- Rewritten plot_interval_scaling with baseline correction.
- Added fit_poly_line.
- Added scale_bar control to plot_cluster_mean_sem.
- Rewritten plot_cross_epoch with bin interval.
- Now plot_3d_latent_dynamics has optional cmap.
- Now adjust_layout_3d_latent can specify if add colorbar.
- Added plot_standard_latent.
- Added plot_interval_scaling to plot_cluster_interval_bin_all.
- Added plot_interval_bin_latent.
- Added adjust_layout_2d_latent.

## 2025.07.06
- Fixed title labeling in plot_cluster_interval_bin_all.
- Added plot_standard_heatmap to plot_cluster_adapt_all.
- Rewritten plot_win_mag_scatter_epoch.
- Now baseline_correction is given in plot_cluster_win_mag_scatter.
- Added print info for all sub functions.
- Added cate_list and cate_gap to control plotting.
- Removed all memmap operations.

## 2025.07.10
- Now plot_win_mag_quant_win_eval can control whether plot baseline.
- Added plot_oddball_time_eclapse.
- Added multi_sess_decoding_time_eclapse.

## 2025.07.14
- Changed coloring of plot_cluster_oddball_fix_all.
- Now plot_cate_fraction can specify color.
- Now plot_cluster_oddball_fix_all colors are defined within panel.
- Added plot_oddball_fix_quant.
- Rewritten plot_oddball_latent_fix_all.
- Added plot_oddball_jitter_latent.
- Removed plot_oddball_jitter_variability.

## 2025.07.17
- Moved plot_random_bin to plot_cluster_oddball_fix_all.
- Removed plot_glm_kernel from plot_cluster_oddball_jitter_global_all.
- Now plot_glm_kernel only plots a dashline at 0.
- Added plot_isi_seting.

## 2025.07.28
- Rewritten plot_transition_latent_individual.
- Removed plot_oddball_jitter_latent.
- Now plot_oddball_latent_all plots all latent dynamics.
- Changed coloring for plot_block_win_decode in plot_cluster_oddball_jitter_global_all.
- Changed coloring for plot_oddball_jitter in plot_cluster_oddball_jitter_global_all.
- Deleted plot_oddball_jitter_latent in plot_cluster_oddball_jitter_global_all.
- Changed coloring for plot_win_mag_dist in plot_cluster_oddball_jitter_global_all.

## 2025.07.29
- Now plot_3d_latent_dynamics starts from 0.
- Now plot_3d_latent_dynamics plots critical points.
- Deleted axis limit settings in adjust_layout_3d_latent.
- Now plot_3d_latent_dynamics plots stimulus as scatter.
- Deleted plot_random_bin from plot_cluster_oddball_jitter_local_all.
- Changed coloring for plot_oddball_jitter in plot_cluster_oddball_jitter_local_all.
- Deleted plot_oddball_jitter_latent from plot_cluster_oddball_jitter_local_all.
- Added random_bin_cmap.

## 2025.07.31
- Deleted plot_heatmap_neuron_cate.
- Now apply_colormap uses default cmap.
- Deleted plot_random_bin in plot_cluster_all.
- Added interpolation and smoothing in plot_3d_latent_dynamics.
- Added plot_standard to plot_latent_individual.
- Now plot_standard in plot_cluster_heatmap_all has options to specify clustering or not.
- Now pick_trial can specify fraction of trials.
- Added heatmap_sort_frac.
- Now plot_standard uses fraction of trials for sorting.

## 2025.08.05
- Now sort_heatmap_neuron uses rastermap for sorting.
- Rewritten plot_heatmap_neuron with rastermap.
- Fixed add_mark in plot_3d_latent_dynamics.
- Now plot_3d_latent_dynamics plots stimulus as square.
- Added plot_corr_mat.

## 2025.08.08
- Adjusted layout of plot_tansition.
- Now fit_poly_line can handle nan values.
- Added plot_interval_scaling to plot_cluster_oddball_jitter_local_all.
- Added legend for plot_interval_scaling.
- Added legend for plot_block_win_decode.
- Added plot_standard to plot_cluster_oddball_jitter_global_all.
- Added plot_standard_scale to plot_cluster_all.
- Added trial correction to plot_tansition_trial_heatmap.
- Rewritten plot_cross_epoch with trial average.

## 2025.08.09
- Fixed legend of plot_cluster_oddball_jitter_local_all.
- Fixed plot_sorted_heatmaps_fix_all.
- Now clustering uses k-shape.
- Removed PCA in clsutering.
- Rewritten remap_cluster_id.
- Changed random_bin_cmap.
- Fixed plot_cross_sess_adapt legend.
- Now plot_interval_bin uses line as stimulus.
- Fixed plot_standard_scale layout problem.
- Added xlim constraint for stimulus in plot_cluster_mean_sem.
- Added plot_sorted_heatmaps_all to short long.
- Added plot_isi_example_epoch.

## 2025.08.11
- Fixed plot_cross_sess_adapt color.
- Added plot_cluster_local_all.
- Reduced max_pixel in plot_heatmap_neuron.
- Fixed plot_tansition_trial_heatmap layout.
- Decoupled trials_around and trials_eval for plot_tansition_trial_heatmap.
- Changed plot_dis_mat cmap.
- Rewritten plot_trial_corr.
- Added red line to plot_tansition_trial_heatmap.
- Removed mean correction in get_row_corr.

## 2025.08.15
- Now plot_dist_cluster_fraction averages across subjects.
- Removed plot_interval_scaling in plot_cross_sess_adapt.
- Added plot_heatmap_trial.
- Added plot_interval_heatmap.
- Removed scale factor in apply_colormap.
- Added percentile scale in norm01.
- Rewritten plot_cross_epoch.
- Added plot_cross_day.
- Now plot_standard can plot superimpose.
- Rewritten plot_latent_individual.

## 2025.08.16
- Removed epoch in plot_cross_day.
- Now plot_heatmap plots cluster heatmap.
- Added plot_standard_heatmap.
- Added get_peak_time.
- Now plot_interval_scaling uses peak magnitude evaluation.
- Fixed plot_tansition colors.

## 2025.08.19
- Added sub_sampling_trial.
- Now get_multi_sess_neu_trial has option on trial subsampling.
- Added norm_gauss.
- Deleted neu_pop_sample_decoding_slide_win.
- Added decoding_time_confusion.
- Added time_decoding_evaluation.
- Added plot_standard_time_decode.
- Adjusted layouts for random.

## 2025.08.21
- Deleted get_sub_time_idx.
- Now decoding_time_confusion also returns timestamps.
- Improved plot_standard_time_decode layouts.
- Fixed plot_standard_time_decode time bug.
- Now bin_times is a parameter in decoding_time_confusion
- Adjusted random layouts.
- Now plot_time_decode_confusion_matrix receives t_range for extent.
- Improved add_heatmap_colorbar with cax.

## 2025.08.23
- Removed plot_heatmap_trial label rotation.
- Adjusted add_heatmap_colorbar cax layouts.
- Added colorbar to plot_heatmap_trial.
- Now cmap is defined completely by base_color in get_cmap_color.
- Now add_heatmap_colorbar has norm as optional.
- Now add_heatmap_colorbar can add yticklabels.
- Changed percentile to nanpercentile in apply_colormap.
- Removed plot_latent_individual in random.
- Added colorbar for plot_heatmap_neuron.
- Fixed plot_standard_heatmap layouts.

## 2025.08.24
- Fixed plot_cluster_local_all layouts.
- Added hide_all_axis to add_heatmap_colorbar.
- Fixed plot_interval_bin_latent_all bin_num.
- Fixed plot_latent_individual layouts.
- Fixed plot_cluster_adapt_all layouts.
- Added plot_cluster_heatmap_trial colorbar.
- Fixed plot_sorted_heatmaps_all.

## 2025.08.27
- Now remap_cluster_id uses rastermap to sort cluster_id.
- Reversed cross_day_cmap.

## 2025.08.28
- Added get_random_rotate_mat_3d.
- Now all 3d dynamics has random rotation.
- Made plot_latent_all independent.
- Removed plot_fix_oddball.
- Changed fix jitter color.
- Added plot_cluster_pred_mod_index_compare.
- Rewritten plot_win_mag_quant_stat.

## 2025.08.29
- Added regression_time.
- Added plot_decode_all.
- Now sort_heatmap_neuron has multiple options for sorting.
- Now plot_standard_heatmap uses shuffle as sort_method.
- Fixed plot_interval_heatmap yticks.
- Added plot_standard_heatmap to random.
- Now decoding_time_confusion also returns overall accuracy score.
- Removed all inside tick.
- Added plot_oddball_time_decode.

## 2025.08.30
- Fixed plot_block_win_decode layouts.
- Fixed plot_standard_time_decode layouts.
- Fixed plot_heatmap_neuron yticklabels.
- Now add_heatmap_colorbar has option to ignore ax.
- Added plot_standard_ramp_params.
- Now plot_dist discard the first and last element.
- Removed plot_standard_latent.
- Removed grid line on adjust_layout_isi_example_epoch.
- Added unexpected omission to plot_tansition.
- Fixed plot_trial_quant trial number.
- Rewritten plot_trial_quant layouts.

## 2025.08.31
- Removed baseline correction in plot_block_win_decode.
- Removed calcium transient.
- Rewritten plot_sess_example_traces.
- Fixed plot_sess_example_traces.
- Added neuron subsampling to plot_standard_time_regress_drop_neu_all.
- Adjusted plot_cluster_mean_sem layouts.
- Now get_full_html will not delete temp folder.
- Removed clean_memap_path.
- Fixed plot_win_mag_quant_stat neg.
- Added plot_block_win_decode_all.
- Added plot_cluster_type_percentage.
- Removed plot_standard_time_decode.

## 2025.09.01
- Now plot_standard_time_decode plots superimpose lineas as well.
- Now plot_dist plots normalized cumulative distribution.
- Fixed plot_stim.
- Added window to plot_win_mag_quant_stat.
- Adjusted plot_time_decode_confusion_matrix layouts.
- Added plot_half_violin.
- Now plot_cluster_pred_mod_index_compare plots violin.
- Adjusted plot_pred_mod_index_box layouts.
- Added plot_cluster_stim_all.

## 2025.09.02
- Deleted plot_cluster_ca_transient.
- Deleted plot_cluster_fraction.
- Moved plot_glm_kernel to utils.
- Removed y=0 in plot_cluster_mean_sem.
- Added cluster id to plot_glm_kernel.
- Now plot_cross_day has option of scaled or not.
- Now plot_cross_epoch has option of scaled or not.
- Removed plot_oddball.
- Fixed DFF labels.
- Adjusted adjust_layout_cluster_neu layouts.
- Now plot_standard_heatmap plots both standard.
- Now plot_standard has option of scaled or not.
- Adjusted plot_standard layouts.
- Now clustering_neu_response_mode can use both kshape and kmeans.
- Rewritten run_clustering.

## 2025.09.03
- Now exception gives full error log.
- Now plot_jitter_global_oddball can plot individual or both conditions.
- Fixed stimulus in plot_latent_all.
- Adjusted plot_standard layouts.
- Added plot_standard_pred.
- Added plot_cluster_box.
- Added plot_cluster_standard_params_box.

## 2025.09.04
- Fixed plot_decode_all layouts.
- Fixed plot_cluster_adapt_all layouts.
- Now plot_3d_latent_dynamics can specify the end point color.
- Improved plot_interval_bin_latent_all layouts.
- Rewritten decoding_time_confusion.

## 2025.09.05
- Changed names of model parameters.
- Changed plot_decode_all colors.
- Changed plot_cluster_standard_time_decode layouts.
- Changed plot_cluster_standard_pred layouts.
- Rewritten plot_dist.

## 2025.09.07
- Added smoothing to decoding accuracy line.
- Fixed plot_standard_ramp_params_all range.
- Now plot_standard_ramp_params_all can plot violin.
- Adjusted plot_cluster_standard_pred layouts.
- Added plot_cluster_standard_time_decode subsampling.
- Now plot_time_decode_confusion_matrix can specify vmax.
- Removed diag plot in plot_cluster_standard_time_decode_confusion.
- Removed average across splits in decoding_time_confusion.
- Now plot_standard_ramp_params_all plots cummulative distribution.
- Added plot_neuron_fraction.

## 2025.09.09
- Now plot_standard_ramp_params_all latency plots negative.
- Now add_heatmap_colorbar plots 2 digits.
- Added plot_neu_Frac_all.
- Fixed plot_standard_heatmap colorbar.
- Changed colors in plot_decode_all.
- Set plot_cluster_standard_time_decode_single ylim.

## 2025.09.10
- Fixed y label in plot_transition_trial.
- Changed plot_cluster_standard_time_decode_confusion colorbar.
