
# Update note

## 2024.07.27
- First release.

## 2024.11.05
- Update with omission codes.

## 2024.11.12
- Update with omission codes.
- Completed plot_normal_latent.
- Completed plot_normal_cluster.
- Completed plot_change_select.
- Completed plot_change_prepost.
- Completed plot_change_latent.

## 2024.11.13
- Added get_bin_stat into utils.
- Completed plot_normal_peak.
- Completed plot_change_decode.
- Added run_multi_sess_decoding_num_neu into utils.

## 2024.11.18
- Now plot_normal_cluster plots the scale traces with sem.
- Completed plot_normal_cluster.

## 2024.11.22
- Moved all machine learning codes into the modeling folder.
- Completed multi_sess_decoding_slide_win.
- Completed plot_select_decode_slide_win.
- Completed plot_select_box.
- Now plot_win_mag_box also plots the baseline.
- Now plot_change_latent plots the colormap.
- Improved plot_heatmap_neuron on labeling.
- Completed plot_normal_spectral and plot_odd_normal_spectral.
- Deleted plot_normal_component.
- Modified get_odd_stim_prepost_idx to oddball paradigm specific.
- Changed get_roi_label_color coloring.

## 2024.11.27
- Added cross cluster correlation and metrics for clustering.
- Completed plot_odd_latent.
- Completed plot_odd_prepost.
- Now decoding is based on pytorch.
- Simplified get_multi_sess_neu_trial_average.
- Corrected get_stim_labels if the last stimulus is oddball.