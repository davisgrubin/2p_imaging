
# Update note

## 2024.07.27
- First release.

## 2024.10.19
- Rewritten to incorporate multiple sessions
- Added function to show used sessions for alignment.
- Moved read_ops and reset_significance to ReadResults.
- Now data reading reads multiple sessions into lists.
- Now plot_significance and plot_roi_significance take list data as input.
- Added tqdm for alignment.
- Used np.searchsorted to improve efficiency for voltage alignment.