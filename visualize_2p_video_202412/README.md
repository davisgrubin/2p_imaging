# Usage

Download ffmpeg into the project folder from https://github.com/FFmpeg/FFmpeg/tags.

Run on windows command line window:
```
cd C:\Users\yhuang887\Projects\visualize_2p_video_202412

python get_raw.py `
--target_frame 5000 `
--target_rois '[12,65,2,46,11,51,10,56,26,56]' `
--roi_labels '[1,1,1,0,0,0,0,0,0,0]' `
--img_path './test' `
--suite2p_path './test/FN14_PPC_20241209_seq1131_t' `
--n_channels 2 `

python ./SRDTrans/train.py `
--GPU 0 `

python ./SRDTrans/test.py `
--GPU 0 `

python get_videos.py `
--labels '[exc, inh]' `
```

# Update note

## 2024.12.17
- First release.
