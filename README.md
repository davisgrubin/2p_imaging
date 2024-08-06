# Usage

Run on windows command line window:
```
python run_postprocess.py `
--session_data_path 'FN16_P_20240410_omi_t' `
--range_skew '-5,5' `
--max_connect '1' `
--max_apect '1.35' `
--range_footprint '1,2' `
--range_compact '0,1.05' `
--diameter '6' `
```

Run on linux server:
```
sbatch run_postprocess.sh
```

# Update note

## 2024.04.10
- Separated from suit2p procecssing pipeline.

## 2024.04.18
- First release.

## 2024.08.06
- Added median filtering for opto/shutter results.
