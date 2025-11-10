#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <session directory | parent directory>" >&2
  exit 1
fi

FILES_PATH="$1"

if [ ! -d "$FILES_PATH" ]; then
  echo "Error: $FILES_PATH is not a directory" >&2
  exit 1
fi

# Build the list of session directories to process.
declare -a SESSION_DIRS=()
if [ -d "$FILES_PATH/suite2p" ]; then
  SESSION_DIRS+=("$FILES_PATH")
else
  while IFS= read -r -d '' session_file; do
    if [ -d "$session_file/suite2p" ]; then
      SESSION_DIRS+=("$session_file")
    fi
  done < <(find "$FILES_PATH" -maxdepth 1 -mindepth 1 -type d -print0)
fi

if [ ${#SESSION_DIRS[@]} -eq 0 ]; then
  echo "No session directories with suite2p outputs found under $FILES_PATH" >&2
  exit 0
fi

# Run the python script for each session separately.
for session_file in "${SESSION_DIRS[@]}"; do
  echo "Processing $session_file"
  python3 /Users/davisgrubin/Downloads/TCA_Block_Transition/2p_imaging/2p_post_process_module_202404/run_postprocess.py \
    --session_data_path="$session_file" \
    --range_skew="-5.0,5.0" \
    --range_aspect="0.0,5.0" \
    --max_connect=1 \
    --range_footprint="1.0,2.0" \
    --range_compact="0,1.06" \
    --diameter=6
done
