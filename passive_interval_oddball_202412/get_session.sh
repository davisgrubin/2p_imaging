#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 2 ]; then
  echo "Usage: $0 <source_path> <destination_path> [destination_endpoint]" >&2
  exit 1
fi

# Define the source and destination endpoint IDs.
# Source defaults to Cedar unless SOURCE_EP is provided.
source_ep="${SOURCE_EP:-6df312ab-ad7c-4bbc-9369-450c82f0cb92}"
dest_ep="${DEST_EP:-${3:-}}"

if [ -z "$dest_ep" ]; then
  echo "Destination endpoint not provided. Set DEST_EP or pass as third argument." >&2
  exit 1
fi

session_path="$1"
dest_root="$2"
session_name="$(basename "$session_path")"
session_dest="${dest_root%/}/$session_name"

mkdir -p "$session_dest"

source_path="${session_path%/}/"
dest_path="${session_dest%/}/"

# Capture the task ID from the transfer command for status checking.
if ! task_id=$(globus transfer "${source_ep}:${source_path}" "${dest_ep}:${dest_path}" --recursive --jmespath 'task_id' --format=UNIX --notify failed,inactive); then
  echo "Globus transfer failed to start for ${session_path} -> ${dest_path}" >&2
  exit 1
fi

task_id=$(echo "$task_id" | tr -d '[:space:]')

if [ -z "$task_id" ]; then
  echo "Globus transfer did not return a task ID for ${session_path} -> ${dest_path}" >&2
  exit 1
fi

echo "Transfer initiated with Task ID: $task_id"
