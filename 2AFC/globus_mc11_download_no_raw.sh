#!/usr/bin/env bash
set -euo pipefail

# Required environment variables:
#   GT_ENDPOINT, LOCAL_ENDPOINT, LOCAL_BASE
# Optional:
#   EXCLUDE_PATTERN (default: raw_voltages.h5)
#   LABEL (default below)

: "${GT_ENDPOINT:?Set GT_ENDPOINT to your source endpoint ID}"
: "${LOCAL_ENDPOINT:?Set LOCAL_ENDPOINT to your local endpoint ID}"
: "${LOCAL_BASE:?Set LOCAL_BASE to destination base path on LOCAL_ENDPOINT}"

EXCLUDE_PATTERN="${EXCLUDE_PATTERN:-raw_voltages.h5}"
LABEL="${LABEL:-MC11 2AFC processed outputs without raw voltages}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BATCH_TEMPLATE="$SCRIPT_DIR/mc11_no_raw_batch.txt"
BATCH_FILE="$SCRIPT_DIR/mc11_no_raw_batch.resolved.txt"

if ! command -v globus >/dev/null 2>&1; then
  echo "Error: globus CLI not found in PATH." >&2
  exit 1
fi

if [[ ! -f "$BATCH_TEMPLATE" ]]; then
  echo "Error: batch template not found at $BATCH_TEMPLATE" >&2
  exit 1
fi

# Resolve destination base path into a concrete batch file.
sed "s|{LOCAL_BASE}|${LOCAL_BASE}|g" "$BATCH_TEMPLATE" > "$BATCH_FILE"

echo "Using:"
echo "  GT_ENDPOINT=$GT_ENDPOINT"
echo "  LOCAL_ENDPOINT=$LOCAL_ENDPOINT"
echo "  LOCAL_BASE=$LOCAL_BASE"
echo "  BATCH_FILE=$BATCH_FILE"
echo "  EXCLUDE_PATTERN=$EXCLUDE_PATTERN"

globus transfer "$GT_ENDPOINT" "$LOCAL_ENDPOINT" \
  --batch "$BATCH_FILE" \
  --exclude "$EXCLUDE_PATTERN" \
  --label "$LABEL"

echo "Transfer submitted."
