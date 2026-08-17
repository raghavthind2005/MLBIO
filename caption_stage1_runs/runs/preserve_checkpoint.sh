#!/bin/bash
# Copy a training checkpoint from scratch to the persistent project store, and
# PROVE the copy is intact.
#
# WHY THIS EXISTS
# ---------------
# All 21 PAPO checkpoints (3 arms x 7 steps) were lost from
# /iopsstor/scratch on 2026-08-11. The directory skeletons survived, so the
# loss was invisible to `ls` -- `global_step_60/actor/huggingface/` still
# existed, containing nothing. The frozen offline perception-KL probe cannot
# run without those weights, and re-creating them costs ~3 twelve-hour slots
# per arm.
#
# /capstor is a different filesystem and was untouched by that event (models
# there date from June and survive), so it is the preservation target.
#
# Two rules this script enforces, both learned the hard way:
#   1. Verify the copy. A directory listing is not evidence -- that is exactly
#      what made the dataset loss invisible.
#   2. Never delete the scratch copy. This adds a replica; it does not move.
#
# Usage:
#   preserve_checkpoint.sh <run_name> <global_step_dir>
# Example:
#   preserve_checkpoint.sh cs1_real /iopsstor/.../checkpoints/global_step_10

set -euo pipefail

RUN_NAME=${1:?usage: preserve_checkpoint.sh <run_name> <global_step_dir>}
SRC=${2:?usage: preserve_checkpoint.sh <run_name> <global_step_dir>}
DEST_ROOT=${CS1_PRESERVE_ROOT:-/capstor/store/cscs/swissai/a0174/caption_stage1_ckpts}

[ -d "$SRC" ] || { echo "FATAL: source does not exist: $SRC"; exit 1; }

STEP=$(basename "$SRC")
DEST="$DEST_ROOT/$RUN_NAME/$STEP"
mkdir -p "$DEST"

echo "=== preserve $RUN_NAME/$STEP ==="
echo "  src : $SRC"
echo "  dest: $DEST"

# Refuse to preserve an empty checkpoint. This is precisely the PAPO failure
# signature: directories present, payload absent.
SRC_FILES=$(find "$SRC" -type f | wc -l)
SRC_BYTES=$(du -sb "$SRC" | cut -f1)
echo "  source: $SRC_FILES files, $(numfmt --to=iec "$SRC_BYTES" 2>/dev/null || echo "$SRC_BYTES bytes")"
if [ "$SRC_FILES" -eq 0 ]; then
    echo "FATAL: source checkpoint contains ZERO files -- nothing to preserve"
    exit 1
fi

# Space check before copying, since /capstor is a shared 1 TB project store.
AVAIL_KB=$(df -Pk "$DEST_ROOT" | awk 'NR==2{print $4}')
NEED_KB=$(( SRC_BYTES / 1024 ))
if [ "$NEED_KB" -gt "$(( AVAIL_KB - 52428800 ))" ]; then   # keep 50 GB headroom
    echo "FATAL: insufficient space on /capstor (need ${NEED_KB}K, avail ${AVAIL_KB}K, 50G reserve)"
    exit 1
fi

# Checksum manifest from the SOURCE first, so verification compares against
# what we intended to copy rather than against the copy itself.
MANIFEST="$DEST/.source_sha256"
( cd "$SRC" && find . -type f -print0 | sort -z | xargs -0 sha256sum ) > "$MANIFEST.tmp"

cp -r --preserve=timestamps "$SRC/." "$DEST/"

# Verify: every source file must be present at the destination with a matching
# digest. `cp` exiting 0 is not sufficient evidence.
FAILED=0
( cd "$DEST" && sha256sum -c --quiet "$MANIFEST.tmp" ) || FAILED=1

DEST_FILES=$(find "$DEST" -type f ! -name ".source_sha256*" | wc -l)
if [ "$DEST_FILES" -ne "$SRC_FILES" ]; then
    echo "FATAL: file-count mismatch -- src=$SRC_FILES dest=$DEST_FILES"
    FAILED=1
fi

if [ "$FAILED" -ne 0 ]; then
    echo "FATAL: verification FAILED for $RUN_NAME/$STEP -- destination is NOT trustworthy"
    exit 1
fi

mv "$MANIFEST.tmp" "$MANIFEST"
echo "  VERIFIED: $DEST_FILES files, all sha256 match"
echo "  capstor free after: $(df -h "$DEST_ROOT" | awk 'NR==2{print $4}')"
