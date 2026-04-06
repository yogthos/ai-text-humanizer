#!/bin/bash
# Keeps only the last N checkpoints, deletes the rest.
# Run as cron: */5 * * * * /workspace/revenant/scripts/cleanup_checkpoints.sh
CHECKPOINT_DIR="saves/Qwen2.5-32B/lora/howard_russell"
KEEP=3

cd /workspace/revenant 2>/dev/null || cd /workspace/howard_russell 2>/dev/null || exit 0

[ -d "$CHECKPOINT_DIR" ] || exit 0

# List checkpoint dirs sorted by number, delete all but last N
ls -d "$CHECKPOINT_DIR"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | head -n -${KEEP} | while read dir; do
    echo "$(date '+%H:%M:%S') Removing $dir"
    rm -rf "$dir"
done
