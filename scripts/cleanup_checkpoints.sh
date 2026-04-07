#!/bin/bash
# Keeps only the last N COMPLETE checkpoints, deletes the rest.
# Avoids deleting during active saves by checking for write locks.
# Run as cron or background loop.
CHECKPOINT_DIR="saves/Qwen2.5-32B/lora/howard_russell"
KEEP=3

cd /workspace/revenant 2>/dev/null || cd /workspace/howard_russell 2>/dev/null || exit 0

[ -d "$CHECKPOINT_DIR" ] || exit 0

# Skip cleanup if a checkpoint is actively being written
# (trainer_state.json is written last — if newest checkpoint lacks it, save is in progress)
NEWEST=$(ls -d "$CHECKPOINT_DIR"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
if [ -n "$NEWEST" ] && [ ! -f "$NEWEST/trainer_state.json" ]; then
    echo "$(date '+%H:%M:%S') Save in progress ($NEWEST), skipping cleanup"
    exit 0
fi

# List checkpoint dirs sorted by number, delete all but last N
ls -d "$CHECKPOINT_DIR"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | head -n -${KEEP} | while read dir; do
    echo "$(date '+%H:%M:%S') Removing $dir"
    rm -rf "$dir"
done
