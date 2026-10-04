#!/bin/bash
#
# Run one side-channel collection trial: the attacker collector and the victim
# command run concurrently, each pinned to its own MIG slice of the same GH200.
#
# The attacker (migration_delay_side_channel) starts first and writes its two
# output files; when it finishes, the victim is terminated. This script is
# normally invoked by auto_collection.py, once per (app, run).
#
# Usage:
#   ./collection_script.sh "<victim_command>" <time_out> <counter_out> <mem_size> <N>
#
# Set the two MIG slice UUIDs below (or via the ATTACKER_MIG / VICTIM_MIG
# environment variables). Find them with:  nvidia-smi -L
set -u

if [ $# -ne 5 ]; then
    echo "Usage: $0 \"<victim_command>\" <time_out> <counter_out> <mem_size> <N>" >&2
    exit 1
fi

VICTIM_CMD="$1"
PATH1="$2"          # attacker output: migration-delay timestamps
PATH2="$3"          # attacker output: access counters
MemorySize="$4"
N="$5"

# ── MIG slices ──────────────────────────────────────────────────────────────
# Replace these with two DIFFERENT slice UUIDs from `nvidia-smi -L`, or export
# ATTACKER_MIG / VICTIM_MIG before running.
ATTACKER_MIG="${ATTACKER_MIG:-MIG-877b8ba2-afff-58bd-a00d-cb554fd9ceaa}"
VICTIM_MIG="${VICTIM_MIG:-MIG-f6579648-12dd-59cb-ad9d-ddf6c487156b}"

if [ "$ATTACKER_MIG" = "$VICTIM_MIG" ]; then
    echo "ERROR: ATTACKER_MIG and VICTIM_MIG are the same slice." >&2
    echo "       Set two different MIG UUIDs (see 'nvidia-smi -L')." >&2
    exit 1
fi

# Attacker / collector executable (built by ../build.sh).
EXECUTABLE1="../build/migration_delay_side_channel"

[ -f "$EXECUTABLE1" ] || { echo "Error: $EXECUTABLE1 not found (run ../build.sh)" >&2; exit 1; }
[ -e "$PATH1" ]       || { echo "Error: $PATH1 not found" >&2; exit 1; }
[ -e "$PATH2" ]       || { echo "Error: $PATH2 not found" >&2; exit 1; }

echo "Starting execution on MIG instances..."

# Run the attacker collector on its slice.
CUDA_VISIBLE_DEVICES="$ATTACKER_MIG" \
    "$EXECUTABLE1" "$PATH1" "$PATH2" "$MemorySize" "$N" &
PID1=$!

sleep 1

# Run the victim command on the other slice.
CUDA_VISIBLE_DEVICES="$VICTIM_MIG" \
    bash -c "$VICTIM_CMD" &
PID2=$!

# Wait for the attacker to finish, then stop the victim if still running.
wait $PID1
EXIT1=$?

if kill -0 $PID2 2>/dev/null; then
    echo "Attacker finished, terminating victim..."
    kill -TERM $PID2 2>/dev/null
    sleep 1
    if kill -0 $PID2 2>/dev/null; then
        kill -KILL $PID2 2>/dev/null
    fi
fi

wait $PID2 2>/dev/null
EXIT2=$?

echo "Execution completed"
echo "Attacker exit code: $EXIT1"
echo "Victim exit code:   $EXIT2"

exit $EXIT1
