#!/bin/bash
#
# Launch the eviction-based covert channel: receiver and sender run
# concurrently, each pinned to its own MIG slice of the same GH200 GPU.
#
# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Find your MIG slice UUIDs and paste them below.
#
#   Run:   nvidia-smi -L
#
#   Look for lines like:
#     GPU 0: NVIDIA GH200 ...
#       MIG 1g.12gb  Device  0:  (UUID: MIG-48b2f4b8-5695-51e2-9150-cf3c0ad3e8ff)
#       MIG 1g.12gb  Device  1:  (UUID: MIG-97f521d2-93bb-5765-bea5-ee40a03dd91a)
#
#   Copy two different MIG UUIDs into RECEIVER_MIG and SENDER_MIG below.
#   The sender and receiver MUST run on different MIG slices.
# ─────────────────────────────────────────────────────────────────────────────

RECEIVER_MIG="MIG-48b2f4b8-5695-51e2-9150-cf3c0ad3e8ff"   # <-- replace with your receiver slice
SENDER_MIG="MIG-97f521d2-93bb-5765-bea5-ee40a03dd91a"     # <-- replace with your sender slice

# Number of parallel channels (sets) to run; defaults to 1.
num_sets=${1:-1}

# Sanity check: the two slices must differ.
if [ "$RECEIVER_MIG" = "$SENDER_MIG" ]; then
    echo "ERROR: RECEIVER_MIG and SENDER_MIG are the same slice." >&2
    echo "       Edit this script and set two different MIG UUIDs (see 'nvidia-smi -L')." >&2
    exit 1
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Launch. Receiver starts first (it waits for the sender's sync bits).
# ─────────────────────────────────────────────────────────────────────────────
# Both binaries write their per-channel traces here; create it up front so the
# first fopen() does not fail (which would crash on the following fprintf).
mkdir -p texts/parallel_channel

CUDA_VISIBLE_DEVICES="$RECEIVER_MIG" ./build/receiver_eviction_parallel_sets \
    --threshold 256 --num-pages 20480 --num-bits 17000 --gpu 0 --delta-n 1 --num-sets "$num_sets" &

CUDA_VISIBLE_DEVICES="$SENDER_MIG" ./build/sender_eviction_parallel_sets \
    --threshold 256 --num-eviction-sets 2 --gpu 0 --delta-n 10 --num-bits 10000 --num-sets "$num_sets"

wait
echo "Both processes finished"
