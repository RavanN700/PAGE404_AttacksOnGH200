#!/usr/bin/env bash
#
# Build script for the covert-channel tools (sender + receiver).
#
# Self-contained: invokes nvcc directly so it works even when cmake is not
# installed. Target hardware: NVIDIA GH200 (Hopper, sm_90).
#
# Binary names match what launch_overt_channel.sh expects:
#   sender.cu   -> build/sender_eviction_parallel_sets
#   receiver.cu -> build/receiver_eviction_parallel_sets
#
# Usage:
#   ./build.sh            # build both tools into ./build/
#   ./build.sh clean      # remove build artifacts
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

OUT_DIR="build"
ARCH="${ARCH:-sm_90}"            # GH200 = Hopper = sm_90
NVCC="${NVCC:-nvcc}"
# Host C++ compiler for nvcc. Defaults to g++-11: on this machine the default
# 'gcc' is v12 but only gcc-11 ships a working cc1plus. Override with CCBIN=.
CCBIN="${CCBIN:-g++-11}"

CCBIN_FLAG=()
if command -v "$CCBIN" >/dev/null 2>&1; then
  CCBIN_FLAG=(-ccbin "$CCBIN")
fi

if [[ "${1:-}" == "clean" ]]; then
  echo "Removing $OUT_DIR ..."
  rm -rf "$OUT_DIR"
  exit 0
fi

if ! command -v "$NVCC" >/dev/null 2>&1; then
  echo "error: '$NVCC' not found in PATH. Load your CUDA toolkit first" >&2
  echo "       (e.g. export PATH=/usr/local/cuda/bin:\$PATH)." >&2
  exit 1
fi

# source.cu : output-binary-name
TOOLS=(
  "sender.cu:sender_eviction_parallel_sets"
  "receiver.cu:receiver_eviction_parallel_sets"
)

mkdir -p "$OUT_DIR"

echo "Compiler : $($NVCC --version | tail -1)"
echo "Arch     : $ARCH"
echo "Host cxx : ${CCBIN_FLAG[*]:-<nvcc default>}"
echo

for entry in "${TOOLS[@]}"; do
  SRC="${entry%%:*}"
  OUT_BIN="$OUT_DIR/${entry##*:}"
  echo "==> $SRC -> $OUT_BIN"
  set -x
  "$NVCC" -O3 -std=c++17 -arch="$ARCH" "${CCBIN_FLAG[@]}" "$SRC" -o "$OUT_BIN"
  set +x
  echo
done

echo "Build complete."
