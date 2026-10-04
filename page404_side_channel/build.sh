#!/usr/bin/env bash
#
# Build script for the shared side-channel collector used by both PoCs
# (prompt_length_estimate and llm_fingerprinting).
#
# Self-contained: invokes nvcc directly so it works even when cmake is not
# installed. Target hardware: NVIDIA GH200 (Hopper, sm_90).
#
# Produces the binary the collection scripts expect:
#   migration_delay_side_channel.cu -> build/migration_delay_side_channel
#
# Usage:
#   ./build.sh            # build into ./build/
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

SRC="migration_delay_side_channel.cu"
OUT_BIN="$OUT_DIR/migration_delay_side_channel"

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

CCBIN_FLAG=()
if command -v "$CCBIN" >/dev/null 2>&1; then
  CCBIN_FLAG=(-ccbin "$CCBIN")
fi

mkdir -p "$OUT_DIR"

echo "Compiler : $($NVCC --version | tail -1)"
echo "Arch     : $ARCH"
echo "Host cxx : ${CCBIN_FLAG[*]:-<nvcc default>}"
echo

echo "==> $SRC -> $OUT_BIN"
set -x
"$NVCC" -O3 -std=c++17 -arch="$ARCH" "${CCBIN_FLAG[@]}" "$SRC" -o "$OUT_BIN"
set +x

echo
echo "Build complete."
