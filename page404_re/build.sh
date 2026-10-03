#!/usr/bin/env bash
#
# Build script for the CUDA tools in this folder.
#
# Self-contained: invokes nvcc directly so it works even when cmake is not
# installed. Target hardware: NVIDIA GH200 (Hopper, sm_90).
#
# Usage:
#   ./build.sh                       # build every *.cu -> ./build/<name>
#   ./build.sh eviction_set_builder.cu   # build just one source
#   ARCH=sm_90 ./build.sh            # override GPU architecture
#   ./build.sh clean                 # remove build artifacts
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

# Sources: the one named on the command line, or every *.cu in the folder.
if [[ -n "${1:-}" ]]; then
  SOURCES=("$1")
else
  SOURCES=(*.cu)
fi

mkdir -p "$OUT_DIR"

echo "Compiler : $($NVCC --version | tail -1)"
echo "Arch     : $ARCH"
echo "Host cxx : ${CCBIN_FLAG[*]:-<nvcc default>}"
echo

for SRC in "${SOURCES[@]}"; do
  OUT_BIN="$OUT_DIR/$(basename "$SRC" .cu)"
  echo "==> $SRC -> $OUT_BIN"
  set -x
  "$NVCC" -O3 -std=c++17 -arch="$ARCH" "${CCBIN_FLAG[@]}" "$SRC" -o "$OUT_BIN"
  set +x
  echo
done

echo "Build complete."
