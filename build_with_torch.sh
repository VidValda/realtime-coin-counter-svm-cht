#!/usr/bin/env bash
# Download LibTorch (CPU) if needed and build coin_counter with Torch support.
# Usage: ./build_with_torch.sh [path-to-existing-libtorch]
#   With no args: downloads LibTorch into build/libtorch and builds.
#   With path: uses that LibTorch and builds (e.g. ./build_with_torch.sh /opt/libtorch).
#
# Note: Use LibTorch 2.5+ so it can load .pt files exported by recent PyTorch (format version 3).
# "latest" on the server can point to an old build; we pin to 2.5.1 for compatibility.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
# Pin to 2.5.1 so TorchScript .pt files (export_torchscript.py) load correctly (format v3)
LIBTORCH_VERSION="${LIBTORCH_VERSION:-2.5.1}"
LIBTORCH_URL="https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-${LIBTORCH_VERSION}%2Bcpu.zip"
LIBTORCH_ZIP="${BUILD_DIR}/libtorch.zip"
LIBTORCH_DIR="${BUILD_DIR}/libtorch"

if [[ -n "$1" ]]; then
  # Use user-provided path
  if [[ ! -d "$1" ]]; then
    echo "Error: directory does not exist: $1"
    exit 1
  fi
  LIBTORCH_DIR="$(cd "$1" && pwd)"
  echo "Using LibTorch at: ${LIBTORCH_DIR}"
else
  # Re-download if existing LibTorch is too old (1.x cannot load .pt from recent PyTorch)
  if [[ -f "${LIBTORCH_DIR}/share/cmake/Torch/TorchConfigVersion.cmake" ]]; then
    if grep -q 'PACKAGE_VERSION "1\.' "${LIBTORCH_DIR}/share/cmake/Torch/TorchConfigVersion.cmake" 2>/dev/null; then
      echo "Removing old LibTorch 1.x (cannot load models from recent PyTorch)."
      rm -rf "${LIBTORCH_DIR}"
    fi
  fi
  # Download and extract if not present
  if [[ ! -d "${LIBTORCH_DIR}/share/cmake/Torch" ]]; then
    echo "LibTorch not found at ${LIBTORCH_DIR}. Downloading LibTorch ${LIBTORCH_VERSION} (CPU, ~200MB)..."
    mkdir -p "${BUILD_DIR}"
    if command -v wget &>/dev/null; then
      wget -q --show-progress -O "${LIBTORCH_ZIP}" "${LIBTORCH_URL}" || true
    elif command -v curl &>/dev/null; then
      curl -# -L -o "${LIBTORCH_ZIP}" "${LIBTORCH_URL}" || true
    else
      echo "Error: need wget or curl to download LibTorch."
      exit 1
    fi
    if [[ ! -f "${LIBTORCH_ZIP}" ]]; then
      echo "Download failed. Get LibTorch manually from https://pytorch.org/get-started/locally/ (C++/LibTorch/CPU), extract to ${LIBTORCH_DIR}, then run:"
      echo "  cmake -DUSE_TORCH=ON -DCMAKE_PREFIX_PATH=${LIBTORCH_DIR} .. && make"
      exit 1
    fi
    echo "Extracting..."
    unzip -q -o "${LIBTORCH_ZIP}" -d "${BUILD_DIR}"
    rm -f "${LIBTORCH_ZIP}"
    if [[ ! -d "${LIBTORCH_DIR}/share/cmake/Torch" ]]; then
      # zip may contain a single top-level folder named libtorch
      if [[ -d "${BUILD_DIR}/libtorch/share/cmake/Torch" ]]; then
        : # already there
      else
        echo "Unexpected zip layout. Check ${BUILD_DIR} and set CMAKE_PREFIX_PATH to the dir that contains share/cmake/Torch."
        exit 1
      fi
    fi
    echo "LibTorch ready at ${LIBTORCH_DIR}"
  else
    echo "Using existing LibTorch at ${LIBTORCH_DIR}"
  fi
fi

# If LibTorch path contains spaces, use a symlink so the linker gets a path without spaces
LIBTORCH_CMAKE="${LIBTORCH_DIR}"
if [[ "${LIBTORCH_DIR}" = *" "* ]]; then
  LIBTORCH_LINK="${BUILD_DIR}/libtorch_link"
  rm -f "${LIBTORCH_LINK}"
  ln -s "${LIBTORCH_DIR}" "${LIBTORCH_LINK}"
  LIBTORCH_CMAKE="${LIBTORCH_LINK}"
  echo "Using symlink for CMake (path has spaces)"
fi

cd "${BUILD_DIR}"
cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_TORCH=ON -DCMAKE_PREFIX_PATH="${LIBTORCH_CMAKE}"
make -j"$(nproc 2>/dev/null || echo 4)"
echo "Done. Run: ./coin_counter (keys 5/6 for CNN/ResNet18) or ./coin_counter_dl (DL-only, keys 1/2 for CNN/ResNet)"
