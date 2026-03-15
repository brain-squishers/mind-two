#!/bin/bash

set -euo pipefail

BASE_URL="https://huggingface.co/depth-anything"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

download_checkpoint() {
    local filename="$1"
    local url="$2"
    echo "Downloading ${filename} checkpoint..."
    wget -O "${SCRIPT_DIR}/${filename}" "${url}"
}

download_checkpoint \
    "depth_anything_v2_metric_hypersim_vits.pth" \
    "${BASE_URL}/Depth-Anything-V2-Metric-Hypersim-Small/resolve/main/depth_anything_v2_metric_hypersim_vits.pth?download=true"
download_checkpoint \
    "depth_anything_v2_metric_hypersim_vitb.pth" \
    "${BASE_URL}/Depth-Anything-V2-Metric-Hypersim-Base/resolve/main/depth_anything_v2_metric_hypersim_vitb.pth?download=true"
download_checkpoint \
    "depth_anything_v2_metric_hypersim_vitl.pth" \
    "${BASE_URL}/Depth-Anything-V2-Metric-Hypersim-Large/resolve/main/depth_anything_v2_metric_hypersim_vitl.pth?download=true"

echo "All indoor metric checkpoints are downloaded successfully."
