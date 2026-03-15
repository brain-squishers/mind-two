#!/bin/bash

set -euo pipefail

BASE_URL="https://huggingface.co/depth-anything"
vits_url="${BASE_URL}/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth?download=true"
vitb_url="${BASE_URL}/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth?download=true"
vitl_url="${BASE_URL}/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Downloading depth_anything_v2_vits.pth checkpoint..."
wget -O "${SCRIPT_DIR}/depth_anything_v2_vits.pth" "${vits_url}"

echo "Downloading depth_anything_v2_vitb.pth checkpoint..."
wget -O "${SCRIPT_DIR}/depth_anything_v2_vitb.pth" "${vitb_url}"

echo "Downloading depth_anything_v2_vitl.pth checkpoint..."
wget -O "${SCRIPT_DIR}/depth_anything_v2_vitl.pth" "${vitl_url}"

echo "All checkpoints are downloaded successfully."
