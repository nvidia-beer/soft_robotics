#!/bin/bash
# PID Tuning Script
#
# Simple empirical tuning:
#   1. Sweep Kp to find best P-only response
#   2. Add Ki = Kp/5 and Kd = Kp*0.1
#
# Usage:
#   ./run_tuning.sh              # Run tuning
#   ./run_tuning.sh --validate   # With validation

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
IMAGE_NAME="spring-mass-nengo"

echo ""
echo "========================================================================"
echo -e "${CYAN}  PID Tuning${NC}"
echo "========================================================================"
echo ""

# Build Docker if needed
if ! docker image inspect "$IMAGE_NAME" &> /dev/null; then
    echo "Building Docker image..."
    "$PROJECT_DIR/build.sh"
fi

# GPU check
GPU_FLAGS=""
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    GPU_FLAGS="--gpus all"
    DEVICE="cuda"
    echo -e "${GREEN}✓ GPU detected${NC}"
else
    DEVICE="cpu"
    echo "Using CPU"
fi
echo ""

# Build command
CMD="python tuning.py --device $DEVICE $@"

echo -e "${BLUE}Running: $CMD${NC}"
echo ""

# Run
docker run -it --rm $GPU_FLAGS --ipc=host --shm-size=2g \
    -e PYTHONUNBUFFERED=1 \
    -v "$PROJECT_DIR:/workspace/soft_robotics" \
    -w /workspace/soft_robotics/inflation_control \
    "$IMAGE_NAME" \
    $CMD

echo ""
echo "========================================================================"
echo -e "${GREEN}Done!${NC}"
echo "========================================================================"
echo ""
echo "Next: ./create_figure.sh 2"
echo ""
