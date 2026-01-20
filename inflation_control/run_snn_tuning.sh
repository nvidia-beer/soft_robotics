#!/bin/bash
# SNN PID Tuning Script
#
# Finds optimal Kp for SNN controller (much lower than classic PID)
# Then computes Ki and Kd using ratios tuned for neural dynamics.
#
# Usage:
#   ./run_snn_tuning.sh              # Run tuning
#   ./run_snn_tuning.sh --validate   # With validation of all P/PI/PD/PID

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
NC='\033[0m'

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
IMAGE_NAME="spring-mass-nengo"

echo ""
echo "========================================================================"
echo -e "${CYAN}  SNN PID Tuning${NC}"
echo "========================================================================"
echo ""
echo -e "${YELLOW}Note: SNN needs MUCH lower gains than classic PID${NC}"
echo "      (typically Kp=1-5 vs classic Kp=20-30)"
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
CMD="python snn_tuning.py --device $DEVICE $@"

echo -e "${BLUE}Running: $CMD${NC}"
echo ""

# Run (use -i for interactive, no -t for non-TTY compatibility)
docker run -i --rm $GPU_FLAGS --ipc=host --shm-size=2g \
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
echo "Next: ./create_figure.sh 1"
echo ""
