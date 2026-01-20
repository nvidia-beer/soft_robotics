#!/bin/bash
# Compare SNN_PID vs SNN_Stress Controllers
#
# Demonstrates that PES learning in SNN_Stress improves performance over time
# by learning the strain→pressure mapping.
#
# Expected results:
# - Early phase: Both controllers similar (PES hasn't learned yet)
# - Late phase: SNN_Stress should outperform SNN_PID (PES has learned)
#
# Usage:
#   ./run_comparison.sh           # Interactive mode
#   ./run_comparison.sh --quick   # Quick comparison with defaults

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
PURPLE='\033[0;35m'
NC='\033[0m'

# Get script directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
IMAGE_NAME="spring-mass-nengo"

echo ""
echo "========================================================================"
echo -e "${PURPLE}  SNN_PID vs SNN_Stress Controller Comparison${NC}"
echo "========================================================================"
echo ""
echo -e "${CYAN}This comparison demonstrates PES feedforward learning:${NC}"
echo ""
echo "  SNN_PID:    Pure spiking PD control (no learning)"
echo "  SNN_Stress: Spiking PD + PES feedforward (learns strain→pressure)"
echo ""
echo -e "${YELLOW}Expected results:${NC}"
echo "  - Early phase: Both similar (PES hasn't learned yet)"
echo "  - Late phase:  SNN_Stress should outperform SNN_PID"
echo ""

# Check for quick mode
QUICK_MODE=false
for arg in "$@"; do
    if [ "$arg" == "--quick" ]; then
        QUICK_MODE=true
        break
    fi
done

# Default parameters
TOTAL_TIME=20.0
PATTERN="multi_step"
MAX_VOLUME=2.0
PES_LR="1e-4"
NEURONS=100
DEVICE="cuda"

if [ "$QUICK_MODE" = false ]; then
    echo -e "${BLUE}Simulation Parameters:${NC}"
    read -p "  Total time in seconds (default: 20.0): " input
    TOTAL_TIME=${input:-$TOTAL_TIME}
    
    echo ""
    echo -e "${BLUE}Target Pattern:${NC}"
    echo "  1) step       - Single step change"
    echo "  2) ramp       - Linear ramp"
    echo "  3) sine       - Sinusoidal oscillation"
    echo "  4) multi_step - Multiple steps (default, best for learning)"
    read -p "  Enter choice [1-4] (default: 4): " pattern_choice
    pattern_choice=${pattern_choice:-4}
    case $pattern_choice in
        1) PATTERN="step" ;;
        2) PATTERN="ramp" ;;
        3) PATTERN="sine" ;;
        *) PATTERN="multi_step" ;;
    esac
    
    echo ""
    read -p "  Max volume ratio (default: 2.0): " input
    MAX_VOLUME=${input:-$MAX_VOLUME}
    
    echo ""
    echo -e "${BLUE}PES Learning Parameters:${NC}"
    echo -e "  ${YELLOW}Higher learning rate = faster learning but may overshoot${NC}"
    read -p "  PES learning rate (default: 1e-4): " input
    PES_LR=${input:-$PES_LR}
    
    echo ""
    read -p "  Neurons per ensemble (default: 100): " input
    NEURONS=${input:-$NEURONS}
    
    echo ""
    read -p "  Device [cuda/cpu] (default: cuda): " input
    DEVICE=${input:-$DEVICE}
fi

# Check for GPU
GPU_FLAGS=""
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    GPU_FLAGS="--gpus all"
    echo -e "${GREEN}✓ GPU detected - using CUDA${NC}"
else
    DEVICE="cpu"
    echo -e "${YELLOW}⚠ No GPU detected - using CPU${NC}"
fi

# Summary
echo ""
echo "========================================================================"
echo -e "${GREEN}Configuration Summary:${NC}"
echo "========================================================================"
echo "  Total time:    ${TOTAL_TIME}s"
echo "  Pattern:       ${PATTERN}"
echo "  Max volume:    ${MAX_VOLUME}x"
echo "  PES lr:        ${PES_LR}"
echo "  Neurons:       ${NEURONS}"
echo "  Device:        ${DEVICE}"
echo "========================================================================"
echo ""

if [ "$QUICK_MODE" = false ]; then
    read -p "Press Enter to start comparison..."
fi

# Build Docker if needed
if ! docker image inspect "$IMAGE_NAME" &> /dev/null; then
    echo -e "${YELLOW}Building Docker image...${NC}"
    "$PROJECT_DIR/build.sh"
fi

# Build command
CMD="python3 compare_controllers.py"
CMD="$CMD --total-time $TOTAL_TIME"
CMD="$CMD --pattern $PATTERN"
CMD="$CMD --max-volume $MAX_VOLUME"
CMD="$CMD --pes-lr $PES_LR"
CMD="$CMD --neurons $NEURONS"
CMD="$CMD --device $DEVICE"

echo ""
echo -e "${CYAN}Running comparison...${NC}"
echo -e "${YELLOW}$CMD${NC}"
echo ""

# Run in Docker
docker run -it --rm $GPU_FLAGS --ipc=host --shm-size=2g \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v "$PROJECT_DIR:/workspace" \
    -w /workspace/inflation_control \
    "$IMAGE_NAME" \
    $CMD

echo ""
echo -e "${GREEN}Comparison complete!${NC}"
echo ""
echo "Results saved to:"
echo "  - comparison_*.npz  (raw data)"
echo "  - comparison_*.png  (plot)"
echo ""
echo "To re-plot with different settings:"
echo "  python compare_controllers.py --plot-file comparison_*.npz --show"
echo ""
