#!/bin/bash
# Compare Simple CPG vs SNN CPG
#
# This script runs both CPG implementations with the same parameters
# to verify the SNN produces the correct traveling wave pattern.
#
# Usage:
#   ./compare_cpg.sh          # Interactive comparison
#   ./compare_cpg.sh simple   # Run simple CPG only
#   ./compare_cpg.sh snn      # Run SNN CPG only

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
PURPLE='\033[0;35m'
RED='\033[0;31m'
NC='\033[0m'

# Get script directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
IMAGE_NAME="spring-mass-nengo"

echo ""
echo "========================================================================"
echo -e "${PURPLE}CPG Comparison: Simple vs SNN${NC}"
echo "========================================================================"
echo ""
echo "This script compares the rate-coded CPG with the spiking neural CPG"
echo "to verify both produce the same traveling wave pattern."
echo ""

# Common parameters for fair comparison
GRID_ROWS=3
GRID_COLS=6
FREQUENCY=4.0
AMPLITUDE=1.0
FORCE_SCALE=20.0
DIR_X=1
DIR_Y=0
DURATION=20

echo -e "${CYAN}Common Parameters:${NC}"
echo "  Grid: ${GRID_COLS}x${GRID_ROWS} ($(( (GRID_ROWS-1)*(GRID_COLS-1) )) groups)"
echo "  Frequency: ${FREQUENCY} Hz"
echo "  Amplitude: ${AMPLITUDE}"
echo "  Direction: (${DIR_X}, ${DIR_Y})"
echo "  Force scale: ${FORCE_SCALE}"
echo ""

# Enable X11
xhost +local:docker 2>/dev/null || true

# Check for GPU
GPU_FLAGS=""
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    GPU_FLAGS="--gpus all"
    DEVICE="cuda"
else
    DEVICE="cpu"
fi

run_simple() {
    echo ""
    echo "========================================================================"
    echo -e "${GREEN}Running: Simple CPG (rate-coded HopfCPG)${NC}"
    echo "========================================================================"
    echo ""
    echo "Watch for:"
    echo "  - Traveling wave pattern in CPG matrix (bottom-right)"
    echo "  - Radial force arrows on each group"
    echo "  - Body should move RIGHT (direction 1,0)"
    echo "  - Note the displacement after ${DURATION}s"
    echo ""
    echo "Press Q or ESC to quit early"
    echo ""
    read -p "Press Enter to start Simple CPG..."
    
    docker run -it --rm $GPU_FLAGS \
        --ipc=host \
        -e DISPLAY=$DISPLAY \
        -e PYTHONPATH=/workspace \
        -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
        -v "$PROJECT_DIR:/workspace" \
        -w /workspace/rl_locomotion \
        "$IMAGE_NAME" \
        python demo_simple_cpg.py \
            --rows $GRID_ROWS \
            --cols $GRID_COLS \
            --frequency $FREQUENCY \
            --amplitude $AMPLITUDE \
            --force-scale $FORCE_SCALE \
            --direction $DIR_X $DIR_Y \
            --duration $DURATION
    
    echo ""
    echo -e "${GREEN}Simple CPG complete!${NC}"
}

run_snn() {
    echo ""
    echo "========================================================================"
    echo -e "${PURPLE}Running: SNN CPG (spiking Hopf oscillators)${NC}"
    echo "========================================================================"
    echo ""
    echo "Watch for:"
    echo "  - Same traveling wave pattern as Simple CPG"
    echo "  - Same radial force arrows"
    echo "  - Body should move in SAME direction as Simple CPG"
    echo "  - Similar displacement after same time"
    echo ""
    echo "Press Q or ESC to quit early"
    echo ""
    read -p "Press Enter to start SNN CPG..."
    
    # Use demo_snn_simple.py for FAST comparison (no Nengo GUI overhead)
    docker run -it --rm $GPU_FLAGS \
        --ipc=host \
        -e DISPLAY=$DISPLAY \
        -e PYTHONPATH=/workspace \
        -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
        -v "$PROJECT_DIR:/workspace" \
        -w /workspace/rl_locomotion \
        "$IMAGE_NAME" \
        python demo_snn_simple.py \
            --rows $GRID_ROWS \
            --cols $GRID_COLS \
            --frequency $FREQUENCY \
            --amplitude $AMPLITUDE \
            --force-scale $FORCE_SCALE \
            --direction $DIR_X $DIR_Y \
            --duration $DURATION \
            --n-neurons 30
    
    echo ""
    echo -e "${PURPLE}SNN CPG complete!${NC}"
}

# Handle command line arguments
if [ "$1" == "simple" ]; then
    run_simple
    exit 0
elif [ "$1" == "snn" ]; then
    run_snn
    exit 0
fi

# Interactive comparison
echo -e "${BLUE}Comparison Options:${NC}"
echo "  1) Run Simple CPG first, then SNN CPG"
echo "  2) Run Simple CPG only"
echo "  3) Run SNN CPG only"
echo ""
read -p "Select [1/2/3] (default: 1): " choice
choice=${choice:-1}

case $choice in
    1)
        run_simple
        echo ""
        echo "========================================================================"
        echo -e "${YELLOW}Now running SNN CPG for comparison...${NC}"
        echo "========================================================================"
        run_snn
        
        echo ""
        echo "========================================================================"
        echo -e "${GREEN}Comparison Complete!${NC}"
        echo "========================================================================"
        echo ""
        echo "Check:"
        echo "  ✓ Both showed traveling wave in CPG matrix?"
        echo "  ✓ Force arrows pointed same directions?"
        echo "  ✓ Body moved in same direction (RIGHT)?"
        echo "  ✓ Similar displacement values?"
        echo ""
        echo "If SNN moved OPPOSITE direction:"
        echo "  → Phase rotation may need to be negated"
        echo "  → Check demo_snn_gui.py make_cpg_output_func()"
        echo ""
        ;;
    2)
        run_simple
        ;;
    3)
        run_snn
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo -e "${GREEN}Done!${NC}"

