#!/bin/bash
# Model Order Reduction Demo
#
# Complete MOR workflow:
# 1. TRAINING: Collect snapshots, compute POD basis
# 2. RUNNING: Simulate with reduced-order solver
#
# Same visuals as pygame_renderer/test_renderer.py

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

echo "========================================================================"
echo -e "${PURPLE}Model Order Reduction Demo${NC}"
echo "========================================================================"
echo ""
echo "Workflow:"
echo "  1. TRAINING: Run full simulation, collect snapshots, compute POD"
echo "  2. RUNNING: Simulate with reduced-order solver"
echo ""
echo -e "${YELLOW}Note: MOR benefits appear with LARGE models (100+ particles).${NC}"
echo -e "${YELLOW}For small models (54 particles), full GPU solver is faster.${NC}"
echo ""

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
IMAGE_NAME="spring-mass-nengo"

echo -e "${BLUE}Select Model Size:${NC}"
echo "  1) Small  (~54 particles)  - fast, for testing"
echo "  2) Large  (~150 particles) - same density, larger radius"
echo ""
read -p "Select [1-2] (default: 1): " size_choice
size_choice=${size_choice:-1}

case $size_choice in
    1)
        # Same as pygame_renderer default
        num_boundary=20
        num_rings=3
        radius=0.5
        ;;
    2)
        # 1.5x larger circle, EXACT same triangle size
        # Radial spacing: 0.75 / 6 = 0.125 (same as 0.5 / 4)
        # Boundary edge: 2π × 0.75 / 30 = π/20 (same as 2π × 0.5 / 20)
        num_boundary=30
        num_rings=5
        radius=0.75
        ;;
esac

echo ""
echo -e "${BLUE}Select Mode:${NC}"
echo "  1) Full workflow (train + run)"
echo "  2) Training only"
echo "  3) Run only (requires pre-trained model)"
echo "  4) Compare full vs reduced solver"
echo "  5) Custom configuration"
echo ""
read -p "Select [1-5] (default: 1): " choice
choice=${choice:-1}

# Defaults
train_duration=60
run_duration=30
tolerance=0.05  # Higher = fewer modes = faster (but less accurate)
snapshot_interval=5
boxsize=2.5
mode_flag=""
no_fem=""
no_sdf=""

case $choice in
    1)
        mode_flag=""
        ;;
    2)
        mode_flag="--train-only"
        ;;
    3)
        mode_flag="--run-only"
        ;;
    4)
        mode_flag="--compare"
        ;;
    5)
        echo ""
        echo -e "${CYAN}Custom Configuration:${NC}"
        echo ""
        
        echo "Mode:"
        echo "  1) Full workflow"
        echo "  2) Train only"
        echo "  3) Run only"
        echo "  4) Compare"
        read -p "  Select [1-4]: " m
        case $m in
            2) mode_flag="--train-only" ;;
            3) mode_flag="--run-only" ;;
            4) mode_flag="--compare" ;;
            *) mode_flag="" ;;
        esac
        
        echo ""
        echo "Training:"
        read -p "  Training duration (default: 60s): " train_duration
        train_duration=${train_duration:-60}
        read -p "  Snapshot interval (default: 5): " snapshot_interval
        snapshot_interval=${snapshot_interval:-5}
        read -p "  POD tolerance (default: 0.001 = 99.9%): " tolerance
        tolerance=${tolerance:-0.001}
        
        echo ""
        echo "Running:"
        read -p "  Run duration (default: 30s): " run_duration
        run_duration=${run_duration:-30}
        
        echo ""
        echo "Model:"
        read -p "  Circle radius (default: 0.5): " radius
        radius=${radius:-0.5}
        read -p "  Boundary points (default: 20): " num_boundary
        num_boundary=${num_boundary:-20}
        
        echo ""
        read -p "  Disable FEM? [y/N]: " d
        [[ "$d" =~ ^[Yy]$ ]] && no_fem="--no-fem"
        
        read -p "  Disable SDF? [y/N]: " d
        [[ "$d" =~ ^[Yy]$ ]] && no_sdf="--no-sdf"
        ;;
esac

echo ""
echo "========================================================================"
echo -e "${GREEN}Configuration:${NC}"
echo "  Mode: $([ -z "$mode_flag" ] && echo "Full workflow" || echo "${mode_flag#--}")"
echo ""
echo "  Training: ${train_duration}s, snapshot every ${snapshot_interval} steps"
echo "  POD tolerance: ${tolerance} (energy captured)"
echo "  Run duration: ${run_duration}s"
echo ""
echo "  Circle: radius=${radius}, ${num_boundary} pts"
echo "  FEM: $([ -z "$no_fem" ] && echo "ENABLED" || echo "disabled")"
echo "  SDF: $([ -z "$no_sdf" ] && echo "ENABLED" || echo "disabled")"
echo "========================================================================"
echo ""
read -p "Press Enter to start..."
echo ""

CMD="python demo_mor.py \
    ${mode_flag} \
    --train-duration ${train_duration} \
    --run-duration ${run_duration} \
    --tolerance ${tolerance} \
    --snapshot-interval ${snapshot_interval} \
    --radius ${radius} \
    --num-boundary ${num_boundary} \
    --num-rings ${num_rings} \
    --boxsize ${boxsize} \
    ${no_fem} \
    ${no_sdf}"

echo -e "${YELLOW}Starting MOR Demo...${NC}"
echo ""
echo "Controls:"
echo "  Q/ESC - Quit"
echo "  SPACE - Pause/Resume"
echo "  R     - Reset"
echo ""

xhost +local:docker 2>/dev/null || true

mkdir -p "$SCRIPT_DIR/reduced_data"

docker run -it --rm \
    --gpus all \
    --ipc=host \
    -e DISPLAY=$DISPLAY \
    -e PYTHONPATH=/workspace \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v "$PROJECT_DIR:/workspace" \
    -w /workspace/soft_model \
    "$IMAGE_NAME" \
    $CMD

echo ""
echo -e "${GREEN}MOR Demo complete!${NC}"
echo "Reduced model saved to: ${SCRIPT_DIR}/reduced_data/"
