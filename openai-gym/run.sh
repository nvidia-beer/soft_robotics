#!/bin/bash
# Run spring mass simulation in Docker with multiple model types
# Supports: tessellation, circle, and grid models

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================================================"
echo -e "${CYAN}Spring Mass Warp - Interactive Simulation${NC}"
echo "========================================================================"
echo ""

# Get script directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
IMAGE_NAME="spring-mass-nengo"

# Model type selection
echo -e "${BLUE}Select model type:${NC}"
echo "  1) Tessellation (load from JSON file)"
echo "  2) Circle (procedural circular mesh)"
echo "  3) Grid (regular NxN grid)"
echo ""
read -p "Enter choice [1-3] (default: 1): " model_choice
model_choice=${model_choice:-1}

MODEL_ARGS=""
case $model_choice in
    1)
        echo ""
        echo -e "${BLUE}Tessellation Mode${NC}"
        read -p "Tessellation file (default: /workspace/tessellation/model.json): " tess_file
        tess_file=${tess_file:-/workspace/tessellation/model.json}
        MODEL_ARGS="--tessellation ${tess_file}"
        echo "  Model: Tessellation from ${tess_file}"
        ;;
    2)
        echo ""
        echo -e "${YELLOW}⚠️  Note: Circle mode requires demo.py to support --circle flag${NC}"
        echo -e "${BLUE}Circle Mode${NC}"
        read -p "Radius (default: 0.75): " radius
        radius=${radius:-0.75}
        read -p "Boundary points (default: 16): " num_boundary
        num_boundary=${num_boundary:-16}
        read -p "Interior rings (default: 3): " num_rings
        num_rings=${num_rings:-3}
        MODEL_ARGS="--circle --radius ${radius} --num-boundary ${num_boundary} --num-rings ${num_rings}"
        echo "  Model: Circle (r=${radius}, boundary=${num_boundary}, rings=${num_rings})"
        ;;
    3)
        echo ""
        echo -e "${BLUE}Grid Mode${NC}"
        read -p "Grid size N (NxN particles, default: 10): " grid_n
        grid_n=${grid_n:-10}
        MODEL_ARGS="--N ${grid_n}"
        echo "  Model: ${grid_n}x${grid_n} grid"
        ;;
    *)
        echo "Invalid choice, using tessellation mode"
        MODEL_ARGS="--tessellation /workspace/tessellation/model.json"
        ;;
esac

echo ""

# Solver configuration
echo -e "${BLUE}Solver Configuration:${NC}"
echo "  1) Semi-Implicit (springs only, fastest)"
echo "  2) Implicit (springs implicit, FEM explicit)"
echo "  3) Fully Implicit FEM (K_fem in system matrix, most stable)"
echo ""
read -p "Enter choice [1-3] (default: 2): " solver_choice
solver_choice=${solver_choice:-2}

SOLVER_FLAG=""
case $solver_choice in
    1)
        echo "  Solver: Semi-Implicit (springs only)"
        ;;
    2)
        SOLVER_FLAG="--implicit"
        echo "  Solver: Implicit (springs implicit, FEM explicit)"
        ;;
    3)
        SOLVER_FLAG="--implicit-fem"
        echo "  Solver: Fully Implicit FEM ⚡⚡ (K_fem in system matrix)"
        ;;
    *)
        SOLVER_FLAG="--implicit"
        echo "  Using default: Implicit solver"
        ;;
esac

use_implicit="n"
if [ "$solver_choice" == "2" ] || [ "$solver_choice" == "3" ]; then
    use_implicit="y"
fi

# Simulation parameters
echo ""
echo -e "${BLUE}Simulation Parameters:${NC}"
read -p "Time step dt (default: 0.05): " dt
dt=${dt:-0.05}

read -p "Number of steps (default: 2000): " steps
steps=${steps:-2000}

read -p "Window width in pixels (default: 1000): " winwidth
winwidth=${winwidth:-1000}

read -p "Window height in pixels (default: 500): " winheight
winheight=${winheight:-500}

echo ""
echo -e "${YELLOW}Note: Boxsize determines simulation height. Width is auto-calculated.${NC}"
echo "  Simulation width = boxsize × (window_width / window_height)"
echo "  Example: boxsize=2.5, 1000×500 window → 5.0×2.5 simulation area"
read -p "Bounding box size (height, default: 2.5): " boxsize
boxsize=${boxsize:-2.5}

read -p "Device [cuda/cpu] (default: cuda): " device
device=${device:-cuda}

# Physics parameters (optional)
echo ""
read -p "Configure advanced physics parameters? [y/n] (default: n): " advanced
advanced=${advanced:-n}

PHYSICS_ARGS=""
if [ "$advanced" == "y" ]; then
    echo ""
    echo -e "${BLUE}Advanced Physics:${NC}"
    read -p "Spring stiffness (default: 40.0): " spring_k
    spring_k=${spring_k:-40.0}
    read -p "Spring damping (default: 0.5): " spring_d
    spring_d=${spring_d:-0.5}
    
    PHYSICS_ARGS="--spring-stiffness ${spring_k} --spring-damping ${spring_d}"
    
    if [ "$use_implicit" == "y" ]; then
        read -p "FEM Young's modulus (default: 50.0): " fem_young
        fem_young=${fem_young:-50.0}
        read -p "FEM Poisson ratio (default: 0.3): " fem_poisson
        fem_poisson=${fem_poisson:-0.3}
        read -p "FEM damping (default: 2.0): " fem_damp
        fem_damp=${fem_damp:-2.0}
        
        PHYSICS_ARGS="$PHYSICS_ARGS --fem-young ${fem_young} --fem-poisson ${fem_poisson} --fem-damping ${fem_damp}"
    fi
fi

# Summary
echo ""
echo "========================================================================"
echo -e "${GREEN}Configuration Summary:${NC}"
case $model_choice in
    1) echo "  Model: Tessellation from ${tess_file}" ;;
    2) echo "  Model: Circle (r=${radius}, boundary=${num_boundary}, rings=${num_rings})" ;;
    3) echo "  Model: ${grid_n}x${grid_n} grid" ;;
esac
case $solver_choice in
    1) echo "  Solver: Semi-Implicit (springs only)" ;;
    2) echo "  Solver: Implicit (springs implicit, FEM explicit)" ;;
    3) echo "  Solver: Fully Implicit FEM (K_fem in system matrix)" ;;
esac
echo "  Time step: ${dt}"
echo "  Steps: ${steps}"
echo "  Window: ${winwidth}×${winheight}"
echo "  Bounding box: height=${boxsize} (width auto-calculated from aspect ratio)"
echo "  Device: ${device}"
if [ ! -z "$PHYSICS_ARGS" ]; then
    echo "  Advanced physics: Enabled"
fi
echo "========================================================================"
echo ""
read -p "Press Enter to start simulation..."
echo ""

# Build command
CMD="python3 demo.py ${MODEL_ARGS}"
CMD="$CMD $SOLVER_FLAG"
CMD="$CMD --dt $dt --steps $steps"
CMD="$CMD --device $device"
CMD="$CMD --window-width $winwidth"
CMD="$CMD --window-height $winheight"
CMD="$CMD --boxsize $boxsize"
if [ ! -z "$PHYSICS_ARGS" ]; then
    CMD="$CMD $PHYSICS_ARGS"
fi

echo -e "${BLUE}Executing:${NC} $CMD"
echo ""

# Run in Docker
docker run -it --rm \
    --gpus all \
    --ipc=host \
    --shm-size=2g \
    -e DISPLAY=$DISPLAY \
    -e SDL_VIDEO_X11_NET_WM_BYPASS_COMPOSITOR=0 \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v "$PROJECT_DIR:/workspace" \
    -w /workspace/openai-gym \
    "$IMAGE_NAME" \
    $CMD

echo ""
echo -e "${GREEN}Simulation complete!${NC}"
echo ""

