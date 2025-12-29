#!/bin/bash
# Run simple CPG demo - validates forces without SNN complexity
#
# This is a quick test to verify:
# 1. Ground friction is working
# 2. CPG forces are applied correctly
# 3. The body moves in the expected direction

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
PURPLE='\033[0;35m'
NC='\033[0m'

echo "========================================================================"
echo -e "${PURPLE}Simple CPG Demo - Force Validation (No SNN)${NC}"
echo "========================================================================"
echo ""
echo "This script tests the basic CPG + physics without Nengo complexity."
echo "Use this to validate that forces and ratchet friction work correctly."
echo ""

# Get script directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
IMAGE_NAME="spring-mass-nengo"

# Quick config or custom
echo -e "${BLUE}Configuration:${NC}"
echo "  1) Quick test (defaults: 4x4 grid, locomotion mode, friction=0.5)"
echo "  2) Custom configuration"
echo ""
read -p "Select [1/2] (default: 1): " config_choice
config_choice=${config_choice:-1}

if [ "$config_choice" == "2" ]; then
    # Custom configuration
    echo ""
    read -p "Grid size N (default: 4): " grid_n
    grid_n=${grid_n:-4}
    
    read -p "CPG frequency Hz (default: 4.0): " frequency
    frequency=${frequency:-4.0}
    
    read -p "CPG amplitude (default: 1.0): " amplitude
    amplitude=${amplitude:-1.0}
    
    read -p "Force scale (default: 20.0): " force_scale
    force_scale=${force_scale:-20.0}
    
    echo ""
    echo "Direction (2D vector):"
    echo "  (1, 0) = right,  (-1, 0) = left"
    echo "  (0, 1) = up,     (0, -1) = down"
    echo "  (1, 1) = diagonal up-right"
    read -p "Direction X (default: 1): " dir_x
    dir_x=${dir_x:-1}
    read -p "Direction Y (default: 0): " dir_y
    dir_y=${dir_y:-0}
    
    read -p "Duration seconds (default: 30): " duration
    duration=${duration:-30}
else
    # Quick defaults
    grid_n=4
    frequency=4.0
    amplitude=1.0
    force_scale=20.0
    dir_x=1
    dir_y=0
    duration=30
fi

echo ""
echo "========================================================================"
echo -e "${GREEN}Configuration Summary:${NC}"
echo "  Grid: ${grid_n}x${grid_n}"
echo "  Force mode: radial (balloon)"
echo "  Direction: (${dir_x}, ${dir_y})"
echo "  Ratchet friction: enabled"
echo "  CPG frequency: ${frequency} Hz"
echo "  CPG amplitude: ${amplitude}"
echo "  Force scale: ${force_scale}"
echo "  Duration: ${duration}s"
echo "  Note: Forces always applied (radial deformation)"
echo "========================================================================"
echo ""
read -p "Press Enter to start..."
echo ""

# Build command
CMD="python demo_simple_cpg.py \
    --grid-size ${grid_n} \
    --direction ${dir_x} ${dir_y} \
    --frequency ${frequency} \
    --amplitude ${amplitude} \
    --force-scale ${force_scale} \
    --duration ${duration}"

echo -e "${YELLOW}Starting Simple CPG Demo...${NC}"
echo ""
echo "Controls:"
echo "  Q/ESC - Quit"
echo "  R     - Reset"
echo "  SPACE - Pause/Resume"
echo ""

# Allow X11 connections from Docker
echo "Setting up X11 display..."
xhost +local:docker 2>/dev/null || true

# Run in Docker
docker run -it --rm \
    --gpus all \
    --ipc=host \
    -e DISPLAY=$DISPLAY \
    -e PYTHONPATH=/workspace \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v "$PROJECT_DIR:/workspace" \
    -w /workspace/rl_locomotion \
    "$IMAGE_NAME" \
    $CMD

echo ""
echo -e "${GREEN}Demo complete!${NC}"

