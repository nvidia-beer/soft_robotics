#!/bin/bash
# =============================================================================
# Test script for Pygame Renderer - Circle Soft Body
# =============================================================================
#
# Tests the Renderer class with circle soft body visualization:
# 1. FEM / Spring strain legends
# 2. Tension visualization (springs + FEM colored by strain gradient)
# 3. Optional SDF background (brown=collision, white=passable)
#
# Usage:
#   ./run.sh              # Run interactive demo with defaults
#   ./run.sh --help       # Show options
#
# =============================================================================

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

echo "========================================================================"
echo -e "${PURPLE}Pygame Renderer Test - Circle Soft Body${NC}"
echo "========================================================================"
echo ""

# Get script directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
IMAGE_NAME="spring-mass-nengo"

# Configuration
echo -e "${BLUE}Configuration:${NC}"
echo "  1) Quick test (defaults)"
echo "  2) Custom configuration"
echo ""
read -p "Select [1/2] (default: 1): " config_choice
config_choice=${config_choice:-1}

if [ "$config_choice" == "2" ]; then
    # Custom configuration
    echo ""
    echo -e "${BLUE}Circle Configuration:${NC}"
    read -p "Circle radius (default: 0.5): " radius
    radius=${radius:-0.5}
    read -p "Number of boundary points (default: 20): " num_boundary
    num_boundary=${num_boundary:-20}
    read -p "Number of interior rings (default: 3): " num_rings
    num_rings=${num_rings:-3}
    
    echo ""
    echo -e "${BLUE}Visualization Options:${NC}"
    read -p "Enable FEM triangles? [y/n] (default: y): " use_fem
    use_fem=${use_fem:-y}
    
    read -p "Show SDF background + collision? [y/n] (default: y): " use_sdf
    use_sdf=${use_sdf:-y}
    
    echo ""
    echo -e "${BLUE}Window Configuration:${NC}"
    read -p "Window width in pixels (default: 1000): " winwidth
    winwidth=${winwidth:-1000}
    
    read -p "Window height in pixels (default: 600): " winheight
    winheight=${winheight:-600}
    
    read -p "Bounding box size (default: 2.5): " boxsize
    boxsize=${boxsize:-2.5}
    
    echo ""
    read -p "Device [cuda/cpu] (default: cuda): " device
    device=${device:-cuda}
    
    read -p "Duration in seconds (default: 30): " duration
    duration=${duration:-30}
else
    # Quick defaults
    radius=0.5
    num_boundary=20
    num_rings=3
    use_fem="y"
    use_sdf="y"
    winwidth=1000
    winheight=600
    boxsize=2.5
    device="cuda"
    duration=30
fi

# Build extra args
EXTRA_ARGS=""
if [ "$use_fem" != "y" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --no-fem"
fi
if [ "$use_sdf" == "y" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --with-sdf"
fi

echo ""
echo "========================================================================"
echo -e "${GREEN}Configuration Summary:${NC}"
echo "  Circle: radius=${radius}, boundary=${num_boundary}, rings=${num_rings}"
echo "  FEM: $([ "$use_fem" == "y" ] && echo "Enabled" || echo "Disabled")"
echo "  SDF + Collision: $([ "$use_sdf" == "y" ] && echo "Enabled" || echo "Disabled")"
echo "  Window: ${winwidth}×${winheight}"
echo "  Bounding box: ${boxsize}"
echo "  Device: ${device}"
echo "  Duration: ${duration}s"
echo ""
echo -e "${CYAN}Features being tested:${NC}"
echo "  ✓ Circle soft body mesh"
echo "  ✓ FEM triangles with strain coloring (blue→cyan→green)"
echo "  ✓ Springs with strain coloring (orange→yellow→red)"
echo "  ✓ Strain legends (Spring + FEM)"
if [ "$use_sdf" == "y" ]; then
    echo "  ✓ SDF background (brown=walls, white=passable)"
    echo "  ✓ Collision with walls"
fi
echo "========================================================================"
echo ""
read -p "Press Enter to start..."
echo ""

# Build command
CMD="python test_renderer.py \
    --radius ${radius} \
    --num-boundary ${num_boundary} \
    --num-rings ${num_rings} \
    --window-width ${winwidth} \
    --window-height ${winheight} \
    --boxsize ${boxsize} \
    --device ${device} \
    --duration ${duration} \
    ${EXTRA_ARGS}"

echo -e "${YELLOW}Starting Renderer Test...${NC}"
echo ""
echo -e "${CYAN}Controls:${NC}"
echo "  Q/ESC  - Quit"
echo "  SPACE  - Pause/Resume"
echo "  R      - Reset"
echo ""

# Allow X11 connections from Docker
echo "Setting up X11 display..."
xhost +local:docker 2>/dev/null || true

# Run in Docker
docker run -it --rm \
    --gpus all \
    --ipc=host \
    -e DISPLAY=$DISPLAY \
    -e PYTHONUNBUFFERED=1 \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v "$PROJECT_DIR:/workspace" \
    -w /workspace/pygame_renderer \
    "$IMAGE_NAME" \
    $CMD

echo ""
echo -e "${GREEN}Renderer test complete!${NC}"
echo ""
