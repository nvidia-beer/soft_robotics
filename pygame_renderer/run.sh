#!/bin/bash
# =============================================================================
# Test script for Pygame Renderer
# =============================================================================
#
# Tests the Renderer class with all visualization features:
# 1. FEM / Spring strain legends
# 2. Tension visualization (springs + FEM colored by strain gradient)
# 3. Groups rendered in hot pink
# 4. External forces rendered by arrows
# 5. Optional SDF background (brown=collision, white=passable)
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
echo -e "${PURPLE}Pygame Renderer Test${NC}"
echo "========================================================================"
echo ""
echo "Tests the Renderer class with all visualization features."
echo ""

# Get script directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
IMAGE_NAME="spring-mass-nengo"

# Configuration
echo -e "${BLUE}Configuration:${NC}"
echo "  1) Quick test (defaults: 3x6 grid, FEM enabled)"
echo "  2) Custom configuration"
echo ""
read -p "Select [1/2] (default: 1): " config_choice
config_choice=${config_choice:-1}

if [ "$config_choice" == "2" ]; then
    # Custom configuration
    echo ""
    echo -e "${BLUE}Grid Configuration:${NC}"
    read -p "Grid rows (height, default: 3): " grid_rows
    grid_rows=${grid_rows:-3}
    read -p "Grid cols (width, default: 6): " grid_cols
    grid_cols=${grid_cols:-6}
    
    echo ""
    echo -e "${BLUE}Visualization Options:${NC}"
    read -p "Enable FEM triangles? [y/n] (default: y): " use_fem
    use_fem=${use_fem:-y}
    
    read -p "Show SDF background + collision (brown=walls)? [y/n] (default: y): " use_sdf
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
    grid_rows=3
    grid_cols=6
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
echo "  Grid: ${grid_cols}x${grid_rows} (wide x tall)"
echo "  FEM: $([ "$use_fem" == "y" ] && echo "Enabled" || echo "Disabled")"
echo "  SDF + Collision: $([ "$use_sdf" == "y" ] && echo "Enabled (brown walls)" || echo "Disabled")"
echo "  Window: ${winwidth}×${winheight}"
echo "  Bounding box: ${boxsize}"
echo "  Device: ${device}"
echo "  Duration: ${duration}s"
echo ""
echo -e "${CYAN}Features being tested:${NC}"
echo "  ✓ FEM triangles with strain coloring (blue→cyan→green)"
echo "  ✓ Springs with strain coloring (orange→yellow→red)"
echo "  ✓ Group centroids (hot pink circles with labels)"
echo "  ✓ Force arrows with magnitude gradient"
echo "  ✓ Strain legends (Spring + FEM)"
echo "  ✓ Force legend"
echo "  ✓ Group forces matrix display"
if [ "$use_sdf" == "y" ]; then
    echo "  ✓ SDF background (brown=walls, white=passable)"
    echo "  ✓ Collision with walls (particles bounce off brown areas)"
    echo "  ✓ Collision debug (blue=safe, cyan=near, green=colliding)"
fi
echo "========================================================================"
echo ""
read -p "Press Enter to start..."
echo ""

# Build command
CMD="python test_renderer.py \
    --rows ${grid_rows} \
    --cols ${grid_cols} \
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
echo "  F      - Apply random force pulse to groups"
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
