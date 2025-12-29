#!/bin/bash
# =============================================================================
# World Map Module - SDF Collision Detection
# =============================================================================
#
# This module provides:
#   - WorldMap class for loading bitmap images
#   - SDF (Signed Distance Field) computation
#   - Collision detection and normal calculation
#
# For visual demos with rendering, use pygame_renderer:
#   cd ../pygame_renderer && ./run.sh
#
# =============================================================================

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "========================================================================"
echo -e "${CYAN}World Map Module${NC}"
echo "========================================================================"
echo ""
echo "This module provides SDF-based collision detection."
echo ""

# Get script directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
IMAGE_NAME="spring-mass-nengo"

echo -e "${BLUE}Options:${NC}"
echo "  1) Run unit tests (matplotlib visualization)"
echo "  2) Go to pygame_renderer for demos"
echo ""
read -p "Select [1/2] (default: 2): " choice
choice=${choice:-2}

if [ "$choice" == "1" ]; then
    echo ""
    echo -e "${YELLOW}Running WorldMap tests...${NC}"
    echo ""
    
    docker run -it --rm \
        --gpus all \
        -e DISPLAY=$DISPLAY \
        -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
        -v "$PROJECT_DIR:/workspace" \
        -w /workspace/world_map \
        "$IMAGE_NAME" \
        python test_world_map.py
    
    echo ""
    echo -e "${GREEN}Tests complete!${NC}"
else
    echo ""
    echo -e "${CYAN}For visual demos, run:${NC}"
    echo ""
    echo "  cd ../pygame_renderer && ./run.sh"
    echo ""
    echo "The pygame_renderer includes:"
    echo "  - SDF background (brown=walls, white=passable)"
    echo "  - Collision with walls"
    echo "  - FEM + spring visualization"
    echo "  - All rendering features"
    echo ""
fi
