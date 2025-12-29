#!/bin/bash
# =============================================================================
# Build Docker Image for Soft Robotics Simulations
# =============================================================================
#
# This script builds the spring-mass-nengo Docker image used by all modules.
#
# Usage:
#   ./build.sh           # Build if image doesn't exist
#   ./build.sh --force   # Force rebuild (removes existing image first)
#
# =============================================================================

set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Get script directory (project root)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
IMAGE_NAME="spring-mass-nengo"

echo "========================================================================"
echo -e "${CYAN}Soft Robotics - Docker Build${NC}"
echo "========================================================================"
echo ""
echo -e "  ${BLUE}Image:${NC}      $IMAGE_NAME"
echo -e "  ${BLUE}Dockerfile:${NC} .devcontainer/Dockerfile"
echo -e "  ${BLUE}Context:${NC}    $SCRIPT_DIR"
echo ""

# Check for --force flag
FORCE_REBUILD=false
if [ "$1" == "--force" ] || [ "$1" == "-f" ]; then
    FORCE_REBUILD=true
fi

# Check if image exists
if docker image inspect "$IMAGE_NAME" &> /dev/null; then
    if [ "$FORCE_REBUILD" == true ]; then
        echo -e "${YELLOW}Removing existing image for rebuild...${NC}"
        docker rmi "$IMAGE_NAME"
        echo ""
    else
        echo -e "${GREEN}✓ Docker image '$IMAGE_NAME' already exists.${NC}"
        echo ""
        echo "  To rebuild, run: ./build.sh --force"
        echo ""
        exit 0
    fi
fi

# Build the image
echo -e "${YELLOW}Building Docker image (this may take 5-10 minutes)...${NC}"
echo ""

cd "$SCRIPT_DIR"
docker build -t "$IMAGE_NAME" -f .devcontainer/Dockerfile . || {
    echo ""
    echo -e "${RED}✗ Docker build failed!${NC}"
    exit 1
}

echo ""
echo -e "${GREEN}✓ Docker image '$IMAGE_NAME' built successfully!${NC}"
echo ""
echo "You can now run any module:"
echo "  ./rl_locomotion/run.sh"
echo "  ./trajectory_tracking/run.sh"
echo "  ./openai-gym/run.sh"
echo "  ./pygame_renderer/run.sh"
echo "  ./tessellation/run.sh"
echo ""

