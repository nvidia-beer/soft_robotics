#!/bin/bash
# =============================================================================
# Tessellation - Convert bitmap to mesh
# =============================================================================
#
# Converts model.bmp to a tessellated mesh (JSON) for use with openai-gym.
#
# Output:
#   model.json     - Mesh data (vertices, triangles, springs)
#   model_tes.bmp  - Visualization of the tessellation
#
# Usage in openai-gym:
#   cd ../openai-gym && ./run.sh
#   Select "Load from tessellation" and use: ../tessellation/model.json
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
echo -e "${CYAN}Tessellation - Bitmap to Mesh${NC}"
echo "========================================================================"
echo ""

PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
IMAGE_NAME="spring-mass-nengo"

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Check if model.bmp exists
if [ ! -f "$SCRIPT_DIR/model.bmp" ]; then
    echo -e "${YELLOW}Warning: model.bmp not found${NC}"
    echo ""
    echo "Please create or copy a bitmap image named 'model.bmp' to:"
    echo "  $SCRIPT_DIR/model.bmp"
    echo ""
    echo "Format:"
    echo "  - White (255) = mesh area"
    echo "  - Black (0) = empty/holes"
    echo ""
    exit 1
fi

echo -e "${GREEN}Found: $SCRIPT_DIR/model.bmp${NC}"

echo -e "${BLUE}Input:${NC}  model.bmp"
echo -e "${BLUE}Output:${NC} model.json (mesh data)"
echo -e "${BLUE}        ${NC} model_tes.bmp (visualization)"
echo ""

# Configuration
echo -e "${BLUE}Tessellation Parameters:${NC}"
read -p "Interior spacing (default: 16): " spacing
spacing=${spacing:-16}
read -p "Max aspect ratio (default: 10.0): " aspect
aspect=${aspect:-10.0}
read -p "Scale factor (default: 6): " scale
scale=${scale:-6}

echo ""
echo "========================================================================"
echo -e "${GREEN}Configuration:${NC}"
echo "  Interior spacing: ${spacing}"
echo "  Max aspect ratio: ${aspect}"
echo "  Scale factor: ${scale}"
echo "========================================================================"
echo ""
read -p "Press Enter to start tessellation..."
echo ""

# Build Python command
CMD="python3 -c \"
from refined_delaunay import refined_delaunay_tessellation

result, stats = refined_delaunay_tessellation(
    'model.bmp',
    'model.json',
    'model_tes.bmp',
    interior_spacing=${spacing},
    max_aspect_ratio=${aspect},
    scale_factor=${scale},
    normalize=True,
    min_area_factor=0.1
)

print()
print('✅ TESSELLATION COMPLETE!')
print(f'   {stats[\"valid\"]} triangles in final mesh')
print(f'   Output: model.json, model_tes.bmp')
\""

echo -e "${YELLOW}Running tessellation...${NC}"
echo ""

# Run in Docker
docker run -it --rm \
    -v "$PROJECT_DIR:/workspace" \
    -w /workspace/tessellation \
    "$IMAGE_NAME" \
    bash -c "$CMD"

echo ""
echo "========================================================================"
echo -e "${GREEN}Done!${NC}"
echo ""
echo "To use the mesh in openai-gym:"
echo "  cd ../openai-gym"
echo "  ./run.sh"
echo "  Select 'Load from tessellation' → ../tessellation/model.json"
echo "========================================================================"
echo ""

