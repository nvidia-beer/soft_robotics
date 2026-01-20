#!/bin/bash
# Run the Inflating Circle demo with PID control in Docker
#
# Inflates by scaling FEM rest configuration (rest lengths & rest poses).
# Material naturally deforms to reach its new rest state.
#
# Usage:
#   ./run.sh          # Interactive mode
#   ./run.sh --quick  # Quick start with defaults
#
# Controls:
#   UP/DOWN arrows: Increase/decrease target volume ratio
#   +/-: Adjust PID Kp gain
#   R: Reset volume & PID state
#   SPACE: Toggle auto-inflation (oscillate between min and max)
#   1-5: Set FEM Poisson ratio (0.1-0.45)
#   Q/ESC: Quit

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Get script directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
IMAGE_NAME="spring-mass-nengo"

echo "========================================================================"
echo -e "${CYAN}  INFLATING CIRCLE DEMO - FEM REST CONFIG SCALING${NC}"
echo "========================================================================"
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
MAX_VOLUME=2.0
FEM_E=2000.0    # Young's modulus (stiff, volume-preserving)
FEM_NU=0.45     # Poisson ratio (near-incompressible)
SPRING_K=1000.0
RADIUS=0.5
NUM_BOUNDARY=16
NUM_RINGS=2
GRAVITY=-9.8    # Earth gravity
DT=0.01
STEPS=5000
DEVICE="cuda"
SOLVER_FLAG=""

# PID parameters for FEM rest config scaling
# Classic PID values from figure_classic.py for fair comparison
PID_KP=5.0
PID_KI=2.0
PID_KD=0.3

if [ "$QUICK_MODE" = false ]; then
    # Interactive configuration
    echo -e "${BLUE}Circle Configuration:${NC}"
    read -p "  Radius (default: 0.5): " input
    RADIUS=${input:-$RADIUS}
    read -p "  Boundary points (default: 16): " input
    NUM_BOUNDARY=${input:-$NUM_BOUNDARY}
    read -p "  Interior rings (default: 2): " input
    NUM_RINGS=${input:-$NUM_RINGS}
    echo ""

    echo -e "${BLUE}Inflation Parameters:${NC}"
    read -p "  Max volume ratio (default: 2.0 = double): " input
    MAX_VOLUME=${input:-$MAX_VOLUME}
    echo ""

    echo -e "${BLUE}PID Controller (adjusts FEM rest configuration):${NC}"
    echo -e "  ${YELLOW}Classic PID values from figure_classic.py${NC}"
    read -p "  Kp - proportional (default: ${PID_KP}): " input
    PID_KP=${input:-$PID_KP}
    read -p "  Ki - integral (default: ${PID_KI}): " input
    PID_KI=${input:-$PID_KI}
    read -p "  Kd - derivative (default: ${PID_KD}): " input
    PID_KD=${input:-$PID_KD}
    echo ""

    echo -e "${BLUE}FEM Material Parameters:${NC}"
    echo -e "  ${YELLOW}Higher E = stiffer, resists deformation${NC}"
    echo -e "  ${YELLOW}Higher nu = more incompressible (use 0.45-0.49)${NC}"
    read -p "  Young's modulus E (default: ${FEM_E}): " input
    FEM_E=${input:-$FEM_E}
    read -p "  Poisson ratio nu (default: ${FEM_NU}): " input
    FEM_NU=${input:-$FEM_NU}
    echo ""
    
    echo -e "${BLUE}Solver:${NC}"
    echo "  1) Implicit FEM (default)"
    echo "  2) VBD (Vertex Block Descent - GPU parallel)"
    read -p "  Enter choice [1-2] (default: 1): " solver_choice
    solver_choice=${solver_choice:-1}
    case $solver_choice in
        2)
            SOLVER_FLAG="--vbd"
            echo "  → VBD solver selected"
            ;;
        *)
            echo "  → Implicit FEM solver selected"
            ;;
    esac
    echo ""

    echo -e "${BLUE}Physics:${NC}"
    echo -e "  ${YELLOW}Gravity: 0=none, -5=medium, -9.8=Earth (default)${NC}"
    read -p "  Gravity (default: ${GRAVITY}): " input
    GRAVITY=${input:-$GRAVITY}
    echo ""

    echo -e "${BLUE}Simulation:${NC}"
    read -p "  Time step dt (default: 0.01): " input
    DT=${input:-$DT}
    read -p "  Number of steps (default: 5000): " input
    STEPS=${input:-$STEPS}
    read -p "  Device [cuda/cpu] (default: cuda): " input
    DEVICE=${input:-$DEVICE}
    echo ""
fi

# Summary
echo "========================================================================"
echo -e "${GREEN}Configuration Summary:${NC}"
echo "  Circle: radius=${RADIUS}, boundary=${NUM_BOUNDARY}, rings=${NUM_RINGS}"
echo "  Material: E=${FEM_E}, nu=${FEM_NU}, spring_k=${SPRING_K}"
echo "  Inflation: max=${MAX_VOLUME}x (FEM rest config scaling)"
echo "  PID: Kp=${PID_KP}, Ki=${PID_KI}, Kd=${PID_KD}"
echo "  Solver: $([ -z "$SOLVER_FLAG" ] && echo "Implicit FEM" || echo "VBD")"
echo "  Physics: gravity=${GRAVITY}"
echo "========================================================================"
echo ""

echo -e "${BLUE}Controls:${NC}"
echo "  UP/DOWN: Change target volume"
echo "  SPACE: Toggle auto-oscillation"
echo "  +/-: Adjust PID Kp gain"
echo "  R: Reset volume & PID state"
echo "  1-5: Set Poisson ratio (0.1-0.45)"
echo "  Q/ESC: Quit"
echo ""

if [ "$QUICK_MODE" = false ]; then
    read -p "Press Enter to start..."
    echo ""
fi

# Build command
CMD="python3 demo.py"
CMD="$CMD --radius $RADIUS --num-boundary $NUM_BOUNDARY --num-rings $NUM_RINGS"
CMD="$CMD --max-volume $MAX_VOLUME --spring-k $SPRING_K"
CMD="$CMD --pid-kp $PID_KP --pid-ki $PID_KI --pid-kd $PID_KD"
CMD="$CMD --fem-E $FEM_E --fem-nu $FEM_NU"
CMD="$CMD --dt $DT --steps $STEPS --device $DEVICE"
CMD="$CMD --gravity $GRAVITY"
CMD="$CMD $SOLVER_FLAG"

echo -e "${CYAN}Running: $CMD${NC}"
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
    -w /workspace/inflation_control \
    "$IMAGE_NAME" \
    $CMD

echo ""
echo -e "${GREEN}Demo complete!${NC}"
echo ""
