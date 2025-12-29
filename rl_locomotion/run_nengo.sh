#!/bin/bash
# SNN CPG Locomotion Demo - Nengo GUI Interface
#
# Runs the CPG locomotion simulation with Nengo GUI for spike visualization.
# Shows both the pygame physics window AND the Nengo web interface.
#
# Features:
# - Real-time strain/tension spike visualization (7D per group)
# - Spiking Hopf CPG oscillators (2D per group)
# - CPG toggle and parameter sliders in Nengo GUI
# - Ratchet friction for directional locomotion
#
# Architecture (follows trajectory_tracking pattern):
# - Strain ensembles: 7D per group (5 springs + 2 FEMs)
# - CPG oscillators: 2D Hopf per group with Kuramoto coupling
# - Output ensembles: CPG + PES feedforward learning
# - Force injection: output → horizontal forces
#
# PES Learning:
# - strain[7D] → u(t)[1D] feedforward (learns to improve locomotion)
# - Error signal: velocity error (target - actual velocity)
#
# Usage:
#   ./run_snn.sh                      # Interactive menu (recommended)
#   ./run_snn.sh -N 4                 # 4x4 grid (9 groups)
#   ./run_snn.sh -F 5.0               # 5 Hz CPG frequency
#   ./run_snn.sh -N 3 -F 3.0 -A 0.8   # Grid + frequency + amplitude
#   ./run_snn.sh --coupling 0.5       # Kuramoto coupling strength
#   ./run_snn.sh --pes-lr 1e-4        # PES learning rate

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

# Enable X11 forwarding
xhost +local:docker 2>/dev/null || true

echo ""
echo "========================================================================"
echo -e "${PURPLE}SNN CPG Locomotion - Nengo GUI Interface${NC}"
echo "========================================================================"
echo ""
echo -e "${CYAN}Two-window visualization:${NC}"
echo "  1. Pygame: Physics simulation with soft body locomotion"
echo "  2. Browser: Nengo GUI with spike rasters (http://localhost:8080)"
echo ""

# Build Docker if needed
if ! docker image inspect "$IMAGE_NAME" &> /dev/null; then
    "$PROJECT_DIR/build.sh"
fi

# Check for GPU support
GPU_FLAGS=""
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    GPU_FLAGS="--gpus all"
    DEVICE="cuda"
    echo -e "${GREEN}✓ GPU detected - using CUDA${NC}"
else
    DEVICE="cpu"
    echo -e "${YELLOW}⚠ No GPU detected - using CPU${NC}"
fi

# If arguments provided, parse them
if [ $# -gt 0 ]; then
    echo ""
    echo -e "${BLUE}Running with arguments: $@${NC}"
    
    ENV_ARGS=""
    while [[ $# -gt 0 ]]; do
        case $1 in
            --grid-size|-N)
                ENV_ARGS="$ENV_ARGS -e SNN_GRID_SIZE=$2"
                shift 2
                ;;
            --frequency|-F)
                ENV_ARGS="$ENV_ARGS -e SNN_FREQUENCY=$2"
                shift 2
                ;;
            --amplitude|-A)
                ENV_ARGS="$ENV_ARGS -e SNN_AMPLITUDE=$2"
                shift 2
                ;;
            --neurons)
                ENV_ARGS="$ENV_ARGS -e SNN_N_NEURONS=$2"
                shift 2
                ;;
            --force-scale)
                ENV_ARGS="$ENV_ARGS -e SNN_FORCE_SCALE=$2"
                shift 2
                ;;
            --dir-x)
                ENV_ARGS="$ENV_ARGS -e SNN_DIR_X=$2"
                shift 2
                ;;
            --dir-y)
                ENV_ARGS="$ENV_ARGS -e SNN_DIR_Y=$2"
                shift 2
                ;;
            --coupling|-K)
                ENV_ARGS="$ENV_ARGS -e SNN_COUPLING=$2"
                shift 2
                ;;
            --pes-lr)
                ENV_ARGS="$ENV_ARGS -e SNN_PES_LR=$2"
                shift 2
                ;;
            *)
                shift
                ;;
        esac
    done
    
    ENV_ARGS="$ENV_ARGS -e SNN_DEVICE=$DEVICE"
    
    CMD="nengo --no-browser -l 0.0.0.0 -P 8080 --unsecure demo_snn_gui.py"
    
    docker run -it --rm $GPU_FLAGS \
        --ipc=host \
        --shm-size=2g \
        -p 8080:8080 \
        -e DISPLAY=$DISPLAY \
        -e PYTHONUNBUFFERED=1 \
        -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
        -e SDL_VIDEO_X11_NET_WM_BYPASS_COMPOSITOR=0 \
        $ENV_ARGS \
        -v "$PROJECT_DIR:/workspace/soft_robotics" \
        -w /workspace/soft_robotics/rl_locomotion \
        "$IMAGE_NAME" \
        $CMD
    exit 0
fi

# Interactive menu
echo ""
echo -e "${BLUE}Grid Configuration:${NC}"
echo "  Grid size determines number of control groups:"
echo "    2 = 1 group  (simplest)"
echo "    3 = 4 groups"
echo "    4 = 9 groups (default)"
echo "    5 = 16 groups"
echo ""
read -p "Grid size N (NxN particles, default: 4): " grid_n
grid_n=${grid_n:-4}
if [ "$grid_n" -lt 2 ]; then
    echo -e "${YELLOW}Warning: Grid size must be at least 2. Setting to 2.${NC}"
    grid_n=2
fi

echo ""
echo -e "${BLUE}CPG Configuration:${NC}"
echo ""
read -p "CPG frequency Hz (default: 4.0): " frequency
frequency=${frequency:-4.0}

read -p "CPG amplitude 0-1 (default: 1.0): " amplitude
amplitude=${amplitude:-1.0}

echo ""
echo -e "${BLUE}Direction (2D vector):${NC}"
echo "  (1, 0) = right,  (-1, 0) = left"
echo "  (0, 1) = up,     (0, -1) = down"
echo ""
read -p "Direction X (default: 1): " dir_x
dir_x=${dir_x:-1}
read -p "Direction Y (default: 0): " dir_y
dir_y=${dir_y:-0}

# Optional: Advanced settings
echo ""
read -p "Configure advanced settings? [y/n] (default: n): " advanced
advanced=${advanced:-n}

WINDOW_WIDTH="1000"
WINDOW_HEIGHT="500"
NEURONS="50"
FORCE_SCALE="20.0"
NENGO_DT="0.001"

if [ "$advanced" == "y" ]; then
    echo ""
    echo -e "${BLUE}Advanced Settings:${NC}"
    read -p "Window width (default: 1000): " WINDOW_WIDTH
    WINDOW_WIDTH=${WINDOW_WIDTH:-1000}
    
    read -p "Window height (default: 500): " WINDOW_HEIGHT
    WINDOW_HEIGHT=${WINDOW_HEIGHT:-500}
    
    read -p "Neurons per ensemble (default: 50): " NEURONS
    NEURONS=${NEURONS:-50}
    
    read -p "Force scale (default: 20.0): " FORCE_SCALE
    FORCE_SCALE=${FORCE_SCALE:-20.0}
    
    read -p "Nengo dt (default: 0.001): " NENGO_DT
    NENGO_DT=${NENGO_DT:-0.001}
fi

# Summary
echo ""
echo "========================================================================"
echo -e "${GREEN}Configuration Summary:${NC}"
echo "========================================================================"
echo ""
echo -e "  ${CYAN}Grid:${NC}        ${grid_n}x${grid_n} ($(( (grid_n-1)*(grid_n-1) )) groups)"
echo -e "  ${CYAN}CPG:${NC}         Spiking Hopf oscillators"
echo -e "  ${CYAN}Frequency:${NC}   ${frequency} Hz"
echo -e "  ${CYAN}Amplitude:${NC}   ${amplitude}"
echo -e "  ${CYAN}Direction:${NC}   (${dir_x}, ${dir_y})"
echo -e "  ${CYAN}Force scale:${NC} ${FORCE_SCALE}"
echo -e "  ${CYAN}Window:${NC}      ${WINDOW_WIDTH}x${WINDOW_HEIGHT}"
echo -e "  ${CYAN}Device:${NC}      $DEVICE"
echo ""
echo -e "  ${PURPLE}Nengo GUI:${NC}   http://localhost:8080"
echo -e "  ${PURPLE}Neurons:${NC}     $NEURONS per ensemble"
echo ""
echo "========================================================================"
echo ""
read -p "Press Enter to start simulation..."
echo ""

# Build environment variables
ENV_VARS=""
ENV_VARS="$ENV_VARS -e SNN_GRID_SIZE=$grid_n"
ENV_VARS="$ENV_VARS -e SNN_FREQUENCY=$frequency"
ENV_VARS="$ENV_VARS -e SNN_AMPLITUDE=$amplitude"
ENV_VARS="$ENV_VARS -e SNN_DIR_X=$dir_x"
ENV_VARS="$ENV_VARS -e SNN_DIR_Y=$dir_y"
ENV_VARS="$ENV_VARS -e SNN_DEVICE=$DEVICE"
ENV_VARS="$ENV_VARS -e SNN_WINDOW_WIDTH=$WINDOW_WIDTH"
ENV_VARS="$ENV_VARS -e SNN_WINDOW_HEIGHT=$WINDOW_HEIGHT"
ENV_VARS="$ENV_VARS -e SNN_N_NEURONS=$NEURONS"
ENV_VARS="$ENV_VARS -e SNN_FORCE_SCALE=$FORCE_SCALE"
ENV_VARS="$ENV_VARS -e SNN_NENGO_DT=$NENGO_DT"

# Nengo GUI command
CMD="nengo --no-browser -l 0.0.0.0 -P 8080 --unsecure demo_snn_gui.py"

echo -e "${BLUE}Executing:${NC} $CMD"
echo ""
echo -e "${YELLOW}Starting SNN CPG Locomotion...${NC}"
echo ""
echo "  You'll see TWO windows:"
echo ""
echo -e "  1. ${GREEN}Pygame window${NC} (Physics):"
echo "     - Spring-mass grid with particles, springs, FEM"
echo "     - Soft body locomotion with ratchet friction"
echo "     - CPG status and displacement display"
echo ""
echo -e "  2. ${GREEN}Browser${NC} (http://localhost:8080):"
echo "     - Strain spike rasters (7D per group)"
echo "     - CPG oscillator spikes (2D Hopf)"
echo "     - CPG output values"
echo "     - Parameter sliders"
echo ""
echo "  ${PURPLE}Steps:${NC}"
echo "  1. Wait for 'Nengo server started' message"
echo "  2. Open browser: http://localhost:8080"
echo "  3. Click ${PURPLE}PLAY ▶️${NC} in Nengo GUI to start simulation"
echo "  4. Use sliders to control:"
echo "     - 'CPG [0=OFF, 1=ON]' → Enable/disable CPG"
echo "     - 'Frequency [1-10 Hz]' → CPG oscillation rate"
echo "     - 'Amplitude [0-1]' → Force amplitude"
echo "  5. Right-click ensembles → Add 'Spike raster' for neuron activity"
echo "  6. Press Ctrl+C here to stop"
echo ""
echo "  ${CYAN}Note:${NC} Strain_G* shows spring/FEM strains as spiking neurons"
echo "        CPG_G* shows Hopf oscillator dynamics as spiking neurons"
echo ""

# Run in Docker
docker run -it --rm $GPU_FLAGS \
    --ipc=host \
    --shm-size=2g \
    -p 8080:8080 \
    -e DISPLAY=$DISPLAY \
    -e PYTHONUNBUFFERED=1 \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -e SDL_VIDEO_X11_NET_WM_BYPASS_COMPOSITOR=0 \
    $ENV_VARS \
    -v "$PROJECT_DIR:/workspace/soft_robotics" \
    -w /workspace/soft_robotics/rl_locomotion \
    "$IMAGE_NAME" \
    $CMD

echo ""
echo -e "${GREEN}Simulation complete!${NC}"
echo ""
