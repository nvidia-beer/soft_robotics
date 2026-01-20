#!/bin/bash
# Inflation Control SNN - Nengo GUI Interface
#
# Runs the inflation control simulation with Nengo GUI for spike visualization.
# Shows both the pygame physics window AND the Nengo web interface.
#
# Features:
# - Real-time boundary strain spike visualization (neural encoding)
# - Controller toggle (ON/OFF) via Nengo slider
# - Controller selection (SNN_PID / SNN_Stress) via Nengo slider
# - Full inflation GUI with volume plots and error history
# - Both controllers loaded at startup - switch between them in GUI
#
# Architecture:
# - SNN_PID: NEF-based spiking PD control (Zaidel et al. 2021)
# - SNN_Stress: NEF-based spiking PD + PES feedforward learning from strain
#
# Usage:
#   ./run_nengo.sh                        # Interactive menu (recommended)
#
# Direct run (skip interactive menu):
#   ./run_nengo.sh -C 0                   # Start with SNN_PID controller
#   ./run_nengo.sh -C 1                   # Start with SNN_Stress controller
#   ./run_nengo.sh --max-volume 2.0       # Set max volume ratio
#   ./run_nengo.sh --gravity -9.8         # Enable gravity (default: 0)
#   ./run_nengo.sh --neurons 2000         # Set neurons per ensemble
#   ./run_nengo.sh --pid-kp 4.5           # Set PID Kp gain

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
echo -e "${PURPLE}Inflation Control SNN - Nengo GUI Interface${NC}"
echo "========================================================================"
echo ""
echo -e "${CYAN}Two-window visualization:${NC}"
echo "  1. Pygame: Balloon inflation simulation"
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

# If arguments provided, pass them as environment variables
if [ $# -gt 0 ]; then
    echo -e "${BLUE}Running with arguments: $@${NC}"
    
    ENV_ARGS=""
    while [[ $# -gt 0 ]]; do
        case $1 in
            --controller|-C) ENV_ARGS="$ENV_ARGS -e INFLATION_CONTROLLER=$2"; shift 2 ;;
            --max-volume|-V) ENV_ARGS="$ENV_ARGS -e INFLATION_MAX_VOLUME=$2"; shift 2 ;;
            --gravity|-g) ENV_ARGS="$ENV_ARGS -e INFLATION_GRAVITY=$2"; shift 2 ;;
            --pid-kp) ENV_ARGS="$ENV_ARGS -e INFLATION_PID_KP=$2"; shift 2 ;;
            --pid-ki) ENV_ARGS="$ENV_ARGS -e INFLATION_PID_KI=$2"; shift 2 ;;
            --pid-kd) ENV_ARGS="$ENV_ARGS -e INFLATION_PID_KD=$2"; shift 2 ;;
            --pes-lr) ENV_ARGS="$ENV_ARGS -e INFLATION_PES_LR=$2"; shift 2 ;;
            --neurons|-n) ENV_ARGS="$ENV_ARGS -e INFLATION_NEURONS=$2"; shift 2 ;;
            *) shift ;;
        esac
    done
    
    ENV_ARGS="$ENV_ARGS -e INFLATION_DEVICE=$DEVICE"
    CMD="nengo --no-browser -l 0.0.0.0 -P 8080 --unsecure snn_nengo_inflation_gui.py"
    
    docker run -it --rm $GPU_FLAGS --ipc=host --shm-size=2g -p 8080:8080 \
        -e DISPLAY=$DISPLAY -e PYTHONUNBUFFERED=1 \
        -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
        -e SDL_VIDEO_X11_NET_WM_BYPASS_COMPOSITOR=0 \
        $ENV_ARGS \
        -v "$PROJECT_DIR:/workspace/soft_robotics" \
        -w /workspace/soft_robotics/inflation_control \
        "$IMAGE_NAME" $CMD
    exit 0
fi

# Interactive menu
echo ""
echo -e "${CYAN}Controllers available:${NC}"
echo "  0 = SNN_PID (NEF-based spiking PD)"
echo "  1 = SNN_Stress (NEF-based spiking PD + PES feedforward learning)"
echo ""
read -p "Initial controller [0/1] (default: 0=SNN_PID): " controller_type
controller_type=${controller_type:-0}
# Validate controller type
if [[ ! "$controller_type" =~ ^[01]$ ]]; then
    echo -e "${YELLOW}Invalid controller type. Using default (0=SNN_PID).${NC}"
    controller_type=0
fi
controller_names=("SNN_PID" "SNN_Stress")
echo -e "${GREEN}✓ Selected: ${controller_names[$controller_type]}${NC}"

echo ""
read -p "Max volume ratio (default: 2.0): " max_volume
max_volume=${max_volume:-2.0}

echo ""
echo -e "${BLUE}Gravity:${NC}"
echo "  0 = No gravity (default)"
echo "  -9.8 = Earth gravity (balloon falls)"
read -p "Gravity Y-component (default: 0): " gravity
gravity=${gravity:-0}

# PES learning rate (for SNN_Stress only)
if [ "$controller_type" == "1" ]; then
    echo ""
    echo -e "${BLUE}PES Learning (SNN_Stress only):${NC}"
    read -p "PES learning rate (default: 1e-4): " pes_lr
    pes_lr=${pes_lr:-1e-4}
else
    pes_lr="1e-4"
fi

# Neuron count
echo ""
echo -e "${BLUE}SNN Configuration:${NC}"
read -p "Neurons per ensemble (default: 500): " n_neurons
n_neurons=${n_neurons:-500}

# PID gains (from snn_gains.txt tuning)
echo ""
echo -e "${BLUE}PID Gains:${NC}"
read -p "Kp - proportional (default: 4.5): " pid_kp
pid_kp=${pid_kp:-4.5}
read -p "Ki - integral (default: 1.25): " pid_ki
pid_ki=${pid_ki:-1.25}
read -p "Kd - derivative (default: 0.05): " pid_kd
pid_kd=${pid_kd:-0.05}

# Summary
echo ""
echo "========================================================================"
echo -e "${GREEN}Configuration Summary:${NC}"
echo "========================================================================"
echo ""
echo -e "  ${CYAN}Controller:${NC}  ${controller_names[$controller_type]} (can toggle in GUI)"
echo -e "  ${CYAN}Neurons:${NC}     $n_neurons per ensemble"
echo -e "  ${CYAN}Max Volume:${NC}  ${max_volume}x"
echo -e "  ${CYAN}Gravity:${NC}     $gravity"
echo -e "  ${CYAN}PID Gains:${NC}   Kp=$pid_kp, Ki=$pid_ki, Kd=$pid_kd"
if [ "$controller_type" == "1" ]; then
    echo -e "  ${CYAN}PES Rate:${NC}    $pes_lr"
fi
echo -e "  ${CYAN}Device:${NC}      $DEVICE"
echo ""
echo -e "  ${PURPLE}Nengo GUI:${NC}   http://localhost:8080"
echo ""
echo "========================================================================"
echo ""
read -p "Press Enter to start simulation..."

# Build environment variables
ENV_VARS="-e INFLATION_CONTROLLER=$controller_type"
ENV_VARS="$ENV_VARS -e INFLATION_NEURONS=$n_neurons"
ENV_VARS="$ENV_VARS -e INFLATION_MAX_VOLUME=$max_volume"
ENV_VARS="$ENV_VARS -e INFLATION_GRAVITY=$gravity"
ENV_VARS="$ENV_VARS -e INFLATION_PID_KP=$pid_kp"
ENV_VARS="$ENV_VARS -e INFLATION_PID_KI=$pid_ki"
ENV_VARS="$ENV_VARS -e INFLATION_PID_KD=$pid_kd"
ENV_VARS="$ENV_VARS -e INFLATION_PES_LR=$pes_lr"
ENV_VARS="$ENV_VARS -e INFLATION_DEVICE=$DEVICE"

CMD="nengo --no-browser -l 0.0.0.0 -P 8080 --unsecure snn_nengo_inflation_gui.py"

echo ""
echo -e "${YELLOW}Starting Inflation Control SNN...${NC}"
echo ""
echo "  You'll see TWO windows:"
echo ""
echo -e "  1. ${GREEN}Pygame window${NC} (Balloon inflation):"
echo "     - Left: Info panel with volume, error, pressure"
echo "     - Right: Balloon visualization with strain colors"
echo "     - Progress bar showing volume ratio"
echo ""
echo -e "  2. ${GREEN}Browser${NC} (http://localhost:8080):"
echo "     - Boundary strain spike rasters"
echo "     - SNN ensembles (q, ei, ed, u) or (q, ed, u, s)"
echo "     - Controller toggle slider"
echo ""
echo -e "  ${PURPLE}Steps:${NC}"
echo "  1. Wait for 'Nengo server started' message"
echo "  2. Open browser: http://localhost:8080"
echo "  3. Click PLAY in Nengo GUI to start simulation"
echo "  4. Use sliders to control:"
echo "     - 'Controller [0=OFF, 1=ON]' → Enable/disable controller"
echo "     - 'Target Volume [1.0-${max_volume}]' → Set target volume"
echo "  5. Right-click ensembles → Add 'Spike raster' for neuron activity"
echo "  6. Press Ctrl+C here to stop"
echo ""
if [ "$controller_type" == "1" ]; then
    echo -e "  ${CYAN}Note:${NC} SNN_Stress learns strain→pressure mapping via PES!"
    echo "        Watch the strain ensemble to see learning in action."
    echo ""
fi

docker run -it --rm $GPU_FLAGS --ipc=host --shm-size=2g -p 8080:8080 \
    -e DISPLAY=$DISPLAY -e PYTHONUNBUFFERED=1 \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -e SDL_VIDEO_X11_NET_WM_BYPASS_COMPOSITOR=0 \
    $ENV_VARS \
    -v "$PROJECT_DIR:/workspace/soft_robotics" \
    -w /workspace/soft_robotics/inflation_control \
    "$IMAGE_NAME" $CMD

echo ""
echo -e "${GREEN}Simulation complete!${NC}"
echo ""
