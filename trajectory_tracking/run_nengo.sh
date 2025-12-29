#!/bin/bash
# Trajectory Tracking SNN - Nengo GUI Interface
#
# Runs the trajectory tracking simulation with Nengo GUI for spike visualization.
# Shows both the pygame physics window AND the Nengo web interface.
#
# Features:
# - Real-time strain/tension spike visualization (neural encoding)
# - Controller toggle (ON/OFF) via Nengo slider
# - Controller selection (PID / MPC) via Nengo slider
# - Full TrackingEnv GUI with XY plots and error history
# - Both controllers loaded at startup - switch between them in GUI
#
# Architecture:
# - Classic PID/MPC controllers compute forces (not SNN-based)
# - Nengo GUI visualizes strain signals as spiking neural activity
# - Each strain input is rate-coded and shown as spike rasters
#
# Usage:
#   ./run_snn_gui.sh                        # Interactive menu (recommended)
#
# Direct run (skip interactive menu):
#   ./run_snn_gui.sh -N 4                   # 4x4 grid (9 groups)
#   ./run_snn_gui.sh -C 2                   # Start with SNN_PID controller
#   ./run_snn_gui.sh --trajectory figure8   # figure8 trajectory
#   ./run_snn_gui.sh -N 3 -C 1 -A 0.5       # Grid + MPC + amplitude

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
PURPLE='\033[0;35m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Get script directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
IMAGE_NAME="spring-mass-nengo"

# Enable X11 forwarding
xhost +local:docker 2>/dev/null || true

echo ""
echo "========================================================================"
echo -e "${PURPLE}Trajectory Tracking SNN - Nengo GUI Interface${NC}"
echo "========================================================================"
echo ""
echo -e "${CYAN}Two-window visualization:${NC}"
echo "  1. Pygame: Physics simulation with trajectory tracking"
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
    echo ""
    echo -e "${BLUE}Running with arguments: $@${NC}"
    
    # Parse arguments into environment variables
    ENV_ARGS=""
    while [[ $# -gt 0 ]]; do
        case $1 in
            --grid-size|-N)
                ENV_ARGS="$ENV_ARGS -e TRACKING_N=$2"
                shift 2
                ;;
            --controller|-C)
                ENV_ARGS="$ENV_ARGS -e TRACKING_CONTROLLER=$2"
                shift 2
                ;;
            --trajectory)
                ENV_ARGS="$ENV_ARGS -e TRACKING_TRAJECTORY=$2"
                shift 2
                ;;
            --amplitude|-A)
                ENV_ARGS="$ENV_ARGS -e TRACKING_AMPLITUDE=$2"
                shift 2
                ;;
            --frequency|-f)
                ENV_ARGS="$ENV_ARGS -e TRACKING_FREQUENCY=$2"
                shift 2
                ;;
            *)
                shift
                ;;
        esac
    done
    
    ENV_ARGS="$ENV_ARGS -e TRACKING_DEVICE=$DEVICE"
    
    CMD="nengo --no-browser -l 0.0.0.0 -P 8080 --unsecure snn_nengo_tracking_gui.py"
    
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
        -w /workspace/soft_robotics/trajectory_tracking \
        "$IMAGE_NAME" \
        $CMD
    exit 0
fi

# Interactive menu
echo ""
echo -e "${CYAN}Controllers available:${NC}"
echo "  0 = PID (classic)"
echo "  1 = MPC (classic)"
echo "  2 = SNN_PID (NEF-based spiking PD)"
echo "  3 = SNN_Stress (NEF-based spiking + strain feedback)"
echo ""
read -p "Initial controller [0/1/2/3] (default: 0=PID): " controller_type
controller_type=${controller_type:-0}
# Validate controller type
if [[ ! "$controller_type" =~ ^[0123]$ ]]; then
    echo -e "${YELLOW}Invalid controller type. Using default (0=PID).${NC}"
    controller_type=0
fi
controller_names=("PID" "MPC" "SNN_PID" "SNN_Stress")
echo -e "${GREEN}✓ Selected: ${controller_names[$controller_type]}${NC}"
echo ""
echo -e "${BLUE}Grid Configuration:${NC}"
echo "  Grid size determines number of control groups:"
echo "    2 = 1 group  (simplest)"
echo "    3 = 4 groups (default)"
echo "    4 = 9 groups"
echo "    5 = 16 groups"
echo ""
read -p "Grid size N (NxN particles, default: 3, min: 2): " grid_n
grid_n=${grid_n:-3}
# Ensure minimum grid size of 2
if [ "$grid_n" -lt 2 ]; then
    echo -e "${YELLOW}Warning: Grid size must be at least 2. Setting to 2.${NC}"
    grid_n=2
fi

echo ""
echo -e "${BLUE}Trajectory Configuration:${NC}"
echo "  Types: sinusoidal, circular, figure8"
echo ""
read -p "Trajectory type (default: circular): " trajectory
trajectory=${trajectory:-circular}

read -p "Amplitude (default: 0.3): " amplitude
amplitude=${amplitude:-0.3}

read -p "Frequency Hz (default: 0.2): " frequency
frequency=${frequency:-0.2}

# Optional: Advanced settings
echo ""
read -p "Configure advanced settings? [y/n] (default: n): " advanced
advanced=${advanced:-n}

WINDOW_WIDTH="1500"
WINDOW_HEIGHT="800"
NEURONS="50"
NENGO_DT="0.01"

if [ "$advanced" == "y" ]; then
    echo ""
    echo -e "${BLUE}Advanced Settings:${NC}"
    read -p "Window width (default: 1500): " WINDOW_WIDTH
    WINDOW_WIDTH=${WINDOW_WIDTH:-1500}
    
    read -p "Window height (default: 800): " WINDOW_HEIGHT
    WINDOW_HEIGHT=${WINDOW_HEIGHT:-800}
    
    read -p "Neurons per ensemble (default: 50): " NEURONS
    NEURONS=${NEURONS:-50}
    
    read -p "Nengo dt (default: 0.01): " NENGO_DT
    NENGO_DT=${NENGO_DT:-0.01}
fi

# Summary
echo ""
echo "========================================================================"
echo -e "${GREEN}Configuration Summary:${NC}"
echo "========================================================================"
echo ""
echo -e "  ${CYAN}Controller:${NC}  ${controller_names[$controller_type]} (can toggle in GUI)"
echo -e "  ${CYAN}Grid:${NC}        ${grid_n}x${grid_n} ($(( (grid_n-1)*(grid_n-1) )) groups)"
echo -e "  ${CYAN}Trajectory:${NC}  $trajectory (A=$amplitude, f=${frequency}Hz)"
echo -e "  ${CYAN}Window:${NC}      ${WINDOW_WIDTH}x${WINDOW_HEIGHT}"
echo -e "  ${CYAN}Device:${NC}      $DEVICE"
echo ""
echo -e "  ${PURPLE}Nengo GUI:${NC}   http://localhost:8080"
echo -e "  ${PURPLE}Neurons:${NC}     $NEURONS per strain input"
echo ""
echo "========================================================================"
echo ""
read -p "Press Enter to start simulation..."
echo ""

# Build environment variables
ENV_VARS=""
ENV_VARS="$ENV_VARS -e TRACKING_N=$grid_n"
ENV_VARS="$ENV_VARS -e TRACKING_CONTROLLER=$controller_type"
ENV_VARS="$ENV_VARS -e TRACKING_TRAJECTORY=$trajectory"
ENV_VARS="$ENV_VARS -e TRACKING_AMPLITUDE=$amplitude"
ENV_VARS="$ENV_VARS -e TRACKING_FREQUENCY=$frequency"
ENV_VARS="$ENV_VARS -e TRACKING_DEVICE=$DEVICE"
ENV_VARS="$ENV_VARS -e TRACKING_WINDOW_WIDTH=$WINDOW_WIDTH"
ENV_VARS="$ENV_VARS -e TRACKING_WINDOW_HEIGHT=$WINDOW_HEIGHT"
ENV_VARS="$ENV_VARS -e TRACKING_NEURONS=$NEURONS"
ENV_VARS="$ENV_VARS -e TRACKING_NENGO_DT=$NENGO_DT"

# Nengo GUI command
CMD="nengo --no-browser -l 0.0.0.0 -P 8080 --unsecure snn_nengo_tracking_gui.py"

echo -e "${BLUE}Executing:${NC} $CMD"
echo ""
echo -e "${YELLOW}Starting Trajectory Tracking SNN...${NC}"
echo ""
echo "  You'll see TWO windows:"
echo ""
echo -e "  1. ${GREEN}Pygame window${NC} (from TrackingEnv):"
echo "     - Left: Spring-mass grid with particles, springs, FEM"
echo "     - Right: XY trajectory plot + Error history"
echo "     - Controller status overlay (ON/OFF)"
echo ""
echo -e "  2. ${GREEN}Browser${NC} (http://localhost:8080):"
echo "     - Strain spike rasters (per group)"
echo "     - Decoded strain values"
echo "     - Force display ensembles"
echo "     - Controller toggle slider"
echo ""
echo "  ${PURPLE}Steps:${NC}"
echo "  1. Wait for 'Nengo server started' message"
echo "  2. Open browser: http://localhost:8080"
echo "  3. Click ${PURPLE}PLAY ▶️${NC} in Nengo GUI to start simulation"
echo "  4. Use sliders to control:"
echo "     - 'Controller [0=OFF, 1=ON]' → Enable/disable controller"
echo "     - 'Type [0=PID, 1=MPC, 2=SNN_PID, 3=SNN_Stress]' → Select controller:"
echo "         0 = Classic PID"
echo "         1 = Classic MPC"
echo "         2 = SNN-PID (NEF-based spiking PD)"
echo "         3 = SNN-Stress (NEF-based + strain feedback)"
echo "     - 'PID Kp/Ki/Kd' → Tune PID gains in real-time"
echo "     - 'u_max' → Adjust maximum control force"
echo "     - 'Traj [0-2]' → Change trajectory (0=sin, 1=circle, 2=figure8)"
echo "  5. Right-click ensembles → Add 'Spike raster' for neuron activity"
echo "  6. Press Ctrl+C here to stop"
echo ""
echo "  ${CYAN}Note:${NC} Strain sensors shown as spiking neurons (rate coding)"
echo "        PID gains update in real-time as you drag sliders!"
echo ""

# Run in Docker with port forwarding
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
    -w /workspace/soft_robotics/trajectory_tracking \
    "$IMAGE_NAME" \
    $CMD

echo ""
echo -e "${GREEN}Simulation complete!${NC}"
echo ""

