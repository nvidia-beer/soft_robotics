#!/bin/bash
# Trajectory Tracking - Interactive Runner
#
# The simulation uses FEM (Finite Element Method) for accurate physics.
# The MPC uses a simplified spring-damper model for fast predictions.
# This STRUCTURAL MISMATCH is what the SNN learns to correct!
#
# Usage:
#   ./run_tracking.sh                    # Interactive menu
#   ./run_tracking.sh --controller mpc   # Direct run

set -e

# Get script directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
IMAGE_NAME="spring-mass-nengo"

# Enable X11 forwarding
xhost +local:docker 2>/dev/null || true

# Build Docker if needed
if ! docker image inspect "$IMAGE_NAME" &> /dev/null; then
    "$PROJECT_DIR/build.sh"
fi

# Check for GPU support
GPU_FLAGS=""
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    GPU_FLAGS="--gpus all"
    DEVICE="cuda"
    echo "GPU detected - using CUDA"
else
    DEVICE="cpu"
    echo "No GPU detected - using CPU"
fi

# If arguments provided, pass them directly
if [ $# -gt 0 ]; then
    echo "Running with arguments: $@"
    docker run --rm $GPU_FLAGS \
        --ipc=host \
        --shm-size=2g \
        -v "$PROJECT_DIR:/workspace/soft_robotics" \
        -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
        -e DISPLAY=$DISPLAY \
        -e SDL_VIDEO_X11_NET_WM_BYPASS_COMPOSITOR=0 \
        -w /workspace/soft_robotics/trajectory_tracking \
        "$IMAGE_NAME" \
        python3 run_tracking.py --device $DEVICE "$@"
    exit 0
fi

# Interactive menu
echo ""
echo "================================================================"
echo "    TRAJECTORY TRACKING - 2D FEM Spring Grid"
echo "================================================================"
echo ""
echo "  Simulation: FEM triangles (Neo-Hookean hyperelastic)"
echo "  MPC Model:  Simple spring-damper (fast approximation)"
echo ""
echo "  This STRUCTURAL MISMATCH is what the SNN learns to correct!"
echo ""
echo "Select an option:"
echo ""
echo "  === Classic Controllers ==="
echo "  1. PID"
echo "  2. MPC"
echo "  3. Stress (strain-based modulation)"
echo ""
echo "  === SNN Controllers ==="
echo "  4. SNN-PID (Zaidel et al.)"
echo "  5. SNN-Stress (spiking + strain)"
echo "  6. MPC + SNN (learns FEM dynamics)"
echo ""
echo "  === Comparison ==="
echo "  7. ★ 3-Way: PID vs SNN-PID vs SNN-Stress"
echo ""
echo "  8. Plot from saved .npz file"
echo "  9. Interactive shell"
echo "  0. Exit"
echo ""
read -p "Enter choice [0-9]: " choice

# Exit early for special options
if [ "$choice" = "9" ]; then
    echo ""
    echo "Starting interactive shell..."
    docker run -it --rm $GPU_FLAGS \
        --ipc=host \
        --shm-size=2g \
        -v "$PROJECT_DIR:/workspace/soft_robotics" \
        -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
        -e DISPLAY=$DISPLAY \
        -w /workspace/soft_robotics/trajectory_tracking \
        "$IMAGE_NAME" \
        /bin/bash
    exit 0
fi

if [ "$choice" = "0" ]; then
    echo "Exiting."
    exit 0
fi

# Plot from file option
if [ "$choice" = "8" ]; then
    echo ""
    echo "Available result files:"
    ls -la *.npz 2>/dev/null || echo "  (no .npz files found)"
    echo ""
    read -p "Enter .npz filename: " npz_file
    if [ -z "$npz_file" ]; then
        echo "No file specified."
        exit 1
    fi
    
    read -p "Figure size [width height, default: 14 10]: " figsize
    figsize=${figsize:-"14 10"}
    
    read -p "DPI [default: 150]: " dpi
    dpi=${dpi:-150}
    
    CMD="python3 run_tracking.py --plot-file $npz_file --figsize $figsize --dpi $dpi --device $DEVICE"
    
    echo ""
    echo "Running: $CMD"
    echo ""
    
    docker run --rm $GPU_FLAGS \
        --ipc=host \
        --shm-size=2g \
        -v "$PROJECT_DIR:/workspace/soft_robotics" \
        -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
        -e DISPLAY=$DISPLAY \
        -e SDL_VIDEO_X11_NET_WM_BYPASS_COMPOSITOR=0 \
        -w /workspace/soft_robotics/trajectory_tracking \
        "$IMAGE_NAME" \
        $CMD
    exit 0
fi

# Grid size selection
echo ""
echo "Grid size options:"
echo "  2 = 1 group  (simplest, fastest)"
echo "  3 = 4 groups (default, fast learning)"
echo "  4 = 9 groups (more complex)"
echo "  5 = 16 groups"
echo ""
read -p "Enter grid size [2-10, default: 3]: " grid_size
grid_size=${grid_size:-3}
GRID_ARG="--grid-size $grid_size"

# Simulation time selection
echo ""
echo "Simulation time (seconds):"
echo "  5   = quick test"
echo "  10  = short"
echo "  30  = default (good for SNN learning)"
echo "  60  = long"
echo ""
read -p "Enter time in seconds [default: 30]: " sim_time
sim_time=${sim_time:-30}
TIME_ARG="--total-time $sim_time"

CMD=""
case $choice in
    # Classic controllers
    1) CMD="python3 run_tracking.py --controller pid $TIME_ARG $GRID_ARG" ;;
    2) CMD="python3 run_tracking.py --controller mpc $TIME_ARG $GRID_ARG" ;;
    3) CMD="python3 run_tracking.py --controller stress $TIME_ARG $GRID_ARG" ;;
    # SNN controllers
    4) CMD="python3 run_tracking.py --controller snn_pid $TIME_ARG $GRID_ARG" ;;
    5) CMD="python3 run_tracking.py --controller snn_stress $TIME_ARG $GRID_ARG" ;;
    6) CMD="python3 run_tracking.py --controller snn_mpc $TIME_ARG $GRID_ARG" ;;
    # Comparison (with display - watch all 3 controllers run)
    # Same gains as GUI: Kp=250, Kd=80, Ki=0 (Pure PD)
    7) CMD="python3 run_tracking.py --compare3 --kp 250 --kd 80 --ki 0 $TIME_ARG $GRID_ARG" ;;
    *) echo "Invalid choice [0-9]."; exit 1 ;;
esac

CMD="$CMD --device $DEVICE"

echo ""
echo "Running: $CMD"
echo ""

docker run --rm $GPU_FLAGS \
    --ipc=host \
    --shm-size=2g \
    -v "$PROJECT_DIR:/workspace/soft_robotics" \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -e DISPLAY=$DISPLAY \
    -e SDL_VIDEO_X11_NET_WM_BYPASS_COMPOSITOR=0 \
    -w /workspace/soft_robotics/trajectory_tracking \
    "$IMAGE_NAME" \
    $CMD

echo ""
echo "Done!"
