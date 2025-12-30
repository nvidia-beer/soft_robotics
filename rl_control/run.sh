#!/bin/bash
# Run RL Control Demos - Soft Robot Locomotion Challenges
#
# Modular framework for locomotion testing with SDF terrain.
# New demos can easily be added by extending the DemoBase class.

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
PURPLE='\033[0;35m'
RED='\033[0;31m'
NC='\033[0m'

echo "========================================================================"
echo -e "${PURPLE}RL Control: Soft Robot Locomotion Challenges${NC}"
echo "========================================================================"
echo ""
echo "Available demos (extensible framework):"
echo ""
echo -e "  ${CYAN}1. PLANE${NC}         - Flat ground locomotion (classic, move right)"
echo -e "  ${CYAN}2. SLANT${NC}         - Climb inclined plane (default: 20°)"
echo -e "  ${CYAN}3. TUNNEL${NC}        - Squeeze through passage (default: 90% height)"
echo -e "  ${CYAN}4. BOULDER${NC}       - Climb over obstacle (default: 50% size)"
echo -e "  ${CYAN}5. ANGLED PLANE${NC}  - Uniform tilted plane (default: 20°, no flat start)"
echo ""
echo "All demos use classic CPG control (no SNN)."
echo ""

# Get script directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
IMAGE_NAME="spring-mass-nengo"

# Demo selection
echo -e "${BLUE}Select Demo:${NC}"
echo "  1) Plane (flat ground, classic locomotion)"
echo "  2) Slant (20° incline, configurable)"
echo "  3) Tunnel (90% height, configurable)"
echo "  4) Boulder (50% size, configurable)"
echo "  5) Angled Plane (20° uniform tilt, no flat start)"
echo "  6) Custom configuration"
echo ""
read -p "Select [1/2/3/4/5/6] (default: 1): " demo_choice
demo_choice=${demo_choice:-1}

case "$demo_choice" in
    1)
        echo ""
        echo -e "${CYAN}=== PLANE DEMO (Classic Locomotion) ===${NC}"
        
        echo ""
        echo "Direction (2D vector):"
        echo "  (1, 0) = right,  (-1, 0) = left"
        echo "  (0, 1) = up,     (0, -1) = down"
        read -p "Direction X (default: 1): " dir_x
        dir_x=${dir_x:-1}
        read -p "Direction Y (default: 0): " dir_y
        dir_y=${dir_y:-0}
        
        read -p "CPG frequency Hz (default: 4.0): " frequency
        frequency=${frequency:-4.0}
        
        read -p "Duration seconds (default: 30): " duration
        duration=${duration:-30}
        
        CMD="python demo_plane.py \
            --direction ${dir_x} ${dir_y} \
            --frequency ${frequency} \
            --duration ${duration}"
        
        DEMO_NAME="Plane (${dir_x}, ${dir_y})"
        ;;
        
    2)
        echo ""
        echo -e "${CYAN}=== SLANT DEMO ===${NC}"
        read -p "Slant angle (degrees, default: 20): " angle
        angle=${angle:-20}
        
        read -p "CPG frequency Hz (default: 4.0): " frequency
        frequency=${frequency:-4.0}
        
        read -p "Duration seconds (default: 60): " duration
        duration=${duration:-60}
        
        CMD="python demo_slant.py \
            --angle ${angle} \
            --frequency ${frequency} \
            --duration ${duration}"
        
        DEMO_NAME="Slant ${angle}°"
        ;;
        
    3)
        echo ""
        echo -e "${CYAN}=== TUNNEL DEMO ===${NC}"
        read -p "Tunnel height ratio (0.0-1.0, default: 0.9): " tunnel_ratio
        tunnel_ratio=${tunnel_ratio:-0.9}
        
        read -p "Tunnel length (default: 3.0): " tunnel_length
        tunnel_length=${tunnel_length:-3.0}
        
        read -p "CPG frequency Hz (default: 4.0): " frequency
        frequency=${frequency:-4.0}
        
        read -p "Duration seconds (default: 60): " duration
        duration=${duration:-60}
        
        CMD="python demo_tunnel.py \
            --tunnel-ratio ${tunnel_ratio} \
            --tunnel-length ${tunnel_length} \
            --frequency ${frequency} \
            --duration ${duration}"
        
        DEMO_NAME="Tunnel ${tunnel_ratio}"
        ;;
        
    4)
        echo ""
        echo -e "${CYAN}=== BOULDER DEMO ===${NC}"
        read -p "Boulder size ratio (0.0-1.0, default: 0.5): " boulder_ratio
        boulder_ratio=${boulder_ratio:-0.5}
        
        read -p "Boulder position X (default: 3.0): " boulder_pos
        boulder_pos=${boulder_pos:-3.0}
        
        read -p "CPG frequency Hz (default: 4.0): " frequency
        frequency=${frequency:-4.0}
        
        read -p "Duration seconds (default: 60): " duration
        duration=${duration:-60}
        
        read -p "Debug SDF collision? (y/N): " debug_sdf
        DEBUG_FLAG=""
        if [[ "$debug_sdf" == "y" || "$debug_sdf" == "Y" ]]; then
            DEBUG_FLAG="--debug-sdf"
        fi
        
        CMD="python demo_boulder.py \
            --boulder-ratio ${boulder_ratio} \
            --boulder-position ${boulder_pos} \
            --frequency ${frequency} \
            --duration ${duration} \
            ${DEBUG_FLAG}"
        
        DEMO_NAME="Boulder ${boulder_ratio}"
        ;;
        
    5)
        echo ""
        echo -e "${CYAN}=== ANGLED PLANE DEMO (20° default) ===${NC}"
        read -p "Plane angle (degrees, default: 20): " angle
        angle=${angle:-20}
        
        read -p "CPG frequency Hz (default: 4.0): " frequency
        frequency=${frequency:-4.0}
        
        read -p "Force scale (default: 20.0): " force_scale
        force_scale=${force_scale:-20.0}
        
        read -p "Duration seconds (default: 60): " duration
        duration=${duration:-60}
        
        CMD="python demo_angled_plane.py \
            --angle ${angle} \
            --frequency ${frequency} \
            --force-scale ${force_scale} \
            --duration ${duration}"
        
        DEMO_NAME="Angled Plane ${angle}°"
        ;;
        
    6)
        echo ""
        echo -e "${CYAN}=== CUSTOM CONFIGURATION ===${NC}"
        echo ""
        echo "Select demo type:"
        echo "  p) Plane (flat ground)"
        echo "  s) Slant"
        echo "  t) Tunnel"
        echo "  b) Boulder"
        echo "  a) Angled Plane (uniform tilt)"
        read -p "Type [p/s/t/b/a]: " demo_type
        
        # Common parameters
        read -p "Grid rows/height (default: 3): " grid_rows
        grid_rows=${grid_rows:-3}
        read -p "Grid cols/width (default: 6): " grid_cols
        grid_cols=${grid_cols:-6}
        
        read -p "CPG frequency Hz (default: 4.0): " frequency
        frequency=${frequency:-4.0}
        
        read -p "CPG amplitude (default: 1.0): " amplitude
        amplitude=${amplitude:-1.0}
        
        read -p "Force scale (default: 25.0): " force_scale
        force_scale=${force_scale:-25.0}
        
        read -p "Duration seconds (default: 60): " duration
        duration=${duration:-60}
        
        case "$demo_type" in
            p|P)
                echo ""
                echo "Direction (2D vector):"
                read -p "Direction X (default: 1): " dir_x
                dir_x=${dir_x:-1}
                read -p "Direction Y (default: 0): " dir_y
                dir_y=${dir_y:-0}
                CMD="python demo_plane.py \
                    --rows ${grid_rows} --cols ${grid_cols} \
                    --direction ${dir_x} ${dir_y} \
                    --frequency ${frequency} \
                    --amplitude ${amplitude} \
                    --force-scale ${force_scale} \
                    --duration ${duration}"
                DEMO_NAME="Plane (custom)"
                ;;
            s|S)
                read -p "Slant angle (degrees, default: 20): " angle
                angle=${angle:-20}
                CMD="python demo_slant.py \
                    --rows ${grid_rows} --cols ${grid_cols} \
                    --angle ${angle} \
                    --frequency ${frequency} \
                    --amplitude ${amplitude} \
                    --force-scale ${force_scale} \
                    --duration ${duration}"
                DEMO_NAME="Slant ${angle}° (custom)"
                ;;
            t|T)
                read -p "Tunnel height ratio (default: 0.9): " tunnel_ratio
                tunnel_ratio=${tunnel_ratio:-0.9}
                read -p "Tunnel length (default: 3.0): " tunnel_length
                tunnel_length=${tunnel_length:-3.0}
                CMD="python demo_tunnel.py \
                    --rows ${grid_rows} --cols ${grid_cols} \
                    --tunnel-ratio ${tunnel_ratio} \
                    --tunnel-length ${tunnel_length} \
                    --frequency ${frequency} \
                    --amplitude ${amplitude} \
                    --force-scale ${force_scale} \
                    --duration ${duration}"
                DEMO_NAME="Tunnel (custom)"
                ;;
            b|B)
                read -p "Boulder size ratio (default: 0.5): " boulder_ratio
                boulder_ratio=${boulder_ratio:-0.5}
                read -p "Boulder position X (default: 3.0): " boulder_pos
                boulder_pos=${boulder_pos:-3.0}
                CMD="python demo_boulder.py \
                    --rows ${grid_rows} --cols ${grid_cols} \
                    --boulder-ratio ${boulder_ratio} \
                    --boulder-position ${boulder_pos} \
                    --frequency ${frequency} \
                    --amplitude ${amplitude} \
                    --force-scale ${force_scale} \
                    --duration ${duration}"
                DEMO_NAME="Boulder (custom)"
                ;;
            a|A)
                read -p "Plane angle (degrees, default: 20): " angle
                angle=${angle:-20}
                CMD="python demo_angled_plane.py \
                    --rows ${grid_rows} --cols ${grid_cols} \
                    --angle ${angle} \
                    --frequency ${frequency} \
                    --amplitude ${amplitude} \
                    --force-scale ${force_scale} \
                    --duration ${duration}"
                DEMO_NAME="Angled Plane ${angle}° (custom)"
                ;;
            *)
                echo -e "${RED}Invalid demo type${NC}"
                exit 1
                ;;
        esac
        ;;
        
    *)
        echo -e "${RED}Invalid selection${NC}"
        exit 1
        ;;
esac

echo ""
echo "========================================================================"
echo -e "${GREEN}Configuration Summary:${NC}"
echo "========================================================================"
echo -e "  Demo: ${CYAN}${DEMO_NAME}${NC}"
echo ""
echo -e "${YELLOW}Force System:${NC}"
echo "  Mode: Balloon (radial inflate/deflate)"
echo "  Ratchet friction: enabled"
echo "  Locomotion: CPG traveling wave + ground friction"
echo ""
echo -e "${YELLOW}Debug Visualization:${NC}"
echo "  SDF collision: cyan circles + arrows on ground-touching particles"
echo ""
echo -e "${YELLOW}CPG Matrix (6x3 grid = 10 groups):${NC}"
echo ""
echo "    Groups numbered bottom-to-top, left-to-right:"
echo ""
echo "    +-----+-----+-----+-----+-----+"
echo "    |  5  |  6  |  7  |  8  |  9  |  <- top row"
echo "    +-----+-----+-----+-----+-----+"
echo "    |  0  |  1  |  2  |  3  |  4  |  <- bottom row (touches ground)"
echo "    +-----+-----+-----+-----+-----+"
echo ""
echo "    Phase wave travels: bottom -> top (for forward motion)"
echo ""
echo -e "${YELLOW}Balloon Forces:${NC}"
echo "  CPG output > 0: INFLATE (push outward from centroid)"
echo "  CPG output < 0: DEFLATE (pull inward to centroid)"
echo ""
echo "  Traveling wave + ratchet friction = locomotion"
echo "========================================================================"
echo ""
echo "Controls:"
echo "  Q/ESC - Quit"
echo "  R     - Reset"
echo "  SPACE - Pause/Resume"
echo ""
read -p "Press Enter to start..."
echo ""

# Allow X11 connections from Docker
echo "Setting up X11 display..."
xhost +local:docker 2>/dev/null || true

# Run in Docker
docker run -it --rm \
    --gpus all \
    --ipc=host \
    -e DISPLAY=$DISPLAY \
    -e PYTHONPATH=/workspace \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v "$PROJECT_DIR:/workspace" \
    -w /workspace/rl_control \
    "$IMAGE_NAME" \
    $CMD

echo ""
echo -e "${GREEN}Demo complete!${NC}"
