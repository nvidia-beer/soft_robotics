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
echo -e "  ${CYAN}1. PLANE${NC}    - Flat ground locomotion (classic, move right)"
echo -e "  ${CYAN}2. SLANT${NC}    - Climb inclined plane (default: 45°)"
echo -e "  ${CYAN}3. TUNNEL${NC}   - Squeeze through passage (default: 90% height)"
echo -e "  ${CYAN}4. BOULDER${NC}  - Climb over obstacle (default: 50% size)"
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
echo "  2) Slant (45° incline, configurable)"
echo "  3) Tunnel (90% height, configurable)"
echo "  4) Boulder (50% size, configurable)"
echo "  5) Custom configuration"
echo ""
read -p "Select [1/2/3/4/5] (default: 1): " demo_choice
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
        read -p "Slant angle (degrees, default: 45): " angle
        angle=${angle:-45}
        
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
        
        CMD="python demo_boulder.py \
            --boulder-ratio ${boulder_ratio} \
            --boulder-position ${boulder_pos} \
            --frequency ${frequency} \
            --duration ${duration}"
        
        DEMO_NAME="Boulder ${boulder_ratio}"
        ;;
        
    5)
        echo ""
        echo -e "${CYAN}=== CUSTOM CONFIGURATION ===${NC}"
        echo ""
        echo "Select demo type:"
        echo "  p) Plane (flat ground)"
        echo "  s) Slant"
        echo "  t) Tunnel"
        echo "  b) Boulder"
        read -p "Type [p/s/t/b]: " demo_type
        
        # Common parameters
        read -p "Grid size N (default: 4): " grid_n
        grid_n=${grid_n:-4}
        
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
                    --grid-size ${grid_n} \
                    --direction ${dir_x} ${dir_y} \
                    --frequency ${frequency} \
                    --amplitude ${amplitude} \
                    --force-scale ${force_scale} \
                    --duration ${duration}"
                DEMO_NAME="Plane (custom)"
                ;;
            s|S)
                read -p "Slant angle (degrees, default: 45): " angle
                angle=${angle:-45}
                CMD="python demo_slant.py \
                    --grid-size ${grid_n} \
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
                    --grid-size ${grid_n} \
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
                    --grid-size ${grid_n} \
                    --boulder-ratio ${boulder_ratio} \
                    --boulder-position ${boulder_pos} \
                    --frequency ${frequency} \
                    --amplitude ${amplitude} \
                    --force-scale ${force_scale} \
                    --duration ${duration}"
                DEMO_NAME="Boulder (custom)"
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
echo -e "${GREEN}Running: ${DEMO_NAME}${NC}"
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
