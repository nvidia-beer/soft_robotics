#!/bin/bash
# Figure Generation Script for PID Control Analysis
#
# This script generates publication-ready figures for the inflation control study.
# Each figure demonstrates different aspects of PID controller implementations.
#
# Output files:
#   figure_snn.png/pdf - SNN PID control (P, PI, PD, PID)
#   figure_classic.png/pdf - Classic PID control (P, PI, PD, PID)
#   figure_neuron_count.png/pdf - Effect of neuron count on SNN PID
#   figure_pes.png/pdf - PES learning rate comparison
#   figure_combined.png/pdf - All figures combined
#
# Usage:
#   ./create_figure.sh              # Interactive menu
#   ./create_figure.sh snn          # Generate SNN figure
#   ./create_figure.sh classic      # Generate Classic PID figure
#   ./create_figure.sh all          # Generate all figures
#   ./create_figure.sh snn --no-sim # Plot from saved data only

set -e

# Colors for output
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
FIGURES_DIR="$SCRIPT_DIR/figures"
IMAGE_NAME="spring-mass-nengo"

# Enable X11 forwarding (for any GUI components)
xhost +local:docker 2>/dev/null || true

# Create figures directory if needed
mkdir -p "$FIGURES_DIR"

echo ""
echo "========================================================================"
echo -e "${PURPLE}Figure Generation - Neuromorphic PID Control Analysis${NC}"
echo "========================================================================"
echo ""
echo -e "Output directory: ${CYAN}$FIGURES_DIR${NC}"
echo ""

# Build Docker if needed
if ! docker image inspect "$IMAGE_NAME" &> /dev/null; then
    echo -e "${YELLOW}Building Docker image...${NC}"
    "$PROJECT_DIR/build.sh"
fi

# Check for GPU support
GPU_FLAGS=""
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    GPU_FLAGS="--gpus all"
    DEVICE="cuda"
    echo -e "${GREEN}GPU detected - using CUDA${NC}"
else
    DEVICE="cpu"
    echo -e "${YELLOW}No GPU detected - using CPU${NC}"
fi
echo ""

# Available figures: name -> "script|description|output"
declare -A FIGURES
FIGURES[snn]="figure_snn.py|SNN PID control (P, PI, PD, PID)|figure_snn"
FIGURES[classic]="figure_classic.py|Classic PID control (P, PI, PD, PID)|figure_classic"
FIGURES[neuron_count]="figure_neuron_count.py|Effect of neuron count on SNN PID|figure_neuron_count"
FIGURES[pes]="figure_pes.py|PES learning rate comparison|figure_pes"
FIGURES[combined]="figure_combined.py|All figures combined (4 rows)|figure_combined"

# Order for display and 'all'
FIGURE_ORDER=(snn classic neuron_count pes combined)

print_available_figures() {
    echo -e "${CYAN}Available Figures:${NC}"
    echo ""
    for key in "${FIGURE_ORDER[@]}"; do
        IFS='|' read -r script description output <<< "${FIGURES[$key]}"
        echo -e "  ${GREEN}$key${NC} - $description"
        echo -e "      Output: ${BLUE}$output.png/pdf${NC}"
        echo ""
    done
}

run_figure() {
    local fig_name=$1
    shift  # Remove first argument (figure name)
    local extra_args="$@"
    
    if [[ -z "${FIGURES[$fig_name]}" ]]; then
        echo -e "${RED}Error: Figure '$fig_name' not found${NC}"
        echo ""
        print_available_figures
        return 1
    fi
    
    IFS='|' read -r script description output <<< "${FIGURES[$fig_name]}"
    
    echo ""
    echo "========================================================================"
    echo -e "${GREEN}Generating: $description${NC}"
    echo "========================================================================"
    echo ""
    echo -e "Script: ${BLUE}$script${NC}"
    echo -e "Output: ${CYAN}$output.png/pdf${NC}"
    echo -e "Device: ${CYAN}$DEVICE${NC}"
    if [[ -n "$extra_args" ]]; then
        echo -e "Extra args: ${YELLOW}$extra_args${NC}"
    fi
    echo ""
    
    # Run in Docker
    docker run -it --rm $GPU_FLAGS --ipc=host --shm-size=2g \
        -e DISPLAY=$DISPLAY -e PYTHONUNBUFFERED=1 \
        -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
        -v "$PROJECT_DIR:/workspace/soft_robotics" \
        -w /workspace/soft_robotics/inflation_control \
        "$IMAGE_NAME" \
        python "$script" --device "$DEVICE" $extra_args
    
    echo ""
    echo -e "${GREEN}Generation complete: $output${NC}"
    echo ""
}

# Parse command line arguments
if [[ $# -eq 0 ]]; then
    # Interactive menu
    print_available_figures
    
    echo ""
    echo "========================================================================"
    echo -e "${CYAN}Options:${NC}"
    echo "  Enter figure name (snn, classic, neuron_count, pes, combined)"
    echo "  Enter 'all' to generate all figures"
    echo "  Enter 'q' to quit"
    echo ""
    echo -e "${CYAN}Additional flags can be added after the figure name:${NC}"
    echo "  --no-sim     : Plot from previously saved data (skip simulation)"
    echo "  --total-time N : Set simulation time to N seconds"
    echo "  --device cpu   : Force CPU (default: auto-detect)"
    echo ""
    
    read -p "Enter selection: " selection
    
    if [[ "$selection" == "q" || "$selection" == "Q" ]]; then
        echo "Exiting."
        exit 0
    elif [[ "$selection" == "all" ]]; then
        for key in "${FIGURE_ORDER[@]}"; do
            run_figure "$key"
        done
    else
        # Parse selection and extra args
        read -r fig_name extra_args <<< "$selection"
        run_figure "$fig_name" $extra_args
    fi
else
    # Command line mode
    fig_arg="$1"
    shift
    extra_args="$@"
    
    if [[ "$fig_arg" == "all" ]]; then
        for key in "${FIGURE_ORDER[@]}"; do
            run_figure "$key" $extra_args
        done
    else
        run_figure "$fig_arg" $extra_args
    fi
fi

echo ""
echo "========================================================================"
echo -e "${GREEN}Figure generation complete!${NC}"
echo "========================================================================"
echo ""
echo -e "Output files are in: ${CYAN}$FIGURES_DIR${NC}"
echo ""

# Summary of generated figures
if [[ -d "$FIGURES_DIR" ]]; then
    echo "Summary:"
    png_count=$(ls -1 "$FIGURES_DIR"/*.png 2>/dev/null | wc -l)
    pdf_count=$(ls -1 "$FIGURES_DIR"/*.pdf 2>/dev/null | wc -l)
    npz_count=$(ls -1 "$FIGURES_DIR"/*.npz 2>/dev/null | wc -l)
    
    echo "  PNG files: $png_count"
    echo "  PDF files: $pdf_count"
    echo "  Data files (npz): $npz_count"
fi
echo ""
