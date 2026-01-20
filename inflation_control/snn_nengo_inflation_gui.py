#!/usr/bin/env python3
"""
Inflation Control SNN with Nengo GUI

Visualization of volume-based control using Nengo spiking neural networks.
Shows both the pygame physics window AND the Nengo web interface.

Features:
- Volume error → SNN controller → Pressure control
- Controller selection: SNN_PID or SNN_Stress (with PES feedforward learning)
- Real-time PID ensemble visualization (q, ei, ed, u)
- Strain ensemble visualization for PES learning (SNN_Stress)
- Controller toggle (ON/OFF) via Nengo slider
- FEM strain visualization as spike rasters
- PID gain tuning via Nengo sliders

Controllers:
- SNN_PID: NEF-based spiking PD control (Zaidel et al. 2021)
- SNN_Stress: NEF-based spiking PD + PES feedforward learning from strain

Usage:
    # Run with Nengo GUI
    nengo snn_nengo_inflation_gui.py
    
    # Or with environment variables
    INFLATION_MAX_VOLUME=2.0 INFLATION_CONTROLLER=1 nengo snn_nengo_inflation_gui.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'warp'))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import nengo
import warp as wp
import pygame

# Disable Warp CUDA graph capture to avoid stream conflicts with Nengo
wp.config.enable_graph_capture_on_kernels = False

from models import BalloonModel
from solvers import SolverImplicitFEM, SolverVBD
from pygame_renderer import Renderer
from controllers.nengo import SNN_PID_Controller
from controllers.stress import SNN_Stress_Controller

# ============================================================================
# Configuration
# ============================================================================

DT = float(os.environ.get('INFLATION_DT', '0.01'))
DEVICE = os.environ.get('INFLATION_DEVICE', 'cuda')

# Model parameters
RADIUS = float(os.environ.get('INFLATION_RADIUS', '0.5'))
NUM_BOUNDARY = int(os.environ.get('INFLATION_NUM_BOUNDARY', '16'))
NUM_RINGS = int(os.environ.get('INFLATION_NUM_RINGS', '2'))
MAX_VOLUME = float(os.environ.get('INFLATION_MAX_VOLUME', '2.0'))

# Material parameters
FEM_E = float(os.environ.get('INFLATION_FEM_E', '2000.0'))
FEM_NU = float(os.environ.get('INFLATION_FEM_NU', '0.45'))
SPRING_K = float(os.environ.get('INFLATION_SPRING_K', '1000.0'))

# Gravity (0 = no gravity, -9.8 = Earth gravity)
GRAVITY_Y = float(os.environ.get('INFLATION_GRAVITY', '0.0'))

# Controller selection: 0=SNN_PID, 1=SNN_Stress
CONTROLLER_TYPE = int(os.environ.get('INFLATION_CONTROLLER', '0'))
CONTROLLER_NAMES = ['SNN_PID', 'SNN_Stress']

# SNN PID parameters
DEFAULT_KP = float(os.environ.get('INFLATION_PID_KP', '1.0'))
DEFAULT_KI = float(os.environ.get('INFLATION_PID_KI', '0.5'))
DEFAULT_KD = float(os.environ.get('INFLATION_PID_KD', '0.3'))

# PES learning rate (for SNN_Stress only)
PES_LEARNING_RATE = float(os.environ.get('INFLATION_PES_LR', '1e-4'))

# SNN settings
N_NEURONS = int(os.environ.get('INFLATION_NEURONS', '100'))
NENGO_DT = float(os.environ.get('INFLATION_NENGO_DT', '0.001'))

# Window settings
WINDOW_WIDTH = int(os.environ.get('INFLATION_WINDOW_WIDTH', '1000'))
WINDOW_HEIGHT = int(os.environ.get('INFLATION_WINDOW_HEIGHT', '600'))
SIM_WIDTH = 600  # Right panel for simulation
PLOT_WIDTH = WINDOW_WIDTH - SIM_WIDTH  # Left panel for plots

print(f"Configuration:")
print(f"  Max volume ratio: {MAX_VOLUME}x")
print(f"  Gravity: {GRAVITY_Y}")
print(f"  Controller: {CONTROLLER_NAMES[CONTROLLER_TYPE]}")
print(f"  Device: {DEVICE}")
print(f"  Physics dt: {DT}s ({1/DT:.0f} Hz)")
print(f"  Nengo dt: {NENGO_DT}s ({1/NENGO_DT:.0f} Hz)")
if CONTROLLER_TYPE == 1:
    print(f"  PES learning rate: {PES_LEARNING_RATE}")
print()

# ============================================================================
# Initialize Warp
# ============================================================================

wp.init()

# ============================================================================
# Create Balloon Model
# ============================================================================

print("Creating Balloon Model...")
balloon_model = BalloonModel(
    radius=RADIUS,
    num_boundary=NUM_BOUNDARY,
    num_rings=NUM_RINGS,
    max_volume_ratio=MAX_VOLUME,
    device=DEVICE,
    boxsize=3.0,
    spring_stiffness=SPRING_K,
    spring_damping=5.0,
    fem_E=FEM_E,
    fem_nu=FEM_NU,
    fem_damping=10.0,
)

# Set gravity (default: no gravity for balloon inflation)
balloon_model.set_gravity((0.0, GRAVITY_Y))
if GRAVITY_Y != 0:
    print(f"  Gravity enabled: (0, {GRAVITY_Y})")

# Create solver
print("Creating Implicit FEM solver...")
solver = SolverImplicitFEM(
    balloon_model,
    dt=DT,
    mass=1.0,
    preconditioner_type="diag",
    solver_type="bicgstab",
    max_iterations=50,
    tolerance=1e-4,
    rebuild_matrix_every=1,
)

# Create states
state = balloon_model.state()
state_next = balloon_model.state()

# Get model info - use boundary springs only for visualization
n_boundary_springs = balloon_model.boundary_spring_count
boundary_spring_indices = balloon_model.boundary_spring_indices
n_total_strains = n_boundary_springs  # Only boundary springs for Nengo viz

print(f"Model created:")
print(f"  Particles: {balloon_model.particle_count}")
print(f"  Total springs: {balloon_model.spring_count}")
print(f"  Boundary springs (circumference): {n_boundary_springs}")
print(f"  FEM triangles: {balloon_model.tri_count}")
print(f"  Initial volume: {balloon_model.initial_volume:.4f}")
print(f"  Max volume: {balloon_model.max_volume:.4f}")

# ============================================================================
# Create Renderer
# ============================================================================

print("\nCreating Renderer...")
pygame.init()
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Inflation Control SNN - Nengo GUI")
clock = pygame.time.Clock()

renderer = Renderer(
    window_width=SIM_WIDTH,
    window_height=WINDOW_HEIGHT,
    boxsize=balloon_model.boxsize,
)

# Prepare boundary spring indices for thick rendering (these feed into Nengo)
boundary_spring_indices_flat = []
for idx in boundary_spring_indices:
    i = balloon_model.spring_indices.numpy()[idx * 2]
    j = balloon_model.spring_indices.numpy()[idx * 2 + 1]
    boundary_spring_indices_flat.extend([i, j])
boundary_spring_indices_np = np.array(boundary_spring_indices_flat, dtype=np.int32)

spring_indices_np = balloon_model.spring_indices.numpy() if balloon_model.spring_count > 0 else None
tri_indices_np = balloon_model.tri_indices.numpy() if balloon_model.tri_count > 0 else None

# Fonts
font_small = pygame.font.Font(None, 18)
font_normal = pygame.font.Font(None, 24)
font_large = pygame.font.Font(None, 28)

# ============================================================================
# Global State
# ============================================================================

current_strains = np.zeros(n_total_strains, dtype=np.float32)
current_pressure = 0.0
current_volume = balloon_model.initial_volume
target_volume = balloon_model.initial_volume
target_ratio = 1.0
simulation_time = 0.0
controller_enabled = True
controller_status_text = f"Controller: {CONTROLLER_NAMES[CONTROLLER_TYPE]} [ON]"

# SNN controller state
current_error = 0.0  # Volume error for SNN
snn_output = 0.0  # SNN computed pressure (from active controller)

# Track physics steps
last_physics_time = -DT

# Strain input for PES learning (normalized boundary strains)

# ============================================================================
# Physics Step Function
# ============================================================================

def physics_step(t):
    """
    Execute one physics step with SNN controller.
    Returns strain data for Nengo visualization.
    """
    global current_strains, current_pressure, current_volume, target_volume
    global simulation_time, controller_enabled, controller_status_text
    global current_error, snn_output
    global last_physics_time, state, state_next, target_ratio
    
    simulation_time = t
    
    # Update error signal (needed for SNN computation)
    current_volume = balloon_model.compute_current_volume(state)
    target_volume = balloon_model.initial_volume * target_ratio
    current_error = target_volume - current_volume
    
    # Only step physics at physics dt intervals
    if t - last_physics_time < DT - NENGO_DT * 0.5:
        return current_strains
    
    last_physics_time = t
    
    # Controller status
    status = "ON" if controller_enabled else "OFF"
    controller_name = CONTROLLER_NAMES[CONTROLLER_TYPE]
    controller_status_text = f"Controller: {controller_name} [{status}]"
    
    # Compute control using SNN controller
    if controller_enabled:
        current_pressure = snn_output
        # Debug every second
        if int(t) != int(t - DT) and t > 0.1:
            print(f"[t={t:.1f}] {controller_name}: pressure={current_pressure:.2f}")
    else:
        current_pressure = 0.0
    
    # Apply inflation
    _, rest_config = balloon_model.apply_inflation(state, current_pressure, target_ratio)
    
    # Update tracking
    balloon_model.current_volume_ratio = current_volume / balloon_model.initial_volume
    balloon_model.target_volume_ratio = target_ratio
    
    # Physics step
    solver.step(state, state_next, DT, external_forces=None)
    state, state_next = state_next, state
    
    # Extract boundary spring strains (circumference only)
    # Normalize to [-1, 1] using min/max of current values
    if n_boundary_springs > 0 and balloon_model.spring_strains is not None:
        all_spring_strains = balloon_model.spring_strains.numpy()
        boundary_strains = all_spring_strains[boundary_spring_indices]
        
        # Min-max normalization to [-1, 1]
        strain_min = boundary_strains.min()
        strain_max = boundary_strains.max()
        strain_range = strain_max - strain_min
        
        if strain_range > 1e-8:
            # Map [min, max] to [-1, 1]
            normalized = 2.0 * (boundary_strains - strain_min) / strain_range - 1.0
        else:
            # All strains equal - set to 0
            normalized = np.zeros_like(boundary_strains)
        
        current_strains = np.array(normalized, dtype=np.float32)
    else:
        current_strains = np.zeros(n_total_strains, dtype=np.float32)
    
    return current_strains

# ============================================================================
# Render Function
# ============================================================================

def render_frame():
    """Render the current state."""
    global screen
    
    # Sync GPU
    wp.synchronize()
    
    # Get positions
    positions = state.particle_q.numpy()
    spring_strains = balloon_model.spring_strains.numpy() if balloon_model.spring_strains is not None else None
    tri_strains = balloon_model.tri_strains.numpy() if balloon_model.tri_strains is not None else None
    
    # Left panel: Info text (simple for now, could add matplotlib plots)
    info_panel = pygame.Surface((PLOT_WIDTH, WINDOW_HEIGHT))
    info_panel.fill((240, 240, 240))
    
    # Draw info on left panel
    y_offset = 20
    line_height = 22
    
    # Title
    title = font_large.render("Inflation Control SNN", True, (0, 0, 0))
    info_panel.blit(title, (10, y_offset))
    y_offset += line_height + 10
    
    # Status
    status_color = (0, 150, 0) if controller_enabled else (200, 50, 50)
    status_text = font_normal.render(controller_status_text, True, status_color)
    info_panel.blit(status_text, (10, y_offset))
    y_offset += line_height
    
    # Volume info
    vol_ratio = current_volume / balloon_model.initial_volume
    vol_text = font_normal.render(f"Volume: {vol_ratio:.3f} / {target_ratio:.3f}", True, (0, 0, 0))
    info_panel.blit(vol_text, (10, y_offset))
    y_offset += line_height
    
    # Error
    err_text = font_normal.render(f"Error: {current_error:.4f}", True, (200, 50, 50) if abs(current_error) > 0.01 else (0, 0, 0))
    info_panel.blit(err_text, (10, y_offset))
    y_offset += line_height
    
    # Pressure
    press_text = font_normal.render(f"Pressure: {current_pressure:.3f}", True, (255, 140, 0))
    info_panel.blit(press_text, (10, y_offset))
    y_offset += line_height + 10
    
    # PES learning rate (if SNN_Stress)
    if CONTROLLER_TYPE == 1:
        pes_text = font_small.render(f"PES learning rate: {PES_LEARNING_RATE}", True, (0, 100, 200))
        info_panel.blit(pes_text, (10, y_offset))
        y_offset += line_height
    
    y_offset += 10
    
    # Instructions
    instructions = [
        "Nengo GUI Controls:",
        "  - Controller ON/OFF slider",
        "  - Target Volume slider",
        "",
        "Open browser: http://localhost:8080",
    ]
    for line in instructions:
        inst_text = font_small.render(line, True, (100, 100, 200))
        info_panel.blit(inst_text, (10, y_offset))
        y_offset += 16
    
    # Volume progress bar on left panel (under info)
    y_offset += 20
    vol_ratio = current_volume / balloon_model.initial_volume
    current_normalized = vol_ratio - 1.0
    max_normalized = MAX_VOLUME - 1.0
    target_normalized = target_ratio - 1.0
    
    renderer.draw_progress_bar(
        info_panel,
        value=current_normalized,
        max_value=max_normalized,
        x=10, y=y_offset,
        width=PLOT_WIDTH - 20, height=20,
        target_value=target_normalized,
        label=f"Volume: {vol_ratio:.2f}x / {target_ratio:.2f}x"
    )
    
    screen.blit(info_panel, (0, 0))
    
    # Right panel: simulation
    canvas = renderer.create_canvas()
    renderer.draw_grid(canvas)
    
    if tri_indices_np is not None and balloon_model.tri_count > 0:
        renderer.draw_fem_triangles(canvas, tri_indices_np, positions, tri_strains)
    
    if spring_indices_np is not None and balloon_model.spring_count > 0:
        renderer.draw_springs(canvas, spring_indices_np, positions, spring_strains)
    
    # Draw boundary springs thicker (these are Nengo inputs)
    if n_boundary_springs > 0 and spring_strains is not None:
        boundary_strains = spring_strains[boundary_spring_indices]
        # Draw with 2x thickness by temporarily changing renderer settings
        old_min = renderer.spring_min_width
        old_max = renderer.spring_max_width
        renderer.spring_min_width = 4  # 2x default
        renderer.spring_max_width = 10  # 2x default
        renderer.draw_springs(canvas, boundary_spring_indices_np, positions, boundary_strains)
        renderer.spring_min_width = old_min
        renderer.spring_max_width = old_max
    
    # Draw particles
    if balloon_model.particle_colliding is not None:
        particle_colors = renderer.get_collision_colors(balloon_model.particle_colliding.numpy())
        renderer.draw_particles(canvas, positions, per_particle_colors=particle_colors)
    else:
        renderer.draw_particles(canvas, positions)
    
    renderer.draw_strain_legends(canvas, spring_strains=spring_strains, fem_strains=tri_strains, show_fem=True)
    
    screen.blit(canvas, (PLOT_WIDTH, 0))
    
    # Separator line
    pygame.draw.line(screen, (100, 100, 100), (PLOT_WIDTH, 0), (PLOT_WIDTH, WINDOW_HEIGHT), 2)
    
    pygame.display.flip()
    
    # Process pygame events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            print("Window closed by user")
            pygame.quit()
            raise SystemExit

# ============================================================================
# Build Nengo Network
# ============================================================================

print("\nBuilding Nengo Network...")
model = nengo.Network(label="Inflation SNN")

with model:
    
    # =========================================================================
    # Controller Toggle and Type Selection
    # =========================================================================
    
    controller_toggle = nengo.Node([1.0], label="Controller [0=OFF, 1=ON]")
    
    # =========================================================================
    # Target Volume Slider
    # =========================================================================
    
    target_volume_slider = nengo.Node([1.0], label="Target Volume [1.0-2.0]")
    
    # =========================================================================
    # Main Update Node
    # =========================================================================
    
    def update_simulation_with_control(t, x):
        """
        Main update function called by Nengo.
        x[0] = controller enabled
        x[1] = target volume ratio
        """
        global controller_enabled, target_ratio
        
        controller_enabled = x[0] > 0.5
        target_ratio = np.clip(x[1], 1.0, MAX_VOLUME)
        
        # Physics step
        strains = physics_step(t)
        
        # Render
        render_frame()
        
        return strains
    
    strain_input = nengo.Node(
        update_simulation_with_control,
        size_in=2,
        size_out=n_total_strains,
        label="Strain_Input"
    )
    
    # Connect sliders to update node
    nengo.Connection(controller_toggle, strain_input[0], synapse=None)
    nengo.Connection(target_volume_slider, strain_input[1], synapse=None)
    
    # =========================================================================
    # Boundary Strain Ensembles (Circumference Springs)
    # =========================================================================
    
    # Create ensemble for boundary spring strains (circumference of balloon)
    # These springs directly measure the expansion/contraction of the outer ring
    N_STRAINS_VIZ = min(n_boundary_springs, n_total_strains)
    
    strain_ensemble = nengo.Ensemble(
        n_neurons=N_NEURONS * N_STRAINS_VIZ,
        dimensions=N_STRAINS_VIZ,
        max_rates=nengo.dists.Uniform(100, 200),
        intercepts=nengo.dists.Uniform(-0.5, 0.5),
        neuron_type=nengo.LIF(),
        radius=1.0,
        label="Boundary_Strain"
    )
    
    # Connect all boundary strains (they represent circumference expansion)
    for i in range(N_STRAINS_VIZ):
        strain_idx = int(i * n_total_strains / N_STRAINS_VIZ) if N_STRAINS_VIZ < n_total_strains else i
        transform = np.zeros((N_STRAINS_VIZ, 1))
        transform[i, 0] = 1.0
        nengo.Connection(strain_input[strain_idx], strain_ensemble,
                        transform=transform, synapse=0.01)
    
    # =========================================================================
    # SNN Controller Configuration
    # =========================================================================
    
    # Common configuration
    error_scale = 0.5  # Max expected volume error (for internal normalization)
    output_scale = 10.0  # Max pressure output (u_max)
    
    # PID gains from environment/defaults
    SNN_KP = DEFAULT_KP   # 1.0
    SNN_KI = DEFAULT_KI   # 0.5
    SNN_KD = DEFAULT_KD   # 0.3
    
    # Error callback (shared by both controllers)
    def get_error(t):
        return current_error / error_scale
    
    # =========================================================================
    # Build ONLY the selected controller (to avoid duplicate ensembles)
    # =========================================================================
    
    snn_controller = None
    snn_ensembles = []
    
    if CONTROLLER_TYPE == 0:
        # =====================================================================
        # NEF-Based SNN PID Controller (Zaidel et al. 2021)
        # =====================================================================
        print(f"  Building SNN_PID controller...")
        snn_controller = SNN_PID_Controller(
            Kp=SNN_KP,
            Ki=SNN_KI,
            Kd=SNN_KD,
            u_max=output_scale,
            n_neurons=N_NEURONS,
            error_scale=error_scale,
        )
        
        # Build SNN PID
        snn_result = snn_controller.build(
            model=model,
            get_error_callback=get_error,
        )
        snn_ensembles = snn_result['ensembles']
        
    else:
        # =====================================================================
        # NEF-Based SNN Stress Controller (PD + PES Feedforward)
        # 
        # IMPORTANT: Pass the existing strain_ensemble that's connected to
        # the simulator's strain data, so PES learns from actual strains!
        # =====================================================================
        print(f"  Building SNN_Stress controller (PES lr={PES_LEARNING_RATE})...")
        snn_controller = SNN_Stress_Controller(
            Kp=SNN_KP,
            Kd=SNN_KD,  # No Ki for stress control (pure PD)
            u_max=output_scale,
            n_neurons=N_NEURONS,
            error_scale=error_scale,
            strain_dim=N_STRAINS_VIZ,  # Match the existing strain_ensemble dimensions
            pes_learning_rate=PES_LEARNING_RATE,
        )
        
        # Build SNN Stress - pass existing strain_ensemble from simulator!
        # This is how PES learns: strain data → strain_ensemble → PES → u(t)
        snn_result = snn_controller.build(
            model=model,
            get_error_callback=get_error,
            strain_ensemble=strain_ensemble,  # USE the existing ensemble connected to simulator!
        )
        snn_ensembles = snn_result['ensembles']
    
    # =========================================================================
    # Controller Output Sync Node
    # =========================================================================
    
    def sync_snn_output(t):
        """Sync output from active controller."""
        global snn_output
        snn_output = snn_controller.get_output()
    
    nengo.Node(sync_snn_output, size_in=0, size_out=0, label="SNN_Output_Sync")
    
    # =========================================================================
    # Output Display
    # =========================================================================
    
    # Volume display node
    volume_display = nengo.Node(
        lambda t: [current_volume / balloon_model.initial_volume, target_ratio],
        size_out=2,
        label="Volume_Display"
    )
    
    # Pressure display node
    pressure_display = nengo.Node(
        lambda t: [current_pressure],
        size_out=1,
        label="Pressure_Display"
    )
    
    # Error display node
    error_display = nengo.Node(
        lambda t: [current_error / error_scale],
        size_out=1,
        label="Error_Display"
    )
    
    # Active controller display (constant - selected at startup)
    controller_display = nengo.Node(
        lambda t: [float(CONTROLLER_TYPE)],
        size_out=1,
        label="Active_Controller"
    )

print(f"\n{'='*70}")
print("Inflation Control SNN Network Created")
print(f"{'='*70}")
print(f"Controller: {CONTROLLER_NAMES[CONTROLLER_TYPE]}")
print(f"Boundary strain ensemble: {N_STRAINS_VIZ}D (circumference springs)")
print(f"SNN ensembles: {len(snn_ensembles)}")
if CONTROLLER_TYPE == 1:
    print(f"PES learns: strain[{N_STRAINS_VIZ}D] → pressure[1D]")
    print(f"PES learning rate: {PES_LEARNING_RATE}")
print(f"")
print(f"NENGO GUI CONTROLS:")
print(f"  - Controller [0=OFF, 1=ON]: Toggle controller")
print(f"  - Target Volume [1.0-{MAX_VOLUME}]: Target volume ratio")
print(f"")
print(f"  Gains (set at build time): Kp={DEFAULT_KP}, Ki={DEFAULT_KI}, Kd={DEFAULT_KD}")
print(f"{'='*70}")

# ============================================================================
# Create Nengo GUI Configuration
# ============================================================================

config_filename = os.path.join(os.path.dirname(__file__), 'snn_nengo_inflation_gui.py.cfg')
print(f"\nCreating Nengo GUI config: {config_filename}")

# Find ensemble indices
strain_ens_idx = None
for idx, ens in enumerate(model.all_ensembles):
    if ens is strain_ensemble:
        strain_ens_idx = idx
        break

snn_ens_indices = []
snn_ens_names = []
for snn_ens in snn_ensembles:
    for idx, ens in enumerate(model.all_ensembles):
        if ens is snn_ens:
            snn_ens_indices.append(idx)
            snn_ens_names.append(snn_ens.label)
            break

# Find node indices
toggle_node_idx = None
target_vol_idx = None
volume_display_idx = None
pressure_display_idx = None
error_display_idx = None
controller_display_idx = None

for idx, node in enumerate(model.all_nodes):
    if node is controller_toggle:
        toggle_node_idx = idx
    if node is target_volume_slider:
        target_vol_idx = idx
    if node is volume_display:
        volume_display_idx = idx
    if node is pressure_display:
        pressure_display_idx = idx
    if node is error_display:
        error_display_idx = idx
    if node is controller_display:
        controller_display_idx = idx

print(f"Found boundary strain ensemble at index: {strain_ens_idx}")
print(f"Found {len(snn_ens_indices)} SNN ensembles: {snn_ens_names}")

# Generate config file
with open(config_filename, 'w') as f:
    f.write("# Nengo GUI Configuration - Inflation Control SNN\n")
    f.write("_viz_net_graph = nengo_gui.components.NetGraph()\n\n")
    
    # Layout
    plot_x = 0.0
    plot_width = 0.12
    plot_height = 0.15
    
    # Boundary strain value and raster (circumference springs)
    if strain_ens_idx is not None:
        f.write(f"_viz_strain_value = nengo_gui.components.Value(model.all_ensembles[{strain_ens_idx}])\n")
        f.write(f"_viz_config[_viz_strain_value].x = {plot_x:.4f}\n")
        f.write(f"_viz_config[_viz_strain_value].y = 0.02\n")
        f.write(f"_viz_config[_viz_strain_value].width = {plot_width:.4f}\n")
        f.write(f"_viz_config[_viz_strain_value].height = {plot_height:.4f}\n\n")
        
        f.write(f"_viz_strain_raster = nengo_gui.components.Raster(model.all_ensembles[{strain_ens_idx}].neurons)\n")
        f.write(f"_viz_config[_viz_strain_raster].x = {plot_x:.4f}\n")
        f.write(f"_viz_config[_viz_strain_raster].y = {0.02 + plot_height + 0.01:.4f}\n")
        f.write(f"_viz_config[_viz_strain_raster].width = {plot_width:.4f}\n")
        f.write(f"_viz_config[_viz_strain_raster].height = {plot_height:.4f}\n\n")
    
    # SNN controller ensembles (q, ei, ed, u for PID; q, ed, u, s for Stress)
    if CONTROLLER_TYPE == 0:
        snn_labels = ["q(t)", "ei(t)", "ed(t)", "u(t)"]
    else:
        snn_labels = ["q(t)", "ed(t)", "u(t)", "s(t)"]
    
    snn_x = plot_x + plot_width + 0.02
    for i, ens_idx in enumerate(snn_ens_indices[:5]):
        label = snn_labels[i] if i < len(snn_labels) else f"ens{i}"
        safe_label = label.replace("(", "").replace(")", "").replace("[", "").replace("]", "")
        
        f.write(f"_viz_snn_{safe_label}_value = nengo_gui.components.Value(model.all_ensembles[{ens_idx}])\n")
        f.write(f"_viz_config[_viz_snn_{safe_label}_value].x = {snn_x:.4f}\n")
        f.write(f"_viz_config[_viz_snn_{safe_label}_value].y = {0.02 + i * (plot_height + 0.01):.4f}\n")
        f.write(f"_viz_config[_viz_snn_{safe_label}_value].width = {plot_width:.4f}\n")
        f.write(f"_viz_config[_viz_snn_{safe_label}_value].height = {plot_height:.4f}\n")
        f.write(f"_viz_config[_viz_snn_{safe_label}_value].max_value = 1.0\n")
        f.write(f"_viz_config[_viz_snn_{safe_label}_value].min_value = -1.0\n\n")
    
    # Slider positions
    slider_x = snn_x + plot_width + 0.02
    slider_width = 0.12
    slider_height = 0.06
    
    # Controller toggle slider
    if toggle_node_idx is not None:
        f.write(f"_viz_slider_toggle = nengo_gui.components.Slider(model.all_nodes[{toggle_node_idx}])\n")
        f.write(f"_viz_config[_viz_slider_toggle].x = {slider_x:.4f}\n")
        f.write(f"_viz_config[_viz_slider_toggle].y = 0.02\n")
        f.write(f"_viz_config[_viz_slider_toggle].width = {slider_width:.4f}\n")
        f.write(f"_viz_config[_viz_slider_toggle].height = {slider_height:.4f}\n")
        f.write(f"_viz_config[_viz_slider_toggle].min_value = 0\n")
        f.write(f"_viz_config[_viz_slider_toggle].max_value = 1\n\n")
    
    # Target volume slider (placed after toggle)
    target_slider_y = 0.02 + (slider_height + 0.01) + 0.02
    if target_vol_idx is not None:
        f.write(f"_viz_slider_target = nengo_gui.components.Slider(model.all_nodes[{target_vol_idx}])\n")
        f.write(f"_viz_config[_viz_slider_target].x = {slider_x:.4f}\n")
        f.write(f"_viz_config[_viz_slider_target].y = {target_slider_y:.4f}\n")
        f.write(f"_viz_config[_viz_slider_target].width = {slider_width:.4f}\n")
        f.write(f"_viz_config[_viz_slider_target].height = {slider_height:.4f}\n")
        f.write(f"_viz_config[_viz_slider_target].min_value = 1.0\n")
        f.write(f"_viz_config[_viz_slider_target].max_value = {MAX_VOLUME}\n\n")
    
    # Display nodes (volume, pressure, error)
    display_x = slider_x + slider_width + 0.02
    
    if volume_display_idx is not None:
        f.write(f"_viz_volume = nengo_gui.components.Value(model.all_nodes[{volume_display_idx}])\n")
        f.write(f"_viz_config[_viz_volume].x = {display_x:.4f}\n")
        f.write(f"_viz_config[_viz_volume].y = 0.02\n")
        f.write(f"_viz_config[_viz_volume].width = {plot_width:.4f}\n")
        f.write(f"_viz_config[_viz_volume].height = {plot_height:.4f}\n")
        f.write(f"_viz_config[_viz_volume].max_value = {MAX_VOLUME}\n")
        f.write(f"_viz_config[_viz_volume].min_value = 0.5\n\n")
    
    if pressure_display_idx is not None:
        f.write(f"_viz_pressure = nengo_gui.components.Value(model.all_nodes[{pressure_display_idx}])\n")
        f.write(f"_viz_config[_viz_pressure].x = {display_x:.4f}\n")
        f.write(f"_viz_config[_viz_pressure].y = {0.02 + plot_height + 0.01:.4f}\n")
        f.write(f"_viz_config[_viz_pressure].width = {plot_width:.4f}\n")
        f.write(f"_viz_config[_viz_pressure].height = {plot_height:.4f}\n")
        f.write(f"_viz_config[_viz_pressure].max_value = 10\n")
        f.write(f"_viz_config[_viz_pressure].min_value = -10\n\n")
    
    if error_display_idx is not None:
        f.write(f"_viz_error = nengo_gui.components.Value(model.all_nodes[{error_display_idx}])\n")
        f.write(f"_viz_config[_viz_error].x = {display_x:.4f}\n")
        f.write(f"_viz_config[_viz_error].y = {0.02 + 2 * (plot_height + 0.01):.4f}\n")
        f.write(f"_viz_config[_viz_error].width = {plot_width:.4f}\n")
        f.write(f"_viz_config[_viz_error].height = {plot_height:.4f}\n")
        f.write(f"_viz_config[_viz_error].max_value = 1\n")
        f.write(f"_viz_config[_viz_error].min_value = -1\n\n")
    
    # Network graph
    network_x = display_x + plot_width + 0.02
    f.write(f"_viz_config[model].pos = ({network_x + 0.15:.3f}, 0.50)\n")
    f.write(f"_viz_config[model].size = (0.30, 0.96)\n")
    f.write(f"_viz_config[model].expanded = False\n\n")
    
    f.write("_viz_sim_control = nengo_gui.components.SimControl()\n")
    f.write("_viz_config[_viz_sim_control].shown_time = 0.5\n")
    f.write("_viz_config[_viz_sim_control].kept_time = 4.0\n\n")

print(f"✅ Config created!")
print(f"\n🌐 Starting Nengo GUI server...")
print()
