#!/usr/bin/env python3
"""
Spring-Mass Trajectory Tracking SNN with Nengo GUI

Visualization of strain/tension signals from the trajectory tracking environment
with Nengo PID and Nengo MPC controller integration.

Features:
- Per-group strain ensembles with spike visualization
- Controller toggle (ON/OFF) via Nengo slider
- Controller selection (Nengo PID / Nengo MPC) via Nengo slider
- No CPG - controller provides the control signal
- Strain visualization is NOT connected to controller (sensory only)
- No gravity in simulation

Usage:
    # Run with Nengo GUI
    nengo snn_nengo_tracking_gui.py
    
    # Or with environment variables for grid/trajectory config
    TRACKING_N=4 TRACKING_TRAJECTORY=figure8 nengo snn_nengo_tracking_gui.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'warp'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'openai-gym'))

import numpy as np
import nengo
import warp as wp
import pygame

# Disable Warp CUDA graph capture to avoid stream conflicts with Nengo
wp.config.enable_graph_capture_on_kernels = False

from tracking_env import TrackingEnv
from snn_nengo_tracking_interface import SNNTrackingInterface
from controllers.nengo.pid import SNN_PID_Controller
from controllers.nengo.stress import SNN_Stress_Controller

# ============================================================================
# Configuration
# ============================================================================

N = int(os.environ.get('TRACKING_N', '3'))  # Grid size (3x3 default)
N = max(N, 2)  # Minimum 2x2 grid required for at least 1 group
DT = float(os.environ.get('TRACKING_DT', '0.01'))
DEVICE = os.environ.get('TRACKING_DEVICE', 'cuda')

# Trajectory settings
TRAJECTORY_TYPE = os.environ.get('TRACKING_TRAJECTORY', 'circular')
TRAJECTORY_AMPLITUDE = float(os.environ.get('TRACKING_AMPLITUDE', '0.3'))
TRAJECTORY_FREQUENCY = float(os.environ.get('TRACKING_FREQUENCY', '0.2'))

# SNN settings
N_NEURONS = int(os.environ.get('TRACKING_NEURONS', '50'))
# Nengo dt should be much smaller than physics dt for proper spike dynamics
# Typical: 0.001s (1ms) for smooth decoded values from spikes
NENGO_DT = float(os.environ.get('TRACKING_NENGO_DT', '0.001'))

# Window settings
WINDOW_WIDTH = int(os.environ.get('TRACKING_WINDOW_WIDTH', '1500'))
WINDOW_HEIGHT = int(os.environ.get('TRACKING_WINDOW_HEIGHT', '800'))

CONTROLLER_NAMES = {0: 'PID', 1: 'MPC', 2: 'SNN_PID', 3: 'SNN_Stress'}
INITIAL_CONTROLLER = int(os.environ.get('TRACKING_CONTROLLER', '0'))
print(f"Configuration:")
print(f"  Grid size: {N}x{N}")
print(f"  Initial controller: {CONTROLLER_NAMES.get(INITIAL_CONTROLLER, 'PID')} (toggle via GUI slider 0/1/2)")
print(f"  Trajectory: {TRAJECTORY_TYPE}")
print(f"  Device: {DEVICE}")
print(f"  Physics dt: {DT}s ({1/DT:.0f} Hz)")
print(f"  Nengo dt: {NENGO_DT}s ({1/NENGO_DT:.0f} Hz) - finer for smooth spike decoding")
print()

# ============================================================================
# Create Tracking Environment (NO GRAVITY)
# ============================================================================

print("Creating Tracking Environment (no gravity)...")
env = TrackingEnv(
    render_mode='human',
    N=N,
    dt=DT,
    spring_stiffness=40.0,
    spring_damping=0.5,
    device=DEVICE,
    trajectory_type=TRAJECTORY_TYPE,
    trajectory_amplitude=TRAJECTORY_AMPLITUDE,
    trajectory_frequency=TRAJECTORY_FREQUENCY,
    use_fem=True,
    boxsize=2.5,
    window_width=WINDOW_WIDTH,
    window_height=WINDOW_HEIGHT,
)

# Verify no gravity (already set in TrackingEnv but double-check)
if hasattr(env, 'gravity'):
    env.gravity = 0.0

# Set max_time for plot scaling (important for tracking plots to display correctly)
env.max_time = 60.0  # 60 seconds of simulation time for plot scaling

env.reset(seed=42)

# Get model info
n_spring_strains = env.model.spring_count
n_fem_strains = env.model.tri_count if hasattr(env.model, 'tri_count') else 0
n_total_strains = n_spring_strains + n_fem_strains
num_groups = env.num_groups
num_groups_per_side = N - 1

print(f"Environment created:")
print(f"  Particles: {N}x{N} = {N*N}")
print(f"  Groups: {num_groups} ({num_groups_per_side}x{num_groups_per_side})")
print(f"  Springs: {n_spring_strains}")
print(f"  FEM triangles: {n_fem_strains}")
print(f"  Total strains: {n_total_strains}")

# ============================================================================
# Create SNN Tracking Interface
# ============================================================================

print("\nCreating SNN Tracking Interface...")
interface = SNNTrackingInterface(
    env=env,
    n_neurons=N_NEURONS,
    dt=NENGO_DT,
    verbose=True,
)

# ============================================================================
# Create Controllers (switchable via Nengo GUI)
# ============================================================================
#
# Available controllers:
#   - PID: Classic proportional-integral-derivative
#   - MPC: Model predictive control
#   - SNN_PID: NEF-based spiking PD (Zaidel et al. 2021)
#   - SNN_Stress: NEF-based spiking PD + strain feedback
#
# The SNN controllers are built using SNN_PID_Controller / SNN_Stress_Controller
# from controllers/nengo/. They create ensembles within this Nengo model.
# ============================================================================

import traceback

controller_pid = None
controller_mpc = None
controller_enabled = True  # Toggle state (ON/OFF)
# Initial controller type set from environment (0=PID, 1=MPC, 2=SNN_PID, 3=SNN_Stress)
controller_type = max(0, min(3, INITIAL_CONTROLLER))  # Clamp to valid range

print("\nCreating controllers...")
print(f"  num_groups = {num_groups}")
print(f"  (Using classic PID/MPC - compatible with Nengo GUI)")

# Create classic PID
try:
    from controllers.pid import PID
    controller_pid = PID(
        num_groups=num_groups,
        dt=DT,
        u_max=500.0,
        Kp=200.0,
        Ki=10.0,
        Kd=50.0,
    )
    print(f"  ✓ PID controller created")
except Exception as e:
    print(f"  ⚠ Could not create PID controller: {e}")
    traceback.print_exc()
    controller_pid = None

# Create classic MPC
try:
    from controllers.mpc import MPC
    controller_mpc = MPC(
        num_groups=num_groups,
        dt=DT,
        u_max=500.0,
        horizon=10,
    )
    print(f"  ✓ MPC controller created")
except Exception as e:
    print(f"  ⚠ Could not create MPC controller: {e}")
    traceback.print_exc()
    controller_mpc = None

# Summary of what was created
print(f"\nController status:")
print(f"  PID: {'✓ Ready' if controller_pid else '✗ Failed'}")
print(f"  MPC: {'✓ Ready' if controller_mpc else '✗ Failed'}")

def get_active_controller():
    """Get the currently active controller based on controller_type."""
    if controller_type == 0:
        return controller_pid
    elif controller_type == 1:
        return controller_mpc
    else:
        return None  # SNN_PID/SNN_Stress handled by Nengo network, not external controller

def get_controller_name():
    """Get name of current controller with availability status."""
    if controller_type == 0:
        if controller_pid is None:
            return "PID (N/A)"
        return "PID"
    elif controller_type == 1:
        if controller_mpc is None:
            return "MPC (N/A)"
        return "MPC"
    elif controller_type == 2:
        return "SNN_PID"  # NEF-based spiking PD
    else:
        return "SNN_Stress"  # NEF-based spiking + strain

# ============================================================================
# Global State
# ============================================================================

current_strains = np.zeros(n_total_strains, dtype=np.float32)
current_forces = np.zeros(num_groups * 2, dtype=np.float32)
current_tracking_error = 0.0
simulation_time = 0.0
controller_status_text = "Controller: PID [ON]"

# SNN_PID specific state
current_error = np.zeros(num_groups * 2, dtype=np.float32)  # Error signal for SNN
snn_pid_output = np.zeros(num_groups * 2, dtype=np.float32)  # SNN computed forces
snn_pid_enabled = False  # Flag to track if SNN_PID is active

# Initialize pygame fonts
pygame.init()
font_small = pygame.font.Font(None, 18)
font_normal = pygame.font.Font(None, 24)
font_large = pygame.font.Font(None, 28)

# ============================================================================
# Physics Step Function (separate from Nengo)
# ============================================================================

# Track last physics step time to handle different dt values
# Nengo runs at NENGO_DT (e.g., 0.001s), physics at DT (e.g., 0.01s)
last_physics_time = -DT  # Force first step at t=0

def physics_step(t):
    """
    Execute one physics step with optional controller.
    Returns strain data for Nengo visualization.
    
    NOTE: This function is called at NENGO_DT intervals (e.g., 0.001s)
    but only steps physics at DT intervals (e.g., 0.01s) to avoid
    running physics too fast.
    """
    global current_strains, current_forces, current_tracking_error, simulation_time
    global controller_enabled, controller_status_text, controller_type
    global current_error, snn_pid_output, snn_pid_enabled
    global last_physics_time
    
    simulation_time = t
    
    # Always update error signal (needed for SNN computation at each Nengo step)
    state_dict = env.get_state_for_controller()
    centroids = state_dict.get('group_centroids', np.zeros((num_groups, 2)))
    targets = state_dict.get('group_targets', np.zeros((num_groups, 2)))
    error = (targets - centroids).flatten()
    current_error = error.astype(np.float32)
    
    # Only step physics at physics dt intervals
    if t - last_physics_time < DT - NENGO_DT * 0.5:
        # Not time for physics step yet - just return current strains
        return current_strains
    
    last_physics_time = t
    
    # Get active controller
    controller = get_active_controller()
    controller_name = get_controller_name()
    
    # Update controller status text
    status = "ON" if controller_enabled else "OFF"
    controller_status_text = f"Controller: {controller_name} [{status}]"
    
    # Compute control forces based on controller type
    if controller_type >= 2 and controller_enabled:
        # SNN_PID or SNN_Stress: Use neural network output (computed by Nengo network)
        snn_pid_enabled = True
        current_forces = snn_pid_output.copy()
        # Debug output every second
        if int(t) != int(t - DT) and t > 0.1:
            force_mag = np.linalg.norm(current_forces)
            ctrl_name = "SNN_PID" if controller_type == 2 else "SNN_Stress"
            print(f"[t={t:.1f}] {ctrl_name} ACTIVE: |F|={force_mag:.1f} (NEF-based)")
    elif controller is not None and controller_enabled:
        # Classic PID or MPC
        snn_pid_enabled = False
        try:
            forces = controller.compute_control(state_dict, env.get_target_position)
            current_forces = forces.flatten()
            # Debug output every second
            if int(t) != int(t - DT) and t > 0.1:
                force_mag = np.linalg.norm(current_forces)
                print(f"[t={t:.1f}] {controller_name} ACTIVE: |F|={force_mag:.1f}")
        except Exception as e:
            print(f"Controller error: {e}")
            current_forces = np.zeros(num_groups * 2, dtype=np.float32)
    else:
        snn_pid_enabled = False
        current_forces = np.zeros(num_groups * 2, dtype=np.float32)
        # Debug: show when controller is OFF
        if int(t) != int(t - DT) and t > 0.1:
            print(f"[t={t:.1f}] Controller OFF (enabled={controller_enabled}, ctrl={controller_name})")
    
    # Step physics (only happens at DT intervals)
    obs, reward, terminated, truncated, info = env.step(current_forces)
    
    # Get tracking error
    current_tracking_error = info.get('tracking_error', 0.0)
    
    # Extract strains
    strains = []
    if n_spring_strains > 0 and env.model.spring_strains_normalized is not None:
        spring_strains = env.model.spring_strains_normalized.numpy()
        strains.extend(spring_strains)
    if n_fem_strains > 0 and env.model.tri_strains_normalized is not None:
        fem_strains = env.model.tri_strains_normalized.numpy()
        strains.extend(fem_strains)
    
    if len(strains) == 0:
        current_strains = np.zeros(n_total_strains, dtype=np.float32)
    else:
        current_strains = np.array(strains, dtype=np.float32)
    
    return current_strains

# ============================================================================
# Render Function
# ============================================================================

def render_frame():
    """
    Render the current state with full TrackingEnv GUI.
    This includes:
    - Spring-mass grid visualization
    - Trajectory targets and tracking
    - XY plot panel
    - Error history plot
    - Force arrows
    - Group centroids
    - Controller status overlay
    """
    if env.render_mode != 'human':
        return
    
    # Sync GPU data to CPU
    wp.synchronize()
    env._sync_to_cpu()
    
    # Full render (includes plots, overlays, and pygame.display.flip())
    env.render()
    
    # Draw controller status overlay on top (after env.render() but need another flip)
    if env.window is not None:
        # Controller status box (top-left, below env's UI text)
        status_y = 110  # Below existing UI text
        
        # Background box for visibility
        box_width = 300
        box_height = 55
        pygame.draw.rect(env.window, (240, 240, 240), (5, status_y - 2, box_width, box_height))
        pygame.draw.rect(env.window, (100, 100, 100), (5, status_y - 2, box_width, box_height), 1)
        
        # Status text with color coding
        active_controller = get_active_controller()
        if controller_enabled and active_controller is not None:
            color = (0, 150, 0)  # Green when ON
        else:
            color = (200, 50, 50)  # Red when OFF
        
        status_surface = font_normal.render(controller_status_text, True, color)
        env.window.blit(status_surface, (10, status_y))
        
        # Show current PID parameters
        if controller_pid is not None:
            kp_val = controller_pid.Kp
            ki_val = controller_pid.Ki
            kd_val = controller_pid.Kd
            umax_val = controller_pid.u_max
            pid_text = f"Kp={kp_val:.0f}  Ki={ki_val:.1f}  Kd={kd_val:.0f}  u_max={umax_val:.0f}"
            pid_surface = font_small.render(pid_text, True, (50, 50, 50))
            env.window.blit(pid_surface, (10, status_y + 22))
        
        # Nengo GUI indicator
        nengo_text = "Running via Nengo GUI (adjust PID sliders in browser)"
        nengo_surface = font_small.render(nengo_text, True, (100, 100, 200))
        env.window.blit(nengo_surface, (10, status_y + 38))
        
        # Update display with overlay
        pygame.display.flip()
    
    # Process pygame events to keep window responsive
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            print("Window closed by user")
            env.close()
            raise SystemExit

# ============================================================================
# Build Nengo Network
# ============================================================================

model = nengo.Network(label=f"Tracking SNN ({N}x{N})")

with model:
    
    # =========================================================================
    # Controller Toggle Widgets
    # =========================================================================
    
    # Controller enable/disable toggle [0=OFF, 1=ON]
    controller_toggle = nengo.Node([1.0], label="Controller [0=OFF, 1=ON]")
    
    # Controller type selector [0=PID, 1=MPC, 2=SNN_PID, 3=SNN_Stress]
    controller_type_selector = nengo.Node([float(INITIAL_CONTROLLER)], label="Type [PID=0,MPC=1,SNN_PD=2,Stress=3]")
    
    # =========================================================================
    # PID Parameter Sliders (adjustable in real-time)
    # =========================================================================
    
    # Default PID gains
    DEFAULT_KP = 200.0
    DEFAULT_KI = 10.0
    DEFAULT_KD = 50.0
    DEFAULT_UMAX = 500.0
    
    # PID Proportional gain [0-500]
    pid_kp_slider = nengo.Node([DEFAULT_KP], label="PID Kp [0-500]")
    
    # PID Integral gain [0-50]
    pid_ki_slider = nengo.Node([DEFAULT_KI], label="PID Ki [0-50]")
    
    # PID Derivative gain [0-200]
    pid_kd_slider = nengo.Node([DEFAULT_KD], label="PID Kd [0-200]")
    
    # Maximum control force [100-1000]
    pid_umax_slider = nengo.Node([DEFAULT_UMAX], label="u_max [100-1000]")
    
    # =========================================================================
    # Trajectory Selector (switchable in real-time)
    # =========================================================================
    
    # Map trajectory types to indices
    TRAJECTORY_TYPES = ['sinusoidal', 'circular', 'figure8']
    
    # Get initial trajectory index
    initial_traj_idx = TRAJECTORY_TYPES.index(TRAJECTORY_TYPE) if TRAJECTORY_TYPE in TRAJECTORY_TYPES else 1
    
    # Trajectory selector [0=sinusoidal, 1=circular, 2=figure8]
    trajectory_slider = nengo.Node([float(initial_traj_idx)], label="Traj [0=sin, 1=circ, 2=fig8]")
    
    # =========================================================================
    # Strain Input from Simulation (reads ALL slider values via input)
    # =========================================================================
    
    def update_simulation_with_control(t, x):
        """
        Main update function called by Nengo.
        Receives slider values as input to ensure correct execution order.
        x[0] = controller enabled (0=OFF, 1=ON)
        x[1] = controller type (0=PID, 1=MPC, 2=SNN_PID, 3=SNN_Stress)
        x[2] = Kp (proportional gain)
        x[3] = Ki (integral gain)
        x[4] = Kd (derivative gain)
        x[5] = u_max (max force)
        x[6] = trajectory type (0=sinusoidal, 1=circular, 2=figure8)
        """
        global controller_enabled, controller_type
        
        # Update controller state from slider inputs
        controller_enabled = x[0] > 0.5
        
        # Controller type: 0=PID, 1=MPC, 2=SNN_PID, 3=SNN_Stress
        type_val = x[1]
        if type_val < 0.5:
            controller_type = 0  # PID
        elif type_val < 1.5:
            controller_type = 1  # MPC
        elif type_val < 2.5:
            controller_type = 2  # SNN_PID
        else:
            controller_type = 3  # SNN_Stress
        
        # Update PID parameters in real-time (for classic PID)
        if controller_pid is not None:
            controller_pid.Kp = np.clip(x[2], 0.0, 500.0)
            controller_pid.Ki = np.clip(x[3], 0.0, 50.0)
            controller_pid.Kd = np.clip(x[4], 0.0, 200.0)
            controller_pid.u_max = np.clip(x[5], 100.0, 1000.0)
        
        # Also update MPC u_max if available
        if controller_mpc is not None:
            controller_mpc.u_max = np.clip(x[5], 100.0, 1000.0)
        
        # Update trajectory type in real-time
        traj_idx = int(np.clip(np.round(x[6]), 0, 2))
        env.trajectory_type = TRAJECTORY_TYPES[traj_idx]
        
        # Physics step
        strains = physics_step(t)
        
        # Render
        render_frame()
        
        # Return strains for SNN visualization
        return strains
    
    strain_input = nengo.Node(
        update_simulation_with_control,
        size_in=7,  # Receives all slider values
        size_out=n_total_strains,
        label="Strain_Input"
    )
    
    # Connect all sliders to strain_input so they're processed BEFORE physics
    nengo.Connection(controller_toggle, strain_input[0], synapse=None)
    nengo.Connection(controller_type_selector, strain_input[1], synapse=None)
    nengo.Connection(pid_kp_slider, strain_input[2], synapse=None)
    nengo.Connection(pid_ki_slider, strain_input[3], synapse=None)
    nengo.Connection(pid_kd_slider, strain_input[4], synapse=None)
    nengo.Connection(pid_umax_slider, strain_input[5], synapse=None)
    nengo.Connection(trajectory_slider, strain_input[6], synapse=None)
    
    # =========================================================================
    # 7D Strain Ensembles (per group: 5 springs + 2 FEMs)
    # =========================================================================
    
    group_averages = []  # Will hold 7D strain ensembles (one per group)
    
    # Assign strains to groups (same logic as interface)
    def particle_to_grid(idx, N):
        return (idx // N, idx % N)
    
    def get_particle_groups(idx, N):
        row, col = particle_to_grid(idx, N)
        groups = []
        if row < N-1 and col < N-1:
            groups.append((row, col))
        if row < N-1 and col > 0:
            groups.append((row, col-1))
        if row > 0 and col < N-1:
            groups.append((row-1, col))
        if row > 0 and col > 0:
            groups.append((row-1, col-1))
        return groups
    
    # Get spring and FEM indices
    spring_indices_np = env.model.spring_indices.numpy()
    tri_indices_np = env.model.tri_indices.numpy() if hasattr(env.model, 'tri_indices') and env.model.tri_indices is not None else np.array([])
    
    # Assign springs to groups
    spring_groups = {g: [] for g in range(num_groups)}
    for spring_idx in range(0, len(spring_indices_np), 2):
        if spring_idx + 1 >= len(spring_indices_np):
            break
        p0 = spring_indices_np[spring_idx]
        p1 = spring_indices_np[spring_idx + 1]
        groups0 = set(get_particle_groups(p0, N))
        groups1 = set(get_particle_groups(p1, N))
        common = groups0 & groups1
        for (gr, gc) in common:
            gid = gr * num_groups_per_side + gc
            if gid < num_groups:
                spring_groups[gid].append(spring_idx // 2)
    
    # Assign FEMs to groups
    fem_groups = {g: [] for g in range(num_groups)}
    for fem_idx in range(0, len(tri_indices_np), 3):
        if fem_idx + 2 >= len(tri_indices_np):
            break
        p0, p1, p2 = tri_indices_np[fem_idx:fem_idx+3]
        groups0 = set(get_particle_groups(p0, N))
        groups1 = set(get_particle_groups(p1, N))
        groups2 = set(get_particle_groups(p2, N))
        common = groups0 & groups1 & groups2
        for (gr, gc) in common:
            gid = gr * num_groups_per_side + gc
            if gid < num_groups:
                fem_groups[gid].append(fem_idx // 3)
    
    # Create 7D strain ensembles (one per group)
    # Each ensemble represents: [spring_0, spring_1, ..., spring_4, fem_0, fem_1]
    N_SPRINGS_VIZ = 5
    N_FEMS_VIZ = 2
    STRAIN_DIM_VIZ = N_SPRINGS_VIZ + N_FEMS_VIZ  # 7
    
    all_ensembles = []
    
    for group_id in range(num_groups):
        group_springs = spring_groups.get(group_id, [])
        group_fems = fem_groups.get(group_id, [])
        
        group_row = group_id // num_groups_per_side
        group_col = group_id % num_groups_per_side
        
        # 7D strain ensemble (5 springs + 2 FEMs)
        # 
        # HIGH-DIMENSIONAL INTERCEPT FIX:
        # With uniform intercepts in 7D, most neurons either never fire or always fire.
        # A neuron with intercept=0.5 fires for only ~2-3% of inputs in 7D space.
        # 
        # Solution: Use triangular distribution → inverse beta transform
        # This concentrates intercepts where neurons achieve ~50% firing probability,
        # maximizing information capacity. See: Zaidel et al. 2021 (NEF-based IK/PID).
        #
        n_ens_neurons = N_NEURONS * STRAIN_DIM_VIZ
        # Zaidel et al. 2021: Triangular distribution for high-D intercepts
        # θ = 1 - 2·I⁻¹((d+1)/2, 1/2, 1-p) where p ~ Triangular
        # Paper uses roughly (0.3, 0.5, 0.7) → spread ≈ 0.2
        d = STRAIN_DIM_VIZ  # 7
        spread = 0.2  # Fixed spread from paper (empirical, not dimension-dependent)
        mode = 0.5   # Center at 50% firing probability
        triangular_samples = np.random.triangular(mode - spread, mode, mode + spread, n_ens_neurons)
        triangular_intercepts = nengo.dists.CosineSimilarity(d + 2).ppf(1 - triangular_samples)
        
        strain_7d_ensemble = nengo.Ensemble(
            n_neurons=n_ens_neurons,
            dimensions=STRAIN_DIM_VIZ,
            max_rates=nengo.dists.Uniform(100, 200),
            intercepts=triangular_intercepts,  # Triangular for 7D (was Uniform - bad!)
            neuron_type=nengo.LIF(),
            radius=1.0,
            label=f"G{group_id}[{group_row},{group_col}]_Strain[7D]"
        )
        
        # Connect springs to dimensions 0-4
        for i, spring_idx in enumerate(group_springs[:N_SPRINGS_VIZ]):
            transform = np.zeros((STRAIN_DIM_VIZ, 1))
            transform[i, 0] = 1.0
            nengo.Connection(strain_input[spring_idx], strain_7d_ensemble, 
                           transform=transform, synapse=0.01)
        
        # Connect FEMs to dimensions 5-6
        for i, fem_idx in enumerate(group_fems[:N_FEMS_VIZ]):
            strain_idx = n_spring_strains + fem_idx
            transform = np.zeros((STRAIN_DIM_VIZ, 1))
            transform[N_SPRINGS_VIZ + i, 0] = 1.0
            nengo.Connection(strain_input[strain_idx], strain_7d_ensemble,
                           transform=transform, synapse=0.01)
        
        group_averages.append(strain_7d_ensemble)
        all_ensembles.append(strain_7d_ensemble)
    
    # =========================================================================
    # NEF-Based SNN Controllers (SNN_PID / SNN_Stress)
    # 
    # Uses SNN_Stress_Controller which includes:
    #   - SNN_PID: Pure NEF-based PD (Zaidel et al. 2021)
    #   - SNN_Stress: Adds strain-rate damping for compliance
    # =========================================================================
    
    # Configuration
    snn_n_neurons = 500     # More neurons = smoother output
    error_scale = 2.0       # meters - max expected error
    output_scale = 500.0    # Newtons - max output force (u_max)
    pes_learning_rate = 1e-4  # PES feedforward learning rate
    
    # Create SNN controller (SNN_Stress = PD + PES feedforward)
    # Gains tuned for NEF (Zaidel et al. 2021)
    snn_pd_controller = SNN_Stress_Controller(
        num_groups=num_groups,
        Kp=250.0,   # Tuned for NEF (compensates synaptic smoothing)
        Kd=80.0,    # Higher for neural differentiator
        u_max=output_scale,
        n_neurons=snn_n_neurons,
        error_scale=error_scale,
        pes_learning_rate=pes_learning_rate,
    )
    
    # Create error callback factory (for each group)
    def make_error_callback(group_id):
        """Create error function for a specific group."""
        def get_error(t):
            idx = group_id * 2
            err = current_error[idx:idx+2]
            return err / error_scale
        return get_error
    
    # Build all SNN networks with 7D strain ensembles (group_averages is now 7D!)
    # The group_averages list now contains 7D ensembles: [s0, s1, s2, s3, s4, f0, f1]
    # FULLY NEURAL: Uses NEF differentiator for D term (Zaidel et al. 2021)
    snn_result = snn_pd_controller.build_all(
        model=model,
        get_error_callback=make_error_callback,
        strain_ensembles=group_averages,  # Reuse 7D strain ensembles for PES
        dt=DT,
    )
    
    # Get ensembles for visualization
    pid_snn_ensembles = snn_result['ensembles']
    
    # Update global output reference (used by physics_step)
    def sync_snn_output(t):
        global snn_pid_output
        snn_pid_output[:] = snn_pd_controller.get_output()
    
    nengo.Node(sync_snn_output, size_in=0, size_out=0, label="SNN_Output_Sync")
    
    print(f"    PES feedforward: lr={pes_learning_rate}, strain_dim=7")
    print(f"    📊 To see PES learning: watch q(t) error decrease over time")
    
    # =========================================================================
    # Control Output Display (shows controller forces)
    # =========================================================================
    
    # Node to display current control forces
    control_display = nengo.Node(
        lambda t: current_forces,
        size_out=num_groups * 2,
        label="Control_Forces"
    )
    
    # Force ensembles for visualization (just display, not control)
    force_display_ensembles = []
    for group_id in range(num_groups):
        group_row = group_id // num_groups_per_side
        group_col = group_id % num_groups_per_side
        
        # Create ensemble for force display (2D per group)
        force_ens = nengo.Ensemble(
            n_neurons=100,
            dimensions=2,
            max_rates=nengo.dists.Uniform(100, 200),
            intercepts=nengo.dists.Uniform(-0.5, 0.5),
            neuron_type=nengo.LIF(),
            radius=1.0,
            label=f"Force_G{group_id}[{group_row},{group_col}]"
        )
        
        # Connect control forces (normalized by force_scale)
        nengo.Connection(
            control_display[group_id*2:(group_id+1)*2],
            force_ens,
            transform=1.0/500.0,  # Normalize to [-1, 1]
            synapse=0.01
        )
        force_display_ensembles.append(force_ens)
    
print(f"\n{'='*70}")
print("Trajectory Tracking SNN Network Created")
print(f"{'='*70}")
pid_status = "✓" if controller_pid is not None else "✗"
mpc_status = "✓" if controller_mpc is not None else "✗"
print(f"Controllers:")
print(f"  [{pid_status}] PID (classic)")
print(f"  [{mpc_status}] MPC (classic)")
print(f"Strain ensembles: {len(group_averages)} (7D each: 5 springs + 2 FEMs)")
print(f"Force display ensembles: {len(force_display_ensembles)}")
print(f"Structure: {num_groups_per_side}x{num_groups_per_side} groups")
print(f"")
print(f"🎛️  NENGO GUI CONTROLS:")
print(f"  • Controller [0=OFF, 1=ON]: Toggle controller on/off")
print(f"  • Type [0-3]: Select controller type")
print(f"    - 0 = Classic PID")
print(f"    - 1 = Classic MPC")
print(f"    - 2 = SNN_PID (NEF-based spiking PD)")
print(f"    - 3 = SNN_Stress (NEF-based + strain feedback)")
print(f"  • PID Kp [0-500]: Proportional gain (default: 200)")
print(f"  • PID Ki [0-50]:  Integral gain (default: 10)")
print(f"  • PID Kd [0-200]: Derivative gain (default: 50)")
print(f"  • u_max [100-1000]: Maximum control force (default: 500)")
print(f"  • Traj [0-2]: Trajectory type (0=sinusoidal, 1=circular, 2=figure8)")
print(f"")
print(f"  SNN_PID/SNN_Stress use neural integrator & differentiator (NEF)")
print(f"  TIP: Drag sliders while simulation runs to see real-time effect!")
print(f"{'='*70}")

# ============================================================================
# Create Nengo GUI Configuration
# ============================================================================

config_filename = os.path.join(os.path.dirname(__file__), 'snn_nengo_tracking_gui.py.cfg')
print(f"\n📝 Creating Nengo GUI config: {config_filename}")

# Find ensemble indices
group_avg_indices = []
for avg_ens in group_averages:
    for idx, ens in enumerate(model.all_ensembles):
        if ens is avg_ens:
            group_avg_indices.append(idx)
            break

force_ens_indices = []
for force_ens in force_display_ensembles:
    for idx, ens in enumerate(model.all_ensembles):
        if ens is force_ens:
            force_ens_indices.append(idx)
            break

# Find SNN PID ensemble indices
snn_pid_ens_indices = []
snn_pid_ens_names = []
for snn_ens in pid_snn_ensembles:
    for idx, ens in enumerate(model.all_ensembles):
        if ens is snn_ens:
            snn_pid_ens_indices.append(idx)
            snn_pid_ens_names.append(snn_ens.label)
            break


# Find controller toggle nodes, PID parameter nodes, and trajectory node
toggle_node_idx = None
type_selector_node_idx = None
kp_node_idx = None
ki_node_idx = None
kd_node_idx = None
umax_node_idx = None
trajectory_node_idx = None

for idx, node in enumerate(model.all_nodes):
    if node is controller_toggle:
        toggle_node_idx = idx
    if node is controller_type_selector:
        type_selector_node_idx = idx
    if node is pid_kp_slider:
        kp_node_idx = idx
    if node is pid_ki_slider:
        ki_node_idx = idx
    if node is pid_kd_slider:
        kd_node_idx = idx
    if node is pid_umax_slider:
        umax_node_idx = idx
    if node is trajectory_slider:
        trajectory_node_idx = idx

print(f"Found {len(group_avg_indices)} group strain ensembles (7D)")
print(f"Found {len(force_ens_indices)} force display ensembles")
print(f"Found {len(snn_pid_ens_indices)} SNN PID ensembles: {snn_pid_ens_names}")
print(f"Controller toggle node at index: {toggle_node_idx}")
print(f"Controller type selector at index: {type_selector_node_idx}")
print(f"PID sliders: Kp={kp_node_idx}, Ki={ki_node_idx}, Kd={kd_node_idx}, u_max={umax_node_idx}")
print(f"Trajectory slider at index: {trajectory_node_idx}")

# Generate config file
with open(config_filename, 'w') as f:
    f.write("# Nengo GUI Configuration - Trajectory Tracking SNN\n")
    f.write("_viz_net_graph = nengo_gui.components.NetGraph()\n\n")
    
    # Layout dimensions
    plot_size = 0.10
    plot_gap = 0.01
    margin_left = 0.0
    margin_top = 0.02
    pair_gap = 0.02
    
    pair_height = plot_size + plot_gap + plot_size
    
    num_plots = min(len(group_avg_indices), len(force_ens_indices))
    
    print(f"Creating config for {num_plots} groups...")
    
    for i in range(num_plots):
        y_base = margin_top + i * (pair_height + plot_gap)
        y_value = y_base
        y_spike = y_value + plot_size + plot_gap
        
        strain_x = margin_left
        force_x = margin_left + plot_size + pair_gap
        
        strain_ens_idx = group_avg_indices[i]
        force_ens_idx = force_ens_indices[i]
        
        # Strain value (7D: spring0-4, fem0-1)
        f.write(f"_viz_strain_value_{i} = nengo_gui.components.Value(model.all_ensembles[{strain_ens_idx}])\n")
        f.write(f"_viz_config[_viz_strain_value_{i}].x = {strain_x:.4f}\n")
        f.write(f"_viz_config[_viz_strain_value_{i}].y = {y_value:.4f}\n")
        f.write(f"_viz_config[_viz_strain_value_{i}].width = {plot_size:.4f}\n")
        f.write(f"_viz_config[_viz_strain_value_{i}].height = {plot_size:.4f}\n")
        f.write(f"_viz_config[_viz_strain_value_{i}].max_value = 1.0\n")
        f.write(f"_viz_config[_viz_strain_value_{i}].min_value = -1.0\n")
        f.write(f"_viz_config[_viz_strain_value_{i}].show_legend = True\n")
        f.write(f"_viz_config[_viz_strain_value_{i}].label_visible = True\n")
        f.write(f"_viz_config[_viz_strain_value_{i}].legend_labels = ['spring0', 'spring1', 'spring2', 'spring3', 'spring4', 'fem0', 'fem1']\n\n")
        
        # Strain raster
        f.write(f"_viz_strain_raster_{i} = nengo_gui.components.Raster(model.all_ensembles[{strain_ens_idx}].neurons)\n")
        f.write(f"_viz_config[_viz_strain_raster_{i}].x = {strain_x:.4f}\n")
        f.write(f"_viz_config[_viz_strain_raster_{i}].y = {y_spike:.4f}\n")
        f.write(f"_viz_config[_viz_strain_raster_{i}].width = {plot_size:.4f}\n")
        f.write(f"_viz_config[_viz_strain_raster_{i}].height = {plot_size:.4f}\n")
        f.write(f"_viz_config[_viz_strain_raster_{i}].show_legend = False\n")
        f.write(f"_viz_config[_viz_strain_raster_{i}].label_visible = False\n\n")
        
        # Force XY-Value (2D: Fx vs Fy)
        f.write(f"_viz_force_xy_{i} = nengo_gui.components.XYValue(model.all_ensembles[{force_ens_idx}])\n")
        f.write(f"_viz_config[_viz_force_xy_{i}].x = {force_x:.4f}\n")
        f.write(f"_viz_config[_viz_force_xy_{i}].y = {y_value:.4f}\n")
        f.write(f"_viz_config[_viz_force_xy_{i}].width = {plot_size:.4f}\n")
        f.write(f"_viz_config[_viz_force_xy_{i}].height = {plot_size:.4f}\n")
        f.write(f"_viz_config[_viz_force_xy_{i}].max_value = 1.0\n")
        f.write(f"_viz_config[_viz_force_xy_{i}].min_value = -1.0\n")
        f.write(f"_viz_config[_viz_force_xy_{i}].label_visible = True\n\n")
        
        # Force raster
        f.write(f"_viz_force_raster_{i} = nengo_gui.components.Raster(model.all_ensembles[{force_ens_idx}].neurons)\n")
        f.write(f"_viz_config[_viz_force_raster_{i}].x = {force_x:.4f}\n")
        f.write(f"_viz_config[_viz_force_raster_{i}].y = {y_spike:.4f}\n")
        f.write(f"_viz_config[_viz_force_raster_{i}].width = {plot_size:.4f}\n")
        f.write(f"_viz_config[_viz_force_raster_{i}].height = {plot_size:.4f}\n")
        f.write(f"_viz_config[_viz_force_raster_{i}].show_legend = False\n")
        f.write(f"_viz_config[_viz_force_raster_{i}].label_visible = False\n\n")
        
        print(f"  G{i}: Strain (7D) + Force (XY)")
    
    # Network graph
    plots_end_x = force_x + plot_size if num_plots > 0 else 0.25
    network_start_x = plots_end_x + 0.05
    network_width = 1.0 - network_start_x - 0.01
    network_center_x = network_start_x + network_width / 2
    
    f.write(f"_viz_config[model].pos = ({network_center_x:.3f}, 0.50)\n")
    f.write(f"_viz_config[model].size = ({network_width:.3f}, 0.96)\n")
    f.write(f"_viz_config[model].expanded = False\n")
    f.write(f"_viz_config[model].has_layout = False\n\n")
    
    f.write("_viz_sim_control = nengo_gui.components.SimControl()\n")
    f.write("_viz_config[_viz_sim_control].shown_time = 0.5\n")
    f.write("_viz_config[_viz_sim_control].kept_time = 4.0\n\n")
    
    # Controller toggle slider (ON/OFF)
    slider_x = 0.25
    slider_y = 0.02
    slider_width = 0.15
    slider_height = 0.08
    
    if toggle_node_idx is not None:
        f.write("# Controller Toggle Slider (ON/OFF)\n")
        f.write(f"_viz_slider_toggle = nengo_gui.components.Slider(model.all_nodes[{toggle_node_idx}])\n")
        f.write(f"_viz_config[_viz_slider_toggle].x = {slider_x:.4f}\n")
        f.write(f"_viz_config[_viz_slider_toggle].y = {slider_y:.4f}\n")
        f.write(f"_viz_config[_viz_slider_toggle].width = {slider_width:.4f}\n")
        f.write(f"_viz_config[_viz_slider_toggle].height = {slider_height:.4f}\n")
        f.write(f"_viz_config[_viz_slider_toggle].min_value = 0\n")
        f.write(f"_viz_config[_viz_slider_toggle].max_value = 1\n\n")
        print(f"  Controller toggle slider added")
    
    # Controller type selector slider (PID/MPC/SNN_PID/SNN_Stress)
    if type_selector_node_idx is not None:
        slider_y2 = slider_y + slider_height + 0.02
        f.write("# Controller Type Selector (0=PID, 1=MPC, 2=SNN_PID, 3=SNN_Stress)\n")
        f.write(f"_viz_slider_type = nengo_gui.components.Slider(model.all_nodes[{type_selector_node_idx}])\n")
        f.write(f"_viz_config[_viz_slider_type].x = {slider_x:.4f}\n")
        f.write(f"_viz_config[_viz_slider_type].y = {slider_y2:.4f}\n")
        f.write(f"_viz_config[_viz_slider_type].width = {slider_width:.4f}\n")
        f.write(f"_viz_config[_viz_slider_type].height = {slider_height:.4f}\n")
        f.write(f"_viz_config[_viz_slider_type].min_value = 0\n")
        f.write(f"_viz_config[_viz_slider_type].max_value = 3\n\n")  # 0=PID, 1=MPC, 2=SNN_PID, 3=SNN_Stress
        print(f"  Controller type selector slider added [0=PID, 1=MPC, 2=SNN_PID, 3=SNN_Stress]")
    
    # =========================================================================
    # PID Parameter Sliders (Real-time tuning)
    # =========================================================================
    f.write("# PID Parameter Sliders (Real-time tuning)\n")
    
    pid_slider_x = slider_x + slider_width + 0.05  # Next column
    pid_slider_height = 0.06
    
    # Kp slider [0-500]
    if kp_node_idx is not None:
        kp_y = 0.02
        f.write(f"_viz_slider_kp = nengo_gui.components.Slider(model.all_nodes[{kp_node_idx}])\n")
        f.write(f"_viz_config[_viz_slider_kp].x = {pid_slider_x:.4f}\n")
        f.write(f"_viz_config[_viz_slider_kp].y = {kp_y:.4f}\n")
        f.write(f"_viz_config[_viz_slider_kp].width = {slider_width:.4f}\n")
        f.write(f"_viz_config[_viz_slider_kp].height = {pid_slider_height:.4f}\n")
        f.write(f"_viz_config[_viz_slider_kp].min_value = 0\n")
        f.write(f"_viz_config[_viz_slider_kp].max_value = 500\n\n")
        print(f"  PID Kp slider added [0-500]")
    
    # Ki slider [0-50]
    if ki_node_idx is not None:
        ki_y = kp_y + pid_slider_height + 0.02
        f.write(f"_viz_slider_ki = nengo_gui.components.Slider(model.all_nodes[{ki_node_idx}])\n")
        f.write(f"_viz_config[_viz_slider_ki].x = {pid_slider_x:.4f}\n")
        f.write(f"_viz_config[_viz_slider_ki].y = {ki_y:.4f}\n")
        f.write(f"_viz_config[_viz_slider_ki].width = {slider_width:.4f}\n")
        f.write(f"_viz_config[_viz_slider_ki].height = {pid_slider_height:.4f}\n")
        f.write(f"_viz_config[_viz_slider_ki].min_value = 0\n")
        f.write(f"_viz_config[_viz_slider_ki].max_value = 50\n\n")
        print(f"  PID Ki slider added [0-50]")
    
    # Kd slider [0-200]
    if kd_node_idx is not None:
        kd_y = ki_y + pid_slider_height + 0.02
        f.write(f"_viz_slider_kd = nengo_gui.components.Slider(model.all_nodes[{kd_node_idx}])\n")
        f.write(f"_viz_config[_viz_slider_kd].x = {pid_slider_x:.4f}\n")
        f.write(f"_viz_config[_viz_slider_kd].y = {kd_y:.4f}\n")
        f.write(f"_viz_config[_viz_slider_kd].width = {slider_width:.4f}\n")
        f.write(f"_viz_config[_viz_slider_kd].height = {pid_slider_height:.4f}\n")
        f.write(f"_viz_config[_viz_slider_kd].min_value = 0\n")
        f.write(f"_viz_config[_viz_slider_kd].max_value = 200\n\n")
        print(f"  PID Kd slider added [0-200]")
    
    # u_max slider [100-1000]
    if umax_node_idx is not None:
        umax_y = kd_y + pid_slider_height + 0.02
        f.write(f"_viz_slider_umax = nengo_gui.components.Slider(model.all_nodes[{umax_node_idx}])\n")
        f.write(f"_viz_config[_viz_slider_umax].x = {pid_slider_x:.4f}\n")
        f.write(f"_viz_config[_viz_slider_umax].y = {umax_y:.4f}\n")
        f.write(f"_viz_config[_viz_slider_umax].width = {slider_width:.4f}\n")
        f.write(f"_viz_config[_viz_slider_umax].height = {pid_slider_height:.4f}\n")
        f.write(f"_viz_config[_viz_slider_umax].min_value = 100\n")
        f.write(f"_viz_config[_viz_slider_umax].max_value = 1000\n\n")
        print(f"  u_max slider added [100-1000]")
    
    # =========================================================================
    # Trajectory Selector Slider
    # =========================================================================
    f.write("# Trajectory Selector (0=sinusoidal, 1=circular, 2=figure8)\n")
    
    if trajectory_node_idx is not None:
        traj_y = umax_y + pid_slider_height + 0.04  # Extra space before trajectory
        f.write(f"_viz_slider_traj = nengo_gui.components.Slider(model.all_nodes[{trajectory_node_idx}])\n")
        f.write(f"_viz_config[_viz_slider_traj].x = {pid_slider_x:.4f}\n")
        f.write(f"_viz_config[_viz_slider_traj].y = {traj_y:.4f}\n")
        f.write(f"_viz_config[_viz_slider_traj].width = {slider_width:.4f}\n")
        f.write(f"_viz_config[_viz_slider_traj].height = {pid_slider_height:.4f}\n")
        f.write(f"_viz_config[_viz_slider_traj].min_value = 0\n")
        f.write(f"_viz_config[_viz_slider_traj].max_value = 2\n\n")
        print(f"  Trajectory slider added [0=sin, 1=circ, 2=fig8]")
    
    # =========================================================================
    # SNN PID Ensemble Visualizations
    # Show first group's PID components (others visible in network graph)
    # =========================================================================
    f.write("# SNN PID Ensembles (Group 0 - representative)\n")
    f.write("# All 2D ensembles shown as XY-Value plots (X vs Y)\n\n")
    
    snn_col_x = pid_slider_x + slider_width + 0.05
    snn_plot_size = 0.08
    snn_plot_gap = 0.01
    
    # All PID ensembles as XY plots (2D: X vs Y)
    snn_labels = ["q(t)", "ei(t)", "ed(t)", "u(t)"]
    for i in range(min(4, len(snn_pid_ens_indices))):
        ens_idx = snn_pid_ens_indices[i]
        label = snn_labels[i]
        safe_label = label.replace("(", "").replace(")", "")
        
        row = i // 2
        col = i % 2
        
        xy_x = snn_col_x + col * (snn_plot_size + snn_plot_gap)
        xy_y = 0.02 + row * (snn_plot_size + snn_plot_gap)
        
        # XYValue plot - shows X vs Y (2D phase space)
        f.write(f"_viz_snn_{safe_label}_xy = nengo_gui.components.XYValue(model.all_ensembles[{ens_idx}])\n")
        f.write(f"_viz_config[_viz_snn_{safe_label}_xy].x = {xy_x:.4f}\n")
        f.write(f"_viz_config[_viz_snn_{safe_label}_xy].y = {xy_y:.4f}\n")
        f.write(f"_viz_config[_viz_snn_{safe_label}_xy].width = {snn_plot_size:.4f}\n")
        f.write(f"_viz_config[_viz_snn_{safe_label}_xy].height = {snn_plot_size:.4f}\n")
        f.write(f"_viz_config[_viz_snn_{safe_label}_xy].max_value = 1.0\n")
        f.write(f"_viz_config[_viz_snn_{safe_label}_xy].min_value = -1.0\n")
        f.write(f"_viz_config[_viz_snn_{safe_label}_xy].label_visible = True\n\n")
        
        print(f"  G0_{label}: XY-Value plot (X vs Y)")

print(f"✅ Config created!")
print(f"\n🌐 Starting Nengo GUI server...")
print()

