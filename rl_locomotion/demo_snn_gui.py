#!/usr/bin/env python3
"""
SNN CPG Locomotion Demo with Nengo GUI

Architecture (follows trajectory_tracking pattern):
- 7D strain ensembles per group (5 springs + 2 FEMs)
- Spiking Hopf CPG oscillators per group (replaces PID controller)
- CPG → radial force injection
- Ratchet friction for locomotion

Author: NBEL
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'warp'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'openai-gym'))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import nengo
import warp as wp
import pygame

# Disable Warp CUDA graph capture to avoid stream conflicts with Nengo
wp.config.enable_graph_capture_on_kernels = False

from spring_mass_env import SpringMassEnv
from solvers import SolverImplicit
from balloon_forces import BalloonForces
from pygame_renderer import Renderer

# ============================================================================
# Configuration
# ============================================================================

ROWS = int(os.environ.get('SNN_ROWS', '3'))
COLS = int(os.environ.get('SNN_COLS', '6'))
ROWS = max(ROWS, 2)
COLS = max(COLS, 2)
DT = float(os.environ.get('SNN_DT', '0.01'))
DEVICE = os.environ.get('SNN_DEVICE', 'cuda')

# CPG settings (match classic demo_simple_cpg.py)
FREQUENCY = float(os.environ.get('SNN_FREQUENCY', '4.0'))
AMPLITUDE = float(os.environ.get('SNN_AMPLITUDE', '1.0'))
DIR_X = float(os.environ.get('SNN_DIR_X', '1.0'))
DIR_Y = float(os.environ.get('SNN_DIR_Y', '0.0'))
DIRECTION = np.array([DIR_X, DIR_Y])

# SNN settings
N_NEURONS = int(os.environ.get('SNN_N_NEURONS', '50'))
NENGO_DT = float(os.environ.get('SNN_NENGO_DT', '0.001'))

# Force settings (match classic demo_simple_cpg.py = 20.0)
FORCE_SCALE = float(os.environ.get('SNN_FORCE_SCALE', '20.0'))

# Window settings
WINDOW_WIDTH = int(os.environ.get('SNN_WINDOW_WIDTH', '1000'))
WINDOW_HEIGHT = int(os.environ.get('SNN_WINDOW_HEIGHT', '500'))
BOXSIZE = float(os.environ.get('SNN_BOXSIZE', '2.5'))

# Derived
OMEGA = 2 * np.pi * FREQUENCY
num_groups = (ROWS - 1) * (COLS - 1)
num_groups_rows = ROWS - 1
num_groups_cols = COLS - 1

print(f"{'='*70}")
print(f"SNN CPG Locomotion - Nengo GUI")
print(f"{'='*70}")
print(f"Configuration:")
print(f"  Grid size: {COLS}x{ROWS} ({ROWS*COLS} particles)")
print(f"  Groups: {num_groups} ({num_groups_cols}x{num_groups_rows})")
print(f"  CPG frequency: {FREQUENCY} Hz")
print(f"  CPG amplitude: {AMPLITUDE}")
print(f"  Direction: ({DIR_X}, {DIR_Y})")
print(f"  Force scale: {FORCE_SCALE}")
print(f"  Device: {DEVICE}")
print(f"  Physics dt: {DT}s ({1/DT:.0f} Hz)")
print(f"  Nengo dt: {NENGO_DT}s ({1/NENGO_DT:.0f} Hz)")
print()

# ============================================================================
# Create Environment
# ============================================================================

print("Creating Spring-Mass Environment...")
pygame.init()

env = SpringMassEnv(
    render_mode='human',
    rows=ROWS, cols=COLS,
    dt=DT,
    spring_coeff=50.0,
    spring_damping=0.3,
    gravity=-0.5,
    boxsize=BOXSIZE,
    device=DEVICE,
    with_fem=True,
    with_springs=True,
    window_width=WINDOW_WIDTH,
    window_height=WINDOW_HEIGHT,
)

# Replace solver with ratchet friction
print(f"Setting up solver with ratchet friction, direction = ({DIR_X}, {DIR_Y})...")
env.solver = SolverImplicit(
    env.model,
    dt=DT,
    mass=1.0,
    preconditioner_type="diag",
    solver_type="bicgstab",
    max_iterations=30,
    tolerance=1e-3,
    ratchet_friction=True,
    locomotion_direction=(DIR_X, DIR_Y),
)

env.reset(seed=42)

# Lower body to ground (matching demo_simple_cpg.py)
initial_pos = env.state_in.particle_q.numpy()  # Use state_in, not model
min_y_initial = np.min(initial_pos[:, 1])
ground_margin = 0.02
y_offset = min_y_initial - ground_margin
if y_offset > 0:
    print(f"  Lowering body by {y_offset:.3f} to touch ground...")
    initial_pos[:, 1] -= y_offset
    new_pos_wp = wp.array(initial_pos, dtype=wp.vec2, device=env.state_in.particle_q.device)
    env.state_in.particle_q = new_pos_wp
    env.state_out.particle_q = wp.clone(new_pos_wp)
    # Also update model (for consistency)
    env.model.particle_q = wp.clone(new_pos_wp)

# Get model info
n_spring_strains = env.model.spring_count
n_fem_strains = env.model.tri_count if hasattr(env.model, 'tri_count') else 0
n_total_strains = n_spring_strains + n_fem_strains

print(f"Environment created:")
print(f"  Particles: {COLS}x{ROWS} = {ROWS*COLS}")
print(f"  Groups: {num_groups}")
print(f"  Springs: {n_spring_strains}")
print(f"  FEM triangles: {n_fem_strains}")
print(f"  Total strains: {n_total_strains}")

# ============================================================================
# Create Force Injector
# ============================================================================

print("\nCreating Force Injector...")
injector = BalloonForces(
    env.model, 
    group_size=2, 
    device=DEVICE,
    force_scale=FORCE_SCALE,
)

initial_positions = env.model.particle_q.numpy()
initial_centroid_x = np.mean(initial_positions[:, 0])
injector.calculate_centroids(initial_positions)

print(f"  Mode: horizontal (like classic demo)")
print(f"  Scale: {FORCE_SCALE}")

# ============================================================================
# Create Shared Renderer
# ============================================================================

renderer = Renderer(
    window_width=WINDOW_WIDTH,
    window_height=WINDOW_HEIGHT,
    boxsize=BOXSIZE,
)

# ============================================================================
# Global State
# ============================================================================

current_strains = np.zeros(n_total_strains, dtype=np.float32)
cpg_outputs = np.zeros(num_groups, dtype=np.float32)
simulation_time = 0.0
cpg_enabled = True
current_frequency = FREQUENCY
current_amplitude = AMPLITUDE

# ============================================================================
# Physics Step Function
# ============================================================================

last_physics_time = -DT

def physics_step(t):
    """
    Execute one physics step with CPG forces.
    Returns strain data for Nengo visualization.
    """
    global current_strains, simulation_time, last_physics_time
    
    simulation_time = t
    
    # Only step physics at DT intervals
    if t - last_physics_time < DT - NENGO_DT * 0.5:
        return current_strains
    
    last_physics_time = t
    
    # Get current positions
    current_pos = env.state_in.particle_q.numpy()
    injector.calculate_centroids(current_pos)
    
    # Apply CPG forces (only after initialization period)
    # During first 0.3s, oscillators are being kick-started onto limit cycle
    INIT_PERIOD = 0.3
    injector.reset()
    if cpg_enabled and t > INIT_PERIOD:
        for group_id in range(num_groups):
            cpg_val = cpg_outputs[group_id] * current_amplitude
            if abs(cpg_val) > 0.001:
                injector.inject(group_id, cpg_val)
    
    # Get forces and step physics
    forces = injector.get_array()
    action = forces.flatten().astype(np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Debug: print CPG matrix every second
    if int(t * 10) % 10 == 0 and t > 0.1:
        max_force = np.max(np.abs(forces))
        total_fx_actual = np.sum(forces[:, 0])
        total_fy_actual = np.sum(forces[:, 1])
        print(f"\n[SNN CPG] t={t:.2f}s")
        print(f"  Forces: max={max_force:.4f}, sum_fx={total_fx_actual:.4f}, sum_fy={total_fy_actual:.4f}")
        # Print CPG matrix
        grid_side = num_groups_cols  # For rendering
        print(f"  CPG Matrix ({grid_side}x{grid_side}):")
        for row in range(grid_side - 1, -1, -1):  # Top to bottom
            row_vals = []
            for col in range(grid_side):
                gid = row * grid_side + col
                row_vals.append(f"{cpg_outputs[gid]:+.2f}")
            print(f"    [{' '.join(row_vals)}]")
    
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
# Render Function (using shared pygame_renderer)
# ============================================================================

def render_frame():
    """Render the current state with CPG visualization using shared Renderer."""
    if env.render_mode != 'human':
        return
    
    wp.synchronize()
    env._sync_to_cpu()
    
    env._init_window()
    if env.window is None:
        return
    
    # Draw base scene from env
    scale = env.window_height / BOXSIZE
    canvas = env._create_canvas()
    env._draw_grid(canvas)
    env._draw_fem_triangles(canvas, scale)
    env._draw_springs(canvas, scale)
    env._draw_particles(canvas, scale)
    env._draw_centroids(canvas, scale)
    env._draw_ui_text(canvas)
    env._draw_legends(canvas)
    
    # Draw CPG overlays using shared Renderer
    if injector.centroids is not None:
        # Draw group centroids with labels (hot pink)
        renderer.draw_group_centroids(canvas, injector.centroids)
        
        # Draw radial force arrows for each group (balloon inflate/deflate)
        for gid in range(num_groups):
            if gid < len(injector.centroids):
                renderer.draw_radial_force_arrows(
                    canvas,
                    injector.centroids[gid],
                    cpg_outputs[gid],
                    num_directions=4,
                )
        
        # Draw CPG matrix display with direction indicator (bottom-right)
        renderer.draw_group_forces_matrix(
            canvas,
            cpg_outputs,
            num_groups_cols,
            title="CPG:",
            direction=DIRECTION,
        )
    
    # Status overlay (bottom-left) - SNN specific info
    status_color = (0, 150, 0) if cpg_enabled else (200, 50, 50)
    status_lines = [
        (f"SNN CPG: {current_frequency:.1f}Hz | {'ON' if cpg_enabled else 'OFF'}", status_color),
    ]
    
    # Displacement info
    new_pos = env.state_in.particle_q.numpy()
    current_x = np.mean(new_pos[:, 0])
    displacement = current_x - initial_centroid_x
    status_lines.append((f"Displacement: {displacement:+.4f}m", renderer.BLACK))
    
    renderer.draw_info_text(
        canvas,
        status_lines,
        position=(10, env.window_height - 50),
        line_spacing=20,
    )
    
    # Blit and flip
    env.window.blit(canvas, canvas.get_rect())
    pygame.event.pump()
    pygame.display.flip()
    
    # Process events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            env.close()
            raise SystemExit

# ============================================================================
# Build Nengo Network
# ============================================================================

print(f"\n{'='*70}")
print("Building Nengo Network...")
print(f"{'='*70}")

model = nengo.Network(label=f"SNN CPG ({COLS}x{ROWS})")

with model:
    
    # =========================================================================
    # CPG Control Widgets (sliders in Nengo GUI)
    # =========================================================================
    
    # CPG enable/disable toggle [0=OFF, 1=ON]
    cpg_toggle = nengo.Node([1.0], label="CPG [0=OFF, 1=ON]")
    
    # Frequency slider [1-10 Hz]
    freq_slider = nengo.Node([FREQUENCY], label="Frequency [1-10 Hz]")
    
    # Amplitude slider [0-1]
    amp_slider = nengo.Node([AMPLITUDE], label="Amplitude [0-1]")
    
    # Force scale slider [1-50]
    force_slider = nengo.Node([FORCE_SCALE], label="Force Scale [1-50]")
    
    # PES learning rate slider [0-0.001] (0 = disabled by default)
    pes_slider = nengo.Node([float(os.environ.get('SNN_PES_LR', '0'))], 
                           label="PES LR [0-0.001]")
    
    # =========================================================================
    # Strain Input from Simulation
    # =========================================================================
    
    current_pes_lr = float(os.environ.get('SNN_PES_LR', '1e-4'))
    
    def update_simulation_with_control(t, x):
        """
        Main update function called by Nengo.
        x[0] = CPG enabled (0=OFF, 1=ON)
        x[1] = frequency (Hz)
        x[2] = amplitude (0-1)
        x[3] = force scale
        x[4] = PES learning rate
        """
        global cpg_enabled, current_frequency, current_amplitude, OMEGA
        global FORCE_SCALE, current_pes_lr
        
        cpg_enabled = x[0] > 0.5
        
        # Update frequency (with bounds)
        new_freq = np.clip(x[1], 1.0, 10.0)
        if abs(new_freq - current_frequency) > 0.01:
            current_frequency = new_freq
            OMEGA = 2 * np.pi * current_frequency
        
        # Update PES learning rate (stored for display, actual rate is fixed at build time)
        current_pes_lr = np.clip(x[4], 0.0, 0.001)
        
        # Update amplitude
        current_amplitude = np.clip(x[2], 0.0, 1.0)
        
        # Update force scale
        new_scale = np.clip(x[3], 1.0, 50.0)
        if abs(new_scale - FORCE_SCALE) > 0.1:
            FORCE_SCALE = new_scale
            injector.force_scale = FORCE_SCALE
        
        # Physics step
        strains = physics_step(t)
        
        # Render
        render_frame()
        
        return strains
    
    strain_input = nengo.Node(
        update_simulation_with_control,
        size_in=5,  # Receives all slider values including PES
        size_out=n_total_strains,
        label="Strain_Input"
    )
    
    # Connect sliders to strain_input
    nengo.Connection(cpg_toggle, strain_input[0], synapse=None)
    nengo.Connection(freq_slider, strain_input[1], synapse=None)
    nengo.Connection(amp_slider, strain_input[2], synapse=None)
    nengo.Connection(force_slider, strain_input[3], synapse=None)
    nengo.Connection(pes_slider, strain_input[4], synapse=None)
    
    # =========================================================================
    # 7D Strain Ensembles (per group: 5 springs + 2 FEMs)
    # =========================================================================
    
    strain_ensembles = []  # Will hold 7D strain ensembles
    
    # Assign strains to groups
    def particle_to_grid(idx, cols):
        return (idx // cols, idx % cols)
    
    def get_particle_groups(idx, rows, cols):
        row, col = particle_to_grid(idx, cols)
        groups = []
        if row < rows-1 and col < cols-1:
            groups.append((row, col))
        if row < rows-1 and col > 0:
            groups.append((row, col-1))
        if row > 0 and col < cols-1:
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
        groups0 = set(get_particle_groups(p0, ROWS, COLS))
        groups1 = set(get_particle_groups(p1, ROWS, COLS))
        common = groups0 & groups1
        for (gr, gc) in common:
            gid = gr * num_groups_cols + gc
            if gid < num_groups:
                spring_groups[gid].append(spring_idx // 2)
    
    # Assign FEMs to groups
    fem_groups = {g: [] for g in range(num_groups)}
    for fem_idx in range(0, len(tri_indices_np), 3):
        if fem_idx + 2 >= len(tri_indices_np):
            break
        p0, p1, p2 = tri_indices_np[fem_idx:fem_idx+3]
        groups0 = set(get_particle_groups(p0, ROWS, COLS))
        groups1 = set(get_particle_groups(p1, ROWS, COLS))
        groups2 = set(get_particle_groups(p2, ROWS, COLS))
        common = groups0 & groups1 & groups2
        for (gr, gc) in common:
            gid = gr * num_groups_cols + gc
            if gid < num_groups:
                fem_groups[gid].append(fem_idx // 3)
    
    # Create 7D strain ensembles
    N_SPRINGS_VIZ = 5
    N_FEMS_VIZ = 2
    STRAIN_DIM = N_SPRINGS_VIZ + N_FEMS_VIZ  # 7D
    
    for group_id in range(num_groups):
        group_springs = spring_groups.get(group_id, [])
        group_fems = fem_groups.get(group_id, [])
        
        group_row = group_id // num_groups_cols
        group_col = group_id % num_groups_cols
        
        # 7D strain ensemble with triangular intercepts (Zaidel et al. 2021)
        n_ens_neurons = N_NEURONS * STRAIN_DIM
        d = STRAIN_DIM
        spread = 0.2
        mode = 0.5
        triangular_samples = np.random.triangular(mode - spread, mode, mode + spread, n_ens_neurons)
        triangular_intercepts = nengo.dists.CosineSimilarity(d + 2).ppf(1 - triangular_samples)
        
        strain_ens = nengo.Ensemble(
            n_neurons=n_ens_neurons,
            dimensions=STRAIN_DIM,
            max_rates=nengo.dists.Uniform(100, 200),
            intercepts=triangular_intercepts,
            neuron_type=nengo.LIF(),
            radius=1.0,
            label=f"G{group_id}[{group_row},{group_col}]_Strain"
        )
        
        # Connect springs to dimensions 0-4
        for i, spring_idx in enumerate(group_springs[:N_SPRINGS_VIZ]):
            transform = np.zeros((STRAIN_DIM, 1))
            transform[i, 0] = 1.0
            nengo.Connection(strain_input[spring_idx], strain_ens,
                           transform=transform, synapse=0.01)
        
        # Connect FEMs to dimensions 5-6
        for i, fem_idx in enumerate(group_fems[:N_FEMS_VIZ]):
            strain_idx = n_spring_strains + fem_idx
            transform = np.zeros((STRAIN_DIM, 1))
            transform[N_SPRINGS_VIZ + i, 0] = 1.0
            nengo.Connection(strain_input[strain_idx], strain_ens,
                           transform=transform, synapse=0.01)
        
        strain_ensembles.append(strain_ens)
        
    print(f"  Created {len(strain_ensembles)} strain ensembles (7D each)")
    
    # =========================================================================
    # CPG Oscillators - Hopf + Kuramoto in NEF
    # 
    # IDENTICAL to cpg.py equations:
    #   ṙ = a(μ - r²)r              # Amplitude dynamics
    #   θ̇ = ω + Σⱼ K·sin(θⱼ - θᵢ - φᵢⱼ)  # Phase with Kuramoto coupling
    #   output = r·cos(θ)
    #
    # In Cartesian (x = r·cos(θ), y = r·sin(θ)):
    #   ẋ = a(μ - r²)x - ωy + coupling_x
    #   ẏ = a(μ - r²)y + ωx + coupling_y
    #
    # Kuramoto coupling in Cartesian:
    #   sin(θⱼ - θᵢ - φ) = xᵢ·y_rot - yᵢ·x_rot  (cross product)
    #   where (x_rot, y_rot) = rotate(xⱼ, yⱼ, -φ)
    #   coupling = K · sin_error · tangent = K · sin_error · [-yᵢ, xᵢ]
    # =========================================================================
    
    # Parameters matching cpg.py exactly
    TAU_SYN = 0.01      # Synaptic time constant (10ms) 
    HOPF_A = 15.0       # Same as cpg.py
    HOPF_MU = 1.0       # Same as cpg.py
    COUPLING_K = float(os.environ.get('SNN_COUPLING', '2.0'))  # Same as cpg.py!
    
    phase_per_cell = np.pi / 2.0  # Same as cpg.py
    
    # Compute phase offsets (same as cpg.py._init_phases)
    phase_offsets = np.zeros(num_groups)
    for gid in range(num_groups):
        row = gid // num_groups_cols
        col = gid % num_groups_cols
        phase_offsets[gid] = (col * DIRECTION[0] + row * DIRECTION[1]) * phase_per_cell
    
    # Build neighbor structure (same as cpg.py._build_neighbors)
    neighbors = [[] for _ in range(num_groups)]
    target_phase_diff = {}
    for gid in range(num_groups):
        row = gid // num_groups_cols
        col = gid % num_groups_cols
        
        # 4-connected neighbors
        neighbor_offsets = []
        if col > 0:
            neighbor_offsets.append((gid - 1, -1, 0))           # left
        if col < num_groups_cols - 1:
            neighbor_offsets.append((gid + 1, 1, 0))            # right
        if row > 0:
            neighbor_offsets.append((gid - num_groups_cols, 0, -1))  # down
        if row < num_groups_rows - 1:
            neighbor_offsets.append((gid + num_groups_cols, 0, 1))   # up
        
        for (jid, col_diff, row_diff) in neighbor_offsets:
            neighbors[gid].append(jid)
            # Target phase: j should lead i by this amount
            target_phase_diff[(gid, jid)] = phase_per_cell * (
                col_diff * DIRECTION[0] + row_diff * DIRECTION[1]
            )
    
    print(f"  Hopf parameters: a={HOPF_A}, μ={HOPF_MU}, ω={OMEGA:.2f} rad/s")
    print(f"  Kuramoto coupling: K={COUPLING_K}, phase_per_cell={np.degrees(phase_per_cell):.0f}°")
    
    # =========================================================================
    # Create CPG Ensembles (2D Hopf oscillators)
    # =========================================================================
    
    cpg_ensembles = []
    
    for group_id in range(num_groups):
        group_row = group_id // num_groups_cols
        group_col = group_id % num_groups_cols
        initial_phase = phase_offsets[group_id]
        
        cpg_ens = nengo.Ensemble(
            n_neurons=N_NEURONS * 2,
            dimensions=2,
            max_rates=nengo.dists.Uniform(100, 200),
            intercepts=nengo.dists.Uniform(-0.3, 0.3),
            neuron_type=nengo.LIF(),
            radius=1.2,
            label=f"G{group_id}[{group_row},{group_col}]_CPG"
        )
        
        # Initialize on limit cycle at correct phase (same as cpg.py)
        x0 = np.cos(initial_phase)
        y0 = np.sin(initial_phase)
        
        def make_init(x_init, y_init):
            def init_fn(t):
                if t < 0.3:
                    return [x_init * 3, y_init * 3]  # Kick onto limit cycle
                return [0, 0]
            return init_fn
        
        init_node = nengo.Node(make_init(x0, y0), size_out=2, label=f"G{group_id}_init")
        nengo.Connection(init_node, cpg_ens, synapse=0.01)
        
        # Hopf dynamics (same equations as cpg.py)
        def make_hopf_fn(omega, tau, a, mu):
            def hopf_fn(x):
                r_sq = x[0]**2 + x[1]**2
                # ẋ = a(μ - r²)x - ωy
                # ẏ = a(μ - r²)y + ωx
                dx = a * (mu - r_sq) * x[0] - omega * x[1]
                dy = a * (mu - r_sq) * x[1] + omega * x[0]
                # NEF integration: return x + tau * ẋ
                return [x[0] + tau * dx, x[1] + tau * dy]
            return hopf_fn
        
        hopf_fn = make_hopf_fn(OMEGA, TAU_SYN, HOPF_A, HOPF_MU)
        nengo.Connection(cpg_ens, cpg_ens, function=hopf_fn, synapse=TAU_SYN)
        
        cpg_ensembles.append(cpg_ens)
    
    print(f"  Created {len(cpg_ensembles)} CPG oscillators (2D Hopf)")
    
    # =========================================================================
    # Kuramoto Coupling - PROPER NEF implementation
    #
    # For each pair (i, j), we need to compute:
    #   sin(θⱼ - θᵢ - φᵢⱼ) = xᵢ·y_rot - yᵢ·x_rot
    #   coupling_i = K · sin_error · [-yᵢ, xᵢ] / r  (tangential)
    #
    # This requires a 4D ensemble to see both [xᵢ, yᵢ] and [xⱼ_rot, yⱼ_rot]
    # =========================================================================
    
    coupling_ensembles = []
    coupling_count = 0
    
    for gid in range(num_groups):
        for jid in neighbors[gid]:
            phi = target_phase_diff[(gid, jid)]
            cos_phi = np.cos(phi)
            sin_phi = np.sin(phi)
            
            # 4D coupling ensemble: [x_i, y_i, x_j_rot, y_j_rot]
            coupling_ens = nengo.Ensemble(
                n_neurons=N_NEURONS * 4,
                dimensions=4,
                max_rates=nengo.dists.Uniform(100, 200),
                intercepts=nengo.dists.Uniform(-0.5, 0.5),
                neuron_type=nengo.LIF(),
                radius=1.5,
                label=f"Coupling_{gid}<-{jid}"
            )
            
            # Connect oscillator i to dimensions [0, 1]
            nengo.Connection(cpg_ensembles[gid], coupling_ens[:2], synapse=TAU_SYN)
            
            # Connect rotated oscillator j to dimensions [2, 3]
            def make_rotate_fn(c, s):
                def rotate_fn(x):
                    # Rotate by -phi: [cos(-φ), sin(-φ)] = [cos(φ), -sin(φ)]
                    return [x[0] * c + x[1] * s, -x[0] * s + x[1] * c]
                return rotate_fn
            
            rotate_fn = make_rotate_fn(cos_phi, sin_phi)
            nengo.Connection(cpg_ensembles[jid], coupling_ens[2:4], 
                           function=rotate_fn, synapse=TAU_SYN)
            
            # Compute Kuramoto coupling: K · sin(θⱼ - θᵢ - φ) · [-yᵢ, xᵢ]
            def make_kuramoto_fn(K, tau):
                def kuramoto_fn(state):
                    x_i, y_i, x_rot, y_rot = state[0], state[1], state[2], state[3]
                    
                    # sin(θⱼ - φ - θᵢ) = xᵢ·y_rot - yᵢ·x_rot (cross product)
                    sin_error = x_i * y_rot - y_i * x_rot
                    
                    # Tangential direction: [-yᵢ, xᵢ] (perpendicular to radial)
                    # This adds to dθ, matching cpg.py: dtheta[i] += K * sin(error)
                    # Scale by tau for NEF integration
                    coupling_x = K * tau * sin_error * (-y_i)
                    coupling_y = K * tau * sin_error * x_i
                    
                    return [coupling_x, coupling_y]
                return kuramoto_fn
            
            kuramoto_fn = make_kuramoto_fn(COUPLING_K, TAU_SYN)
            nengo.Connection(coupling_ens, cpg_ensembles[gid], 
                           function=kuramoto_fn, synapse=TAU_SYN)
            
            coupling_ensembles.append(coupling_ens)
            coupling_count += 1
    
    print(f"  Created {coupling_count} Kuramoto coupling ensembles (4D each)")
    print(f"  Total coupling neurons: {coupling_count * N_NEURONS * 4}")
    
    # =========================================================================
    # PES Learning: strain → force adjustment
    # 
    # Architecture (like trajectory_tracking/stress.py):
    # 
    #   CPG[x-component] ───────────────────────────────┐
    #                                                   ├──> u(t) ──> output
    #   strain[7D] ──> s_ens ──[PES learns from vel]───┘
    # 
    # Error signal: velocity error (target velocity - actual velocity)
    # PES learns: strain → force correction to improve locomotion
    # =========================================================================
    
    PES_LEARNING_RATE = float(os.environ.get('SNN_PES_LR', '0'))  # Disabled by default
    TARGET_VELOCITY = 0.1  # Target horizontal velocity (m/s)
    
    # Velocity error tracking - synchronized with physics timestep
    velocity_error = np.zeros(2)  # [vx_error, vy_error]
    last_centroid_x = initial_centroid_x
    last_velocity_time = 0.0
    smoothed_velocity = 0.0
    VELOCITY_SMOOTHING = 0.9  # Exponential smoothing factor
    
    def compute_velocity_error(t):
        """
        Compute velocity error for PES learning signal.
        
        IMPORTANT: Only update velocity on physics timesteps to avoid
        mismatch between Nengo dt (1ms) and physics dt (10ms).
        """
        global last_centroid_x, last_velocity_time, smoothed_velocity
        
        # Don't compute during initialization period
        if t < 0.5:
            current_pos = env.state_in.particle_q.numpy()
            last_centroid_x = np.mean(current_pos[:, 0])
            last_velocity_time = t
            return np.zeros(2)
        
        # Only update velocity when physics has stepped (every DT seconds)
        time_since_last = t - last_velocity_time
        if time_since_last < DT * 0.9:  # Not a physics step yet
            return velocity_error  # Return cached value
        
        # Physics step occurred - compute new velocity
        current_pos = env.state_in.particle_q.numpy()
        current_x = np.mean(current_pos[:, 0])
        
        # Compute instantaneous velocity
        if time_since_last > 0:
            instant_vx = (current_x - last_centroid_x) / time_since_last
        else:
            instant_vx = 0.0
        
        # Exponential smoothing to reduce noise
        smoothed_velocity = VELOCITY_SMOOTHING * smoothed_velocity + (1 - VELOCITY_SMOOTHING) * instant_vx
        
        last_centroid_x = current_x
        last_velocity_time = t
        
        # Target velocity in direction of movement
        target_vx = TARGET_VELOCITY * DIR_X
        target_vy = TARGET_VELOCITY * DIR_Y
        
        # Error: target - actual (positive error = need more force)
        # Clip to prevent large transients
        velocity_error[0] = np.clip(target_vx - smoothed_velocity, -0.3, 0.3)
        velocity_error[1] = np.clip(target_vy, -0.3, 0.3)
        
        return velocity_error
    
    velocity_error_node = nengo.Node(compute_velocity_error, size_out=2, label="Velocity_Error")
    
    # Output ensembles (u_ens): combine CPG + PES feedforward
    output_ensembles = []
    pes_connections = []
    
    print(f"  Adding PES learning (rate={PES_LEARNING_RATE})...")
    
    for gid in range(num_groups):
        group_row = gid // num_groups_cols
        group_col = gid % num_groups_cols
        
        # Output ensemble: 1D (force magnitude in CPG direction)
        u_ens = nengo.Ensemble(
            n_neurons=N_NEURONS,
            dimensions=1,
            max_rates=nengo.dists.Uniform(100, 200),
            intercepts=nengo.dists.Uniform(-0.5, 0.5),
            neuron_type=nengo.LIF(),
            radius=2.0,  # Allow values up to ±2
            label=f"G{gid}[{group_row},{group_col}]_u(t)"
        )
        
        # CPG → u_ens (baseline oscillation)
        nengo.Connection(cpg_ensembles[gid][0], u_ens, synapse=0.01)
        
        # PES: strain → u_ens (learned feedforward)
        if PES_LEARNING_RATE > 0 and gid < len(strain_ensembles):
            s_ens = strain_ensembles[gid]
            
            # PES connection: strain[7D] → u_ens[1D]
            pes_conn = nengo.Connection(
                s_ens, u_ens,
                transform=np.zeros((1, STRAIN_DIM)),  # Starts at zero, PES learns
                synapse=TAU_SYN,
                learning_rule_type=nengo.PES(learning_rate=PES_LEARNING_RATE),
            )
            
            # Learning signal: velocity error (just x-component)
            # Negative error: if we're too slow, increase force
            nengo.Connection(velocity_error_node[0], pes_conn.learning_rule, 
                           transform=-1, synapse=None)
            
            pes_connections.append(pes_conn)
        
        output_ensembles.append(u_ens)
    
    print(f"  Created {len(output_ensembles)} output ensembles with PES feedforward")
    
    # =========================================================================
    # Output → Force Injection
    # =========================================================================
    
    def make_output_func(gid):
        """Create output function - read from u_ens."""
        def output_func(t, x):
            cpg_outputs[gid] = np.clip(x[0], -1, 1)
        return output_func
    
    for i, u_ens in enumerate(output_ensembles):
        output_node = nengo.Node(make_output_func(i), size_in=1)
        nengo.Connection(u_ens, output_node, synapse=0.01)
    
    # =========================================================================
    # CPG Output Display (for visualization)
    # =========================================================================
    
    cpg_display = nengo.Node(lambda t: cpg_outputs, size_out=num_groups, label="CPG_Outputs")

# ============================================================================
# Create Nengo GUI Configuration File
# ============================================================================

config_filename = os.path.join(os.path.dirname(__file__), 'demo_snn_gui.py.cfg')
print(f"\n📝 Creating Nengo GUI config: {config_filename}")

# Find ensemble indices
strain_ens_indices = []
for strain_ens in strain_ensembles:
    for idx, ens in enumerate(model.all_ensembles):
        if ens is strain_ens:
            strain_ens_indices.append(idx)
            break

cpg_ens_indices = []
for cpg_ens in cpg_ensembles:
    for idx, ens in enumerate(model.all_ensembles):
        if ens is cpg_ens:
            cpg_ens_indices.append(idx)
            break

# Find slider node indices
toggle_node_idx = None
freq_node_idx = None
amp_node_idx = None
force_node_idx = None
pes_node_idx = None

for idx, node in enumerate(model.all_nodes):
    if node is cpg_toggle:
        toggle_node_idx = idx
    if node is freq_slider:
        freq_node_idx = idx
    if node is amp_slider:
        amp_node_idx = idx
    if node is force_slider:
        force_node_idx = idx
    if node is pes_slider:
        pes_node_idx = idx

print(f"Found {len(strain_ens_indices)} strain ensembles (7D)")
print(f"Found {len(cpg_ens_indices)} CPG ensembles (2D)")
print(f"Slider nodes: toggle={toggle_node_idx}, freq={freq_node_idx}, amp={amp_node_idx}, force={force_node_idx}, pes={pes_node_idx}")

# Generate config file (layout based on trajectory_tracking/snn_nengo_tracking_gui.py)
with open(config_filename, 'w') as f:
    f.write("# Nengo GUI Configuration - SNN CPG Locomotion\n")
    f.write("# Auto-generated - do not edit manually\n\n")
    f.write("_viz_net_graph = nengo_gui.components.NetGraph()\n\n")
    
    # Layout dimensions (matching trajectory_tracking style)
    plot_size = 0.10       # Readable plot size
    plot_gap = 0.01        # Gap between plots
    margin_left = 0.0
    margin_top = 0.02
    pair_gap = 0.02        # Gap between strain/CPG pairs
    
    # Each group: XY value + raster stacked vertically
    pair_height = plot_size + plot_gap + plot_size
    
    num_plots = min(len(strain_ens_indices), len(cpg_ens_indices))
    
    print(f"Creating config for {num_plots} groups (trajectory_tracking layout)...")
    
    for i in range(num_plots):
        group_row = i // num_groups_cols
        group_col = i % num_groups_cols
        
        # Vertical stacking: each group gets XY + Raster
        y_base = margin_top + i * (pair_height + plot_gap)
        y_xy = y_base
        y_raster = y_xy + plot_size + plot_gap
        
        # Two columns: Strain (left), CPG (right)
        strain_x = margin_left
        cpg_x = margin_left + plot_size + pair_gap
        
        strain_ens_idx = strain_ens_indices[i]
        cpg_ens_idx = cpg_ens_indices[i]
        
        # Strain Value (7D: spring0-4, fem0-1)
        f.write(f"# G{i}[{group_row},{group_col}] Strain\n")
        f.write(f"_viz_strain_value_{i} = nengo_gui.components.Value(model.all_ensembles[{strain_ens_idx}])\n")
        f.write(f"_viz_config[_viz_strain_value_{i}].x = {strain_x:.4f}\n")
        f.write(f"_viz_config[_viz_strain_value_{i}].y = {y_xy:.4f}\n")
        f.write(f"_viz_config[_viz_strain_value_{i}].width = {plot_size:.4f}\n")
        f.write(f"_viz_config[_viz_strain_value_{i}].height = {plot_size:.4f}\n")
        f.write(f"_viz_config[_viz_strain_value_{i}].max_value = 1.0\n")
        f.write(f"_viz_config[_viz_strain_value_{i}].min_value = -1.0\n")
        f.write(f"_viz_config[_viz_strain_value_{i}].show_legend = True\n")
        f.write(f"_viz_config[_viz_strain_value_{i}].label_visible = True\n\n")
        
        # Strain Raster
        f.write(f"_viz_strain_raster_{i} = nengo_gui.components.Raster(model.all_ensembles[{strain_ens_idx}].neurons)\n")
        f.write(f"_viz_config[_viz_strain_raster_{i}].x = {strain_x:.4f}\n")
        f.write(f"_viz_config[_viz_strain_raster_{i}].y = {y_raster:.4f}\n")
        f.write(f"_viz_config[_viz_strain_raster_{i}].width = {plot_size:.4f}\n")
        f.write(f"_viz_config[_viz_strain_raster_{i}].height = {plot_size:.4f}\n")
        f.write(f"_viz_config[_viz_strain_raster_{i}].label_visible = False\n\n")
        
        # CPG XY-Value (2D: shows limit cycle)
        f.write(f"# G{i}[{group_row},{group_col}] CPG\n")
        f.write(f"_viz_cpg_xy_{i} = nengo_gui.components.XYValue(model.all_ensembles[{cpg_ens_idx}])\n")
        f.write(f"_viz_config[_viz_cpg_xy_{i}].x = {cpg_x:.4f}\n")
        f.write(f"_viz_config[_viz_cpg_xy_{i}].y = {y_xy:.4f}\n")
        f.write(f"_viz_config[_viz_cpg_xy_{i}].width = {plot_size:.4f}\n")
        f.write(f"_viz_config[_viz_cpg_xy_{i}].height = {plot_size:.4f}\n")
        f.write(f"_viz_config[_viz_cpg_xy_{i}].max_value = 1.2\n")
        f.write(f"_viz_config[_viz_cpg_xy_{i}].min_value = -1.2\n")
        f.write(f"_viz_config[_viz_cpg_xy_{i}].label_visible = True\n\n")
        
        # CPG Raster
        f.write(f"_viz_cpg_raster_{i} = nengo_gui.components.Raster(model.all_ensembles[{cpg_ens_idx}].neurons)\n")
        f.write(f"_viz_config[_viz_cpg_raster_{i}].x = {cpg_x:.4f}\n")
        f.write(f"_viz_config[_viz_cpg_raster_{i}].y = {y_raster:.4f}\n")
        f.write(f"_viz_config[_viz_cpg_raster_{i}].width = {plot_size:.4f}\n")
        f.write(f"_viz_config[_viz_cpg_raster_{i}].height = {plot_size:.4f}\n")
        f.write(f"_viz_config[_viz_cpg_raster_{i}].label_visible = False\n\n")
        
        print(f"  G{i}[{group_row},{group_col}]: Strain (7D) + CPG (XY)")
    
    # Network graph (right side)
    plots_end_x = cpg_x + plot_size if num_plots > 0 else 0.25
    network_start_x = plots_end_x + 0.05
    network_width = 1.0 - network_start_x - 0.01
    network_center_x = network_start_x + network_width / 2
    
    f.write(f"# Network graph\n")
    f.write(f"_viz_config[model].pos = ({network_center_x:.3f}, 0.50)\n")
    f.write(f"_viz_config[model].size = ({network_width:.3f}, 0.96)\n")
    f.write(f"_viz_config[model].expanded = False\n")
    f.write(f"_viz_config[model].has_layout = False\n\n")
    
    f.write("_viz_sim_control = nengo_gui.components.SimControl()\n")
    f.write("_viz_config[_viz_sim_control].shown_time = 1.0\n")
    f.write("_viz_config[_viz_sim_control].kept_time = 4.0\n\n")
    
    # CPG Control Sliders (top-right area)
    slider_x = 0.25
    slider_y = 0.02
    slider_width = 0.15
    slider_height = 0.08
    
    f.write("# CPG Control Sliders\n")
    
    if toggle_node_idx is not None:
        f.write(f"_viz_toggle = nengo_gui.components.Slider(model.all_nodes[{toggle_node_idx}])\n")
        f.write(f"_viz_config[_viz_toggle].x = {slider_x:.4f}\n")
        f.write(f"_viz_config[_viz_toggle].y = {slider_y:.4f}\n")
        f.write(f"_viz_config[_viz_toggle].width = {slider_width:.4f}\n")
        f.write(f"_viz_config[_viz_toggle].height = {slider_height:.4f}\n")
        f.write(f"_viz_config[_viz_toggle].min_value = 0\n")
        f.write(f"_viz_config[_viz_toggle].max_value = 1\n\n")
        print(f"  CPG toggle slider added")
    
    slider_y2 = slider_y + slider_height + 0.02
    if freq_node_idx is not None:
        f.write(f"_viz_freq = nengo_gui.components.Slider(model.all_nodes[{freq_node_idx}])\n")
        f.write(f"_viz_config[_viz_freq].x = {slider_x:.4f}\n")
        f.write(f"_viz_config[_viz_freq].y = {slider_y2:.4f}\n")
        f.write(f"_viz_config[_viz_freq].width = {slider_width:.4f}\n")
        f.write(f"_viz_config[_viz_freq].height = {slider_height:.4f}\n")
        f.write(f"_viz_config[_viz_freq].min_value = 1\n")
        f.write(f"_viz_config[_viz_freq].max_value = 10\n\n")
        print(f"  Frequency slider added [1-10 Hz]")
    
    slider_y3 = slider_y2 + slider_height + 0.02
    if amp_node_idx is not None:
        f.write(f"_viz_amp = nengo_gui.components.Slider(model.all_nodes[{amp_node_idx}])\n")
        f.write(f"_viz_config[_viz_amp].x = {slider_x:.4f}\n")
        f.write(f"_viz_config[_viz_amp].y = {slider_y3:.4f}\n")
        f.write(f"_viz_config[_viz_amp].width = {slider_width:.4f}\n")
        f.write(f"_viz_config[_viz_amp].height = {slider_height:.4f}\n")
        f.write(f"_viz_config[_viz_amp].min_value = 0\n")
        f.write(f"_viz_config[_viz_amp].max_value = 1\n\n")
        print(f"  Amplitude slider added [0-1]")
    
    slider_y4 = slider_y3 + slider_height + 0.02
    if force_node_idx is not None:
        f.write(f"_viz_force = nengo_gui.components.Slider(model.all_nodes[{force_node_idx}])\n")
        f.write(f"_viz_config[_viz_force].x = {slider_x:.4f}\n")
        f.write(f"_viz_config[_viz_force].y = {slider_y4:.4f}\n")
        f.write(f"_viz_config[_viz_force].width = {slider_width:.4f}\n")
        f.write(f"_viz_config[_viz_force].height = {slider_height:.4f}\n")
        f.write(f"_viz_config[_viz_force].min_value = 1\n")
        f.write(f"_viz_config[_viz_force].max_value = 50\n\n")
        print(f"  Force scale slider added [1-50]")
    
    slider_y5 = slider_y4 + slider_height + 0.02
    if pes_node_idx is not None:
        f.write(f"_viz_pes = nengo_gui.components.Slider(model.all_nodes[{pes_node_idx}])\n")
        f.write(f"_viz_config[_viz_pes].x = {slider_x:.4f}\n")
        f.write(f"_viz_config[_viz_pes].y = {slider_y5:.4f}\n")
        f.write(f"_viz_config[_viz_pes].width = {slider_width:.4f}\n")
        f.write(f"_viz_config[_viz_pes].height = {slider_height:.4f}\n")
        f.write(f"_viz_config[_viz_pes].min_value = 0\n")
        f.write(f"_viz_config[_viz_pes].max_value = 0.001\n\n")
        print(f"  PES learning rate slider added [0-0.001]")

print(f"  ✓ Config file created")

# ============================================================================
# Summary
# ============================================================================

print(f"\n{'='*70}")
print("SNN CPG Network Created with PES Learning")
print(f"{'='*70}")
print(f"Strain ensembles: {len(strain_ensembles)} (7D: 5 springs + 2 FEMs)")
print(f"CPG oscillators: {len(cpg_ensembles)} (2D Hopf)")
print(f"Output ensembles: {len(output_ensembles)} (1D with PES feedforward)")
print(f"PES connections: {len(pes_connections)}")
print(f"Total neurons: ~{num_groups * (N_NEURONS * 7 + N_NEURONS * 2 + N_NEURONS)}")
print()
print(f"🎛️  NENGO GUI CONTROLS:")
print(f"  • CPG [0=OFF, 1=ON]: Toggle CPG on/off")
print(f"  • Frequency [1-10 Hz]: CPG oscillation frequency")
print(f"  • Amplitude [0-1]: Force amplitude")
print(f"  • Force Scale [1-50]: Force multiplier")
print(f"  • PES LR [0-0.001]: PES learning rate (strain→force)")
print()
print(f"📊 VISUALIZATION (per group):")
print(f"  • G*_Strain: 7D strain values + spike rasters")
print(f"  • G*_CPG: 2D XY phase plot + spike rasters")
print(f"  • CPG_Outputs: All CPG output values")
print()
print(f"🧠 PES LEARNING:")
print(f"  • strain[7D] → u(t)[1D] feedforward")
print(f"  • Error signal: velocity error (target - actual)")
print(f"  • PES learns to adjust forces based on strain feedback")
print(f"{'='*70}")
