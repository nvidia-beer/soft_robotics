"""
Spring FEM Grid Environment for Trajectory Tracking

RIGID GRID TRACKING: The entire grid of group centroids moves as a rigid body.
Each group centroid tracks its own target = initial_position + trajectory_offset.
Control is at the GROUP CENTROID level (not individual particles).

Requires grid size N >= 2 (2x2, 3x3, 4x4, etc.)

Uses the same rendering as openai-gym/rate_coding with implicit FEM solver.

Example: 4x4 grid with 9 overlapping 2x2 groups (3x3 arrangement):

    0──1──2──3         Groups (3x3):
    │╲/│╲/│╲/│         [0][1][2]  <- Each centroid has its own target!
    4──5──6──7         [3][4][5]  <- Group 4 is CENTER (trajectory reference)
    │╲/│╲/│╲/│         [6][7][8]
    8──9─10─11
    │╲/│╲/│╲/│
   12─13─14─15

RIGID GRID MODE:
- ALL group centroids track their own targets
- Targets = initial_centroid + trajectory_offset (same offset for all)
- Control: Force per GROUP CENTROID [fx0,fy0,fx1,fy1,...] (2*num_groups values)
- Forces distributed to particles within each group

This creates a complex control problem:
- FEM physics will deform the grid
- Controller must coordinate ALL groups to maintain rigidity
- SNN must learn model-world mismatch to improve control
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'warp'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'openai-gym'))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import warp as wp
import pygame
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from collections import deque

from spring_mass_env import SpringMassEnv
from sim import Model
from solvers import SolverImplicit
from pygame_renderer import Renderer


class TrackingEnv(SpringMassEnv):
    """
    Spring FEM Grid for RIGID GRID Trajectory Tracking.
    
    ALL group centroids track their own targets:
    - Each target = initial_centroid[group] + trajectory_offset + rotation
    - Control at GROUP CENTROID level (force per centroid, not per particle)
    
    Requires grid size N >= 2 (2x2, 3x3, 4x4, etc.)
    
    Extends SpringMassEnv to add trajectory tracking functionality.
    Uses overlapping 2x2 groups, each containing:
        - 4 particles
        - 2 FEM triangles
        - 5 springs
    
    Example: 4x4 grid with 9 groups (3x3 arrangement):
        0──1──2──3     Groups:     ALL centroids track targets!
        │╲/│╲/│╲/│     [0][1][2]   Target[i] = init[i] + offset
        4──5──6──7     [3][4][5]   Grid moves as rigid body
        │╲/│╲/│╲/│     [6][7][8]
        8──9─10─11
        │╲/│╲/│╲/│
       12─13─14─15
    
    State: positions and velocities of all particles
    Action: forces per GROUP CENTROID [fx0,fy0,fx1,fy1,...] (2 * num_groups values)
           Forces are distributed equally to particles within each group.
    Reward: negative total tracking error (sum of all centroid errors)
    
    The grid moves as a rigid body. FEM physics will try to deform it,
    but the controller must maintain rigidity. This tests SNN learning.
    """
    
    def __init__(
        self,
        render_mode='human',
        rows=3,  # Grid rows
        cols=3,  # Grid cols
        dt=0.01,
        spring_stiffness=40.0,
        spring_damping=0.5,
        device='cuda',
        trajectory_type='sinusoidal',
        trajectory_amplitude=0.5,  # Larger amplitude for visible motion
        trajectory_frequency=0.3,  # Slower for easier tracking
        use_fem=True,
        boxsize=2.5,
        window_width=800,
        window_height=800,
    ):
        """
        Initialize the tracking environment.
        
        Args:
            render_mode: 'human', 'rgb_array', or None
            N: Grid size (NxN)
            dt: Physics timestep
            spring_stiffness: Spring stiffness coefficient
            spring_damping: Spring damping coefficient
            device: 'cuda' or 'cpu'
            trajectory_type: 'sinusoidal', 'circular', 'figure8'
            trajectory_amplitude: Amplitude of trajectory motion
            trajectory_frequency: Frequency of trajectory (Hz)
            use_fem: Whether to use FEM triangles
            boxsize: Simulation box size
            window_width: Render window width
            window_height: Render window height
        """
        # Reduce spring stiffness if using FEM (they work together)
        spring_coeff = spring_stiffness * 0.25 if use_fem else spring_stiffness
        
        # Initialize parent class
        super().__init__(
            render_mode=render_mode,
            rows=rows, cols=cols,
            dt=dt,
            spring_coeff=spring_coeff,
            spring_damping=spring_damping,
            gravity=0.0,  # No gravity for trajectory tracking
            boxsize=boxsize,
            device=device,
            with_fem=use_fem,
            with_springs=True,
            window_width=window_width,
            window_height=window_height,
        )
        
        # Replace with implicit solver for FEM
        if use_fem:
            print("  Using Implicit FEM solver...")
            # Set FEM material properties
            if hasattr(self.model, 'tri_materials') and self.model.tri_materials is not None:
                E = 50.0  # Young's modulus
                nu = 0.3  # Poisson ratio
                k_damp = 2.0  # FEM damping
                k_mu = E / (2.0 * (1.0 + nu))
                k_lambda = E * nu / (1.0 - nu * nu)
                self.model.tri_materials.fill_(wp.vec3(k_mu, k_lambda, k_damp))
            
            self.solver = SolverImplicit(
                self.model,
                dt=dt,
                mass=1.0,
                preconditioner_type="diag",
                solver_type="bicgstab",
                max_iterations=30,
                tolerance=1e-3
            )
        
        # Trajectory parameters
        self.trajectory_type = trajectory_type
        self.trajectory_amplitude = trajectory_amplitude
        self.trajectory_frequency = trajectory_frequency
        
        # Rotation parameters for grid (radians per second)
        # Negative = clockwise rotation
        self.rotation_speed = -trajectory_frequency * 0.5 * np.pi  # Half turn per cycle, clockwise
        self.enable_rotation = True  # Enable grid rotation
        
        # Identify particle groups
        self._setup_particle_groups()
        
        # Store initial positions for trajectory reference
        self.initial_positions = None
        self.time = 0.0
        
        # Initial group positions (for computing grid targets)
        self.initial_group_centroids = None
        
        # Track current control forces for visualization
        # Action: [fx_g0, fy_g0, fx_g1, fy_g1, ...] for each group centroid
        self.action_dim = self.num_groups * 2
        self.current_forces = np.zeros(self.action_dim, dtype=np.float32)
        
        # Per-group forces for visualization
        self.group_forces = np.zeros((self.num_groups, 2), dtype=np.float32)
        
        # Wind force (unknown to model, for visualization)
        self.wind_force = np.array([0.0, 0.0])
        
        # Force visualization constants
        self.FORCE_ARROW_SCALE = 10.0  # Pixels per unit force
        self.FORCE_ARROW_WIDTH = 3
        self.FORCE_ARROW_HEAD_SIZE = 8
        
        # History tracking for plots
        self.time_history = deque()
        self.centroid_x_history = deque()
        self.centroid_y_history = deque()
        self.target_x_history = deque()
        self.target_y_history = deque()
        
        # Per-group error history (list of deques, one per group)
        self.group_error_history = None  # Will be initialized in reset()
        
        # Colors for each group (distinct colors)
        self.group_colors = [
            '#FF6B6B',  # Red
            '#4ECDC4',  # Teal
            '#45B7D1',  # Sky blue
            '#96CEB4',  # Sage green
            '#FFEAA7',  # Yellow
            '#DDA0DD',  # Plum
            '#98D8C8',  # Mint
            '#F7DC6F',  # Gold
            '#BB8FCE',  # Purple
            '#85C1E9',  # Light blue
            '#F8B500',  # Orange
            '#00CED1',  # Dark cyan
        ]
        
        # Matplotlib plotting
        self.fig = None
        self.axes = None
        self.plot_surface = None
        self.max_time = 30.0  # Default max time for plots
        self.plot_update_interval = 2  # Update plots every N frames (lower = more frequent)
        self.frame_count = 0
        
        # Enlarge window to fit plots - LARGER for rotation visualization
        self.plot_width = 700  # Even wider plot panel for per-group errors
        self.sim_width = max(window_width, 800)  # Width of simulation area
        self.window_width = self.sim_width + self.plot_width  # Total width
        self.window_height = max(window_height, 950)  # Taller window for more plots
        
        # Create shared renderer for overlays (uses sim_width, not full window)
        self.renderer = Renderer(
            window_width=self.sim_width,
            window_height=self.window_height,
            boxsize=boxsize,
        )
        
        print(f"✓ TrackingEnv initialized")
        print(f"  Grid: {N}×{N} = {self.model.particle_count} particles")
        print(f"  RIGID GRID: ALL {self.num_groups} group centroids controlled")
        print(f"  Action dim: {self.action_dim} (2 per group = force per centroid)")
        print(f"  Trajectory: {trajectory_type} (A={trajectory_amplitude}, f={trajectory_frequency}Hz)")
        print(f"  FEM: {'Enabled' if use_fem else 'Disabled'}")
    
    def _setup_particle_groups(self):
        r"""
        Setup particle groups as 2x2 overlapping cells.
        
        GRID MODE: All groups receive external forces and track their own targets.
        Requires N >= 2 (any grid size works).
        
        For 4x4 grid (N=4) -> 3x3 = 9 groups:
            Grid:               Groups (3x3):
            0──1──2──3          [0][1][2]
            │╲/│╲/│╲/│          [3][4][5]   <- Group 4 is CENTER
            4──5──6──7          [6][7][8]
            │╲/│╲/│╲/│
            8──9─10─11
            │╲/│╲/│╲/│
           12─13─14─15
        
        Center group (4) particles: [5, 6, 9, 10]
        In GRID MODE: ALL groups receive external forces!
        
        For 6x6 grid (N=6) -> 5x5 = 25 groups:
            Center group (12) is the middle of the 5x5 arrangement.
        """
        # Validate: rows/cols must be >= 2 for at least one group
        if self.rows < 2 or self.cols < 2:
            raise ValueError(f"Grid must be at least 2x2 (got {self.rows}x{self.cols}). "
                           f"Need at least 2x2 particles for 1 group.")
        
        # Number of 2x2 groups = (rows-1) x (cols-1)
        self.groups_rows = self.rows - 1
        self.groups_cols = self.cols - 1
        self.num_groups = self.groups_rows * self.groups_cols
        
        # Build group membership: group_id -> [particle_indices]
        self.group_info = {}
        for group_id in range(self.num_groups):
            group_row = group_id // self.groups_cols
            group_col = group_id % self.groups_cols
            
            # 4 particles in 2x2 cell
            particles = [
                group_row * self.cols + group_col,           # top-left
                group_row * self.cols + group_col + 1,       # top-right
                (group_row + 1) * self.cols + group_col,     # bottom-left
                (group_row + 1) * self.cols + group_col + 1, # bottom-right
            ]
            self.group_info[group_id] = particles
        
        # Use GROUP 0 as the anchor for rotation/translation
        self.center_group_id = 0
        
        # center_indices = the center group's particles
        self.center_indices = np.array(self.group_info[self.center_group_id])
        
        # No fixed particles - all particles free (soft body tracking)
        self.boundary_indices = np.array([])
        
        print(f"  Group structure (2x2 overlapping cells):")
        print(f"    Number of groups: {self.num_groups} ({self.groups_rows}x{self.groups_cols})")
        print(f"    ANCHOR GROUP: 0 (particles {list(self.center_indices)}) - rotation center")
        print(f"    ALL group centroids track targets with rotation!")
    
    def get_trajectory_offset(self, t=None):
        """
        Get the trajectory offset at time t.
        This offset is added to ALL group centroids for rigid grid motion.
        
        Returns offset from equilibrium position (same for all groups).
        """
        if t is None:
            t = self.time
        
        A = self.trajectory_amplitude
        f = self.trajectory_frequency
        omega = 2 * np.pi * f
        
        if self.trajectory_type == 'sinusoidal':
            offset_x = A * np.sin(omega * t)
            offset_y = 0.0
        elif self.trajectory_type == 'circular':
            offset_x = A * np.cos(omega * t)
            offset_y = A * np.sin(omega * t)
        elif self.trajectory_type == 'figure8':
            offset_x = A * np.sin(omega * t)
            offset_y = A * np.sin(2 * omega * t) / 2
        else:
            offset_x = 0.0
            offset_y = 0.0
        
        return np.array([offset_x, offset_y], dtype=np.float32)
    
    def get_target_position(self, t=None):
        """
        Get the target position for the center group at time t.
        (Alias for get_trajectory_offset for backwards compatibility)
        """
        return self.get_trajectory_offset(t)
    
    def get_rotation_angle(self, t=None):
        """
        Get the rotation angle for the grid at time t.
        
        Returns:
            angle: rotation in radians
        """
        if t is None:
            t = self.time
        
        if not self.enable_rotation:
            return 0.0
        
        return self.rotation_speed * t
    
    def get_all_group_targets(self, t=None):
        """
        Get target positions for ALL group centroids.
        
        Each target = rotate(initial_centroid - center, angle) + center + trajectory_offset
        
        The grid:
        1. Rotates around its center
        2. Translates following the trajectory
        
        Returns:
            targets: shape (num_groups, 2) - target positions in world coordinates
        """
        if t is None:
            t = self.time
            
        if self.initial_group_centroids is None:
            # Compute initial centroids on first call
            self._sync_to_cpu()
            self.initial_group_centroids = np.zeros((self.num_groups, 2), dtype=np.float32)
            for group_id, particle_indices in self.group_info.items():
                self.initial_group_centroids[group_id] = np.mean(self.pos_np[particle_indices], axis=0)
        
        # Get trajectory offset (translation)
        offset = self.get_trajectory_offset(t)
        
        # Get rotation angle
        angle = self.get_rotation_angle(t)
        
        # Center of rotation = initial center group centroid
        center = self.initial_group_centroids[self.center_group_id]
        
        # Rotation matrix
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        
        targets = np.zeros((self.num_groups, 2), dtype=np.float32)
        
        for group_id in range(self.num_groups):
            # Vector from center to this group (in initial config)
            rel = self.initial_group_centroids[group_id] - center
            
            # Rotate around center
            rotated_rel = np.array([
                cos_a * rel[0] - sin_a * rel[1],
                sin_a * rel[0] + cos_a * rel[1]
            ])
            
            # Final target = rotated position + center + translation offset
            targets[group_id] = center + rotated_rel + offset
        
        return targets
    
    def get_group_centroids(self):
        """
        Get centroid positions for all 4 groups.
        Each group centroid = average position of its 4 particles.
        
        Returns:
            centroids: shape (num_groups, 2) - centroid positions
        """
        self._sync_to_cpu()
        centroids = np.zeros((self.num_groups, 2), dtype=np.float32)
        
        for group_id, particle_indices in self.group_info.items():
            group_positions = self.pos_np[particle_indices]
            centroids[group_id] = np.mean(group_positions, axis=0)
        
        return centroids
    
    def get_center_centroid(self):
        """
        Get the CENTER GROUP centroid.
        This is the point that tracks the trajectory.
        """
        self._sync_to_cpu()
        return np.mean(self.pos_np[self.center_indices], axis=0)
    
    def get_tracking_error(self):
        """
        Get the current tracking error (sum of errors for ALL group centroids).
        """
        return self.get_total_grid_error()
    
    def get_all_group_errors(self):
        """
        Get tracking error for each group centroid.
        
        Returns:
            errors: shape (num_groups,) - error for each group
        """
        targets = self.get_all_group_targets()
        centroids = self.get_group_centroids()
        
        errors = np.linalg.norm(centroids - targets, axis=1)
        return errors
    
    def get_total_grid_error(self):
        """
        Get total tracking error across all groups (sum of centroid errors).
        """
        errors = self.get_all_group_errors()
        return np.sum(errors)
    
    def get_mean_grid_error(self):
        """
        Get mean tracking error across all groups.
        """
        errors = self.get_all_group_errors()
        return np.mean(errors)
    
    def reset(self, seed=None, options=None):
        """Reset the environment."""
        obs, info = super().reset(seed=seed, options=options)
        
        self.time = 0.0
        self._sync_to_cpu()
        self.initial_positions = self.pos_np.copy()
        
        # Compute initial group centroids for grid tracking
        self.initial_group_centroids = np.zeros((self.num_groups, 2), dtype=np.float32)
        for group_id, particle_indices in self.group_info.items():
            self.initial_group_centroids[group_id] = np.mean(self.pos_np[particle_indices], axis=0)
        
        # Reset group forces
        self.group_forces = np.zeros((self.num_groups, 2), dtype=np.float32)
        self.current_forces = np.zeros(self.action_dim, dtype=np.float32)
        
        # Clear history
        self.time_history.clear()
        self.centroid_x_history.clear()
        self.centroid_y_history.clear()
        self.target_x_history.clear()
        self.target_y_history.clear()
        
        # Initialize per-group error history
        self.group_error_history = [deque() for _ in range(self.num_groups)]
        
        # Add initial data point
        target = self.get_target_position()
        self.time_history.append(0.0)
        self.centroid_x_history.append(0.0)
        self.centroid_y_history.append(0.0)
        self.target_x_history.append(target[0])
        self.target_y_history.append(target[1])
        
        # Initialize per-group errors at t=0
        for g in range(self.num_groups):
            self.group_error_history[g].append(0.0)
        
        return obs, info
    
    def step(self, action):
        """
        Step the environment with trajectory tracking.
        
        Args:
            action: Forces per group centroid [fx0,fy0,fx1,fy1,...] (2*num_groups)
                   Forces are distributed to particles within each group.
                   If None or wrong size, applies zero forces.
        
        Returns:
            observation, reward, terminated, truncated, info
        """
        # Build full action array (forces for all particles)
        full_action = np.zeros(self.model.particle_count * 2, dtype=np.float32)
        
        # Action is per GROUP CENTROID - distribute to particles
        if action is not None:
            action = np.array(action).flatten()
            self.current_forces = action.copy()
            
            for group_id in range(self.num_groups):
                if group_id * 2 + 1 < len(action):
                    fx = action[group_id * 2]
                    fy = action[group_id * 2 + 1]
                    self.group_forces[group_id] = [fx, fy]
                    
                    # Distribute force equally to 4 particles in group
                    particles = self.group_info[group_id]
                    fx_per_particle = fx / len(particles)
                    fy_per_particle = fy / len(particles)
                    
                    for particle_idx in particles:
                        full_action[particle_idx * 2] += fx_per_particle
                        full_action[particle_idx * 2 + 1] += fy_per_particle
        else:
            self.current_forces = np.zeros(self.action_dim, dtype=np.float32)
            self.group_forces = np.zeros((self.num_groups, 2), dtype=np.float32)
        
        # Call parent step
        obs, reward, terminated, truncated, info = super().step(full_action)
        
        self.time += self.dt
        
        # Get per-group errors
        group_errors = self.get_all_group_errors()
        target = self.get_target_position()
        
        # Add tracking info
        info['group_errors'] = group_errors
        info['tracking_error'] = np.sum(group_errors)  # Total for compatibility
        info['target'] = target
        info['time'] = self.time
        
        # Track history for plots
        self.time_history.append(self.time)
        
        # Track per-group errors
        for g in range(self.num_groups):
            self.group_error_history[g].append(group_errors[g])
        
        # Keep centroid history for XY plot
        centroid = self.get_center_centroid()
        if self.initial_positions is not None:
            init_center = np.mean(self.initial_positions[self.center_indices], axis=0)
            relative_centroid = centroid - init_center
        else:
            relative_centroid = np.zeros(2)
        self.centroid_x_history.append(relative_centroid[0])
        self.centroid_y_history.append(relative_centroid[1])
        self.target_x_history.append(target[0])
        self.target_y_history.append(target[1])
        
        # Reward is negative sum of group errors
        reward = -np.sum(group_errors)
        
        return obs, reward, terminated, truncated, info
    
    def get_state_for_controller(self):
        """
        Get state information formatted for the controller.
        Provides info for all group centroids + strain data for sensory feedback.
        """
        self._sync_to_cpu()
        
        # Compute group velocities (mean velocity of particles in each group)
        group_velocities = np.zeros((self.num_groups, 2), dtype=np.float32)
        for group_id, particle_indices in self.group_info.items():
            group_velocities[group_id] = np.mean(self.vel_np[particle_indices], axis=0)
        
        # Get strain data (sensory feedback for SNN)
        spring_strains = None
        fem_strains = None
        if hasattr(self.model, 'spring_strains_normalized') and self.model.spring_strains_normalized is not None:
            spring_strains = self.model.spring_strains_normalized.numpy()
        if hasattr(self.model, 'tri_strains_normalized') and self.model.tri_strains_normalized is not None:
            fem_strains = self.model.tri_strains_normalized.numpy()
        
        state = {
            'positions': self.pos_np.copy(),
            'velocities': self.vel_np.copy(),
            'center_indices': self.center_indices,
            'target': self.get_target_position(),
            'initial_positions': self.initial_positions,
            'time': self.time,
            'dt': self.dt,  # Physics dt for SNN timing sync
            'num_groups': self.num_groups,
            'group_info': self.group_info,
            'group_centroids': self.get_group_centroids(),
            'group_targets': self.get_all_group_targets(),
            'initial_group_centroids': self.initial_group_centroids.copy(),
            'trajectory_offset': self.get_trajectory_offset(),
            'rotation_angle': self.get_rotation_angle(),
            'group_velocities': group_velocities,
            # Strain feedback (normalized [-1, 1])
            'spring_strains': spring_strains,
            'fem_strains': fem_strains,
        }
        
        return state
    
    def _draw_legends(self, canvas):
        """Override parent's legend drawing to stay in simulation area (not plot area)."""
        # Only show FEM legend if using implicit solver
        solver_uses_fem = self.solver.__class__.__name__ == "SolverImplicit"
        
        # Get strain scales from model
        spring_scale = self.model.spring_strain_scale.numpy()[0] if hasattr(self.model, 'spring_strain_scale') else 0.01
        fem_scale = self.model.fem_strain_scale.numpy()[0] if hasattr(self.model, 'fem_strain_scale') else 0.01
        
        # Calculate position for sim_width (not full window_width)
        legend_start_x = self.sim_width - (220 if solver_uses_fem else 120)
        
        # Delegate to renderer with custom position
        self.renderer.draw_strain_legends(
            canvas,
            spring_scale=spring_scale,
            fem_scale=fem_scale,
            show_fem=solver_uses_fem,
            position=(legend_start_x, 10)
        )
    
    def _finalize_render(self, canvas, t_start, t_sync, t_springs, t_particles, t_text):
        """Override to NOT flip display - we'll do it ourselves in render() after adding plots."""
        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            # DON'T call pygame.display.flip() here - we'll do it in render()
            return None
        else:  # rgb_array mode
            import numpy as np
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))
    
    def _draw_ui_text(self, canvas):
        """Override parent's UI text with tracking-specific info (like rate_coding style)."""
        if not pygame.font.get_init():
            return
        
        font = pygame.font.Font(None, 24)
        font_small = pygame.font.Font(None, 18)
        
        # Get tracking info
        error = self.get_tracking_error() if self.initial_positions is not None else 0.0
        target = self.get_target_position()
        
        # Line 1: Trajectory info (blue, like rate_coding's CPG line)
        rotation_deg = np.degrees(self.get_rotation_angle()) % 360
        track_text = f"Trajectory: {self.trajectory_type} | A={self.trajectory_amplitude:.1f} | f={self.trajectory_frequency:.1f}Hz | θ={rotation_deg:.0f}°"
        canvas.blit(font.render(track_text, True, (0, 100, 200)), (10, 10))
        
        # Line 2: Time and error (grey, like rate_coding's Active Forces line)
        status_text = f"Time: {self.time:.1f}s | Error: {error:.4f} | Target: ({target[0]:+.2f}, {target[1]:+.2f})"
        canvas.blit(font_small.render(status_text, True, (100, 100, 100)), (10, 35))
        
        # Line 3: Force status (green if active, grey if not)
        if self.current_forces is not None:
            total_mag = np.sqrt(np.sum(self.current_forces[0::2])**2 + np.sum(self.current_forces[1::2])**2)
            force_status = f"Control Force: |F|={total_mag:.1f}"
            force_color = (0, 150, 0) if total_mag > 1.0 else (150, 150, 150)
        else:
            force_status = "Control Force: waiting..."
            force_color = (150, 150, 150)
        canvas.blit(font_small.render(force_status, True, force_color), (10, 52))
        
        # Line 4: Solver info
        solver_text = self._get_solver_display_text()
        canvas.blit(font_small.render(solver_text, True, (100, 100, 100)), (10, 69))
        
        # Line 5: Model info  
        has_fem = hasattr(self.model, 'tri_count') and self.model.tri_count > 0
        if has_fem:
            info_text = f"Grid: {self.cols}x{self.rows} | Springs: {self.model.spring_count} | FEM: {self.model.tri_count}"
        else:
            info_text = f"Grid: {self.cols}x{self.rows} | Springs: {self.model.spring_count}"
        canvas.blit(font_small.render(info_text, True, (100, 100, 100)), (10, 86))
    
    def _create_tracking_plots(self):
        """Create matplotlib figure for tracking history plots with LARGE grid view."""
        fig_width = self.plot_width / 100.0
        fig_height = self.window_height / 100.0
        
        self.fig = plt.figure(figsize=(fig_width, fig_height), dpi=100)
        self.fig.canvas = FigureCanvasAgg(self.fig)
        
        # Layout: Large XY grid view (top 70%), Error plot (bottom 30%)
        self.axes = []
        # Large XY plot showing rotating grid - takes more space
        ax_xy = self.fig.add_axes([0.1, 0.35, 0.85, 0.60])  # [left, bottom, width, height]
        self.axes.append(ax_xy)
        # Error over time - smaller at bottom
        ax_err = self.fig.add_axes([0.1, 0.08, 0.85, 0.22])
        self.axes.append(ax_err)
    
    def _update_tracking_plots(self):
        """Update tracking plots with current history data - shows rotating grid."""
        if len(self.time_history) < 2:
            return
        
        if self.fig is None:
            self._create_tracking_plots()
        
        t = np.array(self.time_history)
        cx = np.array(self.centroid_x_history)
        cy = np.array(self.centroid_y_history)
        tx = np.array(self.target_x_history)
        ty = np.array(self.target_y_history)
        
        # Colors (matching simulation)
        target_color = 'black'
        actual_color = '#FF69B4'  # Hot pink
        
        # Fixed axis limits - larger to show full rotation
        pos_limit = self.trajectory_amplitude * 2.5
        
        # ===== LARGE XY GRID VIEW =====
        ax = self.axes[0]
        ax.clear()
        
        # Draw center trajectory path
        ax.plot(tx, ty, 'k--', label='Target path', linewidth=1.5, alpha=0.5)
        ax.plot(cx, cy, color=actual_color, label='Actual path', linewidth=1.5, alpha=0.6)
        
        # Get current group targets and centroids for grid visualization
        if self.initial_group_centroids is not None:
            group_targets = self.get_all_group_targets()
            group_centroids = self.get_group_centroids()
            
            # Draw target grid (BLACK crosshairs - matching simulation)
            # Convert targets to relative coordinates (relative to initial center)
            init_center = self.initial_group_centroids[self.center_group_id]
            target_xs = group_targets[:, 0] - init_center[0]
            target_ys = group_targets[:, 1] - init_center[1]
            
            # Draw BLACK circles with numbers for targets (same size)
            gr, gc = self.groups_rows, self.groups_cols  # group rows and cols
            for group_id in range(self.num_groups):
                gx, gy = target_xs[group_id], target_ys[group_id]  # Renamed to avoid conflict with tx, ty
                circle_size = 0.025  # Same size for all
                # Draw filled black circle
                circle = plt.Circle((gx, gy), circle_size, fill=True, facecolor='black',
                                    edgecolor='black', linewidth=1.5, zorder=4)
                ax.add_patch(circle)
                # Add group number (white text)
                ax.annotate(str(group_id), (gx, gy), ha='center', va='center', 
                           fontsize=7, fontweight='bold', color='white', zorder=5)
            
            # Connect target grid points with black lines
            for row in range(gr):
                row_indices = [row * gc + col for col in range(gc)]
                ax.plot(target_xs[row_indices], target_ys[row_indices], 'k-', alpha=0.3, linewidth=1)
            for col in range(gc):
                col_indices = [row * gc + col for row in range(gr)]
                ax.plot(target_xs[col_indices], target_ys[col_indices], 'k-', alpha=0.3, linewidth=1)
            
            # Draw actual grid (HOT PINK circles - matching simulation)
            offset = self.initial_group_centroids[self.center_group_id]
            actual_xs = group_centroids[:, 0] - offset[0]
            actual_ys = group_centroids[:, 1] - offset[1]
            
            # Draw hot pink circles for all groups (like simulation)
            hot_pink = '#FF69B4'
            ax.scatter(actual_xs, actual_ys, c=hot_pink, s=350, marker='o', alpha=0.9,
                      edgecolors='black', linewidths=2, label='Actual', zorder=5)
            
            # Add group numbers (white text on pink)
            for group_id in range(self.num_groups):
                ax.annotate(str(group_id), (actual_xs[group_id], actual_ys[group_id]),
                           ha='center', va='center', fontsize=10, fontweight='bold',
                           color='white', zorder=6)
            
            # Connect actual grid points (pink lines)
            for row in range(gr):
                row_indices = [row * gc + col for col in range(gc)]
                ax.plot(actual_xs[row_indices], actual_ys[row_indices], color=hot_pink, 
                       alpha=0.5, linewidth=2)
            for col in range(gc):
                col_indices = [row * gc + col for row in range(gr)]
                ax.plot(actual_xs[col_indices], actual_ys[col_indices], color=hot_pink, 
                       alpha=0.5, linewidth=2)
            
            # Show rotation angle
            angle_deg = np.degrees(self.get_rotation_angle()) % 360
            ax.set_title(f'Rotating Grid View  |  θ = {angle_deg:.1f}°', fontsize=12)
        else:
            ax.set_title('Grid Tracking', fontsize=12)
        
        # Mark origin
        ax.axhline(0, color='gray', linestyle='-', alpha=0.3, linewidth=0.5)
        ax.axvline(0, color='gray', linestyle='-', alpha=0.3, linewidth=0.5)
        ax.plot(0, 0, 'k+', markersize=10, markeredgewidth=2)
        
        ax.set_xlabel('X', fontsize=10)
        ax.set_ylabel('Y', fontsize=10)
        ax.set_xlim(-pos_limit, pos_limit)
        ax.set_ylim(-pos_limit, pos_limit)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.2)
        ax.legend(loc='upper right', fontsize=8)
        
        # ===== PER-GROUP ERROR PLOT =====
        ax = self.axes[1]
        ax.clear()
        
        # Plot each group's error with its own color
        max_err = 0.01  # Minimum scale
        if self.group_error_history is not None:
            for group_id in range(self.num_groups):
                if len(self.group_error_history[group_id]) > 0:
                    group_err = np.array(self.group_error_history[group_id])
                    color = self.group_colors[group_id % len(self.group_colors)]
                    ax.plot(t[:len(group_err)], group_err, color=color, 
                           linewidth=1.5, label=f'G{group_id}', alpha=0.8)
                    max_err = max(max_err, np.max(group_err))
        
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel('Error per Group', fontsize=10)
        ax.set_xlim(0, self.max_time)
        ax.set_ylim(0, max_err * 1.2)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=7, ncol=3)
        ax.set_title('Per-Group Tracking Error', fontsize=10)
        
        self.fig.tight_layout(pad=1.5)
        
        # Convert to pygame surface
        self.fig.canvas.draw()
        w, h = self.fig.canvas.get_width_height()
        buf = np.frombuffer(self.fig.canvas.buffer_rgba(), dtype=np.uint8)
        buf = buf.reshape((h, w, 4))
        buf = buf[:, :, :3]  # Remove alpha
        self.plot_surface = pygame.surfarray.make_surface(buf.swapaxes(0, 1))
    
    def render(self):
        """Render with target overlay and tracking plots."""
        # Call parent render (draws the simulation area)
        result = super().render()
        
        self.frame_count += 1
        
        # Draw target and tracking info on top
        if self.render_mode == 'human' and self.window is not None:
            # Draw a white background for the plot area (clear it)
            plot_x = self.sim_width
            pygame.draw.rect(self.window, (255, 255, 255), 
                           (plot_x, 0, self.plot_width, self.window_height))
            
            # Update plots only every N frames (for performance)
            if self.frame_count % self.plot_update_interval == 0:
                self._update_tracking_plots()
            
            # Draw plot surface if available
            if self.plot_surface is not None:
                self.window.blit(self.plot_surface, (plot_x, 0))
            
            # Draw separator line
            pygame.draw.line(self.window, (100, 100, 100),
                           (plot_x, 0), (plot_x, self.window_height), 2)
            
            # Draw target overlay (crosshair, centroid, legend)
            self._draw_target_overlay()
            
            # Draw wind arrow if wind is set
            self._draw_wind_arrow(scale=self.window_height / self.boxsize)
            
            # Single display flip at the end
            pygame.display.flip()
        
        return result
    
    def _draw_target_overlay(self):
        """Draw targets, group centroids (pink), and force arrows using shared Renderer."""
        if self.initial_positions is None:
            return
        
        # Sync state for drawing
        self._sync_to_cpu()
        
        # Get all group centroids and targets
        group_centroids = self.get_group_centroids()
        group_targets = self.get_all_group_targets()
        
        # Draw force arrows using renderer
        if self.current_forces is not None and len(self.current_forces) > 0:
            # Convert group_forces to origins and force vectors
            origins = group_centroids
            forces = self.group_forces  # shape (num_groups, 2)
            self.renderer.draw_force_arrows(
                self.window,
                origins,
                forces,
                max_arrow_length=60.0,
                min_magnitude=0.1,
                force_scale=20.0,
            )
        
        # Draw targets for ALL groups (black filled circles with white numbers)
        # Use renderer's draw_group_centroids with black color for targets
        self.renderer.draw_group_centroids(
            self.window,
            group_targets,
            show_labels=True,
            fill_color=self.renderer.BLACK,
            outline_color=self.renderer.BLACK,
        )
        
        # Draw actual group centroids (hot pink)
        self.renderer.draw_group_centroids(
            self.window,
            group_centroids,
            show_labels=True,
        )
        
        # Draw tracking legend (in simulation area, not plot area)
        self._draw_tracking_legend()
    
    def set_wind(self, wind_force):
        """Set wind force for visualization."""
        self.wind_force = np.array(wind_force)
    
    def _draw_wind_arrow(self, scale):
        """Draw wind force arrow using shared Renderer."""
        self.renderer.draw_wind_arrow(self.window, self.wind_force)
    
    def _draw_tracking_legend(self):
        """Draw legend for target, center group centroid, and control force."""
        hot_pink = self.renderer.HOT_PINK
        black = self.renderer.BLACK
        white = self.renderer.WHITE
        grey = (50, 50, 50)
        
        legend_x = 10
        legend_y = self.window_height - 150
        
        # Target legend (black crosshair)
        pygame.draw.circle(self.window, black, (legend_x + 8, legend_y), 8, 2)
        pygame.draw.line(self.window, black, (legend_x, legend_y), (legend_x + 16, legend_y), 2)
        pygame.draw.line(self.window, black, (legend_x + 8, legend_y - 8), (legend_x + 8, legend_y + 8), 2)
        label = self.renderer.font_small.render("Target", True, grey)
        self.window.blit(label, (legend_x + 25, legend_y - 8))
        
        # Group centroid legend (pink circles)
        legend_y += 25
        pygame.draw.circle(self.window, black, (legend_x + 8, legend_y), 8)
        pygame.draw.circle(self.window, hot_pink, (legend_x + 8, legend_y), 6)
        label = self.renderer.font_small.render("Group centroids", True, grey)
        self.window.blit(label, (legend_x + 25, legend_y - 8))
        
        # Force arrow legend
        legend_y += 25
        pygame.draw.line(self.window, (180, 120, 80), (legend_x, legend_y), (legend_x + 20, legend_y), 3)
        self.renderer._draw_arrowhead(self.window, (legend_x, legend_y), (legend_x + 20, legend_y), (180, 120, 80))
        label = self.renderer.font_small.render("Control Force", True, grey)
        self.window.blit(label, (legend_x + 30, legend_y - 8))
        
        # Force magnitude legend (gradient bar)
        if self.current_forces is not None:
            total_fx = np.sum(self.current_forces[0::2])
            total_fy = np.sum(self.current_forces[1::2])
            current_force = np.sqrt(total_fx**2 + total_fy**2)
        else:
            current_force = None
        
        self.renderer.draw_force_legend(
            self.window,
            max_force=50.0,
            position=(self.sim_width - 120, 100),
            current_force=current_force,
        )
    


if __name__ == '__main__':
    # Quick test
    print("Testing TrackingEnv (4x4 grid, rotating rigid grid)...")
    env = TrackingEnv(
        render_mode='human',
        N=4,  # 4x4 grid = 9 groups
        dt=0.01,
        trajectory_amplitude=0.5,
        trajectory_frequency=0.3,
        device='cuda',
    )
    
    obs, _ = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Number of groups: {env.num_groups}")
    print(f"Action dim: {env.action_dim} (2 per group centroid)")
    print(f"Center group: {env.center_group_id}")
    
    for i in range(500):
        state = env.get_state_for_controller()
        
        # GRID MODE: Compute force for EACH group centroid
        # Each group tracks: target = initial_centroid + trajectory_offset
        action = np.zeros(env.action_dim, dtype=np.float32)
        
        group_targets = state['group_targets']
        group_centroids = state['group_centroids']
        group_velocities = state['group_velocities']
        
        Kp = 50.0  # Proportional gain
        Kd = 5.0   # Derivative gain
        
        for group_id in range(env.num_groups):
            # Position error
            error = group_targets[group_id] - group_centroids[group_id]
            # Velocity (for damping)
            vel = group_velocities[group_id]
            
            # PD control per group
            force = Kp * error - Kd * vel
            
            action[group_id * 2] = force[0]
            action[group_id * 2 + 1] = force[1]
        
        obs, reward, done, trunc, info = env.step(action)
        env.render()
        
        if i % 100 == 0:
            print(f"Step {i}: total_error={info['tracking_error']:.4f}, mean={info['mean_error']:.4f}")
    
    env.close()
    print("Test complete!")
