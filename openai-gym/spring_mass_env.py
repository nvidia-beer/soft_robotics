import sys
import os
# Add warp directory to path (sibling directory)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'warp'))
# Add pygame_renderer to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame
import warp as wp
import time

from sim import State
from models import GridModel
from solvers import SolverSemiImplicit
from pygame_renderer import Renderer


class SpringMassEnv(gym.Env):
    """
    Warp-accelerated Spring Mass System Environment for Gymnasium.
    
    GPU-accelerated 2D spring network simulation using modular solver architecture.
    Follows newton.sim.Model + newton.solvers pattern.
    
    Architecture:
        - sim.Model: Static system description (topology, parameters)
        - sim.State: Time-varying state (positions, velocities, forces)
        - solvers.Solver*: Time integration (semi-implicit or implicit)
    
    Features:
        - GPU acceleration via Warp
        - FEM triangular elements with Neo-Hookean material
        - Spring network with damping
        - Real-time strain visualization
        - Momentum-conserving physics
    """
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 60}
    
    # ========================================================================
    # INITIALIZATION
    # ========================================================================
    
    def __init__(
        self, 
        render_mode=None, 
        rows=3,  # Grid height (Y direction)
        cols=6,  # Grid width (X direction)
        dt=0.1, 
        spring_coeff=40.0, 
        spring_damping=0.5,
        gravity=-0.1, 
        boxsize=2.5,
        device='cuda',
        model=None,  # Optional: provide pre-built Model
        with_fem=True,  # Enable/disable FEM triangles
        with_springs=True,  # Enable/disable springs
        window_width=1000,  # Window width in pixels
        window_height=500,  # Window height in pixels
    ):
        """
        Initialize the Spring Mass Environment.
        
        Args:
            render_mode: 'human', 'rgb_array', or None
            rows: Grid height (number of rows, Y direction). Default 4.
            cols: Grid width (number of columns, X direction). Default 4.
            dt: Physics timestep
            spring_coeff: Spring stiffness (uniform if model not provided)
            spring_damping: Spring damping (uniform if model not provided)
            gravity: Gravity strength (negative = downward)
            boxsize: Size of simulation box
            device: Warp device ('cuda' or 'cpu')
            model: Optional pre-built Model object (overrides rows/cols, etc.)
            with_fem: Enable/disable FEM triangles
            with_springs: Enable/disable springs
            window_width: Window width in pixels
            window_height: Window height in pixels
        
        Examples:
            >>> env = SpringMassEnv()                 # 3x6 default (stable)
            >>> env = SpringMassEnv(rows=4, cols=4)   # 4x4 square grid
        """
        super(SpringMassEnv, self).__init__()
        
        # Initialize Warp
        wp.init()
        self.device = device
        
        # Physics parameters
        self.rows = rows
        self.cols = cols
        self.dt = dt
        self.boxsize = boxsize
        
        # Render mode
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        
        # Pygame rendering setup
        self.window_width = window_width
        self.window_height = window_height
        # Calculate boxsize_x based on aspect ratio to maintain uniform scale
        self.boxsize_x = boxsize * (window_width / window_height)
        self.window = None
        self.clock = None
        
        # Create shared renderer (consolidates all drawing logic)
        self.renderer = Renderer(
            window_width=window_width,
            window_height=window_height,
            boxsize=boxsize,
        )
        
        # Create or use provided model
        if model is None:
            # Calculate particle spacing
            particle_spacing = 1.0 / (5 - 1)  # Reference spacing from N=5
            
            # Create model using GridModel
            self.model = GridModel(
                rows=rows, cols=cols, spacing=particle_spacing, 
                device=device, boxsize=boxsize, 
                with_fem=with_fem, with_springs=with_springs
            )
            
            # Set custom properties
            self.model.spring_stiffness.fill_(spring_coeff)
            self.model.spring_damping.fill_(spring_damping)
            self.model.set_gravity((0.0, gravity))
        else:
            self.model = model
            if hasattr(model, 'grid_rows') and hasattr(model, 'grid_cols'):
                self.rows = model.grid_rows
                self.cols = model.grid_cols
            else:
                # Assume square grid from particle count
                n = int(np.sqrt(model.particle_count))
                self.rows = n
                self.cols = n
        
        
        # Create solver
        self.solver = SolverSemiImplicit(self.model)
        
        # Define action and observation space
        self.action_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(self.model.particle_count * 2,), 
            dtype=np.float32
        )
        
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.model.particle_count * 4,),
            dtype=np.float32
        )
        
        # State management
        self.state_in = None
        self.state_out = None
        self._state_dirty = True
        
        # Host arrays for rendering (lazy sync)
        self.pos_np = None
        self.vel_np = None
        self.spring_strains_normalized_np = None  # Normalized spring strains [-1, 1] from solver
        self.tri_strains_normalized_np = None     # Normalized FEM strains [-1, 1] from solver
        
        # Group centroid tracking for hierarchical visualization
        self.group_info = None  # Will be set by external code (demo_snn_gui.py)
        self.centroids_np = None  # Cached centroid positions
        
        # Cache static data that never changes (compute once)
        self.spring_indices_np = self.model.spring_indices.numpy()  # Static, cache once
        self.tri_indices_np = None
        if hasattr(self.model, 'tri_indices') and self.model.tri_indices is not None:
            self.tri_indices_np = self.model.tri_indices.numpy()  # Static, cache once
        
        # Cached energy (updated less frequently for performance)
        self._cached_energy = 0.0
        self._energy_update_counter = 0
        self._energy_update_interval = 10  # Update every N frames
        
        # Performance profiling
        self._profile_render = False  # Set to True to enable profiling
        self._profile_physics = False  # Set to True to enable physics profiling
        self._render_times = {'sync': [], 'springs': [], 'particles': [], 'text': [], 'blit': [], 'flip': []}
        self._physics_times = {'solver': [], 'sync_gpu': [], 'numpy_copy': []}
        self._profile_counter = 0
        self._physics_profile_counter = 0
        
        self.t = 0
    
    # ========================================================================
    # ENVIRONMENT INTERFACE (Gymnasium API)
    # ========================================================================
    
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state"""
        super().reset(seed=seed)
        
        # Set random seed for reproducibility
        if seed is not None:
            np.random.seed(seed)
        else:
            np.random.seed(17)
        
        # Initialize all particles at rest (zero velocity)
        # Particles are positioned at their natural rest configuration
        vel_zero = np.zeros((self.model.particle_count, 2), dtype=np.float32)
        self.model.particle_qd.assign(wp.array(vel_zero, dtype=wp.vec2, device=self.device))
        
        # Create states
        self.state_in = self.model.state()
        self.state_out = self.model.state()
        
        self.t = 0
        self._state_dirty = True
        
        return self._get_obs(), self._get_info()
    
    def step(self, action):
        """Execute one time step of physics simulation"""
        
        t0 = time.perf_counter() if self._profile_physics else None
        
        # Convert action to external forces
        external_forces_np = action.reshape(self.model.particle_count, 2).astype(np.float32)
        
        # Step solver (GPU computation)
        self.solver.step(
            state_in=self.state_in,
            state_out=self.state_out,
            dt=self.dt,
            external_forces=external_forces_np
        )
        
        t1 = time.perf_counter() if self._profile_physics else None
        
        # Swap state buffers
        self.state_in, self.state_out = self.state_out, self.state_in
        
        self._state_dirty = True
        self.t += self.dt
        
        if self._profile_physics and t0 is not None:
            self._physics_times['solver'].append((t1 - t0) * 1000)
        
        return self._get_obs(), self._calculate_reward(), False, False, self._get_info()
    
    def _sync_to_cpu(self):
        """Lazy synchronization: only transfer GPU->CPU when needed."""
        if self._state_dirty:
            t0 = time.perf_counter() if self._profile_physics else None
            
            # Wait for GPU to finish all pending work
            wp.synchronize()
            
            t1 = time.perf_counter() if self._profile_physics else None
            
            # Transfer data from GPU to CPU
            self.pos_np = self.state_in.particle_q.numpy()
            self.vel_np = self.state_in.particle_qd.numpy()
            
            # Cache normalized spring strains (computed by solver in [-1, 1] range)
            if hasattr(self.model, 'spring_strains_normalized') and self.model.spring_strains_normalized is not None:
                self.spring_strains_normalized_np = self.model.spring_strains_normalized.numpy()
            
            # Cache normalized FEM strains (computed by solver in [-1, 1] range)
            if hasattr(self.model, 'tri_strains_normalized') and self.model.tri_strains_normalized is not None:
                self.tri_strains_normalized_np = self.model.tri_strains_normalized.numpy()
            
            t2 = time.perf_counter() if self._profile_physics else None
            
            if self._profile_physics and t0 is not None:
                self._physics_times['sync_gpu'].append((t1 - t0) * 1000)
                self._physics_times['numpy_copy'].append((t2 - t1) * 1000)
                
                self._physics_profile_counter += 1
                if self._physics_profile_counter >= 100:
                    print("\n=== PHYSICS PROFILING (avg over 100 steps, ms) ===")
                    for key, times in self._physics_times.items():
                        if times:
                            print(f"{key:12s}: {np.mean(times):6.2f} ms")
                    total = sum(np.mean(times) for times in self._physics_times.values() if times)
                    print(f"{'TOTAL':12s}: {total:6.2f} ms")
                    print("=" * 50)
                    # Reset
                    self._physics_times = {k: [] for k in self._physics_times.keys()}
                    self._physics_profile_counter = 0
            
            self._state_dirty = False
    
    # ========================================================================
    # STATE & OBSERVATION METHODS
    # ========================================================================
    
    def _get_obs(self):
        self._sync_to_cpu()
        return np.concatenate([self.pos_np.flatten(), self.vel_np.flatten()]).astype(np.float32)
    
    def _get_info(self):
        return {
            'time': self.t, 
            'total_energy': self._calculate_energy(),
            'center_of_mass_velocity': self._calculate_com_velocity()
        }
    
    def _calculate_com_velocity(self):
        """Calculate center-of-mass velocity (should be ~zero if no drift)"""
        self._sync_to_cpu()
        # Total momentum / total mass (assuming unit mass per particle)
        com_vel = np.mean(self.vel_np, axis=0)
        return com_vel
    
    def _calculate_reward(self):
        return 0.0
    
    def _calculate_energy(self):
        """Calculate total system energy (kinetic + potential)"""
        self._sync_to_cpu()
        
        # Kinetic energy
        ke = 0.5 * np.sum(self.vel_np ** 2)
        
        # Spring potential energy
        spring_pe = 0.0
        # Use cached spring_indices_np (already cached in __init__)
        spring_rest_np = self.model.spring_rest_length.numpy()
        spring_stiff_np = self.model.spring_stiffness.numpy()
        
        for i in range(self.model.spring_count):
            idx_i, idx_j = self.spring_indices_np[i * 2:i * 2 + 2]
            pos_i, pos_j = self.pos_np[idx_i], self.pos_np[idx_j]
            l = np.linalg.norm(pos_i - pos_j)
            rest = spring_rest_np[i]
            k = spring_stiff_np[i]
            spring_pe += 0.5 * k * (l - rest) ** 2
        
        # Gravitational potential energy
        gravity_y = self.model.gravity.numpy()[0][1]
        gpe = -gravity_y * np.sum(self.pos_np[:, 1])
        
        return ke + spring_pe + gpe
    
    # ========================================================================
    # RENDERING INTERFACE
    # ========================================================================
    
    def render(self):
        """Public rendering interface"""
        if self.render_mode is None:
            return
        return self._render_frame()
    
    # ========================================================================
    # RENDERING METHODS
    # ========================================================================
    
    # --- Main Rendering Pipeline ---
    
    def _render_frame(self):
        """Main rendering method - orchestrates all drawing operations"""
        t_start = time.perf_counter() if self._profile_render else None
        
        self._sync_to_cpu()
        t_sync = time.perf_counter() if self._profile_render else None
        
        # Initialize pygame window if needed
        self._init_window()
        
        # Create canvas and draw all elements
        canvas = self._create_canvas()
        # Use uniform scale based on height to avoid distortion
        scale = self.window_height / self.boxsize
        
        self._draw_grid(canvas)
        self._draw_fem_triangles(canvas, scale)
        t_springs = self._draw_springs(canvas, scale)
        t_particles = self._draw_particles(canvas, scale)
        self._draw_centroids(canvas, scale)  # Draw group centroids
        self._draw_ui_text(canvas)
        self._draw_legends(canvas)
        
        t_text = time.perf_counter() if self._profile_render else None
        
        # Display or return the rendered frame
        result = self._finalize_render(canvas, t_start, t_sync, t_springs, t_particles, t_text)
        return result
    
    def _init_window(self):
        """Initialize pygame window for human rendering mode"""
        if self.render_mode == "human" and self.window is None:
            pygame.init()
            pygame.display.init()
            flags = pygame.HWSURFACE | pygame.DOUBLEBUF
            self.window = pygame.display.set_mode((self.window_width, self.window_height), flags, vsync=0)
            pygame.display.set_caption("Spring Mass System (Warp - Modular)")
    
    def _create_canvas(self):
        """Create and initialize the drawing canvas"""
        return self.renderer.create_canvas()
    
    def _draw_grid(self, canvas):
        """Draw background grid lines"""
        self.renderer.draw_grid(canvas)
    
    # --- FEM Triangle Rendering ---
    
    def _draw_fem_triangles(self, canvas, scale):
        """
        Draw FEM triangles with strain-based coloring.
        
        Only draws triangles if the solver actually computes FEM forces.
        Semi-implicit solver doesn't evaluate FEM, so triangles are not drawn
        even if they exist in the mesh topology.
        """
        if not (hasattr(self.model, 'tri_count') and self.model.tri_count > 0 and self.tri_indices_np is not None):
            return
        
        # Only draw FEM if using implicit solver or VBD (semi-implicit doesn't compute FEM)
        solver_name = self.solver.__class__.__name__
        solver_uses_fem = solver_name in ("SolverImplicit", "SolverImplicitFEM", "SolverVBD")
        
        if not solver_uses_fem:
            return  # Skip drawing FEM triangles for non-FEM solvers
        
        # Check if we have strain data
        has_strains = self.tri_strains_normalized_np is not None and len(self.tri_strains_normalized_np) > 0
        
        if not has_strains:
            return  # No strain data available
        
        # Delegate to renderer
        self.renderer.draw_fem_triangles(
            canvas,
            self.tri_indices_np,
            self.pos_np,
            self.tri_strains_normalized_np
        )
    
    # --- Spring Rendering ---
    
    def _draw_springs(self, canvas, scale):
        """Draw springs with strain-based coloring (Yellow -> Red)"""
        if self.model.spring_count == 0:
            return time.perf_counter() if self._profile_render else None
        
        # Get strain data (pre-normalized by solver)
        strains = self.spring_strains_normalized_np if self.spring_strains_normalized_np is not None else None
        
        # Delegate to renderer
        self.renderer.draw_springs(
            canvas,
            self.spring_indices_np,
            self.pos_np,
            strains
        )
        
        return time.perf_counter() if self._profile_render else None
    
    # --- Particle Rendering ---
    
    def _draw_particles(self, canvas, scale):
        """Draw particles as circles with outlines"""
        # Check for NaN and warn
        nan_mask = np.any(np.isnan(self.pos_np), axis=1)
        if np.any(nan_mask):
            nan_indices = np.where(nan_mask)[0]
            for idx in nan_indices:
                print(f"WARNING: Particle {idx} has NaN position: {self.pos_np[idx]}, skipping render")
        
        # Delegate to renderer
        self.renderer.draw_particles(canvas, self.pos_np)
        
        return time.perf_counter() if self._profile_render else None
    
    def _draw_centroids(self, canvas, scale):
        """
        Draw group centroids as hot pink circles.
        
        Centroids represent the center of mass for each 2x2 particle group.
        """
        if self.group_info is None or len(self.group_info) == 0:
            return  # No group information available
        
        # Calculate centroids dynamically
        self._calculate_centroids()
        
        if self.centroids_np is None:
            return
        
        # Delegate to renderer (without labels for base env)
        self.renderer.draw_group_centroids(canvas, self.centroids_np, show_labels=False)
    
    def _calculate_centroids(self):
        """Calculate centroid positions for each group dynamically"""
        if self.group_info is None or self.pos_np is None:
            return
        
        num_groups = len(self.group_info)
        self.centroids_np = np.zeros((num_groups, 2), dtype=np.float32)
        
        for group_id, particle_indices in self.group_info.items():
            if len(particle_indices) == 0:
                continue
            
            # Calculate average position of all particles in this group
            group_positions = self.pos_np[particle_indices]
            centroid = np.mean(group_positions, axis=0)
            self.centroids_np[group_id] = centroid
    
    # --- UI Text & Info ---
    
    def _draw_ui_text(self, canvas):
        """Draw UI text (time, energy, solver info, etc.)"""
        # Update cached energy periodically
        self._energy_update_counter += 1
        if self._energy_update_counter >= self._energy_update_interval:
            self._cached_energy = self._calculate_energy()
            self._energy_update_counter = 0
        
        # Build info lines
        has_fem = hasattr(self.model, 'tri_count') and self.model.tri_count > 0
        if has_fem:
            model_info = f"Springs: {self.model.spring_count}, FEM: {self.model.tri_count} triangles"
        else:
            model_info = f"Springs: {self.model.spring_count} (no FEM)"
        
        lines = [
            (f"Time: {self.t:.2f}s", self.renderer.BLACK),
            (f"Energy: {self._cached_energy:.2f}", self.renderer.BLACK),
            (f"Device: {self.device}", (0, 150, 0)),
            (self._get_solver_display_text(), self.renderer.GREY),
            (model_info, self.renderer.GREY),
        ]
        
        self.renderer.draw_info_text(canvas, lines, position=(10, 10), line_spacing=25)
    
    def _get_solver_display_text(self):
        """Get solver display name"""
        solver_name = self.solver.__class__.__name__
        
        if solver_name == "SolverVBD":
            return "Solver: VBD (Vertex Block Descent)"
        elif solver_name == "SolverImplicitFEM":
            has_fem = hasattr(self.model, 'tri_count') and self.model.tri_count > 0
            return "Solver: Fully Implicit FEM" if has_fem else "Solver: Implicit FEM"
        elif solver_name == "SolverImplicit":
            has_fem = hasattr(self.model, 'tri_count') and self.model.tri_count > 0
            return "Solver: Implicit (FEM)" if has_fem else "Solver: Implicit"
        elif solver_name == "SolverSemiImplicit":
            return "Solver: Semi-Implicit"
        else:
            return f"Solver: {solver_name}"
    
    # --- Strain Legends ---
    
    def _draw_legends(self, canvas):
        """Draw strain legends (top right corner)"""
        # Only show FEM legend if using implicit solver or VBD
        solver_name = self.solver.__class__.__name__
        solver_uses_fem = solver_name in ("SolverImplicit", "SolverImplicitFEM", "SolverVBD")
        
        # Get strain scales from model (computed by solver)
        spring_scale = self.model.spring_strain_scale.numpy()[0] if hasattr(self.model, 'spring_strain_scale') else 0.01
        fem_scale = self.model.fem_strain_scale.numpy()[0] if hasattr(self.model, 'fem_strain_scale') else 0.01
        
        # Delegate to renderer
        self.renderer.draw_strain_legends(
            canvas,
            spring_scale=spring_scale,
            fem_scale=fem_scale,
            show_fem=solver_uses_fem
        )
    
    # --- Rendering Finalization & Profiling ---
    
    def _finalize_render(self, canvas, t_start, t_sync, t_springs, t_particles, t_text):
        """Finalize rendering and handle profiling"""
        if self.render_mode == "human":
            t_blit_start = time.perf_counter() if self._profile_render else None
            
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            
            t_flip_start = time.perf_counter() if self._profile_render else None
            pygame.display.flip()
            t_end = time.perf_counter() if self._profile_render else None
            
            # Profiling
            if self._profile_render and t_start:
                self._update_render_profiling(t_start, t_sync, t_springs, t_particles, t_text, t_blit_start, t_flip_start, t_end)
            
            return None
        else:  # rgb_array mode
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))
    
    def _update_render_profiling(self, t_start, t_sync, t_springs, t_particles, t_text, t_blit_start, t_flip_start, t_end):
        """Update and display render profiling statistics"""
        self._render_times['sync'].append((t_sync - t_start) * 1000)
        self._render_times['springs'].append((t_springs - t_sync) * 1000 if t_springs else 0)
        self._render_times['particles'].append((t_particles - t_springs) * 1000 if t_particles else 0)
        self._render_times['text'].append((t_text - t_particles) * 1000 if t_text else 0)
        self._render_times['blit'].append((t_flip_start - t_blit_start) * 1000 if t_blit_start else 0)
        self._render_times['flip'].append((t_end - t_flip_start) * 1000 if t_flip_start else 0)
        
        self._profile_counter += 1
        if self._profile_counter >= 100:
            print("\n=== RENDER PROFILING (avg over 100 frames, ms) ===")
            for key, times in self._render_times.items():
                if times:
                    print(f"{key:12s}: {np.mean(times):6.2f} ms")
            total = sum(np.mean(times) for times in self._render_times.values() if times)
            print(f"{'TOTAL':12s}: {total:6.2f} ms ({1000/total:.1f} FPS max)")
            print("=" * 50)
            self._render_times = {k: [] for k in self._render_times.keys()}
            self._profile_counter = 0
    
    # ========================================================================
    # CLEANUP
    # ========================================================================
    
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
