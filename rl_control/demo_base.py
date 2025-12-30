#!/usr/bin/env python3
"""
Base Class for Locomotion Demos

Provides a reusable framework for soft robot locomotion demos.
Subclasses only need to implement terrain creation and custom rendering.

Uses:
- SolverImplicit from warp/solvers for physics
- SDF collision kernels from warp/world for terrain collision

Author: NBEL
License: Apache-2.0
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'warp'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'openai-gym'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'rl_locomotion'))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import warp as wp
import pygame
import argparse
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

from spring_mass_env import SpringMassEnv
from cpg import HopfCPG
from balloon_forces import BalloonForces
from pygame_renderer import Renderer
from solvers import SolverImplicit
from world import apply_sdf_boundary_with_friction_2d
from world_map import WorldMap


@dataclass
class DemoConfig:
    """Configuration for a locomotion demo."""
    # Grid
    grid_size: int = 4
    
    # Physics
    dt: float = 0.01
    device: str = 'cuda'
    spring_coeff: float = 50.0
    spring_damping: float = 0.3
    gravity: float = -0.5
    
    # CPG
    frequency: float = 4.0
    amplitude: float = 1.0
    direction: Tuple[float, float] = (1.0, 0.0)
    coupling_strength: float = 2.0
    
    # Forces
    force_scale: float = 20.0
    
    # Display
    window_width: int = 1200
    window_height: int = 400
    boxsize: float = 2.5
    
    # Simulation
    duration: float = 60.0
    
    # Robot positioning
    start_x: float = 1.0
    start_y_offset: float = 0.2
    
    # Collision
    ratchet_friction: bool = True
    restitution: float = 0.0


class DemoBase(ABC):
    """
    Abstract base class for locomotion demos.
    
    Subclasses must implement:
        - create_terrain(): Create the SDF terrain for this demo
        - get_demo_name(): Return the demo name for display
        
    Optional overrides:
        - get_info_lines(): Custom info text for HUD
        - on_step(): Called after each physics step
        - on_reset(): Called after reset
        - draw_custom(): Custom rendering on top of base scene
    
    Example:
        class MyDemo(DemoBase):
            def __init__(self, my_param: float = 1.0):
                super().__init__()
                self.my_param = my_param
            
            def create_terrain(self) -> WorldMap:
                return create_my_terrain(self.my_param)
            
            def get_demo_name(self) -> str:
                return f"My Demo (param={self.my_param})"
    """
    
    def __init__(self, config: Optional[DemoConfig] = None):
        """
        Initialize the demo base.
        
        Args:
            config: Demo configuration (uses defaults if None)
        """
        self.config = config or DemoConfig()
        
        # Will be initialized in setup()
        self.env: Optional[SpringMassEnv] = None
        self.terrain: Optional[WorldMap] = None
        self.cpg: Optional[HopfCPG] = None
        self.injector: Optional[BalloonForces] = None
        self.renderer: Optional[Renderer] = None
        self.terrain_surface: Optional[pygame.Surface] = None
        
        # SDF collision data (Warp arrays)
        self.sdf_wp = None
        self.sdf_grad_x_wp = None
        self.sdf_grad_y_wp = None
        self.sdf_resolution: float = 50.0
        self.sdf_origin_x: float = 0.0
        self.sdf_origin_y: float = 0.0
        self.sdf_width: int = 0
        self.sdf_height: int = 0
        self.forward_dir_wp = None
        
        # State tracking
        self.t: float = 0.0
        self.frame_count: int = 0
        self.paused: bool = False
        self.running: bool = True
        
        # Initial state
        self.initial_centroid_x: float = 0.0
        self.initial_centroid_y: float = 0.0
        
        # Calculated
        self.num_groups: int = 0
        self.robot_height: float = 0.0
        self.robot_size: float = 0.0
        self.world_width: float = 0.0
    
    @abstractmethod
    def create_terrain(self) -> WorldMap:
        """
        Create the terrain for this demo.
        
        Returns:
            WorldMap object for collision detection
        """
        pass
    
    @abstractmethod
    def get_demo_name(self) -> str:
        """
        Get the display name of this demo.
        
        Returns:
            String name for UI display
        """
        pass
    
    def get_info_lines(self) -> list:
        """
        Get custom info lines for HUD display.
        
        Override to add demo-specific information.
        
        Returns:
            List of (text, color) tuples
        """
        positions = self.env.state_in.particle_q.numpy()
        cx = np.mean(positions[:, 0])
        cy = np.mean(positions[:, 1])
        dx = cx - self.initial_centroid_x
        dy = cy - self.initial_centroid_y
        
        return [
            (self.get_demo_name(), (0, 0, 0)),
            (f"Time: {self.t:.1f}s", (0, 0, 0)),
            (f"X displacement: {dx:+.3f}m", (0, 0, 0)),
            (f"Y displacement: {dy:+.3f}m", (0, 0, 0)),
            (f"CPG: {self.config.frequency}Hz", (100, 100, 100)),
        ]
    
    def on_step(self, positions: np.ndarray, cpg_output: np.ndarray) -> None:
        """
        Called after each physics step.
        
        Override to add demo-specific logic (e.g., progress tracking).
        
        Args:
            positions: Current particle positions [N, 2]
            cpg_output: Current CPG output [num_groups]
        """
        pass
    
    def on_reset(self) -> None:
        """
        Called after simulation reset.
        
        Override to reset demo-specific state.
        """
        pass
    
    def draw_custom(self, canvas: pygame.Surface) -> None:
        """
        Draw custom elements on top of the scene.
        
        Override to add demo-specific visualization.
        
        Args:
            canvas: Pygame surface to draw on
        """
        pass
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics at end of simulation.
        
        Override to add demo-specific summary data.
        
        Returns:
            Dictionary of summary statistics
        """
        positions = self.env.state_in.particle_q.numpy()
        cx = np.mean(positions[:, 0])
        cy = np.mean(positions[:, 1])
        
        return {
            'duration': self.t,
            'total_displacement_x': cx - self.initial_centroid_x,
            'total_displacement_y': cy - self.initial_centroid_y,
        }
    
    def _setup_sdf_collision(self) -> None:
        """Setup SDF collision arrays for GPU kernel."""
        cfg = self.config
        
        data = self.terrain.to_numpy_arrays()
        
        self.sdf_resolution = data['resolution']
        self.sdf_origin_x = float(data['origin'][0])
        self.sdf_origin_y = float(data['origin'][1])
        self.sdf_width = data['sdf'].shape[1]
        self.sdf_height = data['sdf'].shape[0]
        
        # Create Warp arrays for GPU collision
        self.sdf_wp = wp.array2d(data['sdf'], dtype=float, device=cfg.device)
        self.sdf_grad_x_wp = wp.array2d(data['sdf_grad_x'], dtype=float, device=cfg.device)
        self.sdf_grad_y_wp = wp.array2d(data['sdf_grad_y'], dtype=float, device=cfg.device)
        
        # Forward direction for ratchet friction
        self.forward_dir_wp = wp.vec2(float(cfg.direction[0]), float(cfg.direction[1]))
        
        # Debug: check SDF range
        sdf_min = float(data['sdf'].min())
        sdf_max = float(data['sdf'].max())
        
        print(f"  SDF collision setup:")
        print(f"    Size: {self.sdf_width}x{self.sdf_height} pixels")
        print(f"    Resolution: {self.sdf_resolution} px/unit")
        print(f"    SDF range: [{sdf_min:.1f}, {sdf_max:.1f}] (negative = wall)")
        print(f"    Ratchet friction: {'enabled' if cfg.ratchet_friction else 'disabled'}")
    
    def _apply_sdf_collision(self) -> None:
        """Apply SDF collision after physics step."""
        cfg = self.config
        
        if self.sdf_wp is None:
            return
        
        wp.launch(
            kernel=apply_sdf_boundary_with_friction_2d,
            dim=self.env.model.particle_count,
            inputs=[
                self.env.state_in.particle_q,
                self.env.state_in.particle_qd,
                self.sdf_wp,
                self.sdf_grad_x_wp,
                self.sdf_grad_y_wp,
                self.sdf_resolution,
                self.sdf_origin_x,
                self.sdf_origin_y,
                self.sdf_width,
                self.sdf_height,
                cfg.restitution,
                self.forward_dir_wp,
                1 if cfg.ratchet_friction else 0,
            ],
            device=self.env.model.device,
        )
    
    def setup(self) -> None:
        """Initialize environment, terrain, CPG, and renderer."""
        cfg = self.config
        
        # Calculate robot dimensions
        particle_spacing = 1.0 / (5 - 1)  # Reference spacing
        self.robot_height = (cfg.grid_size - 1) * particle_spacing
        self.robot_size = self.robot_height
        self.world_width = cfg.boxsize * (cfg.window_width / cfg.window_height)
        self.num_groups = (cfg.grid_size - 1) ** 2
        
        print("=" * 70)
        print(self.get_demo_name())
        print("=" * 70)
        print()
        
        # Create terrain
        print("Creating terrain...")
        self.terrain = self.create_terrain()
        print(f"  World size: {self.terrain.world_size[0]:.1f} x {self.terrain.world_size[1]:.1f}")
        print()
        
        # Create environment
        print("Creating environment...")
        self.env = SpringMassEnv(
            render_mode='human',
            N=cfg.grid_size,
            dt=cfg.dt,
            spring_coeff=cfg.spring_coeff,
            spring_damping=cfg.spring_damping,
            gravity=cfg.gravity,
            boxsize=cfg.boxsize,
            device=cfg.device,
            with_fem=True,
            with_springs=True,
            window_width=cfg.window_width,
            window_height=cfg.window_height,
        )
        
        # Use standard SolverImplicit with ratchet friction
        print("Setting up solver...")
        self.env.solver = SolverImplicit(
            self.env.model,
            dt=cfg.dt,
            mass=1.0,
            preconditioner_type="diag",
            solver_type="bicgstab",
            max_iterations=30,
            tolerance=1e-3,
            ratchet_friction=cfg.ratchet_friction,
            locomotion_direction=cfg.direction,
        )
        
        # Setup SDF collision (separate from physics solver)
        self._setup_sdf_collision()
        
        self.env.reset(seed=42)
        
        # Position robot
        self._position_robot()
        
        print(f"  Grid: {cfg.grid_size}x{cfg.grid_size} = {cfg.grid_size**2} particles")
        print()
        
        # Create CPG
        self.cpg = HopfCPG(
            num_groups=self.num_groups,
            frequency=cfg.frequency,
            amplitude=cfg.amplitude,
            direction=list(cfg.direction),
            coupling_strength=cfg.coupling_strength,
            dt=cfg.dt,
        )
        print(f"CPG: {cfg.frequency} Hz, amplitude {cfg.amplitude}")
        print()
        
        # Create force injector
        self.injector = BalloonForces(
            self.env.model,
            group_size=2,
            device=cfg.device,
            force_scale=cfg.force_scale,
        )
        
        # Initialize pygame and renderer
        pygame.init()
        self.renderer = Renderer(
            window_width=cfg.window_width,
            window_height=cfg.window_height,
            boxsize=cfg.boxsize,
        )
        
        # Pre-render terrain
        self.terrain_surface = self.renderer.create_sdf_surface(
            self.terrain.sdf,
            self.terrain.resolution,
            alpha=200,
            max_distance=30.0,
        )
        
        # Store initial position
        positions = self.env.state_in.particle_q.numpy()
        self.initial_centroid_x = np.mean(positions[:, 0])
        self.initial_centroid_y = np.mean(positions[:, 1])
    
    def _position_robot(self) -> None:
        """Position robot at starting location."""
        cfg = self.config
        
        initial_pos = self.env.model.particle_q.numpy()
        initial_pos[:, 0] = initial_pos[:, 0] - np.mean(initial_pos[:, 0]) + cfg.start_x
        initial_pos[:, 1] = initial_pos[:, 1] - np.min(initial_pos[:, 1]) + cfg.start_y_offset
        
        new_pos_wp = wp.array(initial_pos, dtype=wp.vec2, 
                               device=self.env.state_in.particle_q.device)
        self.env.state_in.particle_q = new_pos_wp
        self.env.state_out.particle_q = wp.clone(new_pos_wp)
    
    def reset(self) -> None:
        """Reset simulation to initial state."""
        self.env.reset(seed=42)
        self._position_robot()
        self.cpg.reset()
        self.t = 0.0
        self.frame_count = 0
        
        positions = self.env.state_in.particle_q.numpy()
        self.initial_centroid_x = np.mean(positions[:, 0])
        self.initial_centroid_y = np.mean(positions[:, 1])
        
        self.on_reset()
        print("Reset!")
    
    def step(self) -> None:
        """Execute one simulation step."""
        # Get current positions
        current_positions = self.env.state_in.particle_q.numpy()
        self.injector.calculate_centroids(current_positions)
        
        # Get CPG output
        cpg_output = self.cpg(self.t)
        
        # Apply balloon forces based on CPG
        self.injector.reset()
        for group_id in range(self.num_groups):
            cpg_val = cpg_output[group_id]
            if abs(cpg_val) > 0.001:
                self.injector.inject(group_id, cpg_val)
        
        # Step physics
        forces = self.injector.get_array()
        action = forces.flatten().astype(np.float32)
        self.env.step(action)
        
        # Apply SDF collision after physics step
        self._apply_sdf_collision()
        
        # Update state
        new_positions = self.env.state_in.particle_q.numpy()
        self.injector.calculate_centroids(new_positions)
        
        # Demo-specific step callback
        self.on_step(new_positions, cpg_output)
        
        self.t += self.config.dt
        self.frame_count += 1
    
    def render(self) -> None:
        """Render the current frame."""
        cfg = self.config
        cpg_output = self.cpg.last_output
        
        self.env._sync_to_cpu()
        self.env._init_window()
        
        if self.env.window is None:
            return
        
        canvas = self.env._create_canvas()
        scale = cfg.window_height / cfg.boxsize
        
        # Draw terrain
        canvas.blit(self.terrain_surface, (0, 0))
        
        # Draw robot
        self.env._draw_fem_triangles(canvas, scale)
        self.env._draw_springs(canvas, scale)
        self.env._draw_particles(canvas, scale)
        
        # Draw CPG overlays
        if self.injector.centroids is not None:
            self.renderer.draw_group_centroids(canvas, self.injector.centroids)
            for gid in range(self.num_groups):
                if gid < len(self.injector.centroids):
                    self.renderer.draw_radial_force_arrows(
                        canvas,
                        self.injector.centroids[gid],
                        cpg_output[gid],
                        num_directions=4,
                    )
        
        # Draw info HUD
        font = pygame.font.Font(None, 28)
        for i, (line, color) in enumerate(self.get_info_lines()):
            text = font.render(line, True, color)
            canvas.blit(text, (10, 10 + i * 25))
        
        # Demo-specific custom drawing
        self.draw_custom(canvas)
        
        # Flip
        self.env.window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.flip()
    
    def handle_events(self) -> None:
        """Handle pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    self.running = False
                elif event.key == pygame.K_r:
                    self.reset()
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                    print("Paused" if self.paused else "Resumed")
    
    def run(self) -> Dict[str, Any]:
        """
        Run the demo simulation.
        
        Returns:
            Summary dictionary with simulation results
        """
        self.setup()
        
        print("=" * 70)
        print("SIMULATION STARTED")
        print("=" * 70)
        print("Press Q/ESC to quit, R to reset, SPACE to pause")
        print()
        
        start_time = time.time()
        
        while self.running and self.t < self.config.duration:
            self.handle_events()
            
            if self.paused:
                pygame.time.wait(50)
                continue
            
            self.step()
            self.render()
            
            # Progress output
            if self.frame_count % 100 == 0:
                fps = self.frame_count / max(time.time() - start_time, 0.01)
                positions = self.env.state_in.particle_q.numpy()
                cx = np.mean(positions[:, 0])
                cy = np.mean(positions[:, 1])
                dx = cx - self.initial_centroid_x
                print(f"t={self.t:.2f}s | X={dx:+.3f} | Y={cy:.3f} | fps={fps:.1f}")
        
        # Summary
        summary = self.get_summary()
        
        print()
        print("=" * 70)
        print("SIMULATION COMPLETE")
        print("=" * 70)
        print(f"  Duration: {summary['duration']:.2f}s")
        print(f"  Total X displacement: {summary['total_displacement_x']:+.4f}m")
        print(f"  Total Y displacement: {summary['total_displacement_y']:+.4f}m")
        print()
        
        self.env.close()
        pygame.quit()
        
        return summary
    
    @classmethod
    def add_common_args(cls, parser: argparse.ArgumentParser) -> None:
        """Add common command-line arguments to parser."""
        parser.add_argument('--grid-size', '-n', type=int, default=4,
                            help='Grid size NxN (default: 4)')
        parser.add_argument('--dt', type=float, default=0.01,
                            help='Time step (default: 0.01)')
        parser.add_argument('--device', type=str, default='cuda',
                            choices=['cuda', 'cpu'], help='Device (default: cuda)')
        parser.add_argument('--frequency', type=float, default=4.0,
                            help='CPG frequency in Hz (default: 4.0)')
        parser.add_argument('--amplitude', type=float, default=1.0,
                            help='CPG amplitude 0-1 (default: 1.0)')
        parser.add_argument('--force-scale', type=float, default=20.0,
                            help='Force scale multiplier (default: 20.0)')
        parser.add_argument('--window-width', type=int, default=1200,
                            help='Window width (default: 1200)')
        parser.add_argument('--window-height', type=int, default=400,
                            help='Window height (default: 400)')
        parser.add_argument('--boxsize', type=float, default=2.5,
                            help='Simulation box height (default: 2.5)')
        parser.add_argument('--duration', '-t', type=float, default=60.0,
                            help='Simulation duration in seconds (default: 60)')
    
    @classmethod
    def config_from_args(cls, args) -> DemoConfig:
        """Create DemoConfig from parsed arguments."""
        return DemoConfig(
            grid_size=args.grid_size,
            dt=args.dt,
            device=args.device,
            frequency=args.frequency,
            amplitude=args.amplitude,
            force_scale=args.force_scale,
            window_width=args.window_width,
            window_height=args.window_height,
            boxsize=args.boxsize,
            duration=args.duration,
        )
