#!/usr/bin/env python3
"""
Demo: Force Injection with Group Centroids

This demo showcases the InjectForces class with visual rendering of:
- Grid model with spring-mass system
- Group centroids (hot pink circles)
- Force vectors applied to groups
- Interactive force application patterns

Features:
- Automatic grid partitioning into groups
- Centroid visualization
- Multiple force injection patterns
- Real-time rendering with pygame

Author: NBEL
License: Apache-2.0
"""

import sys
import os
# Add warp directory to path (sibling directory)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'warp'))

import argparse
import numpy as np
import pygame
import warp as wp
import time

from sim import Model, State
from solvers import SolverSemiImplicit, SolverImplicit
from inject_forces_class import InjectForces


class InjectForcesDemo:
    """
    Demo application for force injection with centroid visualization.
    
    Similar to openai-gym rendering, but with force injection capabilities.
    """
    
    def __init__(
        self,
        N=6,
        group_size=2,
        dt=0.01,
        spring_coeff=40.0,
        spring_damping=0.5,
        gravity=-0.1,
        boxsize=2.5,
        device='cuda',
        window_width=1000,
        window_height=500,
        use_implicit=True,
    ):
        """
        Initialize the demo.
        
        Args:
            N: Grid size (NxN particles)
            group_size: Size of each group (e.g., 2 means 2x2 groups)
            dt: Physics timestep
            spring_coeff: Spring stiffness
            spring_damping: Spring damping
            gravity: Gravity strength (negative = downward)
            boxsize: Size of simulation box (height)
            device: Warp device ('cuda' or 'cpu')
            window_width: Window width in pixels
            window_height: Window height in pixels
        """
        # Initialize Warp
        wp.init()
        self.device = device
        
        # Physics parameters
        self.N = N
        self.dt = dt
        self.boxsize = boxsize  # Height of simulation box
        self.window_width = window_width
        self.window_height = window_height
        # Calculate boxsize_x based on aspect ratio to maintain uniform scale
        self.boxsize_x = boxsize * (window_width / window_height)
        
        # Create model using grid builder (SAME AS OPENAI-GYM)
        particle_spacing = 1.0 / (5 - 1)  # Reference spacing from N=5
        
        self.model = Model.from_grid(
            N=N,
            spacing=particle_spacing,
            device=device,
            boxsize=boxsize,
            with_fem=use_implicit,  # Enable FEM only for implicit solver
            with_springs=True
        )
        
        # Set custom properties
        if use_implicit:
            # Reduce spring stiffness when using FEM to avoid double-stiffness
            self.model.spring_stiffness.fill_(spring_coeff * 0.25)
        else:
            self.model.spring_stiffness.fill_(spring_coeff)
        
        self.model.spring_damping.fill_(spring_damping)
        self.model.set_gravity((0.0, gravity))
        
        # Create solver based on user choice
        if use_implicit:
            # Implicit solver (unconditionally stable, handles FEM)
            self.solver = SolverImplicit(
                self.model,
                dt=dt,
                mass=1.0,
                preconditioner_type="diag",
                solver_type="bicgstab",
                max_iterations=30,
                tolerance=1e-3
            )
        else:
            # Semi-implicit solver (fast, springs only)
            self.solver = SolverSemiImplicit(self.model)
        
        # Create force injector with group structure
        self.injector = InjectForces(self.model, group_size=group_size, device=device)
        
        # State management
        self.state_in = None
        self.state_out = None
        
        # Host arrays for rendering
        self.pos_np = None
        self.vel_np = None
        self.spring_strains_normalized_np = None
        self.tri_strains_normalized_np = None
        
        # Cache static data
        self.spring_indices_np = self.model.spring_indices.numpy()
        self.tri_indices_np = None
        if hasattr(self.model, 'tri_indices') and self.model.tri_indices is not None:
            self.tri_indices_np = self.model.tri_indices.numpy()
        
        # Pygame setup
        pygame.init()
        pygame.display.init()
        self.window = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("Force Injection Demo - Group Centroids")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.font_small = pygame.font.Font(None, 18)
        
        # Simulation state
        self.t = 0
        self.paused = False
        self.show_force_vectors = True
        
        # Force application state
        self.active_forces = {}  # {group_id: magnitude_scalar}
        self.selected_group = 0  # Currently selected group for force control
        self.force_increment = 0.5  # Increment for +/- keys
        
        # Performance tracking
        self.fps = 60
    
    def reset(self):
        """Reset the simulation to initial state."""
        # Initialize all particles at rest
        vel_zero = np.zeros((self.model.particle_count, 2), dtype=np.float32)
        self.model.particle_qd.assign(wp.array(vel_zero, dtype=wp.vec2, device=self.device))
        
        # Create states
        self.state_in = self.model.state()
        self.state_out = self.model.state()
        
        self.t = 0
        self.active_forces = {}
        
        # Clear injector forces
        self.injector.reset()
        
        print("\n=== Simulation Reset ===")
        print(f"Grid: {self.N}x{self.N} = {self.model.particle_count} particles")
        print(f"Groups: {self.injector.num_groups}")
        print(f"Device: {self.device}")
        print("========================\n")
    
    def step(self):
        """Execute one physics step."""
        if self.paused:
            return
        
        # Reset force injector
        self.injector.reset()
        
        # Apply active forces to groups
        for group_id, magnitude in self.active_forces.items():
            self.injector.inject_forces_to_group(group_id, magnitude)
        
        # Get external forces from injector
        external_forces = self.injector.get_forces_array()
        
        # Step solver
        self.solver.step(
            state_in=self.state_in,
            state_out=self.state_out,
            dt=self.dt,
            external_forces=external_forces
        )
        
        # Swap state buffers
        self.state_in, self.state_out = self.state_out, self.state_in
        self.t += self.dt
    
    def sync_to_cpu(self):
        """Synchronize GPU data to CPU for rendering."""
        wp.synchronize()
        
        # Transfer data
        self.pos_np = self.state_in.particle_q.numpy()
        self.vel_np = self.state_in.particle_qd.numpy()
        
        # Get normalized spring strains
        if hasattr(self.model, 'spring_strains_normalized') and self.model.spring_strains_normalized is not None:
            self.spring_strains_normalized_np = self.model.spring_strains_normalized.numpy()
        
        # Get normalized FEM strains
        if hasattr(self.model, 'tri_strains_normalized') and self.model.tri_strains_normalized is not None:
            self.tri_strains_normalized_np = self.model.tri_strains_normalized.numpy()
        
        # Calculate centroids
        self.injector.calculate_centroids(self.pos_np)
    
    def render(self):
        """Render the current state."""
        self.sync_to_cpu()
        
        # Clear screen
        self.window.fill((255, 255, 255))
        # Use uniform scale based on height to avoid distortion
        scale = self.window_height / self.boxsize
        
        # Draw grid
        self._draw_grid()
        
        # Draw FEM triangles (drawn first, behind everything)
        self._draw_fem_triangles(scale)
        
        # Draw springs
        self._draw_springs(scale)
        
        # Draw particles
        self._draw_particles(scale)
        
        # Draw centroids
        self._draw_centroids(scale)
        
        # Draw force vectors
        if self.show_force_vectors:
            self._draw_force_vectors(scale)
        
        # Draw UI
        self._draw_ui()
        
        # Draw force legend
        self._draw_force_legend()
        
        # Draw strain legends (spring and FEM)
        self._draw_strain_legends()
        
        # Update display
        pygame.display.flip()
        self.clock.tick(self.fps)
    
    def _draw_grid(self):
        """Draw background grid."""
        scale = self.window_height / self.boxsize
        # Vertical lines
        for i in range(int(self.boxsize_x) + 1):
            x_pos = int(i * scale)
            pygame.draw.line(self.window, (230, 230, 230), (x_pos, 0), (x_pos, self.window_height), 1)
        # Horizontal lines
        for i in range(int(self.boxsize) + 1):
            y_pos = int(i * scale)
            pygame.draw.line(self.window, (230, 230, 230), (0, y_pos), (self.window_width, y_pos), 1)
    
    def _draw_fem_triangles(self, scale):
        """
        Draw FEM triangles with strain-based coloring.
        
        Strains are pre-computed by the solver in [-1, 1] range using adaptive normalization.
        """
        if not (hasattr(self.model, 'tri_count') and self.model.tri_count > 0 and self.tri_indices_np is not None):
            return
        
        # Only draw FEM if using implicit solver
        solver_uses_fem = self.solver.__class__.__name__ == "SolverImplicit"
        
        if not solver_uses_fem:
            return
        
        # Check if we have strain data
        has_strains = self.tri_strains_normalized_np is not None and len(self.tri_strains_normalized_np) > 0
        
        if not has_strains:
            return
        
        # Draw all FEM triangles with strain-based colors
        for i in range(self.model.tri_count):
            # Get triangle vertices
            idx0, idx1, idx2 = self.tri_indices_np[i * 3:i * 3 + 3]
            pos0, pos1, pos2 = self.pos_np[idx0], self.pos_np[idx1], self.pos_np[idx2]
            
            # Convert to screen coordinates
            screen_pos0 = (int(pos0[0] * scale), int(self.window_height - pos0[1] * scale))
            screen_pos1 = (int(pos1[0] * scale), int(self.window_height - pos1[1] * scale))
            screen_pos2 = (int(pos2[0] * scale), int(self.window_height - pos2[1] * scale))
            
            # Get color from pre-computed normalized strain
            normalized_strain = self.tri_strains_normalized_np[i]
            fill_color, outline_color = self._get_fem_color(normalized_strain)
            
            # Draw triangle
            pygame.draw.polygon(self.window, fill_color, [screen_pos0, screen_pos1, screen_pos2], 0)
            pygame.draw.polygon(self.window, outline_color, [screen_pos0, screen_pos1, screen_pos2], 1)
    
    def _get_fem_color(self, strain_normalized):
        """
        Get FEM triangle color based on strain (diverging scheme).
        
        Diverging Color Scheme:
            ε_norm = -1 (compression) → Light Blue (180, 220, 255)
            ε_norm =  0 (no strain)   → Cyan       (100, 255, 255)
            ε_norm = +1 (tension)     → Light Green (150, 255, 150)
        """
        # Map [-1, 1] to [0, 1] for interpolation
        t = (strain_normalized + 1.0) / 2.0
        
        # Use shared diverging color method
        r, g, b = self._get_diverging_color(t, 'fem')
        
        fill_color = (r, g, b)
        outline_color = (max(0, r - 30), max(0, g - 30), max(0, b - 30))
        return fill_color, outline_color
    
    def _get_diverging_color(self, t, gradient_type):
        """
        Get diverging gradient color for normalized position t in [0, 1].
        
        Three-point diverging gradient:
            t=0.0 → Compression color
            t=0.5 → Neutral/rest color
            t=1.0 → Tension color
        
        Args:
            t: Position in gradient [0, 1]
            gradient_type: 'spring' or 'fem'
        
        Returns:
            (r, g, b): RGB color tuple
        """
        # Define three-point color palettes
        palettes = {
            'spring': [(255, 165, 0), (255, 255, 0), (255, 0, 0)],    # Orange → Yellow → Red
            'fem':    [(180, 220, 255), (100, 255, 255), (150, 255, 150)]  # Light Blue → Cyan → Light Green
        }
        
        c0, c1, c2 = palettes.get(gradient_type, [(200, 200, 200), (255, 255, 255), (200, 200, 200)])
        
        # Linear interpolation between color points
        if t < 0.5:
            # Interpolate between compression (c0) and rest (c1)
            t2 = t * 2  # Map [0, 0.5] to [0, 1]
            r = int(c0[0] + (c1[0] - c0[0]) * t2)
            g = int(c0[1] + (c1[1] - c0[1]) * t2)
            b = int(c0[2] + (c1[2] - c0[2]) * t2)
        else:
            # Interpolate between rest (c1) and tension (c2)
            t2 = (t - 0.5) * 2  # Map [0.5, 1] to [0, 1]
            r = int(c1[0] + (c2[0] - c1[0]) * t2)
            g = int(c1[1] + (c2[1] - c1[1]) * t2)
            b = int(c1[2] + (c2[2] - c1[2]) * t2)
        
        return r, g, b
    
    def _draw_springs(self, scale):
        """Draw springs with strain-based coloring."""
        if self.model.spring_count == 0:
            return
        
        indices_i, indices_j = self.spring_indices_np[0::2], self.spring_indices_np[1::2]
        pos_i_all, pos_j_all = self.pos_np[indices_i], self.pos_np[indices_j]
        
        # Convert to screen coordinates
        screen_pos_i = np.column_stack([
            (pos_i_all[:, 0] * scale).astype(int),
            (self.window_height - pos_i_all[:, 1] * scale).astype(int)
        ])
        screen_pos_j = np.column_stack([
            (pos_j_all[:, 0] * scale).astype(int),
            (self.window_height - pos_j_all[:, 1] * scale).astype(int)
        ])
        
        # Get strain-based colors and thicknesses
        normalized_strains = self.spring_strains_normalized_np if self.spring_strains_normalized_np is not None else np.zeros(self.model.spring_count)
        colors, thicknesses = self._compute_spring_visuals(normalized_strains)
        
        # Draw all springs
        for i in range(self.model.spring_count):
            pygame.draw.line(self.window, tuple(colors[i]), 
                           tuple(screen_pos_i[i]), 
                           tuple(screen_pos_j[i]), 
                           int(thicknesses[i]))
    
    def _compute_spring_visuals(self, normalized_strains):
        """
        Compute spring colors and thicknesses from normalized strain data.
        
        Diverging Color Scheme:
            ε_norm = -1 (compression) → Orange (255, 165, 0)
            ε_norm =  0 (no strain)   → Yellow (255, 255, 0)
            ε_norm = +1 (tension)     → Red    (255, 0, 0)
        
        Line Width:
            |ε_norm| ∈ [0, 1] → thickness [3, 8] pixels
        """
        # Map [-1, 1] → [0, 1] for color interpolation
        t_values = (normalized_strains + 1.0) / 2.0
        
        # Compute colors using shared diverging gradient
        colors = np.array([self._get_diverging_color(t, 'spring') for t in t_values])
        
        # Thickness based on absolute strain magnitude
        abs_strains = np.abs(normalized_strains)
        thicknesses = np.clip(3 + (abs_strains * 5).astype(int), 3, 8)
        
        return colors, thicknesses
    
    def _draw_particles(self, scale):
        """Draw particles as circles."""
        for i in range(self.model.particle_count):
            pos = self.pos_np[i]
            
            if np.isnan(pos[0]) or np.isnan(pos[1]):
                continue
            
            screen_pos = (int(pos[0] * scale), int(self.window_height - pos[1] * scale))
            pygame.draw.circle(self.window, (0, 0, 0), screen_pos, 7)      # Black outline
            pygame.draw.circle(self.window, (50, 50, 255), screen_pos, 5)  # Blue fill
    
    def _draw_centroids(self, scale):
        """
        Draw group centroids as hot pink circles.
        
        Similar to openai-gym rendering functionality.
        """
        if self.injector.centroids is None:
            return
        
        # Hot pink color (highly visible)
        hot_pink = (255, 105, 180)
        
        for group_id, centroid in enumerate(self.injector.centroids):
            if np.isnan(centroid[0]) or np.isnan(centroid[1]):
                continue
            
            screen_pos = (int(centroid[0] * scale), int(self.window_height - centroid[1] * scale))
            
            # Highlight if force is active on this group
            if group_id in self.active_forces:
                # Draw larger with green outline for active groups
                pygame.draw.circle(self.window, (0, 255, 0), screen_pos, 14)  # Green outline
                pygame.draw.circle(self.window, hot_pink, screen_pos, 12)     # Hot pink fill
            else:
                # Normal centroid
                pygame.draw.circle(self.window, (0, 0, 0), screen_pos, 10)    # Black outline
                pygame.draw.circle(self.window, hot_pink, screen_pos, 8)      # Hot pink fill
            
            # Draw group ID label
            label = self.font_small.render(str(group_id), True, (0, 0, 0))
            self.window.blit(label, (screen_pos[0] + 12, screen_pos[1] - 8))
    
    def _draw_force_vectors(self, scale):
        """
        Draw force vectors with normalized length and magnitude-based coloring.
        
        Color scheme:
            - Grey (128,128,128): Zero force
            - Black (0,0,0): Medium expansion/contraction
            - Brown (139,69,19): Strong expansion
            - Dark Blue (0,0,139): Strong contraction
        """
        if self.injector.centroids is None:
            return
        
        # Fixed arrow length (normalized visualization)
        arrow_length = 30.0
        
        for group_id, magnitude in self.active_forces.items():
            if group_id >= len(self.injector.centroids):
                continue
            
            centroid = self.injector.centroids[group_id]
            particle_indices = self.injector.group_info[group_id]
            
            # Get color based on force magnitude
            arrow_color = self._get_force_color(magnitude)
            
            # Draw force vector from centroid to each particle in the group
            for particle_idx in particle_indices:
                particle_pos = self.pos_np[particle_idx]
                
                if np.isnan(particle_pos[0]) or np.isnan(particle_pos[1]):
                    continue
                
                # Calculate direction from centroid to particle
                direction = particle_pos - centroid
                direction_norm = np.linalg.norm(direction)
                
                if direction_norm > 1e-6:
                    direction_normalized = direction / direction_norm
                else:
                    direction_normalized = np.array([1.0, 0.0])
                
                # For contraction (negative magnitude), reverse direction
                if magnitude < 0:
                    direction_normalized = -direction_normalized
                
                # Normalized arrow (fixed length)
                arrow_vec = direction_normalized * arrow_length
                
                # Screen coordinates
                screen_particle = (int(particle_pos[0] * scale), 
                                  int(self.window_height - particle_pos[1] * scale))
                arrow_end = (int(screen_particle[0] + arrow_vec[0]),
                           int(screen_particle[1] - arrow_vec[1]))  # Flip y for screen coords
                
                # Draw arrow from particle in force direction
                pygame.draw.line(self.window, arrow_color, screen_particle, arrow_end, 3)
                
                # Draw arrowhead
                self._draw_arrowhead(screen_particle, arrow_end, arrow_color)
    
    def _get_force_color(self, magnitude):
        """
        Get color based on force magnitude (diverging color scheme).
        
        Diverging color scheme:
            magnitude = -max:  Light Grey (180, 180, 180) - Contraction
            magnitude = 0:     White (255, 255, 255) - Rest
            magnitude = +max:  Light Brown (210, 180, 140) - Expansion
        
        Args:
            magnitude: Force magnitude (scalar)
        
        Returns:
            (r, g, b): RGB color tuple
        """
        # Define magnitude range
        max_magnitude = 5.0  # Maximum expected magnitude
        
        # Normalize magnitude to [-1, 1]
        t = np.clip(magnitude / max_magnitude, -1.0, 1.0)
        
        # Map [-1, 1] to [0, 1] for interpolation
        t_normalized = (t + 1.0) / 2.0  # t_normalized: 0 (contraction) -> 0.5 (rest) -> 1 (expansion)
        
        # Define three-point color palette
        c_contraction = (180, 180, 180)  # Light Grey (contraction)
        c_rest = (255, 255, 255)         # White (rest/zero)
        c_expansion = (210, 180, 140)    # Light Brown/Tan (expansion)
        
        # Linear interpolation between color points
        if t_normalized < 0.5:
            # Interpolate between contraction (c_contraction) and rest (c_rest)
            t2 = t_normalized * 2  # Map [0, 0.5] to [0, 1]
            r = int(c_contraction[0] + (c_rest[0] - c_contraction[0]) * t2)
            g = int(c_contraction[1] + (c_rest[1] - c_contraction[1]) * t2)
            b = int(c_contraction[2] + (c_rest[2] - c_contraction[2]) * t2)
        else:
            # Interpolate between rest (c_rest) and expansion (c_expansion)
            t2 = (t_normalized - 0.5) * 2  # Map [0.5, 1] to [0, 1]
            r = int(c_rest[0] + (c_expansion[0] - c_rest[0]) * t2)
            g = int(c_rest[1] + (c_expansion[1] - c_rest[1]) * t2)
            b = int(c_rest[2] + (c_expansion[2] - c_rest[2]) * t2)
        
        return (r, g, b)
    
    def _draw_arrowhead(self, start, end, color):
        """Draw an arrowhead at the end of a line."""
        # Calculate arrow direction
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = np.sqrt(dx**2 + dy**2)
        
        if length < 1e-6:
            return
        
        # Normalize
        dx /= length
        dy /= length
        
        # Arrowhead size
        head_length = 10
        head_width = 6
        
        # Arrowhead points
        perp_x = -dy
        perp_y = dx
        
        p1 = (int(end[0] - dx * head_length + perp_x * head_width),
              int(end[1] - dy * head_length + perp_y * head_width))
        p2 = (int(end[0] - dx * head_length - perp_x * head_width),
              int(end[1] - dy * head_length - perp_y * head_width))
        
        pygame.draw.polygon(self.window, color, [end, p1, p2])
    
    def _draw_strain_legends(self):
        """
        Draw strain legends for spring and FEM (top right corner).
        
        Similar to openai-gym strain visualization.
        """
        if not pygame.font.get_init():
            return
        
        font_small = pygame.font.Font(None, 18)
        
        # Only show FEM legend if using implicit solver
        solver_uses_fem = self.solver.__class__.__name__ == "SolverImplicit"
        
        # Layout configuration
        bar_height, bar_width, spacing = 60, 25, 100
        legend_start_x = self.window_width - (220 if solver_uses_fem else 120)
        legend_y = 10
        
        # Get strain scales from model
        spring_scale = self.model.spring_strain_scale.numpy()[0] if hasattr(self.model, 'spring_strain_scale') else 0.01
        fem_scale = self.model.fem_strain_scale.numpy()[0] if hasattr(self.model, 'fem_strain_scale') else 0.01
        
        # Draw spring strain legend
        self._draw_strain_legend_bar(
            font_small, legend_start_x, legend_y, bar_width, bar_height,
            "Spring:", spring_scale, gradient_type='spring'
        )
        
        # Draw FEM strain legend (only if implicit solver)
        if solver_uses_fem:
            fem_x = legend_start_x + spacing
            self._draw_strain_legend_bar(
                font_small, fem_x, legend_y, bar_width, bar_height,
                "FEM:", fem_scale, gradient_type='fem'
            )
    
    def _draw_strain_legend_bar(self, font, x, y, width, height, title, max_strain, gradient_type='spring'):
        """
        Draw strain legend bar with diverging gradient.
        
        Args:
            font: Pygame font object
            x, y: Top-left position
            width, height: Bar dimensions
            title: Legend title
            max_strain: Current maximum strain value
            gradient_type: 'spring' or 'fem'
        """
        self.window.blit(font.render(title, True, (0, 0, 0)), (x, y))
        
        bar_x, bar_y = x + 5, y + 20
        
        # Draw gradient
        for y_offset in range(height):
            t = 1.0 - (y_offset / height)  # t=0 at bottom, t=1 at top
            r, g, b = self._get_diverging_color(t, gradient_type)
            pygame.draw.line(self.window, (r, g, b), (bar_x, bar_y + y_offset), (bar_x + width, bar_y + y_offset), 1)
        
        pygame.draw.rect(self.window, (0, 0, 0), (bar_x, bar_y, width, height), 2)
        
        # Show strain range (convert to percentage)
        max_pct = max_strain * 100
        
        # Diverging gradients: show +max at top, -max at bottom
        label_high = f"+{max_pct:.0f}%" if max_pct >= 1 else f"+{max_pct:.1f}%"
        label_low = f"-{max_pct:.0f}%" if max_pct >= 1 else f"-{max_pct:.1f}%"
        self.window.blit(font.render(label_high, True, (0, 0, 0)), (bar_x + width + 3, bar_y - 5))
        self.window.blit(font.render(label_low, True, (0, 0, 0)), (bar_x + width + 3, bar_y + height - 10))
    
    def _draw_force_legend(self):
        """
        Draw force magnitude legend with diverging gradient.
        
        Diverging gradient (similar to FEM/spring strain legends):
            Light Grey (contraction) -> White (rest) -> Light Brown (expansion)
        """
        if not pygame.font.get_init():
            return
        
        font_small = pygame.font.Font(None, 18)
        
        # Legend layout (bottom right corner, below strain legends)
        legend_x = self.window_width - 120
        legend_y = 100  # Below the strain legends
        bar_width = 25
        bar_height = 100
        
        # Draw single diverging legend
        self._draw_force_legend_bar(
            font_small, legend_x, legend_y, bar_width, bar_height,
            "Force:", max_magnitude=5.0
        )
    
    def _draw_force_legend_bar(self, font, x, y, width, height, title, max_magnitude):
        """
        Draw force legend bar with diverging gradient.
        
        Args:
            font: Pygame font object
            x, y: Top-left position
            width, height: Bar dimensions
            title: Legend title
            max_magnitude: Maximum magnitude value
        """
        # Draw title
        self.window.blit(font.render(title, True, (0, 0, 0)), (x, y))
        
        bar_x, bar_y = x + 5, y + 20
        
        # Draw diverging gradient from bottom (-max) to top (+max)
        for y_offset in range(height):
            # t = -1 at bottom (contraction), t = +1 at top (expansion)
            # t = 0 at center (rest)
            t = 1.0 - (2.0 * y_offset / height)  # Maps [0, height] to [1, -1]
            
            # Get color for this magnitude
            magnitude = t * max_magnitude
            r, g, b = self._get_force_color(magnitude)
            
            pygame.draw.line(self.window, (r, g, b), 
                           (bar_x, bar_y + y_offset), 
                           (bar_x + width, bar_y + y_offset), 1)
        
        # Draw border
        pygame.draw.rect(self.window, (0, 0, 0), (bar_x, bar_y, width, height), 2)
        
        # Draw magnitude labels
        label_high = f"+{max_magnitude:.1f}"  # Top = expansion
        label_mid = "0.0"  # Middle = rest
        label_low = f"-{max_magnitude:.1f}"  # Bottom = contraction
        
        self.window.blit(font.render(label_high, True, (0, 0, 0)), 
                        (bar_x + width + 3, bar_y - 5))
        self.window.blit(font.render(label_mid, True, (0, 0, 0)), 
                        (bar_x + width + 3, bar_y + height // 2 - 5))
        self.window.blit(font.render(label_low, True, (0, 0, 0)), 
                        (bar_x + width + 3, bar_y + height - 10))
    
    def _draw_ui(self):
        """Draw UI text and instructions."""
        # Top left - simulation info
        y_offset = 10
        self.window.blit(self.font.render(f"Time: {self.t:.2f}s", True, (0, 0, 0)), (10, y_offset))
        y_offset += 25
        self.window.blit(self.font.render(f"Groups: {self.injector.num_groups}", True, (0, 0, 0)), (10, y_offset))
        y_offset += 25
        
        # Show selected group and its force
        current_force = self.active_forces.get(self.selected_group, 0.0)
        self.window.blit(self.font.render(f"Group {self.selected_group}: {current_force:+.1f}", True, (0, 100, 200)), (10, y_offset))
        y_offset += 25
        
        if self.paused:
            self.window.blit(self.font.render("PAUSED", True, (255, 0, 0)), (10, y_offset))
            y_offset += 25
        
        # Bottom left - instructions
        instructions = [
            "Controls:",
            "SPACE - Pause/Resume",
            "R - Reset simulation",
            "F - Toggle force vectors",
            f"1-9 - Select group (current: {self.selected_group})",
            "UP/+ - Increase force (+0.5)",
            "DOWN/- - Decrease force (-0.5)",
            "0 - Set force to zero",
            "C - Clear all forces",
        ]
        
        y_offset = self.window_height - len(instructions) * 20 - 10
        for instruction in instructions:
            self.window.blit(self.font_small.render(instruction, True, (100, 100, 100)), (10, y_offset))
            y_offset += 20
    
    def handle_input(self):
        """Handle keyboard and mouse input."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            if event.type == pygame.KEYDOWN:
                # Pause/Resume
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                    print(f"{'Paused' if self.paused else 'Resumed'}")
                
                # Reset
                elif event.key == pygame.K_r:
                    self.reset()
                
                # Toggle force vectors
                elif event.key == pygame.K_f:
                    self.show_force_vectors = not self.show_force_vectors
                    print(f"Force vectors: {'ON' if self.show_force_vectors else 'OFF'}")
                
                # Select group (1-9 keys)
                elif pygame.K_1 <= event.key <= pygame.K_9:
                    group_id = event.key - pygame.K_1
                    if group_id < self.injector.num_groups:
                        self.selected_group = group_id
                        print(f"Selected group {self.selected_group}")
                
                # Increase force (UP arrow or + key)
                elif event.key in [pygame.K_UP, pygame.K_PLUS, pygame.K_EQUALS]:
                    current_mag = self.active_forces.get(self.selected_group, 0.0)
                    new_mag = min(current_mag + self.force_increment, 5.0)  # Clamp to max +5.0
                    self.active_forces[self.selected_group] = new_mag
                    print(f"Group {self.selected_group}: force = {new_mag:.1f}")
                
                # Decrease force (DOWN arrow or - key)
                elif event.key in [pygame.K_DOWN, pygame.K_MINUS]:
                    current_mag = self.active_forces.get(self.selected_group, 0.0)
                    new_mag = max(current_mag - self.force_increment, -5.0)  # Clamp to min -5.0
                    self.active_forces[self.selected_group] = new_mag
                    print(f"Group {self.selected_group}: force = {new_mag:.1f}")
                
                # Set force to zero (0 key)
                elif event.key == pygame.K_0:
                    if self.selected_group in self.active_forces:
                        del self.active_forces[self.selected_group]
                        print(f"Group {self.selected_group}: force = 0.0 (removed)")
                    else:
                        print(f"Group {self.selected_group}: already at zero")
                
                # Clear all forces
                elif event.key == pygame.K_c:
                    self.active_forces.clear()
                    print("Cleared all forces")
                
                # Quit
                elif event.key == pygame.K_ESCAPE:
                    return False
        
        return True
    
    def run(self):
        """Main simulation loop."""
        print("\n" + "=" * 60)
        print("Force Injection Demo - Group Centroids")
        print("=" * 60)
        print(f"Grid: {self.N}x{self.N} = {self.model.particle_count} particles")
        print(f"Groups: {self.injector.num_groups} ({self.N//self.injector.group_size}x{self.N//self.injector.group_size})")
        print(f"Device: {self.device}")
        print("=" * 60)
        
        self.reset()
        
        running = True
        while running:
            # Handle input
            running = self.handle_input()
            
            # Physics step
            self.step()
            
            # Render
            self.render()
        
        pygame.quit()
        print("\nDemo finished!")


def main():
    """Run the force injection demo."""
    parser = argparse.ArgumentParser(description="Force Injection Demo with Centroids")
    parser.add_argument('--N', type=int, default=6,
                       help='Grid size (NxN particles, default: 6)')
    parser.add_argument('--group-size', type=int, default=2,
                       help='Group size (default: 2 for 2x2 groups)')
    parser.add_argument('--implicit', action='store_true', default=True,
                       help='Use implicit solver with FEM (default: True)')
    parser.add_argument('--semi-implicit', dest='implicit', action='store_false',
                       help='Use semi-implicit solver (faster, springs only)')
    parser.add_argument('--dt', type=float, default=None,
                       help='Physics timestep (auto: 0.02 for implicit, 0.01 for semi-implicit)')
    parser.add_argument('--spring-stiffness', type=float, default=40.0,
                       help='Spring stiffness (default: 40.0)')
    parser.add_argument('--spring-damping', type=float, default=0.5,
                       help='Spring damping (default: 0.5)')
    parser.add_argument('--gravity', type=float, default=-0.1,
                       help='Gravity strength (default: -0.1)')
    parser.add_argument('--window-width', type=int, default=1000,
                       help='Window width in pixels (default: 1000)')
    parser.add_argument('--window-height', type=int, default=500,
                       help='Window height in pixels (default: 500)')
    parser.add_argument('--boxsize', type=float, default=2.5,
                       help='Bounding box size (height, default: 2.5)')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'], help='Computation device')
    args = parser.parse_args()
    
    # Set default timestep based on solver
    if args.dt is None:
        args.dt = 0.02 if args.implicit else 0.01
    
    demo = InjectForcesDemo(
        N=args.N,
        group_size=args.group_size,
        dt=args.dt,
        spring_coeff=args.spring_stiffness,
        spring_damping=args.spring_damping,
        gravity=args.gravity,
        boxsize=args.boxsize,
        device=args.device,
        window_width=args.window_width,
        window_height=args.window_height,
        use_implicit=args.implicit,
    )
    
    demo.run()


if __name__ == "__main__":
    main()

