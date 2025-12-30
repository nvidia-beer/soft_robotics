"""
Pygame Renderer for Spring Mass System

Consolidates rendering functionality from:
- openai-gym/spring_mass_env.py
- rl_locomotion/demo_simple_cpg.py, demo_snn_gui.py
- trajectory_tracking/tracking_env.py
- world_map/demo_world_map.py

Features:
1. FEM / Spring strain legends with diverging gradients
2. Tension visualization (springs + FEM triangles colored by strain)
3. Group centroids rendered in hot pink with labels
4. External forces rendered as arrows with magnitude-based colors
5. SDF collision world map rendering (bitmap, SDF overlay, normals)

Usage:
    from pygame_renderer import Renderer
    
    renderer = Renderer(window_width=1000, window_height=500, boxsize=2.5)
    
    # In render loop:
    canvas = renderer.create_canvas()
    renderer.draw_grid(canvas)
    renderer.draw_fem_triangles(canvas, tri_indices, positions, strains)
    renderer.draw_springs(canvas, spring_indices, positions, strains)
    renderer.draw_particles(canvas, positions)
    renderer.draw_group_centroids(canvas, centroids, group_ids)
    renderer.draw_force_arrows(canvas, centroids, forces)
    renderer.draw_strain_legends(canvas, spring_scale, fem_scale, show_fem=True)
    renderer.draw_force_legend(canvas, max_force=50.0)
    
    # World map / SDF collision:
    map_surface = renderer.create_world_map_surface(bitmap, world_size, resolution)
    sdf_surface = renderer.create_sdf_surface(sdf, resolution)
    renderer.draw_world_map(canvas, map_surface)
    renderer.draw_sdf_overlay(canvas, sdf_surface)
    renderer.draw_sdf_normals(canvas, sdf, sdf_grad_x, sdf_grad_y, world_size, resolution)
    renderer.draw_forward_direction(canvas, direction)
"""

import numpy as np
import pygame
from typing import Dict, List, Optional, Tuple, Union


class Renderer:
    """
    Pygame renderer for Spring Mass System visualization.
    
    Provides consistent rendering across all demos and environments.
    All methods work with pygame surfaces and numpy arrays.
    """
    
    # ========================================================================
    # COLOR CONSTANTS
    # ========================================================================
    
    # Standard colors
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    GREY = (100, 100, 100)
    LIGHT_GREY = (230, 230, 230)
    
    # Group/centroid colors
    HOT_PINK = (255, 105, 180)
    
    # Particle colors
    PARTICLE_FILL = (50, 50, 255)  # Blue
    PARTICLE_OUTLINE = (0, 0, 0)   # Black
    
    # Force arrow colors (gradient endpoints)
    FORCE_LOW = (150, 150, 150)    # Grey (low magnitude)
    FORCE_MID = (255, 255, 255)    # White (medium)
    FORCE_HIGH = (180, 120, 80)    # Brown (high magnitude)
    
    # Wind indicator color
    WIND_COLOR = (0, 150, 200)     # Cyan
    
    # ========================================================================
    # STRAIN COLOR PALETTES (Diverging gradients)
    # ========================================================================
    
    # Spring: Orange (compression) -> Yellow (rest) -> Red (tension)
    SPRING_COLORS = [(255, 165, 0), (255, 255, 0), (255, 0, 0)]
    
    # FEM: Light Blue (compression) -> Cyan (rest) -> Light Green (tension)
    FEM_COLORS = [(180, 220, 255), (100, 255, 255), (150, 255, 150)]
    
    # ========================================================================
    # INITIALIZATION
    # ========================================================================
    
    def __init__(
        self,
        window_width: int = 1000,
        window_height: int = 500,
        boxsize: float = 2.5,
        particle_radius: int = 5,
        particle_outline: int = 7,
        centroid_radius: int = 8,
        centroid_outline: int = 10,
        spring_min_width: int = 3,
        spring_max_width: int = 8,
        arrow_head_size: int = 8,
        arrow_line_width: int = 3,
        font_size: int = 24,
        font_size_small: int = 18,
    ):
        """
        Initialize the renderer.
        
        Args:
            window_width: Window width in pixels
            window_height: Window height in pixels
            boxsize: Simulation box size (height in world units)
            particle_radius: Particle fill circle radius
            particle_outline: Particle outline circle radius
            centroid_radius: Group centroid fill circle radius
            centroid_outline: Group centroid outline circle radius
            spring_min_width: Minimum spring line width
            spring_max_width: Maximum spring line width (high strain)
            arrow_head_size: Force arrow head size
            arrow_line_width: Force arrow line width
            font_size: Main font size
            font_size_small: Small font size for labels
        """
        self.window_width = window_width
        self.window_height = window_height
        self.boxsize = boxsize
        
        # Calculate boxsize_x based on aspect ratio
        self.boxsize_x = boxsize * (window_width / window_height)
        
        # Scale factor (pixels per world unit)
        self.scale = window_height / boxsize
        
        # Visual parameters
        self.particle_radius = particle_radius
        self.particle_outline = particle_outline
        self.centroid_radius = centroid_radius
        self.centroid_outline = centroid_outline
        self.spring_min_width = spring_min_width
        self.spring_max_width = spring_max_width
        self.arrow_head_size = arrow_head_size
        self.arrow_line_width = arrow_line_width
        
        # Fonts (initialized lazily)
        self._font = None
        self._font_small = None
        self._font_size = font_size
        self._font_size_small = font_size_small
    
    @property
    def font(self):
        """Lazy font initialization."""
        if self._font is None:
            if not pygame.font.get_init():
                pygame.font.init()
            self._font = pygame.font.Font(None, self._font_size)
        return self._font
    
    @property
    def font_small(self):
        """Lazy small font initialization."""
        if self._font_small is None:
            if not pygame.font.get_init():
                pygame.font.init()
            self._font_small = pygame.font.Font(None, self._font_size_small)
        return self._font_small
    
    # ========================================================================
    # COORDINATE CONVERSION
    # ========================================================================
    
    def world_to_screen(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to screen coordinates."""
        return (int(x * self.scale), int(self.window_height - y * self.scale))
    
    def world_to_screen_array(self, positions: np.ndarray) -> np.ndarray:
        """
        Convert array of world positions to screen coordinates.
        
        Args:
            positions: Array of shape (N, 2) with [x, y] world coordinates
            
        Returns:
            Array of shape (N, 2) with [screen_x, screen_y] pixel coordinates
        """
        screen = np.zeros_like(positions, dtype=np.int32)
        screen[:, 0] = (positions[:, 0] * self.scale).astype(int)
        screen[:, 1] = (self.window_height - positions[:, 1] * self.scale).astype(int)
        return screen
    
    # ========================================================================
    # CANVAS CREATION
    # ========================================================================
    
    def create_canvas(self, background_color=None) -> pygame.Surface:
        """
        Create a new canvas (pygame Surface) with background color.
        
        Args:
            background_color: RGB tuple or None for white
            
        Returns:
            pygame.Surface
        """
        canvas = pygame.Surface((self.window_width, self.window_height))
        canvas.fill(background_color or self.WHITE)
        return canvas
    
    # ========================================================================
    # GRID RENDERING
    # ========================================================================
    
    def draw_grid(self, canvas: pygame.Surface, color=None):
        """
        Draw background grid lines.
        
        Args:
            canvas: pygame Surface to draw on
            color: Grid line color (default: light grey)
        """
        color = color or self.LIGHT_GREY
        
        # Vertical lines
        for i in range(int(self.boxsize_x) + 1):
            x_pos = int(i * self.scale)
            pygame.draw.line(canvas, color, (x_pos, 0), (x_pos, self.window_height), 1)
        
        # Horizontal lines
        for i in range(int(self.boxsize) + 1):
            y_pos = int(i * self.scale)
            pygame.draw.line(canvas, color, (0, y_pos), (self.window_width, y_pos), 1)
    
    # ========================================================================
    # FEM TRIANGLE RENDERING
    # ========================================================================
    
    def draw_fem_triangles(
        self,
        canvas: pygame.Surface,
        tri_indices: np.ndarray,
        positions: np.ndarray,
        strains: Optional[np.ndarray] = None,
    ):
        """
        Draw FEM triangles with strain-based coloring.
        
        Args:
            canvas: pygame Surface to draw on
            tri_indices: Flat array of triangle vertex indices [i0,j0,k0, i1,j1,k1, ...]
            positions: Array of shape (N, 2) with particle positions
            strains: Normalized strain values in [-1, 1] per triangle, or None
        """
        if tri_indices is None or len(tri_indices) == 0:
            return
        
        num_triangles = len(tri_indices) // 3
        
        for i in range(num_triangles):
            # Get triangle vertices
            idx0 = tri_indices[i * 3]
            idx1 = tri_indices[i * 3 + 1]
            idx2 = tri_indices[i * 3 + 2]
            
            pos0 = positions[idx0]
            pos1 = positions[idx1]
            pos2 = positions[idx2]
            
            # Convert to screen coordinates
            screen_pos0 = self.world_to_screen(pos0[0], pos0[1])
            screen_pos1 = self.world_to_screen(pos1[0], pos1[1])
            screen_pos2 = self.world_to_screen(pos2[0], pos2[1])
            
            # Get color from strain
            if strains is not None and i < len(strains):
                fill_color, outline_color = self._get_fem_color(strains[i])
            else:
                fill_color = self.FEM_COLORS[1]  # Neutral cyan
                outline_color = tuple(max(0, c - 30) for c in fill_color)
            
            # Draw triangle
            pygame.draw.polygon(canvas, fill_color, [screen_pos0, screen_pos1, screen_pos2], 0)
            pygame.draw.polygon(canvas, outline_color, [screen_pos0, screen_pos1, screen_pos2], 1)
    
    def _get_fem_color(self, strain_normalized: float) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
        """
        Get FEM triangle color based on normalized strain.
        
        Args:
            strain_normalized: Strain in [-1, 1] (negative=compression, positive=tension)
            
        Returns:
            (fill_color, outline_color) RGB tuples
        """
        # Map [-1, 1] to [0, 1]
        t = (strain_normalized + 1.0) / 2.0
        r, g, b = self._get_diverging_color(t, 'fem')
        
        fill_color = (r, g, b)
        outline_color = (max(0, r - 30), max(0, g - 30), max(0, b - 30))
        return fill_color, outline_color
    
    # ========================================================================
    # SPRING RENDERING
    # ========================================================================
    
    def draw_springs(
        self,
        canvas: pygame.Surface,
        spring_indices: np.ndarray,
        positions: np.ndarray,
        strains: Optional[np.ndarray] = None,
    ):
        """
        Draw springs with strain-based coloring.
        
        Args:
            canvas: pygame Surface to draw on
            spring_indices: Flat array of spring endpoint indices [i0,j0, i1,j1, ...]
            positions: Array of shape (N, 2) with particle positions
            strains: Normalized strain values in [-1, 1] per spring, or None
        """
        if spring_indices is None or len(spring_indices) == 0:
            return
        
        num_springs = len(spring_indices) // 2
        
        # Get strain-based colors and thicknesses
        if strains is not None:
            colors, thicknesses = self._compute_spring_visuals(strains)
        else:
            # Default: yellow, thin lines
            colors = np.full((num_springs, 3), self.SPRING_COLORS[1], dtype=np.uint8)
            thicknesses = np.full(num_springs, self.spring_min_width, dtype=np.int32)
        
        # Vectorized position extraction
        indices_i = spring_indices[0::2]
        indices_j = spring_indices[1::2]
        
        # Convert to screen coordinates
        screen_i = self.world_to_screen_array(positions[indices_i])
        screen_j = self.world_to_screen_array(positions[indices_j])
        
        # Draw all springs
        for i in range(num_springs):
            pygame.draw.line(
                canvas, 
                tuple(colors[i]),
                tuple(screen_i[i]),
                tuple(screen_j[i]),
                int(thicknesses[i])
            )
    
    def _compute_spring_visuals(self, normalized_strains: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute spring colors and thicknesses from normalized strain data.
        
        Args:
            normalized_strains: Strain values in [-1, 1]
            
        Returns:
            colors: RGB color array shape (N, 3)
            thicknesses: Line thickness array shape (N,)
        """
        # Map [-1, 1] to [0, 1] for color interpolation
        t_values = (normalized_strains + 1.0) / 2.0
        
        # Compute colors
        colors = np.array([self._get_diverging_color(t, 'spring') for t in t_values], dtype=np.uint8)
        
        # Thickness based on absolute strain magnitude
        abs_strains = np.abs(normalized_strains)
        thickness_range = self.spring_max_width - self.spring_min_width
        thicknesses = np.clip(
            self.spring_min_width + (abs_strains * thickness_range).astype(int),
            self.spring_min_width,
            self.spring_max_width
        )
        
        return colors, thicknesses
    
    # ========================================================================
    # PARTICLE RENDERING
    # ========================================================================
    
    def draw_particles(
        self,
        canvas: pygame.Surface,
        positions: np.ndarray,
        fill_color=None,
        outline_color=None,
    ):
        """
        Draw particles as circles with outlines.
        
        Args:
            canvas: pygame Surface to draw on
            positions: Array of shape (N, 2) with particle positions
            fill_color: Particle fill color (default: blue)
            outline_color: Particle outline color (default: black)
        """
        fill_color = fill_color or self.PARTICLE_FILL
        outline_color = outline_color or self.PARTICLE_OUTLINE
        
        for pos in positions:
            if np.isnan(pos[0]) or np.isnan(pos[1]):
                continue
            
            screen_pos = self.world_to_screen(pos[0], pos[1])
            pygame.draw.circle(canvas, outline_color, screen_pos, self.particle_outline)
            pygame.draw.circle(canvas, fill_color, screen_pos, self.particle_radius)
    
    # ========================================================================
    # GROUP CENTROID RENDERING (HOT PINK)
    # ========================================================================
    
    def draw_group_centroids(
        self,
        canvas: pygame.Surface,
        centroids: np.ndarray,
        group_ids: Optional[List[int]] = None,
        show_labels: bool = True,
        fill_color=None,
        outline_color=None,
    ):
        """
        Draw group centroids as hot pink circles with optional labels.
        
        Args:
            canvas: pygame Surface to draw on
            centroids: Array of shape (num_groups, 2) with centroid positions
            group_ids: Optional list of group IDs for labeling (default: 0, 1, 2, ...)
            show_labels: Whether to show group ID labels
            fill_color: Centroid fill color (default: hot pink)
            outline_color: Centroid outline color (default: black)
        """
        if centroids is None or len(centroids) == 0:
            return
        
        fill_color = fill_color or self.HOT_PINK
        outline_color = outline_color or self.BLACK
        
        if group_ids is None:
            group_ids = list(range(len(centroids)))
        
        for i, centroid in enumerate(centroids):
            if np.isnan(centroid[0]) or np.isnan(centroid[1]):
                continue
            
            screen_pos = self.world_to_screen(centroid[0], centroid[1])
            
            # Draw hot pink centroid with black outline
            pygame.draw.circle(canvas, outline_color, screen_pos, self.centroid_outline)
            pygame.draw.circle(canvas, fill_color, screen_pos, self.centroid_radius)
            
            # Draw group label
            if show_labels and i < len(group_ids):
                label = self.font_small.render(str(group_ids[i]), True, self.WHITE)
                label_rect = label.get_rect(center=screen_pos)
                canvas.blit(label, label_rect)
    
    # ========================================================================
    # FORCE ARROW RENDERING
    # ========================================================================
    
    def draw_force_arrows(
        self,
        canvas: pygame.Surface,
        origins: np.ndarray,
        forces: np.ndarray,
        max_arrow_length: float = 40.0,
        min_magnitude: float = 0.5,
        force_scale: float = 1.0,
    ):
        """
        Draw force arrows from origins with magnitude-based coloring.
        
        Uses gradient: Grey (low) -> White (medium) -> Brown (high)
        
        Args:
            canvas: pygame Surface to draw on
            origins: Array of shape (N, 2) with arrow origin positions (world coords)
            forces: Array of shape (N, 2) with force vectors [fx, fy]
            max_arrow_length: Maximum arrow length in pixels
            min_magnitude: Minimum force magnitude to draw
            force_scale: Scale factor for forces
        """
        if origins is None or forces is None or len(origins) == 0:
            return
        
        for i in range(len(origins)):
            fx, fy = forces[i]
            magnitude = np.sqrt(fx**2 + fy**2)
            
            if magnitude < min_magnitude:
                continue
            
            # Normalize direction
            direction = np.array([fx, fy]) / magnitude
            
            # Arrow length proportional to magnitude
            arrow_length = min(max_arrow_length, max_arrow_length * magnitude / force_scale)
            
            # Get color based on magnitude
            arrow_color = self._get_force_color(magnitude, force_scale)
            
            # Screen coordinates
            origin = origins[i]
            screen_origin = self.world_to_screen(origin[0], origin[1])
            
            # Arrow end (flip y for screen coords)
            arrow_vec = direction * arrow_length
            end_x = int(screen_origin[0] + arrow_vec[0])
            end_y = int(screen_origin[1] - arrow_vec[1])
            screen_end = (end_x, end_y)
            
            # Draw arrow line
            pygame.draw.line(canvas, arrow_color, screen_origin, screen_end, self.arrow_line_width)
            
            # Draw arrow head
            if arrow_length > 8:
                self._draw_arrowhead(canvas, screen_origin, screen_end, arrow_color)
    
    def draw_radial_force_arrows(
        self,
        canvas: pygame.Surface,
        center: np.ndarray,
        magnitude: float,
        num_directions: int = 4,
        base_radius: float = 14.0,
        arrow_scale: float = 30.0,
    ):
        """
        Draw radial force arrows (balloon inflate/deflate pattern).
        
        Positive magnitude: outward arrows (inflate)
        Negative magnitude: inward arrows (deflate)
        
        Args:
            canvas: pygame Surface to draw on
            center: Center position (world coords)
            magnitude: Force magnitude (positive=outward, negative=inward)
            num_directions: Number of radial directions
            base_radius: Starting radius for arrows (pixels)
            arrow_scale: Arrow length scale factor
        """
        if abs(magnitude) < 0.01:
            return
        
        arrow_len = abs(magnitude) * arrow_scale
        if arrow_len < 5:
            return
        
        # Get color based on magnitude
        arrow_color = self._get_force_color(abs(magnitude), 1.0)
        
        # Screen center
        screen_center = self.world_to_screen(center[0], center[1])
        
        # Draw arrows in radial directions
        angles = [2 * np.pi * i / num_directions for i in range(num_directions)]
        
        for angle in angles:
            if magnitude > 0:
                # Outward arrows (inflate)
                start_x = screen_center[0] + np.cos(angle) * base_radius
                start_y = screen_center[1] - np.sin(angle) * base_radius
                end_x = screen_center[0] + np.cos(angle) * (base_radius + arrow_len)
                end_y = screen_center[1] - np.sin(angle) * (base_radius + arrow_len)
            else:
                # Inward arrows (deflate)
                end_x = screen_center[0] + np.cos(angle) * base_radius
                end_y = screen_center[1] - np.sin(angle) * base_radius
                start_x = screen_center[0] + np.cos(angle) * (base_radius + arrow_len)
                start_y = screen_center[1] - np.sin(angle) * (base_radius + arrow_len)
            
            start = (int(start_x), int(start_y))
            end = (int(end_x), int(end_y))
            
            # Draw arrow line
            pygame.draw.line(canvas, arrow_color, start, end, self.arrow_line_width)
            
            # Draw arrow head
            if arrow_len > 8:
                self._draw_arrowhead(canvas, start, end, arrow_color)
    
    def _draw_arrowhead(
        self,
        canvas: pygame.Surface,
        start: Tuple[int, int],
        end: Tuple[int, int],
        color: Tuple[int, int, int],
        size: Optional[int] = None,
    ):
        """
        Draw arrow head at end position.
        
        Args:
            canvas: pygame Surface to draw on
            start: Arrow start position (screen coords)
            end: Arrow end position (screen coords)
            color: Arrow color
            size: Head size (default: self.arrow_head_size)
        """
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = np.sqrt(dx**2 + dy**2)
        
        if length < 1:
            return
        
        # Normalize
        dx, dy = dx / length, dy / length
        
        # Perpendicular
        px, py = -dy, dx
        
        # Head points
        head_size = size if size else self.arrow_head_size
        p1 = (int(end[0] - dx * head_size + px * head_size * 0.5),
              int(end[1] - dy * head_size + py * head_size * 0.5))
        p2 = (int(end[0] - dx * head_size - px * head_size * 0.5),
              int(end[1] - dy * head_size - py * head_size * 0.5))
        
        pygame.draw.polygon(canvas, color, [end, p1, p2])
    
    def _get_force_color(self, magnitude: float, max_force: float = 50.0) -> Tuple[int, int, int]:
        """
        Get color for force arrow based on magnitude.
        
        Gradient: Grey (low) -> White (medium) -> Brown (high)
        
        Args:
            magnitude: Force magnitude
            max_force: Maximum force for normalization
            
        Returns:
            RGB color tuple
        """
        t = min(abs(magnitude) / max_force, 1.0)
        
        if t < 0.5:
            # Grey to white
            t2 = t * 2
            r = int(150 + 105 * t2)
            g = int(150 + 105 * t2)
            b = int(150 + 105 * t2)
        else:
            # White to brown
            t2 = (t - 0.5) * 2
            r = int(255 - 75 * t2)
            g = int(255 - 135 * t2)
            b = int(255 - 175 * t2)
        
        return (r, g, b)
    
    def _get_force_color_signed(self, magnitude: float, max_force: float = 50.0) -> Tuple[int, int, int]:
        """
        Get color for signed force magnitude (diverging scheme).
        
        Negative (contraction): Light Grey
        Zero (rest): White
        Positive (expansion): Light Brown
        
        Args:
            magnitude: Signed force magnitude
            max_force: Maximum force for normalization
            
        Returns:
            RGB color tuple
        """
        t = np.clip(magnitude / max_force, -1.0, 1.0)
        
        if t < 0:
            # Grey to white (contraction)
            blend = 1.0 + t  # t=-1 -> blend=0, t=0 -> blend=1
            r = int(180 + 75 * blend)
            g = int(180 + 75 * blend)
            b = int(180 + 75 * blend)
        else:
            # White to brown (expansion)
            blend = t
            r = int(255 - 75 * blend)
            g = int(255 - 115 * blend)
            b = int(255 - 155 * blend)
        
        return (r, g, b)
    
    # ========================================================================
    # WIND INDICATOR
    # ========================================================================
    
    def draw_wind_arrow(
        self,
        canvas: pygame.Surface,
        wind_force: np.ndarray,
        origin: Optional[Tuple[int, int]] = None,
    ):
        """
        Draw wind force indicator arrow.
        
        Args:
            canvas: pygame Surface to draw on
            wind_force: Wind force vector [wx, wy]
            origin: Screen position for wind arrow (default: bottom-left)
        """
        wind_mag = np.linalg.norm(wind_force)
        if wind_mag < 0.1:
            return
        
        # Default position: bottom-left
        if origin is None:
            origin = (20, self.window_height - 15)
        
        # Arrow length proportional to wind magnitude
        max_length = 30.0
        arrow_length = min(wind_mag * 2, max_length)
        
        # Wind direction
        wind_dir = wind_force / wind_mag
        
        # End point (flip y for screen coordinates)
        end_x = int(origin[0] + wind_dir[0] * arrow_length)
        end_y = int(origin[1] - wind_dir[1] * arrow_length)
        
        # Draw cyan arrow
        pygame.draw.line(canvas, self.WIND_COLOR, origin, (end_x, end_y), 3)
        
        # Arrow head
        if arrow_length > 8:
            self._draw_arrowhead(canvas, origin, (end_x, end_y), self.WIND_COLOR)
        
        # Label
        label = self.font_small.render(f"Wind [{wind_force[0]:.0f},{wind_force[1]:.0f}]", True, self.WIND_COLOR)
        canvas.blit(label, (origin[0] + 40, origin[1] - 8))
    
    # ========================================================================
    # STRAIN LEGEND RENDERING
    # ========================================================================
    
    def draw_strain_legends(
        self,
        canvas: pygame.Surface,
        spring_scale: float = 0.01,
        fem_scale: float = 0.01,
        show_fem: bool = True,
        position: Optional[Tuple[int, int]] = None,
    ):
        """
        Draw strain legends (FEM and Spring) in top-right corner.
        
        Args:
            canvas: pygame Surface to draw on
            spring_scale: Current maximum spring strain value
            fem_scale: Current maximum FEM strain value
            show_fem: Whether to show FEM legend
            position: Optional (x, y) top-left position for legends
        """
        bar_height = 60
        bar_width = 25
        spacing = 100
        
        if position is None:
            legend_start_x = self.window_width - (220 if show_fem else 120)
            legend_y = 10
        else:
            legend_start_x, legend_y = position
        
        # Draw spring strain legend
        self._draw_strain_legend(
            canvas, legend_start_x, legend_y, bar_width, bar_height,
            "Spring:", spring_scale, gradient_type='spring'
        )
        
        # Draw FEM strain legend
        if show_fem:
            fem_x = legend_start_x + spacing
            self._draw_strain_legend(
                canvas, fem_x, legend_y, bar_width, bar_height,
                "FEM:", fem_scale, gradient_type='fem'
            )
    
    def _draw_strain_legend(
        self,
        canvas: pygame.Surface,
        x: int, y: int,
        width: int, height: int,
        title: str,
        max_strain: float,
        gradient_type: str = 'spring',
    ):
        """
        Draw a single strain legend bar.
        
        Args:
            canvas: pygame Surface to draw on
            x, y: Top-left position
            width, height: Bar dimensions
            title: Legend title
            max_strain: Maximum strain value for labels
            gradient_type: 'spring' or 'fem'
        """
        canvas.blit(self.font_small.render(title, True, self.BLACK), (x, y))
        
        bar_x, bar_y = x + 5, y + 20
        
        # Draw gradient
        for y_offset in range(height):
            t = 1.0 - (y_offset / height)  # t=0 at bottom, t=1 at top
            r, g, b = self._get_diverging_color(t, gradient_type)
            pygame.draw.line(canvas, (r, g, b), (bar_x, bar_y + y_offset), (bar_x + width, bar_y + y_offset), 1)
        
        # Draw border
        pygame.draw.rect(canvas, self.BLACK, (bar_x, bar_y, width, height), 2)
        
        # Strain range labels (as percentage)
        max_pct = max_strain * 100
        label_high = f"+{max_pct:.0f}%" if max_pct >= 1 else f"+{max_pct:.1f}%"
        label_low = f"-{max_pct:.0f}%" if max_pct >= 1 else f"-{max_pct:.1f}%"
        
        canvas.blit(self.font_small.render(label_high, True, self.BLACK), (bar_x + width + 3, bar_y - 5))
        canvas.blit(self.font_small.render(label_low, True, self.BLACK), (bar_x + width + 3, bar_y + height - 10))
    
    def draw_force_legend(
        self,
        canvas: pygame.Surface,
        max_force: float = 50.0,
        position: Optional[Tuple[int, int]] = None,
        current_force: Optional[float] = None,
    ):
        """
        Draw force magnitude legend with gradient bar.
        
        Args:
            canvas: pygame Surface to draw on
            max_force: Maximum force for normalization
            position: Optional (x, y) position
            current_force: Optional current force value to display
        """
        bar_width = 25
        bar_height = 100
        
        if position is None:
            legend_x = self.window_width - 120
            legend_y = 100
        else:
            legend_x, legend_y = position
        
        # Title
        canvas.blit(self.font_small.render("Force:", True, self.BLACK), (legend_x, legend_y))
        
        bar_x = legend_x + 5
        bar_y = legend_y + 20
        
        # Draw diverging gradient
        for y_offset in range(bar_height):
            t = 1.0 - (2.0 * y_offset / bar_height)  # t from +1 to -1
            color = self._get_force_color_signed(t * max_force, max_force)
            pygame.draw.line(canvas, color, (bar_x, bar_y + y_offset), (bar_x + bar_width, bar_y + y_offset), 1)
        
        # Border
        pygame.draw.rect(canvas, self.BLACK, (bar_x, bar_y, bar_width, bar_height), 2)
        
        # Labels
        canvas.blit(self.font_small.render(f"+{max_force:.0f}", True, self.BLACK), 
                   (bar_x + bar_width + 3, bar_y - 5))
        canvas.blit(self.font_small.render("0", True, self.BLACK), 
                   (bar_x + bar_width + 3, bar_y + bar_height // 2 - 5))
        canvas.blit(self.font_small.render(f"-{max_force:.0f}", True, self.BLACK), 
                   (bar_x + bar_width + 3, bar_y + bar_height - 10))
        
        # Current force value
        if current_force is not None:
            info_y = bar_y + bar_height + 10
            canvas.blit(self.font_small.render(f"|F|={current_force:.1f}", True, self.BLACK), (legend_x, info_y))
    
    # ========================================================================
    # SHARED COLOR UTILITIES
    # ========================================================================
    
    def _get_diverging_color(self, t: float, gradient_type: str) -> Tuple[int, int, int]:
        """
        Get diverging gradient color for normalized position t in [0, 1].
        
        Three-point gradient:
            t=0.0 → Compression color
            t=0.5 → Neutral/rest color
            t=1.0 → Tension color
        
        Args:
            t: Position in gradient [0, 1]
            gradient_type: 'spring' or 'fem'
            
        Returns:
            (r, g, b) color tuple
        """
        if gradient_type == 'spring':
            colors = self.SPRING_COLORS
        else:
            colors = self.FEM_COLORS
        
        c0, c1, c2 = colors
        
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
    
    # ========================================================================
    # GROUP INFO DISPLAY
    # ========================================================================
    
    def draw_group_forces_matrix(
        self,
        canvas: pygame.Surface,
        force_values: np.ndarray,
        groups_per_side: int,
        center_group_id: Optional[int] = None,
        position: Optional[Tuple[int, int]] = None,
        title: str = "Group Forces:",
        direction: Optional[np.ndarray] = None,
        group_rows: Optional[int] = None,
    ):
        """
        Draw a matrix display of group force values with optional direction indicator.
        
        Args:
            canvas: pygame Surface to draw on
            force_values: Array of force values per group
            groups_per_side: Number of columns (groups per row)
            center_group_id: Optional center group to highlight
            position: Optional (x, y) position
            title: Matrix title
            direction: Optional 2D direction vector [dx, dy] to show inside the box
            group_rows: Optional number of rows (defaults to groups_per_side for square)
        """
        cell_size = 35
        group_cols = groups_per_side
        group_rows = group_rows if group_rows is not None else groups_per_side
        
        # Calculate panel height (taller if direction is included)
        base_height = group_rows * cell_size + 60
        panel_height = base_height + 50 if direction is not None else base_height
        
        if position is None:
            matrix_x = self.window_width - (group_cols * cell_size + 25)
            matrix_y = self.window_height - panel_height - 10
        else:
            matrix_x, matrix_y = position
        
        panel_width = group_cols * cell_size + 15
        
        # Background
        bg_rect = pygame.Rect(matrix_x, matrix_y, panel_width, panel_height)
        pygame.draw.rect(canvas, self.WHITE, bg_rect)
        pygame.draw.rect(canvas, self.GREY, bg_rect, 2)
        
        # Title
        canvas.blit(self.font.render(title.replace(":", ""), True, self.BLACK), (matrix_x + 5, matrix_y + 3))
        
        # Draw cells
        cells_x = matrix_x + 5
        matrix_start_y = matrix_y + 22
        
        num_groups = group_cols * group_rows
        
        for gid in range(min(len(force_values), num_groups)):
            row = gid // group_cols
            col = gid % group_cols
            
            val = force_values[gid]
            
            # Cell position (bottom-up to match simulation)
            cell_x = cells_x + col * cell_size
            cell_y = matrix_start_y + (group_rows - 1 - row) * cell_size
            
            # Color based on value
            cell_color = self._get_force_color(abs(val))
            
            # Draw cell
            pygame.draw.rect(canvas, cell_color, 
                           (cell_x + 1, cell_y + 1, cell_size - 2, cell_size - 2))
            pygame.draw.rect(canvas, self.GREY,
                           (cell_x + 1, cell_y + 1, cell_size - 2, cell_size - 2), 1)
            
            # Highlight center group
            if center_group_id is not None and gid == center_group_id:
                pygame.draw.rect(canvas, self.BLACK,
                               (cell_x + 1, cell_y + 1, cell_size - 2, cell_size - 2), 2)
            
            # Value text
            val_text = self.font_small.render(f"{val:+.1f}", True, self.BLACK)
            text_rect = val_text.get_rect(center=(cell_x + cell_size // 2, cell_y + cell_size // 2))
            canvas.blit(val_text, text_rect)
        
        # Legend
        legend_y = matrix_start_y + group_rows * cell_size + 3
        canvas.blit(self.font_small.render("+out/-in", True, self.BLACK), (cells_x, legend_y))
        
        # Direction indicator (inside the box, below legend)
        if direction is not None:
            dir_color = (0, 180, 220)  # Light cyan
            
            # Normalize direction
            dir_len = np.linalg.norm(direction)
            if dir_len > 1e-6:
                dir_norm = direction / dir_len
            else:
                dir_norm = np.array([1.0, 0.0])
            
            # Arrow center position (centered in panel, below legend)
            arrow_cx = cells_x + (group_cols * cell_size) // 2
            arrow_cy = legend_y + 28
            arrow_length = 25
            
            # Arrow end point
            end_x = arrow_cx + int(dir_norm[0] * arrow_length)
            end_y = arrow_cy - int(dir_norm[1] * arrow_length)  # Flip Y
            
            # Draw arrow line (bold)
            pygame.draw.line(canvas, dir_color, (arrow_cx, arrow_cy), (end_x, end_y), 4)
            
            # Draw arrowhead
            angle = np.arctan2(-dir_norm[1], dir_norm[0])
            head_len = 10
            for da in [2.5, -2.5]:
                hx = end_x - int(head_len * np.cos(angle + da))
                hy = end_y + int(head_len * np.sin(angle + da))
                pygame.draw.line(canvas, dir_color, (end_x, end_y), (hx, hy), 3)
            
            # Direction label
            dir_label = f"Dir[{direction[0]:.1f},{direction[1]:.1f}]"
            label_surface = self.font_small.render(dir_label, True, dir_color)
            label_rect = label_surface.get_rect(center=(arrow_cx, arrow_cy + 20))
            canvas.blit(label_surface, label_rect)
    
    # ========================================================================
    # UI TEXT
    # ========================================================================
    
    def draw_info_text(
        self,
        canvas: pygame.Surface,
        lines: List[Tuple[str, Tuple[int, int, int]]],
        position: Tuple[int, int] = (10, 10),
        line_spacing: int = 17,
    ):
        """
        Draw multiple lines of info text.
        
        Args:
            canvas: pygame Surface to draw on
            lines: List of (text, color) tuples
            position: Top-left position
            line_spacing: Vertical spacing between lines
        """
        x, y = position
        
        for i, (text, color) in enumerate(lines):
            text_surface = self.font_small.render(text, True, color)
            canvas.blit(text_surface, (x, y + i * line_spacing))
    
    # ========================================================================
    # WORLD MAP / SDF COLLISION RENDERING
    # ========================================================================
    
    def create_world_map_surface(
        self,
        bitmap: np.ndarray,
        world_size: Tuple[float, float],
        resolution: float,
        wall_color: Tuple[int, int, int] = (50, 50, 55),
        passable_color: Tuple[int, int, int] = (245, 245, 245),
    ) -> pygame.Surface:
        """
        Create a pygame surface from a world map bitmap.
        
        The bitmap uses convention:
            - 0 = wall (obstacle)
            - 1 = passable area
        
        Args:
            bitmap: 2D numpy array (height, width) with 0=wall, 1=passable
            world_size: (width, height) in world units
            resolution: Pixels per world unit in the bitmap
            wall_color: RGB color for walls
            passable_color: RGB color for passable areas
            
        Returns:
            pygame.Surface scaled to fit the renderer window
        """
        h, w = bitmap.shape
        
        # Calculate scale to fit in window
        scaled_w = int(w * self.scale / resolution)
        scaled_h = int(h * self.scale / resolution)
        
        # Flip Y for screen coordinates (origin at bottom-left in world)
        bitmap_flipped = np.flipud(bitmap)
        
        # Convert to RGB
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        
        walls = bitmap_flipped < 0.5
        rgb[walls, 0] = wall_color[0]
        rgb[walls, 1] = wall_color[1]
        rgb[walls, 2] = wall_color[2]
        
        passable = bitmap_flipped >= 0.5
        rgb[passable, 0] = passable_color[0]
        rgb[passable, 1] = passable_color[1]
        rgb[passable, 2] = passable_color[2]
        
        # Create and scale surface
        surf = pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))
        return pygame.transform.scale(surf, (scaled_w, scaled_h))
    
    def create_sdf_surface(
        self,
        sdf: np.ndarray,
        resolution: float,
        alpha: int = 255,
        max_distance: float = 20.0,
    ) -> pygame.Surface:
        """
        Create a pygame surface visualizing the Signed Distance Field.
        
        SDF convention:
            - Positive = inside passable area (distance to nearest wall)
            - Negative = inside wall (distance to nearest passable)
            - Zero = exactly on boundary
        
        Color scheme (brown/white gradient):
            - Brown = inside walls (negative SDF) - darker deeper into wall
            - White = passable area (positive SDF) - light grey at boundary
            - Dark brown = boundary line
        
        Args:
            sdf: 2D numpy array with signed distance values
            resolution: Pixels per world unit
            alpha: Transparency (0-255), default 255 for full opacity
            max_distance: Distance for full color saturation
            
        Returns:
            pygame.Surface scaled to fit window
        """
        h, w = sdf.shape
        
        scaled_w = int(w * self.scale / resolution)
        scaled_h = int(h * self.scale / resolution)
        
        # Flip and normalize
        sdf_flipped = np.flipud(sdf)
        sdf_norm = np.clip(sdf_flipped / max_distance, -1, 1)
        
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Walls (negative SDF) - Brown gradient: darker deeper into wall
        # Deep wall: dark brown (80, 50, 30), Surface: light brown (180, 140, 100)
        neg_mask = sdf_norm < 0
        wall_depth = -sdf_norm[neg_mask]  # 0 at surface, 1 deep inside
        rgb[neg_mask, 0] = (180 - wall_depth * 100).astype(np.uint8)  # R: 180 -> 80
        rgb[neg_mask, 1] = (140 - wall_depth * 90).astype(np.uint8)   # G: 140 -> 50
        rgb[neg_mask, 2] = (100 - wall_depth * 70).astype(np.uint8)   # B: 100 -> 30
        
        # Inside world (positive SDF, passable) - White/light grey
        # Fade from light grey at boundary to white deep inside
        pos_mask = sdf_norm >= 0
        interior_depth = sdf_norm[pos_mask]  # 0 at surface, 1 deep inside
        rgb[pos_mask, 0] = (240 + interior_depth * 15).astype(np.uint8)  # R: 240 -> 255
        rgb[pos_mask, 1] = (240 + interior_depth * 15).astype(np.uint8)  # G: 240 -> 255
        rgb[pos_mask, 2] = (240 + interior_depth * 15).astype(np.uint8)  # B: 240 -> 255
        
        # Boundary highlight (very thin line at SDF ~= 0)
        boundary_mask = np.abs(sdf_norm) < 0.05
        rgb[boundary_mask, 0] = 120
        rgb[boundary_mask, 1] = 90
        rgb[boundary_mask, 2] = 60
        
        surf = pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))
        scaled = pygame.transform.scale(surf, (scaled_w, scaled_h))
        scaled.set_alpha(alpha)
        return scaled
    
    def draw_world_map(
        self,
        canvas: pygame.Surface,
        surface: pygame.Surface,
        offset: Tuple[int, int] = (0, 0),
    ):
        """
        Draw a pre-rendered world map surface onto the canvas.
        
        Args:
            canvas: pygame Surface to draw on
            surface: Pre-rendered world map surface (from create_world_map_surface)
            offset: (x, y) pixel offset for positioning
        """
        canvas.blit(surface, offset)
    
    def draw_sdf_overlay(
        self,
        canvas: pygame.Surface,
        surface: pygame.Surface,
        offset: Tuple[int, int] = (0, 0),
    ):
        """
        Draw a pre-rendered SDF overlay onto the canvas.
        
        Args:
            canvas: pygame Surface to draw on
            surface: Pre-rendered SDF surface (from create_sdf_surface)
            offset: (x, y) pixel offset for positioning
        """
        canvas.blit(surface, offset)
    
    def draw_sdf_normals(
        self,
        canvas: pygame.Surface,
        sdf: np.ndarray,
        sdf_grad_x: np.ndarray,
        sdf_grad_y: np.ndarray,
        world_size: Tuple[float, float],
        resolution: float,
        n_samples: Tuple[int, int] = (30, 15),
        boundary_threshold: float = 0.5,
        arrow_length: int = 15,
        color: Tuple[int, int, int] = (220, 80, 80),
    ):
        """
        Draw surface normals along SDF boundaries.
        
        Normals point toward passable areas (positive SDF gradient direction).
        
        Args:
            canvas: pygame Surface to draw on
            sdf: Signed distance field array
            sdf_grad_x: X component of SDF gradient
            sdf_grad_y: Y component of SDF gradient
            world_size: (width, height) in world units
            resolution: Pixels per world unit
            n_samples: (nx, ny) number of sample points
            boundary_threshold: Draw normals where |SDF| < threshold
            arrow_length: Arrow length in pixels
            color: Arrow color
        """
        nx, ny = n_samples
        
        for i in range(nx):
            for j in range(ny):
                # World position
                x = (i + 0.5) / nx * world_size[0]
                y = (j + 0.5) / ny * world_size[1]
                
                # Get SDF value at this position
                px = int(x * resolution)
                py = int(y * resolution)
                
                if 0 <= px < sdf.shape[1] and 0 <= py < sdf.shape[0]:
                    sdf_val = sdf[py, px] / resolution  # Convert to world units
                    
                    # Only draw near boundaries
                    if abs(sdf_val) < boundary_threshold:
                        # Get gradient (normal direction)
                        gx = sdf_grad_x[py, px]
                        gy = sdf_grad_y[py, px]
                        length = np.sqrt(gx**2 + gy**2)
                        
                        if length > 1e-6:
                            # Normalize
                            nx_dir = gx / length
                            ny_dir = gy / length
                            
                            # Screen coordinates
                            sx, sy = self.world_to_screen(x, y)
                            ex = sx + int(nx_dir * arrow_length)
                            ey = sy - int(ny_dir * arrow_length)  # Flip Y
                            
                            # Draw arrow
                            pygame.draw.line(canvas, color, (sx, sy), (ex, ey), 2)
                            pygame.draw.circle(canvas, color, (ex, ey), 3)
    
    def draw_collision_particles(
        self,
        canvas: pygame.Surface,
        positions: np.ndarray,
        velocities: Optional[np.ndarray] = None,
        radius: float = 0.08,
        base_color: Tuple[int, int, int] = (230, 100, 50),
        speed_color_scale: float = 50.0,
    ):
        """
        Draw particles with optional velocity-based coloring.
        
        Args:
            canvas: pygame Surface to draw on
            positions: Array of shape (N, 2) with particle positions
            velocities: Optional array of shape (N, 2) with velocities
            radius: Particle radius in world units
            base_color: Base particle color (R, G, B)
            speed_color_scale: Speed value for full color shift
        """
        radius_px = max(3, int(radius * self.scale))
        
        for i in range(len(positions)):
            pos = positions[i]
            if np.isnan(pos[0]) or np.isnan(pos[1]):
                continue
            
            screen_pos = self.world_to_screen(pos[0], pos[1])
            
            # Color based on velocity if provided
            if velocities is not None:
                speed = np.linalg.norm(velocities[i])
                hue_shift = min(speed * speed_color_scale, 200)
                color = (
                    base_color[0],
                    max(0, int(base_color[1] - hue_shift * 0.4)),
                    base_color[2]
                )
            else:
                color = base_color
            
            pygame.draw.circle(canvas, color, screen_pos, radius_px)
            pygame.draw.circle(canvas, (40, 40, 50), screen_pos, radius_px, 1)
    
    def draw_sdf_collision_debug(
        self,
        canvas: pygame.Surface,
        positions: np.ndarray,
        sdf: np.ndarray,
        sdf_grad_x: np.ndarray,
        sdf_grad_y: np.ndarray,
        resolution: float,
        origin: Tuple[float, float] = (0.0, 0.0),
        normal_color: Tuple[int, int, int] = (0, 200, 200),  # Cyan
        arrow_length: int = 20,
        near_threshold: float = 0.15,  # Increased to show arrows on ground-touching particles
    ):
        """
        Draw debug visualization for SDF collision detection.
        
        Shows:
        - Cyan circles around particles near/touching terrain
        - Cyan arrows showing collision normals (orthogonal to SDF surface)
        
        Args:
            canvas: pygame Surface to draw on
            positions: Array of shape (N, 2) with particle world positions
            sdf: Signed distance field array
            sdf_grad_x: X gradient of SDF
            sdf_grad_y: Y gradient of SDF
            resolution: Pixels per world unit (SDF resolution)
            origin: World coordinates of SDF origin (x, y)
            normal_color: Color for collision normals (default: cyan)
            arrow_length: Length of normal arrows in pixels
            near_threshold: Distance threshold for "near collision" in world units
        """
        sdf_height, sdf_width = sdf.shape
        collision_count = 0
        near_count = 0
        
        for i, pos in enumerate(positions):
            if np.isnan(pos[0]) or np.isnan(pos[1]):
                continue
            
            # Convert world to SDF pixel coordinates
            px = (pos[0] - origin[0]) * resolution
            py = (pos[1] - origin[1]) * resolution
            
            # Check bounds
            if px < 0 or px >= sdf_width or py < 0 or py >= sdf_height:
                continue
            
            # Sample SDF with bilinear interpolation
            x0, y0 = int(px), int(py)
            x1 = min(x0 + 1, sdf_width - 1)
            y1 = min(y0 + 1, sdf_height - 1)
            fx, fy = px - x0, py - y0
            
            v00 = sdf[y0, x0]
            v01 = sdf[y0, x1]
            v10 = sdf[y1, x0]
            v11 = sdf[y1, x1]
            
            v0 = v00 * (1 - fx) + v01 * fx
            v1 = v10 * (1 - fx) + v11 * fx
            sdf_value = (v0 * (1 - fy) + v1 * fy) / resolution  # Convert to world units
            
            # Get screen position
            screen_pos = self.world_to_screen(pos[0], pos[1])
            
            # Draw cyan circle around particles near or touching boundary
            if sdf_value < 0:
                # In collision - cyan circle around particle
                collision_count += 1
                pygame.draw.circle(canvas, normal_color, screen_pos, 14, 3)  # Outer ring
                pygame.draw.circle(canvas, normal_color, screen_pos, 10, 2)  # Inner ring
                    
            elif sdf_value < near_threshold:
                # Near collision - cyan circle around particle
                near_count += 1
                pygame.draw.circle(canvas, normal_color, screen_pos, 12, 2)  # Single ring
            
            # Draw normal arrow for particles touching or near the ground
            if sdf_value < near_threshold:
                gx = sdf_grad_x[y0, x0]
                gy = sdf_grad_y[y0, x0]
                grad_len = np.sqrt(gx**2 + gy**2)
                if grad_len > 1e-6:
                    nx_dir = gx / grad_len
                    ny_dir = gy / grad_len
                    ex = screen_pos[0] + int(nx_dir * arrow_length)
                    ey = screen_pos[1] - int(ny_dir * arrow_length)  # Flip Y
                    pygame.draw.line(canvas, normal_color, screen_pos, (ex, ey), 3)
                    pygame.draw.circle(canvas, normal_color, (ex, ey), 4)
        
    
    def draw_forward_direction(
        self,
        canvas: pygame.Surface,
        direction: np.ndarray,
        position: Optional[Tuple[int, int]] = None,
        arrow_length: int = 30,
        circle_radius: int = 40,
        color: Optional[Tuple[int, int, int]] = None,
        label: str = "Forward",
    ):
        """
        Draw forward direction indicator (for locomotion/ratchet friction).
        
        Args:
            canvas: pygame Surface to draw on
            direction: 2D direction vector [dx, dy]
            position: Screen position (default: top-right, below strain legends)
            arrow_length: Arrow length in pixels
            circle_radius: Background circle radius
            color: Arrow color (default: light cyan)
            label: Text label below arrow
        """
        # Default position: top-right but below the strain legends (y=120)
        if position is None:
            cx, cy = self.window_width - 60, 130
        else:
            cx, cy = position
        
        # Default color: light cyan/blue
        if color is None:
            color = (0, 180, 220)  # Light cyan
        
        # Normalize direction
        length = np.linalg.norm(direction)
        if length > 1e-6:
            direction = direction / length
        else:
            direction = np.array([1.0, 0.0])
        
        # Draw circle background (bolder - thicker outline)
        pygame.draw.circle(canvas, color, (cx, cy), circle_radius, 3)
        
        # Draw arrow (bolder - thicker line)
        ex = cx + int(direction[0] * arrow_length)
        ey = cy - int(direction[1] * arrow_length)  # Flip Y
        
        pygame.draw.line(canvas, color, (cx, cy), (ex, ey), 5)
        
        # Arrowhead (bolder)
        angle = np.arctan2(-direction[1], direction[0])
        head_len = 12
        for da in [2.5, -2.5]:
            hx = ex - int(head_len * np.cos(angle + da))
            hy = ey + int(head_len * np.sin(angle + da))
            pygame.draw.line(canvas, color, (ex, ey), (hx, hy), 4)
        
        # Label (using same color)
        label_surface = self.font_small.render(label, True, color)
        label_rect = label_surface.get_rect(center=(cx, cy + circle_radius + 15))
        canvas.blit(label_surface, label_rect)
    
    # ========================================================================
    # WORLD MAP IMAGE LOADING
    # ========================================================================
    
    def load_world_map_image(
        self,
        image_path: str,
        threshold: int = 128,
        wall_color: Tuple[int, int, int] = (50, 50, 55),
        passable_color: Tuple[int, int, int] = (245, 245, 245),
    ) -> Tuple[pygame.Surface, np.ndarray, Tuple[float, float]]:
        """
        Load a world map from an image file and create a pygame surface.
        
        The image uses convention:
            - Dark pixels (< threshold) = walls
            - Light pixels (>= threshold) = passable areas
        
        NOTE: SDF computation and collision detection should use WorldMap
        from world_map module. This method only handles RENDERING.
        
        Args:
            image_path: Path to bitmap image (PNG, JPG, etc.)
            threshold: Pixel value threshold (< threshold = wall)
            wall_color: RGB color for walls
            passable_color: RGB color for passable areas
            
        Returns:
            Tuple of:
                - pygame.Surface: Rendered map surface (scaled to window)
                - np.ndarray: Binary bitmap array (for passing to WorldMap)
                - Tuple[float, float]: World size (width, height)
                
        Example:
            # Load image for rendering
            map_surface, bitmap, world_size = renderer.load_world_map_image('map.png')
            
            # For collision, use WorldMap from world_map module:
            from world_map import WorldMap
            world_map = WorldMap(image_path='map.png', resolution=10.0)
            
            # Render
            renderer.draw_world_map(canvas, map_surface)
        """
        from PIL import Image
        import os
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"World map image not found: {image_path}")
        
        # Load and convert to grayscale
        img = Image.open(image_path).convert('L')
        img_array = np.array(img)
        
        # Flip Y axis so origin is bottom-left (standard physics convention)
        img_array = np.flipud(img_array)
        
        # Convert to binary: 1 = passable (white), 0 = wall (black)
        bitmap = (img_array >= threshold).astype(np.float32)
        
        h, w = bitmap.shape
        
        # Calculate resolution to fit the image in the window
        # Use window dimensions to determine scale
        aspect_img = w / h
        aspect_window = self.window_width / self.window_height
        
        if aspect_window > aspect_img:
            # Window is wider - fit to height
            resolution = h / self.boxsize
        else:
            # Window is taller - fit to width
            resolution = w / self.boxsize_x
        
        world_size = (w / resolution, h / resolution)
        
        # Create the surface
        surface = self.create_world_map_surface(
            bitmap, world_size, resolution,
            wall_color=wall_color,
            passable_color=passable_color
        )
        
        return surface, bitmap, world_size
    
    def create_world_map_from_image(
        self,
        image_path: str,
        resolution: float = 10.0,
        threshold: int = 128,
    ) -> dict:
        """
        Load a world map image and prepare all data needed for rendering.
        
        Returns a dict containing:
            - bitmap: Binary numpy array
            - world_size: (width, height) in world units
            - resolution: Pixels per world unit
            - map_surface: Pre-rendered pygame surface
            
        This data can be used with:
            - draw_world_map() for rendering
            - WorldMap class for collision detection (from world_map module)
        
        Args:
            image_path: Path to bitmap image
            resolution: Pixels per world unit
            threshold: Pixel value threshold for wall detection
            
        Returns:
            Dict with rendering data
            
        Example:
            # Load map for rendering
            map_data = renderer.create_world_map_from_image('map.png', resolution=10.0)
            
            # Render in loop
            renderer.draw_world_map(canvas, map_data['map_surface'])
            
            # For collision (separate):
            from world_map import WorldMap
            world_map = WorldMap(image_path='map.png', resolution=10.0)
        """
        from PIL import Image
        import os
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"World map image not found: {image_path}")
        
        # Load and convert to grayscale
        img = Image.open(image_path).convert('L')
        img_array = np.array(img)
        
        # Flip Y for physics convention (origin at bottom-left)
        img_array = np.flipud(img_array)
        
        # Binary: 1 = passable, 0 = wall
        bitmap = (img_array >= threshold).astype(np.float32)
        
        h, w = bitmap.shape
        world_size = (w / resolution, h / resolution)
        
        # Create rendering surface
        map_surface = self.create_world_map_surface(
            bitmap, world_size, resolution
        )
        
        return {
            'bitmap': bitmap,
            'world_size': world_size,
            'resolution': resolution,
            'map_surface': map_surface,
            'image_path': image_path,
        }    
