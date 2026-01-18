# SPDX-FileCopyrightText: Copyright (c) 2025 NBEL
# SPDX-License-Identifier: Apache-2.0
#
# Grid-based spring-mass model

import numpy as np
import warp as wp

from sim.model import Model


class GridModel(Model):
    """
    Spring-mass model with rectangular grid geometry.
    
    Creates a uniform grid with horizontal, vertical, and diagonal springs.
    Supports both spring and FEM-based simulation.
    
    Args:
        rows: Number of rows (height, Y direction). Default 3.
        cols: Number of columns (width, X direction). Default 6.
        spacing: Distance between adjacent particles
        device: Warp device ('cuda' or 'cpu')
        boxsize: Size of simulation box (grid will be centered in box)
        with_fem: If True, add FEM triangles; if False, spring-only
        with_springs: If True, add springs; if False, FEM-only
        spring_stiffness: Spring stiffness (uniform). Default 40.0.
        spring_damping: Spring damping (uniform). Default 0.5.
        fem_E: FEM Young's modulus (uniform). Default 50.0.
        fem_nu: FEM Poisson's ratio (uniform). Default 0.3.
        fem_damping: FEM damping (uniform). Default 2.0.
    
    Example:
        >>> model = GridModel(rows=4, cols=4)
        >>> model = GridModel(rows=3, cols=6, spring_stiffness=100.0)
        >>> state = model.state()
    """
    
    def __init__(self, rows: int = 3, cols: int = 6,
                 spacing: float = 0.25, device='cuda', boxsize: float = 3.0,
                 with_fem: bool = True, with_springs: bool = True,
                 spring_stiffness: float = 40.0, spring_damping: float = 0.5,
                 fem_E: float = 50.0, fem_nu: float = 0.3, fem_damping: float = 2.0):
        
        super().__init__(device=device)
        self.boxsize = boxsize
        self.grid_rows = rows
        self.grid_cols = cols
        
        n_particles = rows * cols
        self.particle_count = n_particles
        
        # Create grid positions (centered in box)
        positions = self._create_grid_positions(rows, cols, spacing, boxsize)
        
        # Initialize particle arrays
        pos_np = np.array(positions, dtype=np.float32)
        vel_np = np.zeros((n_particles, 2), dtype=np.float32)
        
        self.particle_q = wp.array(pos_np, dtype=wp.vec2, device=device)
        self.particle_qd = wp.array(vel_np, dtype=wp.vec2, device=device)
        self.particle_mass = wp.ones(n_particles, dtype=float, device=device)
        self.particle_inv_mass = wp.ones(n_particles, dtype=float, device=device)
        
        # Create spring network
        if with_springs:
            self._setup_springs(rows, cols, spacing, spring_stiffness, spring_damping, device)
            print(f"✓ Created {self.spring_count} springs")
        else:
            self._setup_empty_springs(device)
            print("✓ Created model without springs (FEM-only)")
        
        # Set gravity
        self.set_gravity((0.0, -0.1))
        
        # Add FEM triangles
        if with_fem:
            self.add_triangles_from_grid(rows, cols, E=fem_E, nu=fem_nu, k_damp=fem_damping)
        else:
            print("✓ Created spring-only model (no FEM)")
        
        # Print grid info
        if rows == cols:
            print(f"✓ Created {rows}x{cols} square grid = {n_particles} particles")
        else:
            print(f"✓ Created {cols}x{rows} rectangular grid = {n_particles} particles (wide x tall)")
    
    def _create_grid_positions(self, rows, cols, spacing, boxsize):
        """Create grid positions centered in box."""
        mesh_width = spacing * (cols - 1)
        mesh_height = spacing * (rows - 1)
        offset_x = (boxsize - mesh_width) / 2.0
        offset_y = (boxsize - mesh_height) / 2.0
        
        positions = []
        for r in range(rows):
            for c in range(cols):
                x = c * spacing + offset_x
                y = r * spacing + offset_y
                positions.append([x, y])
        
        return positions
    
    def _setup_springs(self, rows, cols, spacing, stiffness, damping, device):
        """Create spring network with horizontal, vertical, and diagonal springs."""
        spring_indices = []
        spring_lengths = []
        
        def sub2ind(r, c):
            return r * cols + c
        
        # Horizontal springs
        for r in range(rows):
            for c in range(cols - 1):
                i = sub2ind(r, c)
                j = sub2ind(r, c + 1)
                spring_indices.extend([i, j])
                spring_lengths.append(spacing)
        
        # Vertical springs
        for r in range(rows - 1):
            for c in range(cols):
                i = sub2ind(r, c)
                j = sub2ind(r + 1, c)
                spring_indices.extend([i, j])
                spring_lengths.append(spacing)
        
        # Diagonal springs (checkerboard pattern)
        for r in range(rows - 1):
            for c in range(cols - 1):
                if (r + c) % 2 == 0:
                    i = sub2ind(r, c)
                    j = sub2ind(r + 1, c + 1)
                else:
                    i = sub2ind(r + 1, c)
                    j = sub2ind(r, c + 1)
                spring_indices.extend([i, j])
                spring_lengths.append(spacing * np.sqrt(2))
        
        self.spring_count = len(spring_lengths)
        self.spring_indices = wp.array(np.array(spring_indices, dtype=np.int32), dtype=int, device=device)
        self.spring_rest_length = wp.array(np.array(spring_lengths, dtype=np.float32), dtype=float, device=device)
        
        self.spring_stiffness = wp.full(self.spring_count, stiffness, dtype=float, device=device)
        self.spring_damping = wp.full(self.spring_count, damping, dtype=float, device=device)
        
        self.spring_strains = wp.zeros(self.spring_count, dtype=float, device=device)
        self.spring_strains_normalized = wp.zeros(self.spring_count, dtype=float, device=device)
        
        self.spring_strain_scale = wp.array([0.01], dtype=float, device=device)
        self.fem_strain_scale = wp.array([0.01], dtype=float, device=device)
    
    def _setup_empty_springs(self, device):
        """Setup empty spring arrays for FEM-only mode."""
        self.spring_count = 0
        self.spring_indices = wp.array(np.array([], dtype=np.int32), dtype=int, device=device)
        self.spring_rest_length = wp.array(np.array([], dtype=np.float32), dtype=float, device=device)
        self.spring_stiffness = wp.array(np.array([], dtype=np.float32), dtype=float, device=device)
        self.spring_damping = wp.array(np.array([], dtype=np.float32), dtype=float, device=device)
        self.spring_strains = wp.array(np.array([], dtype=np.float32), dtype=float, device=device)
        self.spring_strains_normalized = wp.array(np.array([], dtype=np.float32), dtype=float, device=device)
        
        self.spring_strain_scale = wp.array([0.01], dtype=float, device=device)
        self.fem_strain_scale = wp.array([0.01], dtype=float, device=device)
