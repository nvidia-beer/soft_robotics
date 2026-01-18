# SPDX-FileCopyrightText: Copyright (c) 2025 NBEL
# SPDX-License-Identifier: Apache-2.0
#
# Tessellation-based spring-mass model

import json
import numpy as np
import warp as wp

from sim.model import Model


class TessellationModel(Model):
    """
    Spring-mass model created from a tessellation JSON file.
    
    Loads geometry from a pre-computed tessellation with boundary and interior springs.
    Supports both spring and FEM-based simulation.
    
    Args:
        json_path: Path to tessellation_springs.json file
        device: Warp device ('cuda' or 'cpu')
        boxsize: Size of simulation box (vertices will be scaled to fit)
        scale: Additional scaling factor for the mesh
        spring_stiffness_boundary: Stiffness for boundary springs. Default 60.0.
        spring_stiffness_interior: Stiffness for interior springs. Default 40.0.
        spring_damping: Damping for all springs (uniform). Default 0.5.
        fem_E: FEM Young's modulus (uniform). Default 50.0.
        fem_nu: FEM Poisson's ratio (uniform). Default 0.3.
        fem_damping: FEM damping (uniform). Default 2.0.
    
    Example:
        >>> model = TessellationModel("mesh.json", device='cuda')
        >>> state = model.state()
    """
    
    def __init__(self, json_path: str, device='cuda', boxsize: float = 3.0, scale: float = 1.0,
                 spring_stiffness_boundary: float = 60.0, spring_stiffness_interior: float = 40.0,
                 spring_damping: float = 0.5,
                 fem_E: float = 50.0, fem_nu: float = 0.3, fem_damping: float = 2.0):
        
        super().__init__(device=device)
        self.boxsize = boxsize
        self.json_path = json_path
        
        # Load tessellation data
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Extract data
        vertices_norm = np.array(data['vertices_normalized'], dtype=np.float32)
        triangles = np.array(data['triangles'], dtype=np.int32).flatten()
        boundary_springs = np.array(data['boundary_springs'], dtype=np.int32)
        interior_springs = np.array(data['interior_springs'], dtype=np.int32)
        corner_vertices = data.get('corner_vertices', [])
        
        n_particles = len(vertices_norm)
        self.particle_count = n_particles
        
        # Scale vertices from [0,1] to fit in box
        vertices_scaled = vertices_norm * scale * boxsize
        offset = (boxsize - scale * boxsize) / 2.0
        vertices_scaled += offset
        
        # Initialize particle arrays
        vel_np = np.zeros((n_particles, 2), dtype=np.float32)
        self.particle_q = wp.array(vertices_scaled, dtype=wp.vec2, device=device)
        self.particle_qd = wp.array(vel_np, dtype=wp.vec2, device=device)
        self.particle_mass = wp.ones(n_particles, dtype=float, device=device)
        self.particle_inv_mass = wp.ones(n_particles, dtype=float, device=device)
        
        # Filter and setup springs
        self._setup_springs(vertices_scaled, boundary_springs, interior_springs,
                           spring_stiffness_boundary, spring_stiffness_interior, spring_damping)
        
        # Setup FEM triangles
        initial_tri_count = len(triangles) // 3
        self.tri_count = initial_tri_count
        self.tri_indices = wp.array(triangles, dtype=int, device=device)
        self.compute_triangle_rest_configuration(vertices_scaled, triangles,
                                                  E=fem_E, nu=fem_nu, k_damp=fem_damping)
        
        # Set gravity
        self.set_gravity((0.0, -0.1))
        
        # Print summary
        print(f"✓ Loaded tessellation from {json_path}")
        print(f"  - {n_particles} vertices")
        print(f"  - {self.spring_count} springs")
        print(f"  - {self.tri_count} triangles")
        print(f"  - {len(corner_vertices)} corner vertices protected")
    
    def _setup_springs(self, vertices_scaled, boundary_springs, interior_springs,
                       stiffness_boundary, stiffness_interior, damping):
        """Filter and setup spring arrays."""
        device = self.device
        min_spring_length = 1e-6
        
        valid_boundary_springs = []
        valid_interior_springs = []
        boundary_lengths = []
        interior_lengths = []
        degenerate_count = 0
        
        # Filter boundary springs
        for spring in boundary_springs:
            v0_idx, v1_idx = spring
            v0 = vertices_scaled[v0_idx]
            v1 = vertices_scaled[v1_idx]
            length = np.linalg.norm(v1 - v0)
            
            if length >= min_spring_length:
                valid_boundary_springs.append(spring)
                boundary_lengths.append(length)
            else:
                degenerate_count += 1
        
        # Filter interior springs
        for spring in interior_springs:
            v0_idx, v1_idx = spring
            v0 = vertices_scaled[v0_idx]
            v1 = vertices_scaled[v1_idx]
            length = np.linalg.norm(v1 - v0)
            
            if length >= min_spring_length:
                valid_interior_springs.append(spring)
                interior_lengths.append(length)
            else:
                degenerate_count += 1
        
        if degenerate_count > 0:
            print(f"  ⚠ Removed {degenerate_count} degenerate springs (length < {min_spring_length})")
        
        # Combine valid springs
        if valid_boundary_springs and valid_interior_springs:
            all_springs = np.vstack([valid_boundary_springs, valid_interior_springs])
        else:
            all_springs = np.array(valid_boundary_springs if valid_boundary_springs else valid_interior_springs, dtype=np.int32)
        
        spring_indices = all_springs.flatten()
        all_lengths = boundary_lengths + interior_lengths
        
        self.spring_count = len(all_springs)
        self.spring_indices = wp.array(spring_indices, dtype=int, device=device)
        self.spring_rest_length = wp.array(np.array(all_lengths, dtype=np.float32), dtype=float, device=device)
        
        # Spring properties (boundary vs interior stiffness)
        n_boundary = len(valid_boundary_springs)
        stiffnesses = np.full(self.spring_count, stiffness_interior, dtype=np.float32)
        if n_boundary > 0:
            stiffnesses[:n_boundary] = stiffness_boundary
        
        self.spring_stiffness = wp.array(stiffnesses, dtype=float, device=device)
        self.spring_damping = wp.full(self.spring_count, damping, dtype=float, device=device)
        self.spring_strains = wp.zeros(self.spring_count, dtype=float, device=device)
        self.spring_strains_normalized = wp.zeros(self.spring_count, dtype=float, device=device)
        
        # Initialize adaptive normalization scales
        self.spring_strain_scale = wp.array([0.01], dtype=float, device=device)
        self.fem_strain_scale = wp.array([0.01], dtype=float, device=device)
