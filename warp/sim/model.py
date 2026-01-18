# SPDX-FileCopyrightText: Copyright (c) 2025 NBEL
# SPDX-License-Identifier: Apache-2.0
#
# 2D Model class for spring mass simulations
# Adapted from newton/newton/_src/sim/model.py

import numpy as np
import warp as wp

from .state import State


class Model:
    """
    Base class for 2D spring-mass systems.
    
    Stores all geometry, constraints, and parameters for simulation.
    Adapted from Newton's Model class for 2D systems.
    
    Key Features:
        - Particle properties (mass, position, velocity)
        - Spring network topology and properties
        - Triangulated mesh elements
        - Physical parameters (gravity, damping, etc.)
    
    To create models, use the model classes in warp/models/:
        - GridModel: Create rectangular grid mesh
        - CircleModel: Create circular mesh with Delaunay triangulation
        - TessellationModel: Load from tessellation JSON file
    """
    
    def __init__(self, device='cuda'):
        """
        Initialize a 2D Model object.
        
        Args:
            device (str): Device on which the Model's data will be allocated ('cuda' or 'cpu')
        """
        self.device = wp.get_device(device)
        
        # Particle properties
        self.particle_q = None              # Initial positions, shape [particle_count], vec2
        self.particle_qd = None             # Initial velocities, shape [particle_count], vec2
        self.particle_mass = None           # Particle mass, shape [particle_count], float
        self.particle_inv_mass = None       # Particle inverse mass, shape [particle_count], float
        self.particle_count = 0             # Total number of particles
        
        # Spring network properties
        self.spring_indices = None          # Spring connectivity [i0, j0, i1, j1, ...], shape [spring_count*2], int
        self.spring_rest_length = None      # Rest length per spring, shape [spring_count], float
        self.spring_stiffness = None        # Stiffness per spring, shape [spring_count], float
        self.spring_damping = None          # Damping per spring, shape [spring_count], float
        self.spring_count = 0               # Total number of springs
        self.spring_strains = None          # Raw strain values ε = (L - L₀)/L₀, shape [spring_count], float
        self.spring_strains_normalized = None  # Normalized strains in [-1, 1], shape [spring_count], float
        
        # Strain normalization parameters (computed dynamically by solver)
        self.spring_strain_scale = None     # Adaptive normalization scale [1], float
        self.fem_strain_scale = None        # Adaptive normalization scale for FEM [1], float
        
        # Triangle mesh (optional, for visualization/FEM)
        self.tri_indices = None             # Triangle vertex indices, shape [tri_count*3], int
        self.tri_count = 0                  # Total number of triangles
        self.tri_materials = None           # Material properties [tri_count], vec3(k_mu, k_lambda, k_damp)
        self.tri_poses = None               # Rest pose inverse [tri_count], mat22
        self.tri_areas = None               # Rest areas for each triangle [tri_count], float
        self.tri_strains = None             # Raw FEM strain values (combined metric), shape [tri_count], float
        self.tri_strains_normalized = None  # Normalized strains in [-1, 1], shape [tri_count], float
        
        # Physical parameters
        self.gravity = None                 # Gravity vector, shape [1], vec2
        self.dt = 0.01                      # Default timestep
        self.boxsize = 3.0                  # Simulation box size
        
    def state(self) -> State:
        """
        Create and return a new State object for this model.
        
        The returned state is initialized with the initial configuration
        from the model description.
        
        Returns:
            State: The state object
        """
        s = State()
        
        if self.particle_count > 0:
            s.particle_q = wp.clone(self.particle_q)
            s.particle_qd = wp.clone(self.particle_qd)
            s.particle_f = wp.zeros(self.particle_count, dtype=wp.vec2, device=self.device)
        
        return s
    
    def set_gravity(self, gravity):
        """
        Set gravity for runtime modification.
        
        Args:
            gravity: Gravity vector as tuple (gx, gy) or wp.vec2
                     Common values: (0.0, -9.81) for Y-down
        """
        if self.gravity is None:
            self.gravity = wp.zeros(1, dtype=wp.vec2, device=self.device)
        
        if isinstance(gravity, tuple):
            self.gravity.assign([wp.vec2(gravity[0], gravity[1])])
        else:
            self.gravity.assign([gravity])
    
    def compute_triangle_rest_configuration(self, vertices: np.ndarray, tri_indices: np.ndarray,
                                             E=50.0, nu=0.3, k_damp=2.0):
        """
        Compute rest poses (Dm inverse) and areas for triangles.
        Filters out degenerate triangles (zero or near-zero area).
        
        Args:
            vertices: Vertex positions, shape [n_vertices, 2]
            tri_indices: Triangle indices, shape [tri_count * 3]
            E: Young's modulus - scalar (uniform) or array (per-triangle)
            nu: Poisson's ratio - scalar (uniform) or array (per-triangle)
            k_damp: Damping coefficient - scalar (uniform) or array (per-triangle)
        """
        tri_count = len(tri_indices) // 3
        tri_poses = []
        tri_areas = []
        valid_indices = []
        degenerate_count = 0
        min_area_threshold = 1e-10  # Minimum valid triangle area
        
        for i in range(tri_count):
            # Get triangle vertex indices
            i0 = tri_indices[i * 3 + 0]
            i1 = tri_indices[i * 3 + 1]
            i2 = tri_indices[i * 3 + 2]
            
            # Get vertex positions
            p0 = vertices[i0]
            p1 = vertices[i1]
            p2 = vertices[i2]
            
            # Compute edge vectors in rest configuration
            e1 = p1 - p0
            e2 = p2 - p0
            
            # Compute rest area
            cross_product = e1[0] * e2[1] - e1[1] * e2[0]
            rest_area = 0.5 * abs(cross_product)
            
            # Filter out degenerate triangles
            if rest_area < min_area_threshold:
                degenerate_count += 1
                continue
            
            # Build deformation matrix Dm = [e1, e2]
            Dm = np.array([[e1[0], e2[0]],
                          [e1[1], e2[1]]], dtype=np.float32)
            
            # Compute inverse
            try:
                Dm_inv = np.linalg.inv(Dm)
                # Successfully inverted - this is a valid triangle
                tri_areas.append(rest_area)
                tri_poses.append(Dm_inv.flatten())
                valid_indices.extend([i0, i1, i2])
            except np.linalg.LinAlgError:
                # Degenerate triangle - skip it
                degenerate_count += 1
                continue
        
        if degenerate_count > 0:
            print(f"  ⚠ Filtered {degenerate_count} degenerate triangles during FEM setup")
            # Update triangle count and indices
            self.tri_count = len(tri_areas)
            self.tri_indices = wp.array(np.array(valid_indices, dtype=np.int32), dtype=int, device=self.device)
        
        # Convert to warp arrays
        tri_poses_np = np.array(tri_poses, dtype=np.float32)
        tri_areas_np = np.array(tri_areas, dtype=np.float32)
        
        self.tri_poses = wp.array(tri_poses_np, dtype=wp.mat22, device=self.device)
        self.tri_areas = wp.array(tri_areas_np, dtype=float, device=self.device)
        
        # FEM material properties - handle scalar or per-triangle arrays
        valid_tri_count = len(tri_areas)
        
        # Convert to arrays if scalar
        E_arr = np.full(valid_tri_count, E, dtype=np.float32) if np.isscalar(E) else np.asarray(E, dtype=np.float32)
        nu_arr = np.full(valid_tri_count, nu, dtype=np.float32) if np.isscalar(nu) else np.asarray(nu, dtype=np.float32)
        k_damp_arr = np.full(valid_tri_count, k_damp, dtype=np.float32) if np.isscalar(k_damp) else np.asarray(k_damp, dtype=np.float32)
        
        # Compute Lamé parameters per triangle
        k_mu_arr = E_arr / (2.0 * (1.0 + nu_arr))
        k_lambda_arr = E_arr * nu_arr / (1.0 - nu_arr * nu_arr)
        
        # Build per-triangle material array: vec3(k_mu, k_lambda, k_damp)
        materials_np = np.stack([k_mu_arr, k_lambda_arr, k_damp_arr], axis=1).astype(np.float32)
        self.tri_materials = wp.array(materials_np, dtype=wp.vec3, device=self.device)
        
        # Initialize FEM strain arrays for solver computation
        self.tri_strains = wp.zeros(valid_tri_count, dtype=float, device=self.device)
        self.tri_strains_normalized = wp.zeros(valid_tri_count, dtype=float, device=self.device)
        
        avg_rest_area = np.mean(tri_areas_np) if len(tri_areas_np) > 0 else 0
        E_info = f"E={E}" if np.isscalar(E) else f"E=array[{len(E_arr)}]"
        nu_info = f"nu={nu}" if np.isscalar(nu) else f"nu=array[{len(nu_arr)}]"
        print(f"  - FEM: {E_info}, {nu_info}, avg_area={avg_rest_area:.6f}")
    
    def add_triangles_from_grid(self, rows: int, cols: int,
                                 E: float = 50.0, nu: float = 0.3, k_damp: float = 2.0):
        """
        Add FEM triangles to a grid model.
        Each grid cell is split into 2 triangles.
        
        IMPORTANT: Diagonal pattern must match spring network (checkerboard)!
        Springs use: (r+c) % 2 == 0 → v0-v3 diagonal, else v2-v1 diagonal
        FEM must use the same pattern for alignment.
        
        Args:
            rows: Number of rows (height)
            cols: Number of columns (width)
            E: Young's modulus (uniform)
            nu: Poisson's ratio (uniform)
            k_damp: Damping coefficient (uniform)
        """
        def sub2ind(r, c):
            return r * cols + c
        
        tri_indices = []
        
        # For each grid cell, create 2 triangles
        # Use CHECKERBOARD pattern to match spring diagonals!
        for r in range(rows - 1):
            for c in range(cols - 1):
                # Get the 4 corners of the cell
                v0 = sub2ind(r, c)
                v1 = sub2ind(r, c + 1)
                v2 = sub2ind(r + 1, c)
                v3 = sub2ind(r + 1, c + 1)
                
                if (r + c) % 2 == 0:
                    # v0---v1      Spring diagonal: v0→v3 (/)
                    # | / |        FEM triangles split same way
                    # v2---v3
                    # Triangle 1: v0, v3, v2 (counter-clockwise)
                    tri_indices.extend([v0, v3, v2])
                    # Triangle 2: v0, v1, v3 (counter-clockwise)
                    tri_indices.extend([v0, v1, v3])
                else:
                    # v0---v1      Spring diagonal: v2→v1 (\)
                    # | \ |        FEM triangles split same way
                    # v2---v3
                    # Triangle 1: v0, v1, v2 (counter-clockwise)
                    tri_indices.extend([v0, v1, v2])
                    # Triangle 2: v2, v1, v3 (counter-clockwise)
                    tri_indices.extend([v2, v1, v3])
        
        initial_tri_count = (rows - 1) * (cols - 1) * 2
        self.tri_count = initial_tri_count
        tri_indices_np = np.array(tri_indices, dtype=np.int32)
        self.tri_indices = wp.array(tri_indices_np, dtype=int, device=self.device)
        
        # Use shared method to compute rest configuration (may filter degenerate triangles)
        pos_np = self.particle_q.numpy()
        self.compute_triangle_rest_configuration(pos_np, tri_indices_np, E=E, nu=nu, k_damp=k_damp)
        
        # Report actual triangle count (may be less if degenerates were filtered)
        if self.tri_count < initial_tri_count:
            print(f"✓ Added {self.tri_count} FEM triangles (filtered from {initial_tri_count})")
        else:
            print(f"✓ Added {self.tri_count} FEM triangles to model")
