# SPDX-FileCopyrightText: Copyright (c) 2025 NBEL
# SPDX-License-Identifier: Apache-2.0
#
# 2D Model class for spring mass simulations
# Adapted from newton/newton/_src/sim/model.py

import numpy as np
import warp as wp


class State:
    """
    Represents the time-varying state of a 2D simulation.
    
    Contains particle positions, velocities, and forces.
    """
    
    def __init__(self):
        self.particle_q = None    # Positions (vec2)
        self.particle_qd = None   # Velocities (vec2)
        self.particle_f = None    # Forces (vec2)


class Model:
    """
    Represents the static definition of a 2D spring-mass system.
    
    Stores all geometry, constraints, and parameters for simulation.
    Adapted from Newton's Model class for 2D systems.
    
    Key Features:
        - Particle properties (mass, position, velocity)
        - Spring network topology and properties
        - Triangulated mesh elements
        - Physical parameters (gravity, damping, etc.)
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
    
    @classmethod
    def from_tessellation(cls, json_path: str, device='cuda', boxsize: float = 3.0, scale: float = 1.0):
        """
        Create a spring-mass model from tessellation JSON file.
        
        Args:
            json_path: Path to tessellation_springs.json file
            device: Warp device ('cuda' or 'cpu')
            boxsize: Size of simulation box (vertices will be scaled to fit)
            scale: Additional scaling factor for the mesh
        
        Returns:
            Model: The initialized model with tessellation geometry
        """
        import json
        
        model = cls(device=device)
        model.boxsize = boxsize
        
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
        model.particle_count = n_particles
        
        # Scale vertices from [0,1] to fit in box
        # Center the mesh in the box
        vertices_scaled = vertices_norm * scale * boxsize
        offset = (boxsize - scale * boxsize) / 2.0
        vertices_scaled += offset
        
        # Initialize particle arrays
        vel_np = np.zeros((n_particles, 2), dtype=np.float32)
        model.particle_q = wp.array(vertices_scaled, dtype=wp.vec2, device=device)
        model.particle_qd = wp.array(vel_np, dtype=wp.vec2, device=device)
        model.particle_mass = wp.ones(n_particles, dtype=float, device=device)
        model.particle_inv_mass = wp.ones(n_particles, dtype=float, device=device)
        
        # Filter springs to remove degenerate ones (zero or near-zero length)
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
        all_springs = np.vstack([valid_boundary_springs, valid_interior_springs]) if valid_boundary_springs and valid_interior_springs else \
                      np.array(valid_boundary_springs if valid_boundary_springs else valid_interior_springs, dtype=np.int32)
        spring_indices = all_springs.flatten()
        all_lengths = boundary_lengths + interior_lengths
        
        model.spring_count = len(all_springs)
        model.spring_indices = wp.array(spring_indices, dtype=int, device=device)
        model.spring_rest_length = wp.array(np.array(all_lengths, dtype=np.float32), dtype=float, device=device)
        
        # Default spring properties (differentiate boundary from interior)
        n_boundary = len(valid_boundary_springs)
        stiffnesses = np.full(model.spring_count, 40.0, dtype=np.float32)
        if n_boundary > 0:
            stiffnesses[:n_boundary] = 60.0  # Stiffer boundary springs
        dampings = np.full(model.spring_count, 0.5, dtype=np.float32)
        
        model.spring_stiffness = wp.array(stiffnesses, dtype=float, device=device)
        model.spring_damping = wp.array(dampings, dtype=float, device=device)
        model.spring_strains = wp.zeros(model.spring_count, dtype=float, device=device)
        model.spring_strains_normalized = wp.zeros(model.spring_count, dtype=float, device=device)
        
        # Initialize adaptive normalization scales
        model.spring_strain_scale = wp.array([0.01], dtype=float, device=device)
        model.fem_strain_scale = wp.array([0.01], dtype=float, device=device)
        
        # Add triangles for FEM
        initial_tri_count = len(triangles) // 3
        model.tri_count = initial_tri_count
        model.tri_indices = wp.array(triangles, dtype=int, device=device)
        
        # Compute rest poses and areas for FEM (this may filter out degenerate triangles)
        model._compute_triangle_rest_configuration(vertices_scaled, triangles)
        
        # Set gravity
        model.set_gravity((0.0, -0.1))
        
        print(f"✓ Loaded tessellation from {json_path}")
        print(f"  - {n_particles} vertices")
        print(f"  - {len(valid_boundary_springs)} boundary springs (filtered from {len(boundary_springs)})")
        print(f"  - {len(valid_interior_springs)} interior springs (filtered from {len(interior_springs)})")
        if model.tri_count < initial_tri_count:
            print(f"  - {model.tri_count} triangles (filtered from {initial_tri_count})")
        else:
            print(f"  - {model.tri_count} triangles")
        print(f"  - {len(corner_vertices)} corner vertices protected")
        
        return model
    
    @classmethod
    def from_circle(cls, radius: float = 0.75, num_boundary: int = 16, num_rings: int = 3, 
                    device='cuda', boxsize: float = 3.0, center: tuple = None):
        """
        Create a spring-mass model from a circle with optimal Delaunay triangulation.
        
        Uses concentric rings of points for uniform, high-quality triangulation.
        Based on best practices for circle meshing with Delaunay triangulation.
        
        Args:
            radius: Circle radius in normalized space (0-1 range)
            num_boundary: Number of points on the boundary
            num_rings: Number of interior concentric rings (0 = boundary only)
            device: Warp device ('cuda' or 'cpu')
            boxsize: Size of simulation box
            center: Center of circle (cx, cy) in box coordinates, defaults to box center
        
        Returns:
            Model: The initialized model with circle geometry
        """
        from scipy.spatial import Delaunay
        
        model = cls(device=device)
        model.boxsize = boxsize
        
        # Default center to middle of box
        if center is None:
            center = (boxsize / 2.0, boxsize / 2.0)
        
        cx, cy = center
        
        print(f"Creating circle mesh:")
        print(f"  Radius: {radius}")
        print(f"  Boundary points: {num_boundary}")
        print(f"  Interior rings: {num_rings}")
        print(f"  Center: ({cx:.2f}, {cy:.2f})")
        
        # Generate points - boundary + concentric interior rings
        all_points = []
        
        # Boundary points
        angles = np.linspace(0, 2 * np.pi, num_boundary, endpoint=False)
        boundary_pts = np.c_[radius * np.cos(angles), radius * np.sin(angles)]
        all_points.append(boundary_pts)
        
        # Interior rings for optimal mesh quality
        for ring in range(1, num_rings + 1):
            ring_radius = radius * (num_rings - ring + 1) / (num_rings + 1)
            # Offset angles for better triangulation
            angle_offset = np.pi / num_boundary * ring
            num_pts_ring = max(8, int(num_boundary * ring_radius / radius))
            ring_angles = np.linspace(0, 2 * np.pi, num_pts_ring, endpoint=False) + angle_offset
            ring_pts = np.c_[ring_radius * np.cos(ring_angles), ring_radius * np.sin(ring_angles)]
            all_points.append(ring_pts)
        
        # Add center point for best triangulation
        all_points.append(np.array([[0.0, 0.0]]))
        
        # Combine all points
        points_normalized = np.vstack(all_points)
        n_points = len(points_normalized)
        
        print(f"  Generated {n_points} points ({num_boundary} boundary + {n_points - num_boundary - 1} interior + 1 center)")
        
        # Delaunay triangulation
        tri = Delaunay(points_normalized)
        triangles_raw = tri.simplices
        
        print(f"  Delaunay: {len(triangles_raw)} initial triangles")
        
        # Filter triangles - keep only those inside or on the circle
        valid_triangles = []
        for simplex in triangles_raw:
            # Check if triangle centroid is inside circle
            tri_points = points_normalized[simplex]
            centroid = np.mean(tri_points, axis=0)
            dist_from_center = np.linalg.norm(centroid)
            
            # Keep triangles with centroid inside circle (with small tolerance)
            if dist_from_center <= radius * 1.01:
                # Check area (filter degenerates)
                p0, p1, p2 = tri_points
                e1 = p1 - p0
                e2 = p2 - p0
                cross = e1[0] * e2[1] - e1[1] * e2[0]
                area = 0.5 * abs(cross)
                
                if area >= 1e-8:  # Filter degenerate triangles
                    valid_triangles.append(simplex)
        
        print(f"  Filtered: {len(valid_triangles)} valid triangles (removed {len(triangles_raw) - len(valid_triangles)})")
        
        # Scale and center in box
        points_scaled = points_normalized * radius * boxsize / 2.0  # Scale to fit in box
        points_scaled[:, 0] += cx
        points_scaled[:, 1] += cy
        
        n_particles = len(points_scaled)
        model.particle_count = n_particles
        
        # Initialize particle arrays
        vel_np = np.zeros((n_particles, 2), dtype=np.float32)
        model.particle_q = wp.array(points_scaled.astype(np.float32), dtype=wp.vec2, device=device)
        model.particle_qd = wp.array(vel_np, dtype=wp.vec2, device=device)
        model.particle_mass = wp.ones(n_particles, dtype=float, device=device)
        model.particle_inv_mass = wp.ones(n_particles, dtype=float, device=device)
        
        # Build springs from triangle edges
        edge_set = set()
        for tri in valid_triangles:
            edges = [
                tuple(sorted([tri[0], tri[1]])),
                tuple(sorted([tri[1], tri[2]])),
                tuple(sorted([tri[2], tri[0]]))
            ]
            for edge in edges:
                edge_set.add(edge)
        
        springs = list(edge_set)
        spring_indices = []
        spring_lengths = []
        
        for v0_idx, v1_idx in springs:
            spring_indices.extend([v0_idx, v1_idx])
            v0 = points_scaled[v0_idx]
            v1 = points_scaled[v1_idx]
            length = np.linalg.norm(v1 - v0)
            spring_lengths.append(length)
        
        model.spring_count = len(springs)
        model.spring_indices = wp.array(np.array(spring_indices, dtype=np.int32), dtype=int, device=device)
        model.spring_rest_length = wp.array(np.array(spring_lengths, dtype=np.float32), dtype=float, device=device)
        
        # Uniform spring properties
        model.spring_stiffness = wp.full(model.spring_count, 40.0, dtype=float, device=device)
        model.spring_damping = wp.full(model.spring_count, 0.5, dtype=float, device=device)
        model.spring_strains = wp.zeros(model.spring_count, dtype=float, device=device)
        model.spring_strains_normalized = wp.zeros(model.spring_count, dtype=float, device=device)
        
        # Initialize adaptive normalization scales
        model.spring_strain_scale = wp.array([0.01], dtype=float, device=device)
        model.fem_strain_scale = wp.array([0.01], dtype=float, device=device)
        
        # Add FEM triangles
        tri_indices_flat = []
        for tri in valid_triangles:
            tri_indices_flat.extend(tri)
        
        model.tri_count = len(valid_triangles)
        model.tri_indices = wp.array(np.array(tri_indices_flat, dtype=np.int32), dtype=int, device=device)
        
        # Compute rest configuration for FEM
        model._compute_triangle_rest_configuration(points_scaled, np.array(tri_indices_flat, dtype=np.int32))
        
        # Set gravity
        model.set_gravity((0.0, -0.1))
        
        print(f"✓ Created circle mesh:")
        print(f"  - {n_particles} particles")
        print(f"  - {model.spring_count} springs")
        print(f"  - {model.tri_count} triangles")
        print(f"  - Center: ({cx:.2f}, {cy:.2f}), Radius: {radius * boxsize / 2.0:.2f}")
        
        return model
    
    @classmethod
    def from_grid(cls, rows: int = 3, cols: int = 6, 
                  spacing: float = 0.25, device='cuda', boxsize: float = 3.0, 
                  with_fem: bool = True, with_springs: bool = True):
        """
        Create a grid spring-mass model.
        
        Args:
            rows: Number of rows (height, Y direction). Default 3.
            cols: Number of columns (width, X direction). Default 6.
            spacing: Distance between adjacent particles
            device: Warp device ('cuda' or 'cpu')
            boxsize: Size of simulation box (grid will be centered in box)
            with_fem: If True, add FEM triangles; if False, spring-only
            with_springs: If True, add springs; if False, FEM-only
        
        Returns:
            Model: The initialized model
        
        Examples:
            >>> model = Model.from_grid()                 # 6x3 default (stable)
            >>> model = Model.from_grid(rows=4, cols=4)   # 4x4 square grid
        """
        
        model = cls(device=device)
        model.boxsize = boxsize
        model.grid_rows = rows
        model.grid_cols = cols
        
        n_particles = rows * cols
        model.particle_count = n_particles
        
        # Create grid positions (centered in box)
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
        
        pos_np = np.array(positions, dtype=np.float32)
        vel_np = np.zeros((n_particles, 2), dtype=np.float32)
        
        # Initialize particle arrays
        model.particle_q = wp.array(pos_np, dtype=wp.vec2, device=device)
        model.particle_qd = wp.array(vel_np, dtype=wp.vec2, device=device)
        model.particle_mass = wp.ones(n_particles, dtype=float, device=device)
        model.particle_inv_mass = wp.ones(n_particles, dtype=float, device=device)
        
        # Create spring network (horizontal, vertical, diagonal)
        if with_springs:
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
            
            model.spring_count = len(spring_lengths)
            model.spring_indices = wp.array(np.array(spring_indices, dtype=np.int32), dtype=int, device=device)
            model.spring_rest_length = wp.array(np.array(spring_lengths, dtype=np.float32), dtype=float, device=device)
            
            # Default spring properties (uniform)
            model.spring_stiffness = wp.full(model.spring_count, 40.0, dtype=float, device=device)
            model.spring_damping = wp.full(model.spring_count, 0.5, dtype=float, device=device)
            
            # Allocate strain cache for visualization
            model.spring_strains = wp.zeros(model.spring_count, dtype=float, device=device)
            model.spring_strains_normalized = wp.zeros(model.spring_count, dtype=float, device=device)
            
            # Initialize adaptive normalization scale (will be updated by solver)
            model.spring_strain_scale = wp.array([0.01], dtype=float, device=device)  # Initial guess
            model.fem_strain_scale = wp.array([0.01], dtype=float, device=device)     # Initial guess
            
            print(f"✓ Created {model.spring_count} springs")
        else:
            # No springs
            model.spring_count = 0
            model.spring_indices = wp.array(np.array([], dtype=np.int32), dtype=int, device=device)
            model.spring_rest_length = wp.array(np.array([], dtype=np.float32), dtype=float, device=device)
            model.spring_stiffness = wp.array(np.array([], dtype=np.float32), dtype=float, device=device)
            model.spring_damping = wp.array(np.array([], dtype=np.float32), dtype=float, device=device)
            model.spring_strains = wp.array(np.array([], dtype=np.float32), dtype=float, device=device)
            model.spring_strains_normalized = wp.array(np.array([], dtype=np.float32), dtype=float, device=device)
            
            # Initialize normalization scales
            model.spring_strain_scale = wp.array([0.01], dtype=float, device=device)
            model.fem_strain_scale = wp.array([0.01], dtype=float, device=device)
            print("✓ Created model without springs (FEM-only)")
        
        # Set gravity
        model.set_gravity((0.0, -0.1))
        
        # Optionally add FEM triangles (split each grid cell into 2 triangles)
        if with_fem:
            model._add_triangles_from_grid(rows, cols)
        else:
            print("✓ Created spring-only model (no FEM)")
        
        # Print grid info
        if rows == cols:
            print(f"✓ Created {rows}x{cols} square grid = {n_particles} particles")
        else:
            print(f"✓ Created {cols}x{rows} rectangular grid = {n_particles} particles (wide x tall)")
        
        return model
    
    def _compute_triangle_rest_configuration(self, vertices: np.ndarray, tri_indices: np.ndarray):
        """
        Compute rest poses (Dm inverse) and areas for triangles.
        Filters out degenerate triangles (zero or near-zero area).
        
        Args:
            vertices: Vertex positions, shape [n_vertices, 2]
            tri_indices: Triangle indices, shape [tri_count * 3]
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
        
        # FEM material properties
        E = 50.0
        nu = 0.3
        k_damp = 2.0
        
        k_mu = E / (2.0 * (1.0 + nu))
        k_lambda = E * nu / (1.0 - nu * nu)
        
        # Use actual number of valid triangles
        valid_tri_count = len(tri_areas)
        self.tri_materials = wp.full(valid_tri_count, wp.vec3(k_mu, k_lambda, k_damp), 
                                      dtype=wp.vec3, device=self.device)
        
        # Initialize FEM strain arrays for solver computation
        self.tri_strains = wp.zeros(valid_tri_count, dtype=float, device=self.device)
        self.tri_strains_normalized = wp.zeros(valid_tri_count, dtype=float, device=self.device)
        
        avg_rest_area = np.mean(tri_areas_np) if len(tri_areas_np) > 0 else 0
        print(f"  - FEM: E={E}, nu={nu}, avg_area={avg_rest_area:.6f}")
    
    def _add_triangles_from_grid(self, rows: int, cols: int):
        """
        Add FEM triangles to a grid model.
        Each grid cell is split into 2 triangles.
        
        IMPORTANT: Diagonal pattern must match spring network (checkerboard)!
        Springs use: (r+c) % 2 == 0 → v0-v3 diagonal, else v2-v1 diagonal
        FEM must use the same pattern for alignment.
        
        Args:
            rows: Number of rows (height)
            cols: Number of columns (width)
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
        self._compute_triangle_rest_configuration(pos_np, tri_indices_np)
        
        # Report actual triangle count (may be less if degenerates were filtered)
        if self.tri_count < initial_tri_count:
            print(f"✓ Added {self.tri_count} FEM triangles (filtered from {initial_tri_count})")
        else:
            print(f"✓ Added {self.tri_count} FEM triangles to model")

