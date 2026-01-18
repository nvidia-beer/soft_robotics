# SPDX-FileCopyrightText: Copyright (c) 2025 NBEL
# SPDX-License-Identifier: Apache-2.0
#
# Circle-based spring-mass model with Delaunay triangulation

import numpy as np
import warp as wp

from sim.model import Model


class CircleModel(Model):
    """
    Spring-mass model with circular geometry using Delaunay triangulation.
    
    Creates a high-quality mesh using concentric rings of points.
    Supports both spring and FEM-based simulation.
    
    Args:
        radius: Circle radius in normalized space (0-1 range)
        num_boundary: Number of points on the boundary
        num_rings: Number of interior concentric rings (0 = boundary only)
        device: Warp device ('cuda' or 'cpu')
        boxsize: Size of simulation box
        center: Center of circle (cx, cy) in box coordinates, defaults to box center
        spring_stiffness: Spring stiffness (uniform). Default 40.0.
        spring_damping: Spring damping (uniform). Default 0.5.
        fem_E: FEM Young's modulus (uniform). Default 50.0.
        fem_nu: FEM Poisson's ratio (uniform). Default 0.3.
        fem_damping: FEM damping (uniform). Default 2.0.
    
    Example:
        >>> model = CircleModel(radius=0.75, num_boundary=16, num_rings=3)
        >>> state = model.state()
    """
    
    def __init__(self, radius: float = 0.75, num_boundary: int = 16, num_rings: int = 3,
                 device='cuda', boxsize: float = 3.0, center: tuple = None,
                 spring_stiffness: float = 40.0, spring_damping: float = 0.5,
                 fem_E: float = 50.0, fem_nu: float = 0.3, fem_damping: float = 2.0):
        
        from scipy.spatial import Delaunay
        
        super().__init__(device=device)
        self.boxsize = boxsize
        self.radius = radius
        
        # Default center to middle of box
        if center is None:
            center = (boxsize / 2.0, boxsize / 2.0)
        
        cx, cy = center
        self.center = center
        
        print(f"Creating circle mesh:")
        print(f"  Radius: {radius}")
        print(f"  Boundary points: {num_boundary}")
        print(f"  Interior rings: {num_rings}")
        print(f"  Center: ({cx:.2f}, {cy:.2f})")
        
        # Generate points
        points_normalized = self._generate_points(radius, num_boundary, num_rings)
        n_points = len(points_normalized)
        
        print(f"  Generated {n_points} points ({num_boundary} boundary + {n_points - num_boundary - 1} interior + 1 center)")
        
        # Delaunay triangulation
        tri = Delaunay(points_normalized)
        triangles_raw = tri.simplices
        
        print(f"  Delaunay: {len(triangles_raw)} initial triangles")
        
        # Filter triangles
        valid_triangles = self._filter_triangles(points_normalized, triangles_raw, radius)
        
        print(f"  Filtered: {len(valid_triangles)} valid triangles (removed {len(triangles_raw) - len(valid_triangles)})")
        
        # Scale and center in box
        points_scaled = points_normalized * radius * boxsize / 2.0
        points_scaled[:, 0] += cx
        points_scaled[:, 1] += cy
        
        # Initialize particles
        self._setup_particles(points_scaled, device)
        
        # Build springs from triangle edges
        self._setup_springs_from_triangles(points_scaled, valid_triangles,
                                           spring_stiffness, spring_damping, device)
        
        # Setup FEM triangles
        self._setup_fem_triangles(points_scaled, valid_triangles, fem_E, fem_nu, fem_damping, device)
        
        # Set gravity
        self.set_gravity((0.0, -0.1))
        
        print(f"✓ Created circle mesh:")
        print(f"  - {self.particle_count} particles")
        print(f"  - {self.spring_count} springs")
        print(f"  - {self.tri_count} triangles")
        print(f"  - Center: ({cx:.2f}, {cy:.2f}), Radius: {radius * boxsize / 2.0:.2f}")
    
    def _generate_points(self, radius, num_boundary, num_rings):
        """Generate points for circle mesh."""
        all_points = []
        
        # Boundary points
        angles = np.linspace(0, 2 * np.pi, num_boundary, endpoint=False)
        boundary_pts = np.c_[radius * np.cos(angles), radius * np.sin(angles)]
        all_points.append(boundary_pts)
        
        # Interior rings
        for ring in range(1, num_rings + 1):
            ring_radius = radius * (num_rings - ring + 1) / (num_rings + 1)
            angle_offset = np.pi / num_boundary * ring
            num_pts_ring = max(8, int(num_boundary * ring_radius / radius))
            ring_angles = np.linspace(0, 2 * np.pi, num_pts_ring, endpoint=False) + angle_offset
            ring_pts = np.c_[ring_radius * np.cos(ring_angles), ring_radius * np.sin(ring_angles)]
            all_points.append(ring_pts)
        
        # Center point
        all_points.append(np.array([[0.0, 0.0]]))
        
        return np.vstack(all_points)
    
    def _filter_triangles(self, points, triangles_raw, radius):
        """Filter triangles to keep only valid ones inside the circle."""
        valid_triangles = []
        
        for simplex in triangles_raw:
            tri_points = points[simplex]
            centroid = np.mean(tri_points, axis=0)
            dist_from_center = np.linalg.norm(centroid)
            
            if dist_from_center <= radius * 1.01:
                p0, p1, p2 = tri_points
                e1 = p1 - p0
                e2 = p2 - p0
                cross = e1[0] * e2[1] - e1[1] * e2[0]
                area = 0.5 * abs(cross)
                
                if area >= 1e-8:
                    valid_triangles.append(simplex)
        
        return valid_triangles
    
    def _setup_particles(self, points_scaled, device):
        """Initialize particle arrays."""
        n_particles = len(points_scaled)
        self.particle_count = n_particles
        
        vel_np = np.zeros((n_particles, 2), dtype=np.float32)
        self.particle_q = wp.array(points_scaled.astype(np.float32), dtype=wp.vec2, device=device)
        self.particle_qd = wp.array(vel_np, dtype=wp.vec2, device=device)
        self.particle_mass = wp.ones(n_particles, dtype=float, device=device)
        self.particle_inv_mass = wp.ones(n_particles, dtype=float, device=device)
    
    def _setup_springs_from_triangles(self, points_scaled, valid_triangles,
                                       stiffness, damping, device):
        """Build springs from triangle edges."""
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
        
        self.spring_count = len(springs)
        self.spring_indices = wp.array(np.array(spring_indices, dtype=np.int32), dtype=int, device=device)
        self.spring_rest_length = wp.array(np.array(spring_lengths, dtype=np.float32), dtype=float, device=device)
        
        self.spring_stiffness = wp.full(self.spring_count, stiffness, dtype=float, device=device)
        self.spring_damping = wp.full(self.spring_count, damping, dtype=float, device=device)
        self.spring_strains = wp.zeros(self.spring_count, dtype=float, device=device)
        self.spring_strains_normalized = wp.zeros(self.spring_count, dtype=float, device=device)
        
        self.spring_strain_scale = wp.array([0.01], dtype=float, device=device)
        self.fem_strain_scale = wp.array([0.01], dtype=float, device=device)
    
    def _setup_fem_triangles(self, points_scaled, valid_triangles, fem_E, fem_nu, fem_damping, device):
        """Setup FEM triangle data."""
        tri_indices_flat = []
        for tri in valid_triangles:
            tri_indices_flat.extend(tri)
        
        self.tri_count = len(valid_triangles)
        self.tri_indices = wp.array(np.array(tri_indices_flat, dtype=np.int32), dtype=int, device=device)
        
        self.compute_triangle_rest_configuration(points_scaled, np.array(tri_indices_flat, dtype=np.int32),
                                                  E=fem_E, nu=fem_nu, k_damp=fem_damping)
