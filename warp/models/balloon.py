# SPDX-FileCopyrightText: Copyright (c) 2025 NBEL
# SPDX-License-Identifier: Apache-2.0
#
# Balloon model - inflatable soft body with PID pressure control
# 
# Inflates by scaling the FEM rest configuration (rest lengths & rest poses).
# The material naturally deforms to reach its new rest state.

import numpy as np
import warp as wp
from scipy.spatial import Delaunay

from sim.model import Model


# =============================================================================
# WARP KERNELS
# =============================================================================

@wp.kernel
def compute_volume_kernel(
    positions: wp.array(dtype=wp.vec2),
    tri_indices: wp.array(dtype=int),
    tri_areas: wp.array(dtype=float),
):
    """Compute area of each triangle (2D volume)."""
    tid = wp.tid()
    
    i0 = tri_indices[tid * 3 + 0]
    i1 = tri_indices[tid * 3 + 1]
    i2 = tri_indices[tid * 3 + 2]
    
    p0, p1, p2 = positions[i0], positions[i1], positions[i2]
    e1, e2 = p1 - p0, p2 - p0
    
    # 2D cross product = signed area * 2
    tri_areas[tid] = 0.5 * wp.abs(e1[0] * e2[1] - e1[1] * e2[0])


@wp.kernel
def scale_rest_lengths_kernel(
    original_rest_lengths: wp.array(dtype=float),
    scaled_rest_lengths: wp.array(dtype=float),
    scale: float,
):
    """Scale spring rest lengths by a factor."""
    sid = wp.tid()
    scaled_rest_lengths[sid] = original_rest_lengths[sid] * scale


@wp.kernel
def scale_tri_poses_kernel(
    original_poses: wp.array(dtype=wp.mat22),
    scaled_poses: wp.array(dtype=wp.mat22),
    scale: float,
):
    """
    Scale triangle rest poses.
    
    The rest pose is the inverse of the rest shape matrix Dm.
    To scale the rest shape by 's', we scale Dm by 's', so Dm_inv scales by '1/s'.
    """
    tid = wp.tid()
    inv_scale = 1.0 / scale
    orig = original_poses[tid]
    scaled_poses[tid] = wp.mat22(
        orig[0, 0] * inv_scale, orig[0, 1] * inv_scale,
        orig[1, 0] * inv_scale, orig[1, 1] * inv_scale
    )


@wp.kernel
def scale_tri_areas_kernel(
    original_areas: wp.array(dtype=float),
    scaled_areas: wp.array(dtype=float),
    scale: float,
):
    """Scale triangle rest areas by scale^2 (area scales quadratically)."""
    tid = wp.tid()
    scaled_areas[tid] = original_areas[tid] * scale * scale


# =============================================================================
# BALLOON MODEL
# =============================================================================

class BalloonModel(Model):
    """
    Balloon (inflatable circle) with PID pressure control.
    
    Inflation mechanism:
    - Scales spring rest lengths: springs "want" to be longer
    - Scales FEM rest poses: triangles "want" to be larger
    - Material naturally deforms to reach new rest state
    - Stable with stiff materials (E=2000, nu=0.45)
    
    Args:
        radius: Balloon radius (0-1 normalized, scaled by boxsize/2)
        num_boundary: Points on boundary circle
        num_rings: Interior concentric rings
        max_volume_ratio: Maximum inflation (2.0 = double initial volume)
        fem_E: Young's modulus (stiffness)
        fem_nu: Poisson's ratio (0.45 = nearly incompressible)
        fem_damping: FEM damping
    """
    
    def __init__(
        self,
        radius: float = 0.5,
        num_boundary: int = 24,
        num_rings: int = 4,
        max_volume_ratio: float = 2.0,
        device: str = 'cuda',
        boxsize: float = 3.0,
        center: tuple = None,
        spring_stiffness: float = 1000.0,
        spring_damping: float = 5.0,
        fem_E: float = 2000.0,
        fem_nu: float = 0.45,
        fem_damping: float = 10.0,
    ):
        super().__init__(device=device)
        
        self.boxsize = boxsize
        self.radius = radius
        self.max_volume_ratio = max_volume_ratio
        self.target_volume_ratio = 1.0
        self.current_pressure = 1.0  # Current pressure (rest config ratio)
        
        # Center defaults to box center
        if center is None:
            center = (boxsize / 2.0, boxsize / 2.0)
        self.center = center
        
        # Generate mesh
        points_norm, point_ring_ids = self._generate_points(radius, num_boundary, num_rings)
        triangles = self._triangulate(points_norm, radius)
        
        # Scale to world coordinates
        scale = radius * boxsize / 2.0
        points = points_norm * scale + np.array(center)
        
        # Track ring IDs for visualization
        self.point_ring_ids = point_ring_ids
        self.is_outer_point = point_ring_ids == 0  # Boundary ring
        
        # Setup simulation arrays
        self._setup_particles(points, device)
        self._setup_springs(points, triangles, spring_stiffness, spring_damping, device)
        self._setup_fem(points, triangles, fem_E, fem_nu, fem_damping, device)
        
        # Store original rest configuration for scaling
        self._store_original_rest_config()
        
        # Volume tracking
        self.initial_volume = self._compute_volume(points, triangles)
        self.max_volume = self.initial_volume * max_volume_ratio
        self.current_tri_areas = wp.zeros(self.tri_count, dtype=float, device=device)
        
        # No gravity by default
        self.set_gravity((0.0, 0.0))
        
        print(f"BalloonModel: {self.particle_count} particles, {self.spring_count} springs, {self.tri_count} triangles")
        print(f"  Volume: {self.initial_volume:.4f} → {self.max_volume:.4f} (max {self.max_volume_ratio}x)")
    
    # =========================================================================
    # MESH GENERATION
    # =========================================================================
    
    def _generate_points(self, radius, num_boundary, num_rings):
        """Generate concentric ring points with ring ID tracking."""
        points = []
        ring_ids = []
        
        # Ring 0: Boundary (outermost)
        angles = np.linspace(0, 2 * np.pi, num_boundary, endpoint=False)
        boundary_pts = np.c_[radius * np.cos(angles), radius * np.sin(angles)]
        points.append(boundary_pts)
        ring_ids.extend([0] * len(boundary_pts))
        
        # Rings 1 to num_rings: Interior rings (from outer to inner)
        for ring in range(1, num_rings + 1):
            r = radius * (num_rings - ring + 1) / (num_rings + 1)
            n = max(8, int(num_boundary * r / radius))
            a = np.linspace(0, 2 * np.pi, n, endpoint=False) + np.pi / num_boundary * ring
            ring_pts = np.c_[r * np.cos(a), r * np.sin(a)]
            points.append(ring_pts)
            ring_ids.extend([ring] * len(ring_pts))
        
        # Ring num_rings+1: Center point (innermost)
        points.append([[0.0, 0.0]])
        ring_ids.append(num_rings + 1)
        
        return np.vstack(points), np.array(ring_ids)
    
    def _triangulate(self, points, radius):
        """Delaunay triangulation, filtered to circle interior."""
        tri = Delaunay(points)
        valid = []
        
        for simplex in tri.simplices:
            centroid = np.mean(points[simplex], axis=0)
            if np.linalg.norm(centroid) <= radius * 1.01:
                p0, p1, p2 = points[simplex]
                e1, e2 = p1 - p0, p2 - p0
                if abs(e1[0] * e2[1] - e1[1] * e2[0]) >= 1e-8:
                    valid.append(simplex)
        
        return valid
    
    def _compute_volume(self, points, triangles):
        """Compute total area from triangles."""
        total = 0.0
        for tri in triangles:
            p0, p1, p2 = points[tri]
            e1, e2 = p1 - p0, p2 - p0
            total += 0.5 * abs(e1[0] * e2[1] - e1[1] * e2[0])
        return total
    
    # =========================================================================
    # SIMULATION SETUP
    # =========================================================================
    
    def _setup_particles(self, points, device):
        """Initialize particle arrays."""
        n = len(points)
        self.particle_count = n
        self.particle_q = wp.array(points.astype(np.float32), dtype=wp.vec2, device=device)
        self.particle_qd = wp.zeros(n, dtype=wp.vec2, device=device)
        self.particle_mass = wp.ones(n, dtype=float, device=device)
        self.particle_inv_mass = wp.ones(n, dtype=float, device=device)
    
    def _setup_springs(self, points, triangles, stiffness, damping, device):
        """Build springs from triangle edges."""
        edges = set()
        for tri in triangles:
            edges.add(tuple(sorted([tri[0], tri[1]])))
            edges.add(tuple(sorted([tri[1], tri[2]])))
            edges.add(tuple(sorted([tri[2], tri[0]])))
        
        indices, lengths = [], []
        boundary_spring_indices = []  # Track which springs are on boundary
        spring_idx = 0
        
        for i, j in edges:
            indices.extend([i, j])
            lengths.append(np.linalg.norm(points[j] - points[i]))
            
            # Check if both endpoints are on boundary (ring 0)
            if self.is_outer_point[i] and self.is_outer_point[j]:
                boundary_spring_indices.append(spring_idx)
            spring_idx += 1
        
        self.spring_count = len(edges)
        self.spring_indices = wp.array(np.array(indices, dtype=np.int32), dtype=int, device=device)
        self.spring_rest_length = wp.array(np.array(lengths, dtype=np.float32), dtype=float, device=device)
        self.spring_stiffness = wp.full(self.spring_count, stiffness, dtype=float, device=device)
        self.spring_damping = wp.full(self.spring_count, damping, dtype=float, device=device)
        self.spring_strains = wp.zeros(self.spring_count, dtype=float, device=device)
        
        # Store boundary spring indices for circumference strain visualization
        self.boundary_spring_indices = np.array(boundary_spring_indices, dtype=np.int32)
        self.boundary_spring_count = len(boundary_spring_indices)
    
    def _setup_fem(self, points, triangles, E, nu, damping, device):
        """Setup FEM triangles."""
        indices = []
        for tri in triangles:
            indices.extend(tri)
        
        self.tri_count = len(triangles)
        self.tri_indices = wp.array(np.array(indices, dtype=np.int32), dtype=int, device=device)
        self.tri_strains = wp.zeros(self.tri_count, dtype=float, device=device)
        self.fem_E, self.fem_nu, self.fem_damping = E, nu, damping
        
        self.compute_triangle_rest_configuration(
            points, np.array(indices, dtype=np.int32), E=E, nu=nu, k_damp=damping
        )
    
    def _store_original_rest_config(self):
        """Store original rest configuration for scaling."""
        self.original_spring_rest_length = wp.clone(self.spring_rest_length)
        self.original_tri_poses = wp.clone(self.tri_poses)
        self.original_tri_areas = wp.clone(self.tri_areas)
    
    # =========================================================================
    # VOLUME TRACKING
    # =========================================================================
    
    def compute_current_volume(self, state) -> float:
        """Compute current total area."""
        wp.launch(
            kernel=compute_volume_kernel,
            dim=self.tri_count,
            inputs=[state.particle_q, self.tri_indices, self.current_tri_areas],
            device=self.device
        )
        return float(np.sum(self.current_tri_areas.numpy()))
    
    def get_volume_ratio(self, state) -> float:
        """Current volume / initial volume."""
        return self.compute_current_volume(state) / self.initial_volume
    
    def set_target_volume_ratio(self, ratio: float):
        """Set target volume ratio (clamped to [1.0, max_volume_ratio])."""
        self.target_volume_ratio = np.clip(ratio, 1.0, self.max_volume_ratio)
    
    def get_inflation_info(self, state) -> dict:
        """Get volume info dictionary."""
        current = self.compute_current_volume(state)
        return {
            'initial_volume': self.initial_volume,
            'current_volume': current,
            'max_volume': self.max_volume,
            'current_ratio': current / self.initial_volume,
            'target_ratio': self.target_volume_ratio,
            'max_ratio': self.max_volume_ratio,
            'pressure': self.current_pressure,
        }
    
    # =========================================================================
    # INFLATION (PRESSURE CONTROL)
    # =========================================================================
    
    def apply_inflation(self, state, pid_output: float, target_ratio: float):
        """
        Apply inflation pressure via FEM rest configuration scaling.
        
        Sets pressure = target_ratio + pid_output.
        Material naturally converges to its rest state.
        PID provides corrections to track dynamic targets.
        
        Args:
            state: Current simulation state
            pid_output: PID controller output (pressure adjustment)
            target_ratio: Target area ratio (baseline)
            
        Returns:
            tuple: (None - no external forces, pressure for display)
        """
        pressure = target_ratio + pid_output
        pressure = np.clip(pressure, 1.0, self.max_volume_ratio)
        
        self._set_pressure(pressure)
        
        return None, pressure
    
    def _set_pressure(self, pressure: float):
        """
        Set internal pressure (rest configuration scale).
        
        pressure=1.0: original size (no inflation)
        pressure=2.0: balloon "wants" to be 2.0x larger in area
        """
        pressure = float(np.clip(pressure, 1.0, self.max_volume_ratio))
        
        if abs(pressure - self.current_pressure) < 1e-6:
            return
        
        self.current_pressure = pressure
        
        # For area scaling, use sqrt(pressure) for linear dimensions
        linear_scale = np.sqrt(pressure)
        
        # Scale spring rest lengths
        wp.launch(
            kernel=scale_rest_lengths_kernel,
            dim=self.spring_count,
            inputs=[self.original_spring_rest_length, self.spring_rest_length, linear_scale],
            device=self.device
        )
        
        # Scale FEM rest poses (inverse shape matrix)
        wp.launch(
            kernel=scale_tri_poses_kernel,
            dim=self.tri_count,
            inputs=[self.original_tri_poses, self.tri_poses, linear_scale],
            device=self.device
        )
        
        # Scale FEM rest areas
        wp.launch(
            kernel=scale_tri_areas_kernel,
            dim=self.tri_count,
            inputs=[self.original_tri_areas, self.tri_areas, linear_scale],
            device=self.device
        )
    
    # =========================================================================
    # FEM PARAMETERS
    # =========================================================================
    
    def set_fem_parameters(self, E: float = None, nu: float = None, k_damp: float = None):
        """Update FEM material parameters at runtime."""
        if E is not None:
            self.fem_E = E
        if nu is not None:
            self.fem_nu = nu
        if k_damp is not None:
            self.fem_damping = k_damp
        
        k_mu = self.fem_E / (2.0 * (1.0 + self.fem_nu))
        k_lambda = self.fem_E * self.fem_nu / ((1.0 + self.fem_nu) * (1.0 - 2.0 * self.fem_nu))
        self.tri_materials.fill_(wp.vec3(k_mu, k_lambda, self.fem_damping))
