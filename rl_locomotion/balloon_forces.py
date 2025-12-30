#!/usr/bin/env python3
"""
BalloonForces - Balloon-style force injection for soft body locomotion

Based on CPG-RL paper (Bellegarda & Ijspeert, 2022):
- Groups inflate/deflate like balloons
- Ground friction converts deformation into locomotion

Author: NBEL
License: Apache-2.0
"""

import numpy as np
import warp as wp


class BalloonForces:
    """
    Balloon force injection for 2D spring-mass locomotion.
    
    Each group of particles acts like a balloon:
    - Positive CPG output: Inflate (particles push outward)
    - Negative CPG output: Deflate (particles pull inward)
    
    Locomotion emerges from:
    1. CPG traveling wave (phase gradient across groups)
    2. Ground friction (inflated = more grip)
    
    Example:
        >>> forces = BalloonForces(model, force_scale=20.0)
        >>> forces.calculate_centroids(positions)
        >>> forces.inject(group_id=0, cpg_output=0.5)
        >>> external_forces = forces.get_array()
    
    Based on: CPG-RL (Bellegarda & Ijspeert, 2022)
    """
    
    def __init__(
        self,
        model,
        group_size: int = 2,
        device: str = 'cuda',
        force_scale: float = 1.0,
    ):
        """
        Initialize balloon force injector.
        
        Args:
            model: The 2D Model object (assumes grid topology)
            group_size: Size of each group (e.g., 2 means 2x2 groups)
            device: Warp device ('cuda' or 'cpu')
            force_scale: Global force scaling factor
        """
        self.model = model
        self.device = wp.get_device(device)
        self.group_size = group_size
        self.force_scale = force_scale
        
        # Infer grid dimensions
        self.N = int(np.sqrt(model.particle_count))
        assert self.N * self.N == model.particle_count, "Model must be a square grid"
        
        # Force accumulator
        self.forces_np = np.zeros((model.particle_count, 2), dtype=np.float32)
        self.forces_wp = wp.zeros(model.particle_count, dtype=wp.vec2, device=self.device)
        
        # Group information
        self.group_info = {}
        self.centroids = None
        self.current_positions = None
        self.num_groups = 0
        
        # Build group structure
        self._build_groups()
        
        print(f"\n{'='*60}")
        print(f"BalloonForces Initialized")
        print(f"{'='*60}")
        print(f"  Grid: {self.N}x{self.N} = {model.particle_count} particles")
        print(f"  Groups: {self.num_groups} ({self.N - self.group_size + 1}x{self.N - self.group_size + 1})")
        print(f"  Force scale: {force_scale}")
        print(f"{'='*60}\n")
    
    def _build_groups(self):
        """Build overlapping group structure."""
        groups_per_dim = self.N - self.group_size + 1
        
        if groups_per_dim <= 0:
            groups_per_dim = 1
        
        self.num_groups = groups_per_dim * groups_per_dim
        
        # Build from bottom to top
        group_id = 0
        for group_row in range(groups_per_dim - 1, -1, -1):
            for group_col in range(groups_per_dim):
                particle_indices = []
                for local_row in range(self.group_size):
                    for local_col in range(self.group_size):
                        particle_row = group_row + local_row
                        particle_col = group_col + local_col
                        if particle_row < self.N and particle_col < self.N:
                            particle_idx = particle_row * self.N + particle_col
                            particle_indices.append(particle_idx)
                
                self.group_info[group_id] = particle_indices
                group_id += 1
    
    def calculate_centroids(self, positions_np):
        """Calculate centroid positions for all groups.
        
        Must be called each step before injecting forces.
        """
        centroids = np.zeros((self.num_groups, 2), dtype=np.float32)
        self.current_positions = np.array(positions_np, dtype=np.float32)
        
        for group_id, particle_indices in self.group_info.items():
            if len(particle_indices) > 0:
                group_positions = positions_np[particle_indices]
                centroids[group_id] = np.mean(group_positions, axis=0)
        
        self.centroids = centroids
        return centroids
    
    def inject(self, group_id: int, cpg_output: float):
        """
        Inject balloon force to a group.
        
        Args:
            group_id: Which group to apply forces to
            cpg_output: CPG output in [-1, 1]
                       Positive = inflate
                       Negative = deflate
        """
        if group_id not in self.group_info:
            return
        if self.centroids is None or self.current_positions is None:
            return
        
        particle_indices = self.group_info[group_id]
        centroid = self.centroids[group_id]
        positions = self.current_positions[particle_indices]
        
        magnitude = cpg_output * self.force_scale
        
        for idx, pos in zip(particle_indices, positions):
            if idx < 0 or idx >= self.model.particle_count:
                continue
            
            direction = pos - centroid
            direction_norm = np.linalg.norm(direction)
            
            if direction_norm > 1e-6:
                direction_unit = direction / direction_norm
            else:
                direction_unit = np.array([1.0, 0.0])
            
            force = direction_unit * magnitude
            self.forces_np[idx] += force.astype(np.float32)
    
    def reset(self):
        """Clear all accumulated forces."""
        self.forces_np.fill(0.0)
    
    def get_array(self):
        """Get accumulated forces as numpy array."""
        return self.forces_np
    
    def get_warp(self):
        """Get accumulated forces as Warp array."""
        self.forces_wp.assign(wp.array(self.forces_np, dtype=wp.vec2, device=self.device))
        return self.forces_wp
    
    def get_group_info(self):
        """Get group membership information."""
        return self.group_info
    
    def set_force_scale(self, scale: float):
        """Change force scale at runtime."""
        self.force_scale = scale

