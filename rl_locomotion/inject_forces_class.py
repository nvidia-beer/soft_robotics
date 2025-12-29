#!/usr/bin/env python3
"""
InjectForces Class - Group-based force injection for 2D grid models

This class provides:
1. Automatic splitting of grid models into groups with centroids
2. Interface for injecting forces to specific particles by indices
3. Centroid calculation and tracking for visualization

Author: NBEL
License: Apache-2.0
"""

import numpy as np
import warp as wp


class InjectForces:
    """
    Manages group-based force injection for 2D spring-mass grid models.
    
    Features:
        - Automatic grid partitioning into groups
        - Centroid calculation for each group
        - Force injection interface with particle indices and force vectors
        - Support for both individual and group-based force application
    
    Example:
        >>> model = Model.from_grid(N=5, spacing=0.25)
        >>> injector = InjectForces(model, group_size=2)
        >>> 
        >>> # Apply force to specific particles
        >>> particle_indices = [0, 1, 2]
        >>> forces = [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]
        >>> injector.inject_forces(particle_indices, forces)
        >>> 
        >>> # Get external forces for solver
        >>> external_forces = injector.get_forces_array()
    """
    
    def __init__(self, model, group_size=2, device='cuda'):
        """
        Initialize the InjectForces manager.
        
        Args:
            model: The 2D Model object (assumes grid topology)
            group_size: Size of each group (e.g., 2 means 2x2 groups)
            device: Warp device ('cuda' or 'cpu')
        """
        self.model = model
        self.device = wp.get_device(device)
        self.group_size = group_size
        
        # Infer grid dimensions from particle count
        self.N = int(np.sqrt(model.particle_count))
        assert self.N * self.N == model.particle_count, "Model must be a square grid"
        
        # Force accumulator (cleared each step)
        self.forces_np = np.zeros((model.particle_count, 2), dtype=np.float32)
        self.forces_wp = wp.zeros(model.particle_count, dtype=wp.vec2, device=self.device)
        
        # Group information
        self.group_info = {}      # {group_id: [particle_indices]}
        self.centroids = None     # Array of centroid positions
        self.num_groups = 0
        
        # Build group structure
        self._build_groups()
    
    def _build_groups(self):
        """
        Partition the grid into OVERLAPPING groups based on group_size.
        
        Uses a sliding window approach with stride=1, so groups overlap.
        
        For a grid of N x N particles with group_size=2:
            - Creates (N-1) x (N-1) groups (stride=1, overlapping)
            - Each group contains 2x2 = 4 particles
        
        Group indexing:
            Groups are numbered sequentially from left-to-right, BOTTOM-to-TOP
        
        Example (N=4, group_size=2):
            Grid:           Groups (overlapping):
            0  1  2  3      [6 6][7 7][8 8]
            4  5  6  7      [6 6][7 7][8 8]
                            [3 3][4 4][5 5]
            8  9  10 11     [3 3][4 4][5 5]
            12 13 14 15     [0 0][1 1][2 2]
                            [0 0][1 1][2 2]
            
            → 9 overlapping groups (3x3), numbered from bottom to top
        """
        # Overlapping groups: stride=1
        groups_per_dim = self.N - self.group_size + 1
        
        if groups_per_dim <= 0:
            print(f"Error: Grid size {self.N} too small for group_size {self.group_size}")
            groups_per_dim = 1
        
        self.num_groups = groups_per_dim * groups_per_dim
        
        # Build group membership with overlapping windows
        # Start from BOTTOM row (highest row index) to TOP row (lowest row index)
        group_id = 0
        for group_row in range(groups_per_dim - 1, -1, -1):  # Reverse order: bottom to top
            for group_col in range(groups_per_dim):
                # Get particle indices for this group
                particle_indices = []
                for local_row in range(self.group_size):
                    for local_col in range(self.group_size):
                        particle_row = group_row + local_row
                        particle_col = group_col + local_col
                        
                        # Check bounds (should always be valid with overlapping)
                        if particle_row < self.N and particle_col < self.N:
                            particle_idx = particle_row * self.N + particle_col
                            particle_indices.append(particle_idx)
                
                self.group_info[group_id] = particle_indices
                group_id += 1
        
        print(f"\n=== InjectForces: Group Structure (Overlapping) ===")
        print(f"Grid size: {self.N}x{self.N} = {self.model.particle_count} particles")
        print(f"Group size: {self.group_size}x{self.group_size}")
        print(f"Number of groups: {self.num_groups} ({groups_per_dim}x{groups_per_dim})")
        print(f"Particles per group: {len(self.group_info[0])}")
        print(f"Groups overlap with stride=1")
        print(f"====================================================\n")
    
    def calculate_centroids(self, positions_np):
        """
        Calculate centroid positions for all groups.
        
        Args:
            positions_np: Particle positions as numpy array, shape [particle_count, 2]
        
        Returns:
            centroids: Array of centroid positions, shape [num_groups, 2]
        """
        centroids = np.zeros((self.num_groups, 2), dtype=np.float32)
        
        for group_id, particle_indices in self.group_info.items():
            if len(particle_indices) == 0:
                continue
            
            # Calculate average position of all particles in this group
            group_positions = positions_np[particle_indices]
            centroid = np.mean(group_positions, axis=0)
            centroids[group_id] = centroid
        
        self.centroids = centroids
        return centroids
    
    def inject_forces(self, particle_indices, forces):
        """
        Inject forces to specific particles.
        
        This is the main interface for applying forces. Forces accumulate
        until reset() is called.
        
        Args:
            particle_indices: List/array of particle indices to apply forces to
            forces: List/array of force vectors, shape [len(particle_indices), 2]
                    Each force is [fx, fy]
        
        Example:
            >>> # Apply upward force to particles 0, 1, 2
            >>> injector.inject_forces([0, 1, 2], [[0, 1], [0, 1], [0, 1]])
        """
        particle_indices = np.array(particle_indices, dtype=np.int32)
        forces = np.array(forces, dtype=np.float32)
        
        # Validate inputs
        assert len(particle_indices) == len(forces), \
            f"Mismatch: {len(particle_indices)} indices but {len(forces)} forces"
        
        # Accumulate forces
        for idx, force in zip(particle_indices, forces):
            if 0 <= idx < self.model.particle_count:
                self.forces_np[idx] += force
            else:
                print(f"Warning: Invalid particle index {idx}, skipping")
    
    def inject_forces_to_group(self, group_id, magnitude=1.0):
        """
        Apply radial forces to all particles in a specific group.
        
        Force calculation:
            force[j] = magnitude * normalize(particle[j].position - centroid[i])
        
        Each particle in the group receives a force proportional to its normalized
        direction vector from the group's centroid.
        
        Args:
            group_id: ID of the group to apply forces to
            magnitude: Scalar force magnitude (can be positive or negative)
                      - Positive: pushes particles away from centroid (expansion)
                      - Negative: pulls particles toward centroid (contraction)
        
        Example:
            >>> # Expand group 0 with magnitude 1.0
            >>> injector.inject_forces_to_group(0, magnitude=1.0)
            >>> 
            >>> # Contract group 1 with magnitude -0.5
            >>> injector.inject_forces_to_group(1, magnitude=-0.5)
        """
        if group_id not in self.group_info:
            print(f"Warning: Invalid group_id {group_id}")
            return
        
        if self.centroids is None:
            print(f"Warning: Centroids not calculated. Call calculate_centroids() first.")
            return
        
        particle_indices = self.group_info[group_id]
        centroid = self.centroids[group_id]
        
        # Get current particle positions
        positions = self.model.particle_q.numpy()[particle_indices]
        
        # Calculate forces: magnitude * normalize(position - centroid)
        forces = []
        for pos in positions:
            # Direction vector from centroid to particle
            direction = pos - centroid
            direction_norm = np.linalg.norm(direction)
            
            # Normalize direction (handle zero case)
            if direction_norm > 1e-6:
                direction_normalized = direction / direction_norm
            else:
                # Particle at centroid, use arbitrary direction
                direction_normalized = np.array([1.0, 0.0])
            
            # Force = scalar * normalized_direction
            force = direction_normalized * magnitude
            forces.append(force)
        
        # Inject the calculated forces
        self.inject_forces(particle_indices, forces)
    
    def reset(self):
        """
        Clear all accumulated forces.
        
        Call this at the beginning of each simulation step before
        injecting new forces.
        """
        self.forces_np.fill(0.0)
    
    def get_forces_array(self):
        """
        Get the accumulated forces as a numpy array.
        
        Returns:
            forces_np: Force array, shape [particle_count, 2]
        """
        return self.forces_np
    
    def get_forces_warp(self):
        """
        Get the accumulated forces as a Warp array.
        
        Returns:
            forces_wp: Warp array of forces
        """
        # Copy from numpy to warp
        self.forces_wp.assign(wp.array(self.forces_np, dtype=wp.vec2, device=self.device))
        return self.forces_wp
    
    def get_group_info(self):
        """
        Get the group membership information.
        
        Returns:
            group_info: Dictionary {group_id: [particle_indices]}
        """
        return self.group_info
    
    def print_group_info(self):
        """Print detailed group information for debugging."""
        print("\n=== Group Membership ===")
        for group_id in sorted(self.group_info.keys()):
            indices = self.group_info[group_id]
            print(f"Group {group_id}: particles {indices}")
        print("========================\n")

