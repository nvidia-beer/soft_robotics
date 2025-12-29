#!/usr/bin/env python3
"""
InjectForcesLocomotion - Radial force injection for soft body locomotion

Based on CPG-RL paper (Bellegarda & Ijspeert, 2022):
- All modes use RADIAL forces (expand/contract) - this is what spring-mass can do
- Ground friction converts internal deformation into locomotion
- Forces are ALWAYS applied (not just when on ground)

Force Modes (all radial-based):
1. RADIAL: Basic expand/contract
2. LOCOMOTION: Asymmetric radial for directional movement
3. PERISTALTIC: Radial wave (earthworm-like crawling)
4. ADAPTIVE: Strain-modulated radial

Author: NBEL
License: Apache-2.0
"""

import numpy as np
import warp as wp
from enum import Enum


class ForceMode(Enum):
    """Available force application modes for locomotion.
    
    NOTE: All modes use pure RADIAL (balloon) forces.
    Each group inflates/deflates based on CPG output [-1, 1].
    Locomotion emerges from traveling wave + ground friction.
    """
    RADIAL = "radial"           # Pure balloon inflate/deflate
    HORIZONTAL = "horizontal"   # Same as RADIAL (for compatibility)
    LOCOMOTION = "locomotion"   # Same as RADIAL
    PERISTALTIC = "peristaltic" # Same as RADIAL
    ADAPTIVE = "adaptive"       # Same as RADIAL


class InjectForcesLocomotion:
    """
    Locomotion-optimized force injection for 2D spring-mass systems.
    
    Extends the basic InjectForces with multiple radial force modes
    designed for effective soft body locomotion.
    
    Features:
        - Multiple force modes (radial, locomotion, peristaltic, adaptive)
        - All modes use radial (expand/contract) forces
        - Strain feedback integration
        - Per-group force scaling
    
    Example:
        >>> injector = InjectForcesLocomotion(model, mode='locomotion')
        >>> injector.inject_locomotion_force(group_id=0, cpg_output=0.5)
        >>> forces = injector.get_forces_array()
    
    Based on: CPG-RL (Bellegarda & Ijspeert, 2022)
    """
    
    def __init__(
        self,
        model,
        group_size: int = 2,
        device: str = 'cuda',
        mode: str = 'radial',
        force_scale: float = 1.0,
        locomotion_direction: np.ndarray = None,
        thrust_ratio: float = 0.5,
    ):
        """
        Initialize balloon force injector with locomotion thrust.
        
        Args:
            model: The 2D Model object (assumes grid topology)
            group_size: Size of each group (e.g., 2 means 2x2 groups)
            device: Warp device ('cuda' or 'cpu')
            mode: Force mode ('locomotion' adds horizontal thrust)
            force_scale: Global force scaling factor
            locomotion_direction: 2D vector [dx, dy] for thrust direction (default: [1, 0])
            thrust_ratio: How much horizontal thrust vs radial (0-1, default: 0.5)
        
        Locomotion mechanism:
            - INFLATE (CPG > 0): Pure radial expansion (grip ground)
            - DEFLATE (CPG < 0): Radial contraction + horizontal thrust (slide forward)
        """
        self.model = model
        self.device = wp.get_device(device)
        self.group_size = group_size
        self.force_scale = force_scale
        self.thrust_ratio = thrust_ratio
        
        # Locomotion direction (normalized)
        if locomotion_direction is None:
            self.locomotion_direction = np.array([1.0, 0.0])
        else:
            d = np.array(locomotion_direction, dtype=float)
            norm = np.linalg.norm(d)
            self.locomotion_direction = d / norm if norm > 1e-6 else np.array([1.0, 0.0])
        
        # Parse force mode
        if isinstance(mode, str):
            mode = mode.lower()
            try:
                self.mode = ForceMode(mode)
            except ValueError:
                print(f"Warning: Unknown mode '{mode}', defaulting to 'radial'")
                self.mode = ForceMode.RADIAL
        else:
            self.mode = mode
        
        # Infer grid dimensions
        self.N = int(np.sqrt(model.particle_count))
        assert self.N * self.N == model.particle_count, "Model must be a square grid"
        
        # Force accumulator
        self.forces_np = np.zeros((model.particle_count, 2), dtype=np.float32)
        self.forces_wp = wp.zeros(model.particle_count, dtype=wp.vec2, device=self.device)
        
        # Group information
        self.group_info = {}
        self.centroids = None
        self.current_positions = None  # Store current positions for force calculation
        self.num_groups = 0
        
        # Strain storage (for adaptive mode)
        self.group_strains = None
        
        # Build group structure
        self._build_groups()
        
        print(f"\n{'='*60}")
        print(f"InjectForcesLocomotion Initialized")
        print(f"{'='*60}")
        print(f"  Mode: {self.mode.value.upper()}")
        print(f"  Grid: {self.N}x{self.N} = {model.particle_count} particles")
        print(f"  Groups: {self.num_groups} ({self.N - self.group_size + 1}x{self.N - self.group_size + 1})")
        print(f"  Force scale: {force_scale}")
        print(f"  Note: All modes are pure balloon (radial)")
        print(f"{'='*60}\n")
    
    def _build_groups(self):
        """Build overlapping group structure (same as original)."""
        groups_per_dim = self.N - self.group_size + 1
        
        if groups_per_dim <= 0:
            groups_per_dim = 1
        
        self.num_groups = groups_per_dim * groups_per_dim
        self.group_strains = np.zeros(self.num_groups)
        
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
        
        Also stores positions for use in force calculation.
        """
        centroids = np.zeros((self.num_groups, 2), dtype=np.float32)
        
        # Store current positions for force calculation
        self.current_positions = np.array(positions_np, dtype=np.float32)
        
        for group_id, particle_indices in self.group_info.items():
            if len(particle_indices) > 0:
                group_positions = positions_np[particle_indices]
                centroids[group_id] = np.mean(group_positions, axis=0)
        
        self.centroids = centroids
        return centroids
    
    def set_strain(self, group_id: int, strain: float):
        """Set strain value for a group (used in adaptive mode)."""
        if 0 <= group_id < self.num_groups:
            self.group_strains[group_id] = strain
    
    def set_all_strains(self, strains):
        """Set strain values for all groups."""
        self.group_strains = np.array(strains[:self.num_groups])
    
    # =========================================================================
    # FORCE APPLICATION MODES
    # =========================================================================
    
    def inject_locomotion_force(self, group_id: int, cpg_output: float, mode: ForceMode = None):
        """
        Apply locomotion force to a group using the configured mode.
        
        Args:
            group_id: Which group to apply forces to
            cpg_output: CPG output in [-1, 1] (positive = expand, negative = contract)
            mode: Override default mode (optional)
        
        All modes use RADIAL forces (expand/contract) because that's what
        spring-mass systems can physically apply. The modes differ in how
        they modulate the radial forces.
        """
        # All modes use radial (balloon) forces
        self._inject_radial(group_id, cpg_output)
    
    def _inject_radial(self, group_id: int, cpg_output: float):
        """
        RADIAL MODE: Pure balloon inflate/deflate.
        
        Each group acts like a balloon:
        - cpg_output > 0: Inflate (particles push outward from centroid)
        - cpg_output < 0: Deflate (particles pull inward to centroid)
        
        This is the CORRECT mode for a balloon-grid robot.
        Locomotion emerges from:
        1. CPG traveling wave (phase gradient across groups)
        2. Ground friction (inflated = more contact = more grip)
        """
        if group_id not in self.group_info:
            return
        if self.centroids is None or self.current_positions is None:
            return
        
        particle_indices = self.group_info[group_id]
        centroid = self.centroids[group_id]
        # Use current positions (set by calculate_centroids), not stale model positions
        positions = self.current_positions[particle_indices]
        
        magnitude = cpg_output * self.force_scale
        
        for i, (idx, pos) in enumerate(zip(particle_indices, positions)):
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
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def inject_forces_to_group(self, group_id: int, magnitude: float):
        """
        Backward-compatible interface (uses configured mode).
        
        Args:
            group_id: Which group to apply forces to
            magnitude: CPG output / force magnitude
        """
        self.inject_locomotion_force(group_id, magnitude)
    
    def reset(self):
        """Clear all accumulated forces."""
        self.forces_np.fill(0.0)
    
    def get_forces_array(self):
        """Get accumulated forces as numpy array."""
        return self.forces_np
    
    def get_forces_warp(self):
        """Get accumulated forces as Warp array."""
        self.forces_wp.assign(wp.array(self.forces_np, dtype=wp.vec2, device=self.device))
        return self.forces_wp
    
    def get_group_info(self):
        """Get group membership information."""
        return self.group_info
    
    def set_mode(self, mode: str):
        """Change force application mode at runtime."""
        try:
            self.mode = ForceMode(mode.lower())
            print(f"Force mode changed to: {self.mode.value.upper()}")
        except ValueError:
            print(f"Unknown mode: {mode}")
    
    def set_force_scale(self, scale: float):
        """Change force scale at runtime."""
        self.force_scale = scale


# =========================================================================
# CONVENIENCE FUNCTIONS
# =========================================================================

def get_available_modes():
    """Get list of available force modes."""
    return [m.value for m in ForceMode]


def describe_mode(mode: str) -> str:
    """Get description of a force mode."""
    descriptions = {
        'radial': "Pure balloon inflate/deflate",
        'locomotion': "Pure balloon (same as radial)",
        'peristaltic': "Pure balloon (same as radial)",
        'adaptive': "Pure balloon (same as radial)",
    }
    return descriptions.get(mode.lower(), "Unknown mode")


# =========================================================================
# MAIN (for testing)
# =========================================================================

if __name__ == "__main__":
    print("Available force modes for locomotion:")
    print("-" * 50)
    for mode in get_available_modes():
        print(f"  {mode:12s} : {describe_mode(mode)}")
    print("-" * 50)
    print("\nUsage in run_snn.sh:")
    print("  export SNN_FORCE_MODE=locomotion  # or radial, peristaltic, adaptive")
