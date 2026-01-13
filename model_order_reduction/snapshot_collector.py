# SPDX-FileCopyrightText: Copyright (c) 2025 NBEL
# SPDX-License-Identifier: Apache-2.0
"""
Snapshot Collector for Model Order Reduction Training

Collects simulation snapshots (displacements) during actuation sequences
for POD basis computation.

The quality of the reduced basis depends on the diversity of snapshots:
- Use varied actuations (different magnitudes, directions, frequencies)
- Collect at multiple timesteps during dynamic response
- Include both transient and steady-state behavior
"""

import numpy as np
from typing import List, Optional, Callable, Any
import os


class SnapshotCollector:
    """
    Collect simulation snapshots for POD training.
    
    Example:
        >>> collector = SnapshotCollector()
        >>> 
        >>> # Run multiple simulations with different actuations
        >>> for actuation in actuation_sequence:
        >>>     positions = simulate(actuation, n_steps=100)
        >>>     collector.add_snapshots(positions, sample_interval=5)
        >>> 
        >>> # Get data for POD
        >>> snapshots = collector.get_snapshots()
        >>> rest_pos = collector.rest_position
    """
    
    def __init__(
        self,
        rest_position: Optional[np.ndarray] = None,
        snapshot_interval: int = 1,
        max_snapshots: int = 10000,
    ):
        """
        Initialize snapshot collector.
        
        Args:
            rest_position: Rest configuration (will be set from first snapshot if None)
            snapshot_interval: Collect every N timesteps (1 = every step)
            max_snapshots: Maximum number of snapshots to store
        """
        self.rest_position = rest_position
        self.snapshot_interval = snapshot_interval
        self.max_snapshots = max_snapshots
        
        self._snapshots: List[np.ndarray] = []
        self._step_count = 0
        
    @property
    def n_snapshots(self) -> int:
        """Number of collected snapshots."""
        return len(self._snapshots)
    
    @property
    def n_dof(self) -> int:
        """Number of degrees of freedom (2 * n_particles for 2D)."""
        if len(self._snapshots) > 0:
            return len(self._snapshots[0])
        elif self.rest_position is not None:
            return len(self.rest_position)
        return 0
    
    def set_rest_position(self, positions: np.ndarray):
        """
        Set rest position explicitly.
        
        Args:
            positions: Rest configuration, shape (n_dof,) or (n_particles, 2)
        """
        self.rest_position = positions.flatten().astype(np.float32)
    
    def add_snapshot(self, positions: np.ndarray, velocities: np.ndarray = None):
        """
        Add a single snapshot (optionally with velocities for Cotangent Lift).
        
        Args:
            positions: Current positions, shape (n_dof,) or (n_particles, 2)
            velocities: Current velocities, shape (n_dof,) or (n_particles, 2)
                        If provided, enables Cotangent Lift for better force retention
        """
        pos_flat = positions.flatten().astype(np.float32)
        
        # Set rest position from first snapshot if not provided
        if self.rest_position is None:
            self.rest_position = pos_flat.copy()
            print(f"  Set rest position from first snapshot ({len(pos_flat)} DOF)")
        
        # Respect max_snapshots limit
        if len(self._snapshots) >= self.max_snapshots:
            # Replace oldest snapshot (circular buffer behavior)
            idx = self._step_count % self.max_snapshots
            self._snapshots[idx] = pos_flat
        else:
            self._snapshots.append(pos_flat)
        
        # Store velocity snapshot for Cotangent Lift (if provided)
        if velocities is not None:
            vel_flat = velocities.flatten().astype(np.float32)
            if not hasattr(self, '_velocity_snapshots'):
                self._velocity_snapshots = []
            if len(self._velocity_snapshots) >= self.max_snapshots:
                idx = self._step_count % self.max_snapshots
                self._velocity_snapshots[idx] = vel_flat
            else:
                self._velocity_snapshots.append(vel_flat)
        
        self._step_count += 1
    
    def add_snapshots(
        self,
        positions_sequence: np.ndarray,
        sample_interval: Optional[int] = None,
    ):
        """
        Add multiple snapshots from a sequence.
        
        Args:
            positions_sequence: Array of positions, shape (n_timesteps, n_dof) 
                                or (n_timesteps, n_particles, 2)
            sample_interval: Override snapshot_interval for this call
        """
        interval = sample_interval if sample_interval is not None else self.snapshot_interval
        
        # Reshape if needed
        if positions_sequence.ndim == 3:
            n_steps = positions_sequence.shape[0]
            positions_sequence = positions_sequence.reshape(n_steps, -1)
        
        for i in range(0, len(positions_sequence), interval):
            self.add_snapshot(positions_sequence[i])
    
    def get_snapshots(self) -> np.ndarray:
        """
        Get collected snapshots as matrix.
        
        Returns:
            Snapshot matrix, shape (n_dof, n_snapshots)
            Columns are snapshots (positions), rows are DOF
        """
        if len(self._snapshots) == 0:
            raise ValueError("No snapshots collected")
        
        return np.array(self._snapshots).T
    
    def get_displacement_snapshots(self) -> np.ndarray:
        """
        Get displacement snapshots (relative to rest).
        
        Returns:
            Displacement matrix, shape (n_dof, n_snapshots)
        """
        snapshots = self.get_snapshots()
        return snapshots - self.rest_position.reshape(-1, 1)
    
    def get_augmented_snapshots(self, velocity_weight: float = 1.0) -> np.ndarray:
        """
        Get Cotangent Lift augmented snapshots (position + scaled velocity).
        
        This improves force retention in POD by including velocity information.
        The augmented snapshot is [position; velocity_weight * velocity].
        
        Args:
            velocity_weight: Scaling factor for velocities (default 1.0)
                            Higher values emphasize velocity modes
        
        Returns:
            Augmented snapshot matrix, shape (2*n_dof, n_snapshots)
            or (n_dof, n_snapshots) if no velocities collected
        """
        pos_snapshots = self.get_snapshots()  # (n_dof, n_snapshots)
        
        if not hasattr(self, '_velocity_snapshots') or len(self._velocity_snapshots) == 0:
            print("  No velocity snapshots - returning position-only")
            return pos_snapshots
        
        vel_snapshots = np.array(self._velocity_snapshots).T  # (n_dof, n_snapshots)
        
        if vel_snapshots.shape[1] != pos_snapshots.shape[1]:
            print(f"  Warning: velocity count mismatch ({vel_snapshots.shape[1]} vs {pos_snapshots.shape[1]})")
            # Truncate to minimum
            n = min(vel_snapshots.shape[1], pos_snapshots.shape[1])
            pos_snapshots = pos_snapshots[:, :n]
            vel_snapshots = vel_snapshots[:, :n]
        
        # Augment: [position; scaled_velocity]
        augmented = np.vstack([pos_snapshots, velocity_weight * vel_snapshots])
        print(f"  Cotangent Lift: {pos_snapshots.shape[0]} pos + {vel_snapshots.shape[0]} vel = {augmented.shape[0]} augmented DOF")
        
        return augmented
    
    def has_velocity_snapshots(self) -> bool:
        """Check if velocity snapshots were collected."""
        return hasattr(self, '_velocity_snapshots') and len(self._velocity_snapshots) > 0
    
    def clear(self):
        """Clear all collected snapshots (keep rest position)."""
        self._snapshots = []
        self._velocity_snapshots = []
        self._step_count = 0
    
    def simulate_and_record(
        self,
        model: Any,
        solver: Any,
        state_in: Any,
        state_out: Any,
        n_steps: int,
        dt: float,
        external_forces_fn: Optional[Callable[[int], np.ndarray]] = None,
        record_interval: int = 1,
    ):
        """
        Run simulation and record snapshots.
        
        This is a convenience method that handles the simulation loop.
        
        Args:
            model: Warp Model object
            solver: Warp Solver object (e.g., SolverImplicit)
            state_in: Initial state
            state_out: Output state buffer
            n_steps: Number of simulation steps
            dt: Time step
            external_forces_fn: Optional function (step) -> forces array
            record_interval: Record every N steps
        """
        import warp as wp
        
        # Set rest position from initial state
        if self.rest_position is None:
            self.set_rest_position(state_in.particle_q.numpy())
        
        for step in range(n_steps):
            # Get external forces if provided
            ext_forces = None
            if external_forces_fn is not None:
                ext_forces = external_forces_fn(step)
            
            # Step simulation
            solver.step(state_in, state_out, dt, external_forces=ext_forces)
            
            # Record snapshot
            if step % record_interval == 0:
                positions = state_out.particle_q.numpy()
                self.add_snapshot(positions)
            
            # Swap states
            state_in, state_out = state_out, state_in
        
        return state_in  # Return final state
    
    def save(self, filepath: str):
        """
        Save snapshots to file.
        
        Args:
            filepath: Output path (.npz)
        """
        np.savez(
            filepath,
            snapshots=self.get_snapshots(),
            rest_position=self.rest_position,
        )
        print(f"Saved {self.n_snapshots} snapshots to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> "SnapshotCollector":
        """
        Load snapshots from file.
        
        Args:
            filepath: Path to .npz file
            
        Returns:
            SnapshotCollector with loaded data
        """
        data = np.load(filepath)
        
        collector = cls(rest_position=data['rest_position'])
        
        snapshots = data['snapshots']
        for i in range(snapshots.shape[1]):
            collector._snapshots.append(snapshots[:, i])
        
        print(f"Loaded {collector.n_snapshots} snapshots from {filepath}")
        return collector


class GIECollector:
    """
    Collect Generalized Internal Energy contributions for ECSW.
    
    During each simulation step, records the force contribution of each
    element (spring/triangle) for hyperreduction training.
    
    Example:
        >>> gie_collector = GIECollector(n_elements=500)
        >>> 
        >>> for step in range(n_steps):
        >>>     forces_per_element = compute_element_forces(state)
        >>>     gie_collector.add_sample(forces_per_element)
        >>> 
        >>> gie_data = gie_collector.get_data()
    """
    
    def __init__(self, n_elements: int, max_samples: int = 10000):
        """
        Initialize GIE collector.
        
        Args:
            n_elements: Number of force-contributing elements
            max_samples: Maximum samples to store
        """
        self.n_elements = n_elements
        self.max_samples = max_samples
        self._samples: List[np.ndarray] = []
    
    @property
    def n_samples(self) -> int:
        return len(self._samples)
    
    def add_sample(self, element_forces: np.ndarray):
        """
        Add one sample of element force contributions.
        
        Args:
            element_forces: Force magnitude per element, shape (n_elements,)
        """
        forces = element_forces.flatten().astype(np.float32)
        
        if len(forces) != self.n_elements:
            raise ValueError(f"Expected {self.n_elements} elements, got {len(forces)}")
        
        if len(self._samples) < self.max_samples:
            self._samples.append(forces)
        else:
            # Circular buffer
            idx = len(self._samples) % self.max_samples
            self._samples[idx] = forces
    
    def get_data(self) -> np.ndarray:
        """
        Get GIE data matrix.
        
        Returns:
            GIE matrix, shape (n_samples, n_elements)
        """
        if len(self._samples) == 0:
            raise ValueError("No samples collected")
        return np.array(self._samples)
    
    def clear(self):
        """Clear collected samples."""
        self._samples = []
    
    def save(self, filepath: str):
        """Save GIE data to file."""
        np.save(filepath, self.get_data())
        print(f"Saved {self.n_samples} GIE samples to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> "GIECollector":
        """Load GIE data from file."""
        data = np.load(filepath)
        collector = cls(n_elements=data.shape[1])
        for i in range(data.shape[0]):
            collector._samples.append(data[i])
        print(f"Loaded {collector.n_samples} GIE samples from {filepath}")
        return collector

