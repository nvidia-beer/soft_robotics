#!/usr/bin/env python3
"""
Nengo SNN Tracking Interface for Trajectory Tracking

Generic interface for connecting Nengo spiking neural network controllers
to the spring-mass trajectory tracking environment. Provides standardized
inputs for trajectory targets and sensor data, allowing any Nengo controller
to hook up to the system.

Architecture:
    TrackingEnv → SNNTrackingInterface → [Your Nengo Controller] → Control Forces

The interface provides:
    - Trajectory inputs: target positions, velocities, and angles
    - Sensor inputs: current positions, velocities, strains, errors
    - Per-strain spike ensembles with probes (like rate-coding interface)
    - Group-organized strain visualization
    - Normalized encoding suitable for Nengo ensembles
    - Recording and monitoring utilities
    - Generic nodes that any controller can connect to
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'warp'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'openai-gym'))

import numpy as np
import nengo
from collections import deque
from typing import Optional, Dict, Any, Callable, List, Tuple


class SNNTrackingInterface:
    """
    Generic Nengo SNN interface for trajectory tracking.
    
    Provides standardized input/output nodes for connecting any Nengo
    controller to the spring-mass tracking environment.
    
    Input Channels (read from environment):
        - trajectory: target positions and velocities for each group
        - sensors: current state (positions, velocities, strains)
        - errors: tracking errors (position and velocity)
    
    Output Channels (write to environment):
        - control: force commands per group centroid
    
    Strain Visualization:
        - Per-strain ensembles with spike probes (like rate-coding interface)
        - Group-averaged strain ensembles
        - Spike rasters and decoded values available for GUI
    
    Usage:
        1. Create interface with environment
        2. Build network (creates Nengo nodes and strain ensembles)
        3. Create your controller and connect to interface nodes
        4. Start simulation
        5. Call step() each timestep to update inputs
    
    Attributes:
        env: TrackingEnv instance
        num_groups: Number of control groups
        input_dim: Total input dimension
        output_dim: Control output dimension (num_groups * 2)
        strain_ensembles: List of per-strain LIF ensembles
        group_strain_ensembles: List of group-averaged strain ensembles
        spike_probes: Probes for spike activity
        decoded_probes: Probes for decoded strain values
    """
    
    def __init__(
        self,
        env,
        n_neurons: int = 50,
        max_frequency: float = 100.0,
        neuron_type=None,
        encoding_method: str = 'linear',
        threshold: float = 0.1,
        dt: float = 0.01,
        record_buffer_size: int = 5000,
        verbose: bool = True,
        # Normalization scales
        position_scale: float = 0.5,
        velocity_scale: float = 2.0,
        error_scale: float = 0.5,
        strain_scale: float = 1.0,
        force_scale: float = 500.0,
    ):
        """
        Initialize the SNN tracking interface.
        
        Args:
            env: TrackingEnv instance with trajectory tracking capabilities
            n_neurons: Number of LIF neurons per strain input
            max_frequency: Maximum spike frequency (Hz) for rate encoding
            neuron_type: Nengo neuron type (default: LIF)
            encoding_method: 'linear', 'rectified', or 'threshold'
            threshold: Magnitude threshold for 'threshold' encoding
            dt: Nengo simulation timestep (should match env.dt)
            record_buffer_size: History length for recording
            verbose: Print initialization info
            position_scale: Scale for normalizing positions to [-1, 1]
            velocity_scale: Scale for normalizing velocities to [-1, 1]
            error_scale: Scale for normalizing errors to [-1, 1]
            strain_scale: Scale for strain values (already normalized in env)
            force_scale: Maximum force magnitude (for output denormalization)
        """
        self.env = env
        self.model = env.model
        self.n_neurons = n_neurons
        self.max_frequency = max_frequency
        self.neuron_type = neuron_type or nengo.LIF()
        self.encoding_method = encoding_method
        self.threshold = threshold
        self.dt = dt
        self.verbose = verbose
        self.record_buffer_size = record_buffer_size
        
        # Normalization scales
        self.position_scale = position_scale
        self.velocity_scale = velocity_scale
        self.error_scale = error_scale
        self.strain_scale = strain_scale
        self.force_scale = force_scale
        
        # Get dimensions from environment
        self.num_groups = env.num_groups
        self.output_dim = self.num_groups * 2  # fx, fy per group
        self.N = env.N  # Grid size
        
        # Count strain inputs
        self._count_strain_inputs()
        self._setup_strain_groups()
        
        # Compute input dimensions
        # Per group: pos_error(2) + vel(2) + target_offset(2) = 6
        # Plus strains: n_spring + n_fem
        # Plus rotation angle: 1
        self.per_group_dim = 6  # pos_error(2) + velocity(2) + target_offset(2)
        self.input_dim = (
            self.num_groups * self.per_group_dim +  # Group state
            self.n_total_strains +                   # Strain feedback
            1                                        # Rotation angle
        )
        
        # Current values (updated by update_from_env)
        self._current_input = np.zeros(self.input_dim, dtype=np.float32)
        self._current_output = np.zeros(self.output_dim, dtype=np.float32)
        self._control_output = np.zeros(self.output_dim, dtype=np.float32)
        
        # Breakdown of input components
        self._input_breakdown = {
            'position_errors': (0, self.num_groups * 2),
            'velocities': (self.num_groups * 2, self.num_groups * 4),
            'target_offsets': (self.num_groups * 4, self.num_groups * 6),
            'strains': (self.num_groups * 6, self.num_groups * 6 + self.n_total_strains),
            'rotation': (self.num_groups * 6 + self.n_total_strains,
                        self.num_groups * 6 + self.n_total_strains + 1),
        }
        
        # Recording buffers (like rate-coding interface)
        self.time_history = deque(maxlen=record_buffer_size)
        self.strain_history = deque(maxlen=record_buffer_size)
        self.spike_rate_history = deque(maxlen=record_buffer_size)
        self.spike_count_history = deque(maxlen=record_buffer_size)
        self.decoded_value_history = deque(maxlen=record_buffer_size)
        self.input_history = deque(maxlen=record_buffer_size)
        self.output_history = deque(maxlen=record_buffer_size)
        self.error_history = deque(maxlen=record_buffer_size)
        
        # Current state (updated each step)
        self.current_strains = np.zeros(self.n_total_strains, dtype=np.float32)
        self.current_spike_rates = np.zeros(self.n_total_strains, dtype=np.float32)
        self.current_spike_counts = np.zeros(self.n_total_strains, dtype=np.float32)
        self.current_decoded_values = np.zeros(self.n_total_strains, dtype=np.float32)
        self.current_group_strains = np.zeros(self.num_groups, dtype=np.float32)
        self.current_time = 0.0
        self.step_count = 0
        
        # Nengo network components
        self.net = None
        self.input_node = None
        self.strain_input_node = None
        self.output_node = None
        self.sim = None
        
        # Strain ensembles (per-strain, like rate-coding)
        self.strain_ensembles = []
        self.spike_probes = []
        self.decoded_probes = []
        
        # Group-averaged ensembles
        self.group_strain_ensembles = []
        self.group_strain_probes = []
        
        if verbose:
            print(f"\n[SNNTrackingInterface] Initialized:")
            print(f"  Grid: {self.N}×{self.N}")
            print(f"  Groups: {self.num_groups}")
            print(f"  Springs: {self.n_spring_strains}")
            print(f"  FEM triangles: {self.n_fem_strains}")
            print(f"  Total strain inputs: {self.n_total_strains}")
            print(f"  Neurons per strain: {self.n_neurons}")
            print(f"  Max frequency: {self.max_frequency} Hz")
            print(f"  Encoding: {self.encoding_method}")
            print(f"  Input dimension: {self.input_dim}")
            print(f"  Output dimension: {self.output_dim} (forces per group)")
            print(f"  Nengo dt: {self.dt}s")
    
    def _count_strain_inputs(self):
        """Count available strain sensors from environment model."""
        self.n_spring_strains = self.model.spring_count if hasattr(self.model, 'spring_count') else 0
        self.n_fem_strains = self.model.tri_count if hasattr(self.model, 'tri_count') else 0
        self.n_total_strains = self.n_spring_strains + self.n_fem_strains
    
    def _setup_strain_groups(self):
        """Assign springs and FEMs to groups (2x2 overlapping)."""
        N = self.N
        num_groups_per_side = N - 1
        
        # Get spring indices
        spring_indices = self.model.spring_indices.numpy() if hasattr(self.model, 'spring_indices') else np.array([])
        tri_indices = self.model.tri_indices.numpy() if hasattr(self.model, 'tri_indices') and self.model.tri_indices is not None else np.array([])
        
        # Helper: particle index to grid position
        def particle_to_grid(idx):
            return (idx // N, idx % N)
        
        # Helper: get groups a particle belongs to
        def get_particle_groups(idx):
            row, col = particle_to_grid(idx)
            groups = []
            if row < N-1 and col < N-1:
                groups.append((row, col))
            if row < N-1 and col > 0:
                groups.append((row, col-1))
            if row > 0 and col < N-1:
                groups.append((row-1, col))
            if row > 0 and col > 0:
                groups.append((row-1, col-1))
            return groups
        
        # Assign springs to groups
        self.spring_groups = {g: [] for g in range(self.num_groups)}
        for spring_idx in range(0, len(spring_indices), 2):
            if spring_idx + 1 >= len(spring_indices):
                break
            p0 = spring_indices[spring_idx]
            p1 = spring_indices[spring_idx + 1]
            groups0 = set(get_particle_groups(p0))
            groups1 = set(get_particle_groups(p1))
            common = groups0 & groups1
            for (gr, gc) in common:
                gid = gr * num_groups_per_side + gc
                if gid < self.num_groups:
                    self.spring_groups[gid].append(spring_idx // 2)
        
        # Assign FEMs to groups
        self.fem_groups = {g: [] for g in range(self.num_groups)}
        for fem_idx in range(0, len(tri_indices), 3):
            if fem_idx + 2 >= len(tri_indices):
                break
            p0, p1, p2 = tri_indices[fem_idx:fem_idx+3]
            groups0 = set(get_particle_groups(p0))
            groups1 = set(get_particle_groups(p1))
            groups2 = set(get_particle_groups(p2))
            common = groups0 & groups1 & groups2
            for (gr, gc) in common:
                gid = gr * num_groups_per_side + gc
                if gid < self.num_groups:
                    self.fem_groups[gid].append(fem_idx // 3)
        
        if self.verbose:
            print(f"  Strain groups assigned:")
            for g in range(min(4, self.num_groups)):
                print(f"    Group {g}: {len(self.spring_groups.get(g, []))} springs, {len(self.fem_groups.get(g, []))} FEMs")
    
    def _get_current_strains(self) -> np.ndarray:
        """
        Fetch current normalized strains from the model.
        
        Returns:
            array of shape [n_total_strains] with normalized strains in [-1, 1]
        """
        strains = []
        
        # Get spring strains
        if self.n_spring_strains > 0 and self.model.spring_strains_normalized is not None:
            spring_strains = self.model.spring_strains_normalized.numpy()
            strains.extend(spring_strains)
        
        # Get FEM strains
        if self.n_fem_strains > 0 and self.model.tri_strains_normalized is not None:
            fem_strains = self.model.tri_strains_normalized.numpy()
            strains.extend(fem_strains)
        
        if len(strains) == 0:
            return np.zeros(self.n_total_strains, dtype=np.float32)
        
        return np.array(strains, dtype=np.float32)
    
    def _rate_encode(self, strain: float) -> float:
        """
        Encode a normalized strain to a spike rate (Hz).
        
        Args:
            strain: normalized strain in [-1, 1]
        
        Returns:
            spike rate in [0, max_frequency] Hz
        """
        if self.encoding_method == 'linear':
            rate = self.max_frequency * (strain + 1.0) / 2.0
        elif self.encoding_method == 'rectified':
            rate = self.max_frequency * max(0.0, (strain + 1.0) / 2.0)
        elif self.encoding_method == 'threshold':
            rate = self.max_frequency if abs(strain) > self.threshold else 0.0
        else:
            raise ValueError(f"Unknown encoding method: {self.encoding_method}")
        
        return float(np.clip(rate, 0.0, self.max_frequency))
    
    def build_network(self, label: str = "SNN_Tracking", build_strain_ensembles: bool = True):
        """
        Build the Nengo network with input/output nodes and strain ensembles.
        
        Creates:
        - input_node: Full state input (errors, velocities, targets, strains, rotation)
        - strain_input_node: Direct strain input for visualization
        - output_node: Control force output
        - strain_ensembles: Per-strain LIF ensembles with probes
        - group_strain_ensembles: Group-averaged strain ensembles
        
        Args:
            label: Network label for debugging
            build_strain_ensembles: Whether to build per-strain ensembles for visualization
        """
        if self.verbose:
            print(f"\n[SNNTrackingInterface] Building Nengo network '{label}'...")
        
        self.net = nengo.Network(label=label)
        num_groups_per_side = self.N - 1
        
        with self.net:
            # Main input node: provides full state (for controllers)
            self.input_node = nengo.Node(
                output=lambda t: self._current_input,
                size_out=self.input_dim,
                label="tracking_input"
            )
            
            # Strain-only input node: provides strain data (for visualization)
            self.strain_input_node = nengo.Node(
                output=lambda t: self.current_strains,
                size_out=self.n_total_strains,
                label="strain_input"
            )
            
            # Output node: receives control commands
            self.output_node = nengo.Node(
                output=self._receive_control,
                size_in=self.output_dim,
                label="control_output"
            )
            
            # Build per-strain ensembles (for visualization)
            if build_strain_ensembles and self.n_total_strains > 0:
                self._build_strain_ensembles(num_groups_per_side)
        
        if self.verbose:
            print(f"  ✓ Created input node: {self.input_dim}D")
            print(f"  ✓ Created strain input node: {self.n_total_strains}D")
            print(f"  ✓ Created output node: {self.output_dim}D")
            if build_strain_ensembles:
                print(f"  ✓ Created {len(self.strain_ensembles)} strain ensembles")
                print(f"  ✓ Created {len(self.group_strain_ensembles)} group average ensembles")
            print(f"  → Connect your controller: input_node → [ensemble] → output_node")
    
    def _build_strain_ensembles(self, num_groups_per_side: int):
        """Build per-strain and group-averaged ensembles."""
        
        # Per-strain ensembles (like rate-coding interface)
        for i in range(self.n_total_strains):
            if i < self.n_spring_strains:
                label = f"Spring_{i}"
            else:
                label = f"FEM_{i - self.n_spring_strains}"
            
            ensemble = nengo.Ensemble(
                n_neurons=self.n_neurons,
                dimensions=1,
                max_rates=nengo.dists.Uniform(100, 200),
                intercepts=nengo.dists.Uniform(-0.5, 0.5),
                neuron_type=self.neuron_type,
                radius=1.0,
                label=label
            )
            self.strain_ensembles.append(ensemble)
            
            # Connect strain input to ensemble
            nengo.Connection(
                self.strain_input_node[i],
                ensemble,
                synapse=0.01
            )
            
            # Probe: spike counts
            spike_probe = nengo.Probe(
                ensemble.neurons,
                sample_every=self.dt,
                synapse=0.01
            )
            self.spike_probes.append(spike_probe)
            
            # Probe: decoded value
            decoded_probe = nengo.Probe(
                ensemble,
                sample_every=self.dt,
                synapse=None
            )
            self.decoded_probes.append(decoded_probe)
        
        # Group-averaged ensembles
        for group_id in range(self.num_groups):
            group_springs = self.spring_groups.get(group_id, [])
            group_fems = self.fem_groups.get(group_id, [])
            
            group_strain_indices = group_springs + [self.n_spring_strains + f for f in group_fems]
            
            if len(group_strain_indices) == 0:
                continue
            
            group_row = group_id // num_groups_per_side
            group_col = group_id % num_groups_per_side
            
            avg_ensemble = nengo.Ensemble(
                n_neurons=100,
                dimensions=1,
                max_rates=nengo.dists.Uniform(100, 200),
                intercepts=nengo.dists.Uniform(-0.5, 0.5),
                neuron_type=nengo.LIF(),
                radius=1.0,
                label=f"G{group_id}[{group_row},{group_col}]_Avg"
            )
            
            # Connect member ensembles to average
            for strain_idx in group_strain_indices:
                if strain_idx < len(self.strain_ensembles):
                    nengo.Connection(
                        self.strain_ensembles[strain_idx],
                        avg_ensemble,
                        transform=1.0 / len(group_strain_indices),
                        synapse=0.01
                    )
            
            self.group_strain_ensembles.append(avg_ensemble)
            
            # Probe for group average
            group_probe = nengo.Probe(avg_ensemble, synapse=0.01)
            self.group_strain_probes.append(group_probe)
    
    def _receive_control(self, t, x):
        """Callback to receive control output from network."""
        self._control_output = np.array(x, dtype=np.float32)
    
    def start_simulation(self, progress_bar: bool = False):
        """Start the Nengo simulator."""
        if self.net is None:
            raise RuntimeError("Call build_network() first!")
        
        if self.verbose:
            print(f"\n[SNNTrackingInterface] Starting Nengo simulator...")
        
        try:
            self.sim = nengo.Simulator(self.net, dt=self.dt, progress_bar=progress_bar)
            if self.verbose:
                print(f"  ✓ Simulator started (dt={self.dt}s)")
        except Exception as e:
            print(f"  ✗ Error starting simulator: {e}")
            raise
    
    def update_from_env(self, state_dict: Dict[str, Any] = None):
        """
        Update interface inputs from environment state.
        
        Args:
            state_dict: State dictionary from env.get_state_for_controller()
                       If None, fetches current strains directly from model
        """
        # Always update strain data from model
        self.current_strains = self._get_current_strains()
        
        # Compute spike rates
        self.current_spike_rates = np.array([
            self._rate_encode(s) for s in self.current_strains
        ], dtype=np.float32)
        
        if state_dict is None:
            return
        
        # Extract and normalize state components
        
        # 1. Position errors (target - current)
        group_centroids = state_dict.get('group_centroids', np.zeros((self.num_groups, 2)))
        group_targets = state_dict.get('group_targets', np.zeros((self.num_groups, 2)))
        position_errors = (group_targets - group_centroids).flatten()
        position_errors_norm = np.clip(position_errors / self.error_scale, -1, 1)
        
        # 2. Group velocities
        group_velocities = state_dict.get('group_velocities', np.zeros((self.num_groups, 2)))
        velocities_flat = group_velocities.flatten()
        velocities_norm = np.clip(velocities_flat / self.velocity_scale, -1, 1)
        
        # 3. Target offsets
        trajectory_offset = state_dict.get('trajectory_offset', np.zeros(2))
        target_offsets = np.tile(trajectory_offset, self.num_groups)
        target_offsets_norm = np.clip(target_offsets / self.position_scale, -1, 1)
        
        # 4. Strain feedback (already updated above)
        strains_norm = np.clip(self.current_strains / self.strain_scale, -1, 1)
        
        # Pad or truncate
        if len(strains_norm) < self.n_total_strains:
            strains_norm = np.pad(strains_norm, (0, self.n_total_strains - len(strains_norm)))
        elif len(strains_norm) > self.n_total_strains:
            strains_norm = strains_norm[:self.n_total_strains]
        
        # 5. Rotation angle
        rotation_angle = state_dict.get('rotation_angle', 0.0)
        rotation_norm = np.clip(rotation_angle / np.pi, -1, 1)
        
        # Assemble input vector
        self._current_input = np.concatenate([
            position_errors_norm.astype(np.float32),
            velocities_norm.astype(np.float32),
            target_offsets_norm.astype(np.float32),
            strains_norm.astype(np.float32),
            np.array([rotation_norm], dtype=np.float32)
        ])
        
        # Store raw values
        self._last_position_errors = position_errors
        self._last_velocities = velocities_flat
    
    def step(self, record: bool = True):
        """Advance the Nengo simulation by one timestep."""
        if self.sim is None:
            raise RuntimeError("Call start_simulation() first!")
        
        self.sim.step()
        self.current_time += self.dt
        self.step_count += 1
        
        if record:
            self._record_state()
    
    def _record_state(self):
        """Record current state for analysis."""
        self.time_history.append(self.current_time)
        self.strain_history.append(self.current_strains.copy())
        self.spike_rate_history.append(self.current_spike_rates.copy())
        self.input_history.append(self._current_input.copy())
        self.output_history.append(self._control_output.copy())
        
        # Extract spike counts from probes
        if self.sim is not None and len(self.spike_probes) > 0:
            spike_counts = np.array([
                np.sum(self.sim.data[probe][-1]) if len(self.sim.data[probe]) > 0 else 0.0
                for probe in self.spike_probes
            ])
            self.current_spike_counts = spike_counts
            self.spike_count_history.append(spike_counts.copy())
            
            # Extract decoded values
            decoded_values = np.array([
                self.sim.data[probe][-1][0] if len(self.sim.data[probe]) > 0 else 0.0
                for probe in self.decoded_probes
            ])
            self.current_decoded_values = decoded_values
            self.decoded_value_history.append(decoded_values.copy())
        
        # Extract group averages
        if len(self.group_strain_probes) > 0:
            self.current_group_strains = np.array([
                self.sim.data[probe][-1][0] if len(self.sim.data[probe]) > 0 else 0.0
                for probe in self.group_strain_probes
            ], dtype=np.float32)
        
        if hasattr(self, '_last_position_errors'):
            self.error_history.append(self._last_position_errors.copy())
    
    def get_control_output(self, denormalize: bool = True) -> np.ndarray:
        """Get the current control output (forces)."""
        output = self._control_output.copy()
        if denormalize:
            output = output * self.force_scale
        return output
    
    def get_control_output_2d(self, denormalize: bool = True) -> np.ndarray:
        """Get control output reshaped to (num_groups, 2)."""
        return self.get_control_output(denormalize).reshape(self.num_groups, 2)
    
    def get_spike_counts(self) -> np.ndarray:
        """Get most recent spike counts (one per strain input)."""
        return self.current_spike_counts.copy()
    
    def get_decoded_values(self) -> np.ndarray:
        """Get most recent decoded values from ensembles."""
        return self.current_decoded_values.copy()
    
    def get_spike_rates(self) -> np.ndarray:
        """Get most recent encoded spike rates."""
        return self.current_spike_rates.copy()
    
    def get_strains(self) -> np.ndarray:
        """Get most recent normalized strains."""
        return self.current_strains.copy()
    
    def get_group_strains(self) -> np.ndarray:
        """Get most recent group-averaged strains."""
        return self.current_group_strains.copy()
    
    def get_input_breakdown(self) -> Dict[str, np.ndarray]:
        """Get current input broken down by component."""
        breakdown = {}
        for name, (start, end) in self._input_breakdown.items():
            breakdown[name] = self._current_input[start:end].copy()
        return breakdown
    
    def get_error_input(self) -> np.ndarray:
        """Get just the position error portion of input."""
        start, end = self._input_breakdown['position_errors']
        return self._current_input[start:end].copy()
    
    def get_velocity_input(self) -> np.ndarray:
        """Get just the velocity portion of input."""
        start, end = self._input_breakdown['velocities']
        return self._current_input[start:end].copy()
    
    def get_strain_input(self) -> np.ndarray:
        """Get just the strain portion of input."""
        start, end = self._input_breakdown['strains']
        return self._current_input[start:end].copy()
    
    def reset(self):
        """Reset interface state and simulator."""
        self._current_input = np.zeros(self.input_dim, dtype=np.float32)
        self._current_output = np.zeros(self.output_dim, dtype=np.float32)
        self._control_output = np.zeros(self.output_dim, dtype=np.float32)
        self.current_strains = np.zeros(self.n_total_strains, dtype=np.float32)
        self.current_spike_rates = np.zeros(self.n_total_strains, dtype=np.float32)
        self.current_spike_counts = np.zeros(self.n_total_strains, dtype=np.float32)
        self.current_decoded_values = np.zeros(self.n_total_strains, dtype=np.float32)
        self.current_group_strains = np.zeros(self.num_groups, dtype=np.float32)
        self.current_time = 0.0
        self.step_count = 0
        
        # Clear histories
        self.time_history.clear()
        self.strain_history.clear()
        self.spike_rate_history.clear()
        self.spike_count_history.clear()
        self.decoded_value_history.clear()
        self.input_history.clear()
        self.output_history.clear()
        self.error_history.clear()
        
        if self.sim is not None:
            self.sim.reset()
    
    def print_summary(self):
        """Print current state summary."""
        print("\n" + "=" * 70)
        print("SNNTrackingInterface Summary")
        print("=" * 70)
        print(f"Grid: {self.N}×{self.N}")
        print(f"Groups: {self.num_groups}")
        print(f"Spring inputs: {self.n_spring_strains}")
        print(f"FEM inputs: {self.n_fem_strains}")
        print(f"Total strains: {self.n_total_strains}")
        print(f"Neurons per strain: {self.n_neurons}")
        print(f"Total neurons: {self.n_total_strains * self.n_neurons}")
        print(f"\nCurrent state:")
        print(f"  Strains:       min={self.current_strains.min():.3f}, max={self.current_strains.max():.3f}")
        print(f"  Spike rates:   min={self.current_spike_rates.min():.1f}, max={self.current_spike_rates.max():.1f} Hz")
        print(f"  Spike counts:  min={self.current_spike_counts.min():.3f}, max={self.current_spike_counts.max():.3f}")
        print(f"  Decoded:       min={self.current_decoded_values.min():.3f}, max={self.current_decoded_values.max():.3f}")
        print(f"  Group strains: min={self.current_group_strains.min():.3f}, max={self.current_group_strains.max():.3f}")
        print(f"\nHistory buffer: {len(self.time_history)} / {self.record_buffer_size}")
        print("=" * 70)
    
    def close(self):
        """Clean up resources."""
        if self.sim is not None:
            try:
                self.sim.close()
            except:
                pass
            self.sim = None
        if self.verbose:
            print("[SNNTrackingInterface] Closed")
    
    def __del__(self):
        self.close()


class ControllableNode(nengo.Node):
    """Nengo Node whose output can be set directly."""
    
    def __init__(self, size_out: int, label: str = None):
        self._value = np.zeros(size_out, dtype=np.float32)
        super().__init__(
            output=lambda t: self._value,
            size_out=size_out,
            label=label
        )
    
    @property
    def value(self) -> np.ndarray:
        return self._value
    
    @value.setter
    def value(self, v):
        self._value = np.asarray(v, dtype=np.float32)
