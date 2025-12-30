#!/usr/bin/env python3
"""
Hopf CPG - Central Pattern Generator for Spring-Mass Locomotion

A clean Hopf oscillator implementation for coordinated locomotion.
Each oscillator produces rhythmic output, coupled to neighbors for traveling waves.

Author: NBEL
License: Apache-2.0
"""

import numpy as np


class HopfCPG:
    """
    Hopf Oscillator Central Pattern Generator.
    
    Produces coordinated rhythmic patterns for 2x2 force groups on a particle grid.
    For a 4x4 particle grid → 3x3 = 9 overlapping 2x2 groups.
    
    Dynamics:
        ṙ = a(μ - r²)r              # Amplitude converges to √μ
        θ̇ = d·ω + Σⱼ wᵢⱼ sin(θⱼ - θᵢ - φᵢⱼ)  # Phase with coupling
        x = r·cos(θ)                # Output
    
    Features:
        - Output: 1 scalar per group in [-1, 1]
        - Inter-oscillator coupling for traveling waves
        - Direction control (+1 = body moves right, -1 = left)
        - RL-friendly interface for amplitude/frequency control
    
    Example:
        >>> cpg = HopfCPG(num_groups=9, frequency=2.0, direction=[1.0, 0.0])
        >>> forces = cpg(t)  # shape (9,) in [-1, 1]
    
    Example (RL):
        >>> cpg.set_rl_params(amplitude=0.8, frequency=1.5, direction=[1.0, 0.0])
        >>> forces = cpg(t)
    """
    
    def __init__(
        self,
        num_groups: int,
        frequency: float = 2.0,
        amplitude: float = 0.5,
        direction = None,
        coupling_strength: float = 2.0,
        dt: float = 0.001,
        group_rows: int = None,
        group_cols: int = None,
    ):
        """
        Initialize Hopf CPG.
        
        Args:
            num_groups: Number of 2x2 groups (e.g., 9 for 4x4 particles)
            frequency: Oscillation frequency in Hz
            amplitude: Output amplitude [0, 1]
            direction: 2D normalized vector [dx, dy] for wave direction.
                       Wave travels OPPOSITE to direction, body moves WITH direction.
                       Default: [1, 0] (move right)
            coupling_strength: How strongly oscillators influence each other
            dt: Integration timestep
            group_rows: Number of group rows (for rectangular grids)
            group_cols: Number of group columns (for rectangular grids)
        """
        self.num_groups = num_groups
        self.frequency = frequency
        self.amplitude = amplitude
        self.coupling_strength = coupling_strength
        self.dt = dt
        
        # Validate and normalize direction
        if direction is None:
            self.direction = np.array([1.0, 0.0])
        else:
            d = np.array(direction, dtype=float)
            if d.shape != (2,):
                raise ValueError(f"Direction must be 2D vector [dx, dy], got {d}")
            norm = np.linalg.norm(d)
            if norm < 1e-6:
                raise ValueError(f"Direction cannot be zero")
            self.direction = d / norm
        
        # Grid layout (support rectangular grids)
        if group_rows is not None and group_cols is not None:
            self.grid_rows = group_rows
            self.grid_cols = group_cols
        else:
            # Square grid (backward compatible)
            side = int(np.sqrt(num_groups))
            self.grid_rows = side
            self.grid_cols = side
        
        # Hopf parameters - tuned for fast locomotion
        self.hopf_a = 15.0                         # Fast convergence
        self.mu = np.ones(num_groups) * 1.0        # Full amplitude
        self.omega = np.ones(num_groups) * 2.0 * np.pi * frequency
        
        # Oscillator state
        self.r = np.ones(num_groups) * 1.0         # Start at full amplitude
        self.theta = np.zeros(num_groups)          # Phase
        
        # Initialize phases for traveling wave
        self._init_phases()
        
        # Build neighbor structure and coupling
        self.neighbors = self._build_neighbors()
        self._build_coupling()
        
        # Time and output
        self.time = 0.0
        self.last_output = np.zeros(num_groups)
        
        print(f"[HopfCPG] Initialized:")
        print(f"  Groups: {num_groups} ({self.grid_cols}x{self.grid_rows})")
        print(f"  Frequency: {frequency} Hz")
        print(f"  Direction: [{self.direction[0]:.2f}, {self.direction[1]:.2f}]")
        print(f"  Coupling: {coupling_strength}")
        print(f"  Initial phases (degrees):")
        for i in range(min(num_groups, 9)):
            row, col = i // self.grid_cols, i % self.grid_cols
            print(f"    Group {i} (r{row},c{col}): {np.degrees(self.theta[i]):.1f}°")
    
    def _init_phases(self):
        """Initialize phases for traveling wave in current direction."""
        # Phase step per grid cell - creates traveling wave
        # Using π/2 per cell = quarter wave = smooth crawling locomotion
        # (2π would create standing wave: 0° → 180° → 360° = 0°)
        phase_per_cell = np.pi / 2.0  # 90° per cell = proper traveling wave
        
        for i in range(self.num_groups):
            row = i // self.grid_cols
            col = i % self.grid_cols
            
            # Phase gradient: HIGHER phase in direction → wave travels OPPOSITE → body moves WITH direction
            col_phase = col * phase_per_cell * self.direction[0]
            row_phase = row * phase_per_cell * self.direction[1]
            
            self.theta[i] = col_phase + row_phase
    
    def _build_neighbors(self):
        """Build 4-connected neighbor list for grid layout."""
        rows = self.grid_rows
        cols = self.grid_cols
        neighbors = [[] for _ in range(self.num_groups)]
        
        for r in range(rows):
            for c in range(cols):
                i = r * cols + c
                if c > 0:
                    neighbors[i].append(i - 1)      # left
                if c < cols - 1:
                    neighbors[i].append(i + 1)      # right
                if r > 0:
                    neighbors[i].append(i - cols)   # up
                if r < rows - 1:
                    neighbors[i].append(i + cols)   # down
        
        return neighbors
    
    def _build_coupling(self):
        """Build phase coupling for traveling wave in 2D direction."""
        self.phase_coupling = np.zeros((self.num_groups, self.num_groups))
        self.target_phase_diff = np.zeros((self.num_groups, self.num_groups))
        
        # Phase difference between adjacent cells (same as _init_phases)
        phase_per_cell = np.pi / 2.0  # Must match _init_phases!
        
        for i in range(self.num_groups):
            for j in self.neighbors[i]:
                row_i, col_i = i // self.grid_cols, i % self.grid_cols
                row_j, col_j = j // self.grid_cols, j % self.grid_cols
                
                self.phase_coupling[i, j] = self.coupling_strength
                
                # Neighbor offset in grid
                col_diff = col_j - col_i
                row_diff = row_j - row_i
                
                # Target phase difference: j should lead i if j is toward direction
                # (consistent with _init_phases)
                self.target_phase_diff[i, j] = phase_per_cell * (
                    col_diff * self.direction[0] +
                    row_diff * self.direction[1]
                )
    
    def __call__(self, t: float) -> np.ndarray:
        """
        Step the CPG and get output.
        
        Args:
            t: Current simulation time (seconds)
        
        Returns:
            Array of shape (num_groups,) with values in [-1, 1]
        """
        self.time = t
        
        # Amplitude dynamics: ṙ = a(μ - r²)r
        dr = self.hopf_a * (self.mu - self.r**2) * self.r
        self.r += self.dt * dr
        self.r = np.clip(self.r, 0.01, 2.0)
        
        # Phase dynamics with coupling
        # Direction is encoded in target_phase_diff, not omega
        dtheta = self.omega.copy()
        for i in range(self.num_groups):
            for j in self.neighbors[i]:
                phase_error = self.theta[j] - self.theta[i] - self.target_phase_diff[i, j]
                dtheta[i] += self.phase_coupling[i, j] * np.sin(phase_error)
        
        self.theta += self.dt * dtheta
        self.theta = np.mod(self.theta, 2.0 * np.pi)
        
        # Output: x = amplitude * r * cos(θ)
        output = self.amplitude * self.r * np.cos(self.theta)
        self.last_output = np.clip(output, -1.0, 1.0)
        
        return self.last_output
    
    def reset(self):
        """Reset CPG to initial state."""
        self.r = np.ones(self.num_groups) * 0.5
        self._init_phases()  # Use 2D direction-aware init
        self.time = 0.0
        self.last_output[:] = 0.0
    
    def set_direction(self, direction):
        """
        Set movement direction (for RL control).
        
        Args:
            direction: 2D vector [dx, dy], will be normalized
        """
        d = np.array(direction, dtype=float)
        if d.shape != (2,):
            raise ValueError(f"Direction must be 2D vector [dx, dy], got {d}")
        norm = np.linalg.norm(d)
        if norm < 1e-6:
            raise ValueError(f"Direction cannot be zero")
        self.direction = d / norm
        self._init_phases()
        self._build_coupling()
    
    def set_rl_params(self, amplitude: float = None, frequency: float = None,
                      direction = None, coupling: float = None):
        """
        RL-friendly interface to control CPG.
        
        Args:
            amplitude: Output amplitude [0, 2]
            frequency: Frequency scaling [0.1, 5]
            direction: 2D vector [dx, dy], will be normalized
            coupling: Coupling strength [0, 5]
        """
        if amplitude is not None:
            self.amplitude = np.clip(amplitude, 0.0, 2.0)
        if frequency is not None:
            freq_scale = np.clip(frequency, 0.1, 5.0)
            self.omega = np.ones(self.num_groups) * 2.0 * np.pi * self.frequency * freq_scale
        if direction is not None:
            self.set_direction(direction)  # Uses 2D-aware parsing
        if coupling is not None:
            self.coupling_strength = np.clip(coupling, 0.0, 5.0)
            self._build_coupling()
    
    def get_state(self) -> dict:
        """Get current CPG state for RL observation."""
        return {
            'phases': self.theta.copy(),
            'amplitudes': self.r.copy(),
            'output': self.last_output.copy(),
        }


# Alias for backward compatibility
CPG = HopfCPG
