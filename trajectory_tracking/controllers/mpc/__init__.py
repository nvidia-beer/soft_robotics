"""
Model Predictive Control (MPC) with COUPLED spring-mass model.

Simplified spring network that approximates FEM:
- Groups connected to adjacent neighbors by springs
- Linear springs (vs FEM's nonlinear Neo-Hookean)
- Grid topology (vs FEM's triangular mesh)

This is simpler than FEM but captures the coupling between groups.
"""

from .. import BaseController
import numpy as np
from scipy.optimize import minimize


class MPC(BaseController):
    """
    MPC with coupled spring-mass model between group centroids.
    
    Spring network:
        [G0]====[G1]====[G2]
          ‖       ‖       ‖
        [G3]====[G4]====[G5]
          ‖       ‖       ‖
        [G6]====[G7]====[G8]
    
    Each group connected to 4-neighbors by springs.
    """
    
    def __init__(
        self,
        num_groups: int,
        dt: float = 0.01,
        u_max: float = 500.0,
        horizon: int = 10,
        spring_k: float = 40.0,   # Spring constant between groups
        damping: float = 0.5,     # Damping coefficient
        mass: float = 1.0,
        Q: float = 500.0,
        R: float = 0.001,
        S: float = 0.01,
        **kwargs
    ):
        self.num_groups = num_groups
        super().__init__(num_groups, dt, u_max)
        
        self.horizon = horizon
        self.k = spring_k
        self.c = damping
        self.m = mass * 4  # 4 particles per group
        
        self.Q = Q
        self.R = R
        self.S = S
        
        self.u_prev = np.zeros(num_groups * 2)
        self.action_dim = num_groups * 2
        
        # Build adjacency for coupled springs
        self.groups_per_side = int(np.sqrt(num_groups))
        self._build_adjacency()
        
        print(f"✓ MPC initialized ({num_groups} groups, {self.groups_per_side}x{self.groups_per_side})")
        print(f"  Model: COUPLED spring network (k={spring_k}, c={damping})")
        print(f"  Topology: {len(self.adjacency)} spring connections")
        print(f"  Horizon: {horizon}, Q={Q}, R={R}, S={S}, u_max={u_max}")
    
    def _build_adjacency(self):
        """Build adjacency list for spring connections between groups."""
        self.adjacency = {}  # group_id -> list of neighbor group_ids
        gs = self.groups_per_side
        
        for g in range(self.num_groups):
            row = g // gs
            col = g % gs
            neighbors = []
            
            # 4-connected neighbors (up, down, left, right)
            if row > 0:
                neighbors.append((row - 1) * gs + col)  # up
            if row < gs - 1:
                neighbors.append((row + 1) * gs + col)  # down
            if col > 0:
                neighbors.append(row * gs + (col - 1))  # left
            if col < gs - 1:
                neighbors.append(row * gs + (col + 1))  # right
            
            self.adjacency[g] = neighbors
        
        # Rest distances (assume unit spacing initially, will be set from initial config)
        self.rest_distance = 1.0  # Will be updated when we see actual positions
        self.rest_distance_set = False
    
    def _compute_spring_forces(self, positions):
        """
        Compute spring forces from neighboring groups.
        
        Args:
            positions: (num_groups, 2) array of group positions
        
        Returns:
            forces: (num_groups, 2) array of spring forces
        """
        forces = np.zeros((self.num_groups, 2))
        
        for g in range(self.num_groups):
            for neighbor in self.adjacency[g]:
                # Vector from g to neighbor
                dx = positions[neighbor, 0] - positions[g, 0]
                dy = positions[neighbor, 1] - positions[g, 1]
                
                # Current distance
                dist = np.sqrt(dx**2 + dy**2) + 1e-8  # Avoid division by zero
                
                # Spring force: F = k * (dist - rest_dist) * direction
                stretch = dist - self.rest_distance
                force_mag = self.k * stretch
                
                # Add force toward neighbor (if stretched) or away (if compressed)
                forces[g, 0] += force_mag * dx / dist
                forces[g, 1] += force_mag * dy / dist
        
        return forces
    
    def _step(self, group_states, u_t):
        """
        One step prediction with coupled springs. SHARED by vanilla MPC and NengoMPC.
        
        Args:
            group_states: (num_groups, 4) - [px, py, vx, vy] per group
            u_t: (num_groups * 2,) - [fx, fy] per group
        
        Returns:
            next_states: (num_groups, 4)
        """
        px = group_states[:, 0]
        py = group_states[:, 1]
        vx = group_states[:, 2]
        vy = group_states[:, 3]
        
        fx = u_t[0::2]
        fy = u_t[1::2]
        
        # Coupled spring forces from neighbors
        positions = np.column_stack([px, py])
        spring_forces = self._compute_spring_forces(positions)
        
        # Damping
        fx_damp = -self.c * vx
        fy_damp = -self.c * vy
        
        # Acceleration
        ax = (fx + spring_forces[:, 0] + fx_damp) / self.m
        ay = (fy + spring_forces[:, 1] + fy_damp) / self.m
        
        # Semi-implicit Euler
        new_vx = vx + ax * self.dt
        new_vy = vy + ay * self.dt
        new_px = px + new_vx * self.dt
        new_py = py + new_vy * self.dt
        
        return np.column_stack([new_px, new_py, new_vx, new_vy])
    
    def _predict_all_groups(self, group_states, u_sequence, target_sequence):
        """
        Forward prediction over horizon using _step.
        """
        u_reshaped = u_sequence.reshape((self.horizon, self.num_groups, 2))
        states = group_states.copy()
        
        # Set rest distance from initial configuration (once)
        if not self.rest_distance_set and self.num_groups > 1:
            if 1 in self.adjacency[0]:
                dx = states[1, 0] - states[0, 0]
                dy = states[1, 1] - states[0, 1]
                self.rest_distance = np.sqrt(dx**2 + dy**2)
                self.rest_distance_set = True
        
        total_error = 0.0
        
        for t in range(self.horizon):
            targets_t = target_sequence[t]
            total_error += np.sum((states[:, 0] - targets_t[:, 0])**2 + 
                                  (states[:, 1] - targets_t[:, 1])**2)
            
            u_t = u_reshaped[t].flatten()  # (num_groups * 2,)
            states = self._step(states, u_t)
        
        return total_error
    
    def _cost_function(self, u_flat, group_states, target_sequence):
        """MPC cost: tracking + control effort + smoothness."""
        tracking_cost = self.Q * self._predict_all_groups(group_states, u_flat, target_sequence)
        control_cost = self.R * np.sum(u_flat ** 2)
        
        # Smoothness cost
        u_first = u_flat[:self.action_dim]
        smooth_cost = self.S * np.sum((u_first - self.u_prev) ** 2)
        
        if self.horizon > 1:
            u_reshaped = u_flat.reshape((self.horizon, self.action_dim))
            smooth_cost += self.S * np.sum(np.diff(u_reshaped, axis=0) ** 2)
        
        return tracking_cost + control_cost + smooth_cost
    
    def compute_control(self, state_dict, target_func):
        """Compute optimal control using scipy SLSQP."""
        group_centroids = state_dict['group_centroids']
        group_velocities = state_dict['group_velocities']
        group_targets = state_dict['group_targets']
        num_groups = state_dict['num_groups']
        
        # Build state array
        group_states = np.column_stack([
            group_centroids[:, 0],
            group_centroids[:, 1],
            group_velocities[:, 0],
            group_velocities[:, 1]
        ])
        
        # Target sequence (constant over horizon)
        target_sequence = np.tile(group_targets, (self.horizon, 1, 1))
        
        # Optimize
        result = minimize(
            fun=self._cost_function,
            x0=np.zeros(self.horizon * self.action_dim),
            args=(group_states, target_sequence),
            method='SLSQP',
            bounds=[(-self.u_max, self.u_max)] * (self.horizon * self.action_dim),
            options={'maxiter': 20, 'ftol': 1e-4}
        )
        
        u_optimal = result.x[:self.action_dim]
        self.u_prev = u_optimal.copy()
        self.step_count += 1
        
        return u_optimal.astype(np.float32)
    
    def reset(self):
        """Reset controller state."""
        super().reset()
        self.u_prev = np.zeros(self.num_groups * 2)
        self.rest_distance_set = False
    
    def __str__(self):
        return f"MPC(coupled, groups={self.num_groups}, horizon={self.horizon})"
