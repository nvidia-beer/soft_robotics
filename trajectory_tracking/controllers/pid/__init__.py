"""
PID Controller for Grid Trajectory Tracking.

Task: Track a moving trajectory with a deformable FEM grid.
      Each of the 9 group centroids follows its own target position
      as the entire grid moves along a trajectory (e.g., circular path).

Grid layout (9 groups from 4×4 particles):
    [G0]───[G1]───[G2]
      │      │      │
    [G3]───[G4]───[G5]    ← Each group = 4 particles (2×2)
      │      │      │       connected by FEM elements
    [G6]───[G7]───[G8]

Control: Independent PID for each group centroid.
         Same gains (Kp, Ki, Kd) but separate error tracking.

    u_g(t) = Kp * e_g + Ki * ∫e_g + Kd * de_g/dt

    where for group g:
        e_g = target_g - centroid_g
        de_g/dt = target_velocity_g - velocity_g

Output: 18 forces [fx0, fy0, fx1, fy1, ..., fx8, fy8]
        Applied to group centroids, distributed to particles.
"""

from .. import BaseController
import numpy as np


class PID(BaseController):
    """
    PID controller for grid trajectory tracking.
    
    Computes independent control for each of 9 group centroids.
    Groups are coupled through FEM physics, but controlled independently.
    """
    
    def __init__(
        self,
        num_groups: int,
        dt: float = 0.01,
        u_max: float = 500.0,
        Kp: float = 200.0,
        Ki: float = 10.0,
        Kd: float = 50.0,
        integral_limit: float = 100.0,
        **kwargs
    ):
        self.num_groups = num_groups
        super().__init__(num_groups, dt, u_max)
        
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.integral_limit = integral_limit
        
        self.error_integral = np.zeros((num_groups, 2))
        self.last_target = None
        
        self.action_dim = num_groups * 2
        self.groups_per_side = int(np.sqrt(num_groups))
        
        print(f"✓ PID initialized ({num_groups} groups)")
        print(f"  Gains: Kp={Kp}, Ki={Ki}, Kd={Kd}")
        print(f"  u_max: {u_max}")
    
    # ==================== Helper Methods ====================
    
    def _extract_state(self, state_dict):
        """
        Extract centroids, velocities, targets from state_dict.
        
        Returns:
            centroids: (num_groups, 2)
            velocities: (num_groups, 2)
            targets: (num_groups, 2)
        """
        return (
            state_dict['group_centroids'],
            state_dict['group_velocities'],
            state_dict['group_targets']
        )
    
    def _compute_error(self, targets, centroids):
        """
        Compute position error: e = target - position
        """
        return targets - centroids
    
    def _compute_target_velocity(self, targets):
        """
        Compute target velocity from consecutive targets.
        
        Returns:
            target_velocity: (num_groups, 2)
        """
        if self.last_target is not None:
            return (targets - self.last_target) / self.dt
        else:
            return np.zeros_like(targets)
    
    def _compute_error_derivative(self, targets, velocities):
        """
        Compute error derivative: de/dt = target_velocity - velocity
        """
        target_velocity = self._compute_target_velocity(targets)
        return target_velocity - velocities
    
    def _clip_output(self, u):
        """
        Clip force magnitude per group to u_max.
        """
        u_mag = np.linalg.norm(u, axis=1, keepdims=True)
        scale = np.where(u_mag > self.u_max, self.u_max / u_mag, 1.0)
        return u * scale
    
    def _compute_pid_raw(self, state_dict):
        """
        Compute raw PID output (unclipped) and return intermediate values.
        
        This is the core PID computation, extracted for reuse by subclasses.
        
        Returns:
            u_pid: (num_groups, 2) unclipped PID forces
            info: dict with intermediate values for subclass use:
                - error: position error
                - error_derivative: de/dt
                - target_velocity: for feedforward learning
                - velocities: current velocities
        """
        centroids, velocities, targets = self._extract_state(state_dict)
        
        # Error
        error = self._compute_error(targets, centroids)
        
        # Target velocity (for feedforward learning in subclasses)
        target_velocity = self._compute_target_velocity(targets)
        
        # Error derivative
        error_derivative = target_velocity - velocities
        
        # P term
        P = self.Kp * error
        
        # I term with anti-windup
        self.error_integral += error * self.dt
        self.error_integral = np.clip(
            self.error_integral, -self.integral_limit, self.integral_limit
        )
        I = self.Ki * self.error_integral
        
        # D term
        D = self.Kd * error_derivative
        
        # Store for next iteration
        self.last_target = targets.copy()
        
        # Total PID (unclipped)
        u_pid = P + I + D
        
        # Return intermediate values for subclass use
        info = {
            'error': error,
            'error_derivative': error_derivative,
            'target_velocity': target_velocity,
            'velocities': velocities,
        }
        
        return u_pid, info
    
    # ==================== Main Control ====================
    
    def compute_control(self, state_dict, target_func=None):
        """
        Compute PID control for all groups.
        
        Returns:
            Control forces [fx0, fy0, ..., fx8, fy8] shape (18,)
        """
        # Use the shared PID computation
        u_pid, info = self._compute_pid_raw(state_dict)
        
        # Clip and return
        u = self._clip_output(u_pid)
        
        # Print progress every 200 steps
        if self.step_count % 200 == 0:
            err_mag = np.linalg.norm(info['error'])
            pid_mag = np.linalg.norm(u)
            print(f"\n  step={self.step_count:5d} | err={err_mag:.4f} | PID={pid_mag:5.0f}", flush=True)
        
        self.step_count += 1
        return u.flatten().astype(np.float32)
    
    def reset(self):
        super().reset()
        self.error_integral = np.zeros((self.num_groups, 2))
        self.last_target = None
    
    def __str__(self):
        return f"PID(Kp={self.Kp}, Ki={self.Ki}, Kd={self.Kd})"
