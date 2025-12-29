"""
Classic Stress-Adaptive Controller

PD control with strain-based stiffness modulation.
High stress/strain → reduce stiffness → be compliant.

ARCHITECTURE:
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  PD CONTROL (from PID with Ki=0):                                  │
│  ────────────────────────────────                                  │
│                                                                     │
│    F_pd = Kp × error + Kd × error_dot                             │
│                                                                     │
│  COMPLIANCE MODULATION (strain-based):                             │
│  ──────────────────────────────────────                            │
│                                                                     │
│    compliance_factor = 1 - α × |strain|                            │
│    F_modulated = F_pd × compliance_factor                          │
│                                                                     │
│  STRAIN-RATE DAMPING:                                              │
│  ────────────────────                                              │
│                                                                     │
│    F_damping = -β × strain_rate                                    │
│                                                                     │
│  OUTPUT:                                                           │
│  ───────                                                           │
│                                                                     │
│    F_total = F_modulated + F_damping                               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
"""

import numpy as np
from ..pid import PID


class Stress(PID):
    """
    Classic Stress-Adaptive Controller with strain-based modulation.
    
    Inherits PD control from PID (with Ki=0) and adds:
    - Strain-based compliance modulation (high stress → soft)
    - Strain-rate damping for stability
    """
    
    def __init__(
        self,
        num_groups: int,
        dt: float = 0.01,
        u_max: float = 500.0,
        Kp: float = 200.0,
        Kd: float = 50.0,
        alpha: float = 0.5,
        beta: float = 20.0,
        **kwargs
    ):
        """
        Initialize stress-adaptive controller.
        
        Args:
            num_groups: Number of control groups
            dt: Time step (seconds)
            u_max: Maximum control force (N)
            Kp: Proportional gain (N/m) - virtual stiffness
            Kd: Derivative gain (N·s/m) - virtual damping
            alpha: Compliance modulation factor (0 to 1)
                   0 = no modulation (always stiff)
                   1 = full compliance at max strain
            beta: Strain-rate damping coefficient (N·s per unit strain/s)
        """
        # Initialize PID with Ki=0 (pure PD)
        super().__init__(
            num_groups=num_groups,
            dt=dt,
            u_max=u_max,
            Kp=Kp,
            Ki=0.0,  # No integral term
            Kd=Kd,
            **kwargs
        )
        
        # Stress parameters
        self.alpha = alpha
        self.beta = beta
        
        # State for strain-rate damping
        self._prev_strain = np.zeros(num_groups)
        
        print(f"  + Stress control: α={alpha} (compliance), β={beta} (damping)")
    
    def compute_control(self, state_dict, get_target_fn=None):
        """
        Compute control forces using stress-adaptive control.
        
        Extends PID by adding strain-based modulation.
        
        Args:
            state_dict: Dict with 'group_centroids', 'group_targets',
                       'spring_strains', 'fem_strains', etc.
            get_target_fn: Optional (unused, targets in state_dict)
        
        Returns:
            np.ndarray: Control forces [num_groups * 2]
        """
        # Get raw PD output from parent (unclipped)
        u_pd, info = self._compute_pid_raw(state_dict)
        
        # Get strain values
        current_strain = self._get_group_strains(state_dict)
        
        # Compute strain rate
        if self.step_count == 0:
            strain_rate = np.zeros(self.num_groups)
        else:
            strain_rate = (current_strain - self._prev_strain) / self.dt
        self._prev_strain = current_strain.copy()
        
        # =====================================================================
        # COMPLIANCE MODULATION (strain-based)
        # High strain → reduce stiffness (be compliant)
        # =====================================================================
        output = np.zeros((self.num_groups, 2))
        
        for g in range(self.num_groups):
            strain_mag = np.abs(current_strain[g])
            
            # Compliance factor: 1 at zero strain, (1-alpha) at max strain
            compliance_factor = 1.0 - self.alpha * strain_mag
            
            # Apply compliance modulation to PD output
            output[g] = u_pd[g] * compliance_factor
            
            # =====================================================================
            # STRAIN-RATE DAMPING
            # Fast strain changes → add damping force
            # =====================================================================
            damping_force = -self.beta * strain_rate[g]
            output[g, 0] += damping_force * 0.5  # Split between x and y
            output[g, 1] += damping_force * 0.5
        
        # Clip using parent's method
        output = self._clip_output(output)
        
        # Print progress every 200 steps
        if self.step_count % 200 == 0:
            err_mag = np.linalg.norm(info['error'])
            strain_avg = np.mean(np.abs(current_strain))
            out_mag = np.linalg.norm(output)
            print(f"\n  step={self.step_count:5d} | err={err_mag:.4f} | strain={strain_avg:.3f} | F={out_mag:5.0f}", flush=True)
        
        self.step_count += 1
        return output.flatten().astype(np.float32)
    
    def _get_group_strains(self, state_dict):
        """Extract average strain per group."""
        spring_strains = state_dict.get('spring_strains', None)
        fem_strains = state_dict.get('fem_strains', None)
        
        # Combine all strains
        all_strains = []
        if spring_strains is not None and len(spring_strains) > 0:
            all_strains.extend(spring_strains)
        if fem_strains is not None and len(fem_strains) > 0:
            all_strains.extend(fem_strains)
        
        if len(all_strains) == 0:
            return np.zeros(self.num_groups)
        
        all_strains = np.array(all_strains)
        n_strains = len(all_strains)
        
        # Distribute strains across groups
        group_size = max(1, n_strains // self.num_groups)
        group_strains = np.zeros(self.num_groups)
        
        for g in range(self.num_groups):
            start = g * group_size
            end = start + group_size if g < self.num_groups - 1 else n_strains
            if start < n_strains:
                group_strains[g] = np.clip(np.mean(all_strains[start:end]), -1.0, 1.0)
        
        return group_strains
    
    def reset(self):
        """Reset controller state."""
        super().reset()
        self._prev_strain = np.zeros(self.num_groups)
    
    def __str__(self):
        return f"Stress(Kp={self.Kp}, Kd={self.Kd}, α={self.alpha}, β={self.beta})"

