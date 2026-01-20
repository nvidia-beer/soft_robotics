"""
NEF-Based Stress Controller with PES Feedforward Learning for Volume Control.

Based on: "Neuromorphic NEF-Based Inverse Kinematics and PID Control"
Zaidel et al., Frontiers in Neurorobotics, 2021
https://doi.org/10.3389/fnbot.2021.631159

ARCHITECTURE (Volume Control - Scalar PID + PES):

┌──────────────────────────────────────────────────────────────────┐
│                                                                  │
│  error ──[÷e]──> q(t) ───┬────[Kp]────> u(t) ──[×u]──> pressure │
│                   r=1    │               r=1                     │
│                          │                ↑                      │
│                          ├──> ed(t) ─[Kd]─┤                      │
│                          │    r=1         │                      │
│                          │                │                      │
│                          │  strain[N] ──> s(t) ──[PES]──┘        │
│                          │                r=1                    │
│                          │                  ↑                    │
│                          └──────────────────┘                    │
│                             -error (learning signal)             │
│                                                                  │
│  u(t) = Kp*q + Kd*ed + PES_learned(strain)                      │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘

PES learns: strain[N-D] → pressure[1D] directly into u(t) ensemble

Key differences from trajectory tracking:
- SCALAR (1D) output: pressure instead of 2D force vectors
- SCALAR (1D) error: volume error instead of position error per group
- No integral term (pure PD + PES feedforward)
"""

import numpy as np
import nengo

from ..pid import PID
from ..nengo import NengoPID, SNN_PID_Controller


# Default strain dimensions (can be overridden)
DEFAULT_STRAIN_DIM = 7  # 5 springs + 2 FEMs (same as trajectory_tracking)


# =============================================================================
# CLASSIC STRESS CONTROLLER (Non-Spiking)
# =============================================================================

class Stress(PID):
    """
    Classic Stress-Adaptive Controller for Volume Control.
    
    PD control (PID with Ki=0) plus strain-based compliance modulation.
    High stress/strain → reduce stiffness → be compliant.
    
    Control Law:
        pressure_raw = Kp * volume_error + Kd * d(volume_error)/dt
        compliance_factor = 1 - α × |avg_strain|
        pressure = pressure_raw × compliance_factor + strain_rate_damping
    """
    
    def __init__(
        self,
        dt: float = 0.01,
        u_max: float = 10.0,
        Kp: float = 1.0,
        Kd: float = 0.3,
        alpha: float = 0.5,
        beta: float = 2.0,
        **kwargs
    ):
        """
        Initialize stress-adaptive controller for volume control.
        
        Args:
            dt: Time step (seconds)
            u_max: Maximum pressure magnitude
            Kp: Proportional gain
            Kd: Derivative gain
            alpha: Compliance modulation factor (0-1)
                   0 = no modulation (always stiff)
                   1 = full compliance at max strain
            beta: Strain-rate damping coefficient
        """
        # Initialize PID with Ki=0 (pure PD)
        super().__init__(
            dt=dt,
            u_max=u_max,
            Kp=Kp,
            Ki=0.0,  # No integral term for stress control
            Kd=Kd,
            **kwargs
        )
        
        # Stress parameters
        self.alpha = alpha
        self.beta = beta
        
        # State for strain tracking
        self._prev_strain = 0.0
        self._current_strain = 0.0
        
        print(f"  + Stress control: α={alpha} (compliance), β={beta} (damping)")
    
    def compute(self, target_volume: float, current_volume: float, 
                strain: float = None) -> float:
        """
        Compute pressure from volume error with strain-based modulation.
        
        Args:
            target_volume: Target volume (area in 2D)
            current_volume: Current volume (area in 2D)
            strain: Optional average strain value (from FEM/springs)
                    If None, uses stored _current_strain
        
        Returns:
            Pressure value (positive = inflate, negative = deflate)
        """
        # Update strain if provided
        if strain is not None:
            self._current_strain = np.clip(strain, -1.0, 1.0)
        
        # Compute error
        error = target_volume - current_volume
        
        # P term
        P = self.Kp * error
        
        # D term (on error, not volume)
        if self.step_count > 0:
            error_derivative = (error - self.last_error) / self.dt
        else:
            error_derivative = 0.0
        D = self.Kd * error_derivative
        
        # Raw PD output
        pressure_raw = P + D
        
        # =====================================================================
        # COMPLIANCE MODULATION (strain-based)
        # High strain → reduce stiffness (be compliant)
        # =====================================================================
        strain_mag = np.abs(self._current_strain)
        compliance_factor = 1.0 - self.alpha * strain_mag
        pressure_modulated = pressure_raw * compliance_factor
        
        # =====================================================================
        # STRAIN-RATE DAMPING
        # Fast strain changes → add damping
        # =====================================================================
        if self.step_count > 0:
            strain_rate = (self._current_strain - self._prev_strain) / self.dt
        else:
            strain_rate = 0.0
        damping = -self.beta * strain_rate
        
        # Total output
        pressure = pressure_modulated + damping
        pressure = np.clip(pressure, -self.u_max, self.u_max)
        
        # Update state
        self._prev_strain = self._current_strain
        self.last_error = error
        self.last_volume = current_volume
        self.step_count += 1
        
        # Debug output every 100 steps
        if self.step_count % 100 == 0:
            print(f"  [Stress step {self.step_count}] "
                  f"err={error:.4f} strain={self._current_strain:.3f} "
                  f"cf={compliance_factor:.2f} → pressure={pressure:.1f}")
        
        return pressure
    
    def set_strain(self, strain: float):
        """
        Update the current strain value.
        
        Call this before compute() if strain is not passed directly.
        
        Args:
            strain: Average strain value (clipped to [-1, 1])
        """
        self._current_strain = np.clip(strain, -1.0, 1.0)
    
    def reset(self):
        """Reset controller state."""
        super().reset()
        self._prev_strain = 0.0
        self._current_strain = 0.0
    
    def __str__(self):
        return f"Stress(Kp={self.Kp}, Kd={self.Kd}, α={self.alpha}, β={self.beta})"


# =============================================================================
# STANDALONE NENGO CONTROLLER (creates its own simulator)
# =============================================================================

class NengoStress(NengoPID):
    """
    NEF-based Stress Controller with PES Feedforward for Volume Control.
    
    - Spiking PD control (from NengoPID with Ki=0)
    - Spiking PES feedforward (learns strain[N-D] → pressure[1D])
    
    Formula: pressure = P + D + PES_learned(strain)
    
    Creates its own Nengo model and simulator internally.
    For Nengo GUI integration, use SNN_Stress_Controller instead.
    """
    
    def __init__(
        self,
        dt: float = 0.01,
        u_max: float = 10.0,
        Kp: float = 1.0,
        Kd: float = 0.3,
        n_neurons: int = 100,
        error_scale: float = 0.5,
        strain_dim: int = DEFAULT_STRAIN_DIM,
        pes_learning_rate: float = 1e-4,
        **kwargs
    ):
        """
        Initialize stress controller with PES feedforward for volume control.
        
        Args:
            dt: Simulation time step
            u_max: Maximum pressure output
            Kp: Proportional gain
            Kd: Derivative gain
            n_neurons: Neurons per dimension
            error_scale: Max expected volume error
            strain_dim: Dimensions of strain input (default 7: 5 springs + 2 FEMs)
            pes_learning_rate: PES learning rate (1e-4 default)
        """
        self.pes_learning_rate = pes_learning_rate
        self.strain_dim = strain_dim
        self._current_strain = np.zeros(strain_dim)
        
        # Initialize parent with Ki=0 (pure PD)
        super().__init__(
            dt=dt,
            u_max=u_max,
            Kp=Kp,
            Ki=0.0,  # No integral for stress control
            Kd=Kd,
            n_neurons=n_neurons,
            error_scale=error_scale,
            **kwargs
        )
        
        print(f"  ✓ NengoStress (spiking PD + PES feedforward)")
        print(f"    Strain dim: {strain_dim}")
        print(f"    PES learning rate: {pes_learning_rate}")
    
    def _build_model(self):
        """Build Nengo model with PD and PES feedforward."""
        
        # Build parent's PD network
        super()._build_model()
        
        tau_syn = 0.01
        
        with self.model:
            # =====================================================================
            # STRAIN INPUT NODE (N-D)
            # =====================================================================
            def get_strain(t):
                return self._current_strain
            
            strain_node = nengo.Node(
                get_strain,
                size_out=self.strain_dim,
                label="Strain[N-D]"
            )
            
            # =====================================================================
            # STRAIN ENSEMBLE (N-D for PES source)
            # 
            # HIGH-DIMENSIONAL INTERCEPT FIX (Zaidel et al. 2021):
            # With uniform intercepts in high-D, neurons fire for only ~2-3% of inputs.
            # Triangular distribution → inverse beta transform concentrates intercepts
            # where neurons achieve ~50% firing probability, maximizing capacity.
            # =====================================================================
            n_strain_neurons = self.n_neurons * self.strain_dim
            triangular_samples = np.random.triangular(0.3, 0.5, 0.7, n_strain_neurons)
            strain_intercepts = nengo.dists.CosineSimilarity(self.strain_dim + 2).ppf(1 - triangular_samples)
            
            s_ens = nengo.Ensemble(
                n_neurons=n_strain_neurons,
                dimensions=self.strain_dim,
                max_rates=nengo.dists.Uniform(100, 200),
                intercepts=strain_intercepts,  # Triangular for high-D
                neuron_type=nengo.LIF(),
                radius=1.0,
                label="s(t)[N-D]"
            )
            nengo.Connection(strain_node, s_ens, synapse=tau_syn)
            self._strain_ensemble = s_ens
            
            # =====================================================================
            # PES FEEDFORWARD CONNECTION (strain → output)
            # 
            # Direct connection to u_ens - fully neural!
            # Transform: (1, strain_dim) - starts at zero, PES learns
            # =====================================================================
            if self.pes_learning_rate > 0:
                q_ens = self._snn_controller.components['q_ens']
                u_ens = self._snn_controller.components['u_ens']
                
                # PES: strain[N-D] → u_ens[1D]
                ff_conn = nengo.Connection(
                    s_ens, u_ens,
                    transform=np.zeros((1, self.strain_dim)),
                    synapse=tau_syn,
                    learning_rule_type=nengo.PES(learning_rate=self.pes_learning_rate),
                )
                
                # Learning signal: -error (positive error → increase pressure)
                nengo.Connection(q_ens, ff_conn.learning_rule, transform=-1)
                
                self._pes_connection = ff_conn
                print(f"    + PES connection: s(t)[{self.strain_dim}D] ──[PES]──> u(t)[1D]")
    
    def compute(self, target_volume: float, current_volume: float,
                strain: np.ndarray = None) -> float:
        """
        Compute pressure using NEF-based PD + PES feedforward.
        
        Args:
            target_volume: Target volume (area in 2D)
            current_volume: Current volume (area in 2D)
            strain: Optional strain array [strain_dim] (default: zeros)
        
        Returns:
            Pressure value (positive = inflate, negative = deflate)
        """
        # Update strain if provided
        if strain is not None:
            strain = np.asarray(strain).flatten()
            if len(strain) >= self.strain_dim:
                self._current_strain = np.clip(strain[:self.strain_dim], -1.0, 1.0)
            else:
                self._current_strain[:len(strain)] = np.clip(strain, -1.0, 1.0)
        
        # Compute error
        error = target_volume - current_volume
        self._current_error = error
        
        # Step simulation
        self.sim.step()
        self._sim_time += self.dt
        
        # Get output from SNN controller (includes PES feedforward)
        pressure = self._snn_controller.get_output()
        
        # Update state
        self._last_error = error
        self._last_volume = current_volume
        self.step_count += 1
        
        # Debug output every 100 steps
        if self.step_count % 100 == 0:
            strain_avg = np.mean(np.abs(self._current_strain))
            print(f"  [NengoStress step {self.step_count}] "
                  f"err={error:.4f} strain_avg={strain_avg:.3f} → pressure={pressure:.2f}")
        
        return pressure
    
    def set_strain(self, strain: np.ndarray):
        """
        Update the current strain values.
        
        Call this before compute() if strain is not passed directly.
        
        Args:
            strain: Strain array [strain_dim] (clipped to [-1, 1])
        """
        strain = np.asarray(strain).flatten()
        if len(strain) >= self.strain_dim:
            self._current_strain = np.clip(strain[:self.strain_dim], -1.0, 1.0)
        else:
            self._current_strain[:len(strain)] = np.clip(strain, -1.0, 1.0)
    
    def reset(self):
        """Reset controller state."""
        super().reset()
        self._current_strain = np.zeros(self.strain_dim)
    
    def __str__(self):
        return f"NengoStress(Kp={self.Kp}, Kd={self.Kd}, PES={self.pes_learning_rate})"


# =============================================================================
# GUI CONTROLLER (uses external Nengo model/simulator)
# =============================================================================

class SNN_Stress_Controller(SNN_PID_Controller):
    """
    NEF-based Stress Controller with PES Feedforward for Nengo GUI.
    
    Designed to work within an existing Nengo model (for Nengo GUI).
    
    - Spiking PD control (from parent with Ki=0)
    - Spiking PES feedforward (learns strain[N-D] → pressure[1D])
    
    Formula: pressure = P + D + PES_learned(strain)
    
    Can either:
    - REUSE existing strain ensemble from GUI
    - CREATE own N-D strain ensemble using get_strain_callback
    """
    
    def __init__(
        self,
        Kp: float = 1.0,
        Kd: float = 0.3,
        u_max: float = 10.0,
        n_neurons: int = 100,
        error_scale: float = 0.5,
        strain_dim: int = DEFAULT_STRAIN_DIM,
        pes_learning_rate: float = 1e-4,
    ):
        """
        Initialize stress controller for GUI (Zaidel et al. 2021 + PES).
        
        Args:
            Kp: Proportional gain
            Kd: Derivative gain
            u_max: Maximum pressure output
            n_neurons: Neurons per dimension
            error_scale: Max expected volume error
            strain_dim: Dimensions of strain input (default 7)
            pes_learning_rate: PES learning rate (0 to disable)
        """
        self.pes_learning_rate = pes_learning_rate
        self.strain_dim = strain_dim
        self.tau_syn = 0.01
        
        # Initialize parent with Ki=0 (pure PD)
        super().__init__(
            Kp=Kp,
            Ki=0.0,  # No integral for stress control
            Kd=Kd,
            u_max=u_max,
            n_neurons=n_neurons,
            error_scale=error_scale,
        )
        
        # Strain storage (if creating own ensemble)
        self._current_strain = np.zeros(strain_dim)
        self._strain_ensemble = None
        self.pes_connection = None
        
        print(f"  + PES feedforward: lr={pes_learning_rate}, strain_dim={strain_dim}")
    
    def build(
        self,
        model,
        get_error_callback,
        strain_ensemble=None,
        get_strain_callback=None,
        dt=0.01,
    ):
        """
        Build PD network and PES feedforward for volume control.
        
        FULLY NEURAL architecture (Zaidel et al. 2021).
        
        Can either:
        - REUSE strain_ensemble from GUI (auto-detects dimensions)
        - CREATE own N-D strain ensemble using get_strain_callback
        
        Args:
            model: Parent nengo.Network
            get_error_callback: Callable returning normalized error (1D scalar)
            strain_ensemble: Existing strain ensemble from GUI (optional)
            get_strain_callback: Callable returning strain array [strain_dim] (optional)
            dt: Time step (unused, kept for compatibility)
        
        Returns:
            dict with ensembles, output, components
        """
        # Build parent's PD network
        result = super().build(model, get_error_callback)
        
        if self.pes_learning_rate <= 0:
            print(f"    PES disabled (lr=0)")
            return result
        
        # Determine strain source
        use_external = strain_ensemble is not None
        use_callback = get_strain_callback is not None
        
        if not use_external and not use_callback:
            print(f"    ⚠ No strain source - PES disabled")
            print(f"       Provide strain_ensemble or get_strain_callback")
            return result
        
        print(f"  Building PES feedforward (lr={self.pes_learning_rate})...")
        
        with model:
            # Determine strain ensemble and dimensions
            if use_external:
                # Use provided strain ensemble
                s_ens = strain_ensemble
                strain_dim = s_ens.dimensions
                print(f"    Using external s_ens (dim={strain_dim})")
            else:
                # Create own strain ensemble
                strain_dim = self.strain_dim
                
                strain_node = nengo.Node(
                    get_strain_callback,
                    size_out=strain_dim,
                    label="Strain[N-D]"
                )
                
                # Triangular intercepts for high-D (Zaidel et al. 2021)
                n_s_neurons = self.n_neurons * strain_dim
                tri_samples = np.random.triangular(0.3, 0.5, 0.7, n_s_neurons)
                s_intercepts = nengo.dists.CosineSimilarity(strain_dim + 2).ppf(1 - tri_samples)
                
                s_ens = nengo.Ensemble(
                    n_neurons=n_s_neurons,
                    dimensions=strain_dim,
                    max_rates=nengo.dists.Uniform(100, 200),
                    intercepts=s_intercepts,  # Triangular for high-D
                    neuron_type=nengo.LIF(),
                    radius=1.0,
                    label="s(t)[N-D]"
                )
                nengo.Connection(strain_node, s_ens, synapse=self.tau_syn)
                self._strain_ensemble = s_ens
                print(f"    Created own s_ens (dim={strain_dim})")
            
            # Store reference
            self._strain_ensemble = s_ens
            self.all_ensembles.append(s_ens)
            
            # Get q_ens (error) and u_ens (output) from PD controller
            q_ens = self.components['q_ens']
            u_ens = self.components['u_ens']
            
            # =====================================================================
            # PES FEEDFORWARD CONNECTION
            # 
            # strain[N-D] → u_ens[1D] (learns feedforward directly to output!)
            # Transform shape: (1, strain_dim) - starts at zero, PES learns
            # =====================================================================
            ff_conn = nengo.Connection(
                s_ens, u_ens,
                transform=np.zeros((1, strain_dim)),
                synapse=self.tau_syn,
                learning_rule_type=nengo.PES(learning_rate=self.pes_learning_rate),
            )
            
            # Learning signal: -error (positive error → increase pressure)
            nengo.Connection(q_ens, ff_conn.learning_rule, transform=-1)
            
            # Store connection reference
            self.pes_connection = ff_conn
            
            print(f"    s(t)[{strain_dim}D] ──[PES]──> u(t)[1D]")
        
        print(f"  ✓ PES feedforward connected")
        return result
    
    def set_strain(self, strain: np.ndarray):
        """
        Update the current strain values (for callback).
        
        Args:
            strain: Strain array [strain_dim] (clipped to [-1, 1])
        """
        strain = np.asarray(strain).flatten()
        if len(strain) >= self.strain_dim:
            self._current_strain = np.clip(strain[:self.strain_dim], -1.0, 1.0)
        else:
            self._current_strain[:len(strain)] = np.clip(strain, -1.0, 1.0)
    
    def get_strain_callback(self):
        """
        Get a callback function for strain input.
        
        Use this with build() if you want the controller to manage its own
        strain ensemble:
        
            controller = SNN_Stress_Controller(...)
            controller.build(model, get_error_fn, 
                           get_strain_callback=controller.get_strain_callback())
            
            # In simulation loop:
            controller.set_strain(current_strain)
        
        Returns:
            Callable that returns current strain values
        """
        def get_strain(t):
            return self._current_strain
        return get_strain
    
    def __str__(self):
        return f"SNN_Stress_Controller(Kp={self.Kp}, Kd={self.Kd}, PES={self.pes_learning_rate})"


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'Stress',
    'NengoStress', 
    'SNN_Stress_Controller',
    'DEFAULT_STRAIN_DIM',
]
