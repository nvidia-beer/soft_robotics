"""
NEF-Based Spiking Neural Network PID Controller for Volume Control.

Based on: "Neuromorphic NEF-Based Inverse Kinematics and PID Control"
Zaidel et al., Frontiers in Neurorobotics, 2021
https://doi.org/10.3389/fnbot.2021.631159

ARCHITECTURE (4 ensembles, all with UNIFORM radius=1.0):

┌─────────────────────────────────────────────────────────┐
│                                                         │
│  in ──[÷e]──> q(t) ───┬────[Kp]────> u(t) ──[×u]──> out │
│               r=1     │               r=1               │
│                       │                ↑                │
│                       ├──> ei(t) ─[Ki]─┤                │
│                       │    r=1    ↺    │                │
│                       │                │                │
│                       └──> ed(t) ─[Kd]─┘                │
│                            r=1                          │
│                                                         │
│  Legend: [÷e]=÷error_scale  [×u]=×output_scale          │
│          All r=1 (uniform radius, signals in [-1,1])    │
│                                                         │
└─────────────────────────────────────────────────────────┘

This is a SCALAR (1D) PID for volume control:
- Input: volume error (target - current)
- Output: pressure (positive = inflate, negative = deflate)

NEF BEST PRACTICES (following Zaidel et al. 2021):
  1. ALL ensembles have SAME radius (~1.0) - uniform precision
  2. Normalize at INPUT boundary (divide by error_scale)
  3. Denormalize at OUTPUT boundary (multiply by output_scale)
  4. Gains in CONNECTION TRANSFORMS, not ensemble radii
  5. All internal signals stay in [-1, 1] range
"""

from .. import BaseController
import numpy as np
import nengo


def build_snn_pid_network(
    model,
    error_input_node,
    output_node,
    prefix: str = "PID",
    Kp: float = 1.0,         # Same as Classic PID default
    Ki: float = 0.5,         # Same as Classic PID default
    Kd: float = 0.3,         # Same as Classic PID default
    error_scale: float = 0.5,      # Max expected volume error
    output_scale: float = 10.0,    # Max pressure output (u_max)
    n_neurons: int = 100,
    tau_syn: float = 0.001,   # 1ms - minimal delay
    tau_int: float = 0.01,    # 10ms - fast integrator
    tau_fast: float = 0.0005, # 0.5ms - very fast
    tau_slow: float = 0.005,  # 5ms - fast derivative
    radius: float = 1.0,
):
    """
    Build a NEF-based PID controller within an existing Nengo model.
    
    Implements the architecture from Zaidel et al. 2021:
    - q(t): Error input ensemble (1D)
    - ei(t): Integrator ensemble (1D, with recurrent connection)
    - ed(t): Differentiator ensemble (1D, fast/slow pathway)
    - u(t): Output ensemble (1D, sums P + I + D)
    
    All computation is FULLY NEURAL (spiking).
    
    Args:
        model: Parent nengo.Network to build within
        error_input_node: Node providing 1D normalized error signal
        output_node: Node to receive 1D control output (scaled pressure)
        prefix: Label prefix for ensembles
        Kp: Proportional gain
        Ki: Integral gain
        Kd: Derivative gain
        error_scale: Max expected error (for normalization)
        output_scale: Max output pressure (u_max)
        n_neurons: Neurons per ensemble
        tau_syn: Standard synaptic time constant (10ms)
        tau_int: Integrator time constant (100ms)
        tau_fast: Fast synapse for derivative (2ms)
        tau_slow: Slow synapse for derivative (20ms)
        radius: UNIFORM radius for all ensembles (textbook NEF: 1.0)
    
    Returns:
        dict with references to created ensembles
    """
    
    # ==========================================================================
    # GAIN NORMALIZATION for NEF (Zaidel et al. 2021)
    # ==========================================================================
    # 
    # Classic PID:  u = Kp*e + Ki*∫e·dt + Kd*de/dt
    # NEF PID:      u = Kp_norm*e_norm + Ki_norm*ei + Kd_norm*ed
    #
    # Where:
    #   e_norm = e / error_scale (input normalization)
    #   u = u_norm * output_scale (output denormalization)
    #
    # For equivalent gains:
    #   Kp_norm = Kp * error_scale / output_scale
    #   Ki_norm = Ki * error_scale / output_scale
    #   Kd_norm = Kd * error_scale / output_scale
    #
    # ==========================================================================
    
    # Normalized gains (textbook NEF: all internal signals in [-1, 1])
    Kp_norm = Kp * error_scale / output_scale
    Ki_norm = Ki * error_scale / output_scale
    Kd_norm = Kd * error_scale / output_scale
    
    # Differentiator scale and normalization (from trajectory_tracking)
    diff_scale = 1.0 / (tau_slow - tau_fast)  # ~55.6 for trajectory_tracking tau (0.002, 0.02)
    diff_norm = diff_scale * 0.1  # Scale down to keep derivative in [-1, 1]
    
    components = {}
    
    with model:
        # =====================================================================
        # ENSEMBLE 1: q(t) - Normalized error (1D)
        # Input is ALREADY normalized to [-1, 1] by error_input_node
        # =====================================================================
        q_ens = nengo.Ensemble(
            n_neurons=n_neurons,
            dimensions=1,
            radius=radius,
            label=f"{prefix}_q(t)"
        )
        nengo.Connection(error_input_node, q_ens, synapse=None)
        components['q_ens'] = q_ens
        
        # =====================================================================
        # ENSEMBLE 2: eᵢ(t) - Integral term (NEF integrator, 1D)
        # Leaky integrator to prevent windup
        # =====================================================================
        ei_ens = nengo.Ensemble(
            n_neurons=n_neurons,
            dimensions=1,
            radius=radius,
            label=f"{prefix}_ei(t)"
        )
        # Leaky integrator: 0.999 decay (much less leaky for steady-state accuracy)
        nengo.Connection(ei_ens, ei_ens, transform=0.999, synapse=tau_int)
        nengo.Connection(q_ens, ei_ens, transform=tau_int * 2.0, synapse=tau_int)  # Higher gain
        components['ei_ens'] = ei_ens
        
        # =====================================================================
        # ENSEMBLE 3: eᵈ(t) - Derivative term (NEF differentiator, 1D)
        # More neurons for smoother derivative (from trajectory_tracking)
        # =====================================================================
        ed_ens = nengo.Ensemble(
            n_neurons=n_neurons * 4,  # More neurons for smoother derivative
            dimensions=1,
            radius=radius,
            label=f"{prefix}_ed(t)"
        )
        # NEF differentiator: fast - slow ≈ derivative
        # diff_norm keeps output in [-1, 1] range
        nengo.Connection(q_ens, ed_ens, transform=diff_norm, synapse=tau_fast)
        nengo.Connection(q_ens, ed_ens, transform=-diff_norm, synapse=tau_slow)
        components['ed_ens'] = ed_ens
        
        # =====================================================================
        # ENSEMBLE 4: u(t) - Output ensemble (sums P + I + D, 1D)
        # TEXTBOOK NEF: uniform radius=1.0
        # =====================================================================
        u_ens = nengo.Ensemble(
            n_neurons=n_neurons,
            dimensions=1,
            radius=radius,  # Uniform radius
            label=f"{prefix}_u(t)"
        )
        # P-term: Kp_norm keeps contribution bounded
        nengo.Connection(q_ens, u_ens, transform=Kp_norm, synapse=tau_syn)
        # I-term: Ki_norm × normalized integral
        nengo.Connection(ei_ens, u_ens, transform=Ki_norm, synapse=tau_syn)
        # D-term: compensate for diff_norm scaling in NEF differentiator
        Kd_effective = Kd_norm / diff_norm * diff_scale if diff_norm > 0 else 0
        nengo.Connection(ed_ens, u_ens, transform=Kd_effective, synapse=tau_syn)
        components['u_ens'] = u_ens
        
        # =====================================================================
        # OUTPUT: Scale from normalized [-1, 1] to real units
        # =====================================================================
        def make_output_scale(scale):
            def scale_output(x):
                return np.clip(x * scale, -scale, scale)
            return scale_output
        
        # Output with longer synapse for smoothing (5x tau_syn)
        # This adds slight delay but reduces spikiness
        output_synapse = tau_syn * 5  # 5ms smoothing
        nengo.Connection(u_ens, output_node,
                        function=make_output_scale(output_scale),
                        synapse=output_synapse)
    
    return components


class SNN_PID_Controller:
    """
    NEF-based Spiking PID Controller for Nengo GUI integration.
    
    SCALAR (1D) PID for volume control - single input/output.
    
    This class manages the creation and configuration of the NEF-based PID
    controller for volume-based inflation control.
    
    Designed to work within an existing Nengo model (for Nengo GUI).
    For standalone use (without GUI), see NengoPID.
    """
    
    def __init__(
        self,
        Kp: float = 1.0,         # Same as Classic PID default
        Ki: float = 0.5,         # Same as Classic PID default
        Kd: float = 0.3,         # Same as Classic PID default
        u_max: float = 10.0,
        n_neurons: int = 100,
        error_scale: float = 0.5,
    ):
        """
        Initialize SNN PID controller configuration (Zaidel et al. 2021).
        
        Gains match Classic PID defaults for equivalent behavior:
        - Kp = 1.0, Ki = 0.5, Kd = 0.3, u_max = 10.0
        
        NEF normalization: Kp_norm = Kp * error_scale / u_max
        The normalization cancels out in the output, so effective gain = Kp.
        
        Args:
            Kp: Proportional gain (default 1.0, same as Classic PID)
            Ki: Integral gain (default 0.5, same as Classic PID)
            Kd: Derivative gain (default 0.3, same as Classic PID)
            u_max: Maximum pressure output (default 10.0)
            n_neurons: Neurons per ensemble
            error_scale: Max expected volume error (for internal normalization)
        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.u_max = u_max
        self.n_neurons = n_neurons
        self.error_scale = error_scale
        
        # Minimal time constants - balloon is very responsive, need fast SNN
        self.tau_syn = 0.001   # 1ms - minimal synaptic delay
        self.tau_int = 0.01    # 10ms - fast integrator
        self.tau_fast = 0.0005 # 0.5ms - very fast
        self.tau_slow = 0.005  # 5ms - fast derivative
        
        # TEXTBOOK NEF: uniform radius
        self.radius = 1.0
        
        # Normalized gains
        self.Kp_norm = Kp * error_scale / u_max
        self.Ki_norm = Ki * error_scale / u_max
        self.Kd_norm = Kd * error_scale / u_max
        
        # Storage for built components
        self.components = {}
        self.all_ensembles = []
        self.error_input_node = None
        self.output_node = None
        
        # Output storage (updated by Nengo nodes)
        self._output = np.array([0.0], dtype=np.float32)
        
        print(f"  SNN_PID_Controller (Volume Control):")
        print(f"    Gains: Kp={Kp}, Ki={Ki}, Kd={Kd}")
        print(f"    Normalized: Kp_norm={self.Kp_norm:.2f}, Ki_norm={self.Ki_norm:.2f}, Kd_norm={self.Kd_norm:.2f}")
        print(f"    Radius: {self.radius} (uniform)")
        print(f"    Neurons: {n_neurons * 4} (4 ensembles × {n_neurons})")
    
    def build(
        self,
        model,
        get_error_callback,
    ):
        """
        Build the SNN PID network within a Nengo model.
        
        Args:
            model: Parent nengo.Network to build within
            get_error_callback: Callable that returns normalized error (1D scalar)
                Example: f(t) -> error / error_scale
        
        Returns:
            dict with:
                - 'ensembles': list of all ensembles (for visualization)
                - 'output': reference to output array
                - 'components': dict of ensemble components
        """
        print(f"  Building SNN PID controller (Zaidel et al. 2021)...")
        
        with model:
            # Create error input node
            self.error_input_node = nengo.Node(
                get_error_callback,
                size_out=1,
                label="ErrorIn"
            )
            
            # Create output store node
            def store_output(t, x):
                self._output[0] = np.clip(x[0], -self.u_max, self.u_max)
            
            self.output_node = nengo.Node(
                store_output,
                size_in=1,
                label="PressureOut"
            )
            
            # Build the PID network
            self.components = build_snn_pid_network(
                model=model,
                error_input_node=self.error_input_node,
                output_node=self.output_node,
                prefix="PID",
                Kp=self.Kp,
                Ki=self.Ki,
                Kd=self.Kd,
                error_scale=self.error_scale,
                output_scale=self.u_max,
                n_neurons=self.n_neurons,
                tau_syn=self.tau_syn,
                tau_int=self.tau_int,
                tau_fast=self.tau_fast,
                tau_slow=self.tau_slow,
                radius=self.radius,
            )
            
            # Collect ensembles for visualization
            for key in ['q_ens', 'ei_ens', 'ed_ens', 'u_ens']:
                if key in self.components:
                    self.all_ensembles.append(self.components[key])
            
            print(f"    4 ensembles created (NEF differentiator)")
        
        print(f"  ✓ SNN PID created")
        print(f"    Total ensembles: {len(self.all_ensembles)}")
        
        return {
            'ensembles': self.all_ensembles,
            'output': self._output,
            'components': self.components,
        }
    
    def get_output(self):
        """Get the current pressure output (scalar)."""
        return float(self._output[0])
    
    def get_all_ensembles(self):
        """Return list of all ensembles for visualization."""
        return self.all_ensembles
    
    def __str__(self):
        return f"SNN_PID_Controller(Kp={self.Kp}, Ki={self.Ki}, Kd={self.Kd})"


class NengoPID(BaseController):
    """
    NEF-based Spiking PID Controller for standalone use (Volume Control).
    
    SCALAR (1D) PID implementation from Zaidel et al. 2021.
    
    This creates its own Nengo model and simulator internally,
    providing the same interface as the classic PID controller.
    
    Based on: Zaidel et al., Frontiers in Neurorobotics, 2021
    """
    
    def __init__(
        self,
        dt: float = 0.01,
        u_max: float = 10.0,
        Kp: float = 1.0,         # Same as Classic PID default
        Ki: float = 0.5,         # Same as Classic PID default
        Kd: float = 0.3,         # Same as Classic PID default
        n_neurons: int = 100,
        error_scale: float = 0.5,
        integral_limit: float = 20.0,  # Kept for API compatibility (unused)
        deadband: float = 0.001,       # Kept for API compatibility (unused)
        **kwargs
    ):
        """
        Initialize standalone NEF PID controller (Zaidel et al. 2021).
        
        FULLY NEURAL - all computation in spiking neurons.
        
        Gains match Classic PID defaults for equivalent behavior.
        The NEF normalization cancels out, so effective gain = Kp.
        
        Args:
            dt: Simulation time step
            u_max: Maximum pressure output
            Kp: Proportional gain
            Ki: Integral gain
            Kd: Derivative gain
            n_neurons: Neurons per ensemble
            error_scale: Max expected error
            integral_limit: (ignored - handled by NEF leaky integrator)
            deadband: (ignored - handled by neural threshold)
        """
        super().__init__(dt, u_max)
        
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.n_neurons = n_neurons
        self.error_scale = error_scale
        
        # Internal state
        self._current_error = 0.0
        self._sim_time = 0.0
        self._last_error = 0.0
        self._last_volume = None
        
        # Build Nengo model
        self._build_model()
        
        print(f"✓ NengoPID initialized (Volume Control)")
        print(f"  Gains: Kp={Kp}, Ki={Ki}, Kd={Kd}")
        print(f"  u_max: {u_max}, error_scale: {error_scale}")
        print(f"  Neurons: {n_neurons * 4} (4 ensembles × {n_neurons})")
    
    def _build_model(self):
        """Build the internal Nengo model."""
        
        # Create inner SNN_PID_Controller
        self._snn_controller = SNN_PID_Controller(
            Kp=self.Kp,
            Ki=self.Ki,
            Kd=self.Kd,
            u_max=self.u_max,
            n_neurons=self.n_neurons,
            error_scale=self.error_scale,
        )
        
        # Error callback
        def get_error(t):
            return self._current_error / self.error_scale
        
        # Build Nengo model
        self.model = nengo.Network(label="NengoPID_Volume")
        
        with self.model:
            self._snn_controller.build(
                model=self.model,
                get_error_callback=get_error,
            )
        
        # Create simulator
        self._create_simulator()
    
    def _create_simulator(self):
        """Create the Nengo simulator."""
        self.sim = nengo.Simulator(self.model, dt=self.dt)
        print(f"    Nengo simulator created (dt={self.dt}s)")
    
    def compute(self, target_volume: float, current_volume: float) -> float:
        """
        Compute pressure from volume error using NEF-based PID.
        
        Compatible interface with classic PID controller.
        
        Args:
            target_volume: Target volume (area in 2D)
            current_volume: Current volume (area in 2D)
        
        Returns:
            Pressure value (positive = inflate, negative = deflate)
        """
        # Compute error
        error = target_volume - current_volume
        self._current_error = error
        
        # Step simulation
        self.sim.step()
        self._sim_time += self.dt
        
        # Get output from SNN controller
        pressure = self._snn_controller.get_output()
        
        # Update state
        self._last_error = error
        self._last_volume = current_volume
        self.step_count += 1
        
        # Debug output every 100 steps
        if self.step_count % 100 == 0:
            print(f"  [NengoPID step {self.step_count}] "
                  f"err={error:.4f} → pressure={pressure:.2f}")
        
        return pressure
    
    def reset(self):
        """Reset controller state."""
        super().reset()
        self._current_error = 0.0
        self._sim_time = 0.0
        self._last_error = 0.0
        self._last_volume = None
        
        # Reset SNN controller output
        self._snn_controller._output[:] = 0.0
        
        # Reset simulator
        self.sim.close()
        self._create_simulator()
    
    def set_gains(self, Kp: float = None, Ki: float = None, Kd: float = None):
        """
        Update PID gains at runtime.
        
        Note: For NEF-based PID, this requires rebuilding the network
        to change connection transforms. Not supported at runtime.
        """
        print(f"  ⚠ NengoPID: Runtime gain update not supported (requires network rebuild)")
        print(f"    Current gains: Kp={self.Kp}, Ki={self.Ki}, Kd={self.Kd}")
    
    def get_state(self) -> dict:
        """Get current controller state for debugging."""
        return {
            'error': self._current_error,
            'last_error': self._last_error,
            'step_count': self.step_count,
            'sim_time': self._sim_time,
        }
    
    def close(self):
        """Close the simulator."""
        if hasattr(self, 'sim'):
            self.sim.close()
    
    def __del__(self):
        """Cleanup on deletion."""
        self.close()
    
    def __str__(self):
        return f"NengoPID(Kp={self.Kp}, Ki={self.Ki}, Kd={self.Kd})"
